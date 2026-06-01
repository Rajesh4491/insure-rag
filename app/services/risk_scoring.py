"""
Insurance Risk Scoring Ensemble
Combines XGBoost (tabular), LLM (narrative), and GNN (fraud network) models.
MLflow tracking + SHAP explanations for SR 11-7 compliance.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np

from app.core.config import settings
from app.models.schemas import (
    EnsembleScoreBreakdown,
    FraudIndicator,
    RiskAssessmentRequest,
    RiskAssessmentResponse,
    RiskSeverity,
    SHAPExplanation,
)

logger = logging.getLogger(__name__)


def _severity_from_score(score: float) -> RiskSeverity:
    if score >= settings.RISK_SCORE_CRITICAL_THRESHOLD:
        return RiskSeverity.CRITICAL
    if score >= settings.RISK_SCORE_HIGH_THRESHOLD:
        return RiskSeverity.HIGH
    if score >= settings.RISK_SCORE_MEDIUM_THRESHOLD:
        return RiskSeverity.MEDIUM
    return RiskSeverity.LOW


class XGBoostRiskScorer:
    """Tabular risk scorer — claim velocity, loss history, geo features."""

    def __init__(self):
        self._model = None
        self._explainer = None

    def load(self):
        model_path = Path(settings.XGB_MODEL_PATH)
        if model_path.exists():
            try:
                import xgboost as xgb
                import shap

                self._model = xgb.Booster()
                self._model.load_model(str(model_path))
                self._explainer = shap.TreeExplainer(self._model)
                logger.info("XGBoost model loaded from %s", model_path)
            except Exception as e:
                logger.warning("XGBoost model load failed: %s — using mock scorer", e)
        else:
            logger.warning("XGBoost model not found at %s — using mock scorer", model_path)

    def score(self, features: Dict[str, Any]) -> Tuple[float, Optional[List[SHAPExplanation]]]:
        """Returns (score 0–100, shap_explanations or None)."""
        if self._model is None:
            return self._mock_score(features), None

        try:
            import xgboost as xgb
            import shap

            feat_arr = np.array(list(features.values())).reshape(1, -1)
            dmatrix = xgb.DMatrix(feat_arr, feature_names=list(features.keys()))
            raw_score = float(self._model.predict(dmatrix)[0])
            score = min(100.0, max(0.0, raw_score * 100))

            shap_vals = self._explainer.shap_values(feat_arr)
            base_val = float(self._explainer.expected_value)
            explanations = []
            for i, (feat_name, feat_val) in enumerate(features.items()):
                sv = float(shap_vals[0][i])
                explanations.append(
                    SHAPExplanation(
                        feature_name=feat_name,
                        shap_value=round(sv, 4),
                        base_value=round(base_val, 4),
                        feature_value=feat_val,
                        direction="increases" if sv > 0 else "decreases",
                        percentage_contribution=round(abs(sv) / (sum(abs(s) for s in shap_vals[0]) + 1e-9) * 100, 2),
                    )
                )
            explanations.sort(key=lambda x: abs(x.shap_value), reverse=True)
            return score, explanations

        except Exception as e:
            logger.error("XGBoost inference error: %s", e)
            return self._mock_score(features), None

    @staticmethod
    def _mock_score(features: Dict[str, Any]) -> float:
        """Deterministic mock based on known risk features."""
        score = 30.0
        score += features.get("claim_velocity_30d", 0) * 12
        score += features.get("late_fnol_days", 0) * 1.5
        score += features.get("prior_claim_count", 0) * 3
        score += features.get("network_fraud_flag", 0) * 25
        score -= features.get("credit_score_norm", 0) * 5
        return min(100.0, max(0.0, score))


class LLMRiskScorer:
    """Narrative-based risk scorer using Azure GPT-4o structured output."""

    def __init__(self, llm):
        self.llm = llm

    async def score(self, policy_id: str, context_chunks: List[str]) -> Tuple[float, List[str]]:
        """Returns (score 0–100, top_signals)."""
        if not context_chunks:
            return 50.0, ["Insufficient context for narrative analysis"]

        context = "\n\n".join(context_chunks[:5])
        prompt = f"""Analyze the following insurance policy evidence and return a JSON risk assessment.

Policy ID: {policy_id}
Context:
{context}

Return ONLY valid JSON:
{{
  "risk_score": <0-100 float>,
  "top_signals": ["signal1", "signal2", "signal3"],
  "fraud_indicators": ["indicator1"],
  "confidence": <0-1 float>
}}"""

        try:
            from langchain_core.messages import HumanMessage
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            import json, re
            match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if match:
                result = json.loads(match.group())
                score = float(result.get("risk_score", 50.0))
                signals = result.get("top_signals", [])
                return min(100.0, max(0.0, score)), signals
        except Exception as e:
            logger.error("LLM risk scoring failed: %s", e)

        return 50.0, ["LLM scoring unavailable"]


class GNNFraudScorer:
    """Graph Neural Network fraud ring detector."""

    def __init__(self):
        self._model = None

    def load(self):
        model_path = Path(settings.GNN_MODEL_PATH)
        if model_path.exists():
            try:
                import torch
                self._model = torch.load(str(model_path), map_location="cpu")
                self._model.eval()
                logger.info("GNN fraud model loaded from %s", model_path)
            except Exception as e:
                logger.warning("GNN model load failed: %s — using mock scorer", e)
        else:
            logger.warning("GNN model not found at %s — using mock scorer", model_path)

    def score(self, policy_id: str, graph_features: Dict[str, Any]) -> float:
        """Returns fraud ring probability score 0–100."""
        if self._model is None:
            return self._mock_score(graph_features)
        try:
            import torch
            feat_tensor = torch.tensor(
                list(graph_features.values()), dtype=torch.float32
            ).unsqueeze(0)
            with torch.no_grad():
                prob = float(torch.sigmoid(self._model(feat_tensor)).item())
            return min(100.0, max(0.0, prob * 100))
        except Exception as e:
            logger.error("GNN inference error: %s", e)
            return self._mock_score(graph_features)

    @staticmethod
    def _mock_score(features: Dict[str, Any]) -> float:
        score = 20.0
        score += features.get("shared_provider_count", 0) * 8
        score += features.get("connected_claims_count", 0) * 5
        score += features.get("ring_membership_prob", 0) * 40
        return min(100.0, max(0.0, score))


class RiskScoringService:
    """Orchestrates XGBoost + LLM + GNN ensemble with MLflow tracking."""

    def __init__(self, llm=None):
        self.xgb_scorer = XGBoostRiskScorer()
        self.llm_scorer = LLMRiskScorer(llm) if llm else None
        self.gnn_scorer = GNNFraudScorer()
        self.weights = settings.ENSEMBLE_WEIGHTS  # [xgb, llm, gnn]

        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    def initialize(self):
        """Load models at startup."""
        self.xgb_scorer.load()
        self.gnn_scorer.load()
        logger.info("Risk scoring ensemble initialized")

    async def assess(
        self,
        request: RiskAssessmentRequest,
        context_chunks: Optional[List[str]] = None,
        tabular_features: Optional[Dict[str, Any]] = None,
        graph_features: Optional[Dict[str, Any]] = None,
    ) -> RiskAssessmentResponse:
        """Run full ensemble and return scored risk assessment."""
        t0 = time.perf_counter()

        # Defaults
        tab_feat = tabular_features or self._default_tabular_features(request.policy_id)
        graph_feat = graph_features or self._default_graph_features(request.policy_id)
        chunks = context_chunks or []

        # Run all three scorers (XGB + GNN sync, LLM async)
        xgb_score, shap_exps = self.xgb_scorer.score(tab_feat)
        gnn_score = self.gnn_scorer.score(request.policy_id, graph_feat)

        if self.llm_scorer:
            llm_score, llm_signals = await self.llm_scorer.score(request.policy_id, chunks)
        else:
            llm_score = (xgb_score + gnn_score) / 2
            llm_signals = ["LLM scorer not available"]

        # Weighted ensemble
        w_xgb, w_llm, w_gnn = self.weights
        final_score = w_xgb * xgb_score + w_llm * llm_score + w_gnn * gnn_score
        final_score = round(min(100.0, max(0.0, final_score)), 2)

        severity = _severity_from_score(final_score)
        confidence = self._compute_confidence(xgb_score, llm_score, gnn_score)
        fraud_indicators = self._detect_fraud_indicators(tab_feat, graph_feat)
        latency_ms = (time.perf_counter() - t0) * 1000

        # SR 11-7 regulatory notice
        reg_notice = None
        if final_score >= settings.RISK_SCORE_CRITICAL_THRESHOLD:
            reg_notice = (
                f"ADVERSE ACTION NOTICE (SR 11-7): Policy {request.policy_id} scored "
                f"{final_score:.0f}/100 ({severity.value.upper()}). "
                "Automated model decision — human review required before adverse action."
            )

        breakdown = EnsembleScoreBreakdown(
            xgboost_score=round(xgb_score, 2),
            llm_score=round(llm_score, 2),
            gnn_score=round(gnn_score, 2),
            xgboost_weight=w_xgb,
            llm_weight=w_llm,
            gnn_weight=w_gnn,
            final_score=final_score,
        )

        # MLflow logging
        self._log_to_mlflow(request.policy_id, breakdown, final_score, confidence, latency_ms)

        return RiskAssessmentResponse(
            policy_id=request.policy_id,
            risk_score=final_score,
            severity=severity,
            confidence=round(confidence, 3),
            top_signals=llm_signals[:5],
            fraud_indicators=fraud_indicators,
            ensemble_breakdown=breakdown,
            shap_explanations=shap_exps[:10] if request.include_shap and shap_exps else None,
            rag_chunks_used=len(chunks),
            latency_ms=round(latency_ms, 1),
            regulatory_notice=reg_notice,
        )

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _compute_confidence(xgb: float, llm: float, gnn: float) -> float:
        """Confidence = 1 - normalized std deviation of scorer agreement."""
        vals = np.array([xgb, llm, gnn])
        std = float(np.std(vals))
        return max(0.5, 1.0 - std / 100.0)

    @staticmethod
    def _detect_fraud_indicators(
        tab_feat: Dict[str, Any], graph_feat: Dict[str, Any]
    ) -> List[FraudIndicator]:
        indicators = []
        if tab_feat.get("claim_velocity_30d", 0) > 3:
            indicators.append(FraudIndicator.CLAIM_VELOCITY)
        if tab_feat.get("late_fnol_days", 0) > 7:
            indicators.append(FraudIndicator.LATE_FNOL)
        if graph_feat.get("connected_claims_count", 0) > 5:
            indicators.append(FraudIndicator.NETWORK_RING)
        if tab_feat.get("injury_severity_outlier", False):
            indicators.append(FraudIndicator.INJURY_PATTERN)
        if graph_feat.get("shared_provider_count", 0) > 2:
            indicators.append(FraudIndicator.PROVIDER_FRAUD)
        return indicators

    @staticmethod
    def _default_tabular_features(policy_id: str) -> Dict[str, Any]:
        """Stub feature extraction — replace with real feature store lookup."""
        seed = sum(ord(c) for c in policy_id)
        rng = np.random.default_rng(seed)
        return {
            "claim_velocity_30d": float(rng.integers(0, 8)),
            "late_fnol_days": float(rng.integers(0, 20)),
            "prior_claim_count": float(rng.integers(0, 5)),
            "network_fraud_flag": float(rng.integers(0, 2)),
            "credit_score_norm": float(rng.uniform(0.3, 1.0)),
            "policy_age_months": float(rng.integers(1, 120)),
            "injury_severity_outlier": bool(rng.integers(0, 2)),
            "geographic_risk_index": float(rng.uniform(0.1, 0.9)),
        }

    @staticmethod
    def _default_graph_features(policy_id: str) -> Dict[str, Any]:
        seed = sum(ord(c) for c in policy_id) + 1
        rng = np.random.default_rng(seed)
        return {
            "shared_provider_count": float(rng.integers(0, 6)),
            "connected_claims_count": float(rng.integers(0, 12)),
            "ring_membership_prob": float(rng.uniform(0.0, 0.5)),
            "degree_centrality": float(rng.uniform(0.0, 0.3)),
        }

    def _log_to_mlflow(
        self,
        policy_id: str,
        breakdown: EnsembleScoreBreakdown,
        final_score: float,
        confidence: float,
        latency_ms: float,
    ):
        try:
            with mlflow.start_run(run_name=f"risk_score_{policy_id}", nested=True):
                mlflow.log_metrics({
                    "xgb_score": breakdown.xgboost_score,
                    "llm_score": breakdown.llm_score,
                    "gnn_score": breakdown.gnn_score,
                    "final_score": final_score,
                    "confidence": confidence,
                    "latency_ms": latency_ms,
                })
                mlflow.log_params({
                    "policy_id": policy_id,
                    "xgb_weight": breakdown.xgboost_weight,
                    "llm_weight": breakdown.llm_weight,
                    "gnn_weight": breakdown.gnn_weight,
                })
        except Exception as e:
            logger.debug("MLflow logging failed (non-critical): %s", e)
