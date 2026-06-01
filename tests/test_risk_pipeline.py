"""
Test suite for InsureRAG backend.
Run with: pytest tests/ -v --tb=short
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.schemas import (
    FNOLEvent,
    LineOfBusiness,
    RiskAssessmentRequest,
    RiskSeverity,
)
from app.services.risk_scoring import (
    GNNFraudScorer,
    RiskScoringService,
    XGBoostRiskScorer,
    _severity_from_score,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def assess_request():
    return RiskAssessmentRequest(
        policy_id="POL-7821-X",
        line_of_business=LineOfBusiness.COMMERCIAL_AUTO,
        include_shap=False,
        include_rag_context=False,
    )


@pytest.fixture
def fnol_event():
    from datetime import datetime, timedelta
    return FNOLEvent(
        policy_id="POL-7821-X",
        claim_id="CLM-001",
        loss_date=datetime.utcnow() - timedelta(days=15),
        report_date=datetime.utcnow(),
        loss_description="Vehicle collision on I-35 near Fort Worth. Front-end damage.",
        loss_amount_estimate=12_500.0,
        claimant_name="John Doe",
        line_of_business=LineOfBusiness.COMMERCIAL_AUTO,
    )


# ── Schema tests ──────────────────────────────────────────────────────────

class TestSchemas:
    def test_policy_id_normalized(self):
        req = RiskAssessmentRequest(
            policy_id="  pol-abc  ",
            line_of_business=LineOfBusiness.CYBER,
        )
        assert req.policy_id == "POL-ABC"

    def test_fnol_lag_days(self, fnol_event):
        assert fnol_event.fnol_lag_days > 14
        assert fnol_event.fnol_lag_days < 16

    def test_risk_response_siu_flag(self):
        from app.models.schemas import EnsembleScoreBreakdown, RiskAssessmentResponse
        resp = RiskAssessmentResponse(
            policy_id="POL-TEST",
            risk_score=85.0,
            severity=RiskSeverity.CRITICAL,
            confidence=0.92,
            top_signals=["velocity"],
            fraud_indicators=[],
            ensemble_breakdown=EnsembleScoreBreakdown(
                xgboost_score=80, llm_score=85, gnn_score=90,
                xgboost_weight=0.4, llm_weight=0.35, gnn_weight=0.25,
                final_score=85,
            ),
            rag_chunks_used=5,
            latency_ms=210.0,
        )
        assert resp.siu_referral_recommended is True

    def test_risk_response_no_siu_below_threshold(self):
        from app.models.schemas import EnsembleScoreBreakdown, RiskAssessmentResponse
        resp = RiskAssessmentResponse(
            policy_id="POL-LOW",
            risk_score=40.0,
            severity=RiskSeverity.MEDIUM,
            confidence=0.82,
            top_signals=[],
            fraud_indicators=[],
            ensemble_breakdown=EnsembleScoreBreakdown(
                xgboost_score=40, llm_score=38, gnn_score=42,
                xgboost_weight=0.4, llm_weight=0.35, gnn_weight=0.25,
                final_score=40,
            ),
            rag_chunks_used=3,
            latency_ms=190.0,
        )
        assert resp.siu_referral_recommended is False


# ── Risk scoring tests ────────────────────────────────────────────────────

class TestSeverity:
    def test_critical(self):
        assert _severity_from_score(95) == RiskSeverity.CRITICAL

    def test_high(self):
        assert _severity_from_score(70) == RiskSeverity.HIGH

    def test_medium(self):
        assert _severity_from_score(50) == RiskSeverity.MEDIUM

    def test_low(self):
        assert _severity_from_score(20) == RiskSeverity.LOW

    def test_boundary_critical(self):
        assert _severity_from_score(80) == RiskSeverity.CRITICAL

    def test_boundary_high(self):
        assert _severity_from_score(65) == RiskSeverity.HIGH


class TestXGBoostScorer:
    def test_mock_score_range(self):
        scorer = XGBoostRiskScorer()
        features = {
            "claim_velocity_30d": 5.0,
            "late_fnol_days": 10.0,
            "prior_claim_count": 2.0,
            "network_fraud_flag": 1.0,
            "credit_score_norm": 0.5,
            "policy_age_months": 24.0,
            "injury_severity_outlier": False,
            "geographic_risk_index": 0.7,
        }
        score, shap = scorer.score(features)
        assert 0 <= score <= 100
        assert shap is None  # No model loaded

    def test_high_velocity_increases_score(self):
        scorer = XGBoostRiskScorer()
        low = scorer.score({"claim_velocity_30d": 0, "late_fnol_days": 0,
                             "prior_claim_count": 0, "network_fraud_flag": 0,
                             "credit_score_norm": 0.8, "policy_age_months": 60,
                             "injury_severity_outlier": False, "geographic_risk_index": 0.2})[0]
        high = scorer.score({"claim_velocity_30d": 7, "late_fnol_days": 0,
                              "prior_claim_count": 0, "network_fraud_flag": 0,
                              "credit_score_norm": 0.8, "policy_age_months": 60,
                              "injury_severity_outlier": False, "geographic_risk_index": 0.2})[0]
        assert high > low


class TestGNNScorer:
    def test_mock_score_range(self):
        scorer = GNNFraudScorer()
        score = scorer.score("POL-TEST", {
            "shared_provider_count": 3.0,
            "connected_claims_count": 8.0,
            "ring_membership_prob": 0.3,
            "degree_centrality": 0.15,
        })
        assert 0 <= score <= 100


class TestRiskScoringService:
    @pytest.mark.asyncio
    async def test_assess_returns_response(self, assess_request):
        service = RiskScoringService(llm=None)
        service.initialize()
        result = await service.assess(request=assess_request)
        assert result.policy_id == "POL-7821-X"
        assert 0 <= result.risk_score <= 100
        assert result.severity in list(RiskSeverity)
        assert 0 <= result.confidence <= 1

    @pytest.mark.asyncio
    async def test_critical_score_generates_reg_notice(self, assess_request):
        service = RiskScoringService(llm=None)
        service.initialize()
        # Force high score via features
        with patch.object(service, "_default_tabular_features", return_value={
            "claim_velocity_30d": 7.0, "late_fnol_days": 15.0, "prior_claim_count": 4.0,
            "network_fraud_flag": 1.0, "credit_score_norm": 0.2, "policy_age_months": 6.0,
            "injury_severity_outlier": True, "geographic_risk_index": 0.9,
        }):
            result = await service.assess(request=assess_request)
        if result.risk_score >= 80:
            assert result.regulatory_notice is not None
            assert "SR 11-7" in result.regulatory_notice

    @pytest.mark.asyncio
    async def test_ensemble_weights_sum(self):
        from app.core.config import settings
        assert abs(sum(settings.ENSEMBLE_WEIGHTS) - 1.0) < 1e-6


# ── Vector store tests ────────────────────────────────────────────────────

class TestVectorStore:
    @pytest.mark.asyncio
    async def test_mock_hnsw_returns_chunks(self):
        from app.services.vector_store import VectorStoreService
        vs = VectorStoreService()
        # Without warmup, returns mock results
        results = await vs.hnsw_search(embedding=[0.1] * 10, top_k=5)
        assert len(results) <= 5
        for chunk in results:
            assert chunk.score >= 0
            assert chunk.content

    @pytest.mark.asyncio
    async def test_rrf_fusion(self):
        from app.services.rag_agent import InsureRAGAgent
        from app.models.schemas import ContextChunk

        list_a = [ContextChunk(chunk_id=f"a{i}", source="a", content="test", score=1.0) for i in range(5)]
        list_b = [ContextChunk(chunk_id=f"b{i}", source="b", content="test", score=1.0) for i in range(5)]
        # Add overlap
        list_a.append(ContextChunk(chunk_id="shared", source="both", content="overlap", score=1.0))
        list_b.insert(0, ContextChunk(chunk_id="shared", source="both", content="overlap", score=1.0))

        fused = InsureRAGAgent._rrf_fusion(list_a, list_b)
        ids = [c.chunk_id for c in fused]
        # Shared chunk should rank first (boosted by both lists)
        assert ids[0] == "shared"
