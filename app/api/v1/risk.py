"""
Risk Assessment API endpoints.
POST /api/v1/risk/assess   — Score a single policy
POST /api/v1/risk/batch    — Score up to 50 policies in parallel
GET  /api/v1/risk/{policy_id}/history — Historical risk scores
"""
from __future__ import annotations

import asyncio
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.core.config import settings
from app.models.schemas import (
    RiskAssessmentRequest,
    RiskAssessmentResponse,
)
from app.services.risk_scoring import RiskScoringService
from app.services.rag_agent import InsureRAGAgent
from app.models.schemas import QueryRequest

router = APIRouter()
logger = logging.getLogger(__name__)


def get_risk_scorer(request: Request) -> RiskScoringService:
    llm = getattr(request.app.state, "llm", None)
    scorer = RiskScoringService(llm=llm.client if llm else None)
    scorer.initialize()
    return scorer


def get_rag_agent(request: Request) -> InsureRAGAgent:
    vs = request.app.state.vector_store
    from app.services.cache import SemanticCacheService
    cache = SemanticCacheService()
    return InsureRAGAgent(vector_store=vs, cache=cache)


@router.post(
    "/assess",
    response_model=RiskAssessmentResponse,
    summary="Assess risk for a single policy",
    description=(
        "Runs XGBoost + LLM + GNN ensemble scoring with RAG-retrieved context. "
        "Returns risk score 0–100, severity tier, SHAP explanations, and fraud indicators. "
        "SR 11-7 adverse action notice generated for scores ≥ 80."
    ),
)
async def assess_policy_risk(
    body: RiskAssessmentRequest,
    scorer: RiskScoringService = Depends(get_risk_scorer),
    agent: InsureRAGAgent = Depends(get_rag_agent),
) -> RiskAssessmentResponse:
    logger.info("Risk assessment request: %s (%s)", body.policy_id, body.line_of_business.value)

    # Retrieve supporting context chunks if requested
    context_text: List[str] = []
    if body.include_rag_context:
        try:
            rag_result = await agent.run(
                QueryRequest(
                    question=f"What are the risk factors for policy {body.policy_id} in {body.line_of_business.value}?",
                    policy_id=body.policy_id,
                    line_of_business=body.line_of_business,
                    top_k=body.max_context_chunks,
                )
            )
            context_text = [c.content for c in rag_result.context_chunks]
        except Exception as e:
            logger.warning("RAG context retrieval failed: %s — scoring without context", e)

    result = await scorer.assess(request=body, context_chunks=context_text)
    return result


@router.post(
    "/batch",
    response_model=List[RiskAssessmentResponse],
    summary="Batch assess up to 50 policies",
)
async def batch_assess(
    requests: List[RiskAssessmentRequest],
    scorer: RiskScoringService = Depends(get_risk_scorer),
) -> List[RiskAssessmentResponse]:
    if len(requests) > 50:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Batch size limit is 50 policies per request.",
        )
    tasks = [scorer.assess(request=r) for r in requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    responses = []
    for r in results:
        if isinstance(r, Exception):
            logger.error("Batch assessment error: %s", r)
            raise HTTPException(status_code=500, detail=f"Scoring error: {r}")
        responses.append(r)
    return responses


@router.get(
    "/{policy_id}/history",
    summary="Retrieve historical risk score assessments for a policy",
)
async def get_risk_history(policy_id: str) -> dict:
    # In production, query from Azure Cosmos DB or PostgreSQL
    return {
        "policy_id": policy_id.upper(),
        "assessments": [
            {"date": "2025-09-01", "score": 45.2, "severity": "medium"},
            {"date": "2025-10-01", "score": 58.7, "severity": "high"},
            {"date": "2025-11-01", "score": 71.3, "severity": "high"},
            {"date": "2025-12-01", "score": 82.4, "severity": "critical"},
            {"date": "2026-01-01", "score": 91.0, "severity": "critical"},
        ],
        "trend": "deteriorating",
        "assessment_count": 5,
    }
