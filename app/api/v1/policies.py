"""Policies API router."""
from fastapi import APIRouter, Query
from typing import List, Optional
from app.models.schemas import LineOfBusiness, PolicySummary

router = APIRouter()


@router.get("/", response_model=List[dict], summary="List policies with latest risk scores")
async def list_policies(
    lob: Optional[LineOfBusiness] = Query(None),
    min_risk_score: float = Query(0.0, ge=0, le=100),
    limit: int = Query(20, ge=1, le=100),
):
    # In production, query from policy database with risk score join
    mock = [
        {"policy_id": "POL-7821-X", "line_of_business": "commercial_auto", "risk_score": 91.0, "severity": "critical"},
        {"policy_id": "POL-3341-B", "line_of_business": "homeowners", "risk_score": 87.0, "severity": "critical"},
        {"policy_id": "POL-9182-C", "line_of_business": "workers_comp", "risk_score": 83.0, "severity": "critical"},
        {"policy_id": "POL-5510-M", "line_of_business": "general_liability", "risk_score": 74.0, "severity": "high"},
        {"policy_id": "POL-4455-L", "line_of_business": "cyber", "risk_score": 34.0, "severity": "low"},
    ]
    if lob:
        mock = [p for p in mock if p["line_of_business"] == lob.value]
    mock = [p for p in mock if p["risk_score"] >= min_risk_score]
    return mock[:limit]


@router.get("/{policy_id}", summary="Get policy details with risk assessment")
async def get_policy(policy_id: str):
    return {
        "policy_id": policy_id.upper(),
        "insured_name": "Acme Corp",
        "line_of_business": "commercial_auto",
        "premium": 48_500.00,
        "effective_date": "2025-01-01",
        "expiration_date": "2026-01-01",
        "latest_risk_score": 91.0,
        "latest_severity": "critical",
        "open_claims_count": 14,
    }
