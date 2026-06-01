"""
Pydantic models for request / response schemas and domain types.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── Enums ─────────────────────────────────────────────────────────────────

class RiskSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LineOfBusiness(str, Enum):
    COMMERCIAL_AUTO = "commercial_auto"
    HOMEOWNERS = "homeowners"
    WORKERS_COMP = "workers_comp"
    GENERAL_LIABILITY = "general_liability"
    COMMERCIAL_PROPERTY = "commercial_property"
    PERSONAL_AUTO = "personal_auto"
    LIFE = "life"
    CYBER = "cyber"
    MARINE = "marine"
    DO = "do"
    TECH_EO = "tech_eo"


class ClaimStatus(str, Enum):
    OPEN = "open"
    UNDER_REVIEW = "under_review"
    PENDING_SIU = "pending_siu"
    CLOSED = "closed"
    DENIED = "denied"


class FraudIndicator(str, Enum):
    CLAIM_VELOCITY = "claim_velocity"
    LATE_FNOL = "late_fnol"
    NETWORK_RING = "network_ring"
    INJURY_PATTERN = "injury_pattern"
    STAGED_ACCIDENT = "staged_accident"
    PROVIDER_FRAUD = "provider_fraud"
    PHANTOM_VEHICLE = "phantom_vehicle"


# ── Base ───────────────────────────────────────────────────────────────────

class TimestampedModel(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


# ── Risk Scoring ───────────────────────────────────────────────────────────

class SHAPExplanation(BaseModel):
    """Feature attribution from SHAP values for regulatory compliance (SR 11-7)."""
    feature_name: str
    shap_value: float
    base_value: float
    feature_value: Any
    direction: str = Field(..., pattern="^(increases|decreases)$")
    percentage_contribution: float


class EnsembleScoreBreakdown(BaseModel):
    xgboost_score: float = Field(..., ge=0, le=100)
    llm_score: float = Field(..., ge=0, le=100)
    gnn_score: float = Field(..., ge=0, le=100)
    xgboost_weight: float
    llm_weight: float
    gnn_weight: float
    final_score: float = Field(..., ge=0, le=100)


class RiskAssessmentRequest(BaseModel):
    policy_id: str = Field(..., min_length=3, max_length=50)
    line_of_business: LineOfBusiness
    include_shap: bool = Field(True, description="Generate SHAP explanations (SR 11-7)")
    include_rag_context: bool = Field(True, description="Retrieve supporting evidence")
    max_context_chunks: int = Field(10, ge=1, le=50)

    @field_validator("policy_id")
    @classmethod
    def normalize_policy_id(cls, v: str) -> str:
        return v.strip().upper()


class RiskAssessmentResponse(TimestampedModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    policy_id: str
    risk_score: float = Field(..., ge=0, le=100)
    severity: RiskSeverity
    confidence: float = Field(..., ge=0, le=1)
    top_signals: List[str]
    fraud_indicators: List[FraudIndicator]
    ensemble_breakdown: EnsembleScoreBreakdown
    shap_explanations: Optional[List[SHAPExplanation]] = None
    rag_chunks_used: int
    latency_ms: float
    regulatory_notice: Optional[str] = Field(
        None, description="SR 11-7 adverse action notice if score >= 80"
    )
    siu_referral_recommended: bool = False

    @model_validator(mode="after")
    def set_siu_referral(self) -> "RiskAssessmentResponse":
        if self.risk_score >= 80:
            self.siu_referral_recommended = True
        return self


# ── RAG Query ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000)
    policy_id: Optional[str] = None
    line_of_business: Optional[LineOfBusiness] = None
    top_k: int = Field(10, ge=1, le=50)
    include_agent_trace: bool = Field(False)

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        return v.strip()


class ContextChunk(BaseModel):
    chunk_id: str
    source: str
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentNode(BaseModel):
    node_name: str
    input_summary: str
    output_summary: str
    latency_ms: float
    iteration: int = 1


class QueryResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    answer: str
    context_chunks: List[ContextChunk]
    agent_trace: Optional[List[AgentNode]] = None
    confidence: float
    total_latency_ms: float
    retrieval_latency_ms: float
    generation_latency_ms: float
    cache_hit: bool = False
    rag_sources_count: int

    @model_validator(mode="after")
    def count_sources(self) -> "QueryResponse":
        self.rag_sources_count = len(self.context_chunks)
        return self


# ── Policies ──────────────────────────────────────────────────────────────

class PolicySummary(BaseModel):
    policy_id: str
    line_of_business: LineOfBusiness
    insured_name: str
    premium: float
    effective_date: datetime
    expiration_date: datetime
    latest_risk_score: Optional[float] = None
    latest_severity: Optional[RiskSeverity] = None
    open_claims_count: int = 0


class PolicyRiskHistory(BaseModel):
    policy_id: str
    assessments: List[Dict[str, Any]]
    trend: str = Field(..., pattern="^(improving|stable|deteriorating)$")
    assessment_count: int


# ── Ingestion ─────────────────────────────────────────────────────────────

class FNOLEvent(BaseModel):
    """First Notice of Loss — Kafka event schema."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    policy_id: str
    claim_id: str
    loss_date: datetime
    report_date: datetime = Field(default_factory=datetime.utcnow)
    loss_description: str
    loss_amount_estimate: Optional[float] = None
    claimant_name: str
    line_of_business: LineOfBusiness
    geolocation: Optional[Dict[str, float]] = None  # {lat, lon}
    telematics_attached: bool = False

    @property
    def fnol_lag_days(self) -> float:
        """Days between loss and report — key fraud signal."""
        return (self.report_date - self.loss_date).total_seconds() / 86400


# ── Health ────────────────────────────────────────────────────────────────

class ServiceStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    name: str
    status: ServiceStatus
    latency_ms: Optional[float] = None
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: ServiceStatus
    version: str
    environment: str
    uptime_seconds: float
    components: List[ComponentHealth]
    vector_index_size: Optional[int] = Field(None, description="Total indexed vectors")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
