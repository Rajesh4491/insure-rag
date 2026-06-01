"""
Central configuration — all settings from environment variables.
Uses Pydantic BaseSettings for validation and dotenv support.
"""
from typing import List, Optional
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ──────────────────────────────────────────────────────────────
    APP_ENV: str = Field("development", description="development | staging | production")
    LOG_LEVEL: str = Field("INFO")
    ALLOWED_ORIGINS: List[str] = Field(default=["*"])
    API_KEY_HEADER: str = Field("X-API-Key")
    API_KEYS: List[str] = Field(default=["dev-key-change-in-prod"])

    # ── Azure OpenAI ─────────────────────────────────────────────────────
    AZURE_OPENAI_ENDPOINT: str = Field(..., description="https://<resource>.openai.azure.com/")
    AZURE_OPENAI_API_KEY: str = Field(...)
    AZURE_OPENAI_API_VERSION: str = Field("2024-08-01-preview")
    AZURE_OPENAI_DEPLOYMENT_GPT4O: str = Field("gpt-4o-insurerag")
    AZURE_OPENAI_DEPLOYMENT_EMBED: str = Field("text-embedding-3-large")
    AZURE_OPENAI_EMBED_DIM: int = Field(3072)

    # ── Azure AI Search (Vector Store) ───────────────────────────────────
    AZURE_SEARCH_ENDPOINT: str = Field(...)
    AZURE_SEARCH_KEY: str = Field(...)
    AZURE_SEARCH_INDEX_NAME: str = Field("insurerag-vectors")
    AZURE_SEARCH_SEMANTIC_CONFIG: str = Field("insurerag-semantic")
    VECTOR_SEARCH_TOP_K: int = Field(30, ge=1, le=200)
    HNSW_EF_SEARCH: int = Field(64, description="HNSW efSearch parameter")

    # ── Redis (Semantic Cache) ────────────────────────────────────────────
    REDIS_URL: str = Field("redis://localhost:6379/0")
    CACHE_TTL_SECONDS: int = Field(300, ge=0)
    CACHE_SIMILARITY_THRESHOLD: float = Field(0.97, ge=0.0, le=1.0)

    # ── MLflow ───────────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = Field("http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME: str = Field("insurerag-risk-scoring")
    MLFLOW_REGISTRY_MODEL_NAME: str = Field("insurerag-ensemble")

    # ── Azure Blob (Raw data lake) ────────────────────────────────────────
    AZURE_STORAGE_CONN_STR: Optional[str] = Field(None)
    AZURE_STORAGE_CONTAINER: str = Field("insurerag-datalake")

    # ── Kafka (Streaming ingestion) ───────────────────────────────────────
    KAFKA_BOOTSTRAP_SERVERS: str = Field("localhost:9092")
    KAFKA_CLAIMS_TOPIC: str = Field("claims-fnol")
    KAFKA_CONSUMER_GROUP: str = Field("insurerag-ingest")

    # ── Risk Scoring ─────────────────────────────────────────────────────
    RISK_SCORE_CRITICAL_THRESHOLD: int = Field(80)
    RISK_SCORE_HIGH_THRESHOLD: int = Field(65)
    RISK_SCORE_MEDIUM_THRESHOLD: int = Field(45)
    XGB_MODEL_PATH: str = Field("models/xgb_risk_v2.ubj")
    GNN_MODEL_PATH: str = Field("models/gnn_fraud_v1.pt")
    ENSEMBLE_WEIGHTS: List[float] = Field(default=[0.4, 0.35, 0.25])  # XGB, LLM, GNN

    # ── LangGraph ────────────────────────────────────────────────────────
    LANGGRAPH_MAX_RETRIES: int = Field(2, ge=0, le=5)
    LANGGRAPH_REFLECTION_CONFIDENCE_THRESHOLD: float = Field(0.70)
    CONTEXT_GRADE_MIN_RELEVANCE: float = Field(0.65)

    # ── Observability ────────────────────────────────────────────────────
    AZURE_MONITOR_CONNECTION_STRING: Optional[str] = Field(None)
    ENABLE_TELEMETRY: bool = Field(True)

    @field_validator("ENSEMBLE_WEIGHTS")
    @classmethod
    def weights_sum_to_one(cls, v: List[float]) -> List[float]:
        if abs(sum(v) - 1.0) > 1e-6:
            raise ValueError(f"ENSEMBLE_WEIGHTS must sum to 1.0, got {sum(v)}")
        return v

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"

    @property
    def risk_tier(self) -> dict:
        return {
            "critical": self.RISK_SCORE_CRITICAL_THRESHOLD,
            "high": self.RISK_SCORE_HIGH_THRESHOLD,
            "medium": self.RISK_SCORE_MEDIUM_THRESHOLD,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
