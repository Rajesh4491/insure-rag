"""
InsureRAG — Financial Insurance Risk Assessment RAG Pipeline
FastAPI application entry point
"""
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import risk, health, query, policies
from app.core.config import settings
from app.core.logging import setup_logging
from app.services.vector_store import VectorStoreService
from app.services.llm import LLMService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    setup_logging()
    logger.info("🚀 InsureRAG starting — petabyte-scale risk intelligence pipeline")

    # Warm up vector store connection
    vs = VectorStoreService()
    await vs.warmup()
    app.state.vector_store = vs

    # Warm up LLM client
    llm = LLMService()
    await llm.warmup()
    app.state.llm = llm

    logger.info("✅ All services online — ready to serve")
    yield

    logger.info("🛑 InsureRAG shutting down")
    await vs.close()


app = FastAPI(
    title="InsureRAG Risk Intelligence API",
    description=(
        "Petabyte-scale financial insurance risk assessment pipeline. "
        "RAG-powered with Azure OpenAI, LangGraph, HNSW vector search, "
        "and XGBoost + LLM + GNN ensemble scoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ──────────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# ── Routers ─────────────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(risk.router, prefix="/api/v1/risk", tags=["Risk Assessment"])
app.include_router(query.router, prefix="/api/v1/query", tags=["RAG Query"])
app.include_router(policies.router, prefix="/api/v1/policies", tags=["Policies"])


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "InsureRAG Risk Intelligence API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "operational",
    }
