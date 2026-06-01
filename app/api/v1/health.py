"""Health check router."""
import time
from fastapi import APIRouter, Request
from app.models.schemas import ComponentHealth, HealthResponse, ServiceStatus

router = APIRouter()
_start_time = time.time()


@router.get("/", response_model=HealthResponse, summary="Health check")
async def health(request: Request) -> HealthResponse:
    components = []

    # Vector store
    vs = getattr(request.app.state, "vector_store", None)
    if vs:
        t0 = time.perf_counter()
        try:
            stats = await vs.get_index_stats()
            latency = (time.perf_counter() - t0) * 1000
            components.append(ComponentHealth(
                name="azure_ai_search",
                status=ServiceStatus.HEALTHY,
                latency_ms=round(latency, 1),
                detail=f"{stats.get('document_count', '?')} vectors indexed",
            ))
        except Exception as e:
            components.append(ComponentHealth(name="azure_ai_search", status=ServiceStatus.UNHEALTHY, detail=str(e)))
    else:
        components.append(ComponentHealth(name="azure_ai_search", status=ServiceStatus.DEGRADED, detail="Not initialized"))

    overall = ServiceStatus.HEALTHY
    if any(c.status == ServiceStatus.UNHEALTHY for c in components):
        overall = ServiceStatus.UNHEALTHY
    elif any(c.status == ServiceStatus.DEGRADED for c in components):
        overall = ServiceStatus.DEGRADED

    from app.core.config import settings
    return HealthResponse(
        status=overall,
        version="1.0.0",
        environment=settings.APP_ENV,
        uptime_seconds=round(time.time() - _start_time, 1),
        components=components,
    )


@router.get("/live", summary="Kubernetes liveness probe")
async def liveness():
    return {"status": "alive"}


@router.get("/ready", summary="Kubernetes readiness probe")
async def readiness(request: Request):
    vs = getattr(request.app.state, "vector_store", None)
    if vs is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Vector store not ready")
    return {"status": "ready"}
