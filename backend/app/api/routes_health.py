from fastapi import APIRouter

from app.api.dependencies import get_ollama_client, get_vector_store
from app.config import get_settings
from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    settings = get_settings()
    ollama_ok = get_ollama_client().healthcheck()
    qdrant_ok = get_vector_store().healthcheck()
    return HealthResponse(
        status="ok" if ollama_ok and qdrant_ok else "degraded",
        app_name=settings.app_name,
        environment=settings.environment,
        ollama_reachable=ollama_ok,
        qdrant_reachable=qdrant_ok,
    )
