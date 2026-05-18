import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes_chat import router as chat_router
from app.api.routes_documents import router as documents_router
from app.api.routes_eval import router as eval_router
from app.api.routes_health import router as health_router
from app.config import get_settings
from app.logging_config import configure_logging
from app.retrieval.vector_store import VectorStoreError
from app.generation.ollama_client import OllamaServiceError

configure_logging()
settings = get_settings()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(documents_router)
app.include_router(chat_router)
app.include_router(eval_router)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    logger.info("request_started", extra={"path": request.url.path, "method": request.method})
    response = await call_next(request)
    logger.info(
        "request_completed",
        extra={"path": request.url.path, "status_code": response.status_code},
    )
    return response


@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(OllamaServiceError)
async def ollama_error_handler(_: Request, exc: OllamaServiceError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(VectorStoreError)
async def qdrant_error_handler(_: Request, exc: VectorStoreError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
