from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # config.py lives at backend/app/config.py, so parents[1] resolves to backend/
    base_dir: Path = Path(__file__).resolve().parents[1]
    app_name: str = "local-production-rag-agent"
    environment: str = "local"
    debug: bool = False

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:8501",
            "http://127.0.0.1:8501",
            "http://localhost:3000",
        ]
    )

    ollama_base_url: str = "http://host.docker.internal:11434"
    ollama_llm_model: str = "qwen3:8b"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_eval_model: str | None = None
    ollama_timeout_seconds: float = 90.0
    ollama_max_retries: int = 3
    ollama_num_predict: int = 256
    ollama_eval_num_predict: int = 120
    ollama_eval_fast_num_predict: int = 80

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection_name: str = "rag_chunks"
    qdrant_vector_size: int = 768
    qdrant_timeout_seconds: float = 10.0

    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.2
    eval_default_question_limit: int = 20
    eval_fast_question_limit: int = 8

    data_dir: Path = base_dir / "data"
    upload_dir: Path = data_dir / "uploads"
    eval_questions_path: Path = data_dir / "eval" / "eval_questions.json"
    sample_docs_dir: Path = data_dir / "sample_docs"

    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
