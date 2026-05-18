from functools import lru_cache

from app.agent.rag_agent import RAGAgent
from app.agent.tools import AgentTools
from app.evaluation.evaluator import Evaluator
from app.generation.ollama_client import OllamaClient
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import QdrantVectorStore


@lru_cache
def get_ollama_client() -> OllamaClient:
    return OllamaClient()


@lru_cache
def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore()


@lru_cache
def get_embeddings() -> EmbeddingService:
    return EmbeddingService(get_ollama_client())


@lru_cache
def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline(get_embeddings(), get_vector_store())


@lru_cache
def get_retriever() -> Retriever:
    return Retriever(get_embeddings(), get_vector_store())


@lru_cache
def get_agent_tools() -> AgentTools:
    return AgentTools(get_retriever(), get_vector_store(), get_ollama_client())


@lru_cache
def get_rag_agent() -> RAGAgent:
    return RAGAgent(get_agent_tools())


@lru_cache
def get_evaluator() -> Evaluator:
    return Evaluator(get_rag_agent())
