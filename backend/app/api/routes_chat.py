from fastapi import APIRouter

from app.api.dependencies import get_rag_agent
from app.models.schemas import ChatRequest, ChatResponse

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    return get_rag_agent().run(payload.question, top_k=payload.top_k)
