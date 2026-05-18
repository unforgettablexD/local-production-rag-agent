import re

from app.agent.tools import AgentTools
from app.agent.verifier import REFUSAL_MESSAGE, build_citations
from app.config import get_settings
from app.models.schemas import ChatResponse, RetrievedChunk


class RAGAgent:
    def __init__(self, tools: AgentTools) -> None:
        self.settings = get_settings()
        self.tools = tools

    def run(
        self,
        question: str,
        top_k: int | None = None,
        model_override: str | None = None,
        num_predict: int | None = None,
    ) -> ChatResponse:
        needs_retrieval = self._needs_retrieval(question)
        rewritten_query = self._rewrite_query(question) if needs_retrieval else None
        retrieved_chunks: list[RetrievedChunk] = []
        answer = REFUSAL_MESSAGE
        citations = []

        if needs_retrieval:
            retrieved_chunks = self.tools.retrieve_context(
                rewritten_query or question, top_k=top_k or self.settings.retrieval_top_k
            )
            if retrieved_chunks:
                answer, citations = self.tools.answer_with_context(
                    question,
                    retrieved_chunks,
                    model_override=model_override,
                    num_predict=num_predict,
                )

        grounded = self.tools.evaluate_answer_grounding(answer, retrieved_chunks)
        refusal = answer.strip() == REFUSAL_MESSAGE or not grounded
        if refusal:
            answer = REFUSAL_MESSAGE
            citations = []
        elif not citations and retrieved_chunks:
            citations = build_citations(answer, retrieved_chunks)

        confidence = self._estimate_confidence(retrieved_chunks, len(citations), grounded, refusal)
        return ChatResponse(
            answer=answer,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            confidence=confidence,
            needs_retrieval=needs_retrieval,
            rewritten_query=rewritten_query,
            refusal=refusal,
        )

    def _needs_retrieval(self, question: str) -> bool:
        normalized = question.strip().lower()
        if normalized in {"hi", "hello", "hey"}:
            return False
        greeting_match = re.fullmatch(r"(hi|hello|hey)[!. ]*", normalized)
        return greeting_match is None

    def _rewrite_query(self, question: str) -> str:
        normalized = re.sub(r"\s+", " ", question).strip()
        return normalized.replace("policy", "policy guidance").replace("doc", "document")

    @staticmethod
    def _estimate_confidence(
        retrieved_chunks: list[RetrievedChunk], citation_count: int, grounded: bool, refusal: bool
    ) -> float:
        if refusal or not retrieved_chunks:
            return 0.2
        average_score = sum(chunk.score for chunk in retrieved_chunks) / len(retrieved_chunks)
        citation_bonus = min(0.2, citation_count * 0.05)
        grounded_bonus = 0.2 if grounded else 0.0
        return max(0.0, min(1.0, average_score + citation_bonus + grounded_bonus))
