import re

from app.generation.ollama_client import OllamaClient
from app.generation.prompts import (
    ANSWER_SYSTEM_PROMPT,
    build_compare_prompt,
    build_grounded_answer_prompt,
    build_summary_prompt,
)
from app.models.schemas import Citation, RetrievedChunk
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import QdrantVectorStore
from app.agent.verifier import REFUSAL_MESSAGE, build_citations, evaluate_answer_grounding


class AgentTools:
    def __init__(
        self, retriever: Retriever, vector_store: QdrantVectorStore, ollama_client: OllamaClient
    ) -> None:
        self.retriever = retriever
        self.vector_store = vector_store
        self.ollama_client = ollama_client

    def retrieve_context(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return self.retriever.retrieve(query, top_k=top_k)

    def answer_with_context(
        self,
        question: str,
        context: list[RetrievedChunk],
        model_override: str | None = None,
        num_predict: int | None = None,
    ) -> tuple[str, list[Citation]]:
        prompt = build_grounded_answer_prompt(question, context)
        answer = self.ollama_client.generate(
            prompt=prompt,
            system=ANSWER_SYSTEM_PROMPT,
            model=model_override,
            num_predict=num_predict,
        )
        citations = build_citations(answer, context)
        if self._should_use_fallback(answer, citations) and context:
            fallback_answer = self._extractive_fallback_answer(question, context)
            if fallback_answer:
                answer = fallback_answer
                citations = build_citations(answer, context)
        return answer, citations

    @staticmethod
    def _should_use_fallback(answer: str, citations: list[Citation]) -> bool:
        normalized = answer.strip()
        if normalized == REFUSAL_MESSAGE:
            return True
        if not citations:
            return True
        analysis_markers = (
            "mention the same",
            "the relevant part",
            "the key sentence",
            "the context",
            "both mention",
        )
        return any(marker in normalized.lower() for marker in analysis_markers)

    @staticmethod
    def _extractive_fallback_answer(question: str, context: list[RetrievedChunk]) -> str | None:
        if not context:
            return None

        question_terms = AgentTools._tokenize(question)
        best_score = -1
        best_sentence: str | None = None
        best_chunk_index: int | None = None

        for chunk in context:
            for sentence in AgentTools._split_sentences(chunk.text):
                sentence_terms = AgentTools._tokenize(sentence)
                if not sentence_terms:
                    continue
                overlap = len(question_terms & sentence_terms)
                numeric_bonus = 1 if re.search(r"\b\d+\b", sentence) else 0
                score = overlap * 10 + numeric_bonus
                if score > best_score:
                    best_score = score
                    best_sentence = sentence
                    best_chunk_index = chunk.index

        if not best_sentence or best_chunk_index is None:
            return None

        cleaned_sentence = re.sub(r"\s+", " ", best_sentence).strip().rstrip(".")
        if "[" in cleaned_sentence and "]" in cleaned_sentence:
            return cleaned_sentence
        return f"{cleaned_sentence}. [{best_chunk_index}]"

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if part.strip()]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        stopwords = {
            "what",
            "when",
            "where",
            "which",
            "who",
            "whom",
            "whose",
            "why",
            "how",
            "many",
            "much",
            "does",
            "do",
            "did",
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "to",
            "of",
            "for",
            "in",
            "on",
            "with",
            "and",
            "or",
            "per",
            "within",
        }
        terms = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        return {term for term in terms if term not in stopwords and len(term) > 1}

    def summarize_document(self, document_id: str) -> tuple[str, list[Citation]]:
        chunks = self.vector_store.get_document_chunks(document_id)
        if not chunks:
            raise ValueError(f"Document '{document_id}' was not found.")
        prompt = build_summary_prompt(chunks[0].filename, chunks[:8])
        answer = self.ollama_client.generate(prompt=prompt, system=ANSWER_SYSTEM_PROMPT)
        citations = build_citations(answer, chunks[:8])
        return answer, citations

    def compare_documents(self, document_id_a: str, document_id_b: str) -> tuple[str, list[Citation]]:
        chunks_a = self.vector_store.get_document_chunks(document_id_a)
        chunks_b = self.vector_store.get_document_chunks(document_id_b)
        if not chunks_a or not chunks_b:
            raise ValueError("Both documents must exist before comparison.")
        offset_chunks_b = [
            chunk.model_copy(update={"index": chunk.index + len(chunks_a[:4])}) for chunk in chunks_b[:4]
        ]
        prompt = build_compare_prompt(
            chunks_a[0].filename,
            chunks_b[0].filename,
            chunks_a[:4],
            offset_chunks_b,
        )
        combined_chunks = chunks_a[:4] + offset_chunks_b
        answer = self.ollama_client.generate(prompt=prompt, system=ANSWER_SYSTEM_PROMPT)
        citations = build_citations(answer, combined_chunks)
        return answer, citations

    def evaluate_answer_grounding(self, answer: str, retrieved_chunks: list[RetrievedChunk]) -> bool:
        return evaluate_answer_grounding(answer, retrieved_chunks)
