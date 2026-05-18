import json
from pathlib import Path

from app.agent.rag_agent import RAGAgent
from app.agent.verifier import REFUSAL_MESSAGE
from app.config import get_settings
from app.models.schemas import (
    EvaluationCaseResult,
    EvaluationMetrics,
    EvaluationResponse,
)


class Evaluator:
    def __init__(self, rag_agent: RAGAgent) -> None:
        self.settings = get_settings()
        self.rag_agent = rag_agent

    def run(
        self,
        top_k: int | None = None,
        eval_path: Path | None = None,
        max_questions: int | None = None,
        fast_mode: bool = False,
    ) -> EvaluationResponse:
        path = eval_path or self.settings.eval_questions_path
        with path.open("r", encoding="utf-8") as file:
            cases = json.load(file)
        if max_questions is None:
            max_questions = (
                self.settings.eval_fast_question_limit
                if fast_mode
                else self.settings.eval_default_question_limit
            )
        cases = cases[:max_questions]

        model_override = self.settings.ollama_eval_model or None
        num_predict = (
            self.settings.ollama_eval_fast_num_predict
            if fast_mode
            else self.settings.ollama_eval_num_predict
        )

        results: list[EvaluationCaseResult] = []
        for case in cases:
            response = self.rag_agent.run(
                case["question"],
                top_k=top_k,
                model_override=model_override,
                num_predict=num_predict,
            )
            expected_source = case.get("expected_source_document")
            cited_documents = {citation.filename for citation in response.citations}
            retrieved_documents = {chunk.filename for chunk in response.retrieved_chunks}
            results.append(
                EvaluationCaseResult(
                    question=case["question"],
                    expected_behavior=case["expected_behavior"],
                    expected_source_document=expected_source,
                    actual_answer=response.answer,
                    refusal=response.refusal,
                    citation_present=bool(response.citations),
                    retrieval_hit=(expected_source in retrieved_documents)
                    if expected_source
                    else response.refusal,
                    grounded=response.answer == REFUSAL_MESSAGE or bool(cited_documents),
                    metadata={
                        "retrieved_documents": sorted(retrieved_documents),
                        "cited_documents": sorted(cited_documents),
                    },
                )
            )

        total = len(results) or 1
        metrics = EvaluationMetrics(
            total_questions=len(results),
            retrieval_hit_rate=sum(result.retrieval_hit for result in results) / total,
            citation_presence_rate=sum(result.citation_present for result in results) / total,
            refusal_accuracy=sum(
                (
                    result.refusal
                    if result.expected_behavior == "refuse"
                    else not result.refusal
                )
                for result in results
            )
            / total,
            groundedness_rate=sum(result.grounded for result in results) / total,
        )
        return EvaluationResponse(metrics=metrics, results=results)
