from fastapi import APIRouter

from app.api.dependencies import get_evaluator
from app.models.schemas import EvaluationRequest, EvaluationResponse

router = APIRouter(tags=["evaluation"])


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate(payload: EvaluationRequest) -> EvaluationResponse:
    return get_evaluator().run(
        top_k=payload.top_k,
        max_questions=payload.max_questions,
        fast_mode=payload.fast_mode,
    )
