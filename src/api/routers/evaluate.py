from datetime import datetime

from fastapi import APIRouter, HTTPException

from src.api.schemas.transaction import TransactionPayload
from src.api.schemas.response import RiskResponse
from src.api.services.ml_scorer import (
    predict_with_shap, score_to_action, is_model_loaded, get_optimal_threshold
)
from src.api.services.rule_engine import evaluate_with_rules

router = APIRouter(prefix="/api", tags=["evaluation"])


@router.post("/evaluate", response_model=RiskResponse)
async def evaluate_transaction(payload: TransactionPayload) -> RiskResponse:
    # total_past_orders == 0 means truly new user → rule engine
    # any previous orders → ML model has enough signal
    if payload.total_past_orders == 0:
        risk_score, reasons = evaluate_with_rules(payload)
        routing = "rule_engine"
    else:
        if not is_model_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded yet")

        risk_score, reasons = predict_with_shap(payload)
        routing = "ml_pipeline"

    action = score_to_action(risk_score)

    return RiskResponse(
        risk_score=round(risk_score, 4),
        action=action,
        routing=routing,
        reasons=reasons,
        threshold_used=get_optimal_threshold(),
        timestamp=datetime.now(),
        model_version="rf_v1.0",
    )
