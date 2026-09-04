
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class RiskReason(BaseModel):
    """A single factor contributing to the risk score."""
    
    feature: str = Field(
        ...,
        description="Name of the contributing feature",
        examples=["transaction_velocity_24h"],
    )
    
    value: float = Field(
        ...,
        description="The actual value of this feature for the transaction",
        examples=[5.0],
    )
    
    contribution: float = Field(
        ...,
        description="How much this feature contributed to the risk score (SHAP value or rule weight)",
        examples=[0.15],
    )
    
    description: str = Field(
        ...,
        description="Human-readable explanation of why this factor matters",
        examples=["Transaction velocity in last 24h is 300% higher than average"],
    )


class RiskResponse(BaseModel):
    """
    Structured risk evaluation response.
    
    Returned by POST /api/evaluate. Contains the risk score,
    recommended action, routing path, and explainability reasons.
    """
    
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated return-risk score (0.0 = safe, 1.0 = high risk)",
        examples=[0.73],
    )
    
    action: Literal["ALLOW", "VERIFY_OTP", "MANDATE_PREPAID", "BLOCK"] = Field(
        ...,
        description="Recommended action for the merchant",
        examples=["VERIFY_OTP"],
    )
    
    routing: Literal["ml_pipeline", "rule_engine"] = Field(
        ...,
        description="Which engine evaluated this transaction (ML for returning users, Rule for new users)",
        examples=["ml_pipeline"],
    )
    
    reasons: list[RiskReason] = Field(
        ...,
        description="Top contributing factors explaining the risk score",
    )
    
    threshold_used: float = Field(
        ...,
        description="The FPC-optimal decision threshold used",
        examples=[0.42],
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the evaluation",
    )
    
    model_version: str = Field(
        default="rf_v1.0",
        description="Version identifier of the scoring model",
    )

