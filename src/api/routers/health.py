"""
Health Router  GET /api/health
================================
Container health check endpoint for Docker and load balancers.
"""

from fastapi import APIRouter

from src.api.services.ml_scorer import is_model_loaded

router = APIRouter(prefix="/api", tags=["health"])


@router.get(
    "/health",
    summary="Service health check",
    description="Returns service health status including model readiness.",
)
async def health_check():
    """
    Check if the service is healthy and the ML model is loaded.
    
    Returns:
        dict: Health status with model_loaded flag
    """
    return {
        "status": "healthy",
        "model_loaded": is_model_loaded(),
        "service": "ai-return-risk-scorer",
        "version": "1.0.0",
    }

