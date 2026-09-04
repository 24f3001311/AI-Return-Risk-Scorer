from contextlib import asynccontextmanager
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.routers import evaluate, health
from src.api.services.ml_scorer import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up, loading model...")
    load_model()
    print("Ready!")
    yield
    print("Shutting down")


app = FastAPI(
    title="AI Return-Risk Scorer",
    description="Scores e-commerce transactions for return/fraud risk using ML + rule engine",
    version="1.0.0",
    lifespan=lifespan,
)

# allow the vue frontend to call the api
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(evaluate.router)
app.include_router(health.router)


@app.get("/api")
async def root():
    return {
        "service": "AI Return-Risk Scorer",
        "version": "1.0.0",
        "docs": "/docs",
    }


# serve frontend build if it exists (for docker deployment)
frontend_dist = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist')
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
else:
    @app.get("/")
    async def fallback():
        return {"message": "Run npm run build in src/frontend or use docker-compose"}
