from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from src.schemas import (
    BatchPredictionRequest,
    HealthResponse,
    HistoryRecord,
    ModelInfoResponse,
    PredictionResponse,
    SkillGapRequest,
    SkillGapResponse,
    StudentProfileRequest,
)
from src.student_career_willingness_predictor import StudentCareerWillingnessPredictor

MODEL_PATH = Path("models/student_career_willingness_model.pkl")

predictor = StudentCareerWillingnessPredictor()
model_loaded = False


def load_predictor() -> bool:
    global model_loaded
    try:
        predictor.load_model(str(MODEL_PATH))
    except (FileNotFoundError, KeyError, ValueError):
        model_loaded = False
    else:
        model_loaded = True
    return model_loaded


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    load_predictor()
    yield


app = FastAPI(
    title="Student Career Readiness Predictor API",
    version="1.0.0",
    lifespan=lifespan,
)


def _require_model() -> None:
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Prediction model is not loaded")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=model_loaded)


@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentProfileRequest) -> PredictionResponse:
    _require_model()
    result = predictor.predict_profile(student.model_dump())
    return PredictionResponse.model_validate(result)


@app.post("/batch_predict", response_model=list[PredictionResponse])
def batch_predict(request: BatchPredictionRequest) -> list[PredictionResponse]:
    _require_model()
    frame = pd.DataFrame([student.model_dump() for student in request.students])
    return [
        PredictionResponse.model_validate(predictor.predict_profile(row.to_dict()))
        for _, row in frame.iterrows()
    ]


@app.post("/skill_gap", response_model=SkillGapResponse)
def skill_gap(request: SkillGapRequest) -> SkillGapResponse:
    _require_model()
    result = predictor.skill_gap_analysis(
        request.student_profile.model_dump(),
        target_role=request.target_role,
    )
    recommendations = [f"Strengthen {skill} for {request.target_role}." for skill in result["missing"]]
    return SkillGapResponse(**result, recommendations=recommendations)


@app.get("/history", response_model=list[HistoryRecord])
def history(limit: int = Query(default=50, ge=1, le=100)) -> list[HistoryRecord]:
    del limit
    return []


@app.get("/model_info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    _require_model()
    importance = predictor.get_feature_importance()
    records = [] if importance is None else importance.to_dict(orient="records")
    return ModelInfoResponse(
        model_name=predictor.model_name,
        residual_std=predictor.residual_std,
        cv_results=predictor.cv_results,
        dataset_counts=predictor.dataset_counts,
        feature_importance=records,
    )
