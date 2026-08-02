from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import PredictionRecord, StudentProfile, get_db, init_db
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
    await init_db()
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
async def predict(
    student: StudentProfileRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> PredictionResponse:
    _require_model()
    result = predictor.predict_profile(student.model_dump())
    values = student.model_dump()
    student_record = StudentProfile(
        academic_scores_json={
            "tenth_mark": values["tenth_mark"],
            "twelfth_mark": values["twelfth_mark"],
            "college_mark": values["college_mark"],
        },
        engagement_metrics_json={
            key: values[key]
            for key in (
                "certification_course",
                "daily_studying_time",
                "likes_degree",
                "travel_time",
                "stress_level",
                "part_time_job",
            )
        },
        github_activity=values["github_activity_proxy"],
        projects_count=int(values["side_projects_count"]),
        internships_count=int(values["internship_experience"]),
        tech_skills_json={
            "self_rating": values["tech_skills_self_rating"],
            "languages_known_count": values["languages_known_count"],
        },
    )
    db.add(student_record)
    await db.flush()
    prediction_record = PredictionRecord(
        student_id=student_record.id,
        readiness_score=result["readiness_score"],
        domain_fit_json=result["domain_scores"],
        shap_features_json={},
        model_version=predictor.model_name or "unknown",
    )
    db.add(prediction_record)
    await db.commit()
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
    recommendations = [
        f"Strengthen {skill} for {request.target_role}." for skill in result["missing"]
    ]
    return SkillGapResponse(**result, recommendations=recommendations)


@app.get("/history", response_model=list[HistoryRecord])
async def history(
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(default=50, ge=1, le=100),
) -> list[HistoryRecord]:
    query = select(PredictionRecord).order_by(PredictionRecord.created_at.desc()).limit(limit)
    records = (await db.scalars(query)).all()
    history_records: list[HistoryRecord] = []
    for record in records:
        level = (
            "High"
            if record.readiness_score >= 70
            else "Moderate"
            if record.readiness_score >= 40
            else "Low"
        )
        best_domain = max(record.domain_fit_json, key=lambda name: record.domain_fit_json[name])
        history_records.append(
            HistoryRecord(
                id=record.id,
                readiness_score=record.readiness_score,
                readiness_level=level,
                best_domain=best_domain,
                created_at=record.created_at.isoformat(),
            )
        )
    return history_records


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
