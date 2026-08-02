from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StudentProfileRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenth_mark: float = Field(ge=0, le=100)
    twelfth_mark: float = Field(ge=0, le=100)
    college_mark: float = Field(ge=0, le=100)
    certification_course: float = Field(default=0, ge=0, le=1)
    daily_studying_time: float = Field(default=2, ge=0, le=24)
    likes_degree: float = Field(default=1, ge=0, le=1)
    travel_time: float = Field(default=2, ge=0, le=24)
    stress_level: float = Field(default=2, ge=0, le=3)
    part_time_job: float = Field(default=0, ge=0, le=1)
    github_activity_proxy: float = Field(default=4, ge=0, le=10)
    side_projects_count: float = Field(default=1, ge=0, le=100)
    internship_experience: float = Field(default=0, ge=0, le=1)
    tech_skills_self_rating: float = Field(default=5, ge=1, le=10)
    languages_known_count: float = Field(default=2, ge=0, le=50)


class PredictionResponse(BaseModel):
    readiness_score: float
    readiness_level: Literal["Low", "Moderate", "High"]
    confidence_interval: list[float]
    best_domain: Literal["Web Dev", "Data", "DevOps", "Mobile"]
    domain_scores: dict[str, float]
    feature_values: dict[str, float]


class BatchPredictionRequest(BaseModel):
    students: list[StudentProfileRequest] = Field(min_length=1, max_length=100)


class SkillGapRequest(BaseModel):
    student_profile: StudentProfileRequest
    target_role: Literal["Web Dev", "Data", "DevOps", "Mobile"]


class SkillGapResponse(BaseModel):
    target_role: str
    met: list[str]
    missing: list[str]
    recommendations: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool


class HistoryRecord(BaseModel):
    id: int
    readiness_score: float
    readiness_level: str
    best_domain: str
    created_at: str


class MetricsResponse(BaseModel):
    total_predictions: int
    total_batch_requests: int
    average_response_time_ms: float
