from __future__ import annotations

import os
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL", "sqlite:///./student_career.db")
    if url.startswith("sqlite:///") and not url.startswith("sqlite+aiosqlite:///"):
        return url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
    return url


class Base(DeclarativeBase):
    pass


class StudentProfile(Base):
    __tablename__ = "student_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    academic_scores_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    engagement_metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    github_activity: Mapped[float] = mapped_column(Float, nullable=False)
    projects_count: Mapped[int] = mapped_column(Integer, nullable=False)
    internships_count: Mapped[int] = mapped_column(Integer, nullable=False)
    tech_skills_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    predictions: Mapped[list[PredictionRecord]] = relationship(
        back_populates="student", cascade="all, delete-orphan"
    )


class PredictionRecord(Base):
    __tablename__ = "prediction_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("student_profiles.id"), nullable=False)
    readiness_score: Mapped[float] = mapped_column(Float, nullable=False)
    domain_fit_json: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    shap_features_json: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False
    )
    student: Mapped[StudentProfile] = relationship(back_populates="predictions")


engine = create_async_engine(get_database_url())
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db() -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
