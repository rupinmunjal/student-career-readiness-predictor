from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import (  # noqa: E402
    CANONICAL_FEATURES,
    TARGET_COLUMN,
    prepare_training_dataset,
)

PROFILE_COUNT = 2_000
RANDOM_SEED = 42
BINARY_FEATURES = {
    "certification_course",
    "likes_degree",
    "part_time_job",
    "internship_experience",
}
INTEGER_FEATURES = {
    "side_projects_count",
    "languages_known_count",
}
FEATURE_BOUNDS: dict[str, tuple[float, float]] = {
    "certification_course": (0, 1),
    "tenth_mark": (0, 100),
    "twelfth_mark": (0, 100),
    "college_mark": (0, 100),
    "daily_studying_time": (0, 5),
    "likes_degree": (0, 1),
    "travel_time": (0, 6),
    "stress_level": (0, 3),
    "part_time_job": (0, 1),
    "github_activity_proxy": (0, 10),
    "side_projects_count": (0, 10),
    "internship_experience": (0, 1),
    "tech_skills_self_rating": (1, 10),
    "languages_known_count": (0, 10),
}


def _sample_feature(
    source: pd.Series,
    feature: str,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    mean = float(source.mean())
    std = float(source.std(ddof=0))
    if feature in BINARY_FEATURES:
        return rng.binomial(1, np.clip(mean, 0, 1), PROFILE_COUNT).astype(float)
    values = rng.normal(mean, max(std, 0.01), PROFILE_COUNT)
    lower, upper = FEATURE_BOUNDS[feature]
    values = np.clip(values, lower, upper)
    if feature in INTEGER_FEATURES:
        values = np.rint(values)
    return values


def _generate_skills(profiles: pd.DataFrame, rng: np.random.Generator) -> list[str]:
    generated: list[str] = []
    for rating in profiles["tech_skills_self_rating"]:
        react = rng.random() < min(0.15 + float(rating) / 25, 0.55)
        javascript = rng.random() < (0.90 if react else 0.35)
        python = rng.random() < min(0.20 + float(rating) / 20, 0.70)
        sql = python and rng.random() < 0.75
        docker = rng.random() < min(0.05 + float(rating) / 30, 0.40)
        skills = [
            name
            for name, selected in (
                ("JavaScript", javascript),
                ("React", react),
                ("Python", python),
                ("SQL", sql),
                ("Docker", docker),
            )
            if selected
        ]
        generated.append(",".join(skills))
    return generated


def generate_profiles(output_dir: Path = Path("data/synthetic")) -> tuple[Path, Path]:
    rng = np.random.default_rng(RANDOM_SEED)
    Faker.seed(RANDOM_SEED)
    fake = Faker()
    real_data = prepare_training_dataset().data

    profiles = pd.DataFrame(
        {
            feature: _sample_feature(real_data[feature], feature, rng)
            for feature in CANONICAL_FEATURES
        }
    )
    profiles.insert(0, "student_id", [fake.uuid4() for _ in range(PROFILE_COUNT)])
    profiles["tech_skills"] = _generate_skills(profiles, rng)

    signal = (
        profiles["college_mark"] * 0.45
        + profiles["tech_skills_self_rating"] * 3.0
        + profiles["github_activity_proxy"] * 1.5
        + profiles["internship_experience"] * 8.0
    )
    signal = (signal - signal.mean()) / max(float(signal.std(ddof=0)), 0.01)
    target_mean = float(real_data[TARGET_COLUMN].mean())
    target_std = float(real_data[TARGET_COLUMN].std(ddof=0))
    readiness = np.clip(target_mean + signal * target_std + rng.normal(0, 4, PROFILE_COUNT), 0, 100)
    outcomes = pd.DataFrame(
        {
            "student_id": profiles["student_id"],
            TARGET_COLUMN: readiness,
            "placement_status": np.where(readiness >= 70, "Placed", "Not Placed"),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_path = output_dir / "synthetic_students.csv"
    outcomes_path = output_dir / "synthetic_outcomes.csv"
    profiles.to_csv(profiles_path, index=False)
    outcomes.to_csv(outcomes_path, index=False)
    return profiles_path, outcomes_path


if __name__ == "__main__":
    students_path, outcomes_path = generate_profiles()
    print(f"Generated {PROFILE_COUNT} profiles: {students_path} and {outcomes_path}")
