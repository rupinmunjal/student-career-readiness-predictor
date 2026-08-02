from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CANONICAL_FEATURES: list[str] = [
    "certification_course",
    "tenth_mark",
    "twelfth_mark",
    "college_mark",
    "daily_studying_time",
    "likes_degree",
    "travel_time",
    "stress_level",
    "part_time_job",
    "github_activity_proxy",
    "side_projects_count",
    "internship_experience",
    "tech_skills_self_rating",
    "languages_known_count",
]

DERIVED_FEATURES: list[str] = [
    "academic_relevance_score",
    "experience_relevance_score",
    "engagement_relevance_score",
]

MODEL_FEATURES: list[str] = CANONICAL_FEATURES + DERIVED_FEATURES

DOMAIN_COLUMNS: list[str] = ["domain_web_dev", "domain_data", "domain_devops", "domain_mobile"]

TARGET_COLUMN = "junior_readiness_score"

DEFAULT_FEATURE_VALUES: dict[str, float] = {
    "certification_course": 0.0,
    "tenth_mark": 70.0,
    "twelfth_mark": 70.0,
    "college_mark": 70.0,
    "daily_studying_time": 2.0,
    "likes_degree": 1.0,
    "travel_time": 2.0,
    "stress_level": 2.0,
    "part_time_job": 0.0,
    "github_activity_proxy": 4.0,
    "side_projects_count": 1.0,
    "internship_experience": 0.0,
    "tech_skills_self_rating": 5.0,
    "languages_known_count": 2.0,
}

STUDY_TIME_MAP = {
    "0 - 30 minute": 0.0,
    "30 - 60 minute": 1.0,
    "1 - 2 hour": 2.0,
    "2 - 3 hour": 3.0,
    "3 - 4 hour": 4.0,
    "more than 4 hour": 5.0,
}

TRAVEL_TIME_MAP = {
    "0 - 30 minutes": 0.0,
    "30 - 60 minutes": 1.0,
    "1 - 1.30 hour": 2.0,
    "1.30 - 2 hour": 3.0,
    "2 - 2.30 hour": 4.0,
    "2.30 - 3 hour": 5.0,
    "more than 3 hour": 6.0,
}

STRESS_LEVEL_MAP = {"bad": 0.0, "awful": 1.0, "good": 2.0, "fabulous": 3.0}

DATASET_SOURCES: list[dict[str, Any]] = [
    {
        "name": "Student Attitude and Behavior",
        "local_file": "Student Attitude and Behavior.csv",
        "source": "https://www.kaggle.com/datasets/sahilislam007/student-attitude-and-behavior",
        "notes": (
            "Primary dataset with self-reported career willingness (%). Contains academic "
            "performance, study habits, and stress metrics."
        ),
        "target_type": "self-reported",
        "rows_used": 235,
        "target_description": (
            "Student self-assessment of willingness to pursue career based on degree (0-100%)"
        ),
        "limitations": "Self-reported data may not correlate perfectly with actual job readiness",
    },
    {
        "name": "Campus Placement Records",
        "local_file": "Placement_Data_Full_Class.csv",
        "source": "https://www.kaggle.com/datasets/benroshan/factors-affecting-campus-placement",
        "notes": (
            "MBA student placement data with REAL outcomes. Contains academic scores, work "
            "experience, and actual placement results."
        ),
        "target_type": "real_outcome",
        "rows_used": 215,
        "target_description": "Actual placement status (Placed/Not Placed) with salary data",
        "limitations": "MBA students (different population), but provides real career outcomes",
        "new_signal": "Placement test scores, work experience, real salary outcomes",
    },
]


@dataclass
class PipelineResult:
    data: pd.DataFrame
    dataset_counts: dict[str, int]
    used_files: list[str]


def _to_yes_no_flag(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    val = str(value).strip().lower()
    if val in {"yes", "y", "1", "1.0", "true", "placed"}:
        return 1.0
    if val in {"no", "n", "0", "0.0", "false", "unplaced"}:
        return 0.0
    return np.nan


def _to_float(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    cleaned = str(value).replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([np.nan] * len(df), dtype=float)


def _safe_fill_defaults(df: pd.DataFrame, default_dict: dict[str, float]) -> pd.DataFrame:
    for col, default in default_dict.items():
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(default)
    return df


def _add_relevance_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["academic_relevance_score"] = (
        df["tenth_mark"].fillna(70.0) * 0.2
        + df["twelfth_mark"].fillna(70.0) * 0.3
        + df["college_mark"].fillna(70.0) * 0.5
    )
    df["experience_relevance_score"] = (
        df["github_activity_proxy"].fillna(4.0) * 0.4
        + df["side_projects_count"].fillna(1.0) * 0.3
        + df["internship_experience"].fillna(0.0) * 10.0
        + df["tech_skills_self_rating"].fillna(5.0) * 0.8
    )
    df["engagement_relevance_score"] = (
        df["daily_studying_time"].fillna(2.0) * 8.0
        + df["likes_degree"].fillna(1.0) * 12.0
        - df["travel_time"].fillna(2.0) * 3.0
        - df["stress_level"].fillna(2.0) * 5.0
    )
    return df


def _add_domain_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["domain_web_dev"] = (
        df["github_activity_proxy"].fillna(4.0) * 8.0
        + df["side_projects_count"].fillna(1.0) * 12.0
        + df["languages_known_count"].fillna(2.0) * 10.0
    )
    df["domain_data"] = (
        df["college_mark"].fillna(70.0) * 0.8
        + df["daily_studying_time"].fillna(2.0) * 6.0
        + df["tech_skills_self_rating"].fillna(5.0) * 7.0
    )
    df["domain_devops"] = (
        df["internship_experience"].fillna(0.0) * 20.0
        + df["github_activity_proxy"].fillna(4.0) * 9.0
        + df["side_projects_count"].fillna(1.0) * 8.0
    )
    df["domain_mobile"] = (
        df["side_projects_count"].fillna(1.0) * 14.0
        + df["languages_known_count"].fillna(2.0) * 11.0
        + df["tech_skills_self_rating"].fillna(5.0) * 6.0
    )
    return df


def _normalize_for_model(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    df["source_dataset"] = source
    df = _safe_fill_defaults(df, DEFAULT_FEATURE_VALUES)
    df = _add_relevance_scores(df)
    df = _add_domain_scores(df)
    return df


def load_primary_dataset(path: Path) -> pd.DataFrame:
    df = _normalize_column_names(pd.read_csv(path))
    drop_columns = [
        "Gender",
        "Department",
        "Height(CM)",
        "Weight(KG)",
        "hobbies",
        "salary expectation",
        "social medai & video",
        "Financial Status",
        "prefer to study in",
    ]
    for col in drop_columns:
        if col in df.columns:
            df = df.drop(columns=[col])

    out = pd.DataFrame()
    out["certification_course"] = _col(df, "Certification Course").map(_to_yes_no_flag)
    out["tenth_mark"] = _col(df, "10th Mark").map(_to_float)
    out["twelfth_mark"] = _col(df, "12th Mark").map(_to_float)
    out["college_mark"] = _col(df, "college mark").map(_to_float)
    out["daily_studying_time"] = (
        _col(df, "daily studing time").astype(str).str.strip().str.lower().map(STUDY_TIME_MAP)
    )
    out["likes_degree"] = _col(df, "Do you like your degree?").map(_to_yes_no_flag)
    out["travel_time"] = (
        _col(df, "Travelling Time ").astype(str).str.strip().str.lower().map(TRAVEL_TIME_MAP)
    )
    out["stress_level"] = (
        _col(df, "Stress Level ").astype(str).str.strip().str.lower().map(STRESS_LEVEL_MAP)
    )
    out["part_time_job"] = _col(df, "part-time job").map(_to_yes_no_flag)

    # Proxies for junior-level profile from existing data
    out["github_activity_proxy"] = (
        np.clip(out["daily_studying_time"].fillna(2.0), 0, 5) * 1.3
        + out["certification_course"].fillna(0.0) * 2.0
        + out["likes_degree"].fillna(0.0) * 1.5
    )
    out["side_projects_count"] = (
        out["certification_course"].fillna(0.0) * 2.0 + out["daily_studying_time"].fillna(1.0) * 0.7
    )
    out["internship_experience"] = (out["part_time_job"].fillna(0.0) > 0).astype(float)
    out["tech_skills_self_rating"] = np.clip(
        (
            out["college_mark"].fillna(65.0) * 0.06
            + out["certification_course"].fillna(0.0) * 1.8
            + out["likes_degree"].fillna(0.0) * 1.6
        ),
        1,
        10,
    )
    out["languages_known_count"] = np.clip(
        out["tech_skills_self_rating"] * 0.35 + out["certification_course"].fillna(0.0),
        1,
        8,
    )
    out[TARGET_COLUMN] = _col(df, "willingness to pursue a career based on their degree").map(
        _to_float
    )
    return _normalize_for_model(out, "Student Attitude and Behavior")


def load_campus_placement_dataset(path: Path) -> pd.DataFrame:
    df = _normalize_column_names(pd.read_csv(path))
    out = pd.DataFrame()
    out["certification_course"] = _col(df, "specialisation").map(
        lambda x: 1.0 if str(x).strip().lower() in {"mkt&fin", "mkt&hr"} else 0.0
    )
    out["tenth_mark"] = _col(df, "ssc_p").map(_to_float)
    out["twelfth_mark"] = _col(df, "hsc_p").map(_to_float)
    out["college_mark"] = _col(df, "degree_p").map(_to_float)
    out["daily_studying_time"] = np.clip(_col(df, "mba_p").map(_to_float) / 20.0, 0, 5)
    out["likes_degree"] = 1.0
    out["travel_time"] = 2.0
    out["stress_level"] = np.clip(3.0 - (_col(df, "etest_p").map(_to_float) / 33.3), 0, 3)
    out["part_time_job"] = _col(df, "workex").map(_to_yes_no_flag)
    out["github_activity_proxy"] = np.clip(_col(df, "etest_p").map(_to_float) / 10.0, 0, 10)
    out["side_projects_count"] = np.clip(_col(df, "degree_p").map(_to_float) / 20.0, 0, 10)
    out["internship_experience"] = _col(df, "workex").map(_to_yes_no_flag)
    out["tech_skills_self_rating"] = np.clip(_col(df, "mba_p").map(_to_float) / 10.0, 1, 10)
    out["languages_known_count"] = np.clip(
        out["tech_skills_self_rating"].fillna(5.0) * 0.4 + out["internship_experience"].fillna(0.0),
        1,
        8,
    )
    status = _col(df, "status").astype(str).str.strip().str.lower()
    out[TARGET_COLUMN] = np.where(status.eq("placed"), 85.0, 40.0)
    return _normalize_for_model(out, "Campus Placement Records")


def prepare_training_dataset(
    base_dataset_path: str = "data/Student Attitude and Behavior.csv",
    data_dir: str = "data",
) -> PipelineResult:
    base_path = Path(base_dataset_path)
    datasets: list[pd.DataFrame] = []
    dataset_counts: dict[str, int] = {}
    used_files: list[str] = []

    if not base_path.exists():
        raise FileNotFoundError(f"Base dataset not found: {base_path}")

    primary = load_primary_dataset(base_path)
    datasets.append(primary)
    dataset_counts["Student Attitude and Behavior"] = len(primary)
    used_files.append(str(base_path))

    placement_path = Path(data_dir) / "Placement_Data_Full_Class.csv"
    if placement_path.exists():
        placement = load_campus_placement_dataset(placement_path)
        datasets.append(placement)
        dataset_counts["Campus Placement Records"] = len(placement)
        used_files.append(str(placement_path))

    merged = pd.concat(datasets, ignore_index=True)

    if TARGET_COLUMN not in merged.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from all datasets")

    initial_count = len(merged)
    merged = merged.dropna(subset=[TARGET_COLUMN])
    dropped_count = initial_count - len(merged)
    if dropped_count > 0:
        print(f"Note: Dropped {dropped_count} rows with missing target values")

    merged = merged.fillna({col: DEFAULT_FEATURE_VALUES.get(col, 0.0) for col in MODEL_FEATURES})

    if len(merged) == 0:
        raise ValueError("No valid training data after preprocessing")

    merged = merged.reset_index(drop=True)

    return PipelineResult(data=merged, dataset_counts=dataset_counts, used_files=used_files)


def normalize_input_frame(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=MODEL_FEATURES + DOMAIN_COLUMNS)

    frame = df.copy()
    for col in MODEL_FEATURES:
        if col not in frame.columns:
            frame[col] = DEFAULT_FEATURE_VALUES.get(col, 0.0)

    frame = _safe_fill_defaults(frame, DEFAULT_FEATURE_VALUES)
    frame = _add_relevance_scores(frame)
    frame = _add_domain_scores(frame)

    return frame


def run_pipeline_and_save(output_path: str = "data/merged_student_career_data.csv") -> None:
    result = prepare_training_dataset()
    result.data.to_csv(output_path, index=False)
    print(f"✅ Saved merged dataset to {output_path}")


if __name__ == "__main__":
    run_pipeline_and_save()
