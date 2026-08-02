from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data_pipeline import (
    CANONICAL_FEATURES,
    DOMAIN_COLUMNS,
    MODEL_FEATURES,
    TARGET_COLUMN,
    load_primary_dataset,
    prepare_training_dataset,
)


def test_csv_merging_preserves_both_dataset_sources(tmp_path: Path) -> None:
    primary = pd.read_csv("data/Student Attitude and Behavior.csv")
    placement = pd.read_csv("data/Placement_Data_Full_Class.csv")
    primary_path = tmp_path / "Student Attitude and Behavior.csv"
    placement_path = tmp_path / "Placement_Data_Full_Class.csv"
    primary.to_csv(primary_path, index=False)
    placement.to_csv(placement_path, index=False)

    result = prepare_training_dataset(str(primary_path), str(tmp_path))

    assert len(result.data) == 450
    assert result.dataset_counts == {
        "Student Attitude and Behavior": 235,
        "Campus Placement Records": 215,
    }


def test_canonical_feature_alignment() -> None:
    result = prepare_training_dataset()

    expected = set(MODEL_FEATURES + DOMAIN_COLUMNS + [TARGET_COLUMN, "source_dataset"])
    assert expected.issubset(result.data.columns)
    assert not result.data[MODEL_FEATURES].isna().any().any()


def test_unexpected_columns_are_ignored(tmp_path: Path) -> None:
    primary = pd.read_csv("data/Student Attitude and Behavior.csv")
    primary["unexpected_external_field"] = "ignored"
    path = tmp_path / "primary.csv"
    primary.to_csv(path, index=False)

    normalized = load_primary_dataset(path)

    assert "unexpected_external_field" not in normalized.columns
    assert set(MODEL_FEATURES).issubset(normalized.columns)


def test_synthetic_data_is_loaded_only_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    primary = pd.read_csv("data/Student Attitude and Behavior.csv")
    placement = pd.read_csv("data/Placement_Data_Full_Class.csv")
    primary_path = tmp_path / "Student Attitude and Behavior.csv"
    primary.to_csv(primary_path, index=False)
    placement.to_csv(tmp_path / "Placement_Data_Full_Class.csv", index=False)
    synthetic_dir = tmp_path / "synthetic"
    synthetic_dir.mkdir()
    profile = dict.fromkeys(CANONICAL_FEATURES, 1.0)
    pd.DataFrame([{"student_id": "synthetic-1", **profile}]).to_csv(
        synthetic_dir / "synthetic_students.csv", index=False
    )
    pd.DataFrame([{"student_id": "synthetic-1", TARGET_COLUMN: 75.0}]).to_csv(
        synthetic_dir / "synthetic_outcomes.csv", index=False
    )
    monkeypatch.setenv("USE_SYNTHETIC", "1")

    result = prepare_training_dataset(str(primary_path), str(tmp_path))

    assert len(result.data) == 451
    assert result.dataset_counts["Synthetic Student Profiles"] == 1
