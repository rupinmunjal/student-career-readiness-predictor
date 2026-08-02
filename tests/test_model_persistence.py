from __future__ import annotations

from pathlib import Path

import pytest

from src.student_career_willingness_predictor import StudentCareerWillingnessPredictor


def test_pickle_round_trip_is_deterministic(
    tmp_path: Path,
    valid_profile: dict[str, float],
) -> None:
    original = StudentCareerWillingnessPredictor()
    original.load_model("models/student_career_willingness_model.pkl")
    before = original.predict_profile(valid_profile)

    model_path = tmp_path / "round_trip.pkl"
    original.save_model(str(model_path))
    restored = StudentCareerWillingnessPredictor()
    restored.load_model(str(model_path))
    after = restored.predict_profile(valid_profile)

    assert after["readiness_score"] == pytest.approx(before["readiness_score"], abs=1e-12)
    assert after["domain_scores"] == before["domain_scores"]
