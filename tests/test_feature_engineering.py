from __future__ import annotations

import pandas as pd
import pytest

from src.data_pipeline import DOMAIN_COLUMNS, normalize_input_frame


def test_derived_scores_preserve_explicit_zero_values() -> None:
    frame = pd.DataFrame(
        [
            {
                "tenth_mark": 0,
                "twelfth_mark": 0,
                "college_mark": 0,
                "github_activity_proxy": 0,
                "side_projects_count": 0,
                "internship_experience": 0,
                "tech_skills_self_rating": 0,
                "daily_studying_time": 0,
                "likes_degree": 0,
                "travel_time": 0,
                "stress_level": 0,
                "languages_known_count": 0,
            }
        ]
    )

    result = normalize_input_frame(frame).iloc[0]

    assert result["academic_relevance_score"] == 0
    assert result["experience_relevance_score"] == 0
    assert result["engagement_relevance_score"] == 0
    assert all(result[column] == 0 for column in DOMAIN_COLUMNS)


def test_missing_optional_fields_use_documented_defaults() -> None:
    result = normalize_input_frame(
        pd.DataFrame([{"tenth_mark": 80, "twelfth_mark": 70, "college_mark": 90}])
    ).iloc[0]

    assert result["academic_relevance_score"] == pytest.approx(82.0)
    assert result["experience_relevance_score"] == pytest.approx(5.9)
    assert result["engagement_relevance_score"] == pytest.approx(12.0)


def test_domain_scores_follow_feature_formulas() -> None:
    result = normalize_input_frame(
        pd.DataFrame(
            [
                {
                    "college_mark": 50,
                    "daily_studying_time": 2,
                    "github_activity_proxy": 2,
                    "side_projects_count": 1,
                    "internship_experience": 1,
                    "tech_skills_self_rating": 4,
                    "languages_known_count": 3,
                }
            ]
        )
    ).iloc[0]

    assert result["domain_web_dev"] == pytest.approx(58)
    assert result["domain_data"] == pytest.approx(80)
    assert result["domain_devops"] == pytest.approx(46)
    assert result["domain_mobile"] == pytest.approx(71)
