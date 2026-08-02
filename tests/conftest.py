from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api import app


@pytest.fixture
def valid_profile() -> dict[str, float]:
    return {
        "certification_course": 1.0,
        "tenth_mark": 78.0,
        "twelfth_mark": 76.0,
        "college_mark": 80.0,
        "daily_studying_time": 3.0,
        "likes_degree": 1.0,
        "travel_time": 1.0,
        "stress_level": 2.0,
        "part_time_job": 0.0,
        "github_activity_proxy": 6.0,
        "side_projects_count": 3.0,
        "internship_experience": 1.0,
        "tech_skills_self_rating": 7.0,
        "languages_known_count": 3.0,
    }


@pytest.fixture
def api_client() -> Iterator[TestClient]:
    with TestClient(app) as client:
        yield client
