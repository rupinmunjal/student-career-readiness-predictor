from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src import tts as tts_module


def test_health_reports_loaded_model(api_client: TestClient) -> None:
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_readiness(api_client: TestClient, valid_profile: dict[str, float]) -> None:
    response = api_client.post("/predict", json=valid_profile)

    assert response.status_code == 200
    payload = response.json()
    assert 0 <= payload["readiness_score"] <= 100
    assert payload["best_domain"] in {"Web Dev", "Data", "DevOps", "Mobile"}


def test_predict_rejects_empty_and_invalid_payloads(api_client: TestClient) -> None:
    assert api_client.post("/predict", json={}).status_code == 422
    response = api_client.post(
        "/predict",
        json={"tenth_mark": 101, "twelfth_mark": 70, "college_mark": 80},
    )
    assert response.status_code == 422


def test_batch_predict_returns_one_result_per_student(
    api_client: TestClient,
    valid_profile: dict[str, float],
) -> None:
    history_before = api_client.get("/history?limit=100")
    metrics_before = api_client.get("/metrics")
    response = api_client.post("/batch_predict", json={"students": [valid_profile, valid_profile]})
    history_after = api_client.get("/history?limit=100")
    metrics_after = api_client.get("/metrics")

    assert response.status_code == 200
    assert len(response.json()) == 2
    previous_ids = {record["id"] for record in history_before.json()}
    current_ids = {record["id"] for record in history_after.json()}
    assert len(current_ids - previous_ids) == 2
    assert (
        metrics_after.json()["total_predictions"] == metrics_before.json()["total_predictions"] + 2
    )
    assert (
        metrics_after.json()["total_batch_requests"]
        == metrics_before.json()["total_batch_requests"] + 1
    )


def test_batch_size_limit(api_client: TestClient, valid_profile: dict[str, float]) -> None:
    allowed: dict[str, list[dict[str, Any]]] = {"students": [valid_profile] * 100}
    too_large: dict[str, list[dict[str, Any]]] = {"students": [valid_profile] * 101}

    assert api_client.post("/batch_predict", json=allowed).status_code == 200
    assert api_client.post("/batch_predict", json=too_large).status_code == 422


def test_batch_rejects_empty_payload(api_client: TestClient) -> None:
    assert api_client.post("/batch_predict", json={"students": []}).status_code == 422


def test_metrics_reports_request_activity(api_client: TestClient) -> None:
    api_client.get("/health")

    response = api_client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["average_response_time_ms"] >= 0


@pytest.fixture
def narrate_payload() -> dict[str, Any]:
    return {
        "student_name": "Alex",
        "readiness_score": 7.2,
        "top_features": [
            {"feature": "backend projects", "impact": 18.0},
            {"feature": "internship months", "impact": 12.0},
        ],
        "recommended_track": "DevOps",
        "improvement_tip": "Add an AWS certification",
    }


def test_narrate_explanation_returns_503_when_not_configured(
    api_client: TestClient,
    narrate_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    response = api_client.post("/narrate-explanation", json=narrate_payload)

    assert response.status_code == 503


def test_narrate_explanation_rejects_invalid_payload(api_client: TestClient) -> None:
    assert api_client.post("/narrate-explanation", json={}).status_code == 422


def test_narrate_explanation_returns_audio_when_configured(
    api_client: TestClient,
    narrate_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-key")
    monkeypatch.setattr(tts_module, "AUDIO_DIR", tmp_path)

    class _FakeTextToSpeech:
        def convert(self, **_: Any) -> list[bytes]:
            return [b"fake-mp3-bytes"]

    class _FakeClient:
        text_to_speech = _FakeTextToSpeech()

    monkeypatch.setattr(tts_module, "_ElevenLabsClient", lambda api_key: _FakeClient())

    response = api_client.post("/narrate-explanation", json=narrate_payload)

    assert response.status_code == 200
    payload = response.json()
    assert payload["audio_path"] == str(tmp_path / "explanation_Alex.mp3")
    assert "Alex" in payload["narration_text"]
    assert "DevOps" in payload["narration_text"]
    assert payload["audio_base64"]
    assert (tmp_path / "explanation_Alex.mp3").read_bytes() == b"fake-mp3-bytes"
