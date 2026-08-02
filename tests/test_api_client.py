from __future__ import annotations

import pytest

from src.api_client import PredictionApiClient


def test_embedded_client_loads_model_and_predicts(valid_profile: dict[str, float]) -> None:
    with PredictionApiClient(base_url=None) as client:
        model_info = client.get_object("/model_info")
        prediction = client.post_object("/predict", valid_profile)

    assert model_info["model_name"]
    assert 0 <= prediction["readiness_score"] <= 100
    assert prediction["best_domain"] in {"Web Dev", "Data", "DevOps", "Mobile"}


def test_embedded_client_supports_batch_predictions(valid_profile: dict[str, float]) -> None:
    with PredictionApiClient(base_url=None) as client:
        predictions = client.post_list("/batch_predict", {"students": [valid_profile]})

    assert len(predictions) == 1
    assert predictions[0]["readiness_level"] in {"Low", "Moderate", "High"}


def test_embedded_client_close_is_idempotent() -> None:
    client = PredictionApiClient(base_url=None)

    assert client.__enter__() is client
    client.__exit__(None, None, None)
    client.close()

    with pytest.raises(RuntimeError, match="already closed"):
        client.__enter__()
