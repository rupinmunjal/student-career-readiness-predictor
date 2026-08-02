from __future__ import annotations

from types import TracebackType
from typing import Any, Protocol, cast

import requests
from fastapi.testclient import TestClient


class HttpResponse(Protocol):
    def raise_for_status(self) -> None: ...

    def json(self) -> Any: ...


class HttpClient(Protocol):
    def get(self, url: str, *, timeout: float) -> HttpResponse: ...

    def post(self, url: str, *, json: object, timeout: float) -> HttpResponse: ...


class PredictionApiClient:
    """Call a remote prediction API or the same FastAPI app in-process."""

    def __init__(self, base_url: str | None) -> None:
        self._embedded_client: TestClient | None = None
        if base_url:
            self._base_url = base_url.rstrip("/")
            self._client = cast(HttpClient, requests.Session())
        else:
            from api import app

            embedded_client = TestClient(app)
            embedded_client.__enter__()
            self._embedded_client = embedded_client
            self._client = cast(HttpClient, embedded_client)
            self._base_url = ""

    def close(self) -> None:
        if self._embedded_client is not None:
            self._embedded_client.__exit__(None, None, None)
            self._embedded_client = None

    def __enter__(self) -> PredictionApiClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def get_object(self, path: str, *, timeout: float = 10) -> dict[str, Any]:
        response = self._client.get(self._url(path), timeout=timeout)
        response.raise_for_status()
        return self._object_payload(response)

    def post_object(
        self,
        path: str,
        payload: object,
        *,
        timeout: float = 15,
    ) -> dict[str, Any]:
        response = self._client.post(self._url(path), json=payload, timeout=timeout)
        response.raise_for_status()
        return self._object_payload(response)

    def post_list(
        self,
        path: str,
        payload: object,
        *,
        timeout: float = 30,
    ) -> list[dict[str, Any]]:
        response = self._client.post(self._url(path), json=payload, timeout=timeout)
        response.raise_for_status()
        data: object = response.json()
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise TypeError("Prediction API returned an invalid list response")
        return cast(list[dict[str, Any]], data)

    def _url(self, path: str) -> str:
        normalized_path = path if path.startswith("/") else f"/{path}"
        return f"{self._base_url}{normalized_path}"

    @staticmethod
    def _object_payload(response: HttpResponse) -> dict[str, Any]:
        data: object = response.json()
        if not isinstance(data, dict):
            raise TypeError("Prediction API returned an invalid object response")
        return cast(dict[str, Any], data)
