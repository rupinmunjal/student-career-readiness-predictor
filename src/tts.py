"""ElevenLabs text-to-speech narration for SHAP explanations.

This module is designed to fail soft: if the `elevenlabs` package isn't
installed or `ELEVENLABS_API_KEY` isn't set, `get_client()` returns `None`
and callers should surface a 503 rather than crash the app.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from src.schemas import NarrateExplanationRequest

if TYPE_CHECKING:
    from elevenlabs.client import ElevenLabs

logger = logging.getLogger("student_career_api")

MODEL_ID = "eleven_flash_v2_5"
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # "George" - calm, professional
OUTPUT_FORMAT = "mp3_44100_128"
AUDIO_DIR = Path("/tmp")  # noqa: S108 - intentional, ephemeral demo storage

_ElevenLabsClient: type[Any] | None
try:
    from elevenlabs.client import ElevenLabs as _ElevenLabsClient
except Exception:  # pragma: no cover - optional dependency
    _ElevenLabsClient = None


class TTSNotConfiguredError(RuntimeError):
    """Raised when the ElevenLabs client isn't available (missing key/package)."""


class TTSGenerationError(RuntimeError):
    """Raised when narration synthesis fails while calling the ElevenLabs API."""


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    return slug[:60] or "student"


def get_client() -> ElevenLabs | None:
    """Build an ElevenLabs client, or None if the feature isn't configured."""
    if _ElevenLabsClient is None:
        return None
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return None
    try:
        client = _ElevenLabsClient(api_key=api_key)
    except Exception:
        logger.warning("tts_client_init_failed", exc_info=True)
        return None
    return cast("ElevenLabs", client)


def build_narration_text(request: NarrateExplanationRequest) -> str:
    """Turn a SHAP explanation payload into a natural narration script."""
    name = request.student_name.strip()
    first_name = name.split()[0] if name else "there"

    sentences = [
        f"Hi {first_name}.",
        f"Your career readiness score is {request.readiness_score:.1f} out of 10.",
    ]

    if request.top_features:
        strengths = ", ".join(
            f"{item.feature} {'boosted' if item.impact >= 0 else 'reduced'} "
            f"your score by {abs(item.impact):.0f}%"
            for item in request.top_features
        )
        sentences.append(f"Your top strengths: {strengths}.")

    sentences.append(f"Based on your profile, your best fit track is {request.recommended_track}.")
    sentences.append(f"One quick win: {request.improvement_tip}.")
    sentences.append("Good luck with your search!")

    return " ".join(sentences)


def synthesize_explanation(request: NarrateExplanationRequest, narration_text: str) -> Path:
    """Generate narration audio via ElevenLabs and save it to a temp mp3 file."""
    client = get_client()
    if client is None:
        raise TTSNotConfiguredError(
            "Text-to-speech is not configured. Set ELEVENLABS_API_KEY to enable narration."
        )

    try:
        audio_stream = client.text_to_speech.convert(
            voice_id=VOICE_ID,
            model_id=MODEL_ID,
            output_format=OUTPUT_FORMAT,
            text=narration_text,
        )
        audio_bytes = b"".join(chunk for chunk in audio_stream if chunk)
    except Exception as exc:
        logger.warning("tts_generation_failed", exc_info=True)
        raise TTSGenerationError(f"Failed to generate narration audio: {exc}") from exc

    if not audio_bytes:
        raise TTSGenerationError("ElevenLabs returned no audio data.")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    output_path = AUDIO_DIR / f"explanation_{_slugify(request.student_name)}.mp3"
    try:
        output_path.write_bytes(audio_bytes)
    except OSError as exc:
        raise TTSGenerationError(f"Failed to save narration audio: {exc}") from exc

    return output_path
