"""The canonical progress payload and the guarded emitter.

The key set is a cross-surface contract (TUI, relay, desktop, native read it
through one adapter each): every key present on every update, ``None`` for
anything a provider did not supply. Pinning it here means a future field
addition fails loudly at the producer before a surface can ignore it silently.
"""

from __future__ import annotations

from typing import Any

from local_operator.artifacts.progress import emit_progress, progress_details

#: The frozen key set every update carries (design §6; Q7 wire-side split).
CANONICAL_KEYS = {
    "tool_name",
    "stage",
    "provider",
    "model",
    "elapsed_s",
    "num_images",
    "queue_position",
    "progress_fraction",
    "log_lines",
    "error",
    "error_type",
}


def test_progress_details_emits_the_full_canonical_key_set() -> None:
    details = progress_details(tool="generate_image", stage="queued")

    assert set(details) == CANONICAL_KEYS
    assert details["tool_name"] == "generate_image"
    assert details["stage"] == "queued"
    # Nulls are PRESENT, never absent — a value no provider supplied is None.
    assert details["provider"] is None
    assert details["progress_fraction"] is None
    assert details["log_lines"] is None


def test_progress_details_passes_every_field_through() -> None:
    details = progress_details(
        tool="generate_video",
        stage="in_progress",
        provider="alpha",
        model="m1",
        elapsed_s=14,
        num_images=2,
        queue_position=3,
        log_lines=[{"message": "hi", "timestamp": 1}],
        error="boom",
        error_type="upstream",
    )

    assert details == {
        "tool_name": "generate_video",
        "stage": "in_progress",
        "provider": "alpha",
        "model": "m1",
        "elapsed_s": 14,
        "num_images": 2,
        "queue_position": 3,
        "progress_fraction": None,
        "log_lines": [{"message": "hi", "timestamp": 1}],
        "error": "boom",
        "error_type": "upstream",
    }


def test_emit_progress_is_guarded_and_optional() -> None:
    seen: list[tuple[str, dict[str, Any]]] = []
    emit_progress(lambda text, details: seen.append((text, details)), "hello", stage="queued")
    assert seen == [("hello", {"stage": "queued"})]

    # A None emitter is a no-op; a raising emitter never propagates.
    emit_progress(None, "hello", stage="queued")

    def broken(text: str, details: dict[str, Any]) -> None:
        raise RuntimeError("consumer died")

    emit_progress(broken, "hello", stage="queued")
