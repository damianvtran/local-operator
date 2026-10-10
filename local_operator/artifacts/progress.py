"""The canonical live-progress payload every artifact tool emits.

One producer (``progress_details``) and one guarded emitter
(``emit_progress``): the payload's key set is a CONTRACT the four image
surfaces already read (TUI, relay, desktop, native), and the guards here are
what keep a broken consumer from ever taking a generation down. The emitted
key set is pinned by tests so any future field addition fails loudly here
first, before a surface can ignore it silently.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: The tool's live-progress callback: one bounded line plus a JSON-safe mapping.
ProgressFn = Callable[[str, dict[str, Any]], None]

__all__ = ["ProgressFn", "emit_progress", "progress_details"]


def emit_progress(emit: ProgressFn | None, text: str, **details: Any) -> None:
    """One progress line; a broken emitter must never break a generation.

    Public beside the private helpers because the walk's failure updates
    (another module) emit through it — one guarded spelling for "progress is
    presentation, never control flow". The tool's own terminal updates route
    through ITS guarded emitter (``image_tool._progress_emitter``'s closure),
    which carries the same contract.
    """
    if emit is None:
        return
    try:
        emit(text, details)
    except Exception:  # noqa: BLE001 - progress is presentation, never control flow
        logger.debug("artifact progress emitter raised; continuing", exc_info=True)


def progress_details(
    *,
    tool: str,
    stage: str | None,
    provider: str | None = None,
    model: str | None = None,
    elapsed_s: int | None = None,
    num_images: int | None = None,
    queue_position: int | None = None,
    log_lines: list[dict[str, Any]] | None = None,
    error: str | None = None,
    error_type: str | None = None,
) -> dict[str, Any]:
    """The canonical payload every artifact-tool update carries.

    The canonical field set — ``stage``, ``queue_position``,
    ``progress_fraction``, ``log_lines``, ``error``, ``error_type`` — is
    emitted by the artifact lane; the surfaces align their adapters afterwards
    (Q7 wire-side split, manager scope 2026-10-09). Every key is PRESENT on
    every update; a value no provider supplied is ``None``, never a synthesized
    stand-in. ``tool`` names the emitting tool (the frozen ``tool_name`` key
    keeps its historical slot — surfaces gate detection on it). Constraints,
    each from what the rungs actually receive:

    - ``stage`` vocabulary: ``queued`` / ``in_progress`` / ``completed`` /
      ``cancelled`` / ``cancelling`` (the cancel-confirmation hold), and
      ``None`` on a mid-walk failure update whose semantics ride
      ``error``/``error_type`` instead.
    - ``progress_fraction`` stays ``None`` until a provider reports one:
      neither the hub's media route nor FAL's queue status carries a fraction
      today, and elapsed-vs-budget is a TIMEOUT, not progress — it is
      deliberately never synthesized into a bar.
    - ``log_lines`` is the provider's own ``logs`` list passed through
      verbatim (``[{message, timestamp}]``), ``None`` where the payload
      carried none.
    - ``error``/``error_type`` are the platform's sentence and the structured
      code beside it; on a rung failure they carry the SAME classification as
      that attempt's ``reason_class`` so the two can never disagree.
    """
    return {
        "tool_name": tool,
        "stage": stage,
        "provider": provider,
        "model": model,
        "elapsed_s": elapsed_s,
        "num_images": num_images,
        "queue_position": queue_position,
        "progress_fraction": None,
        "log_lines": log_lines,
        "error": error,
        "error_type": error_type,
    }
