"""Failure evidence for the armed-probe cell, and ONLY for it.

WHY THIS EXISTS: ``test_the_armed_probe_drains_and_exits_when_the_loaded_tree_vanishes``
(``tests/e2e/test_runtime_refresh_e2e.py``) went red on CI (run 42826633627,
job 129873496093) at "the latch must land while the turn is STILL running",
and its report said only that both completion rows were ``complete``. A local
diagnostic run then showed the evidence that report drops: the turn's ``bash``
tool result was a SYNTHETIC abort (``__fault: aborted``, ``__synthetic: true``,
~0.01 s) — a row ``_completion_attention`` never reads, because it keeps
``completion_attention`` custom rows only. The child also ran with
``stderr=DEVNULL``, so anything it printed on the way down was lost.

WHAT IT DOES AND DOES NOT DO: it reads what the test can ALREADY see — the
session's durable transcript and a stderr file the test owns — and turns it
into a bounded, credential-scrubbed paragraph appended to assertion MESSAGES.
It does not name who aborted the tool call (nothing test-visible records the
caller today), it adds no import, hook or patch to the child, and it changes no
assertion, deadline or success path: a pytest assertion message is evaluated
only when the assertion fails, so a green run never calls into this module.

WHEN TO REMOVE: once the armed-probe failure is root-caused and fixed, fold
whatever proved useful into the cell's own report or delete this module.

Pure on purpose (stdlib + the product's scrubber): the unit contract in
``tests/unit/test_armed_probe_diagnostics.py`` imports it without booting a
runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from local_operator.redaction_shapes import REDACTION_MARKER, is_credential_name, scrub_secrets

#: Per-field and whole-report bounds. A CI log line, not an archive: the row
#: count bounds a runaway transcript, the excerpt bounds one huge tool output,
#: the stderr tail keeps the END of the child's output (where a dying process
#: writes its traceback) without pasting a whole boot log into the failure.
MAX_FAULT_ROWS = 20
MAX_EXCERPT_CHARS = 600
MAX_STDERR_BYTES = 4000


def _redact(value: Any) -> Any:
    """Scrub credential SHAPES from strings and mask credential-NAMED keys.

    The shape scrubber catches a token spelled like one; masking by key name
    catches the opaque value a ``token``/``secret`` field carries without any
    shape around it (the completion rows carry a ``token`` field).
    """
    if isinstance(value, str):
        return scrub_secrets(value)
    if isinstance(value, dict):
        return {
            key: (
                REDACTION_MARKER
                if isinstance(key, str) and is_credential_name(key) and item not in ("", None)
                else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _bounded(text: str, limit: int = MAX_EXCERPT_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f"… [{len(text) - limit} chars cut]"


def _rows(transcript: Path) -> list[dict[str, Any]]:
    try:
        durable = transcript.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    rows: list[dict[str, Any]] = []
    for line in durable.splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            # Same race as ``_completion_attention``: a half-written last line.
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def tool_fault_rows(transcript: Path) -> list[dict[str, Any]]:
    """Tool-result rows that ERRORED or were SYNTHESISED, with their metadata.

    These are the rows that vanished from the original report: an aborted tool
    call is recorded as a ``message``/``role: tool`` row, not as a completion
    row. Only what the transcript itself carries is reported — envelope
    ``id``/``ts``, the tool call identity, ``is_error``, the provider details
    (``__fault``/``__synthetic`` and anything else the runtime put there) and
    ``duration_s`` — redacted and bounded.
    """
    out: list[dict[str, Any]] = []
    for row in _rows(transcript):
        payload = row.get("payload")
        if row.get("type") != "message" or not isinstance(payload, dict):
            continue
        if payload.get("role") != "tool":
            continue
        provider = payload.get("provider_payload")
        provider = provider if isinstance(provider, dict) else {}
        details = provider.get("details")
        details = details if isinstance(details, dict) else {}
        if not (payload.get("is_error") or "__fault" in details or details.get("__synthetic")):
            continue
        out.append(
            {
                "id": row.get("id"),
                "ts": row.get("ts"),
                "tool_name": payload.get("tool_name"),
                "tool_call_id": payload.get("tool_call_id"),
                "is_error": payload.get("is_error"),
                "details": _redact(details),
                "duration_s": provider.get("duration_s"),
                "content": _bounded(scrub_secrets(json.dumps(payload.get("content"), default=str))),
            }
        )
    return out


def stderr_tail(path: Path | None, limit: int = MAX_STDERR_BYTES) -> str:
    """The last ``limit`` bytes of the child's stderr file, scrubbed."""
    if path is None:
        return "(no stderr file)"
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > limit:
                handle.seek(size - limit)
            raw = handle.read()
    except OSError as error:
        return f"(stderr unreadable: {error!r})"
    if not raw:
        return "(empty)"
    prefix = f"[first {size - limit} bytes cut] " if size > limit else ""
    return prefix + scrub_secrets(raw.decode("utf-8", errors="replace"))


def evidence(transcript: Path, stderr_path: Path | None) -> str:
    """The extra paragraph every armed-probe failure message carries."""
    faults = tool_fault_rows(transcript)
    shown = faults[-MAX_FAULT_ROWS:]
    cut = len(faults) - len(shown)
    lines = [
        "",
        "ARMED-PROBE DIAGNOSTICS (failure-only evidence; does not name the abort's caller):",
        f"errored/synthetic tool-result rows ({len(faults)}"
        + (f", oldest {cut} omitted" if cut else "")
        + "):",
        *(json.dumps(row, default=str) for row in shown),
        "child stderr tail:",
        stderr_tail(stderr_path),
    ]
    return "\n".join(lines)


def redacted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The completion rows as a REPORT shows them (the assertion keeps the raw ones)."""
    return _redact(rows)
