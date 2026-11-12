"""Model-facing text for the music tools and the music job's completion message.

WHY THIS EXISTS. A music ToolResult carries two audiences. ``details`` is the UI
contract (cards, players, the spend rail) and the model never reads it; the
model reads only ``content`` and the job manager's ``result_text``/``error_text``
(the "background job ... completed: ..." delivery and ``wait``'s output). Before
this module those carried one-line labels ("Music", "Music ready — job …"), so
the agent could not tell the user the tariff, what a request would cost, what a
finished job was charged, or which attachment holds the track (installed-runtime
QA of v0.101.48, finding Q1). Every sentence here is derived from the same
server payloads the UI renders — the ``/tools/music/models`` capability row and
each track's versioned ``cost`` object — so the two audiences never disagree.

CONSTRAINTS.
- Model-facing, so English by design (AGENTS.md "Internationalization": model-
  facing prompts are the legitimate exception). User-facing labels stay keyed in
  ``wire.music``; nothing here is rendered as UI chrome.
- Money authority is the server's ``cost`` object. A quote stays "estimated"
  until the server says ``billed``; an absent cost is "unknown", never "$0".
- Never echo signed URLs, bearer grants or keys: only IDs, display names and
  attachment digests (the durable identity ``artifact_details`` records).
- Compact: these strings re-enter the context window, so one line per track.
"""

from __future__ import annotations

from typing import Any

_MODEL_ID = "radient/music"


def money(amount_usd: float) -> str:
    """``$0.15`` / ``$0.0075`` / ``$0.07625``: two decimals minimum, micro-dollars kept.

    Music quotes are fractions of a cent per second, so rounding to cents would
    turn a 3 s quote ($0.0075) into "$0.01" — a figure the server never stated.
    Six decimals because the server rounds ``duration_ms × unit_price`` to the
    micro-dollar (docs/MUSIC.md "Money and cancellation"): a 30 500 ms track is
    $0.07625, and four decimals would report a billed $0.0763 nobody charged.
    """
    text = f"{amount_usd:.6f}".rstrip("0")
    whole, _, fraction = text.partition(".")
    return f"${whole}.{fraction.ljust(2, '0')}"


def _seconds(duration_ms: Any) -> str:
    if not isinstance(duration_ms, (int, float)):
        return "?"
    return f"{duration_ms / 1000:g} s"


def _model_row(capabilities: Any, model_id: str = _MODEL_ID) -> dict[str, Any]:
    rows = capabilities.get("models", []) if isinstance(capabilities, dict) else []
    return next((row for row in rows if row.get("id") == model_id), {})


def _unit_price_ms(row: dict[str, Any]) -> float | None:
    """USD per requested millisecond, only when the server prices in ms."""
    pricing = row.get("pricing")
    if not isinstance(pricing, dict) or pricing.get("unit") != "millisecond":
        return None
    price = pricing.get("unit_price_usd")
    return float(price) if isinstance(price, (int, float)) else None


def _tariff(row: dict[str, Any]) -> str:
    price = _unit_price_ms(row)
    if price is None:
        # The caller prefixes "price ", so this reads "price not reported…".
        return "not reported by the server"
    basis = str((row.get("pricing") or {}).get("basis") or "quote").replace("_", " ")
    version = (row.get("pricing") or {}).get("price_version")
    tail = f", {version}" if version else ""
    return (
        f"{money(price * 60000)}/min per variation ({money(price * 1000)}/s; "
        f"{basis} on requested duration{tail})"
    )


def models_text(capabilities: Any) -> str:
    """``music_status(action="models")``: availability, tariff, limits, operations."""
    rows = capabilities.get("models", []) if isinstance(capabilities, dict) else []
    # Read-only diagnostics should survive a malformed row the way job_text
    # does, rather than raising on ``row.get`` and hiding the good rows.
    rows = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    if not rows:
        return "Music models: none reported by the server."
    lines = ["Music models:"]
    for row in rows:
        enabled = bool(row.get("enabled"))
        state = "enabled" if enabled else f"disabled ({row.get('disabled_reason') or 'no reason'})"
        lines.append(
            f"- {row.get('id', '?')}: {state}; price {_tariff(row)}."
        )  # i18n: ignore model-facing tool result
        limits = row.get("duration_ms")
        formats = row.get("formats")
        # A bare string is one format; joining it would spell it out per letter.
        formats = [formats] if isinstance(formats, str) else formats
        formats = [str(f) for f in formats] if isinstance(formats, list) else []
        if isinstance(limits, dict):
            lines.append(  # i18n: ignore model-facing tool result
                f"  Duration {_seconds(limits.get('min'))}–{_seconds(limits.get('max'))}"
                + (f"; formats {', '.join(formats)}." if formats else ".")
            )
        operations = row.get("operations") or {}
        reasons = row.get("operation_disabled_reasons") or {}
        available = [name for name, on in operations.items() if on is True]
        unavailable = [
            f"{name} ({reasons.get(name) or 'unavailable'})"
            for name, on in operations.items()
            if on is not True
        ]
        lines.append(  # i18n: ignore model-facing tool result
            f"  Operations available: {', '.join(available) or 'none'}"
            + (f"; unavailable: {', '.join(unavailable)}." if unavailable else ".")
        )
        if row.get("attribution"):
            lines.append(
                f"  Attribution: {row['attribution']}"
            )  # i18n: ignore model-facing tool result
    return "\n".join(lines)


def started_text(job_id: str, params: dict[str, Any], capabilities: Any) -> str:
    """generate/modify start: job id, the fixed quote for this request, how to follow it."""
    row = _model_row(capabilities, str(params.get("model") or _MODEL_ID))
    duration_ms = params.get("duration_ms")
    variations = params.get("variations") or 1
    price = _unit_price_ms(row)
    plural = "s" if variations != 1 else ""
    if price is not None and isinstance(duration_ms, (int, float)):
        basis = str((row.get("pricing") or {}).get("basis") or "quote").replace("_", " ")
        estimate = (
            f"Estimated cost {money(price * duration_ms * variations)} "
            f"at {money(price * 60000)}/min per variation ({basis}); "
            "the server's billed cost is authoritative."
        )
    else:
        estimate = "Estimated cost unknown (the server reported no millisecond tariff)."
    return (
        f"Music job {job_id} started ({row.get('id') or params.get('model') or _MODEL_ID}, "
        f"{_seconds(duration_ms)} × {variations} variation{plural}). {estimate} "
        f'It runs in the background; use music_status(job_id="{job_id}") or '
        f'wait(job_id="{job_id}") for the tracks, costs and saved files.'
    )


def _cost_text(track: dict[str, Any]) -> tuple[str, float | None]:
    """The track's cost phrase plus its billed amount (None unless billed)."""
    cost = track.get("cost")
    if not isinstance(cost, dict) or not isinstance(cost.get("amount_usd"), (int, float)):
        return "cost unknown", None
    amount = float(cost["amount_usd"])
    if cost.get("basis") == "billed":
        return f"cost {money(amount)} billed", amount
    return f"cost {money(amount)} estimated (quote, not yet billed)", None


def _track_line(index: int, track: dict[str, Any], files: list[dict[str, Any]]) -> str:
    title = track.get("title")
    head = f"- Track {index}" + (f' "{title}"' if title else "")
    cost, _billed = _cost_text(track)
    if track.get("error_type"):
        return f"{head}: failed ({track['error_type']}); {cost}."
    if track.get("registration_error"):
        return (
            f"{head}: finished but its audio could not be saved locally "
            f"({track['registration_error']}); {cost}. music_status retries without "
            "buying it again."
        )
    if not track.get("settled"):
        status = str(track.get("status") or "pending").lower().replace("_", " ")
        return f"{head}: {status}; {cost}."
    if not files:
        return f"{head}: {str(track.get('status') or 'settled').lower()}; {cost}."
    saved = "; ".join(
        f"{row.get('name') or 'audio'} (attachment {row.get('digest')}"
        + (f", {row['duration_s']:g} s" if isinstance(row.get("duration_s"), (int, float)) else "")
        + ")"
        for row in files
    )
    return f"{head}: completed, saved as {saved}; {cost}."


def job_text(record: dict[str, Any]) -> str:
    """Status and completion text: per-track outcome, files, costs, total billed.

    Used for ``music_status(job_id=…)``, the job's completion ``result_text`` and
    its no-playable-results failure message, so ``wait``, the completion delivery
    and a status read all say the same thing.
    """
    job = record.get("job", "?")
    tracks = [track for track in record.get("tracks") or [] if isinstance(track, dict)]
    artifacts = [row for row in record.get("artifacts") or [] if isinstance(row, dict)]
    state = str(record.get("state") or "registered")
    words = {
        "registered": "queued",
        "running": "running",
        "cancel_requested": "cancellation requested",
        "interrupted": "interrupted",
        "completed": "completed",
        "failed": "failed",
    }
    saved = sum(
        1 for track in tracks if any(a.get("track_id") == track.get("id") for a in artifacts)
    )
    head = f"Music job {job}: {words.get(state, state)}"
    if tracks:
        head += f" ({len(tracks)} track{'s' if len(tracks) != 1 else ''}, {saved} saved)"
    lines = [head + "."]
    if not tracks and record.get("error"):
        lines.append(f"Error: {record['error']}.")  # i18n: ignore model-facing tool result
    billed_total = 0.0
    any_billed = False
    # A settled track with no usable cost object is server contract drift; it
    # must surface as "unknown" in the total rather than silently read as $0
    # (or understate a partly billed job). Pending tracks are counted apart:
    # their cost is unknown only because it is not settled yet.
    unknown = 0
    for index, track in enumerate(tracks, start=1):
        files = [row for row in artifacts if row.get("track_id") == track.get("id")]
        lines.append(_track_line(index, track, files))  # i18n: ignore model-facing tool result
        phrase, billed = _cost_text(track)
        if billed is not None:
            billed_total += billed
            any_billed = True
        elif track.get("settled") and phrase == "cost unknown":
            unknown += 1
    if tracks:
        pending = sum(1 for track in tracks if not track.get("settled"))
        if any_billed:
            total = f"Total billed: {money(billed_total)}"
        elif unknown:
            total = "Total billed: unknown"
        else:
            total = "Total billed: $0.00"
        notes = []
        if unknown and any_billed:
            notes.append(f"{unknown} track(s) with unknown cost")
        if pending:
            notes.append(f"{pending} track(s) not yet settled")
        so_far = " so far" if pending else ""
        lines.append(
            total + so_far + (f" ({'; '.join(notes)})." if notes else ".")
        )  # i18n: ignore model-facing tool result
    if record.get("cancellation_requested"):
        # Same promise as wire.music.cancel_requested and the UI: a request,
        # never a refund or a guaranteed upstream stop.
        if state in {"completed", "interrupted"} and saved:
            lines.append(  # i18n: ignore model-facing tool result
                "Cancellation was requested, but the composition had already started: "
                "it finished and was charged, and its audio is retained."
            )
        elif state in {"completed", "failed"}:
            lines.append(
                "Cancellation was requested; no audio was produced."
            )  # i18n: ignore model-facing tool result
        else:
            lines.append(  # i18n: ignore model-facing tool result
                "Cancellation was requested; running generation may still complete and be "
                "charged, and its results remain retrievable."
            )
    if record.get("attribution"):
        lines.append(
            f"Attribution: {record['attribution']}"
        )  # i18n: ignore model-facing tool result
    return "\n".join(lines)
