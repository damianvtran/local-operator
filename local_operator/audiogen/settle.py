"""The bounded post-cancel settlement poll (audio contract, security round 1).

**Why this exists.** The settled cancel contract draws a hard line: a QUEUED
job can be stopped for certain and is never charged, but once the provider has
been called there is NO cancel API — the work continues and is charged if it
produces audio. The tool call that asked to stop has returned by then, so
something has to answer the question the receipt leaves open: *did it finish,
and what did it cost?* This module is that something.

**What it is NOT.** It is not a job system. The poll is one bounded
observation loop (every ~3 s, up to ~2 minutes) that the SESSION owns as a
background task and cancels on dispose (``ToolContext.spawn_background``): no
detached manager, no wake schedule, no restart durability. If the runtime dies
mid-poll the server's own settlement is still authoritative and the ledger row
simply is not written — the same audited gap a killed tool call leaves.

**Money rules.** The STATUS read alone carries a late charge (the contract's
own words): ``settled: true`` plus a billed cost is recorded as a ledger
revision — never the artifact. The artifact is a bonus: when the result read
and the sha256-verified download succeed it is registered through
``cache_media`` exactly like a first-class generation, so a paid clip does not
vanish because the user pressed stop. When nothing terminal is observed inside
the bound the row says "settlement not observed" and NO amount is invented.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from local_operator.audiogen.rungs import (
    SettlementFacts,
    _client_scope,
    fetch_radient_settlement,
)

logger = logging.getLogger(__name__)

#: The transcript row type for a post-cancel settlement. A bookkeeping row
#: (``transcript.BOOKKEEPING_CUSTOM_TYPES``) because it lands minutes after the
#: user left — it must not re-rank the session as freshly worked.
AUDIO_SETTLE_CUSTOM_TYPE = "audio_settle_v1"

#: Poll pacing and the hard bound. The contract asks for ~2-5 s and ~2 min; a
#: fixed interval sits inside that band and the deadline is the promise that no
#: watcher outlives the turn that spawned it by more than this.
SETTLE_POLL_INTERVAL_S = 3.0
SETTLE_DEADLINE_S = 120.0

#: The sentence a row carries when the bound expired and the job was still
#: running. It is deliberately not a charge and not a failure.
SETTLEMENT_NOT_OBSERVED = "settlement not observed"


@dataclass(frozen=True)
class SettleOutcome:
    """What the bounded poll observed, before any ledger or transcript write."""

    status: str
    terminal: bool
    cost_usd: float | None = None
    billing_basis: str | None = None
    usage_record_id: str | None = None
    units: float | None = None
    unit: str | None = None
    error_type: str = ""
    message: str = ""
    #: Set only when the late artifact was downloaded AND verified.
    digest: str | None = None
    duration_s: float | None = None
    content_type: str = ""
    size_bytes: int = 0
    asset_error: str = ""
    #: The verified bytes, when the late artifact was fetched. Opaque here:
    #: registration (``cache_media``) is the only consumer.
    asset: Any = None

    @property
    def settled_charge(self) -> bool:
        """Whether this observation is a charge the ledger must record."""
        return self.terminal and self.cost_usd is not None


async def observe_audio_settlement(
    *,
    base_url: str,
    credential: str,
    request_id: str,
    poll_interval_s: float = SETTLE_POLL_INTERVAL_S,
    deadline_s: float = SETTLE_DEADLINE_S,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    clock: Callable[[], float] = time.monotonic,
    client: httpx.AsyncClient | None = None,
) -> SettleOutcome:
    """Poll a Radient job until it is terminal or the bound expires.

    Read-only: status first, then (only when terminal and successful) the
    result read and the verified download. ``sleep`` is injected so tests can
    drive the loop without the wall clock (with ``clock``, its virtual twin —
    a test whose fake sleeper returns instantly must not spin for two real
    minutes); ``client`` is the test seam for
    ``httpx.MockTransport``. A transport blip is NOT a terminal answer — the
    loop keeps its bound and retries the next tick (the per-request helper
    already spends one bounded retry of its own).
    """
    started = clock()
    last_status = ""
    last: Any = None
    async with _client_scope(client) as http:
        while True:
            try:
                last = await fetch_radient_settlement(
                    http,
                    base_url=base_url,
                    credential=credential,
                    request_id=request_id,
                )
            except Exception as exc:  # noqa: BLE001 - an observation never raises out
                logger.debug("audio settlement poll failed", exc_info=True)
                last = None
                if clock() - started >= deadline_s:
                    return SettleOutcome(
                        status=last_status,
                        terminal=False,
                        message=str(exc).strip() or "settlement read failed",
                    )
            else:
                last_status = last.status
                if last.terminal:
                    return _outcome_from_facts(last)
            remaining = deadline_s - (clock() - started)
            if remaining <= 0:
                return SettleOutcome(
                    status=last_status or (last.status if last is not None else ""),
                    terminal=False,
                )
            await sleep(min(poll_interval_s, remaining))


def _outcome_from_facts(facts: SettlementFacts) -> SettleOutcome:
    """Narrow ``SettlementFacts`` into this module's result (keeps the import)."""
    return SettleOutcome(
        status=facts.status,
        terminal=True,
        cost_usd=facts.cost_usd,
        billing_basis=facts.billing_basis,
        usage_record_id=facts.usage_record_id,
        units=facts.units,
        unit=facts.unit,
        error_type=facts.error_type or "",
        message=facts.message,
        digest=None,
        duration_s=facts.asset.duration_s if facts.asset is not None else None,
        content_type=facts.asset.content_type if facts.asset is not None else "",
        size_bytes=len(facts.asset.data) if facts.asset is not None else 0,
        asset_error=facts.asset_error,
        asset=facts.asset,
    )


async def settle_cancelled_audio(
    payload: dict[str, Any],
    *,
    record_spend: Callable[[Any], None] | None,
    journal: Callable[[dict[str, Any]], Awaitable[None]] | None,
) -> None:
    """Observe a cancelled-but-running job and record what actually happened.

    Runs as a SESSION background task (spawned by the tool through
    ``ToolContext.spawn_background``). Every step fails soft: this is an
    after-the-fact record, and a failure here must never surface as an unhandled
    task exception or undo the cancel receipt the user already has.

    The payload is the tool's frozen handover (see ``audio_tool``): the request
    id, what the call would have cost in units, and the deterministic ledger
    record id the cancel row used, so a settled charge lands as a REVISION of
    that row rather than a second charge-shaped row.
    """
    request_id = str(payload.get("request_id") or "")
    base_url = str(payload.get("base_url") or "")
    credential = str(payload.get("credential") or "")
    if not request_id or not base_url or not credential:
        return
    try:
        outcome = await observe_audio_settlement(
            base_url=base_url, credential=credential, request_id=request_id
        )
    except Exception:  # noqa: BLE001 - the observer itself never raises
        logger.debug("audio settlement observation failed", exc_info=True)
        return

    recorded = False
    if outcome.settled_charge and record_spend is not None:
        try:
            record_spend(_settled_record(payload, outcome))
            recorded = True
        except Exception:  # noqa: BLE001 - the ledger recorder never fails a turn
            logger.debug("audio settlement ledger row failed", exc_info=True)

    # The artifact is a bonus on top of the money: a settled clip registers
    # only when the observation returned sha256-verified bytes, and the ONE
    # registration path (``cache_media``) is used either way.
    prompt = str(payload.get("prompt") or "") or None
    digest, artifact_name = _register_asset(outcome, prompt)

    if journal is not None:
        row = {
            "version": 1,
            "tool": str(payload.get("tool") or "generate_voiceover"),
            "request_id": request_id,
            "status": outcome.status or SETTLEMENT_NOT_OBSERVED,
            "terminal": outcome.terminal,
            "settlement_observed": outcome.terminal,
            "cost_usd": outcome.cost_usd,
            "billing_basis": outcome.billing_basis,
            "usage_record_id": outcome.usage_record_id or None,
            "recorded": recorded,
            "error_type": outcome.error_type or None,
            "message": outcome.message or None,
            "asset_error": outcome.asset_error or None,
            "digest": digest,
            "duration_s": outcome.duration_s,
            "content_type": outcome.content_type or None,
            "size_bytes": outcome.size_bytes or None,
            "detail": "" if outcome.terminal else SETTLEMENT_NOT_OBSERVED,
        }
        if digest:
            row["artifacts"] = [
                {
                    "digest": digest,
                    "kind": "audio",
                    "name": artifact_name,
                    "content_type": outcome.content_type or None,
                    "duration_s": outcome.duration_s,
                }
            ]
        try:
            await journal(row)
        except Exception:  # noqa: BLE001 - a bookkeeping row must never crash the task
            logger.debug("audio settlement journal row failed", exc_info=True)


def _register_asset(
    outcome: SettleOutcome, prompt: str | None = None
) -> tuple[str | None, str | None]:
    """Download/verify already happened; this is the ONE registration path.

    Returns ``(digest, display name)``. The name comes from the REQUEST (the
    prompt carried in the settle payload), exactly like a first-class
    generation: this task runs after the turn that asked for the audio is
    gone, so it carries what the asset IS rather than reading turn state.

    Kept separate from the observation so a registration failure cannot make a
    settled charge look unobserved (the money is recorded regardless).
    """
    asset = outcome.asset
    if asset is None:
        return None, None
    try:
        from local_operator.artifacts.naming import media_filename
        from local_operator.session.attachments import cache_media

        block = cache_media(
            asset.data,
            asset.content_type,
            kind="audio",
            name=media_filename(
                prompt,
                getattr(asset, "content_type", None) or outcome.content_type,
                kind="audio",
            ),
        )
        if block is None:
            return None, None
        return block.attachment, block.name
    except Exception:  # noqa: BLE001 - registration is best-effort
        logger.debug("audio settlement registration failed", exc_info=True)
        return None, None


def _settled_record(payload: dict[str, Any], outcome: SettleOutcome) -> Any:
    """The ledger REVISION for a late-settled charge (the cancel row's id)."""
    from local_operator.session.channel_spend import (
        ChannelSpendRecord,
        map_audio_cost_labels,
        usd_to_micro,
    )

    basis, source, price_version = map_audio_cost_labels(
        str(payload.get("provider") or ""),
        "reported",
        outcome.billing_basis,
        None,
        has_amount=outcome.cost_usd is not None,
    )
    return ChannelSpendRecord(
        record_id=str(payload.get("record_id") or ""),
        rev=int(payload.get("rev") or 0) + 1,
        channel="audio",
        provider=str(payload.get("provider") or ""),
        model=str(payload.get("model") or ""),
        units=outcome.units if outcome.units is not None else float(payload.get("units") or 0.0),
        unit=outcome.unit or str(payload.get("unit") or ""),
        amount_micro=usd_to_micro(outcome.cost_usd) if outcome.cost_usd is not None else None,
        billing_basis=basis,
        cost_source=source,
        price_version=price_version,
        status="ok" if outcome.error_type == "" else "failed",
        request_id=str(payload.get("request_id") or ""),
        usage_record_id=outcome.usage_record_id or "",
        detail=(
            "settled after a best-effort cancel"
            if outcome.error_type == ""
            else f"settled after a best-effort cancel ({outcome.error_type})"
        ),
    )


__all__ = [
    "AUDIO_SETTLE_CUSTOM_TYPE",
    "SETTLEMENT_NOT_OBSERVED",
    "SETTLE_DEADLINE_S",
    "SETTLE_POLL_INTERVAL_S",
    "SettleOutcome",
    "observe_audio_settlement",
    "settle_cancelled_audio",
]
