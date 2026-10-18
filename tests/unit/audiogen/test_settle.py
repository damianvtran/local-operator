"""The bounded post-cancel settlement poll — money first, artifact second.

The contract this pins: a RUNNING cancel cannot stop the work, so the harness
observes it for a bounded window and records what actually happened. What is
under test is the OBSERVATION LOOP (terminal detection, the deadline, injected
sleep), and the two writes it may produce — a ledger revision carrying the late
charge, and one bookkeeping transcript row (with the digest when the paid
artifact could be registered). No paid call: the transport is scripted and the
registration seam is patched.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from local_operator.audiogen import rungs as audio_rungs
from local_operator.audiogen import settle as audio_settle

HUB = "https://hub.example/v1"
AUDIO_BYTES = b"ID3\x04\x00\x00\x00settled-late"
AUDIO_SHA = hashlib.sha256(AUDIO_BYTES).hexdigest()


def _transport(statuses: list[dict[str, Any]], *, with_result: bool = True):
    """A scripted Radient: one status answer per poll, then the result."""
    script = list(statuses)
    seen = {"status": 0, "result": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/tools/audio/status":
            seen["status"] += 1
            payload = script.pop(0) if len(script) > 1 else script[0]
            return httpx.Response(200, json=payload)
        if path == "/v1/tools/audio/result":
            seen["result"] += 1
            if not with_result:
                return httpx.Response(500, json={"error": "no result"})
            return httpx.Response(
                200,
                json={
                    "status": "COMPLETED",
                    "audio": {
                        "url": f"{HUB}/media/late.mp3",
                        "content_type": "audio/mpeg",
                        "duration_seconds": 7.5,
                        "sha256": AUDIO_SHA,
                    },
                    "cost": {
                        "v": 1,
                        "amount_usd": 0.03,
                        "basis": "billed",
                        "units": 44,
                        "unit": "chars",
                        "usage_record_id": "ur-late",
                    },
                },
            )
        if path == "/v1/media/late.mp3":
            return httpx.Response(200, content=AUDIO_BYTES)
        return httpx.Response(404)

    return httpx.MockTransport(handler), seen


@pytest.mark.asyncio
async def test_the_poll_settles_on_the_first_terminal_answer():
    transport, seen = _transport(
        [
            {"status": "IN_PROGRESS"},
            {"status": "IN_PROGRESS"},
            {
                "status": "COMPLETED",
                "settled": True,
                "cost": {"v": 1, "amount_usd": 0.03, "basis": "billed"},
            },
        ]
    )
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    async with httpx.AsyncClient(transport=transport) as client:
        outcome = await audio_settle.observe_audio_settlement(
            base_url=HUB,
            credential="tok",
            request_id="req-late",
            sleep=fake_sleep,
            client=client,
        )
    assert outcome.terminal is True
    assert outcome.status == "COMPLETED"
    assert outcome.cost_usd == 0.03
    assert outcome.billing_basis == "billed"
    assert outcome.asset is not None and outcome.asset.data == AUDIO_BYTES
    assert outcome.duration_s == 7.5
    assert outcome.settled_charge is True
    assert sleeps == [3.0, 3.0], "the contract's 2-5 s tick"


@pytest.mark.asyncio
async def test_the_poll_gives_up_at_the_bound_without_inventing_a_charge():
    transport, seen = _transport([{"status": "IN_QUEUE"}])
    sleeps: list[float] = []
    virtual = {"now": 0.0}

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        virtual["now"] += seconds

    def fake_clock() -> float:
        return virtual["now"]

    async with httpx.AsyncClient(transport=transport) as client:
        outcome = await audio_settle.observe_audio_settlement(
            base_url=HUB,
            credential="tok",
            request_id="req-late",
            poll_interval_s=3.0,
            deadline_s=9.0,
            sleep=fake_sleep,
            clock=fake_clock,
            client=client,
        )
    assert outcome.terminal is False
    assert outcome.status == "IN_QUEUE"
    assert outcome.cost_usd is None
    assert outcome.settled_charge is False
    # Polls at t=0,3,6,9: the deadline check runs AFTER an observation, so the
    # bound is four reads and 9 seconds of waiting — never a fifth tick.
    assert seen["status"] == 4, "bounded by the deadline, not by luck"
    assert sum(sleeps) <= 9.0


@pytest.mark.asyncio
async def test_a_late_failure_is_terminal_with_zero_charge():
    transport, _seen = _transport(
        [
            {
                "status": "COMPLETED",
                "error_type": "audio_rate_limited",
                "error": "Audio generation is busy right now. Wait a moment and try again.",
                "cost": {"v": 1, "amount_usd": 0.0, "basis": "billed"},
            }
        ]
    )

    async def fake_sleep(seconds: float) -> None:
        return None

    async with httpx.AsyncClient(transport=transport) as client:
        outcome = await audio_settle.observe_audio_settlement(
            base_url=HUB, credential="tok", request_id="req-late", sleep=fake_sleep, client=client
        )
    assert outcome.terminal is True
    assert outcome.error_type == "audio_rate_limited"
    assert outcome.cost_usd == 0.0
    # A stated zero IS a settled charge: the ledger records the server's word.
    assert outcome.settled_charge is True


@pytest.mark.asyncio
async def test_a_settled_charge_is_recorded_even_when_the_artifact_cannot_be_fetched():
    """Status alone carries the money; the artifact is a bonus."""
    transport, _seen = _transport(
        [{"status": "COMPLETED", "cost": {"v": 1, "amount_usd": 0.02, "basis": "billed"}}],
        with_result=False,
    )

    async def fake_sleep(seconds: float) -> None:
        return None

    async with httpx.AsyncClient(transport=transport) as client:
        outcome = await audio_settle.observe_audio_settlement(
            base_url=HUB, credential="tok", request_id="req-late", sleep=fake_sleep, client=client
        )
    assert outcome.settled_charge is True
    assert outcome.asset is None and outcome.asset_error


@pytest.mark.asyncio
async def test_settle_cancelled_audio_records_the_revision_and_the_row(monkeypatch):
    """The late charge lands as a REVISION of the cancel row, not a second row."""
    observed = audio_settle.SettleOutcome(
        status="COMPLETED",
        terminal=True,
        cost_usd=0.03,
        billing_basis="billed",
        usage_record_id="ur-late",
        duration_s=7.5,
        content_type="audio/mpeg",
        size_bytes=len(AUDIO_BYTES),
        asset=SimpleNamespace(data=AUDIO_BYTES, content_type="audio/mpeg", duration_s=7.5),
    )

    async def fake_observe(**_kwargs):
        return observed

    monkeypatch.setattr(audio_settle, "observe_audio_settlement", fake_observe)
    registered: list[tuple[bytes, str, str]] = []

    def fake_cache_media(data, content_type, *, kind):
        registered.append((data, content_type, kind))
        return SimpleNamespace(attachment="a" * 32)

    import local_operator.session.attachments as attachments

    monkeypatch.setattr(attachments, "cache_media", fake_cache_media)

    spent: list[Any] = []
    rows: list[dict[str, Any]] = []

    async def journal(row: dict[str, Any]) -> None:
        rows.append(row)

    await audio_settle.settle_cancelled_audio(
        {
            "request_id": "req-late",
            "base_url": HUB,
            "credential": "tok",
            "provider": "radient",
            "model": "radient/voiceover",
            "record_id": "audio:req-late",
            "rev": 0,
            "units": 44.0,
            "unit": "chars",
        },
        record_spend=spent.append,
        journal=journal,
    )

    assert registered == [(AUDIO_BYTES, "audio/mpeg", "audio")], "one registration path"
    assert len(spent) == 1
    record = spent[0]
    assert record.record_id == "audio:req-late" and record.rev == 1
    assert record.channel == "audio"
    assert record.amount_micro == 30_000
    assert record.billing_basis == "billed"
    assert record.cost_source == "server_reported"
    assert record.status == "ok"
    assert record.request_id == "req-late"

    assert len(rows) == 1
    row = rows[0]
    assert row["version"] == 1 and row["settlement_observed"] is True
    assert row["digest"] == "a" * 32
    assert row["artifacts"][0]["kind"] == "audio"
    assert row["cost_usd"] == 0.03


@pytest.mark.asyncio
async def test_settle_cancelled_audio_records_no_charge_when_unobserved(monkeypatch):
    """Nothing terminal inside the bound: a row saying so, and NO money."""
    observed = audio_settle.SettleOutcome(status="IN_PROGRESS", terminal=False)

    async def fake_observe(**_kwargs):
        return observed

    monkeypatch.setattr(audio_settle, "observe_audio_settlement", fake_observe)
    spent: list[Any] = []
    rows: list[dict[str, Any]] = []

    async def journal(row: dict[str, Any]) -> None:
        rows.append(row)

    await audio_settle.settle_cancelled_audio(
        {
            "request_id": "req-late",
            "base_url": HUB,
            "credential": "tok",
            "provider": "radient",
            "model": "radient/voiceover",
            "record_id": "audio:req-late",
            "rev": 0,
            "units": 44.0,
            "unit": "chars",
        },
        record_spend=spent.append,
        journal=journal,
    )
    assert spent == [], "no invented charge"
    assert rows[0]["settlement_observed"] is False
    assert rows[0]["detail"] == audio_settle.SETTLEMENT_NOT_OBSERVED
    assert rows[0]["cost_usd"] is None


def test_the_settle_row_type_is_bookkeeping_and_pinned():
    """The literal is duplicated in ``transcript`` (no import); pin the pair."""
    from local_operator.session import transcript

    assert (
        audio_settle.AUDIO_SETTLE_CUSTOM_TYPE in transcript.BOOKKEEPING_CUSTOM_TYPES
    ), "a late settle must not re-rank the session as freshly worked"


def test_the_observation_uses_the_rungs_settlement_read():
    """The wire code lives in ONE module (rungs); settle only observes it."""
    assert audio_settle.fetch_radient_settlement is audio_rungs.fetch_radient_settlement
    assert audio_rungs.SETTLEMENT_WATCH_NOTE == "Checking for settlement."
