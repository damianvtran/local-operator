"""Channel spend at the session seam: ingest, journal, marker, backfill, publish.

These are the design §7 claims that only an assembled ``Session`` can answer:
the transcript round-trip, the idempotent backfill (run twice, assert
identical), the one-time start marker, and the published ``spend_channels``
object reaching the frontend state.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from local_operator.harness.types import Message, ModelSpec
from local_operator.session.channel_spend import (
    CHANNEL_SPEND_CUSTOM_TYPE,
    ChannelSpendRecord,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

MODEL = ModelSpec(
    provider="deepseek",
    model_id="deepseek-chat",
    display_name="DeepSeek Chat",
    context_window=64_000,
)


async def _no_stream(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover - never runs
    raise AssertionError("no stream expected")


def make_session(tmp_path: Path, **kwargs: Any) -> Session:
    return Session(
        model=kwargs.pop("model", MODEL),
        stream_fn=_no_stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: [],
        **kwargs,
    )


async def _settle(session: Session) -> None:
    """Wait for the session's channel appends to finish (bounded, structural)."""
    for _ in range(400):
        if not session._channel_tasks:
            return
        await asyncio.sleep(0.005)
    raise AssertionError("channel appends never settled")


def search_record(record_id: str = "search:one", **kwargs: Any) -> ChannelSpendRecord:
    return ChannelSpendRecord(
        record_id=record_id,
        channel=kwargs.pop("channel", "search"),
        provider=kwargs.pop("provider", "tavily"),
        units=kwargs.pop("units", 1),
        unit=kwargs.pop("unit", "searches"),
        amount_micro=kwargs.pop("amount_micro", 8000),
        billing_basis=kwargs.pop("billing_basis", "estimated"),
        cost_source=kwargs.pop("cost_source", "catalogue"),
        price_version=kwargs.pop("price_version", "client-search-table-2026-09"),
        **kwargs,
    )


def test_record_folds_journals_marker_and_publishes(tmp_path: Path) -> None:
    """One ingest: fold, journal row + start marker, published object."""

    async def main() -> None:
        session = make_session(tmp_path)
        session.record_channel_spend(search_record())
        await _settle(session)

        transcript = session._transcript
        rows = transcript.channel_spend_rows()
        assert [row["record_id"] for row in rows] == ["search:one"]
        assert rows[0]["session_id"] == session.session_id, "the SESSION stamps identity"
        assert transcript.channel_spend_tracked() is True, "the marker was written first"
        assert session.channels.total_known_micro() == 8000

        state = session._frontend_state_store.state
        assert state.spend_channels is not None
        assert state.spend_channels.total_micro == 8000
        assert state.spend_channels.tracked is True
        assert state.spend_channels.knowledge == "exact"

    asyncio.run(main())


def test_a_replayed_record_changes_nothing(tmp_path: Path) -> None:
    """Idempotence: the same record twice is one row, one total."""

    async def main() -> None:
        session = make_session(tmp_path)
        for _ in range(3):
            session.record_channel_spend(search_record())
        await _settle(session)
        assert len(session._transcript.channel_spend_rows()) == 1
        assert session.channels.total_known_micro() == 8000
        # ... and no second marker, either.
        markers = [
            entry
            for entry in session._transcript.entries()
            if entry.payload.get("custom_type") == CHANNEL_SPEND_CUSTOM_TYPE
            and isinstance(entry.payload.get("details"), dict)
            and entry.payload["details"].get("kind") == "start"
        ]
        assert len(markers) == 1

    asyncio.run(main())


def test_a_rev_upgrade_wins_the_fold_and_appends_a_second_row(tmp_path: Path) -> None:
    async def main() -> None:
        session = make_session(tmp_path)
        session.record_channel_spend(
            search_record(record_id="image:x", channel="image", unit="images")
        )
        session.record_channel_spend(
            search_record(
                record_id="image:x",
                channel="image",
                unit="images",
                amount_micro=53000,
                billing_basis="billed",
                rev=1,
            )
        )
        await _settle(session)
        assert session.channels.get("image:x").amount_micro == 53000
        assert len(session._transcript.channel_spend_rows()) == 2, "each revision is its own row"

    asyncio.run(main())


def test_resume_folds_the_journal_and_keeps_tracked(tmp_path: Path) -> None:
    async def main() -> None:
        first = make_session(tmp_path)
        first.record_channel_spend(search_record())
        await _settle(first)

        resumed = make_session(tmp_path)
        assert resumed.channels.total_known_micro() == 8000
        assert resumed.channels_started is True

    asyncio.run(main())


def test_backfill_recovers_legacy_rows_without_a_marker_and_is_idempotent(
    tmp_path: Path,
) -> None:
    """Run it twice: identical fold, identical journal — the §7 claim."""
    directory = tmp_path / "sess"
    transcript = Transcript(directory)

    async def seed() -> None:
        await transcript.append_message(
            Message(
                role="tool",
                tool_call_id="c1",
                tool_name="web_search",
                provider_payload={
                    "details": {
                        "provider": "tavily",
                        "search_cost": {"usd": 0.008, "basis": "published per-search rate"},
                    }
                },
            )
        )
        await transcript.append_message(
            Message(
                role="tool",
                tool_call_id="c2",
                tool_name="generate_image",
                provider_payload={
                    "details": {
                        "provider": "radient",
                        "model": "gpt-image-2",
                        "generation_id": "req_7f3a",
                        "cost_usd": 0.053,
                    }
                },
            )
        )

    asyncio.run(seed())

    async def main() -> None:
        session = make_session(tmp_path)
        assert session.channels_started is False, "a legacy journal has no marker"
        session.rebuild_channels_if_needed()
        await _settle(session)
        entry_ids = [entry.id for entry in session._transcript.entries() if entry.type == "message"]
        ids = sorted(session.channels._records)
        # One legacy search row (keyed by its tool entry) and one image row
        # (keyed by the provider's generation id, which is the stable identity).
        assert ids == sorted(["image:req_7f3a", f"legacy:{entry_ids[0]}"]), ids
        assert session.channels.total_known_micro() == 8_000 + 53_000
        rows_after_first = len(session._transcript.channel_spend_rows())

        # Second run: the gate has fired once per process, so a second process
        # (which is what the design's "run twice" means) must also change
        # nothing. Simulate it by re-running the worker directly.
        await session._rebuild_channels()
        await _settle(session)
        assert sorted(session.channels._records) == ids
        assert len(session._transcript.channel_spend_rows()) == rows_after_first
        assert session.channels_started is False, "backfill never claims tracking"
        state = session._frontend_state_store.state
        assert state.spend_channels is not None
        assert state.spend_channels.tracked is False
        assert (
            state.spend_channels.knowledge == "partial"
        ), "recovered history is partial: rows exist without the tracking marker"

    asyncio.run(main())


def test_ingest_reaches_analytics_through_the_recorder(tmp_path: Path) -> None:
    """The whole chain in one test: fold -> queue -> writer thread -> SQLite."""
    import sqlite3

    from local_operator.analytics import recorder as recorder_module
    from local_operator.analytics.store import AnalyticsStore

    async def main() -> None:
        db = tmp_path / "analytics.db"
        recorder = recorder_module.reset_recorder_for_test(AnalyticsStore(db_path=db))
        try:
            session = make_session(tmp_path)
            session.record_channel_spend(search_record())
            await _settle(session)
            recorder.flush_for_test()
            conn = sqlite3.connect(db)
            rows = conn.execute(
                "SELECT record_id, channel, amount_micro, session_id, billing_basis "
                "FROM channel_calls"
            ).fetchall()
            assert len(rows) == 1, rows
            record_id, channel, amount, session_id, basis = rows[0]
            assert record_id == "search:one"
            assert channel == "search" and amount == 8000
            assert session_id == session.session_id
            assert basis == "estimated"
        finally:
            recorder_module.reset_recorder_for_test()

    asyncio.run(main())


def test_cost_channels_capability_key_is_advertised() -> None:
    """The UI's gate: an old backend has no key, this one advertises 1."""
    from local_operator.server.features import feature_flags

    assert feature_flags()["cost_channels"] == 1
