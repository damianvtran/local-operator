"""Channel spend at the session seam: ingest, journal, marker, backfill, publish.

These are the design §7 claims that only an assembled ``Session`` can answer:
the transcript round-trip, the idempotent backfill (run twice, assert
identical), the one-time start marker, and the published ``spend_channels``
object reaching the frontend state.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from local_operator.harness.types import Message, ModelSpec, Usage
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


async def _no_stream(*_args: Any, **_kwargs: Any) -> AsyncIterator[Any]:
    """A stream_fn-shaped async generator that must never be pulled."""
    raise AssertionError("no stream expected")
    yield  # pragma: no cover — unreachable; makes this an async generator


def published(store: Any) -> Any:
    """The store's published channel object, asserted present (test shorthand)."""
    payload = store.state.spend_channels
    assert payload is not None, "this host must publish the object"
    return payload


def make_session(root: Path, **kwargs: Any) -> Session:
    """A real Session over ``root/<session_name>``.

    ``session_name`` is a parameter because a session's id IS its transcript
    directory's name: two sessions built as ``<a>/sess`` and ``<b>/sess`` would
    share the id "sess", which the child-relay guard (correctly) reads as a
    self-relay.
    """
    root.mkdir(parents=True, exist_ok=True)
    name = str(kwargs.pop("session_name", "sess"))
    return Session(
        model=kwargs.pop("model", MODEL),
        stream_fn=_no_stream,
        tools=[],
        transcript=Transcript(root / name),
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

        channels_obj = published(session._frontend_state_store)
        assert channels_obj.total_micro == 8000
        assert channels_obj.tracked is True
        assert channels_obj.knowledge == "exact"

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
        held = session.channels.get("image:x")
        assert held is not None and held.amount_micro == 53000
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
        channels_obj = published(session._frontend_state_store)
        assert channels_obj.tracked is False
        assert (
            channels_obj.knowledge == "partial"
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


# -- round-1 remediation: origin semantics, publishing, relay -----------------


def test_fresh_session_is_tracked_from_creation(tmp_path: Path) -> None:
    """M2: a NEW session on this build is tracked before any channel event.

    The old marker-on-first-record rule left every fresh conversation reading
    "channels not tracked" until its first search or image — a false alarm on
    the exact sessions that have nothing to declare.
    """

    async def main() -> None:
        session = make_session(tmp_path)
        assert session._channels_origin == "fresh"
        assert session.channels_tracked is True
        assert published(session._frontend_state_store).tracked is True

    asyncio.run(main())


def test_fresh_session_writes_its_marker_at_the_adopt_seam(tmp_path: Path) -> None:
    """A fresh session's ``start`` marker lands at adopt, once, with no records."""

    async def main() -> None:
        session = make_session(tmp_path)
        session.rebuild_channels_if_needed()
        await _settle(session)
        assert session.channels_started is True
        assert session._transcript.channel_spend_tracked() is True
        session.rebuild_channels_if_needed()
        await _settle(session)
        markers = [
            entry
            for entry in session._transcript.entries()
            if entry.payload.get("custom_type") == CHANNEL_SPEND_CUSTOM_TYPE
            and isinstance(entry.payload.get("details"), dict)
            and entry.payload["details"].get("kind") == "start"
        ]
        assert len(markers) == 1, "the marker is one-time"

    asyncio.run(main())


def test_legacy_journal_stays_untracked_after_a_live_record(tmp_path: Path) -> None:
    """M2: recording does not forgive history — tracked stays false, partial stays.

    The old live-record path wrote the marker on ANY unmarked journal, so one
    new search flipped a backfilled legacy session to ``tracked/exact`` and
    silently forgave the image row whose cost was never recorded.
    """
    directory = tmp_path / "sess"
    transcript = Transcript(directory)

    async def seed() -> None:
        await transcript.append_message(
            Message(
                role="tool",
                tool_call_id="c1",
                tool_name="generate_image",
                provider_payload={
                    "details": {
                        "provider": "radient",
                        "model": "gpt-image-2",
                        "generation_id": "req_legacy",
                        "cost_usd": 0.053,
                    }
                },
            )
        )

    asyncio.run(seed())

    async def main() -> None:
        session = make_session(tmp_path)
        assert session._channels_origin == "legacy"
        assert session.channels_tracked is False
        session.rebuild_channels_if_needed()
        await _settle(session)
        session.record_channel_spend(search_record(record_id="search:live"))
        await _settle(session)
        assert session.channels_tracked is False, "a live record must not flip legacy history"
        channels_obj = published(session._frontend_state_store)
        assert channels_obj.tracked is False
        assert channels_obj.knowledge == "partial"
        assert (
            session._transcript.channel_spend_tracked() is False
        ), "no marker may claim history this build never watched"

    asyncio.run(main())


def test_spend_writers_republish_the_channel_object(tmp_path: Path) -> None:
    """M1's probe, at the session seam: the published total follows the writers.

    The band reads ONLY the published object, so a writer that moves inference
    money without republishing leaves the band behind by exactly that delta —
    round 1 measured the re-price and the aside accrual doing exactly that
    (``cumulative_parent_cost=3.0`` beside ``spend_channels.total_micro=8000``).
    """

    async def main() -> None:
        session = make_session(tmp_path)
        store = session._frontend_state_store
        session.record_channel_spend(search_record())
        await _settle(session)
        assert published(store).total_micro == 8000

        # The aside/detached-call writer: a provider call accrued outside the
        # agent event stream.
        usage = Usage(
            provider="deepseek",
            model_id="deepseek-chat",
            input_tokens=100,
            output_tokens=20,
            context_tokens=100,
            usd_cost=2.5,
        )
        store.accrue_usage(session, usage)
        # ``cumulative_parent_cost`` stays inference-only by contract; the
        # PUBLISHED total is the one that must include both halves.
        assert store.state.cumulative_parent_cost == 2.5
        assert (
            published(store).total_micro == 2_508_000
        ), "the published object must follow the accrual"

        # The re-price writer, through the adopt seam's early arm (a session
        # with no restored usage still republishes the fold's object).
        session.spend.accrue(500_000, {"provider": "deepseek", "model_id": "deepseek-chat"})
        store.refresh_restored_usage(session)
        assert (
            published(store).total_micro == 3_008_000
        ), "the published object must follow the re-price"

    asyncio.run(main())


def test_child_channel_spend_reaches_the_parent_total_once(tmp_path: Path) -> None:
    """Q2/Q3: a live child's record lands in the parent's children block, once."""

    async def main() -> None:
        parent = make_session(tmp_path / "p")
        child = make_session(
            tmp_path / "c",
            session_name="child",
            parent_session=parent,
            parent_session_id=str(parent.session_id),
        )
        child.record_channel_spend(search_record(record_id="search:kid"))
        await _settle(child)
        await _settle(parent)

        assert child.channels.total_known_micro() == 8000
        rows = child._transcript.channel_spend_rows()
        assert rows[0]["parent_session_id"] == str(
            parent.session_id
        ), "the child's own journal carries the parent link"

        state = parent._frontend_state_store.state
        assert state.spend_channels is not None
        assert published(parent._frontend_state_store).children.total_micro == 8000
        assert published(parent._frontend_state_store).total_micro == 8000
        assert [
            row.channel for row in published(parent._frontend_state_store).rows
        ] == [], "a child's record is not the parent's own row"
        # Replaying the same child record changes nothing (record_id dedup).
        parent._absorb_child_channel_spend(child.channels.rows()[0])
        assert published(parent._frontend_state_store).children.total_micro == 8000

    asyncio.run(main())


def test_headless_checkpoint_restore_publishes_the_fold(tmp_path: Path) -> None:
    """Q1: a headless resume rebuilds ``spend_channels`` from the fold.

    The checkpoint is written at turn end; a settle landing after it is in the
    journal and the fold but not in the checkpointed object, and publishing the
    stale copy made a daemon-shaped resume disagree with the live owner.
    """
    from local_operator.session.frontend_state import FrontendStateStore

    async def main() -> None:
        session = make_session(tmp_path)
        session.record_channel_spend(search_record())
        await _settle(session)
        session.accrue_spend(10_000, {"provider": "openrouter", "model_id": "x"})
        session.refresh_frontend_usage()
        store = session._frontend_state_store
        await store.checkpoint(session._transcript)
        assert published(store).total_micro == 18_000

        # ... and the settle lands AFTER the checkpoint.
        session.record_channel_spend(
            search_record(record_id="search:one", amount_micro=15_000, rev=1)
        )
        await _settle(session)
        assert published(store).total_micro == 25_000

        resumed = FrontendStateStore.from_checkpoint(session)
        assert (
            published(resumed).total_micro == 25_000
        ), "the restored object must come from the fold, not the checkpoint"

    asyncio.run(main())


def test_backfill_ignores_non_image_rows_that_report_a_cost(tmp_path: Path) -> None:
    """MINOR 2: only a ``generate_image`` row may become an image record."""
    directory = tmp_path / "sess"
    transcript = Transcript(directory)

    async def seed() -> None:
        await transcript.append_message(
            Message(
                role="tool",
                tool_call_id="c9",
                tool_name="classify",
                provider_payload={"details": {"provider": "openrouter", "cost_usd": 0.01}},
            )
        )

    asyncio.run(seed())
    rows = Transcript(directory).channel_backfill_rows()
    assert rows == [], "a non-image tool's per-call cost is not an image record"
