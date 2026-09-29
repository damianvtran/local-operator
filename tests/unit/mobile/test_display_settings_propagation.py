"""A ``display.*`` write from another process reaches a RUNNING daemon.

Design review round 1 on #1746 (D1/D2): a ``config.yml`` edit is the phone's
only toggle (there is no settings UI on mobile), and the daemon's folds read
the process-cached display flags (``tui.settings``). Nothing in ``mobile/``
subscribed the config watcher, so a running daemon kept the value from its
first read, and ``/history`` kept a pre-flip render in the durable fold cache
(whose ``render`` is re-derived only when the transcript GROWS — that is D2),
until a restart.

These tests drive the exact seam production uses — ``MobileDaemon
.watch_display_settings`` plus a watcher tick — with the write shaped like
another process's (``settings_io._store``, below the notify hook), and never
the clock: ``poll_now()`` is the tick. The frame this pins is "no restart".
The retained-route cells extend that to QA round 2's Q1: a route the daemon
has served live keeps a per-route projection, and the flip must drop that too
(the stale branch would otherwise re-serve the retained frame on reconnect).
A second retained-route cell pins QA round 3's Q2: a clear that lands WHILE
the seed's fold is in flight must not re-materialize the fold in hand — the
resumed seed recomputes for the state the flip established.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.config_watch import (
    _reset_for_tests,
    existing_watcher,
    process_watcher,
)
from local_operator.harness.message_types import PEER_MESSAGE_MESSAGE_TYPE
from local_operator.harness.types import CustomMessage, Message, TextContent, ToolCall
from local_operator.mobile.daemon import (
    MobileDaemon,
    SessionEntry,
    _durable_projection,
    _entry_for_session,
    _history_page,
)
from local_operator.mobile.types import SessionProjection
from local_operator.session.runtime import registry
from local_operator.session.transcript import Transcript

SESSION = "prop-s1"
KEY = "display.hide_cross_session"


@pytest.fixture(autouse=True)
def _fresh_registry():
    _reset_for_tests()
    yield
    _reset_for_tests()


@pytest.fixture(autouse=True)
def _fresh_display_cache():
    """The reader is process-global; pin the tests to their own config dir."""
    from local_operator.tui.settings import settings_reload

    settings_reload()
    yield
    settings_reload()


def _write_elsewhere(config_dir: Path, key: str, value: object) -> None:
    """A write shaped like another process's: a fresh manager, no notify hook."""
    setting = settings_io.resolve_key(key)
    assert setting is not None, key
    settings_io._store(ConfigManager(config_dir), setting.path, value)


async def _seed_session(config_dir: Path, *, extra_turn: bool = False) -> Path:
    """A durable user session with one peer receipt and one `send` pair."""
    directory = config_dir / "sessions" / SESSION
    directory.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(directory)

    async def run() -> None:
        await transcript.append_message(Message.user("run the sweep", id="u1"))
        await transcript.append_message(
            CustomMessage(
                custom_type=PEER_MESSAGE_MESSAGE_TYPE,
                attribution="user",
                details={
                    "text": "<wrapped>status?</wrapped>",
                    "body": "peer status?",
                    "sender": {"pid": 4242, "conversation_name": "peer-session"},
                },
            )
        )
        for n, call_id in ((1, "cs1"),) if not extra_turn else ((1, "cs1"), (2, "cs2")):
            await transcript.append_message(
                Message(
                    role="assistant",
                    content=[TextContent(text="")],
                    tool_calls=[ToolCall(id=call_id, name="send", arguments={"text": f"hi {n}"})],
                    stop_reason="toolUse",
                )
            )
            await transcript.append_message(
                Message(
                    role="tool",
                    content=[TextContent(text="delivered")],
                    tool_call_id=call_id,
                    tool_name="send",
                )
            )
            await transcript.append_message(Message.assistant(f"done {n}", id=f"a{n}"))
        if extra_turn:
            await transcript.append_message(
                CustomMessage(
                    custom_type=PEER_MESSAGE_MESSAGE_TYPE,
                    attribution="user",
                    details={
                        "text": "<wrapped>again</wrapped>",
                        "body": "peer again",
                        "sender": {"pid": 4242, "conversation_name": "peer-session"},
                    },
                )
            )
            await transcript.append_message(Message.assistant("done 3", id="a3"))

    await run()
    return directory


def _kinds(page) -> list[str]:
    return [entry.kind for entry in page]


async def _retained_route(daemon: MobileDaemon, *, live_version: int = 53) -> int:
    """The state a phone's live view leaves behind, then the route's death.

    Two captures, both production shapes: the attach repaint that retains a
    live route's summary (``_connect_phone.repaint`` ->
    ``capture_subagent_details(..., record=record)``), then the reaper's
    terminal capture on death (``_scan_once``: ``_durable_projection`` ->
    ``capture(..., record=..., terminal=True)``). ``live_version`` stands in
    for the live owner's fold counter — restart-at-zero per owner, so strictly
    above the fresh durable fold's single bump; 53 is the number the QA
    round 2 repro observed on the wire for the frozen `W` route. Returns the
    retained epoch the summary carries.
    """
    record = registry.SessionRecord(
        pid=4242,
        kind="daemon",
        session_id=SESSION,
        conversation_name="retained",
        cwd="/tmp",
        model_label="",
        control_port=1,
        control_key="k",
        started_at=1.0,
        heartbeat_at=1.0,
    )
    entry = SessionEntry(record)
    daemon.table.entries[record.pid] = entry
    daemon.table.session_subscribers.setdefault(SESSION, set()).add(asyncio.Queue())

    live = await asyncio.to_thread(_durable_projection, SESSION)
    assert live is not None
    live.version = live_version
    entry.projection = daemon.capture_subagent_details(live, record=record)
    entry.ended = True
    dead = await asyncio.to_thread(_durable_projection, SESSION)
    assert dead is not None
    daemon.capture_subagent_details(dead, record=record, terminal=True)
    daemon._prune_projection_generation(SESSION)
    return live_version + 1


async def _reconnect_seed(daemon: MobileDaemon) -> SessionProjection:
    """``api_session_events``' opening seed, in the handler's own order: the
    live projection when a live entry has one, else the stamped durable
    capture — the same ``_capture_durable_projection`` call the handler makes,
    so the cells cannot drift from the production order they exercise."""
    live = _entry_for_session(daemon, SESSION)
    projection: SessionProjection | None = live.projection if live is not None else None
    if projection is None:
        projection = await daemon._capture_durable_projection(SESSION)
        assert projection is not None  # the fixture seeds this session on disk
    return projection


@pytest.mark.asyncio
async def test_a_disk_write_reaches_a_running_daemons_history_without_a_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The design review's probe E, inverted: the rows are GONE after the write.

    The first read is the daemon's boot shape — it caches the reader AND
    builds the durable render — and the only thing that may change the state
    afterwards is the watcher contract this test drives.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    await _seed_session(config_dir)

    page, _more = await asyncio.to_thread(_history_page, SESSION, None, 50)
    assert _kinds(page) == ["user", "peer_message", "tool", "assistant"]

    MobileDaemon(port=0, password="pw").watch_display_settings()
    _write_elsewhere(config_dir, KEY, True)
    change = process_watcher(config_dir).poll_now()
    assert change is not None and KEY in change.changed_keys

    # No restart, no transcript growth: the CACHED fold re-derived.
    page, _more = await asyncio.to_thread(_history_page, SESSION, None, 50)
    assert _kinds(page) == ["user", "assistant"]

    # And the fresh-fold path (the SSE seed) agrees with it.
    projection = await asyncio.to_thread(_durable_projection, SESSION)
    assert projection is not None
    assert [entry.kind for entry in projection.transcript] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_the_flip_is_reversible_without_a_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Probe D/D2 inverted: a write back restores the rows, no append needed."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    await _seed_session(config_dir)

    daemon = MobileDaemon(port=0, password="pw")
    daemon.watch_display_settings()

    _write_elsewhere(config_dir, KEY, True)
    assert process_watcher(config_dir).poll_now() is not None
    page, _more = await asyncio.to_thread(_history_page, SESSION, None, 50)
    assert _kinds(page) == ["user", "assistant"]

    _write_elsewhere(config_dir, KEY, False)
    assert process_watcher(config_dir).poll_now() is not None
    page, _more = await asyncio.to_thread(_history_page, SESSION, None, 50)
    assert _kinds(page) == ["user", "peer_message", "tool", "assistant"]


@pytest.mark.asyncio
async def test_scroll_back_pages_come_from_the_re_derived_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D2's leak is in the PAGES: every page off the cache must be post-flip.

    Two pages are walked after the write — the seeded tail and one fetched by
    an ``anchor`` — because the pre-flip leak served the rows from a render
    the flag never reached, and a page fetched that way is the exact thing a
    scrolling phone sees.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    await _seed_session(config_dir, extra_turn=True)

    MobileDaemon(port=0, password="pw").watch_display_settings()

    # Warm the cache under the OLD value, then flip.
    page, _more = await asyncio.to_thread(_history_page, SESSION, None, 2)
    assert _kinds(page) == ["peer_message", "assistant"]
    _write_elsewhere(config_dir, KEY, True)
    assert process_watcher(config_dir).poll_now() is not None

    tail, has_more = await asyncio.to_thread(_history_page, SESSION, None, 2)
    assert _kinds(tail) == ["assistant", "assistant"]
    assert has_more is True
    older, _has_more = await asyncio.to_thread(_history_page, SESSION, tail[0].id, 2)
    assert _kinds(older) == ["user", "assistant"]
    for entry in [*tail, *older]:
        assert entry.kind != "peer_message"
        assert not (entry.kind == "tool" and entry.tool_name == "send")


@pytest.mark.asyncio
async def test_a_non_display_write_leaves_the_fold_cache_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The follower filters to ``display.*``: unrelated edits must not re-fold.

    A dropped cache costs one full fold per session on its next open, so the
    filter is load-bearing rather than cosmetic — and the identity assertion
    is what proves the state was NOT re-derived behind the test's back.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    await _seed_session(config_dir)

    MobileDaemon(port=0, password="pw").watch_display_settings()
    page, _more = await asyncio.to_thread(_history_page, SESSION, None, 50)
    assert _kinds(page) == ["user", "peer_message", "tool", "assistant"]

    from local_operator.mobile.daemon import _durable_fold_cache

    directory = config_dir / "sessions" / SESSION
    state = _durable_fold_cache().get(directory)

    _write_elsewhere(config_dir, "compaction.threshold_percent", 0.5)
    change = process_watcher(config_dir).poll_now()
    assert change is not None and KEY not in change.changed_keys

    assert _durable_fold_cache().get(directory) is state, "an unrelated write dropped the fold"

    # The display write is the positive control: it DOES drop it.
    _write_elsewhere(config_dir, KEY, True)
    assert process_watcher(config_dir).poll_now() is not None
    assert _durable_fold_cache().get(directory) is not state


@pytest.mark.asyncio
async def test_a_retained_routes_sse_seed_follows_the_flip_without_a_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA round 2, cell 4, inverted: a route the phone has VIEWED live re-seeds
    from the post-flip fold on reconnect — not from its retained pre-flip frame.

    The freeze this pins (Q1): ``session_projections[sid]`` outlives the flip,
    and the stale branch of ``capture_subagent_details`` re-serves it in place
    of the fresh durable fold, so every reconnect carried the last live frame —
    ``["user", "peer_message", "tool", "assistant"]`` — until a restart.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    await _seed_session(config_dir)

    daemon = MobileDaemon(port=0, password="pw")
    daemon.watch_display_settings()
    epoch = await _retained_route(daemon)

    # The route is genuinely retained — the pre-flip seed IS the live frame.
    seed = await _reconnect_seed(daemon)
    assert _kinds(seed.transcript) == ["user", "peer_message", "tool", "assistant"]
    assert seed.version == epoch

    _write_elsewhere(config_dir, KEY, True)
    assert process_watcher(config_dir).poll_now() is not None

    seed = await _reconnect_seed(daemon)
    assert _kinds(seed.transcript) == ["user", "assistant"]
    # Re-materialized at the retained epoch: a browser has already observed
    # this version, so the flip must not renumber the route under it.
    assert seed.version == epoch

    # And the retained path is reversible the same way (consecutive writes
    # stay clean: each tick converges the next seed from the fresh fold).
    _write_elsewhere(config_dir, KEY, False)
    assert process_watcher(config_dir).poll_now() is not None
    seed = await _reconnect_seed(daemon)
    assert _kinds(seed.transcript) == ["user", "peer_message", "tool", "assistant"]
    assert seed.version == epoch


@pytest.mark.asyncio
async def test_a_fold_that_straddles_the_flip_is_not_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA round 3 Q2 / agent review round 3 MAJOR: the clear can land DURING
    the seed's fold, and the resumed capture must not re-materialize the
    pre-flip fold it is holding.

    The seed awaits its durable fold off-loop (``asyncio.to_thread``), so the
    loop is free for the follower tick between the fold starting and its
    capture; the fold in hand is then pre-flip, and the missing-payload branch
    would re-materialize it — every reconnect after that re-serving the stale
    frame through the stale branch (the probe's ``v54`` frozen shape). The
    interleave here is deterministic: the fold is computed, then PARKED while
    the flip's clear runs, so the resumed seed sees a display generation that
    moved under it. It must recompute once for the state the flip established,
    retain THAT, and the reconnects that follow must serve the clean frame.
    The refusal backstop is pinned directly too: a capture stamped before the
    last clear is dropped, not re-inserted.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    await _seed_session(config_dir)

    daemon = MobileDaemon(port=0, password="pw")
    daemon.watch_display_settings()
    epoch = await _retained_route(daemon)
    seed = await _reconnect_seed(daemon)
    assert _kinds(seed.transcript) == ["user", "peer_message", "tool", "assistant"]

    # The seed's fold: computed while the flag is still OFF, then parked until
    # AFTER the flip's clear — exactly the window ``to_thread`` opens.
    import local_operator.mobile.daemon as daemon_mod

    real = daemon_mod._durable_projection
    fold_in_hand = threading.Event()
    release_fold = threading.Event()
    calls: list[int] = []
    folds: list[SessionProjection] = []

    def gated(session_id: str) -> SessionProjection | None:
        calls.append(1)
        projection = real(session_id)
        if len(calls) == 1:
            assert projection is not None  # the fixture seeds this session on disk
            folds.append(projection)
            fold_in_hand.set()
            assert release_fold.wait(timeout=10), "the test never released the fold"
        return projection

    monkeypatch.setattr(daemon_mod, "_durable_projection", gated)
    gen_before = daemon.display_generation

    seed_task = asyncio.create_task(_reconnect_seed(daemon))
    assert await asyncio.to_thread(fold_in_hand.wait, 10), "the fold never started"
    _write_elsewhere(config_dir, KEY, True)
    assert process_watcher(config_dir).poll_now() is not None
    release_fold.set()
    seed = await seed_task

    # The recompute happens exactly once, and it is what the seed serves: the
    # post-flip frame at the retained epoch.
    assert len(calls) == 2
    assert _kinds(seed.transcript) == ["user", "assistant"]
    assert seed.version == epoch

    # ``session_projections`` holds the recomputed payload, not the pre-flip
    # fold the seed was holding when the clear landed.
    retained = daemon.session_projections[SESSION]
    assert _kinds(retained.transcript) == ["user", "assistant"]

    # The refusal backstop: the pre-flip fold, stamped before the clear, is
    # dropped rather than re-inserted (and leaves the retained payload alone).
    from local_operator.mobile.daemon import _StaleProjection

    with pytest.raises(_StaleProjection):
        daemon.capture_subagent_details(folds[0], display_generation=gen_before)
    assert daemon.session_projections[SESSION] is retained

    # The reconnects that follow the race are served the clean frame.
    for _ in range(2):
        seed = await _reconnect_seed(daemon)
        assert _kinds(seed.transcript) == ["user", "assistant"]
        assert seed.version == epoch


@pytest.mark.asyncio
async def test_a_cold_routes_sse_seed_is_untouched_by_the_flip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA round 2, cell 3: the route with no retained projection was right
    before this fix and stays right — the seed is the fresh fold at v1."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    await _seed_session(config_dir)

    daemon = MobileDaemon(port=0, password="pw")
    daemon.watch_display_settings()
    assert SESSION not in daemon.session_projections

    _write_elsewhere(config_dir, KEY, True)
    assert process_watcher(config_dir).poll_now() is not None

    seed = await _reconnect_seed(daemon)
    assert _kinds(seed.transcript) == ["user", "assistant"]
    assert seed.version == 1


@pytest.mark.asyncio
async def test_an_unreadable_config_write_leaves_the_retained_state_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA cell 10's unreadable probe, at the daemon's half of the contract: a
    hand-edit failure fans out nothing, so no cache is dropped and the last
    good value keeps being served — and the next VALID write still converges,
    with no restart."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    await _seed_session(config_dir)

    daemon = MobileDaemon(port=0, password="pw")
    daemon.watch_display_settings()
    epoch = await _retained_route(daemon)

    from local_operator.mobile.daemon import _durable_fold_cache

    # Warm the durable fold so the identity assertions below have a state.
    page, _more = await asyncio.to_thread(_history_page, SESSION, None, 50)
    assert _kinds(page) == ["user", "peer_message", "tool", "assistant"]
    directory = config_dir / "sessions" / SESSION
    state = _durable_fold_cache().get(directory)

    (config_dir / "config.yml").write_text("display: [unclosed\n", encoding="utf-8")
    assert process_watcher(config_dir).poll_now() is None
    assert _durable_fold_cache().get(directory) is state
    assert SESSION in daemon.session_projections

    # The last good snapshot still serves (OFF), from the retained frame.
    seed = await _reconnect_seed(daemon)
    assert _kinds(seed.transcript) == ["user", "peer_message", "tool", "assistant"]
    assert seed.version == epoch

    # The hand-edit is repaired with the flipped value; the flip still lands.
    _write_elsewhere(config_dir, KEY, True)
    assert process_watcher(config_dir).poll_now() is not None
    seed = await _reconnect_seed(daemon)
    assert _kinds(seed.transcript) == ["user", "assistant"]


@pytest.mark.asyncio
async def test_the_follower_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Boot calls it once; a second call must not stack listeners (a stacked
    listener would run the cache drops twice per change for no benefit)."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))

    import local_operator.tui.settings as settings_mod

    reloads: list[int] = []
    real = settings_mod.settings_reload

    def counted() -> None:
        reloads.append(1)
        real()

    monkeypatch.setattr(settings_mod, "settings_reload", counted)

    daemon = MobileDaemon(port=0, password="pw")
    daemon.watch_display_settings()
    daemon.watch_display_settings()

    _write_elsewhere(config_dir, KEY, True)
    assert process_watcher(config_dir).poll_now() is not None
    assert len(reloads) == 1

    # A watcher exists and is the one the daemon subscribed (the attach
    # started it rather than dropping the change on the floor).
    assert existing_watcher(config_dir) is not None
