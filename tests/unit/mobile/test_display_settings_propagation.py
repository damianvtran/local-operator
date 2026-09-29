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
"""

from __future__ import annotations

import asyncio
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
    _durable_projection,
    _history_page,
)
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
