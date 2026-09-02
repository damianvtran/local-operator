"""A running ``Session`` follows ``config.yml``: the registry's LIVE label is true.

The enforcement behind ``settings_io.Scope.LIVE``. For every key in every
section the registry calls LIVE, a write from a SECOND ``ConfigManager`` (what
another process's edit looks like) followed by one watcher tick must change
the subscribed session's typed view — ``_compaction_settings``,
``routing_settings``, the job cap — or, for the groups that are live because
they are read per use (``fork.*``, ``subagents.models.*``, ``web_*`` knobs),
must be readable from the watcher's snapshot the consumer will switch to.
Then a registry-honesty test closes the loop: every LIVE section has a key
exercised here, and every key exercised here sits in a LIVE section. A section
relabelled LIVE without wiring fails by name, and so does one wired without
being relabelled.

Nothing here waits on the clock: ``poll_now()`` is the tick.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from local_operator import settings_io
from local_operator.compaction.thresholds import CompactionSettings
from local_operator.config import ConfigManager
from local_operator.config_watch import ConfigWatcher, _reset_for_tests, process_watcher
from local_operator.harness.jobs import DEFAULT_MAX_RUNNING_JOBS
from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
)
from local_operator.model.configure import SessionStreamFn
from local_operator.providers.failover import RetrySettings
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


@pytest.fixture(autouse=True)
def _fresh_registry():
    _reset_for_tests()
    yield
    _reset_for_tests()


class RebindableStream:
    """A stream fn exposing the two things ``_apply_config_change`` reads."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self._settings: Any = settings
        self.applied: list[Any] = []

    @property
    def routing_settings(self):
        return self._settings

    def apply_settings(self, values) -> None:
        self.applied.append(values)
        self._settings = values

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        async def gen():
            yield StreamTextDelta(delta="ok")
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def make_session(tmp_path, stream, **kwargs) -> Session:
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda: ["stable"],
        **kwargs,
    )


def write_from_another_process(config_dir, key: str, value: Any) -> None:
    """The shape of an edit made elsewhere: a fresh manager, the facade's merge
    rule, and no in-process notification (``_store`` is below the hook)."""
    setting = settings_io.resolve_key(key)
    assert setting is not None, key
    settings_io._store(ConfigManager(config_dir), setting.path, value)


def _stored(watcher: ConfigWatcher, path: list[str]) -> Any:
    """What a read-per-use consumer sees in the watcher's snapshot (default-aware)."""
    found = settings_io._walk(watcher.values, path)
    return None if found is settings_io._MISSING else found


def subscribe(session: Session, watcher: ConfigWatcher) -> None:
    session.add_dispose_hook(watcher.subscribe(session._apply_config_change))


def compaction_of(session: Session) -> CompactionSettings:
    """The session's typed compaction view, asserted present for the checker."""
    settings = session._compaction_settings
    assert isinstance(settings, CompactionSettings)
    return settings


# ---------------------------------------------------------------------------
# What a non-default value for each LIVE key looks like, and how to observe it
# ---------------------------------------------------------------------------

#: ``key -> (non-default value, observer)``. The observer reads the SESSION's
#: typed view after one tick and returns what it sees; the test compares that
#: to the value written. Keys whose consumer reads config per use (no session
#: attribute to observe) read the watcher's snapshot — the mapping those
#: consumers are being pointed at.
LIVE_KEY_PROBES: dict[str, tuple[Any, Any]] = {
    # -- compaction: the session's typed settings object -----------------------
    "compaction.enabled": (False, lambda s, w: compaction_of(s).enabled),
    "compaction.strategy": ("snapcompact", lambda s, w: compaction_of(s).strategy),
    "compaction.threshold_percent": (0.5, lambda s, w: compaction_of(s).threshold_percent),
    "compaction.threshold_tokens": (123_456, lambda s, w: compaction_of(s).threshold_tokens),
    "compaction.keep_recent_tokens": (
        4_321,
        lambda s, w: compaction_of(s).keep_recent_tokens,
    ),
    "compaction.auto_continue": (False, lambda s, w: compaction_of(s).auto_continue),
    "compaction.mid_turn_enabled": (False, lambda s, w: compaction_of(s).mid_turn_enabled),
    # -- failover: what RetrySettings.from_settings reads off the stream --------
    "retry.enabled": (False, lambda s, w: RetrySettings.from_settings(s.routing_settings).enabled),
    "retry.maxRetries": (
        3,
        lambda s, w: RetrySettings.from_settings(s.routing_settings).max_retries,
    ),
    "retry.baseDelayMs": (
        1_234,
        lambda s, w: RetrySettings.from_settings(s.routing_settings).base_delay_ms,
    ),
    "retry.connectivityMaxRetries": (
        7,
        lambda s, w: RetrySettings.from_settings(s.routing_settings).connectivity_max_retries,
    ),
    "retry.connectivityBackoffCapMs": (
        9_999,
        lambda s, w: RetrySettings.from_settings(s.routing_settings).connectivity_backoff_cap_ms,
    ),
    "retry.modelFallback": (
        False,
        lambda s, w: RetrySettings.from_settings(s.routing_settings).model_fallback,
    ),
    "retry.usageAwareFallback": (
        True,
        lambda s, w: RetrySettings.from_settings(s.routing_settings).usage_aware_fallback,
    ),
    "retry.usageReservePercent": (
        25.0,
        lambda s, w: RetrySettings.from_settings(s.routing_settings).usage_reserve_percent,
    ),
    "retry.fallbackChains": (
        {"default": ["zai/glm-5.3"]},
        lambda s, w: dict(RetrySettings.from_settings(s.routing_settings).fallback_chains),
    ),
    # -- subagents ---------------------------------------------------------------
    "subagents.max_running": (3, lambda s, w: s.jobs.max_running),
    "subagents.models.lo": ("p/lo-model", lambda s, w: _stored(w, ["subagents", "models", "lo"])),
    "subagents.models.med": (
        "p/med-model",
        lambda s, w: _stored(w, ["subagents", "models", "med"]),
    ),
    "subagents.models.hi": ("p/hi-model", lambda s, w: _stored(w, ["subagents", "models", "hi"])),
    # -- read-per-use consumers: the snapshot they are handed ------------------
    "fork.mode": ("here", lambda s, w: _stored(w, ["fork", "mode"])),
    "fork.cmux_placement": ("split", lambda s, w: _stored(w, ["fork", "cmux_placement"])),
    "web_search.strategy": ("ordered", lambda s, w: _stored(w, ["web_search", "strategy"])),
    "web_search.providers": (["brave"], lambda s, w: _stored(w, ["web_search", "providers"])),
    "web_search.timeout_seconds": (5.0, lambda s, w: _stored(w, ["web_search", "timeout_seconds"])),
    "web_search.searxng_endpoint": (
        "http://searx.local",
        lambda s, w: _stored(w, ["web_search", "searxng_endpoint"]),
    ),
    "web_fetch.timeout_seconds": (5.0, lambda s, w: _stored(w, ["web_fetch", "timeout_seconds"])),
    "web_fetch.max_bytes": (1_000, lambda s, w: _stored(w, ["web_fetch", "max_bytes"])),
    "web_fetch.max_redirects": (1, lambda s, w: _stored(w, ["web_fetch", "max_redirects"])),
    "web_fetch.cache_ttl_seconds": (0, lambda s, w: _stored(w, ["web_fetch", "cache_ttl_seconds"])),
    "web_fetch.allow_private": (True, lambda s, w: _stored(w, ["web_fetch", "allow_private"])),
    "web_fetch.render_backend": ("plain", lambda s, w: _stored(w, ["web_fetch", "render_backend"])),
    "web_fetch.enrich": (False, lambda s, w: _stored(w, ["web_fetch", "enrich"])),
}

#: LIVE sections whose keys have no session-side apply because the TUI owns
#: them (``appearance``: ``OperatorApp._on_config_change`` applies
#: ``display.*``/``tui.theme``; covered in the TUI suite).
TUI_OWNED_LIVE_SECTIONS = {"appearance"}


def _live_sections() -> set[str]:
    return {s.name for s in settings_io.SECTIONS if s.scope is settings_io.Scope.LIVE}


@pytest.mark.asyncio
@pytest.mark.parametrize("key", sorted(LIVE_KEY_PROBES))
async def test_a_write_from_another_process_reaches_the_running_session(tmp_path, key) -> None:
    value, observe = LIVE_KEY_PROBES[key]
    setting = settings_io.resolve_key(key)
    assert setting is not None
    assert value != setting.default, f"{key}: the probe must write a NON-default value"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    ConfigManager(config_dir).set_config_value("hosting", "")  # a real file to diff against

    stream = RebindableStream(dict(ConfigManager(config_dir).get_config().values))
    session = make_session(tmp_path, stream, compaction_settings=CompactionSettings())
    watcher = ConfigWatcher(config_dir)
    subscribe(session, watcher)
    try:
        before = observe(session, watcher)
        assert before != value, f"{key}: the session already held the probe value"

        write_from_another_process(config_dir, key, value)
        change = watcher.poll_now()

        assert change is not None and key in change.changed_keys
        assert observe(session, watcher) == value
    finally:
        await session.dispose()


def test_every_live_section_is_exercised_and_every_probe_is_live() -> None:
    """Registry honesty, both directions, failing BY NAME.

    A section relabelled LIVE needs a probe here (or an entry in the explicit
    TUI-owned list); a probe here for a key in a non-LIVE section means the
    code went live and the label did not follow.
    """
    live = _live_sections()
    probed_sections = {
        cast(settings_io.Setting, settings_io.resolve_key(k)).section for k in LIVE_KEY_PROBES
    }
    unprobed = live - probed_sections - TUI_OWNED_LIVE_SECTIONS
    assert not unprobed, f"LIVE sections with no live-apply probe: {sorted(unprobed)}"
    mislabelled = {
        k
        for k in LIVE_KEY_PROBES
        if cast(settings_io.Setting, settings_io.resolve_key(k)).section not in live
    }
    assert not mislabelled, f"probed as live but not in a LIVE section: {sorted(mislabelled)}"
    # And every key of every probed LIVE section is probed, not just one per
    # section: a new key added to a LIVE section must prove it applies.
    for section in probed_sections:
        keys = {s.key for s in settings_io.settings_for(section)}
        missing = keys - set(LIVE_KEY_PROBES)
        assert not missing, f"{section}: LIVE keys without a probe: {sorted(missing)}"


def test_the_deliberately_build_time_keys_stay_new_sessions() -> None:
    """The design keeps these OUT of live on purpose; a future relabel must be
    a decision, not a side effect of the section split."""
    scope_of = {s.name: s.scope for s in settings_io.SECTIONS}
    for key in ("tool_approval_mode", "auto_save_conversation"):
        setting = settings_io.resolve_key(key)
        assert setting is not None
        assert scope_of[setting.section] is settings_io.Scope.NEW_SESSIONS, key
    for key in ("web_search.enabled", "web_fetch.enabled"):
        setting = settings_io.resolve_key(key)
        assert setting is not None
        assert setting.section == "web_tools"
        assert scope_of["web_tools"] is settings_io.Scope.NEW_SESSIONS


# ---------------------------------------------------------------------------
# The three apply paths in detail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_compaction_is_recoerced_into_a_fresh_object(tmp_path) -> None:
    """The read sites hold the attribute, not the object: a pass in flight
    keeps what it captured, the next check sees the new one."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    ConfigManager(config_dir).set_config_value("hosting", "")
    session = make_session(tmp_path, RebindableStream({}), compaction_settings=CompactionSettings())
    watcher = ConfigWatcher(config_dir)
    subscribe(session, watcher)
    try:
        captured = compaction_of(session)
        write_from_another_process(config_dir, "compaction.threshold_percent", 0.5)
        watcher.poll_now()
        assert compaction_of(session) is not captured
        assert captured.threshold_percent == CompactionSettings().threshold_percent
        assert compaction_of(session).threshold_percent == 0.5
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_real_stream_fn_rebinds_and_keeps_its_route_state(tmp_path) -> None:
    """``SessionStreamFn.apply_settings`` on the production class: the mapping
    the per-call ``from_settings`` reads moves, and nothing else does — a
    pinned fallback must survive a threshold edit."""

    class FakeAuth:
        async def get_oauth_access(self, *args, **kwargs):
            return None

    stream = SessionStreamFn(cast(Any, FakeAuth()), {"retry": {"maxRetries": 10}}, "session-x")
    try:
        route_state = stream._route_state
        stream.apply_settings({"retry": {"maxRetries": 2}})
        assert RetrySettings.from_settings(stream.routing_settings).max_retries == 2
        assert stream._route_state is route_state
    finally:
        await stream.close()


@pytest.mark.asyncio
async def test_lowering_max_running_evicts_nothing_and_raising_it_promotes_the_queue(
    tmp_path,
) -> None:
    """Both directions of ``set_max_running`` through the session listener."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    ConfigManager(config_dir).set_config_value("subagents", {"max_running": 1})
    session = make_session(tmp_path, RebindableStream({}))
    watcher = ConfigWatcher(config_dir)
    subscribe(session, watcher)
    try:
        session.jobs.set_max_running(1)
        started: list[str] = []

        async def runner(job_id, signal, on_update):
            started.append(job_id)
            return "done"

        first = session.jobs.register("task", "a", runner)
        assert session.jobs.at_capacity()
        parked = session.jobs.register("task", "b", runner, queued=True)
        assert parked in session.jobs.queued_ids()

        # Lower below the running count: nothing is cancelled.
        write_from_another_process(config_dir, "subagents.max_running", 1)
        write_from_another_process(config_dir, "subagents.max_running", 2)
        watcher.poll_now()
        assert session.jobs.max_running == 2
        # Raising promoted the parked job without waiting for a completion.
        assert parked not in session.jobs.queued_ids()
        assert session.jobs.get(first) is not None
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_unsetting_max_running_restores_the_built_in_default(tmp_path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    ConfigManager(config_dir).set_config_value("subagents", {"max_running": 3})
    session = make_session(tmp_path, RebindableStream({}))
    session.jobs.set_max_running(3)
    watcher = ConfigWatcher(config_dir)
    subscribe(session, watcher)
    try:
        setting = settings_io.resolve_key("subagents.max_running")
        assert setting is not None
        settings_io._delete(ConfigManager(config_dir), setting.path)
        watcher.poll_now()
        assert session.jobs.max_running == DEFAULT_MAX_RUNNING_JOBS
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_disposed_session_ignores_changes_and_unsubscribes(tmp_path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    ConfigManager(config_dir).set_config_value("hosting", "")
    stream = RebindableStream({})
    session = make_session(tmp_path, stream, compaction_settings=CompactionSettings())
    watcher = ConfigWatcher(config_dir)
    subscribe(session, watcher)
    await session.dispose()
    assert watcher._listeners == []
    write_from_another_process(config_dir, "retry.maxRetries", 1)
    watcher.poll_now()
    assert stream.applied == []


@pytest.mark.asyncio
async def test_the_local_fast_path_applies_on_the_writers_own_call_stack(tmp_path) -> None:
    """A ``/settings`` toggle in THIS process reaches THIS session before the
    write facade returns — no poll interval, no loop turn."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    ConfigManager(config_dir).set_config_value("hosting", "")
    session = make_session(tmp_path, RebindableStream({}), compaction_settings=CompactionSettings())
    watcher = process_watcher(config_dir)
    watcher.start()
    subscribe(session, watcher)
    try:
        setting = settings_io.resolve_key("compaction.enabled")
        assert setting is not None
        settings_io.write_setting(ConfigManager(config_dir), setting, False)
        assert compaction_of(session).enabled is False
    finally:
        await session.dispose()
        await watcher.stop()


# ---------------------------------------------------------------------------
# 8. Subagents follow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_subagent_follows_config_and_does_not_alias_its_parents_seed(
    tmp_path, monkeypatch
) -> None:
    """The ``model_copy`` at build is the child's INITIAL value; the
    subscription is what keeps it current. Both must hold: the child moves
    with the file, and the parent's object is not the same object."""
    from local_operator.harness import subagent as subagent_mod

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    ConfigManager(config_dir).set_config_value("hosting", "")

    parent = make_session(
        tmp_path,
        RebindableStream({}),
        compaction_settings=CompactionSettings(threshold_tokens=250_000),
    )
    watcher = process_watcher(config_dir)
    subscribe(parent, watcher)
    child = await subagent_mod._build_child_session(
        label="sub",
        prompt="do the thing",
        parent_session=parent,
        model_spec=None,
        job_id="job-1",
    )
    try:
        assert compaction_of(child) is not compaction_of(parent)
        assert compaction_of(child).threshold_tokens == 250_000
        assert len(watcher._listeners) == 2

        write_from_another_process(config_dir, "compaction.threshold_tokens", 99_000)
        watcher.poll_now()

        assert compaction_of(child).threshold_tokens == 99_000
        assert compaction_of(parent).threshold_tokens == 99_000
        assert compaction_of(child) is not compaction_of(parent)
    finally:
        await subagent_mod._dispose_child(child)
        assert len(watcher._listeners) == 1  # the child unsubscribed itself
        await parent.dispose()
        assert watcher._listeners == []
        await watcher.stop()
