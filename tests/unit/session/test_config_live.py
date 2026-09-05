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

import inspect
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
from local_operator.model.configure import (
    SessionStreamFn,
    _anthropic_cache_ttl_1h_min_context_tokens,
    _openai_api_mode,
    _openai_use_max_context_window,
)
from local_operator.providers.failover import RetrySettings
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.spawn.policy import fork_cmux_placement, fork_mode

MODEL = ModelSpec(provider="test", model_id="m", context_window=100_000)


@pytest.fixture(autouse=True)
def _fresh_registry():
    _reset_for_tests()
    yield
    _reset_for_tests()


class FakeAuthStore:
    """Just enough auth store to observe the PUSHED setting.

    ``retry.usageAwareAccountPick`` is the one ``retry.*`` key the cascade does
    not read back off the settings mapping: the real ``AuthStore`` copies it
    into ``_usage_aware_pick`` when the stream fn pushes it. A probe that read
    the mapping instead would pass while the cascade still used the old value,
    which is precisely the gap review round 2 (B1) found — so the double keeps
    its own copy, exactly as the real store does.
    """

    def __init__(self) -> None:
        self._usage_aware_pick = True  # the real store defaults ON

    def configure_usage_aware_pick(self, enabled: bool) -> None:
        self._usage_aware_pick = bool(enabled)


class RebindableStream:
    """A stream fn exposing the two things ``_apply_config_change`` reads."""

    def __init__(self, settings: dict[str, Any]) -> None:
        self._settings: Any = settings
        self.applied: list[Any] = []
        self.auth_store = FakeAuthStore()

    @property
    def routing_settings(self):
        return self._settings

    def apply_settings(self, values) -> None:
        self.applied.append(values)
        self._settings = values
        # Mirrors the real ``SessionStreamFn.apply_settings``, which re-pushes
        # this one key rather than relying on the rebind.
        from local_operator.providers.failover import RetrySettings

        self.auth_store.configure_usage_aware_pick(
            RetrySettings.from_settings(values).usage_aware_account_pick
        )

    def __call__(self, request: ChatRequest, signal: AbortSignal | None):
        async def gen():
            yield StreamTextDelta(delta="ok")
            yield StreamEndEvent(stop_reason="stop")

        return gen()


def make_session(tmp_path, stream, **kwargs) -> Session:
    # Both web tools OFFERED at build, the way the factory builds a session on
    # default config, so the ``web_*.enabled`` probes can observe a disable as
    # the tool leaving the inventory at the next turn boundary.
    from local_operator.harness.types import ToolContext
    from local_operator.tools.registry import create_tools

    tools = kwargs.pop(
        "tools",
        create_tools(
            ToolContext(
                cwd=str(tmp_path),
                web_search_settings={"enabled": True},
                web_fetch_settings={"enabled": True},
            ),
            enabled=("web_search", "web_fetch"),
        ),
    )
    return Session(
        model=MODEL,
        stream_fn=stream,
        tools=tools,
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


def _search_settings(watcher: ConfigWatcher):
    """What `execute_web_search` builds per call, from the watcher's directory."""
    from local_operator.web_search.tool import load_search_settings

    return load_search_settings(ConfigManager(watcher.config_dir))


def _fetch_settings(watcher: ConfigWatcher):
    """What the fetch engine builds per call, from the watcher's directory."""
    from local_operator.web_fetch.tool import load_fetch_settings

    return load_fetch_settings(ConfigManager(watcher.config_dir))


def _spawn_model(session: Session, tier: str, watcher: ConfigWatcher | None = None) -> str:
    """The subagent ModelSpec the session resolves for an effort tier.

    `_resolve_subagent_model` returns None when config names no model for the
    tier, which is "inherit the parent" rather than a value — rendered as the
    empty string so a probe comparing to a `provider/model` string reads a
    miss as a miss instead of as an exception.
    """
    spec = session._resolve_subagent_model("task", tier)
    return "" if spec is None else f"{spec.provider}/{spec.model_id}"


def subscribe(session: Session, watcher: ConfigWatcher) -> None:
    session.add_dispose_hook(watcher.subscribe(session._apply_config_change))


def compaction_of(session: Session) -> CompactionSettings:
    """The session's typed compaction view, asserted present for the checker."""
    settings = session._compaction_settings
    assert isinstance(settings, CompactionSettings)
    return settings


async def _settled_model(session: Session) -> ModelSpec:
    """``session.model`` after the background model adopt has landed.

    ``_apply_config_change`` spawns the switch (``build_model_spec`` may hit
    the network for a real provider), so the observer AWAITS the session's
    background tasks rather than the clock. The probe table's other observers
    are sync; the live-apply test awaits whatever an observer returns.
    """
    import asyncio

    pending = [task for task in session._background_tasks if not task.done()]
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    return session.model


async def _settled_model_attr(session: Session, attr: str) -> Any:
    return getattr(await _settled_model(session), attr)


def _web_tool_offered(session: Session, name: str) -> bool:
    """Whether ``name`` is in the inventory after the next turn-boundary reconcile.

    Calls the same hook ``_run_turn`` calls, so the probe measures the real
    path rather than the dirty flag. The probe value is ``False`` (disabled),
    so the session under test must START with the tool offered — see
    ``_web_session`` for how the table's session is built.
    """
    session._reconcile_web_tools()
    return any(tool.name == name for tool in session._tools)


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
    # The two BYTE knobs. LIVE for the same reason as their neighbours — the
    # session re-coerces CompactionSettings on every config change — and it
    # matters more here: an operator raising the budget is usually reacting to
    # a session that is shedding right now, and a NEW_SESSIONS scope would make
    # them restart to get the frames back.
    "compaction.wire_bytes_budget": (
        12_345_678,
        lambda s, w: compaction_of(s).wire_bytes_budget,
    ),
    "compaction.wire_bytes_trigger": (
        8_765_432,
        lambda s, w: compaction_of(s).wire_bytes_trigger,
    ),
    # -- providers: read off the rebound mapping at the NEXT client build ------
    "providers.anthropic.cache_ttl_1h_min_context_tokens": (
        999_999,
        lambda s, w: _anthropic_cache_ttl_1h_min_context_tokens(s.routing_settings),
    ),
    # Observed through the same function the client builder calls, not through
    # the raw mapping: what makes this key live is that ``_openai_api_mode``
    # reads the REBOUND settings, so that is what must move.
    "providers.openai.use_max_context_window": (
        False,
        lambda s, w: _openai_use_max_context_window(s.routing_settings),
    ),
    "providers.openai.api": (
        "chat_completions",
        lambda s, w: _openai_api_mode(s.routing_settings),
    ),
    # -- failover: what RetrySettings.from_settings reads off the stream --------
    # NOTE the odd one out below: `retry.usageAwareAccountPick` is observed on
    # the auth STORE, not the mapping, because the store holds its own copy.
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
    # Observed on the auth STORE, deliberately (review round 2, B1). Every
    # other `retry.*` key is live because `from_settings` re-reads the mapping
    # per call; this one is pushed into the store, which then owns the value.
    # Probing the mapping here would have passed while the cascade kept using
    # the stale flag — the exact false-green the finding described.
    "retry.usageAwareAccountPick": (
        False,
        lambda s, w: s._stream_fn.auth_store._usage_aware_pick,
    ),
    # -- subagents ---------------------------------------------------------------
    "subagents.max_running": (3, lambda s, w: s.jobs.max_running),
    "subagents.models.lo": ("openai/lo-model", lambda s, w: _spawn_model(s, "lo")),
    "subagents.models.med": ("openai/med-model", lambda s, w: _spawn_model(s, "med")),
    "subagents.models.hi": ("openai/hi-model", lambda s, w: _spawn_model(s, "hi")),
    # -- read-per-use consumers: observed THROUGH the consumer ------------------
    # Not through the watcher's snapshot (review round 3, M2). Asserting that a
    # key reached `watcher.values` is bookkeeping: it cannot tell "this key is
    # live" from "this key is parsed and nothing reads it", which is exactly the
    # blind spot that let M3 and round-2's B1 through. These consumers are all
    # genuinely read-per-use today, and the point of routing the probe through
    # them is that the guard notices the day one of them starts caching at
    # construction — the very change the deferred web-tools follow-up proposes.
    "fork.mode": ("window", lambda s, w: fork_mode(w.values)),
    "fork.cmux_placement": ("surface", lambda s, w: fork_cmux_placement(w.values)),
    "web_search.strategy": ("ordered", lambda s, w: _search_settings(w).strategy),
    "web_search.providers": (["brave"], lambda s, w: list(_search_settings(w).providers)),
    "web_search.timeout_seconds": (5.0, lambda s, w: _search_settings(w).timeout_seconds),
    "web_search.searxng_endpoint": (
        "http://searx.local",
        lambda s, w: _search_settings(w).searxng_endpoint,
    ),
    "web_fetch.timeout_seconds": (5.0, lambda s, w: _fetch_settings(w).timeout_seconds),
    "web_fetch.max_bytes": (1_048_576, lambda s, w: _fetch_settings(w).max_bytes),
    "web_fetch.max_redirects": (1, lambda s, w: _fetch_settings(w).max_redirects),
    "web_fetch.cache_ttl_seconds": (0, lambda s, w: _fetch_settings(w).cache_ttl_seconds),
    "web_fetch.allow_private": (True, lambda s, w: _fetch_settings(w).allow_private),
    "web_fetch.render_backend": ("stdlib", lambda s, w: _fetch_settings(w).render_backend),
    "web_fetch.enrich": (False, lambda s, w: _fetch_settings(w).enrich),
    # -- model: the session's OWN spec, after the background adopt settles -----
    # Observed on ``session.model``, not the snapshot: what makes these keys
    # live is that a config-sourced session SWITCHES. ``model_name`` probes
    # against the ``test`` provider the session boots on; ``hosting`` moves to
    # ``openai`` with no ``model_name`` set, which exercises the
    # default-model fallback and resolves from the static registry (verified
    # with sockets blocked: ~4 ms, no network). The observer awaits the
    # background adopt so the switch has landed before it reads.
    "hosting": ("openai", lambda s, w: _settled_model_attr(s, "provider")),
    "model_name": ("probe-model", lambda s, w: _settled_model_attr(s, "model_id")),
    # -- web_tools: the inventory after the next turn boundary -----------------
    # Observed through the SAME reconcile the turn start runs, on a session
    # whose inventory starts with both tools, so a disable is seen as the tool
    # leaving. The per-call gate inside the tools is covered separately.
    "web_search.enabled": (False, lambda s, w: _web_tool_offered(s, "web_search")),
    "web_fetch.enabled": (False, lambda s, w: _web_tool_offered(s, "web_fetch")),
}

#: LIVE sections whose keys have no session-side apply because a HOST owns
#: them (``appearance``: ``OperatorApp._on_config_change`` applies
#: ``display.*``/``tui.theme``; covered in the TUI suite).
#:
#: ``approvals`` is here because the approval MODE lives in the host's gate,
#: not in the ``Session``: the runtime's ``OwnedSessionHandle._auto_approve``
#: (``tests/unit/session/runtime/test_owned_approvals_live.py``) and the
#: TUI's ``_approve_all`` (``tests/unit/tui/test_config_change_notice.py``).
#: The session only holds whatever gate closure the host installed.
#:
#: ``runtime`` is here for the same reason and a sharper one: its keys are
#: read at COMMAND time by the surface that acts on them, so there is no
#: session attribute for a probe to observe moving.
#: ``runtime.background_on_resume`` is read by ``/resume`` in the TUI when it
#: decides what to do with the session being left, and
#: ``runtime.unattended_gate_timeout`` is read by the RUNTIME's gate at the
#: moment a question parks — a different process from the ``Session`` this
#: file drives. Both are covered where they live: the resume behaviour in the
#: TUI suite, the gate policy in ``tests/unit/session/runtime/test_parked_gates.py``.
HOST_OWNED_LIVE_SECTIONS = {"appearance", "runtime", "approvals"}


def _live_sections() -> set[str]:
    return {s.name for s in settings_io.SECTIONS if s.scope is settings_io.Scope.LIVE}


@pytest.mark.asyncio
@pytest.mark.parametrize("key", sorted(LIVE_KEY_PROBES))
async def test_a_write_from_another_process_reaches_the_running_session(
    tmp_path, key, monkeypatch
) -> None:
    value, observe = LIVE_KEY_PROBES[key]
    setting = settings_io.resolve_key(key)
    assert setting is not None
    assert value != setting.default, f"{key}: the probe must write a NON-default value"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    # Some consumers resolve their own directory through `paths.config_dir()`
    # rather than taking one (`Session._resolve_subagent_model` builds its own
    # `ConfigManager`). Pointing the env at the watched directory is what makes
    # those probes read the file under test instead of the operator's real one.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    # A real file to diff against. ``hosting`` is seeded to the session's own
    # provider (``MODEL`` is ``test/m``) so the ``model_name`` probe exercises
    # the common case — a bare ``model_name`` edit with ``hosting`` unchanged —
    # and resolves to a provider that needs no network.
    ConfigManager(config_dir).set_config_value("hosting", MODEL.provider)

    stream = RebindableStream(dict(ConfigManager(config_dir).get_config().values))
    session = make_session(tmp_path, stream, compaction_settings=CompactionSettings())
    # Through the process registry, not a bare ``ConfigWatcher``: the web-tool
    # reconcile reads ``existing_watcher()`` for its snapshot, which is the
    # contract the module docstring imposes on every listener-side consumer.
    watcher = process_watcher(config_dir)
    subscribe(session, watcher)

    async def observed() -> Any:
        result = observe(session, watcher)
        return await result if inspect.isawaitable(result) else result

    try:
        before = await observed()
        assert before != value, f"{key}: the session already held the probe value"

        write_from_another_process(config_dir, key, value)
        change = watcher.poll_now()

        assert change is not None and key in change.changed_keys
        assert await observed() == value
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
    unprobed = live - probed_sections - HOST_OWNED_LIVE_SECTIONS
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


#: Non-LIVE keys whose write DOES reach the session, with the reason that is
#: legitimate. Kept explicit so the guard below fails by name on a new one
#: rather than being widened silently.
#:
#: ``subagents.*`` and ``web_*.enabled`` cannot appear here — both are LIVE
#: now, so the honesty test above covers them from the other direction.
_NON_LIVE_KEYS_ALLOWED_TO_APPLY: dict[str, str] = {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key",
    sorted(
        s.key
        for s in settings_io.SETTINGS
        if s.section not in _live_sections() and s.kind is not settings_io.Kind.READONLY
    ),
)
async def test_a_non_live_key_does_not_quietly_apply_to_a_running_session(tmp_path, key) -> None:
    """The direction the other honesty test cannot see (review round 1, M3).

    ``test_every_live_section_is_exercised_and_every_probe_is_live`` walks LIVE
    sections DOWNWARD: every LIVE section must have a key that provably applies.
    Nothing walked UPWARD — a key with a real live apply sitting in a NEW_LAUNCH
    section passed both directions of that test, which is exactly how
    ``providers.openai.api`` came to be applied live while the notice told the
    user it "takes effect on /new". A label that overstates is a nuisance; one
    that UNDERSTATES is the painted lie, because the user acts on it.

    Asserted BEHAVIOURALLY rather than by reading the source: write the key
    from a second ``ConfigManager``, tick, and require that the session's
    observable state did not move. That catches a future apply added anywhere
    in ``_apply_config_change``'s fan-out, not just a string this test knows to
    grep for.
    """
    setting = settings_io.resolve_key(key)
    assert setting is not None
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    ConfigManager(config_dir).set_config_value("hosting", "")

    stream = RebindableStream(dict(ConfigManager(config_dir).get_config().values))
    session = make_session(tmp_path, stream, compaction_settings=CompactionSettings())
    watcher = ConfigWatcher(config_dir)
    subscribe(session, watcher)

    def observe() -> tuple[Any, ...]:
        """Everything a non-LIVE key must NOT move.

        `retry.max_retries` is excluded deliberately: the co-write below moves
        it on purpose, so including it would make every case fail. The provider
        readers ARE included — without them the guard could not see M6, whose
        whole shape is a `providers.*` value riding along on the mapping rebind
        that a `retry.*` edit triggers. Observed through the same functions the
        client builder calls, not off the raw mapping.
        """
        retry = RetrySettings.from_settings(session.routing_settings)
        return (
            compaction_of(session).model_dump(),
            {k: v for k, v in retry.__dict__.items() if k != "max_retries"},
            session.jobs.max_running,
            _openai_api_mode(session.routing_settings),
            _anthropic_cache_ttl_1h_min_context_tokens(session.routing_settings),
            stream.auth_store._usage_aware_pick,
        )

    try:
        before = observe()
        # A value that differs from whatever is stored, typed to the setting.
        # Branched rather than built as a dict literal: a dict evaluates EVERY
        # value eagerly, so the INT arm ran `int("ask")` for an ENUM key.
        probe: Any
        if setting.kind is settings_io.Kind.ENUM:
            others = [c.value for c in setting.resolved_choices if c.value != setting.default]
            if not others:
                pytest.skip(f"{key} has a single choice; nothing to change it to")
            probe = others[0]
        elif setting.kind is settings_io.Kind.BOOL:
            probe = not bool(setting.default)
        elif setting.kind is settings_io.Kind.INT:
            probe = int(setting.default or 0) + 7
        elif setting.kind is settings_io.Kind.FLOAT:
            probe = float(setting.default or 0) + 3.5
        elif setting.kind is settings_io.Kind.TEXT:
            probe = "probe-value"
        elif setting.kind is settings_io.Kind.LIST:
            probe = ["probe"]
        elif setting.kind is settings_io.Kind.CASCADE:
            probe = {"default": ["probe/model"]}
        else:
            pytest.skip(f"{key}: no probe value for {setting.kind}")

        # Written TOGETHER with a LIVE `retry.*` key, in one tick (review round
        # 2, M6). Writing the non-LIVE key alone was the guard's blind spot:
        # the session rebinds the whole settings mapping when a `retry.*` key
        # moves, so a non-LIVE key sharing that mapping applies as a side
        # effect of its neighbour — invisible to a case that only ever changes
        # one key. The co-write reproduces the real shape (a `/settings` page
        # or an editor saving several keys at once) and is strictly stronger:
        # anything the single write caught, this catches too.
        write_from_another_process(config_dir, key, probe)
        live_probe = 1 + int(RetrySettings.from_settings(session.routing_settings).max_retries)
        write_from_another_process(config_dir, "retry.maxRetries", live_probe)
        change = watcher.poll_now()
        assert (
            change is not None and key in change.changed_keys
        ), f"{key}: the watcher did not see the write, so this test proved nothing"
        assert "retry.maxRetries" in change.changed_keys, (
            "the co-written LIVE key did not land in the same tick, so the "
            "mapping-rebind path this case exists to exercise never fired"
        )

        after = observe()
        if key in _NON_LIVE_KEYS_ALLOWED_TO_APPLY:
            return
        assert after == before, (
            f"{key} is in section {setting.section!r} (not LIVE), so the config-change "
            f"notice tells the user it 'takes effect on /new' — but writing it MOVED the "
            f"running session. Either give it a LIVE section (as providers.openai.api "
            f"got) or stop applying it."
        )
    finally:
        await session.dispose()


def test_the_scope_of_every_reclassified_key_is_the_one_its_consumer_earns() -> None:
    """The reversal of the original build-time design, pinned as a DECISION.

    ``tool_approval_mode``, ``hosting``/``model_name`` and ``web_*.enabled``
    were kept out of LIVE on purpose and are now IN it on purpose: the gate,
    the model and the web tools all follow the file in a running session. A
    future relabel in either direction must be a decision, not a side effect
    of a section split. ``auto_save_conversation`` and ``session.cleanup.*``
    go the other way — nothing reads them after process start, so the honest
    label is NEW_LAUNCH, not the "new sessions" the old section claimed.
    """
    scope_of = {s.name: s.scope for s in settings_io.SECTIONS}
    for key, section in (
        ("tool_approval_mode", "approvals"),
        ("hosting", "model"),
        ("model_name", "model"),
        ("web_search.enabled", "web_tools"),
        ("web_fetch.enabled", "web_tools"),
    ):
        setting = settings_io.resolve_key(key)
        assert setting is not None
        assert setting.section == section, key
        assert scope_of[section] is settings_io.Scope.LIVE, key
    for key in ("auto_save_conversation", "session.cleanup.enabled"):
        setting = settings_io.resolve_key(key)
        assert setting is not None
        assert setting.section == "session", key
        assert scope_of["session"] is settings_io.Scope.NEW_LAUNCH, key


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


# ---------------------------------------------------------------------------
# 9. The model rule: follow the file iff the file chose the model
# ---------------------------------------------------------------------------


#: Events captured per session by ``_capture_events``, keyed by id.
_EVENTS: dict[int, list[Any]] = {}


def _notices(session: Session) -> list[str]:
    """Texts of every ``NoticeEvent`` the session emitted, via a real subscriber."""
    return [getattr(e, "text", "") for e in _EVENTS[id(session)] if e.type == "notice"]


def _capture_events(session: Session) -> None:
    events: list[Any] = []
    _EVENTS[id(session)] = events
    session.subscribe(events.append)


async def _settle(session: Session) -> None:
    import asyncio

    for _ in range(3):
        pending = [t for t in session._background_tasks if not t.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_a_model_name_edit_alone_switches_a_config_sourced_session(
    tmp_path, monkeypatch
) -> None:
    """The operator's exact report: ``/model default`` (or ``lop config edit
    model_name``) in one pane printed "model_name needs a relaunch" in every
    other. A bare ``model_name`` diff, ``hosting`` untouched, is the common
    shape and must switch on its own — the pair is re-read as a whole."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    ConfigManager(config_dir).set_config_value("hosting", MODEL.provider)
    session = make_session(tmp_path, RebindableStream({}), compaction_settings=CompactionSettings())
    _capture_events(session)
    watcher = process_watcher(config_dir)
    subscribe(session, watcher)
    try:
        write_from_another_process(config_dir, "model_name", "m-next")
        change = watcher.poll_now()
        assert change is not None and change.changed_keys == {"model_name"}
        await _settle(session)
        assert (session.model.provider, session.model.model_id) == ("test", "m-next")
        # The receipt names both ends and the cause, and comes from the session.
        assert any("test/m → test/m-next" in n and "config.yml" in n for n in _notices(session))
        # Adopting the file is not the user choosing: a SECOND edit applies too.
        write_from_another_process(config_dir, "model_name", "m-after")
        watcher.poll_now()
        await _settle(session)
        assert session.model.model_id == "m-after"
        # And the boot selector re-based, so the journal row for this switch is
        # skipped on resume (the new default IS the boot).
        assert session._boot_selector == "test/m-after"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_explicit_model_choice_keeps_its_model_and_says_so(tmp_path, monkeypatch) -> None:
    """A session the user pointed somewhere with ``/model`` is not moved by a
    default edit; it prints the keep notice naming ``/model saved``."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    ConfigManager(config_dir).set_config_value("hosting", MODEL.provider)
    session = make_session(tmp_path, RebindableStream({}), compaction_settings=CompactionSettings())
    _capture_events(session)
    watcher = process_watcher(config_dir)
    subscribe(session, watcher)
    try:
        session.set_model(MODEL.model_copy(update={"model_id": "chosen"}), explicit=True)
        assert session._explicit_model_choice is True

        write_from_another_process(config_dir, "model_name", "m-next")
        watcher.poll_now()
        await _settle(session)
        assert session.model.model_id == "chosen"
        keep = [n for n in _notices(session) if "keeping test/chosen" in n]
        assert keep and "chosen with /model" in keep[0] and "/model saved" in keep[0], _notices(
            session
        )
    finally:
        await session.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("source, phrase", [("agent", "an agent profile"), ("flag", "a flag")])
async def test_an_agent_or_flag_sourced_session_gets_a_notice_only(
    tmp_path, monkeypatch, source: str, phrase: str
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    ConfigManager(config_dir).set_config_value("hosting", MODEL.provider)
    session = make_session(
        tmp_path,
        RebindableStream({}),
        compaction_settings=CompactionSettings(),
        model_source=source,
    )
    _capture_events(session)
    watcher = process_watcher(config_dir)
    subscribe(session, watcher)
    try:
        write_from_another_process(config_dir, "model_name", "m-next")
        watcher.poll_now()
        await _settle(session)
        assert session.model.model_id == "m"
        assert any(phrase in n and "keeping test/m" in n for n in _notices(session)), _notices(
            session
        )
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_child_session_never_follows_the_default_model(tmp_path, monkeypatch) -> None:
    """Silent, not even a notice: the child's spec was chosen at spawn and a
    review child losing its cache prefix mid-task is pure cost."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    ConfigManager(config_dir).set_config_value("hosting", MODEL.provider)
    session = make_session(
        tmp_path,
        RebindableStream({}),
        compaction_settings=CompactionSettings(),
        job_id="job-1",
        model_source="child",
    )
    _capture_events(session)
    watcher = process_watcher(config_dir)
    subscribe(session, watcher)
    try:
        write_from_another_process(config_dir, "model_name", "m-next")
        watcher.poll_now()
        await _settle(session)
        assert session.model.model_id == "m"
        assert _notices(session) == []
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_unknown_provider_in_config_warns_and_does_not_switch(
    tmp_path, monkeypatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    ConfigManager(config_dir).set_config_value("hosting", MODEL.provider)
    session = make_session(tmp_path, RebindableStream({}), compaction_settings=CompactionSettings())
    _capture_events(session)
    watcher = process_watcher(config_dir)
    subscribe(session, watcher)
    try:
        write_from_another_process(config_dir, "hosting", "no-such-provider")
        watcher.poll_now()
        await _settle(session)
        assert (session.model.provider, session.model.model_id) == ("test", "m")
        warnings = [
            e
            for e in _EVENTS[id(session)]
            if e.type == "notice" and getattr(e, "kind", "") == "warning"
        ]
        assert warnings and "unknown provider 'no-such-provider'" in warnings[0].text
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_writing_pane_is_a_no_op(tmp_path, monkeypatch) -> None:
    """``/model default p/id`` in THIS session: the runtime already switched
    (explicit) and the disk change names the model it is on. No receipt owed
    — the `/model default` receipt was it — and no journal churn."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    ConfigManager(config_dir).set_config_value("hosting", MODEL.provider)
    session = make_session(tmp_path, RebindableStream({}), compaction_settings=CompactionSettings())
    _capture_events(session)
    watcher = process_watcher(config_dir)
    subscribe(session, watcher)
    try:
        write_from_another_process(config_dir, "model_name", "m")
        change = watcher.poll_now()
        assert change is not None
        await _settle(session)
        assert session.model.model_id == "m"
        assert _notices(session) == []
    finally:
        await session.dispose()


# ---------------------------------------------------------------------------
# 10. Web tools: inventory at the turn boundary, never mid-turn
# ---------------------------------------------------------------------------


def _offered(session: Session) -> set[str]:
    return {t.name for t in session._tools}


@pytest.mark.asyncio
async def test_web_tools_leave_and_return_at_the_turn_boundary(tmp_path, monkeypatch) -> None:
    """The tick only marks dirty; the inventory moves when the next turn
    starts (``_reconcile_web_tools`` is what ``_run_turn`` calls). Both
    directions, and a re-enabled tool comes back through the same createIf
    builder the factory used."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    ConfigManager(config_dir).set_config_value("hosting", "")
    session = make_session(tmp_path, RebindableStream({}), compaction_settings=CompactionSettings())
    watcher = process_watcher(config_dir)
    subscribe(session, watcher)
    try:
        assert {"web_search", "web_fetch"} <= _offered(session)

        write_from_another_process(config_dir, "web_search.enabled", False)
        watcher.poll_now()
        # Mid-turn view: still offered (a call already emitted must not 404).
        assert "web_search" in _offered(session)
        assert session._web_tools_dirty is True

        session._reconcile_web_tools()
        assert "web_search" not in _offered(session)
        assert "web_fetch" in _offered(session)
        assert session._web_tools_dirty is False

        write_from_another_process(config_dir, "web_search.enabled", True)
        write_from_another_process(config_dir, "web_fetch.enabled", False)
        watcher.poll_now()
        session._reconcile_web_tools()
        assert "web_search" in _offered(session)
        assert "web_fetch" not in _offered(session)
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_child_keeps_its_spawn_inventory(tmp_path, monkeypatch) -> None:
    """Per-call gate only for a child: no dirty mark, no reconcile."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    ConfigManager(config_dir).set_config_value("hosting", "")
    session = make_session(
        tmp_path,
        RebindableStream({}),
        compaction_settings=CompactionSettings(),
        job_id="job-1",
        model_source="child",
    )
    watcher = process_watcher(config_dir)
    subscribe(session, watcher)
    try:
        write_from_another_process(config_dir, "web_search.enabled", False)
        watcher.poll_now()
        assert session._web_tools_dirty is False
        session._reconcile_web_tools()
        assert "web_search" in _offered(session)
    finally:
        await session.dispose()
