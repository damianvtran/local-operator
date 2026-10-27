"""A model/effort switch costs O(1) on the session loop, whatever the roster's size.

WHY THIS FILE EXISTS. The operator's ``/model did not run: the app waits up to 20
seconds`` came from long manager sessions (a 2.7 MB subagent roster, a 110 MB
transcript). Beyond the connection's op chain (pinned in ``test_server.py``), the
switch itself ran two roster-sized passes on the session loop: the full canonical
``refresh_from_session`` (twice per genuine switch: once directly, once through the
tier-tool rebuild's ``refresh_tools``) and the projection's
``set_subagent_details`` registry pass. Measured with
``scripts/bench_model_switch.py`` at a 1,000-child roster: 19 ms p50 / 133 ms max
per ``refresh_from_session``, ``set_model`` p50 rising from 21 ms to 190 ms.

STRUCTURAL SPIES, not time bounds (AGENTS.md, "Prefer a structural invariant to a
numeric one"): the assertions are about WHICH roster-scaling call runs, which no
host load can flake. The positive halves check the selection fields still land, so
the cheap path cannot pass by publishing nothing.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from local_operator.harness.types import ModelSpec
from local_operator.session.frontend_state import FrontendStateStore
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript

_LADDER = ("low", "medium", "high")
MODEL_A = ModelSpec(
    provider="test",
    model_id="switch-a",
    context_window=100_000,
    reasoning_efforts=_LADDER,
    reasoning_default_effort="medium",
    reasoning_effort="medium",
)
MODEL_B = MODEL_A.model_copy(update={"model_id": "switch-b", "context_window": 64_000})


def _session(tmp_path: Any) -> Session:
    def never(_request: Any, _signal: Any = None) -> Any:  # pragma: no cover - no turn runs
        raise AssertionError("no provider call in this test")

    return Session(
        model=MODEL_A,
        stream_fn=never,
        tools=[],
        transcript=Transcript(tmp_path / "sess"),
        system_blocks_provider=lambda *a, **k: ["stable"],
    )


def _count(monkeypatch: pytest.MonkeyPatch, owner: Any, name: str) -> list[int]:
    calls = [0]
    original = getattr(owner, name)

    def spy(self: Any, *a: Any, **kw: Any) -> Any:
        calls[0] += 1
        return original(self, *a, **kw)

    monkeypatch.setattr(owner, name, spy)
    return calls


async def _drain(rounds: int = 5) -> None:
    """Let the switch's background work run: the ``ModelChangeEvent`` emit
    (``_spawn_background``) and the scheduled access-claim publish."""
    for _ in range(rounds):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_session_set_model_publishes_the_selection_without_a_full_rescan(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every ``Session.set_model`` arm publishes in O(1) and still publishes,
    INCLUDING the ``ModelChangeEvent`` the switch emits, with a viewer attached.

    The subscriber matters (review round 1, R1-1): ``Session._emit`` skips
    ``observe_event`` when nobody is subscribed to the store, and the event's
    full rescan was invisible to a test that never attached one — while an
    attached desktop always is. Fails on the old code: a genuine switch called
    ``refresh_from_session`` twice directly and once more from the event.
    """
    session = _session(tmp_path)
    try:
        session._frontend_state_store.subscribe(lambda *_a, **_k: None)
        rescans = _count(monkeypatch, FrontendStateStore, "refresh_from_session")

        session.set_model(MODEL_B, explicit=True)  # genuine switch: emits the event
        session.set_model(
            MODEL_B.model_copy(update={"reasoning_effort": "high"}), effort_choice="high"
        )  # same pair: the /effort path
        await _drain()

        assert rescans[0] == 0, "a switch, or its model_change event, must not rescan"
        state = session._frontend_state_store.state
        assert state.selected_model is not None
        assert state.selected_model.model_id == "switch-b"
        assert state.selected_model.reasoning_effort == "high"
        assert state.context_window == 64_000
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_serving_switch_skips_the_roster_pass_and_reads_credentials_off_loop(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``set_model_effort``/``set_effort`` on the real handle, SUBSCRIBED the way
    the runtime server subscribes it, with the switch's event drained: no roster
    pass, no full rescan, exactly ONE credential read, and none on the loop.

    Fails on the old code: ``_refresh_state()`` ran ``set_subagent_details`` on
    every switch and ``publish_model_access`` read SQLite on the loop. Fails on
    this PR's first head (review round 1, R1-1): the event handler re-read the
    store synchronously on the loop, beside the off-loop read — two reads, one
    of them on the loop thread.
    """
    import threading

    from local_operator.mobile.projection import ProjectionFold
    from local_operator.providers import model_access

    session = _session(tmp_path)
    loop = asyncio.get_running_loop()
    handle = ServingSessionHandle(session, loop, cwd=str(tmp_path), config_dir=tmp_path)
    unsubscribe = handle.subscribe(lambda: None)
    session._frontend_state_store.subscribe(lambda *_a, **_k: None)
    try:
        roster = _count(monkeypatch, ProjectionFold, "set_subagent_details")
        rescans = _count(monkeypatch, FrontendStateStore, "refresh_from_session")
        loop_thread = threading.get_ident()
        read_threads: list[int] = []
        original = model_access.serving_model_access_here

        def recording(*a: Any, **kw: Any) -> Any:
            read_threads.append(threading.get_ident())
            return original(*a, **kw)

        monkeypatch.setattr(model_access, "serving_model_access_here", recording)
        monkeypatch.setattr(
            "local_operator.model.configure.build_model_spec",
            lambda provider, model_id, *a, **k: MODEL_B.model_copy(
                update={"provider": provider, "model_id": model_id}
            ),
        )

        await handle.set_model_effort("test", "switch-b", "high")
        await _drain()
        # The reply does not wait for the claim; wait for the read it scheduled.
        # ``getattr`` so the negative control against older code fails on the
        # read count below, not on a missing attribute.
        pending = getattr(handle, "_model_access_publish", None)
        if pending is not None:
            await pending
        await handle.set_effort("low")
        await _drain()

        assert roster[0] == 0, "a switch must not run the roster registry pass"
        assert rescans[0] == 0, "the switch's model_change event must not rescan"
        assert len(read_threads) == 1, f"one credential read per switch, got {read_threads}"
        assert loop_thread not in read_threads, "the credential read ran on the session loop"
        assert handle._projection.model_label.endswith("switch-b")
        assert session.model.reasoning_effort == "low"
    finally:
        unsubscribe()
        await handle.dispose()
        await session.dispose()


@pytest.mark.asyncio
async def test_a_warm_spec_builds_inline_and_a_cold_one_leaves_the_default_executor(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spec build never queues on the loop's DEFAULT executor (Q1-1).

    A warm model (its metadata already memoised) builds on the loop thread, with
    no hop at all; a cold one builds on the switch's own ``lop-switch`` pool.
    Fails on the old code: every build went through ``asyncio.to_thread``, the
    pool the busy lanes' blocking work saturates.
    """
    import threading

    from local_operator.model import configure

    session = _session(tmp_path)
    loop = asyncio.get_running_loop()
    handle = ServingSessionHandle(session, loop, cwd=str(tmp_path), config_dir=tmp_path)
    try:
        build_threads: list[str] = []

        def build(provider: str, model_id: str, *a: Any, **k: Any) -> ModelSpec:
            build_threads.append(threading.current_thread().name)
            return MODEL_B.model_copy(update={"provider": provider, "model_id": model_id})

        monkeypatch.setattr(configure, "build_model_spec", build)
        warm = {"switch-b"}
        monkeypatch.setattr(
            configure, "model_spec_is_warm", lambda _provider, model_id: model_id in warm
        )

        await handle.set_model_effort("test", "switch-b", None)
        await handle.set_model_effort("test", "switch-cold", None)

        assert build_threads[0] == threading.current_thread().name, build_threads
        assert build_threads[1].startswith("lop-switch"), build_threads
        assert session.model.model_id == "switch-cold"
    finally:
        await handle.dispose()
        await session.dispose()


def test_model_spec_is_warm_only_after_a_full_resolve_in_this_bucket() -> None:
    """The warm predicate answers from the full resolver's own memo, and refuses
    every provider whose resolution reads a credential store (``openai``)."""
    from local_operator.model import configure

    configure.invalidate_model_info_cache()
    try:
        assert not configure.model_spec_is_warm("anthropic", "claude-opus-5")
        configure.resolve_model_info("anthropic", "claude-opus-5")
        assert configure.model_spec_is_warm("anthropic", "claude-opus-5")
        # Canonicalised like the build: the hosting-qualified spelling is the
        # same key.
        assert configure.model_spec_is_warm("anthropic", "anthropic/claude-opus-5")
        assert not configure.model_spec_is_warm("openai", "gpt-5")
    finally:
        configure.invalidate_model_info_cache()
