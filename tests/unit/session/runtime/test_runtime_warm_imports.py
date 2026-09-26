"""The runtime child pays the composition root's imports OFF its own loop.

``create_session`` is a coroutine whose body is one long SYNCHRONOUS stretch, so
the loop that calls it is frozen for the whole of it. In the runtime child that
loop is the one the operator waits through between sending a message and the
session starting, and the stretch measures 630.4 ms of contiguous stall against
a heartbeat probe (206.9 ms with the imports resident) — see
``.perf/bench/lane11/``. ``serving.spawn_owned_session`` therefore warms the
factory's imports in a worker thread immediately before it enters the factory.

Three properties are worth defending, and none of them is a timing bound:

* the warm runs OFF the loop thread (a timing assertion here would flake on this
  fleet; a thread identity cannot);
* a warm that fails costs the warm and nothing else;
* the ``await`` it adds sits OUTSIDE the session factory's lease window, which
  ``session_factory`` states as an invariant in its own words: the acquire ->
  claim pair stays synchronous and on the loop, because a yield between the two
  is how two cold resumes lose the race the lease exists to arbitrate.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

#: Upper bound on an awaited call, never a budget to sleep through.
GUARD_S = 20.0

#: Loop turns used to prove the prober is live before the lease window opens. A
#: turn count rather than a sleep, for the reason AGENTS.md gives: a window
#: measured in seconds is a bet on machine load, and the work being waited out
#: is exactly the work that stretches under load.
PROBER_PRIMING_TURNS = 20


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Config, home and cwd out of the way of the developer's real ones."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.mark.asyncio
async def test_the_warm_runs_off_the_loop_thread_before_the_composition_root(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The warm is on a WORKER thread, and it is finished before the factory runs.

    Asserted on ``threading.get_ident()`` rather than on elapsed time: the
    property is "the loop thread did not do this work", which thread identity
    answers exactly and a duration can only answer probabilistically — this
    host's heartbeat floor alone swings 6.7 -> 27 ms under sibling load. The
    assertion fails deterministically if the ``asyncio.to_thread`` around the
    warm is dropped, which is the regression it exists for.

    The ORDER half is asserted in the same breath because off-loop warm that
    happened after ``create_session`` had already imported everything would
    satisfy the thread assertion and warm nothing at all.
    """
    from local_operator import session_factory
    from local_operator.session.runtime import serving

    loop_thread = threading.get_ident()
    order: list[str] = []
    warm_threads: list[int] = []

    def warm_spy() -> None:
        warm_threads.append(threading.get_ident())
        order.append("warm")

    class ReachedTheFactory(RuntimeError):
        """Sentinel: stops the spawn once the factory has been entered.

        Raised from a stand-in rather than letting the real factory build a
        session, so this test costs one config parse instead of a full
        construction — and so the order it observes is the spawn's, not the
        factory's.
        """

    async def factory_stub(*args: Any, **kwargs: Any) -> Any:
        order.append("factory")
        raise ReachedTheFactory

    monkeypatch.setattr(session_factory, "warm_session_imports", warm_spy)
    monkeypatch.setattr(session_factory, "create_session", factory_stub)

    with pytest.raises(ReachedTheFactory):
        await asyncio.wait_for(
            serving.spawn_owned_session(
                asyncio.get_running_loop(),
                cwd=str(isolated_config),
                provider="test",
                model_id="mock",
            ),
            timeout=GUARD_S,
        )

    assert warm_threads, "spawn_owned_session never warmed the composition root"
    assert warm_threads[0] != loop_thread, (
        "the composition root's imports ran ON the loop thread; the warm must go "
        "through asyncio.to_thread or the runtime child stalls for its whole duration"
    )
    assert order == ["warm", "factory"], (
        "the warm must complete BEFORE the factory is entered, or the factory "
        "pays the import cost itself"
    )


@pytest.mark.asyncio
async def test_a_raised_warm_does_not_fail_the_session_build(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A warm failure must leave the factory as the only reporter.

    ``warm_session_imports`` never raises by contract, but the thread hop around
    it can — a shut-down executor, an interpreter tearing down — and a boot that
    dies with a traceback about import machinery instead of the factory's own
    message about the session is strictly worse than a missed warm. The
    sentinel below is what makes the assertion exact: the build must fail for
    its OWN reason, not for the warm's.
    """
    from local_operator import session_factory
    from local_operator.session.runtime import serving

    def exploding_warm() -> None:
        raise RuntimeError("prewarm exploded")

    class FactoryReported(RuntimeError):
        """Sentinel: the factory was reached and raised its own failure."""

    async def factory_stub(*args: Any, **kwargs: Any) -> Any:
        raise FactoryReported

    monkeypatch.setattr(session_factory, "warm_session_imports", exploding_warm)
    monkeypatch.setattr(session_factory, "create_session", factory_stub)

    with pytest.raises(FactoryReported):
        await asyncio.wait_for(
            serving.spawn_owned_session(
                asyncio.get_running_loop(),
                cwd=str(isolated_config),
                provider="test",
                model_id="mock",
            ),
            timeout=GUARD_S,
        )


def test_the_warm_list_covers_the_registry_and_the_classification_seam() -> None:
    """The two groups the warm used to omit, pinned by name AND by effect.

    Membership alone would pass for a stale string that no longer resolves: a
    rename leaves a dead entry behind, and the warm's own per-name ``except``
    means the only symptom is the stall coming back. So the list is asserted and
    then the warm is run over it, which is what proves the names still import
    into ``sys.modules``.
    """
    from local_operator.session_factory import _WARM_IMPORTS, warm_session_imports

    for name in ("local_operator.tools.registry", "local_operator.classification"):
        assert name in _WARM_IMPORTS, f"{name} is what the runtime child's stall was made of"

    # The shipped list, not a patched one — that is the thing under test.
    warm_session_imports()

    missing = [
        name
        for name in ("local_operator.tools.registry", "local_operator.classification")
        if name not in sys.modules
    ]
    assert missing == [], f"warmed but absent from sys.modules: {missing}"


@pytest.mark.asyncio
async def test_no_loop_turn_happens_between_the_lease_and_the_claim(
    isolated_config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard for the ordering the warm site must not violate.

    ``session_factory._prepare`` takes the session lease and writes the claim
    marker as one synchronous pair, and its comment states why: a yield between
    the two is how two cold resumes lose the race the lease exists to
    arbitrate. This test measures that property directly instead of trusting a
    comment — a prober task turns the loop as fast as it can, and the turn count
    is sampled around both calls. Any ``await`` introduced inside the window
    (the obvious wrong way to warm the factory: put the ``to_thread`` where the
    imports are) lets the prober run and the two samples diverge.

    Driven through ``spawn_owned_session`` rather than ``_prepare`` so it also
    covers the half this change touches: the warm's own ``await`` must land
    before ``create_session`` is entered, which is why the prober is allowed to
    turn freely up to the acquire.
    """
    from local_operator import session_factory, session_lease
    from local_operator.session import retention
    from local_operator.session.runtime import serving

    turns = 0
    running = asyncio.Event()

    async def prober() -> None:
        nonlocal turns
        running.set()
        while True:
            await asyncio.sleep(0)
            turns += 1

    marks: dict[str, int] = {}

    real_acquire = session_lease.acquire_session_lease
    real_claim = retention.claim_session

    def acquire_spy(directory: Path) -> Any:
        marks["at_acquire"] = turns
        return real_acquire(directory)

    def claim_spy(directory: Path) -> Any:
        marks["at_claim"] = turns
        return real_claim(directory)

    async def no_wiring(session: Any, tools: Any, cwd: str, **kwargs: Any) -> None:
        # The MCP wiring is a real await on the session's loop and is not part
        # of the lease window; degrading it keeps this test about the lease.
        await asyncio.sleep(0)
        return None

    monkeypatch.setattr(session_factory, "wire_mcp_into_session", no_wiring)
    monkeypatch.setattr(session_lease, "acquire_session_lease", acquire_spy)
    monkeypatch.setattr(retention, "claim_session", claim_spy)

    prober_task = asyncio.create_task(prober())
    try:
        await asyncio.wait_for(running.wait(), timeout=GUARD_S)
        while turns < PROBER_PRIMING_TURNS:
            await asyncio.sleep(0)

        handle = await asyncio.wait_for(
            serving.spawn_owned_session(
                asyncio.get_running_loop(),
                cwd=str(isolated_config),
                provider="test",
                model_id="mock",
            ),
            timeout=GUARD_S,
        )
        try:
            assert "at_acquire" in marks and "at_claim" in marks, (
                "the lease window did not run: this test would pass vacuously if "
                f"the fixture stopped producing a claimed session directory ({marks})"
            )
            assert marks["at_claim"] == marks["at_acquire"], (
                "the loop turned between acquire_session_lease() and claim_session() "
                f"({marks['at_acquire']} -> {marks['at_claim']}): a yield in that window is how "
                "two cold resumes lose the race the lease exists to arbitrate"
            )
        finally:
            await handle._session.dispose()
    finally:
        prober_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await prober_task
