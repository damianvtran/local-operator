"""The runtime's own build watch retires it, and no third party kills it.

WHY THIS FILE EXISTS (I2 of the mass-kill work). Three fleet-wide deaths are
documented in this repo — the 2026-09-15 19:41 sweep of 36 runtimes, the
`libpython` dylib pin that killed 113 processes the same day, and the 2026-09-18
vanishing of 25 runtimes inside 13 seconds — and the runtime's build watch is the
one path that LOOKS like all three from the outside: a runtime disappears
mid-fleet, on a build move, without an exit record of the ordinary kind. It is not
any of them. It calls itself: the ladder below retires ONE runtime, at ITS OWN
turn boundary, and leaves through the graceful disposal, logging
``retiring for <build>`` and then ``exiting cleanly``.

So the fix for the two hazards (I1) must not touch this path, and the pin is the
test: what the path does, in order, and the two things it must never do — take a
turn with it, or signal anybody. The ``lop serve`` daemon's half (announce only,
never exit, production supplies no callback) is pinned in
``tests/unit/server/test_serve_retire.py``; this is the session-runtime half.
"""

from __future__ import annotations

import ast
import asyncio
import logging
from typing import Awaitable, TypeVar

import pytest

from local_operator import buildwatch
from local_operator import update as update_mod
from local_operator.session.runtime import process
from local_operator.update import BuildStamp

_R = TypeVar("_R")

BOOT = BuildStamp(version="0.51.0", source_ref="abc1234567890")
NEW = BuildStamp(version="0.52.0", source_ref="def1234567890")


@pytest.fixture()
def moved(monkeypatch: pytest.MonkeyPatch) -> None:
    """A newer, SETTLED build on disk, and no stagger to sleep out.

    The readers are the shared ones (``buildwatch``/``update``) because both build
    watchers must obey one rule; the stagger is shortened because it is jitter for
    a fleet, not a behaviour under test.
    """
    monkeypatch.setattr(update_mod, "installed_build", lambda *_a, **_k: NEW)
    monkeypatch.setattr(update_mod, "disk_build", lambda *_a, **_k: NEW)
    monkeypatch.setattr(update_mod, "build_marker_age_s", lambda *_a, **_k: 999.0)
    monkeypatch.delenv("LOP_BUILD_SETTLE_S", raising=False)
    monkeypatch.delenv("LOP_BUILD_STAGGER_S", raising=False)
    monkeypatch.delenv("LOP_BUILD_PREFIX", raising=False)
    monkeypatch.setattr(process, "_build_stagger_seconds", lambda: 0.0)


class IdleHandle:
    """The two seams the retire path uses, and a tripwire for the one it must not.

    ``may_refresh`` is the product's own idle predicate (``ServingSessionHandle``);
    ``begin_retire`` is the LATCH that commits the runtime to leaving in one
    synchronous step. ``request_stop`` exists to raise: nothing here may stop the
    runtime from the outside, and a signal sent by this path would be the
    third-party kill this whole change set exists to rule out.
    """

    def __init__(self, *, admits: bool = True) -> None:
        self.admits = admits
        self.latched: list[tuple[str, str]] = []

    def may_refresh(self) -> str:
        return ""

    def begin_retire(self, cause: str, detail: str = "") -> bool:
        self.latched.append((cause, detail))
        return self.admits

    def request_stop(self) -> None:
        raise AssertionError("the build watch must never signal anyone")


class IdleRuntime:
    """The session under the handle: the announce seam and the boot build."""

    def __init__(self) -> None:
        self._boot_build = BOOT
        self.announced: list[tuple[str, str]] = []

    async def announce_retiring(self, cause: str, *, to: str = "") -> None:
        self.announced.append((cause, to))


@pytest.mark.asyncio
async def test_an_idle_runtime_retires_itself_gracefully_on_a_build_move(
    moved: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Announce, latch, then leave through the GRACEFUL disposal — in that order.

    Every assertion here is a different promise, and the order is the one the
    reference investigation could not reconstruct afterwards: the announcement is
    what makes a viewer re-engage instead of reading the exit as owner death, the
    latch is what makes the decision atomic against a turn arriving during the
    announce, and the reason string is what makes the exit self-explaining in the
    log (``retiring for 0.52.0`` — the sentence design §1.6 asks for, because an
    exiting runtime that logged nothing about itself is how a refresh retirement,
    a SIGTERM and a torn install became one story).
    """
    handle, runtime, stop = IdleHandle(), IdleRuntime(), asyncio.Event()
    exits: list[str] = []

    async def recording_exit(
        _handle: object, _runtime: object, *, reason: str, detail: str = ""
    ) -> None:
        # ``detail`` accepted (and unused) because the real ``_clean_exit`` now
        # carries the why-now into the durable exit record; this cell is about
        # the REASON string the log names.
        exits.append(reason)

    monkeypatch.setattr(process, "_clean_exit", recording_exit)

    with caplog.at_level(logging.INFO, logger=process.__name__):
        retired = await process._refresh_for(NEW, handle, runtime, stop)

    assert retired is True
    assert handle.latched == [("runtime-retired", " (0.51.0@abc1234 → 0.52.0@def1234)")]
    assert runtime.announced == [("stale-build", NEW.label())]
    assert exits == [f"retiring for {NEW.label()}"], exits
    assert stop.is_set(), "the exit ends the run: amain's wait() must return"
    assert any("retiring for 0.52.0" in record.getMessage() for record in caplog.records), [
        record.getMessage() for record in caplog.records
    ]


@pytest.mark.asyncio
async def test_a_turn_arriving_during_the_announce_keeps_the_runtime(
    moved: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The latch refusing is the whole safety property, and nothing else may proceed.

    ``begin_retire`` returns False when any work is in flight or any admission
    would be refused, so the refresh must keep the runtime rather than abort a turn
    it had just decided not to disturb — the shape a "sample the predicate, then
    exit" watcher gets wrong. Nothing was disposed and nothing was exited.
    """
    handle, runtime, stop = IdleHandle(admits=False), IdleRuntime(), asyncio.Event()
    exits: list[str] = []

    async def recording_exit(
        _handle: object, _runtime: object, *, reason: str, detail: str = ""
    ) -> None:
        # ``detail`` accepted (and unused) because the real ``_clean_exit`` now
        # carries the why-now into the durable exit record; this cell is about
        # the REASON string the log names.
        exits.append(reason)

    monkeypatch.setattr(process, "_clean_exit", recording_exit)

    assert await process._refresh_for(NEW, handle, runtime, stop) is False
    assert exits == [], "a refused latch must not leave"
    assert not stop.is_set(), "the runtime is still serving"


@pytest.mark.asyncio
async def test_a_stop_landing_during_the_stagger_leaves_the_exit_to_the_stop(
    moved: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stagger is an await, and a stop inside it owns the exit rather than this.

    A stop has already staged its own evidence (``control._write_stop_marker``) and
    chosen its rung; a second exit path completing here would attribute the death
    twice, to two different parties. The watcher stands down instead.
    """
    handle, runtime, stop = IdleHandle(), IdleRuntime(), asyncio.Event()
    stop.set()
    exits: list[str] = []

    async def recording_exit(
        _handle: object, _runtime: object, *, reason: str, detail: str = ""
    ) -> None:
        # ``detail`` accepted (and unused) because the real ``_clean_exit`` now
        # carries the why-now into the durable exit record; this cell is about
        # the REASON string the log names.
        exits.append(reason)

    monkeypatch.setattr(process, "_clean_exit", recording_exit)

    assert await process._refresh_for(NEW, handle, runtime, stop) is False
    assert handle.latched == [], "a stopter's runtime is not retired by the watcher too"
    assert exits == []


def test_the_serve_daemon_can_only_ever_exit_with_an_injected_callback() -> None:
    """The other graceful retirement, pinned at ITS seam: production only announces.

    ``retirement_poll`` writes the handover into the daemon's record and keeps
    serving; the only thing that can make it exit is an ``exit_process`` callback,
    which production never supplies — the live ``run/serve/61225.json`` still
    reading ``retiring_from 0.59.0`` hours later is exactly that behaviour and not a
    stuck daemon. ``tests/unit/server/test_serve_retire.py`` pins the sequence in
    full (announce while serving, never request shutdown, no callback); this pins the
    default that makes it true, so a future signature that defaults the callback to
    something exit-shaped reddens here.
    """
    import inspect

    from local_operator.server import retire

    parameter = inspect.signature(retire.retirement_poll).parameters["exit_process"]
    assert parameter.default is None
    assert "must not exit on marker drift" in (retire.retirement_poll.__doc__ or "")


def test_the_watchers_share_one_build_rule() -> None:
    """Both watchers read ``buildwatch``, so neither can drift onto its own copy.

    The session runtime's retire and the serve daemon's announcement must agree on
    what "the build moved" means, or one of them fires where the other refuses —
    and the disagreement shows up as a fleet behaviour rather than as a defect.
    """
    from local_operator.server import retire

    assert process._buildwatch is buildwatch
    assert retire.buildwatch is buildwatch
    assert process.BUILD_CHECK_S == buildwatch.BUILD_CHECK_S
    assert process.BUILD_SETTLE_S == buildwatch.BUILD_SETTLE_S
    # NOTHING MORE TO PIN ON THE DAEMON'S SIDE (review round 1, NIT 1): the identity
    # assertions above are what hold both readers on the one rule, and the line that
    # used to sit here compared ``retire.buildwatch.BUILD_CHECK_S`` with
    # ``buildwatch.BUILD_CHECK_S`` — the same object read twice, which could never
    # redden for the drift it claimed to catch.


def test_a_handle_without_the_idle_predicate_is_never_retired() -> None:
    """Unknown state is not an invitation to leave (the reduced-host contract).

    A handle that cannot answer ``may_refresh`` — an older host, a stripped test
    handle — must never retire: the whole path is decoration on a process listing
    until the runtime can prove it would lose nothing.
    """

    class NoProbe:
        pass

    assert process._idle_for_refresh(NoProbe()) is False
    assert process._should_refresh(NoProbe(), BOOT) is None


def test_the_watcher_never_signals_anyone() -> None:
    """A structural pin for the symptom this change set is about: nothing kills.

    ``IdleHandle.request_stop`` raises, and the fall-through of every guard above
    is the graceful disposal, so the only way this path can end a runtime is by the
    runtime leaving itself. The count of kill sites is what an investigation reads
    (``grep -rn 'os.kill'`` over this module is how the 2026-09-18 sweep was looked
    for), so the module is read as code rather than as text: an AST walk for
    kill-SHAPED calls, in every spelling, reported with the line that holds one.

    WHAT THIS REPLACED, and why (review round 1, MINOR 3). The text pin matched
    three literal substrings over the source: a comment mentioning ``os.kill``
    turned it red, while ``os.killpg``, ``Popen.kill``, ``Process.terminate``,
    ``signal.raise_signal`` and ``subprocess.run(["kill", …])`` all stayed green —
    brittle in the direction that costs a re-run and blind in the direction that
    costs a runtime.
    """
    from pathlib import Path

    source = Path(process.__file__).read_text(encoding="utf-8")
    sites = _kill_shaped_sites(source)
    assert sites == [], "this module must not be able to end a process: " + "; ".join(sites)


#: Call names that SPAWN something, checked for a killer in their argv: a shell
#: ``kill`` reaches the same place as ``os.kill`` while carrying none of its names,
#: and this is the spelling a name-based pin cannot see.
_SPAWN_CALLS = frozenset(
    {"run", "call", "check_call", "check_output", "system", "popen", "Popen", "execv", "execvp"}
)

#: Calls that reach a name through a STRING, so the name check has to read the
#: argument: ``getattr(os, "kill")(…)`` and ``getattr(signal, "raise_signal")(…)``
#: are the same sites as their attribute spellings and carry no attribute to match.
#: ``__getattr__``/``__getattribute__`` are here for the same reason.
_INDIRECT_NAMES = frozenset({"getattr", "__getattr__", "__getattribute__"})

#: ``signal.pthread_kill`` and ``signal.raise_signal`` end a process and belong to
#: the same shape as ``os.kill`` (review round 2, NIT 2): the pin's own docstring
#: says anything it cannot see is reported as nothing rather than as a pass, which
#: is exactly why the reach is widened rather than assumed.
_KILL_CALLS = frozenset(
    {
        "kill",
        "killpg",
        "pthread_kill",
        "terminate",
        "send_signal",
        "raise_signal",
    }
)
_KILLER_TOKENS = ("kill",)


def _kill_shaped_sites(source: str) -> list[str]:
    """Every ``line N: …`` in ``source`` that could end a process, or ``[]``.

    Reads the module as CODE (review round 1, MINOR 3): a comment saying ``os.kill``
    is not a kill site, and ``os.killpg`` is, whatever the text looks like. Three
    spellings are covered because each is a way a runtime really dies here — the
    signal call, the process-object method, and a spawned ``kill`` — and anything
    the walk cannot see is reported as nothing rather than as a pass, which is why
    the call SHAPES are listed above rather than a distance to the nearest ``kill``.
    """

    def called_name(node: ast.Call) -> str:
        func = node.func
        if isinstance(func, ast.Attribute):
            return func.attr
        return func.id if isinstance(func, ast.Name) else ""

    def name_from_literal(node: ast.Call) -> str:
        """The name a call reaches by a STRING rather than by an attribute.

        ``getattr(os, "kill")(pid, 9)`` ends a process exactly as ``os.kill``
        does, and it carries no attribute for the walk above to read — the same
        blind spot the spawn check covers for a shelled ``kill``, in the one
        other spelling a name-based pin cannot see (review round 2, NIT 2). Only
        ``getattr`` is read this way: an arbitrary call's string arguments are
        full of words like "terminate", and reporting those would make the pin
        fire on prose in an argument list — the failure mode the text pin this
        replaced had.
        """
        if len(node.args) < 2:  # a malformed getattr; nothing to read
            return ""
        func = node.func
        reached = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id if isinstance(func, ast.Name) else ""
        )
        if reached not in _INDIRECT_NAMES:
            return ""
        target = node.args[1]
        if isinstance(target, ast.Constant) and isinstance(target.value, str):
            return target.value
        return ""

    def literals(node: ast.Call) -> list[str]:
        """Every string constant in the call's own arguments, nested lists included."""
        out: list[str] = []
        for arg in [*node.args, *node.keywords]:
            for inner in ast.walk(arg):
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                    out.append(inner.value)
        return out

    sites: list[str] = []
    for node in ast.walk(ast.parse(source)):
        # The control ladder's own SIGTERM/SIGKILL rungs: reaching into them from a
        # build watch is exactly the coupling this pin forbids, however it is
        # spelled (``control._signal_and_confirm`` or an imported bare name).
        if isinstance(node, ast.Name) and node.id == "_signal_and_confirm":
            sites.append(f"line {node.lineno}: references _signal_and_confirm")
            continue
        if isinstance(node, ast.Attribute) and node.attr == "_signal_and_confirm":
            sites.append(f"line {node.lineno}: references _signal_and_confirm")
            continue
        if not isinstance(node, ast.Call):
            continue
        name = called_name(node)
        if name in _KILL_CALLS:
            sites.append(f"line {node.lineno}: calls {name}()")
            continue
        if name in _INDIRECT_NAMES:
            indirect = name_from_literal(node)
            if indirect in _KILL_CALLS:
                sites.append(f"line {node.lineno}: calls {indirect}() by name")
                continue
        if name in _SPAWN_CALLS and any(
            token in word for word in literals(node) for token in _KILLER_TOKENS
        ):
            sites.append(f"line {node.lineno}: spawns something called a killer ({name})")
    return sites


@pytest.mark.parametrize(
    "snippet",
    [
        "import os\nos.kill(1, 0)\n",
        "import os\nos.killpg(1, 0)\n",
        "handle.kill()\n",
        "child.terminate()\n",
        "import signal\nsignal.raise_signal(signal.SIGKILL)\n",
        "conn.send_signal(9)\n",
        'import subprocess\nsubprocess.run(["kill", "-9", "1"])\n',
        "control._signal_and_confirm(record, SIGTERM, 1.0)\n",
        # Reaching the same two by NAME rather than by attribute (review round 2,
        # NIT 2). Neither carries an attribute for the walk to read, and both end
        # a process exactly as their attribute spellings do.
        'import os\ngetattr(os, "kill")(1, 9)\n',
        'import signal\ngetattr(signal, "pthread_kill")(1, 9)\n',
    ],
)
def test_the_no_kill_pin_sees_every_spelling(snippet: str) -> None:
    """The instrument finds what the text pin could not (review round 1, MINOR 3).

    Each spelling here is a way a runtime really dies, and the pin it replaced —
    three literal substrings over the source — was blind to all but the first.
    Asserted on the WALKER rather than on the module, because the module is
    supposed to be clean: a guard nobody has seen fail is not a guard.
    """
    assert _kill_shaped_sites(snippet) != [], snippet


def test_the_no_kill_pin_ignores_prose() -> None:
    """...and it does not redden on a COMMENT, which is what the text pin did.

    The comment below names the exact call the pin exists to forbid; the module is
    allowed to talk about it (this file does), and only a CALL is a site.
    """
    assert _kill_shaped_sites("# never call os.kill / signal.SIGKILL here\nx = 1\n") == []


def test_the_no_kill_pin_ignores_a_string_that_is_not_a_name() -> None:
    """The widened reach must not become the text pin it replaced (NIT 2).

    Reading a string argument is what makes ``getattr(os, "kill")(…)`` visible, and
    it is also the way this could fire on prose: an ordinary call whose ARGUMENT
    happens to contain the word. Only the name position of the three indirect
    calls is read, so a message mentioning a kill is not a site — which is the
    same distinction the comment case above draws, one layer in.
    """
    assert _kill_shaped_sites('print("we never terminate anything")\n') == []
    assert _kill_shaped_sites('log.warning("os.kill was not called")\n') == []
    assert _kill_shaped_sites('getattr(obj, "terminate_on_idle")\n') == []
    # ...and a one-argument ``getattr`` must not be an index error inside the pin.
    assert _kill_shaped_sites("getattr(obj)\n") == []


@pytest.mark.asyncio
async def test_a_stuck_disk_read_neither_blocks_the_reaper_loop_nor_refreshes(
    moved: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reaper's build check reads the disk OFF the loop (2026-11-17: a 145 s
    on-loop metadata read wedged a ``serve`` loop; the runtime's check is the same
    read). While it is stuck nothing is judged — no refresh, no decline — and
    once it answers, the settled move is acted on as before."""
    import threading
    import time

    gate = threading.Event()

    def stuck(*_a: object, **_k: object) -> BuildStamp:
        gate.wait(10.0)
        return NEW

    refreshed: list[BuildStamp] = []

    async def recording_refresh(newer, _handle, _runtime, stop, **_kw):  # noqa: ANN001
        # ``**_kw``: the reaper now hands its own reader in (``read=``).
        refreshed.append(newer)
        stop.set()
        return True

    monkeypatch.setattr(update_mod, "disk_build", stuck)
    monkeypatch.setattr(buildwatch, "BUILD_READ_TIMEOUT_S", 0.05)
    monkeypatch.setattr(process, "REAP_CHECK_S", 0.01)
    monkeypatch.setattr(process, "BUILD_CHECK_S", 0.02)
    monkeypatch.setattr(process, "_should_exit", lambda *_a: False)
    monkeypatch.setattr(process, "_refresh_for", recording_refresh)
    handle, runtime, stop = IdleHandle(), IdleRuntime(), asyncio.Event()
    reaper = asyncio.ensure_future(process._reaper(handle, runtime, stop))
    try:
        ticks, deadline = 0, time.monotonic() + 0.3
        while time.monotonic() < deadline:
            await asyncio.sleep(0.01)
            ticks += 1
        assert ticks >= 10, f"the loop made only {ticks} ticks while the read was stuck"
        assert refreshed == [], "an unanswered read is not a build to refresh onto"
        assert not reaper.done()
        gate.set()
        assert await asyncio.wait_for(reaper, 2.0) is True
    finally:
        gate.set()
        stop.set()
    assert refreshed == [NEW]


async def _loop_ticks_while(work: "Awaitable[_R]") -> "tuple[int, _R]":
    """Await ``work`` while counting 5 ms ticks of a sibling coroutine.

    The tick count is the observable for "the loop was not blocked": a read run ON
    the loop holds every other coroutine for as long as the disk does, so the count
    stays near zero; a read run off it lets the sibling tick throughout.
    """
    ticks = 0

    async def tick() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.005)
            ticks += 1

    ticker = asyncio.ensure_future(tick())
    try:
        result = await asyncio.wait_for(work, 5.0)
    finally:
        ticker.cancel()
    return ticks, result


class _StuckDisk:
    """``update.disk_build`` that answers ``answered`` times, then blocks on a gate.

    The 2026-11-17 stall (an ``importlib.metadata`` read wedged for 145 s) as a
    function: the gate stands in for the swapped-out page, and the test releases it
    in ``finally`` so the abandoned thread ends with the test.
    """

    def __init__(self, *, answered: int = 0) -> None:
        import threading

        self.gate = threading.Event()
        self.calls = 0
        self._answered = answered

    def __call__(self, *_a: object, **_k: object) -> BuildStamp:
        self.calls += 1
        if self.calls > self._answered:
            self.gate.wait(10.0)
        return NEW


@pytest.mark.asyncio
async def test_a_stuck_re_check_after_the_stagger_keeps_without_blocking_the_loop(
    moved: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 1, M1: ``_refresh_for`` re-reads the disk after its stagger.

    That re-check used to be the sync ``_should_refresh`` on the loop, so the first
    read moving off the loop only moved the stall one await later. Now it goes
    through the reader: the loop keeps ticking, and an unanswered re-check KEEPS —
    no latch, no announcement, no exit (a refresh can always wait a check).
    """
    disk = _StuckDisk()
    monkeypatch.setattr(update_mod, "disk_build", disk)
    monkeypatch.setattr(buildwatch, "BUILD_READ_TIMEOUT_S", 0.05)
    exits: list[str] = []

    async def recording_exit(
        _handle: object, _runtime: object, *, reason: str, detail: str = ""
    ) -> None:
        exits.append(reason)

    monkeypatch.setattr(process, "_clean_exit", recording_exit)
    handle, runtime, stop = IdleHandle(), IdleRuntime(), asyncio.Event()
    try:
        ticks, retired = await _loop_ticks_while(process._refresh_for(NEW, handle, runtime, stop))
    finally:
        disk.gate.set()
    assert ticks >= 3, f"the loop made only {ticks} ticks while the re-check was stuck"
    assert retired is False, "an unanswered re-check is not a build to leave for"
    assert disk.calls == 1, "the re-check read the disk once"
    assert handle.latched == [] and runtime.announced == [] and exits == []
    assert not stop.is_set()


@pytest.mark.asyncio
async def test_a_stuck_re_check_on_a_latchless_handle_keeps_without_blocking_the_loop(
    moved: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same M1 site's second half: a handle without ``begin_retire`` re-checks
    after the announce instead of latching. The first re-check answers (so the
    announce happens), the second sticks: the runtime keeps, and the loop ran."""

    class LatchlessHandle:
        def may_refresh(self) -> str:
            return ""

        def request_stop(self) -> None:
            raise AssertionError("the build watch must never signal anyone")

    disk = _StuckDisk(answered=1)
    monkeypatch.setattr(update_mod, "disk_build", disk)
    monkeypatch.setattr(buildwatch, "BUILD_READ_TIMEOUT_S", 0.05)
    exits: list[str] = []

    async def recording_exit(
        _handle: object, _runtime: object, *, reason: str, detail: str = ""
    ) -> None:
        exits.append(reason)

    monkeypatch.setattr(process, "_clean_exit", recording_exit)
    runtime, stop = IdleRuntime(), asyncio.Event()
    try:
        ticks, retired = await _loop_ticks_while(
            process._refresh_for(NEW, LatchlessHandle(), runtime, stop)
        )
    finally:
        disk.gate.set()
    assert ticks >= 3, f"the loop made only {ticks} ticks while the re-check was stuck"
    assert retired is False and exits == [] and not stop.is_set()
    assert runtime.announced == [("stale-build", NEW.label())], "the first re-check answered"
    assert disk.calls == 2, "the post-announce re-check is the one that stuck"
