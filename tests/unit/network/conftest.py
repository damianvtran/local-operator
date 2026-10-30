"""Fixtures and helpers for the mesh core's unit tests.

Every test builds its OWN config root: the modules take a ``root`` argument
precisely so a test never has to monkeypatch ``HOME``, and the one place that
matters is identity — a test that minted a key into the operator's real store
would be a defect wearing a green tick.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
import socket
import time
from pathlib import Path
from typing import Any, Callable

import pytest

_LOG = logging.getLogger(__name__)

#: The one backstop budget for waits on real network/handshake/process progress here.
#:
#: WHY ONE CONSTANT, AND WHY IT IS GENEROUS. Every wait in this package is already
#: condition-based: the predicate is the assertion, and the budget only decides how long
#: a GENUINE hang takes to report instead of blocking the run (AGENTS.md, "Wait on the
#: event, never on the clock"). A per-call literal (``timeout_s=10.0`` at 14 sites in
#: ``test_membership_convergence`` alone) re-coupled that backstop to a number picked on
#: an idle machine, and this fleet runs at load 50-110 across ~25 concurrent worktrees:
#: the establishment pull, the admit-time pass and the returning handshake slots all DID
#: become true, just after ten seconds of contention, and the cells reported a product
#: failure that never happened. Raising the number at each site would work and would also
#: guarantee the next new wait reintroduces the class, so the budget lives here and call
#: sites that want it say nothing at all.
#:
#: ``LO_TEST_NET_WAIT_S`` has exactly one job: the negative control. Set it low
#: (``LO_TEST_NET_WAIT_S=0.5``) and this package must stay GREEN wherever the predicate is
#: already satisfied when the wait is reached — measured: 21/21 in
#: ``test_membership_convergence`` at 0.5 s AND at 0.05 s — because that is what proves the
#: budget is a backstop and not a sleep. To watch it BITE, aim it at a wait whose predicate
#: genuinely needs real progress: the cross-process cells in ``test_move_queue`` red at
#: ``LO_TEST_NET_WAIT_S=0.001``, and ``test_wait_budget``'s delayed predicate proves the
#: same thing deterministically in one cell. A green low-budget run is therefore the
#: expected result on a loaded host, not evidence that the guard cannot fail. Never raise
#: it to make a cell green.
NET_WAIT_BUDGET_S: float = float(os.environ.get("LO_TEST_NET_WAIT_S") or "60")


def _describe(predicate: Callable[[], bool]) -> str:
    """What the caller was waiting for, as one line, for a failure report.

    The predicate IS the assertion here, so its own source is the most useful thing to
    print beside the caller's sentence. Best-effort by design: a predicate built by
    ``exec``, or a callable object with no source, still has to render, and this only
    runs on the reporting path.
    """
    try:
        source = inspect.getsource(predicate).strip()
    except (OSError, TypeError):
        source = getattr(predicate, "__qualname__", None) or repr(predicate)
    source = " ".join(source.split())
    return source if len(source) <= 300 else source[:297] + "..."


class WaitResult:
    """One wait's outcome, in the shape an ``assert`` at the call site renders.

    WHY NOT A PLAIN BOOL: ``assert wait_for(...), "<caller's sentence>"`` prints that
    sentence AND this object's repr when it fails, so a red shard log carries the elapsed
    time and the predicate that was still false next to the caller's own description. A
    bool prints the sentence alone, which is exactly the ambiguity that made tonight's
    fleet-wide reds readable as a product defect: nothing in the message distinguished
    "this machine was slow" from "the work never happened". Truthiness is the whole
    contract for existing call sites — every one of them uses the result in a boolean
    context.
    """

    __slots__ = ("budget_s", "elapsed_s", "ok", "what")

    def __init__(self, ok: bool, elapsed_s: float, budget_s: float, what: str = "") -> None:
        # Coerced here, not in ``__bool__``: call sites are promised a truthiness
        # contract, and a predicate that returns ``1`` or ``[]`` through a future
        # ``wait_outcome(ok=...)`` would otherwise raise ``TypeError: __bool__ should
        # return bool`` at the assert instead of reporting which condition failed.
        self.ok = bool(ok)
        self.elapsed_s = elapsed_s
        self.budget_s = budget_s
        self.what = what

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:
        if self.ok:
            return f"<net wait satisfied in {self.elapsed_s:.2f}s (budget {self.budget_s:.1f}s)>"
        return (
            f"<net wait EXPIRED after {self.elapsed_s:.2f}s "
            f"(budget {self.budget_s:.1f}s): {self.what} still false>"
        )


def wait_outcome(
    ok: bool, started: float, budget_s: float, predicate: Callable[[], bool]
) -> WaitResult:
    """The one place a wait becomes a result: measured, logged, described.

    ``wait_for`` is the spelling for a synchronous predicate. A caller that owns its own
    polling loop — the async spelling in ``test_remote_viewer`` awaits its predicate in a
    thread — still reports through here, which is what keeps "expired after N s" meaning
    the same thing everywhere, and keeps the drift log on the passing path off a per-file
    copy that goes stale.
    """
    elapsed = time.monotonic() - started
    if ok:
        _LOG.debug("net wait satisfied in %.2fs (budget %.1fs)", elapsed, budget_s)
        return WaitResult(True, elapsed, budget_s)
    what = _describe(predicate)
    _LOG.error(
        "net wait expired after %.2fs (budget %.1fs): %s still false", elapsed, budget_s, what
    )
    return WaitResult(False, elapsed, budget_s, what)


def wait_for(predicate: Callable[[], bool], timeout_s: float = NET_WAIT_BUDGET_S) -> WaitResult:
    """Poll ``predicate`` until it holds, or until the deadline passes; report which.

    THE DEADLINE IS A BACKSTOP, NOT THE ASSERTION. The wait lasts exactly as long as
    the work does, and the bound exists so that a genuine hang fails the run instead of
    blocking it (see AGENTS.md, "Wait on the event, never on the clock"). What a caller
    has to get right is the PREDICATE: it must test the condition the assertion is
    about, not a proxy that merely happens to become true first.

    That distinction is what two cells in this package got wrong, and a loaded shard
    runner is what found it. The relay RECORDS on the thread that ACTS, and it acts
    first: ``_close_stream`` pops the stream table and closes the dial before it calls
    ``_report_stream_closed``, and the pairing listener sends its abort frame before it
    records ``pairing_refused``. A test that waits for the effect and then reads the
    trail once is therefore racing the row's own write with a margin of ZERO rather
    than a margin of its bound — measured at 1 ms of injected scheduling delay on one
    cell and 100 ms on the other, against starvation gaps of 525-668 ms recorded for
    this fleet.

    ONE SPELLING, ONE DEFAULT, ONE BUDGET. This replaces the two module-local copies it
    was moved out of (``test_session_plane._wait_for``, 10 s, and
    ``test_credentials_real_link._wait_for``, 15 s) *and* the per-call ``timeout_s=10.0``
    literals that survived that move: pass a budget only when the work genuinely has a
    shorter latency by construction, and otherwise let ``NET_WAIT_BUDGET_S`` decide.

    PASSING WAITS ARE MEASURED TOO. Every satisfied wait logs its elapsed time at DEBUG,
    so a cell creeping toward the backstop is visible before it reds on a loaded host —
    the alternative is discovering drift as the outage. Nothing in ``pyproject.toml`` or
    the CI workflows raises that level, so on a normal run the line is NOT emitted: read
    it with ``-o log_level=DEBUG`` (or ``--log-cli-level=DEBUG``). It is a control you
    switch on, not telemetry that arrives by default, and this docstring said otherwise.
    """
    started = time.monotonic()
    deadline = started + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return wait_outcome(True, started, timeout_s, predicate)
        time.sleep(0.05)
    # The deadline passing is not proof the work never landed: re-test once, so a
    # condition that became true during the final sleep is reported as satisfied rather
    # than as a hang (the polling loop's last iteration can end on the clock, not the
    # predicate). This single re-test is also where the old form's trailing ``return
    # predicate()`` went.
    return wait_outcome(predicate(), started, timeout_s, predicate)


def subcommands_of(parser: argparse.ArgumentParser) -> dict[str, Any]:
    """``{verb: subparser}`` for ``parser``, proven to be the dict argparse built.

    argparse types an action's ``choices`` as ``Iterable[Any] | None``, so a reader
    that KNOWS this mapping is the one ``add_subparsers`` created has to prove it:
    a re-derived ``getattr`` per use is not something a checker can narrow, and
    subscripting the union blind is the error this replaces. Shared because three
    of this package's test modules ask the same question of the real parser.
    """
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict):
            return choices
    raise AssertionError(f"{parser.prog}: no subcommands registered")


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    """A private config root for one test."""
    return tmp_path


@pytest.fixture()
def socketpair() -> Any:
    """A connected pair of TCP sockets on loopback — real sockets, not a stub.

    The handshake's failure modes are about what goes on the wire (a silent close,
    a pipelined frame, a truncated read), so a fake stream would test the fake
    rather than the protocol.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.create_connection(server.getsockname()[:2], timeout=5)
    accepted, _addr = server.accept()
    try:
        yield client, accepted
    finally:
        for sock in (client, accepted, server):
            try:
                sock.close()
            except OSError:
                pass


@pytest.fixture(autouse=True)
def _no_real_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test in this suite may DISCOVER the real gh (review round 2, n3).

    ``github.find_gh`` is the only discovery seam for the forge ladder's gh arm,
    and the real gh's keychain lookup is user-global: a stray discovery in a
    test process could reach the operator's login. Pinning the seam to "not
    found" here covers every file in this directory — present and future —
    rather than the single file that happened to need it first; cells that
    exercise the ask-gh path install their own stub and override this pin
    explicitly, and the discovery itself never runs the binary.
    """
    from local_operator.network.credentials import github as github_mod

    monkeypatch.setattr(github_mod, "_resolve_program", lambda name, path=None: None)
