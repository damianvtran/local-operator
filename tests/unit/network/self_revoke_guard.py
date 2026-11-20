"""One teardown guard for the borrower's self-revoke timers, shared by the modules arming them.

WHY. ``borrowed_git_env`` arms ``github_mod._schedule_self_revoke``: a daemon
``threading.Timer`` that DELETEs the token at the grant's window end, against
the module's real ``GITHUB_API_BASE`` unless a cell points it at a loopback.
Nothing in the product cancels those timers (they live as long as the process),
so in a test process they outlive the cell that armed them and fire inside
whatever cell the same xdist worker runs next. Two shapes of that leak were
measured:

* ``test_credentials_github.py`` serves grants whose window ends 600 s out.
  Full-suite runs 43596174675, 43641441274, 43643694388 and 43656258089 each
  failed ``test_desktop_team_bridge.py`` teardowns ~10 min later: its wire
  tripwire counted the timer thread's ``httpx.HTTPTransport`` (or, when the
  client was built just before the cell, ``socket.connect``) as the cell's own.
* ``test_github_designation_refresh.py`` serves ``grant_expires_at_ms: 0``
  grants, so the timer's delay is the module's 0.1 s floor. A scratch probe on
  the unguarded head saw one of those timers still alive after its cell and its
  DELETE to ``https://api.github.com`` finish inside the NEXT cell
  (``test_two_relays_refresh_delivers_widen_narrow_and_no_op``).

PACKAGE-WIDE BACKSTOP (2026-11-19). Opt-in by import left every OTHER arming
module unguarded: ``borrowed_git_env`` is reached from at least a dozen modules
across ``tests/unit`` (network, code_requests, guardrails, monitors, session,
tui). Full-suite run 44408211857 (attempt 2, shard 4) failed
``test_cli_remote.py``'s teardown tripwire with ``DELETE
https://api.github.com/installation/token`` from a 600 s belt that the unguarded
``test_github_helper_diagnostic.py`` armed on the same xdist worker (its
``[thread-leak]`` line names that cell and a ``Thread-N`` timer). So
``tests/unit/conftest.py`` now runs :func:`stop_every_borrower_revoke` at every
unit cell's teardown. It does NOT replace ``github_mod.threading``: it uses the
product's own joinable handles (``cancel_pending_self_revokes(join_timeout_s=)``
and ``pending_self_revoke_threads``), so the constraint below still holds for
the rest of the package. The by-import fixture stays for what it adds on top:
a per-cell dedupe set and a record of every timer the cell built.

HOW TO USE. Import :func:`no_self_revoke_outlives_its_cell` into the test module
(``# noqa: F401`` — an autouse fixture is active in every module whose namespace
holds it, the same "fixtures by import" convention these modules already use).
Keeping it a by-import fixture rather than a package-wide ``conftest.py``
autouse scopes it to the modules that actually arm the timer; the rest of the
package never sees ``github_mod.threading`` replaced.

CONSTRAINTS. The timer's behaviour inside its own cell is untouched — a cell
that waits for its window end (``test_borrower_self_revoke_fires_at_window_end``)
still sees the real DELETE; only the leftover is cancelled, then joined so a
fire already in flight finishes before the next cell starts, and a join that
runs out fails the cell by name instead of letting the thread drift on. Only
``github_mod``'s timers are recorded: everything else on its ``threading``
attribute is the real module, read at call time.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any

import pytest

from local_operator.network.credentials import github as github_mod

#: Per-timer join bound at teardown. A cancelled, unfired timer exits at once;
#: only a fire already running waits. That fire is one DELETE on an
#: ``httpx.Client(timeout=github_mod.HTTP_TIMEOUT_S)`` — a 10 s cap on EACH
#: phase (connect, write, read, pool acquire), not on the request total — so a
#: pathologically slow fire can in principle outlast this bound. When it does,
#: the assertion after the join names the timer rather than letting it drift
#: into the next cell. Observed fires against the real host took 0.079-0.158 s
#: (n=4, scratch probe on this branch's base).
JOIN_TIMEOUT_S = 30.0


class _TimerRecordingThreading:
    """``github_mod``'s view of :mod:`threading`, recording every ``Timer`` it builds.

    Everything but ``Timer`` is the real module, read at call time, so the
    module's locks, events and threads behave exactly as in production; only
    the constructed timers are remembered so the cell's teardown can stop them.
    """

    def __init__(self, real: Any, sink: list[threading.Timer]) -> None:
        self._real = real
        self._sink = sink

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def Timer(self, *args: Any, **kwargs: Any) -> threading.Timer:  # noqa: N802
        timer = self._real.Timer(*args, **kwargs)
        self._sink.append(timer)
        return timer


def _describe(timer: threading.Timer) -> str:
    target = getattr(timer.function, "__qualname__", repr(timer.function))
    return f"{timer.name} ({target}, interval={timer.interval:.1f}s)"


def stop_recorded_timers(
    timers: list[threading.Timer], *, join_timeout_s: float = JOIN_TIMEOUT_S
) -> None:
    """Cancel, then join, every recorded timer; fail naming any still alive.

    Split from the fixture so the bounded-join assertion has a direct
    regression cell: the join's expiry is otherwise silent, and a guard whose
    failure mode is "passes while the thread lives on" is the leak again.
    """
    for timer in timers:
        timer.cancel()
    for timer in timers:
        # Bounded join, not a sleep: see JOIN_TIMEOUT_S for why it is bounded
        # and why the bound can, rarely, run out.
        timer.join(timeout=join_timeout_s)
    alive = [_describe(timer) for timer in timers if timer.is_alive()]
    assert not alive, (
        f"self-revoke timer(s) still alive {join_timeout_s:g}s after cancel+join, "
        f"and would fire into the next cell: {alive}"
    )


def stop_every_borrower_revoke(*, join_timeout_s: float = JOIN_TIMEOUT_S) -> None:
    """Cancel and join every borrower-side revoke this process has in flight.

    The package-wide teardown behind ``tests/unit/conftest.py``: pending belt
    timers are cancelled, then joined (a fire already running finishes inside the
    cell that armed it), and :func:`github_mod.revoke_now` threads are joined. Any
    thread still alive after the bound fails the cell by name, for the same
    reason :func:`stop_recorded_timers` does: a silent expiry is the leak again.
    """
    github_mod.cancel_pending_self_revokes(join_timeout_s=join_timeout_s)
    for thread in github_mod.pending_self_revoke_threads():
        thread.join(timeout=join_timeout_s)
    alive = [
        _describe(t) if isinstance(t, threading.Timer) else t.name
        for t in github_mod.pending_self_revoke_threads()
    ]
    assert not alive, (
        f"borrower revoke thread(s) still alive {join_timeout_s:g}s after cancel+join, "
        f"and would DELETE inside the next cell: {alive}"
    )


@pytest.fixture(autouse=True)
def no_self_revoke_outlives_its_cell(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stop every borrower self-revoke timer a cell scheduled, at that cell's teardown.

    The per-token dedupe set is per cell too: a token string reused by a later
    cell (``"tok"``) must not have its scheduling suppressed by a timer this
    teardown cancelled (a cancelled timer never runs the ``discard`` in its
    ``_fire``'s ``finally``).
    """
    timers: list[threading.Timer] = []
    monkeypatch.setattr(github_mod, "threading", _TimerRecordingThreading(threading, timers))
    monkeypatch.setattr(github_mod, "_SELF_REVOKE_SCHEDULED", set())
    yield
    stop_recorded_timers(timers)
