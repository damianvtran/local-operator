"""The console capability, read off the desktop app's EXISTING discovery record.

Why this module reads ``run/ui-browser/host.json`` rather than publishing its own
file: the console rides the host the app already runs (design §10.1) — one
endpoint, one key, one heartbeat, one teardown. The record's *lifecycle* does not
differ from the browser host's (same process, same heartbeat, same writer), and a
second namespace is justified only when a different PROCESS owns the directory,
which is the argument ``ui_browser/state.py`` itself makes for keeping away from
``run/browser/bridge.json``. So the namespace here is a MODEL, not a directory:
``ConsoleHostState`` extends the app host's record with the console fields the app
writes (``console``, ``console_surfaces``, ``console_agent_surfaces``,
``console_proto`` — additive, all defaulted, so a record written by an app that
predates the console reads as "no console" through this model AND through the
unmodified ``UiHostState``, whose ``extra="ignore"`` drops fields it does not
know).

That is also why there is no ``remove``/``run_dir`` wrapper: the app owns its
record's lifecycle, exactly as the browser's reader states for its own namespace.

The predicate differs from the browser's in ONE way, and it is the whole gate:
advertisability requires the ``console`` capability bit as well as a live
heartbeat. A running app whose console feature is off (``LOCAL_OPERATOR_UI_CONSOLE_HOST=0``,
a settings toggle, or a ``node-pty`` load failure) must NOT put a ``console`` tool
in the inventory — advertising a tool whose every action errors is worse than
offering none (design §15, first rule), and the app says which of the three it
was in the record rather than in a sentence the harness has to parse.
"""

from __future__ import annotations

from pathlib import Path

from local_operator.browser_bridge import state as _bridge_state
from local_operator.ui_browser import state as _host_state

#: Aliases, not copies. The console has no directory of its own (see the module
#: docstring), so re-spelling the path arithmetic here would be a second place to
#: get it wrong; what is *this* module's is the model and the console clause.
RUN_DIRNAME = _host_state.RUN_DIRNAME
STATE_FILENAME = _host_state.STATE_FILENAME
HEARTBEAT_INTERVAL_S = _host_state.HEARTBEAT_INTERVAL_S
HEARTBEAT_TIMEOUT_S = _host_state.HEARTBEAT_TIMEOUT_S
HOST = _host_state.HOST

Liveness = _host_state.Liveness
pid_alive = _host_state.pid_alive


class ConsoleHostState(_host_state.UiHostState):
    """The app host's record, plus what it says about its console.

    ``console`` is the capability bit the gate reads. The two counters are
    diagnostics the tool can report (how many surfaces exist, how many an agent
    opened) and deliberately NOT inputs to any decision — a host with zero
    surfaces is healthy; ``console_create`` makes one. ``console_proto`` is the
    console namespace's own floor if it ever needs one; unused today, read by
    nothing.
    """

    console: bool = False
    console_surfaces: int = 0
    console_agent_surfaces: int = 0
    console_proto: int = 0


def state_path(root: Path | None = None) -> Path:
    """Where the shared discovery file lives. Pure path arithmetic; creates NOTHING."""
    return _host_state.state_path(root)


def publish(state: ConsoleHostState, root: Path | None = None) -> Path:
    """Write the record atomically, 0600 under a 0700 directory.

    Not called by this runtime — the app writes its own file — but the honest
    counterpart of :func:`read` and what tests use to produce a record for the
    reader to classify. A ``ConsoleHostState`` serialises its console fields, so a
    test's record is byte-for-byte the shape the app publishes.
    """
    return _bridge_state.publish(state, root, dirname=RUN_DIRNAME, filename=STATE_FILENAME)


def read(root: Path | None = None) -> ConsoleHostState | None:
    """Read without mutating, with THIS namespace's model so no field is dropped."""
    return _bridge_state.read(
        root, dirname=RUN_DIRNAME, filename=STATE_FILENAME, model=ConsoleHostState
    )


def heartbeat_age(current: ConsoleHostState, *, now: float | None = None) -> float:
    """Seconds since the host last republished. Negative ages clamp to 0."""
    return _bridge_state.heartbeat_age(current, now=now)


def liveness(
    root: Path | None = None, *, now: float | None = None
) -> tuple[Liveness, ConsoleHostState | None]:
    """Classify the host from the file alone: no socket, no subprocess.

    Byte-for-byte the browser host's classifier, for the reason that module
    states: a live pid plus a fresh heartbeat is the predicate, and the file
    yields three states because the heartbeat lies in both directions and only
    ``STALE`` is worth a socket probe for. The console clause is applied by
    :func:`available`/:func:`advertisable`, not here — a host that is running is
    running, and folding the capability into the liveness answer would make
    ``console_unavailable`` (the feature is off) indistinguishable from ``ABSENT``
    (there is no app), which are two different remedies.
    """
    current = read(root)
    if current is None or not pid_alive(current.pid):
        return Liveness.ABSENT, current
    if _bridge_state.heartbeat_age(current, now=now) <= HEARTBEAT_TIMEOUT_S:
        return Liveness.FRESH, current
    return Liveness.STALE, current


def available(root: Path | None = None, *, now: float | None = None) -> bool:
    """Cheap file-only "known-good right now" gate: FRESH **and** console on."""
    status, current = liveness(root, now=now)
    return status is Liveness.FRESH and current is not None and current.console


def advertisable(root: Path | None = None, *, now: float | None = None) -> bool:
    """Whether the ``console`` TOOL should be offered: FRESH-or-STALE **and** on.

    The weaker commitment the browser's gate also makes — "advertising only
    promises the agent can ASK" — because a host whose heartbeat writer stopped is
    still a host whose RPC socket answers, and hiding the tool from it would leave
    the agent with no console and no explanation for an app that is running
    (design §14.1). The capability bit, by contrast, is NOT a liveness question
    and is required here: see the module docstring.
    """
    status, current = liveness(root, now=now)
    return status in (Liveness.FRESH, Liveness.STALE) and current is not None and current.console
