"""Atomic discovery state for the desktop app's browser host.

Same shape and the same two guarantees as :mod:`local_operator.browser_bridge.
state` — 0600 under a 0700 directory, staged write + ``os.replace``, and a read
that creates nothing — in its OWN namespace, because the primitives are shared
and only the namespace, the record type and the liveness predicate differ.

Why a separate namespace rather than a sibling of ``run/browser/bridge.json``:
``lop browser status`` reads the bridge's path through ``browser_state.read()``,
and the bridge's install/cleanup/health paths all assume that directory belongs
to the daemon. A second, unrelated process's record beside it invites exactly
the confusion a ``lop browser cleanup`` sweep would then cause. The same reason,
one level up, as ``docs/design-daemon-discovery.md``'s "a new namespace, not the
session one".

Why the predicate is NOT ``browser_bridge.state.liveness``: that function folds
``extension_connected`` into its classification, because the bridge's browser is a
separate process the user can close while the daemon stays up. The UI host has no
such field and no such third party — its views are its own children, so if the
process is alive and heartbeating it can create a browser tab on demand. Reusing
the bridge's predicate verbatim would classify a perfectly healthy UI host as
ABSENT forever.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import ConfigDict, Field

from local_operator.browser_bridge import state as _bridge_state
from local_operator.browser_bridge.state import HeartbeatState

RUN_DIRNAME = "run/ui-browser"
STATE_FILENAME = "host.json"

#: The heartbeat constants are SHARED with the bridge rather than re-declared.
#: They describe one mechanism (a file heartbeat that may lag its process), and
#: two spellings of "how stale is too stale" would drift silently.
HEARTBEAT_INTERVAL_S = _bridge_state.HEARTBEAT_INTERVAL_S
HEARTBEAT_TIMEOUT_S = _bridge_state.HEARTBEAT_TIMEOUT_S

#: The value of the record's ``host`` field, and the pyright-visible marker that
#: this file is the UI host's. Written by the app; checked by nothing on the read
#: path (the namespace already separates the files), but kept so `lop`-side
#: diagnostics can print WHICH host answered without inferring it from a path.
HOST = "ui"

Liveness = _bridge_state.Liveness
pid_alive = _bridge_state.pid_alive


class UiHostState(HeartbeatState):
    """The desktop app's published browser-host record.

    Extends the shared `HeartbeatState` rather than declaring its own
    `heartbeat_at`/`started_at`, because the heartbeat IS the shared mechanism: a
    second declaration would be a second default and a second spelling of "how
    stale is too stale" for one published fact.

    ``profile_dir`` and ``agent_tabs`` are diagnostic only and deliberately NOT
    inputs to any decision: the first answers "where are my logins" (it is
    ``ses.getStoragePath()``, §5.3 of the design), and the second makes the
    agent-tab cap observable. ``tabs`` is not part of the liveness predicate
    either — a host with zero tabs is healthy; `open` creates one.
    """

    model_config = ConfigDict(extra="ignore")

    pid: int
    port: int
    session_key: str = Field(min_length=32)
    proto: int
    host: str = HOST
    app_version: str = ""
    profile_dir: str = ""
    tabs: int = 0
    agent_tabs: int = 0


def state_path(root: Path | None = None) -> Path:
    """Where the discovery file lives. Pure path arithmetic: creates NOTHING."""
    return _bridge_state.state_path(root, dirname=RUN_DIRNAME, filename=STATE_FILENAME)


def publish(state: UiHostState, root: Path | None = None) -> Path:
    """Write the record atomically, 0600 under a 0700 directory.

    Not called by this runtime — the app writes its own file — but the honest
    counterpart of :func:`read`, and what tests use to produce a record for the
    reader to classify without weakening the publisher (the same rule the
    bridge's own state tests state explicitly).
    """
    return _bridge_state.publish(state, root, dirname=RUN_DIRNAME, filename=STATE_FILENAME)


def read(root: Path | None = None) -> UiHostState | None:
    """Read without mutating; the UI host's own model, so no field is dropped."""
    return _bridge_state.read(root, dirname=RUN_DIRNAME, filename=STATE_FILENAME, model=UiHostState)


#: Deliberately no `run_dir`/`remove` wrappers here: nothing in this runtime
#: creates or deletes the app's record (the app owns its own file's lifecycle). A
#: namespace-bound wrapper per primitive is how a module grows an API nothing
#: calls and nobody dares delete; the shared primitive is one call away if a
#: cleanup path ever needs it.


def heartbeat_age(current: UiHostState, *, now: float | None = None) -> float:
    """Seconds since the host last republished. Negative ages clamp to 0."""
    return _bridge_state.heartbeat_age(current, now=now)


def liveness(
    root: Path | None = None, *, now: float | None = None
) -> tuple[Liveness, UiHostState | None]:
    """Classify the host from the file alone: no socket, no subprocess.

    Clause by clause this mirrors the bridge's classifier minus the one clause
    that has no analogue here (see the module docstring): a live pid plus a
    heartbeat inside the timeout is the whole predicate, and the file still
    yields THREE states rather than two, because the heartbeat lies in both
    directions and only ``STALE`` is worth paying a socket probe for.

    A protocol mismatch is deliberately NOT folded in here. A version-skewed host
    is a real host that can explain itself: it is advertisable, its socket
    answers, and the per-action path returns the typed ``proto_mismatch`` whose
    copy names the remedy. Silently unadvertising would leave the agent with no
    browser tool and no explanation for a host that is running.
    """
    current = read(root)
    if current is None or not pid_alive(current.pid):
        return Liveness.ABSENT, current
    if _bridge_state.heartbeat_age(current, now=now) <= HEARTBEAT_TIMEOUT_S:
        return Liveness.FRESH, current
    return Liveness.STALE, current


def available(root: Path | None = None, *, now: float | None = None) -> bool:
    """Cheap file-only "known-good right now" gate: FRESH only, never probes."""
    return liveness(root, now=now)[0] is Liveness.FRESH


def advertisable(root: Path | None = None, *, now: float | None = None) -> bool:
    """Whether the `browser` TOOL should be offered: FRESH or STALE-but-alive.

    The same weaker-commitment rule the bridge's gate uses, for the same reason:
    advertising only promises the agent can ASK, and `execute_browser` still
    decides with a real probe whether the host answers.
    """
    return liveness(root, now=now)[0] in (Liveness.FRESH, Liveness.STALE)
