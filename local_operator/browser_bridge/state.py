"""Atomic discovery state for the browser bridge daemon.

The state file is both cheap createIf discovery and session-leg authorization.
It therefore stays 0600 under a 0700 directory and is replaced atomically so a
session can never consume a half-written key or port.
"""

from __future__ import annotations

import enum
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from local_operator import procstate
from local_operator.paths import config_dir

RUN_DIRNAME = "run/browser"
STATE_FILENAME = "bridge.json"
HEARTBEAT_INTERVAL_S = 15.0
HEARTBEAT_TIMEOUT_S = 45.0


class HeartbeatState(BaseModel):
    """Base for a host discovery record: every host stamps its own heartbeat.

    Exists so the primitives below (``state_path``/``publish``/``read``) are
    generic over WHICH host's file they are addressing while still owning the
    one thing they must own — the atomic write and the heartbeat stamp. The
    UI browser host's record (``local_operator/ui_browser/state.py``) is the
    second implementation, and it deliberately reuses these primitives rather
    than copying them, because a second atomic-write implementation is a second
    chance to publish a half-written key or a non-0600 file.
    """

    heartbeat_at: float = Field(default_factory=time.time)
    started_at: float = Field(default_factory=time.time)


#: The record type ``read`` returns unless a caller names its own model. Bound
#: to a plain ``BaseModel`` rather than to ``HeartbeatState`` so a future host
#: whose file carries no heartbeat is not forced to invent one.
StateT = TypeVar("StateT", bound=BaseModel)


class HeartbeatStamp(Protocol):
    """Anything carrying a heartbeat stamp: what :func:`heartbeat_age` needs.

    Structural rather than a base class so a host can be stamped without being
    forced into this module's hierarchy.
    """

    heartbeat_at: float


class BridgeState(HeartbeatState):
    model_config = ConfigDict(extra="ignore")

    pid: int
    port: int
    session_key: str = Field(min_length=32)
    proto: int
    extension_connected: bool = False
    paired: bool = False
    extension_id: str = ""
    browser_name: str = ""
    #: Whether the daemon has latched "the attached extension stopped answering"
    #: and dropped its link for it. Related to `/health`'s `extension_unresponsive`
    #: but NOT equal to it by design (review round 5, NIT 4): `/health` reports the
    #: latch only while no proven link is serving, so for the TTL window after a
    #: promotion this file says "latched" where `/health` says "the driver is
    #: answering". Published because the reader that needs it most
    #: cannot ask: `_execute_browser`'s demotion guard runs on the ABSENT side of
    #: `liveness`, where the contract forbids a socket probe, and since a drop
    #: writes `extension_connected=false` the file alone would otherwise look
    #: exactly like a host with no bridge at all — which is how a paired, running
    #: bridge got told to run `lop browser install` (design D3-2).
    #:
    #: Defaults false, so a file written by an older daemon (and every fixture)
    #: reads as "no latch" — the conservative answer, since a false positive here
    #: would claim a wedge the daemon never reported.
    extension_unresponsive: bool = False
    #: The attached extension's OWN reported version, as the daemon last saw it
    #: in `hello`, and its protocol version. Published because the session-side
    #: decision they drive — whether this link can use the `owner_*` lifecycle
    #: at all (see `browser_bridge/resources.py`) — must not cost a socket
    #: round-trip, and because the file is the only surface a session between
    #: dials can read. Blank/0 when no link is proven, so a stamp outliving its
    #: socket can never drive that decision.
    extension_version: str = ""
    extension_proto: int = 0
    #: The wire methods the attached extension ADVERTISED, sorted (design §6.3).
    #:
    #: Published for the same reason `extension_version` is: the session-side
    #: decision it drives — whether `download`/`upload` may be sent at all — must
    #: not cost a socket round-trip, and the file is the only surface a session
    #: between dials can read. Empty when no link is proven, so a stamp outliving
    #: its socket can never authorise a method nobody serves.
    #:
    #: `extra="ignore"` on this model is what makes the field safe in BOTH
    #: directions: an old harness ignores a key it does not know, and a new
    #: harness reads a record written by an older daemon as the empty default —
    #: "the host told us nothing", which is exactly the refusal case.
    capabilities: list[str] = []
    #: Whether the DAEMON that wrote this file speaks capability advertisement at
    #: all — its own build stamp, NOT the extension's.
    #:
    #: `capabilities` alone cannot attribute an empty list: a daemon that predates
    #: the advertisement writes a record with no such key, and a current daemon
    #: writing "the peer advertised nothing" produces the very same empty list.
    #: Those two causes have OPPOSITE remedies — restart the bridge versus toggle
    #: the extension — and design §6.4 promises the model is told which one it is
    #: (review round 1, R4). A daemon at or after the advertisement always sets
    #: this, so its ABSENCE names the writer. Defaults false, so a record written
    #: by an older daemon (and every fixture) reads as "an old writer".
    capabilities_known: bool = False
    #: The servable methods the OPERATOR has switched off in the extension's own
    #: options, as the extension last reported them (protocol.CapabilitySwitches).
    #:
    #: Published for the same reason `capabilities` is: the session-side decision it
    #: drives — WHICH remedy the refusal names — must not cost a socket round-trip.
    #: It is a separate list rather than "absent from capabilities" because the two
    #: absences have opposite remedies: a method the build cannot serve needs an
    #: UPDATE, and one the operator has not enabled needs the switch. Blanked when
    #: no link is proven, like every other proven-only fact here.
    disabled_capabilities: list[str] = []
    #: Whether the DAEMON that wrote this file speaks the switch advertisement at
    #: all — its own build stamp, NOT the extension's, exactly like
    #: `capabilities_known` and for the same reason: an empty `disabled` list has
    #: two causes (the operator enabled everything / nobody told us about switches)
    #: and only the writer can say which. Absent (false) means a daemon that
    #: predates the switches, whose records must keep reading as "no switch answer",
    #: never as "the operator enabled it".
    switches_known: bool = False
    #: Whether a KNOWN extension version is strictly below the one this runtime
    #: ships with (`protocol.EXPECTED_EXTENSION_VERSION`). The predicate lives
    #: in the daemon (see `BridgeService.publish`) and is published rather than
    #: recomputed here, so there is exactly one spelling of "an update is
    #: available". Defaults false, so a file written by an older daemon never
    #: nags.
    extension_update_available: bool = False


def run_dir(root: Path | None = None, *, dirname: str = RUN_DIRNAME) -> Path:
    """The run directory, CREATED and locked down. Only writers may call this.

    ``dirname`` is a keyword-only default so a SECOND host's namespace can reuse
    this without a second implementation of the mkdir/chmod pair. Every
    parameter defaults to today's value, so no existing call site changes
    behaviour (the ``AGENTS.md`` tool-surface ladder's "extend, do not invent a
    parallel mechanism" rule, applied to a directory rather than a tool).
    """
    directory = (root or config_dir()) / dirname
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    return directory


def state_path(
    root: Path | None = None,
    *,
    dirname: str = RUN_DIRNAME,
    filename: str = STATE_FILENAME,
) -> Path:
    """Where the discovery file lives. Pure path arithmetic: creates NOTHING.

    It used to route through :func:`run_dir`, which mkdirs and chmods, so every
    READER and every diagnostic performed a write. That turned the ENOSPC log
    line in ``BridgeService.publish_safely`` into a second ``OSError`` raised
    from inside the handler for the first one: on a fresh config dir with a
    full disk the daemon failed to boot, in precisely the disk-full scenario
    this module is meant to survive. An error path may never perform the
    operation that is failing, and detection may never mutate the filesystem
    (see :func:`read`), so the path is now derived without touching disk and
    only the writer (:func:`publish`) asks for the directory to exist.
    """
    return (root or config_dir()) / dirname / filename


def publish(
    state: HeartbeatState,
    root: Path | None = None,
    *,
    dirname: str = RUN_DIRNAME,
    filename: str = STATE_FILENAME,
) -> Path:
    """Staged write + ``os.replace``, 0600 under a 0700 directory.

    The temporary file's prefix names the host whose file is being replaced, so
    an interrupted write is attributable to one namespace.
    """
    directory = run_dir(root, dirname=dirname)
    state.heartbeat_at = time.time()
    fd, temporary = tempfile.mkstemp(
        dir=directory, prefix=f".{Path(filename).stem}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(state.model_dump(mode="json"), handle)
        os.chmod(temporary, 0o600)
        target = directory / filename
        os.replace(temporary, target)
        return target
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def read(
    root: Path | None = None,
    *,
    dirname: str = RUN_DIRNAME,
    filename: str = STATE_FILENAME,
    model: type[StateT] = BridgeState,
) -> StateT | None:
    """Read without mutating or reaping; detection must have no side effects.

    ``model`` is selectable because this module's own ``BridgeState`` declares
    ``extra="ignore"``: reading another host's file through it would silently
    DROP that host's own fields (the UI host's ``host``, ``app_version``,
    ``profile_dir``, ``agent_tabs``), which is harmless but blind — the file
    would parse and every host-specific fact would be missing with no error.
    """
    try:
        raw: Any = json.loads(
            state_path(root, dirname=dirname, filename=filename).read_text(encoding="utf-8")
        )
        return model.model_validate(raw)
    except (OSError, ValueError, TypeError):
        return None


def pid_alive(pid: int) -> bool:
    """Whether some process still holds this pid.

    Delegates to :func:`local_operator.procstate.pid_alive`, which asks the
    right question per platform: `os.kill(pid, 0)` here would TERMINATE the
    bridge daemon it is probing on Windows (signal 0 is `TerminateProcess`
    there), and this decides whether the daemon's state file is trusted.
    """
    return procstate.pid_alive(pid)


def heartbeat_age(current: HeartbeatStamp, *, now: float | None = None) -> float:
    """Seconds since the daemon last republished. Negative ages clamp to 0.

    Clock skew (or a state file written by a daemon whose clock ran ahead) must
    never read as "fresher than fresh" and must never render as a negative age
    in diagnostics, so the floor is 0.
    """
    timestamp = time.time() if now is None else now
    return max(0.0, timestamp - current.heartbeat_at)


class Liveness(enum.Enum):
    """What the discovery FILE alone can honestly conclude about the daemon.

    The file heartbeat is a liveness PROXY, and it lies in both directions: it
    goes stale while the daemon is perfectly healthy (the heartbeat writer can
    die on its own — see ``BridgeService._supervise``, or the daemon can be
    SIGSTOPped) and it stays fresh for a few seconds after a daemon is killed.
    Collapsing that into one bool is what made a healthy daemon read as a
    permanent "no": every session silently fell back to cmux while
    ``lop browser status`` — which reads the LIVE ``/health`` socket — kept
    reporting the extension connected, and nothing reconciled the two.

    So the file answers three states, not two, and the caller decides how much
    a given answer is worth paying for:

    - ``ABSENT``  no daemon, or no browser attached. A definite no; never probe.
    - ``FRESH``   heartbeat inside the timeout. A definite yes; never probe.
    - ``STALE``   heartbeat expired but the pid is ALIVE and an extension was
      attached when the file was last written. Genuinely unknown from the file:
      only a socket round-trip can settle it (``bridge_browser_reachable``).
    """

    ABSENT = "absent"
    FRESH = "fresh"
    STALE = "stale"


def liveness(
    root: Path | None = None, *, now: float | None = None
) -> tuple[Liveness, BridgeState | None]:
    """Classify the daemon from the file alone: no socket, no subprocess."""
    current = read(root)
    if current is None or not current.extension_connected or not pid_alive(current.pid):
        return Liveness.ABSENT, current
    if heartbeat_age(current, now=now) <= HEARTBEAT_TIMEOUT_S:
        return Liveness.FRESH, current
    return Liveness.STALE, current


def available(root: Path | None = None, *, now: float | None = None) -> bool:
    """Cheap file-only availability gate: FRESH only, no socket, never probes.

    Deliberately still FILE-ONLY and still false for a stale heartbeat. It
    answers "is the bridge known-good right now", which is the question the
    backend-selection paths ask. A stale-but-alive daemon is acquitted on the
    browser path instead, by :func:`~local_operator.browser_bridge.backend.
    bridge_browser_reachable`, which pays for one bounded probe only when it is
    about to condemn the bridge.

    For TOOL GATING use :func:`advertisable` instead — see why there.
    """
    return liveness(root, now=now)[0] is Liveness.FRESH


def advertisable(root: Path | None = None, *, now: float | None = None) -> bool:
    """Whether the `browser` TOOL should be offered. FRESH or STALE-but-alive.

    Gating is a weaker commitment than execution: advertising the tool only
    promises the agent can ASK, and `execute_browser` still decides — with a
    real socket probe — whether the extension answers, falling back to cmux or
    returning the typed demotion diagnostic. So the gate must not apply the
    stricter :func:`available` test.

    It did, and that is a hole in the RC2 rescue: gating ran the FRESH-only
    check, so on an extension-only host (no cmux, the ordinary configuration
    for the extension) a daemon whose heartbeat writer had died was never
    advertised at all. The tool list is built once per session, so that session
    had NO browser tool for its lifetime, `execute_browser` was never reached,
    the socket probe never ran, and the demotion hint — whose every call site
    is inside `execute_browser` — could not fire. The agent got no fallback and
    no diagnostic: strictly worse than the incident this fixes, since it cannot
    even discover that a healthy daemon is sitting there.

    The CONSTRAINT that made the gate file-only still holds and is respected:
    this is synchronous and runs while constructing EVERY session, so it must
    not block or do unbounded I/O. It does neither. STALE is already
    established by :func:`liveness` from one file read plus ``os.kill(pid, 0)``
    — no socket is opened here and no subprocess is spawned, so the hot path
    keeps its measured sub-millisecond cost. When in doubt this errs toward
    advertising: a tool that explains why it cannot reach the bridge beats a
    tool that silently does not exist.
    """
    return liveness(root, now=now)[0] in (Liveness.FRESH, Liveness.STALE)


def remove(
    root: Path | None = None,
    *,
    dirname: str = RUN_DIRNAME,
    filename: str = STATE_FILENAME,
) -> None:
    try:
        state_path(root, dirname=dirname, filename=filename).unlink()
    except OSError:
        pass
