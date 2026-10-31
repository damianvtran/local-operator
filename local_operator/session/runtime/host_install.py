"""Install-on-demand for the session host's launchd unit.

THE UNIT IS A SIBLING OF ``com.local-operator.wakes``, NOT AN EXTENSION OF IT
(design-surface-independence §2, option (B)). Three reasons, each measured or
read in code, and each is a property this file has to preserve:

1. The wake supervisor SELF-RETIRES: ``KeepAlive: {SuccessfulExit: False}`` plus
   an exit-0 on an empty wake index means a machine with no scheduled wakes runs
   no supervisor, so it cannot be the always-available spawner.
2. Its STOP semantics are the inverse of what is required. On stop it SIGTERMs
   the runtimes it spawned (``wakes/owned_runtimes.sweep``, written for #3028);
   a routine supervisor restart would become a fleet stop — rule R3. The session
   host has NO such sweep, and one must not be added: this module's plist and
   ``host_service``'s exit path signal nothing.
3. Import weight: the wake job's justification is being import-light.

The shape is reuse, not invention: the plist renderer follows
``wakes.install.render_plist``; "installed means RUNNING" is the repair the wakes
installer made (a job launchd merely knows about still prints and still exits 0,
so state is read from ``launchctl print``, never from a file existing); every
``launchctl`` call goes through :mod:`local_operator.launchd`, which refuses to
address the operator's real ``gui/<uid>`` domain from a sandbox.

KEEPALIVE. ``{SuccessfulExit: False}``: a CRASHED host restarts; a host that
EXITED CLEANLY stays down. A clean exit is what a second host does after losing
the single-instance election (the first is serving — restarting the loser would
be a launchd respawn loop against a lock it can never take) and what a host
stopped by SIGTERM does. Neither is a fault, and the cold-start ladder
(:func:`ensure_host_installed`, called from every engage that finds no host)
kickstarts a cleanly-stopped host on demand, so "stays down" costs one
``launchctl kickstart`` on the next engage and never a lost runtime.

``AbandonProcessGroup``. launchd's default on stopping a job is to signal what is
left in the job's process group. Runtimes are ``setsid`` children and are not in
it, and the 2026-10-30 measurement is that ``launchctl kill SIGKILL`` of a
submitted job did not sweep them — but that survival should not rest on one
launchd behaviour. Declaring the abandonment makes "a host stop reaches no
runtime" a property of the plist (rule R3) rather than of an OS detail.

PLATFORM. launchd only in this slice. A machine whose supervisor is systemd or
Task Scheduler gets ``unsupported`` — an HONEST refusal, never a local fork — so
the host route stays opt-in (``session_host.SPAWN_AUTHORITY_ENV``) until those
platforms have a unit.
"""

from __future__ import annotations

import logging
import plistlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from local_operator import launchd, procname, supervisors

logger = logging.getLogger(__name__)

#: Beside ``com.local-operator.wakes`` / ``.tunnel`` / ``.mobile`` in
#: ``launchctl list`` so the supervised units read as siblings.
LABEL = "com.local-operator.session-host"

#: What the host runs as. A module entry point, so Gate A (design §8) can run the
#: very same code as an ordinary subprocess; the launchd wiring is not what that
#: gate tests, and saying so is honest.
HOST_MODULE = "local_operator.session.runtime.host_service"


def plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def log_path(config_dir: Path) -> Path:
    return config_dir / "logs" / "session-host.log"


def render_plist(config_dir: Path) -> dict[str, object]:
    """The whole supervised-unit plan in one pure function.

    What tests assert is what launchd is handed.
    """
    return {
        "Label": LABEL,
        **procname.launchd_job(
            HOST_MODULE,
            label=procname.branded_argv0(procname.LABEL_SESSION_HOST),
        ),
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "AbandonProcessGroup": True,
        "StandardOutPath": str(log_path(config_dir)),
        "StandardErrorPath": str(log_path(config_dir)),
        # The store is part of the contract, exactly as for the wake supervisor.
        # Deliberately NOTHING ELSE: launchd's bare PATH is repaired per request
        # from the engaging surface's environment (design §2), not baked here —
        # a PATH in the plist would vary per calling session and turn the
        # content comparison below into a restart loop.
        "EnvironmentVariables": {"LOCAL_OPERATOR_CONFIG_DIR": str(config_dir)},
    }


@dataclass(frozen=True)
class HostInstallOutcome:
    """Never an exception; ``code`` is a log/test token, not user copy.

    ``code`` is ``""`` on success, else ``unsupported`` (no launchd here) or
    ``unavailable`` (launchd declined: sandboxed home, not the default store, a
    failed bootstrap).
    """

    ok: bool
    code: str = ""
    reason: str = ""


def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["launchctl", *args], capture_output=True, text=True, timeout=15
    )


def ensure_host_installed(config_dir: Path) -> HostInstallOutcome:
    """Make the unit installed AND running for ``config_dir``; never raises.

    Idempotent, and cheap when it is already right (one ``launchctl print``). The
    repairs mirror ``wakes.install.ensure_supervisor_installed``:

    * plist missing or content-different → write it and ``reload_job``
      (bootout + wait + bootstrap, with launchd's transient-EIO retries);
    * plist right, job LIVE → nothing to do;
    * plist right, job loaded but stopped → ``kickstart -k``;
    * plist right, job not registered → ``reload_job`` (only a bootstrap
      registers a label launchd has forgotten).

    This function STARTS THE HOST UNDER LAUNCHD. It never starts it as a child of
    the caller: a host that is the app's child is the bug.
    """
    if supervisors.supervisor() != supervisors.LAUNCHCTL:
        return HostInstallOutcome(
            False, "unsupported", "the session host needs launchd; no supported supervisor here"
        )
    try:
        refusal = launchd.shared_label_refusal(config_dir)
        if refusal is not None:
            logger.warning("session host: %s", refusal)
            return HostInstallOutcome(False, "unavailable", refusal)
        path = plist_path()
        wanted = render_plist(config_dir)
        current = None
        if path.exists():
            try:
                current = plistlib.loads(path.read_bytes())
            except Exception:  # noqa: BLE001 — an unreadable plist is a stale one
                current = None
        if current != wanted:
            path.parent.mkdir(parents=True, exist_ok=True)
            log_path(config_dir).parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(plistlib.dumps(wanted))
        else:
            observation = launchd.observe_job(label=LABEL, path=path, run=_launchctl)
            if observation.state == launchd.JobState.LIVE:
                return HostInstallOutcome(True)
            if observation.state == launchd.JobState.UNKNOWN:
                # AN UNANSWERED ``launchctl print`` IS NOT A DEAD HOST (#3222 R-4):
                # under load it times out on a perfectly healthy job, and
                # ``kickstart -k`` would SIGKILL it and the request in flight.
                # Declining here costs nothing -- the caller's bounded connect is
                # the real liveness test and raises honestly if nobody answers.
                return HostInstallOutcome(True)
            if launchd.kickstart(label=LABEL, path=path, run=_launchctl):
                return HostInstallOutcome(True)
            # kickstart declined: launchd has forgotten the label (or refused);
            # fall through to the bootstrap, which reports its own failure.
        reloaded = launchd.reload_job(label=LABEL, path=path, runner=_launchctl)
        if not reloaded.ok:
            return HostInstallOutcome(
                False,
                "unavailable",
                f"launchctl could not load the session host: {reloaded.detail}",
            )
        return HostInstallOutcome(True)
    except Exception as exc:  # noqa: BLE001 — never raises; the engage reports it
        logger.debug("session host install failed", exc_info=True)
        return HostInstallOutcome(False, "unavailable", f"install failed: {exc}")
