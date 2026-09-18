"""User-level service lifecycle; no root installer or global cloudflared login.

One daemon, three supervisors (see :mod:`local_operator.supervisors`): a
LaunchAgent plist on macOS, a ``systemd --user`` unit on Linux, a Task Scheduler
task on Windows. All three record the store the connector serves
(``LOCAL_OPERATOR_CONFIG_DIR``), because a tunnel belongs to a config root.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

from local_operator import launchd, procname, supervisors
from local_operator.paths import CONFIG_DIR_ENV, config_dir
from local_operator.tunnels import config

LABEL = "com.local-operator.tunnel"

#: Linux user unit name, and the Task Scheduler task name (Windows). Fixed, like
#: ``LABEL``: the store travels in the unit's environment, so a second config
#: root retargets this connector rather than shadowing it.
SYSTEMD_UNIT = "lop-tunnel.service"
TASK_NAME = "Local Operator tunnel"

#: The refusal every entry point shares when the machine has no user supervisor.
NO_SUPERVISOR_ERROR = supervisors.no_supervisor_error("lop tunnel serve")


def task_record_path() -> Path:
    """Our own copy of the Windows task definition (Task Scheduler keeps the original)."""
    return config.directory() / "service-task.xml"


def service_path() -> Path:
    kind = supervisors.supervisor()
    if kind == supervisors.LAUNCHCTL:
        return Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
    if kind == supervisors.SYSTEMCTL:
        return supervisors.systemd_unit_path(SYSTEMD_UNIT)
    if kind == supervisors.SCHTASKS:
        return task_record_path()
    raise ValueError(NO_SUPERVISOR_ERROR)


def _quoted(value: str) -> str:
    """Quote a value for systemd's OWN grammar, which is not shell escaping.

    Percent is doubled because unit specifiers expand even inside quoted
    strings; a path or a store with a ``%`` in it would otherwise be silently
    mangled. Hoisted out of ``install`` so the tunnel's unit and the shared
    renderer's caller agree on one spelling.
    """
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%") + '"'


def render_systemd(config_base: Path | None = None) -> str:
    """The systemd user unit, for the store ``config_base`` names.

    ``config_base`` defaults to this process's config dir, which is what
    ``install`` wants; a repair would pass the store recorded in the unit it is
    replacing (see :func:`local_operator.tunnels.config.directory`).

    ``UMask=0077`` is load-bearing rather than tidy: the connector's
    ``cloudflared.token`` is written 0600 by ``config.private_write`` and
    nothing else protects it, so the unit must not widen it at write time.
    """
    base = config_base if config_base is not None else config_dir()
    # The stable shim when this machine has the generation layout, else this
    # process's interpreter: a unit restarts, and a path inside a tree a flip or
    # a prune replaced is a daemon that dies at load.
    image = procname.supervised_image() or sys.executable
    return supervisors.render_systemd_unit(
        description="Radient personal tunnel",
        after="network-online.target",
        pre_lines=["Type=simple"],
        exec_start=f"{_quoted(str(image))} -m local_operator.tunnels.service",
        post_lines=[
            f"Environment={_quoted(f'{CONFIG_DIR_ENV}={base}')}",
            "UMask=0077",
        ],
        restart_sec=10,
    )


def render_task_xml(config_base: Path | None = None) -> str:
    """The Windows Task Scheduler task for the connector, store included."""
    base = config_base if config_base is not None else config_dir()
    image = procname.supervised_image() or sys.executable
    return supervisors.render_task_xml(
        description="Radient personal tunnel connector",
        image=str(image),
        argv=["-m", "local_operator.tunnels.service"],
        environment={CONFIG_DIR_ENV: str(base)},
        log=config.directory(base) / "service.log",
        user_id=supervisors.current_user_id(),
    )


def _run(args: list[str], *, checked: bool = True) -> None:
    """Run a supervisor command, refusing LEGIBLY when the binary is absent.

    ``subprocess.run(check=False)`` suppresses a non-zero exit status but NOT
    ``FileNotFoundError`` for a missing executable, so a Linux without systemd
    (Devuan, Alpine, most containers, WSL2 without systemd) used to escape as an
    ``OSError`` from ``lop tunnel install`` — which ``tunnels/cli.py`` renders as
    *"check network access and your Radient login"*. The diagnosis was wrong,
    and it is precisely the misreport ``browser_bridge._supervisor`` exists to
    prevent: a missing user service manager is not a network problem.
    """
    if shutil.which(args[0]) is None:
        raise ValueError(NO_SUPERVISOR_ERROR)
    result = subprocess.run(args, capture_output=True, timeout=20)  # noqa: S603 — fixed argv
    if checked and result.returncode:
        # ``capture_output=True`` without ``text=True`` answers in BYTES, but a
        # test double (or a future `text=True`) answers in str — one adapter
        # rather than a shape every caller has to remember, same as
        # ``launchd._text``.
        raw = result.stderr if result.stderr is not None else b""
        stderr = (raw.decode(errors="replace") if isinstance(raw, bytes) else raw).strip()
        raise ValueError(
            "Tunnel service action failed; check your user service manager."
            + (f"\n{stderr[:200]}" if stderr else "")
        )


def _launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    """This daemon's ``launchctl``, threaded into :func:`launchd.reload_job`.

    ``_run`` above stays for systemd, whose failures carry their own message.
    This one answers with launchd's stderr intact, because the reload reports
    launchd's own reason to the operator rather than a generic sentence.
    """
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["launchctl", *args], capture_output=True, text=True, timeout=20
    )


def render_plist(config_base: Path | None = None) -> dict[str, object]:
    """The whole supervised-unit plan, for the store ``config_base`` names.

    ``config_base`` defaults to this process's config dir, which is what
    ``install`` wants. The LaunchAgent repair passes the store recorded in the
    plist it is replacing instead — see
    :func:`local_operator.tunnels.config.directory`.

    ``launchd_job`` rather than ``launchd_program``: ``Program`` is the branded
    image and ``ProgramArguments[0]`` is this daemon's role label, which is what
    stops the tunnel reading as the same bare ``Local Operator`` row as the
    other three daemons (and, before branding, as ``python3``). macOS names a
    background item by the basename of ``ProgramArguments[0]``, so the old
    ``sys.executable`` here is what made installing notify that 'python3 is
    running in the background'. The trade-off that shape accepts is recorded in
    ``procname.launchd_job``; with no link to plant this is the same plist as
    before. On a machine with the generation layout ``Program`` is the stable
    shim instead (see ``procname.supervised_image``) and the role label is
    unchanged.
    """
    base = config_base if config_base is not None else config_dir()
    return {
        "Label": LABEL,
        **procname.launchd_job(
            "local_operator.tunnels.service",
            label=procname.branded_argv0(procname.LABEL_TUNNEL),
        ),
        "EnvironmentVariables": {"LOCAL_OPERATOR_CONFIG_DIR": str(base)},
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False},
        "ThrottleInterval": 10,
        "StandardOutPath": str(config.directory(base) / "service.log"),
        "StandardErrorPath": str(config.directory(base) / "service.log"),
    }


def refresh_plist_if_stale() -> launchd.PlistRefresh:
    """Rewrite this daemon's LaunchAgent when an older build wrote it.

    The tunnel had no repair path at all: ``lop-update`` bounced the mobile
    daemon and nothing else, so a plist written before branding kept running a
    bare ``python3.14 -m local_operator.tunnels.service`` forever — measured on
    the operator's machine as ``com.local-operator.tunnel.plist`` and pid 92821.

    Two guards, both inherited from ``wakes.install`` rather than reinvented:
    the plist must be the one the real passwd home produces (``launchctl``
    always addresses the real user's session, whatever ``HOME`` says), and the
    store it records must live under the real home (otherwise this would point
    the operator's real LaunchAgent at a sandbox store that disappears).

    Never raises; a repair that cannot run leaves the upgrade that called it
    exactly as it was.
    """
    name = "tunnel"
    try:
        if supervisors.supervisor() != supervisors.LAUNCHCTL:
            # The systemd user unit has no plist to repair; it re-reads its unit
            # file on every start, so it has never had this failure mode.
            return launchd.PlistRefresh(name=name, kind="unsupported")
        path = service_path()
        if not launchd.is_own_plist(path, LABEL):
            return launchd.PlistRefresh(name=name, kind="not-addressable")
        current = launchd.load(path)
        base = launchd.config_dir_from_plist(current) or config_dir()
        if not launchd.config_lives_in_real_home(base):
            return launchd.PlistRefresh(name=name, kind="not-addressable")
        outcome = launchd.rewrite_if_stale(name=name, path=path, rendered=render_plist(base))
        if outcome.kind != "repaired":
            return outcome
        # bootout + bootstrap through the shared helper, NOT kickstart -k: a
        # kickstart restarts the job from launchd's in-memory definition, so it
        # would keep running the old argv after this rewrite. Measured; see
        # :mod:`local_operator.launchd`.
        reloaded = launchd.reload_job(label=LABEL, path=path, runner=_launchctl)
        if not reloaded.ok:
            # Names the recovery, because the job is DOWN at this point: see
            # `launchd.reload_failure`.
            return reloaded.as_refresh_failure(name=name, path=path, recovery="lop tunnel install")
        return outcome
    except Exception as exc:  # noqa: BLE001 — a repair must never fail an upgrade
        return launchd.PlistRefresh(name=name, kind="failed", detail=str(exc))


def install() -> None:
    kind = supervisors.supervisor()
    if kind is None:
        raise ValueError(NO_SUPERVISOR_ERROR)
    path = service_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == supervisors.LAUNCHCTL:
        log = config.directory() / "service.log"
        config.private_write(log, "")
        value = render_plist()
        path.write_bytes(plistlib.dumps(value))
        path.chmod(0o600)
        # Reload through the shared helper, and REPORT what launchd said. This
        # used to issue bootout+bootstrap and raise a generic "check your user
        # service manager" that discarded launchd's stderr while the CLI went on
        # to print "Tunnel service started." — a false success over a daemon
        # that had just been booted out and never loaded back. See
        # :mod:`local_operator.launchd` for the measurement.
        reloaded = launchd.reload_job(label=LABEL, path=path, runner=_launchctl)
        if not reloaded.ok:
            # The same sentence the upgrade path prints, because the state is
            # the same one it describes: the plist is written and the job is
            # not loaded. `lop tunnel install` is the working recovery.
            raise ValueError(
                launchd.reload_failure("tunnel", path, "lop tunnel install", reloaded.detail).detail
            )
    elif kind == supervisors.SYSTEMCTL:
        path.write_text(render_systemd())
        path.chmod(0o600)
        _run(["systemctl", "--user", "daemon-reload"])
        _run(["systemctl", "--user", "enable", "--now", path.name])
    else:  # schtasks
        xml = render_task_xml()
        # The record first: it is what `uninstall` and a future diff read, and it
        # is the only visible copy of a definition Task Scheduler keeps in the
        # registry.
        config.private_write(path, xml)
        ok, detail = supervisors.create_task(TASK_NAME, xml)
        if not ok:
            raise ValueError(f"schtasks could not register the tunnel service: {detail}")
        started = supervisors.schtasks(*supervisors.task_run_args(TASK_NAME))
        if started.returncode:
            raise ValueError(
                "the tunnel task was registered but could not be started: "
                f"{((started.stderr or started.stdout) or '').strip()[:200]}"
            )


def action(name: str) -> None:
    kind = supervisors.supervisor()
    if kind is None:
        raise ValueError(NO_SUPERVISOR_ERROR)
    path = service_path()
    # Windows has no registration FILE this installer owns (Task Scheduler keeps
    # its own copy), so "the record is gone" must not read as "not installed".
    if not path.exists() and kind != supervisors.SCHTASKS:
        raise ValueError(
            "Tunnel service not installed. Run lop tunnel install or lop tunnel serve."
        )
    if kind == supervisors.SYSTEMCTL and shutil.which("systemctl") is None:
        # Reachable through `lop tunnel start` on a systemd-less Linux: the unit
        # file may exist from a machine backup, but nothing here can run it.
        raise ValueError(NO_SUPERVISOR_ERROR)
    if kind == supervisors.LAUNCHCTL:
        if name == "stop":
            # A bare bootout, deliberately: stopping is not a reload, and there
            # is nothing to bootstrap afterwards.
            _run(["launchctl", "bootout", f"gui/{os.getuid()}/{LABEL}"], checked=False)
        else:
            # start/restart both mean "load the plist that is on disk now", so
            # both go through the shared reload: the old shape bootstrapped with
            # its failure ignored and then kickstarted, which reported success
            # for a job that was never registered.
            reloaded = launchd.reload_job(label=LABEL, path=path, runner=_launchctl)
            if not reloaded.ok:
                raise ValueError(
                    launchd.reload_failure(
                        "tunnel", path, "lop tunnel install", reloaded.detail
                    ).detail
                )
    elif kind == supervisors.SYSTEMCTL:
        _run(["systemctl", "--user", name, path.name])
    else:  # schtasks
        if name in ("stop", "restart"):
            ended = supervisors.schtasks(*supervisors.task_end_args(TASK_NAME))
            if ended.returncode and "running" not in ((ended.stderr or ended.stdout) or "").lower():
                raise ValueError(((ended.stderr or ended.stdout) or "").strip()[:200])
        if name in ("start", "restart"):
            started = supervisors.schtasks(*supervisors.task_run_args(TASK_NAME))
            if started.returncode:
                raise ValueError(((started.stderr or started.stdout) or "").strip()[:200])


def uninstall() -> None:
    kind = supervisors.supervisor()
    if kind is None:
        # Nothing this installer could have registered; the connector runs in the
        # foreground there, and saying so is the whole answer.
        return
    if kind == supervisors.SCHTASKS:
        _uninstall_task()
        return
    path = service_path()
    if not path.exists():
        return
    action("stop")
    if kind == supervisors.SYSTEMCTL:
        _run(["systemctl", "--user", "disable", path.name])
    path.unlink()


def _uninstall_task() -> None:
    """Deregister the Windows task and delete our copy of its definition.

    A separate function so ``uninstall`` keeps exactly ONE file-removal call:
    ``tests/unit/session/test_no_session_deletion.py`` inventories every
    ``<path>.unlink`` by function and count, and a second one riding on that
    row would be a new unreviewed remover — which is what the inventory exists
    to catch.
    """
    supervisors.delete_task(TASK_NAME)
    task_record_path().unlink(missing_ok=True)
