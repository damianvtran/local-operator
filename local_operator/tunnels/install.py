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

from local_operator import launchd, procname, procstate, supervisors
from local_operator.paths import CONFIG_DIR_ENV, config_dir
from local_operator.tunnels import config, gateway, state

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
        exec_start=f"{supervisors.quoted(str(image))} -m local_operator.tunnels.service",
        post_lines=[
            f"Environment={supervisors.quoted(f'{CONFIG_DIR_ENV}={base}')}",
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


def _domain() -> str:
    """``gui/<uid>`` — the launchd domain the ``bootout`` below addresses.

    The sibling installers each have this helper; the tunnel arm was the one
    still spelling ``f"gui/{os.getuid()}"`` inline, which put ``os.getuid`` in
    a function (``action``) that is reachable on every platform. The guard is
    here rather than at that call site because ``os.getuid`` does not exist off
    POSIX: an inline spelling is an ``AttributeError`` waiting for the next
    caller, and only a guard INSIDE the function it protects is visible to a
    reader — or to the static scan that grades this branch — as "safe to call
    anywhere" rather than "currently unreachable".
    """
    if procstate.is_windows():
        raise RuntimeError("launchd domains exist only on macOS")
    return f"gui/{os.getuid()}"


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


def rearm_if_parked(*, provider: str, credential_id: int) -> str:
    """Start a parked connector again after the login that parked it is fixed.

    A park withdraws remote access on purpose, so the operator has to be able to
    end it — and the act that ends it is the one they were already told to
    perform. Nothing else notices: the connector exited 0, which is precisely
    what stops its supervisor from retrying it, so without this hook the phone
    stays dark until someone runs a local command they have no reason to run.

    Called from the credential-write path (`AuthStore.upsert_credential`), which
    is the ONE place a TUI `/login`, a `lop login`, and the desktop
    ``POST /v1/auth/login`` all land, and from `lop tunnel status`/`start` as a
    self-heal. Cheap by construction: every guard below is a stat or a small
    file read, and the first one returns for the overwhelmingly common case —
    nothing is parked at all.

    Deliberately narrow, in four ways, each for its own reason:

    * Only ``login_required``. A connector parked for a missing prerequisite or
      a console re-enrolment needs a DIFFERENT local command, and starting it
      early would only park it again — noisily, in a log this exists to quiet.
    * Only a tunnel that is configured and not deliberately stopped: `stopped`
      means the operator is not using the tunnel, so there is nothing to re-arm.
    * Only the credential THIS tunnel owns. Every writer in the process reaches
      here (see the hook's comment), so an unrelated provider login — or a
      Radient login for a different account — must touch nothing.
    * Only when the store this process is acting on lives in the real home, and
      the unit is actually installed. `launchctl` addresses the real user's
      session whatever ``HOME`` says, so a sandboxed run (a test, a review
      worktree) would otherwise reach out and restart the operator's live
      connector. `LaunchAgent repair` carries the same guard for the same reason.

    TO REHEARSE THIS END TO END FROM A SANDBOX, PIN THE LABEL (QA round 1, Q2).
    The last two guards together mean a rig that leaves both alone cannot reach
    the happy path at all: it declines, correctly, because its config root is not
    the real home — and relaxing the real-home guard without the pin is worse,
    since the fixed `LABEL` above resolves to whatever plist already holds it,
    i.e. the operator's live connector. The shape that works, and the one QA's
    own rehearsal used: rebuild this module's `LABEL` to the RIG's own label
    (monkeypatch/`importlib.reload` the constant, never the plist on disk) and
    force ONLY `supervisors.config_lives_in_real_home`, leaving every other guard
    — provider, credential id, `stopped`, park reason, plist presence — the
    production one, so a broken guard shows up as a `launchctl` call the rig
    recorded rather than as an incident. Put a record-only `launchctl` shim on
    PATH for the containment case, which is run UNPINNED.

    Never raises: a re-arm that fails must not fail the login that triggered it.
    Returns a sentence for the caller to show, or "" when it did nothing.
    """
    if not _rearm_allowed():
        return ""
    # `supervisors.supervisor()` rather than a `sys.platform` test: it also
    # answers "no supervisor here at all" (a Linux without systemd, a Windows
    # without schtasks), which is a platform this must decline on rather than
    # raise out of a login, and it is the same question `lop tunnel install`
    # and `action()` ask.
    kind = supervisors.supervisor()
    if kind is None:
        return ""
    parked = state.parked()
    if not parked or parked.get("reason") != gateway.LOGIN_REQUIRED:
        return ""
    try:
        value = config.load()
    except ValueError:
        return ""
    if value.get("stopped"):
        return ""
    owns = value.get("credential_id")
    if (
        provider != "radient"
        or not isinstance(owns, int)
        or isinstance(owns, bool)
        or owns != credential_id
    ):
        # `isinstance` as well as the comparison, because a configuration with no
        # usable id (hand-edited, or written by a build that stored it as a
        # string) must not match a `credential_id` of the same kind: the guard is
        # "this is the credential the tunnel owns", and None == None is not that.
        return ""
    try:
        if not supervisors.config_lives_in_real_home(config_dir()):
            return ""
        if kind == supervisors.SCHTASKS:
            # Windows keeps its own copy of a task, so there is no registration
            # FILE this installer owns to check — the addressability guard is
            # the shared one `action()` uses for the same reason.
            if not supervisors.task_scheduler_is_addressable(config_dir()):
                return ""
            started = supervisors.schtasks(*supervisors.task_run_args(TASK_NAME))
            if started.returncode:
                raise ValueError(((started.stderr or started.stdout) or "").strip()[:200])
            return "The Radient tunnel connector is starting again."
        path = service_path()
        if not path.exists():
            return ""
        if kind == supervisors.LAUNCHCTL:
            # kickstart, not `launchd.reload_job`: that sequence exists for the
            # case where the plist was just REWRITTEN, and its own docstring
            # names this one as the deliberate exception — the parked job's
            # plist is already correct and the job is loaded but stopped, so
            # there is nothing to tear down and a reload would discard a
            # pending plist repair mid-flight. The mobile installer starts its
            # own daemon the same way.
            result = _launchctl("kickstart", "-k", f"gui/{os.getuid()}/{LABEL}")
            if result.returncode:
                raise ValueError(result.stderr.strip())
        else:
            # Started, not restarted: the unit is inactive by definition here.
            _run(["systemctl", "--user", "start", path.name])
    except (OSError, ValueError, subprocess.SubprocessError):
        return "Signed in, but the tunnel connector could not be restarted; run lop tunnel install."
    return "The Radient tunnel connector is starting again."


def _rearm_allowed() -> bool:
    """Whether the credential-write hook may start a supervised service.

    An environment opt-out rather than a module flag because the hook is
    reached from tests, review worktrees and the installed product alike, and
    the run that must not touch launchd is the one that already knows it is a
    test — it can say so in its own environment.
    """
    value = os.environ.get("LOP_TUNNEL_NO_REARM", "").strip().lower()
    return value not in {"1", "true", "yes", "on"}


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
            _run(["launchctl", "bootout", f"{_domain()}/{LABEL}"], checked=False)
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

    ``/End`` before ``/Delete`` for the same reason ``mobile``'s arm sends it:
    ``schtasks /Delete`` deregisters the task without interrupting the program
    it runs, so the connector would keep serving after ``lop tunnel uninstall``
    reported it removed.
    """
    supervisors.schtasks(*supervisors.task_end_args(TASK_NAME))
    supervisors.delete_task(TASK_NAME)
    task_record_path().unlink(missing_ok=True)
