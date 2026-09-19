"""The launchd control commands have to work from whatever state launchd is
actually in — a plist can exist while the agent was never bootstrapped, and
`restart` failing with "Could not find service" is exactly that gap.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from local_operator.mobile import install


def _steps(result: dict[str, object]) -> list[str]:
    """A result's ``steps`` as text — the declared shape, asserted here.

    ``install``/``uninstall``/``service_action`` are declared
    ``dict[str, object]``: the keys are known, the values are not, so an ``in``
    test on one of these fields reads as ``str in object`` to the type checker.
    Narrowing through a helper keeps the assertions readable and keeps them
    HONEST — a field that is not the shape these tests pin is a contract break,
    not something to ``str()`` into passing.
    """
    value = result["steps"]
    assert isinstance(value, list), f"steps is {type(value).__name__}"
    return [str(step) for step in value]


def _error(result: dict[str, object]) -> str:
    """A result's ``error`` as text. See :func:`_steps` for why this exists."""
    value = result["error"]
    assert isinstance(value, str), f"error is {type(value).__name__}"
    return value


class FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakePlistPath:
    """A plist path that reports existing without touching the filesystem.

    ``service_action`` bootstraps only when the plist exists; pointing the
    mock at a real path couples the test to whatever happens to be in
    ~/Library/LaunchAgents, so the fake stands in for the file."""

    def __init__(self, exists: bool = True) -> None:
        self._exists = exists

    def exists(self) -> bool:
        return self._exists

    def __str__(self) -> str:
        return "/tmp/fake.plist"

    def __fspath__(self) -> str:
        return str(self)


@pytest.fixture(autouse=True)
def _driving_launchd(monkeypatch) -> None:  # noqa: ANN001
    """These tests fake ``launchctl``, so they must be driving the launchd arm.

    They used to rely on the module's discovery answering "launchctl" by being
    run on macOS; now that the same code drives systemd and Task Scheduler, the
    supervisor the installer would call is stated rather than inherited from
    the runner's OS.
    """
    from local_operator import supervisors

    monkeypatch.setattr(supervisors, "supervisor", lambda: "launchctl")


def test_restart_bootstraps_when_the_agent_is_missing() -> None:
    calls: list[list[str]] = []

    def run(*cmd: str) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        # _launchctl is called with the arguments only ("print", ...), not
        # the "launchctl" argv[0].
        if list(cmd)[0] == "print":
            # Real launchctl: "Bad request." on stdout, the service message on
            # stderr, exit 113.
            return FakeProc(  # type: ignore[return-value]
                returncode=113, stdout="Bad request.", stderr='Could not find service "x"'
            )
        return FakeProc(returncode=0)  # type: ignore[return-value]

    with (
        patch.object(install, "_launchctl", side_effect=run),
        patch.object(install, "plist_path", return_value=FakePlistPath()),
    ):
        result = install.service_action("restart")

    assert result["ok"] is True
    verbs = [c[0] for c in calls]
    # print (missing) -> bootstrap -> kickstart -k, in that order.
    assert verbs == ["print", "bootstrap", "kickstart"]


def test_restart_does_not_bootstrap_a_live_agent() -> None:
    calls: list[list[str]] = []

    def run(*cmd: str) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        if list(cmd)[0] == "print":
            return FakeProc(returncode=0, stdout="pid = 4242")  # type: ignore[return-value]
        return FakeProc(returncode=0)  # type: ignore[return-value]

    with (
        patch.object(install, "_launchctl", side_effect=run),
        patch.object(install, "plist_path", return_value=FakePlistPath()),
    ):
        result = install.service_action("restart")

    assert result["ok"] is True
    assert not any(c[1:2] == ["bootstrap"] for c in calls)


# ---------------------------------------------------------------------------
# The three arms (A1/A2/A8)
#
# A2 is the crash: `lop mobile start|stop|restart|uninstall` called
# `_launchctl` (and `_domain()`, which needs `os.getuid`) with no platform
# guard, so on Linux it raised `FileNotFoundError: 'launchctl'` and on Windows
# `AttributeError: module 'os' has no attribute 'getuid'` — a stack trace where
# the user asked to remove something. Only `install()` had a guard.
# ---------------------------------------------------------------------------


class _FakeSystemctl:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *args: str, **kwargs: object):  # noqa: ANN204
        self.calls.append(args)
        return subprocess.CompletedProcess(list(args), 0, "", "")


@pytest.fixture
def no_supervisor(monkeypatch) -> None:  # noqa: ANN001
    """Devuan/Alpine/slim container/Windows-without-schtasks: nothing to drive."""
    from local_operator import supervisors

    monkeypatch.setattr(supervisors, "supervisor", lambda: None)


def test_every_verb_refuses_instead_of_crashing_without_a_supervisor(
    no_supervisor: None,
) -> None:
    """THE A2 REGRESSION TEST: uninstall and service_action used to raise."""
    actions = [install.uninstall(), install.service_action("start"), install.service_action("stop")]

    for result in actions:
        assert result["ok"] is False
        assert result["error"] == install.NO_SUPERVISOR_ERROR
    # The refusal is ALSO on the step line, because the CLI's uninstall prints
    # `steps` and never `error` — a refusal that only lived in `error` would be
    # invisible on exactly the platform this exists for.
    assert install.NO_SUPERVISOR_ERROR in _steps(install.uninstall())
    # And install(), which was already guarded, still answers in the same words.
    assert _error(install.install(dry_run=True)) == install.NO_SUPERVISOR_ERROR


def test_the_refusal_names_the_platforms_that_do_work(no_supervisor: None) -> None:
    """A refusal that does not say what WOULD work is not actionable."""
    message = _error(install.install(dry_run=True))

    assert "launchctl" in message and "systemctl --user" in message and "schtasks" in message
    assert "`lop mobile serve`" in message


def test_the_linux_arm_writes_a_unit_and_enables_the_service(
    monkeypatch, tmp_path  # noqa: ANN001
) -> None:
    """A1: Linux had no supervisor for this daemon at all."""
    from local_operator import supervisors

    fake = _FakeSystemctl()
    monkeypatch.setattr(supervisors, "supervisor", lambda: "systemctl")
    monkeypatch.setattr(supervisors, "systemctl_user", fake)
    monkeypatch.setattr(supervisors, "enable_linger", lambda: True)
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda _unit: True)
    monkeypatch.setattr(supervisors, "systemd_version", lambda: 255)
    monkeypatch.setattr(install, "load_password", lambda: "hunter2")
    monkeypatch.setattr(install, "ensure_bundle", lambda build=True: (True, "bundle present"))
    monkeypatch.setattr(install, "_our_daemon_listening", lambda _port: True)
    monkeypatch.setattr(install, "health", lambda port=4098, timeout=3.0: {"ok": True})
    monkeypatch.setattr(install, "gate_closed", lambda port=4098, timeout=3.0: True)

    result = install.install(4099)

    assert result["ok"] is True, result
    unit = install.systemd_path()
    text = unit.read_text(encoding="utf-8")
    assert "ExecStart=" in text and "local_operator.mobile.service" in text
    assert "--port 4099" in text
    # Restart=on-failure, not always: a daemon that exits 2 because it has no
    # password must stay down rather than flap.
    assert "Restart=on-failure" in text
    assert ("enable", "--now", install.SYSTEMD_UNIT) in fake.calls
    assert ("daemon-reload",) in fake.calls


def test_the_linux_arm_does_not_address_a_redirected_home(
    monkeypatch, tmp_path  # noqa: ANN001
) -> None:
    """The file half runs; the half that loads a unit refuses."""
    from local_operator import supervisors

    fake = _FakeSystemctl()
    monkeypatch.setattr(supervisors, "supervisor", lambda: "systemctl")
    monkeypatch.setattr(supervisors, "systemctl_user", fake)
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda _unit: False)
    monkeypatch.setattr(install, "load_password", lambda: "hunter2")
    monkeypatch.setattr(install, "ensure_bundle", lambda build=True: (True, "bundle present"))

    result = install.install(4099)

    assert result["ok"] is False
    assert "not the unit the real home owns" in _error(result)
    assert install.systemd_path().exists()
    assert fake.calls == [], "the live user manager was addressed from a redirected home"


def test_a_failing_systemctl_is_reported_with_its_own_words(
    monkeypatch, tmp_path  # noqa: ANN001
) -> None:
    from local_operator import supervisors

    def failing(*args: str, **kwargs: object):  # noqa: ANN202
        if args[0] == "enable":
            return subprocess.CompletedProcess(list(args), 1, "", "Failed to connect to bus")
        return subprocess.CompletedProcess(list(args), 0, "", "")

    monkeypatch.setattr(supervisors, "supervisor", lambda: "systemctl")
    monkeypatch.setattr(supervisors, "systemctl_user", failing)
    monkeypatch.setattr(supervisors, "enable_linger", lambda: True)
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda _unit: True)
    monkeypatch.setattr(install, "load_password", lambda: "hunter2")
    monkeypatch.setattr(install, "ensure_bundle", lambda build=True: (True, "bundle present"))

    result = install.install(4099)

    assert result["ok"] is False
    assert "loginctl enable-linger" in _error(result), "the remedy must be named"


def test_service_action_uses_each_platforms_own_verbs(monkeypatch) -> None:  # noqa: ANN001
    from local_operator import supervisors

    fake = _FakeSystemctl()
    monkeypatch.setattr(supervisors, "supervisor", lambda: "systemctl")
    monkeypatch.setattr(supervisors, "systemctl_user", fake)
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda _unit: True)

    assert install.service_action("restart")["ok"] is True
    assert fake.calls[-1] == ("restart", install.SYSTEMD_UNIT)

    runs: list[tuple[str, ...]] = []
    monkeypatch.setattr(supervisors, "supervisor", lambda: "schtasks")
    monkeypatch.setattr(supervisors, "task_state", lambda _name: (True, True, "Running"))
    monkeypatch.setattr(
        supervisors,
        "schtasks",
        lambda *args, **kw: runs.append(args) or subprocess.CompletedProcess(list(args), 0, "", ""),
    )

    install.service_action("restart")

    assert runs[0] == ("/End", "/TN", install.TASK_NAME)
    assert runs[1] == ("/Run", "/TN", install.TASK_NAME)


def test_the_windows_arm_registers_a_task_and_reports_refusals(
    monkeypatch, tmp_path  # noqa: ANN001
) -> None:
    """A8: no task scheduling was attempted anywhere in this repo before."""
    from local_operator import supervisors

    created: list[tuple[str, str]] = []
    runs: list[tuple[str, ...]] = []
    monkeypatch.setattr(supervisors, "supervisor", lambda: "schtasks")
    monkeypatch.setattr(install, "load_password", lambda: "hunter2")
    monkeypatch.setattr(install, "ensure_bundle", lambda build=True: (True, "bundle present"))
    monkeypatch.setattr(install, "_our_daemon_listening", lambda _port: True)
    monkeypatch.setattr(install, "health", lambda port=4098, timeout=3.0: {"ok": True})
    monkeypatch.setattr(install, "gate_closed", lambda port=4098, timeout=3.0: True)

    def fake_create(name: str, xml: str) -> tuple[bool, str]:
        created.append((name, xml))
        return True, "registered"

    monkeypatch.setattr(supervisors, "create_task", fake_create)
    monkeypatch.setattr(
        supervisors,
        "schtasks",
        lambda *args, **kw: runs.append(args) or subprocess.CompletedProcess(list(args), 0, "", ""),
    )

    result = install.install(4099)

    assert result["ok"] is True, result
    name, xml = created[0]
    assert name == install.TASK_NAME
    assert "local_operator.mobile.service" in xml
    assert "--port" in xml and "4099" in xml
    assert ("/Run", "/TN", install.TASK_NAME) in runs

    # A refusal is schtasks' own words, never a claimed success.
    monkeypatch.setattr(
        supervisors, "create_task", lambda _n, _x: (False, "ERROR: Access is denied.")
    )
    refused = install.install(4099)

    assert refused["ok"] is False
    assert "Access is denied." in _error(refused)


def test_the_linux_unit_quotes_the_interpreter_path(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """A5: an interpreter under a path with a space made the unit unstartable.

    Measured on systemd 255 with the text this renderer emitted unquoted:
    ``Command /home/a is not executable: No such file or directory``.
    """
    import sys as _sys

    from local_operator import procname

    monkeypatch.setattr(procname, "supervised_image", lambda: None)
    monkeypatch.setattr(_sys, "executable", "/home/a b/python3")

    text = install.render_systemd(4098)

    assert 'ExecStart="/home/a b/python3" -m local_operator.mobile.service --port 4098' in text


def test_every_supervisor_arm_carries_the_store(monkeypatch, tmp_path) -> None:  # noqa: ANN001
    """The daemon must read the store the INSTALLER wrote the password into.

    Found by the Windows battery (`mobile.install`, PASS=20/FAIL=1 at the
    round-1 head): the install stored the portal password under the redirected
    store, the task it registered started a daemon that looked in the DEFAULT
    one, and the daemon exited 2 with "no mobile password set. Run `lop mobile
    install`" — naming the store the installer had just written to. A supervised
    process inherits nothing from the installer, so every arm has to say which
    store it serves; `wakes` and `tunnels` have always passed it and `mobile`
    was the one daemon that did not. It was latent on macOS and Linux too, and
    nothing in the mocked-platform suite could see it.
    """
    from local_operator import paths
    from local_operator.paths import CONFIG_DIR_ENV

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path / "store"))
    store = str(paths.config_dir())
    assert store == str(tmp_path / "store"), "the override did not take"

    plist = install.render_plist(4098)
    assert plist["EnvironmentVariables"] == {CONFIG_DIR_ENV: store}

    unit = install.render_systemd(4098)
    environment_lines = [line for line in unit.splitlines() if line.startswith("Environment=")]
    assert environment_lines, unit
    assert f"{CONFIG_DIR_ENV}=" in environment_lines[0]
    assert store in environment_lines[0], environment_lines[0]

    task = install.render_task_xml(4098)
    assert store in task, "Task Scheduler has no env element; the launcher must carry it"


def test_the_node_refusal_names_node_and_where_to_get_it(monkeypatch) -> None:  # noqa: ANN001
    """Q6: the container reading said what was missing, not what to do.

    The old detail ("node is not installed; the bundle needs a one-time `pnpm
    build`") named a command that cannot be run without the thing that is
    missing, so the Ubuntu and Mint readings could not tell an operator how to
    get past it.
    """
    monkeypatch.setattr(install.shutil, "which", lambda _name: None)

    detail = install._build_bundle()

    assert detail is not None
    assert "node is not installed" in detail
    assert "nodejs.org" in detail or "install node" in detail.lower()


def test_a_missing_bundle_refuses_before_the_store_is_touched(
    monkeypatch, tmp_path  # noqa: ANN001
) -> None:
    """Q6: the bundle check used to run AFTER the password was written.

    So an install that could not succeed on a source checkout without Node had
    already created ``mobile/password`` — the mutated-store shape the container
    readings showed, where the FAIL detail was the password progress line.
    Asserted on the write, not on the message.
    """
    from local_operator import supervisors

    written: list[str] = []
    monkeypatch.setattr(supervisors, "supervisor", lambda: "systemctl")
    monkeypatch.setattr(install, "load_password", lambda: None)
    monkeypatch.setattr(install, "store_password", lambda value: written.append(value))
    monkeypatch.setattr(
        install, "ensure_bundle", lambda build=True: (False, "node is not installed")
    )

    result = install.install(port=4098)

    assert result["ok"] is False, result
    assert "node is not installed" in _error(result)
    assert written == [], "the store was mutated by an install that then refused"


def test_the_windows_uninstall_ends_the_task_before_deleting_it(
    monkeypatch, tmp_path  # noqa: ANN001
) -> None:
    """A2: ``/Delete`` deregisters without interrupting the program it runs.

    Microsoft's own wording for the verb is "This command doesn't delete the
    program that the task runs or interrupt a running program", so an uninstall
    that only deleted left the portal daemon serving on its port with the
    password still loaded — while the other two platforms stop the job as part
    of the uninstall (``bootout``, ``disable --now``). No test covered this path
    at all before this one.
    """
    from local_operator import supervisors

    calls: list[tuple[str, ...]] = []
    record = tmp_path / "mobile-task.xml"
    record.write_text("<Task/>", encoding="utf-8")
    monkeypatch.setattr(supervisors, "supervisor", lambda: "schtasks")
    monkeypatch.setattr(install, "task_record_path", lambda: record)
    monkeypatch.setattr(
        supervisors,
        "schtasks",
        lambda *args, **kw: calls.append(args)
        or subprocess.CompletedProcess(list(args), 0, "", ""),
    )
    monkeypatch.setattr(
        supervisors,
        "delete_task",
        lambda name: calls.append(("/Delete", "/TN", name)) or (True, "deleted"),
    )

    result = install.uninstall()

    assert result["ok"] is True, result
    assert [call[0] for call in calls] == ["/End", "/Delete"], calls
    assert calls[0] == tuple(supervisors.task_end_args(install.TASK_NAME))
    assert not record.exists()


def test_a_refused_windows_uninstall_is_reported_as_a_failure(
    monkeypatch, tmp_path  # noqa: ANN001
) -> None:
    """The refusal only reached a ``steps`` line, and ``ok`` still said True."""
    from local_operator import supervisors

    record = tmp_path / "mobile-task.xml"
    record.write_text("<Task/>", encoding="utf-8")
    monkeypatch.setattr(supervisors, "supervisor", lambda: "schtasks")
    monkeypatch.setattr(install, "task_record_path", lambda: record)
    monkeypatch.setattr(
        supervisors,
        "schtasks",
        lambda *args, **kw: subprocess.CompletedProcess(list(args), 0, "", ""),
    )
    monkeypatch.setattr(
        supervisors, "delete_task", lambda _name: (False, "ERROR: Access is denied.")
    )

    result = install.uninstall()

    assert result["ok"] is False, result
    assert "Access is denied." in _error(result)
    assert record.exists(), "the record was deleted for a task that is still registered"


def test_the_plist_repair_reports_unsupported_off_macos(monkeypatch) -> None:  # noqa: ANN001
    """The repair's gate is the supervisor's IDENTITY, not ``is_supported()``.

    ``is_supported()`` now answers True on Linux and Windows too, so a repair
    gated on it would go looking for a LaunchAgent that cannot exist on those
    platforms and report "not addressable" — a diagnosis about a sandbox rather
    than about a platform.
    """
    from local_operator import supervisors

    monkeypatch.setattr(supervisors, "supervisor", lambda: "systemctl")

    assert install.refresh_plist_if_stale().kind == "unsupported"


# ---------------------------------------------------------------------------
# The bundle build: pnpm on a platform whose shims are batch files (C10).
# ---------------------------------------------------------------------------


def test_the_pnpm_launcher_is_the_bare_name_on_posix(monkeypatch) -> None:  # noqa: ANN001
    """POSIX is unchanged: ``which`` found it, and ``execve`` runs it."""
    monkeypatch.setattr(install.shutil, "which", lambda name: f"/usr/local/bin/{name}")
    assert install._shim_argv("pnpm") == ["pnpm"]
    assert install._shim_argv("corepack") == ["corepack"]


def test_an_absent_tool_resolves_to_none(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(install.shutil, "which", lambda name: None)
    assert install._shim_argv("pnpm") is None


def test_the_windows_launcher_goes_through_the_command_interpreter(
    monkeypatch,
) -> None:  # noqa: ANN001
    """``CreateProcess`` cannot run the ``.CMD`` shim npm installs.

    It appends only ``.exe`` to a name with no extension, so the bare ``"pnpm"``
    this used to pass never reaches the shim at all — the build "failed" with no
    stated reason. The documented route is ``cmd /c`` plus the batch file, and
    ``call`` is there because cmd strips the quotes of a command line it cannot
    disambiguate, which is every ``C:\\Program Files\\nodejs\\pnpm.CMD``.
    """
    shim = r"C:\Program Files\nodejs\pnpm.CMD"

    monkeypatch.delenv("COMSPEC", raising=False)
    assert install._windows_shim_argv(shim) == ["cmd.exe", "/c", "call", shim]

    # ``COMSPEC`` is how a console says where its own interpreter is, so it wins
    # over the fallback rather than being decoration.
    monkeypatch.setenv("COMSPEC", r"D:\Windows\System32\cmd.exe")
    assert install._windows_shim_argv(shim) == [
        r"D:\Windows\System32\cmd.exe",
        "/c",
        "call",
        shim,
    ]
