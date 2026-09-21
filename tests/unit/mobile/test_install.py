"""The launchd control commands have to work from whatever state launchd is
actually in — a plist can exist while the agent was never bootstrapped, and
`restart` failing with "Could not find service" is exactly that gap.
"""

from __future__ import annotations

import plistlib
import subprocess
from pathlib import Path
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
    ~/Library/LaunchAgents, so the fake stands in for the file — and its path is
    SHAPED like a home's, because the verbs now pass the pair they address
    through ``launchd.is_own_plist`` (round 2, R-8). A placeholder path would be
    refused before any call, which would make bootstrap-on-demand untestable
    rather than guarded; ``owned_plist_home`` below names the home it stands
    for."""

    #: The home this fake plist belongs to; the identity test is asked about it.
    HOME = Path("/tmp/lop-fake-home")

    def __init__(self, exists: bool = True) -> None:
        self._exists = exists

    def exists(self) -> bool:
        return self._exists

    def __str__(self) -> str:
        return str(self.HOME / "Library" / "LaunchAgents" / f"{install.LABEL}.plist")

    def __fspath__(self) -> str:
        return str(self)


@pytest.fixture
def owned_plist_home(monkeypatch) -> Path:  # noqa: ANN001
    """Pin the launchd-level home to the fake plist's, so the guard passes.

    The same seam ``_InstallRig(own=True)`` uses, one level down: without it the
    identity test refuses before any call and the cells below would assert the
    refusal instead of the bootstrap they exist for.
    """
    monkeypatch.setattr(install.launchd, "real_home", lambda: FakePlistPath.HOME)
    return FakePlistPath.HOME


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


def test_restart_bootstraps_when_the_agent_is_missing(owned_plist_home: Path) -> None:
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


def test_restart_does_not_bootstrap_a_live_agent(owned_plist_home: Path) -> None:
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


def test_build_failure_reports_the_compiler_error_not_the_script_echo() -> None:
    """The failure message has to carry what an operator can act on.

    pnpm writes its `$ tsc -b && vite build` script echo to stderr and `tsc`
    writes the reason to ITS stdout, so the old "last line of stderr" reported
    the echo — measured on a source-snapshot install, where the phone showed
    503 and the only clue was a line naming the script that had just run.
    """
    result: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        ["pnpm", "build"],
        2,
        stdout=(
            "src/model-sheet.order.test.tsx(31,20): error TS2307: Cannot find module "
            "'./fixtures/models.ranked.json' or its corresponding type declarations.\n"
        ),
        stderr="$ tsc -b && vite build\n[ELIFECYCLE] Command failed with exit code 2.\n",
    )

    detail = install._failure_detail(result)

    assert "TS2307" in detail
    assert "./fixtures/models.ranked.json" in detail
    # The echo and the lifecycle summary are what made the old message
    # useless; neither survives when a real error was printed.
    assert "$ tsc -b" not in detail
    assert "Command failed with exit code" not in detail


def test_failure_detail_is_bounded_and_falls_back_to_the_tail() -> None:
    """A tool that fails without printing an error still gets quoted, and no
    amount of output makes the message unbounded."""
    noisy: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        ["pnpm", "install"], 1, stdout="\n".join(f"line {n}" for n in range(200)), stderr=""
    )

    detail = install._failure_detail(noisy)

    assert detail == "line 197 | line 198 | line 199"

    many_errors: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        ["pnpm", "build"],
        2,
        stdout="\n".join(f"error TS{n}: detail {n}" for n in range(20)),
        stderr="",
    )
    bounded = install._failure_detail(many_errors)
    assert bounded.count(" | ") == 2
    assert len(bounded) <= install._BUILD_DETAIL_CHARS


def test_build_bundle_returns_the_compiler_error_on_one_line() -> None:
    """The wiring, not just the helper: what `_build_bundle` hands `install`.

    The step runner and the pin guard are patched here rather than exercised: this
    test is about the FAILURE's shape (one line, the compiler's reason), which is
    the half a build error reaches an operator through. The runner and the guard
    have their own tests in `test_bundle_build_bound.py`.
    """

    def step(argv: list[str], _cwd: Path, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        if "install" in argv:
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(
            argv,
            2,
            "src/fixture.test.ts(1,1): error TS2307: Cannot find module './fixtures/x.json'.\n",
            "$ tsc -b && vite build\n",
        )

    with (
        patch.object(install.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"),
        patch.object(install, "_run_build_step", side_effect=step),
        patch.object(install, "_pin_mismatch", return_value=None),
    ):
        error = install._build_bundle()

    assert error is not None
    assert "pnpm build failed: " in error
    assert "TS2307" in error
    assert "\n" not in error


def _snapshot_web(tmp_path: Path, *, dist: bool = False) -> Path:
    """A snapshot tree's web dir: package.json always, dist/ only if asked."""
    web = tmp_path / "local_operator" / "mobile" / "web"
    web.mkdir(parents=True)
    (web / "package.json").write_text("{}", encoding="utf-8")
    if dist:
        (web / "dist").mkdir()
        (web / "dist" / "index.html").write_text("<html></html>", encoding="utf-8")
    return web


def test_snapshot_bundle_builds_sources_that_have_no_dist(tmp_path: Path) -> None:
    """The gap that left a phone on 503: a snapshot carries the web SOURCES,
    dist/ is gitignored, and nothing built it — so the installed generation
    had no UI and every authed GET answered "bundle not built"."""
    web = _snapshot_web(tmp_path)

    with (
        patch.object(install.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"),
        patch.object(install, "_build_bundle", return_value=None) as build,
    ):
        status = install.snapshot_bundle(web)

    assert status == "built"
    assert build.call_args.args[0] == web


def test_snapshot_bundle_leaves_an_already_built_snapshot_alone(tmp_path: Path) -> None:
    web = _snapshot_web(tmp_path, dist=True)

    with patch.object(install, "_build_bundle") as build:
        status = install.snapshot_bundle(web)

    assert status == "already built"
    build.assert_not_called()


def test_snapshot_bundle_skips_without_node_and_does_not_fail(tmp_path: Path) -> None:
    """No Node is a documented skip, not an update failure: the daemon heals
    itself at `lop mobile install` on any host that has it."""
    web = _snapshot_web(tmp_path)

    with patch.object(install.shutil, "which", return_value=None):
        status = install.snapshot_bundle(web)

    assert status == "skipped (node not installed; build at `lop mobile install`)"


def test_snapshot_bundle_reports_a_failed_build_without_raising(tmp_path: Path) -> None:
    web = _snapshot_web(tmp_path)

    with (
        patch.object(install.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}"),
        patch.object(install, "_build_bundle", return_value="pnpm build failed: boom"),
    ):
        status = install.snapshot_bundle(web)

    assert status == "FAILED (pnpm build failed: boom)"


def test_snapshot_bundle_skips_a_tree_with_no_web_sources(tmp_path: Path) -> None:
    assert install.snapshot_bundle(tmp_path) == "skipped (no web sources in snapshot)"


def _guard_script(web: Path) -> None:
    (web / "scripts").mkdir(parents=True, exist_ok=True)
    (web / "scripts" / "check-bundle.mjs").write_text("", encoding="utf-8")


def test_verify_bundle_reports_the_guard_message(tmp_path: Path) -> None:
    web = tmp_path / "web"
    _guard_script(web)
    failed = FakeProc(
        returncode=1,
        stderr="error: the stylesheet names 19 classes; this bundle has 235.\n",
    )

    with (
        patch.object(install.shutil, "which", return_value="/usr/bin/node"),
        patch.object(install.subprocess, "run", return_value=failed),
    ):
        detail = install._verify_bundle(web)

    assert detail is not None
    assert detail.startswith("bundle check failed: ")
    assert "19 classes" in detail


def test_verify_bundle_is_optional_without_node_or_the_script(tmp_path: Path) -> None:
    """Nothing to run is not an error: refusing to install would be worse than
    the blind spot, and an older tree simply has no guard script."""
    assert install._verify_bundle(tmp_path) is None

    web = tmp_path / "web"
    _guard_script(web)
    with patch.object(install.shutil, "which", return_value=None):
        assert install._verify_bundle(web) is None


def test_build_bundle_reports_what_its_guard_rejects(tmp_path: Path) -> None:
    """Exit 0 is not proof the bundle is servable, and this is the wiring that
    makes a degenerate one loud instead of serving an unstyled phone.

    The step runner is patched rather than ``subprocess.run``: the steps of a
    build are run by `_run_build_step` (its own process group, see
    `test_bundle_build_bound.py`), so patching the old call would leave this test
    running the REAL pnpm against a temp tree.
    """
    web = tmp_path / "web"
    (web / "dist").mkdir(parents=True)
    (web / "dist" / "index.html").write_text("<html></html>", encoding="utf-8")

    with (
        patch.object(install, "_run_build_step", return_value=FakeProc(returncode=0)),
        patch.object(install, "_verify_bundle", return_value="bundle check failed: nope"),
    ):
        error = install._build_bundle(web, ["pnpm"])

    assert error == "bundle check failed: nope"
    # And the refused bundle is not left where `_bundle_state` would read it as
    # "built" and keep serving it.
    assert not (web / "dist").exists()


def test_ensure_bundle_rebuilds_a_present_bundle_its_guard_rejects(tmp_path: Path) -> None:
    """The state a FAILED build leaves: vite writes dist/, the guard fails,
    the installer removed nothing, and `_bundle_state` reads the leftover
    index.html as "built" — so every later install says "bundle present" while
    the phone renders unstyled."""
    web = _snapshot_web(tmp_path, dist=True)

    with (
        patch.object(install, "_verify_bundle", return_value="bundle check failed: 62 classes"),
        patch.object(install, "_build_bundle", return_value=None) as build,
    ):
        ok, detail = install.ensure_bundle(web_dir=web)

    assert (ok, detail) == (True, "built the web bundle")
    build.assert_called_once()
    # The rejected bundle is gone, so a later `_bundle_state` cannot read it as
    # "built" while the rebuild is what actually produced the bundle.
    assert not (web / "dist").exists()


def test_ensure_bundle_trusts_a_bundle_its_guard_accepts(tmp_path: Path) -> None:
    web = _snapshot_web(tmp_path, dist=True)

    with (
        patch.object(install, "_verify_bundle", return_value=None),
        patch.object(install, "_build_bundle") as build,
    ):
        ok, detail = install.ensure_bundle(web_dir=web)

    assert (ok, detail) == (True, "bundle present")
    build.assert_not_called()


def test_ensure_bundle_reports_a_rejected_bundle_when_it_may_not_build(tmp_path: Path) -> None:
    web = _snapshot_web(tmp_path, dist=True)

    with (
        patch.object(install, "_verify_bundle", return_value="bundle check failed: 62 classes"),
        patch.object(install, "_build_bundle") as build,
    ):
        ok, detail = install.ensure_bundle(build=False, web_dir=web)

    assert ok is False
    assert detail == "bundle check failed: 62 classes"
    build.assert_not_called()


def test_ensure_bundle_uses_the_installs_own_tree_by_default(tmp_path: Path) -> None:
    """The default path, which is the one `lop mobile install` takes: `web_dir`
    is None there, and a helper that cannot take None crashes the CLI instead
    of rebuilding (measured on the real command, not in a test)."""
    web = _snapshot_web(tmp_path, dist=True)

    with (
        patch.object(install, "_WEB_DIR", web),
        patch.object(install, "_verify_bundle", return_value="bundle check failed: 62 classes"),
        patch.object(install, "_build_bundle", return_value=None) as build,
    ):
        ok, _ = install.ensure_bundle()

    assert ok is True
    build.assert_called_once_with(web)


def test_snapshot_bundle_rebuilds_a_bundle_its_guard_rejects(tmp_path: Path) -> None:
    """A snapshot that ARRIVES with a `dist/` is not thereby a snapshot with a
    usable one: a tree packed with a pre-fix (utility-less) bundle would install
    unstyled and report success — the silent half of the defect this path
    exists to close. Same standard as `ensure_bundle`: verify, then drop and
    rebuild."""
    web = _snapshot_web(tmp_path, dist=True)

    with (
        patch.object(install, "_verify_bundle", return_value="bundle check failed: 62 classes"),
        patch.object(install, "_build_bundle", return_value=None) as build,
    ):
        status = install.snapshot_bundle(web)

    assert status == "built"
    build.assert_called_once()
    # The refused bundle is gone before the rebuild, so nothing can read it as
    # "built" in the meantime.
    assert not (web / "dist").exists()


def test_snapshot_bundle_trusts_a_present_bundle_its_guard_accepts(tmp_path: Path) -> None:
    web = _snapshot_web(tmp_path, dist=True)

    with (
        patch.object(install, "_verify_bundle", return_value=None),
        patch.object(install, "_build_bundle") as build,
    ):
        status = install.snapshot_bundle(web)

    assert status == "already built"
    build.assert_not_called()


# --------------------------------------------------------------------------
# An install that would change nothing must do nothing.
#
# `install` rewrote the plist and bootout+bootstrapped unconditionally, and
# with the generation shim the plist path is stable — so the common re-run
# wrote IDENTICAL bytes and churned launchd behind them, which is precisely the
# pair of signals an EDR reads as "Persistence: launchd job / plist file
# modification" (MITRE T1543.001). Both directions are pinned below, with the
# launchctl call log asserted directly rather than inferred from the steps.
# --------------------------------------------------------------------------


class _InstallRig:
    """Everything ``install()`` reaches outside itself, faked and recorded.

    The password store is faked rather than read: this test has no business
    touching the operator's live portal password, and reading it would make the
    result depend on the host.
    """

    def __init__(self, monkeypatch, tmp_path, *, own: bool = True) -> None:  # noqa: ANN001
        # WHERE A HOME WOULD OWN IT: `<home>/Library/LaunchAgents/<label>.plist`,
        # so `own` can be spelled by pointing `real_home` at this tmpdir rather
        # than by stubbing the guard. A stubbed guard restored the ORIGINAL over
        # a runtime mutation of `is_own_plist` and hid it, which is why the
        # installer-level sandbox cell could not be mutation-verified (review
        # round 2, R-10). `own=False` pins nothing, so the real operator home is
        # compared against and the refusal is the genuine one.
        home = tmp_path
        home.joinpath("Library", "LaunchAgents").mkdir(parents=True, exist_ok=True)
        plist = home / "Library" / "LaunchAgents" / f"{install.LABEL}.plist"
        self.plist = plist
        self.calls: list[list[str]] = []
        self.serving = True
        monkeypatch.setattr(install, "plist_path", lambda: plist)
        monkeypatch.setattr(install, "load_password", lambda: "a-password")
        monkeypatch.setattr(install, "ensure_bundle", lambda build=True: (True, "bundle present"))
        monkeypatch.setattr(install, "store_description", lambda: "test store")
        monkeypatch.setattr(install, "_launchctl", self._launchctl)
        monkeypatch.setattr(install, "_our_daemon_listening", lambda _port: self.serving)
        monkeypatch.setattr(install, "health", lambda port=4098, timeout=3.0: {"ok": True})
        monkeypatch.setattr(install, "gate_closed", lambda port=4098, timeout=3.0: True)
        if own:
            monkeypatch.setattr(install.launchd, "real_home", lambda: home)

    def _launchctl(self, *cmd: str) -> FakeProc:
        self.calls.append(list(cmd))
        if cmd[:2] == ("kickstart", "-k"):
            self.serving = True  # the repair worked
        return FakeProc(returncode=0, stdout="pid = 4242")

    def write_current_plist(self, port: int = install.DEFAULT_PORT) -> None:
        self.plist.parent.mkdir(parents=True, exist_ok=True)
        self.plist.write_bytes(plistlib.dumps(install.render_plist(port)))


def test_a_sandboxed_repair_never_reaches_the_real_job(
    monkeypatch, tmp_path  # noqa: ANN001
) -> None:
    """R-1, the reviewer's reproduction: a redirected HOME must reach nothing.

    The label is a fixed module constant while the plist path moves with
    ``$HOME``, so a loaded-but-stopped job found at a SANDBOX path used to send
    `kickstart -k gui/<uid>/com.local-operator.mobile` at the operator's live
    daemon. The identity guard now refuses both helpers, and the reload that
    follows refuses in its own words — so the decline is REPORTED, not silent,
    and no launchctl call is issued at all.
    """
    rig = _InstallRig(monkeypatch, tmp_path, own=False)
    rig.write_current_plist()
    rig.serving = False

    result = install.install()

    assert rig.calls == [], f"a sandboxed install reached launchctl: {rig.calls}"
    assert result["ok"] is False, result
    assert "not the LaunchAgent the real home owns" in _error(result), _error(result)


def test_an_install_that_changes_nothing_does_not_write_or_reload(
    monkeypatch, tmp_path  # noqa: ANN001
) -> None:
    """Equal content ⇒ no write, NO launchctl call, and no false claim.

    The two assertions a mutation cannot survive: the file's mtime must not
    move (a rewrite is the EDR signal) and the call log must be EMPTY (a
    bootout/bootstrap is the other one).
    """
    rig = _InstallRig(monkeypatch, tmp_path)
    rig.write_current_plist()
    before = rig.plist.stat().st_mtime_ns

    result = install.install()

    assert result["ok"] is True
    steps = _steps(result)
    assert rig.plist.stat().st_mtime_ns == before, "the plist was rewritten"
    assert rig.calls == [], f"install reached launchctl for no reason: {rig.calls}"
    assert any("already current" in step for step in steps), steps
    assert not any(
        "loaded the LaunchAgent" in step for step in steps
    ), "an install that skipped the load must not report one"


def test_an_install_that_changes_the_plist_still_writes_and_reloads(
    monkeypatch, tmp_path  # noqa: ANN001
) -> None:
    """Changed content ⇒ today's write + reload, unchanged."""
    from local_operator import launchd

    rig = _InstallRig(monkeypatch, tmp_path)
    rig.write_current_plist(port=install.DEFAULT_PORT + 1)
    reloads: list[object] = []

    def fake_reload(**kwargs) -> "launchd.JobReload":  # noqa: ANN003
        reloads.append(kwargs["path"])
        return launchd.JobReload(label=str(kwargs["label"]), outcome="reloaded")

    monkeypatch.setattr(install.launchd, "reload_job", fake_reload)

    result = install.install()

    assert result["ok"] is True
    assert plistlib.loads(rig.plist.read_bytes()) == install.render_plist(install.DEFAULT_PORT)
    assert reloads == [rig.plist], "a changed plist must still be reloaded"
    assert not any("already current" in step for step in _steps(result))


def test_a_loaded_but_dead_job_is_restarted_without_rewriting_the_plist(
    monkeypatch, tmp_path  # noqa: ANN001
) -> None:
    """The state the old unconditional rewrite repaired, still repaired.

    A stopped-but-loaded job is exactly what a compare-then-skip could leave
    dead if it only looked at the file. It is restarted with `kickstart -k` —
    the narrower operation, which does not briefly unregister the label — and
    the file is left untouched.
    """
    from local_operator import launchd

    rig = _InstallRig(monkeypatch, tmp_path)
    rig.write_current_plist()
    rig.serving = False
    before = rig.plist.stat().st_mtime_ns

    result = install.install()

    assert result["ok"] is True
    assert rig.plist.stat().st_mtime_ns == before, "the file was already correct"
    assert rig.calls == [["kickstart", "-k", f"{launchd.job_domain()}/{install.LABEL}"]]
    assert not any(
        "loaded the LaunchAgent" in step for step in _steps(result)
    ), "a kickstart is not a reload and must not be reported as one"


def test_a_redirected_home_asks_launchd_nothing_about_the_operator(
    monkeypatch, tmp_path  # noqa: ANN001
) -> None:
    """Q-1 (QA round 2): the read-only probe was the last unguarded call.

    ``_supervised_pid`` asked ``launchctl print gui/<uid>/<label>`` by label
    alone, so a redirected ``HOME`` read the operator's job — the read-only half
    of the same mistake R-1 fixed on the acting half, on the path
    ``_our_daemon_listening`` takes to decide whether to skip a reload. The
    identity test now sits inside the probe, so this asserts NO call at all: a
    refusal that merely ignored the answer would still be their job inspected.
    """
    calls: list[list[str]] = []

    def fake_launchctl(*cmd: str) -> FakeProc:
        calls.append(list(cmd))
        return FakeProc(returncode=0, stdout="\tpid = 4242\n")

    monkeypatch.setattr(install, "plist_path", lambda: tmp_path / f"{install.LABEL}.plist")
    monkeypatch.setattr(install, "_launchctl", fake_launchctl)
    monkeypatch.setattr(install.supervisors, "supervisor", lambda: install.supervisors.LAUNCHCTL)

    assert install._supervised_pid() is None
    assert calls == [], f"a redirected home asked launchd about the operator: {calls}"


def test_the_probe_still_answers_for_the_job_this_run_owns(
    monkeypatch, tmp_path  # noqa: ANN001
) -> None:
    """The other direction of the same guard, and it is the REAL one.

    ``real_home`` is pinned to this tmpdir and the plist is written where that
    home owns it, so ``is_own_plist`` genuinely answers True — a stubbed guard
    here would prove nothing about the real-home path staying alive.
    """
    home = tmp_path
    home.joinpath("Library", "LaunchAgents").mkdir(parents=True, exist_ok=True)
    plist = home / "Library" / "LaunchAgents" / f"{install.LABEL}.plist"
    calls: list[list[str]] = []

    def fake_launchctl(*cmd: str) -> FakeProc:
        calls.append(list(cmd))
        return FakeProc(returncode=0, stdout="\tpid = 4242\n")

    monkeypatch.setattr(install.launchd, "real_home", lambda: home)
    monkeypatch.setattr(install, "plist_path", lambda: plist)
    monkeypatch.setattr(install, "_launchctl", fake_launchctl)
    monkeypatch.setattr(install.supervisors, "supervisor", lambda: install.supervisors.LAUNCHCTL)

    assert install._supervised_pid() == 4242
    assert calls == [["print", f"{install._domain()}/{install.LABEL}"]], calls


def test_the_verbs_refuse_a_redirected_home_and_act_for_the_owned_one(
    monkeypatch, tmp_path  # noqa: ANN001
) -> None:
    """R-8: ``service_action``'s launchd arm addressed the label from anywhere.

    ``bootstrap <domain> <plist>`` is resolved by launchd to the Label INSIDE the
    file, so a redirected home's `lop mobile restart` EVICTED and replaced the
    operator's daemon rather than merely restarting it — the measurement
    ``browser_bridge._root_suffix`` records — and ``LABEL`` is a fixed constant
    here while ``plist_path()`` moves with ``$HOME``. The mirrored half keeps the
    real-home verb reaching launchd.
    """
    sandbox = _InstallRig(monkeypatch, tmp_path / "sandbox", own=False)
    sandbox.write_current_plist()

    refused = install.service_action("restart")

    assert refused["ok"] is False, refused
    assert sandbox.calls == [], f"a redirected home reached launchd: {sandbox.calls}"
    assert "not the LaunchAgent the real home owns" in _error(refused)

    owned = _InstallRig(monkeypatch, tmp_path / "home")
    owned.write_current_plist()

    assert install.service_action("restart")["ok"] is True
    assert [call[:2] for call in owned.calls] == [
        ["print", f"{install._domain()}/{install.LABEL}"],
        ["kickstart", "-k"],
    ], owned.calls
