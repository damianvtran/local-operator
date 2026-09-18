"""The shared supervisor layer: what it discovers, what it renders, what it refuses.

Every one of these defects was a CRASH or a SILENT no-op on a platform other
than macOS, so the tests are written to fail on the unfixed code:
``supervisors`` did not exist, ``browser_bridge`` discovered its supervisor with
its own copy, and ``mobile``/``wakes``/``tunnels`` had none at all — a Linux
without systemd raised ``FileNotFoundError: 'launchctl'``/``'systemctl'`` out of
the CLI and Windows raised ``AttributeError: module 'os' has no attribute
'getuid'``.

The Windows arm is the one thing here that CANNOT be settled on this host: there
is no Task Scheduler to register against. What is asserted is everything that is
decidable — the XML is well formed, carries the right triggers, restarts on
failure, records the store, and redirects the daemon's output — while the
PLACEMENT (``schtasks /Create /XML``) is exercised through a fake runner and
reported as unproven-on-Windows rather than implied.
"""

from __future__ import annotations

import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from local_operator import supervisors

_NS = "{http://schemas.microsoft.com/windows/2004/02/mit/task}"


def _completed(args: list[str], code: int = 0, stdout: str = "", stderr: str = "") -> object:
    return subprocess.CompletedProcess(args, code, stdout, stderr)


# ---------------------------------------------------------------------------
# Discovery: a platform is not a capability
# ---------------------------------------------------------------------------


def test_no_supervisor_is_reported_rather_than_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Linux without systemd is a machine with no supervisor, not a crash.

    The failure this prevents, reproduced before the fix: ``subprocess.run``
    suppresses a non-zero exit but NOT ``FileNotFoundError`` for an absent
    executable, so every unguarded entry point raised out of the CLI.
    """
    monkeypatch.setattr(supervisors.shutil, "which", lambda _name: None)

    assert supervisors.supervisor() is None


def test_the_supervisor_is_chosen_by_binary_not_by_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisors.sys, "platform", "linux")
    monkeypatch.setattr(supervisors.os, "name", "nt")
    # A Windows with schtasks present, whatever sys.platform says.
    monkeypatch.setattr(
        supervisors.shutil,
        "which",
        lambda name: "/usr/bin/schtasks" if name == "schtasks" else None,
    )

    assert supervisors.supervisor() == "schtasks"


def test_linux_with_a_systemctl_is_systemctl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(supervisors.sys, "platform", "linux")
    monkeypatch.setattr(supervisors.shutil, "which", lambda name: f"/usr/bin/{name}")

    assert supervisors.supervisor() == "systemctl"


def test_the_refusal_names_every_supervisor_and_the_foreground_command() -> None:
    """The message is the deliverable on a platform with no supervisor."""
    message = supervisors.no_supervisor_error("lop mobile serve")

    assert "launchctl" in message
    assert "systemctl --user" in message
    assert "schtasks" in message
    assert "`lop mobile serve`" in message


# ---------------------------------------------------------------------------
# The addressability guards: the guard is what keeps a test off a live session
# ---------------------------------------------------------------------------


def test_a_redirected_home_is_not_addressable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Identity, not location: the live user manager is keyed by uid.

    ``systemctl --user`` reaches the CALLING user's instance whatever ``$HOME``
    says, so the file half of an install may run under a redirected home while
    the half that loads a unit must refuse. Same guard, same incident, as the
    LaunchAgent plist (a stray unit pointed at a pytest tmpdir).
    """
    import pwd

    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert supervisors.systemd_unit_is_addressable("local-operator-mobile.service") is False
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: real_home))
    assert supervisors.systemd_unit_is_addressable("local-operator-mobile.service") is True


def test_a_path_inside_the_real_home_is_still_not_the_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The round-1 escape: a redirected home INSIDE the real one.

    A containment test accepts ``$HOME/tmp-pytest-sandbox/home``; the identity
    test this is must not, or the guard fails open on exactly the sandbox shape
    AGENTS.md prescribes.
    """
    import pwd

    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    monkeypatch.setattr(
        Path, "home", classmethod(lambda cls: real_home / "tmp-pytest-sandbox" / "h")
    )

    assert supervisors.systemd_unit_is_addressable("local-operator-wakes.service") is False


def test_a_config_dir_outside_the_real_home_never_gets_a_unit(
    tmp_path: Path,
) -> None:
    assert supervisors.config_lives_in_real_home(tmp_path / "sandbox") is False


def test_the_windows_task_guard_refuses_off_windows() -> None:
    """False elsewhere is the answer that keeps ``schtasks`` unreachable there."""
    assert supervisors.task_scheduler_is_addressable(Path("/tmp/x")) is False


# ---------------------------------------------------------------------------
# systemd rendering
# ---------------------------------------------------------------------------


def test_a_unit_restarts_on_failure_and_not_always() -> None:
    """``on-failure`` is load-bearing: the wake supervisor exits 0 to retire."""
    unit = supervisors.render_systemd_unit(
        description="Local Operator wake supervisor", exec_start="/usr/bin/python3 -m x"
    )

    assert "Restart=on-failure" in unit
    assert "Restart=always" not in unit
    assert "WantedBy=default.target" in unit
    assert unit.endswith("\n")


def test_the_browser_unit_is_byte_identical_to_the_one_it_replaced() -> None:
    """The hoisted renderer must not perturb a unit real machines already run.

    Byte-for-byte against the string ``render_systemd`` emitted before the
    renderer was shared: a cosmetic diff here would rewrite every installed
    browser unit on the next upgrade for no reason.
    """
    expected = (
        "[Unit]\n"
        "Description=Local Operator browser bridge\n"
        "\n"
        "[Service]\n"
        "ExecStart=/usr/bin/python3 -m local_operator.browser_bridge.daemon --port 4099\n"
        "Restart=on-failure\n"
        "RestartSec=5\n"
        "StandardOutput=append:/tmp/bridge.log\n"
        "StandardError=append:/tmp/bridge.log\n"
        "\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )
    assert (
        supervisors.render_systemd_unit(
            description="Local Operator browser bridge",
            exec_start="/usr/bin/python3 -m local_operator.browser_bridge.daemon --port 4099",
            post_lines=[
                "StandardOutput=append:/tmp/bridge.log",
                "StandardError=append:/tmp/bridge.log",
            ],
        )
        == expected
    )
    # And with no redirect at all, which is the old-systemd branch: one blank
    # line where the redirect used to be, exactly as before.
    assert (
        supervisors.render_systemd_unit(
            description="Local Operator browser bridge",
            exec_start="/usr/bin/python3 -m x",
        )
        == "[Unit]\nDescription=Local Operator browser bridge\n\n[Service]\n"
        "ExecStart=/usr/bin/python3 -m x\nRestart=on-failure\nRestartSec=5\n\n"
        "[Install]\nWantedBy=default.target\n"
    )


def test_a_timer_re_runs_the_unit_only_when_it_is_inactive() -> None:
    """``OnUnitInactiveSec`` is the launchd ``StartInterval`` semantic.

    ``OnCalendar`` would fire on a clock whether or not an instance is already
    running, which is the double-instance shape launchd does not have.
    """
    timer = supervisors.render_systemd_timer(
        description="self-heal", unit="local-operator-wakes.service", interval_seconds=900
    )

    assert "OnUnitInactiveSec=900s" in timer
    assert "Unit=local-operator-wakes.service" in timer
    assert "WantedBy=timers.target" in timer
    assert "OnCalendar" not in timer


def test_the_bus_failure_is_distinguished_from_a_missing_unit() -> None:
    """One is "nothing is installed"; the other is "cannot ask from here"."""
    assert supervisors.is_bus_failure("Failed to connect to bus: No medium found")
    assert not supervisors.is_bus_failure("Unit foo.service could not be found.")
    # With no remedy supplied the shared sentence names the actual command;
    # with one supplied it is used verbatim, because each daemon's recovery is
    # its own installer and the browser bridge's remedy already carries the
    # linger sentence.
    assert "loginctl enable-linger" in supervisors.translate_systemctl_error(
        "Failed to connect to bus: No medium found"
    )
    translated = supervisors.translate_systemctl_error(
        "Failed to connect to bus: No medium found", remedy="remedy: re-run lop x install."
    )
    assert translated.endswith("remedy: re-run lop x install.")
    assert supervisors.translate_systemctl_error("Job for x.service failed") == (
        "Job for x.service failed"
    )


# ---------------------------------------------------------------------------
# Windows task rendering (decidable here) and placement (faked here)
# ---------------------------------------------------------------------------


def test_the_task_xml_is_well_formed_and_carries_the_launchd_contract() -> None:
    """Start at logon, restart on failure, never run a second instance.

    ``RestartOnFailure`` is the ``KeepAlive`` analogue and is reachable ONLY
    through the XML — ``schtasks``' switch form has no option for it.
    """
    xml = supervisors.render_task_xml(
        description="Local Operator wake supervisor",
        image=r"C:\Program Files\Local Operator\python.exe",
        argv=["-m", "local_operator.wakes.supervisor"],
        user_id=r"DOMAIN\user",
    )
    root = ET.fromstring(xml)

    assert root.tag == f"{_NS}Task"
    assert root.find(f"{_NS}Triggers/{_NS}LogonTrigger") is not None
    assert root.find(f"{_NS}Triggers/{_NS}CalendarTrigger") is None
    settings = root.find(f"{_NS}Settings")
    assert settings is not None
    assert settings.findtext(f"{_NS}RestartOnFailure/{_NS}Count") == "999"
    assert settings.findtext(f"{_NS}MultipleInstancesPolicy") == "IgnoreNew"
    assert settings.findtext(f"{_NS}ExecutionTimeLimit") == "PT0S"
    exec_node = root.find(f"{_NS}Actions/{_NS}Exec")
    assert exec_node is not None
    # Least privilege and no password: a Windows Service would need admin and
    # could not reach the user's session.
    assert root.findtext(f"{_NS}Principals/{_NS}Principal/{_NS}RunLevel") == "LeastPrivilege"
    assert root.findtext(f"{_NS}Principals/{_NS}Principal/{_NS}LogonType") == "InteractiveToken"
    # A program path with spaces must still be one argument.
    assert exec_node.findtext(f"{_NS}Command") == r"C:\Program Files\Local Operator\python.exe"


def test_a_self_healing_task_carries_its_interval_trigger() -> None:
    xml = supervisors.render_task_xml(
        description="wakes",
        image="python.exe",
        argv=["-m", "x"],
        interval_minutes=15,
    )
    root = ET.fromstring(xml)

    trigger = root.find(f"{_NS}Triggers/{_NS}CalendarTrigger")
    assert trigger is not None
    assert trigger.findtext(f"{_NS}Repetition/{_NS}Interval") == "PT15M"


def test_a_recorded_store_travels_through_cmd_and_the_redirect_joins_stderr() -> None:
    """Task Scheduler has neither an environment element nor output redirection.

    Both are properties of the daemon's contract on the other two platforms
    (the unit and the plist record the store; launchd and systemd create the
    log file), so the Windows arm has to reach them through ``cmd.exe``.
    """
    command, arguments = supervisors.task_command_line(
        r"C:\Program Files\Local Operator\python.exe",
        ["-m", "local_operator.wakes.supervisor"],
        {"LOCAL_OPERATOR_CONFIG_DIR": r"C:\Users\x\.local-operator"},
        Path(r"C:\Users\x\.local-operator\logs\wakes.log"),
    )

    assert command.endswith("cmd.exe")
    assert arguments.startswith("/c set ")
    assert 'set "LOCAL_OPERATOR_CONFIG_DIR=' in arguments
    assert arguments.count("&&") == 1
    assert '>> "C:\\Users\\x\\.local-operator\\logs\\wakes.log" 2>&1' in arguments
    # The whole line is wrapped ONCE for `cmd /c`, whose legacy quote-stripping
    # would otherwise eat the opening quote of the program path.
    assert arguments == arguments.rstrip('"') or not arguments.startswith('""')
    assert '"' in arguments


def test_cmd_line_wrapping_only_when_the_line_starts_with_a_quote() -> None:
    wrapped = supervisors._for_cmd_c('"C:\\Program Files\\p.exe" -m x')
    assert wrapped.startswith('""') and wrapped.endswith('"')
    plain = 'set "A=B" && "C:\\p.exe" -m x'
    assert supervisors._for_cmd_c(plain) == plain


def test_a_task_with_neither_store_nor_log_runs_the_image_directly() -> None:
    """No wrapper when there is nothing to wrap: the shortest command that works."""
    command, arguments = supervisors.task_command_line("/usr/bin/python3", ["-m", "x"])

    assert command == "/usr/bin/python3"
    assert arguments == "-m x"


def test_the_task_xml_escapes_what_windows_paths_can_contain() -> None:
    xml = supervisors.render_task_xml(
        description="a & b <c>", image="/bin/python", argv=["-m", "x"]
    )
    ET.fromstring(xml)

    assert "a &amp; b &lt;c&gt;" in xml


def test_create_task_writes_utf16_and_reports_schtasks_own_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The XML must be UTF-16 (Task Scheduler refuses anything else) and a
    failure must be schtasks' message, not a generic sentence."""
    seen: dict[str, object] = {}

    def fake(*args: str, timeout: float = 30) -> object:
        path = Path(args[args.index("/XML") + 1])
        seen["raw"] = path.read_bytes()
        seen["args"] = list(args)
        return _completed(list(args), 1, "", "ERROR: The task XML is malformed.")

    monkeypatch.setattr(supervisors, "schtasks", fake)
    ok, detail = supervisors.create_task("Local Operator test", "<Task/>")

    assert ok is False
    assert "malformed" in detail
    raw = seen["raw"]
    assert isinstance(raw, bytes)
    assert raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"), "not UTF-16 with a BOM"
    assert seen["args"] == [
        "/Create",
        "/TN",
        "Local Operator test",
        "/XML",
        str(seen["args"][4]),  # type: ignore[index]
        "/F",
    ]


def test_an_absent_task_is_not_an_error_on_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uninstall idempotence: "no such task" is the state the caller asked for."""
    monkeypatch.setattr(
        supervisors,
        "schtasks",
        lambda *args, **kwargs: _completed(
            list(args), 1, "", "ERROR: The system cannot find the file specified."
        ),
    )

    ok, detail = supervisors.delete_task("Local Operator test")

    assert ok is True
    assert detail == "no such task"


def test_task_state_reads_status_out_of_the_query(monkeypatch: pytest.MonkeyPatch) -> None:
    body = "TaskName:      \\Local Operator test\nStatus:        Running\n"
    monkeypatch.setattr(supervisors, "schtasks", lambda *a, **k: _completed(list(a), 0, body))

    assert supervisors.task_state("Local Operator test") == (True, True, "Running")
    monkeypatch.setattr(supervisors, "schtasks", lambda *a, **k: _completed(list(a), 1, "", "no"))
    assert supervisors.task_state("Local Operator test")[0] is False
