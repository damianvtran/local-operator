"""Supervisor-layer guards for the browser bridge installer.

There was no coverage for ``browser_bridge/install.py`` at all before this
file, which is how four separate defects survived in the Linux path: a log
destination the product could not read, an uncaught ``FileNotFoundError`` on
every systemd-less machine, an uninstall that reported success it had not
achieved, and a supervisor label that collided across config roots.

The systemd branch is exercised on macOS by monkeypatching ``sys.platform``
and ``shutil.which`` — the same technique ``tests/unit/mcp/test_manager.py``
already uses — because the code under test dispatches on exactly those two
things. Where a claim can only be settled by a real service manager (whether
``append:`` actually redirects, whether a non-lingering user has a manager at
all) it was verified by hand against systemd 255 in a container and is cited
in the docstrings; those are deliberately NOT asserted here, since a test that
mocks systemd cannot prove what systemd does.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from local_operator.browser_bridge import install


@pytest.fixture
def linux(monkeypatch: pytest.MonkeyPatch) -> None:
    """Present as Linux WITH systemd available."""
    monkeypatch.setattr(install.sys, "platform", "linux")
    monkeypatch.setattr(install.shutil, "which", lambda name: f"/usr/bin/{name}")


@pytest.fixture
def linux_without_systemd(monkeypatch: pytest.MonkeyPatch) -> None:
    """Devuan/Alpine/Void/OpenRC/WSL2-without-systemd, and every container."""
    monkeypatch.setattr(install.sys, "platform", "linux")
    monkeypatch.setattr(install.shutil, "which", lambda name: None)


@pytest.fixture
def isolated_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A config root that is NOT the uid's default — i.e. an isolated run."""
    root = tmp_path / "isolated-config"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    return root


# --------------------------------------------------------------------------
# L1 — the generated unit has to send output somewhere the product reads.
# --------------------------------------------------------------------------


def test_systemd_unit_redirects_output_to_the_log_file_on_modern_systemd() -> None:
    """Without this, output goes to the journal while three surfaces cite a file.

    Reproduced on real systemd 255 before the fix: the unit produced output,
    ``tail`` of ``log_path()`` reported "No such file or directory", and the
    lines were in ``journalctl`` instead.
    """
    unit = install.render_systemd(4099, version=255)
    assert f"StandardOutput=append:{install.log_path()}" in unit
    assert f"StandardError=append:{install.log_path()}" in unit


def test_systemd_unit_omits_append_on_systemd_too_old_for_it() -> None:
    """``append:`` is a 240+ value, and an older systemd drops it SILENTLY.

    Measured on systemd 255 with a deliberately invalid specifier: the unit
    still starts and logs "Failed to parse output specifier, ignoring". So
    emitting it blindly would not break the daemon — it would leave the output
    in the journal while the product kept citing a file nothing writes, which
    is the defect being fixed. The gate is what keeps the unit and
    ``log_location()`` agreeing on every systemd.
    """
    unit = install.render_systemd(4099, version=239)
    assert "append:" not in unit
    # Still a well-formed unit that would start.
    assert "ExecStart=" in unit
    assert "WantedBy=default.target" in unit


def test_systemd_unit_omits_append_when_the_version_cannot_be_read() -> None:
    """Unknown degrades to "assume old": keep the unit loadable."""
    assert "append:" not in install.render_systemd(4099, version=None)


def test_logs_command_reads_the_journal_when_the_log_file_is_not_written(
    linux: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``lop browser logs`` must not tail a path nothing writes."""
    monkeypatch.setattr(install, "systemd_version", lambda: 239)
    command = install.logs_command(50)
    assert command[:2] == ["journalctl", "--user"]
    assert install.systemd_unit() in command
    assert "-f" not in command


def test_logs_command_follows_the_journal_when_asked(
    linux: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install, "systemd_version", lambda: 239)
    assert install.logs_command(50, follow=True)[-1] == "-f"


def test_logs_command_tails_the_file_when_systemd_does_redirect(
    linux: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(install, "systemd_version", lambda: 255)
    command = install.logs_command(50)
    assert command[0] == "tail"
    assert command[-1] == str(install.log_path())


def test_status_reports_the_journal_as_the_log_location_on_old_systemd(
    linux: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``log:`` line users are told to look at must be true.

    This is the surface that made the empty directory read as "the daemon
    produced no output".
    """
    monkeypatch.setattr(install, "systemd_version", lambda: 239)
    assert install.log_location() == f"journalctl --user -u {install.systemd_unit()}"


# --------------------------------------------------------------------------
# L2 — no entry point may raise on a machine without a supervisor.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("action", ["start", "stop", "restart"])
def test_service_action_returns_an_error_instead_of_raising(
    action: str, linux_without_systemd: None
) -> None:
    """Pre-fix this raised ``FileNotFoundError: 'systemctl'`` out of the CLI.

    ``check=False`` suppresses a non-zero exit status, not a missing binary —
    the distinction the original code was relying on and did not have.
    """
    result = install.service_action(action)
    assert result["ok"] is False
    assert "no supported user service supervisor" in str(result["error"])


def test_uninstall_returns_an_error_instead_of_raising(linux_without_systemd: None) -> None:
    result = install.uninstall()
    assert result["ok"] is False
    assert "no supported user service supervisor" in str(result["error"])


def test_install_refuses_without_a_supervisor_using_the_same_message(
    linux_without_systemd: None,
) -> None:
    """One shared string: the entry points cannot describe one machine differently."""
    result = install.install()
    assert result["ok"] is False
    assert result["error"] == install.NO_SUPERVISOR_ERROR


def test_service_action_never_shells_out_when_there_is_no_supervisor(
    linux_without_systemd: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard must come BEFORE the subprocess call, not wrap it."""

    def explode(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not invoke a subprocess without a supervisor")

    monkeypatch.setattr(install.subprocess, "run", explode)
    assert install.service_action("start")["ok"] is False


# --------------------------------------------------------------------------
# L2 (recovery) — Linux gains the reload-and-retry macOS already had.
# --------------------------------------------------------------------------


def test_start_reloads_and_retries_when_the_unit_is_not_yet_known(
    linux: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A unit on disk the manager has not read fails until something reloads it."""
    monkeypatch.setattr(install, "systemd_path", lambda: tmp_path / "unit.service")
    (tmp_path / "unit.service").write_text("[Service]\n", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(cmd[2:]))
        failed = cmd[2] == "start" and ("daemon-reload",) not in calls
        return subprocess.CompletedProcess(
            cmd, 1 if failed else 0, "", "Unit not found." if failed else ""
        )

    monkeypatch.setattr(install.subprocess, "run", fake_run)
    result = install.service_action("start")
    assert result["ok"] is True, "a reload-then-retry should recover this"
    assert ("daemon-reload",) in calls


def test_stop_does_not_reload_and_retry(
    linux: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Recovery is for start/restart; a failed stop is not fixed by a reload."""
    monkeypatch.setattr(install, "systemd_path", lambda: tmp_path / "unit.service")
    (tmp_path / "unit.service").write_text("[Service]\n", encoding="utf-8")
    calls: list[tuple[str, ...]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(cmd[2:]))
        return subprocess.CompletedProcess(cmd, 1, "", "boom")

    monkeypatch.setattr(install.subprocess, "run", fake_run)
    assert install.service_action("stop")["ok"] is False
    assert ("daemon-reload",) not in calls


# --------------------------------------------------------------------------
# L3 — uninstall must not claim a removal it did not perform.
# --------------------------------------------------------------------------


def test_uninstall_reports_failure_when_disable_fails(
    linux: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pre-fix this returned ``ok: True`` and the CLI exited 0.

    Reproduced with a real ``systemctl`` stub exiting 1: the result was
    ``{'ok': True, 'steps': ['removed the systemd user service']}``.
    """
    unit = tmp_path / "unit.service"
    unit.write_text("[Service]\n", encoding="utf-8")
    monkeypatch.setattr(install, "systemd_path", lambda: unit)
    monkeypatch.setattr(
        install.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "Failed to disable"),
    )
    result = install.uninstall()
    assert result["ok"] is False
    steps = result["steps"]
    assert isinstance(steps, list)
    assert any("failed" in str(step) for step in steps)


def test_uninstall_says_nothing_was_installed_rather_than_claiming_removal(
    linux: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(install, "systemd_path", lambda: tmp_path / "absent.service")
    monkeypatch.setattr(
        install.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "Unit not loaded"),
    )
    result = install.uninstall()
    assert result["ok"] is True
    steps = result["steps"]
    assert isinstance(steps, list)
    assert any("no systemd user service" in str(step) for step in steps)


# --------------------------------------------------------------------------
# L4 — lingering, and the bus error that names its remedy.
# --------------------------------------------------------------------------


def test_install_enables_lingering_before_enabling_the_unit(
    linux: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Order matters: without a user manager the enable itself fails."""
    monkeypatch.setattr(install, "systemd_path", lambda: tmp_path / "unit.service")
    monkeypatch.setattr(install, "log_path", lambda: tmp_path / "bridge.log")
    monkeypatch.setattr(install, "health", lambda *a, **k: {"ok": True})
    order: list[str] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        order.append("linger" if "enable-linger" in cmd else " ".join(cmd[2:3]))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(install.subprocess, "run", fake_run)
    result = install.install(4099)
    assert result["ok"] is True
    assert "linger" in order, "install must attempt loginctl enable-linger"
    assert order.index("linger") < order.index("enable")


def test_a_refused_linger_does_not_fail_the_install(
    linux: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """It can legitimately be refused; the install otherwise succeeded."""
    monkeypatch.setattr(install, "systemd_path", lambda: tmp_path / "unit.service")
    monkeypatch.setattr(install, "log_path", lambda: tmp_path / "bridge.log")
    monkeypatch.setattr(install, "health", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(install, "enable_linger", lambda: False)
    monkeypatch.setattr(
        install.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, "", ""),
    )
    result = install.install(4099)
    assert result["ok"] is True
    steps = result["steps"]
    assert isinstance(steps, list)
    assert any("lingering" in str(step) for step in steps)


def test_the_bus_error_is_translated_into_the_linger_remedy() -> None:
    """Surfacing systemd's stderr verbatim was truthful but not actionable.

    Nothing in the product named ``loginctl enable-linger`` before this.
    """
    translated = install._translate_systemctl_error("Failed to connect to bus: No medium found")
    assert "loginctl enable-linger" in translated
    assert "No medium found" in translated, "keep the original cause too"


def test_an_unrelated_systemctl_error_is_passed_through_unchanged() -> None:
    """Do not attach the linger advice to failures it does not explain."""
    translated = install._translate_systemctl_error("Job for x.service failed")
    assert "loginctl enable-linger" not in translated


def test_enable_linger_is_absent_rather_than_raising_without_loginctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(install.shutil, "which", lambda name: None)
    assert install.enable_linger() is False


# --------------------------------------------------------------------------
# L11 — the supervisor name is global per-uid, so it must follow the root.
# --------------------------------------------------------------------------


def test_the_default_config_root_keeps_the_historical_label_byte_for_byte(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BACKWARD COMPATIBILITY, and the sharp edge of this change.

    Every already-installed user's daemon is registered under exactly these
    names. If they moved, that daemon becomes an orphan: a running process the
    CLI can no longer stop, start or uninstall, with no way back short of
    hand-editing launchd/systemd. The default root must therefore be
    indistinguishable from the pre-change build.
    """
    monkeypatch.delenv("LOCAL_OPERATOR_CONFIG_DIR", raising=False)
    # Point the config root at the uid's REAL default location. The suite's
    # autouse isolation redirects HOME to a tmp dir, which is itself a
    # non-default root — so without this the fixture, not the code, is what
    # makes the label differ, and the compatibility claim goes untested.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(install._default_config_root()))
    assert install.label() == "com.local-operator.browser"
    assert install.systemd_unit() == "local-operator-browser.service"


def test_the_default_label_is_anchored_on_the_uid_not_on_HOME(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The collision namespace is ``gui/<uid>``, which does not move with $HOME.

    Deriving "is this the default install?" from ``Path.home()`` would let an
    isolated ``HOME=/tmp/... lop browser install`` compare its root against its
    own home, conclude it IS the default, and reuse the real daemon's label —
    the exact collision this fixes.
    """
    monkeypatch.delenv("LOCAL_OPERATOR_CONFIG_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert install.label() != "com.local-operator.browser"


def test_an_isolated_config_root_gets_its_own_label(isolated_root: Path) -> None:
    """Verified against launchd: bootout resolves a plist to the Label INSIDE it,
    so a same-label plist at another path evicts the incumbent. An isolated run
    killed the operator's live daemon on this machine before this fix."""
    assert install.label().startswith("com.local-operator.browser.")
    assert install.label() != "com.local-operator.browser"
    assert install.systemd_unit() != "local-operator-browser.service"
    assert install.systemd_unit().endswith(".service")


def test_two_different_isolated_roots_do_not_collide(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "a"))
    (tmp_path / "a").mkdir()
    first = install.label()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "b"))
    (tmp_path / "b").mkdir()
    assert install.label() != first


def test_the_label_is_stable_across_equivalent_spellings_of_one_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Otherwise ``/tmp/x`` and ``/tmp/./x`` would manage two different daemons."""
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    plain = install.label()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root) + "/./")
    assert install.label() == plain


def test_the_plist_and_unit_paths_follow_the_label(isolated_root: Path) -> None:
    assert install.plist_path().name == f"{install.label()}.plist"
    assert install.systemd_path().name == install.systemd_unit()


def test_the_rendered_plist_carries_this_roots_label(isolated_root: Path) -> None:
    """The Label INSIDE the plist is what launchd registers, so it is the one
    that must be per-root — the filename alone is not enough."""
    assert install.render_plist(4099)["Label"] == install.label()


def test_a_default_install_reports_no_orphan(monkeypatch: pytest.MonkeyPatch) -> None:
    """The default root's name did not change, so its install is adopted, not orphaned."""
    monkeypatch.delenv("LOCAL_OPERATOR_CONFIG_DIR", raising=False)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(install._default_config_root()))
    assert install.legacy_registration() is None


def test_an_isolated_root_names_an_older_builds_shared_registration(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """MIGRATION: a pre-change build under this root wrote the shared name.

    That file is no longer addressable by the CLI, so it is named rather than
    left as a running daemon nobody can stop.
    """
    home = tmp_path / "home"
    (home / "Library" / "LaunchAgents").mkdir(parents=True)
    (home / "Library" / "LaunchAgents" / f"{install.LABEL}.plist").write_text("x")
    monkeypatch.setattr(install.sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    assert install.legacy_registration() is not None


def test_status_names_the_supervisor_and_root_it_reports_on(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two isolated daemons otherwise produce identical status output."""
    monkeypatch.setattr(install, "health", lambda *a, **k: None)
    result = install.status()
    assert result["supervisor"] == install.label() or result["supervisor"] == (
        install.systemd_unit()
    )
    assert result["config_root"] == str(isolated_root)


# --------------------------------------------------------------------------
# CLI edges: a non-zero exit must never be silent.
# --------------------------------------------------------------------------


def test_uninstall_carries_an_error_string_for_the_cli_to_print(
    linux_without_systemd: None,
) -> None:
    """The no-supervisor case produces no steps, so without this the CLI
    exited 1 having printed nothing at all."""
    result = install.uninstall()
    assert result["ok"] is False
    assert str(result.get("error", "")).strip(), "a failed uninstall must say why"
