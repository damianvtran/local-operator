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

import os
import plistlib
import subprocess
import sys
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
    """Two paths naming ONE directory must produce one daemon, not two.

    Through a real SYMLINK, which is what makes this a test of ``.resolve()``.
    An earlier version compared ``/tmp/x`` against ``/tmp/x/./`` and passed with
    ``.resolve()`` deleted, because ``pathlib`` collapses ``./`` in its own
    constructor (``str(Path("/tmp/x/./")) == "/tmp/x"``) — it named symlink
    canonicalisation and measured string normalisation ``Path`` did for free.
    A symlink survives that constructor, so only a real ``resolve()`` collapses
    it.
    """
    root = tmp_path / "root"
    root.mkdir()
    link = tmp_path / "link-to-root"
    link.symlink_to(root, target_is_directory=True)
    # Guard the guard: if these were already equal as strings the assertion
    # below would hold with no canonicalisation at all.
    assert str(link) != str(root)

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    through_real_path = install.label()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(link))
    assert install.label() == through_real_path


def test_the_digest_is_stable_across_processes(tmp_path: Path) -> None:
    """The label must survive a restart, so its digest cannot be salted.

    ``hash()`` is randomised per process by ``PYTHONHASHSEED``, so swapping
    sha256 for it would give the SAME process one stable label — passing every
    in-process assertion — while the next ``lop`` invocation computed a
    different one and lost track of the daemon it installed. Nothing else in
    the suite would catch that, so this pins the value across a real
    interpreter boundary rather than within one.
    """
    root = tmp_path / "stable-root"
    root.mkdir()
    program = (
        "import os, sys;"
        "sys.path.insert(0, os.environ['LO_REPO']);"
        "from local_operator.browser_bridge import install;"
        "print(install.label())"
    )
    env = {
        **os.environ,
        "LOCAL_OPERATOR_CONFIG_DIR": str(root),
        "LO_REPO": str(Path(install.__file__).resolve().parents[2]),
    }
    seen = set()
    for seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True,
            text=True,
            env={**env, "PYTHONHASHSEED": seed},
        )
        assert result.returncode == 0, result.stderr
        seen.add(result.stdout.strip())
    assert len(seen) == 1, f"label changed across PYTHONHASHSEED values: {seen}"
    assert seen.pop().startswith("com.local-operator.browser.")


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

    An evidence-free file is NAMED, never claimed. It is not addressable by this
    build, so leaving it silent would present a running daemon nobody can stop
    as "nothing installed" — but claiming it on location alone is how a
    non-default root came to delete the default root's live LaunchAgent.
    """
    home = tmp_path / "home"
    (home / "Library" / "LaunchAgents").mkdir(parents=True)
    (home / "Library" / "LaunchAgents" / f"{install.LABEL}.plist").write_text("x")
    monkeypatch.setattr(install.sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(install, "_passwd_home", lambda: home)
    assert install.legacy_registration() is None, "no evidence means not ours to claim"
    reported = install.legacy_ambiguity()
    assert reported is not None and str(install.LABEL) in reported


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


# --------------------------------------------------------------------------
# A1 — a config root that INHERITS a pre-per-root build's registration.
#
# `LOCAL_OPERATOR_CONFIG_DIR` is a documented setting, and a `$HOME` that
# differs from the passwd home suffixes the label too. For those users the
# released build registered the DEFAULT name while this build looks for a
# suffixed one, so every entry point has to resolve the inherited one or it
# reports a success it did not achieve.
# --------------------------------------------------------------------------


@pytest.fixture
def inherited_darwin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_root: Path) -> Path:
    """A CLAIMABLE legacy plist: written as a released build writes it, carrying
    the evidence that makes it provably this root's.

    ``StandardOutPath`` is what the released ``render_plist`` records
    (``str(log_path())`` — verified against v0.51.14) and it is the ownership
    evidence, because it derives from the CONFIG ROOT: a plist written under a
    different root records a different path. The default root's directory is
    deliberately absent, since while it exists the shared-name registration may
    be the default install's and is refused. Both conditions have their own
    negative tests below.
    """
    home = tmp_path / "home"
    (home / "Library" / "LaunchAgents").mkdir(parents=True)
    monkeypatch.setattr(install.sys, "platform", "darwin")
    monkeypatch.setattr(install.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(install, "_passwd_home", lambda: home)
    legacy = home / "Library" / "LaunchAgents" / f"{install.LABEL}.plist"
    legacy.write_bytes(
        plistlib.dumps({"Label": install.LABEL, "StandardOutPath": str(install.log_path())})
    )
    return legacy


def test_uninstall_removes_the_registration_an_older_build_left(
    inherited_darwin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reproduced before this fix: ``{'ok': True, 'steps': ['no LaunchAgent was
    installed']}`` while the plist stayed on disk and its daemon kept running —
    the same false success ``uninstall`` exists to prevent."""
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        install,
        "_launchctl",
        lambda *a: calls.append(a) or subprocess.CompletedProcess(list(a), 0, "", ""),
    )
    result = install.uninstall()
    assert result["ok"] is True
    assert not inherited_darwin.exists(), "the inherited plist must actually be removed"
    assert any(a[0] == "bootout" and str(inherited_darwin) in a[-1] for a in calls)
    steps = result["steps"]
    assert isinstance(steps, list)
    assert any("older build" in str(step) for step in steps)


def test_uninstall_does_not_claim_a_removal_it_did_not_make(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No own install and nothing inherited: still says so plainly."""
    home = tmp_path / "empty-home"
    (home / "Library" / "LaunchAgents").mkdir(parents=True)
    monkeypatch.setattr(install.sys, "platform", "darwin")
    monkeypatch.setattr(install.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(install, "_passwd_home", lambda: home)
    monkeypatch.setattr(
        install, "_launchctl", lambda *a: subprocess.CompletedProcess(list(a), 0, "", "")
    )
    result = install.uninstall()
    steps = result["steps"]
    assert isinstance(steps, list)
    assert any("no LaunchAgent was installed" in str(step) for step in steps)
    assert not any("older build" in str(step) for step in steps)


def test_status_counts_an_inherited_registration_as_installed(
    inherited_darwin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "installed: no" beside a running daemon sends the user to reinstall on
    top of what they already have."""
    monkeypatch.setattr(install, "health", lambda *a, **k: None)
    result = install.status()
    assert result["installed"] is True
    assert result["legacy_registration"] == str(inherited_darwin)


def test_status_reports_no_legacy_registration_for_a_default_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default root is adopted, never orphaned, so nothing to report."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(install._default_config_root()))
    monkeypatch.setattr(install, "health", lambda *a, **k: None)
    assert install.status()["legacy_registration"] is None


@pytest.mark.parametrize("action", ["start", "stop", "restart"])
def test_service_action_targets_the_inherited_label(
    action: str, inherited_darwin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise start/stop/restart address a service that was never registered
    while the real daemon keeps running."""
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        install,
        "_launchctl",
        lambda *a: calls.append(a) or subprocess.CompletedProcess(list(a), 0, "", ""),
    )
    install.service_action(action)
    targets = [a[-1] for a in calls]
    assert any(t.endswith(f"/{install.LABEL}") or t == str(inherited_darwin) for t in targets)
    assert not any(install.label() in t for t in targets), "must not use the suffixed name"


def test_an_own_registration_wins_over_an_inherited_one(
    inherited_darwin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adoption is only for a root with no install of its own."""
    install.plist_path().write_bytes(b"<plist>this root</plist>")
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        install,
        "_launchctl",
        lambda *a: calls.append(a) or subprocess.CompletedProcess(list(a), 0, "", ""),
    )
    install.service_action("start")
    assert any(install.label() in a[-1] for a in calls)


def test_a_redirected_home_never_adopts_the_passwd_homes_registration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_root: Path
) -> None:
    """THE A3 GUARD. An isolated root must not claim another root's supervisor.

    ``HOME=/tmp/... LOCAL_OPERATOR_CONFIG_DIR=...`` is the isolation AGENTS.md
    prescribes for any bridge work. An earlier version of this fix also searched
    the passwd home, so that pattern resolved the OPERATOR's live plist; the
    isolated root has no install of its own, so it was ADOPTED and would then be
    booted out, unlinked and SIGTERM'd. The fake passwd-home plist must satisfy
    the other ownership checks: valid matching-log evidence and no default-root
    directory. A malformed or no-log fixture is rejected by the evidence guard
    even if HOME discovery widens, so it cannot test this boundary.

    The lookup is HOME-keyed on purpose ("did MY predecessor write this?"),
    unlike ``_default_config_root``, which is UID-keyed ("does this root own the
    default NAME?"). A registration under a different ``$HOME`` belongs to a
    different run.

    Uses a FAKE passwd home: this must never depend on, or touch, the real one.
    """
    passwd_home = tmp_path / "passwd-home"
    (passwd_home / "Library" / "LaunchAgents").mkdir(parents=True)
    other_root_plist = passwd_home / "Library" / "LaunchAgents" / f"{install.LABEL}.plist"
    other_root_plist.write_bytes(
        plistlib.dumps({"Label": install.LABEL, "StandardOutPath": str(install.log_path())})
    )
    original_plist = other_root_plist.read_bytes()
    redirected = tmp_path / "redirected-home"
    (redirected / "Library" / "LaunchAgents").mkdir(parents=True)
    monkeypatch.setattr(install.sys, "platform", "darwin")
    monkeypatch.setattr(install.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: redirected))
    monkeypatch.setattr(install, "_passwd_home", lambda: passwd_home)

    # Isolate the discovery boundary: none of the later checks may reject this
    # candidate and conceal a passwd-home fallback regression.
    assert not install._default_config_root().exists()
    assert not install._own_registration_exists()
    recorded = install._recorded_log_path(other_root_plist)
    assert recorded is not None
    assert install._canonical(recorded) == install._canonical(install.log_path())
    assert install.legacy_registration() is None, "must not resolve another HOME's plist"

    # And the consequences, not merely the lookup: nothing may target or delete
    # a registration this root does not own.
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        install,
        "_launchctl",
        lambda *a: calls.append(a) or subprocess.CompletedProcess(list(a), 0, "", ""),
    )
    install.uninstall()
    assert (
        other_root_plist.read_bytes() == original_plist
    ), "uninstall must preserve another root's plist byte-for-byte"
    install.service_action("stop")
    assert not any(str(other_root_plist) in c[-1] for c in calls)
    assert not any(
        c[-1].endswith(f"/{install.LABEL}") for c in calls
    ), "must not address the default label it does not own"


def test_the_legacy_unit_is_resolved_on_linux_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_root: Path
) -> None:
    """A JOURNAL-ONLY legacy unit is never auto-adopted, on name alone.

    This is the Linux case that has no macOS twin. The released
    ``render_systemd`` emitted ``ExecStart`` only — no ``StandardOutput=`` —
    so every unit written before this build carries NO ownership evidence at
    all, and its output went to the journal. Adopting one because its NAME
    matches is precisely the reasoning that let a root claim another root's
    supervisor, so it is refused and reported instead.
    """
    home = tmp_path / "linux-home"
    unit_dir = home / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True)
    legacy = unit_dir / install.SYSTEMD_UNIT
    # Exactly what v0.51.14's render_systemd produced.
    legacy.write_text(
        "[Unit]\nDescription=Local Operator browser bridge\n\n"
        "[Service]\nExecStart=/usr/bin/python -m local_operator.browser_bridge.daemon\n"
        "Restart=on-failure\nRestartSec=5\n\n[Install]\nWantedBy=default.target\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(install.sys, "platform", "linux")
    monkeypatch.setattr(install.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(install, "_passwd_home", lambda: home)

    assert install.legacy_registration() is None, "journal-only unit carries no evidence"
    reported = install.legacy_ambiguity()
    assert reported is not None and "journal" in reported

    calls: list[tuple[str, ...]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(install.subprocess, "run", fake_run)
    install.service_action("start")
    # Never addresses the legacy unit name it cannot prove is its own.
    assert not any(install.SYSTEMD_UNIT in c and install.systemd_unit() not in c for c in calls)
    install.uninstall()
    assert legacy.exists(), "an unattributable unit must not be deleted"


# --------------------------------------------------------------------------
# A4 — the systemd half of the inherited-registration handling.
#
# On a PR whose whole purpose is the Linux path, this branch was unguarded:
# deleting it entirely, and separately swapping its by-unit-name disable for
# the suffixed name, both left 187 tests green.
# --------------------------------------------------------------------------


@pytest.fixture
def inherited_linux(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_root: Path) -> Path:
    """A CLAIMABLE legacy unit: carrying the ``StandardOutput=append:`` line that
    is the only ownership evidence a unit can hold.

    Note what this fixture implies about the Linux upgrade path, which is
    covered by its own test below: the RELEASED ``render_systemd`` emitted
    ``ExecStart`` only, so a unit from any build before this one is journal-only
    and carries NO evidence. Those are never auto-adopted; this fixture
    represents a unit written by a build new enough to record the destination
    (systemd >= 240).
    """
    home = tmp_path / "linux-home"
    (home / ".config" / "systemd" / "user").mkdir(parents=True)
    monkeypatch.setattr(install.sys, "platform", "linux")
    monkeypatch.setattr(install.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(install, "_passwd_home", lambda: home)
    legacy = home / ".config" / "systemd" / "user" / install.SYSTEMD_UNIT
    legacy.write_text(
        "[Service]\nExecStart=/bin/true\n" f"StandardOutput=append:{install.log_path()}\n",
        encoding="utf-8",
    )
    return legacy


def _systemctl_spy(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    """Record every ``systemctl`` argv without running one."""
    calls: list[tuple[str, ...]] = []

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(tuple(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(install.subprocess, "run", fake_run)
    return calls


def test_uninstall_removes_the_systemd_unit_an_older_build_left(
    inherited_linux: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Linux twin of the launchd case: the file must actually go, and the
    step must say so, or `uninstall` reports a success it did not achieve."""
    _systemctl_spy(monkeypatch)
    result = install.uninstall()
    assert not inherited_linux.exists(), "the inherited unit file must be removed"
    steps = result["steps"]
    assert isinstance(steps, list)
    assert any("older build" in str(step) for step in steps)


def test_uninstall_disables_the_legacy_unit_by_its_own_name(
    inherited_linux: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """systemd addresses units by NAME, so disabling the suffixed name leaves the
    legacy unit enabled and its daemon running — deleting the file alone does not
    stop it. Swapping this for ``systemd_unit()`` previously passed every test."""
    calls = _systemctl_spy(monkeypatch)
    install.uninstall()
    disables = [c for c in calls if "disable" in c]
    assert any(
        install.SYSTEMD_UNIT in c for c in disables
    ), f"no disable targeted the legacy unit name; saw {disables}"


def test_uninstall_leaves_no_systemd_unit_untouched_when_nothing_was_inherited(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The mirror of the above: with nothing inherited, the legacy name is never
    disabled — otherwise this would disable a unit belonging to someone else."""
    home = tmp_path / "bare-home"
    (home / ".config" / "systemd" / "user").mkdir(parents=True)
    monkeypatch.setattr(install.sys, "platform", "linux")
    monkeypatch.setattr(install.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    calls = _systemctl_spy(monkeypatch)
    install.uninstall()
    assert not any(install.SYSTEMD_UNIT in c and install.systemd_unit() not in c for c in calls)


# --------------------------------------------------------------------------
# Q1/Q2 — ownership must be PROVEN before a legacy registration is claimed.
#
# QA hit Q1 for real: from a non-default config root, `uninstall` deleted the
# operator's live default-root LaunchAgent (restored byte-for-byte). On a user's
# machine that takes RunAtLoad/KeepAlive with it, so the bridge never returns
# after a reboot — while `uninstall` reports success.
# --------------------------------------------------------------------------


def _plant(home: Path, *, log_path: str | None) -> Path:
    """A default-named plist, optionally carrying ownership evidence."""
    (home / "Library" / "LaunchAgents").mkdir(parents=True, exist_ok=True)
    body: dict[str, object] = {"Label": install.LABEL}
    if log_path is not None:
        body["StandardOutPath"] = log_path
    target = home / "Library" / "LaunchAgents" / f"{install.LABEL}.plist"
    target.write_bytes(plistlib.dumps(body))
    return target


def test_a_non_default_root_never_claims_the_default_roots_registration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_root: Path
) -> None:
    """Q1. While the default root exists, the shared-name file may be its live
    install, so no other root may claim — or delete — it."""
    home = tmp_path / "home"
    (home / ".local-operator").mkdir(parents=True)  # the default root, present
    monkeypatch.setattr(install.sys, "platform", "darwin")
    monkeypatch.setattr(install.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(install, "_passwd_home", lambda: home)
    # Evidence that would otherwise satisfy ownership: still refused.
    live = _plant(home, log_path=str(install.log_path()))
    monkeypatch.setattr(
        install, "_launchctl", lambda *a: subprocess.CompletedProcess(list(a), 0, "", "")
    )

    assert install.legacy_registration() is None
    result = install.uninstall()
    assert live.exists(), "must never delete the default root's registration"
    assert result.get("warning"), "and must say what it found and left"


def test_a_registration_written_by_another_root_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_root: Path
) -> None:
    """Ownership is decided by recorded evidence, not by location."""
    home = tmp_path / "home"
    monkeypatch.setattr(install.sys, "platform", "darwin")
    monkeypatch.setattr(install.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(install, "_passwd_home", lambda: home)
    other = _plant(home, log_path="/some/other/root/logs/browser-bridge.log")
    monkeypatch.setattr(
        install, "_launchctl", lambda *a: subprocess.CompletedProcess(list(a), 0, "", "")
    )

    assert install.legacy_registration() is None
    install.uninstall()
    assert other.exists()
    reported = install.legacy_ambiguity()
    assert reported is not None and "different config root" in reported


def test_ownership_evidence_is_compared_canonically(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_root: Path
) -> None:
    """A symlinked or dot-segmented spelling of ONE path is the same path.

    Without canonicalisation a genuine owner is refused on macOS, where
    ``/tmp`` is a symlink to ``/private/tmp`` — the recorded string and the
    computed one then differ while naming one file.
    """
    home = tmp_path / "home"
    monkeypatch.setattr(install.sys, "platform", "darwin")
    monkeypatch.setattr(install.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(install, "_passwd_home", lambda: home)
    # A real symlink, since `Path` collapses `./` in its own constructor and
    # would make this pass without any canonicalisation at all.
    real_logs = install.log_path().parent
    real_logs.mkdir(parents=True, exist_ok=True)
    link = tmp_path / "logs-via-symlink"
    link.symlink_to(real_logs, target_is_directory=True)
    through_link = str(link / install.log_path().name)
    assert through_link != str(install.log_path())
    _plant(home, log_path=through_link)
    assert install.legacy_registration() is not None


def test_this_roots_own_registration_takes_precedence(
    inherited_darwin: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q2. With an own install present, uninstall removes ITS registration and
    leaves the other one alone — uninstalling root A must not take down root B."""
    install.plist_path().write_bytes(plistlib.dumps({"Label": install.label()}))
    monkeypatch.setattr(
        install, "_launchctl", lambda *a: subprocess.CompletedProcess(list(a), 0, "", "")
    )
    result = install.uninstall()
    assert inherited_darwin.exists(), "must not remove a registration it does not own"
    assert not install.plist_path().exists(), "must remove its own"
    assert result.get("warning"), "and must name what it left"


def test_an_unclaimable_registration_is_reported_not_silently_ignored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, isolated_root: Path
) -> None:
    """Unknown ownership must reach the user, never collapse into the
    'no LaunchAgent was installed' success that hides it."""
    home = tmp_path / "home"
    (home / ".local-operator").mkdir(parents=True)
    monkeypatch.setattr(install.sys, "platform", "darwin")
    monkeypatch.setattr(install.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(install, "_passwd_home", lambda: home)
    _plant(home, log_path=str(install.log_path()))
    monkeypatch.setattr(
        install, "_launchctl", lambda *a: subprocess.CompletedProcess(list(a), 0, "", "")
    )
    monkeypatch.setattr(install, "health", lambda *a, **k: None)

    warning = install.uninstall().get("warning")
    assert warning is not None and str(install.LABEL) in str(warning)
    assert install.status()["legacy_ambiguity"] is not None


def test_ambiguous_uninstall_advice_preserves_configuration_and_sessions(
    inherited_darwin: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A5: removing persistent user data is not a remedy for an ambiguous job.

    Exercise the printed CLI warning, not only the helper's return value. The
    ownership heuristic must still refuse, but its advice must concern only the
    named supervisor registration, never the retained config/session root.
    """
    from argparse import Namespace

    from local_operator.cli import browser_command

    default_root = install._default_config_root()
    default_root.mkdir()
    transcript = default_root / "sessions" / "synthetic" / "transcript.jsonl"
    transcript.parent.mkdir(parents=True)
    transcript.write_text('{"synthetic": true}\n', encoding="utf-8")
    original = inherited_darwin.read_bytes()
    monkeypatch.setattr(
        install, "_launchctl", lambda *a: subprocess.CompletedProcess(list(a), 0, "", "")
    )

    # Preserve the previously approved return contract; this change is advice
    # only, not a redesign of ambiguity handling.
    assert browser_command(Namespace(browser_command="uninstall", purge=False)) == 0
    output = capsys.readouterr().out
    assert str(inherited_darwin) in output
    assert "remove that config root" not in output
    assert "confirm ownership" in output
    assert "Keep all configuration and session data" in output
    assert inherited_darwin.read_bytes() == original
    assert transcript.read_text(encoding="utf-8") == '{"synthetic": true}\n'
