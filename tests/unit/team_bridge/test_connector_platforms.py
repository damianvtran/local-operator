"""The receiver on Linux (systemd --user) and Windows (Task Scheduler), plus the trigger.

The operator's directive: the Team Bridge receiver must work on Mac, Windows
and Linux, through every surface, under ONE contract. ``test_connector_autoinstall``
pins the macOS arm; this file pins the other two arms against the SAME
properties, through the supervisor-runner seams only:

* install writes the unit/task and starts it; a repeat is a no-op (``current``);
* a current registration the supervisor forgot is re-registered, not rewritten;
* Turn off (the wanted state flipping) removes it, through the same lock;
* the guards hold on every arm: an isolated (non-default) store is refused
  before anything is written, a source checkout never repoints the unit, a
  redirected home never reaches the real supervisor;
* Linux with no user manager answers a distinct code with the remedy, and the
  lingering outcome is READ back, never assumed;
* ``connector_state`` is truthful per arm;
* the cross-process lock and the connector singleton use the portable
  primitive (no ``fcntl`` import on the path);
* every surface's start converges on ``catch_up_at_surface_start`` →
  ``sync_connector_service``; ``lop team-bridge receiver`` is the same call.

No test reaches a real ``systemctl``, ``schtasks`` or ``launchctl``: the seams
are module attributes (``install._systemctl``/``_schtasks``/``_launchctl``),
and the package-wide guard in ``tests/unit/conftest.py`` makes the default a
hard failure. Windows-only behaviour that cannot be faked on POSIX (the
``msvcrt`` lock, ``pythonw.exe``) is asserted natively by the cells marked
``windows_native`` in the ``filesystem-boundaries-windows`` CI job.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator import launchd, procname, supervisors
from local_operator.team_bridge.connector import install
from local_operator.team_bridge.store import OwnerReceipt, OwnerSnapshot, Store

OWNER = OwnerSnapshot(
    credential_id=1,
    account_id="acc_1",
    tenant_id="tnt_1",
    endpoint_id="bep_1",
    canonical_base_url="https://hub.example/v1",
    enrollment_generation=1,
)


def _commit(root: Path) -> None:
    Store(root).commit_enrollment(OWNER, OwnerReceipt("bridge_consent_v1", "bep_1"))


def _set_enabled(root: Path, value: bool) -> None:
    """The real settings writer, as the explicit OFF/ON events use it."""

    from local_operator import settings_io
    from local_operator.config import ConfigManager

    settings_io.write_setting(ConfigManager(root), settings_io.BY_KEY["team_bridge.enabled"], value)


def _done(args: tuple[str, ...], code: int = 0, out: str = "", err: str = ""):
    return subprocess.CompletedProcess(list(args), code, out, err)


class FakeSystemctl:
    """One in-memory ``systemctl --user`` for the connector's unit.

    ``bus_down`` models a login with no user manager (a container, WSL without
    systemd, SSH without lingering): every call answers systemd's own bus
    failure text, which is what ``supervisors.is_bus_failure`` matches.
    """

    BUS = "Failed to connect to bus: No medium found"

    def __init__(self) -> None:
        self.enabled = False
        self.active = False
        self.bus_down = False
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *args: str) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        if self.bus_down:
            return _done(args, 1, "", self.BUS)
        verb = args[0]
        if verb == "is-active":
            return _done(args, 0 if self.active else 3, "active" if self.active else "inactive")
        if verb == "is-enabled":
            return _done(args, 0 if self.enabled else 1, "enabled" if self.enabled else "disabled")
        if verb == "enable":
            self.enabled = True
        elif verb in ("restart", "start"):
            self.active = True
        elif verb == "disable":
            self.enabled = False
            if "--now" in args:
                self.active = False
        return _done(args)


class FakeSchtasks:
    """One in-memory Task Scheduler task with the verbs the installer uses.

    The ``/Query`` body is the real ``/FO CSV`` shape ``supervisors.task_state``
    parses by POSITION (header row, then a row whose first field starts with
    the root backslash).
    """

    def __init__(self) -> None:
        self.registered = False
        self.running = False
        self.xml: str | None = None
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *args: str) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        verb = args[0]
        if verb == "/Create":
            xml_path = Path(args[args.index("/XML") + 1])
            self.xml = xml_path.read_text(encoding=supervisors.TASK_XML_ENCODING)
            self.registered = True
        elif verb == "/Query":
            if not self.registered:
                return _done(args, 1, "", "ERROR: The system cannot find the file specified.")
            status = "Running" if self.running else "Ready"
            body = '"TaskName","Next Run Time","Status"\n'
            body += f'"\\{install.TASK_NAME}","N/A","{status}"\n'
            return _done(args, 0, body)
        elif verb == "/Run":
            if not self.registered:
                return _done(args, 1, "", "ERROR: The system cannot find the file specified.")
            self.running = True
        elif verb == "/End":
            self.running = False
        elif verb == "/Delete":
            if not self.registered:
                return _done(args, 1, "", "ERROR: The system cannot find the file specified.")
            self.registered = False
            self.running = False
        return _done(args)


@pytest.fixture
def user_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A real-user-shaped home + default store, every machine-side effect contained."""

    home = tmp_path / "home"
    root = home / ".local-operator"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(procname, "supervised_image", lambda: Path("/stable/lop/bin/python3"))
    return home, root


@pytest.fixture
def linux(user_home, monkeypatch: pytest.MonkeyPatch) -> tuple[FakeSystemctl, Path, Path]:
    """The systemd arm with the ownership + addressability guards OPENED.

    Opened at the module attributes production reads, exactly as the macOS
    lifecycle cells open theirs; the isolation cells below leave them shut.
    """

    home, root = user_home
    fake = FakeSystemctl()
    monkeypatch.setattr(supervisors, "supervisor", lambda: supervisors.SYSTEMCTL)
    monkeypatch.setattr(install, "_ownership_refusal", lambda config_dir: None)
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda unit: True)
    # The unit render asks systemd's version for the output lines; pin "new".
    monkeypatch.setattr(supervisors, "systemd_version", lambda: 255)
    monkeypatch.setattr(install, "_systemctl", fake)
    linger = {"on": False, "enable_calls": 0}

    def enable() -> bool:
        linger["enable_calls"] += 1
        return False  # a distro whose polkit refuses self-linger

    monkeypatch.setattr(install, "_enable_linger", enable)
    monkeypatch.setattr(install, "_linger_enabled", lambda: linger["on"])
    fake.linger = linger  # type: ignore[attr-defined]
    return fake, home, root


@pytest.fixture
def windows(user_home, monkeypatch: pytest.MonkeyPatch) -> tuple[FakeSchtasks, Path, Path]:
    """The Task Scheduler arm with its guards opened; ``_schtasks`` is the fake."""

    home, root = user_home
    fake = FakeSchtasks()
    monkeypatch.setattr(install, "_is_windows", lambda: False)  # Simulated lifecycle.
    monkeypatch.setattr(supervisors, "supervisor", lambda: supervisors.SCHTASKS)
    monkeypatch.setattr(install, "_ownership_refusal", lambda config_dir: None)
    monkeypatch.setattr(supervisors, "task_scheduler_is_addressable", lambda config_dir: True)
    monkeypatch.setattr(install, "_schtasks", fake)
    return fake, home, root


def _unit(home: Path) -> Path:
    return home / ".config" / "systemd" / "user" / install.SYSTEMD_UNIT


# -- Linux: systemd --user ------------------------------------------------------


def test_linux_install_writes_enables_and_starts_the_unit(linux) -> None:
    fake, home, root = linux
    _commit(root)
    outcome = install.sync_connector_service(root)
    assert outcome.code == install.CODE_INSTALLED, outcome
    unit = _unit(home)
    assert unit.read_text(encoding="utf-8") == install.render_systemd(root)
    assert fake.enabled and fake.active
    verbs = [call[0] for call in fake.calls]
    assert verbs == ["daemon-reload", "enable", "restart"], fake.calls
    # The always-on contract, in systemd's words, naming the STABLE shim.
    text = unit.read_text(encoding="utf-8")
    assert "Restart=always" in text and "WantedBy=default.target" in text
    assert f'{supervisors.quoted(str(Path("/stable/lop/bin/python3")))} -m {install.MODULE}' in text
    assert f"--config-dir {supervisors.quoted(str(root))}" in text
    # Lingering was ATTEMPTED and its refusal is reported, not assumed.
    assert fake.linger["enable_calls"] == 1
    assert "lingering is off" in outcome.detail and "loginctl enable-linger" in outcome.detail
    assert install.connector_state(root) == install.STATE_LOADED


def test_linux_linger_that_took_is_reported_as_such(linux) -> None:
    fake, home, root = linux
    fake.linger["on"] = True
    _commit(root)
    outcome = install.sync_connector_service(root)
    assert outcome.code == install.CODE_INSTALLED
    assert "lingering is on" in outcome.detail


def test_linux_repeat_is_one_query_and_no_write(linux) -> None:
    fake, home, root = linux
    _commit(root)
    install.sync_connector_service(root)
    before = _unit(home).stat().st_mtime_ns
    fake.calls.clear()
    assert install.sync_connector_service(root).code == install.CODE_CURRENT
    assert _unit(home).stat().st_mtime_ns == before
    assert [call[0] for call in fake.calls] == ["is-active", "is-enabled"], fake.calls


def test_linux_stopped_unit_is_started_not_rewritten(linux) -> None:
    fake, home, root = linux
    _commit(root)
    install.sync_connector_service(root)
    fake.active = False  # the user stopped it, or the manager forgot it
    before = _unit(home).stat().st_mtime_ns
    assert install.sync_connector_service(root).code == install.CODE_REGISTERED
    assert _unit(home).stat().st_mtime_ns == before
    assert fake.active


def test_linux_turn_off_disables_and_removes_the_unit(linux) -> None:
    fake, home, root = linux
    _commit(root)
    install.sync_connector_service(root)
    _set_enabled(root, False)
    assert install.sync_connector_service(root).code == install.CODE_REMOVED
    assert not _unit(home).exists()
    assert not fake.active and not fake.enabled
    assert ("disable", "--now", install.SYSTEMD_UNIT) in fake.calls
    assert install.connector_state(root) == install.STATE_MISSING


def test_linux_with_no_user_manager_is_a_distinct_code_with_the_remedy(linux) -> None:
    fake, home, root = linux
    fake.bus_down = True
    _commit(root)
    outcome = install.sync_connector_service(root)
    assert outcome.code == install.CODE_NO_USER_MANAGER, outcome
    assert outcome.code != install.CODE_UNSUPPORTED
    assert "loginctl enable-linger" in outcome.detail
    assert "python -m local_operator.team_bridge.connector --config-dir" in outcome.detail
    # The unit file is written (it takes effect when a user manager exists);
    # the state says WHY it is not running rather than "missing".
    assert _unit(home).exists()
    assert install.connector_state(root) == install.CODE_NO_USER_MANAGER


def test_linux_redirected_home_never_addresses_the_user_manager(linux, monkeypatch) -> None:
    fake, home, root = linux
    monkeypatch.setattr(supervisors, "systemd_unit_is_addressable", lambda unit: False)
    _commit(root)
    assert install.sync_connector_service(root).code == install.CODE_NOT_ADDRESSABLE
    assert fake.calls == [], "a redirected home reached systemctl"
    assert install.connector_state(root) == install.STATE_INSTALLED


def test_linux_source_checkout_never_writes_the_unit(linux, monkeypatch) -> None:
    fake, home, root = linux
    from local_operator import update

    monkeypatch.setattr(procname, "supervised_image", lambda: None)
    monkeypatch.setattr(update, "install_kind", lambda: update.InstallKind.EDITABLE)
    _commit(root)
    assert install.sync_connector_service(root).code == install.CODE_SOURCE_CHECKOUT
    assert not _unit(home).exists() and fake.calls == []


def test_a_source_checkout_can_still_turn_the_receiver_off(linux, monkeypatch) -> None:
    """The checkout guard refuses to INSTALL; it must never strand a removal.

    A developer whose install became a checkout after the receiver was placed
    (or who runs from a worktree) has to be able to turn agent messages off.
    Refusing ``remove`` with the install refusal is the "Turn off does not turn
    it off" failure this path exists to prevent.
    """

    fake, home, root = linux
    from local_operator import update

    _commit(root)
    install.sync_connector_service(root)
    assert _unit(home).exists() and fake.active
    monkeypatch.setattr(procname, "supervised_image", lambda: None)
    monkeypatch.setattr(update, "install_kind", lambda: update.InstallKind.EDITABLE)
    assert install.remove_connector_service(root).code == install.CODE_REMOVED
    assert not _unit(home).exists() and not fake.active
    _set_enabled(root, False)
    assert install.sync_connector_service(root).code != install.CODE_SOURCE_CHECKOUT


def test_linux_isolated_store_is_refused_before_anything(tmp_path, user_home, monkeypatch) -> None:
    """Guards SHUT: the real ``launchd.shared_label_refusal`` decides."""

    home, _root = user_home
    fake = FakeSystemctl()
    monkeypatch.setattr(supervisors, "supervisor", lambda: supervisors.SYSTEMCTL)
    monkeypatch.setattr(install, "_systemctl", fake)
    scratch = tmp_path / "qa-scratch-store"
    _commit(scratch)
    assert install.sync_connector_service(scratch).code == install.CODE_ISOLATED
    assert install.remove_connector_service(scratch).code == install.CODE_ISOLATED
    assert install.connector_state(scratch) == install.CODE_ISOLATED
    assert not _unit(home).exists() and fake.calls == []


def test_linux_turn_off_racing_the_catch_up_wins(linux, monkeypatch) -> None:
    """R2-1 on this arm: the wanted state is decided again INSIDE the lock."""

    fake, home, root = linux
    _commit(root)
    answers = iter([True, False])
    monkeypatch.setattr(install, "_wanted", lambda _c: next(answers, False))
    assert install.sync_connector_service(root).code == install.CODE_ABSENT
    assert not _unit(home).exists() and not fake.active


# -- Windows: Task Scheduler ------------------------------------------------------


def test_windows_install_registers_and_runs_the_logon_task(windows, monkeypatch) -> None:
    fake, home, root = windows
    monkeypatch.setenv("USERNAME", "ada")
    monkeypatch.setenv("USERDOMAIN", "LAB")
    _commit(root)
    outcome = install.sync_connector_service(root)
    assert outcome.code == install.CODE_INSTALLED, outcome
    assert fake.registered and fake.running
    xml = fake.xml or ""
    # Logon trigger for THIS user, least privilege, interactive token (no
    # stored password, no elevation), restart on failure, one instance.
    assert "<LogonTrigger>" in xml and "<UserId>LAB\\ada</UserId>" in xml
    assert "<RunLevel>LeastPrivilege</RunLevel>" in xml
    assert "<LogonType>InteractiveToken</LogonType>" in xml
    assert "<RestartOnFailure>" in xml
    assert "<MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>" in xml
    assert "<ExecutionTimeLimit>PT0S</ExecutionTimeLimit>" in xml
    # The interpreter runs directly — no cmd.exe wrapper, so no console host.
    assert "cmd.exe" not in xml
    assert "-m local_operator.team_bridge.connector --config-dir" in xml
    # Registered without /RU: the task belongs to the creating user.
    create = next(call for call in fake.calls if call[0] == "/Create")
    assert "/RU" not in create and "/F" in create
    assert install.task_record_path(root).read_text(encoding="utf-8") == xml
    assert install.connector_state(root) == install.STATE_LOADED


def test_windows_repeat_is_one_query(windows) -> None:
    fake, home, root = windows
    _commit(root)
    install.sync_connector_service(root)
    fake.calls.clear()
    assert install.sync_connector_service(root).code == install.CODE_CURRENT
    assert [call[0] for call in fake.calls] == ["/Query"], fake.calls


def test_windows_idle_task_is_run_not_reregistered(windows) -> None:
    fake, home, root = windows
    _commit(root)
    install.sync_connector_service(root)
    fake.running = False
    fake.calls.clear()
    assert install.sync_connector_service(root).code == install.CODE_REGISTERED
    assert [call[0] for call in fake.calls] == ["/Query", "/Run"], fake.calls


def test_windows_turn_off_ends_and_deletes_the_task(windows) -> None:
    fake, home, root = windows
    _commit(root)
    install.sync_connector_service(root)
    _set_enabled(root, False)
    assert install.sync_connector_service(root).code == install.CODE_REMOVED
    assert not fake.registered and not fake.running
    verbs = [call[0] for call in fake.calls]
    assert verbs.index("/End") < verbs.index("/Delete"), fake.calls
    assert not install.task_record_path(root).exists()
    assert install.connector_state(root) == install.STATE_MISSING


def test_windows_a_failed_create_leaves_no_record(windows, monkeypatch) -> None:
    fake, home, root = windows
    _commit(root)

    def refusing(*args: str):
        if args[0] == "/Create":
            return _done(args, 1, "", "ERROR: Access is denied.")
        return fake(*args)

    monkeypatch.setattr(install, "_schtasks", refusing)
    outcome = install.sync_connector_service(root)
    assert outcome.code == install.CODE_FAILED and "Access is denied" in outcome.detail
    # No record means the next sync retries instead of reading "current".
    assert not install.task_record_path(root).exists()


def test_windows_outside_the_real_profile_never_reaches_schtasks(windows, monkeypatch) -> None:
    fake, home, root = windows
    monkeypatch.setattr(supervisors, "task_scheduler_is_addressable", lambda config_dir: False)
    _commit(root)
    assert install.sync_connector_service(root).code == install.CODE_NOT_ADDRESSABLE
    assert fake.calls == []


def test_windows_ownership_uses_the_profile_not_passwd(tmp_path, monkeypatch) -> None:
    """``launchd.real_home`` reads passwd, which Windows lacks; this arm must not.

    Otherwise the rule refuses EVERY store on Windows — the default included —
    and the receiver could never be placed there. Simulated through the
    ``_is_windows`` seam and ``supervisors.real_home`` because the identity
    comparison is pure.
    """

    profile = tmp_path / "Users" / "ada"
    (profile / ".local-operator").mkdir(parents=True)
    monkeypatch.setattr(install, "_is_windows", lambda: True)
    monkeypatch.setattr(supervisors, "windows_profile_home", lambda: profile)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: profile))
    monkeypatch.setenv("HOME", str(profile))
    assert install._ownership_refusal(profile / ".local-operator") is None
    refused = install._ownership_refusal(tmp_path / "scratch")
    assert refused is not None and "not the default one" in refused
    # ``$HOME`` is IGNORED on Windows, deliberately: Git Bash, MSYS2 and Cygwin
    # export it as a POSIX path that means nothing to this product, and refusing
    # on it denied the receiver to every developer in those shells (review round
    # 2 fix #1). The identity test is the PROFILE, which is ``Path.home()``.
    monkeypatch.setenv("HOME", str(tmp_path / "git-bash-style-home"))
    assert install._ownership_refusal(profile / ".local-operator") is None
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "redirected-profile"))
    assert "redirected" in (install._ownership_refusal(profile / ".local-operator") or "")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: profile))
    monkeypatch.setattr(supervisors, "windows_profile_home", lambda: None)
    assert install._ownership_refusal(profile / ".local-operator") is not None


def test_windowless_interpreter_is_chosen_only_when_it_exists(tmp_path) -> None:
    scripts = tmp_path / "Scripts"
    scripts.mkdir()
    python = scripts / "python.exe"
    python.write_bytes(b"")
    assert install._windowless(str(python)) == str(python)
    (scripts / "pythonw.exe").write_bytes(b"")
    assert install._windowless(str(python)) == str(scripts / "pythonw.exe")
    assert install._windowless("/usr/bin/python3") == "/usr/bin/python3"


# -- shared: guards, lock, state ---------------------------------------------------


@pytest.mark.parametrize("kind", [supervisors.SYSTEMCTL, supervisors.SCHTASKS])
def test_an_unenrolled_machine_is_absent_with_no_artifact(user_home, monkeypatch, kind) -> None:
    home, root = user_home
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(supervisors, "supervisor", lambda: kind)
    monkeypatch.setattr(launchd, "shared_label_refusal", lambda config_dir: None)
    monkeypatch.setattr(install, "_ownership_refusal", lambda config_dir: None)
    monkeypatch.setattr(install, "_systemctl", lambda *a: calls.append(a))
    monkeypatch.setattr(install, "_schtasks", lambda *a: calls.append(a))
    assert install.sync_connector_service(root).code == install.CODE_ABSENT
    assert not (root / "team_bridge").exists() and calls == []


def test_remove_forces_off_without_changing_the_wanted_state(linux) -> None:
    fake, home, root = linux
    _commit(root)
    install.sync_connector_service(root)
    assert install.remove_connector_service(root).code == install.CODE_REMOVED
    assert not _unit(home).exists()
    # The wanted state is untouched: the next sync places it again.
    assert install._wanted(root) is True
    assert install.sync_connector_service(root).code == install.CODE_INSTALLED


def test_the_sync_lock_and_singleton_never_import_fcntl(tmp_path, monkeypatch) -> None:
    """Both locks go through the portable primitive (``wakes.lock``).

    ``fcntl`` does not exist on Windows; the first cut imported it directly in
    ``_sync_exclusive`` (a silent no-op there) and in ``ConnectorLock`` (a
    crash). Made unimportable here, both still take and release their lock.
    """

    from local_operator.team_bridge.connector import lock as lock_mod
    from local_operator.wakes import lock as wakes_lock

    real_try, real_unlock = wakes_lock._try_lock, wakes_lock._unlock
    seen: list[str] = []

    def tracking_try(fd: int) -> bool:
        seen.append("try")
        return real_try(fd)

    monkeypatch.setattr(wakes_lock, "_try_lock", tracking_try)
    monkeypatch.setattr(lock_mod, "_try_lock", tracking_try)
    monkeypatch.setattr(lock_mod, "_unlock", real_unlock)
    import builtins

    original_import = builtins.__import__

    def no_fcntl(name: str, *args: Any, **kwargs: Any):
        if name == "fcntl":
            raise AssertionError("the connector lock path imported fcntl")
        return original_import(name, *args, **kwargs)

    # The primitive itself imports fcntl on POSIX; that is the portable
    # helper's business. What must not happen is THIS module importing it.
    monkeypatch.setattr(install, "_SYNC_FILE_POLL_S", 0.01)
    with install._sync_exclusive(tmp_path):
        pass
    assert seen, "the sync lock did not use the portable primitive"
    seen.clear()
    with lock_mod.ConnectorLock(tmp_path, deadline_s=0.2):
        pass
    assert seen, "the connector singleton did not use the portable primitive"
    for module in (install, lock_mod):
        source = Path(module.__file__ or "").read_text(encoding="utf-8")
        assert "import fcntl" not in source, f"{module.__name__} imports fcntl directly"
    assert no_fcntl  # the import trap documents the property the source check pins


def test_the_sync_file_lock_serialises_two_holders(tmp_path, monkeypatch) -> None:
    """The cross-process half really excludes: a second holder waits it out."""

    from local_operator.procstate import O_BINARY
    from local_operator.wakes.lock import _try_lock, _unlock

    lock_path = tmp_path / "team_bridge" / "connector-service.lock"
    lock_path.parent.mkdir(parents=True)
    fd = os.open(lock_path, O_BINARY | os.O_CREAT | os.O_RDWR, 0o600)
    assert _try_lock(fd)
    monkeypatch.setattr(install, "_SYNC_FILE_LOCK_TIMEOUT_S", 0.3)
    monkeypatch.setattr(install, "_SYNC_FILE_POLL_S", 0.02)
    import time

    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="file lock is busy"):
            with install._sync_exclusive(tmp_path):
                pytest.fail("sync mutated while another process held its lock")
        waited = time.monotonic() - started
    finally:
        _unlock(fd)
        os.close(fd)
    with install._sync_exclusive(tmp_path):
        pass  # Both locks were released by the failed attempt.
    # The holder above is THIS process; flock is per open file description on
    # POSIX and msvcrt per handle on Windows, so the second open still contends.
    assert waited >= 0.25, f"the file lock did not wait for its holder ({waited:.2f}s)"


def test_no_path_returns_the_retired_unsupported_code(user_home, monkeypatch) -> None:
    home, root = user_home
    _commit(root)
    for kind in (None, supervisors.SYSTEMCTL, supervisors.SCHTASKS, supervisors.LAUNCHCTL):
        monkeypatch.setattr(supervisors, "supervisor", lambda kind=kind: kind)
        assert install.sync_connector_service(root).code != install.CODE_UNSUPPORTED
        assert install.connector_state(root) != install.CODE_UNSUPPORTED


@pytest.mark.skipif(os.name != "nt", reason="native Windows only (filesystem-boundaries-windows)")
def test_windows_native_connector_lock_excludes_a_second_holder(tmp_path) -> None:
    """On real Windows the singleton takes an ``msvcrt`` lock and refuses a second."""

    from local_operator.team_bridge.connector.lock import ConnectorLock, ConnectorLockBusy

    first = ConnectorLock(tmp_path, deadline_s=0.3)
    first.acquire()
    try:
        with pytest.raises(ConnectorLockBusy):
            ConnectorLock(tmp_path, deadline_s=0.3).acquire()
    finally:
        first.release()
    with ConnectorLock(tmp_path, deadline_s=0.3):
        pass


@pytest.mark.skipif(os.name != "nt", reason="native Windows only (filesystem-boundaries-windows)")
def test_windows_native_task_render_names_a_windowless_interpreter(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(procname, "supervised_image", lambda: None)
    xml = install.render_task_xml(tmp_path / ".local-operator")
    expected = Path(sys.executable).with_name("pythonw.exe")
    if expected.exists():
        assert "pythonw.exe" in xml
    assert "cmd.exe" not in xml


def test_windows_stop_handler_turns_sigbreak_into_a_clean_stop(monkeypatch) -> None:
    """``__main__``: SIGBREAK raises KeyboardInterrupt (the Ctrl+C path)."""

    import signal

    from local_operator.team_bridge.connector import __main__ as entry

    installed: dict[int, Any] = {}
    monkeypatch.setattr(entry.os if hasattr(entry, "os") else os, "name", "nt", raising=False)
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(signal, "SIGBREAK", 21, raising=False)
    monkeypatch.setattr(signal, "signal", lambda sig, handler: installed.__setitem__(sig, handler))
    assert entry._install_windows_stop() is True
    with pytest.raises(KeyboardInterrupt):
        installed[21](21, None)


def test_posix_installs_no_windows_stop_handler(monkeypatch) -> None:
    from local_operator.team_bridge.connector import __main__ as entry

    monkeypatch.setattr(os, "name", "posix")
    assert entry._install_windows_stop() is False


# -- the trigger: every surface converges on one function -------------------------


async def _settle(coro):
    return await coro


def test_every_surface_start_calls_the_one_catch_up(tmp_path, monkeypatch) -> None:
    """``catch_up_at_surface_start`` runs ``sync_connector_service`` off the loop.

    And never raises: a surface boots whether or not the receiver was placed.
    """

    seen: list[tuple[str, str]] = []

    def fake_sync(root: Path):
        import threading

        seen.append((str(root), threading.current_thread().name))
        return install.ServiceOutcome(install.CODE_CURRENT)

    monkeypatch.setattr(install, "sync_connector_service", fake_sync)
    outcome = asyncio.run(install.catch_up_at_surface_start(tmp_path, surface="tui"))
    assert outcome.code == install.CODE_CURRENT
    assert seen and seen[0][0] == str(tmp_path)
    assert seen[0][1] != "MainThread", "the catch-up ran on the event loop's thread"

    def broken(root: Path):
        raise RuntimeError("supervisor exploded")

    monkeypatch.setattr(install, "sync_connector_service", broken)
    outcome = asyncio.run(install.catch_up_at_surface_start(tmp_path, surface="mobile"))
    assert outcome.code == install.CODE_FAILED


@pytest.mark.parametrize(
    ("path", "needle"),
    [
        ("local_operator/server/app.py", 'catch_up_at_surface_start(config_dir, surface="serve")'),
        ("local_operator/tui/app.py", 'catch_up_at_surface_start(config_dir(), surface="tui")'),
        (
            "local_operator/mobile/service.py",
            'catch_up_at_surface_start(_config_dir(), surface="mobile")',
        ),
    ],
)
def test_each_surface_entry_names_the_shared_catch_up(path: str, needle: str) -> None:
    """A STRUCTURAL pin: each surface's start reaches the one helper.

    Behavioural cells cover the serve lifespan (``tests/unit/server``) and the
    helper itself (above); this row catches a surface that drifts to a private
    copy of the sync, which is the one-function rule the operator set.
    """

    root = Path(__file__).resolve().parents[3]
    assert needle in (root / path).read_text(encoding="utf-8"), f"{path} lost the catch-up"


def test_the_desktop_app_start_boundary_reaches_the_same_sync() -> None:
    """The desktop's own trigger is ``StatusDiscipline``'s APP_START branch."""

    root = Path(__file__).resolve().parents[3]
    client = (root / "local_operator/team_bridge/client.py").read_text(encoding="utf-8")
    assert "Boundary.parse(boundary) is Boundary.APP_START" in client
    assert "await asyncio.to_thread(sync_connector_service, self._config_dir)" in client


# -- the CLI ----------------------------------------------------------------------


def _cli(argv: list[str]) -> Any:
    from local_operator.cli import build_cli_parser

    return build_cli_parser().parse_args(argv)


def test_cli_install_is_the_shared_sync_and_never_enrols(linux, capsys, monkeypatch) -> None:
    """Not enrolled: nothing installed, nothing enrolled, and the way forward said."""

    from local_operator.team_bridge.connector import cli as tb_cli

    fake, home, root = linux
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    code = tb_cli.main(_cli(["team-bridge", "receiver", "install"]))
    out = capsys.readouterr().out
    assert code == 1
    # The way forward is the APP, not a terminal (round 4: consent is UI-only),
    # so the hint names the app and then this group's own readiness check. The
    # old copy told the user to run a terminal consent command, which no longer
    # exists.
    assert "nothing was installed" in out
    assert "enable Agent messages" in out and "lop team-bridge setup" in out
    assert "/login radient" not in out
    assert not _unit(home).exists() and fake.calls == []
    assert Store(root).active_owner_readonly() is None, "install enrolled silently"


def test_cli_install_status_remove_round_trip(linux, capsys, monkeypatch) -> None:
    from local_operator.team_bridge.connector import cli as tb_cli

    fake, home, root = linux
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _commit(root)
    assert tb_cli.main(_cli(["team-bridge", "receiver", "install"])) == 0
    assert "installed and running" in capsys.readouterr().out
    assert tb_cli.main(_cli(["team-bridge", "receiver", "status", "--json"])) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["state"] == install.STATE_LOADED
    assert status["supervisor"] == supervisors.SYSTEMCTL
    assert status["enrolment"] == "on"
    assert status["log"].endswith("bridge-connector.log")
    assert status["linger"] is False
    assert tb_cli.main(_cli(["team-bridge", "receiver", "status"])) == 0
    human = capsys.readouterr().out
    assert "Receiver: running" in human and "systemd --user" in human
    assert "loginctl enable-linger" in human
    assert tb_cli.main(_cli(["team-bridge", "receiver", "remove"])) == 0
    removed = capsys.readouterr().out
    assert "stopped and removed" in removed and "team_bridge.enabled false" in removed
    assert not _unit(home).exists()


def test_cli_refuses_an_isolated_root_with_the_way_to_run_it(tmp_path, capsys, monkeypatch) -> None:
    from local_operator.team_bridge.connector import cli as tb_cli

    scratch = tmp_path / "scratch"
    _commit(scratch)
    calls: list[Any] = []
    monkeypatch.setattr(supervisors, "supervisor", lambda: supervisors.SYSTEMCTL)
    monkeypatch.setattr(install, "_systemctl", lambda *a: calls.append(a))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(scratch))
    assert tb_cli.main(_cli(["team-bridge", "receiver", "install"])) == 1
    out = capsys.readouterr().out
    assert "not your default one" in out and str(scratch) in out
    assert "python -m local_operator.team_bridge.connector --config-dir" in out
    assert calls == []


def test_cli_logs_reads_the_receivers_own_log(tmp_path, capsys, monkeypatch) -> None:
    from local_operator.team_bridge.connector import cli as tb_cli

    root = tmp_path / "cfg"
    log = install.log_path(root)
    log.parent.mkdir(parents=True)
    log.write_text("\n".join(f"line {n}" for n in range(80)) + "\n", encoding="utf-8")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setattr(supervisors, "supervisor", lambda: supervisors.LAUNCHCTL)
    assert tb_cli.main(_cli(["team-bridge", "receiver", "logs", "--lines", "3"])) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[-3:] == ["line 77", "line 78", "line 79"]


def test_linux_active_but_disabled_is_reenabled(linux):
    fake, home, root = linux
    _commit(root)
    install.sync_connector_service(root)
    fake.enabled = False
    assert install.sync_connector_service(root).code == install.CODE_REGISTERED
    assert fake.enabled and fake.active


def test_linux_failed_reload_is_retried_even_when_disk_matches(linux, monkeypatch):
    fake, home, root = linux
    _commit(root)

    def fail_reload(*args):
        if args[0] == "daemon-reload":
            return _done(args, 1, err="permission denied")
        return fake(*args)

    monkeypatch.setattr(install, "_systemctl", fail_reload)
    assert install.sync_connector_service(root).code == install.CODE_FAILED
    assert _unit(home).exists()
    fake.active = True  # An older definition may still be serving.
    monkeypatch.setattr(install, "_systemctl", fake)
    assert install.sync_connector_service(root).code == install.CODE_REGISTERED
    assert ("daemon-reload",) in fake.calls


def test_linux_failed_stop_keeps_registration(linux, monkeypatch):
    fake, home, root = linux
    _commit(root)
    install.sync_connector_service(root)

    def fail_stop(*args):
        if args[0] == "disable":
            return _done(args, 1, err="access denied")
        return fake(*args)

    monkeypatch.setattr(install, "_systemctl", fail_stop)
    assert install.remove_connector_service(root).code == install.CODE_FAILED
    assert _unit(home).exists() and fake.active


def test_windows_failed_end_does_not_delete_or_replace_task(windows, monkeypatch):
    fake, home, root = windows
    _commit(root)
    install.sync_connector_service(root)
    original = install.task_record_path(root).read_text()
    fake.calls.clear()

    def fail_end(*args):
        if args[0] == "/End":
            return _done(args, 1, err="access denied")
        return fake(*args)

    monkeypatch.setattr(install, "_schtasks", fail_end)
    assert install.remove_connector_service(root).code == install.CODE_FAILED
    assert fake.running and fake.registered
    assert install.task_record_path(root).read_text() == original
    monkeypatch.setattr(install, "render_task_xml", lambda root: original + "<!--changed-->")
    assert install.sync_connector_service(root).code == install.CODE_FAILED
    assert not any(c[0] in ("/Delete", "/Create") for c in fake.calls)


def test_windows_unknown_localized_status_is_not_claimed_running(windows):
    fake, home, root = windows
    _commit(root)
    install.sync_connector_service(root)
    fake.localized = True

    # Patch only the query wire, preserving the installed record.
    def localized(*args):
        return _done(args, out='"TaskName","Next Run Time","Status"\n"receiver","N/A","未知"\n')

    from unittest.mock import patch

    with patch.object(install, "_schtasks", localized):
        assert install.connector_state(root) == install.CODE_UNKNOWN


@pytest.mark.skipif(os.name != "nt", reason="native Windows profile API")
def test_windows_native_redirected_profile_never_reaches_scheduler(tmp_path, monkeypatch):
    real = supervisors.windows_profile_home()
    assert real is not None
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert supervisors.windows_profile_home() == real
    assert install._ownership_refusal(real / ".local-operator") is not None


def test_task_hidden_and_per_user(windows, monkeypatch, tmp_path):
    fake, home, root = windows
    assert "<Hidden>true</Hidden>" in install.render_task_xml(root)
    first = install.task_name()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "another-user"))
    assert first != install.task_name()
