"""F4 + F5 + the staging-swap race, pinned on POSIX.

WHY THESE LIVE APART FROM ``test_operator_authority.py``: each test here pins one
of the three trust-anchor holes the pairing-presence architecture review found
(F4: the Windows anchor was env-located and unchecked; F5: a device-admitting
write could take a stored admin password; the race: root installed a
user-writable staged file it never re-checked). The Windows half is driven
through the ``trust._on_windows`` seam with FAKE security descriptors so the
Ubuntu suite proves the verifier's logic; ``test_anchor_windows_native.py`` runs
the same verifier over real ACLs on windows-latest.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_operator.operator import OperatorAnchor, handlers, trust
from local_operator.operator.trust import WindowsAce, WindowsSecurity, anchor_bytes
from local_operator.operator.verify import key_id_for

SYSTEM, ADMINS, USERS = "S-1-5-18", "S-1-5-32-544", "S-1-5-32-545"
EVERYONE, AUTH_USERS = "S-1-1-0", "S-1-5-11"
ME = "S-1-5-21-1111111111-2222222222-3333333333-1001"
ALLOW, DENY = 0, 1
FULL, READ = 0x001F01FF, 0x001200A9
WRITE_DAC = 0x00040000


def _anchor() -> OperatorAnchor:
    spki = b"\x04" + bytes(range(64))
    return OperatorAnchor(key_id=key_id_for(spki), spki=spki, backend="file-only", presence=False)


def _with(*extra: WindowsAce, owner: str = ADMINS) -> WindowsSecurity:
    """The restrictive descriptor plus ``extra`` ACEs."""
    return WindowsSecurity(owner=owner, aces=(*(_restrictive().aces or ()), *extra))


def _restrictive() -> WindowsSecurity:
    """What the elevated install leaves: Administrators own; SYSTEM/Admins full; Users read."""
    return WindowsSecurity(
        owner=ADMINS,
        aces=(
            WindowsAce(ALLOW, 0, FULL, SYSTEM),
            WindowsAce(ALLOW, 0, FULL, ADMINS),
            WindowsAce(ALLOW, 0, READ, USERS),
        ),
    )


# ---------------------------------------------------------------------------
# F4, problem 1: the Windows root and account come from the OS, never the env
# ---------------------------------------------------------------------------


def test_windows_anchor_path_ignores_programdata_and_username(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hostile ``PROGRAMDATA``/``USERNAME`` must not move the Windows anchor."""
    known = tmp_path / "KnownFolder" / "ProgramData"
    monkeypatch.setattr(trust, "_on_windows", lambda: True, raising=False)
    monkeypatch.setattr(trust, "_windows_program_data", lambda: known, raising=False)
    monkeypatch.setattr(trust, "_windows_account_name", lambda: "alice", raising=False)
    monkeypatch.setattr(trust.os, "name", "nt")
    monkeypatch.delattr(trust.os, "getuid", raising=False)
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "attacker"))
    monkeypatch.setenv("USERNAME", "mallory")
    monkeypatch.setenv("USER", "mallory")

    path = trust.anchor_path()

    assert path == known / "local-operator" / "operators" / "alice.json"
    assert "attacker" not in str(path) and "mallory" not in str(path)


def test_windows_root_unresolved_fails_closed_not_to_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unresolvable Known Folder reads a path that never exists, not ``%PROGRAMDATA%``."""
    monkeypatch.setattr(trust, "_on_windows", lambda: True)
    monkeypatch.setattr(trust, "_windows_program_data", lambda: None)
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path))
    assert trust.anchor_dir() == trust._UNRESOLVED_ROOT


# ---------------------------------------------------------------------------
# F4, problem 2: the DACL verifier, over fake descriptors
# ---------------------------------------------------------------------------


def test_admin_owned_restrictive_acl_is_accepted() -> None:
    assert trust.windows_acl_refusal("anchor", _restrictive()) is None


@pytest.mark.parametrize(
    ("security", "needle"),
    [
        pytest.param(
            _with(WindowsAce(ALLOW, 0, 0x2, USERS)),
            USERS,
            id="users-writable-ace",
        ),
        pytest.param(
            WindowsSecurity(owner=ME, aces=_restrictive().aces),
            "not SYSTEM or Administrators",
            id="non-admin-owner",
        ),
        pytest.param(
            WindowsSecurity(
                owner=ADMINS,
                aces=(*(_restrictive().aces or ()), WindowsAce(ALLOW, 0, WRITE_DAC, EVERYONE)),
            ),
            EVERYONE,
            id="everyone-write-dac",
        ),
        pytest.param(
            WindowsSecurity(
                owner=ADMINS, aces=(*(_restrictive().aces or ()), WindowsAce(ALLOW, 0x10, FULL, ME))
            ),
            ME,
            id="current-user-full-inherited",
        ),
        pytest.param(
            WindowsSecurity(
                owner=ADMINS,
                aces=(*(_restrictive().aces or ()), WindowsAce(ALLOW, 0, 0x40000000, AUTH_USERS)),
            ),
            AUTH_USERS,
            id="authenticated-users-generic-write",
        ),
        pytest.param(WindowsSecurity(owner=ADMINS, aces=None), "no access list", id="null-dacl"),
        pytest.param(
            WindowsSecurity(
                owner=ADMINS, aces=(*(_restrictive().aces or ()), WindowsAce(5, 0, FULL, ""))
            ),
            "kind the runtime does not read",
            id="unknown-ace-type",
        ),
    ],
)
def test_unsafe_windows_acl_is_refused(security: WindowsSecurity, needle: str) -> None:
    refusal = trust.windows_acl_refusal("anchor", security)
    assert refusal is not None and needle in refusal, refusal


def test_inherit_only_and_deny_aces_grant_nothing_here() -> None:
    """``CREATOR OWNER``'s inherit-only template and a deny ACE are not grants on this object."""
    security = WindowsSecurity(
        owner=SYSTEM,
        aces=(
            *(_restrictive().aces or ()),
            WindowsAce(ALLOW, 0x08 | 0x01 | 0x02, FULL, "S-1-3-0"),
            WindowsAce(DENY, 0, FULL, EVERYONE),
        ),
    )
    assert trust.windows_acl_refusal("anchor", security) is None


def _windows_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, by_target: dict[str, WindowsSecurity]
) -> trust.AnchorLoad:
    """``load_anchor`` with the Windows branch on and ``_windows_security`` faked per object."""
    root = tmp_path / "local-operator" / "operators"
    root.mkdir(parents=True)
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", root)
    path = trust.anchor_path("alice")
    path.write_bytes(anchor_bytes(_anchor()))
    monkeypatch.setattr(trust, "_on_windows", lambda: True)

    def fake(target: Path | int) -> WindowsSecurity | None:
        key = "file" if isinstance(target, int) else Path(target).name
        return by_target.get(key, _restrictive())

    monkeypatch.setattr(trust, "_windows_security", fake)
    return trust.load_anchor("alice")


def test_load_anchor_on_windows_accepts_a_hardened_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = _windows_load(tmp_path, monkeypatch, {})
    assert loaded.usable, loaded.reason
    assert loaded.root_owned


@pytest.mark.parametrize("which", ["file", "operators", "local-operator"])
def test_load_anchor_on_windows_refuses_a_user_writable_link_in_the_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    """The file AND both directories lop owns are checked; any writable one refuses."""
    writable = WindowsSecurity(owner=ME, aces=(WindowsAce(ALLOW, 0, FULL, ME),))
    loaded = _windows_load(tmp_path, monkeypatch, {which: writable})
    assert not loaded.usable
    assert not loaded.root_owned
    assert "not SYSTEM or Administrators" in loaded.reason, loaded.reason


def test_load_anchor_on_windows_refuses_an_unreadable_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "local-operator" / "operators"
    root.mkdir(parents=True)
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", root)
    trust.anchor_path("alice").write_bytes(anchor_bytes(_anchor()))
    monkeypatch.setattr(trust, "_on_windows", lambda: True)
    monkeypatch.setattr(trust, "_windows_security", lambda target: None)
    loaded = trust.load_anchor("alice")
    assert not loaded.usable and "could not be read" in loaded.reason


# ---------------------------------------------------------------------------
# F4, problem 3: the Windows install is elevated and sets a restrictive DACL
# ---------------------------------------------------------------------------


def _inner_script(command: list[list[str]]) -> str:
    """The elevated script a printed Windows command carries (gzip+base64 blob)."""
    outer = command[0][-1]
    blob = outer.split("FromBase64String('", 1)[1].split("'", 1)[0]
    return gzip.decompress(base64.b64decode(blob)).decode("utf-8")


def test_windows_install_command_is_elevated_and_hardens_the_dacl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(trust, "_on_windows", lambda: True)
    staged = Path("C:/Users/alice/.local-operator/operator/anchor.json")
    target = Path("C:/ProgramData/local-operator/operators/alice.json")
    digest = "ab" * 32

    command = trust.install_commands(staged, target, digest)

    assert len(command) == 1
    outer = " ".join(command[0])
    assert "-Verb RunAs" in outer and "Copy-Item" not in outer
    assert ".ExitCode" in outer, "the elevated child's exit code must propagate"
    script = _inner_script(command)
    # R1-2: the DACL is REPLACED whole (protected, SYSTEM/Admins full, Users read)
    # through a handle opened without following reparse points; no icacls.
    assert "icacls" not in script
    assert "O:BAD:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;0x1200a9;;;BU)" in script
    assert "O:BAD:P(A;;FA;;;SY)(A;;FA;;;BA)(A;;0x1200a9;;;BU)" in script
    assert "0x02200000" in script, "FILE_FLAG_OPEN_REPARSE_POINT | BACKUP_SEMANTICS"
    assert "is a junction or symbolic link, so it was refused" in script
    assert "not SYSTEM or Administrators, so it was refused" in script
    assert "Add-Type" not in script, "Add-Type compiles through the user's writable %TEMP%"
    assert f"$expected='{digest}'" in script, "the elevated step re-checks the digest"
    assert "if($actual -ne $expected)" in script


@pytest.mark.parametrize(
    "staged",
    [
        "C:/Users/o'brien/.local-operator/operator/anchor.json",
        "C:/Users/100%done/.local-operator/operator/anchor.json",
        "C:/Users/$p!x/.local-operator/operator/anchor.json",
    ],
)
def test_windows_printed_line_has_nothing_either_shell_expands(
    monkeypatch: pytest.MonkeyPatch, staged: str
) -> None:
    """R1-3: pasted into PowerShell OR cmd.exe, the printed line must reach PowerShell intact.

    The values live only inside the gzip blob, so even a path carrying ``$``,
    ``%`` or ``'`` adds nothing either shell rewrites; the line also stays under
    cmd.exe's 8191-character limit.
    """
    monkeypatch.setattr(trust, "_on_windows", lambda: True)
    target = Path("C:/ProgramData/local-operator/operators/alice.json")

    command = trust.install_commands(Path(staged), target, "cd" * 32)
    line = trust.format_install_commands(command)

    argument = command[0][-1]
    assert not set('$`"%!') & set(argument), argument[:120]
    assert line == f'powershell -NoProfile -NonInteractive -Command "{argument}"'
    assert len(line) < 8191
    assert f"$staged={trust._ps_quote(str(Path(staged)))}" in _inner_script(command)


def test_install_anchor_on_windows_prints_and_never_runs_the_elevation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """lop never raises a UAC prompt itself: the operator runs the printed command."""
    monkeypatch.setattr(trust, "_on_windows", lambda: True)
    monkeypatch.setattr(handlers, "_is_privileged", lambda: False)
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", tmp_path / "anchor-root")
    ran: list[Any] = []
    monkeypatch.setattr(handlers.subprocess, "run", lambda *a, **k: ran.append(a))
    handlers.stage_anchor(tmp_path / "config", _anchor())

    assert handlers.install_anchor(tmp_path / "config") == 1
    assert ran == []
    assert "-Verb RunAs" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# F5: a device-admitting write refuses a stored admin password
# ---------------------------------------------------------------------------


def test_admit_device_refuses_a_stored_sudo_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    resolved: list[str] = []
    monkeypatch.setattr(handlers, "_resolve_secret_bytes", lambda name: resolved.append(name))
    monkeypatch.setattr(handlers.subprocess, "run", lambda *a, **k: pytest.fail("ran a command"))
    handlers.stage_anchor(tmp_path / "config", _anchor())

    code = handlers.install_anchor(
        tmp_path / "config", sudo_secret="ADMIN_PW", purpose=handlers.PURPOSE_ADMIT_DEVICE
    )

    assert code == 2
    assert resolved == [], "the stored password must not even be resolved"
    assert "stored admin password cannot be used" in capsys.readouterr().err


def test_onboarding_install_still_takes_the_sudo_secret(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``purpose``: the interim ask-sudo path is unchanged (narrowing it is not this PR)."""
    fed: list[str] = []
    monkeypatch.setattr(handlers, "_is_privileged", lambda: False)
    monkeypatch.setattr(handlers.shutil, "which", lambda name: "/usr/bin/sudo")
    monkeypatch.setattr(
        handlers, "_install_with_sudo_stdin", lambda command, name: fed.append(name) or 0
    )
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", tmp_path / "anchor-root")
    handlers.stage_anchor(tmp_path / "config", _anchor())

    handlers.install_anchor(tmp_path / "config", sudo_secret="ADMIN_PW")

    assert fed == ["ADMIN_PW"]


def test_admit_device_without_a_secret_takes_the_interactive_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard refuses the SECRET, not the write: the human-prompt path still runs."""
    ran: list[list[str]] = []
    monkeypatch.setattr(handlers, "_is_privileged", lambda: False)
    monkeypatch.setattr(handlers.shutil, "which", lambda name: "/usr/bin/sudo")
    monkeypatch.setattr(
        handlers.subprocess,
        "run",
        lambda argv, **k: ran.append(argv) or subprocess.CompletedProcess(argv, 0),
    )
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", tmp_path / "anchor-root")
    handlers.stage_anchor(tmp_path / "config", _anchor())

    handlers.install_anchor(tmp_path / "config", purpose=handlers.PURPOSE_ADMIT_DEVICE)

    assert ran and all(argv[0] == "sudo" for argv in ran)


# ---------------------------------------------------------------------------
# The staging-swap race: the privileged step installs only the digest lop read
# ---------------------------------------------------------------------------


def _unprivileged_install_shim(tmp_path: Path) -> dict[str, str]:
    """A PATH whose ``install`` drops ``-o``/``-g``, so the REAL script runs as this user.

    Only the ownership flags need root; the digest check and the rename — the
    part under test — are the generated script's own lines, unchanged.
    """
    real = shutil.which("install")
    assert real is not None
    bin_dir = tmp_path / "shim-bin"
    bin_dir.mkdir()
    shim = bin_dir / "install"
    shim.write_text(
        "#!/bin/sh\n"
        "args=''\n"
        'while [ $# -gt 0 ]; do case "$1" in -o|-g) shift 2;; '
        '*) args="$args \\"$1\\""; shift;; esac; done\n'
        f'eval "exec {real} $args"\n'
    )
    shim.chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}


def _run_privileged_half(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    assert argv[0] == "sudo"
    return subprocess.run(argv[1:], env=env, capture_output=True, text=True, check=False)


def test_the_privileged_step_installs_the_bytes_it_was_given_the_digest_of(
    tmp_path: Path,
) -> None:
    env = _unprivileged_install_shim(tmp_path)
    staged = tmp_path / "staged.json"
    staged.write_bytes(anchor_bytes(_anchor()))
    target = tmp_path / "anchor-dir" / "501.json"
    command = trust.install_commands(
        staged, target, hashlib.sha256(staged.read_bytes()).hexdigest()
    )

    for argv in command:
        result = _run_privileged_half(argv, env)
        assert result.returncode == 0, result.stderr

    assert target.read_bytes() == anchor_bytes(_anchor())
    assert oct(target.stat().st_mode & 0o777) == oct(0o644)
    assert [p.name for p in target.parent.iterdir()] == ["501.json"], "no temp file left"


def test_a_staged_file_swapped_after_it_was_read_is_refused(tmp_path: Path) -> None:
    env = _unprivileged_install_shim(tmp_path)
    staged = tmp_path / "staged.json"
    staged.write_bytes(anchor_bytes(_anchor()))
    target = tmp_path / "anchor-dir" / "501.json"
    command = trust.install_commands(
        staged, target, hashlib.sha256(staged.read_bytes()).hexdigest()
    )
    # The swap a same-user process makes between lop's read and root's install.
    staged.write_bytes(b'{"v": 1, "alg": "ES256", "spki": "attacker"}\n')

    results = [_run_privileged_half(argv, env) for argv in command]

    assert results[-1].returncode == 3, results[-1]
    assert "changed after it was read" in results[-1].stderr
    assert not target.exists()
    assert list(target.parent.iterdir()) == [], "the refused copy is cleaned up"


def test_install_anchor_end_to_end_refuses_a_swap_between_read_and_sudo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Through ``install_anchor`` itself: the swap lands between its read and the sudo step."""
    env = _unprivileged_install_shim(tmp_path)
    root = tmp_path / "config"
    anchor_root = tmp_path / "anchor-root"
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", anchor_root)
    monkeypatch.setattr(handlers, "_is_privileged", lambda: False)
    monkeypatch.setattr(handlers.shutil, "which", lambda name: "/usr/bin/sudo")
    staged = handlers.stage_anchor(root, _anchor())
    real_run = subprocess.run

    def swap_then_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        staged.write_bytes(b"attacker bytes\n")
        return real_run(argv[1:], env=env, check=False)

    monkeypatch.setattr(handlers.subprocess, "run", swap_then_run)

    assert handlers.install_anchor(root) == 3
    assert not trust.anchor_path().exists()


# ---------------------------------------------------------------------------
# Round 1 remediation: R1-1 (pinned root), R1-4 (device delta), R1-5 (junctions),
# R1-6 (the security reader never raises)
# ---------------------------------------------------------------------------


@pytest.fixture
def windows_root(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Drive the Windows root from two FAKED sources, with the process cache cleared."""

    def drive(system_dir: str | None, known: str | None) -> None:
        monkeypatch.setattr(trust, "_on_windows", lambda: True)
        monkeypatch.setattr(trust, "_system_windows_directory", lambda: system_dir)
        monkeypatch.setattr(trust, "_known_folder_program_data", lambda: known)
        trust._windows_program_data_resolution.cache_clear()

    yield drive
    trust._windows_program_data_resolution.cache_clear()


def test_windows_root_is_the_system_drive_when_both_sources_agree(windows_root: Any) -> None:
    windows_root("D:\\Windows", "d:\\programdata\\")
    assert trust.anchor_dir() == Path("D:\\ProgramData") / "local-operator" / "operators"


@pytest.mark.parametrize(
    ("system_dir", "known", "needle"),
    [
        # %SystemDrive% steered by the environment: the Known Folder follows it.
        ("C:\\Windows", "C:\\Users\\me\\x\\ProgramData", "not 'C:\\\\ProgramData'"),
        ("C:\\Windows", "<cwd>\\%SystemDrive%\\ProgramData", "not 'C:\\\\ProgramData'"),
        ("C:\\Windows", None, "could not be resolved"),
        (None, "C:\\ProgramData", "system directory could not be read"),
        ("\\\\server\\share\\Windows", "C:\\ProgramData", "names no drive letter"),
    ],
)
def test_windows_root_mismatch_refuses_with_a_reason(
    windows_root: Any, system_dir: str | None, known: str | None, needle: str
) -> None:
    """R1-1: any disagreement loads as no anchor, and the status line says why."""
    windows_root(system_dir, known)

    assert trust.anchor_dir() == trust._UNRESOLVED_ROOT
    loaded = trust.load_anchor("alice")

    assert not loaded.usable and not loaded.root_owned
    assert needle in loaded.reason, loaded.reason


def _installed(monkeypatch: pytest.MonkeyPatch, *devices: dict[str, Any]) -> None:
    """Make ``load_anchor`` report a usable INSTALLED anchor carrying ``devices``."""
    from dataclasses import replace

    installed = replace(_anchor(), devices=tuple(devices))
    monkeypatch.setattr(
        handlers,
        "load_anchor",
        lambda *a, **k: trust.AnchorLoad(
            anchor=installed, path=Path("/x"), root_owned=True, reason="ok", exists=True
        ),
    )


def _sudo_secret_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *staged_devices: dict[str, Any]
) -> tuple[int, list[str]]:
    from dataclasses import replace

    fed: list[str] = []
    monkeypatch.setattr(handlers, "_is_privileged", lambda: False)
    monkeypatch.setattr(handlers.shutil, "which", lambda name: "/usr/bin/sudo")
    monkeypatch.setattr(
        handlers, "_install_with_sudo_stdin", lambda command, name: fed.append(name) or 0
    )
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", tmp_path / "anchor-root")
    handlers.stage_anchor(tmp_path / "config", replace(_anchor(), devices=tuple(staged_devices)))
    # No ``purpose``: the guard must judge the write by what it does (R1-4).
    return handlers.install_anchor(tmp_path / "config", sudo_secret="ADMIN_PW"), fed


@pytest.mark.parametrize(
    ("installed", "staged"),
    [
        # Un-revoking a phone: the staged copy drops the revocation.
        ([{"device_id": "phone-1", "revoked": True}], []),
        # Adding a device entry the installed list does not carry.
        ([], [{"device_id": "phone-2"}]),
    ],
)
def test_a_stored_secret_cannot_admit_a_device_even_without_a_purpose(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    installed: list[dict[str, Any]],
    staged: list[dict[str, Any]],
) -> None:
    _installed(monkeypatch, *installed)

    code, fed = _sudo_secret_install(tmp_path, monkeypatch, *staged)

    assert code == 2 and fed == []
    assert "admits a device" in capsys.readouterr().err


def test_a_stored_secret_may_still_revoke_a_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tightening is not admitting: adding a revocation keeps the interim path."""
    _installed(monkeypatch)

    code, fed = _sudo_secret_install(
        tmp_path, monkeypatch, {"device_id": "phone-1", "revoked": True}
    )

    assert code == 0 and fed == ["ADMIN_PW"]


def test_a_windows_junction_in_the_anchor_path_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-5: a junction is a reparse point ``S_ISLNK`` does not see."""
    root = tmp_path / "local-operator" / "operators"
    root.mkdir(parents=True)
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", root)
    (root / "alice.json").write_bytes(anchor_bytes(_anchor()))
    real_lstat = os.lstat

    class _Junction:
        def __init__(self, info: os.stat_result) -> None:
            self.st_mode = info.st_mode  # a directory, not S_IFLNK
            self.st_file_attributes = 0x10 | 0x400  # DIRECTORY | REPARSE_POINT

    def lstat(path: Any, **kwargs: Any) -> Any:
        info = real_lstat(path, **kwargs)
        return _Junction(info) if not kwargs and Path(path) == root else info

    monkeypatch.setattr(trust.os, "lstat", lstat)

    loaded = trust.load_anchor("alice")

    assert not loaded.usable and loaded.exists
    assert "junction" in loaded.reason, loaded.reason


def test_load_anchor_never_raises_when_the_security_reader_breaks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-6: the REAL ``_windows_security`` on a host without ``ctypes.WinDLL``.

    ``getattr(ctypes, "WinDLL")`` raises ``AttributeError`` off Windows \u2014 exactly
    the kind of reader fault that used to propagate out of ``load_anchor``.
    """
    root = tmp_path / "local-operator" / "operators"
    root.mkdir(parents=True)
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", root)
    trust.anchor_path("alice").write_bytes(anchor_bytes(_anchor()))
    monkeypatch.setattr(trust, "_on_windows", lambda: True)
    monkeypatch.delattr(trust.ctypes, "WinDLL", raising=False)

    loaded = trust.load_anchor("alice")

    assert not loaded.usable and not loaded.root_owned
    assert "could not be read" in loaded.reason, loaded.reason
