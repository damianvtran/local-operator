"""The Windows anchor verifier against REAL ACLs (F4) — runs only on Windows.

WHY A NATIVE JOB AND NOT ONLY THE FAKE-DESCRIPTOR TESTS: ``test_anchor_hardening``
proves the verifier's DECISION over values it is handed; only a real Windows
kernel proves that ``_windows_security`` reads the owner and DACL that
``icacls`` actually set, from an open descriptor, and that the Known Folder
lookup ignores the environment. ``.github/workflows/windows-anchor.yml`` runs
this file on windows-latest, where the runner account is an administrator, so it
can set the Administrators owner the hardened layout needs.

WHAT THIS DOES NOT PROVE: the UAC prompt itself, or that the secure desktop
resists same-user UI automation — a CI runner has UAC disabled. The elevated
install script is exercised here without elevation for the same reason.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator.operator import trust
from local_operator.operator.trust import OperatorAnchor, anchor_bytes
from local_operator.operator.verify import key_id_for

pytestmark = pytest.mark.skipif(os.name != "nt", reason="native Windows ACLs only")

SYSTEM, ADMINS, USERS = "*S-1-5-18", "*S-1-5-32-544", "*S-1-5-32-545"


def _anchor() -> OperatorAnchor:
    spki = b"\x04" + bytes(range(64))
    return OperatorAnchor(key_id=key_id_for(spki), spki=spki, backend="file-only", presence=False)


def _icacls(*args: str) -> None:
    result = subprocess.run(["icacls", *args], capture_output=True, text=True, check=False)
    assert result.returncode == 0, (args, result.stdout, result.stderr)


def _harden(path: Path, inherit: str) -> None:
    _icacls(
        str(path),
        "/inheritance:r",
        "/grant:r",
        f"{SYSTEM}:{inherit}F",
        f"{ADMINS}:{inherit}F",
        f"{USERS}:{inherit}R",
    )
    _icacls(str(path), "/setowner", ADMINS)


def _tree(tmp_path: Path) -> Path:
    """``<tmp>/local-operator/operators/<account>.json`` with the default (user) ACL."""
    operators = tmp_path / "local-operator" / "operators"
    operators.mkdir(parents=True)
    (operators / "anchor.json").write_bytes(anchor_bytes(_anchor()))
    return operators


def test_a_default_acl_anchor_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Created the way a same-user process would: the user owns it and may write it."""
    operators = _tree(tmp_path)
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", operators)

    load = trust.load_anchor("anchor")

    print("default ACL ->", load.usable, load.reason)
    assert not load.usable
    assert not load.root_owned
    assert "not SYSTEM or Administrators" in load.reason or "writable by" in load.reason


def test_an_icacls_hardened_anchor_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    operators = _tree(tmp_path)
    _harden(operators.parent, "(OI)(CI)")
    _harden(operators, "(OI)(CI)")
    _harden(operators / "anchor.json", "")
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", operators)

    load = trust.load_anchor("anchor")

    print("hardened ACL ->", load.usable, load.reason)
    assert load.usable, load.reason
    assert load.root_owned


def test_hardened_file_under_a_user_writable_directory_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file alone is not enough: a writable parent lets the user replace it."""
    operators = _tree(tmp_path)
    _harden(operators / "anchor.json", "")
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", operators)

    load = trust.load_anchor("anchor")

    print("hardened file, default dir ->", load.usable, load.reason)
    assert not load.usable
    assert "directory" in load.reason


def test_known_folder_ignores_a_hostile_programdata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trust._windows_program_data_resolution.cache_clear()
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "attacker"))
    monkeypatch.setenv("USERNAME", "mallory")

    resolved = trust.anchor_dir()

    print("anchor_dir with hostile PROGRAMDATA ->", resolved)
    assert "attacker" not in str(resolved)
    assert resolved.parts[-2:] == ("local-operator", "operators")
    assert resolved.parent.parent.is_dir(), "the Known Folder resolves to a real directory"
    assert "mallory" not in trust.anchor_path().name


# ---------------------------------------------------------------------------
# Round 1 remediation, on real Windows: R1-1 (SystemDrive), R1-5 (junction on
# the load side), R1-2 (the elevated script refuses planted targets) and R1-3
# (the PRINTED line runs from both PowerShell and cmd.exe).
# ---------------------------------------------------------------------------

_PROBE = (
    "from local_operator.operator import trust\n"
    "print('ROOT', trust.anchor_dir())\n"
    "print('REASON', trust.load_anchor('probe').reason)\n"
)


def _expected_root() -> Path:
    system_dir = trust._system_windows_directory()
    assert system_dir, "GetSystemWindowsDirectoryW answered nothing"
    return Path(system_dir[:2].upper() + "\\ProgramData") / "local-operator" / "operators"


def test_a_hostile_systemdrive_cannot_move_the_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-1: ``SystemDrive`` set, or removed, in-process AND in a fresh process."""
    expected = _expected_root()
    trust._windows_program_data_resolution.cache_clear()
    monkeypatch.setenv("SystemDrive", str(tmp_path / "attacker"))
    in_process = trust.anchor_dir()
    print("in-process hostile SystemDrive ->", in_process, "| expected", expected)
    assert in_process in (expected, trust._UNRESOLVED_ROOT)
    assert "attacker" not in str(in_process)
    trust._windows_program_data_resolution.cache_clear()

    for label, value in (("hostile", str(tmp_path / "attacker")), ("removed", None)):
        env = {k: v for k, v in os.environ.items() if k.upper() != "SYSTEMDRIVE"}
        if value is not None:
            env["SystemDrive"] = value
        result = subprocess.run(
            [sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, check=False
        )
        print(f"subprocess {label} SystemDrive ->", result.stdout.strip().replace("\n", " | "))
        assert result.returncode == 0, result.stderr
        root = result.stdout.split("ROOT ", 1)[1].splitlines()[0]
        assert root in (str(expected), str(trust._UNRESOLVED_ROOT)), root
        if root == str(trust._UNRESOLVED_ROOT):
            assert "ProgramData" in result.stdout, "a refusal names its reason"


def _junction(link: Path, target: Path) -> None:
    result = subprocess.run(
        ["cmd", "/d", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_junction_in_the_anchor_path_is_refused_on_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-5: a junction is not ``S_ISLNK``; the load must still refuse it."""
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    (decoy / "anchor.json").write_bytes(anchor_bytes(_anchor()))
    (tmp_path / "local-operator").mkdir()
    operators = tmp_path / "local-operator" / "operators"
    _junction(operators, decoy)
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", operators)

    load = trust.load_anchor("anchor")

    print("junction on load ->", load.usable, load.reason)
    assert not load.usable and "junction" in load.reason


def _acl(path: Path) -> str:
    return subprocess.run(
        ["icacls", str(path)], capture_output=True, text=True, check=False
    ).stdout.replace(str(path), "<p>")


def _staged(tmp_path: Path) -> tuple[Path, str]:
    staged = tmp_path / "staged" / "anchor.json"
    staged.parent.mkdir()
    staged.write_bytes(anchor_bytes(_anchor()))
    return staged, hashlib.sha256(staged.read_bytes()).hexdigest()


def _run_inner(command: list[list[str]]) -> subprocess.CompletedProcess[str]:
    """The ELEVATED script, run directly (the runner is already elevated)."""
    blob = command[0][-1].split("FromBase64String('", 1)[1].split("'", 1)[0]
    script = gzip.decompress(base64.b64decode(blob)).decode("utf-8")
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )


def test_install_refuses_a_planted_junction_and_leaves_the_decoy_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-2: ``operators`` pre-planted as a junction to a decoy directory."""
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", None)
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    before = _acl(decoy)
    (tmp_path / "ProgramData" / "local-operator").mkdir(parents=True)
    # Administrators-owned (what an elevated runner creates anyway), so the ONLY
    # thing wrong with the tree is the junction.
    _icacls(str(tmp_path / "ProgramData" / "local-operator"), "/setowner", ADMINS)
    _junction(tmp_path / "ProgramData" / "local-operator" / "operators", decoy)
    staged, digest = _staged(tmp_path)
    target = tmp_path / "ProgramData" / "local-operator" / "operators" / "anchor.json"

    result = _run_inner(trust.install_commands(staged, target, digest))

    after = _acl(decoy)
    print("planted junction -> rc", result.returncode, "|", result.stderr.strip().splitlines()[0])
    print("decoy ACL unchanged ->", before == after, "| decoy entries:", os.listdir(decoy))
    assert result.returncode != 0
    assert "is a junction or symbolic link, so it was refused" in result.stderr
    assert before == after, (before, after)
    assert os.listdir(decoy) == []


def test_install_replaces_a_planted_grant_and_refuses_a_foreign_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-2: an attacker's explicit ACE is GONE after install; a foreign owner refuses."""
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", None)
    operators = tmp_path / "ProgramData" / "local-operator" / "operators"
    operators.mkdir(parents=True)
    _icacls(str(operators.parent), "/setowner", ADMINS)
    _icacls(str(operators), "/setowner", ADMINS)
    _icacls(str(operators), "/grant", "*S-1-1-0:(OI)(CI)F")
    staged, digest = _staged(tmp_path)
    target = operators / "anchor.json"
    print("planted grant before ->", "Everyone" in _acl(operators))

    result = _run_inner(trust.install_commands(staged, target, digest))

    print("planted grant -> rc", result.returncode, result.stdout.strip(), result.stderr.strip())
    after = _acl(operators)
    print("planted grant after ->", "Everyone" in after or "S-1-1-0" in after)
    assert result.returncode == 0, result.stderr
    assert "Everyone" not in after and "S-1-1-0" not in after, after
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", operators)
    assert trust.load_anchor("anchor").usable

    foreign = tmp_path / "foreign" / "ProgramData"
    (foreign / "local-operator").mkdir(parents=True)
    _icacls(str(foreign / "local-operator"), "/setowner", USERS)
    result = _run_inner(
        trust.install_commands(staged, foreign / "local-operator" / "operators" / "a.json", digest)
    )
    print("foreign owner -> rc", result.returncode, "|", result.stderr.strip().splitlines()[0])
    assert result.returncode != 0 and "not SYSTEM or Administrators" in result.stderr


@pytest.mark.parametrize("shell", ["powershell", "cmd"])
def test_the_printed_line_installs_when_pasted_into_either_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shell: str
) -> None:
    """R1-3 / QA Q1: execute the EXACT printed line, as each shell would parse it.

    PowerShell: the line is the body of a ``.ps1`` (parsed as if typed at the
    prompt). cmd.exe: ``cmd /d /c <line>``. ``Start-Process -Verb RunAs`` does
    not prompt on the runner (administrator, UAC off). The target is a scratch
    root via the test seam, never the real ProgramData anchor.
    """
    operators = tmp_path / "ProgramData" / "local-operator" / "operators"
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", operators)
    # The elevated script creates only the two directories it owns (CreateDirectoryW
    # is not recursive); ProgramData itself always exists on a real host.
    operators.parent.parent.mkdir(parents=True)
    staged, digest = _staged(tmp_path)
    target = trust.anchor_path("anchor")
    line = trust.format_install_commands(trust.install_commands(staged, target, digest))

    if shell == "powershell":
        script = tmp_path / "pasted.ps1"
        script.write_text(line + "\nexit $LASTEXITCODE\n", encoding="utf-8")
        argv = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)]
        result = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=180)
    else:
        result = subprocess.run(
            f"cmd /d /c {line}", capture_output=True, text=True, check=False, timeout=180
        )

    load = trust.load_anchor("anchor")
    print(f"printed line via {shell} -> rc", result.returncode, "| load", load.usable, load.reason)
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert target.read_bytes() == staged.read_bytes()
    assert load.usable, load.reason


# ---------------------------------------------------------------------------
# Round 2 remediation, on real Windows: R2-1 (no command for an unresolved root;
# the elevated script refuses a drive-root target) and R2-4 (full path).
# ---------------------------------------------------------------------------

_INSTALL_PROBE = (
    "import sys\n"
    "from pathlib import Path\n"
    "from local_operator.operator import handlers, trust\n"
    "from local_operator.operator.trust import OperatorAnchor\n"
    "from local_operator.operator.verify import key_id_for\n"
    "spki = b'\\x04' + bytes(range(64))\n"
    "a = OperatorAnchor(key_id=key_id_for(spki), spki=spki, backend='file-only', presence=False)\n"
    "root = Path(sys.argv[1])\n"
    "handlers.stage_anchor(root, a)\n"
    "print('ROOT', trust.anchor_dir())\n"
    "print('RC', handlers.install_anchor(root, print_only=True))\n"
)


def test_an_unresolved_root_prints_no_command(tmp_path: Path) -> None:
    """R2-1: a fresh process with ``SystemDrive`` removed resolves no root, and
    ``install --print-only`` then prints the reason and NO elevated command."""
    env = {k: v for k, v in os.environ.items() if k.upper() != "SYSTEMDRIVE"}
    result = subprocess.run(
        [sys.executable, "-c", _INSTALL_PROBE, str(tmp_path / "config")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    out = result.stdout + result.stderr
    print("unresolved root install ->", out.strip().replace("\n", " | "))
    assert result.returncode == 0, result.stderr
    if f"ROOT {trust._UNRESOLVED_ROOT}" not in result.stdout:
        pytest.skip("this host resolved ProgramData without SystemDrive")
    assert "RC 1" in result.stdout
    assert "RunAs" not in out and "powershell" not in out.lower()
    assert "cannot be installed on this computer" in result.stderr


def test_the_elevated_script_refuses_a_drive_root_target(tmp_path: Path) -> None:
    """R2-1 defence in depth: handed the unresolved root anyway (``$root`` = the
    drive root), the elevated script refuses before touching anything."""
    staged, digest = _staged(tmp_path)
    drive_root = Path(Path.cwd().anchor)
    before = _acl(drive_root)
    target = Path(str(trust._UNRESOLVED_ROOT)) / "probe.json"

    result = _run_inner(trust.install_commands(staged, target, digest))

    after = _acl(drive_root)
    print("drive-root target -> rc", result.returncode, "|", result.stderr.strip()[:200])
    print("drive root ACL unchanged ->", before == after)
    assert result.returncode != 0
    assert "refusing to install into" in result.stderr
    assert before == after
    assert not Path(str(trust._UNRESOLVED_ROOT)).exists()


def test_the_printed_line_names_powershell_by_full_path() -> None:
    """R2-4: the printed launcher and the elevated child are the System32 binary."""
    command = trust.install_commands(
        Path("C:/s/anchor.json"), Path("C:/ProgramData/local-operator/operators/a.json"), "ab" * 32
    )
    line = trust.format_install_commands(command)
    print("printed line head ->", line[:140])
    expected = trust.windows_powershell_path()
    assert expected is not None and Path(expected).is_file()
    assert command[0][0] == expected and f"-FilePath '{expected}'" in command[0][-1]
