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

import os
import subprocess
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
    trust._windows_program_data.cache_clear()
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "attacker"))
    monkeypatch.setenv("USERNAME", "mallory")

    resolved = trust.anchor_dir()

    print("anchor_dir with hostile PROGRAMDATA ->", resolved)
    assert "attacker" not in str(resolved)
    assert resolved.parts[-2:] == ("local-operator", "operators")
    assert resolved.parent.parent.is_dir(), "the Known Folder resolves to a real directory"
    assert "mallory" not in trust.anchor_path().name
