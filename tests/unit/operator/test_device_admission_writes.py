"""UNRUN: S2 admission writes, plant monotonicity and migration guards.

Written under the fleet disk/CI hold. These cells use only the in-process anchor
root seam and fake root ownership; no OS prompt or real privileged write runs.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.operator import handlers, trust
from local_operator.operator.elevate import ElevationResult
from local_operator.operator.trust import anchor_bytes
from local_operator.operator.verify import key_id_for

ID_X, ID_Y = "a" * 32, "b" * 32


def _anchor(**changes: Any) -> trust.OperatorAnchor:
    spki = b"\x04" + bytes(range(64))
    return replace(
        trust.OperatorAnchor(
            key_id=key_id_for(spki), spki=spki, backend="file-only", presence=False
        ),
        **changes,
    )


def _install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, anchor: trust.OperatorAnchor
) -> tuple[Path, Path]:
    root = tmp_path / "config"
    anchor_root = tmp_path / "anchors"
    anchor_root.mkdir(exist_ok=True)
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", anchor_root)
    target = trust.anchor_path()
    target.write_bytes(anchor_bytes(anchor))
    real_open = trust._open_no_follow

    def as_root(path: Path) -> Any:
        opened = real_open(path)
        assert opened is not None
        descriptor, _ = opened
        return descriptor, SimpleNamespace(st_uid=0, st_mode=0o100644)

    monkeypatch.setattr(trust, "_open_no_follow", as_root)
    return root, target


def _plant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, anchor: trust.OperatorAnchor
) -> Path:
    source = tmp_path / "received.json"
    source.write_bytes(anchor_bytes(anchor))
    monkeypatch.setattr(handlers, "install_anchor", lambda *a, **k: 0)
    return source


def test_r1_refuses_revocation_lift_and_allow_control_admits_new_phone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    installed = _anchor(devices=({"device_id": ID_X, "revoked": True},))
    root, target = _install(tmp_path, monkeypatch, installed)
    staged = trust.staging_path(root)
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"sentinel")
    incoming = replace(installed, devices=({"device_id": ID_Y},))
    source = _plant(tmp_path, monkeypatch, incoming)

    assert handlers.install_anchor_from(source, root) == 2
    assert target.read_bytes() == anchor_bytes(installed)
    assert staged.read_bytes() == b"sentinel", "R1 refusal happens before staging"
    assert ID_X in capsys.readouterr().err

    # ALLOW control: an ordinary incoming admission must propagate; R1 gates only
    # the un-revoke half, not the statement's complete list.
    retaining = replace(installed, devices=(*installed.devices, {"device_id": ID_Y}))
    allowed = _plant(tmp_path, monkeypatch, retaining)
    assert handlers.install_anchor_from(allowed, root) == 0
    assert json.loads(staged.read_bytes())["devices"][-1] == {"device_id": ID_Y}

    assert handlers.install_anchor_from(source, root, allow_revocation_lift=True) == 0
    assert json.loads(staged.read_bytes())["devices"] == [{"device_id": ID_Y}]


def test_plant_replaces_existing_list_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed = _anchor(devices=({"device_id": ID_X},), admission="")
    root, _ = _install(tmp_path, monkeypatch, installed)
    incoming = replace(
        installed,
        admission=trust.ADMISSION_LISTED,
        devices=({"device_id": ID_Y}, {"device_id": ID_X, "revoked": True}),
    )
    source = _plant(tmp_path, monkeypatch, incoming)

    assert handlers.install_anchor_from(source, root) == 0
    assert trust.staging_path(root).read_bytes() == anchor_bytes(incoming)
    assert {"admitted": True} not in json.loads(trust.staging_path(root).read_bytes())["devices"]


def _confirm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    installed: trust.OperatorAnchor | None = None,
    ids: list[str] | None = None,
    sudo_secret: str = "",
) -> tuple[int, list[list[list[str]]], Path, Path]:
    anchor = installed or _anchor()
    root, target = _install(tmp_path, monkeypatch, anchor)
    monkeypatch.setattr(
        "local_operator.operator.devices.list_devices",
        lambda root: [SimpleNamespace(device_id=ID_X), SimpleNamespace(device_id=ID_Y)],
    )
    commands: list[list[list[str]]] = []

    def succeed(command: list[list[str]]) -> ElevationResult:
        commands.append(command)
        target.write_bytes(trust.staging_path(root).read_bytes())
        return ElevationResult(0)

    code = handlers.confirm_devices(root, ids or [ID_X], runner=succeed, sudo_secret=sudo_secret)
    return code, commands, root, target


def test_confirm_is_idempotent_and_keeps_unselected_phone_listed_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, commands, root, target = _confirm(tmp_path, monkeypatch)
    assert code == 0 and len(commands) == 1
    installed = trust.OperatorAnchor.from_json(json.loads(target.read_bytes()))
    assert installed is not None
    assert installed.admission == trust.ADMISSION_LISTED
    assert {"device_id": ID_X} in installed.devices
    assert ID_Y not in trust.open_device_ids(installed)
    assert all("admitted" not in entry for entry in installed.devices)

    # A repeated selection produces exactly the old bytes and no second gesture.
    code, commands, _, _ = _confirm(
        tmp_path,
        monkeypatch,
        installed=installed,
        ids=[ID_X, ID_X],
    )
    assert code == 0 and commands == []


def test_admit_refuses_migration_when_existing_phone_would_be_de_admitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, target = _install(tmp_path, monkeypatch, _anchor())
    old = target.read_bytes()
    monkeypatch.setattr(
        "local_operator.operator.devices.list_devices",
        lambda root: [SimpleNamespace(device_id=ID_X)],
    )

    assert handlers.admit_device(root, ID_Y) == 2
    assert target.read_bytes() == old
    assert not trust.staging_path(root).exists()
    assert "de-admit the phones you already paired" in capsys.readouterr().err


def test_cancelled_prompt_writes_nothing_and_leaves_no_installable_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed = _anchor()
    root, target = _install(tmp_path, monkeypatch, installed)
    monkeypatch.setattr(
        "local_operator.operator.devices.list_devices",
        lambda root: [SimpleNamespace(device_id=ID_X)],
    )
    calls: list[Any] = []

    def cancelled(command: Any) -> ElevationResult:
        calls.append(command)
        return ElevationResult(1, cancelled=True)

    assert handlers.confirm_devices(root, [ID_X], runner=cancelled) == 1
    assert len(calls) == 1
    assert target.read_bytes() == anchor_bytes(installed)
    assert not trust.staging_path(root).exists(), "a later install must not apply a cancelled admit"


def test_authorise_passes_the_admit_purpose_so_stored_password_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """F5 must cover authorise, not only the new confirm helper."""
    installed = _anchor(
        admission=trust.ADMISSION_LISTED,
        devices=({"device_id": ID_X, "revoked": True},),
    )
    root, target = _install(tmp_path, monkeypatch, installed)
    handlers.stage_anchor(root, _admission_anchor_for_test(installed, ID_X))
    resolved: list[str] = []
    monkeypatch.setattr(handlers, "_resolve_secret_bytes", lambda name: resolved.append(name))
    monkeypatch.setattr(
        "local_operator.operator.pair_handlers.devices.record_revocation", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "local_operator.operator.pair_handlers.devices.is_revoked_here", lambda *a, **k: True
    )
    monkeypatch.setattr(
        "local_operator.operator.pair_handlers.devices.clear_revocation", lambda *a, **k: True
    )
    monkeypatch.setattr(
        "local_operator.operator.pair_handlers.devices.list_devices", lambda *a: []
    )
    monkeypatch.setattr(
        "local_operator.operator.pair_handlers.devices.list_pending", lambda *a: []
    )
    from argparse import Namespace

    from local_operator.operator.pair_handlers import describe_devices

    assert describe_devices(
        Namespace(
            authorise=ID_X,
            revoke="",
            print_only=False,
            sudo_secret="ADMIN_PW",
        )
    ) == 2
    assert resolved == []
    assert target.read_bytes() == anchor_bytes(installed)


def _admission_anchor_for_test(installed: trust.OperatorAnchor, device_id: str) -> Any:
    return replace(
        installed,
        devices=({"device_id": device_id},),
    )


def test_downgrade_refuses_sudo_secret_without_resolving_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root, target = _install(
        tmp_path, monkeypatch, _anchor(admission=trust.ADMISSION_LISTED)
    )
    handlers.stage_anchor(root, _anchor())
    resolved: list[str] = []
    monkeypatch.setattr(handlers, "_resolve_secret_bytes", lambda name: resolved.append(name))

    assert handlers.install_anchor(root, sudo_secret="ADMIN_PW") == 2
    assert resolved == []
    assert target.read_bytes() == anchor_bytes(_anchor(admission=trust.ADMISSION_LISTED))
    assert "turning off phone admission" in capsys.readouterr().err


def test_admission_command_pins_installed_and_staged_digests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed = _anchor()
    code, commands, root, target = _confirm(tmp_path, monkeypatch, installed=installed)
    assert code == 0
    final = commands[0][-1]
    assert final[-1] == hashlib.sha256(anchor_bytes(installed)).hexdigest()
    assert final[-3] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert commands[0][0][0] == "sudo"


def _run_posix_install(
    tmp_path: Path, staged: Path, target: Path, old_hex: str
) -> subprocess.CompletedProcess[str]:
    """Execute the real root script with only ownership flags shimmed out."""
    import shutil

    shim = tmp_path / "bin"
    shim.mkdir()
    real = shutil.which("install")
    assert real is not None
    executable = shim / "install"
    # Ownership flags are the only part this unprivileged control removes.
    executable.write_text(
        "#!/bin/sh\nwhile [ $# -gt 0 ]; do case \"$1\" in -o|-g) shift 2;; "
        "-m) shift 2;; *) break;; esac; done\n"
        f"exec {shlex.quote(real)} \"$@\"\n"
    )
    executable.chmod(0o755)
    command = trust.install_commands(
        staged, target, hashlib.sha256(staged.read_bytes()).hexdigest(), old_hex
    )[-1]
    assert command[0] == "sudo"
    return subprocess.run(
        command[1:], capture_output=True, text=True, env={"PATH": f"{shim}:/usr/bin:/bin"}
    )


def test_privileged_cas_refuses_installed_swap_and_staged_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _anchor()
    _, target = _install(tmp_path, monkeypatch, original)
    staged = tmp_path / "config" / "staged.json"
    staged.write_bytes(anchor_bytes(_anchor(admission=trust.ADMISSION_LISTED)))
    old_hex = hashlib.sha256(target.read_bytes()).hexdigest()

    target.write_bytes(anchor_bytes(replace(original, devices=({"device_id": ID_Y},))))
    assert _run_posix_install(tmp_path / "one", staged, target, old_hex).returncode == 3
    assert ID_Y in target.read_text()

    target.write_bytes(anchor_bytes(original))
    before = target.read_bytes()
    result = _run_posix_install(tmp_path / "two", staged, target, old_hex)
    # The unchanged control is the discriminating setup: it must be able to pass
    # before its staged-file mutation proves the negative is not a broken rig.
    assert result.returncode == 0, result.stderr
    target.write_bytes(before)
    staged.write_bytes(anchor_bytes(_anchor(devices=({"device_id": ID_Y},))))
    assert _run_posix_install(tmp_path / "three", staged, target, old_hex).returncode == 3
    assert target.read_bytes() == before
