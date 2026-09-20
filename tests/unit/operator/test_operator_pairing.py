"""``lop pair`` — the one flow in which a device becomes a signer (stage D).

WHY THIS FILE DRIVES THE HANDLER RATHER THAN SHELLING OUT. The verb's contract is
a sequence of filesystem facts plus ONE signed statement, and a subprocess would
add an environment to reproduce without testing any more of the logic. What must
be exercised for real is the part a subprocess could not fake anyway: the
certificate is produced by the operator's own signer and then VERIFIED against the
same key before it is installed, so a rig that stubbed the signer would be testing
its own stub.

The relay's half — ``POST /api/pair`` — is covered in
``tests/unit/mobile/test_relay_operator_authority.py``; here it is represented by
the one call it makes (``devices.write_pending``), which is the whole of its part.

NOTHING HERE TOUCHES A REAL KEYCHAIN: the ``file-only`` backend, a test-owned
anchor root, and ``--yes`` rather than a terminal prompt.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from local_operator.operator import devices
from local_operator.operator.keychain import FILE_ONLY
from local_operator.operator.sign import anchor_for_handle, create_key
from local_operator.operator.trust import (
    AnchorLoad,
    OperatorAnchor,
    anchor_bytes,
    load_staged_anchor,
    staging_path,
)
from local_operator.operator.verify import (
    key_id_for,
    read_device_cert,
    verify_device_cert,
)


def _config() -> Path:
    """The isolated config root the fixture installed, as the verbs read it."""
    return Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])


@pytest.fixture()
def paired_machine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[OperatorAnchor]:
    """A machine ready to pair: an operator key, an installed anchor, a config root."""
    anchor_root = tmp_path / "anchor-root"
    anchor_root.mkdir()
    monkeypatch.setattr(
        "local_operator.operator.trust._ANCHOR_ROOT_OVERRIDE", anchor_root, raising=False
    )
    handle = create_key(config_root=tmp_path, preference=FILE_ONLY)
    anchor = anchor_for_handle(handle, label="pairing-test")
    (anchor_root / "operator.json").write_bytes(anchor_bytes(anchor))
    # ...AND THE STAGED COPY, because that is the signal the signer's backend
    # resolution actually uses on a host whose anchor is not root-owned: an anchor
    # the runtime will not trust still tells this machine WHICH backend holds the
    # private half. ``lop operator init`` leaves exactly this pair behind, and the
    # pairing verb has to work in that state — it runs BEFORE the privileged
    # install as often as after it.
    staged = staging_path(tmp_path)
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(anchor_bytes(anchor))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    yield anchor


def _args(**overrides: Any) -> argparse.Namespace:
    defaults: dict[str, Any] = {
        "timeout": 0.0,
        "code_only": False,
        "yes": True,
        "device": "",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _phone_point() -> tuple[Any, bytes]:
    key = ec.generate_private_key(ec.SECP256R1())
    point = key.public_key().public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
    return key, point


def _claim(config_root: Path, point: bytes, *, name: str = "Damian's phone") -> str:
    """Exactly what the relay's ``POST /api/pair`` does once the code checks out."""
    device_id = devices.new_device_id(point)
    devices.write_pending(
        config_root,
        device_id=device_id,
        name=name,
        spki=devices.encode_spki(point),
        code=devices.read_pairing(config_root) or "",
    )
    return device_id


def test_code_only_prints_a_code_and_waits_for_nothing(
    paired_machine: OperatorAnchor, capsys: pytest.CaptureFixture[str]
) -> None:
    """The two-step path a scripted or second-terminal flow needs.

    Asserted on the CODE being live rather than merely printed: a ``--code-only``
    that printed a string nothing could claim would leave the operator holding a
    code the machine would refuse, which is worse than printing nothing.
    """
    from local_operator.operator.pair_handlers import dispatch

    assert dispatch(_args(code_only=True)) == 0
    printed = capsys.readouterr().out
    code = devices.read_pairing(_config())
    assert code is not None, "--code-only printed a code it did not publish"
    assert code in printed


def test_pairing_installs_a_verifiable_certificate_and_burns_the_code(
    paired_machine: OperatorAnchor, capsys: pytest.CaptureFixture[str]
) -> None:
    """THE FLOW, end to end, and the three facts that make it one device.

    1. the installed certificate VERIFIES against the anchored operator key — the
       verb checks this itself before writing, and this asserts the result rather
       than the check;
    2. the pairing code is gone, so a second phone cannot claim it;
    3. the pending request is gone, so a later ``lop operator devices`` does not
       offer an already-answered request.
    """
    from local_operator.operator.pair_handlers import dispatch

    root = _config()
    assert dispatch(_args(code_only=True)) == 0
    _, point = _phone_point()
    device_id = _claim(root, point)
    capsys.readouterr()

    assert dispatch(_args(device=device_id)) == 0
    output = capsys.readouterr().out
    assert device_id in output

    stored = devices.read_device(root, device_id)
    assert stored is not None, "the device was not installed"
    assert stored.device_id == key_id_for(point)
    assert stored.scope == ("loosen", "approve")
    parsed = read_device_cert(stored.certificate)
    assert parsed is not None
    assert (
        verify_device_cert(
            stored.certificate, operator_spki=paired_machine.spki, now=int(time.time())
        )
        is not None
    ), "the flow installed a certificate its own anchor refuses"
    assert devices.read_pairing(root) is None, "the code survived its use"
    assert devices.list_pending(root) == []
    # And the relay can now DECLARE it, which is how the runtime learns a phone is
    # paired without being told on every frame.
    assert devices.paired_certificate(root) == stored.certificate


def test_pairing_refuses_a_request_that_names_a_device_that_did_not_ask(
    paired_machine: OperatorAnchor, capsys: pytest.CaptureFixture[str]
) -> None:
    """A named device that is not pending is not silently swapped for the first.

    ``--device`` is an operator saying WHICH request they read. Answering with a
    different one would install authority for a device they never looked at, so the
    absence is a refusal with nothing installed.
    """
    from local_operator.operator.pair_handlers import dispatch

    root = _config()
    assert dispatch(_args(code_only=True)) == 0
    _, point = _phone_point()
    _claim(root, point, name="the one that asked")
    capsys.readouterr()

    assert dispatch(_args(device="deadbeef" * 4)) == 1
    assert devices.list_devices(root) == [], "a device was installed anyway"
    # The pending request SURVIVES a refusal: declining or mis-naming is a
    # statement about the operator's reading, not about the request.
    assert len(devices.list_pending(root)) == 1


def test_pairing_with_nothing_claimed_installs_nothing(
    paired_machine: OperatorAnchor, capsys: pytest.CaptureFixture[str]
) -> None:
    """The timeout path leaves the machine exactly as it was."""
    from local_operator.operator.pair_handlers import dispatch

    root = _config()
    assert dispatch(_args(code_only=True)) == 0
    capsys.readouterr()
    assert dispatch(_args()) == 1
    assert devices.list_devices(root) == []
    assert "Nothing was paired" in capsys.readouterr().out


def test_pairing_refuses_a_request_naming_a_key_this_build_cannot_read(
    paired_machine: OperatorAnchor, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed public point fails BEFORE the signing call.

    The operator is about to make a human gesture; spending it on a request this
    build cannot even parse is a gesture wasted on a refusal. The request is
    dropped so it cannot be re-offered, and nothing is installed.
    """
    from local_operator.operator.pair_handlers import dispatch

    root = _config()
    assert dispatch(_args(code_only=True)) == 0
    devices.write_pending(
        root,
        device_id="notakey",
        name="broken",
        spki="Zm9vYmFy",  # "foobar": not a P-256 point
        code=devices.read_pairing(root) or "",
    )
    capsys.readouterr()

    assert dispatch(_args(device="notakey")) == 1
    assert devices.list_devices(root) == []
    assert devices.list_pending(root) == []


def test_devices_lists_paired_and_pending_and_stages_a_revocation(
    paired_machine: OperatorAnchor,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``lop operator devices`` — the operator's view, and the revocation step.

    The revocation goes into the ROOT-OWNED ANCHOR rather than a file under the
    config root, and the privileged step is the same one the anchor install uses.
    ``--print-only`` is asserted here because running it would write a root-owned
    path from a test.

    ROOT OWNERSHIP ITSELF IS NOT THIS CELL'S CLAIM. A test cannot own a root file,
    so ``load_anchor`` is patched to report the load a real install produces; what
    this cell proves is what the VERB does with it — that the revocation lands in
    the anchor's own device list, that the certificate is dropped at once, and that
    the privileged step is the anchor's own. That an anchor is only trusted when
    root-owned is ``trust``'s claim, asserted by its own tests.
    """
    from local_operator.operator.pair_handlers import describe_devices
    from local_operator.operator.pair_handlers import dispatch as pair_dispatch

    root = _config()
    assert pair_dispatch(_args(code_only=True)) == 0
    _, paired_point = _phone_point()
    paired_id = _claim(root, paired_point, name="paired phone")
    assert pair_dispatch(_args(device=paired_id)) == 0

    _, pending_point = _phone_point()
    pending_id = _claim(root, pending_point, name="waiting phone")
    capsys.readouterr()

    list_args = argparse.Namespace(revoke="", print_only=False)
    assert describe_devices(list_args) == 0
    listing = capsys.readouterr().out
    assert paired_id in listing and "active" in listing
    assert pending_id in listing and "PENDING" in listing

    # Revoke, staged only. ``stage_anchor`` is what the privileged install would
    # move, so the staged file is the assertion that a revocation was prepared.
    monkeypatch.setattr(
        "local_operator.operator.trust.load_anchor",
        lambda *_a, **_k: AnchorLoad(
            anchor=paired_machine, path=root / "anchor.json", root_owned=True, reason=""
        ),
    )
    revoke_args = argparse.Namespace(revoke=paired_id, print_only=True)
    assert describe_devices(revoke_args) == 0
    staged = load_staged_anchor(root)
    assert staged is not None, "the revocation was not staged"
    assert any(
        entry.get("device_id") == paired_id and entry.get("revoked") for entry in staged.devices
    )
    # ...and the certificate is gone from the store immediately, so the refusal is
    # true even on a host whose anchor has not caught up.
    assert devices.read_device(root, paired_id) is None
