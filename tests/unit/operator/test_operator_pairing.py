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
import json
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


def test_the_pairing_receipt_does_not_promise_authority_this_host_lacks(
    paired_machine: OperatorAnchor, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """UX round 6, U2: the receipt has to agree with the refusal about the same fact.

    Measured before this fix, in the default state between `lop operator init`
    (which only STAGES the anchor) and `lop operator install` (the privileged step
    that lands it): `lop pair` printed "The phone can now approve parked cards and
    loosen a running gate" — and every signature that phone then sent was refused,
    because the runtime has no key to verify against. Two surfaces, one fact, two
    answers, and the optimistic one came first.

    Both halves are asserted, because a receipt that hedged on a host WITH an anchor
    would be the same defect mirrored.
    """
    from local_operator.operator import operator_authority_unusable, trust
    from local_operator.operator.pair_handlers import dispatch

    root = _config()
    assert dispatch(_args(code_only=True)) == 0
    _, point = _phone_point()
    device_id = _claim(root, point)
    capsys.readouterr()

    # (1) NO USABLE ANCHOR: the receipt names the step that would change that.
    import local_operator.operator as operator_pkg

    # BOTH NAMES, and the reason is worth stating: the runtime's cache reads
    # `trust.load_anchor` (its own module global), while `operator_authority_unusable`
    # — the predicate the receipts consult — reads the name `local_operator.operator`
    # imported from it. Patching one and not the other is how a test ends up
    # asserting the two ends of the same question in different worlds.
    def absent(uid: Any = None) -> Any:
        return _absent_anchor(trust, uid)

    monkeypatch.setattr(trust, "load_anchor", absent)
    monkeypatch.setattr(operator_pkg, "load_anchor", absent)
    assert operator_authority_unusable() is True
    assert dispatch(_args(device=device_id)) == 0
    captured = capsys.readouterr().out
    assert "lop operator install" in captured, captured
    assert "The phone can now approve parked cards" not in captured, captured
    # The certificate is still installed: the receipt is about what the phone can
    # DO, not about whether the machine recorded it (which the refusal's copy and
    # `lop operator status` both handle).
    assert devices.read_device(root, device_id) is not None

    # (2) A SECOND DEVICE ON A HOST WITH ITS ANCHOR USABLE: the promise is true,
    # and made. The root-owned fact is supplied at the same read seam, because the
    # real anchor is a root-owned file this suite cannot create — the fixture's
    # staged one is the real anchor the product writes, reported as installed.
    def installed(uid: Any = None) -> Any:
        return _installed_anchor(trust, root, uid)

    monkeypatch.setattr(trust, "load_anchor", installed)
    monkeypatch.setattr(operator_pkg, "load_anchor", installed)
    assert operator_authority_unusable() is False
    _, second = _phone_point()
    second_id = _claim(root, second, name="second")
    capsys.readouterr()
    assert dispatch(_args(device=second_id)) == 0
    captured = capsys.readouterr().out
    assert "The phone can now approve parked cards" in captured, captured
    assert "lop operator install" not in captured, captured


def _installed_anchor(trust: Any, root: Path, uid: Any) -> Any:
    """The staged anchor, reported the way an INSTALLED one is.

    The same seam ``_absent_anchor`` uses, in the other direction: the file is the
    one ``lop operator init`` really stages, parsed by the real parser, and only the
    root-owned INSTALL fact is supplied — which is the one thing a test cannot
    create without ``sudo`` and a password prompt on the operator's screen.
    """
    body = json.loads(trust.staging_path(root).read_text())
    parsed = trust.OperatorAnchor.from_json(body)
    assert parsed is not None
    return trust.AnchorLoad(
        anchor=parsed,
        path=trust.anchor_path(uid),
        root_owned=True,
        reason="ok",
        exists=True,
    )


def _absent_anchor(trust: Any, uid: Any) -> Any:
    """An ``AnchorLoad`` for a host that has no usable anchor, without touching disk."""
    return trust.AnchorLoad(
        anchor=None,
        path=trust.anchor_path(uid),
        root_owned=False,
        reason="pinned absent by the test",
        exists=False,
    )


def test_the_revoke_receipt_states_the_window_it_actually_honours(
    paired_machine: OperatorAnchor, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N-1 and M-5: the two copy facts the revoke verb owes an operator.

    N-1 — the receipt ("a session already running picks it up within 30s") is the
    ONE place the propagation window is stated to a human, and it was asserted by
    nothing. It is also the sentence a reader can check against behaviour, so the
    assertion is against the CONSTANT the runtime enforces rather than the digit
    30: the runtime used to carry its own literal (agent round 7, M-2), which is
    exactly how a product ends up quoting a window it does not honour.

    M-5 — the pre-install path (the default state) said only "no operator anchor on
    this machine to record a revocation in" and named no remedy, two lines below a
    sentence that had just gained one. Measured on the real CLI: rc 1 on a
    staged-but-not-installed host, i.e. the state every reader of the pairing docs
    is in.
    """
    import local_operator.operator as operator_pkg
    from local_operator.operator import trust
    from local_operator.operator.pair_handlers import describe_devices
    from local_operator.operator.trust import ANCHOR_REFRESH_S
    from local_operator.session.runtime import server as server_module

    root = _config()
    _, point = _phone_point()
    device_id = devices.new_device_id(point)

    def installed(uid: Any = None) -> Any:
        return _installed_anchor(trust, root, uid)

    monkeypatch.setattr(trust, "load_anchor", installed)
    monkeypatch.setattr(operator_pkg, "load_anchor", installed)

    # M-2's binding, asserted where both names are in hand: the window the receipt
    # quotes is the window the runtime enforces, because it is the same object.
    assert server_module._ANCHOR_REFRESH_S == ANCHOR_REFRESH_S

    assert describe_devices(_args(revoke=device_id, print_only=True)) == 0
    receipt = capsys.readouterr().out
    assert f"within {int(ANCHOR_REFRESH_S)}s" in receipt, receipt
    assert "Revocation lives in the anchor" in receipt, receipt

    # (b) NO USABLE ANCHOR: the message names the step that ends that state.
    monkeypatch.setattr(trust, "load_anchor", lambda uid=None: _absent_anchor(trust, uid))
    monkeypatch.setattr(operator_pkg, "load_anchor", lambda uid=None: _absent_anchor(trust, uid))
    assert describe_devices(_args(revoke=device_id, print_only=True)) == 1
    refusal = capsys.readouterr().err
    assert "lop operator install" in refusal, refusal
    assert "no installed operator anchor" in refusal, refusal


def test_the_pairing_prompt_qualifies_its_promise_on_this_host(
    paired_machine: OperatorAnchor, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UX round 8, U8-2: the moment consent is given carries the same facts as the receipt.

    The `Authorise it? [y/N]` prompt is where the operator decides, and on a host
    whose anchor is staged but not installed it promised authority that cannot be
    exercised yet — the qualification arrived one step LATER, in the receipt, after
    the answer. Same predicate the receipt uses (`operator_authority_unusable`), one
    surface earlier. Both directions are asserted, because a prompt that hedged on an
    anchored host would be the same defect mirrored.
    """
    import local_operator.operator as operator_pkg
    from local_operator.operator import trust
    from local_operator.operator.pair_handlers import _confirm

    root = _config()
    # Annotated, not inferred: `dict[str, str]` is not assignable to the
    # `dict[str, object]` that `_confirm` takes (a mutable value type is invariant),
    # and pyright says so at the call rather than at the literal.
    row: dict[str, object] = {"name": "phone", "device_id": "ab" * 8}
    monkeypatch.setattr("builtins.input", lambda *_args: "n")

    def installed(uid: Any = None) -> Any:
        return _installed_anchor(trust, root, uid)

    monkeypatch.setattr(trust, "load_anchor", installed)
    monkeypatch.setattr(operator_pkg, "load_anchor", installed)
    assert _confirm(row) is False
    qualified = capsys.readouterr().out
    assert "lop operator install" not in qualified, qualified
    assert "This lets that device APPROVE" in qualified, qualified

    absent = lambda uid=None: _absent_anchor(trust, uid)  # noqa: E731 - a seam, not a style
    monkeypatch.setattr(trust, "load_anchor", absent)
    monkeypatch.setattr(operator_pkg, "load_anchor", absent)
    assert _confirm(row) is False
    unready = capsys.readouterr().out
    assert "once this machine's operator authority is installed" in unready, unready
    assert "lop operator install" in unready, unready


def test_authorising_a_device_lifts_both_halves_of_its_revocation(
    paired_machine: OperatorAnchor, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R9-2/Q9-1: the route the phone's refusal names has to actually lift the state.

    Measured before the fix: the relay records a revocation in TWO places — its own
    ``revoked.json`` under the config root and the root-owned anchor — and nothing in
    the product removed an entry from either, so the sentence that sent the operator
    to `lop operator init` + `install` left the phone refused (403 on a genuinely new
    anchor, too; the anchor's own half is asserted in
    ``test_operator_devices.py``). This drives the inverse verb through the same CLI
    entry point ``lop operator devices`` uses and asserts every surface it touches:
    the local record, the anchor's list, and the guard that decides the re-pair.
    """
    import local_operator.operator as operator_pkg
    from local_operator.operator import trust
    from local_operator.operator.pair_handlers import (
        _stage_anchor_revocation,
        describe_devices,
    )

    root = _config()
    _, point = _phone_point()
    device_id = devices.new_device_id(point)
    _, other_point = _phone_point()
    other_id = devices.new_device_id(other_point)

    def installed(uid: Any = None) -> Any:
        return _installed_anchor(trust, root, uid)

    def fake_install(config_root: Path, *, print_only: bool = False) -> int:
        """The privileged step, stubbed: this rig's anchor IS the staged file.

        Needed since round 10 (UX U2) because the SUCCESS receipt now belongs to the
        run that actually lands the change: a `--print-only` run reports the state it
        left, which is the subject of its own cell. The real install is a sudo call and
        no cell here may raise a password prompt.
        """
        return 0

    monkeypatch.setattr(trust, "load_anchor", installed)
    monkeypatch.setattr(operator_pkg, "load_anchor", installed)
    monkeypatch.setattr("local_operator.operator.handlers.install_anchor", fake_install)

    # The state `lop operator devices --revoke` leaves: both halves, plus the card.
    devices.record_revocation(root, device_id)
    devices.record_revocation(root, other_id)
    assert _stage_anchor_revocation(root, device_id, revoked=True) is True
    assert _stage_anchor_revocation(root, other_id, revoked=True) is True
    assert devices.is_revoked(root, device_id) is True

    assert describe_devices(_args(authorise=device_id, print_only=False)) == 0
    receipt = capsys.readouterr().out
    assert f"authorised {device_id}" in receipt, receipt
    assert "lop pair" in receipt, receipt

    assert devices.is_revoked_here(root, device_id) is False, "the local record survived"
    assert devices.is_revoked(root, device_id) is False, "the guard still refuses it"
    assert devices.is_revoked(root, other_id) is True, "another device was released with it"

    # Idempotent: a second run on an unrevoked device says so rather than pretending.
    assert describe_devices(_args(authorise=device_id, print_only=False)) == 0
    assert "nothing recorded a revocation" in capsys.readouterr().out


def test_a_revoked_device_keeps_a_row_and_the_command_that_brings_it_back(
    paired_machine: OperatorAnchor, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UX round 10, U1: the operand every remedy names has to be readable somewhere.

    Measured before: after a revocation the device appeared in NO listing — the
    revocation clears its certificate, so `paired` cannot hold it, and nothing printed
    the anchor-only revocations — which left the 32-hex id readable only in a receipt
    from an earlier day, and left the phone's own user unable to name their device at
    all. The row is asserted where a person looks for it, WITH the command, because
    being brought back is the reason the row exists.
    """
    import local_operator.operator as operator_pkg
    from local_operator.operator import trust
    from local_operator.operator.pair_handlers import (
        _stage_anchor_revocation,
        describe_devices,
    )

    root = _config()
    _, point = _phone_point()
    device_id = devices.new_device_id(point)

    def installed(uid: Any = None) -> Any:
        return _installed_anchor(trust, root, uid)

    monkeypatch.setattr(trust, "load_anchor", installed)
    monkeypatch.setattr(operator_pkg, "load_anchor", installed)

    # The state `--revoke` leaves behind: named in the anchor, certificate gone.
    assert _stage_anchor_revocation(root, device_id, revoked=True) is True
    devices.record_revocation(root, device_id)
    assert devices.list_devices(root) == []

    assert describe_devices(_args(print_only=True)) == 0
    listing = capsys.readouterr().out
    assert device_id in listing, listing
    assert f"lop operator devices --authorise {device_id}" in listing, listing
    assert "revoked" in listing, listing


def test_authorise_print_only_is_a_preview_that_changes_nothing(
    paired_machine: OperatorAnchor, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UX round 11, U5: a dry run that acts is not a dry run.

    TWO ROUNDS OF THIS VERB'S HISTORY ARE IN THIS CELL. Round 10's U2 was that the dry
    run printed the success sentence while `is_revoked` was still True; the fix added the
    caveat — but left the local clear ABOVE the `print_only` branch, so the preview
    really did lift that half. In the state `--revoke --print-only` itself creates (the
    local record is the only holder) the "preview" had therefore already un-revoked the
    device while its own sentence said the device was still refused: a receipt
    contradicting the state it left, one layer down.

    The rig separates STAGED from INSTALLED, which the neighbouring cells do not need to:
    they read the staged file as the anchor, so a staged change looks applied and the
    mutation this cell exists for would be invisible.
    """
    import json

    import local_operator.operator as operator_pkg
    from local_operator.operator import handlers, trust
    from local_operator.operator.pair_handlers import (
        _stage_anchor_revocation,
        describe_devices,
    )

    root = _config()
    _, point = _phone_point()
    device_id = devices.new_device_id(point)
    installed_path = root / "installed-anchor.json"

    def installed(uid: Any = None) -> Any:
        body = json.loads(installed_path.read_text())
        return trust.AnchorLoad(
            anchor=trust.OperatorAnchor.from_json(body),
            path=trust.anchor_path(uid),
            root_owned=True,
            reason="ok",
            exists=True,
        )

    real_install = handlers.install_anchor

    def fake_install(config_root: Path, *, print_only: bool = False) -> int:
        """Models the ONE privileged step — and delegates the PRINT half to the product.

        The preview is asserted to print the privileged command, so the rig must not
        stand in for the code that prints it: `print_only=True` goes to the real
        implementation (which writes nothing), and only the taking of the step is
        simulated, because that is the sudo call no cell here may raise.
        """
        if print_only:
            return real_install(config_root, print_only=True)
        installed_path.write_bytes(trust.staging_path(config_root).read_bytes())
        return 0

    installed_path.write_bytes(trust.staging_path(root).read_bytes())
    monkeypatch.setattr(trust, "load_anchor", installed)
    monkeypatch.setattr(operator_pkg, "load_anchor", installed)
    monkeypatch.setattr(handlers, "install_anchor", fake_install)

    assert _stage_anchor_revocation(root, device_id, revoked=True) is True
    installed_path.write_bytes(trust.staging_path(root).read_bytes())
    devices.record_revocation(root, device_id)
    assert devices.is_revoked(root, device_id) is True

    # THE WHOLE STATE, byte for byte, around the preview.
    before = {
        "record": devices.revoked_path(root).read_bytes(),
        "staged": trust.staging_path(root).read_bytes(),
        "is_revoked": devices.is_revoked(root, device_id),
        "here": devices.is_revoked_here(root, device_id),
    }
    assert describe_devices(_args(authorise=device_id, print_only=True)) == 0
    preview = capsys.readouterr().out
    after = {
        "record": devices.revoked_path(root).read_bytes(),
        "staged": trust.staging_path(root).read_bytes(),
        "is_revoked": devices.is_revoked(root, device_id),
        "here": devices.is_revoked_here(root, device_id),
    }
    assert before == after, f"the preview changed the state: {before} -> {after}"
    # AND THE PROMISED OUTPUT IS THERE. The flag's own help says "print the privileged
    # command instead of running it", and round 11's fix kept the "instead of running it"
    # half while losing this one: the early return removed the command, the listing rows
    # and the next step, so the two previews of one verb family disagreed (measured by
    # both UX and the reviewer — 0 `sudo install` lines against `--revoke`'s one, and no
    # device table at all). Both halves are pinned here so neither can be fixed by
    # breaking the other.
    assert "preview:" in preview, preview
    assert "sudo install" in preview, preview
    assert "NOTHING has been changed" in preview, preview
    assert "without --print-only" in preview, preview
    assert f"{device_id}" in preview and "certificate cleared" in preview, preview
    assert "accepted at once" not in preview, preview
    assert devices.AUTHORISE_COMMAND.format(device_id=device_id) in preview, preview

    # ...and the real run does lift both halves, so the preview is a preview and not a
    # verb that cannot act.
    assert describe_devices(_args(authorise=device_id, print_only=False)) == 0
    capsys.readouterr()
    assert devices.is_revoked(root, device_id) is False
    assert devices.revoked_path(root).exists() is False

    # (b) and with nothing left to lift, the preview says so and offers no command: a
    # reader must not be sent at a privileged step that would do nothing.
    assert describe_devices(_args(authorise=device_id, print_only=True)) == 0
    idle_preview = capsys.readouterr().out
    assert "nothing recorded a revocation" in idle_preview, idle_preview
    assert "no privileged step would be needed" in idle_preview, idle_preview
    assert "sudo install" not in idle_preview, idle_preview


def test_authorise_does_not_claim_the_lift_when_the_local_record_survives(
    paired_machine: OperatorAnchor, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UX round 11 U6 = QA Q11-1 = reviewer R11-1: the stdout half of a failed clear.

    With the operator root unwritable, the run warned on stderr and then printed
    "authorised … a new pairing request is accepted at once" on stdout, two lines above
    a listing row calling the device revoked — and the phone was still refused (403 on a
    fresh code, measured by both rounds). `is_revoked` is the OR of the two halves, so
    the local record alone still refuses the device and the receipt has to say so.
    Checked in BOTH branches, because the reviewer measured both: with an installed
    anchor, and on the no-anchor arm whose sentence claimed "only the local record was
    cleared".
    """
    import json
    import os as _os

    import local_operator.operator as operator_pkg
    from local_operator.operator import handlers, trust
    from local_operator.operator.pair_handlers import (
        _stage_anchor_revocation,
        describe_devices,
    )

    root = _config()
    _, point = _phone_point()
    device_id = devices.new_device_id(point)
    installed_path = root / "installed-anchor.json"

    def installed(uid: Any = None) -> Any:
        body = json.loads(installed_path.read_text())
        return trust.AnchorLoad(
            anchor=trust.OperatorAnchor.from_json(body),
            path=trust.anchor_path(uid),
            root_owned=True,
            reason="ok",
            exists=True,
        )

    def fake_install(config_root: Path, *, print_only: bool = False) -> int:
        if not print_only:
            installed_path.write_bytes(trust.staging_path(config_root).read_bytes())
        return 0

    installed_path.write_bytes(trust.staging_path(root).read_bytes())
    monkeypatch.setattr(trust, "load_anchor", installed)
    monkeypatch.setattr(operator_pkg, "load_anchor", installed)
    monkeypatch.setattr(handlers, "install_anchor", fake_install)
    assert _stage_anchor_revocation(root, device_id, revoked=True) is True
    installed_path.write_bytes(trust.staging_path(root).read_bytes())
    devices.record_revocation(root, device_id)

    operator_dir = devices.operator_root(root)
    try:
        # (a) installed anchor: the anchor half IS lifted, the local half is not. The
        # directory is re-tightened before each run because `stage_anchor` chmods its
        # own parent back to 0700 as it writes (pre-existing behaviour), so one chmod
        # would only make the FIRST run unwritable.
        _os.chmod(operator_dir, 0o500)
        assert describe_devices(_args(authorise=device_id)) == 0
        captured = capsys.readouterr()
        assert "still names" in captured.err, captured.err
        assert "accepted at once" not in captured.out, captured.out
        assert "STILL refused" in captured.out, captured.out
        assert devices.is_revoked(root, device_id) is True, "the guard was released anyway"
        assert devices.revoked_path(root).exists() is True

        # (b) the no-anchor arm, whose sentence claimed a clear that did not happen.
        monkeypatch.setattr(
            trust,
            "load_anchor",
            lambda uid=None: trust.AnchorLoad(
                anchor=None,
                path=trust.anchor_path(uid),
                root_owned=False,
                reason="pinned absent by the test",
                exists=False,
            ),
        )
        monkeypatch.setattr(
            operator_pkg,
            "load_anchor",
            lambda uid=None: trust.AnchorLoad(
                anchor=None,
                path=trust.anchor_path(uid),
                root_owned=False,
                reason="pinned absent by the test",
                exists=False,
            ),
        )
        # An UNSTAMPED record, so it still applies with no anchor installed (the stamp
        # is compared against the installed key id, and "no anchor" answers `""`); this
        # is the half the reviewer measured, where the arm's sentence claimed the local
        # record had been cleared.
        devices.revoked_path(root).write_text(json.dumps({"v": 1, "devices": [device_id]}))
        _os.chmod(operator_dir, 0o500)
        assert describe_devices(_args(authorise=device_id)) == 0
        captured = capsys.readouterr()
        assert "no installed operator anchor" in captured.err, captured.err
        assert "only the local record was cleared" not in captured.err, captured.err
        assert "STILL refused" in captured.err, captured.err
    finally:
        _os.chmod(operator_dir, 0o700)


def test_a_record_that_did_not_apply_is_not_reported_as_never_recorded(
    paired_machine: OperatorAnchor, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agent review round 10, NIT-2: the wording, about a record that DID exist.

    A record stamped with a key that is no longer installed does not apply
    (`is_revoked_here`'s scoping, round 9) while still being a record on disk. The verb
    said "nothing recorded a revocation of X" about it — accurate about the guard,
    wrong about the file — while the same run cleared it.
    """
    import json

    import local_operator.operator as operator_pkg
    from local_operator.operator import trust
    from local_operator.operator.pair_handlers import describe_devices

    root = _config()
    _, point = _phone_point()
    device_id = devices.new_device_id(point)

    def installed(uid: Any = None) -> Any:
        return _installed_anchor(trust, root, uid)

    monkeypatch.setattr(trust, "load_anchor", installed)
    monkeypatch.setattr(operator_pkg, "load_anchor", installed)
    devices.revoked_path(root).write_text(
        json.dumps({"v": 1, "devices": [device_id], "operator_key_id": "not-the-installed-key"})
    )
    assert devices.is_revoked(root, device_id) is False, "the fixture's record does apply"

    # The REAL run, not a preview: since round 11 a `--print-only` run changes nothing,
    # so it is the real one that clears this record (and this state's branch never
    # reaches the install step, so no privileged command is offered or run).
    assert describe_devices(_args(authorise=device_id, print_only=False)) == 0
    said = capsys.readouterr().out
    assert "did not apply under the installed anchor" in said, said
    assert "nothing recorded a revocation" not in said, said
    assert devices.revoked_path(root).exists() is False


def test_the_authorise_preview_promises_exactly_what_its_own_run_does(
    paired_machine: OperatorAnchor, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Design round 13, D1 (MAJOR): the promise must equal the action, in every state.

    Measured on the previous head, with a local record but no USABLE anchor: the preview
    promised a statement write and "ONE privileged step", printed the `sudo install` line
    and sent the reader to re-run without `--print-only` — while the real run in that same
    state wrote nothing, took no step and said "run `lop operator install`". The promise
    outran the run, which is the class of defect rounds 10-12 were about, and it survived
    because each side was pinned by its own wording rather than against the other.

    Pinned as a PROPERTY now, in every state this verb models: the two things the preview
    claims — whether it would clear the local record, and whether a privileged step exists
    — are compared with what the real run actually does. Wording changes cannot make the
    two sides agree falsely, because the comparison is between the command the PRODUCT
    prints and the install the stub was actually asked to perform.
    """
    import dataclasses
    import json

    import local_operator.operator as operator_pkg
    from local_operator.operator import handlers, trust
    from local_operator.operator.pair_handlers import describe_devices

    root = _config()
    _, point = _phone_point()
    device_id = devices.new_device_id(point)
    installed_path = root / "installed-anchor.json"
    calls: list[str] = []

    # The INSTALLED statement is its own file, so a preview that writes nothing cannot
    # move it: the state the run acts on is separated from the state it would write.
    installed_path.write_bytes(
        trust.anchor_bytes(
            dataclasses.replace(
                paired_machine, devices=({"device_id": device_id, "revoked": True},)
            )
        )
    )

    def seam(uid: Any = None) -> Any:
        body = json.loads(installed_path.read_text())
        return trust.AnchorLoad(
            anchor=trust.OperatorAnchor.from_json(body),
            path=trust.anchor_path(uid),
            root_owned=True,
            reason="ok",
            exists=True,
        )

    absent = lambda uid=None: trust.AnchorLoad(  # noqa: E731 — a seam, not a style
        anchor=None,
        path=trust.anchor_path(uid),
        root_owned=False,
        reason="pinned absent by the test",
        exists=False,
    )
    real_install = handlers.install_anchor

    def fake_install(config_root: Path, *, print_only: bool = False) -> int:
        if print_only:
            return real_install(config_root, print_only=True)
        calls.append("installed")
        installed_path.write_bytes(trust.staging_path(config_root).read_bytes())
        return 0

    monkeypatch.setattr(handlers, "install_anchor", fake_install)

    def observe(*, preview: bool, anchor_usable: bool, staged: bool = True) -> dict[str, Any]:
        """One fresh fixture, one invocation, and what it claimed or did."""
        staged_path = trust.staging_path(root)
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        if staged:
            staged_path.write_bytes(trust.anchor_bytes(paired_machine))
        else:
            staged_path.unlink(missing_ok=True)
        devices.record_revocation(root, device_id)
        usable = seam if anchor_usable else absent
        monkeypatch.setattr(trust, "load_anchor", usable)
        monkeypatch.setattr(operator_pkg, "load_anchor", usable)
        calls.clear()
        before = staged_path.read_bytes() if staged else None
        capsys.readouterr()
        assert describe_devices(_args(authorise=device_id, print_only=preview)) == 0
        out = capsys.readouterr().out
        return {
            "claims_local_clear": ("would clear the local revocation record" in out)
            or ("only the local record was cleared" in out),
            "claims_privileged_step": "sudo install" in out,
            "record_gone": not devices.revoked_path(root).exists(),
            "staged_changed": (staged_path.read_bytes() if staged else None) != before,
            "stepped": bool(calls),
            "out": out,
        }

    # (a) the state D1 measured: a local record, and no usable anchor.
    preview = observe(preview=True, anchor_usable=False)
    assert preview["record_gone"] is False and preview["staged_changed"] is False
    assert preview["stepped"] is False, "the preview took the privileged step"
    assert preview["claims_privileged_step"] is False, preview["out"]
    assert "no usable" in preview["out"] and "lop operator install" in preview["out"]
    real = observe(preview=False, anchor_usable=False)
    assert preview["claims_local_clear"] == real["record_gone"] is True
    assert preview["claims_privileged_step"] == real["stepped"], (
        "the preview's promise and the real run's action disagree: "
        f"{preview['claims_privileged_step']} vs {real['stepped']}"
    )
    assert real["stepped"] is False and real["record_gone"] is True

    # (b) the ordinary state: both halves revoked, anchor usable.
    preview = observe(preview=True, anchor_usable=True)
    assert preview["record_gone"] is False and preview["staged_changed"] is False
    real = observe(preview=False, anchor_usable=True)
    assert preview["claims_local_clear"] == real["record_gone"] is True
    assert preview["claims_privileged_step"] == real["stepped"] is True

    # (c) and nothing staged: the privileged step cannot be offered, and the remedy rides
    # the same stream in the right order (design round 13, D5) instead of arriving on
    # stderr before this block in a piped run.
    preview = observe(preview=True, anchor_usable=True, staged=False)
    assert preview["claims_privileged_step"] is False, preview["out"]
    assert "lop operator init" in preview["out"], preview["out"]
    assert "nothing is staged to install yet" not in preview["out"], preview["out"]

    # (d) the caveat travels with the command (design round 13, D2): it used to sit
    # fifteen rows below it at 44 columns.
    ordered = observe(preview=True, anchor_usable=True)["out"]
    assert ordered.index("NOTHING has been changed by this run") < ordered.index("sudo install")
