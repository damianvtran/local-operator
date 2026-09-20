"""The paired-device store and the pairing handshake (stage D, issue #1310).

WHAT THESE CELLS ARE FOR. The device tier's security claim is narrow and must be
stated exactly: the certificate lives under the operator's OWN config root, where
a same-uid subject can write it — so the file's PATH defends nothing, and the
SIGNATURE is the whole of the defence. Two tests therefore matter more than the
rest and are named for it:

* a SUBSTITUTED certificate (the file's contents replaced with a statement the
  operator never signed) fails verification against the anchored operator key;
* a FORGED certificate (signed by a key the machine has never seen) fails the
  same check.

Everything else here is the mechanics those two rest on: the id is DERIVED from
the key so a device cannot relabel itself, the modes are 0644-under-0700, the
pairing code is single-device and expiring, and a revocation is recorded where it
cannot be un-recorded.

NOTHING IN THIS FILE TOUCHES A REAL KEYCHAIN. The ``file-only`` backend is the
only one a test may create (the Secure Enclave cannot be created in a throwaway
keychain at all, and writing the operator's login keychain is forbidden), and the
anchor root is patched in-process so no privileged path is ever written.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from local_operator.operator import devices
from local_operator.operator.keychain import FILE_ONLY
from local_operator.operator.sign import (
    Signer,
    anchor_for_handle,
    create_key,
    issue_device_cert,
    load_signer,
)
from local_operator.operator.trust import (
    OperatorAnchor,
    anchor_bytes,
    device_is_revoked,
)
from local_operator.operator.verify import (
    DeviceCert,
    key_id_for,
    read_device_cert,
    verify_device_cert,
)


@pytest.fixture()
def operator_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[OperatorAnchor]:
    """A real operator key (``file-only``) plus the anchor that names it.

    The anchor is installed at a patched, test-owned root so ``load_anchor()`` —
    which the pairing flow and the verifier both read — sees exactly what a real
    install would put there. Patching the module global rather than an environment
    variable is the seam ``trust`` deliberately exposes: an env-redirectable
    anchor path would be the substitution attack that module exists to prevent.
    """
    anchor_root = tmp_path / "anchor-root"
    anchor_root.mkdir()
    monkeypatch.setattr(
        "local_operator.operator.trust._ANCHOR_ROOT_OVERRIDE", anchor_root, raising=False
    )
    handle = create_key(config_root=tmp_path, preference=FILE_ONLY)
    anchor = anchor_for_handle(handle, label="device-store-test")
    (anchor_root / "operator.json").write_bytes(anchor_bytes(anchor))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    yield anchor


def _new_point() -> tuple[Any, bytes]:
    """A fresh ES256 key and its uncompressed P-256 point (the wire form)."""
    key = ec.generate_private_key(ec.SECP256R1())
    point = key.public_key().public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)
    return key, point


def _operator_signed(config_root: Path, point: bytes, *, label: str = "phone") -> str:
    """A certificate signed by the operator key the fixture installed."""
    signer = load_signer(config_root=config_root, backend_name=FILE_ONLY)
    assert signer is not None, "the fixture's operator key did not load"
    try:
        return _sign_statement(signer, point, label=label)
    finally:
        signer.close()


def _sign_statement(signer: Signer, point: bytes, *, label: str) -> str:
    return issue_device_cert(
        device_spki=point,
        device_id=key_id_for(point),
        label=label,
        signer=signer,
    )


def _store(config_root: Path, certificate: str, *, name: str = "phone") -> devices.StoredDevice:
    parsed = read_device_cert(certificate)
    assert parsed is not None, "the rig produced a certificate it cannot parse"
    return devices.write_device_cert(
        config_root,
        certificate=certificate,
        parsed=parsed,
        operator_key_id="",
        name=name,
    )


# ---------------------------------------------------------------------------
# The pairing code
# ---------------------------------------------------------------------------


def test_a_pairing_code_expires_and_is_burned_by_use(tmp_path: Path) -> None:
    """One code, one device, one window.

    The code is the only thing bounding WHICH phone may claim a pairing, so two
    properties carry the weight: it stops working on its own (an operator who left
    ``lop pair`` running has not left a door open), and it goes away the moment the
    operator says it has been consumed (a code that survived its first use would
    let a second phone pair on a gesture the operator made for the first).
    """
    code = devices.begin_pairing(tmp_path, ttl_s=60, now=1_000.0)
    assert code and len(code) == 32
    assert devices.read_pairing(tmp_path, now=1_030.0) == code
    # Past its window: no code, which is the same answer as never having one.
    assert devices.read_pairing(tmp_path, now=1_061.0) is None

    devices.clear_pairing(tmp_path)
    assert devices.read_pairing(tmp_path, now=1_030.0) is None
    assert devices.pairing_path(tmp_path).exists() is False


def test_pending_requests_are_listed_oldest_first_and_name_their_own_device(
    tmp_path: Path,
) -> None:
    """The listing the operator reads before signing anything.

    ``name`` is attacker-supplied text shown on a screen next to a signing prompt,
    so the LIST must not become a way to smuggle a statement about a different
    device: a file whose name and contents disagree is dropped rather than
    reconciled.
    """
    _, first_point = _new_point()
    _, second_point = _new_point()
    first = devices.new_device_id(first_point)
    second = devices.new_device_id(second_point)
    devices.write_pending(
        tmp_path,
        device_id=first,
        name="first",
        spki=devices.encode_spki(first_point),
        code="c",
        now=100.0,
    )
    devices.write_pending(
        tmp_path,
        device_id=second,
        name="second",
        spki=devices.encode_spki(second_point),
        code="c",
        now=200.0,
    )
    assert [row["device_id"] for row in devices.list_pending(tmp_path)] == [first, second]

    planted = devices.pending_path(tmp_path, first)
    planted.write_text(json.dumps({"device_id": "someone-else", "spki": "x", "name": "y"}))
    assert [row["device_id"] for row in devices.list_pending(tmp_path)] == [second]


def test_a_device_id_cannot_name_a_path_outside_the_operator_root(tmp_path: Path) -> None:
    """A device id arrives over the wire and becomes a FILENAME.

    ``write_pending`` is reached from the relay's pairing endpoint, so a value
    containing a separator or a traversal segment would be a write primitive rather
    than a pairing request. Refused at the boundary that builds the path.
    """
    for bad in ("../escape", "a/b", "", "x" * 200):
        with pytest.raises(ValueError):
            devices.write_pending(tmp_path, device_id=bad, name="n", spki="s", code="c")


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


def test_a_stored_certificate_is_readable_and_says_what_it_authorises(
    tmp_path: Path, operator_key: OperatorAnchor
) -> None:
    """The record the relay declares from and the operator reads.

    Modes are pinned because the design states them (0644 under 0700) and because
    they are the OPPOSITE of the anchor's: this file is readable on purpose — public
    data plus a signature over it — and a later change that quietly made it 0600
    would break the relay without breaking any other assertion here.
    """
    _, point = _new_point()
    certificate = _operator_signed(tmp_path, point)
    stored = _store(tmp_path, certificate, name="Damian's phone")

    assert stored.device_id == key_id_for(point)
    assert stored.scope == devices.DEVICE_SCOPES == ("loosen", "approve")
    assert stored.certificate == certificate

    path = devices.device_path(tmp_path, stored.device_id)
    assert oct(devices.device_certificate_mode(path)) == "0o644"
    assert oct(path.parent.stat().st_mode & 0o777) == "0o700"
    assert devices.read_device(tmp_path, stored.device_id) == stored
    assert devices.list_devices(tmp_path) == [stored]
    assert devices.paired_certificate(tmp_path) == certificate

    # PUBLIC DATA ONLY. A private half here would be a stored credential, which is
    # the one thing this tier must never hold — the whole point is that reading
    # this file is worthless without the phone.
    body = json.loads(path.read_text(encoding="utf-8"))
    assert set(body) == {
        "v",
        "kind",
        "device_id",
        "name",
        "spki",
        "key_id",
        "scope",
        "iat",
        "exp",
        "operator_key_id",
        "certificate",
    }


def test_a_substituted_certificate_fails_verification(
    tmp_path: Path, operator_key: OperatorAnchor
) -> None:
    """THE SECURITY CLAIM OF THIS MODULE, pinned (design §2.1 / §2.2 / §4).

    The store is under the operator's own config root, which a same-uid subject can
    write — including the model-authored tool child #1310 exists for. So this test
    writes exactly what that subject could write: a well-formed certificate naming a
    real P-256 point, signed by a key of THEIR choosing, into the file the runtime
    reads. ``verify_device_cert`` checks it against the ANCHORED operator key and
    refuses.

    That is why the path needs no protection and the signature carries the whole
    claim — and why a change that "simplified" the check to a shape check would be
    caught here rather than in production.
    """
    _, point = _new_point()
    certificate = _operator_signed(tmp_path, point)
    stored = _store(tmp_path, certificate)
    now = int(time.time())
    assert verify_device_cert(certificate, operator_spki=operator_key.spki, now=now) is not None

    # The substitution: same path, same shape, the same point — signed by a key
    # that is not the operator's.
    attacker_key = ec.generate_private_key(ec.SECP256R1())
    statement = DeviceCert(
        device_id=stored.device_id,
        spki=point,
        label="phone",
        issued_at=now,
        not_after=now + 3600,
    )
    substituted = statement.encode(
        signature=attacker_key.sign(statement.payload(), ec.ECDSA(hashes.SHA256()))
    )
    path = devices.device_path(tmp_path, stored.device_id)
    body = json.loads(path.read_text(encoding="utf-8"))
    body["certificate"] = substituted
    path.write_text(json.dumps(body), encoding="utf-8")

    reread = devices.read_device(tmp_path, stored.device_id)
    assert (
        reread is not None and reread.certificate == substituted
    ), "the rig did not actually substitute anything"
    assert (
        verify_device_cert(substituted, operator_spki=operator_key.spki, now=now) is None
    ), "a certificate signed by another key verified against the anchored operator key"


def test_a_forged_and_an_expired_certificate_are_both_refused(
    tmp_path: Path, operator_key: OperatorAnchor
) -> None:
    """The same claim from the other direction, plus the lifetime.

    A certificate signed by a key the machine has never seen, over a real P-256
    point and with a correct derived id, must fail against the anchor. And an
    expired one must fail too, so a certificate's lifetime is a real bound rather
    than a field nobody reads — otherwise a phone from three years ago keeps
    working because nobody deleted its file.
    """
    attacker_key, attacker_point = _new_point()
    now = int(time.time())

    def statement(*, issued_at: int, not_after: int) -> DeviceCert:
        return DeviceCert(
            device_id=key_id_for(attacker_point),
            spki=attacker_point,
            label="not my phone",
            issued_at=issued_at,
            not_after=not_after,
        )

    forged = statement(issued_at=now, not_after=now + 3600)
    assert (
        verify_device_cert(
            forged.encode(signature=attacker_key.sign(forged.payload(), ec.ECDSA(hashes.SHA256()))),
            operator_spki=operator_key.spki,
            now=now,
        )
        is None
    )

    signer = load_signer(config_root=tmp_path, backend_name=FILE_ONLY)
    assert signer is not None
    try:
        expired = statement(issued_at=now - 7200, not_after=now - 3600)
        expired_certificate = expired.encode(signature=signer.sign(expired.payload()))
    finally:
        signer.close()
    assert (
        verify_device_cert(expired_certificate, operator_spki=operator_key.spki, now=now) is None
    ), "an expired certificate verified"


# ---------------------------------------------------------------------------
# Revocation
# ---------------------------------------------------------------------------


def test_revocation_drops_the_certificate_and_the_anchor_is_authoritative(
    tmp_path: Path, operator_key: OperatorAnchor
) -> None:
    """Availability is the residual; UNAVAILABILITY must be complete.

    A revoked device must not come back by pairing again under a new NAME, which is
    why the id is derived from the key rather than chosen — the relay asks the
    revocation list about the derived id and refuses. And the list the RUNTIME
    consults is the root-owned anchor's, recorded through the same privileged step
    as the anchor itself, so the revoked subject cannot un-revoke it.
    """
    _, point = _new_point()
    stored = _store(tmp_path, _operator_signed(tmp_path, point), name="lost phone")

    devices.record_revocation(tmp_path, stored.device_id)
    assert devices.is_revoked_here(tmp_path, stored.device_id) is True
    assert devices.read_device(tmp_path, stored.device_id) is None, "the certificate survived"
    assert devices.paired_certificate(tmp_path) is None

    revoking = dataclasses.replace(
        operator_key, devices=({"device_id": stored.device_id, "revoked": True},)
    )
    assert device_is_revoked(revoking, stored.device_id) is True
    assert device_is_revoked(revoking, "some-other-device") is False
    # An anchor written before device support has NO OPINION, which is deliberately
    # different from a list that names the device.
    assert device_is_revoked(operator_key, stored.device_id) is False


def test_an_unreadable_or_planted_file_is_not_a_device(tmp_path: Path) -> None:
    """Malformed input degrades to ABSENT, never to a partially-trusted device."""
    path = devices.devices_dir(tmp_path) / "planted.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert devices.list_devices(tmp_path) == []
    assert devices.read_device(tmp_path, "planted") is None

    path.write_text(json.dumps({"kind": "operator", "v": 1}), encoding="utf-8")
    assert devices.read_device(tmp_path, "planted") is None
    assert devices.paired_certificate(tmp_path) is None


def test_the_spki_decoder_accepts_only_a_p256_point() -> None:
    """One encoder, one decoder, and only the uncompressed form.

    The pairing endpoint is the first boundary a phone's public key crosses, and a
    length or prefix check that drifted from the verifier's would accept a value
    that later fails deep inside a signing path — after the operator has already
    been asked for a presence gesture.
    """
    _, point = _new_point()
    encoded = devices.encode_spki(point)
    assert devices.decode_spki(encoded) == point
    assert devices.decode_spki("not-base64!!") is None
    assert devices.decode_spki(base64.urlsafe_b64encode(b"\x04" + b"\x00" * 10).decode()) is None
    assert devices.decode_spki("") is None
    assert devices.decode_spki(None) is None
    assert devices.decode_spki(b"bytes") is None
    assert devices.decode_spki("A" * 600) is None


def test_a_local_record_is_scoped_to_the_anchor_that_stamped_it(
    tmp_path: Path, operator_key: OperatorAnchor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R9-2/Q9-1: the local record must not outlive the anchor it was written under.

    Measured defect (round 9): the relay keeps its own revocation copy so the common
    case needs no privileged read, nothing in the product removed an entry from it,
    and it was honoured unconditionally — so an operator who followed the ONLY route
    the product then named (create a genuinely new anchor) had a phone that was still
    refused, because this file still named it. A revocation is a statement by the
    operator key that signed the certificate, so a record stamped with a key that is
    no longer installed describes devices of an anchor that is gone.

    Both directions are asserted, because the wrong way to fail here is to lift a
    revocation an operator really did make: an UNSTAMPED record (one written before
    the stamp existed) keeps being honoured.
    """
    from local_operator.operator import trust
    from local_operator.operator.keychain import FILE_ONLY
    from local_operator.operator.sign import anchor_for_handle, create_key

    anchor_root = tmp_path / "anchor-root"
    anchor_root.mkdir(exist_ok=True)
    monkeypatch.setattr(
        "local_operator.operator.trust._ANCHOR_ROOT_OVERRIDE", anchor_root, raising=False
    )
    _, point = _new_point()
    stored = _store(tmp_path, _operator_signed(tmp_path, point), name="lost phone")

    def stage(anchor: OperatorAnchor) -> None:
        """The staging half of an install: the file the privileged step would move."""
        staged = trust.staging_path(tmp_path)
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_bytes(trust.anchor_bytes(anchor))

    def installed_from_staged(uid: Any = None) -> Any:
        body = json.loads(trust.staging_path(tmp_path).read_text())
        return trust.AnchorLoad(
            anchor=trust.OperatorAnchor.from_json(body),
            path=trust.anchor_path(uid),
            root_owned=True,
            reason="ok",
            exists=True,
        )

    stage(operator_key)
    monkeypatch.setattr(trust, "load_anchor", installed_from_staged)

    devices.record_revocation(tmp_path, stored.device_id)
    assert devices.is_revoked_here(tmp_path, stored.device_id) is True
    assert (
        json.loads(devices.revoked_path(tmp_path).read_text())["operator_key_id"]
        == operator_key.key_id
    ), "the record was written without the key id it belongs to"

    # A GENUINELY NEW ANCHOR: the record is about a key that is no longer installed.
    fresh = anchor_for_handle(create_key(config_root=tmp_path / "fresh", preference=FILE_ONLY))
    stage(fresh)
    assert fresh.key_id != operator_key.key_id
    assert devices.is_revoked_here(tmp_path, stored.device_id) is False

    # ...and an unstamped record is still honoured, which is the safe direction.
    devices.revoked_path(tmp_path).write_text(json.dumps({"v": 1, "devices": [stored.device_id]}))
    assert devices.is_revoked_here(tmp_path, stored.device_id) is True


def test_forget_revocation_lifts_the_record_and_takes_the_file_with_it(tmp_path: Path) -> None:
    """The inverse of ``record_revocation``, host-side, and complete when it is last.

    The file going away matters as much as the entry: a leftover ``{"devices": []}``
    reads as "nothing is revoked, and there was a list", which is a state a future
    reader has to reason about for no benefit.
    """
    _, point = _new_point()
    first = devices.new_device_id(point)
    _, other = _new_point()
    second = devices.new_device_id(other)

    devices.record_revocation(tmp_path, first)
    devices.record_revocation(tmp_path, second)
    assert devices.forget_revocation(tmp_path, first) is True
    assert devices.is_revoked_here(tmp_path, first) is False
    assert devices.is_revoked_here(tmp_path, second) is True, "the other device was released"

    assert devices.forget_revocation(tmp_path, second) is True
    assert devices.revoked_path(tmp_path).exists() is False
    # Idempotent: lifting a revocation that is not recorded is False, not an error.
    assert devices.forget_revocation(tmp_path, second) is False


def test_a_record_that_cannot_be_removed_is_reported_rather_than_swallowed(
    tmp_path: Path,
) -> None:
    """Agent review round 10, NIT-1: the receipt for a clear that did not happen.

    `forget_revocation` swallowed an `OSError` around the unlink and returned `True`,
    so with the operator root unwritable the verb printed "only the local record was
    cleared" while `revoked.json` still named the device and `is_revoked_here` was
    still `True` — on the one message a refused phone's operator is sent to follow.
    (``record_revocation`` swallows the same error for the CERTIFICATE, which is a file
    nothing reads for authority; this file is read, so the two are not the same case.)
    """
    import os as _os

    _, point = _new_point()
    device_id = devices.new_device_id(point)
    devices.record_revocation(tmp_path, device_id)
    assert devices.is_revoked_here(tmp_path, device_id) is True

    operator_dir = devices.operator_root(tmp_path)
    _os.chmod(operator_dir, 0o500)
    try:
        assert devices.forget_revocation(tmp_path, device_id) is False
        assert devices.is_revoked_here(tmp_path, device_id) is True, "the record went anyway"
        assert devices.revoked_path(tmp_path).exists() is True
    finally:
        _os.chmod(operator_dir, 0o700)
