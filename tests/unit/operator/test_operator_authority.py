"""Operator authority: the anchor, the key, the message and the verdict.

Issue #1310, revision 2. The invariant did not change — *a constrained subject
must not be able to mint the authority that removes its own approval
requirement* — but the authority did: it is now a fact about the operator, proved
by an ES256 signature over a per-action challenge, verified against a public key
pinned in a ROOT-OWNED file.

Three groups of claims, and each is the security half of a sentence the runtime
relies on:

1. **the anchor is a boundary only where it is root-owned**, and nothing the
   gated uid can write may redirect it. Both halves are asserted here against the
   real filesystem in ``tmp_path``, because "a same-uid file is not an anchor" is
   the one claim the whole design rests on and it is cheap to lose;
2. **the signed message binds every field**, so a signature harvested for one
   action, session, card or challenge does not verify for another;
3. **the verdict is fail-closed** at every malformed input — no anchor, a bad
   hex signature, a key id that is not the anchor's, a forged or expired device
   certificate, a revoked device.

Deliberately NOT here: the wire and the seam. Those need a real socket and a real
runtime, and they live in
``tests/unit/session/runtime/test_approval_authority_seam.py`` beside the
capability controls they extend. Nothing in this file touches the operator's
login keychain, their config, or their sessions: every key is created in a
``tmp_path`` through the ``file-only`` backend, which is a REAL backend (real
P-256, real ES256) rather than a test double — see ``operator/keychain.py``'s
module docstring for the measurement that makes the Secure Enclave untestable.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.operator import (
    LEVEL_ANCHOR_UNPINNED,
    LEVEL_OPERATOR_FILE_ONLY,
    LEVEL_OPERATOR_PRESENCE,
    LEVEL_SPAWN_ONLY,
    keychain,
    operator_authority_level,
    operator_authority_report,
    trust,
    verify,
)
from local_operator.operator.keychain import FILE_ONLY, FileKeyBackend
from local_operator.operator.sign import (
    anchor_for_handle,
    create_key,
    effect_copy,
    issue_device_cert,
    load_signer,
    sign_challenge,
)

#: The closed set of levels, spelled once for the vocabulary test below. The
#: values live in ``local_operator.operator``; naming them here a second time is
#: deliberate, because a set that grows without the test noticing is how a report
#: starts emitting a level no copy knows how to describe.
LEVELS = frozenset(
    {LEVEL_OPERATOR_PRESENCE, LEVEL_OPERATOR_FILE_ONLY, LEVEL_ANCHOR_UNPINNED, LEVEL_SPAWN_ONLY}
)

# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@pytest.fixture()
def keyed(tmp_path: Path) -> Any:
    """A real operator key in a throwaway file, its anchor, and a signer."""

    class Keyed:
        def __init__(self) -> None:
            self.root = tmp_path
            # The CANONICAL file path, not a fixture-chosen one: `sign_challenge`
            # resolves the backend by name and then asks THAT backend for its
            # default location, so a key written anywhere else would prove the
            # test can sign while the product cannot.
            self.handle = create_key(config_root=tmp_path, preference=FILE_ONLY)
            self.backend = FileKeyBackend(keychain.default_file_path(tmp_path))
            self.anchor = anchor_for_handle(self.handle, label="test")
            # The staged anchor, exactly as `lop operator init` leaves it: sign
            # without this and `resolve_backend_name` would fall to the host's
            # presence ladder and look in the Secure Enclave, which is the real
            # behaviour for a host with no key — and not what this fixture means.
            staged = trust.staging_path(tmp_path)
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(trust.anchor_bytes(self.anchor))
            self.signer = self.backend.load()

        def sign(self, **fields: str) -> dict[str, str]:
            challenge = fields.pop("challenge")
            return sign_challenge(
                challenge=challenge,
                config_root=tmp_path,
                **fields,
            ).as_json()

        def close(self) -> None:
            if self.signer is not None:
                self.signer.close()

    keyed = Keyed()
    try:
        yield keyed
    finally:
        keyed.close()


def _anchor_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    """Point the anchor loader at a throwaway directory.

    Patching the module GLOBAL, not an environment variable — and that asymmetry
    is the design: ``anchor_dir`` reads no environment at all (see
    ``test_the_anchor_path_cannot_be_redirected``), so the only way a test can
    redirect it is to reach into the process, which is exactly what a same-uid
    subject cannot do to a running runtime from outside.
    """
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", root)
    return root


# ---------------------------------------------------------------------------
# 1. The anchor
# ---------------------------------------------------------------------------


def test_a_file_this_uid_owns_is_not_an_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The measured reason the anchor is a root-owned file, pinned as code.

    A ``login`` keychain item is silently substitutable by a same-uid process and
    keychain search order is rewritable without authentication (measured in a
    throwaway keychain; see ``operator/trust.py``). The same thing is true of any
    file this uid can write, so ``load_anchor`` REFUSES a well-formed anchor that
    is not root-owned — it does not warn, and it does not use it at a lower
    level, because a substituted anchor is indistinguishable from a real one to
    everything downstream.
    """
    root = _anchor_root(monkeypatch, tmp_path / "anchors")
    root.mkdir()
    handle = FileKeyBackend(tmp_path / "k.pem").create()
    anchor = anchor_for_handle(handle)
    (root / f"{os.getuid()}.json").write_bytes(trust.anchor_bytes(anchor))

    loaded = trust.load_anchor()
    assert loaded.exists is True, "the file is there"
    assert loaded.usable is False, "a same-uid file must never be trusted"
    assert "not root" in loaded.reason, loaded.reason
    assert operator_authority_level() == LEVEL_ANCHOR_UNPINNED


def test_a_symlinked_anchor_path_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A link anywhere in the path lets the uid point the runtime elsewhere.

    Checked component by component with ``lstat``, and the final open uses
    ``O_NOFOLLOW``: a design that only validated the final file would be
    defeated by a symlinked DIRECTORY, which is the cheaper attack.
    """
    real = tmp_path / "real"
    real.mkdir()
    shim = tmp_path / "shim"
    shim.symlink_to(real)
    handle = FileKeyBackend(tmp_path / "k.pem").create()
    anchor = anchor_for_handle(handle)
    real.joinpath(f"{os.getuid()}.json").write_bytes(trust.anchor_bytes(anchor))
    _anchor_root(monkeypatch, shim)

    loaded = trust.load_anchor()
    assert loaded.usable is False
    assert "symbolic link" in loaded.reason, loaded.reason


def test_the_anchor_path_cannot_be_redirected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No environment variable, no config, no ``PATH`` entry moves the anchor.

    THE PIN THE DESIGN ASKS FOR BY NAME. Every candidate name a redirect might
    plausibly use is set to a directory this test controls, and the resolved path
    must not move — because an env-redirectable anchor is not an anchor: the
    gated subject writes the environment of its own tool subprocesses.
    """
    expected = trust.anchor_path()
    assert expected.is_absolute()
    before = str(expected)
    for name in (
        "LOP_OPERATOR_ANCHOR",
        "LOP_OPERATOR_ANCHOR_DIR",
        "LOCAL_OPERATOR_ANCHOR",
        "LOCAL_OPERATOR_ANCHOR_DIR",
        "LOP_HOME",
        "LOCAL_OPERATOR_HOME",
        "PROGRAMDATA",
    ):
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ.get('PATH', '')}")
    assert str(trust.anchor_path()) == before
    assert trust.anchor_dir() == trust.anchor_dir()
    # And the file NAME is the uid, not anything a caller supplies per call: a
    # path parameter would be one more thing to get wrong at a call site.
    assert trust.anchor_path().name == f"{os.getuid()}.json"


def test_an_anchor_whose_key_id_does_not_match_its_key_is_refused(tmp_path: Path) -> None:
    """The id is DERIVED, so a mismatch means the file was hand-assembled.

    Accepting it would let a frame's ``operator_key_id`` be matched against a
    value unrelated to the key that will actually verify the signature — the
    difference between naming a key and choosing one.
    """
    handle = FileKeyBackend(tmp_path / "k.pem").create()
    anchor = anchor_for_handle(handle)
    body = anchor.to_json()
    body["key_id"] = "0" * 32
    assert trust.OperatorAnchor.from_json(body) is None
    body = anchor.to_json()
    body["alg"] = "HS256"
    assert trust.OperatorAnchor.from_json(body) is None
    body = anchor.to_json()
    body["spki"] = "00" * 65
    assert trust.OperatorAnchor.from_json(body) is None
    assert trust.OperatorAnchor.from_json(anchor.to_json()) == anchor


# ---------------------------------------------------------------------------
# 2. The message
# ---------------------------------------------------------------------------


def test_the_signed_message_binds_every_field() -> None:
    """One field changed, one different message — for each field.

    This is what removes replay as a CATEGORY rather than one case of it, and the
    length prefixes are why it holds even when the fields are attacker-chosen:
    without them, ``("ab", "c")`` and ``("a", "bc")`` would frame identically.
    """
    base = dict(action="loosen", session_id="s1", request_id="", challenge="cd" * 32)
    canonical = verify.signed_message(**base)
    for field, value in (
        ("action", "approve"),
        ("session_id", "s2"),
        ("request_id", "r1"),
        ("challenge", "ef" * 32),
    ):
        changed = {**base, field: value}
        assert verify.signed_message(**changed) != canonical, field
    # The domain tag is part of the message, so a signature over any other
    # protocol's payload cannot be presented here even with identical fields.
    assert canonical.startswith(verify.DOMAIN)
    # Length-prefix framing: shifting a byte between adjacent fields changes it.
    assert verify.signed_message(
        action="a", session_id="bc", request_id="", challenge="z"
    ) != verify.signed_message(action="ab", session_id="c", request_id="", challenge="z")


# ---------------------------------------------------------------------------
# 3. The verdict
# ---------------------------------------------------------------------------


def test_a_signature_verifies_only_for_its_own_fields(keyed: Any) -> None:
    """The positive control, then each negative beside it."""
    challenge = "ab" * 32
    signature = keyed.sign(
        challenge=challenge, purpose="loosen", session_id="sess-1", request_id=""
    )
    assert signature["key_id"] == keyed.handle.key_id

    def verdict(**overrides: Any) -> bool | None:
        fields: dict[str, Any] = dict(
            action="loosen",
            session_id="sess-1",
            request_id="",
            challenge=challenge,
            signature_hex=signature["sig"],
            operator_spki=keyed.anchor.spki,
            operator_key_id=signature["key_id"],
            operator_cert=None,
            device_spki=None,
            now=int(time.time()),
        )
        fields.update(overrides)
        return verify.signature_verdict(**fields)

    assert verdict() is True
    assert verdict(action="approve") is False
    assert verdict(session_id="sess-2") is False
    assert verdict(challenge="aa" * 32) is False
    assert verdict(operator_key_id="0" * 32) is False
    # A signature offered with no anchor to place it against is a REFUSAL, not
    # "not offered": reading it as absent would let a present-but-unverifiable
    # signature slide through on the other source's coat-tails.
    assert verdict(operator_spki=None) is False
    # No signature at all IS "not offered", which is a different answer.
    assert verdict(signature_hex=None) is None
    assert verdict(signature_hex="zz") is False


def test_a_forged_device_certificate_is_rejected(keyed: Any) -> None:
    """A device proves itself with an OPERATOR-SIGNED certificate or not at all.

    The forgery here is the realistic one: the attacker holds a real P-256 key of
    their own (so the signature over the message verifies under ITS public point)
    and mints a certificate claiming that point is the operator's device. Nothing
    about the message is wrong — only the certificate is, and it dies because it
    carries no operator signature.
    """
    attacker = FileKeyBackend(keyed.root / "attacker.pem").create()
    forged = verify.DeviceCert(
        device_id="stolen",
        spki=attacker.spki,
        label="not really",
        issued_at=int(time.time()) - 10,
        not_after=int(time.time()) + 3600,
    )
    forged_cert = forged.encode(signature=b"\x30\x06\x02\x01\x01\x02\x01\x01")
    assert (
        verify.verify_device_cert(
            forged_cert, operator_spki=keyed.anchor.spki, now=int(time.time())
        )
        is None
    )

    # The real one, for contrast, and it carries the device's OWN point.
    device = FileKeyBackend(keyed.root / "device.pem").create()
    assert keyed.signer is not None
    good = issue_device_cert(
        device_spki=device.spki,
        device_id="phone-1",
        label="iPhone",
        signer=keyed.signer,
    )
    resolved = verify.verify_device_cert(
        good, operator_spki=keyed.anchor.spki, now=int(time.time())
    )
    assert resolved == device.spki
    # The SAME certificate under a different operator key does not verify: the
    # certificate is bound to one operator, not to the format.
    other = FileKeyBackend(keyed.root / "other-operator" / "operator-key.pem").create()
    assert verify.verify_device_cert(good, operator_spki=other.spki, now=int(time.time())) is None


def test_an_expired_or_future_device_certificate_is_rejected(keyed: Any) -> None:
    """Both directions: a certificate from the future is as much a lie as an old one."""
    device = FileKeyBackend(keyed.root / "device2.pem").create()
    assert keyed.signer is not None
    issued = issue_device_cert(
        device_spki=device.spki,
        device_id="phone-2",
        label="iPhone",
        signer=keyed.signer,
        lifetime_s=60,
    )
    now = int(time.time())
    assert verify.verify_device_cert(issued, operator_spki=keyed.anchor.spki, now=now) is not None
    assert verify.verify_device_cert(issued, operator_spki=keyed.anchor.spki, now=now + 61) is None
    assert verify.verify_device_cert(issued, operator_spki=keyed.anchor.spki, now=now - 61) is None


def test_a_revoked_device_is_refused_by_the_anchor(keyed: Any) -> None:
    """Revocation is a property of the ANCHOR, not of the certificate.

    The certificate still verifies — it was really signed — and the anchor's list
    is what refuses it. That division is why revocation is an edit to a
    root-owned file rather than an unauthenticated write to a cache.
    """
    anchor = trust.OperatorAnchor(
        key_id=keyed.anchor.key_id,
        spki=keyed.anchor.spki,
        backend=keyed.anchor.backend,
        presence=False,
        devices=({"device_id": "phone-1", "revoked": True},),
    )
    assert trust.device_is_revoked(anchor, "phone-1") is True
    assert trust.device_is_revoked(anchor, "phone-2") is False
    # An anchor that predates device support holds no opinion and therefore
    # revokes nothing: an empty list is not a blanket refusal.
    assert trust.device_is_revoked(keyed.anchor, "phone-1") is False


# ---------------------------------------------------------------------------
# 4. Levels and copy
# ---------------------------------------------------------------------------


def test_the_levels_are_the_closed_set_the_report_uses() -> None:
    """One vocabulary, so the seam and the copy cannot disagree about a level."""
    assert LEVELS == {
        "operator-presence",
        "operator-file-only",
        "anchor-unpinned",
        "spawn-capability-only",
    }


def test_a_host_with_no_anchor_reports_the_old_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``spawn-capability-only``, and the report says why.

    The runtime must WORK in this state — it is what a fresh install is — and the
    report is what stops that state being mistaken for the anchored one.
    """
    empty = _anchor_root(monkeypatch, tmp_path / "none")
    assert operator_authority_level() == LEVEL_SPAWN_ONLY
    report = operator_authority_report()
    assert report["anchor_installed"] is False
    assert report["presence_enforced_by_os"] is False
    assert "no anchor" in report["reason"]
    assert not empty.exists()


def test_the_effect_copy_names_the_session_and_the_effect(keyed: Any) -> None:
    """The prompt copy is the operator's only chance to see what they are approving.

    The OS dialog cannot carry custom text, so this sentence is where "name the
    session and the effect" happens — and the design's own residual (a prompt can
    be spammed and misread) is why it must name both.
    """
    loosen = effect_copy(purpose="loosen", session_id="sess-42")
    assert "LOOSEN" in loosen and "sess-42" in loosen
    approve = effect_copy(purpose="approve", session_id="sess-42", request_id="card-7")
    assert "APPROVE" in approve and "sess-42" in approve and "card-7" in approve
    fallback = effect_copy(purpose="approve", session_id="")
    assert "this session" in fallback


def test_signing_refuses_an_unknown_purpose_or_a_missing_challenge(keyed: Any) -> None:
    """Fail closed, and never silently: the caller must be able to tell the two
    "no" answers apart, because only one of them has a remedy the copy can name."""
    with pytest.raises(keychain.KeyBackendError):
        sign_challenge(
            challenge="ab" * 32,
            purpose="delete-everything",
            config_root=keyed.root,
            session_id="s",
        )
    with pytest.raises(keychain.KeyBackendError):
        sign_challenge(challenge="", purpose="loosen", config_root=keyed.root, session_id="s")
    # And with no key at all, the message names the one-step remedy rather than
    # raising a bare failure the caller has to translate.
    empty = keyed.root / "empty"
    empty.mkdir()
    with pytest.raises(keychain.KeyBackendError) as excinfo:
        sign_challenge(challenge="ab" * 32, purpose="loosen", config_root=empty, session_id="s")
    assert "lop operator init" in str(excinfo.value)


def test_the_staged_anchor_is_only_a_hint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``init`` before ``install`` must still be able to sign.

    The staged file is not authoritative and is deliberately not trust-checked:
    the runtime reads the ROOT-OWNED path, so a hint that lies about the backend
    costs one failed load and nothing else.
    """
    _anchor_root(monkeypatch, tmp_path / "anchors")
    created = create_key(config_root=tmp_path, preference=FILE_ONLY)
    staged = trust.staging_path(tmp_path)
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(trust.anchor_bytes(anchor_for_handle(created)))
    hint = trust.load_staged_anchor(tmp_path)
    assert hint is not None and hint.backend == FILE_ONLY
    # A hint and a trusted anchor are different questions with different answers.
    assert trust.load_anchor().usable is False
    signature = sign_challenge(
        challenge="ab" * 32, purpose="loosen", config_root=tmp_path, session_id="s"
    )
    assert signature.key_id == created.key_id
    assert load_signer(config_root=tmp_path, backend_name=FILE_ONLY) is not None


def test_no_private_key_material_is_in_the_anchor(keyed: Any) -> None:
    """Public data only, asserted on the bytes rather than on the intent.

    The anchor is world-readable (0644) by design — it holds a public key — so a
    private half in it would be a plain leak, and the check is on the serialized
    form so a future field cannot add one quietly.
    """
    blob = trust.anchor_bytes(keyed.anchor)
    body = json.loads(blob)
    assert set(body) == {
        "v",
        "key_id",
        "alg",
        "spki",
        "backend",
        "presence",
        "label",
        "created_at",
        "devices",
    }
    assert len(bytes.fromhex(body["spki"])) == 65
    private = (keyed.root / "operator" / "operator-key.pem").read_text()
    assert "PRIVATE" in private  # the file really does hold one...
    assert private.strip().splitlines()[-2] not in blob.decode()  # ...and none of it is here
