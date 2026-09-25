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

import argparse
import ctypes
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
    # AND IT IS REPORTED AS AN UNPINNED ANCHOR, NOT AS NO ANCHOR (agent review round
    # 6, R6-6). ``exists`` was left False for this refusal, so a host whose anchor is
    # present but behind a link reported ``spawn-capability-only`` — the level for a
    # host that never installed one — in a report whose whole job is to tell those
    # two states apart. The reason field said "symbolic link" while the level said
    # "nothing installed", and the level is what the operator's own `status` prints.
    assert loaded.exists is True, "a present-but-redirected anchor is not a missing one"
    assert operator_authority_level() == LEVEL_ANCHOR_UNPINNED


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


def test_the_signing_verb_shows_the_copy_to_the_person_signing(
    keyed: Any, capsys: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE SENTENCE REACHES A HUMAN FROM THE CLI (UX round 6, U3 = design D3).

    Measured before this fix: ``lop operator sign`` wrote JSON on stdout and
    nothing else, no OS operation prompt was passed at the signing call (there is
    nowhere to pass one — ``SecKeyCreateSignature`` takes no parameters dictionary
    and ``kSecUseOperationPrompt`` was deprecated in macOS 11), and
    ``on_operator_prompt`` was set at no production construction site. So the
    copy the design names as its mitigation for the misread-prompt residual was
    built and shown to nobody, while the doc claimed it was shown.

    ``stderr`` rather than stdout, because stdout is the value and nothing else —
    the property the module docstring states and a caller parsing this verb
    depends on.
    """
    from argparse import Namespace

    from local_operator.operator.handlers import _sign

    # The verb resolves the key the way the product does — through the config root
    # — so the fixture's root has to BE the config root for this call.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(keyed.root))
    # A REAL challenge, not a placeholder: the verb signs whatever it is handed,
    # and this cell is about the human-facing line rather than about validation.
    code = _sign(Namespace(challenge="cd" * 32, purpose="loosen", session="sess-9", request_id=""))
    assert code == 0
    captured = capsys.readouterr()
    assert "LOOSEN" in captured.err and "sess-9" in captured.err, captured.err
    assert "LOOSEN" not in captured.out, captured.out
    # ...and stdout is still exactly the JSON value, so a caller that pipes it is
    # unaffected by the human-facing line.
    assert captured.out.strip().startswith("{") and captured.out.strip().endswith("}")


def test_the_windows_ladder_falls_back_to_the_file_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R6-2/R6-8: ``supported()`` answered for the HOST, not for this BUILD.

    ``choose_backend("auto")`` promises "the first backend this host both HAS and
    can USE", and on ``os.name == "nt"`` it returned ``CngBackend`` without asking
    the second half — whose ``create`` raises "not implemented on this build". So
    ``lop operator init`` on Windows exited 1 with an internal backend name instead
    of creating the ``file-only`` key it could have created, and instead of
    reporting the level that host actually has.

    Read, not run: no Windows host exists here, so this pins the DECISION (which
    backend the ladder selects) and the operator-readable failure, not the CNG
    call. The doc grades Windows accordingly.
    """
    from local_operator.operator import keychain

    assert keychain.CngBackend().supported() is False, (
        "supported() must answer for this build, not for the host: its only caller "
        "uses it to decide whether the host can USE the backend"
    )
    with pytest.raises(keychain.KeyBackendError) as refused:
        keychain.CngBackend().create()
    assert "--backend file-only" in str(refused.value), str(refused.value)

    monkeypatch.setattr(keychain.os, "name", "nt")
    chosen = keychain.choose_backend("auto", config_root=tmp_path)
    assert isinstance(chosen, keychain.FileKeyBackend), type(chosen).__name__
    # ...and asking for it BY NAME still reports the readable reason rather than
    # dying on an internal identifier.
    with pytest.raises(keychain.KeyBackendError) as named:
        keychain.choose_backend("cng-presence", config_root=tmp_path).create()
    assert "not implemented on this build" in str(named.value), str(named.value)


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


def test_lop_operator_init_is_idempotent_and_never_replaces_a_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """R9-2/Q9-2: the first step of every route has to survive being run twice.

    Measured: `lop operator init --backend file-only` on a machine that already had
    an operator key raised an uncaught ``FileExistsError`` from the ``O_EXCL`` create
    and printed a traceback, and on a presence host the same run is a duplicate-item
    create. Neither may REPLACE the key — a new anchor invalidates every paired
    phone — so the verb reports what it found and completes only what is missing.
    The private half is compared byte for byte, because "nothing was replaced" is the
    claim that matters and a key id alone would not catch a re-created key.
    """
    from local_operator.operator import handlers, trust

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "local_operator.operator.trust._ANCHOR_ROOT_OVERRIDE",
        tmp_path / "anchor-root",
        raising=False,
    )
    args = argparse.Namespace(operator_command="init", backend="file-only", label="idem")

    assert handlers.dispatch(args) == 0
    key_path = tmp_path / "operator" / "operator-key.pem"
    first_key = key_path.read_bytes()
    first_anchor = json.loads(trust.staging_path(tmp_path).read_text())
    capsys.readouterr()

    # (a) the second run reports and exits 0, with nothing replaced
    assert handlers.dispatch(args) == 0
    said = capsys.readouterr().out
    assert "already exists" in said, said
    assert "nothing replaced" in said, said
    assert key_path.read_bytes() == first_key, "the private key was rewritten"
    assert json.loads(trust.staging_path(tmp_path).read_text()) == first_anchor

    # (b) and it completes the half that is missing rather than refusing to help
    trust.staging_path(tmp_path).unlink()
    assert handlers.dispatch(args) == 0
    said = capsys.readouterr().out
    assert "the anchor statement was missing, so it was written" in said, said
    assert json.loads(trust.staging_path(tmp_path).read_text())["key_id"] == first_anchor["key_id"]
    assert key_path.read_bytes() == first_key

    # (c) the state where the carrier is absent AND nothing needs writing — an
    # INSTALLED anchor with no staged file — used to omit the line entirely rather
    # than say there is none (UX round 10, U4): a slot that reads as "not reported".
    trust.staging_path(tmp_path).unlink()
    monkeypatch.setattr(
        handlers,
        "load_anchor",
        lambda uid=None: trust.AnchorLoad(
            anchor=trust.OperatorAnchor.from_json(first_anchor),
            path=trust.anchor_path(uid),
            root_owned=True,
            reason="ok",
            exists=True,
        ),
    )
    assert handlers.dispatch(args) == 0
    said = capsys.readouterr().out
    assert "staged : (none at" in said, said
    assert "anchor : " in said and "installed: True" in said, said


def test_the_spawn_only_status_lines_are_wrapped_by_the_terminal_not_by_hand(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """Design round 10, D3: the block that broke mid-word below ~79 columns.

    The first version printed ONE string containing newlines and an 11-space
    continuation indent. At 60 and 44 columns the terminal soft-wrapped those physical
    lines again, stranding the indent mid-answer and splitting words — the only field
    in the report with a hand-set indent, and the only one that looked broken. Every
    other field is one line per fact and lets the terminal wrap; this is asserted as
    that: one line per sentence, starting at column 0, with no interior run of spaces
    for a terminal to strand.
    """
    from local_operator.operator import handlers

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        handlers,
        "operator_authority_report",
        lambda: {
            "level": "spawn-capability-only",
            "anchor_path": str(tmp_path / "anchor-root" / "501.json"),
            "anchor_installed": False,
            "anchor_root_owned": False,
            "backend": "",
            "presence_enforced_by_os": False,
            "capability_guarantee": "spawn-capability",
            "reason": "no anchor installed",
        },
    )
    assert handlers._status() == 0
    out = capsys.readouterr().out.splitlines()
    at = next(index for index, line in enumerate(out) if line.startswith("loosening:"))
    lines = out[at : at + 2]
    assert len(lines) == 2, out
    for line in lines:
        assert line == line.lstrip(), f"a hand-set indent came back: {line!r}"
        assert "  " not in line, f"a doubled space for a terminal to strand: {line!r}"
    assert lines[0].endswith("host.")
    assert "lop operator init" in lines[1], lines


# ---------------------------------------------------------------------------
# 11. The Secure Enclave backend: Apple's flags, the ladder, and the ownership
# ---------------------------------------------------------------------------
#
# Four defects, in the order a caller meets them:
#
# 1. the access-control flag pair was a shift guess (``1 << 0`` / ``1 << 2``)
#    instead of Apple's (``userPresence`` 1<<0, ``privateKeyUsage`` 1<<30), so
#    ``SecAccessControlCreateWithFlags`` answered ``errSecParam`` (-50) for every
#    protection class and ``create`` never once reached key generation;
# 2. the key-generation dictionaries were built with NULL CoreFoundation callbacks
#    (``CFDictionaryCreate(..., None, None)``), which retain nothing and release
#    nothing. Measured on this host with the corrected flags, that is a SIGSEGV
#    INSIDE ``SecKeyCreateRandomKey`` on the FIRST attempt (``TKSEPKey`` /
#    ``objc_retain``), on an unsigned interpreter and on an Apple-signed one alike —
#    so it is not the tag lifetime and not a caller problem;
# 3. ``tag`` was released INSIDE the ladder and then handed to the next iteration's
#    dictionary — a released ``CFDataRef``. This is a SECOND, distinct crash, reachable
#    only once (2) is fixed and key generation then fails on attempt 1, so fixing only
#    the callbacks and flags would leave a use-after-free on the fallback path;
# 4. the ``CFNumberRef`` handed to ``kSecAttrKeySizeInBits`` was never released,
#    because its docstring said the dictionary would do it while the dictionary was
#    built with those NULL callbacks.
#
# Nothing in this section creates a key, raises a prompt, or writes to any keychain:
# the ladder tests drive a fake ``_CF`` that models ownership by substituting the whole
# CoreFoundation surface (the real one would reach the OS), so their subject is
# OWNERSHIP — which ref exists, who releases it, and whether a released one is read.
# The real ``dict()``'s callback ARGUMENTS are pinned separately, by the two dedicated
# tests above it. The test that asks the real framework creates an access-control object
# and nothing else. The one exception is the opt-in test at the end, gated on
# ``LOP_OPERATOR_ENCLAVE_TEST=1``.

#: A real uncompressed P-256 point, exported from a Secure Enclave key created on
#: this machine (public data — the private half never leaves the Enclave). A fixed
#: value rather than a generated one keeps the ladder tests deterministic.
_A_REAL_P256_POINT = bytes.fromhex(
    "04db295b8129a1ba169a538a3b69f1caa1dbdd2ed34332c33ec5d3e0364ecb87"
    "8845a59a4f54c9728f8ed81905ad9765ad752b9e1e9080a528cc37e8098534167b"
)


def test_the_access_control_flags_are_apple_s_values_and_not_a_shift_guess() -> None:
    """The pair that shipped was transposed and two bits off Apple's ``PrivateKeyUsage``.

    ``kSecAccessControlUserPresence = 1u << 0`` and ``kSecAccessControlPrivateKeyUsage
    = 1u << 30`` (``Security.framework/Headers/SecAccessControl.h``); the file shipped
    ``PRIVATE_KEY_USAGE = 1 << 0`` and ``USER_PRESENCE = 1 << 2``, which the framework
    refuses with ``errSecParam``. Asserted as VALUES rather than as "the two flags
    differ", because a pair that is merely self-consistent is exactly what was wrong.
    """
    backend = keychain.SecureEnclaveBackend
    assert backend.USER_PRESENCE == 1 << 0
    assert backend.PRIVATE_KEY_USAGE == 1 << 30
    # The shipped pair, spelled out: a wrong pair must not be able to look right again.
    shipped_wrong = (1 << 0) | (1 << 2)
    assert (backend.USER_PRESENCE | backend.PRIVATE_KEY_USAGE) != shipped_wrong


def _sdk_header() -> Path | None:
    """The Security header that DEFINES the two flags, or ``None`` without an SDK."""
    import subprocess

    try:
        sdk = subprocess.run(
            ["xcrun", "--show-sdk-path"], capture_output=True, text=True, check=True, timeout=60
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    header = Path(sdk) / "System/Library/Frameworks/Security.framework/Headers/SecAccessControl.h"
    return header if header.is_file() else None


def test_the_constants_match_the_sdk_header_that_defines_them() -> None:
    """Read Apple's own header, so the values are checked against the source of truth.

    A test that only compares the constant to a literal this file also contains
    cannot catch the failure it exists for: whoever transposed the pair would write
    the transposed pair in both places. The header is the one copy of the value that
    lives outside this repository, so it is the copy the test reads — and it is read
    from whatever SDK the host actually compiles against.
    """
    import re

    header = _sdk_header()
    if header is None:
        pytest.skip("no macOS SDK on this host (xcrun --show-sdk-path found no header)")

    text = header.read_text()
    assert "kSecAccessControlUserPresence" in text and "kSecAccessControlPrivateKeyUsage" in text
    backend = keychain.SecureEnclaveBackend
    for symbol, constant in (
        ("kSecAccessControlUserPresence", backend.USER_PRESENCE),
        ("kSecAccessControlPrivateKeyUsage", backend.PRIVATE_KEY_USAGE),
    ):
        match = re.search(rf"{symbol}[^\n]*?=\s*1u\s*<<\s*(\d+)", text)
        assert match is not None, f"{symbol} is not defined the way this test reads it"
        assert constant == 1 << int(match.group(1)), (
            f"{symbol} is 1u << {match.group(1)} in {header}, " f"but this build uses {constant}"
        )


@pytest.mark.skipif(os.uname().sysname != "Darwin", reason="SecAccessControl is macOS-only")
def test_every_protection_class_accepts_the_flag_pair_this_build_uses() -> None:
    """THE TEST THAT WOULD HAVE CAUGHT THE RELEASED BUG — so it asks the framework.

    ``SecAccessControlCreateWithFlags`` is not our code, and it is the call that
    returned NULL/-50 for both protection classes with the shipped flags. It writes
    nothing to any keychain and raises no prompt, so it is safe in the default suite,
    and it fails loudly on the pre-fix constants with the framework's own sentence.

    COVERAGE SPLIT — this is a DEV-HOST GUARD, not a CI gate: it is gated on Darwin and
    the framework call cannot be exercised on Linux at all, so a green CI does NOT cover
    the framework boundary. What CI does cover with certainty is the constants, the SDK
    header read and the whole fake-``_CF`` ownership net, none of which need an OS.
    """
    cf = keychain._CF()
    backend = keychain.SecureEnclaveBackend.__new__(keychain.SecureEnclaveBackend)
    for protection in keychain.SecureEnclaveBackend.PROTECTION_LADDER:
        err = ctypes.c_void_p()
        access = cf.S.SecAccessControlCreateWithFlags(
            None,
            cf.const(protection),
            keychain.SecureEnclaveBackend.PRIVATE_KEY_USAGE
            | keychain.SecureEnclaveBackend.USER_PRESENCE,
            ctypes.byref(err),
        )
        if not access:
            pytest.fail(f"{protection} was refused by the framework: {cf.error(err)}")
        cf.release(int(access))
    assert backend is not None  # the ladder is a class attribute; no instance needed


class _FakeCF:
    """A ``_CF`` stand-in that models OWNERSHIP, not the operating system.

    Every CoreFoundation call the backend makes is answered with a synthetic ref, and
    the fake tracks which refs are alive. The part that matters: it REFUSES to read a
    ref that has been released, raising a sentence that names the call site. That is
    the check the real framework cannot give us — the same mistake is a SIGSEGV inside
    ``SecKeyCreateRandomKey`` on this host (measured, 6 of 6 runs), which no unit suite
    can assert against. Borrowed refs (``const``/``boolean``) are negative and never
    tracked, exactly like the framework's singletons.
    """

    def __init__(
        self,
        *,
        access_accepted: dict[str, bool] | None = None,
        keygen: list[tuple[int, str] | None] | None = None,
        point: bytes = _A_REAL_P256_POINT,
        sign_refused: bool = False,
    ) -> None:
        self._next = 1
        self.kinds: dict[int, str] = {}
        self.live: set[int] = set()
        self.created: list[tuple[str, int]] = []
        self.released: list[tuple[str, int]] = []
        self.reuse: list[tuple[str, int]] = []
        self.blobs: dict[int, bytes] = {}
        self.names: dict[int, str] = {}
        self.errors: dict[int, tuple[int, str]] = {}
        self.access_accepted = access_accepted or {
            name: True for name in keychain.SecureEnclaveBackend.PROTECTION_LADDER
        }
        self.keygen = keygen if keygen is not None else [None]
        self.attempts = 0
        self.flags_seen: list[int] = []
        self.point = point
        #: Whether ``SecKeyCreateSignature`` refuses, so the signing path's failure exit is
        #: reachable without an OS: the payload release lives on both exits.
        self.sign_refused = sign_refused
        #: A DER-shaped ES256 signature, so a successful sign returns something real.
        self.signature = b"\x30\x44\x02\x20" + bytes(range(32)) + b"\x02\x20" + bytes(range(32))
        self.S = _FakeSecurity(self)
        self.C = _FakeCoreFoundation(self)

    # -- ownership plumbing -------------------------------------------------
    def _make(self, kind: str) -> int:
        ref = self._next
        self._next += 1
        self.kinds[ref] = kind
        self.live.add(ref)
        self.created.append((kind, ref))
        return ref

    def _touch(self, ref: Any, where: str) -> int:
        value = int(ref or 0)
        if value in self.kinds and value not in self.live:
            self.reuse.append((where, value))
            raise AssertionError(
                f"{where} read {self.kinds[value]} ref {value} after it was released — this "
                "is the freed-memory path the tag-lifetime fix removes"
            )
        return value

    def release(self, *refs: Any) -> None:
        """Release each owned ref once; borrowing a const is a no-op, as in ``_CF``.

        A second release of the same ref raises through :meth:`_touch`, which is what
        makes "released exactly once" a real assertion rather than a counting one.
        """
        for ref in refs:
            value = int(ref or 0)
            if value in self.kinds:
                self._touch(value, "release")
                self.live.discard(value)
                self.released.append((self.kinds[value], value))

    def _fail(self, out: Any, status: int, text: str) -> None:
        """Publish an error the way the framework does: an out-param the caller owns.

        A status of 0 means the framework refused WITHOUT publishing an error — the out
        param is left untouched, so ``status`` and ``error`` see no error object at all.
        That is the state QA round 2 (Q2-1) is about, and modelling it is what makes the
        "no success code in a failure line" assertion mean something.
        """
        if not status:
            return
        ref = self._make("error")
        self.errors[ref] = (status, text)
        if out is not None:
            out._obj.value = ref

    # -- the surface the backend uses --------------------------------------
    def const(self, name: str) -> int:
        for ref, known in self.names.items():
            if known == name:
                return ref
        ref = -(len(self.names) + 1)
        self.names[ref] = name
        return ref

    def boolean(self, value: bool) -> int:
        return self.const(f"bool:{value}")

    def data(self, raw: bytes) -> int:
        ref = self._make("data")
        self.blobs[ref] = raw
        return ref

    def data_bytes(self, ref: Any) -> bytes:
        return self.blobs[self._touch(ref, "data_bytes")]

    def dict(self, pairs: list[tuple[int, int]]) -> int:
        """Assert the contents are ALIVE at build time, then record an owned ref.

        LIMIT, stated because it is the one thing this fake cannot model: it does not
        RETAIN what it is given, so it cannot catch a value released after the
        dictionary was built and then read THROUGH it. That half is not left untested —
        it is asserted where it lives, in the typed-callbacks tests, which is also why
        those read Apple's own callback tables rather than a flag on this fake.
        """
        for key, value in pairs:
            self._touch(key, "CFDictionaryCreate key")
            self._touch(value, "CFDictionaryCreate value")
        return self._make("dict")

    def status(self, err: Any) -> int:
        value = int(err.value or 0)
        # Touch it: reading a code out of an error object that has already been RELEASED is
        # the use-after-free the real ``cf.error`` would hide, and the signing path relies
        # on the order (status first, then the call that releases it).
        self._touch(value, "CFErrorGetCode")
        return self.errors.get(value, (0, ""))[0]

    def error(self, err: Any) -> str:
        value = int(err.value or 0)
        if not value:
            # ``_CF.error`` renders a missing error as "no error reported" and releases
            # nothing, so the fake says the same rather than inventing a code.
            return "no error reported"
        self._touch(value, "CFErrorCopyDescription")
        code, text = self.errors.pop(value, (0, ""))
        if value in self.live:  # _CF.error releases the CFErrorRef; so does the fake
            self.live.discard(value)
            self.released.append((self.kinds[value], value))
        return f"OSStatus error {code} - {text}"

    def counts(self) -> dict[str, tuple[int, int]]:
        """``kind -> (created, released)``, for the accounting test."""
        kinds = sorted(set(self.kinds.values()))
        return {
            kind: (
                sum(1 for k, _ in self.created if k == kind),
                sum(1 for k, _ in self.released if k == kind),
            )
            for kind in kinds
        }


class _FakeSecurity:
    """The ``.S`` half: the calls ``create``, ``load`` and ``sign`` make.

    ``SecKeyCreateSignature`` is here so the signing path has a guard that runs in every
    CI job rather than only under this host's by-hand probes: it was the one changed line
    in the round-1 remediation with nothing exercising it (agent review round 2, R2-3 /
    QA round 2, Q2-3).
    """

    def __init__(self, cf: _FakeCF) -> None:
        self.cf = cf

    def SecAccessControlCreateWithFlags(
        self, allocator: Any, protection: Any, flags: Any, out: Any
    ) -> int:
        cf = self.cf
        cf.flags_seen.append(int(flags))
        name = cf.names.get(int(protection or 0), "")
        if cf.access_accepted.get(name, True):
            return cf._make("access")
        cf._fail(out, -50, "the access control was refused")
        return 0

    def SecKeyCreateRandomKey(self, attrs: Any, out: Any) -> int:
        cf = self.cf
        cf._touch(attrs, "SecKeyCreateRandomKey")
        plan = cf.keygen[min(cf.attempts, len(cf.keygen) - 1)]
        cf.attempts += 1
        if plan is None:
            return cf._make("key")
        cf._fail(out, plan[0], plan[1])
        return 0

    def SecKeyCopyPublicKey(self, key: Any) -> int:
        cf = self.cf
        cf._touch(key, "SecKeyCopyPublicKey")
        return cf._make("public")

    def SecKeyCopyExternalRepresentation(self, public: Any, out: Any) -> int:
        cf = self.cf
        cf._touch(public, "SecKeyCopyExternalRepresentation")
        # The bytes live in the ref, so `data_bytes` reads something real: the point is
        # the one a Secure Enclave key on this host actually exported.
        raw = cf._make("raw")
        cf.blobs[raw] = cf.point
        return raw

    def SecKeyCreateSignature(self, key: Any, algorithm: Any, payload: Any, out: Any) -> int:
        """Sign, or publish a refusal the caller must report.

        ``payload`` is TOUCHED, so a message ref that was released before the call raises
        here — which is exactly the leak's inverse and the reason the payload is bound.
        """
        cf = self.cf
        cf._touch(key, "SecKeyCreateSignature key")
        cf._touch(payload, "SecKeyCreateSignature payload")
        if cf.sign_refused:
            cf._fail(out, -34018, "a signature needs a presence prompt this process cannot raise")
            return 0
        signature = cf._make("signature")
        cf.blobs[signature] = cf.signature
        return signature

    def SecItemCopyMatching(self, query: Any, out: Any) -> int:
        cf = self.cf
        cf._touch(query, "SecItemCopyMatching")
        return -25300  # errSecItemNotFound: this store holds nothing


class _FakeSymbol:
    """One stand-in ctypes function object.

    A plain bound method cannot play this role: the backend sets ``restype`` and
    ``argtypes`` on the symbol before calling it, exactly as it must on a real
    ``_FuncPtr``, and setting an attribute on a bound method raises. This carries those
    two attributes and does nothing with them, so the ownership test exercises the
    real call site rather than a relaxed copy of it.
    """

    def __init__(self, impl: Any) -> None:
        self.restype: Any = None
        self.argtypes: Any = None
        self._impl = impl

    def __call__(self, *args: Any) -> Any:
        return self._impl(*args)


class _FakeCoreFoundation:
    """The ``.C`` half. Only ``CFNumberCreate`` is used, and it counts as an owned ref."""

    def __init__(self, cf: _FakeCF) -> None:
        self.CFNumberCreate = _FakeSymbol(lambda allocator, kind, byref_value: cf._make("number"))


def _enclave_with(monkeypatch: pytest.MonkeyPatch, fake: _FakeCF) -> Any:
    """The real backend, wired to the fake CoreFoundation surface."""
    monkeypatch.setattr(keychain, "_CF", lambda: fake)
    return keychain.SecureEnclaveBackend()


def test_the_dictionaries_are_built_with_the_typed_callbacks() -> None:
    """The dictionary's callbacks are the difference between a clean call and a crash.

    ``CFDictionaryCreate(..., None, None)`` is legal but means "this dictionary manages
    nothing", and the framework's copy/release paths inside key generation do not honour
    that (see the section header). Asserted on the two ARGUMENTS, because they are the
    whole difference — a test that only drove the fake could pass with NULL callbacks,
    since the fake never touches the OS.
    """
    cf = keychain._CF.__new__(keychain._CF)
    cf.TYPED_KEY_CALLBACKS = 0xCA11
    cf.TYPED_VALUE_CALLBACKS = 0xCA22
    seen: list[tuple[Any, ...]] = []

    class _Recorder:
        restype: Any = None
        argtypes: Any = None

        def CFDictionaryCreate(self, *args: Any) -> int:
            seen.append(args)
            return 4242

    # ``Any`` because this stands in for the loaded dylib: the point is the ARGUMENTS
    # ``dict`` passes, and the real ``_CF.C`` is a ``CDLL`` the recorder deliberately
    # is not.
    recorder: Any = _Recorder()
    cf.C = recorder
    assert cf.dict([(1, 2)]) == 4242
    assert len(seen) == 1, seen
    allocator, keys, values, count, key_callbacks, value_callbacks = seen[0]
    assert allocator is None and count == 1 and keys is not None and values is not None
    assert (key_callbacks, value_callbacks) == (0xCA11, 0xCA22), (
        "a dictionary built without the typed callbacks is the second crash: "
        f"got {(key_callbacks, value_callbacks)!r}"
    )


@pytest.mark.skipif(os.uname().sysname != "Darwin", reason="CoreFoundation is macOS-only")
def test_the_typed_callbacks_are_a_live_corefoundation_table_not_a_null_pair() -> None:
    """...and they are Apple's OWN table, not merely two addresses that differ.

    A NULL pair passes any "the arguments are not None" check on the wrong object, so
    this reads the memory the address points at: ``CFDictionaryKeyCallBacks`` /
    ``ValueCallBacks`` begin ``{CFIndex version; void *retain; void *release; ...}``, and
    a wrong or NULL address cannot present version 0 with a live retain/release pair.

    COVERAGE SPLIT — a DEV-HOST GUARD like the framework test above: it needs the loaded
    dylib, so it skips (and proves nothing) on Linux. The callback ARGUMENTS are checked
    portably by ``test_the_dictionaries_are_built_with_the_typed_callbacks``, which is
    the half CI runs everywhere.
    """
    cf = keychain._CF()
    for name, address in (
        ("key", cf.TYPED_KEY_CALLBACKS),
        ("value", cf.TYPED_VALUE_CALLBACKS),
    ):
        assert address, f"the {name} callbacks are NULL — the released defect"
        assert ctypes.c_long.from_address(address).value == 0, f"{name} callbacks version"
        retain = ctypes.c_void_p.from_address(address + 8).value
        release = ctypes.c_void_p.from_address(address + 16).value
        assert retain and release, f"the {name} callback table has no retain/release pair"


def test_a_failed_first_attempt_hands_the_next_one_a_live_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE USE-AFTER-FREE, and why it is coupled to the flag fix.

    The released ladder released ``tag`` on the attempt that failed to make a key —
    and then built the next attempt's dictionary around it. With the shipped flags
    that path was unreachable, because the access control failed one step earlier;
    fixing the flags alone would make it reachable exactly where it matters (a host
    whose policy refuses the strictest protection class), turning a precise error
    into a crash. Modelled in that order deliberately: access control ACCEPTED,
    key generation FAILED on the first attempt, and the second attempt must receive a
    live tag.
    """
    fake = _FakeCF(keygen=[(-34018, "failed to add key to keychain"), None])
    backend = _enclave_with(monkeypatch, fake)

    handle = backend.create()
    assert handle.backend == keychain.SECURE_ENCLAVE
    assert fake.attempts == 2, "the ladder should have taken the second class"
    assert fake.reuse == [], f"a released ref was used again: {fake.reuse}"


def test_every_ref_the_ladder_creates_is_released_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accounting per object class, over a two-attempt ladder, a success, and an access refusal.

    The CFNumber is the one that leaked: its docstring claimed the dictionary would
    release it while the dictionary was built with NULL callbacks and retained
    nothing. The tag is the one that was released twice. Both are invisible to a
    functional test that only checks the OSStatus, which is why they are asserted as
    counts over the same fake that refuses a stale ref.

    Three branches, because they hold three different sets of objects: the ladder that
    falls through, the ladder that succeeds, and the ACCESS-REFUSED branch, whose
    ``CFErrorRef`` is created by the framework and released by ``_CF.error`` — with the
    lifetime nobody would notice leaking (agent review round 1, R1-7).
    """
    ladder = _FakeCF(keygen=[(-34018, "failed to add key to keychain"), None])
    _enclave_with(monkeypatch, ladder).create()
    for kind, (created, released) in ladder.counts().items():
        assert created == released, f"{kind}: created {created}, released {released}"
    assert ladder.live == set(), f"still live after a two-attempt ladder: {ladder.live}"

    success = _FakeCF(keygen=[None])
    _enclave_with(monkeypatch, success).create()
    for kind, (created, released) in success.counts().items():
        assert created == released, f"{kind}: created {created}, released {released}"
    assert success.live == set(), f"still live after a success: {success.live}"
    # ...and the CFNumber really is among them, so "no leak" is not vacuous.
    assert success.counts()["number"] == (1, 1)

    refusal = _FakeCF(
        access_accepted={name: False for name in keychain.SecureEnclaveBackend.PROTECTION_LADDER}
    )
    with pytest.raises(keychain.KeyBackendError):
        _enclave_with(monkeypatch, refusal).create()
    for kind, (created, released) in refusal.counts().items():
        assert created == released, f"{kind}: created {created}, released {released}"
    assert refusal.live == set(), f"still live after an access refusal: {refusal.live}"
    # The error refs are the point of this branch: one per class, each released once.
    assert refusal.counts()["error"] == (2, 2)


def test_the_flags_the_ladder_actually_builds_are_apple_s_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the framework is HANDED, not what the module says about itself.

    ``test_the_access_control_flags_...`` pins the constants; this pins the value
    that reaches the call, so an OR/AND slip between them cannot pass both.
    """
    fake = _FakeCF(keygen=[None])
    _enclave_with(monkeypatch, fake).create()
    expected = keychain.SecureEnclaveBackend.USER_PRESENCE | (
        keychain.SecureEnclaveBackend.PRIVATE_KEY_USAGE
    )
    assert fake.flags_seen == [expected]


def test_both_classes_failing_is_a_precise_error_and_never_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No silent downgrade, no traceback, and both attempts named.

    The two outcomes a caller must tell apart are "this host has no presence store"
    and "the presence store refused", and only the second has a gesture the operator
    can make — so a failure that collapsed them, or that dropped one attempt's text,
    would cost the operator the only sentence they can act on.

    With DIFFERENT framework detail per class the detail cannot be merged, so this is
    also the branch that pins one detail line PER refusal — the merge case is asserted
    in ``test_the_refusal_message_leads_with_the_diagnosis...``. This is the
    ``errSecParam``-at-key-generation site, which is a class this host refused, NOT the
    build defect the same code means at the access-control site (design round 1, D2).
    """
    fake = _FakeCF(keygen=[(-50, "first class refused"), (-50, "second class refused")])
    backend = _enclave_with(monkeypatch, fake)

    with pytest.raises(keychain.KeyBackendError) as refused:
        backend.create()
    message = str(refused.value)
    assert "first class refused" in message and "second class refused" in message
    assert "kSecAttrAccessibleWhenPasscodeSetThisDeviceOnly" in message
    assert "kSecAttrAccessibleWhenUnlockedThisDeviceOnly" in message
    assert message.count("framework detail:") == 2, message
    assert "inconsistent" in message, "the key-generation site, not the flag-pair one"
    assert refused.value.status == -50
    assert fake.attempts == 2


def test_the_refusal_is_classified_by_call_site_and_not_by_the_code_alone() -> None:
    """The same ``OSStatus`` means something different at the two call sites.

    ``errSecParam`` (-50) is the framework's GENERIC parameter refusal. At the
    access-control call it means a flag pair Apple will not accept — a build defect no
    host can work around; at key generation it means "inconsistent private key
    parameters", whose usual cause is a protection class THIS HOST refuses. Reading the
    code alone would tell an operator on a host that merely refuses a class that their
    BUILD is broken (design round 1, D2 / agent review round 1, R1-1).

    ``errSecMissingEntitlement`` (-34018) is about the CALLER, and the measured gate is
    the keychain ENTITLEMENT rather than "being code-signed" in the abstract: a
    signed-but-unentitled caller gets the same code (R1-2).
    """
    access = keychain.secure_enclave_diagnosis(keychain.ACCESS_CONTROL_REFUSED, -50)
    assert "errSecParam" in access[0]
    assert "kSecAccessControlApplicationPassword" in access[0]
    # The site must name a next command, and NOT the one that cannot help there: a
    # file-backed key is a different key, and `file-only` never calls
    # SecAccessControlCreateWithFlags, so pointing at it would send the operator to a
    # command that does not address what failed (design round 2, D2-1). The command it
    # names must be one the READER has, and must say which build carries the fix: this
    # copy ships in a wheel, where only `local-operator` and `lop` exist — `lop-update`
    # is the operator's own host script and is not installed by anything here (design
    # round 1, D1-2).
    assert "lop update" in " ".join(access), "the access-control site names no action"
    assert "from `main`" in " ".join(access), "the copy must say which build has the fix"
    assert "file-only" not in " ".join(access), "file-only does not help at this site"
    keygen = keychain.secure_enclave_diagnosis(keychain.KEY_GENERATION_REFUSED, -50)
    assert keygen and keygen[0] != access[0], "one code read the same way at two sites"
    assert "inconsistent" in keygen[0]
    # Its next action used to be `lop operator status` alone, and on the state a keygen
    # refusal leaves that report's own way out is `lop operator init` — the command that
    # just refused. It is now the downgrade, byte-identical to the sibling entry's, so the
    # two sites cannot drift into describing one fallback two ways (design round 1, D1-1).
    assert keygen[1] == keychain._FILE_ONLY_FALLBACK, keygen
    assert (
        keygen[1]
        == keychain.secure_enclave_diagnosis(
            keychain.KEY_GENERATION_REFUSED, keychain._ERR_SEC_MISSING_ENTITLEMENT
        )[1]
    )
    # A status nobody has classified contributes nothing, rather than a guessed cause.
    assert keychain.secure_enclave_diagnosis(keychain.KEY_GENERATION_REFUSED, -9999) == ()
    assert keychain.secure_enclave_diagnosis("a site that does not exist", -50) == ()
    # ...and an unclassified refusal still reaches the operator, with the code stated by
    # US because the framework's text did not state it.
    unclassified = keychain.secure_enclave_refusal_message(
        {("key generation", -9999, "something new"): ["kSecAttrAccessibleX"]}
    )
    assert unclassified.startswith("the Secure Enclave refused to create an operator key")
    assert unclassified.endswith(
        "framework detail: key generation — kSecAttrAccessibleX: OSStatus -9999 - something new"
    ), unclassified


def test_the_entitlement_diagnosis_names_the_cost_of_the_fallback_it_offers() -> None:
    """``file-only`` may be the fallback, and may not be sold as the good path.

    The first cut ended "…`--backend file-only` creates the level this runtime can
    enforce", which is the overclaim ``docs/design/approval-authority.md`` exists to
    prevent: ``file-only`` is exactly what this product refuses to call a boundary —
    ``describe_level`` says "This is NOT a boundary", and the file-only WARNING says any
    process running as you can read the key. So the sentence names what is GIVEN UP (no
    presence prompt, readable by any process running as you), names the fallback and the
    level ``lop operator status`` will report, and follows the shape ``CngBackend``
    already established for the same situation (design round 1, D1).
    """
    lines = keychain.secure_enclave_diagnosis(
        keychain.KEY_GENERATION_REFUSED, keychain._ERR_SEC_MISSING_ENTITLEMENT
    )
    joined = " ".join(lines)
    assert "errSecMissingEntitlement" in joined
    assert "entitlement" in joined and "data-protection keychain" in joined
    assert "code-signed" not in joined, "the measured gate is the entitlement"
    assert "file-only" in joined and "operator-file-only" in joined
    assert "raises no presence prompt" in joined
    assert "any process running as you can read it" in joined
    assert "enforce" not in joined, "file-only is a fallback, not an enforced level"


def test_the_refusal_message_leads_with_the_diagnosis_and_says_each_thing_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the operator reads: diagnosis, next command, then the framework's detail.

    Three things the first cut got wrong (design round 1, D3/D4): the CLI prefix and the
    exception restated each other ("could not create the operator key: the Secure Enclave
    refused to create an operator key"), the framework's identical clause was printed
    once per protection class, and the recommendation sat at the end of an ~850-character
    line — so this pins the LAYOUT, not just the content.

    The framework's own words are kept as the last line, and its per-run object address
    is removed there because this text is meant to be pasted into a report.
    """
    fake = _FakeCF(
        keygen=[
            (
                -34018,
                "failed to add key to keychain" " <SecKeyRef:('com.apple.setoken')> 0x7f86b0d240",
            ),
            (
                -34018,
                "failed to add key to keychain" " <SecKeyRef:('com.apple.setoken')> 0x7f86b0d3f0",
            ),
        ]
    )
    with pytest.raises(keychain.KeyBackendError) as refused:
        _enclave_with(monkeypatch, fake).create()
    message = str(refused.value)
    first, *rest = message.splitlines()
    assert first.startswith("this runtime cannot create a presence-gated operator key"), first
    assert first.count("errSecMissingEntitlement") == 1, "the diagnosis is said once, first"
    assert any("lop operator init --backend file-only" in line for line in rest[:2]), rest[:2]
    # ONE detail line, with both classes beside it: the same clause is not repeated per
    # class, and the per-run address did not reach the copy.
    assert message.count("framework detail:") == 1, message
    assert (
        "kSecAttrAccessibleWhenPasscodeSetThisDeviceOnly, "
        "kSecAttrAccessibleWhenUnlockedThisDeviceOnly"
    ) in message, message
    assert "0x" not in message, f"a per-run object address reached the copy: {message}"
    assert message.count("-34018") == 1, f"the status is stated once: {message}"
    assert "OSStatus -34018 - OSStatus error" not in message, "our prefix was added on top"
    assert refused.value.status == keychain._ERR_SEC_MISSING_ENTITLEMENT
    # ONE LINE PER SENTENCE, AT COLUMN 0, with no hand-set indent and no interior run of
    # spaces for a terminal to strand mid-wrap: the sibling status block's convention,
    # which design round 2 (D2-2) asked this message to follow rather than introduce a
    # second one. Only the product's lines are checked for double spaces — the framework's
    # own words are kept verbatim and may contain them.
    for line in message.splitlines():
        assert not line.startswith(" "), f"a hand-set indent came back: {line!r}"
        if not line.startswith("framework detail:"):
            assert "  " not in line, f"a doubled space for a terminal to strand: {line!r}"

    # When the framework's own description already states the code — its usual
    # "(OSStatus error -N - …)" shape — the framework's words are kept VERBATIM and ours
    # are not prefixed on top of them: the point of the detail line is that it is what a
    # report should quote (design round 1, D4).
    real_shape = _FakeCF(
        keygen=[
            (
                -34018,
                "The operation couldn\u2019t be completed. (OSStatus error -34018 - failed to "
                "add key to keychain: <SecKeyRef:('com.apple.setoken')>)",
            )
        ]
    )
    with pytest.raises(keychain.KeyBackendError) as refused_real:
        _enclave_with(monkeypatch, real_shape).create()
    detail_line = str(refused_real.value).splitlines()[-1]
    assert detail_line.endswith(
        "(OSStatus error -34018 - failed to add key to keychain: <SecKeyRef:('com.apple.setoken')>)"
    ), detail_line
    assert "OSStatus -34018 - The operation" not in detail_line, "our prefix was added anyway"


def test_a_printed_failure_carries_no_per_run_object_address() -> None:
    r"""The framework's ``> 0x…`` address goes; nothing else does.

    ``0x75929d8380`` differs every run, and stripping it is also what lets two protection
    classes' otherwise identical refusals collapse into one detail line (design round 1,
    D4). The first cut used ``\s*0x[0-9a-fA-F]+``, which ALSO deleted a small hex literal,
    truncated a hex path segment and — because ``\s*`` matches a newline — could join two
    lines. So the rule is anchored to the shape the framework actually prints, and the
    cases below are its contract rather than an aspiration (agent review round 2, R2-2 /
    QA round 2, Q2-2).
    """
    raw = "failed to add key to keychain: <SecKeyRef:('com.apple.setoken')> 0x7F86B0D240"
    cleaned = keychain.without_run_addresses(raw)
    assert "0x" not in cleaned and "7F86B0D240" not in cleaned
    assert cleaned == "failed to add key to keychain: <SecKeyRef:('com.apple.setoken')>"
    # Nine digits is the SHORTEST address the framework has printed on this host
    # (`0x10137ede0`), and it must still go: a threshold of eight would have spared the
    # parameters below instead (agent review round 1, R1-1).
    nine = "keychain: <SecKeyRef:('com.apple.setoken')> 0x10137ede0"
    assert keychain.without_run_addresses(nine) == "keychain: <SecKeyRef:('com.apple.setoken')>"
    # NOT run addresses — every one of these was altered by the first rule.
    for kept in (
        "id=0x7f8 ref=0x1",  # short hex literals: `id= ref=` before
        "invalid parameter 0x00000008 for kSecAttrKeyType",  # a length-8 PARAMETER (R1-1)
        "token 0xdeadbeef is a parameter, not an address",  # ...and another: both were deleted
        "could not read /tmp/0x9f/probe.pem",  # a hex path segment: `/tmp//probe.pem` before
        "OSStatus error -50",  # no hex at all
        "0x75929d8380 begins the description",  # no separator, so nothing to anchor on
        "first line\n    0x1000 second line",  # a small literal AND a line break: must not join
    ):
        assert keychain.without_run_addresses(kept) == kept, kept
    # ...and the two classes' texts DO become identical, which is the dedup benefit: two
    # addresses that differ per run and per class now read the same, so the refusals merge.
    class_a = "keychain: <SecKeyRef:('com.apple.setoken')> 0x75929d8380"
    class_b = "keychain: <SecKeyRef:('com.apple.setoken')> 0x75929d8840"
    assert keychain.without_run_addresses(class_a) == keychain.without_run_addresses(class_b)


def test_the_refusal_status_is_unanimous_or_none_and_never_a_success_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A caller branching on ``status`` must fail CLOSED.

    The attribute exists so the opt-in hardware test can skip without reading prose, so a
    ladder whose classes refused with DIFFERENT codes must not hand it whichever class
    happened to run first — that silently chooses which remedy a caller is offered, and
    the ladder's order would decide it (agent review round 2, R2-1). A refusal that
    published NO error object carries ``0``, a SUCCESS code, and is reported as no code at
    all rather than as "OSStatus 0" (QA round 2, Q2-1).
    """
    mixed = _FakeCF(keygen=[(-50, "inconsistent params"), (-34018, "no entitlement")])
    with pytest.raises(keychain.KeyBackendError) as refused:
        _enclave_with(monkeypatch, mixed).create()
    assert refused.value.status is None, "a mixed ladder decided which code to report"
    # Both refusals are still on the message: failing closed must not lose the detail.
    assert str(refused.value).count("framework detail:") == 2, str(refused.value)
    # ...and swapping the ladder's order cannot change the answer.
    swapped = _FakeCF(keygen=[(-34018, "no entitlement"), (-50, "inconsistent params")])
    with pytest.raises(keychain.KeyBackendError) as refused_swapped:
        _enclave_with(monkeypatch, swapped).create()
    assert refused_swapped.value.status is None

    agreed = _FakeCF(keygen=[(-34018, "no entitlement"), (-34018, "no entitlement")])
    with pytest.raises(keychain.KeyBackendError) as unanimous:
        _enclave_with(monkeypatch, agreed).create()
    assert unanimous.value.status == -34018, "a unanimous ladder must still report its code"

    # A refusal the framework reports WITHOUT an error object: no code to state, and the
    # number 0 must not reach the line as if it were one.
    silent = _FakeCF(keygen=[(0, "ignored — nothing was published"), (0, "ignored")])
    with pytest.raises(keychain.KeyBackendError) as wordless:
        _enclave_with(monkeypatch, silent).create()
    message = str(wordless.value)
    assert wordless.value.status is None
    assert "OSStatus 0 - " not in message, f"a success code in a failure line: {message}"
    assert message.splitlines()[-1].endswith("no error reported"), message


def test_the_signing_payload_is_released_on_both_paths_and_the_status_outlives_the_error() -> None:
    """The one changed line of the round-1 remediation that had no guard anywhere.

    ``_SecureEnclaveSigner.sign`` used to create the message ``CFDataRef`` inline and drop
    it — one leaked CoreFoundation object per signature — and the fix binds it and releases
    it in a ``finally``. Nothing exercised that: ``SecKeyCreateSignature`` appeared in the
    tree only inside docstrings, so a future re-inline would have passed CI exactly as the
    leak did (agent review round 2, R2-3 / QA round 2, Q2-3).

    No OS is needed for this, which is why it belongs in the default suite: the fake now
    carries ``SecKeyCreateSignature``, refuses a released ref, and refuses to answer
    ``status`` out of an error object that has already been released — so the ORDER the
    signing path relies on (code first, then the call that releases the error) is
    asserted rather than assumed.
    """
    handle = keychain.KeyHandle(
        backend=keychain.SECURE_ENCLAVE,
        key_id=verify.key_id_for(_A_REAL_P256_POINT),
        spki=_A_REAL_P256_POINT,
        presence=True,
    )

    # Success: the payload is created once and released once, and so is the signature.
    # ``Any`` for the fake, as elsewhere in this file: it stands in for the loaded dylib,
    # and the real parameter is typed as the ``_CF`` it deliberately is not.
    signed_cf: Any = _FakeCF()
    signed_key = signed_cf._make("key")
    signer = keychain._SecureEnclaveSigner(handle, signed_key, signed_cf)
    assert signer.sign(b"a message") == signed_cf.signature
    assert signed_cf.counts()["data"] == (1, 1), signed_cf.counts()
    assert signed_cf.counts()["signature"] == (1, 1), signed_cf.counts()
    assert signed_cf.live == {signed_key}, "only the key the signer still holds may be live"
    signer.close()
    assert signed_cf.live == set()

    # Failure: the payload is still released exactly once, and the status is read while
    # the error ref is live (a read after `error()` would raise in this fake).
    refused_cf: Any = _FakeCF(sign_refused=True)
    refused_key = refused_cf._make("key")
    refusing = keychain._SecureEnclaveSigner(handle, refused_key, refused_cf)
    with pytest.raises(keychain.KeyBackendError) as raised:
        refusing.sign(b"a message")
    assert raised.value.status == -34018
    assert refused_cf.counts()["data"] == (1, 1), refused_cf.counts()
    assert refused_cf.counts()["error"] == (1, 1), refused_cf.counts()
    assert "0x" not in str(raised.value), str(raised.value)
    assert refused_cf.live == {refused_key}
    refusing.close()
    assert refused_cf.live == set()


class _StubEnclaveBackend:
    """A presence store that answers the two questions ``init`` asks, without a key.

    ``load`` returns ``None`` until ``create`` has run, so ONE object models both the
    "first init" and the "second init" state while ``create`` is counted — which is
    what makes "idempotent" and "never silently falls back" assertable without
    creating anything on the machine.
    """

    def __init__(self) -> None:
        self.handle = keychain.KeyHandle(
            backend=keychain.SECURE_ENCLAVE,
            key_id=verify.key_id_for(_A_REAL_P256_POINT),
            spki=_A_REAL_P256_POINT,
            presence=True,
        )
        self.created = 0

    def create(self) -> Any:
        self.created += 1
        return self.handle

    def load(self) -> Any:
        return keychain.Signer(self.handle) if self.created else None

    def supported(self) -> bool:
        return True


def _stub_enclave(monkeypatch: pytest.MonkeyPatch, stub: _StubEnclaveBackend) -> None:
    """Wire the stub into BOTH callers of ``choose_backend``.

    ``handlers._existing_key`` imports it per call; ``sign.create_key`` bound it at
    import time. Patching one and not the other is how a stub test silently exercises
    the real ladder on a machine that has a Secure Enclave.
    """
    from local_operator.operator import sign

    monkeypatch.setattr(keychain, "choose_backend", lambda preference, *, config_root: stub)
    monkeypatch.setattr(sign, "choose_backend", lambda preference, *, config_root: stub)


def test_init_on_a_presence_host_reports_the_presence_level_and_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """A presence host must not be quietly demoted to ``file-only``.

    The failure this guards is a SETUP STEP that reports success at a level weaker
    than the one asked for: the operator asked for the presence tier, so a key written
    as a readable file instead would be a silent downgrade of the security claim —
    which is why the assertion is the ABSENCE of the file, not just the presence of
    the right stdout line.
    """
    from local_operator.operator import handlers, trust

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "local_operator.operator.trust._ANCHOR_ROOT_OVERRIDE",
        tmp_path / "anchor-root",
        raising=False,
    )
    stub = _StubEnclaveBackend()
    _stub_enclave(monkeypatch, stub)
    args = argparse.Namespace(operator_command="init", backend="secure-enclave", label="enclave")

    assert handlers.dispatch(args) == 0
    captured = capsys.readouterr()
    assert "operator key created in the secure-enclave store" in captured.out, captured.out
    assert "secure-enclave: every signature requires a human gesture" in captured.out
    assert "has no presence store" not in captured.err, captured.err

    anchor = json.loads(trust.staging_path(tmp_path).read_text())
    assert anchor["backend"] == keychain.SECURE_ENCLAVE
    assert anchor["presence"] is True
    assert anchor["key_id"] == verify.key_id_for(_A_REAL_P256_POINT)
    assert stub.created == 1
    # THE SILENT-DOWNGRADE CHECK: no software key was written behind the operator's back.
    assert not (tmp_path / "operator" / "operator-key.pem").exists()


def test_init_twice_on_a_presence_host_replaces_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    """The same idempotency contract the file-only test asserts, on the presence path.

    ``init`` on a host that already has a presence key is a duplicate-item create
    there, and a second key would invalidate every device certificate signed under the
    first anchor. So the second run reports what it found and CREATE IS NOT CALLED
    AGAIN — asserted on the stub's own counter, which is what distinguishes "reported
    as unchanged" from "changed and reported nothing".
    """
    from local_operator.operator import handlers, trust

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "local_operator.operator.trust._ANCHOR_ROOT_OVERRIDE",
        tmp_path / "anchor-root",
        raising=False,
    )
    stub = _StubEnclaveBackend()
    _stub_enclave(monkeypatch, stub)
    args = argparse.Namespace(operator_command="init", backend="secure-enclave", label="enclave")

    assert handlers.dispatch(args) == 0
    first_anchor = json.loads(trust.staging_path(tmp_path).read_text())
    capsys.readouterr()

    assert handlers.dispatch(args) == 0
    said = capsys.readouterr().out
    assert "already exists in the secure-enclave store; nothing replaced" in said, said
    assert json.loads(trust.staging_path(tmp_path).read_text()) == first_anchor
    assert stub.created == 1, "the second init created a second key"
    assert not (tmp_path / "operator" / "operator-key.pem").exists()


@pytest.mark.skipif(
    os.environ.get("LOP_OPERATOR_ENCLAVE_TEST") != "1",
    reason="opt-in: creates ONE item in the operator's login keychain, under a unique tag",
)
def test_a_real_secure_enclave_key_round_trips_and_is_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The only test here that may create a real key: opt-in, uniquely tagged, cleaned up.

    Gated on ``LOP_OPERATOR_ENCLAVE_TEST=1`` because a Secure Enclave key can only live
    in the data-protection keychain, i.e. the operator's login keychain (see the module
    docstring). It uses a UNIQUE application tag so it can never touch the operator's own
    operator key, creates the key through the product's own ``create``, and deletes that
    item in teardown so the machine is left exactly as found.

    On a runtime with no keychain entitlement this SKIPS rather than fails: the measured
    answer there is ``errSecMissingEntitlement`` (-34018) from the OS, which is a fact
    about the CALLER's signature (an ad-hoc/linker-signed interpreter gets it, Apple's
    signed ``python3`` does not), not a defect in this code. A test cannot sign itself,
    so the honest outcome is a skip naming the measurement.

    COVERAGE SPLIT — this is the ONLY test that could prove the real key path end to end,
    and it is opt-in, so NO CI job covers the framework boundary: it skips everywhere CI
    runs. The by-hand round trip in this PR's evidence is what stands in for it, run under
    the one signed interpreter on this machine. CI's guarantee is the
    constants/header/fake-``_CF`` net, which needs no OS and does cover this file's logic.
    """
    unique = f"com.local-operator.operator.test.{os.getpid()}"
    monkeypatch.setattr(keychain, "APPLICATION_TAG", unique)
    cf = keychain._CF()
    tag = cf.data(unique.encode())
    query = cf.dict(
        [
            (cf.const("kSecClass"), cf.const("kSecClassKey")),
            (cf.const("kSecAttrApplicationTag"), tag),
        ]
    )
    try:
        backend = keychain.SecureEnclaveBackend()
        try:
            handle = backend.create()
        except keychain.KeyBackendError as refused:
            # Branch on the STATUS the OS returned, not on the message text: the message is
            # prose that may be reworded, and this skip must keep working through that
            # (agent review round 1, R1-6).
            if refused.status == keychain._ERR_SEC_MISSING_ENTITLEMENT:
                pytest.skip(f"this interpreter cannot use the data-protection keychain: {refused}")
            raise
        try:
            assert len(handle.spki) == 65 and handle.spki[0] == 0x04
            assert verify.decode_point(handle.spki) is not None
            signer = backend.load()
            assert signer is not None, "load() did not find the key create() just made"
            try:
                assert signer.handle.key_id == handle.key_id
                assert signer.handle.spki == handle.spki
            finally:
                signer.close()
        finally:
            # The product ships no delete verb on purpose (see the keychain module
            # docstring); removing the item this TEST created is the test's job.
            cf.S.SecItemDelete(query)
    finally:
        cf.release(query, tag)
