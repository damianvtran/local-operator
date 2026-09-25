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
import ctypes.util
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
from local_operator.operator.macos import keyagent
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

        def sign(self, **fields: Any) -> dict[str, str]:
            # `Any` rather than `str`: this stub forwards whatever the caller passed,
            # and ``sign_challenge`` now takes a numeric ``timeout`` as well as its
            # string fields — a passthrough that narrowed the type would have to know
            # every parameter of the function it exists to stand in for.
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
# WHAT REPLACED THE OWNERSHIP NET, and why. This file used to drive a fake ``_CF``
# that modelled CoreFoundation ownership across a Python implementation of the ladder.
# That implementation is GONE from the product (it could never work — the entitlement
# comes from a code signature and a provisioning profile, see ``keychain.py``), so a
# fake for it would be a net under code that does not exist. What takes its place:
#
#   * the constants vs the SDK HEADER, below (unchanged, and still the one copy of the
#     value that lives outside this repository);
#   * THE C SOURCE, pinned symbol-for-symbol against those same constants — the native
#     sequence now exists once, in ``packaging/macos/lop-keyagent/se-keyagent.c``, and
#     drift between the two copies of it is the hazard this arrangement creates;
#   * the KEY AGENT's OWN ``selftest`` verb, which asserts the release discipline with
#     ``CFGetRetainCount`` against the very constructors ``create`` runs. It needs no
#     entitlement and no keychain, and the release job runs it on the built binary
#     (see ``.github/workflows/publish.yml``); from here it is exercised only when a
#     built helper is pointed at by ``LOP_KEYAGENT_BINARY``;
#   * a real framework call for the ONE assertion a fake cannot make — that
#     ``SecAccessControlCreateWithFlags`` ACCEPTS the flag pair for both classes (the
#     defect that shipped). It writes nothing and prompts nothing.
#
# Nothing in this section creates a key, raises a prompt, or writes to any keychain.
# The one exception is the opt-in test at the end, gated on
# ``LOP_OPERATOR_ENCLAVE_TEST=1``, which creates ONE item under a unique tag through the
# signed key agent and deletes it again.

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


def _framework() -> tuple[Any, Any]:
    """CoreFoundation and Security, bound for the ONE call this file still makes.

    Bound HERE rather than in shipped code on purpose: the product no longer touches
    these functions from Python (the signed key agent does), so a binding in
    ``keychain.py`` would be a second implementation of a native sequence — which is
    exactly what was deleted. A test may hold one because its subject IS the framework.
    """
    cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))
    security = ctypes.CDLL(ctypes.util.find_library("Security"))
    security.SecAccessControlCreateWithFlags.restype = ctypes.c_void_p
    security.SecAccessControlCreateWithFlags.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    cf.CFRelease.argtypes = [ctypes.c_void_p]
    cf.CFStringGetCString.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_long,
        ctypes.c_uint32,
    ]
    cf.CFStringGetCString.restype = ctypes.c_bool
    cf.CFErrorCopyDescription.restype = ctypes.c_void_p
    cf.CFErrorCopyDescription.argtypes = [ctypes.c_void_p]
    return cf, security


def _framework_constant(lib: Any, name: str) -> int:
    """The ``CFTypeRef`` a framework global HOLDS, as the address of the global.

    ``in_dll`` gives the ADDRESS of the symbol; for a ``CFStringRef`` constant the
    value that belongs in a dictionary is what is stored THERE, which is what the
    dereference reads. This is the shape ``keychain._CF.const`` used before it was
    deleted, kept here because the call it feeds needs it.
    """
    value = ctypes.c_void_p.from_address(
        ctypes.addressof((ctypes.c_char * 0).in_dll(lib, name))
    ).value
    assert value is not None, f"{name} holds no pointer"
    return value


def _framework_error(cf: Any, err: Any) -> str:
    """The framework's own sentence for a failed call, releasing the error object."""
    import ctypes

    if not err.value:
        return "no error object was published"
    text = cf.CFErrorCopyDescription(err)
    if not text:
        return "the error had no description"
    buffer = ctypes.create_string_buffer(512)
    cf.CFStringGetCString(text, buffer, 512, 0x08000100)
    cf.CFRelease(text)
    cf.CFRelease(err.value)
    return buffer.value.decode("utf-8", "replace")


@pytest.mark.skipif(os.uname().sysname != "Darwin", reason="SecAccessControl is macOS-only")
def test_every_protection_class_accepts_the_flag_pair_this_build_uses() -> None:
    """THE TEST THAT WOULD HAVE CAUGHT THE RELEASED BUG — so it asks the framework.

    ``SecAccessControlCreateWithFlags`` is not our code, and it is the call that
    returned NULL/-50 for both protection classes with the shipped flags. It writes
    nothing to any keychain and raises no prompt, so it is safe in the default suite,
    and it fails loudly on the pre-fix constants with the framework's own sentence.

    COVERAGE SPLIT — this is a DEV-HOST GUARD, not a CI gate: it is gated on Darwin and
    the framework call cannot be exercised on Linux at all, so a green Linux CI does not
    cover this boundary. What CI covers with certainty is the constants, the SDK header
    read and the C source that makes the same two calls (``keyagent-selftest``).
    """
    import ctypes

    cf, security = _framework()
    backend = keychain.SecureEnclaveBackend
    for protection in backend.PROTECTION_LADDER:
        err = ctypes.c_void_p()
        access = security.SecAccessControlCreateWithFlags(
            None,
            _framework_constant(security, protection),
            backend.PRIVATE_KEY_USAGE | backend.USER_PRESENCE,
            ctypes.byref(err),
        )
        assert access, (
            f"{protection} was refused by the framework with the flag pair "
            f"{backend.USER_PRESENCE} | {backend.PRIVATE_KEY_USAGE}: {_framework_error(cf, err)}"
        )
        cf.CFRelease(access)


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
    operator key, creates the key through the product's own ``backend.create()``, and
    purges that item in teardown so the machine is left exactly as found.

    On an installation with no usable key agent this SKIPS rather than fails: without the
    signed helper there is no process that could hold this key at all, which is a fact
    about the INSTALL (and the reason ``health()`` exists), not a defect in this code.

    The purge is the interesting half. Deletion is gated by the SAME entitlement as
    creation — measured: an unsigned process asking to delete an item that exists gets
    errSecItemNotFound (-25300) — so the cleanup has to go through the entitled process
    too, which is why the key agent has a ``purge`` that refuses the operator's own tag.
    """
    unique = f"com.local-operator.operator.test.{os.getpid()}"
    monkeypatch.setattr(keychain, "APPLICATION_TAG", unique)
    backend = keychain.SecureEnclaveBackend()
    healthy, why = backend.health()
    if not healthy:
        pytest.skip(f"this installation cannot reach an entitled key agent: {why}")
    try:
        handle = backend.create()
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
        deleted = keyagent.KeyagentClient(tag=unique).purge()
        assert deleted in (
            0,
            keychain._ERR_SEC_ITEM_NOT_FOUND,
        ), f"the test's own key was not removed: SecItemDelete={deleted}"


# ---------------------------------------------------------------------------
# 12. The native sequence now lives in C: pin it against this module
# ---------------------------------------------------------------------------

#: The helper's source, in-tree. Its ABSENCE would make this file's checks vacuous, so
#: the path is asserted to exist rather than skipped on.
_HELPER_C = (
    Path(__file__).resolve().parents[3] / "packaging" / "macos" / "lop-keyagent" / "se-keyagent.c"
)


def _helper_code() -> str:
    """The C source with its comments removed, so a prose mention cannot pass a check."""
    import re

    source = _HELPER_C.read_text()
    assert "se-keyagent.c" in source or source, "the helper source is empty"
    return re.sub(r"/\*.*?\*/", "", source, flags=re.S)


def test_the_c_helper_uses_the_same_native_sequence_as_this_module() -> None:
    """Drift between the two copies of one native sequence is this design's hazard.

    The Python implementation of the Secure Enclave sequence is gone — it could never
    work, and two implementations of one sequence, one of which no test on any host can
    exercise, is a second way of doing things rather than a fallback. What remains is the
    C helper, so the constants this module still defines are pinned to what THAT file
    compiles and to the same SDK header the values came from.
    """
    assert _HELPER_C.is_file(), f"no helper source at {_HELPER_C}"
    code = _helper_code()
    backend = keychain.SecureEnclaveBackend

    # 1. THE LADDER, in the order this module documents: strictest first, so a host that
    #    accepts passcode-set never settles for unlocked.
    order = [code.index(protection) for protection in backend.PROTECTION_LADDER]
    assert order == sorted(
        order
    ), f"the helper's ladder order differs from {backend.PROTECTION_LADDER}"

    # 2. APPLE'S SYMBOLS, never a shift literal: the released pair was 1<<0 | 1<<2 and
    #    every protection class was refused with errSecParam.
    assert "kSecAccessControlUserPresence" in code
    assert "kSecAccessControlPrivateKeyUsage" in code
    for wrong in ("1 << 0", "1 << 2", "1 << 30", "1u << 0", "1u << 30"):
        assert wrong not in code, f"a shift literal ({wrong}) crept into the flag pair"

    # 3. THE GENERATION DICTIONARY: EC P-256 on the Enclave token, 256 bits.
    for symbol in (
        "kSecAttrKeyType",
        "kSecAttrKeyTypeECSECPrimeRandom",
        "kSecAttrKeySizeInBits",
        "kSecAttrTokenID",
        "kSecAttrTokenIDSecureEnclave",
        "kSecPrivateKeyAttrs",
        "kSecAttrIsPermanent",
        "kSecAttrApplicationTag",
        "kSecAttrAccessControl",
    ):
        assert symbol in code, f"the helper no longer names {symbol}"
    assert "int bits = 256;" in code

    # 4. TYPED CALLBACKS BY ADDRESS. NULL callbacks retain nothing and release nothing,
    #    and a generation dictionary so built SIGSEGVs inside SecKeyCreateRandomKey
    #    (measured, 6 of 6 runs). This is the one thing a rewrite must not "simplify".
    assert "&kCFTypeDictionaryKeyCallBacks" in code
    assert "&kCFTypeDictionaryValueCallBacks" in code
    assert "NULL, 0, NULL, NULL" not in code

    # 5. THE ALGORITHM, which is exactly what ``verify_signature`` consumes (DER ECDSA
    #    over SHA-256).
    assert "kSecKeyAlgorithmECDSASignatureMessageX962SHA256" in code

    # 6. NO kSecUseDataProtectionKeychain: not in Apple's documented Enclave generation
    #    dictionary, not load-bearing (the passing probe does not set it), and measured
    #    identical with and without it. The helper is built without it.
    assert "kSecUseDataProtectionKeychain" not in code

    # 7. THE TAG IS CREATED ONCE, BY main, AND THE LADDER ONLY BORROWS IT. That is the
    #    structural answer to the use-after-free (#1547): a tag created and released
    #    INSIDE the loop handed the next iteration a released CFDataRef. Asserted as
    #    "exactly one creation, exactly one release, neither inside cmd_create".
    assert code.count("CFDataCreate(NULL, (const UInt8 *)tagtext") == 1
    assert code.count("CFRelease(tag);") == 1
    ladder = code[
        code.index("static int cmd_create(CFDataRef tag) {") : code.index(
            "static int cmd_public_or_exists"
        )
    ]
    assert "CFDataCreate" not in ladder, "the ladder creates a tag of its own"
    assert "CFRelease(tag)" not in ladder, "the ladder releases the tag main owns"

    # 8. THE SITE VOCABULARY and the exit codes the client maps to copy, so a reply
    #    cannot arrive in a dialect this module does not speak.
    for site in (
        keychain.ACCESS_CONTROL_REFUSED,
        keychain.KEY_GENERATION_REFUSED,
        keychain.SIGNATURE_REFUSED,
        keychain.KEY_LOOKUP_REFUSED,
    ):
        assert f'"{site}"' in code, f"the helper does not report the site {site!r}"
    import re

    for name, value in (
        ("EXIT_CANCELLED", 2),
        ("EXIT_NO_KEY", 3),
        ("EXIT_REFUSED", 4),
        ("EXIT_USAGE", 5),
    ):
        assert re.search(rf"#define {name} {value}\b", code), f"{name} != {value} in C"
    protocol = re.search(r"#define PROTOCOL (\d+)", code)
    assert protocol is not None, "the helper has no PROTOCOL define"
    assert int(protocol.group(1)) == keyagent.PROTOCOL


@pytest.mark.skipif(
    not os.environ.get("LOP_KEYAGENT_BINARY"),
    reason="set LOP_KEYAGENT_BINARY to a built lop-keyagent to run its ownership selftest",
)
def test_the_built_helper_asserts_its_own_ownership_model() -> None:
    """The C-side ownership check the release job runs, reachable for a local build.

    It needs no entitlement and no keychain, so a plain ``clang`` build is enough —
    which is what makes it usable in a workflow that has not imported an identity yet.
    Every check it reports must agree: a mismatch is a reference this program created
    and did not give back, or gave back twice.
    """
    import subprocess

    result = subprocess.run(
        [os.environ["LOP_KEYAGENT_BINARY"], "selftest", "--tag", keyagent.HEALTH_TAG],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.stdout.strip(), result.stderr
    report = json.loads(result.stdout)
    mismatched = [check for check in report["checks"] if check["expected"] != check["actual"]]
    assert report["ok"] is True and result.returncode == 0, mismatched
    assert len(report["checks"]) >= 10, "the selftest stopped asserting the model"
