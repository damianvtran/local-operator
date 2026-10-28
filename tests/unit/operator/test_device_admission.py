"""S4a-pair-op S1 — the device-approval split: one admission decision, three callers.

Until this change a certificate had to answer ONE question — "is this device
revoked?" — so an operator-signed certificate naming a device the computer never
listed was let in (the S4a spike minted one from the anchor's own key and the
runtime accepted it). The device-approval design turns a second question on: does
this computer ADMIT the device, which on an enforcing host means the root-owned
anchor names it.

That question is asked at three places and must get the SAME answer at all of
them:

* the runtime's certificate point, on a cache MISS (the verify arm) and on a
  cache HIT (the per-use re-check arm) — ``RuntimeServer._device_cert_point``;
* the relay's command path — ``mobile.daemon._device_refusal``, which answers the
  phone in words that name the action clearing the refusal.

:func:`test_every_site_agrees_on_every_case` is the point of this file: one row
per (anchor, certificate) case, run through every site, so a caller that keeps the
old predicate — or grows a fourth question of its own — fails here rather than in
production.

THE SWITCH IS DORMANT, and two tests say so where a human reads it. An anchor with
no ``admission`` marker behaves exactly as it does today (byte-identical
statements, so no pinned approval digest moves; a paired phone stays admitted with
no re-pair), and the reporting surfaces SAY that rather than let the runtime's
silence read as a boundary.

Two seams only, both in-process by design: ``trust._ANCHOR_ROOT_OVERRIDE`` — a
module global a test patches, never an environment variable, because an
env-redirectable anchor path is the substitution attack the module prevents — and
``trust._open_no_follow``, faked to report ``st_uid=0`` so a throwaway file plays
the root-owned anchor. Nothing here reads the real anchor, opens the operator's
keychain, or starts a service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.operator import trust
from local_operator.operator.devices import new_device_id
from local_operator.operator.keychain import FILE_ONLY, FileKeyBackend
from local_operator.operator.sign import anchor_for_handle, create_key

# ---------------------------------------------------------------------------
# The rig: a real file-only operator key, real device keys, a throwaway anchor
# ---------------------------------------------------------------------------


def _anchor_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    """Point the anchor loader at a throwaway directory (the in-process seam)."""
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", root)
    return root


def _present_as_root(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report the anchor file as root-owned, which a test cannot be.

    Everything else stays real — the symlink walk, the open, the JSON, the parse
    — so the ownership gate is the only synthesized fact, exactly as
    ``test_operator_authority.py`` does it.
    """
    real_open = trust._open_no_follow

    def as_root(path: Path) -> Any:
        opened = real_open(path)
        assert opened is not None
        descriptor, _ = opened
        return descriptor, SimpleNamespace(st_uid=0, st_mode=0o100644)

    monkeypatch.setattr(trust, "_open_no_follow", as_root)


def _install(root: Path, anchor: trust.OperatorAnchor) -> Path:
    """Write the anchor where the loader looks, as the install step would."""
    path = root / f"{os.getuid()}.json"
    path.write_bytes(trust.anchor_bytes(anchor))
    return path


class _Rig:
    """One operator key, three device keys, and the anchors the table varies."""

    def __init__(self, config_root: Path, anchor_root: Path) -> None:
        self.config_root = config_root
        self.anchor_root = anchor_root
        handle = create_key(config_root=config_root, preference=FILE_ONLY)
        self.base = anchor_for_handle(handle, label="s1-admission")
        # Three DEVICE keys: the one an enforcing host lists, one it does not, and
        # one it has revoked. A device key is a file-only P-256 key in a tmp dir;
        # nothing signs with it here, because the certificate is the OPERATOR's
        # statement about the key — the device's own signature is a frame concern.
        self.listed_key = FileKeyBackend(config_root / "listed.pem").create()
        self.unlisted_key = FileKeyBackend(config_root / "unlisted.pem").create()
        self.revoked_key = FileKeyBackend(config_root / "revoked.pem").create()
        self.listed_id = new_device_id(self.listed_key.spki)
        self.unlisted_id = new_device_id(self.unlisted_key.spki)
        self.revoked_id = new_device_id(self.revoked_key.spki)

    def anchor(self, kind: str) -> trust.OperatorAnchor:
        """One anchor per row of the policy table below."""
        listed = {"device_id": self.listed_id}
        revoked = {"device_id": self.revoked_id, "revoked": True}
        if kind == "dormant":
            return self.base
        if kind == "dormant_with_revocation":
            return replace(self.base, devices=(listed, revoked))
        if kind == "enforced":
            return replace(self.base, admission=trust.ADMISSION_LISTED, devices=(listed,))
        if kind == "enforced_with_revocation":
            return replace(
                self.base,
                admission=trust.ADMISSION_LISTED,
                devices=(listed, revoked),
            )
        if kind == "presence":
            return replace(self.base, presence=True)
        if kind == "presence_enforced":
            return replace(
                self.base,
                presence=True,
                admission=trust.ADMISSION_LISTED,
                devices=(listed,),
            )
        raise AssertionError(f"unknown anchor kind {kind}")

    def certificate(self, kind: str) -> tuple[str, str, bytes]:
        """``(certificate, declared device id, the key it really carries)``.

        ``forged`` is the S4a spike's statement: the operator's own key signs a
        certificate that NAMES the listed id while carrying a different key. It
        verifies — domain, signature and expiry all hold — which is why the
        predicate has to compare the id against the key rather than trust the body.
        """
        from local_operator.operator.sign import issue_device_cert, load_signer

        signer = load_signer(config_root=self.config_root, backend_name="file-only")
        assert signer is not None, "the rig's operator key did not load"
        try:
            if kind == "listed":
                key, declared = self.listed_key, self.listed_id
            elif kind == "unlisted":
                key, declared = self.unlisted_key, self.unlisted_id
            elif kind == "revoked":
                key, declared = self.revoked_key, self.revoked_id
            elif kind == "forged":
                key, declared = self.unlisted_key, self.listed_id
            else:  # pragma: no cover - a typo in a parametrization
                raise AssertionError(f"unknown certificate kind {kind}")
            certificate = issue_device_cert(
                device_spki=key.spki,
                device_id=declared,
                label="phone",
                signer=signer,
            )
        finally:
            signer.close()
        return certificate, declared, key.spki


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Rig:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    anchor_root = _anchor_root(monkeypatch, tmp_path / "anchors")
    _present_as_root(monkeypatch)
    return _Rig(tmp_path, anchor_root)


# ---------------------------------------------------------------------------
# The policy table: every (anchor, certificate) pair, and what each site owes
# ---------------------------------------------------------------------------

#: ``(anchor kind, certificate kind, admitted)``.
#:
#: The expectations ARE the design, written once so the three sites cannot
#: disagree about it:
#:
#: * a DORMANT host (no marker) admits any device whose certificate this key
#:   signed — today's behaviour, kept for every phone paired before the switch;
#: * an ENFORCING host admits only the id its anchor lists;
#: * ``forged`` is refused everywhere, including on a dormant host: the
#:   certificate's id must be its key's id, which is what stops a statement from
#:   naming a permitted id while carrying somebody else's key. Every certificate
#:   ``lop pair`` mints derives its id, so this refuses hand-made statements only;
#: * ``presence: true`` admits an UNLISTED device BY DESIGN — that backend costs a
#:   human gesture per signature, so the signature already is the confirmation;
#: * a revoked device is refused at every level, which is the rule that predates
#:   this change and stays exactly as it was.
CASES: tuple[tuple[str, str, bool], ...] = (
    ("dormant", "listed", True),
    ("dormant", "unlisted", True),
    ("dormant", "forged", False),
    ("dormant", "revoked", True),    ("dormant_with_revocation", "listed", True),
    ("dormant_with_revocation", "unlisted", True),
    ("dormant_with_revocation", "forged", False),
    ("dormant_with_revocation", "revoked", False),
    ("enforced", "listed", True),
    ("enforced", "unlisted", False),
    ("enforced", "forged", False),
    ("enforced", "revoked", False),
    ("enforced_with_revocation", "listed", True),
    ("enforced_with_revocation", "unlisted", False),
    ("enforced_with_revocation", "forged", False),
    ("enforced_with_revocation", "revoked", False),
    ("presence", "listed", True),
    ("presence", "unlisted", True),
    ("presence", "forged", False),
    ("presence", "revoked", True),
    ("presence_enforced", "listed", True),
    ("presence_enforced", "unlisted", True),
    ("presence_enforced", "forged", False),
    ("presence_enforced", "revoked", True),
)


def _runtime_stub() -> Any:
    """The smallest object ``RuntimeServer._device_cert_point`` needs.

    The method is pure over the cache it is given — it reads the anchor it is
    passed and touches no socket, no session and no configuration — so this runs
    the REAL body for both arms without standing up a server.
    """
    return SimpleNamespace(_device_certs={})


@pytest.mark.parametrize(("anchor_kind", "cert_kind", "admitted"), CASES)
def test_every_site_agrees_on_every_case(
    rig: _Rig,
    monkeypatch: pytest.MonkeyPatch,
    anchor_kind: str,
    cert_kind: str,
    admitted: bool,
) -> None:
    """The table, through every site that decides whether a device may act."""
    from local_operator.mobile.daemon import (
        OPERATOR_DEVICE_REVOKED_CODE,
        OPERATOR_DEVICE_UNCONFIRMED_CODE,
        _device_refusal,
    )
    from local_operator.operator import devices as device_store
    from local_operator.session.runtime import server as runtime_server
    from local_operator.operator.verify import verify_device_cert

    anchor = rig.anchor(anchor_kind)
    _install(rig.anchor_root, anchor)
    certificate, declared, key = rig.certificate(cert_kind)
    revoked = anchor_kind.endswith("_with_revocation") and cert_kind == "revoked"

    # 1. The predicate itself, on the two facts the certificate carries.
    assert trust.device_is_admitted(anchor, declared, key) is admitted

    # 2. The store wrapper the relay asks (a cert string, parsed here).
    assert device_store.is_admitted(rig.config_root, certificate) is admitted

    # 3. The runtime's certificate point. The FIRST call verifies, the SECOND is a
    #    cache hit — counted, so the per-use arm is proven to have run rather than
    #    assumed from a matching answer.
    stub = _runtime_stub()
    calls: list[int] = []
    real_verify = verify_device_cert

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(runtime_server.operator_verify, "verify_device_cert", counting)
    first = runtime_server.RuntimeServer._device_cert_point(stub, certificate, anchor)
    second = runtime_server.RuntimeServer._device_cert_point(stub, certificate, anchor)
    assert (first is not None) is admitted, "cache-miss arm"
    assert (second is not None) is admitted, "cache-hit arm"
    assert len(calls) == 1, "the second call must take the cache, not re-verify"

    # 4. The relay, which answers the phone rather than parking the command.
    refusal = _device_refusal(certificate)
    if admitted:
        assert refusal is None
    else:
        assert refusal is not None
        assert refusal.status_code == 403
        body = json.loads(refusal.body)
        # TWO CODES, because the two remedies are different: a revocation is
        # lifted from this computer's device list, a not-yet-confirmed phone has
        # already done what pairing asks of it.
        assert body["code"] == (
            OPERATOR_DEVICE_REVOKED_CODE if revoked else OPERATOR_DEVICE_UNCONFIRMED_CODE
        )


def test_the_unconfirmed_copy_does_not_send_the_phone_to_pair_again(rig: _Rig) -> None:
    """A re-pair would mint the same certificate and meet the same refusal.

    The refusal the phone shows is the relay's sentence, so it must name the action
    that clears the state (confirming the phone on the computer) and say that
    pairing will not help — otherwise the human is in a loop with no exit.
    """
    from local_operator.mobile.daemon import _device_refusal

    _install(rig.anchor_root, rig.anchor("enforced"))
    certificate, _, _ = rig.certificate("unlisted")
    refusal = _device_refusal(certificate)
    assert refusal is not None
    sentence = json.loads(refusal.body)["error"]
    assert "confirm" in sentence
    assert "pairing it again will not help" in sentence


# ---------------------------------------------------------------------------
# The dormant switch: no marker, no change — including the bytes
# ---------------------------------------------------------------------------


def test_an_anchor_without_the_marker_is_byte_identical_to_the_old_shape(rig: _Rig) -> None:
    """``statement_digest`` IS the bytes, and live approval cards pin it.

    ``anchor_bytes`` is what ``statement_digest`` hashes and every card records as
    its provenance. Emitting ``"admission": ""`` unconditionally would change the
    digest of every anchor in existence and make every card filed before this
    build refuse with ``approval_anchor_mismatch``. The expected body below is
    spelled out rather than derived from ``to_json``: a test that calls the code
    it is checking would pass on any drift.
    """
    anchor = rig.base
    expected = {
        "v": trust.ANCHOR_VERSION,
        "key_id": anchor.key_id,
        "alg": "ES256",
        "spki": anchor.spki.hex(),
        "backend": anchor.backend,
        "presence": anchor.presence,
        "label": anchor.label,
        "created_at": anchor.created_at,
        "devices": [],
    }
    expected_bytes = json.dumps(expected, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    assert trust.anchor_bytes(anchor) == expected_bytes
    assert "admission" not in json.loads(expected_bytes)
    assert trust.statement_digest(anchor) == (
        "sha256:" + hashlib.sha256(expected_bytes).hexdigest()
    )
    # And a round trip through the parser returns the same statement, digest and all.
    reparsed = trust.OperatorAnchor.from_json(json.loads(expected_bytes))
    assert reparsed is not None
    assert trust.anchor_bytes(reparsed) == expected_bytes


def test_a_marker_set_is_covered_by_the_digest(rig: _Rig) -> None:
    """Once it IS set the field is part of the statement, not decoration.

    The marker decides who may act on this host, so a pinned digest has to change
    when it changes — otherwise a card could be minted against an anchor and
    installed over one with a different admission rule.
    """
    dormant = rig.base
    enforced = rig.anchor("enforced")
    assert trust.statement_digest(dormant) != trust.statement_digest(enforced)
    assert json.loads(trust.anchor_bytes(enforced))["admission"] == "listed"


def test_the_marker_survives_a_revocation_and_an_authorisation_restage(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--revoke`` and ``--authorise`` rebuild the anchor from the INSTALLED one.

    That is the property the marker depends on: the two verbs exist to change the
    revocation list and nothing else, so a re-stage that dropped the marker would
    silently switch enforcement OFF as a side effect of revoking a phone — the
    exact downgrade an agent able to trigger either verb would want. Asserted by
    reading the STAGED file the install step would move, not the in-memory anchor.
    """
    from local_operator.operator.pair_handlers import _stage_anchor_revocation
    from local_operator.operator.trust import staging_path
    from local_operator.paths import config_dir

    _install(rig.anchor_root, rig.anchor("enforced"))
    assert _stage_anchor_revocation(rig.config_root, rig.unlisted_id, revoked=True)
    staged = json.loads(staging_path(config_dir()).read_bytes())
    assert staged["admission"] == "listed"
    assert {"device_id": rig.unlisted_id, "revoked": True} in staged["devices"]

    # Installing the staged statement makes it the anchor the next verb reads.
    _install(rig.anchor_root, trust.OperatorAnchor.from_json(staged))
    assert _stage_anchor_revocation(rig.config_root, rig.unlisted_id, revoked=False)
    restaged = json.loads(staging_path(config_dir()).read_bytes())
    assert restaged["admission"] == "listed"
    assert rig.unlisted_id not in [entry.get("device_id") for entry in restaged["devices"]]
    assert {"device_id": rig.listed_id} in restaged["devices"], "the listed phone kept its entry"


# ---------------------------------------------------------------------------
# Fail-closed on the anchor itself
# ---------------------------------------------------------------------------


def test_a_marker_on_an_anchor_root_does_not_own_is_no_anchor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The marker is only as good as the file it sits in.

    A same-uid file that says ``"admission": "listed"`` is not an anchor at all —
    the loader refuses it before any parse — so the switch cannot be turned on, or
    off, by anything the gated user can write. Asserted with the REAL ownership
    gate (no fake), which is the state a hand-planted file produces.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    root = _anchor_root(monkeypatch, tmp_path / "anchors")
    handle = create_key(config_root=tmp_path, preference=FILE_ONLY)
    planted = replace(anchor_for_handle(handle), admission=trust.ADMISSION_LISTED)
    _install(root, planted)

    loaded = trust.load_anchor()
    assert loaded.usable is False
    assert loaded.anchor is None
    assert "not root" in loaded.reason

    from local_operator.operator import operator_authority_report

    assert operator_authority_report()["device_admission"] is False


# ---------------------------------------------------------------------------
# What the reporting surfaces say — the dormant switch has to be visible
# ---------------------------------------------------------------------------


def test_status_says_admission_is_not_enforced_yet(rig: _Rig, capsys: Any) -> None:
    """A dormant switch that only the source knows about is a claim never made."""
    from local_operator.operator import handlers as operator_handlers

    _install(rig.anchor_root, rig.anchor("dormant"))
    assert operator_handlers.dispatch(argparse.Namespace(operator_command="status")) == 0
    printed = capsys.readouterr().out
    assert "NOT ENFORCED YET" in printed
    assert "without being confirmed on this computer" in printed


def test_status_and_the_listing_name_an_unlisted_phone_as_not_confirmed(
    rig: _Rig, capsys: Any
) -> None:
    """Under enforcement the level does not say WHICH phones it applies to.

    The listing does, and a phone this computer holds a certificate for but does
    not list is refused by the runtime — so a row that read as an ordinary phone
    would be the one wrong reading a human could take from it.
    """
    from local_operator.operator import handlers as operator_handlers
    from local_operator.operator.devices import write_device_cert
    from local_operator.operator.verify import read_device_cert

    _install(rig.anchor_root, rig.anchor("enforced"))
    certificate, declared, _ = rig.certificate("unlisted")
    parsed = read_device_cert(certificate)
    assert parsed is not None
    write_device_cert(
        rig.config_root,
        certificate=certificate,
        parsed=parsed,
        operator_key_id=rig.base.key_id,
        name="old phone",
    )

    assert operator_handlers.dispatch(argparse.Namespace(operator_command="status")) == 0
    printed = capsys.readouterr().out
    assert "phone admission        : enforced" in printed
    assert "not confirmed on this computer" in printed
    assert declared in printed

    # `lop operator devices` — the same verb that revokes them, which is where an
    # operator deciding what to do about it goes next.
    assert (
        operator_handlers.dispatch(
            argparse.Namespace(
                operator_command="devices", revoke="", authorise="", print_only=False
            )
        )
        == 0
    )
    listed = capsys.readouterr().out
    assert "phone admission        : enforced" in listed
    assert "unconfirmed" in listed


def test_the_devices_listing_says_the_switch_is_off_on_a_dormant_host(
    rig: _Rig, capsys: Any
) -> None:
    """The other half of the same honesty: unflipped hosts are told plainly."""
    from local_operator.operator import handlers as operator_handlers
    from local_operator.operator.devices import write_device_cert
    from local_operator.operator.verify import read_device_cert

    _install(rig.anchor_root, rig.anchor("dormant"))
    certificate, _, _ = rig.certificate("unlisted")
    parsed = read_device_cert(certificate)
    assert parsed is not None
    write_device_cert(
        rig.config_root,
        certificate=certificate,
        parsed=parsed,
        operator_key_id=rig.base.key_id,
        name="still works",
    )

    assert (
        operator_handlers.dispatch(
            argparse.Namespace(
                operator_command="devices", revoke="", authorise="", print_only=False
            )
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "NOT ENFORCED YET" in printed
    assert "unconfirmed" not in printed
