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

THE MIGRATION GUARD IS WHAT STAYS OPEN, and two tests say so where a human reads
it. An anchor with no ``admission`` marker was written before the rule existed, so
it behaves exactly as it does today (byte-identical statements, so no pinned
approval digest moves; an already-paired phone stays admitted with no re-pair) —
and the reporting surfaces NAME the guard rather than let the runtime's silence
read as the rule.

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
import time
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
        if kind == "pre_rule":
            return self.base
        if kind == "pre_rule_with_revocation":
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
#: * a host whose anchor PREDATES the rule (no marker) admits any device whose
#:   certificate this key signed — today's behaviour, kept by the migration guard so
#:   that every phone paired before the rule is not de-admitted by the upgrade;
#: * an ENFORCING host admits only the id its anchor lists;
#: * ``forged`` is refused everywhere, including where the migration guard applies: the
#:   certificate's id must be its key's id, which is what stops a statement from
#:   naming a permitted id while carrying somebody else's key. Every certificate
#:   ``lop pair`` mints derives its id, so this refuses hand-made statements only;
#: * ``presence: true`` admits an UNLISTED device BY DESIGN — that backend costs a
#:   human gesture per signature, so the signature already is the confirmation;
#: * a revoked device is refused at every level, which is the rule that predates
#:   this change and stays exactly as it was.
CASES: tuple[tuple[str, str, bool], ...] = (
    ("pre_rule", "listed", True),
    ("pre_rule", "unlisted", True),
    ("pre_rule", "forged", False),
    ("pre_rule", "revoked", True),
    ("pre_rule_with_revocation", "listed", True),
    ("pre_rule_with_revocation", "unlisted", True),
    ("pre_rule_with_revocation", "forged", False),
    ("pre_rule_with_revocation", "revoked", False),
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
    from local_operator.operator.verify import verify_device_cert
    from local_operator.session.runtime import server as runtime_server

    anchor = rig.anchor(anchor_kind)
    _install(rig.anchor_root, anchor)
    certificate, declared, key = rig.certificate(cert_kind)
    revoked = anchor_kind.endswith("_with_revocation") and cert_kind == "revoked"

    # 1. The predicate itself, on the two facts the certificate carries.
    assert trust.device_is_admitted(anchor, declared, key) is admitted

    # 2. The store wrapper the relay asks (a cert string, parsed here).
    assert device_store.is_admitted(certificate) is admitted

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
        body = json.loads(bytes(refusal.body))
        # TWO CODES, because the two remedies are different: a revocation is
        # lifted from this computer's device list, a not-yet-confirmed phone has
        # already done what pairing asks of it.
        assert body["code"] == (
            OPERATOR_DEVICE_REVOKED_CODE if revoked else OPERATOR_DEVICE_UNCONFIRMED_CODE
        )
        if revoked:
            assert (
                "Removing it on one computer does not remove it on your other computers."
                in body["error"]
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
    sentence = json.loads(bytes(refusal.body))["error"]
    assert "confirm" in sentence
    assert "Pairing it again will not help." in sentence


# ---------------------------------------------------------------------------
# The migration guard: no marker, no change — including the bytes
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
    pre_rule = rig.base
    enforced = rig.anchor("enforced")
    assert trust.statement_digest(pre_rule) != trust.statement_digest(enforced)
    assert json.loads(trust.anchor_bytes(enforced))["admission"] == "listed"


def test_the_marker_survives_a_revocation_and_an_authorisation_restage(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--revoke`` and ``--authorise`` rebuild the anchor from the INSTALLED one.

    That is the property the marker depends on: the two verbs exist to change the
    revocation list and nothing else, so a re-stage that dropped the marker would
    roll the host back to the migration guard as a side effect of revoking a phone —
    the exact downgrade an agent able to trigger either verb would want. Asserted by
    reading the STAGED file the install step would move, not the in-memory anchor.

    AND UN-REVOKING LISTS (agent review round 1, M1): on an enforcing host, dropping
    the entry would leave the phone unlisted, which the predicate refuses — so the
    un-revoke writes it back as a LISTED device instead. On a host under the migration
    guard there is no list to keep, so the entry is dropped there.
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
    restaged = trust.OperatorAnchor.from_json(staged)
    assert restaged is not None
    _install(rig.anchor_root, restaged)
    assert _stage_anchor_revocation(rig.config_root, rig.unlisted_id, revoked=False)
    restaged = json.loads(staging_path(config_dir()).read_bytes())
    assert restaged["admission"] == "listed"
    assert {"device_id": rig.unlisted_id} in restaged["devices"], "un-revoking LISTS it"
    assert {"device_id": rig.listed_id} in restaged["devices"], "the listed phone kept its entry"

    # The migration guard's host keeps the older shape: no list, so nothing to write.
    _install(rig.anchor_root, rig.anchor("pre_rule_with_revocation"))
    assert _stage_anchor_revocation(rig.config_root, rig.revoked_id, revoked=False)
    guarded = json.loads(staging_path(config_dir()).read_bytes())
    assert rig.revoked_id not in [entry.get("device_id") for entry in guarded["devices"]]


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
# What the reporting surfaces say — the guard state has to be visible
# ---------------------------------------------------------------------------


def test_status_names_the_migration_guard_on_a_pre_rule_anchor(rig: _Rig, capsys: Any) -> None:
    """The guard is named where a human reads it, and never as an "off" state.

    A guard only the source knows about is a claim never made: this surface states
    the RULE and says which statements the migration guard still carries, so an
    operator on an upgraded host is not told a boundary exists that their own anchor
    does not apply yet.
    """
    from local_operator.operator import handlers as operator_handlers

    _install(rig.anchor_root, rig.anchor("pre_rule"))
    assert operator_handlers.dispatch(argparse.Namespace(operator_command="status")) == 0
    printed = capsys.readouterr().out
    assert "phone admission        : waiting for you to confirm your phones" in printed
    assert "phones paired before this version keep working until you do" in printed
    assert "NOT ENFORCED" not in printed


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
    assert "phone admission        : on" in printed
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
    assert "phone admission        : on" in listed
    assert "unconfirmed" in listed
    # m1: the row names the verb that clears it, not only the state.
    assert "run `lop operator devices --approve" in listed, listed


def test_the_devices_listing_names_the_migration_guard_on_a_pre_rule_anchor(
    rig: _Rig, capsys: Any
) -> None:
    """The other half of the same honesty — and it reaches a host with no phones.

    A zero-phone host is the state the guard exists for, and it is the one host the
    line used to be hidden from (agent review round 1, n3), so the listing says it
    too rather than print nothing at all.
    """
    from local_operator.operator import handlers as operator_handlers

    _install(rig.anchor_root, rig.anchor("pre_rule"))
    assert (
        operator_handlers.dispatch(
            argparse.Namespace(
                operator_command="devices", revoke="", authorise="", print_only=False
            )
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "phone admission        : waiting for you to confirm your phones" in printed
    assert "phones paired before this version keep working until you do" in printed
    assert "NOT ENFORCED" not in printed
    assert "unconfirmed" not in printed
    # D6: the state is the whole discovery path, so the verb that ends it is named
    # UNDER it — the delta's first-launch prompt is not on this head.
    assert "run `lop operator devices --confirm` (it asks for your approval)" in printed


def test_the_copy_states_the_presence_exemption_the_predicate_applies(
    rig: _Rig, capsys: Any
) -> None:
    """m1: neither surface may claim a list requirement a presence host does not make.

    ``anchor.presence`` exempts a device BEFORE the marker is read, so on a
    Secure-Enclave host an unlisted phone IS admitted — the table's
    ``("presence_enforced", "unlisted", True)`` row is the runtime half of this. The
    copy has to agree with it, or the product makes exactly the over-claim this
    change exists to remove, in the other direction.
    """
    from local_operator.operator import handlers as operator_handlers
    from local_operator.operator.devices import write_device_cert
    from local_operator.operator.verify import read_device_cert

    _install(rig.anchor_root, rig.anchor("presence_enforced"))
    certificate, declared, _ = rig.certificate("unlisted")
    parsed = read_device_cert(certificate)
    assert parsed is not None
    write_device_cert(
        rig.config_root,
        certificate=certificate,
        parsed=parsed,
        operator_key_id=rig.base.key_id,
        name="this mac's phone",
    )

    assert operator_handlers.dispatch(argparse.Namespace(operator_command="status")) == 0
    printed = capsys.readouterr().out
    assert "not needed on this computer" in printed
    assert "must be confirmed on this computer" not in printed
    assert "not confirmed on this computer" not in printed

    assert (
        operator_handlers.dispatch(
            argparse.Namespace(
                operator_command="devices", revoke="", authorise="", print_only=False
            )
        )
        == 0
    )
    listed = capsys.readouterr().out
    assert "not needed on this computer" in listed
    assert "unconfirmed" not in listed
    assert declared in listed


# ---------------------------------------------------------------------------
# The remedy an enforcing host promises, and the fail-closed floor beneath it
# ---------------------------------------------------------------------------


def test_authorise_re_lists_the_phone_so_it_admits_again_with_the_same_id(rig: _Rig) -> None:
    """``--authorise`` must LIST the phone, not merely stop revoking it (M1).

    Under enforcement the revoked row's own promise ("brings this device back") and
    the relay's revoked copy ("allow this phone again, then pair this phone again")
    both depend on the un-revoke leaving the device ADMITTED. Dropping the entry left
    it unlisted — refused by the predicate — and a re-pair mints the SAME derived id,
    so the phone met ``operator_device_unconfirmed`` with no route out of the loop.

    Driven through the installed-anchor path, not the in-memory one: the same id the
    phone already holds has to admit again with no new certificate.
    """
    from local_operator.operator.pair_handlers import _stage_anchor_revocation
    from local_operator.operator.trust import staging_path
    from local_operator.paths import config_dir
    from local_operator.session.runtime import server as runtime_server

    _install(rig.anchor_root, rig.anchor("enforced_with_revocation"))
    certificate, declared, key = rig.certificate("revoked")
    assert trust.device_is_admitted(rig.anchor("enforced_with_revocation"), declared, key) is False

    assert _stage_anchor_revocation(rig.config_root, declared, revoked=False)
    staged = trust.OperatorAnchor.from_json(json.loads(staging_path(config_dir()).read_bytes()))
    assert staged is not None
    _install(rig.anchor_root, staged)

    # The statement the phone already holds is admitted again, same id.
    assert trust.device_is_admitted(staged, declared, key) is True
    assert {"device_id": declared} in staged.devices
    assert {"device_id": rig.listed_id} in staged.devices, "the other listed phone kept its entry"
    stub = _runtime_stub()
    assert runtime_server.RuntimeServer._device_cert_point(stub, certificate, staged) is not None


def test_authorise_does_not_admit_a_different_key_that_reuses_the_listed_id(rig: _Rig) -> None:
    """The negative: re-listing an id must not hand that id to another key.

    The invariant the whole predicate rests on is ``device_id == key_id_for(spki)``,
    and a re-list writes the DERIVED id back rather than a new one. So a statement
    that names the freshly listed id while carrying somebody else's key — the S4a
    spike's forgery, now aimed at a re-listed device — is still refused, at the
    predicate and at the runtime's own certificate point.
    """
    from local_operator.operator.pair_handlers import _stage_anchor_revocation
    from local_operator.operator.sign import issue_device_cert, load_signer
    from local_operator.operator.trust import staging_path
    from local_operator.paths import config_dir
    from local_operator.session.runtime import server as runtime_server

    _install(rig.anchor_root, rig.anchor("enforced_with_revocation"))
    assert _stage_anchor_revocation(rig.config_root, rig.revoked_id, revoked=False)
    staged = trust.OperatorAnchor.from_json(json.loads(staging_path(config_dir()).read_bytes()))
    assert staged is not None
    _install(rig.anchor_root, staged)
    assert rig.revoked_id in [entry.get("device_id") for entry in staged.devices]

    signer = load_signer(config_root=rig.config_root, backend_name="file-only")
    assert signer is not None
    try:
        forged = issue_device_cert(
            device_spki=rig.unlisted_key.spki,
            device_id=rig.revoked_id,
            label="not this phone",
            signer=signer,
        )
    finally:
        signer.close()
    assert trust.device_is_admitted(staged, rig.revoked_id, rig.unlisted_key.spki) is False
    stub = _runtime_stub()
    assert runtime_server.RuntimeServer._device_cert_point(stub, forged, staged) is None


def test_a_refusal_clears_within_the_anchor_window(
    rig: _Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """m2: a phone the human just confirmed may not be refused for five more minutes.

    ``_device_cert_point`` caches a REFUSED certificate, and the only anchor-driven
    transition that existed before listing was admit -> revoked, where caching the
    denial is right. Listing introduced the reverse, caused by the human's own
    gesture, so a refusal that outlived ``ANCHOR_REFRESH_S`` kept the phone out while
    the copy told the person to confirm it — against the split's "accepted within
    30 s" cell. An ADMITTED point keeps the certificate TTL: re-verifying a signature
    cannot change its answer.
    """
    from local_operator.session.runtime import server as runtime_server

    anchor = rig.anchor("enforced")
    _install(rig.anchor_root, anchor)
    certificate, declared, _ = rig.certificate("unlisted")

    clock = {"t": 1000.0}
    monkeypatch.setattr(
        runtime_server,
        "time",
        SimpleNamespace(monotonic=lambda: clock["t"], time=time.time),
    )
    stub = _runtime_stub()
    assert runtime_server.RuntimeServer._device_cert_point(stub, certificate, anchor) is None
    stored = stub._device_certs[certificate]
    assert stored[0] is None, "the refusal is what was cached"
    assert (
        stored[1] - clock["t"] <= trust.ANCHOR_REFRESH_S
    ), "a refusal may not outlive the window the confirming copy promises"

    # The human confirms this phone: the anchor now lists it. No new certificate.
    confirmed = replace(anchor, devices=anchor.devices + ({"device_id": declared},))
    clock["t"] += trust.ANCHOR_REFRESH_S + 1
    point = runtime_server.RuntimeServer._device_cert_point(stub, certificate, confirmed)
    assert point is not None, "the confirmation must take effect inside the anchor window"
    admitted_deadline = stub._device_certs[certificate][1]
    assert (
        admitted_deadline - clock["t"] > trust.ANCHOR_REFRESH_S
    ), "an admitted point keeps the certificate TTL"


def test_no_usable_anchor_refuses_a_device_certificate_at_the_runtime_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The codifying control for the courier's fail-open (review, reserved deviation 2).

    ``devices.is_admitted`` answers "admit" when it can read no anchor, which is only
    safe because the RUNTIME refuses every device certificate in that state before it
    verifies anything. That was reasoned from two files and pinned nowhere, so this
    drives a REAL certificate through the runtime's own certificate point with a real
    empty anchor root (the real ownership gate, no fake): the courier forwards, the
    runtime refuses, and nothing is verified on the way.
    """
    from local_operator.operator import devices as device_store
    from local_operator.operator.sign import issue_device_cert, load_signer
    from local_operator.operator.verify import verify_device_cert
    from local_operator.session.runtime import server as runtime_server

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _anchor_root(monkeypatch, tmp_path / "no-anchors")  # nothing installed, no st_uid fake
    create_key(config_root=tmp_path, preference=FILE_ONLY)  # a real operator key
    phone = FileKeyBackend(tmp_path / "phone.pem").create()
    declared = new_device_id(phone.spki)
    signer = load_signer(config_root=tmp_path, backend_name="file-only")
    assert signer is not None
    try:
        certificate = issue_device_cert(
            device_spki=phone.spki, device_id=declared, label="phone", signer=signer
        )
    finally:
        signer.close()

    loaded = trust.load_anchor()
    assert loaded.usable is False, "the empty root must not read as an anchor"
    assert loaded.anchor is None

    # The courier's answer, which is the state this control is about.
    assert device_store.is_admitted(certificate) is True

    calls: list[int] = []
    real_verify = verify_device_cert

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(runtime_server.operator_verify, "verify_device_cert", counting)
    stub = _runtime_stub()
    assert runtime_server.RuntimeServer._device_cert_point(stub, certificate, None) is None
    assert calls == [], "the refusal happens before verification, so nothing can be admitted"
