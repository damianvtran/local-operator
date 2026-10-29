"""The ``operator_device_pair`` card, its typed code check, and its apply.

WHAT IS REAL HERE, and what is substituted. REAL: the approval store (filing, the write-once
decision, ``request_digest``, the run lease, receipts, the deny), the operator key (a real
file-only ES256 key from ``create_key``, used to sign the decision AND to issue the
certificate), the SAS derivation and its constant-time compare, ``issue_device_cert`` /
``verify_device_cert``, and ``devices.write_device_cert`` / ``devices.is_revoked``.
SUBSTITUTED, and named where it happens:

* ``desktop_approval.assurance`` is pinned per cell -- the assurance LEVEL is a host fact
  (the backend of an installed, root-owned anchor) and no headless rig has one. The table
  that decides what a FILE-ONLY key may sign is the shipped registry, not a stub;
* no cell raises a real Touch ID sheet or a real OS admin prompt. A pairing on a
  presence-gated host costs TWO of those (the decision, then the certificate); these cells
  therefore prove the two SIGNER CALLS, not the gestures behind them. Real-host QA is
  blocked on this Mac -- the anchor install is root-owned -- and the PR says so.
"""

from __future__ import annotations

import base64
import copy
from dataclasses import replace
import time
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

from local_operator.network import approvals as A
from local_operator.network.credentials import desktop_approval
from local_operator.network.types import MeshRefusal
from local_operator.operator import devices, pair_op
from local_operator.operator.verify import (
    key_id_for,
    read_device_cert,
    spki_fp,
    verify_device_cert,
)
from tests.unit.network.test_approvals_store import _make_key, _sign
from tests.unit.network.test_approvals_store import root as approvals_root  # noqa: F401
from tests.unit.network.test_credentials_real_link import _lop_network

FILE_ONLY = ("file_only", "operator-file-only")
PRESENCE = ("presence", "operator-presence")
NAME = "Damian's iPhone"


@pytest.fixture
def root(approvals_root: Path) -> Path:  # noqa: F811 — imported fixture
    """Reuse the approval store's isolated root fixture without a module-level shadow."""
    return approvals_root


@pytest.fixture(autouse=True)
def _isolated_daemon_state() -> Any:
    """The typed engine's attempt/cancel latches are per-daemon process state."""
    desktop_approval._ATTEMPTS.clear()
    desktop_approval._CANCELLED.clear()
    yield
    desktop_approval._ATTEMPTS.clear()
    desktop_approval._CANCELLED.clear()


def _level(monkeypatch: pytest.MonkeyPatch, level: tuple[str | None, str]) -> None:
    """Pin the assurance LEVEL (a host fact) without stubbing the policy that reads it."""
    monkeypatch.setattr(desktop_approval, "assurance", lambda: level)


def _point() -> bytes:
    """A fresh ES256 point: a phone's key, or (for ``computer_id_pub``) the computer's."""
    key = ec.generate_private_key(ec.SECP256R1())
    return key.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode((text + "=" * (-len(text) % 4)).encode("ascii"))


def _trio(where: Path) -> dict[str, Any]:
    trio = A.local_anchor_trio(where)
    assert trio is not None, "the rig's staged operator anchor did not read"
    return trio


def _request(
    *,
    phone: bytes,
    computer: bytes,
    name: str = NAME,
    platform: str = "ios",
    scopes: list[str] | None = None,
    role: str = "approver",
    via: dict[str, Any] | None = None,
    lifetime_s: int = 90 * 24 * 3600,
    request_id: str | None = None,
    np: bytes | None = None,
    nd: bytes | None = None,
    expires_in: float = 300.0,
) -> dict[str, Any]:
    """The argument set ``create_request`` takes, as the filing route builds it."""
    nonce = np if np is not None else b"\x11" * 32
    return {
        "kind": A.KIND_OPERATOR_DEVICE_PAIR,
        "request_id": request_id or A.new_request_id(),
        "requested_by": {"surface": "phone", "device_id": key_id_for(phone), "session_id": ""},
        "device": {
            "device_id": devices.new_device_id(phone),
            "spki": devices.encode_spki(phone),
            "key_fp": spki_fp(phone),
            "name": name,
            "platform": platform,
        },
        "what": {
            pair_op.OP: {
                "role": role,
                "scopes": list(scopes if scopes is not None else devices.DEVICE_SCOPES),
                "cert_lifetime_s": lifetime_s,
                "via": dict(via or {"route": "radient", "account": "acct_test"}),
                "sas": {
                    "commit": pair_op.sas_commit(nonce),
                    "nd": _b64(nd if nd is not None else b"\x22" * 32),
                    "np": _b64(nonce),
                },
                "computer_id_pub": devices.encode_spki(computer),
            },
            "anchor": {},
        },
        "credential_ref": None,
        "created_at": time.time(),
        "expires_at": time.time() + expires_in,
    }


def _file(where: Path, **over: Any) -> dict[str, Any]:
    """File a pairing record through the real store, against this root's anchor."""
    phone = over.pop("phone", None) or _point()
    computer = over.pop("computer", None) or _make_key(where).spki
    request = _request(phone=phone, computer=computer, **over)
    request["what"]["anchor"] = {
        k: _trio(where)[k] for k in ("key_id", "spki_fp", "statement_digest")
    }
    return A.create_request(**request, root=where)


def _phone_code(record: dict[str, Any], where: Path) -> str:
    """The six digits, derived the way the PHONE derives them -- not via a private helper.

    An INDEPENDENT implementation of the published expression (HKDF over
    ``np || nd || device_spki || operator_spki || approval_id``), so the cells prove the
    contract a phone implements rather than echoing the module's own call.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    sas = record["what"][pair_op.OP]["sas"]
    okm = HKDF(algorithm=hashes.SHA256(), length=8, salt=None, info=b"lop-pair-sas-v1").derive(
        _unb64(sas["np"])
        + _unb64(sas["nd"])
        + _unb64(record["device"]["spki"])
        + _trio(where)["spki"]
        + record["approval_id"].encode("utf-8")
    )
    return f"{int.from_bytes(okm, 'big') % 1_000_000:06d}"


def _body(record: dict[str, Any]) -> dict[str, str]:
    """The typed card's acknowledgement: what it showed, as digests."""
    return {
        "operation": pair_op.OP,
        "request_digest": str(record["request_digest"]),
        "disclosure_digest": desktop_approval.disclosure_digest(
            pair_op.disclosure(copy.deepcopy(record))
        ),
    }


def _approve(where: Path, record: dict[str, Any]) -> dict[str, Any]:
    """The typed desktop approve, in-process: the route's own engine call, no HTTP."""
    body = _body(record)
    return desktop_approval.approve(
        where,
        str(record["approval_id"]),
        op=body["operation"],
        request_digest=body["request_digest"],
        disclosure_digest_=body["disclosure_digest"],
    )


def _counted_signer(where: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count decision signatures; the signature itself stays the real software one."""
    calls = {"signer": 0}

    def counted(**kw: Any) -> str:
        calls["signer"] += 1
        return _sign(
            where,
            A.signed_payload(
                kind=kw["kind"],
                request_id=kw["request_id"],
                request_digest=kw["request_digest"],
                decision=kw.get("decision", "approve"),
                decided_at=kw["decided_at"],
            ),
        )

    monkeypatch.setattr(A, "sign_decision", counted)
    return calls


# ---------------------------------------------------------------------------
# The record and its disclosure
# ---------------------------------------------------------------------------


def test_a_filed_pair_record_carries_the_designed_shape_and_disclosure(root: Path) -> None:
    _make_key(root)
    record = _file(root)

    assert record["kind"] == A.KIND_OPERATOR_DEVICE_PAIR
    assert A.KIND_OPERATOR_DEVICE_PAIR in A.KINDS
    assert record["state"] == A.STATE_REQUESTED
    assert pair_op.intent(record) is not None
    assert pair_op.exact_shape(record) is True
    facts = pair_op.disclosure(copy.deepcopy(record))
    assert facts["phone_name"] == NAME
    assert facts["platform"] == "ios"
    assert facts["via_words"] == "through your Radient account"
    assert facts["grants"] == [
        "loosen limits on this computer",
        "approve requests on this computer",
    ]
    assert facts["lifetime_days"] == 90
    assert facts["signing_principal"] == record["what"]["anchor"]["key_id"]
    # NEVER the digits: they are compared in constant time, never stored, never disclosed.
    # The record carries a COMMITMENT to the phone's nonce instead.
    code = _phone_code(record, root)
    assert code not in repr(facts)
    assert code not in repr(record)
    assert record["what"][pair_op.OP]["sas"]["commit"] != record["what"][pair_op.OP]["sas"]["np"]


def test_the_disclosure_follows_the_signed_payload_not_a_default(root: Path) -> None:
    _make_key(root)
    custom = pair_op.disclosure(
        copy.deepcopy(
            _file(
                root,
                name="Pixel",
                platform="android",
                via={"route": "custom"},
                lifetime_s=30 * 24 * 3600,
            )
        )
    )
    assert custom["phone_name"] == "Pixel"
    assert custom["platform"] == "android"
    assert custom["via_words"] == "through your own tunnel address"
    assert custom["lifetime_days"] == 30
    # The Radient route names the account; the custom route has none to name.
    assert custom["via"] == {"route": "custom"}
    assert custom["requested_from"] == "your phone"

# ---------------------------------------------------------------------------
# One decision, one signature, one certificate
# ---------------------------------------------------------------------------


def test_a_typed_approve_with_the_right_code_signs_once_and_installs_a_verifying_cert(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One signer call per DECISION; the certificate is the second, disclosed signature."""
    _make_key(root)
    _level(monkeypatch, FILE_ONLY)
    record = _file(root)
    approval_id = str(record["approval_id"])
    device_id = str(record["device"]["device_id"])
    phone = _unb64(record["device"]["spki"])

    # The typed code, checked BEFORE any signer runs. It leaves a receipt, not a decision.
    # The phone posted its request through the relay before anyone looked at the card.
    devices.write_pending(
        root,
        device_id=device_id,
        name=NAME,
        spki=str(record["device"]["spki"]),
        code="123456",
    )
    assert devices.read_pending(root, device_id) is not None

    pair_op.check_ack(approval_id, code=_phone_code(record, root), root=root)
    calls = _counted_signer(root, monkeypatch)
    admissions: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        "local_operator.operator.handlers.admit_device",
        lambda config_root, phone_id: admissions.append((config_root, phone_id)) or 0,
    )

    approved = _approve(root, record)
    assert approved["state"] == A.STATE_APPROVED
    assert calls["signer"] == 1, "a typed approve must cost exactly one signature"
    assert approved["signature"]["key_id"] == _trio(root)["key_id"]

    pair_op.apply(approval_id, pair_op.expected(A.load_record(approval_id, root=root)), root)

    assert admissions == [(root, device_id)]
    stored = devices.read_device(root, device_id)
    assert stored is not None, "apply did not install the certificate"
    assert stored.operator_key_id == _trio(root)["key_id"]
    assert stored.scope == devices.DEVICE_SCOPES
    # THE RUNTIME'S OWN TWO CALLS: parse the statement for the id, then verify it under the
    # pinned operator key (and refuse it outside its window).
    parsed = read_device_cert(stored.certificate)
    assert parsed is not None
    assert parsed.device_id == device_id
    assert (
        verify_device_cert(
            stored.certificate, operator_spki=_trio(root)["spki"], now=int(time.time())
        )
        == phone
    )
    assert (
        verify_device_cert(
            stored.certificate, operator_spki=_trio(root)["spki"], now=parsed.not_after + 1
        )
        is None
    ), "an expired certificate must not verify"
    assert devices.is_revoked(root, device_id) is False

    finished = A.load_record(approval_id, root=root)
    assert finished["state"] == A.STATE_CONNECTED
    assert any(r.get("step") == pair_op.STEP_INSTALL for r in finished["receipts"])
    # The pairing request the phone posted is SPENT by the install: the phone reads its
    # certificate, so a second install of the same pending request must find nothing.
    assert devices.read_pending(root, device_id) is None


def test_an_approval_without_the_typed_code_is_never_written(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per-kind decision guard: a signature obtained without the code is not recorded."""
    _make_key(root)
    _level(monkeypatch, FILE_ONLY)
    record = _file(root)
    approval_id = str(record["approval_id"])
    _counted_signer(root, monkeypatch)

    with pytest.raises(MeshRefusal) as raised:
        _approve(root, record)
    assert raised.value.code == "pair_code_required"
    assert A.load_record(approval_id, root=root)["state"] == A.STATE_REQUESTED


def test_a_wrong_code_costs_a_strike_and_the_third_closes_the_card(root: Path) -> None:
    _make_key(root)
    record = _file(root)
    approval_id = str(record["approval_id"])
    right = _phone_code(record, root)
    wrong = f"{(int(right) + 1) % 1_000_000:06d}"

    for _ in range(2):
        with pytest.raises(MeshRefusal) as raised:
            pair_op.check_ack(approval_id, code=wrong, root=root)
        assert raised.value.code == "pair_code_mismatch"
        assert "left" in str(raised.value)
        assert A.load_record(approval_id, root=root)["state"] == A.STATE_REQUESTED

    with pytest.raises(MeshRefusal) as raised:
        pair_op.check_ack(approval_id, code=wrong, root=root)
    assert raised.value.code == "pair_code_mismatch"
    assert "now closed" in str(raised.value)
    denied = A.load_record(approval_id, root=root)
    assert denied["state"] == A.STATE_DENIED
    assert pair_op.exact_shape(copy.deepcopy(denied)) is True

    # A decided card accepts no further code: the deny is write-once, and even the RIGHT
    # code cannot reopen it.
    with pytest.raises(MeshRefusal) as raised:
        pair_op.check_ack(approval_id, code=right, root=root)
    assert raised.value.code == "approval_decision_conflict"


def test_a_transcription_that_is_not_six_digits_does_not_burn_a_strike(root: Path) -> None:
    """Not a guess at the value, so it must not close a card: refused, no receipt written."""
    _make_key(root)
    record = _file(root)
    approval_id = str(record["approval_id"])

    with pytest.raises(MeshRefusal) as raised:
        pair_op.check_ack(approval_id, code="48192", root=root)
    assert raised.value.code == "pair_code_invalid"
    assert A.load_record(approval_id, root=root)["receipts"] == []


def test_the_code_is_a_function_of_the_record_and_the_operator_key_alone(root: Path) -> None:
    """The published expression, checked against an independent derivation."""
    _make_key(root)
    record = _file(root)
    trio = _trio(root)
    sas = record["what"][pair_op.OP]["sas"]

    def derive(approval_id: str) -> str:
        return pair_op.sas_code(
            np=_unb64(sas["np"]),
            nd=_unb64(sas["nd"]),
            device_spki=_unb64(record["device"]["spki"]),
            operator_spki=trio["spki"],
            approval_id=approval_id,
        )

    assert derive(str(record["approval_id"])) == _phone_code(record, root)
    assert derive("ap_other") != _phone_code(record, root)


# ---------------------------------------------------------------------------
# Every generic surface refuses; deny stays available
# ---------------------------------------------------------------------------


def test_every_generic_surface_refuses_a_pair_card_and_deny_stays_available(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_key(root)
    # The typed approve at step 5 runs the shipped policy, which reads the host's assurance.
    _level(monkeypatch, FILE_ONLY)
    record = _file(root)
    approval_id = str(record["approval_id"])

    # 1. THE ONE CHECK both generic approvers run, before any signer: the terminal verb.
    from local_operator.network import approval_ops

    for surface in (approval_ops.SURFACE_CLI, approval_ops.SURFACE_DESKTOP):
        with pytest.raises(MeshRefusal) as raised:
            approval_ops.require_generic_approvable(record, approval_id, surface=surface)
        assert raised.value.code == "pair_typed_approval_required"

    # 2. The bodiless desktop `/approve` is that same function.
    from local_operator.server.utils import desktop_approvals as desktop_utils

    with pytest.raises(MeshRefusal) as raised:
        desktop_utils.approve(root, approval_id)
    assert raised.value.code == "pair_typed_approval_required"

    # 3. The real `lop network approvals approve` verb, parsed and dispatched as typed.
    # CLI dispatch converts the typed refusal to its ordinary nonzero result and prints it.
    assert _lop_network("approvals", "approve", approval_id) != 0
    assert "phone pairing request" in capsys.readouterr().err

    # 4. `approvals run` on an undecided card runs nothing, whatever its kind.
    assert _lop_network("approvals", "run", approval_id) != 0
    assert "only an approved" in capsys.readouterr().err
    assert A.load_record(approval_id, root=root)["state"] == A.STATE_REQUESTED

    # 5. Decided, the card still never reaches an onboarding runner: a pairing is not an
    #    SSH install, and the lease refuses a kind it does not own.
    pair_op.check_ack(approval_id, code=_phone_code(record, root), root=root)
    _approve(root, record)
    assert _lop_network("approvals", "run", approval_id) != 0
    assert "onboarding" in capsys.readouterr().err
    assert A.load_record(approval_id, root=root)["state"] == A.STATE_APPROVED

    # 6. DENY is ordinary and stays available on every surface, code or no code -- including
    #    this approved-but-never-applied card (the operator changed their mind).
    assert _lop_network("approvals", "deny", approval_id) == 0
    assert A.load_record(approval_id, root=root)["state"] == A.STATE_DENIED


def test_the_card_lines_name_a_pairing_and_no_terminal_command(root: Path) -> None:
    """The fallthrough would be the ONBOARDING card ("install anchor", "definitions")."""
    _make_key(root)
    record = _file(root)
    lines = pair_op.card_lines(copy.deepcopy(record))
    text = "\n".join(lines)
    assert "phone" in text.lower()
    assert NAME in text
    assert "approve requests on this computer" in text
    # The approve act is on the computer's sheet, so no command is offered for it.
    assert "approvals approve" not in text
    assert "anchor" not in text.lower()
    assert "MCP" not in text


# ---------------------------------------------------------------------------
# Revocation, at filing and at apply
# ---------------------------------------------------------------------------


def test_a_revoked_key_is_refused_at_filing_and_at_apply(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_key(root)
    _level(monkeypatch, FILE_ONLY)
    record = _file(root)
    approval_id = str(record["approval_id"])
    device_id = str(record["device"]["device_id"])
    pair_op.check_ack(approval_id, code=_phone_code(record, root), root=root)
    _approve(root, record)

    # Revoked AFTER the decision: the privileged writer refuses before the second gesture.
    devices.record_revocation(root, device_id)
    signer_calls: list[str] = []
    monkeypatch.setattr(
        pair_op, "_issue_certificate", lambda **kwargs: signer_calls.append("sign")
    )

    with pytest.raises(MeshRefusal) as raised:
        pair_op.apply(approval_id, pair_op.expected(A.load_record(approval_id, root=root)), root)
    assert raised.value.code == "pair_device_revoked"
    assert not signer_calls
    assert devices.read_device(root, device_id) is None, "a refused apply installed a cert"
    # The lease is taken BEFORE the privileged write (agent review round 2, M2), so a
    # refusal releases the run: `failed` is retry-eligible (`begin_run` accepts
    # approved|failed), which keeps the card usable once the phone is brought back.
    assert A.load_record(approval_id, root=root)["state"] == A.STATE_FAILED

    # ... and at FILING: a revoked device cannot even ask again. The id it would install is
    # derived from the revoked key, so re-pairing by approval is not a way back.
    with pytest.raises(MeshRefusal) as raised:
        _file(root, phone=_unb64(record["device"]["spki"]))
    assert raised.value.code == "pair_device_revoked"


def test_admit_refuses_a_phone_revoked_while_its_card_was_open(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M1: the state the product creates — card open, revocation lands — must refuse.

    THE CELL IT REPLACES PASSED FOR THE WRONG REASON (agent review round 2, M1): it
    asserted the refusal `admit_device` gives a phone with no pending request, so a
    revocation would have gone unnoticed. Here every other guard is satisfied — the
    statement enforces its list and the phone HAS a pending pairing request — so the
    revocation is the only thing that can refuse it.
    """
    from types import SimpleNamespace

    from local_operator.operator import trust as trust_module

    _make_key(root)
    _level(monkeypatch, FILE_ONLY)
    record = _file(root)
    approval_id = str(record["approval_id"])
    device_id = str(record["device"]["device_id"])
    pair_op.check_ack(approval_id, code=_phone_code(record, root), root=root)
    _approve(root, record)

    staged = trust_module.load_staged_anchor(root)
    assert staged is not None
    installed = replace(
        staged,
        admission=trust_module.ADMISSION_LISTED,
        devices=({"device_id": device_id, "revoked": True},),
    )
    # The same ownership seam test_device_admission_writes uses: the installed
    # statement is root-owned on a real host and no headless rig has one.
    anchor_root = root / "anchor-root"
    anchor_root.mkdir(exist_ok=True)
    monkeypatch.setattr(trust_module, "_ANCHOR_ROOT_OVERRIDE", anchor_root)
    target = trust_module.anchor_path()
    target.write_bytes(trust_module.anchor_bytes(installed))
    real_open = trust_module._open_no_follow

    def as_root(path: Path) -> Any:
        opened = real_open(path)
        assert opened is not None
        descriptor, _ = opened
        return descriptor, SimpleNamespace(st_uid=0, st_mode=0o100644)

    monkeypatch.setattr(trust_module, "_open_no_follow", as_root)
    # The card is OPEN: the pairing request has not been dropped yet, so the pending
    # guard passes and nothing but the revocation stands in this phone's way.
    devices.write_pending(root, device_id=device_id, name=NAME, spki="00", code="123456")
    # `lop operator init` already staged a statement for the bootstrap install, so the
    # stage exists before this call; what must not change is its CONTENT.
    staged_before = trust_module.staging_path(root).read_bytes()
    # THE RIG PINS "NO INSTALLED ANCHOR" BY DEFAULT — no headless host has a
    # root-owned one — and this cell is ABOUT the installed statement, so the pin is
    # lifted for it. `handlers` bound the reader at import time, which is the only
    # handle on the real function while the fixture's patch is live.
    from local_operator.operator import handlers
    from local_operator.operator.trust import device_is_revoked

    monkeypatch.setattr(trust_module, "load_anchor", handlers.load_anchor)
    # SETUP CHECK: the state under test really revokes this phone, through both the
    # predicate the runtime uses and the product's combined question.
    loaded = trust_module.load_anchor()
    assert loaded.usable and loaded.anchor is not None
    assert device_is_revoked(loaded.anchor, device_id)
    assert devices.is_revoked(root, device_id)

    prompts: list[Any] = []
    monkeypatch.setattr(
        "local_operator.operator.elevate.run_elevated", lambda command: prompts.append(command)
    )
    signer_calls: list[str] = []
    monkeypatch.setattr(pair_op, "_issue_certificate", lambda **kwargs: signer_calls.append("sign"))

    with pytest.raises(MeshRefusal) as raised:
        pair_op.apply(approval_id, pair_op.expected(A.load_record(approval_id, root=root)), root)

    assert raised.value.code == "pair_device_revoked", raised.value.code
    assert signer_calls == [], "a revoked phone's certificate was signed"
    assert devices.read_device(root, device_id) is None, "a refused apply installed a cert"
    assert prompts == [], "a revoked phone spent the administrator prompt"
    assert target.read_bytes() == trust_module.anchor_bytes(installed), (
        "the admission write replaced the revoked entry"
    )
    assert trust_module.staging_path(root).read_bytes() == staged_before, (
        "a refused admit rewrote the staged statement"
    )


@pytest.mark.parametrize("decided", ["requested", "denied"])
def test_a_card_that_cannot_be_applied_costs_no_prompt_and_no_write(
    root: Path, monkeypatch: pytest.MonkeyPatch, decided: str
) -> None:
    """M2: the state guards run BEFORE the privileged write, so a refusal is free.

    At the previous head a card in `requested` or `denied` spent the administrator
    prompt and rewrote the root-owned anchor, and only THEN reached `begin_run` — the
    state machine's real gate — and refused. Both states are reachable: a filed card
    is `requested` until it is approved, and `deny` is offered on every surface.
    """
    from local_operator.operator import handlers

    _make_key(root)
    _level(monkeypatch, FILE_ONLY)
    record = _file(root)
    approval_id = str(record["approval_id"])
    if decided == "denied":
        A.deny(approval_id, root=root)
    admits: list[str] = []
    monkeypatch.setattr(
        handlers, "admit_device", lambda *args: admits.append(str(args[-1])) or 0
    )
    prompts: list[Any] = []
    monkeypatch.setattr(
        "local_operator.operator.elevate.run_elevated", lambda command: prompts.append(command)
    )

    with pytest.raises(MeshRefusal) as raised:
        pair_op.apply(approval_id, pair_op.expected(A.load_record(approval_id, root=root)), root)

    assert raised.value.code == "approval_not_runnable", raised.value.code
    assert admits == [], "a card that cannot be applied still reached the privileged write"
    assert prompts == [], "a card that cannot be applied still asked for approval"
    state = A.load_record(approval_id, root=root)["state"]
    assert state == (A.STATE_REQUESTED if decided == "requested" else A.STATE_DENIED), state


def test_a_deny_during_the_prompt_stops_the_certificate_and_records_the_stop(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 3 m1: the lease precedes the OS prompt, so a deny can land inside it.

    `connecting → denied` is a matrix row, and its receipt promises the runner stops
    at its next step check. At `f49cd3822eee` nothing checked between the prompt and
    the signer: the certificate was signed and written on a `denied` record, and the
    release's `denied → failed` refusal was swallowed, so no receipt showed the stop.
    """
    from local_operator.operator import handlers

    _make_key(root)
    _level(monkeypatch, FILE_ONLY)
    record = _file(root)
    approval_id = str(record["approval_id"])
    device_id = str(record["device"]["device_id"])
    pair_op.check_ack(approval_id, code=_phone_code(record, root), root=root)
    _counted_signer(root, monkeypatch)
    _approve(root, record)

    def deny_while_prompting(config_root: Path, phone_id: str) -> int:
        # The person answers the OS prompt; meanwhile another surface denies the card.
        assert A.load_record(approval_id, root=root)["state"] == A.STATE_CONNECTING
        A.deny(approval_id, root=root)
        return handlers.ADMIT_OK

    monkeypatch.setattr(handlers, "admit_device", deny_while_prompting)
    signed: list[str] = []
    monkeypatch.setattr(pair_op, "_issue_certificate", lambda **kwargs: signed.append("sign"))

    with pytest.raises(MeshRefusal) as raised:
        pair_op.apply(approval_id, pair_op.expected(A.load_record(approval_id, root=root)), root)

    assert raised.value.code == "approval_denied", raised.value.code
    assert signed == [], "a certificate was signed after the card was denied"
    assert devices.read_device(root, device_id) is None, "a denied card installed a cert"
    finished = A.load_record(approval_id, root=root)
    assert finished["state"] == A.STATE_DENIED, "the operator's deny must stay final"
    run_id = str(finished["run"]["run_id"])
    stops = [
        r
        for r in finished["receipts"]
        if r.get("run_id") == run_id and r.get("step") == pair_op.STEP_INSTALL
    ]
    assert stops and stops[-1]["ok"] is False, finished["receipts"]
    assert stops[-1]["detail"] == "approval_denied", stops[-1]


# ---------------------------------------------------------------------------
# The file-only row, and the shape predicate
# ---------------------------------------------------------------------------


def test_a_file_only_key_reads_unproven_human_and_may_still_sign(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operator a-2a8b: "Allow, with the warning and a 'Set up Touch ID' offer"."""
    _make_key(root)
    filed = _file(root)

    _level(monkeypatch, FILE_ONLY)
    from local_operator.network import approval_ops

    assert approval_ops.file_only_signing()["pair_phone"] is True
    row = desktop_approval.row_fields(filed)
    assert row is not None
    assert row["operation"] == pair_op.OP
    assert row["assurance"] == FILE_ONLY[0]
    assert row["unproven_human"] is True
    assert row["signable_here"] is True, "the file-only row must still be signable here"
    assert row["disclosure"]["phone_name"] == NAME
    # The row carries the upgrade offer, because that is the operator's answer: allow, warn,
    # and offer "Set up Touch ID". A row that warned without the offer would be a dead end.
    assert "upgrade" in row

    _level(monkeypatch, PRESENCE)
    presence_row = desktop_approval.row_fields(filed)
    assert presence_row is not None
    assert presence_row["assurance"] == PRESENCE[0]
    assert presence_row["unproven_human"] is False
    assert presence_row["signable_here"] is True
    assert "upgrade" not in presence_row
    # Same card, same facts: only the host fact changed.
    assert presence_row["disclosure_digest"] == row["disclosure_digest"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: record_payload(r).update({"note": "signed but never shown"}),
        lambda r: record_payload(r).update({"role": "viewer"}),
        lambda r: record_payload(r).update({"scopes": ["approve"]}),
        lambda r: record_payload(r).update({"cert_lifetime_s": 400 * 24 * 3600}),
        lambda r: record_payload(r).update({"computer_id_pub": _b64(b"not-a-point")}),
        lambda r: record_payload(r).update({"via": {"route": "carrier-pigeon"}}),
        lambda r: record_payload(r)["sas"].update({"np": _b64(b"\x33" * 32)}),
        lambda r: record_payload(r)["sas"].update({"extra": "x"}),
        lambda r: r["what"].update({"pair_other": {"a": 1}}),
        lambda r: r["device"].update({"host": "10.0.0.9"}),
        lambda r: r["device"].update({"platform": "mac"}),
        lambda r: r["device"].update({"name": "Line\nBreak"}),
        lambda r: r["requested_by"].update({"device_id": "d_other"}),
        lambda r: r.update({"credential_ref": {"kind": "ssh", "ref": "secret:x"}}),
    ],
)
def test_a_tampered_record_or_an_extra_key_refuses_the_one_shape_predicate(
    root: Path, mutate: Any
) -> None:
    """#3087's discipline: an undisclosed key or value is never signed, so it must refuse.

    ONE predicate (``exact_shape``) behind disclosure, apply, the filing guard and the typed
    route's operation lookup -- so no two of them can accept a record another refuses.
    """
    _make_key(root)
    record = copy.deepcopy(_file(root))
    assert pair_op.exact_shape(record) is True
    mutate(record)
    assert pair_op.exact_shape(record) is False
    with pytest.raises(MeshRefusal) as raised:
        pair_op.disclosure(copy.deepcopy(record))
    assert raised.value.code == "pair_payload_invalid"
    refusal = pair_op._PairOp().generic_refusal(record, str(record["approval_id"]))
    assert refusal is not None and refusal.code == "pair_payload_invalid"


def record_payload(record: dict[str, Any]) -> dict[str, Any]:
    return record["what"][pair_op.OP]


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


def test_a_replayed_request_id_with_a_changed_payload_refuses(root: Path) -> None:
    """An idempotent retry is the SAME record; a changed one is a new request id."""
    _make_key(root)
    phone = _point()
    first = _file(root, phone=phone, request_id=A.new_request_id())
    again = _file(root, phone=phone, request_id=first["request_id"])
    assert again["approval_id"] == first["approval_id"]

    with pytest.raises(MeshRefusal) as raised:
        _file(root, phone=_point(), request_id=first["request_id"])
    assert raised.value.code == "approval_request_conflict"


def test_a_stale_window_refuses_at_the_code_check_and_never_signs(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_key(root)
    _level(monkeypatch, FILE_ONLY)
    record = _file(root, expires_in=-1.0)
    approval_id = str(record["approval_id"])

    with pytest.raises(MeshRefusal) as raised:
        pair_op.check_ack(approval_id, code=_phone_code(record, root), root=root)
    assert raised.value.code == "approval_expired"

    calls = _counted_signer(root, monkeypatch)
    with pytest.raises(MeshRefusal) as raised:
        _approve(root, record)
    assert raised.value.code == "approval_expired"
    assert calls["signer"] == 0, "an expired card must not reach a signer"
