"""S4a-core: the typed approval op registry is a PURE REFACTOR of the credential path.

WHAT THIS FILE PROVES, and why each property needs its own test:

* GOLDEN DIGESTS. ``request_digest`` binds the signature and ``disclosure_digest`` binds
  what the card showed. Both were computed on ``origin/main`` f5c43172d3f -- BEFORE the
  registry existed -- from two representative records built by the real rigs in
  ``test_desktop_credential_approval`` (one finite share, one inheritance) and are pinned
  here as literals. A refactor that changes either for an existing record breaks every
  card in flight, so drift must fail loudly. (The ``[redacted]`` values are the rigs' own
  synthetic credential names, not elisions.)
* ROUTE PARITY. ``/approve-operation`` and ``/apply-operation`` are the op-generic twins of
  ``/approve-credential`` and ``/apply-credential``. Every scenario runs through both and
  must answer the same status AND body: success, 401 (no bearer), 503 (plane closed,
  #3039), 404 (unknown id) and 409 (state / digest mismatch).
* ONE GENERIC CHECK. The bodiless desktop ``/approve`` refuses a typed record through
  ``approval_ops.require_generic_approvable`` before any signer call, and the CLI verb runs
  the same check. Negative control: with the check stubbed out, the bodiless route signs.
* REGISTRY CONSISTENCY. ``FILE_ONLY_SIGNING`` is derived from the registry and holds
  exactly the values it held before.
"""

from __future__ import annotations

import copy
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.config import ConfigManager
from local_operator.network import approval_ops
from local_operator.network import approvals as A
from local_operator.network.credentials import (
    desktop_approval,
    finite_share,
    inheritance,
)
from local_operator.network.types import MeshRefusal
from local_operator.server import desktop
from local_operator.server.routes import desktop_approvals as approval_routes
from local_operator.server.routes import desktop_credentials as operation_routes
from local_operator.server.utils.desktop_sessions import DesktopSessions
from tests.unit.network.test_credentials_real_link import _lop_network

# ``_fresh_limits`` is autouse there (the abuse bounds are per-process state) and ``rig`` is a
# fixture: both are imported for pytest, not referenced by name.
from tests.unit.network.test_desktop_credential_approval import (  # noqa: F401
    FILE_ONLY,
    PRESENCE,
    ZERO,
    _ack,
    _counted,
    _finite,
    _fresh_limits,
    _inherit,
    _level,
    _row,
    rig,
)

TOKEN = "synthetic-s4a-core-bearer"

# ---------------------------------------------------------------------------
# Frozen pre-refactor records (captured on origin/main f5c43172d3f, never regenerated)
# ---------------------------------------------------------------------------

_FROZEN_FINITE: dict[str, Any] = {
    "approval_id": "ap_s5g8hm0s4vg4r",
    "audit": ["onboard_requested"],
    "created_at": 1793166592.530283,
    "credential_ref": None,
    "decided_at": 0.0,
    "device": {
        "device_id": "d_9ad8d32b03372d554f765b48d3263765",
        "fingerprint": "sha256:9a6e31865b728bd39c6eeebc119a864f67b9035c217ca1449376cb9257a9ebb3",
        "name": "",
    },
    "expires_at": 1793170192.530283,
    "kind": "device_onboard",
    "receipts": [],
    "request_digest": "sha256:8be5a8d55f911d884b7b00b8a00f6a436af5705e337200a18a8398c788656409",
    "request_id": "req_f84z2bv1c4nsm",
    "requested_by": {
        "device_id": "d_f380a75775e0db577bd6925b49a2efc4",
        "session_id": "",
        "surface": "cli",
    },
    "schema": 1,
    "signature": None,
    "state": "requested",
    "what": {
        "anchor": {
            "key_id": "2d434736b1fb972040872d29b4cf406c",
            "spki_fp": "2D43-4736-B1FB-9720",
            "statement_digest": "sha256:db1f280f720e23920a2b2c430964a36b48497bb633032ce5427f7e34d2c54008",  # noqa: E501 -- a frozen digest literal
        },
        "credential_finite_share": {
            "denies": [
                {
                    "approval_ref": "desktop-one",
                    "reason": "withdrawn",
                    "revision": 2,
                    "selector": {"credential_key": "permissiontest"},
                }
            ],
            "guard": None,
            "key": "permissiontest",
            "network_id": "n_111111111111111111111111",
            "owner_device": "d_f380a75775e0db577bd6925b49a2efc4",
            "policy_revision": 2,
            "public_key_fingerprint": "sha256:9a6e31865b728bd39c6eeebc119a864f67b9035c217ca1449376cb9257a9ebb3",  # noqa: E501 -- a frozen digest literal
            "recipient_device": "d_9ad8d32b03372d554f765b48d3263765",
            "scope": "session",
        },
    },
}
_FROZEN_FINITE_REQUEST_DIGEST = (
    "sha256:8be5a8d55f911d884b7b00b8a00f6a436af5705e337200a18a8398c788656409"
)
_FROZEN_FINITE_DISCLOSURE_DIGEST = (
    "sha256:db53863011ccfb66bd3550c68aa9426ab0b38f266f210a174c26e9c15234f3dc"
)
_FROZEN_INHERITANCE: dict[str, Any] = {
    "approval_id": "ap_96abwnag1hkxp",
    "audit": ["onboard_requested"],
    "created_at": 1793166592.5305521,
    "credential_ref": None,
    "decided_at": 0.0,
    "device": {
        "device_id": "d_7aae7d1ebb4c8018496a4fd3d64eb548",
        "fingerprint": "sha256:356dffc7cf6c0f2bf495511c4255ddf9e8936ff89ce16a9a4d177a1523ebd6a9",
        "name": "",
    },
    "expires_at": 1793170192.5305521,
    "kind": "device_onboard",
    "receipts": [],
    "request_digest": "sha256:24a9027360d1d948050b13e0864bd2f791107c3d59e3e2ce689d504ecfbf5d08",
    "request_id": "req_r9768fzyk0gk8",
    "requested_by": {
        "device_id": "d_6b336db2606ff6055bddc97d19c5c2df",
        "session_id": "",
        "surface": "cli",
    },
    "schema": 1,
    "signature": None,
    "state": "requested",
    "what": {
        "anchor": {
            "key_id": "1a86d02dd54896b4dd7003ea157b8285",
            "spki_fp": "1A86-D02D-D548-96B4",
            "statement_digest": "sha256:ba7bb622cbd79bab86f8579a169690d91a0965321ebff03288501f82bf73a3f9",  # noqa: E501 -- a frozen digest literal
        },
        "credential_inheritance": {
            "action": "grant",
            "inherit": {
                "mcp": {"future_logins": False, "urls": []},
                "origin": "owner-native-login",
                "providers": {"future_logins": True, "ids": ["permissiontest"]},
            },
            "network_id": "n_111111111111111111111111",
            "owner_device": "d_6b336db2606ff6055bddc97d19c5c2df",
            "public_key_fingerprint": "sha256:356dffc7cf6c0f2bf495511c4255ddf9e8936ff89ce16a9a4d177a1523ebd6a9",  # noqa: E501 -- a frozen digest literal
            "recipient_device": "d_7aae7d1ebb4c8018496a4fd3d64eb548",
            "scope": {
                "account_id": "account-one",
                "principal_ref": "1a86d02dd54896b4dd7003ea157b8285",
            },
        },
    },
}
_FROZEN_INHERITANCE_REQUEST_DIGEST = (
    "sha256:24a9027360d1d948050b13e0864bd2f791107c3d59e3e2ce689d504ecfbf5d08"
)
_FROZEN_INHERITANCE_DISCLOSURE_DIGEST = (
    "sha256:f9dfc492a1b21c609eb3b1ec13cae958563bfb75a1c3119d7f03baf2394e118c"
)


@pytest.mark.parametrize(
    ("record", "module", "request_hex", "disclosure_hex"),
    [
        (
            _FROZEN_FINITE,
            finite_share,
            _FROZEN_FINITE_REQUEST_DIGEST,
            _FROZEN_FINITE_DISCLOSURE_DIGEST,
        ),
        (
            _FROZEN_INHERITANCE,
            inheritance,
            _FROZEN_INHERITANCE_REQUEST_DIGEST,
            _FROZEN_INHERITANCE_DISCLOSURE_DIGEST,
        ),
    ],
    ids=["finite_share", "inheritance"],
)
def test_golden_credential_digests(
    record: dict[str, Any], module: Any, request_hex: str, disclosure_hex: str
) -> None:
    frozen = copy.deepcopy(record)
    assert A.request_digest(frozen) == request_hex == frozen["request_digest"]
    # Through the REGISTRY, the path the engine now takes -- not the module directly.
    op = desktop_approval.operation(frozen)
    assert op is not None and op == module.OP
    registered = approval_ops.get(op)
    assert registered is not None
    assert desktop_approval.disclosure_digest(registered.disclosure(frozen)) == disclosure_hex
    # And the module's own disclosure agrees, so the adapter adds nothing.
    assert desktop_approval.disclosure_digest(module.disclosure(frozen)) == disclosure_hex


def test_file_only_signing_is_derived_from_the_registry_unchanged() -> None:
    assert desktop_approval.FILE_ONLY_SIGNING == {finite_share.OP: True, inheritance.OP: False}
    assert dict(desktop_approval.FILE_ONLY_SIGNING) == approval_ops.file_only_signing()
    # Every registered op carries an explicit decision, and only registered ops are listed.
    assert {op.op for op in approval_ops.registered()} == set(desktop_approval.FILE_ONLY_SIGNING)
    for op in approval_ops.registered():
        assert isinstance(op, approval_ops.ApprovalOp)
        assert op.kind == A.KIND_DEVICE_ONBOARD


def test_registering_a_second_op_under_the_same_name_is_refused() -> None:
    existing = approval_ops.get(finite_share.OP)
    assert existing is not None
    impostor = copy.copy(existing)
    # Unsealed so the DUPLICATE check is what answers, not the seal.
    with approval_ops._unsealed_for_tests():
        with pytest.raises(ValueError, match="already registered"):
            approval_ops.register(impostor)
        # A family's refusal answers for every unknown member: never silently replaced.
        with pytest.raises(ValueError, match="already registered"):
            approval_ops.register_family("credential_", lambda *a: None)
    assert approval_ops.get(finite_share.OP) is existing


def test_the_registry_is_sealed_after_the_providers_load() -> None:
    """R2: once loaded, ``register``/``register_family`` refuse -- the registry is closed."""
    approval_ops.registered()  # any lookup finishes loading and seals
    fresh = _SyntheticTypedOp()
    with pytest.raises(RuntimeError, match="sealed"):
        approval_ops.register(fresh)
    with pytest.raises(RuntimeError, match="sealed"):
        approval_ops.register_family("synthetic_", lambda *a: None)
    assert approval_ops.get(fresh.op) is None
    # The test-only unseal restores the tables and the seal exactly.
    with approval_ops._unsealed_for_tests():
        approval_ops.register(fresh)
        assert approval_ops.get(fresh.op) is fresh
    assert approval_ops.get(fresh.op) is None
    with pytest.raises(RuntimeError, match="sealed"):
        approval_ops.register(fresh)


# ---------------------------------------------------------------------------
# Route parity: -credential and -operation answer identically
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _client(
    rig: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    *,
    bearer: bool = True,
    plane_open: bool = True,
) -> AsyncIterator[AsyncClient]:
    """Both routers, as ``server.app`` mounts them, over the rig's isolated store."""
    monkeypatch.setattr(desktop, "_SPAWN_BEARER", None)
    monkeypatch.setattr(desktop, "_CLAIMED", None)
    if plane_open:
        monkeypatch.setenv(desktop.TOKEN_ENV, TOKEN)
    else:
        monkeypatch.delenv(desktop.TOKEN_ENV, raising=False)
    app = FastAPI()
    app.state.config_manager = ConfigManager(rig.root)
    app.state.desktop_sessions = DesktopSessions(rig.root)
    app.include_router(approval_routes.router)
    app.include_router(operation_routes.router)
    headers = {"Authorization": f"Bearer {TOKEN}"} if bearer else {}
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://localhost", headers=headers
    ) as client:
        yield client


async def _scenario(
    rig: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: Any,
    *,
    family: str,
    case: str,
) -> tuple[int, Any, int]:
    """Run ONE scenario against ONE route family; ``(status, body, signer calls)``."""
    kind = "inheritance" if case == "inheritance_success" else "finite"
    record = (
        _inherit(rig, monkeypatch, capsys) if kind == "inheritance" else _finite(rig, monkeypatch)
    )
    _level(monkeypatch, PRESENCE if kind == "inheritance" else FILE_ONLY)
    counters = _counted(monkeypatch)
    aid = record["approval_id"]
    async with _client(rig, monkeypatch) as reader:
        row = await _row(reader, aid)
    body = _ack(row)
    verb = "apply" if case.startswith("apply") else "approve"
    target = aid
    if case == "unknown":
        target = "ap_0000000000000"
    elif case == "digest_mismatch":
        body = _ack(row, disclosure_digest=ZERO)
    elif case == "state_mismatch":
        A.deny(aid, root=rig.root)
    path = f"/v1/desktop/approvals/{target}/{verb}-{family}"
    async with _client(
        rig,
        monkeypatch,
        bearer=case != "no_bearer",
        plane_open=case != "plane_closed",
    ) as client:
        if case in {"apply_retry", "apply_replay"}:
            if case == "apply_retry":
                # "Signed, not applied": the first apply fails transiently (the existing
                # test_signed_not_applied_then_apply_retry pattern), so the retry is a 200.
                real_apply = finite_share.apply
                calls = {"n": 0}

                def flaky(*args: Any, **kw: Any) -> Any:
                    calls["n"] += 1
                    if calls["n"] == 1:
                        raise MeshRefusal("finite_write_unconfirmed", "synthetic transient")
                    return real_apply(*args, **kw)

                monkeypatch.setattr(finite_share, "apply", flaky)
            first = await client.post(f"/v1/desktop/approvals/{aid}/approve-{family}", json=body)
            assert first.status_code == 200, first.text
        response = await client.post(path, json=body)
    # Each family runs on its own fresh rig, so the per-rig identifiers (the card id and the
    # minted operator key) differ by construction; they are masked, and nothing else is.
    text = json.dumps(response.json(), sort_keys=True).replace(aid, "<approval_id>")
    key_id = str((A.load_record(aid, root=rig.root).get("signature") or {}).get("key_id") or "")
    if key_id:
        text = text.replace(key_id, "<key_id>")
    return response.status_code, json.loads(text), counters["signer"]


PARITY_CASES = {
    "finite_success": (200, 1),
    "inheritance_success": (200, 1),
    "apply_retry": (200, 1),
    "apply_replay": (409, 1),
    "apply_unsigned": (409, 0),
    "no_bearer": (401, 0),
    "plane_closed": (503, 0),
    "unknown": (404, 0),
    "state_mismatch": (409, 0),
    "digest_mismatch": (409, 0),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", sorted(PARITY_CASES))
async def test_credential_and_operation_routes_answer_identically(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: str,
) -> None:
    from tests.unit.network import test_credentials_consent

    answers = {}
    for family in ("credential", "operation"):
        # A FRESH isolated rig per family: the success cells consume their card.
        wrapped = test_credentials_consent.rig.__wrapped__  # type: ignore[attr-defined]
        fixture = contextmanager(wrapped)
        with (
            pytest.MonkeyPatch.context() as mp,
            fixture(tmp_path_factory.mktemp(f"{case}-{family}"), mp) as fresh,
        ):
            desktop_approval._ATTEMPTS.clear()
            desktop_approval._CANCELLED.clear()
            answers[family] = await _scenario(fresh, mp, capsys, family=family, case=case)
    status, signer = PARITY_CASES[case]
    assert answers["credential"][0] == status, answers["credential"]
    assert answers["credential"][2] == signer
    assert answers["operation"] == answers["credential"]


# ---------------------------------------------------------------------------
# ONE generic check, both generic approvers, before any signer
# ---------------------------------------------------------------------------


class _SyntheticTypedOp:
    """A registered typed op with NO frozen refusal and no CLI disclosure.

    The shape S4a-pair-op's ``pair_phone`` will take: it must be refused by BOTH generic
    approvers with the registry's own ``typed_approval_required``.
    """

    op = "synthetic_typed_op"
    kind = A.KIND_DEVICE_ONBOARD
    file_only_signing = False
    cli_approvable = False

    def intent(self, record: dict[str, Any]) -> dict[str, Any] | None:
        what = record.get("what") or {}
        return what.get(self.op) if isinstance(what.get(self.op), dict) else None

    def disclosure(self, record: dict[str, Any]) -> dict[str, Any]:
        return {"operation": self.op}

    def expected(self, record: Any) -> dict[str, Any]:
        return {}

    def apply(self, approval_id: str, expected: dict[str, Any], root: Any) -> Any:
        raise AssertionError("a generic approver must never reach apply")

    def generic_refusal(self, record: Any, approval_id: str) -> MeshRefusal | None:
        return approval_ops.typed_required(approval_id)


class _DeferringTypedOp(_SyntheticTypedOp):
    """A registered op OUTSIDE every family whose own refusal DEFERS (returns ``None``).

    The R1 / SC-1 shape: a future op (``pair_phone``) that cannot parse a malformed payload
    and so has no specific answer. Key presence alone must still refuse on BOTH generic
    approvers -- ``None`` defers to the fail-closed default, it never means "signable".
    """

    op = "probe_none_op"

    def generic_refusal(self, record: Any, approval_id: str) -> MeshRefusal | None:
        return None


def _synthetic(
    rig: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    op: Any = None,
    payload: Any = None,
) -> dict[str, Any]:
    from tests.unit.network.test_finite_share import _record_with_what

    op = op or _SyntheticTypedOp()
    template = _finite(rig, monkeypatch)
    # The registry is sealed; the stub enters through the private table and monkeypatch
    # removes it again, which is the test seam the seal docstring names.
    monkeypatch.setitem(approval_ops._OPS, op.op, op)
    what = {
        "anchor": copy.deepcopy(template["what"]["anchor"]),
        op.op: {"payload": "synthetic"} if payload is None else payload,
    }
    return _record_with_what(template, what, rig.root)


@pytest.mark.asyncio
async def test_bodiless_desktop_approve_refuses_a_registered_typed_op_before_the_signer(
    rig: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _synthetic(rig, monkeypatch)
    counters = _counted(monkeypatch)
    aid = record["approval_id"]
    async with _client(rig, monkeypatch) as client:
        response = await client.post(f"/v1/desktop/approvals/{aid}/approve")
        assert response.status_code == 409, response.text
        assert response.json()["detail"]["code"] == approval_ops.TYPED_APPROVAL_REQUIRED
        # Deny stays open: the safe direction is never gated by the registry.
        denied = await client.post(f"/v1/desktop/approvals/{aid}/deny")
        assert denied.status_code == 200, denied.text
    assert counters["signer"] == 0
    assert A.load_record(aid, root=rig.root)["state"] == "denied"


def test_cli_approve_refuses_a_registered_typed_op_before_the_signer(
    rig: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record = _synthetic(rig, monkeypatch)
    counters = _counted(monkeypatch)
    aid = record["approval_id"]
    capsys.readouterr()
    assert _lop_network("approvals", "approve", aid, "--json") != 0
    out = capsys.readouterr()
    assert json.loads(out.out)["code"] == approval_ops.TYPED_APPROVAL_REQUIRED
    assert counters["signer"] == 0
    assert A.load_record(aid, root=rig.root)["state"] == "requested"
    # Deny (and withdraw) stay open on the CLI too.
    assert _lop_network("approvals", "deny", aid, "--json") == 0
    assert A.load_record(aid, root=rig.root)["state"] == "denied"


def test_cli_approve_negative_control_signs_without_the_registry_check(
    rig: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """R5: the CLI twin of the bodiless negative control -- the check is what refuses."""
    record = _synthetic(rig, monkeypatch)
    counters = _counted(monkeypatch)
    aid = record["approval_id"]
    monkeypatch.setattr(approval_ops, "require_generic_approvable", lambda *a, **k: None)
    capsys.readouterr()
    assert _lop_network("approvals", "approve", aid, "--json") == 0
    assert counters["signer"] == 1
    assert A.load_record(aid, root=rig.root)["state"] == "approved"


MALFORMED_PAYLOADS = ["not-a-dict", 7, [], {}]


@pytest.mark.parametrize("payload", MALFORMED_PAYLOADS)
@pytest.mark.asyncio
async def test_bodiless_approve_fails_closed_for_a_deferring_op_with_a_malformed_payload(
    rig: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    payload: Any,
) -> None:
    """R1 / SC-1 (desktop): key presence refuses even when the op's own refusal is ``None``."""
    record = _synthetic(rig, monkeypatch, op=_DeferringTypedOp(), payload=payload)
    _level(monkeypatch, PRESENCE)
    counters = _counted(monkeypatch)
    aid = record["approval_id"]
    async with _client(rig, monkeypatch) as client:
        response = await client.post(f"/v1/desktop/approvals/{aid}/approve")
    assert counters["signer"] == 0
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == approval_ops.TYPED_APPROVAL_REQUIRED
    assert A.load_record(aid, root=rig.root)["state"] == "requested"


@pytest.mark.parametrize("payload", MALFORMED_PAYLOADS)
def test_cli_approve_fails_closed_for_a_deferring_op_with_a_malformed_payload(
    rig: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    payload: Any,
) -> None:
    """R1 / SC-1 (CLI): the same record refuses on ``lop network approvals approve``."""
    record = _synthetic(rig, monkeypatch, op=_DeferringTypedOp(), payload=payload)
    counters = _counted(monkeypatch)
    aid = record["approval_id"]
    capsys.readouterr()
    rc = _lop_network("approvals", "approve", aid, "--json")
    assert counters["signer"] == 0
    assert rc != 0
    assert json.loads(capsys.readouterr().out)["code"] == approval_ops.TYPED_APPROVAL_REQUIRED
    assert A.load_record(aid, root=rig.root)["state"] == "requested"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "code"),
    [
        ("finite", "finite_terminal_approval_required"),
        ("inheritance", "inherit_terminal_approval_required"),
    ],
)
async def test_bodiless_approve_answers_credential_cards_through_the_registry(
    rig: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    kind: str,
    code: str,
) -> None:
    """The frozen per-op refusals now come from the registry, byte for byte.

    The NEGATIVE CONTROL is the second half: with the one registry check stubbed out, the
    same bodiless POST reaches the signer and the card is approved -- which is exactly the
    blind signature the check exists to prevent, and proves the check is what refuses.
    """
    record = _finite(rig, monkeypatch) if kind == "finite" else _inherit(rig, monkeypatch, capsys)
    _level(monkeypatch, PRESENCE)
    counters = _counted(monkeypatch)
    aid = record["approval_id"]
    async with _client(rig, monkeypatch) as client:
        response = await client.post(f"/v1/desktop/approvals/{aid}/approve")
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == code
    assert f"lop network approvals approve {aid}. Nothing was approved." in detail["text"]
    assert counters["signer"] == 0

    monkeypatch.setattr(approval_ops, "require_generic_approvable", lambda *a, **k: None)
    async with _client(rig, monkeypatch) as client:
        blind = await client.post(f"/v1/desktop/approvals/{aid}/approve")
    assert blind.status_code == 200, blind.text
    assert counters["signer"] == 1
    assert A.load_record(aid, root=rig.root)["state"] == "approved"


# ---------------------------------------------------------------------------
# Q3 (QA round 1, folded in agent review round 2): the CLI approver runs the credential
# family scan too, so it never signs a ``credential_*`` key it cannot disclose.
# ---------------------------------------------------------------------------


def _cli_credential_record(
    rig: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    shape: str,
) -> dict[str, Any]:
    from tests.unit.network.test_finite_share import _record_with_what

    template = _finite(rig, monkeypatch)
    anchor = copy.deepcopy(template["what"]["anchor"])
    if shape == "bogus-alone":
        what = {"anchor": anchor, "credential_bogus": {"payload": "unknown"}}
    elif shape == "bogus-beside-finite":
        what = copy.deepcopy(template["what"])
        what["credential_bogus"] = {"payload": "unknown"}
    elif shape == "malformed-finite":  # the finite op key is present, its payload does not parse
        what = {"anchor": anchor, finite_share.OP: "not-a-dict"}
    else:  # "malformed-inheritance": the op key is present, its payload does not parse
        what = {"anchor": anchor, inheritance.OP: "not-a-dict"}
    return _record_with_what(template, what, rig.root)


#: QA round 3, Q4: every credential-bearing record shape the CLI must refuse, including the
#: two that carry a finite share -- a bogus key beside a valid one, and a malformed one --
#: where the CLI used to answer with the family code while the desktop gave the finite op's.
CLI_REFUSED_SHAPES = [
    "bogus-alone",
    "bogus-beside-finite",
    "malformed-inheritance",
    "malformed-finite",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", CLI_REFUSED_SHAPES)
async def test_cli_and_desktop_refuse_each_undisclosable_credential_shape_identically(
    rig: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    shape: str,
) -> None:
    """Q3 + Q4: both generic approvers give the SAME code and sentence, signer untouched.

    The desktop's bodiless ``/approve`` is the reference: its answer is the op's own frozen
    refusal when one applies, else the family refusal. The CLI must not drift from it (QA
    round 3 caught it telling a malformed finite share to use "a typed approval flow" where
    the desktop says to request a new finite-share recovery).
    """
    record = _cli_credential_record(rig, monkeypatch, shape)
    _level(monkeypatch, PRESENCE)
    counters = _counted(monkeypatch)
    aid = record["approval_id"]
    async with _client(rig, monkeypatch) as client:
        response = await client.post(f"/v1/desktop/approvals/{aid}/approve")
    assert response.status_code == 409, response.text
    desktop_body = response.json()["detail"]
    capsys.readouterr()
    rc = _lop_network("approvals", "approve", aid, "--json")
    out = capsys.readouterr()
    assert rc != 0
    cli_body = json.loads(out.out)
    assert cli_body["code"] == desktop_body["code"], (cli_body, desktop_body)
    assert desktop_body["text"] in json.dumps(cli_body), (cli_body, desktop_body)
    assert counters["signer"] == 0
    assert A.load_record(aid, root=rig.root)["state"] == "requested"


def test_cli_approve_still_signs_a_valid_finite_share_after_disclosing_it(
    rig: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Q3 CONTROL: the CLI's own ``cli_approvable`` op signs exactly as before the fold."""
    record = _finite(rig, monkeypatch)
    counters = _counted(monkeypatch)
    aid = record["approval_id"]
    capsys.readouterr()
    rc = _lop_network("approvals", "approve", aid, "--json")
    out = capsys.readouterr()
    assert rc == 0, out
    assert counters["signer"] == 1
    assert json.loads(out.out)["state"] == "approved"
    # The full disclosure precedes the signature on stderr, as on base.
    assert finite_share.details(record)[0].split()[0] in out.err
    assert A.load_record(aid, root=rig.root)["state"] == "approved"
