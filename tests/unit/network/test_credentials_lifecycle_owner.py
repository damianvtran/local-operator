"""Owner-side ``lifecycle_read`` / ``lifecycle_ack`` / unknown-kind dispatch.

Runs the REAL ``MeshCredentialBroker.on_broker`` over the synthetic isolated rig the
consent suite already builds (tmp_path identities, synthetic SQLite auth rows, no
network, no provider endpoint). Each test pins one rule of
``docs/design/mesh-credential-lifecycle.md``: a recipient reads only its own
snapshot, a sticky deny answers ``withdrawn`` BEFORE any counter is read, and an
unknown ``lifecycle_*`` kind is a by-name ``unsupported``, never an ``internal`` fault.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network.credentials import consent
from local_operator.network.credentials import lifecycle as L
from tests.unit.network.test_credentials_consent import N, P
from tests.unit.network.test_credentials_consent import rig as consent_rig

SCOPE = "acct-one"

# The consent suite's synthetic owner/recipient rig, re-exported under the fixture
# name pytest resolves; an assignment (not a shadowing import) keeps flake8 quiet.
rig = consent_rig


def _seed(rig: Any, scope: str = SCOPE, key: str = P) -> L.CounterRow:
    return L.LifecycleStore(N, rig.root).record_native(
        key, scope, L.NativeFacts(present=True, row_updated_at=1.0)
    )


def _deny(rig: Any) -> None:
    consent.restrict_and_revoke(
        N,
        rig.own.device_id,
        rig.peer.device_id,
        {"credential_key": P},
        "synthetic-withdrawal",
        rig.root,
    )


def _call(rig: Any, kind: str, entries: list[dict[str, Any]], *, device: str = "", **extra: Any):
    device = device or rig.peer.device_id
    link = SimpleNamespace(device_id=device, network_id=N, epoch=1)
    frame = {"op": "net_broker", "kind": kind, "req": 1, "from_device": device, **extra}
    if entries is not None:
        frame["entries"] = entries
    return rig.broker.on_broker(link, frame)["detail"]


def _read(rig: Any, key: str = P, scope: str = SCOPE, **kw: Any) -> dict[str, Any]:
    return _call(rig, "lifecycle_read", [{"credential_key": key, "account_scope_ref": scope}], **kw)


def test_read_returns_own_snapshot(rig):
    _seed(rig)
    detail = _read(rig)
    assert detail["kind"] == "lifecycle_snapshot"
    assert detail["recipient_device"] == rig.peer.device_id
    assert detail["entries"] == [
        {
            "credential_key": P,
            "account_scope_ref": SCOPE,
            "state": "available",
            "material_generation": 1,
            "login_chain_revision": 1,
        }
    ]
    assert "access" not in repr(detail) and "synthetic-private-marker" not in repr(detail)


def test_read_refuses_another_recipients_snapshot(rig):
    _seed(rig)
    detail = _read(rig, recipient_device="d_" + "9" * 32)
    assert detail["kind"] == "error" and detail["code"] == "identity_mismatch"
    assert "entries" not in detail


def test_read_by_a_non_holder_is_absent_not_a_leak(rig):
    _seed(rig)
    stranger = "d_" + "8" * 32
    detail = _read(rig, device=stranger)
    # Same answer as a key that does not exist: coverage is not disclosed.
    assert detail["entries"] == [
        {"credential_key": P, "account_scope_ref": SCOPE, "state": "absent"}
    ]


def test_denied_key_is_withdrawn_before_any_state_is_read(rig, monkeypatch):
    _seed(rig)
    _deny(rig)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("lifecycle state was read for a denied key")

    monkeypatch.setattr(L.LifecycleStore, "snapshot", boom)
    monkeypatch.setattr(L.LifecycleStore, "scopes_for", boom)
    monkeypatch.setattr(type(rig.broker), "_lifecycle_candidates", boom)
    detail = _read(rig)
    assert detail["entries"] == [
        {"credential_key": P, "account_scope_ref": SCOPE, "state": "withdrawn"}
    ]


def test_unknown_required_scope_is_scope_unknown_never_a_grant(rig):
    _seed(rig, scope="acct-other")
    detail = _read(rig, scope=SCOPE)
    assert detail["entries"] == [
        {"credential_key": P, "account_scope_ref": SCOPE, "state": "scope_unknown"}
    ]


def test_no_row_is_absent(rig):
    assert _read(rig)["entries"][0]["state"] == "absent"


def test_device_bound_provider_is_non_portable(rig):
    detail = _read(rig, key="kimi")
    assert detail["entries"][0]["state"] == "non_portable"


def test_read_of_a_key_this_device_does_not_own_is_not_owner(rig):
    detail = _read(rig, key="not-a-key")
    assert detail["kind"] == "error" and detail["code"] == "not_owner"


def test_malformed_read_is_refused_before_any_state(rig):
    detail = _call(rig, "lifecycle_read", [{"credential_key": P}])
    assert detail["kind"] == "error" and detail["code"] == "malformed_frame"


def test_ack_is_accepted_and_withdrawn_recipient_is_refused(rig):
    entry = {
        "credential_key": P,
        "account_scope_ref": SCOPE,
        "material_generation": 1,
        "login_chain_revision": 1,
    }
    assert _call(rig, "lifecycle_ack", [entry]) == {"kind": "ack", "entries": 1}
    _deny(rig)
    detail = _call(rig, "lifecycle_ack", [entry])
    assert detail["kind"] == "error" and detail["code"] == "not_a_holder"


@pytest.mark.parametrize("kind", ["lifecycle_future", "lifecycle_"])
def test_unserved_lifecycle_kind_is_unsupported_never_internal(rig, kind):
    detail = _call(rig, kind, None)
    assert detail["code"] == "unsupported"
    assert detail["code"] != "internal"
