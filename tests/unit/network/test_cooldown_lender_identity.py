"""#3066 m4: a row number belongs to the device that actually lent the bearer.

The placement owner may move between a serve and its quota report. Recording the
old row against the new device incorrectly narrows that device's unrelated row
with the same integer id. These tests do not move authority or change reports;
the real-link cell changes only the session client's observed placement view,
while its relay still routes the report to the original owner.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network.credentials import client as client_mod
from local_operator.network.credentials import state as state_mod
from local_operator.network.credentials.client import MeshCredentialClient
from local_operator.network.credentials.state import PlacementState
from local_operator.network.credentials.types import CredentialRef, Grant
from tests.unit.network.test_credentials_real_link import (  # noqa: F401 — fixtures by import
    STUB_PROVIDER,
    _audit,
    _borrower_client,
    _pull,
    _share,
    devices,
    idp,
    mesh,
    stub_provider,
)

_FIXTURES = (devices, idp, mesh, stub_provider)
OLD_OWNER = "d_00000000000000000000000000000001"
NEW_OWNER = "d_00000000000000000000000000000002"
BORROWER = "d_00000000000000000000000000000003"
NETWORK = "n_00000000000000000000000000000004"
NOW = 2_000_000_000.0
ROW = 17
MODEL = "claude-opus-5-5"


def _grant(*, row: int = ROW, owner: str = OLD_OWNER) -> Grant:
    return Grant(
        access_token="synthetic-cooldown-bearer",
        kind="bearer",
        token_expires_at_ms=int(NOW * 1000) + 120_000,
        grant_expires_at_ms=int(NOW * 1000) + 120_000,
        credential_ref=CredentialRef(owner, "old-owner", STUB_PROVIDER, "oauth", row),
        served_by=owner,
        identity={"email": "original@example.test"},
    )


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MeshCredentialClient:
    # Freeze only these modules' clock objects, not the global time module used
    # by relay threads or the test runner's deadline/cleanup machinery.
    clock = SimpleNamespace(time=lambda: NOW, monotonic=time.monotonic)
    monkeypatch.setattr(client_mod, "time", clock)
    monkeypatch.setattr(state_mod, "time", clock)
    instance = MeshCredentialClient(root=tmp_path, self_device=BORROWER, network_id=NETWORK)
    monkeypatch.setattr(instance, "owner_of", lambda _key: NEW_OWNER)
    monkeypatch.setattr(instance, "should_borrow", lambda _key: True)
    return instance


def _report(client: MeshCredentialClient, credential_id: int = ROW) -> None:
    assert (
        client.report_sync(
            STUB_PROVIDER,
            kind="quota",
            session_id="affected-session",
            model_id=MODEL,
            retry_after_ms=86_400_000,
            credential_id=credential_id,
        )
        is None
    )


@pytest.mark.parametrize("slot", ["affected-session", ""])
def test_quota_after_owner_moves_rests_only_the_actual_lender(
    client: MeshCredentialClient, slot: str
) -> None:
    client.grants.put(STUB_PROVIDER, slot, _grant())
    _report(client)
    old = client.state.cooldowns(STUB_PROVIDER, MODEL, owner_device=OLD_OWNER)
    new = client.state.cooldowns(STUB_PROVIDER, MODEL, owner_device=NEW_OWNER)
    assert new == [], "the old owner's numeric id must never hide the new owner's row"
    assert len(old) == 1 and old[0]["credential_id"] == ROW
    assert old[0]["label"] == "original@example.test"
    assert old[0]["until"] == NOW + 3600
    # Persistence, not merely the session's in-memory view, carries the lender.
    reloaded = PlacementState.load(NETWORK, client.root)
    assert reloaded.cooldowns(STUB_PROVIDER, MODEL, owner_device=OLD_OWNER) == old
    assert reloaded.cooldowns(STUB_PROVIDER, MODEL, owner_device=NEW_OWNER) == []
    assert client.grants.peek(STUB_PROVIDER, slot) is None


@pytest.mark.parametrize("shape", ["absent", "unattributed", "different-row", "missing-owner"])
def test_unknown_or_mismatched_provenance_never_guesses_the_current_owner(
    client: MeshCredentialClient, caplog: pytest.LogCaptureFixture, shape: str
) -> None:
    if shape == "different-row":
        client.grants.put(STUB_PROVIDER, "affected-session", _grant(row=ROW + 1))
    elif shape == "missing-owner":
        client.grants.put(STUB_PROVIDER, "affected-session", _grant(owner=""))
    with caplog.at_level(logging.DEBUG, logger=client_mod.__name__):
        _report(client, credential_id=0 if shape == "unattributed" else ROW)
    assert client.state.cooldowns(STUB_PROVIDER, MODEL, owner_device=NEW_OWNER) == []
    assert client.state.cooldowns(STUB_PROVIDER, MODEL, owner_device=OLD_OWNER) == []
    notices = [r for r in caplog.records if "cooldown provenance" in r.message]
    assert len(notices) == 1
    assert "no matching lending grant" in notices[0].message
    assert "synthetic-cooldown-bearer" not in notices[0].message


def test_a_mismatched_session_slot_can_use_the_matching_device_slot(
    client: MeshCredentialClient,
) -> None:
    client.grants.put(STUB_PROVIDER, "affected-session", _grant(row=ROW + 1))
    client.grants.put(STUB_PROVIDER, "", _grant())
    _report(client)
    old = client.state.cooldowns(STUB_PROVIDER, MODEL, owner_device=OLD_OWNER)
    assert len(old) == 1 and old[0]["credential_id"] == ROW
    assert client.state.cooldowns(STUB_PROVIDER, MODEL, owner_device=NEW_OWNER) == []


def test_same_owner_control_preserves_the_entire_serialized_rest(
    client: MeshCredentialClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(client, "owner_of", lambda _key: OLD_OWNER)
    client.grants.put(STUB_PROVIDER, "affected-session", _grant())
    _report(client)
    rows = client.state.cooldowns(STUB_PROVIDER, MODEL, owner_device=OLD_OWNER)
    assert rows == [
        {
            "key": STUB_PROVIDER,
            "owner_device": OLD_OWNER,
            "credential_id": ROW,
            "block_scope": "model:opus",
            "until": NOW + 3600,
            "label": "original@example.test",
            "rested_at": NOW,
        }
    ]
    print("SAME_OWNER_BYTES", json.dumps(client.state.to_json(), sort_keys=True))


def test_a_zero_row_still_records_nothing(client: MeshCredentialClient) -> None:
    client._cool_lent_row(
        STUB_PROVIDER,
        session_id="affected-session",
        credential_id=0,
        model_id=MODEL,
        retry_after_ms=86_400_000,
        block_scope="",
    )
    assert not client.state._cooldowns


def test_real_borrow_and_report_use_the_lender_when_the_session_view_moves(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    pair: Any = request.getfixturevalue("mesh")
    _share(pair)
    _pull(pair)
    borrower = _borrower_client(pair)
    lent = borrower.request_grant_sync(
        STUB_PROVIDER, provider=STUB_PROVIDER, session_id="real-quota", model_id=MODEL
    )
    assert isinstance(lent, Grant)
    assert lent.credential_ref.owner_device == pair.owner
    before = len(_audit(pair.a, "credential.report"))
    # The session has noticed a new placement; its separate relay still sees A.
    # This isolates the m4 write race without inventing a successful new-owner
    # grant or claiming this two-relay fixture certifies a three-device move.
    monkeypatch.setattr(borrower, "owner_of", lambda _key: NEW_OWNER)
    assert (
        borrower.report_sync(
            STUB_PROVIDER,
            kind="quota",
            session_id="real-quota",
            model_id=MODEL,
            retry_after_ms=86_400_000,
        )
        is None
    )
    reports = _audit(pair.a, "credential.report")
    old = borrower.state.cooldowns(STUB_PROVIDER, MODEL, owner_device=pair.owner)
    assert len(reports) == before + 1, "the original owner must actually receive the report"
    assert borrower.state.cooldowns(STUB_PROVIDER, MODEL, owner_device=NEW_OWNER) == []
    assert len(old) == 1 and old[0]["credential_id"] == lent.credential_ref.credential_id
    print(
        "REAL_LINK_RESULT",
        json.dumps(
            {
                "grant_received": True,
                "report_audit_delta": len(reports) - before,
                "lender_rests": len(old),
                "new_owner_rests": 0,
                "refresh_post_count": len(pair.idp.posts),
            },
            sort_keys=True,
        ),
    )
