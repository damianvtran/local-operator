"""Recipient half of ``lifecycle_announce`` and the evidence it may record.

Synthetic ``tmp_path`` rig shared with the consent suite. The headline rule: a
lifecycle observation is METADATA, so no announcement of any state can ever make a
credential ``usable`` — only ``note_resolution`` given a real ``Grant`` does.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network.credentials import effective
from local_operator.network.credentials import lifecycle as L
from local_operator.network.credentials.types import BrokerError
from tests.unit.network.test_credentials_consent import N, P
from tests.unit.network.test_credentials_consent import rig as consent_rig

# The consent suite's rig under the name pytest resolves (assignment, not a shadowing
# import, so flake8 stays quiet).
rig = consent_rig
SCOPE = "acct-one"


@pytest.fixture(autouse=True)
def _clean_receipts():
    effective._receipts.clear()
    yield
    effective._receipts.clear()


def _server(rig: Any) -> Any:
    # The RECIPIENT's relay: root + identity are all member_lifecycle_announce reads.
    return SimpleNamespace(root=rig.root, identity=SimpleNamespace(device_id=rig.peer.device_id))


def _announce(rig: Any, *, state="available", chain=1, material=1, sender="", **over: Any):
    sender = sender or rig.own.device_id
    link = SimpleNamespace(device_id=sender, network_id=N, epoch=1)
    frame = {
        "op": "net_broker",
        "kind": "lifecycle_announce",
        "req": 1,
        "from_device": sender,
        "owner_device": sender,
        "recipient_device": rig.peer.device_id,
        "entries": [
            {
                "credential_key": P,
                "account_scope_ref": SCOPE,
                "state": state,
                "material_generation": material,
                "login_chain_revision": chain,
            }
        ],
    }
    frame.update(over)
    return L.member_lifecycle_announce(_server(rig), link, frame)


def _hw(rig: Any):
    return L.LifecycleStore(N, rig.root).load().high_water


def test_owner_announcement_is_applied_and_acked(rig):
    detail = _announce(rig)
    assert detail["kind"] == "ack"
    assert detail["entries"][0]["applied"] is True
    assert _hw(rig) == {(rig.own.device_id, P, SCOPE): (1, 1)}


def test_replay_and_stale_do_not_move_the_high_water(rig):
    assert _announce(rig, chain=2, material=3)["entries"][0]["applied"] is True
    replay = _announce(rig, chain=2, material=3)["entries"][0]
    stale = _announce(rig, chain=1, material=1)["entries"][0]
    assert (replay["applied"], replay["reason"]) == (False, L.OBSERVED_REPLAY)
    assert (stale["applied"], stale["reason"]) == (False, L.OBSERVED_STALE)
    assert _hw(rig) == {(rig.own.device_id, P, SCOPE): (2, 3)}


def test_withdrawn_is_reapplied_at_an_unchanged_counter(rig):
    # A sticky deny moves neither counter; forward-only alone would drop it.
    _announce(rig, chain=1, material=1)
    detail = _announce(rig, state="withdrawn", chain=1, material=1)
    assert detail["entries"][0]["applied"] is True
    assert effective._receipts[(effective._root(rig.root), P)].state == "withdrawn"


def test_sender_that_is_not_the_placement_owner_changes_nothing(rig):
    # The recipient itself is a holder, not the owner of P: an announce from it is refused.
    detail = _announce(rig, sender=rig.peer.device_id, recipient_device=rig.peer.device_id)
    assert detail["code"] == "not_owner"
    assert _hw(rig) == {} and effective._receipts == {}


def test_frame_addressed_to_another_device_changes_nothing(rig):
    detail = _announce(rig, recipient_device="d_" + "9" * 32)
    assert detail["code"] == "identity_mismatch"
    assert _hw(rig) == {}


def test_unknown_key_changes_nothing(rig):
    link = SimpleNamespace(device_id=rig.own.device_id, network_id=N, epoch=1)
    frame = {
        "kind": "lifecycle_announce",
        "owner_device": rig.own.device_id,
        "recipient_device": rig.peer.device_id,
        "entries": [
            {
                "credential_key": "not-shared",
                "account_scope_ref": SCOPE,
                "state": "available",
                "material_generation": 1,
                "login_chain_revision": 1,
            }
        ],
    }
    assert L.member_lifecycle_announce(_server(rig), link, frame)["code"] == "unknown_key"
    assert _hw(rig) == {}


@pytest.mark.parametrize(
    "bad",
    [
        {"entries": []},
        {"note": "x" * 17000},
        {"access_token": "synthetic"},
    ],
)
def test_malformed_announcement_changes_nothing(rig, bad):
    detail = _announce(rig, **bad)
    assert detail["code"] == "malformed_announce" or detail["kind"] == "error"
    assert _hw(rig) == {} and effective._receipts == {}


def test_scope_unknown_is_not_a_state_an_owner_pushes(rig):
    assert _announce(rig, state="scope_unknown")["code"] == "malformed_announce"


@pytest.mark.parametrize(
    "state", ["available", "absent", "withdrawn", "unsupported", "non_portable", "bogus", ""]
)
def test_no_lifecycle_observation_can_ever_be_usable(rig, state):
    receipt = effective.note_lifecycle_observation(rig.root, P, state, owner=rig.own.device_id)
    assert receipt is None or (receipt.state != "effective_borrow" and not receipt.usable)
    # And through the whole recipient path, whatever state arrives.
    _announce(rig, state="available" if state == "bogus" else state or "available")
    for stored in effective._receipts.values():
        assert not stored.usable


def test_observation_after_a_refused_resolve_stays_pending(rig):
    # The resolution path (a real Grant) is the consent suite's; here: an observation
    # layered on a refused resolve never turns it usable.
    effective.note_resolution(
        rig.root, P, BrokerError(code="owner_offline", owner_device=rig.own.device_id)
    )
    receipt = effective.note_lifecycle_observation(
        rig.root, P, "available", owner=rig.own.device_id
    )
    assert receipt is not None and receipt.state == "pending" and not receipt.usable
