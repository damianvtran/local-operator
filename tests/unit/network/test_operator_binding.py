"""The operator binding: a node forwards node-started asks/notices only to the
device the operator's OWN onboarding signature names (``network/operator_binding``).

Real file-only operator keys, real ``onboard_approvals.request_device`` filing and
real ``approvals.approve`` signatures; the only seam is the node's installed
anchor (``trust.load_anchor``, patched to return a ROOT-OWNED load of the given
anchor, as the onboarding runner's privileged install would leave it). Every
negative row starts from one valid binding and changes one cause, and asserts
both the verdict and that a node-started row reaches nobody.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import approvals as A
from local_operator.network import enrollment
from local_operator.network import operator_binding as ob
from local_operator.network import store as net_store
from local_operator.network import wire
from local_operator.network.identity import load_or_mint
from local_operator.network.types import MemberRecord, NetworkRecord
from local_operator.notifications import remote_attention as ra
from local_operator.notifications.store import NotificationStore
from local_operator.operator import trust
from tests.unit.network.test_approvals_store import _make_key, _sign_decision

NET = "n_test"
NODE = "d_node"
CO = "d_comember"
SID = "s_node_started"
CAPS = frozenset({wire.OPERATOR_BINDING_V1, wire.REMOTE_ATTENTION_V1})


@dataclasses.dataclass
class Rig:
    mac: Path
    node: Path
    mac_id: str
    anchor: Any
    monkeypatch: pytest.MonkeyPatch

    def install_anchor(self, anchor: Any, *, root_owned: bool = True) -> None:
        """What the onboarding runner's privileged step leaves on the node."""

        def load(uid: Any = None) -> Any:
            return trust.AnchorLoad(
                anchor=anchor,
                path=trust.anchor_path(uid),
                root_owned=root_owned,
                reason="test-installed",
                exists=True,
            )

        self.monkeypatch.setattr(trust, "load_anchor", load)
        ob._reset_caches()

    def link(self, device_id: str, network_id: str = NET, caps: frozenset[str] = CAPS) -> Any:
        return SimpleNamespace(device_id=device_id, network_id=network_id, capabilities=caps)


def _network(root: Path, self_id: str, members: tuple[str, ...], network_id: str = NET) -> None:
    record = NetworkRecord(network_id=network_id, name="mesh", self_device_id=self_id)
    record.members = [MemberRecord(device_id=d, name=f"{d}-box") for d in members]
    net_store.save(record, root)


def _onboard(
    rig: Rig, *, network: str = NET, host: str = "node.test", at: float = 0.0
) -> dict[str, Any]:
    """The operator files and SIGNS one onboarding card on the Mac (real path)."""
    from local_operator.network.onboard_approvals import request_device

    record = request_device(
        host=host,
        user="ec2-user",
        host_key_fp="SHA256:fixture",
        network=network,
        credential_ref="ssh-node-ref",
        root=rig.mac,
    )
    decided_at = at or time.time()
    sig = _sign_decision(rig.mac, record, "approve", decided_at)
    return A.approve(record["approval_id"], signature_hex=sig, decided_at=decided_at, root=rig.mac)


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    mac = tmp_path / "mac"
    node = tmp_path / "node"
    mac.mkdir()
    node.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(mac))
    anchor = _make_key(mac)
    mac_id = load_or_mint(mac).device_id
    _network(mac, mac_id, (mac_id, NODE, CO))
    _network(node, NODE, (NODE, mac_id, CO))
    r = Rig(mac=mac, node=node, mac_id=mac_id, anchor=anchor, monkeypatch=monkeypatch)
    r.install_anchor(anchor)
    return r


def _bind(rig: Rig) -> dict[str, Any]:
    _onboard(rig)
    proofs = ob.own_proofs(rig.mac, NET, for_device=NODE)
    assert len(proofs) == 1
    reply = ob.accept_presented(rig.node, rig.link(rig.mac_id), {"proofs": proofs})
    assert reply == {"accepted": True, "reason": ""}
    return proofs[0]


def _node_started_session(node: Path, *, origin: dict[str, Any] | None = None) -> None:
    """A session started ON the node with one open ask and one notify."""
    from local_operator.asks import store as ask_store
    from local_operator.session.placement import (
        MeshStamp,
        SessionPlacement,
        write_stamp,
    )

    (node / "sessions" / SID).mkdir(parents=True, exist_ok=True)
    if origin is not None:
        write_stamp(
            node,
            MeshStamp(
                session_id=SID,
                network_id="",
                home_device=NODE,
                placement=SessionPlacement(
                    mode="local", network_id="", home_device=NODE, stamp_revision=1
                ),
                origin=origin,
            ),
        )
    now = ask_store.now_ms()
    ask_store.append_event(
        ask_store.session_dir(node, SID),
        {
            "v": ask_store.EVENT_SCHEMA,
            "kind": ask_store.EVENT_QUEUED,
            "at": now,
            "ask_id": "ask_site_card",
            "expires_at": now + 3_600_000,
            "questions": [{"id": "q0", "question": "Allow example.com?", "options": []}],
        },
    )
    NotificationStore(node).notify(SID, title="Build done", priority="high", key="build")


def _listing(rig: Rig, viewer: str, *, origin: dict[str, Any] | None = None) -> dict[str, Any]:
    """The node's catalogue row for ``viewer``, exactly as ``_rows_for_viewer`` stamps it."""
    row = {"session_id": SID, "state": "stored", "origin": origin}
    link = rig.link(viewer)
    ra.stamp_rows_for_viewer(
        rig.node,
        [row],
        viewer,
        operator_viewer=ob.is_operator_link(rig.node, link),
        self_device_id=NODE,
    )
    return row


def _reaches_nobody(rig: Rig, viewer: str) -> None:
    _node_started_session(rig.node)
    row = _listing(rig, viewer)
    assert "notices" not in row and "attention" not in row


# ---------------------------------------------------------------------------
# Positive rows
# ---------------------------------------------------------------------------


def test_an_ask_a_site_card_and_a_notify_reach_the_bound_operator_only(rig: Rig) -> None:
    _bind(rig)
    _node_started_session(rig.node)
    assert ob.is_operator_link(rig.node, rig.link(rig.mac_id))
    row = _listing(rig, rig.mac_id)
    keys = [n["key"] for n in row["notices"]]
    assert any(k.startswith("remote-ask:") for k in keys), keys
    assert "build" in keys
    # Notices only: a node-started session's completion is not paged.
    assert "attention" not in row and "attention_unknown" not in row
    # The co-member sees the row without either field.
    co_row = _listing(rig, CO)
    assert "notices" not in co_row and "attention" not in co_row
    # The notify result says FORWARDED, to the operator's device, by name.
    assert ra.notice_forwarding(rig.node, SID) == (ra.FORWARDED, rig.mac_id)
    assert ra.forwarded_notice_sentence(rig.node, SID) == (
        f" Posted on {NODE}-box; it will be forwarded to {rig.mac_id}-box, the operator's device."
    )


def test_the_network_only_kind_is_a_proof_too(rig: Rig) -> None:
    from local_operator.network.onboard_approvals import request_device

    record = request_device(
        host="node.test",
        user="ec2-user",
        host_key_fp="SHA256:fixture",
        network=NET,
        credential_ref="ssh-node-ref",
        network_only=True,
        root=rig.mac,
    )
    A.approve_network_enrollment(
        record["approval_id"],
        authority_version=1,
        scope="network_only",
        digest=record["request_digest"],
        root=rig.mac,
    )
    # FIELD PATH (manager question): the on-disk enrollment file wraps the record,
    # so ``jq`` reads ``.record.what.network_id``; the SIGNED preimage carries it
    # flat at ``what.network_id``, which is the path the binding reads.
    raw = json.loads(A.record_path(record["approval_id"], rig.mac).read_text())
    assert raw["record"]["what"]["network_id"] == NET
    proofs = ob.own_proofs(rig.mac, NET)
    assert proofs and proofs[0]["kind"] == enrollment.KIND
    assert proofs[0]["request"]["what"]["network_id"] == NET
    assert ob.verify_proof(rig.node, proofs[0], network_id=NET, device_id=rig.mac_id) == (True, "")


def test_the_proof_for_that_node_is_preferred_over_another_nodes(rig: Rig) -> None:
    """M3: the decision that onboarded THIS node first; another node's only as fallback."""
    from local_operator.network.onboard_approvals import request_device

    other = request_device(
        host="other.test",
        user="ec2-user",
        host_key_fp="SHA256:fixture",
        device_id="d_other",
        network=NET,
        root=rig.mac,
    )
    t = time.time()
    A.approve(
        other["approval_id"],
        signature_hex=_sign_decision(rig.mac, other, "approve", t + 5),
        decided_at=t + 5,
        root=rig.mac,
    )
    mine = request_device(
        host="node.test",
        user="ec2-user",
        host_key_fp="SHA256:fixture",
        device_id=NODE,
        network=NET,
        root=rig.mac,
    )
    A.approve(
        mine["approval_id"],
        signature_hex=_sign_decision(rig.mac, mine, "approve", t),
        decided_at=t,
        root=rig.mac,
    )
    assert ob.own_proofs(rig.mac, NET, for_device=NODE)[0]["approval_id"] == mine["approval_id"]
    assert ob.own_proofs(rig.mac, NET, for_device="d_x")[0]["approval_id"] == other["approval_id"]


def test_the_binding_proof_in_use_survives_the_30_day_prune(rig: Rig) -> None:
    """M1: the newest signed onboarding per network is exempt; an older one is pruned."""
    old = _onboard(rig, host="old.test", at=time.time() - 1000)
    new = _onboard(rig, host="new.test")
    for record in (old, new):
        path = A.record_path(record["approval_id"], rig.mac)
        raw = json.loads(path.read_text())
        raw["state"] = A.STATE_CONNECTED  # terminal
        raw["decided_at"] = time.time() - A.TERMINAL_PRUNE_AGE_S - (100 if record is new else 200)
        path.write_text(json.dumps(raw))
    A.sweep(root=rig.mac)
    assert not A.record_path(old["approval_id"], rig.mac).exists()
    assert A.record_path(new["approval_id"], rig.mac).exists()
    assert [p["approval_id"] for p in ob.own_proofs(rig.mac, NET)] == [new["approval_id"]]


# ---------------------------------------------------------------------------
# Negative rows: each yields no forwarding and the honest node-only sentence
# ---------------------------------------------------------------------------


def _node_only(rig: Rig, viewer: str) -> None:
    _reaches_nobody(rig, viewer)
    ra.record_forwarding_origin(rig.node, viewer)
    assert ra.notice_forwarding(rig.node, SID) == (ra.LOCAL_ONLY, "")
    assert "only" in ra.forwarded_notice_sentence(rig.node, SID)


def test_a_forged_admin_role_binds_nothing(rig: Rig) -> None:
    record = net_store.load(NET, rig.node)
    for member in record.members:
        if member.device_id == CO:
            member.role = "admin"
    net_store.save(record, rig.node)
    assert not ob.is_operator_link(rig.node, rig.link(CO))
    _node_only(rig, CO)


def test_a_proof_replayed_on_another_network_is_refused(rig: Rig) -> None:
    proof = _bind(rig)
    _network(rig.node, NODE, (NODE, rig.mac_id), network_id="n_other")
    reply = ob.accept_presented(rig.node, rig.link(rig.mac_id, "n_other"), {"proofs": [proof]})
    assert reply == {"accepted": False, "reason": "network_mismatch"}
    # Planted by a same-uid writer under the other network's name: still refused.
    source, planted = ob._stored_path(rig.node, NET), ob._stored_path(rig.node, "n_other")
    assert source is not None and planted is not None
    shutil.copy(source, planted)
    assert not ob.is_operator_link(rig.node, rig.link(rig.mac_id, "n_other"))


def test_a_removed_or_revoked_mac_binds_nothing(rig: Rig) -> None:
    _bind(rig)
    record = net_store.load(NET, rig.node)
    record.removed_ids.append(rig.mac_id)
    net_store.save(record, rig.node)
    assert not ob.is_operator_link(rig.node, rig.link(rig.mac_id))
    _node_only(rig, rig.mac_id)


def test_a_device_the_anchor_revokes_binds_nothing(rig: Rig) -> None:
    _bind(rig)
    revoked = dataclasses.replace(rig.anchor, devices=({"device_id": rig.mac_id, "revoked": True},))
    rig.install_anchor(revoked)
    assert ob.verify_proof(
        rig.node, ob.load_stored(rig.node, NET), network_id=NET, device_id=rig.mac_id
    ) == (False, "device_revoked")
    _node_only(rig, rig.mac_id)


def test_an_old_signer_is_refused_after_a_re_anchor(rig: Rig, tmp_path: Path) -> None:
    _bind(rig)
    rig.install_anchor(_make_key(tmp_path / "other-operator"))
    assert ob.verify_proof(
        rig.node, ob.load_stored(rig.node, NET), network_id=NET, device_id=rig.mac_id
    ) == (False, "old_signer")
    _node_only(rig, rig.mac_id)


def test_a_tampered_stored_proof_is_refused_on_next_use(rig: Rig) -> None:
    _bind(rig)
    path = ob._stored_path(rig.node, NET)
    assert path is not None and oct(path.stat().st_mode & 0o777) == "0o600"
    proof = json.loads(path.read_text())
    proof["request"]["device"]["host"] = "elsewhere.test"
    path.write_text(json.dumps(proof))
    assert not ob.is_operator_link(rig.node, rig.link(rig.mac_id))
    _node_only(rig, rig.mac_id)


def test_a_co_member_cannot_claim_the_macs_proof(rig: Rig) -> None:
    """Presented by CO: ``requested_by`` names the Mac, and editing it breaks the digest."""
    _onboard(rig)
    proof = ob.own_proofs(rig.mac, NET)[0]
    reply = ob.accept_presented(rig.node, rig.link(CO), {"proofs": [proof]})
    assert reply == {"accepted": False, "reason": "device_mismatch"}
    forged = json.loads(json.dumps(proof))
    forged["request"]["requested_by"]["device_id"] = CO
    reply = ob.accept_presented(rig.node, rig.link(CO), {"proofs": [forged]})
    assert reply == {"accepted": False, "reason": "signature_invalid"}
    _node_only(rig, CO)


def test_a_node_holding_the_anchor_key_has_no_approver(rig: Rig) -> None:
    from local_operator.operator.keychain import default_file_path

    proof = _bind(rig)
    target = default_file_path(rig.node)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(default_file_path(rig.mac), target)
    assert ob.verify_proof(rig.node, proof, network_id=NET, device_id=rig.mac_id) == (
        False,
        "no_approver",
    )
    _node_only(rig, rig.mac_id)


def test_a_non_root_owned_anchor_is_never_trusted(rig: Rig) -> None:
    proof = _bind(rig)
    rig.install_anchor(rig.anchor, root_owned=False)
    assert ob.verify_proof(rig.node, proof, network_id=NET, device_id=rig.mac_id) == (
        False,
        "no_installed_anchor",
    )


def test_an_old_mac_without_the_capability_gets_node_only(rig: Rig) -> None:
    _bind(rig)
    old = rig.link(rig.mac_id, caps=frozenset({wire.REMOTE_ATTENTION_V1}))
    assert not ob.is_operator_link(rig.node, old)


def test_agent_children_and_co_member_placed_rows_are_unchanged(rig: Rig) -> None:
    _bind(rig)
    agent = {"kind": ra.AGENT_ORIGIN_KIND, "source_device": NODE, "source_session_id": "s_p"}
    _node_started_session(rig.node, origin=agent)
    assert "notices" not in _listing(rig, rig.mac_id, origin=agent)
    assert ra.notice_forwarding(rig.node, SID)[0] == ra.LOCAL_ONLY
    placed = {"kind": "user", "source_device": CO, "source_session_id": ""}
    assert "notices" not in _listing(rig, rig.mac_id, origin=placed)
    assert "notices" in _listing(rig, CO, origin=placed)


def test_only_the_onboarding_kinds_and_approvals_are_proofs(rig: Rig) -> None:
    approved = _onboard(rig)
    for change in ({"kind": "operator_device_pair"}, {"kind": "profile_save"}):
        assert ob.proof_from_record({**approved, **change}, own_device_id=rig.mac_id) is None
    assert ob.proof_from_record({**approved, "state": "denied"}, own_device_id=rig.mac_id) is None
    assert ob.proof_from_record(approved, own_device_id=CO) is None
    proof = ob.proof_from_record(approved, own_device_id=rig.mac_id)
    assert proof is not None
    relabelled = {**proof, "kind": "operator_device_pair"}
    assert ob.verify_proof(rig.node, relabelled, network_id=NET, device_id=rig.mac_id) == (
        False,
        "kind_not_allowed",
    )


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_r1_the_onboarding_requester_is_computed_not_a_parameter(rig: Rig) -> None:
    """The provenance the binding rests on: no caller can name ``requested_by``."""
    import inspect

    from local_operator.network.onboard_approvals import request_device

    params = set(inspect.signature(request_device).parameters)
    assert not {"requested_by", "requester", "requester_device_id"} & params
    record = _onboard(rig)
    assert record["requested_by"]["device_id"] == rig.mac_id
    assert ob.PROOF_KINDS == {A.KIND_DEVICE_ONBOARD, enrollment.KIND}


def test_r5_the_signature_is_verified_once_not_per_listing(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.operator import verify

    _bind(rig)
    ob._reset_caches()
    calls = []
    real = verify.verify_signature

    def counting(**kwargs: Any) -> bool:
        calls.append(1)
        return real(**kwargs)

    monkeypatch.setattr(verify, "verify_signature", counting)
    for _ in range(20):
        assert ob.is_operator_link(rig.node, rig.link(rig.mac_id))
    assert len(calls) == 1
