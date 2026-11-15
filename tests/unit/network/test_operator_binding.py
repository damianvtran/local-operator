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
    rig: Rig,
    *,
    network: str = NET,
    host: str = "node.test",
    at: float = 0.0,
    admitted: str = NODE,
) -> dict[str, Any]:
    """The operator files and SIGNS one onboarding card on the Mac (real path).

    ``admitted`` is the mesh id the onboarding RUNNER records once it admits the
    node (``onboard._record_operator_binding_target``); ``""`` records nothing.
    """
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
    approved = A.approve(
        record["approval_id"], signature_hex=sig, decided_at=decided_at, root=rig.mac
    )
    if admitted:
        assert ob.record_onboarded(
            rig.mac, approval_id=record["approval_id"], network_id=network, device_id=admitted
        )
    return approved


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
    assert ob.own_proofs(rig.mac, NET, for_device=NODE) == []
    ob.record_onboarded(rig.mac, approval_id=record["approval_id"], network_id=NET, device_id=NODE)
    proofs = ob.own_proofs(rig.mac, NET, for_device=NODE)
    assert proofs and proofs[0]["kind"] == enrollment.KIND
    assert proofs[0]["request"]["what"]["network_id"] == NET
    assert ob.verify_proof(rig.node, proofs[0], network_id=NET, device_id=rig.mac_id) == (True, "")


def test_each_onboarded_node_gets_exactly_its_own_proof(rig: Rig) -> None:
    """S1: the decision that onboarded THAT node, and no other node's, ever."""
    other = _onboard(rig, host="other.test", at=time.time() + 5, admitted="d_other")
    mine = _onboard(rig, host="node.test")
    [proof] = ob.own_proofs(rig.mac, NET, for_device=NODE)
    assert proof["approval_id"] == mine["approval_id"]
    assert proof["request"]["device"]["host"] == "node.test"
    [theirs] = ob.own_proofs(rig.mac, NET, for_device="d_other")
    assert theirs["approval_id"] == other["approval_id"]
    # No fallback to the newest proof for the network: an un-onboarded peer, the
    # wrong network or no device at all gets nothing.
    assert ob.own_proofs(rig.mac, NET, for_device=CO) == []
    assert ob.own_proofs(rig.mac, "n_other", for_device=NODE) == []
    assert ob.own_proofs(rig.mac, NET) == []


class _WireLink:
    """A link whose ``request`` records every frame the Mac would put on the wire."""

    def __init__(self, device_id: str, caps: frozenset[str] = CAPS) -> None:
        self.device_id = device_id
        self.network_id = NET
        self.capabilities = caps
        self.frames: list[dict[str, Any]] = []

    def request(self, frame: dict[str, Any], **_kw: Any) -> dict[str, Any]:
        self.frames.append(frame)
        return {"op": "ack", "detail": {}}


def _mac_server(rig: Rig) -> Any:
    return SimpleNamespace(root=rig.mac, _next_relay_req=lambda: 7)


def test_s1_a_co_member_is_sent_no_proof_even_for_a_device_id_less_card(rig: Rig) -> None:
    """S1: ``request_device`` files ``device.device_id`` empty; no fallback reaches CO."""
    approved = _onboard(rig)
    assert approved["device"]["device_id"] == ""  # the filing shape this guards
    co = _WireLink(CO)
    assert ob.present(_mac_server(rig), co) is None
    assert co.frames == []
    # Not even with NOTHING recorded at all (a pre-upgrade onboarding).
    _onboard(rig, host="legacy.test", at=time.time() + 5, admitted="")
    assert ob.present(_mac_server(rig), co) is None
    assert co.frames == []
    # The onboarded node gets one frame carrying exactly its own proof.
    node = _WireLink(NODE)
    ob.present(_mac_server(rig), node)
    [frame] = node.frames
    assert frame["op"] == ob.OP
    assert [p["approval_id"] for p in frame["proofs"]] == [approved["approval_id"]]


def test_r1_n1_a_node_without_the_capability_is_sent_nothing(rig: Rig) -> None:
    _onboard(rig)
    old = _WireLink(NODE, caps=frozenset({wire.REMOTE_ATTENTION_V1}))
    assert ob.present(_mac_server(rig), old) is None
    assert old.frames == []


def test_record_onboarded_refuses_path_shaped_ids(rig: Rig) -> None:
    for bad in ("..", "../x", "a/b", ""):
        assert not ob.record_onboarded(rig.mac, approval_id="ap_x", network_id=NET, device_id=bad)
        assert not ob.record_onboarded(rig.mac, approval_id=bad, network_id=NET, device_id=NODE)
    assert ob.onboarded_devices(rig.mac) == {}


def test_the_binding_proof_in_use_survives_the_30_day_prune(rig: Rig) -> None:
    """M1: the decision presented to each onboarded node is exempt; others prune."""
    old = _onboard(rig, host="old.test", at=time.time() - 1000)
    new = _onboard(rig, host="new.test")
    unrecorded = _onboard(rig, host="never.test", at=time.time() + 5, admitted="")
    for record in (old, new, unrecorded):
        path = A.record_path(record["approval_id"], rig.mac)
        raw = json.loads(path.read_text())
        raw["state"] = A.STATE_CONNECTED  # terminal
        raw["decided_at"] = time.time() - A.TERMINAL_PRUNE_AGE_S - (100 if record is new else 200)
        path.write_text(json.dumps(raw))
    A.sweep(root=rig.mac)
    assert not A.record_path(old["approval_id"], rig.mac).exists()
    assert not A.record_path(unrecorded["approval_id"], rig.mac).exists()
    assert A.record_path(new["approval_id"], rig.mac).exists()
    assert [p["approval_id"] for p in ob.own_proofs(rig.mac, NET, for_device=NODE)] == [
        new["approval_id"]
    ]


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
    source = ob._stored_path(rig.node, NET, rig.mac_id)
    planted = ob._stored_path(rig.node, "n_other", rig.mac_id)
    assert source is not None and planted is not None
    planted.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(source, planted)
    assert not ob.is_operator_link(rig.node, rig.link(rig.mac_id, "n_other"))


@pytest.mark.parametrize("how", ["removed", "burned"])
def test_a_removed_or_burned_mac_binds_nothing(rig: Rig, how: str) -> None:
    """S3: network membership is THE revocation lever, read live on every use."""
    _bind(rig)
    record = net_store.load(NET, rig.node)
    if how == "burned":
        record.removed_ids.append(rig.mac_id)
    else:
        for member in record.members:
            if member.device_id == rig.mac_id:
                member.lifecycle = "expired"
                member.removed_at = time.time()
    net_store.save(record, rig.node)
    assert not ob.is_operator_link(rig.node, rig.link(rig.mac_id))
    _node_only(rig, rig.mac_id)


def test_s3_the_anchor_device_list_is_not_consulted(rig: Rig) -> None:
    """The anchor's ``devices`` holds operator KEY ids, never mesh ids: not a lever."""
    proof = _bind(rig)
    listed = dataclasses.replace(rig.anchor, devices=({"device_id": rig.mac_id, "revoked": True},))
    rig.install_anchor(listed)
    # Same verdict as without the entry; revocation goes through membership above.
    assert ob.verify_proof(rig.node, proof, network_id=NET, device_id=rig.mac_id) == (True, "")


def test_an_old_signer_is_refused_after_a_re_anchor(rig: Rig, tmp_path: Path) -> None:
    _bind(rig)
    rig.install_anchor(_make_key(tmp_path / "other-operator"))
    assert ob.verify_proof(
        rig.node, ob.load_stored(rig.node, NET, rig.mac_id), network_id=NET, device_id=rig.mac_id
    ) == (False, "old_signer")
    _node_only(rig, rig.mac_id)


def test_a_tampered_stored_proof_is_refused_on_next_use(rig: Rig) -> None:
    _bind(rig)
    path = ob._stored_path(rig.node, NET, rig.mac_id)
    assert path is not None and oct(path.stat().st_mode & 0o777) == "0o600"
    proof = json.loads(path.read_text())
    proof["request"]["device"]["host"] = "elsewhere.test"
    path.write_text(json.dumps(proof))
    assert not ob.is_operator_link(rig.node, rig.link(rig.mac_id))
    _node_only(rig, rig.mac_id)


def test_a_co_member_cannot_claim_the_macs_proof(rig: Rig) -> None:
    """Presented by CO: ``requested_by`` names the Mac, and editing it breaks the digest."""
    _onboard(rig)
    proof = ob.own_proofs(rig.mac, NET, for_device=NODE)[0]
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


def test_r1_m1_a_downgraded_mac_gets_the_node_only_sentence(rig: Rig) -> None:
    """Bound and collecting, then back as an older build: tool text follows the catalogue."""
    _bind(rig)
    _node_started_session(rig.node)
    assert "notices" in _listing(rig, rig.mac_id)
    assert ra.notice_forwarding(rig.node, SID) == (ra.FORWARDED, rig.mac_id)
    old = rig.link(rig.mac_id, caps=frozenset({wire.REMOTE_ATTENTION_V1}))
    row = {"session_id": SID, "state": "stored", "origin": None}
    ra.stamp_rows_for_viewer(
        rig.node,
        [row],
        rig.mac_id,
        operator_viewer=ob.is_operator_link(rig.node, old),
        self_device_id=NODE,
    )
    assert "notices" not in row
    assert ra.notice_forwarding(rig.node, SID) == (ra.LOCAL_ONLY, "")
    assert "only" in ra.forwarded_notice_sentence(rig.node, SID)
    # Upgraded again: forwarding (and the sentence) come back.
    assert "notices" in _listing(rig, rig.mac_id)
    assert ra.notice_forwarding(rig.node, SID) == (ra.FORWARDED, rig.mac_id)


def test_r1_m2_two_bound_devices_keep_separate_slots(rig: Rig, tmp_path: Path) -> None:
    """Each device's link-up stores into its own slot; neither displaces the other."""
    _bind(rig)
    second = tmp_path / "second"
    second.mkdir()
    rig.monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(second))
    from tests.unit.network.test_approvals_store import _sign_decision as sign

    second_id = load_or_mint(second).device_id
    shutil.copytree(rig.mac / "operator", second / "operator", dirs_exist_ok=True)
    _network(second, second_id, (second_id, NODE, rig.mac_id))
    record = net_store.load(NET, rig.node)
    record.members.append(MemberRecord(device_id=second_id, name="second-box"))
    net_store.save(record, rig.node)
    from local_operator.network.onboard_approvals import request_device

    card = request_device(
        host="node.test", user="ec2-user", host_key_fp="SHA256:fixture", network=NET, root=second
    )
    at = time.time()
    A.approve(
        card["approval_id"],
        signature_hex=sign(second, card, "approve", at),
        decided_at=at,
        root=second,
    )
    ob.record_onboarded(second, approval_id=card["approval_id"], network_id=NET, device_id=NODE)
    proofs = ob.own_proofs(second, NET, for_device=NODE)
    reply = ob.accept_presented(rig.node, rig.link(second_id), {"proofs": proofs})
    assert reply == {"accepted": True, "reason": ""}
    # Both stay bound, whatever order their links come up in.
    for _ in range(2):
        assert ob.is_operator_link(rig.node, rig.link(rig.mac_id))
        assert ob.is_operator_link(rig.node, rig.link(second_id))
    assert ob.load_stored(rig.node, NET, rig.mac_id) is not None
    assert ob.load_stored(rig.node, NET, second_id) is not None


def test_r1_m3_concurrent_stores_never_collide(rig: Rig) -> None:
    import threading

    proof = _bind(rig)
    errors: list[BaseException] = []

    def write() -> None:
        try:
            for _ in range(25):
                ob._store(rig.node, NET, rig.mac_id, proof)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=write) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert ob.load_stored(rig.node, NET, rig.mac_id) == proof
    leftovers = list(ob._bindings_dir(rig.node).rglob("*.tmp"))
    assert leftovers == []


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


def test_s2_a_credential_share_shaped_device_onboard_is_no_proof(rig: Rig) -> None:
    """``device_onboard`` filed by the credential flows: refused by SHAPE on both sides.

    Worst case on purpose: the card names the network (what keeps it out today
    is only that it does not), carries the Mac as requester, and the operator
    really signed it.
    """
    from local_operator.network.approvals import _immutable_request

    trio = A.local_anchor_trio(rig.mac)
    assert trio is not None
    now = time.time()
    card = A.create_request(
        kind=A.KIND_DEVICE_ONBOARD,
        request_id=A.new_request_id(),
        requested_by={"device_id": rig.mac_id, "session_id": "", "surface": "cli"},
        device={"device_id": CO, "name": "co", "fingerprint": "fp"},
        what={
            "credential_finite_share": {"ref": "x"},
            "anchor": {k: trio[k] for k in ("key_id", "spki_fp", "statement_digest")},
            "network_id": NET,
        },
        credential_ref=None,
        created_at=now,
        expires_at=now + 3600,
        root=rig.mac,
    )
    approved = A.approve(
        card["approval_id"],
        signature_hex=_sign_decision(rig.mac, card, "approve", now),
        decided_at=now,
        root=rig.mac,
    )
    ob.record_onboarded(rig.mac, approval_id=card["approval_id"], network_id=NET, device_id=NODE)
    # The Mac never selects it ...
    assert ob.proof_from_record(approved, own_device_id=rig.mac_id) is None
    assert ob.own_proofs(rig.mac, NET, for_device=NODE) == []
    # ... and a node handed it anyway (a validly signed proof) refuses it.
    forged = {
        "v": ob.PROOF_VERSION,
        "kind": approved["kind"],
        "approval_id": approved["approval_id"],
        "request_id": approved["request_id"],
        "request": _immutable_request(approved),
        "decided_at": now,
        "sig": approved["signature"]["sig"],
        "key_id": approved["signature"]["key_id"],
    }
    assert ob.verify_proof(rig.node, forged, network_id=NET, device_id=rig.mac_id) == (
        False,
        "not_onboarding",
    )
    reply = ob.accept_presented(rig.node, rig.link(rig.mac_id), {"proofs": [forged]})
    assert reply == {"accepted": False, "reason": "not_onboarding"}
    assert not ob.is_operator_link(rig.node, rig.link(rig.mac_id))


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
