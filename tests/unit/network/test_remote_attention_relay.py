"""Remote attention over two REAL relays: the owner stamps only its origin's rows.

The strongest proof this repo can give without a second machine: two
``RelayServer``s in one process, paired by the shared fixture's real ceremony,
with a real authenticated ``PeerLink`` between them and the production
``net_catalog`` op answering over it. Device B dials device A, so A is the OWNER
answering and B the VIEWER asking; the session lives on A's disk with a stamp
naming its origin.

Pinned here, each over the wire rather than against a helper:

* the origin viewer gets ``attention``/``notices`` on the session's row, and the
  origin's watcher turns that reply into exactly one ``remote-notice:`` row in
  ITS inbox, at the owner item's own time — and NO ``remote-done:`` row (a
  remote turn-end is never a notification, operator rule 2026-11-13);
* a viewer that is NOT the session's origin gets the row without either field;
* an OLD viewer (one that does not advertise ``remote-attention-v1``; the list
  is narrowed exactly as ``test_relay_e2e``'s old-peer cells do) gets the exact
  row shape it always did.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import relay, types, wire
from local_operator.network.projection import PeerRow
from local_operator.notifications import remote_attention as ra
from local_operator.notifications.store import NotificationStore
from local_operator.session import peer_rows
from local_operator.session.attention import AttentionStore
from local_operator.session.placement import MeshStamp, SessionPlacement, write_stamp
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _live_link,
    devices,
)

SID = "c0ffee123456"
# A real UUID: the completion store refuses any other token shape.
TOKEN = str(uuid.uuid5(uuid.NAMESPACE_URL, "remote-attention-relay-test"))

Devices = tuple[relay.RelayServer, relay.RelayServer, str, int]


@pytest.fixture()
def pair(request: pytest.FixtureRequest) -> Devices:
    """The shared two-relay fixture under a name these tests can take (F811)."""
    value: Devices = request.getfixturevalue("devices")
    return value


def _own_session(owner: relay.RelayServer, origin_device: str) -> None:
    """A session on ``owner``'s disk that ``origin_device`` started there."""
    directory = owner.root / "sessions" / SID
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
    write_stamp(
        owner.root,
        MeshStamp(
            session_id=SID,
            network_id="n_test",
            home_device=owner.identity.device_id,
            placement=SessionPlacement(
                mode="peer",
                network_id="n_test",
                home_device=owner.identity.device_id,
                stamp_revision=1,
            ),
            # ``started_by: operator`` is the attestation an operator-started
            # create stamps (#4001): notify items are default-deny without it,
            # so this fixture models the attested case the relay tests cover.
            origin={
                "kind": "user",
                "source_device": origin_device,
                "source_session_id": "",
                "started_by": "operator",
            },
        ),
    )
    AttentionStore(owner.root / "attention.db").publish(
        f"session/{SID}", TOKEN, "anchor-1", "complete"
    )
    NotificationStore(owner.root).notify(
        SID, title="Release cut — sign-off needed", priority="high", key="release"
    )


def _catalog_row(link: relay.PeerLink) -> dict[str, Any]:
    reply = link.request({"op": "net_catalog", "req": 41, "locality": "remote"})
    assert reply is not None and reply.get("op") == "ack", reply
    rows = [row for row in reply["detail"]["sessions"] if row.get("session_id") == SID]
    assert len(rows) == 1, reply["detail"]["sessions"]
    return dict(rows[0])


def test_the_origin_gets_both_fields_and_its_watcher_raises_the_rows(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server_a, server_b, _host, _port = pair
    head = tmp_path / "head-inbox"
    # The head first watched the owner BEFORE the notice was posted there, so the
    # no-backfill rule lets it through.
    ra.origin_baseline_ms(head, server_a.identity.device_id)
    time.sleep(0.002)
    _record, link = _live_link(pair, monkeypatch)
    try:
        _own_session(server_a, server_b.identity.device_id)
        assert wire.REMOTE_ATTENTION_V1 in link.capabilities
        row = _catalog_row(link)
    finally:
        link.close("test")
    assert row["attention"] == {"token": TOKEN, "outcome": "complete"}
    [notice_block] = row["notices"]
    assert notice_block["key"] == "release"
    assert notice_block["priority"] == "high"

    # THE ORIGIN HALF, from the bytes that crossed: the production parse, the
    # production row builder, the production watcher.
    owner = server_a.identity.device_id

    class _Peer:
        device_id = owner
        name = "lop-remote-node-2"
        network_id = "n_test"
        reachable = True
        reason = ""

    class _Catalog:
        def peers(self) -> list[Any]:
            return [_Peer()]

        def rows(self) -> list[Any]:
            return [PeerRow.from_json(row, device_id=owner)]

    monkeypatch.setattr(ra, "_deliver_async", lambda *_a, **_k: None)
    rows = peer_rows.peer_session_rows(None, ttl_s=0, catalog=_Catalog())
    ra.RemoteAttentionWatcher(head).observe(rows)
    items, _ = NotificationStore(head).list_items(state="all", limit=10)
    assert [item["key"] for item in items] == [ra.notice_key(owner, SID, "release")]
    [notice] = items
    assert notice["priority"] == "high"
    # Written at the OWNER item's time, not the sync moment.
    assert notice["created_at"] == notice_block["at"] > 0


#: The owner-side key of the forwarded ask, spelled literally so this cell fails
#: on BEHAVIOUR (no such notice) against a build without the forwarder.
ASK_KEY = "remote-ask:q7"


def _ask_event(owner: relay.RelayServer, kind: str, ask_id: str, **extra: Any) -> None:
    from local_operator.asks import store as ask_store

    now = ask_store.now_ms()
    ask_store.append_event(
        ask_store.session_dir(owner.root, SID),
        {"v": ask_store.EVENT_SCHEMA, "kind": kind, "at": now, "ask_id": ask_id, **extra},
    )


def test_an_open_queued_ask_crosses_and_its_answer_retires_the_origin_row(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """node-ask-forward (2026-11-13): a node session's queued ask over the wire.

    The ask is NOT a NotificationStore row on the owner, so before this nothing
    carried it: the origin never learned an agent on the node was waiting.
    """
    from local_operator.asks import store as ask_store

    server_a, server_b, _host, _port = pair
    owner = server_a.identity.device_id
    head = tmp_path / "head-inbox"
    # The head first watched the owner BEFORE the ask was queued there, so the
    # no-backfill rule lets it through (the notice test's own setup).
    ra.origin_baseline_ms(head, owner)
    _record, link = _live_link(pair, monkeypatch)
    _own_session(server_a, server_b.identity.device_id)
    _ask_event(
        server_a,
        ask_store.EVENT_QUEUED,
        "q7",
        expires_at=ask_store.now_ms() + 3_600_000,
        questions=[{"id": "q0", "question": "Install Chromium here?", "options": []}],
    )

    class _Peer:
        device_id = owner
        name = "lop-remote-node-2"
        network_id = "n_test"
        reachable = True
        reason = ""

    def _origin_rows(row: dict[str, Any]) -> Any:
        class _Catalog:
            def peers(self) -> list[Any]:
                return [_Peer()]

            def rows(self) -> list[Any]:
                return [PeerRow.from_json(row, device_id=owner)]

        return peer_rows.peer_session_rows(None, ttl_s=0, catalog=_Catalog())

    monkeypatch.setattr(ra, "_deliver_async", lambda *_a, **_k: None)
    watcher = ra.RemoteAttentionWatcher(head)
    try:
        row = _catalog_row(link)
        block = row["notices"][0]
        assert {k: v for k, v in block.items() if k != "at"} == {
            "key": ASK_KEY,
            "revision": 1,
            "priority": "high",
            "title": "Install Chromium here?",
        }
        # The ask's own queue time rides the block (the no-backfill rule).
        assert block["at"] > 0
        watcher.observe(_origin_rows(row), listed_at_ms=ra_now())
        key = ra.notice_key(owner, SID, ASK_KEY)
        item = NotificationStore(head).get_by_key(SID, key)
        assert item is not None and item["priority"] == "high" and not item["dismissed_at"]

        # Answered on the node (any settle; dismissal is the simplest to write).
        _ask_event(server_a, ask_store.EVENT_DISMISSED, "q7", reason="answered_in_chat")
        row = _catalog_row(link)
    finally:
        link.close("test")
    assert all(b["key"] != ASK_KEY for b in row.get("notices", []))
    watcher.observe(_origin_rows(row), listed_at_ms=ra_now() + 1)
    item = NotificationStore(head).get_by_key(SID, key)
    assert item is not None and item["dismissed_at"], "the answered ask still asks on the origin"


def test_a_queued_ask_claim_crosses_as_queued_and_never_parks_a_member(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR #3934 Q1: a member that did not start the session is not paged.

    The owner's live record says ``pending="ask"`` for an open QUEUED ask; that
    reaches every member's listing (the "needs" badge), and before the
    ``pending_queued`` fact every member's TUI also turned it into a high park
    row. Over the wire, through the production parse and row builder.
    """
    import subprocess
    import sys

    from local_operator.session.runtime import registry
    from local_operator.session.runtime.types import SessionRecord

    server_a, server_b, _host, _port = pair
    owner = server_a.identity.device_id
    _record, link = _live_link(pair, monkeypatch)
    # Started by a THIRD device: this viewer (B) is not its origin.
    _own_session(server_a, "d_someone_else")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])

    class _Peer:
        device_id = owner
        name = "lop-remote-node-2"
        network_id = "n_test"
        reachable = True
        reason = ""

    def _parks(row: dict[str, Any]) -> list[str]:
        class _Catalog:
            def peers(self) -> list[Any]:
                return [_Peer()]

            def rows(self) -> list[Any]:
                return [PeerRow.from_json(row, device_id=owner)]

        rows = peer_rows.peer_session_rows(None, ttl_s=0, catalog=_Catalog())
        edges, _ = peer_rows.park_edges({}, rows)
        return [edge.kind for edge in edges]

    def _publish(**fields: Any) -> None:
        registry.publish(
            SessionRecord(
                pid=child.pid,
                kind="tui",
                session_id=SID,
                conversation_name="node work",
                cwd=str(server_a.root),
                model_label="test/model",
                control_port=0,
                control_key="",
                **fields,
            ),
            server_a.root,
        )

    try:
        _publish(pending="ask", pending_queued=True)
        row = _catalog_row(link)
        assert row["pending"] == "ask" and row.get("pending_queued") is True
        assert _parks(row) == [], "a queued ask paged a member that did not start it"
        # A BLOCKING gate on the same session is still a park everywhere.
        _publish(pending="ask")
        row = _catalog_row(link)
        assert "pending_queued" not in row
        assert _parks(row) == ["ask"]
    finally:
        link.close("test")
        child.kill()
        child.wait()


def ra_now() -> int:
    from local_operator.notifications.store import now_ms

    return now_ms()


def test_a_non_origin_viewer_gets_neither_field(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, _server_b, _host, _port = pair
    _record, link = _live_link(pair, monkeypatch)
    try:
        # Started by SOME OTHER device: B is a co-member, not the operator's origin.
        _own_session(server_a, "d_someone_else")
        row = _catalog_row(link)
    finally:
        link.close("test")
    assert "attention" not in row and "notices" not in row


def test_an_old_viewer_gets_the_row_shape_it_always_did(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        wire,
        "LINK_CAPABILITIES",
        tuple(c for c in wire.LINK_CAPABILITIES if c != wire.REMOTE_ATTENTION_V1),
    )
    server_a, server_b, _host, _port = pair
    _record, link = _live_link(pair, monkeypatch)
    try:
        _own_session(server_a, server_b.identity.device_id)
        assert wire.REMOTE_ATTENTION_V1 not in link.capabilities
        row = _catalog_row(link)
    finally:
        link.close("test")
    assert "attention" not in row and "notices" not in row


def test_types_module_is_untouched_by_the_capability() -> None:
    """The capability is a LINK advertisement, never a role grant."""
    assert wire.REMOTE_ATTENTION_V1 in wire.LINK_CAPABILITIES
    assert wire.REMOTE_ATTENTION_V1 not in types.GRANTABLE_CAPABILITIES


def test_a_node_started_ask_and_notify_reach_the_operator_bound_by_its_signature(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The operator binding over REAL relays (``network/operator_binding``).

    B plays the operator's Mac: it files and SIGNS a real onboarding card for
    the network (file-only key), and A — the node — has that key's anchor
    installed root-owned (the one seam). B's link-up hook presents the signed
    decision on its own; A verifies and stores it, and then a session STARTED on
    A (no stamp, no origin) lists its open ask and its ``notify`` to B. Without a
    verified proof the same row carries neither (the pre-binding control).
    """
    from local_operator.asks import store as ask_store
    from local_operator.network import operator_binding as ob

    ask_key_q9 = ra.ASK_NOTICE_PREFIX + "q9"
    from local_operator.network.approvals import approve
    from local_operator.network.onboard_approvals import request_device
    from local_operator.network.store import list_networks
    from local_operator.operator import trust
    from tests.unit.network.test_approvals_store import _make_key, _sign_decision

    server_a, server_b, _host, _port = pair
    head = tmp_path / "operator-inbox"
    # The operator's inbox first watched the node BEFORE anything was posted.
    ra.origin_baseline_ms(head, server_a.identity.device_id)
    time.sleep(0.002)
    # The node-started session on A: no stamp at all, an open ask and a notify.
    directory = server_a.root / "sessions" / SID
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
    NotificationStore(server_a.root).notify(SID, title="Build done", priority="high", key="build")
    _ask_event(
        server_a,
        ask_store.EVENT_QUEUED,
        "q9",
        expires_at=ask_store.now_ms() + 3_600_000,
        questions=[{"id": "q0", "question": "Allow example.com?", "options": []}],
    )

    # CONTROL: before any binding, B's listing of A's node-started row is bare.
    _record, link = _live_link(pair, monkeypatch)
    try:
        bare = _catalog_row(link)
    finally:
        link.close("test")
    assert "notices" not in bare and "attention" not in bare

    # The operator signs one onboarding card on B, and A has that key installed
    # (patched BEFORE the approval: the approve path reads the installed anchor
    # first, and this host may carry a real one).
    anchor = _make_key(server_b.root)

    def installed(uid: Any = None) -> Any:
        return trust.AnchorLoad(
            anchor=anchor, path=trust.anchor_path(uid), root_owned=True, reason="", exists=True
        )

    monkeypatch.setattr(trust, "load_anchor", installed)
    ob._reset_caches()
    [network] = list_networks(server_b.root)
    card = request_device(
        host="node.test",
        user="ec2-user",
        host_key_fp="SHA256:fixture",
        network=network.network_id,
        credential_ref="ssh-node-ref",
        root=server_b.root,
    )
    assert card["requested_by"]["device_id"] == server_b.identity.device_id
    decided_at = time.time()
    approve(
        card["approval_id"],
        signature_hex=_sign_decision(server_b.root, card, "approve", decided_at),
        decided_at=decided_at,
        root=server_b.root,
    )

    # The onboarding RUNNER on B records the mesh id that run admitted (S1):
    # only that node is ever presented this proof.
    assert ob.record_onboarded(
        server_b.root,
        approval_id=card["approval_id"],
        network_id=network.network_id,
        device_id=server_a.identity.device_id,
    )
    # A fresh link: B's link-up hook presents the proof by itself.
    link = server_b.dial(network.network_id, host=f"{_host}:{_port}", epoch=network.epoch)[0]
    assert link is not None
    try:
        stored = None
        deadline = time.time() + 10.0
        while time.time() < deadline and stored is None:
            stored = ob.load_stored(server_a.root, network.network_id, server_b.identity.device_id)
            time.sleep(0.05)
        assert stored is not None, "the link-up presentation never reached the node"
        assert wire.OPERATOR_BINDING_V1 in link.capabilities
        row = _catalog_row(link)
    finally:
        link.close("test")
    keys = [notice["key"] for notice in row["notices"]]
    assert keys[0] == ra.ASK_NOTICE_PREFIX + "q9" and "build" in keys
    # Notices only: a node-started session's completions are not forwarded.
    assert "attention" not in row and "attention_unknown" not in row
    # The node's notify result now says where it goes.
    assert ra.notice_forwarding(server_a.root, SID) == (
        ra.FORWARDED,
        server_b.identity.device_id,
    )
    assert "the operator's device" in ra.forwarded_notice_sentence(server_a.root, SID)

    # THE OPERATOR'S HALF, from the bytes that crossed: the production parse and
    # watcher raise the ask and the notice in the bound device's own inbox.
    owner = server_a.identity.device_id

    class _Peer:
        device_id = owner
        name = "node"
        network_id = network.network_id
        reachable = True
        reason = ""

    class _Catalog:
        def peers(self) -> list[Any]:
            return [_Peer()]

        def rows(self) -> list[Any]:
            return [PeerRow.from_json(row, device_id=owner)]

    monkeypatch.setattr(ra, "_deliver_async", lambda *_a, **_k: None)
    # B's own root: the Mac raises a started-on-owner ``notify`` only from a node
    # its records say it onboarded (``peer_rows._bound_nodes``).
    ra.RemoteAttentionWatcher(head).observe(
        peer_rows.peer_session_rows(server_b.root, ttl_s=0, catalog=_Catalog())
    )
    items, _ = NotificationStore(head).list_items(state="all", limit=10)
    assert sorted(item["key"] for item in items) == sorted(
        [ra.notice_key(owner, SID, "build"), ra.notice_key(owner, SID, ask_key_q9)]
    )
    built = next(item for item in items if item["key"] == ra.notice_key(owner, SID, "build"))
    assert built["title"] == "Build done", "the forwarded row carries the node's own text"


def test_s1_a_peer_the_mac_did_not_onboard_gets_no_proof_bytes_on_the_wire(
    pair: Devices, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """PR #4030 security review S1, over REAL relays and the real link-up hook.

    B (the operator's Mac) holds a signed onboarding decision for the network,
    filed the way ``request_device`` files it (``device.device_id`` empty), but
    its records show no onboarding that admitted A: to B, A is a co-member. Every
    frame B's links SEAL for the wire is captured (``PeerLink._write``, the last
    step before the codec): no ``net_operator_binding`` frame and none of the
    preimage's bytes (SSH host, credential-ref name) may appear. Before the fix
    the "newest proof for the network" fallback put them on A's link. The same
    holds when the onboarding runner's already-active path was handed A's id by
    the SSH node's own ``identity show``: that report alone records nothing.
    """
    import json
    import threading

    from local_operator.network import operator_binding as ob
    from local_operator.network.approvals import approve
    from local_operator.network.onboard_approvals import request_device
    from local_operator.network.store import list_networks
    from local_operator.operator import trust
    from tests.unit.network.test_approvals_store import _make_key, _sign_decision

    server_a, server_b, _host, _port = pair
    _record, warm = _live_link(pair, monkeypatch)
    warm.close("test")
    anchor = _make_key(server_b.root)

    def installed(uid: Any = None) -> Any:
        return trust.AnchorLoad(
            anchor=anchor, path=trust.anchor_path(uid), root_owned=True, reason="", exists=True
        )

    monkeypatch.setattr(trust, "load_anchor", installed)
    ob._reset_caches()
    [network] = list_networks(server_b.root)
    card = request_device(
        host="secret-node.test",
        user="ec2-user",
        host_key_fp="SHA256:fixture",
        network=network.network_id,
        credential_ref="ssh-secret-ref",
        root=server_b.root,
    )
    assert card["device"]["device_id"] == ""
    decided_at = time.time()
    approve(
        card["approval_id"],
        signature_hex=_sign_decision(server_b.root, card, "approve", decided_at),
        decided_at=decided_at,
        root=server_b.root,
    )
    # The already-active path of B's onboarding runner, for a node that reports
    # A's id in ``identity show`` (A IS an active member, so the step is
    # satisfied) while no invite this approval minted was admitted for A. Before
    # the fix that self-report alone was recorded, and the link-up hook below then
    # presented the proof to A. Run through the runner's own recording method.
    from types import SimpleNamespace

    from local_operator.network.onboard import OnboardRun

    runner: Any = SimpleNamespace(
        root=server_b.root,
        approval_id=card["approval_id"],
        view=SimpleNamespace(what={"network_id": network.network_id}, receipts=[]),
    )
    OnboardRun._record_operator_binding_target(runner, member_id=server_a.identity.device_id)
    assert ob.onboarded_devices(server_b.root) == {}
    # R2-M1 ruling (b)/(g): B also holds a signed credential card for ANOTHER
    # active member. Its signed recipient is not A, so A's link carries none of it.
    from local_operator.network import store as net_store
    from local_operator.network.types import MemberRecord
    from tests.unit.network.test_operator_binding import CARD_SECRETS, file_card

    third = "d_third_member"
    record_b = net_store.load(network.network_id, server_b.root)
    record_b.members.append(MemberRecord(device_id=third, name="third"))
    net_store.save(record_b, server_b.root)
    file_card(
        server_b.root,
        owner=server_b.identity.device_id,
        recipient=third,
        network=network.network_id,
    )
    assert len(ob.own_proofs(server_b.root, network.network_id, for_device=third)) == 1
    assert (
        ob.own_proofs(server_b.root, network.network_id, for_device=server_a.identity.device_id)
        == []
    )

    sealed: list[dict[str, Any]] = []
    real_write = relay.PeerLink._write

    def capture(self: relay.PeerLink, frame: dict[str, Any]) -> None:
        sealed.append(frame)
        real_write(self, frame)

    monkeypatch.setattr(relay.PeerLink, "_write", capture)
    presented = threading.Event()
    real_present = ob.present

    def watched(server: Any, link: Any) -> Any:
        try:
            return real_present(server, link)
        finally:
            if server is server_b:
                presented.set()

    monkeypatch.setattr(ob, "present", watched)
    link = server_b.dial(network.network_id, host=f"{_host}:{_port}", epoch=network.epoch)[0]
    assert link is not None
    try:
        assert wire.OPERATOR_BINDING_V1 in link.capabilities
        # B's link-up hook ran its presentation decision (off-thread).
        assert presented.wait(10.0), "the link-up presenter never ran"
        reply = link.request({"op": "net_catalog", "req": 41, "locality": "remote"})
        assert reply is not None
    finally:
        link.close("test")
    wire_text = json.dumps(sealed, default=str)
    assert not [f for f in sealed if f.get("op") == ob.OP]
    for secret in ("secret-node.test", "ssh-secret-ref", "SHA256:fixture", *CARD_SECRETS):
        assert secret not in wire_text
    assert ob.load_stored(server_a.root, network.network_id, server_b.identity.device_id) is None


def test_a_head_relay_of_a_forwarded_node_site_card_resurfaces_it_and_the_node_answer_retires_it(
    pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """E2E 44ab2361d833 (lop-remote-node-2, cards a-1f70/a-4452, 2026-11-15).

    A head-node coordinator's ``ask(relay='<node session>/<ask>')`` was refused
    with "names no ask in the original session's log" although the card's
    ``remote-notice:`` row sat in the head's inbox. Over two real relays: the
    node's SITE card (``site_approval``'s own builder) crosses the catalogue, the
    head's watcher raises it, the relay re-surfaces that one row (unread, one new
    revision, no relay row in the coordinator's log), the node's own answer —
    the record ``Session.site_decide`` writes once the signed decision applied —
    settles the ask there, and the next listing retires the head's row; a relay
    after that is refused as expired/answered. The head's root is device B's
    real one, so its network record (A an active member) is the attestation.
    """
    from local_operator.asks import render
    from local_operator.asks import store as ask_store
    from local_operator.asks.queue import AskQueue
    from local_operator.browser_bridge.site_approval import site_ask_question

    server_a, server_b, _host, _port = pair
    owner = server_a.identity.device_id
    head = server_b.root
    ask_id = "a-4452"
    key = ra.notice_key(owner, SID, f"{ra.ASK_NOTICE_PREFIX}{ask_id}")
    ra.origin_baseline_ms(head, owner)
    _record, link = _live_link(pair, monkeypatch)
    _own_session(server_a, server_b.identity.device_id)
    question = site_ask_question(
        origin="https://example.com",
        node_name="lop-remote-node-2",
        node_device_id=owner,
        entry_id="entry-1",
    )
    _ask_event(
        server_a,
        ask_store.EVENT_QUEUED,
        ask_id,
        expires_at=ask_store.now_ms() + 3_600_000,
        questions=[question],
    )

    class _Peer:
        device_id = owner
        name = "lop-remote-node-2"
        network_id = "n_test"
        reachable = True
        reason = ""

    def _origin_rows(row: dict[str, Any]) -> Any:
        class _Catalog:
            def peers(self) -> list[Any]:
                return [_Peer()]

            def rows(self) -> list[Any]:
                return [PeerRow.from_json(row, device_id=owner)]

        return peer_rows.peer_session_rows(None, ttl_s=0, catalog=_Catalog())

    banners: list[int] = []
    monkeypatch.setattr(
        ra, "_deliver_async", lambda _r, outcome, **_k: banners.append(outcome.revision)
    )
    coordinator = AskQueue(None, config_dir=head, session_id="c0decafe0001")
    watcher = ra.RemoteAttentionWatcher(head)
    try:
        # BEFORE the forward: nothing in the head's inbox, so the old sentence.
        assert coordinator.enqueue_relay(f"{SID}/{ask_id}") == {
            "ok": False,
            "error": render.RELAY_ORIGIN_UNKNOWN,
        }
        watcher.observe(_origin_rows(_catalog_row(link)), listed_at_ms=ra_now())
        raised = NotificationStore(head).get_by_key(SID, key)
        assert raised is not None and not raised["dismissed_at"]
        # The operator glanced at it; a relay of a still-unread row writes
        # nothing (R1-2), so read it first to exercise the re-surface.
        NotificationStore(head).apply("read", ids=[(raised["id"], raised["revision"])], via="test")

        relayed = coordinator.enqueue_relay(f"{SID}/{ask_id}")
        assert relayed["ok"], relayed
        assert "ORIGINAL session" in relayed["text"]
        resurfaced = NotificationStore(head).get_by_key(SID, key)
        assert resurfaced is not None and not resurfaced["read"]
        assert resurfaced["id"] == raised["id"]
        assert resurfaced["revision"] == raised["revision"] + 1
        assert resurfaced["title"] == raised["title"]
        assert banners[-1] == resurfaced["revision"]
        assert ask_store.read_events(ask_store.session_dir(head, "c0decafe0001")) == []

        # The node's answer, recorded where ``Session.site_decide`` records it.
        node_queue = AskQueue(None, config_dir=server_a.root, session_id=SID)
        answered = node_queue.respond(ask_id, {"site-access": ["deny"]}, by="operator")
        assert answered.get("ok"), answered
        assert node_queue.find(ask_id)["status"] == ask_store.STATUS_ANSWERED
        row = _catalog_row(link)
    finally:
        link.close("test")
    assert all(b["key"] != f"{ra.ASK_NOTICE_PREFIX}{ask_id}" for b in row.get("notices", []))
    watcher.observe(_origin_rows(row), listed_at_ms=ra_now() + 1)
    retired = NotificationStore(head).get_by_key(SID, key)
    assert retired is not None and retired["dismissed_via"] == "remote-attention"
    after = coordinator.enqueue_relay(f"{SID}/{ask_id}")
    assert after["ok"] is False and "expired or was already answered" in after["error"]
