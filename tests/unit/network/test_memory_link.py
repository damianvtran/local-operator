"""Native relay/PeerLink sockets, two isolated homes, no live mesh mutation."""

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.memory import MemoryError
from local_operator.memory import MemoryError as SensitiveMemoryError
from local_operator.memory import (
    MemoryIdentity,
    MemoryStore,
    guard,
    parse_url,
    safe_segment,
)
from local_operator.network import audit, identity, memory, wire
from tests.unit.network.test_mcpdefs_link import _bind_and_start, _set_peer_endpoints
from tests.unit.network.test_relay_e2e import _pair, serve_shaped_relay

# Fixed PUBLIC Ed25519 bytes from the independent public-origin discriminator.
# This is neither a private key nor host provenance. Derive through production
# device_id_for in the test; never search/mint until a guard happens to accept.
_PUBLIC_ORIGIN_KEY = "8464ccf9ebd8d19a0f63287f11100526a9b9ca0222825fc2ed997f6cfe1821e0"
_PUBLIC_ORIGIN_ID = "d_ac37443550475dfe7a5ffd3028fc5964"


def _public_origin_hazard():
    value = identity.device_id_for(bytes.fromhex(_PUBLIC_ORIGIN_KEY))
    binding_matches = value == _PUBLIC_ORIGIN_ID
    assert binding_matches, "fixed public-key/device-ID binding changed"
    return value


def _assert_value_free_refusal(caught, value):
    # Keep rejected bytes out of failure messages/properties: report only the
    # label/boolean, not an assertion that renders the rejected argument.
    did_not_echo = value not in str(caught.value)
    assert did_not_echo, "sensitive-memory refusal echoed its input"


def test_public_origin_free_text_guards_still_refuse_before_any_put(tmp_path):
    """The free-text refusals survive the merged structural-id repairs.

    This case originally characterized raw-string refusal as the fail-closed
    state of every surface. The merged repairs deliberately accept a minted
    ``d_``+hex device id on the write/sync path (#2492) and exempt a minted
    reference from the recall-query guard (#2511); both are pinned by their own
    regression suites. This control keeps what those repairs did NOT move:
    guard() and a free-text safe_segment() still refuse the raw value, and
    constructing with it (no put) writes no storage artifact; a future
    host-bound change must arrive through a separately reviewed path, never
    this arbitrary naked string. No peer/identity mint/socket fixture is
    requested by these controls.
    """
    value = _public_origin_hazard()
    root = tmp_path / "refused-origin"
    with pytest.raises(SensitiveMemoryError, match="Sensitive value refused") as rejected:
        guard(value)
    _assert_value_free_refusal(rejected, value)
    with pytest.raises(SensitiveMemoryError, match="Sensitive value refused") as rejected:
        safe_segment(value)
    _assert_value_free_refusal(rejected, value)
    # Structural acceptance itself belongs to #2492's regression suite; the
    # only claim here is that construction alone touches no storage (no put).
    MemoryStore(
        root / "config",
        MemoryIdentity("synthetic-session"),
        root / "session",
        origin=value,
    )
    assert not root.exists(), "constructing with a structural origin left a storage artifact"


# A minted-reference recall query is now an admitted read (#2511), pinned by
# that repair's own tests; these carriers are the surfaces that still refuse.
@pytest.mark.parametrize("carrier", ["body", "title", "tags", "source-uri", "url", "owner"])
def test_public_origin_bytes_remain_untrusted_in_other_fields(tmp_path, carrier):
    value = _public_origin_hazard()
    root = tmp_path / "refused-carrier"
    store = MemoryStore(
        root / "config",
        MemoryIdentity("synthetic-session"),
        root / "session",
        origin="synthetic-device",
    )
    fact: dict[str, Any] = {
        "title": "Widget proof",
        "body": "Deployment evidence in docs/widget.md",
    }
    with pytest.raises(SensitiveMemoryError, match="Sensitive value refused") as rejected:
        if carrier in ("body", "title"):
            fact[carrier] = value
            store.put("session", fact)
        elif carrier == "tags":
            fact["tags"] = [value]
            store.put("session", fact)
        elif carrier == "source-uri":
            fact["source"] = {
                "kind": "observed",
                "uri": "https://example.invalid/evidence/" + value,
            }
            store.put("session", fact)
        elif carrier == "url":
            # A valid memory URL/entity reaches the actual safe-segment guard;
            # unknown scope/syntax errors are not a sensitive-value refusal.
            parse_url("memory://session/" + value)
        else:
            MemoryIdentity(session_id=value)
    _assert_value_free_refusal(rejected, value)
    assert not root.exists(), "untrusted carrier refusal left a storage artifact"


def test_benign_origin_and_fact_still_use_the_real_store(tmp_path):
    root = tmp_path / "benign-control"
    store = MemoryStore(
        root / "config",
        MemoryIdentity("synthetic-session"),
        root / "session",
        origin="synthetic-device",
    )
    body = "Widget deployment evidence lives at docs/widget.md"
    item = store.put("session", {"title": "Widget proof", "body": body})
    assert store.path.is_file()
    assert store.show("session", item["id"])["fact"]["body"] == body
    assert store.search("Widget deployment", scope="session")


@pytest.fixture
def devices(tmp_path, monkeypatch):
    servers = []
    for name in ("a", "b"):
        home = tmp_path / ("home-" + name)
        root = home / ".local-operator"
        monkeypatch.setenv("HOME", str(home))
        keypair = identity.mint(root, name="memory-device-" + name)
        servers.append(
            serve_shaped_relay(root, monkeypatch, identity=keypair, audit=audit.AuditLog(root))
        )
    a, b = servers
    host, port = a.bind()
    a.bind_control()
    a.start()
    try:
        yield a, b, host, port
    finally:
        a.stop()
        b.stop()


def enable(server, *, scopes=None):
    (server.root / "config.yml").write_text(
        json.dumps(
            {
                "values": {
                    "memory": {"enabled": True, "capture": True, "suggest": True},
                    "network": {
                        "memory": {"enabled": True, "scopes": scopes or list(memory.SCOPES)}
                    },
                }
            }
        )
    )


def bank(server):
    return MemoryStore(
        server.root,
        MemoryIdentity("memory-session", "memory-team", "memory-agent"),
        server.root / "sessions" / "memory-session",
        origin=server.identity.device_id,
    )


def linked(devices, monkeypatch, *, role="admin"):
    a, b, _, _ = devices
    record, _, _ = _pair(devices, monkeypatch, role=role)
    live = _bind_and_start(b, record)
    _set_peer_endpoints(a, record, b.identity.device_id, [live])
    enable(a)
    enable(b)
    return a, b


def test_promoted_session_uses_real_origin_and_syncs_over_sockets(devices, monkeypatch):
    import shutil

    from local_operator.memory import SESSION_EXPORT_NAME

    a, b = linked(devices, monkeypatch)
    original = bank(a)
    item = original.put(
        "session", {"title": "Promotion proof", "body": "Promotion evidence docs/promoted.md"}
    )
    directory = b.root / "sessions" / "promoted-session"
    directory.mkdir(parents=True)
    assert original.session_dir is not None
    shutil.copyfile(original.session_dir / SESSION_EXPORT_NAME, directory / SESSION_EXPORT_NAME)
    promoted = MemoryStore(
        b.root, MemoryIdentity("promoted-session"), directory, origin=b.identity.device_id
    )
    promoted.import_session()
    rows = [json.loads(line) for line in (directory / SESSION_EXPORT_NAME).read_text().splitlines()]
    assert rows[0]["origin_device"] == b.identity.device_id
    assert rows[0]["owner_key"] == "promoted-session"
    assert item["version"] in rows[0]["parents"]
    snapshot = (directory / SESSION_EXPORT_NAME).read_bytes()
    promoted.import_session()
    assert snapshot == (directory / SESSION_EXPORT_NAME).read_bytes()
    result = memory.push_to_peer(b, a.identity.device_id)
    assert result["code"] == "in_sync", result
    receiving = MemoryStore(a.root, MemoryIdentity("promoted-session"))
    assert receiving.search("Promotion evidence")
    print("MESH promotion real-device origin, canonical restart and socket recall PASS")


def test_late_gap_fill_replays_previously_skipped_history_over_sockets(devices, monkeypatch):
    a, b = linked(devices, monkeypatch)
    left, right = bank(a), bank(b)
    first = left.put(
        "global", {"title": "Predecessor proof", "body": "Predecessor global evidence"}
    )
    second = left.put("session", {"title": "Partial session", "body": "Partial offline evidence"})
    with left.connection() as db:
        records = [json.loads(row[0]) for row in db.execute("SELECT data FROM ops ORDER BY seq")]
        # Synthetic cold receiver: preserve its scoped session export but not
        # the full bank. No operator/live database is touched by this fixture.
        for table in ("recall", "current", "causal_edges", "associations", "ops", "meta"):
            db.execute("DELETE FROM " + table)
        db.commit()
    right.apply_ops([records[1]], contiguous=False)
    result = memory.push_to_peer(b, a.identity.device_id)
    assert result["code"] == "in_sync", result
    assert not left.search("Predecessor global")
    assert left.search("Partial offline")
    assert memory.generation(right) == 0
    right.apply_ops([records[0]], contiguous=False)
    assert memory.generation(right) == 1
    result = memory.push_to_peer(b, a.identity.device_id)
    assert result["code"] == "in_sync", result
    assert left.search("Predecessor global")[0]["id"] == first["id"]
    assert left.search("Partial offline")[0]["id"] == second["id"]
    print("MESH missing-prefix scoped relay then late gap-fill generation replay PASS")


def test_sender_withholds_by_id_and_receiver_refuses_without_secret_residue(devices, monkeypatch):
    import hashlib

    from local_operator.memory import _dump

    a, b = linked(devices, monkeypatch)
    left, right = bank(a), bank(b)
    item = left.put(
        "global", {"title": "Guard proof", "body": "Originally valid synthetic evidence"}
    )
    # Deliberately bypass the store boundary to test the sender's independent
    # guard. This synthetic corruption is NOT an admitted memory API write.
    bad = "passport number AB" + "1234567"
    with left.connection() as db:
        op = json.loads(
            db.execute("SELECT data FROM ops WHERE op_id=?", (item["version"],)).fetchone()[0]
        )
        op["fact"]["body"] = bad
        op["body_sha"] = hashlib.sha256(_dump(op["fact"]).encode()).hexdigest()
        db.execute("UPDATE ops SET data=? WHERE op_id=?", (_dump(op), item["version"]))
        db.commit()
    result = memory.push_to_peer(a, b.identity.device_id)
    assert result["code"] == "withheld", result
    assert result["withheld"] == [item["version"]]
    assert right.search("Guard proof") == []
    assert not any(
        bad.encode() in path.read_bytes() for path in right.root.glob("*") if path.is_file()
    )
    link = a._ensure_link(b.identity.device_id)
    payload = {
        "v": 1,
        "generation": 0,
        "scopes": ["global"],
        "after": {a.identity.device_id: 1},
        "ops": [op],
        "skips": [],
        "proofs": [],
    }
    reply = link.request(
        {
            "op": "net_memory",
            "phase": "push",
            "generation": 0,
            "scopes": ["global"],
            "payload": payload,
            "req": a._next_relay_req(),
            "locality": "remote",
        },
        timeout=5,
    )
    assert reply["op"] == "error"
    # The receiver refuses at the op level: validate_op → validate_fact applies the
    # free-text guard to the fact body, so the sentence names the op REFERENCE (its
    # id) and certainly not the rejected value itself.
    assert "Sensitive value refused" in reply["message"]
    assert bad not in json.dumps(reply)
    with right.connection() as db:
        assert db.execute("SELECT count(*) FROM ops").fetchone()[0] == 0
    print(
        "MESH unsafe sender op withheld by ID; direct receiver refuses payload; "
        "no peer body/index/history residue PASS"
    )


def test_all_scopes_real_transport_restart_and_tombstone_convergence(devices, monkeypatch):
    a, b = linked(devices, monkeypatch)
    left, right = bank(a), bank(b)
    items = []
    for scope in memory.SCOPES:
        items.append(
            left.put(
                scope,
                {
                    "title": "Mercury evidence " + scope,
                    "body": "Mercury proof is in docs/proof.md",
                    "tags": ["mercury", "proof"],
                },
            )
        )
    response = memory.push_to_peer(a, b.identity.device_id)
    assert response["ok"], response
    for scope in memory.SCOPES:
        assert right.search("Mercury proof", scope), scope
    assert memory.push_to_peer(a, b.identity.device_id)["batches"] == 0
    # A restarted store, not a shared object/cache, reads the peer's own SQLite.
    restarted = bank(b)
    current = restarted.show("session", items[1]["id"])
    corrected = restarted.put(
        "session",
        {"title": "Corrected Mercury proof", "body": "Proof location docs/corrected.md"},
        entity=items[1]["id"],
        expected_version=current["version"],
    )
    response = memory.push_to_peer(b, a.identity.device_id)
    assert response["ok"], response
    assert left.show("session", items[1]["id"])["fact"]["body"].endswith("corrected.md")
    left.put(
        "session", {}, entity=items[1]["id"], expected_version=corrected["version"], forget=True
    )
    response = memory.push_to_peer(a, b.identity.device_id)
    assert response["ok"], response
    assert right.show("session", items[1]["id"])["status"] == "forgotten"
    assert not right.search("corrected", "session")
    for source, destination in ((left, right),):
        with source.connection() as db:
            left_rows = db.execute("SELECT op_id,data FROM ops ORDER BY op_id").fetchall()
        with destination.connection() as db:
            right_rows = db.execute("SELECT op_id,data FROM ops ORDER BY op_id").fetchall()
        assert [tuple(row) for row in left_rows] == [tuple(row) for row in right_rows]


def test_scope_filter_reenable_resends_old_rows_without_cursor_loss(devices, monkeypatch):
    a, b = linked(devices, monkeypatch)
    left, right = bank(a), bank(b)
    left.put("team", {"title": "Private team", "body": "Team-only retained evidence"})
    left.put("global", {"title": "Global", "body": "Public install evidence"})
    enable(a, scopes=["global"])
    assert memory.push_to_peer(a, b.identity.device_id)["ok"]
    assert right.search("install", "global")
    assert not right.search("team", "team")
    enable(a)
    assert memory.push_to_peer(a, b.identity.device_id)["ok"]
    assert right.search("team", "team")


def test_iban_shaped_origin_syncs_but_a_secret_body_is_still_refused(tmp_path):
    """The mesh push path is structural-vs-fact, same as a local write.

    R1: an install whose minted device id is IBAN/mod-97-shaped could write
    locally (after the put-path fix) yet still could not sync — the receiver's
    blanket ``guard(payload)`` scanned the STRUCTURAL origin as if it were user
    text. ``receive`` is the handler's push entrypoint, so it is exercised
    directly with a payload built from a real op, not a hand-written stub.
    """
    import hashlib
    import types

    from local_operator.memory import _dump

    iban_shaped = "d_cd474787b1adaa67f8cb5934196c4834"
    sender = MemoryStore(tmp_path / "sender", MemoryIdentity("s", "t", "a"), origin=iban_shaped)
    item = sender.put("global", {"title": "Host", "body": "staging host is stage-2 docs/h.md"})
    with sender.connection() as db:
        op = json.loads(
            db.execute("SELECT data FROM ops WHERE op_id=?", (item["version"],)).fetchone()[0]
        )
    assert op["origin_device"] == iban_shaped

    def payload(operation):
        return {
            "v": 1,
            "generation": 0,
            "scopes": ["global"],
            "after": {iban_shaped: operation["seq"]},
            "ops": [operation],
            "skips": [],
            "proofs": [],
        }

    # A real op whose structural origin is IBAN-shaped now syncs end to end.
    receiver = types.SimpleNamespace(root=tmp_path / "receiver")
    result = memory.receive(receiver, iban_shaped, payload(op), ["global"], {iban_shaped})
    assert result["phase"] == "push" and result["changed"] == 1, result
    landed = MemoryStore(tmp_path / "receiver", MemoryIdentity("s", "t", "a"))
    assert landed.search("stage-2")

    # The structural exemption must not weaken the fact guard: a real secret in a
    # synced body is refused, exactly as an admitted local write is. Built by
    # bypassing the store boundary, so it is NOT an admitted write.
    secret = "passport number AB1234567"
    poisoned = dict(op, entity_id="0123456789abcdef0123456789abcdef")
    poisoned["fact"] = dict(op["fact"], body=secret)
    poisoned["body_sha"] = hashlib.sha256(_dump(poisoned["fact"]).encode()).hexdigest()
    with pytest.raises(MemoryError, match="Sensitive value refused"):
        memory.receive(
            types.SimpleNamespace(root=tmp_path / "receiver2"),
            iban_shaped,
            payload(poisoned),
            ["global"],
            {iban_shaped},
        )
    print("MESH IBAN-shaped origin syncs; real secret in synced body still refused PASS")


def test_old_peer_skips_before_memory_frame(devices, monkeypatch):
    a, b = linked(devices, monkeypatch)
    link = a._ensure_link(b.identity.device_id)
    assert link is not None
    monkeypatch.setattr(
        link, "capabilities", tuple(c for c in link.capabilities if c != wire.MEMORY_SYNC_V1)
    )
    called = []
    monkeypatch.setattr(link, "request", lambda *args, **kwargs: called.append(args))
    result = memory.push_to_peer(a, b.identity.device_id)
    assert result["code"] == "peer_too_old" and not called


def test_nonadmin_receiver_refuses_over_real_link(devices, monkeypatch):
    a, b = linked(devices, monkeypatch, role="drive")
    link = b._ensure_link(a.identity.device_id)
    assert link is not None
    reply = link.request(
        {
            "op": "net_memory",
            "phase": "state",
            "req": b._next_relay_req(),
            "scopes": list(memory.SCOPES),
            "locality": "remote",
        },
        timeout=5,
    )
    assert reply["op"] == "error", reply
    # The shipped authoriser intentionally returns a codeless sentence rather
    # than disclose which guard fired (wire.refusal_frame contract).
    assert isinstance(reply.get("message"), str) and reply["message"], reply
    print("NONADMIN actual receiver refusal:", reply)
    assert memory.push_to_peer(b, a.identity.device_id)["code"] == "skipped:no_admin"
    assert not bank(a).path.exists()


def test_offline_return_restarts_real_relay_same_roots_and_catches_up(devices, monkeypatch):
    """A stopped receiver comes back from its existing disk, not just a fresh store handle.

    The paired keys, network grant and memory vectors survive. Offline writes and forgets
    must converge after the replacement relay returns, including a stale local body on B.
    Every member exchanges its complete permitted operation set with each admin peer;
    this is full-member replication, not an A->B->C forwarding guarantee.
    """
    from local_operator.network import store as network_store

    a, b = linked(devices, monkeypatch)
    left, right = bank(a), bank(b)
    victim = left.put("session", {"title": "Retire locator", "body": "docs/old-runbook.md"})
    assert memory.push_to_peer(a, b.identity.device_id)["code"] == "in_sync"
    assert right.search("old-runbook", "session"), "positive control: B really has the old body"
    records = network_store.list_networks(a.root)
    assert len(records) == 1
    record = records[0]
    b_id, b_root, b_keys = b.identity.device_id, b.root, b.identity
    b.stop()
    fresh = left.put("team", {"title": "Offline note", "body": "docs/offline-catchup.md"})
    left.put("session", {}, entity=victim["id"], expected_version=victim["version"], forget=True)
    offline = memory.push_to_peer(a, b_id)
    assert not offline["ok"], offline
    assert right.show("session", victim["id"])["status"] == "active"
    returned = serve_shaped_relay(
        b_root, monkeypatch, identity=b_keys, audit=audit.AuditLog(b_root)
    )
    try:
        live = _bind_and_start(returned, record)
        _set_peer_endpoints(a, record, b_id, [live])
        result = memory.push_to_peer(a, b_id)
        assert result["code"] == "in_sync", result
        assert bank(returned).show("team", fresh["id"])["fact"]["body"] == "docs/offline-catchup.md"
        assert bank(returned).show("session", victim["id"])["status"] == "forgotten"
        # Reverse exchange and replay cannot resurrect the old put from B's disk.
        assert memory.push_to_peer(returned, a.identity.device_id)["code"] == "in_sync"

        def digest(memory_store):
            import hashlib

            with memory_store.connection() as db:
                rows = [row[0] for row in db.execute("SELECT data FROM ops ORDER BY op_id")]
            return hashlib.sha256("\n".join(rows).encode()).hexdigest()

        assert digest(left) == digest(bank(returned))
        assert not left.search("old-runbook") and not bank(returned).search("old-runbook")
        print(
            "OFFLINE RETURN actual relay stop/new listener same roots; "
            "catchup, equal op digests, no resurrection PASS"
        )
    finally:
        returned.stop()


class _StubMemoryLink:
    """A link that answers memory frames with canned payloads, no socket.

    This is the reviewer/QA shape that exposed the silent refusal: `_push_to_peer`
    talks to the peer only through `link.request`, so a stub is honest here — the
    assertion is about what the RECEIPT keeps, not about the transport.
    """

    def __init__(self, answers: list[dict[str, Any]]) -> None:
        self.capabilities = (wire.MEMORY_SYNC_V1,)
        self.network_id = "n_memory_stub"
        self.answers = list(answers)
        self.frames: list[dict[str, Any]] = []

    def request(self, frame: dict[str, Any], timeout: float = 30) -> dict[str, Any]:
        self.frames.append(frame)
        return self.answers.pop(0)


def _memory_server(root: Path) -> Any:
    from local_operator.network import relay

    return relay.RelayServer(
        root=root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )


def _stub_memory_sync(server: Any, link: _StubMemoryLink, monkeypatch: pytest.MonkeyPatch) -> None:
    from argparse import Namespace

    monkeypatch.setattr(memory, "settings", lambda root: (True, list(memory.SCOPES), []))
    monkeypatch.setattr(server, "_ensure_link", lambda device_id: link)
    monkeypatch.setattr(server, "_next_relay_req", lambda: 1)
    # The push batch is built for the active origins; a one-member namespace is
    # enough and keeps this test off the network store entirely.
    monkeypatch.setattr(
        memory.network_store,
        "load",
        lambda *args, **kwargs: Namespace(active_members=lambda: [Namespace(device_id="device-a")]),
    )


def test_a_peer_refusal_message_reaches_the_receipt_and_status(tmp_path, monkeypatch):
    """Round-2 MAJOR: the actionable sentence was dropped and status claimed success."""
    from local_operator.memory_cli import render_human, run

    store = MemoryStore(tmp_path, MemoryIdentity("s", "", ""), origin="device-a")
    store.put("global", {"title": "Sync proof", "body": "Durable preference for tea"})
    server = _memory_server(tmp_path)
    refusal = "Memory move refused: resolve the conflict on this device, then sync again."
    link = _StubMemoryLink(
        [
            {
                "op": "ack",
                "req": 1,
                "detail": {"v": 1, "generation": 0, "scopes": ["global"], "vector": {}},
            },
            {"op": "error", "req": 1, "code": "bad_memory", "message": refusal},
        ]
    )
    _stub_memory_sync(server, link, monkeypatch)
    device = "d_" + "a" * 32

    result = memory.push_to_peer(server, device)
    assert result == {
        "ok": False,
        "code": "bad_memory",
        "message": refusal,
        "batches": 0,
    }
    receipt = memory.peer_status(tmp_path)[0]
    assert receipt["code"] == "bad_memory"
    assert receipt["message"] == refusal

    text, failed = run("status", config=tmp_path, identity=MemoryIdentity("s", "", ""))
    assert not failed, text
    human = render_human(text, failed)
    assert "Peers synced: 0" in human
    assert "1 failing" in human
    assert device in human and "bad_memory" in human
    # The sentence is wrapped at the row bar, so the assertion is per clause.
    assert "Memory move" in human
    assert "resolve the conflict on this device" in human

    # A healthy answer clears the stale sentence: a recovered peer must not keep
    # showing the failure that was just resolved.
    monkeypatch.setattr(
        memory, "_push_to_peer", lambda server, device_id: {"ok": True, "code": "in_sync"}
    )
    assert memory.push_to_peer(server, device)["code"] == "in_sync"
    recovered = memory.peer_status(tmp_path)[0]
    assert recovered["message"] == ""
    text, failed = run("status", config=tmp_path, identity=MemoryIdentity("s", "", ""))
    assert "Peers synced: 1" in render_human(text, failed)
    assert "Not syncing with" not in render_human(text, failed)


def test_bad_memory_is_a_policy_refusal_not_a_fast_retry():
    """The refusal repeats until the conflict is resolved; 15 s retries only flood logs."""
    from local_operator.network import definitions

    assert "bad_memory" in definitions.POLICY_REFUSAL_CODES
