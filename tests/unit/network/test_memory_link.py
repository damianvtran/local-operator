"""Native relay/PeerLink sockets, two isolated homes, no live mesh mutation."""

import json

import pytest

from local_operator.memory import MemoryError, MemoryIdentity, MemoryStore
from local_operator.network import audit, identity, memory, wire
from tests.unit.network.test_mcpdefs_link import _bind_and_start, _set_peer_endpoints
from tests.unit.network.test_relay_e2e import _pair, serve_shaped_relay


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
