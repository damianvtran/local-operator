"""Logical removal of legacy display-name keys; never a forensic-erasure claim.

Fixtures create disposable databases only. A name-derived operational digest
preserves each old queue identity, not anonymization or forensic erasure. No
aliases collapse even when their payloads match; true target conflicts roll back.
"""

import hashlib
import sqlite3

import pytest

from local_operator.team_bridge.store import (
    OwnerReceipt,
    Store,
    StoreRefused,
    state_db_path,
)
from local_operator.team_bridge.types import OwnerSnapshot, feed_seq

LEGACY = (
    "message_available:conversation_id=bcv_abcdefgh/bcmsg_abcdef01/1/1/"
    "PRIVATE-CHANNEL/PRIVATE-SENDER/human"
)
CANONICAL = "message_available:conversation_id=bcv_abcdefgh/bcmsg_abcdef01/1/1/human"


def alias(raw):
    # Independent expectation: digest the EXACT full persisted old key,
    # including its prefix and percent-encoding spelling, not decoded labels.
    parts = raw.split("/")
    return (
        "/".join(parts[:4] + [parts[6]])
        + "/legacy_alias="
        + hashlib.sha256(raw.encode("utf-8")).hexdigest()
    )


def legacy_store(root):
    owner = OwnerSnapshot(
        credential_id=1,
        account_id="acc_a",
        tenant_id="tnt_a",
        endpoint_id="bep_a",
        canonical_base_url="https://hub.example",
        enrollment_generation=1,
    ).validated()
    Store(root).commit_enrollment(owner, OwnerReceipt("bridge_consent_v1", "receipt"))
    # Fixture-only raw inserts model a v6 file before the new code is loaded.
    with sqlite3.connect(state_db_path(root)) as conn:
        key = conn.execute("SELECT owner_key FROM enrollments").fetchone()[0]
        conn.execute("UPDATE schema_meta SET value='6' WHERE key='schema_version'")
        conn.execute("UPDATE sync SET feed_seq=17, pending_resync=1, inbox_checkpoint='checkpoint'")
        conn.execute("INSERT INTO feed_refs VALUES (?,?,?)", (key, LEGACY, 11))
        conn.execute(
            "INSERT INTO screening_holds VALUES (?,?,?,?,?)",
            (key, LEGACY, "provider_error", 900, 12),
        )
        conn.execute(
            "INSERT INTO claims VALUES (?,?,?,?,?,?,?)",
            (key, "bcmsg_abcdef01", "cos", "claim", 1, 999, "held"),
        )
    return owner, key


def migrate(root):
    # active_owner intentionally swallows StoreRefused (no authority). Exercise
    # the write-capable schema door to assert the exact refusal and rollback.
    store = Store(root)
    with store._connect() as conn:
        store._init(conn)


def snapshot(root):
    with sqlite3.connect(state_db_path(root)) as conn:
        return {
            table: conn.execute(f"SELECT * FROM {table} ORDER BY 1, 2").fetchall()
            for table in (
                "enrollments",
                "sync",
                "claims",
                "feed_refs",
                "screening_holds",
                "schema_meta",
            )
        }


def test_legacy_keys_migrate_atomically_without_touching_owner_cursor_or_claim(tmp_path):
    owner, _ = legacy_store(tmp_path)
    before = snapshot(tmp_path)
    assert Store(tmp_path).active_owner() == owner
    after = snapshot(tmp_path)
    for table in ("enrollments", "sync", "claims"):
        assert after[table] == before[table]
    assert after["feed_refs"][0][1:] == (alias(LEGACY), 11)
    assert after["screening_holds"][0][1:] == (alias(LEGACY), "provider_error", 900, 12)
    assert "PRIVATE-" not in repr(after)
    # Restart is a no-op; no subsequent call recreates the old spelling.
    assert Store(tmp_path).active_owner() == owner
    assert snapshot(tmp_path) == after


@pytest.mark.parametrize("other_key", [CANONICAL, LEGACY.replace("PRIVATE-SENDER", "OTHER-SENDER")])
@pytest.mark.parametrize("different_state", [False, True])
def test_every_alias_preserves_its_own_pending_state(tmp_path, other_key, different_state):
    owner, key = legacy_store(tmp_path)
    seq = 16 if different_state else 11
    code = "policy_stale" if different_state else "provider_error"
    retry = None if different_state else 900
    created = 20 if different_state else 12
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        conn.execute("INSERT INTO feed_refs VALUES (?,?,?)", (key, other_key, seq))
        conn.execute(
            "INSERT INTO screening_holds VALUES (?,?,?,?,?)", (key, other_key, code, retry, created)
        )
    assert Store(tmp_path).active_owner() == owner
    after = snapshot(tmp_path)
    other_target = CANONICAL if other_key == CANONICAL else alias(other_key)
    assert set(after["feed_refs"]) == {(key, alias(LEGACY), 11), (key, other_target, seq)}
    assert set(after["screening_holds"]) == {
        (key, alias(LEGACY), "provider_error", 900, 12),
        (key, other_target, code, retry, created),
    }
    assert "PRIVATE-" not in repr(after) and "OTHER-SENDER" not in repr(after)


@pytest.mark.parametrize("table", ["feed_refs", "screening_holds"])
def test_true_alias_target_collision_rolls_back_even_when_state_matches(tmp_path, table):
    _, key = legacy_store(tmp_path)
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        if table == "feed_refs":
            conn.execute("INSERT INTO feed_refs VALUES (?,?,?)", (key, alias(LEGACY), 11))
        else:
            conn.execute(
                "INSERT INTO screening_holds VALUES (?,?,?,?,?)",
                (key, alias(LEGACY), "provider_error", 900, 12),
            )
    before = snapshot(tmp_path)
    with pytest.raises(StoreRefused) as error:
        migrate(tmp_path)
    assert error.value.code == "ref_migration_conflict"
    assert "PRIVATE-" not in str(error.value)
    assert snapshot(tmp_path) == before


def test_rekey_failure_rolls_back_both_tables_and_version(tmp_path):
    legacy_store(tmp_path)
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        conn.execute(
            "CREATE TRIGGER reject_hold_rekey BEFORE UPDATE OF message_id ON screening_holds "
            "BEGIN SELECT RAISE(ABORT, 'fixture failure'); END"
        )
    before = snapshot(tmp_path)
    with pytest.raises(StoreRefused):
        migrate(tmp_path)
    assert snapshot(tmp_path) == before


def test_readonly_access_never_runs_the_migration(tmp_path):
    owner, _ = legacy_store(tmp_path)
    before = snapshot(tmp_path)
    assert Store(tmp_path).active_owner_readonly() == owner
    assert snapshot(tmp_path) == before
    assert "PRIVATE-" in repr(before)  # no claim that a read cleaned the file


def test_more_than_one_batch_and_other_owner_keys_survive(tmp_path):
    _, key = legacy_store(tmp_path)
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        for number in range(260):
            raw = LEGACY.replace("bcmsg_abcdef01", f"bcmsg_{number:08d}")
            conn.execute("INSERT INTO feed_refs VALUES (?,?,?)", (key, raw, number))
        # Same raw locator under another owner is not a collision at all.
        conn.execute("INSERT INTO feed_refs VALUES (?,?,?)", ("other-owner", LEGACY, 50))
    before = snapshot(tmp_path)
    migrate(tmp_path)
    after = snapshot(tmp_path)
    assert len(after["feed_refs"]) == len(before["feed_refs"])
    assert ("other-owner", alias(LEGACY), 50) in after["feed_refs"]
    assert "PRIVATE-" not in repr(after)
    for table in ("enrollments", "sync", "claims"):
        assert after[table] == before[table]


def test_malformed_legacy_key_refuses_without_claiming_clean_stamp(tmp_path):
    _, key = legacy_store(tmp_path)
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        conn.execute("INSERT INTO feed_refs VALUES (?,?,?)", (key, LEGACY + "/extra", 12))
    before = snapshot(tmp_path)
    with pytest.raises(StoreRefused) as error:
        migrate(tmp_path)
    assert error.value.code == "ref_migration_malformed"
    assert snapshot(tmp_path) == before


def test_old_in_memory_keys_cannot_reintroduce_names_after_upgrade(tmp_path):
    owner, _ = legacy_store(tmp_path)
    migrate(tmp_path)
    store = Store(tmp_path)
    store.record_feed_batch(owner, feed_seq(17), [LEGACY], resync_required=False)
    store.persist_hold(owner, LEGACY, "provider_error", retry_at=901)
    after = snapshot(tmp_path)
    assert "PRIVATE-" not in repr(after)
    assert len(after["feed_refs"]) == len(after["screening_holds"]) == 1


def test_nonpositive_rowids_are_not_skipped(tmp_path):
    legacy_store(tmp_path)
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        conn.execute("UPDATE feed_refs SET rowid=-1")
        conn.execute("UPDATE screening_holds SET rowid=0")
    migrate(tmp_path)
    after = snapshot(tmp_path)
    assert "PRIVATE-" not in repr(after)
    assert len(after["feed_refs"]) == len(after["screening_holds"]) == 1


def test_inflight_old_key_consumption_leaves_canonical_and_other_alias_untouched(tmp_path):
    owner, key = legacy_store(tmp_path)
    sibling = LEGACY.replace("PRIVATE-SENDER", "SIBLING-SENDER")
    with sqlite3.connect(state_db_path(tmp_path)) as conn:
        for raw in (CANONICAL, sibling):
            conn.execute("INSERT INTO feed_refs VALUES (?,?,?)", (key, raw, 11))
            conn.execute(
                "INSERT INTO screening_holds VALUES (?,?,?,?,?)",
                (key, raw, "open_chief_of_staff", None, 25),
            )
    migrate(tmp_path)
    store = Store(tmp_path)
    assert store.advance_hold_retry(owner, LEGACY, retry_at=999)
    assert store.screening_hold(owner, LEGACY) == {
        "hold_code": "provider_error",
        "retry_at": 999,
        "created_at": 12,
    }
    canonical_hold = store.screening_hold(owner, CANONICAL)
    sibling_hold = store.screening_hold(owner, sibling)
    assert canonical_hold is not None and canonical_hold["retry_at"] is None
    assert sibling_hold is not None and sibling_hold["retry_at"] is None
    store.delete_hold(owner, LEGACY)
    assert store.delete_feed_refs(owner, [LEGACY]) == 1
    after = snapshot(tmp_path)
    assert {row[1] for row in after["feed_refs"]} == {CANONICAL, alias(sibling)}
    assert {row[1] for row in after["screening_holds"]} == {CANONICAL, alias(sibling)}
    assert Store(tmp_path).active_owner() == owner
    assert snapshot(tmp_path) == after


def test_readonly_old_key_resolves_its_exact_row_before_and_after_migration(tmp_path):
    owner, _ = legacy_store(tmp_path)
    store = Store(tmp_path)
    before = snapshot(tmp_path)
    hold = store.screening_hold_readonly(owner, LEGACY)
    assert hold == {"hold_code": "provider_error", "retry_at": 900, "created_at": 12}
    assert snapshot(tmp_path) == before
    migrate(tmp_path)
    after = snapshot(tmp_path)
    assert store.screening_hold_readonly(owner, LEGACY) == hold
    assert snapshot(tmp_path) == after


def test_v6_version_guard_refuses_v7_without_downgrading_rows(tmp_path, monkeypatch):
    from local_operator.team_bridge import store as store_mod

    legacy_store(tmp_path)
    migrate(tmp_path)
    before = snapshot(tmp_path)
    # Emulate the old version ceiling at the existing layout validator. An
    # older client must refuse, not rewrite an alias key it cannot understand.
    monkeypatch.setattr(store_mod, "SCHEMA_VERSION", 6)
    with pytest.raises(StoreRefused) as error:
        migrate(tmp_path)
    assert error.value.code == "schema_unsupported"
    assert Store(tmp_path).active_owner() is None
    assert snapshot(tmp_path) == before


def test_alias_round_trip_hashes_exact_utf8_full_key_and_new_refs_stay_unchanged():
    from local_operator.team_bridge.types import parse_conversation_ref_value

    keys = [
        LEGACY,
        LEGACY.replace("PRIVATE-SENDER", "Jos\u00e9"),
        LEGACY.replace("PRIVATE-SENDER", "Jose\u0301"),
        LEGACY.replace("PRIVATE-SENDER", "%2f"),
        LEGACY.replace("PRIVATE-SENDER", "%2F"),
    ]
    aliases = set()
    for raw in keys:
        parsed = parse_conversation_ref_value(raw.split("=", 1)[1])
        assert parsed is not None
        assert parsed.encode() == alias(raw)
        assert parsed.legacy_alias == hashlib.sha256(raw.encode("utf-8")).hexdigest()
        assert parse_conversation_ref_value(parsed.encode().split("=", 1)[1]) == parsed
        assert parsed.message_id == "bcmsg_abcdef01"
        aliases.add(parsed.legacy_alias)
    assert len(aliases) == len(keys)
    for raw in (CANONICAL, CANONICAL.rsplit("/", 1)[0]):
        parsed = parse_conversation_ref_value(raw.split("=", 1)[1])
        assert parsed is not None and parsed.legacy_alias is None
        assert parsed.encode() == raw


@pytest.mark.parametrize(
    "slot",
    [
        "legacy_alias=",
        "legacy_alias=" + "a" * 63,
        "legacy_alias=" + "a" * 65,
        "legacy_alias=" + "A" * 64,
        "legacy_alias=" + "g" * 64,
        "legacy_alias=" + "a" * 64 + "\n",
        "legacy_alias=" + "a" * 64 + "/extra",
        "legacy_alias=" + "a" * 64 + "/human",
    ],
)
def test_malformed_alias_slot_never_becomes_a_pending_ref(slot):
    from local_operator.team_bridge.types import parse_conversation_ref_value

    assert parse_conversation_ref_value(CANONICAL.split("=", 1)[1] + "/" + slot) is None


def test_server_cannot_mint_migration_aliases_in_new_wire_events():
    from local_operator.team_bridge.types import (
        parse_conversation_ref_value,
        project_events_batch,
    )

    _, refs, _ = project_events_batch(
        {
            "cursor": "1",
            "events": [
                {
                    "type": "message_available",
                    "at": "2026-10-23T00:00:00Z",
                    "ids": {
                        "conversation_id": "bcv_abcdefgh",
                        "message_id": "bcmsg_abcdef01",
                        "conversation_seq": 1,
                        "mention": True,
                        "sender_kind": "human",
                        "legacy_alias": "a" * 64,
                    },
                }
            ],
        }
    )
    assert refs == (CANONICAL,)
    parsed = parse_conversation_ref_value(refs[0].split("=", 1)[1])
    assert parsed is not None and parsed.legacy_alias is None


def test_unencodable_legacy_key_is_refused_without_guessing_a_digest():
    from local_operator.team_bridge.types import parse_conversation_ref_value

    raw = LEGACY.replace("PRIVATE-SENDER", "invalid-\ud800")
    assert parse_conversation_ref_value(raw.split("=", 1)[1]) is None


def test_a_thread_root_rides_as_a_tag_beside_the_existing_parts() -> None:
    """The reply's root is a TAGGED slot, so no existing part changes meaning.

    A 5-form ref and a 5-form ref plus ``/thread_root=`` differ by exactly one
    named tag; neither the 6-part alias form nor the old 7-part name layout is
    touched, which is what makes the addition backwards-compatible.
    """

    from local_operator.team_bridge.types import parse_conversation_ref_value

    root = "bcmsg_root0001"
    base = "bcv_abcdefgh/bcmsg_abcdef01/1/1/human"

    plain = parse_conversation_ref_value(base)
    assert plain is not None and plain.thread_root_id is None
    assert plain.encode().endswith(base)

    reply = parse_conversation_ref_value(base + f"/thread_root={root}")
    assert reply is not None
    assert reply.thread_root_id == root
    assert reply.sender_kind == "human"
    assert reply.encode().endswith(base + f"/thread_root={root}")


def test_an_unknown_tag_parses_and_is_ignored() -> None:
    """Extensibility is the point: a tag this release does not know decodes.

    Dropping the ref would drop a durable queue row just because a later
    writer added a field. The tag is never read as a positional slot, and it is
    not preserved on re-encode (it is the writer's field, not ours).
    """

    from local_operator.team_bridge.types import parse_conversation_ref_value

    base = "bcv_abcdefgh/bcmsg_abcdef01/1/1/human"
    parsed = parse_conversation_ref_value(base + "/future_tag=whatever")

    assert parsed is not None
    assert parsed.sender_kind == "human"
    assert parsed.encode().endswith(base)


def test_a_thread_root_never_rides_with_a_migration_alias() -> None:
    """The grammar stays narrow: no tag combination is legal just because it
    parses. A reply has no migration alias, and an alias predates threads, so
    a ref carrying both is refused rather than half-understood."""

    from local_operator.team_bridge.types import parse_conversation_ref_value

    base = "bcv_abcdefgh/bcmsg_abcdef01/1/1/human"
    alias = "a" * 64
    assert (
        parse_conversation_ref_value(base + f"/legacy_alias={alias}/thread_root=bcmsg_root0001")
        is None
    )
    assert parse_conversation_ref_value(base + f"/thread_root=bcmsg_root0001") is not None


def test_a_positional_after_a_tag_is_still_malformed() -> None:
    """The one shape tags must not open up: order-dependent meaning."""

    from local_operator.team_bridge.types import parse_conversation_ref_value

    base = "bcv_abcdefgh/bcmsg_abcdef01/1/1"
    assert parse_conversation_ref_value(base + "/thread_root=bcmsg_root0001/human") is None
