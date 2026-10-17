"""Read-only memory inspection: redaction, preview-only dispositions, paging, identity.

Seeding note: work-shaped GLOBAL rows and a guard-refused row cannot be created
through ``MemoryStore.put`` today — the write path refuses both. They are exactly
the legacy/imported state this inspector exists to show, so the tests insert
those rows directly into the store's own tables, then prove the inspector both
shows them safely and changes nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.compaction.tokens import count_text_tokens
from local_operator.memory import MemoryError, MemoryIdentity, MemoryStore, guard
from local_operator.memory_cli import render_human, run
from local_operator.memory_inspect import PREVIEW_CHARS, SCHEMA, inspect_entries

SENSITIVE_TITLE = "Card number 4111 1111 1111 1111"
SENSITIVE_BODY = "Backup credentials: SSN 123-45-6789"


@pytest.fixture
def rig(tmp_path):
    root = tmp_path / "home" / ".local-operator"
    identity = MemoryIdentity("sess-inspect", "team-inspect", "agent-inspect")
    store = MemoryStore(root, identity, root / "sessions" / "sess-inspect", origin="test-device")
    return root, identity, store


def _seed_legacy_row(
    store: MemoryStore,
    scope: str,
    owner: str,
    *,
    title: str,
    body: str,
    source: dict[str, Any] | None = None,
    status: str = "active",
) -> str:
    """Insert one row the way an older build/import could have left it.

    Deliberately bypasses ``validate_fact``/``guard``: the guard-refused and
    work-shaped rows under test are unrepresentable through today's write path.
    """
    entity = hashlib.sha256(f"{scope}-{title}-{time.time_ns()}".encode()).hexdigest()[:26]
    op_id = hashlib.sha256(f"op-{entity}".encode()).hexdigest()[:26]
    fact = {
        "title": title,
        "body": body,
        "tags": [],
        "aliases": [],
        "source": source or {"kind": "reported", "author": "main"},
        "status": status,
        "verified_at": None,
        "expires_at": None,
        "pinned": False,
        "subject": "",
        "created_at": time.time() - 90,
        "source_time": None,
        "replaces": "",
    }
    wire = json.dumps(fact, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    op = {
        "v": 1,
        "op_id": op_id,
        "entity_id": entity,
        "origin_device": "legacy-device",
        "seq": int(time.time() * 1000),
        "hlc": [int(time.time() * 1000), 0, "legacy-device"],
        "parents": [],
        "kind": "put",
        "scope": scope,
        "owner_key": owner,
        "fact": fact,
        "body_sha": hashlib.sha256(wire.encode()).hexdigest(),
    }
    if not store.path.exists():
        # Initialise the schema through the store's own connection once.
        with store.connection():
            pass
    db = sqlite3.connect(store.path)
    try:
        db.execute(
            "INSERT INTO ops VALUES (?,?,?,?,?,?,?,?)",
            (
                op["op_id"],
                entity,
                op["origin_device"],
                op["seq"],
                scope,
                owner,
                "put",
                wire,
            ),
        )
        db.execute(
            "INSERT INTO current VALUES (?,?,?,?,?,?)",
            (entity, scope, owner, op_id, wire, count_text_tokens(wire)),
        )
        db.commit()
    finally:
        # Close explicitly: a lingering WAL connection changes the file set the
        # read-only identity test compares.
        db.close()
    return entity


def _snapshot(store: MemoryStore) -> dict[str, Any]:
    """Every row that a mutation would move: op log, current heads, metadata."""
    db = sqlite3.connect(f"{store.path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        return {
            "ops": [
                tuple(row)
                for row in db.execute(
                    "SELECT op_id,entity_id,origin,seq,scope,owner,kind,data FROM ops "
                    "ORDER BY op_id"
                )
            ],
            "current": [
                tuple(row)
                for row in db.execute(
                    "SELECT entity_id,scope,owner,op_id,fact,tokens FROM current ORDER BY entity_id"
                )
            ],
            "meta": [tuple(row) for row in db.execute("SELECT * FROM meta ORDER BY key")],
        }
    finally:
        db.close()


def _files(root: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "memory").iterdir())
        if path.is_file()
    }


def _seed_all_scopes(store: MemoryStore, identity: MemoryIdentity) -> dict[str, str]:
    ids = {}
    ids["global"] = store.put(
        "global",
        {
            "title": "Colleague Elara lives in Dunedin",
            "body": "Elara lives in Dunedin, New Zealand.",
            "source": {"kind": "reported", "author": "main"},
        },
    )["id"]
    ids["session"] = store.put(
        "session",
        {"title": "Deploy checkpoint", "body": "Smoke run passed on staging."},
    )["id"]
    ids["team"] = store.put(
        "team",
        {"title": "Team standup time", "body": "Standup is 09:30 on weekdays."},
    )["id"]
    ids["agent"] = store.put(
        "agent",
        {"title": "Agent profile", "body": "Prefers terse summaries."},
    )["id"]
    return ids


def test_lists_every_authorized_scope_with_provenance_and_age(rig):
    root, identity, store = rig
    ids = _seed_all_scopes(store, identity)

    report = inspect_entries(root, identity)

    assert report["schema"] == SCHEMA
    assert report["read_only"] is True
    assert report["counts"]["total"] == 4
    assert {entry["scope"] for entry in report["entries"]} == {
        "global",
        "session",
        "team",
        "agent",
    }
    assert {entry["id"] for entry in report["entries"]} == set(ids.values())
    for entry in report["entries"]:
        assert entry["origin"] == "stated"
        assert entry["status"] == "active"
        assert isinstance(entry["age_seconds"], float) and entry["age_seconds"] >= 0
        assert entry["provenance"]["kind"] == "reported"
        assert entry["provenance"]["session"] == "sess-inspect"
        assert entry["revisions"] == 1
        assert entry["disposition"]["applied"] is False
    assert report["has_more"] is False and report["next_cursor"] is None


def test_guard_refused_title_and_body_are_withheld_without_echo(rig):
    root, identity, store = rig
    # Sanity: these really are values the shared guard refuses.
    for value in (SENSITIVE_TITLE, SENSITIVE_BODY):
        with pytest.raises(MemoryError):
            guard(value)
    entity = _seed_legacy_row(store, "global", "global", title=SENSITIVE_TITLE, body=SENSITIVE_BODY)

    report = inspect_entries(root, identity)
    entry = next(item for item in report["entries"] if item["id"] == entity)

    assert entry["redacted"] is True
    assert entry["title"] is None and entry["title_exported"] is False
    assert entry["body_preview"] is None and entry["body_exported"] is False
    # The safe header survives, so a human can still find and judge the row.
    assert entry["scope"] == "global" and entry["status"] == "active"
    # Nothing refused may leak through a different field of the payload.
    rendered = json.dumps(report)
    assert "4111 1111 1111 1111" not in rendered
    assert "123-45-6789" not in rendered
    assert "hunter" not in rendered


def test_exportable_body_is_a_bounded_preview_not_the_full_body(rig):
    root, identity, store = rig
    body = "Dunedin fact " + ("x" * 400)
    store.put("global", {"title": "Long fact", "body": body, "source": {"kind": "reported"}})

    report = inspect_entries(root, identity)
    entry = report["entries"][0]

    assert entry["body_exported"] is False
    assert entry["body_preview"] == body[:PREVIEW_CHARS]
    assert entry["body_truncated"] is True


def test_preview_never_applies_and_the_store_is_untouched(rig):
    root, identity, store = rig
    _seed_all_scopes(store, identity)
    _seed_legacy_row(
        store,
        "global",
        "global",
        title="PR #2463 is merged; CI green",
        body="Next step: cut the release window.",
        source={"kind": "reported", "author": "main", "session": "sess-inspect"},
    )

    before_rows = _snapshot(store)
    before_files = _files(root)

    # Both entry points a human/agent would use; neither may write.
    raw, failed = run("inspect", config=root, identity=identity, prefix="/memory")
    assert failed is False
    payload = json.loads(raw)
    render_human(raw, prefix="/memory")
    inspect_entries(root, identity, limit=50)

    after_rows = _snapshot(store)
    after_files = _files(root)

    assert after_rows == before_rows  # ops/current/meta byte-for-byte equal
    assert after_files == before_files  # no rewrite, no sidecar left behind
    # IDs and revisions are the fields a migration would move.
    assert [row[:4] for row in after_rows["current"]] == [row[:4] for row in before_rows["current"]]
    assert payload["read_only"] is True
    assert all(item["disposition"]["applied"] is False for item in payload["entries"])
    assert "proposal" in payload["entries"][0]["disposition"]


def test_paging_walks_more_than_one_page_without_gaps_or_duplicates(rig):
    root, identity, store = rig
    expected = {
        store.put("global", {"title": f"Durable fact {n}", "body": f"Body {n}."})["id"]
        for n in range(5)
    }
    seen: list[str] = []
    cursor = None
    pages = 0
    while True:
        report = inspect_entries(root, identity, limit=2, cursor=cursor)
        seen.extend(entry["id"] for entry in report["entries"])
        pages += 1
        if not report["has_more"]:
            assert report["next_cursor"] is None
            break
        cursor = report["next_cursor"]
        assert cursor
        assert pages < 10
    assert pages == 3
    assert len(seen) == len(set(seen)) == 5
    assert set(seen) == expected


def test_dispositions_are_preview_only_and_scope_aware(tmp_path):
    root = tmp_path / "home" / ".local-operator"
    identity = MemoryIdentity("sess-inspect", "team-inspect", "agent-inspect")
    store = MemoryStore(root, identity, origin="test-device")
    store.put("global", {"title": "Colleague Elara", "body": "Lives in Dunedin."})
    store.put("session", {"title": "Sprint checkpoint", "body": "Two tasks left."})
    _seed_legacy_row(
        store,
        "global",
        "global",
        title="PR #2463 is merged; CI green",
        body="Next step: release.",
        source={"kind": "reported", "author": "main", "session": "sess-inspect"},
    )
    _seed_legacy_row(
        store, "global", "global", title="HOLD pending review on PR #12", body="Blocked."
    )

    report = inspect_entries(root, identity)
    by_title = {entry["title"]: entry for entry in report["entries"]}

    keep = by_title["Colleague Elara"]["disposition"]
    assert keep["proposal"] == "keep" and keep["applied"] is False

    assert by_title["Sprint checkpoint"]["disposition"]["proposal"] == "keep"

    scoped = by_title["PR #2463 is merged; CI green"]
    assert scoped["scope_fit_candidate"] is True
    assert "tracker-ref" in scoped["work_markers"]
    assert scoped["disposition"]["proposal"] == "session-scoped"
    assert scoped["disposition"]["destination"] == "session/sess-inspect"
    assert "NOT" not in scoped["disposition"]["reason"]  # the renderer says NOT applied
    assert scoped["disposition"]["applied"] is False

    no_session = by_title["HOLD pending review on PR #12"]
    assert no_session["disposition"]["proposal"] == "team"
    assert no_session["disposition"]["destination"] == "team/team-inspect"

    # Without a bound team the same row needs a human choice, not a guess.
    solo = MemoryIdentity("sess-inspect", "", "agent-inspect")
    solo_report = inspect_entries(root, solo)
    unbound = next(entry for entry in solo_report["entries"] if entry["id"] == no_session["id"])
    assert unbound["disposition"]["proposal"] == "review-work-scope"


def test_missing_store_is_empty_and_not_created(tmp_path):
    root = tmp_path / "home" / ".local-operator"
    identity = MemoryIdentity("sess-inspect", "", "")

    report = inspect_entries(root, identity)

    assert report["store"]["exists"] is False
    assert report["entries"] == [] and report["counts"]["total"] == 0
    assert not (root / "memory").exists()


def test_cli_json_is_parseable_and_human_view_states_nothing_changed(tmp_path, monkeypatch):
    root = tmp_path / "home" / ".local-operator"
    home = root.parent
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    identity = MemoryIdentity("cli-inspect", "", "")
    store = MemoryStore(root, identity, origin="cli-device")
    store.put("global", {"title": "CLI inspect proof", "body": "A short durable fact."})
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_"))}

    def cli(*args):
        response = subprocess.run(
            [sys.executable, "-m", "local_operator.cli", "memory", *args],
            cwd=home,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert response.returncode == 0, (response.returncode, response.stderr)
        return response.stdout

    payload = json.loads(cli("inspect", "--json", "--session", "cli-inspect"))
    assert payload["schema"] == SCHEMA and payload["read_only"] is True
    assert payload["entries"][0]["reference"] == "global/" + payload["entries"][0]["id"]
    assert set(payload) >= {
        "schema",
        "read_only",
        "store",
        "generated_at",
        "authorized_scopes",
        "counts",
        "entries",
        "has_more",
        "next_cursor",
    }

    human = cli("inspect", "--session", "cli-inspect")
    assert "Read-only preview" in human
    assert "No change was made" in human
    assert "NOT applied" in human
    assert '"entries"' not in human


def test_bad_limit_is_refused(rig):
    root, identity, _ = rig
    with pytest.raises(MemoryError):
        inspect_entries(root, identity, limit=0)
    with pytest.raises(MemoryError):
        inspect_entries(root, identity, limit=51)
