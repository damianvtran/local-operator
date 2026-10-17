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
from local_operator.memory import (
    MemoryError,
    MemoryIdentity,
    MemoryStore,
    guard,
    validate_fact,
)
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
    created_at: float | None = None,
) -> str:
    """Insert one hand-built row with arbitrary bytes in the fact columns.

    Deliberately bypasses ``validate_fact``/``guard``: the guard-refused and
    work-shaped rows under test are unrepresentable through today's write path,
    and this surface's premise is arbitrary stored state (an import, a restored
    backup).

    The wire shapes still match production: ``ops.data`` carries the OP
    ENVELOPE (``_dump(op)``, the shape ``MemoryStore._ops`` decodes and
    ``show`` reads ``scope``/``owner_key`` from) and ``current.fact`` carries
    the fact. An earlier version of this helper put the FACT in ``ops.data``,
    which is not a shape any build wrote; ``lop memory show`` then failed with
    ``KeyError: 'scope'`` on rows minted here — a fixture defect, not a
    reachable bug (see the PR thread).
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
        "created_at": time.time() - 90 if created_at is None else created_at,
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
                json.dumps(op, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
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


# --- Round 1 remediation: the opaque write-path rule, provenance locators,
# --- the live-writer read branch, corrupt rows, and the desktop registry.


#: A secret-shaped title and base64 body: the plain guard accepts both, the
#: write path's opaque pass refuses both. This is the rule MAJOR 1 was about.
OPAQUE_TITLE = "Staging deploy key 9f2c7b1e4a5d6f8091a2b3c4d5e6f708192a3b4c7d5e"
OPAQUE_BODY = "copied from the console: aB3dE6gH9jK2mN5pQ8sT1vW4yZ7bC0fI3lO6rU9x"


def test_opaque_rule_withholds_what_the_write_path_refuses(rig):
    """title/body use the SAME guard pass as ``validate_fact``, not the plain one."""
    root, identity, store = rig
    for value in (OPAQUE_TITLE, OPAQUE_BODY):
        guard(value)  # the plain pass genuinely accepts these ...
        with pytest.raises(MemoryError):
            guard(value, opaque=True)  # ... and the write path's pass refuses them
    with pytest.raises(MemoryError):
        validate_fact({"title": OPAQUE_TITLE, "body": "safe body"})
    with pytest.raises(MemoryError):
        validate_fact({"title": "safe title", "body": OPAQUE_BODY})

    entity = _seed_legacy_row(store, "global", "global", title=OPAQUE_TITLE, body=OPAQUE_BODY)

    report = inspect_entries(root, identity)
    entry = next(item for item in report["entries"] if item["id"] == entity)
    assert entry["redacted"] is True
    assert entry["title"] is None and entry["title_exported"] is False
    assert entry["body_preview"] is None and entry["body_exported"] is False
    assert OPAQUE_TITLE not in json.dumps(report)
    assert OPAQUE_BODY not in json.dumps(report)

    raw, failed = run("inspect", config=root, identity=identity, prefix="lop memory")
    assert failed is False
    human = render_human(raw, prefix="lop memory")
    assert OPAQUE_TITLE not in human and OPAQUE_BODY not in human
    assert "title withheld" in human and "Preview withheld" in human
    # D4: a withheld row's proposal must not assert a judgement about content
    # the page is refusing to show.
    assert "No work/lane/progress marker" not in human
    assert "Content withheld by the privacy guard" in human


def test_sensitive_provenance_locator_is_withheld_and_disclosed(rig):
    """``provenance_redacted`` is a sibling of the dict, and the human page says so."""
    root, identity, store = rig
    uri = "file:///Users/damian/.ssh/id_ed25519"
    with pytest.raises(MemoryError):
        guard(uri)
    _seed_legacy_row(
        store,
        "global",
        "global",
        title="Config pointer",
        body="Read it from the key file.",
        source={"kind": "reported", "author": "main", "uri": uri},
    )

    report = inspect_entries(root, identity)
    entry = report["entries"][0]
    assert entry["provenance_redacted"] is True
    assert "uri" not in entry["provenance"]
    assert uri not in json.dumps(report)

    raw, failed = run("inspect", config=root, identity=identity, prefix="lop memory")
    assert failed is False
    human = render_human(raw, prefix="lop memory")
    assert uri not in human
    assert "locator withheld by the privacy guard" in human


def test_live_writer_sidecars_take_the_non_immutable_read_path(rig):
    """With ``-wal``/``-shm`` present the read MUST replay committed WAL frames.

    ``immutable=1`` would miss them; this proves the mode=ro branch is what
    runs, and that reading it changes no durable byte.
    """
    root, identity, store = rig
    _seed_all_scopes(store, identity)
    entity = hashlib.sha256(b"wal-only-row").hexdigest()[:26]
    fact = json.dumps(
        {
            "title": "WAL-only fact",
            "body": "Committed, not yet checkpointed.",
            "tags": [],
            "aliases": [],
            "source": {"kind": "reported", "author": "main"},
            "status": "active",
            "verified_at": None,
            "expires_at": None,
            "pinned": False,
            "subject": "",
            "created_at": int(time.time()),
            "source_time": None,
            "replaces": "",
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    writer = sqlite3.connect(store.path)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        writer.execute(
            "INSERT INTO current VALUES (?,?,?,?,?,?)",
            (entity, "global", "global", "wal-op-id", fact, count_text_tokens(fact)),
        )
        writer.commit()
        assert Path(str(store.path) + "-wal").exists()
        assert Path(str(store.path) + "-shm").exists()
        before = _files(root)

        report = inspect_entries(root, identity)

        assert {entry["id"] for entry in report["entries"]} >= {entity}
        after = _files(root)
        assert set(after) == set(before)
        for name in ("memory.db", "memory.db-wal"):
            assert after[name] == before[name]
    finally:
        writer.rollback()
        writer.close()


def test_corrupt_stored_fact_fails_closed_without_touching_the_store(rig):
    root, identity, store = rig
    _seed_all_scopes(store, identity)
    entity = hashlib.sha256(b"corrupt-row").hexdigest()[:26]
    db = sqlite3.connect(store.path)
    try:
        db.execute(
            "INSERT INTO current VALUES (?,?,?,?,?,?)",
            (entity, "global", "global", "corrupt-op", "{not valid json", 4),
        )
        db.commit()
    finally:
        db.close()
    before = _snapshot(store)

    raw, failed = run("inspect", config=root, identity=identity, prefix="lop memory")

    assert failed is True
    assert raw == "Memory request refused or store unavailable. No unsafe payload included."
    assert _snapshot(store) == before


def test_desktop_registry_admits_every_memory_operation():
    """Pin the desktop vocabulary to the terminal one — nothing else does.

    ``/memory inspect`` typed in the desktop composer is planned as prose (a
    paid model turn) and 422s at the command route unless ``inspect`` is in the
    registry's ``subcommands``; the TUI path reads ``memory_cli.OPERATIONS``
    directly, which is why the gap was invisible to CI.
    """
    from local_operator import memory_cli
    from local_operator.slash_commands import (
        command_argument_refusal,
        slash_command_for,
        whole_draft_command,
    )

    spec = slash_command_for("/memory")
    assert spec is not None
    assert spec.subcommands == memory_cli.OPERATIONS
    planned = whole_draft_command("/memory inspect")
    assert planned is not None
    assert planned[1] == "inspect"
    assert command_argument_refusal(spec, "inspect") is None


def test_page_is_newest_first_marks_suggestions_and_holds_eighty_columns(rig):
    """D1/D2: the order is stated, the actionable rows are scannable, lines fit."""
    root, identity, store = rig
    now = time.time()
    old = _seed_legacy_row(
        store,
        "global",
        "global",
        title="Old imported note",
        body="From nine days ago.",
        created_at=now - 9 * 86400,
    )
    store.put("global", {"title": "Fresh durable fact", "body": "Stated just now."})
    work = _seed_legacy_row(
        store,
        "global",
        "global",
        title="PR #2463 is merged; CI green",
        body="Next step: cut the release window.",
        source={"kind": "reported", "author": "main", "session": "sess-inspect"},
        created_at=now - 3600,
    )

    report = inspect_entries(root, identity)
    stamps = [entry["created_at"] for entry in report["entries"]]
    assert stamps == sorted(stamps, reverse=True)
    assert report["entries"][-1]["id"] == old
    assert report["counts"]["suggestions"] == 1
    work_entry = next(entry for entry in report["entries"] if entry["id"] == work)
    assert work_entry["disposition"]["proposal"] == "session-scoped"

    raw, failed = run("inspect", config=root, identity=identity, prefix="lop memory")
    assert failed is False
    human = render_human(raw, prefix="lop memory")
    flat = " ".join(human.split())
    assert "Order: newest first" in flat
    assert "1 of 3 shown carries a move suggestion (→)" in flat
    marked = [line for line in human.splitlines() if line.startswith("→ ")]
    assert len(marked) == 1 and f"global/{work}" in marked[0]
    # The bar covers prose and rows; a paste-ready command is exempt (it is
    # copied, not read, and wrapping it would split a flag).
    prose = [line for line in human.splitlines() if not line.startswith("More: ")]
    over = [line for line in prose if len(line) > 80]
    assert over == [], over


def test_more_hint_repeats_every_flag_that_shaped_the_page(rig):
    """D3: following the printed hint verbatim must not change the page."""
    root, identity, store = rig
    for n in range(4):
        store.put("global", {"title": f"Fact {n}", "body": f"Body {n}."})

    raw, failed = run(
        "inspect --scope global --limit 2 --session sess-inspect",
        config=root,
        identity=identity,
        prefix="lop memory",
    )
    assert failed is False
    assert json.loads(raw)["has_more"] is True
    human = render_human(raw, prefix="lop memory")
    more = next(line for line in human.splitlines() if line.startswith("More: "))
    assert "--cursor " in more
    assert "--limit 2" in more
    assert "--scope global" in more
    assert "--session sess-inspect" in more

    followed, failed = run(
        more[len("More: lop memory ") :], config=root, identity=identity, prefix="lop memory"
    )
    assert failed is False
    assert json.loads(followed)["counts"]["returned"] == 2


def test_inspect_refuses_a_target_instead_of_silently_ignoring_it(rig):
    """NIT 6: `inspect global/abc` must not read as an applied filter."""
    root, identity, _ = rig
    raw, failed = run("inspect global/abc", config=root, identity=identity, prefix="lop memory")
    assert failed is True
    assert "lists instead of reading one fact" in raw
    assert "`lop memory show global/abc`" in raw
