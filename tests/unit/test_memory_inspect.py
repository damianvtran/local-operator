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
from local_operator.memory_inspect import (
    ENTRY_GATES,
    PREVIEW_CHARS,
    SCHEMA,
    inspect_entries,
)

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
    overrides: dict[str, Any] | None = None,
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
    # Applied last so a caller can poison raw fields the write path would refuse
    # (timestamps, status, ...) — the imported-row state this reader exists for.
    fact.update(overrides or {})
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


def _apply_imported_row(
    store: MemoryStore,
    scope: str,
    entity_id: str,
    op_id: str,
    *,
    title: str,
    body: str = "Imported body.",
    created_at: float | None = None,
    origin: str = "legacy-device",
) -> None:
    """Insert one row through the store's OWN merge path (no table editing).

    ``apply_ops`` is the entry point a peer/imported batch really uses, and it
    accepts any ``safe_segment``-shaped identifier (``validate_op`` skips the
    free-text guard for ids). Driving it here — rather than the direct INSERT
    in ``_seed_legacy_row`` — is what proves the inspector refuses ids the
    WRITE path genuinely accepted and stored, not shapes only this test can
    construct.
    """
    fact = store.prepare_fact(
        {
            "title": title,
            "body": body,
            "created_at": time.time() - 90 if created_at is None else created_at,
        }
    )
    wire = json.dumps(fact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    store.apply_ops(
        [
            {
                "v": 1,
                "op_id": op_id,
                "entity_id": entity_id,
                "origin_device": origin,
                "seq": 1,
                "hlc": [int(time.time() * 1000), 0, origin],
                "parents": [],
                "kind": "put",
                "scope": scope,
                "owner_key": "global" if scope == "global" else store.identity.owner(scope),
                "fact": fact,
                "body_sha": hashlib.sha256(wire.encode()).hexdigest(),
            }
        ]
    )


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
    # D6: the footer is a separate block, not glued to the last row or More:.
    lines = human.splitlines()
    footer = next(i for i, line in enumerate(lines) if line.startswith("No change was made"))
    assert lines[footer - 1] == ""


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


def test_sensitive_author_and_out_of_vocabulary_status_are_not_echoed(rig):
    """Every provenance string is gated, ``status`` by the write vocabulary, and
    every freshness timestamp by the write path's numeric rule.

    ``author`` is free text and ``status`` is a closed vocabulary at write; an
    imported row can hold anything in either, and both reach the screen. The
    PAN here is a value ``guard()`` AND ``validate_fact()`` both refuse, so a
    row carrying it is exactly the import this reader exists for. The same
    poison goes into ``created_at``/``verified_at``/``expires_at``: those reach
    ``--json`` and were echoed raw through three review rounds (R4-1/Q8), while
    ``validate_fact`` refuses a non-numeric value outright.
    """
    root, identity, store = rig
    pan = "4111 1111 1111 1111"
    with pytest.raises(MemoryError):
        guard(pan)
    with pytest.raises(MemoryError):
        validate_fact({"title": "safe", "body": "safe body", "source": {"author": pan}})
    with pytest.raises(MemoryError):
        validate_fact({"title": "safe", "body": "safe body", "status": pan})
    for field in ("created_at", "verified_at", "expires_at"):
        with pytest.raises(MemoryError):
            validate_fact({"title": "safe", "body": "safe body", field: pan})

    entity = _seed_legacy_row(
        store,
        "global",
        "global",
        title="Imported card holder",
        body="Nothing sensitive in this body.",
        source={"kind": "reported", "author": pan, "session": "sess-inspect"},
        status=pan,
        overrides={"verified_at": pan, "expires_at": pan, "created_at": pan},
    )

    report = inspect_entries(root, identity)
    entry = next(item for item in report["entries"] if item["id"] == entity)
    assert entry["provenance_redacted"] is True
    assert "author" not in entry["provenance"]
    assert entry["status"] == "unknown"
    assert entry["verified_at"] is None
    assert entry["expires_at"] is None
    assert entry["created_at"] is None
    assert entry["age_seconds"] is None
    assert pan not in json.dumps(report)

    raw, failed = run("inspect", config=root, identity=identity, prefix="lop memory")
    assert failed is False
    human = render_human(raw, prefix="lop memory")
    assert pan not in human
    assert "by unknown (reported)" in human
    # Legit values are NOT collateral damage: the write path accepts these, so
    # the reader keeps them.
    assert entry["provenance"]["kind"] == "reported"
    assert entry["provenance"]["session"] == "sess-inspect"
    assert "session sess-inspect" in human


def test_every_exported_field_is_gated_by_the_table(rig):
    """The exported row is exactly :data:`ENTRY_GATES`, and every entry has a gate.

    This is the regression test for the CLASS of defect three review rounds kept
    finding one field at a time: a field reaching ``--json`` with no gate in
    front of it. ``_project`` iterates the table (so a key cannot be emitted
    from anywhere else), and this test fails if a table entry is added with no
    gate — ``None`` is not callable, and an inline lambda or a module-function
    passthrough is not a ``_gate_*`` in this module.
    """
    root, identity, store = rig
    _seed_all_scopes(store, identity)
    report = inspect_entries(root, identity)
    assert report["entries"]
    for entry in report["entries"]:
        assert set(entry) == set(ENTRY_GATES)
    for name, gate in ENTRY_GATES.items():
        assert callable(gate), f"{name} is exported with no gate"
        assert gate.__module__ == "local_operator.memory_inspect", name
        assert gate.__name__.startswith("_gate_"), f"{name}: gate must be a named _gate_*"


def test_poison_in_every_stored_field_stays_out_of_the_report(rig):
    """A row whose every stored field is a PAN says nothing but the safe header.

    End-to-end twin of the table test: it poisons every key a stored fact can
    carry (what ``validate_fact`` normalises, plus the unexported ones) and
    asserts the serialised report and the human page contain none of it. A new
    output field sourced from a stored field without a gate fails here.
    """
    root, identity, store = rig
    pan = "4111 1111 1111 1111"
    entity = _seed_legacy_row(
        store,
        "global",
        "global",
        title=pan,
        body=pan,
        source={
            "kind": pan,
            "author": pan,
            "session": pan,
            "tool": pan,
            "uri": pan,
            "entry": pan,
        },
        overrides={
            "subject": pan,
            "tags": [pan],
            "aliases": [pan],
            "status": pan,
            "created_at": pan,
            "verified_at": pan,
            "expires_at": pan,
            "source_time": pan,
            "replaces": pan,
        },
    )

    report = inspect_entries(root, identity)
    entry = next(item for item in report["entries"] if item["id"] == entity)
    assert pan not in json.dumps(report)
    # The row is shown, redacted, rather than dropped: a listing that hides a
    # row entirely is a listing a person cannot act on.
    assert entry["redacted"] is True
    assert entry["title"] is None and entry["title_exported"] is False
    assert entry["body_preview"] is None and entry["body_exported"] is False
    assert entry["provenance"] == {}
    assert entry["provenance_redacted"] is True
    assert entry["status"] == "unknown"

    # Identifier columns are stored fields too: an imported row can carry a
    # PAN/SSN in entity_id/op_id (``validate_op`` shape-checks ids without the
    # free-text guard). Seeded through ``apply_ops`` so this is the writer's
    # own product, then held to the same no-echo promise.
    ssn = "123-45-6789"
    bare_pan = "4111111111111111"
    _apply_imported_row(
        store, "global", bare_pan, ssn, title="Imported identifiers", origin="legacy-ids"
    )
    report = inspect_entries(root, identity)
    imported = next(item for item in report["entries"] if item["id"] is None)
    assert imported["reference"] is None
    assert imported["revision"] is None
    assert bare_pan not in json.dumps(report)
    assert ssn not in json.dumps(report)
    assert imported["title"] == "Imported identifiers"

    raw, failed = run("inspect", config=root, identity=identity, prefix="lop memory")
    assert failed is False
    human = render_human(raw, prefix="lop memory")
    assert pan not in human
    assert bare_pan not in human
    assert ssn not in human


def test_real_minted_identifiers_pass_the_identifier_gate(rig):
    """The identifier gate must not be collateral damage: real ids stay intact.

    A row this build mints carries a canonical UUID; a promoted row carries a
    bare 64-hex digest (``apply_ops``'s remap); a device id carries ``d_`` + 32
    lowercase hex. All three are legitimate values the write path stored, so
    the reader must show them unchanged — an over-strict gate would break
    ``id``/``revision`` and paging for every normal row.
    """
    root, identity, store = rig
    uuid_id = store.put("global", {"title": "Real UUID row", "body": "Body."})["id"]
    device_id = "d_" + "0123456789abcdef" * 2
    digest_id = hashlib.sha256(b"promoted-entity").hexdigest()
    _apply_imported_row(
        store, "global", device_id, digest_id, title="Minted shapes", origin="legacy-minted"
    )

    report = inspect_entries(root, identity)
    by_title = {entry["title"]: entry for entry in report["entries"]}
    real = by_title["Real UUID row"]
    assert real["id"] == uuid_id
    assert real["reference"] == f"global/{uuid_id}"
    assert real["revision"]
    minted = by_title["Minted shapes"]
    assert minted["id"] == device_id
    assert minted["reference"] == f"global/{device_id}"
    assert minted["revision"] == digest_id

    # And the digest branch is not hypothetical: promotion really mints one.
    # ``apply_ops`` rewrites a promoted op/entity id to a bare sha256 digest,
    # so a reader that only admits UUIDs would blank every promoted row.
    peer = MemoryStore(
        root,
        MemoryIdentity("peer-session", "team-inspect", "agent-inspect"),
        root / "sessions" / "peer-session",
        origin="peer-device",
    )
    source = peer.put("session", {"title": "Promoted fact", "body": "Body."})["id"]
    with peer.connection() as db:
        operations = [
            json.loads(row[0])
            for row in db.execute("SELECT data FROM ops WHERE entity_id=?", (source,))
        ]
    store.apply_ops(operations, contiguous=False, portable=True, promote=True)
    promoted = next(
        entry
        for entry in inspect_entries(root, identity)["entries"]
        if entry["title"] == "Promoted fact"
    )
    assert promoted["id"] and len(promoted["id"]) == 64
    assert promoted["revision"] and len(promoted["revision"]) == 64


def test_refused_identifier_is_not_the_paging_cursor(rig):
    """A gate-refused id must not ride out through ``next_cursor``/``More:``.

    The cursor is the row's ordering key for the NEXT page, so reading the raw
    column would re-export the very id the row itself shows as null. Here the
    poisoned row is last on a full page with more rows behind it.
    """
    root, identity, store = rig
    pan = "4111111111111111"
    ssn = "123-45-6789"
    now = time.time()
    _apply_imported_row(
        store,
        "global",
        "01a1b2c3-d4e5-7f60-8123-456789abcdef",
        "01a1b2c3-d4e5-7f60-8123-456789abcdee",
        title="Newest imported",
        created_at=now - 10,
        origin="legacy-new",
    )
    _apply_imported_row(
        store,
        "global",
        pan,
        ssn,
        title="Poisoned middle",
        created_at=now - 600,
        origin="legacy-poison",
    )
    _apply_imported_row(
        store,
        "global",
        "01a1b2c3-d4e5-7f60-8123-456789abcde1",
        "01a1b2c3-d4e5-7f60-8123-456789abcde2",
        title="Oldest imported",
        created_at=now - 3600,
        origin="legacy-old",
    )

    report = inspect_entries(root, identity, limit=2)
    assert [entry["title"] for entry in report["entries"]] == [
        "Newest imported",
        "Poisoned middle",
    ]
    assert report["entries"][-1]["id"] is None
    assert report["has_more"] is True
    # No cursor at all rather than the refused column value.
    assert report["next_cursor"] is None
    assert pan not in json.dumps(report)
    assert ssn not in json.dumps(report)

    raw, failed = run("inspect --limit 2", config=root, identity=identity, prefix="lop memory")
    assert failed is False
    human = render_human(raw, prefix="lop memory")
    assert "More:" not in human
    assert "id withheld" in human
    assert "revision withheld" in human
    assert pan not in human
    assert ssn not in human


def test_stale_cursor_reports_real_totals_and_says_start_from_the_top(rig):
    """A cursor that no longer names a row must not deny the store's contents."""
    root, identity, store = rig
    for n in range(3):
        store.put("global", {"title": f"Fact {n}", "body": f"Body {n}."})

    first = inspect_entries(root, identity, limit=1)
    cursor = first["next_cursor"]
    assert cursor and first["counts"]["total"] == 3
    entry = first["entries"][0]
    store.put(
        "global",
        {},
        entity=entry["id"],
        expected_version=entry["revision"],
        forget=True,
    )

    stale = inspect_entries(root, identity, limit=1, cursor=cursor)

    assert stale["cursor_stale"] is True
    assert stale["entries"] == []
    # Cursor-independent: the store still holds two entries, and the page says so.
    assert stale["counts"]["total"] == 2
    assert stale["counts"]["by_scope"] == {"global": 2}

    raw, failed = run(
        f"inspect --limit 1 --cursor {cursor}",
        config=root,
        identity=identity,
        prefix="lop memory",
    )
    assert failed is False
    human = render_human(raw, prefix="lop memory")
    flat = " ".join(human.split())
    assert "2 entries in the authorized scopes" in flat
    assert "Cursor no longer available" in flat
    assert "Nothing retained" not in human


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
    # D6 on a paged view too: the footer must not touch the More: command.
    lines = human.splitlines()
    footer = next(i for i, line in enumerate(lines) if line.startswith("No change was made"))
    assert lines[footer - 1] == "" and lines[footer - 2].startswith("More: ")

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
