"""The derived index: schema, search, filters, sorts, counts and metadata sync.

Every test here builds the index through :func:`indexer.rebuild_now` — the same
path the daemon takes — so a failure is a real projection/filter bug rather than
a fixture that assembled rows the writer would never produce.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.library import identity, indexer, service
from local_operator.library.metadata import MetadataStore
from local_operator.library.store import LibraryStore, fts_expression, search_tokens
from local_operator.session.attachments import cache_media

from ._fixtures import (
    OTHER,
    SESSION,
    config_root,
    image_rows,
    png,
    session,
    user_row,
    write_rows,
)

THIRD = "111111111111"
SECOND = "222222222222"


def _row(**fields: Any) -> dict[str, Any]:
    """One synthetic scan record, shaped like ``index._record_dict``."""
    base: dict[str, Any] = {
        "key": "",
        "path": None,
        "digest": None,
        "name": "",
        "kind": "other",
        "content_type": None,
        "size_bytes": None,
        "width": None,
        "height": None,
        "duration_s": None,
        "origin": "written",
        "tool": "write",
        "intent": None,
        "first_ts": 1.0,
        "last_ts": 1.0,
        "first_seq": 0,
        "last_seq": 0,
        "touches": 1,
        "highlighted": False,
        "title": "A session",
        "child_id": "",
        "label": None,
    }
    base.update(fields)
    return base


def seed(root: Path, *, rebuild: bool = True) -> None:
    first = root.parent / "quarterly-report.pdf"
    first.write_bytes(b"%PDF-1.4")
    second = root.parent / "notes.md"
    second.write_text("# notes")
    fourth = root.parent / "plans.txt"
    fourth.write_text("plans")
    block = cache_media(png(), "image/png", kind="image", name="cat.png")
    assert block and block.attachment
    session(
        root,
        SESSION,
        [
            user_row("u1", 1.0, "review the quarterly numbers"),
            *write_rows("w1", 2.0, first),
            *write_rows("w2", 3.0, second),
            *image_rows("g1", 4.0, block.attachment),
        ],
    )
    session(root, OTHER, [user_row("u2", 5.0, "look at the notes"), *write_rows("w3", 5.5, fourth)])
    if rebuild:
        indexer.rebuild_now(root)


def query(root: Path, **kwargs: Any) -> dict[str, Any]:
    return service.list_assets(root, service.Query(**kwargs))


def test_search_matches_prefixes_across_names_descriptions_tags_and_titles(
    tmp_path, monkeypatch
) -> None:
    root = config_root(monkeypatch, tmp_path)
    seed(root)
    metadata = MetadataStore(root)

    # A prefix of a name token, and nothing else: "report" is one token of
    # "quarterly-report.pdf" and appears in no conversation title here.
    assert query(root, q="eport")["total"] == 0, "search is token-prefix, not substring"
    assert query(root, q="report")["total"] == 1
    assert query(root, q="quarterly rep")["total"] == 1, "tokens are ANDed"
    # A user rename must not make the file unfindable by its filename.
    report_id = next(r["id"] for r in query(root)["rows"] if r["name"].endswith(".pdf"))
    service.update_asset(root, report_id, {"name": "Q3 Numbers"}, actor="user", source="user")
    assert query(root, q="q3")["total"] == 1
    assert query(root, q="report")["total"] == 1, "the filename must stay searchable"

    # Description.
    service.update_asset(
        root, report_id, {"description": "Revenue by region"}, actor="user", source="user"
    )
    assert query(root, q="reven")["total"] == 1

    # Tag name.
    tag = metadata.create_tag("finance")
    service.set_tags(root, report_id, [tag["id"]])
    assert query(root, q="financ")["total"] == 1
    service.set_tags(root, report_id, [])
    assert query(root, q="financ")["total"] == 0, "an untagged asset must leave the index"

    # Conversation title: every asset that occurred in OTHER is a match.
    assert query(root, q="look")["total"] == 1

    # Multiple tokens AND together.
    assert query(root, q="report plans")["total"] == 0, "tokens AND together"
    assert query(root, q="reporttypo")["total"] == 0


def test_a_query_with_no_token_matches_nothing(tmp_path, monkeypatch) -> None:
    """``!!`` is a query, not an absence of one: it must not answer all rows."""
    root = config_root(monkeypatch, tmp_path)
    seed(root)
    assert query(root)["total"] == 4
    assert query(root, q="!!! ...")["total"] == 0
    # An EMPTY q is still "no filter".
    assert query(root, q="   ")["total"] == 4
    assert search_tokens("!!!") == []
    assert fts_expression("!!!") is None
    assert fts_expression("a b*c") == '"a"* AND "b"* AND "c"*'


def test_every_filter_and_combination(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    seed(root)
    names = lambda frame: {row["name"] for row in frame["rows"]}  # noqa: E731
    everything = query(root)
    assert everything["total"] == 4

    assert names(query(root, categories=frozenset({"image"}))) == {"cat.png"}
    assert names(query(root, kinds=frozenset({"pdf"}))) == {"quarterly-report.pdf"}
    assert names(query(root, origin_groups=frozenset({"generated"}))) == {"cat.png"}
    assert names(query(root, origins=frozenset({"written"}))) == {
        "quarterly-report.pdf",
        "notes.md",
        "plans.txt",
    }
    assert names(query(root, highlighted=True)) == set()
    assert names(query(root, highlighted=False)) == names(everything)

    report_id = next(r["id"] for r in everything["rows"] if r["name"].endswith(".pdf"))
    metadata = MetadataStore(root)
    service.update_asset(root, report_id, {"favorite": True}, actor="user", source="user")
    assert names(query(root, favorite=True)) == {"quarterly-report.pdf"}
    service.update_asset(root, report_id, {"organized": True}, actor="user", source="user")
    assert names(query(root, organized=True)) == {"quarterly-report.pdf"}
    assert names(query(root, organized=False)) == names(everything) - {"quarterly-report.pdf"}

    tag = metadata.create_tag("finance")
    service.set_tags(root, report_id, [tag["id"]])
    assert names(query(root, tag=tag["id"])) == {"quarterly-report.pdf"}

    # Sessions: one id, and a comma list (EXISTS over occurrences).
    assert names(query(root, session=OTHER)) == {"plans.txt"}
    assert names(query(root, session_ids=frozenset({OTHER}))) == {"plans.txt"}
    assert names(query(root, session_ids=frozenset({SESSION, OTHER}))) == names(everything)

    # A combination narrows by AND, not by OR.
    combined = query(
        root,
        categories=frozenset({"image", "document"}),
        origin_groups=frozenset({"generated"}),
        q="cat",
    )
    assert names(combined) == {"cat.png"}

    final = query(root, scope="final")
    assert names(final) == {"cat.png"}, "only the generated artifact is 'kept' here"


def test_all_sorts_and_orders(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    seed(root)
    for sort in service.SORTS:
        for order in (None, "asc", "desc"):
            frame = query(root, sort=sort, order=order, limit=50)
            assert len(frame["rows"]) == frame["total"], (sort, order)
            ids = [row["id"] for row in frame["rows"]]
            assert len(ids) == len(set(ids)), f"{sort}/{order} duplicated a row"

    names = [row["name"] for row in query(root, sort="name", order="asc")["rows"]]
    assert names == sorted(names, key=str.lower)
    names_desc = [row["name"] for row in query(root, sort="name", order="desc")["rows"]]
    assert names_desc == sorted(names, key=str.lower, reverse=True)

    recent = query(root, sort="recent")["rows"]
    stamps = [row["modified_at"] for row in recent]
    assert stamps == sorted(stamps, reverse=True), "recent is newest-first by default"
    oldest_first = [row["modified_at"] for row in query(root, sort="created", order="asc")["rows"]]
    assert oldest_first == sorted(oldest_first)


def test_counts_follow_insert_update_and_delete(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    seed(root)
    store = LibraryStore(root)
    all_counts = store.counts("all")
    assert all_counts["category"]["image"] == 1
    assert all_counts["origin_group"] == {"generated": 1, "created": 3}
    assert store.counts("final") == {
        "category": {"image": 1},
        "kind": {"image": 1},
        "origin_group": {"generated": 1},
    }

    path = root.parent / "extra.txt"
    path.write_text("x")
    key = identity.key_for(str(path), None)
    asset_id = identity.asset_id(key)
    with store.write() as connection:
        store.replace_session_occurrences(
            connection,
            THIRD,
            [_row(asset_id=asset_id, key=key, path=str(path), name="extra.txt")],
        )
        store.recompute_assets(connection, [asset_id])
    assert store.counts("all")["category"]["document"] == 4

    # An is_final transition moves a row BETWEEN scopes, both ways.
    with store.write() as connection:
        store.replace_session_occurrences(
            connection,
            THIRD,
            [
                _row(
                    asset_id=asset_id,
                    key=key,
                    path=str(path),
                    name="extra.txt",
                    highlighted=True,
                )
            ],
        )
        store.recompute_assets(connection, [asset_id])
    assert store.counts("final")["category"]["document"] == 1
    assert store.counts("all")["category"]["document"] == 4

    # Metadata favourite is an authoritative write, not a derived one: it must
    # not disturb the derived counts.
    metadata = MetadataStore(root)
    metadata.ensure_asset(asset_id, key, derived_name="extra.txt")
    metadata.update_fields(asset_id, {"favorite": True}, actor="user", source="user")
    assert store.counts("all")["category"]["document"] == 4

    with store.write() as connection:
        store.forget_session(connection, THIRD)
        store.recompute_assets(connection, [asset_id])
    assert store.counts("all")["category"]["document"] == 3
    assert store.counts("final")["category"] == {"image": 1}
    assert store.asset(asset_id) is None, "an asset with no occurrence is gone"


def test_deleting_a_session_removes_only_its_occurrences(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    seed(root)
    store = LibraryStore(root)
    before = {row["name"]: row for row in query(root)["rows"]}

    with store.write() as connection:
        affected = store.forget_session(connection, OTHER)
        store.recompute_assets(connection, affected)
    after = query(root)["rows"]
    assert len(after) == len(before) - 1
    assert "plans.txt" not in {row["name"] for row in after}
    assert "notes.md" in {row["name"] for row in after}, "another session still has it"
    assert store.counts("all")["origin_group"] == {"created": 2, "generated": 1}


def test_metadata_writes_reach_search_and_the_sorted_name_immediately(
    tmp_path, monkeypatch
) -> None:
    """Triggers, not a rebuild: a rename or a tag edit is visible on the next query."""
    root = config_root(monkeypatch, tmp_path)
    seed(root)
    row = next(r for r in query(root)["rows"] if r["name"] == "notes.md")
    asset_id = row["id"]
    metadata = MetadataStore(root)
    store = LibraryStore(root)

    service.update_asset(root, asset_id, {"name": "Zebra Notes"}, actor="user", source="user")
    assert query(root, q="zebra")["total"] == 1
    assert query(root, q="zeb")["total"] == 1
    last = query(root, sort="name", order="asc")["rows"][-1]
    assert last["id"] == asset_id, "the rename moved the row to the end of a name sort"
    with store.read() as connection:
        stored = connection.execute(
            "SELECT effective_name FROM idx_assets WHERE asset_id = ?", (asset_id,)
        ).fetchone()
    assert stored["effective_name"] == "Zebra Notes"

    # A tag add, then a tag rename, then removal — each reaches the FTS row.
    tag = metadata.create_tag("invoice")
    service.set_tags(root, asset_id, [tag["id"]])
    assert query(root, q="invoic")["total"] == 1
    metadata.rename_tag(tag["id"], "receipts")
    assert query(root, q="invoic")["total"] == 0
    assert query(root, q="receip")["total"] == 1
    service.set_tags(root, asset_id, [])
    assert query(root, q="receip")["total"] == 0

    # A derived-only refresh must NOT overwrite a user name: recomputing an
    # asset used to fall back to its derived name and silently undo the rename.
    with store.write() as connection:
        store.recompute_assets(connection, [asset_id])
    assert query(root, q="zebra")["total"] == 1
    with store.read() as connection:
        stored = connection.execute(
            "SELECT effective_name, derived_name FROM idx_assets WHERE asset_id = ?", (asset_id,)
        ).fetchone()
    assert stored["effective_name"] == "Zebra Notes"
    assert stored["derived_name"] == "notes.md"


def test_the_index_is_rebuildable_and_survives_a_version_move(tmp_path, monkeypatch) -> None:
    """Every ``idx_*`` table is derived: dropping them costs a rebuild, never data."""
    root = config_root(monkeypatch, tmp_path)
    seed(root)
    row = next(r for r in query(root)["rows"] if r["name"] == "notes.md")
    service.update_asset(root, row["id"], {"name": "Kept Name"}, actor="user", source="user")

    store = LibraryStore(root)
    with store.write() as connection:
        for table in (
            "idx_assets_fts",
            "idx_counts",
            "idx_occurrences",
            "idx_assets",
            "idx_sessions",
        ):
            connection.execute(f"DELETE FROM {table}")
        store.set_state(connection, backfill_done="0", revision="0")

    frame = query(root)
    assert frame["total"] == 0 and frame["state"] == "building"
    indexer.rebuild_now(root)
    frame = query(root)
    assert frame["total"] == 4
    assert next(r for r in frame["rows"] if r["id"] == row["id"])["name"] == "Kept Name"


def test_revision_advances_with_index_writes_and_metadata_edits(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    seed(root)
    first = query(root)["revision"]
    path = root.parent / "later.txt"
    path.write_text("x")
    session(root, THIRD, write_rows("w9", 9, path))
    indexer.rebuild_now(root)
    second = query(root)["revision"]
    assert second > first

    row = query(root)["rows"][0]
    service.update_asset(root, row["id"], {"locked": True}, actor="user", source="user")
    assert query(root)["revision"] != second


def test_each_sort_keeps_its_default_direction(tmp_path, monkeypatch) -> None:
    """``order=None`` must mean ``DEFAULT_ORDER[sort]``, not "desc everywhere".

    The TUI and the mobile API leave ``order`` out. Reading ``query.order``
    alone turned ``sort=kind`` into a descending sort for every one of them.
    """
    root = config_root(monkeypatch, tmp_path)
    seed(root)
    store = LibraryStore(root)
    for sort, default in service.DEFAULT_ORDER.items():
        implicit, _ = store.query_page(service.Query(sort=sort), limit=50)
        explicit, _ = store.query_page(service.Query(sort=sort, order=default), limit=50)
        assert [asset.id for asset in implicit] == [asset.id for asset in explicit], sort


def test_the_schema_is_verified_once_per_process(tmp_path, monkeypatch) -> None:
    """A store per request must not re-run DDL on the read path.

    Proven with a CLOSED sqlite connection: once the process has verified the
    path, ``_ensure_schema`` answers True without issuing a single statement —
    if it tried any, the closed connection would raise. The control at the end
    is the same call on an unverified path, which must fail.
    """
    import sqlite3

    from local_operator.library.metadata import MetadataStore

    root = config_root(monkeypatch, tmp_path)
    seed(root)
    for _ in range(3):
        query(root)

    dead = sqlite3.connect(":memory:")
    dead.close()
    assert LibraryStore(root)._ensure_schema(dead) is True
    assert MetadataStore(root)._ensure_schema(dead) is True

    # Negative control: an unverified path must reach SQL and fail.
    fresh = root.parent / "fresh-cfg"
    fresh.mkdir()
    from local_operator.library.store import IndexUnavailable

    with pytest.raises(IndexUnavailable):
        LibraryStore(fresh)._ensure_schema(dead)


def test_thumbnail_liveness_never_hashes_the_source(tmp_path, monkeypatch) -> None:
    """One stat per request, never a full-file read of the image."""
    import hashlib

    from local_operator.library.store import thumb_source_stamp

    root = config_root(monkeypatch, tmp_path)
    image = root.parent / "photo.png"
    image.write_bytes(png())
    session(root, SESSION, write_rows("w1", 1.0, image))
    indexer.rebuild_now(root)
    asset_id = service.list_assets(root, service.Query())["rows"][0]["id"]
    store = LibraryStore(root)

    live = root.parent / "cached.webp"
    live.write_bytes(b"webp")
    with store.write() as connection:
        connection.execute(
            "INSERT OR REPLACE INTO idx_thumbnails(asset_id, width, path, key, src, created_at) "
            "VALUES(?, ?, ?, ?, ?, ?)",
            (asset_id, 480, str(live), "0123456789abcdef-480", thumb_source_stamp(image), 1.0),
        )

    def boom(*args, **kwargs):
        raise AssertionError("liveness must not read the source bytes")

    monkeypatch.setattr(hashlib, "file_digest", boom)
    assert store.thumb_is_live(asset_id, 480) == live
    # A width with no row is a miss, decided without touching the source.
    assert store.thumb_is_live(asset_id, 240) is None


def test_two_widths_are_cached_and_served_separately(tmp_path, monkeypatch) -> None:
    """A single slot per asset could only serve the width generated last."""
    pytest.importorskip("PIL")
    root = config_root(monkeypatch, tmp_path)
    image = root.parent / "photo.png"
    image.write_bytes(png())
    session(root, SESSION, write_rows("w1", 1.0, image))
    indexer.rebuild_now(root)
    asset_id = service.list_assets(root, service.Query())["rows"][0]["id"]
    store = LibraryStore(root)

    first = store.ensure_thumbnail(asset_id, width=240, quality=82)
    second = store.ensure_thumbnail(asset_id, width=480, quality=82)
    assert first and second and first != second
    assert first.endswith("-240.webp") and second.endswith("-480.webp")
    assert Path(first).is_file() and Path(second).is_file()

    # BOTH are served at their own width — the regression the review found.
    assert service.thumbnail(root, asset_id, 240).path == Path(first)
    assert service.thumbnail(root, asset_id, 480).path == Path(second)
    assert service.thumbnail(root, asset_id, 240).content_type == "image/webp"


def test_a_path_token_is_searchable(tmp_path, monkeypatch) -> None:
    """``CameraRoll`` is a folder name: the old haystack matched the display path."""
    root = config_root(monkeypatch, tmp_path)
    folder = root.parent / "CameraRoll"
    folder.mkdir()
    photo = folder / "IMG_0001.png"
    photo.write_bytes(png())
    session(root, SESSION, write_rows("w1", 1.0, photo))
    indexer.rebuild_now(root)

    assert query(root, q="img")["total"] == 1
    assert query(root, q="camera")["total"] == 1
    assert query(root, q="cameraroll")["total"] == 1
    assert query(root, q="nothing")["total"] == 0


def test_name_sort_is_ascii_folded(tmp_path, monkeypatch) -> None:
    """Pins the documented limitation: ``COLLATE NOCASE`` folds A-Z only."""
    root = config_root(monkeypatch, tmp_path)
    rows: list[dict[str, Any]] = []
    for number, name in enumerate(("apple.txt", "Zebra.txt", "École.txt")):
        path = root.parent / name
        path.write_text("x")
        rows += write_rows(f"w{number}", float(number + 1), path)
    session(root, SESSION, rows)
    indexer.rebuild_now(root)

    ordered = [row["name"] for row in query(root, sort="name", order="asc", limit=10)["rows"]]
    assert ordered[0] == "apple.txt"
    assert ordered[1] == "Zebra.txt"
    # Byte order, not Unicode case folding: the accented name sorts last.
    assert ordered[-1] == "École.txt"


def test_a_schema_version_move_discards_the_index_and_its_state(tmp_path, monkeypatch) -> None:
    """A bump rebuilds the tables AND the counters, or the UI says ready on empty."""
    from local_operator.library.store import SCHEMA_VERSION

    root = config_root(monkeypatch, tmp_path)
    seed(root)
    store = LibraryStore(root)
    assert query(root)["state"] == "ready"

    # The lazy worker re-adds the readiness key from its own thread; stop it, or
    # this test races the very cache it is exercising.
    indexer.reset_for_tests()
    with store.write() as connection:
        connection.execute("UPDATE idx_state SET value = '99' WHERE key = 'schema_version'")
        connection.execute("UPDATE idx_state SET value = '2' WHERE key = 'sessions_total'")
    LibraryStore._ready_paths.discard(store._key)
    store.ensure_schema()

    # Read the STATE directly: a service call would start the indexer, which can
    # backfill this tiny config before the assertion runs.
    assert store.listing_state() == ("building", 0, 0)
    assert store.query_page(service.Query(), limit=10)[1] == 0
    assert store.state()["schema_version"] == str(SCHEMA_VERSION)
    assert store.state()["backfill_done"] == "0"

    # The version row is an upsert, so the NEXT start does not drop again: the
    # index the previous process built survives.
    indexer.rebuild_now(root)
    before = query(root)["total"]
    indexer.reset_for_tests()
    LibraryStore._ready_paths.discard(store._key)
    store.ensure_schema()
    assert query(root)["total"] == before


def test_recovery_drops_only_the_derived_tables(tmp_path, monkeypatch) -> None:
    """A corrupt derived index rebuilds; names and tags are never lost."""
    root = config_root(monkeypatch, tmp_path)
    seed(root)
    row = next(r for r in query(root)["rows"] if r["name"] == "notes.md")
    service.update_asset(root, row["id"], {"name": "Kept"}, actor="user", source="user")
    store = LibraryStore(root)
    tag = MetadataStore(root).create_tag("kept-by-tag")
    service.set_tags(root, row["id"], [tag["id"]])

    with store.write() as connection:
        connection.execute("DELETE FROM idx_assets")
        connection.execute("DELETE FROM idx_counts")
    assert query(root)["total"] == 0

    assert store.recover_derived() is True
    assert store.state()["backfill_done"] == "0"
    indexer.rebuild_now(root)
    frame = query(root)
    assert next(r for r in frame["rows"] if r["id"] == row["id"])["name"] == "Kept"
    assert frame["tags"] and frame["tags"][0]["name"] == "kept-by-tag"
    stored = MetadataStore(root).asset(row["id"])
    assert stored is not None and stored["name"] == "Kept"


def test_an_aggregate_does_not_depend_on_scan_order(tmp_path, monkeypatch) -> None:
    """Rescanning ONE session must not flip a shared asset's derived fields.

    Two sessions name the same file differently. The aggregate must be a
    function of the occurrence set ordered by TIME, not of which session was
    scanned last — under ``ORDER BY rowid`` the rescan moved its rows to the end
    and its record won.
    """
    root = config_root(monkeypatch, tmp_path)
    shared = root.parent / "shared.txt"
    shared.write_text("x")
    # The 1.0 session is indexed SECOND, so insertion order and time disagree.
    session(root, SECOND, write_rows("w2", 2.0, shared))
    session(root, OTHER, write_rows("w1", 1.0, shared))
    indexer.rebuild_now(root)
    asset_id = identity.asset_id(identity.key_for(str(shared), None))
    store = LibraryStore(root)
    before = store.asset(asset_id)
    assert before is not None

    # Re-insert the EARLIER session's rows with a DIFFERENT name: only its
    # rowid moves, its time does not.
    with store.write() as connection:
        store.forget_session(connection, OTHER)
        store.replace_session_occurrences(
            connection,
            OTHER,
            [
                {
                    "asset_id": asset_id,
                    "key": before.key,
                    "session_id": OTHER,
                    "title": "earlier",
                    "origin": "written",
                    "child_id": "",
                    "label": None,
                    "tool": "write",
                    "intent": None,
                    "first_ts": 1.0,
                    "last_ts": 1.0,
                    "first_seq": 1,
                    "last_seq": 1,
                    "touches": 1,
                    "highlighted": False,
                    "name": "from-the-earlier-session.txt",
                    "path": str(shared),
                    "kind": "text",
                    "content_type": "text/plain",
                    "size_bytes": 1,
                    "width": None,
                    "height": None,
                    "duration_s": None,
                }
            ],
        )
        store.recompute_assets(connection, [asset_id])
    after = store.asset(asset_id)
    assert after is not None
    # The 2.0 session is still the LAST in time, so it still names the asset.
    assert after.derived_name == "shared.txt"
    assert after.kind == before.kind


def test_recovery_rebuilds_a_malformed_derived_table(tmp_path, monkeypatch) -> None:
    """The case recovery exists for: a table that makes ``_ensure_schema`` fail.

    Reproduced on the round-1 head as a permanent 503 — ``recover_derived``
    opened through ``_connect``, whose schema step hit the same malformed table
    it was trying to drop.
    """
    root = config_root(monkeypatch, tmp_path)
    seed(root)
    store = LibraryStore(root)
    with store.write() as connection:
        connection.execute("DROP TABLE idx_assets")
        connection.execute("CREATE TABLE idx_assets(oops TEXT)")
        connection.execute("DROP TABLE idx_occurrences")
        connection.execute("CREATE TABLE idx_occurrences(oops TEXT)")
    LibraryStore._ready_paths.discard(store._key)

    assert store.recover_derived() is True
    frame = query(root)
    assert frame["total"] == 0
    indexer.rebuild_now(root)
    assert query(root)["total"] == 4


def test_an_empty_config_lists_as_ready(tmp_path, monkeypatch) -> None:
    """An empty journal directory is a finished empty index, not a build."""
    root = config_root(monkeypatch, tmp_path)
    store = LibraryStore(root)
    store.ensure_schema()
    assert store.listing_state()[0] == "ready"

    # A session that exists but has not been indexed yet IS a build.
    session(root, SESSION, write_rows("w1", 1.0, root.parent / "a.txt"))
    assert store.listing_state()[0] == "building"


def test_concurrent_readers_survive_repeated_recovery(tmp_path, monkeypatch) -> None:
    """Six readers against 25 rebuilds: every call answers or raises 503-typed."""
    import threading

    root = config_root(monkeypatch, tmp_path)
    seed(root)
    failures: list[BaseException] = []
    answered: list[int] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            try:
                service.list_assets(root, service.Query(scope="final"))
                answered.append(1)
            except service.LibraryUnavailable:
                answered.append(1)
            except BaseException as error:  # noqa: BLE001 - the point of the test
                failures.append(error)

    def recoverer() -> None:
        for _ in range(25):
            try:
                LibraryStore(root).recover_derived()
            except BaseException as error:  # noqa: BLE001
                failures.append(error)
            time.sleep(0.002)

    readers = [threading.Thread(target=reader) for _ in range(6)]
    for thread in readers:
        thread.start()
    recoverer()
    stop.set()
    for thread in readers:
        thread.join(10)

    assert not failures, failures[:1]
    assert answered, "the readers must have answered requests"
