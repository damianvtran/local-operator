"""The derived index: schema, search, filters, sorts, counts and metadata sync.

Every test here builds the index through :func:`indexer.rebuild_now` — the same
path the daemon takes — so a failure is a real projection/filter bug rather than
a fixture that assembled rows the writer would never produce.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.library import identity, indexer, service
from local_operator.library.metadata import MetadataStore
from local_operator.library.store import LibraryStore, fts_expression, search_tokens

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
from local_operator.session.attachments import cache_media

THIRD = "111111111111"


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


def test_search_ignores_a_query_that_tokenises_to_nothing(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    seed(root)
    assert query(root, q="!!! ...")["total"] == query(root)["total"]
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

    root = config_root(monkeypatch, tmp_path)
    image = root.parent / "photo.png"
    image.write_bytes(png())
    session(root, SESSION, write_rows("w1", 1.0, image))
    indexer.rebuild_now(root)
    asset_id = service.list_assets(root, service.Query())["rows"][0]["id"]
    store = LibraryStore(root)

    with store.write() as connection:
        connection.execute(
            "UPDATE idx_assets SET thumb_path = ?, thumb_key = ?, thumb_src = ? WHERE asset_id = ?",
            (str(root.parent / "cached.webp"), "0123456789abcdef-480", "stamp", asset_id),
        )
    live = root.parent / "cached.webp"
    live.write_bytes(b"webp")
    with store.write() as connection:
        connection.execute(
            "UPDATE idx_assets SET thumb_src = ? WHERE asset_id = ?",
            (
                __import__("local_operator.library.store", fromlist=["x"]).thumb_source_stamp(
                    image
                ),
                asset_id,
            ),
        )

    def boom(*args, **kwargs):
        raise AssertionError("liveness must not read the source bytes")

    monkeypatch.setattr(hashlib, "file_digest", boom)
    assert store.thumb_is_live(asset_id, 480) == live
    # A different width is a different file, decided from the key alone.
    assert store.thumb_is_live(asset_id, 240) is None
