from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator import resume
from local_operator.library import identity
from local_operator.library import index as library_index
from local_operator.library import indexer, service
from local_operator.library.metadata import MetadataStore
from local_operator.library.store import LibraryStore
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


def listing(root: Path, query: service.Query) -> dict[str, Any]:
    """A test asks for a COMPLETE answer; the 1.5 s cold budget is a route concern."""
    return service.list_assets(root, query, budget_s=None)


def build(root: Path, rows: list[dict[str, Any]]) -> None:
    """Write a session and build the index the way the operator's app would.

    The service used to build from the journals inside ``list_assets``. It reads
    the SQLite index now, so a test that wants a populated listing has to say so
    — the same explicitness the daemon gets from its watcher.
    """
    session(root, SESSION, rows)
    indexer.rebuild_now(root)


def test_row_projection_uses_effective_metadata_and_display_form(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    path = tmp_path / "report.md"
    path.write_text("# hi")
    build(root, write_rows("w1", 1, path))
    row = listing(root, service.Query())["rows"][0]
    assert row["ref"]["path"] == str(path)
    assert row["ref"]["display_path"] == "~/report.md"
    assert row["kind"] == "markdown" and row["category"] == "document"
    assert row["occurrences"][0]["session_id"] == SESSION
    occurrence = row["occurrences"][0]
    assert occurrence["first_seq"] <= occurrence["last_seq"]

    store = MetadataStore(root)
    store.ensure_asset(row["id"], f"p:{path}")
    store.update_fields(row["id"], {"name": "Quarterly", "description": "Numbers"}, actor="user")
    assert service.get_asset(root, row["id"])["name"] == "Quarterly"

    phone = service.get_asset(root, row["id"], display=True)
    assert phone["ref"]["path"] is None and phone["ref"]["display_path"] == "~/report.md"


def test_filters_sort_and_session_id_list(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    first = tmp_path / "a.txt"
    first.write_text("a")
    second = tmp_path / "b.png"
    second.write_bytes(b"png")
    build(root, write_rows("w1", 1, first) + write_rows("w2", 2, second, text="x"))
    session(root, "fedcba654321", [user_row("u1", 3, f"{first}")])

    answer = listing(root, service.Query(kinds=frozenset({"image"})))
    assert answer["total"] == 1, answer["counts"]
    # A file both created and mentioned later is CREATED: the strongest origin wins.
    assert listing(root, service.Query(origin_groups=frozenset({"created"})))["total"] == 2
    assert (
        service.list_assets(root, service.Query(session_ids=frozenset({SESSION, "fedcba654321"})))[
            "total"
        ]
        == 2
    )
    names = [row["name"] for row in listing(root, service.Query(sort="name", order="asc"))["rows"]]
    assert names == sorted(names, key=str.lower)
    searched = listing(root, service.Query(q="a.txt"))
    assert searched["total"] == 1


def test_cursor_pages_and_resets_on_a_moved_revision(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    rows: list[dict[str, Any]] = []
    for index in range(4):
        path = tmp_path / f"f{index}.txt"
        path.write_text("x")
        rows += write_rows(f"w{index}", index + 1, path, text="ok")
    build(root, rows)
    first = listing(root, service.Query(limit=2))
    assert len(first["rows"]) == 2 and first["next_cursor"]
    second = service.list_assets(root, service.Query(limit=2, cursor=first["next_cursor"]))
    assert len(second["rows"]) == 2 and second["cursor_reset"] is False
    assert {row["id"] for row in first["rows"]}.isdisjoint({row["id"] for row in second["rows"]})

    extra = tmp_path / "late.txt"
    extra.write_text("late")
    session(root, "fedcba654321", write_rows("late", 9, extra))
    # The indexer's pass is what moves the revision; a cursor minted before it
    # must reset rather than skip the row that crossed the page boundary.
    indexer.rebuild_now(root)
    moved = listing(root, service.Query(limit=2, cursor=first["next_cursor"]))
    assert moved["cursor_reset"] is True
    assert len(moved["rows"]) == 2

    try:
        listing(root, service.Query(limit=2, cursor="deadbeef.1.0"))
    except service.InvalidCursor:
        pass
    else:  # pragma: no cover
        raise AssertionError("a foreign cursor must be refused")


def test_path_content_thumbnail_and_digest_content(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    image = tmp_path / "photo.png"
    image.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
        b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
        b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    build(root, write_rows("w1", 1, image))
    asset = listing(root, service.Query())["rows"][0]
    content = service.resolve_content(root, asset["id"])
    assert content.path == image
    thumb = service.thumbnail(root, asset["id"], 120)
    assert thumb.path.exists() and thumb.content_type in ("image/webp", "image/png")

    # A denied or vanished source is a typed refusal, never a path read.
    image.unlink()
    try:
        service.resolve_content(root, asset["id"])
    except service.ContentUnavailable as error:
        assert error.code == "content_missing"
    else:  # pragma: no cover
        raise AssertionError("a missing source must refuse")


def test_chat_link_verifies_the_session_and_clears_a_dead_one(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    path = tmp_path / "a.txt"
    path.write_text("x")
    build(root, write_rows("w1", 1, path))
    asset_id = listing(root, service.Query())["rows"][0]["id"]

    chat_session = session(root, "111111111111")
    service.set_chat_link(root, asset_id, "111111111111")
    chat = service.chat_link(root, asset_id)
    assert chat is not None and chat["session_id"] == "111111111111"
    detail = service.get_asset(root, asset_id)
    assert detail["chat"] is not None and detail["chat"]["session_id"] == "111111111111"

    # A deleted conversation clears the stale link on the next read.
    for child in sorted(chat_session.rglob("*"), reverse=True):
        child.unlink() if child.is_file() else child.rmdir()
    chat_session.rmdir()
    assert service.get_asset(root, asset_id)["chat"] is None
    assert MetadataStore(root).chat(asset_id) is None

    from local_operator.library.metadata import TagRefused

    try:
        service.set_chat_link(root, asset_id, "not-a-session")
    except TagRefused:
        pass
    else:  # pragma: no cover
        raise AssertionError("a malformed session id must be refused")


def test_user_fields_survive_agent_writes_and_locked_rows(monkeypatch) -> None:
    import tempfile

    from local_operator.library.metadata import MetadataStore, TagRefused

    work = Path(tempfile.mkdtemp())
    root = config_root(monkeypatch, work)
    path = work / "a.txt"
    path.write_text("x")
    build(root, write_rows("w1", 1, path))
    asset_id = listing(root, service.Query())["rows"][0]["id"]

    service.update_asset(root, asset_id, {"name": "Mine", "description": "My words"}, actor="user")
    # A caller that names the writer but omits `source` must not claim a user
    # edit: provenance defaults to the actor.
    service.update_asset(
        root, asset_id, {"name": "Librarian", "description": "Librarian words"}, actor="librarian"
    )
    service.update_asset(root, asset_id, {"name": "Agent"}, actor="agent")
    row = service.get_asset(root, asset_id)
    assert (row["name"], row["description"]) == ("Mine", "My words")
    store = MetadataStore(root)
    stored = store.asset(asset_id)
    assert stored is not None
    assert stored["name_source"] == "user"
    assert stored["description_source"] == "user"

    # The tags path carries actor/provenance too, and must not touch the
    # effective name/description while replacing assignments.
    tag = store.create_tag("shared-vocabulary")
    service.set_tags(root, asset_id, [tag["id"]], actor="librarian")
    assert service.get_asset(root, asset_id)["name"] == "Mine"

    service.update_asset(root, asset_id, {"locked": True}, actor="user")
    for actor in ("librarian", "agent"):
        try:
            service.update_asset(root, asset_id, {"name": f"{actor} name"}, actor=actor)
        except TagRefused as error:
            assert error.code == "asset_locked"
        else:  # pragma: no cover
            raise AssertionError(f"{actor} overwrote a locked asset")
    locked_tag = store.create_tag("locked-test")
    try:
        service.set_tags(root, asset_id, [locked_tag["id"]], actor="librarian")
    except TagRefused as error:
        assert error.code == "asset_locked"
    else:  # pragma: no cover
        raise AssertionError("a non-user tag write touched a locked asset")
    # The user remains able to unlock and edit their own row.
    assert service.update_asset(root, asset_id, {"locked": False}, actor="user")["locked"] is False
    assert service.update_asset(root, asset_id, {"name": "Mine again"}, actor="user")["name"] == (
        "Mine again"
    )


def test_thumbnail_reads_the_index_without_a_session_scan(tmp_path, monkeypatch) -> None:
    """Content/thumbnail/get must not pay a journal scan or an index rebuild.

    A 60-card grid issues 60 thumbnail requests; each one running a build is
    what pushed a list past the desktop's 20 s deadline. The index is now a
    SQLite read, so the negative control is that the journal scanner is never
    even consulted.
    """
    from local_operator.library import index as library_index

    root = config_root(monkeypatch, tmp_path)
    image = tmp_path / "photo.png"
    image.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
        b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
        b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    build(root, write_rows("w1", 1, image))
    asset_id = listing(root, service.Query())["rows"][0]["id"]

    def boom(*args, **kwargs):
        raise AssertionError("a thumbnail read must not touch the journal scanner")

    monkeypatch.setattr(library_index, "_visible_sessions", boom)
    monkeypatch.setattr(library_index, "build_catalog", boom)

    first = service.thumbnail(root, asset_id, 120)
    second = service.thumbnail(root, asset_id, 120)

    # The FIRST read has no cached WebP yet, so it serves the original and asks
    # the worker for one; it must not resize inline. The SECOND may already be
    # the cached file — the worker is a thread — but it is never a fresh resize
    # on the request path, so it is either the original or a file the worker
    # wrote under ``cache/library/thumbs``.
    thumbs = root / "cache" / "library" / "thumbs"
    assert first.path == image
    assert second.path == image or second.path.parent == thumbs


class _NoWorker:
    """Stands in for the indexer where a test wants no background thread."""

    @staticmethod
    def enqueue_thumbnail(_asset_id: str) -> None:
        return None


def test_a_cached_thumbnail_is_served_once_the_worker_made_it(tmp_path, monkeypatch) -> None:
    """The request path serves a live cached WebP and never decodes the source."""
    # No background worker here, and not just from now on: it starts lazily on
    # the FIRST list request, so it must be stubbed before the fixture builds —
    # a worker sweeping at 480 while the assertion runs is a race, not a bug.
    monkeypatch.setattr(service, "ensure_started", lambda *args, **kwargs: _NoWorker())
    root = config_root(monkeypatch, tmp_path)
    image = tmp_path / "photo.png"
    image.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
        b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
        b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    build(root, write_rows("w1", 1, image))
    asset_id = listing(root, service.Query())["rows"][0]["id"]

    pytest.importorskip("PIL")
    # The width the route asks for: one cached size per asset, so a test at a
    # different width would fight the worker that regenerates at 480.
    made = LibraryStore(root).ensure_thumbnail(asset_id, width=service.THUMB_MAX_WIDTH, quality=82)
    assert made is not None and Path(made).is_file()

    served = service.thumbnail(root, asset_id, service.THUMB_MAX_WIDTH)
    assert served.content_type == "image/webp" and served.path == Path(made)


def test_a_thumbnail_is_invalidated_when_its_source_changes(tmp_path, monkeypatch) -> None:
    """``thumb_src`` is the source's identity: a rewritten image is not the old one."""
    root = config_root(monkeypatch, tmp_path)
    image = tmp_path / "photo.png"
    image.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
        b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
        b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    build(root, write_rows("w1", 1, image))
    asset_id = listing(root, service.Query())["rows"][0]["id"]
    store = LibraryStore(root)
    made = store.ensure_thumbnail(asset_id, width=service.THUMB_MAX_WIDTH, quality=82)
    if made is None:
        pytest.skip("Pillow is not installed")
    assert store.thumb_is_live(asset_id, service.THUMB_MAX_WIDTH) is not None

    # Same size and content CAN repeat; the mtime identity is what moves.
    image.write_bytes(image.read_bytes())
    assert store.thumb_is_live(asset_id, service.THUMB_MAX_WIDTH) is None


def test_a_just_indexed_asset_is_found_and_a_hit_never_scans(tmp_path, monkeypatch) -> None:
    """The watcher indexes a new session; a read then answers from SQLite.

    The old rule rebuilt the whole catalog on a miss. That rebuild WAS the
    request-path cost, so the miss path is now the indexer's job and a hit must
    stay a pure index read.
    """
    from local_operator.library import index as library_index

    root = config_root(monkeypatch, tmp_path)
    image = tmp_path / "photo.png"
    image.write_bytes(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
        b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05"
        b"\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    build(root, write_rows("w1", 1, image))
    image_id = listing(root, service.Query())["rows"][0]["id"]

    fresh = tmp_path / "fresh.txt"
    fresh.write_text("x")
    session(root, OTHER, write_rows("w9", 9, fresh))
    fresh_id = identity.asset_id(identity.key_for(str(fresh), None))
    # This is the file-event pass, played synchronously.
    indexer.rebuild_now(root)

    row = service.get_asset(root, fresh_id)
    assert row["name"] == "fresh.txt"

    def boom(*args, **kwargs):
        raise AssertionError("a read hit must not scan sessions")

    monkeypatch.setattr(library_index, "_visible_sessions", boom)
    thumb = service.thumbnail(root, image_id, 120)
    assert thumb.path.is_file()
    assert service.get_asset(root, fresh_id)["id"] == fresh_id


def test_scope_final_keeps_generated_and_highlighted_assets(tmp_path, monkeypatch) -> None:
    """The operator's kept set: generated artifacts and highlights, no scratch.

    A final listing narrows BEFORE metadata materialisation, so total, counts
    and the tag vocabulary describe only the rows it may show.
    """
    root = config_root(monkeypatch, tmp_path)
    block = cache_media(png(), "image/png", kind="image", name="cat.png")
    assert block and block.attachment
    session(root, SESSION, image_rows("g1", 1, block.attachment))

    mentioned = tmp_path / "mentioned.txt"
    mentioned.write_text("x")
    session(root, OTHER, [user_row("u1", 2, f"look at {mentioned}")])

    scratch = tmp_path / "scratch.txt"
    scratch.write_text("x")
    session(root, "111111111111", write_rows("w1", 3, scratch))

    kept = tmp_path / "kept.md"
    kept.write_text("# kept")
    rows = write_rows("w2", 4, kept)
    rows.append(
        {
            "id": "s1",
            "ts": 5,
            "type": "custom",
            "payload": {
                "kind": "custom",
                "custom_type": "supplement_v1",
                "details": {
                    "anchor": "turn-1",
                    "version": 1,
                    "state": "done",
                    "files": [{"path": str(kept), "name": "kept.md", "kind": "markdown"}],
                },
            },
        }
    )
    session(root, "222222222222", rows)
    indexer.rebuild_now(root)

    all_list = listing(root, service.Query())
    final = listing(root, service.Query(scope="final"))
    all_names = {row["name"] for row in all_list["rows"]}
    final_names = {row["name"] for row in final["rows"]}
    assert all_names == {"cat.png", "mentioned.txt", "scratch.txt", "kept.md"}
    assert final_names == {"cat.png", "kept.md"}
    assert (all_list["total"], final["total"]) == (4, 2)
    assert final["counts"]["origin_group"] == {"generated": 1, "created": 1}
    assert all_list["counts"]["origin_group"] == {
        "generated": 1,
        "created": 2,
        "mentioned": 1,
    }

    store = MetadataStore(root)
    keep_tag = store.create_tag("keep")
    scratch_tag = store.create_tag("scratch")
    by_name = {row["name"]: row["id"] for row in all_list["rows"]}
    service.set_tags(root, by_name["cat.png"], [keep_tag["id"]])
    service.set_tags(root, by_name["scratch.txt"], [scratch_tag["id"]])
    final = listing(root, service.Query(scope="final"))
    assert [tag["name"] for tag in final["tags"]] == ["keep"]
    assert final["tags"][0]["count"] == 1
    assert {tag["name"] for tag in listing(root, service.Query())["tags"]} == {
        "keep",
        "scratch",
    }

    first = service.list_assets(root, service.Query(scope="final", limit=1))
    assert len(first["rows"]) == 1 and first["next_cursor"]
    second = service.list_assets(
        root, service.Query(scope="final", limit=1, cursor=first["next_cursor"])
    )
    assert len(second["rows"]) == 1 and second["cursor_reset"] is False
    assert {first["rows"][0]["id"]}.isdisjoint({second["rows"][0]["id"]})

    # A cursor is bound to its scope; reusing an all-scope cursor for final is
    # a client error, exactly like any other changed query.
    all_first = service.list_assets(root, service.Query(limit=1))
    try:
        service.list_assets(root, service.Query(scope="final", cursor=all_first["next_cursor"]))
    except service.InvalidCursor:
        pass
    else:  # pragma: no cover
        raise AssertionError("a cursor minted for another scope must be refused")


def test_occurrence_session_title_is_the_stored_name_not_the_opening_line(
    tmp_path, monkeypatch
) -> None:
    """End to end through the service: the response carries the session's name.

    The UI renders ``occurrences[0].session_title`` as the conversation
    reference, so the index's precedence has to survive the projection. Asserted
    on the wire dict, not on the index internals.
    """
    root = config_root(monkeypatch, tmp_path)
    path = tmp_path / "brief.md"
    path.write_text("# hi")
    opening = "Can you review Victor's message in non-urgent-tech-support"
    build(root, [user_row("u0", 0.5, opening), *write_rows("w1", 1, path)])
    resume.write_session_title(
        root / "sessions" / SESSION, "Victor support triage", user_set=False, past_names=[]
    )
    # A rename moves no journal signature; the indexer re-reads the stored title
    # on the session pass, which is what this exercises.
    indexer.rebuild_now(root)

    row = listing(root, service.Query())["rows"][0]

    assert row["occurrences"][0]["session_title"] == "Victor support triage"


def test_a_rename_reaches_the_next_pass_without_a_journal_change(tmp_path, monkeypatch) -> None:
    """A rename moves no journal signature, so it is applied on the session pass.

    ``/rename`` writes only the title sidecar. The transcript, and therefore the
    signature the indexer compares, is untouched — before the stored title was
    part of the per-session check the Library kept showing the pre-rename label
    until the conversation was used again.
    """
    root = config_root(monkeypatch, tmp_path)
    path = tmp_path / "brief.md"
    path.write_text("# hi")
    opening = "Please review Victor's message"
    directory = session(root, SESSION, [user_row("u0", 0.5, opening), *write_rows("w1", 1, path)])
    indexer.rebuild_now(root)
    first = listing(root, service.Query())
    assert first["rows"][0]["occurrences"][0]["session_title"] == opening

    resume.write_session_title(
        directory, "Victor support triage", user_set=True, past_names=[opening]
    )
    revision_before = first["revision"]
    # ``budget_s`` is still accepted by the service — and ignored, because there
    # is no scan on the request path any more.
    indexer.rebuild_now(root)

    polled = service.list_assets(root, service.Query(), budget_s=library_index.DEFAULT_BUDGET_S)
    assert polled["rows"][0]["occurrences"][0]["session_title"] == "Victor support triage"
    assert polled["revision"] > revision_before
