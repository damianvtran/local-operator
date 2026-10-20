from __future__ import annotations

from pathlib import Path
from typing import Any

from local_operator import resume
from local_operator.library import identity, service
from local_operator.library.metadata import MetadataStore
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
    session(root, SESSION, rows)


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


def test_thumbnail_reads_the_persisted_catalog_without_a_session_scan(
    tmp_path, monkeypatch
) -> None:
    """Content/thumbnail/get must not pay a journal scan or an index rewrite.

    A 60-card grid issues 60 thumbnail requests; each one running
    ``build_catalog`` is what pushed a list past the desktop's 20 s deadline.
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

    # A fresh process: no in-process aggregate, but the index file survives.
    library_index._CACHE.clear()
    scans: list[int] = []
    asset_builds: list[int] = []
    original = library_index._build_assets
    monkeypatch.setattr(
        library_index,
        "_visible_sessions",
        lambda *args, **kwargs: scans.append(1) or [],
    )
    monkeypatch.setattr(
        library_index,
        "_build_assets",
        lambda sessions: asset_builds.append(1) or original(sessions),
    )

    first = service.thumbnail(root, asset_id, 120)
    second = service.thumbnail(root, asset_id, 120)

    assert scans == [], "a thumbnail must not enumerate sessions"
    assert asset_builds == [1], "assets are built once per persisted index"
    assert first.path == second.path and first.path.is_file()


def test_a_read_miss_refreshes_once_for_a_just_indexed_asset(tmp_path, monkeypatch) -> None:
    """A new asset is found without a rebuild, and the hit path still never scans."""
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

    # Indexed after the last write: the read path has never seen this row.
    fresh = tmp_path / "fresh.txt"
    fresh.write_text("x")
    session(root, OTHER, write_rows("w9", 9, fresh))
    fresh_id = identity.asset_id(identity.key_for(str(fresh), None))

    row = service.get_asset(root, fresh_id)
    assert row["name"] == "fresh.txt"

    def boom(*args, **kwargs):
        raise AssertionError("a read hit must not scan sessions")

    monkeypatch.setattr(library_index, "_visible_sessions", boom)
    thumb = service.thumbnail(root, image_id, 120)
    assert thumb.path.is_file()


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

    row = listing(root, service.Query())["rows"][0]

    assert row["occurrences"][0]["session_title"] == "Victor support triage"
