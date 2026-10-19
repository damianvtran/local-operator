from __future__ import annotations

from pathlib import Path
from typing import Any

from local_operator.library import service
from local_operator.library.metadata import MetadataStore

from ._fixtures import SESSION, config_root, session, user_row, write_rows


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
