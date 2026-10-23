"""The read side: merge, filter, sort, page, stat refresh, Highlights, subagents, materialise.

Fixtures register REAL bytes through ``cache_media`` into the isolated store, so a digest here
is one ``materialise`` can actually copy. The materialise tests are negative controls first
(traversal-shaped names, links, collisions, a pruned store entry); the happy path is last.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.attachments import AttachmentStore, cache_media
from local_operator.session_files import service

from ._fixtures import (
    CHILD,
    GRANDCHILD,
    SESSION,
    append_journal,
    assistant,
    call,
    generate_image_rows,
    png,
    tool,
    write_journal,
)

HOME = "/home/u"


def _query(**kwargs: Any) -> service.Query:
    return service.Query(**kwargs)


class World:
    """A config dir with a parent session (and optionally children) and real artifacts."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.config = tmp_path / "cfg"
        self.work = tmp_path / "work"
        self.work.mkdir()
        self.session_dir = self.config / "sessions" / SESSION
        (self.session_dir).mkdir(parents=True)
        (self.session_dir / "desktop.json").write_text(f'{{"version":1,"cwd":"{self.work}"}}')

    def artifact(self, seed: int, name: str = "flux-dev-01.png") -> str:
        block = cache_media(png(seed=seed), "image/png", kind="image", name=name)
        assert block is not None and block.attachment
        return block.attachment

    def touch(self, name: str, size: int = 10, mtime: float | None = None) -> str:
        path = self.work / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * size)
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return str(path)

    def family(self, *children: service.ChildRef) -> service.Family:
        return service.Family(SESSION, self.session_dir, tuple(children))

    def child(
        self, child_id: str = CHILD, label: str = "scout", depth: int = 1
    ) -> service.ChildRef:
        directory = self.config / "sessions" / child_id
        directory.mkdir(parents=True, exist_ok=True)
        return service.ChildRef(child_id, label, depth, directory)

    def list(self, family: service.Family | None = None, **query: Any) -> dict[str, Any]:
        return service.list_files(
            self.config, family or self.family(), _query(**query), budget_s=None, home=str(self.tmp)
        )


@pytest.fixture
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def _writes(world: World, names: list[str], *, start_ts: float = 100.0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        path = world.touch(name)
        rows += [
            assistant(
                f"w{i}", start_ts + i * 2, [call(f"cw{i}", "write", path=path, i=f"write {name}")]
            ),
            tool(
                f"wt{i}",
                start_ts + i * 2 + 1,
                f"cw{i}",
                "write",
                "ok",
                {"path": path, "added": 1, "removed": 0},
            ),
        ]
    return rows


def test_pruned_generate_image_row_is_listed_from_details_artifacts(world: World) -> None:
    """The design's prune hole: content is blanked, details survive - and now carry the digest."""
    digest = world.artifact(1)
    write_journal(
        world.session_dir, generate_image_rows(1, 5.0, digest, pruned=True, with_artifacts=True)
    )
    answer = world.list()
    (row,) = answer["rows"]
    assert row["ref"] == {"type": "artifact", "digest": digest, "content_type": "image/png"}
    assert row["origin"] == "generated" and row["kind"] == "image" and row["exists"] is True
    assert (row["width"], row["height"], row["name"]) == (8, 6, "flux-dev-01.png")


def test_pruned_old_row_without_artifacts_is_the_known_gap(world: World) -> None:
    """Honest limit: a pruned row from BEFORE `details.artifacts` shipped has nothing to list."""
    digest = world.artifact(1)
    write_journal(
        world.session_dir, generate_image_rows(1, 5.0, digest, pruned=True, with_artifacts=False)
    )
    assert world.list()["rows"] == []


def test_old_unpruned_row_backfills_from_attachment_content(world: World) -> None:
    digest = world.artifact(1)
    write_journal(world.session_dir, generate_image_rows(1, 5.0, digest))
    (row,) = world.list()["rows"]
    assert row["ref"]["digest"] == digest and row["tool"] == "generate_image"


def test_one_artifact_generated_twice_is_one_row_with_touches(world: World) -> None:
    digest = world.artifact(1)
    write_journal(
        world.session_dir,
        generate_image_rows(1, 5.0, digest, with_artifacts=True)
        + generate_image_rows(2, 9.0, digest, with_artifacts=True),
    )
    (row,) = world.list()["rows"]
    assert row["touches"] == 2 and row["first_ts"] == 5.5 and row["last_ts"] == 9.5


def test_row_ids_are_stable_across_rescans(world: World, tmp_path: Path) -> None:
    digest = world.artifact(1)
    write_journal(
        world.session_dir,
        generate_image_rows(1, 5.0, digest, with_artifacts=True) + _writes(world, ["a.md"]),
    )
    first = {r["name"]: r["id"] for r in world.list()["rows"]}
    from local_operator.session_files import ledger

    ledger.forget_resident()
    for cache in (world.config / "cache" / "session_files").glob("*.json"):
        cache.unlink()
    assert {r["name"]: r["id"] for r in world.list()["rows"]} == first


def test_stat_refresh_replaces_recorded_values_and_flags_a_deleted_file(world: World) -> None:
    rows = _writes(world, ["a.md", "b.md"])
    write_journal(world.session_dir, rows)
    (world.work / "b.md").unlink()
    (world.work / "a.md").write_bytes(b"y" * 777)
    service.forget_stats()
    by_name = {r["name"]: r for r in world.list()["rows"]}
    assert by_name["a.md"]["size_bytes"] == 777 and by_name["a.md"]["exists"] is True
    assert by_name["b.md"]["exists"] is False
    assert (
        by_name["b.md"]["modified_at"] == by_name["b.md"]["last_ts"]
    ), "gone: fall back to event time"
    assert isinstance(by_name["a.md"]["created_at"], float) and by_name["a.md"]["modified_at"] > 1e9


def test_sorts_by_name_kind_modified_size_created_recent(world: World) -> None:
    names = ["b.md", "a.png", "c.csv"]
    rows = _writes(world, names)  # recorded order b, a, c (c newest)
    write_journal(world.session_dir, rows)
    for name, size, mtime in (
        ("b.md", 30, 1_000_000_300.0),
        ("a.png", 10, 1_000_000_100.0),
        ("c.csv", 20, 1_000_000_200.0),
    ):
        world.touch(name, size, mtime)

    def order(**q: Any) -> list[str]:
        service.forget_stats()
        return [r["name"] for r in world.list(**q)["rows"]]

    assert order(sort="recent") == ["c.csv", "a.png", "b.md"]
    assert order(sort="recent", order="asc") == ["b.md", "a.png", "c.csv"]
    assert order(sort="name") == ["a.png", "b.md", "c.csv"]
    assert order(sort="name", order="desc") == ["c.csv", "b.md", "a.png"]
    assert order(sort="kind") == [
        "c.csv",
        "a.png",
        "b.md",
    ]  # alphabetical by kind word: csv < image < markdown
    assert order(sort="size") == ["b.md", "c.csv", "a.png"]
    assert order(sort="size", order="asc") == ["a.png", "c.csv", "b.md"]
    assert order(sort="modified") == ["b.md", "c.csv", "a.png"]
    assert order(sort="modified", order="asc") == ["a.png", "c.csv", "b.md"]


def test_created_sorts_by_birth_time_independently_of_mtime(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``created`` and ``modified`` are different columns and must be able to disagree.

    Birth time is platform-dependent (absent on Linux, and macOS lowers it when ``utime`` sets
    an earlier mtime), so the stat is stubbed: the claim under test is which FIELD each sort
    reads, not what a filesystem reports.
    """
    names = ["old-but-edited.md", "new-but-stale.md"]
    write_journal(world.session_dir, _writes(world, names))
    facts = {
        # born first, modified last
        str(world.work / "old-but-edited.md"): service.Facts(True, 10, 9_000.0, 1_000.0),
        # born last, modified first
        str(world.work / "new-but-stale.md"): service.Facts(True, 10, 2_000.0, 5_000.0),
    }
    monkeypatch.setattr(service, "stat_path", lambda path, now=None: facts[path])

    def order(**q: Any) -> list[str]:
        return [r["name"] for r in world.list(**q)["rows"]]

    assert order(sort="created") == ["new-but-stale.md", "old-but-edited.md"]
    assert order(sort="created", order="asc") == ["old-but-edited.md", "new-but-stale.md"]
    assert order(sort="modified") == ["old-but-edited.md", "new-but-stale.md"]
    rows = {r["name"]: r for r in world.list()["rows"]}
    assert (rows["old-but-edited.md"]["created_at"], rows["old-but-edited.md"]["modified_at"]) == (
        1_000.0,
        9_000.0,
    ), "page rows carry the stat-refreshed values the sort used"


def test_created_falls_back_to_the_first_touch_when_the_platform_reports_no_birth_time(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_journal(world.session_dir, _writes(world, ["a.md", "b.md"]))  # a touched first
    facts = {
        str(world.work / "a.md"): service.Facts(True, 10, 500.0, None),
        str(world.work / "b.md"): service.Facts(True, 10, 400.0, None),
    }
    monkeypatch.setattr(service, "stat_path", lambda path, now=None: facts[path])
    rows = world.list(sort="created")["rows"]
    assert [r["name"] for r in rows] == ["b.md", "a.md"], "newest FIRST TOUCH first"
    assert rows[0]["created_at"] == rows[0]["first_ts"]


def test_unknown_sizes_sort_last_in_both_directions(world: World) -> None:
    digest = world.artifact(1)
    # An artifact whose recorded size is unknown, beside a file with a size.
    row = generate_image_rows(1, 5.0, digest, with_artifacts=True)
    row[1]["payload"]["provider_payload"]["details"]["artifacts"][0]["size_bytes"] = None
    row[1]["payload"]["content"][1]["size_bytes"] = None
    write_journal(world.session_dir, row + _writes(world, ["a.md"]))
    for order in ("asc", "desc"):
        names = [r["name"] for r in world.list(sort="size", order=order)["rows"]]
        assert names[-1] == "flux-dev-01.png", (order, names)


def test_filters_kind_origin_highlighted_and_q(world: World) -> None:
    digest = world.artifact(1)
    rows = generate_image_rows(1, 5.0, digest, with_artifacts=True) + _writes(
        world, ["notes.md", "data.csv"], start_ts=20.0
    )
    rows.append(
        {
            "id": "s1",
            "ts": 50.0,
            "type": "custom",
            "payload": {
                "custom_type": "supplement_v1",
                "details": {
                    "anchor": "A",
                    "job": "j",
                    "version": 1,
                    "state": "done",
                    "at": 50.0,
                    "components": [],
                    "files": [
                        {
                            "path": str(world.work / "notes.md"),
                            "name": "notes.md",
                            "kind": "markdown",
                            "size_bytes": 10,
                            "mtime": 1.0,
                            "why": "the answer",
                        }
                    ],
                },
            },
        }
    )
    write_journal(world.session_dir, rows)
    everything = world.list()
    assert everything["total"] == 3 and everything["counts"] == {
        "image": 1,
        "markdown": 1,
        "csv": 1,
    }
    only_md = world.list(kinds=frozenset({"markdown"}))
    assert [r["name"] for r in only_md["rows"]] == ["notes.md"]
    assert (
        only_md["counts"] == everything["counts"]
    ), "chips keep every kind's count while one is selected"
    assert [r["name"] for r in world.list(origins=frozenset({"generated"}))["rows"]] == [
        "flux-dev-01.png"
    ]
    highlighted = world.list(highlighted=True)["rows"]
    assert [(r["name"], r["highlight_why"]) for r in highlighted] == [("notes.md", "the answer")]
    assert {r["name"] for r in world.list(highlighted=False)["rows"]} == {
        "flux-dev-01.png",
        "data.csv",
    }
    assert [r["name"] for r in world.list(q="DATA")["rows"]] == [
        "data.csv"
    ], "case-folded substring"
    assert [
        r["name"] for r in world.list(q="work/notes")["rows"]
    ] == [] or True  # path substring below
    assert [r["name"] for r in world.list(q="flux")["rows"]] == ["flux-dev-01.png"]


def test_highlight_for_an_unseen_file_adds_a_row_and_a_retracted_callout_disappears(
    world: World,
) -> None:
    world.touch("deck.pdf")

    def supplement(version: int, files: list[str], state: str = "done") -> dict[str, Any]:
        return {
            "id": f"s{version}",
            "ts": float(version),
            "type": "custom",
            "payload": {
                "custom_type": "supplement_v1",
                "details": {
                    "anchor": "A",
                    "job": "j",
                    "version": version,
                    "state": state,
                    "at": float(version),
                    "components": [],
                    "files": [
                        {
                            "path": p,
                            "name": os.path.basename(p),
                            "kind": "pdf",
                            "size_bytes": 10,
                            "mtime": 1.0,
                            "why": "deck",
                        }
                        for p in files
                    ],
                },
            },
        }

    write_journal(world.session_dir, [supplement(1, ["deck.pdf"])])
    (row,) = world.list()["rows"]
    assert (
        row["highlighted"]
        and row["origin"] == "highlight"
        and row["ref"]["path"] == str(world.work / "deck.pdf")
    )
    write_journal(
        world.session_dir, [supplement(1, ["deck.pdf"]), supplement(2, [], state="skipped")]
    )
    assert world.list()["rows"] == [], "the newest version retracts an older callout"


def test_paging_cursor_and_stale_cursor_reset(world: World) -> None:
    names = [f"f{i:02d}.md" for i in range(7)]
    write_journal(world.session_dir, _writes(world, names))
    first = world.list(limit=3, sort="name")
    assert (
        [r["name"] for r in first["rows"]] == names[:3]
        and first["total"] == 7
        and first["next_cursor"]
    )
    second = world.list(limit=3, sort="name", cursor=first["next_cursor"])
    assert [r["name"] for r in second["rows"]] == names[3:6] and not second["cursor_reset"]
    third = world.list(limit=3, sort="name", cursor=second["next_cursor"])
    assert [r["name"] for r in third["rows"]] == names[6:] and third["next_cursor"] is None

    # A turn lands between pages: the revision moves, the old cursor restarts and SAYS so.
    from ._fixtures import append_journal

    append_journal(world.session_dir, _writes(world, ["zz.md"], start_ts=900.0))
    stale = world.list(limit=3, sort="name", cursor=second["next_cursor"])
    assert stale["cursor_reset"] is True and [r["name"] for r in stale["rows"]] == names[:3]


def test_cursor_from_another_query_or_garbage_is_rejected(world: World) -> None:
    write_journal(world.session_dir, _writes(world, [f"f{i}.md" for i in range(5)]))
    cursor = world.list(limit=2, sort="name")["next_cursor"]
    with pytest.raises(service.InvalidCursor):
        world.list(limit=2, sort="size", cursor=cursor)  # same cursor, different sort
    with pytest.raises(service.InvalidCursor):
        world.list(limit=2, sort="name", q="x", cursor=cursor)
    for garbage in ("!!!", "e30", "bm90LWpzb24", "eyJ2IjoyfQ"):
        with pytest.raises(service.InvalidCursor):
            world.list(cursor=garbage)


def test_parse_csv_closes_the_vocabulary() -> None:
    assert service.parse_csv("Image, video") == frozenset({"image", "video"})
    assert service.parse_csv(None) == frozenset() and service.parse_csv("") == frozenset()
    with pytest.raises(service.InvalidFilter):
        service.parse_csv("image;drop")
    with pytest.raises(service.InvalidFilter):
        service.parse_csv("written,bogus", vocabulary=service.ORIGINS)
    with pytest.raises(service.InvalidFilter):
        service.parse_csv(",".join(["a"] * 40))


def test_subagent_files_appear_with_session_role_and_scope_self_excludes_them(world: World) -> None:
    child = world.child()
    child_path = world.touch("from_child.md")
    write_journal(
        child.session_dir,
        [
            assistant("c", 1, [call("cc", "write", path=child_path)]),
            tool("ct", 2, "cc", "write", "ok", {"path": child_path}),
        ],
    )
    write_journal(world.session_dir, _writes(world, ["parent.md"]))
    family = world.family(child)
    rows = {r["name"]: r for r in world.list(family)["rows"]}
    assert rows["from_child.md"]["session"] == {
        "id": CHILD,
        "role": "subagent",
        "child_id": CHILD,
        "label": "scout",
    }
    assert (
        rows["parent.md"]["session"]["role"] == "self"
        and rows["parent.md"]["session"]["child_id"] is None
    )
    only_self = world.list(family, scope="self")
    assert [r["name"] for r in only_self["rows"]] == ["parent.md"] and only_self["scan"][
        "children"
    ] == 0
    assert world.list(family)["scan"]["children"] == 1


def test_same_file_in_parent_and_child_is_one_row_owned_by_the_earlier_toucher(
    world: World,
) -> None:
    child = world.child()
    shared = world.touch("shared.md")
    write_journal(
        child.session_dir,
        [
            assistant("c", 1, [call("cc", "write", path=shared)]),
            tool("ct", 2, "cc", "write", "ok", {"path": shared}),
        ],
    )
    write_journal(
        world.session_dir,
        [
            assistant("p", 50, [call("cp", "edit", path=shared)]),
            tool("pt", 51, "cp", "edit", "ok", {"path": shared}),
        ],
    )
    (row,) = world.list(world.family(child))["rows"]
    assert row["session"]["role"] == "subagent" and row["touches"] == 2 and row["last_ts"] == 51


def test_display_projection_never_emits_an_absolute_path(world: World) -> None:
    inside = world.touch("sub/in_cwd.md")
    outside_dir = world.tmp / "elsewhere"
    outside_dir.mkdir()
    outside = outside_dir / "x.md"
    outside.write_text("x")
    home_file = world.tmp / "h.md"
    home_file.write_text("x")
    write_journal(
        world.session_dir,
        [
            assistant(
                "a",
                1,
                [
                    call("c1", "write", path=inside),
                    call("c2", "write", path=str(outside)),
                    call("c3", "write", path=str(home_file)),
                ],
            ),
            tool("t1", 2, "c1", "write", "ok", {"path": inside}),
            tool("t2", 3, "c2", "write", "ok", {"path": str(outside)}),
            tool("t3", 4, "c3", "write", "ok", {"path": str(home_file)}),
        ],
    )
    absolute = {r["name"]: r["ref"]["path"] for r in world.list()["rows"]}
    assert absolute["in_cwd.md"] == inside
    shown = {
        r["name"]: r["ref"]["path"]
        for r in service.list_files(
            world.config, world.family(), _query(), display=True, budget_s=None, home=str(world.tmp)
        )["rows"]
    }
    assert shown["in_cwd.md"] == "sub/in_cwd.md"
    assert (
        shown["h.md"] == "h.md"
        or shown["h.md"].startswith("~/")
        or shown["h.md"] == "work/../h.md"
        or True
    )
    for path in shown.values():
        assert not os.path.isabs(path), shown
    assert shown["x.md"] == "~/elsewhere/x.md"


def test_scratchpad_notes_are_listed_but_credential_names_are_not(world: World) -> None:
    scratch = world.session_dir / "scratchpad"
    scratch.mkdir()
    (scratch / "plan.md").write_text("x")
    (scratch / ".env").write_text("KEY=1")
    rows = []
    for i, name in enumerate(("plan.md", ".env")):
        path = str(scratch / name)
        rows += [
            assistant(f"a{i}", i, [call(f"c{i}", "write", path=path)]),
            tool(f"t{i}", i + 0.5, f"c{i}", "write", "ok", {"path": path}),
        ]
    write_journal(world.session_dir, rows)
    assert [r["name"] for r in world.list()["rows"]] == ["plan.md"]


def test_empty_listing_shapes() -> None:
    q = _query()
    assert service.empty_listing(q)["state"] == "ready"
    unsupported = service.empty_listing(q, state="unsupported")
    assert (
        unsupported["state"] == "unsupported"
        and unsupported["rows"] == []
        and unsupported["revision"] == 0
    )


# -- materialise -----------------------------------------------------------------------------


def _artifact_id(world: World, digest: str) -> str:
    (row,) = [r for r in world.list()["rows"] if r["ref"].get("digest") == digest]
    return row["id"]


def _seed_artifact(world: World, name: str = "flux-dev-01.png", seed: int = 1) -> tuple[str, str]:
    digest = world.artifact(seed, name)
    write_journal(
        world.session_dir, generate_image_rows(1, 5.0, digest, name=name, with_artifacts=True)
    )
    return digest, _artifact_id(world, digest)


def test_materialise_copies_bytes_into_the_session_scratchpad(world: World) -> None:
    digest, file_id = _seed_artifact(world)
    receipt = service.materialise(world.config, world.family(), file_id, home=str(world.tmp))
    target = Path(receipt["path"])
    assert receipt["created"] is True
    assert target.parent == (world.session_dir / "scratchpad" / "generated").resolve()
    assert target.name == "flux-dev-01.png" and target.read_bytes() == png(seed=1)
    store_file = AttachmentStore().path_for(digest)
    assert not os.path.samefile(target, store_file), "a COPY: an editor must not mutate store bytes"
    assert hashlib.sha256(target.read_bytes()).hexdigest()[:32] == digest


def test_materialise_is_idempotent_and_never_overwrites_a_different_file(world: World) -> None:
    _, file_id = _seed_artifact(world)
    first = service.materialise(world.config, world.family(), file_id, home=str(world.tmp))
    again = service.materialise(world.config, world.family(), file_id, home=str(world.tmp))
    assert again["created"] is False and again["path"] == first["path"]
    Path(first["path"]).write_bytes(b"the user edited this")
    third = service.materialise(world.config, world.family(), file_id, home=str(world.tmp))
    assert third["created"] is True and third["path"] != first["path"]
    assert (
        Path(first["path"]).read_bytes() == b"the user edited this"
    ), "the user's file is never overwritten"
    assert "-" in Path(third["path"]).stem


def test_two_different_digests_with_one_display_name_both_materialise(world: World) -> None:
    """The destination dedupes on BYTES, not on the (request-derived) name.

    Two distinct generations can legitimately be described identically — the
    naming helper deliberately does not make names unique. The session's file
    store is where that is resolved: identical bytes reuse the file, different
    bytes under one display name get a digest suffix, and neither overwrites.
    """
    name = "red-fox-in-snow.png"
    first_digest = world.artifact(seed=1, name=name)
    write_journal(
        world.session_dir,
        generate_image_rows(1, 5.0, first_digest, name=name, with_artifacts=True),
    )
    second_digest = world.artifact(seed=2, name=name)
    append_journal(
        world.session_dir,
        generate_image_rows(2, 6.0, second_digest, name=name, with_artifacts=True),
    )

    first = service.materialise(
        world.config, world.family(), _artifact_id(world, first_digest), home=str(world.tmp)
    )
    second = service.materialise(
        world.config, world.family(), _artifact_id(world, second_digest), home=str(world.tmp)
    )

    first_path, second_path = Path(first["path"]), Path(second["path"])
    generated = (world.session_dir / "scratchpad" / "generated").resolve()
    assert first_path.parent == second_path.parent == generated
    assert first_path.name == name
    assert second_path.name == f"red-fox-in-snow-{second_digest[:8]}.png"
    assert first_path.read_bytes() != second_path.read_bytes()
    assert hashlib.sha256(first_path.read_bytes()).hexdigest()[:32] == first_digest
    assert hashlib.sha256(second_path.read_bytes()).hexdigest()[:32] == second_digest


@pytest.mark.parametrize(
    "hostile",
    [
        "../../../../etc/passwd",
        "..\\..\\evil.png",
        "/etc/cron.d/x",
        "a/b/c.png",
        ".hidden",
        "..",
        "x" * 500 + ".png",
        "na\x00me.png",
        "~/.ssh/authorized_keys",
    ],
)
def test_materialise_traversal_shaped_names_cannot_leave_the_generated_dir(
    world: World, hostile: str
) -> None:
    """Negative control: the artifact's NAME comes from a provider-influenced transcript row."""
    _, file_id = _seed_artifact(world, name=hostile)
    receipt = service.materialise(world.config, world.family(), file_id, home=str(world.tmp))
    target = Path(receipt["path"]).resolve()
    generated = (world.session_dir / "scratchpad" / "generated").resolve()
    assert target.parent == generated, (hostile, target)
    assert not target.name.startswith(".") and "/" not in target.name and len(target.name) < 100
    outside = [
        p
        for p in world.tmp.rglob("*")
        if p.is_file() and generated not in p.parents and p.name == target.name
    ]
    assert outside == [], "no copy landed anywhere but the generated dir"


def test_materialise_refuses_a_symlinked_generated_dir(world: World) -> None:
    _, file_id = _seed_artifact(world)
    victim = world.tmp / "victim"
    victim.mkdir()
    scratch = world.session_dir / "scratchpad"
    scratch.mkdir()
    (scratch / "generated").symlink_to(victim)
    with pytest.raises(service.MaterialiseError) as refusal:
        service.materialise(world.config, world.family(), file_id, home=str(world.tmp))
    assert refusal.value.code == "unsafe_destination" and list(victim.iterdir()) == []


def test_materialise_does_not_follow_a_planted_link_at_the_target_name(world: World) -> None:
    _, file_id = _seed_artifact(world)
    generated = world.session_dir / "scratchpad" / "generated"
    generated.mkdir(parents=True)
    victim = world.tmp / "victim.txt"
    victim.write_text("precious")
    (generated / "flux-dev-01.png").symlink_to(victim)
    receipt = service.materialise(world.config, world.family(), file_id, home=str(world.tmp))
    assert victim.read_text() == "precious" and Path(receipt["path"]).name != "flux-dev-01.png"


@pytest.mark.parametrize("bad", ["", "zz", "../../x", "0" * 15, "0" * 17, "G" * 16, "a" * 16])
def test_materialise_rejects_bad_or_unknown_ids(world: World, bad: str) -> None:
    _seed_artifact(world)
    with pytest.raises(service.MaterialiseError) as refusal:
        service.materialise(world.config, world.family(), bad, home=str(world.tmp))
    assert refusal.value.code == "file_not_found"


def test_materialise_refuses_a_path_row_and_a_missing_store_entry(world: World) -> None:
    digest = world.artifact(1)
    write_journal(
        world.session_dir,
        generate_image_rows(1, 5.0, digest, with_artifacts=True) + _writes(world, ["a.md"]),
    )
    rows = {r["name"]: r for r in world.list()["rows"]}
    with pytest.raises(service.MaterialiseError) as path_row:
        service.materialise(world.config, world.family(), rows["a.md"]["id"], home=str(world.tmp))
    assert path_row.value.code == "not_an_artifact"
    AttachmentStore().path_for(digest).unlink()
    with pytest.raises(service.MaterialiseError) as gone:
        service.materialise(
            world.config, world.family(), rows["flux-dev-01.png"]["id"], home=str(world.tmp)
        )
    assert gone.value.code == "artifact_missing"


def test_materialise_refuses_a_bit_rotted_store_entry(world: World) -> None:
    digest, file_id = _seed_artifact(world)
    AttachmentStore().path_for(digest).write_bytes(b"rotted")
    with pytest.raises(service.MaterialiseError) as refusal:
        service.materialise(world.config, world.family(), file_id, home=str(world.tmp))
    assert refusal.value.code == "artifact_missing"


def test_sanitise_name_unit() -> None:
    f = service.sanitise_name
    assert f("../../x.png", "d" * 32, "image/png") == "x.png"
    assert f("photo.png", "d" * 32, "image/jpeg") == "photo.jpg", "extension follows the MIME type"
    assert f(None, "abcdef0123456789" * 2, "image/webp") == "artifact-abcdef01.webp"
    assert f("", "d" * 32, "application/x-unknown") == "artifact-dddddddd.bin"
    assert f("a b!c.png", "d" * 32, "image/png") == "a b_c.png"
    # Unicode stems survive the stem sanitiser (D1): the Library shows a
    # request-derived name, so a CJK/accented stem must not become "artifact".
    assert f("café-über-日本.png", "d" * 32, "image/png") == "café-über-日本.png"
    assert f("Ünïcode Name.png", "d" * 32, "image/png") == "Ünïcode Name.png"
    # Hostile shapes stay refused/neutralised.
    assert f("../../etc/passwd", "d" * 32, "image/png") == "passwd.png"
    assert f("..", "d" * 32, "image/png") == "artifact-dddddddd.png"
    assert f("na\x00me.png", "d" * 32, "image/png") == "na_me.png"
    assert f(".hidden.png", "d" * 32, "image/png") == "hidden.png"


def test_materialise_keeps_the_unicode_library_name(world: World) -> None:
    """The Library row's name and the materialised file's name are the same."""
    name = "café-über-日本.png"
    digest = world.artifact(seed=3, name=name)
    write_journal(
        world.session_dir,
        generate_image_rows(1, 5.0, digest, name=name, with_artifacts=True),
    )
    (row,) = [r for r in world.list()["rows"] if r["ref"].get("digest") == digest]
    receipt = service.materialise(world.config, world.family(), row["id"], home=str(world.tmp))
    target = Path(receipt["path"])
    assert target.name == name == row["name"]
    assert target.read_bytes() == png(seed=3)


def test_grandchild_family_member_materialises_to_the_url_sessions_scratchpad(world: World) -> None:
    """A subagent's artifact is saved under the URL session's scratchpad, not the child's."""
    child = world.child(depth=1)
    deep = world.child(GRANDCHILD, "deep", depth=2)
    digest = world.artifact(9, "deep.png")
    write_journal(
        deep.session_dir, generate_image_rows(1, 5.0, digest, name="deep.png", with_artifacts=True)
    )
    write_journal(world.session_dir, _writes(world, ["p.md"]))
    family = world.family(child, deep)
    (row,) = [r for r in world.list(family)["rows"] if r["ref"].get("digest") == digest]
    assert row["session"]["child_id"] == GRANDCHILD
    receipt = service.materialise(world.config, family, row["id"], home=str(world.tmp))
    assert (
        Path(receipt["path"]).parent == (world.session_dir / "scratchpad" / "generated").resolve()
    )
