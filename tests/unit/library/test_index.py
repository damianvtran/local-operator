from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.library import index
from local_operator.session.attachments import cache_media

from ._fixtures import OTHER, SESSION, config_root, image_rows, png, session, user_row, write_rows


def test_assets_dedupe_across_sessions_and_keep_occurrences(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    created = tmp_path / "made.txt"
    created.write_text("hi")
    session(root, SESSION, write_rows("w1", 10, created))
    session(root, OTHER, [user_row("u1", 20, f"look at {created}")])

    catalog = index.build_catalog(root, budget_s=None)
    assert catalog.state == "ready"
    assert catalog.sessions_total == 2 and catalog.sessions_scanned == 2
    assert len(catalog.assets) == 1
    asset = catalog.assets[0]
    assert {occurrence.session_id for occurrence in asset.occurrences} == {SESSION, OTHER}
    assert asset.origin == "written" and asset.origin_group == "created"


def test_digest_assets_are_generated_and_grouped(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    block = cache_media(png(), "image/png", kind="image", name="cat.png")
    assert block and block.attachment
    session(root, SESSION, image_rows("g1", 1, block.attachment))
    catalog = index.build_catalog(root, budget_s=None)
    asset = catalog.assets[0]
    assert asset.digest == block.attachment and asset.ref_type == "artifact"
    assert (asset.kind, asset.origin_group) == ("image", "generated")


def test_cold_budget_answers_building_then_completes(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1, tmp_path / "a.txt"))
    partial = index.build_catalog(root, budget_s=0)
    assert partial.state == "building"
    assert partial.sessions_scanned == 0
    finished = index.build_catalog(root, budget_s=None)
    assert finished.state == "ready" and finished.sessions_scanned == 1
    cached = index.build_catalog(root, budget_s=None)
    assert cached.revision == finished.revision
    assert cached.assets[0].id == finished.assets[0].id


def test_mention_origins_keep_the_strongest_and_highlight_is_a_flag(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    written = tmp_path / "done.md"
    written.write_text("# hi")
    rows = write_rows("w1", 1, written)
    rows.append(
        {
            "id": "s1",
            "ts": 2,
            "type": "custom",
            "payload": {
                "kind": "custom",
                "custom_type": "supplement_v1",
                "details": {
                    "anchor": "turn-1",
                    "version": 1,
                    "state": "done",
                    "files": [{"path": str(written), "name": "done.md", "kind": "markdown"}],
                },
            },
        }
    )
    session(root, SESSION, rows)
    asset = index.build_catalog(root, budget_s=None).assets[0]
    assert asset.origin == "written" and asset.origin_group == "created"
    assert asset.highlighted is True


def test_a_torn_index_is_discarded_without_raising(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    path = index.index_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    session(root, SESSION, write_rows("w1", 1, tmp_path / "a.txt"))
    assert index.build_catalog(root, budget_s=None).state == "ready"


def test_unreadable_journal_does_not_blank_other_sessions(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1, tmp_path / "a.txt"))
    broken = root / "sessions" / OTHER
    broken.mkdir(parents=True)
    (broken / "transcript.jsonl").write_bytes(b"\xff\xfe garbage\n")
    catalog = index.build_catalog(root, budget_s=None)
    assert catalog.state == "ready"
    assert len(catalog.assets) == 1
