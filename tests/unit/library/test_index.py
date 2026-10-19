from __future__ import annotations

import json
import os
import time

import pytest

from local_operator.library import identity, index, service
from local_operator.session.attachments import cache_media

from ._fixtures import (
    CHILD,
    OTHER,
    SESSION,
    config_root,
    image_rows,
    png,
    session,
    user_row,
    write_rows,
)


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


def test_a_child_append_is_seen_without_deleting_the_cache(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    parent = session(root, SESSION, write_rows("p1", 1, tmp_path / "parent.txt"))
    child = session(root, CHILD, write_rows("c1", 2, tmp_path / "child-one.txt"))
    (child / "origin.json").write_text(json.dumps({"origin": "subagent"}))
    (parent / "subagent-roster.v1.json").write_text(
        json.dumps(
            {
                "version": 1,
                "generation": 1,
                "jobs": [],
                "accounting": {},
                "records": [
                    {
                        "job_id": "j1",
                        "label": "scout",
                        "agent_role": "scout",
                        "depth": 1,
                        "session_dir": str(child),
                    }
                ],
            }
        )
    )
    first = index.build_catalog(root, budget_s=None)
    assert {asset.derived_name for asset in first.assets} == {"parent.txt", "child-one.txt"}
    assert index.index_path(root).is_file()

    append = write_rows("c2", 5, tmp_path / "child-two.txt")
    with (child / "transcript.jsonl").open("ab") as handle:
        handle.write(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in append).encode()
        )
    second = index.build_catalog(root, budget_s=None)
    assert {asset.derived_name for asset in second.assets} == {
        "parent.txt",
        "child-one.txt",
        "child-two.txt",
    }
    assert second.revision != first.revision
    assert index.index_path(root).is_file(), "the incremental path must not force a rebuild"


def test_a_roster_only_change_refreshes_the_family(monkeypatch) -> None:
    """A membership edit with an unchanged parent journal must be seen.

    The roster sidecar is the authoritative family list; the parent transcript
    can stay byte-identical while a new child (or grandchild) run is added.
    """
    import tempfile
    from pathlib import Path

    work = Path(tempfile.mkdtemp())
    root = config_root(monkeypatch, work)
    parent = session(root, SESSION, write_rows("p1", 1, work / "parent.txt"))
    first = session(root, CHILD, write_rows("c1", 2, work / "child-one.txt"))
    (first / "origin.json").write_text(json.dumps({"origin": "subagent"}))
    second_id = "111111111111"
    second = session(root, second_id, write_rows("g1", 3, work / "grandchild.txt"))
    (second / "origin.json").write_text(json.dumps({"origin": "subagent"}))

    def roster(extra: bool) -> None:
        records = [
            {
                "job_id": "j1",
                "label": "scout",
                "agent_role": "scout",
                "depth": 1,
                "session_dir": str(first),
            }
        ]
        if extra:
            records.append(
                {
                    "job_id": "j2",
                    "label": "grandchild",
                    "agent_role": "scout",
                    "depth": 2,
                    "session_dir": str(second),
                }
            )
        (parent / "subagent-roster.v1.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "generation": 1,
                    "jobs": [],
                    "accounting": {},
                    "records": records,
                }
            )
        )

    roster(False)
    first_build = index.build_catalog(root, budget_s=None)
    assert {asset.derived_name for asset in first_build.assets} == {
        "parent.txt",
        "child-one.txt",
    }
    parent_bytes = (parent / "transcript.jsonl").read_bytes()

    roster(True)
    second_build = index.build_catalog(root, budget_s=None)
    assert (parent / "transcript.jsonl").read_bytes() == parent_bytes, "parent must be unchanged"
    assert {asset.derived_name for asset in second_build.assets} == {
        "parent.txt",
        "child-one.txt",
        "grandchild.txt",
    }
    assert second_build.revision != first_build.revision

    # Removing the record retracts the journal from the family on the same path.
    roster(False)
    third_build = index.build_catalog(root, budget_s=None)
    assert {asset.derived_name for asset in third_build.assets} == {
        "parent.txt",
        "child-one.txt",
    }


def test_a_second_build_on_an_unchanged_index_does_not_write(tmp_path, monkeypatch) -> None:
    """The aggregate is derived from the index; an unchanged index costs no dump.

    ``state == "ready"`` used to force an 11 MB rewrite on every request, which
    was most of the measured 5.3 s warm build on the operator's corpus.
    """
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1, tmp_path / "a.txt"))
    first = index.build_catalog(root, budget_s=None)

    writes: list[object] = []
    monkeypatch.setattr(index, "_write", lambda *args, **kwargs: writes.append(args))
    second = index.build_catalog(root, budget_s=None)

    assert writes == []
    assert second is first


def test_a_warm_request_path_serves_the_cache_without_a_sweep(tmp_path, monkeypatch) -> None:
    """Inside the freshness window a poll touches neither journals nor assets."""
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1, tmp_path / "a.txt"))
    first = index.build_catalog(root, budget_s=None)

    def boom(*args, **kwargs):
        raise AssertionError("a warm poll must not scan or rebuild")

    monkeypatch.setattr(index, "_visible_sessions", boom)
    monkeypatch.setattr(index, "_build_assets", boom)

    assert index.build_catalog(root) is first


def test_a_partial_merge_keeps_complete_sessions_from_disk(tmp_path, monkeypatch) -> None:
    """A budget-limited build must not truncate another process's progress.

    The desktop backend, the mobile relay and ``lop serve`` all build this
    index; a partial build from one used to overwrite the others' complete rows
    (observed collapsing 490 sessions to 1).
    """
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1, tmp_path / "a.txt"))
    session(root, OTHER, write_rows("w2", 2, tmp_path / "b.txt"))
    index.build_catalog(root, budget_s=None)
    disk = index._load(index.index_path(root))["sessions"]
    assert {SESSION, OTHER} <= set(disk)

    # What a partial build that only reached SESSION would hand to the writer.
    incoming = {SESSION: disk[SESSION]}
    merged = index._merge_sessions(disk, incoming, keep={SESSION, OTHER})

    assert set(merged) == {SESSION, OTHER}
    assert merged[OTHER]["complete"] is True
    # Where both sides have an entry, this build's scan wins regardless of the
    # revision comparison (a retraction lowers it below the stale disk copy).
    stale = {"complete": True, "revision": 99.0}
    fresh = {"complete": True, "revision": 1.0}
    assert index._merge_sessions({"s": stale}, {"s": fresh}, keep={"s"})["s"] is fresh


def _payload_with_extra(root, extra: "os.PathLike[str]") -> dict:
    """A newer index written by "another process": one session, one asset row."""
    record = {
        "key": identity.key_for(str(extra), None),
        "path": str(extra),
        "digest": None,
        "name": extra.name,
        "kind": "document",
        "content_type": "text/plain",
        "size": 1,
        "width": None,
        "height": None,
        "duration_s": None,
        "origin": "written",
        "tool": "write",
        "call_id": None,
        "intent": None,
        "first_seq": 1,
        "last_seq": 1,
        "first_ts": 1.0,
        "last_ts": 1.0,
        "touches": 1,
        "highlighted": False,
    }
    return {
        "version": index.INDEX_SCHEMA,
        "built_at": 2.0,
        "revision": 1,
        "complete": True,
        "sessions": {
            OTHER: {
                "sig": {"size": 0, "mtime": 0.0, "inode": 0},
                "roster_sig": None,
                "parent_revision": 1.0,
                "revision": 1.0,
                "title": "extra",
                "cwd": str(extra.parent),
                "complete": True,
                "entries": [record],
                "children": {},
            }
        },
    }


def test_catalog_for_read_never_caches_assets_under_a_newer_file_stat(
    tmp_path, monkeypatch
) -> None:
    """A write landing between the read and the stat must not be cached as ours.

    Reproduced two-process: the aggregate built from the older payload was
    recorded under the newer stat, so ``refresh=True`` took the "unchanged"
    early return and served the old catalog — the new session's asset was
    absent and ``get_asset`` raised ``AssetNotFound``.
    """
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1, tmp_path / "a.txt"))
    index.build_catalog(root, budget_s=None)
    index._CACHE.clear()

    extra = tmp_path / "extra.txt"
    extra.write_text("x")
    # The session is real so a later refresh can scan it; the synthetic entry
    # only stands in for the file the other process wrote.
    session(root, OTHER, write_rows("w2", 2, extra))
    newer = _payload_with_extra(root, extra)
    newer["sessions"] = {
        **index._load(index.index_path(root))["sessions"],
        **newer["sessions"],
    }
    real_load = index._load
    calls: list[int] = []

    def racing_load(target):
        payload = real_load(target)
        if not calls:
            calls.append(1)
            index._write(target, newer)
        return payload

    monkeypatch.setattr(index, "_load", racing_load)
    catalog = index.catalog_for_read(root)
    asset_id = identity.asset_id(identity.key_for(str(extra), None))
    assert any(asset.id == asset_id for asset in catalog.assets)

    refreshed = index.build_catalog(root, refresh=True)
    assert any(asset.id == asset_id for asset in refreshed.assets)
    assert service.get_asset(root, asset_id)["name"] == "extra.txt"


def test_a_retracted_child_survives_a_partial_merge(tmp_path, monkeypatch) -> None:
    """A fresh parent entry must beat the stale disk entry despite its revision.

    Retracting a child LOWERS the parent's aggregate revision, so the old
    revision tie-break kept the disk copy and the retraction never landed.
    """
    root = config_root(monkeypatch, tmp_path)
    parent = session(root, SESSION, write_rows("p1", 1, tmp_path / "parent.txt"))
    child = session(root, CHILD, write_rows("c1", 2, tmp_path / "child.txt"))
    (child / "origin.json").write_text(json.dumps({"origin": "subagent"}))
    (parent / "subagent-roster.v1.json").write_text(
        json.dumps(
            {
                "version": 1,
                "generation": 1,
                "jobs": [],
                "accounting": {},
                "records": [
                    {
                        "job_id": "j1",
                        "label": "scout",
                        "agent_role": "scout",
                        "depth": 1,
                        "session_dir": str(child),
                    }
                ],
            }
        )
    )
    first = index.build_catalog(root, budget_s=None)
    assert "child.txt" in {asset.derived_name for asset in first.assets}

    # The merge rule itself: the fresh entry wins for a key this build scanned,
    # even though retracting the child made its revision LOWER than disk's.
    stale = {"complete": True, "revision": 100.0, "children": {"child": {"revision": 5.0}}}
    fresh = {"complete": True, "revision": 50.0, "children": {}}
    merged = index._merge_sessions({SESSION: stale}, {SESSION: fresh}, keep={SESSION})
    assert merged[SESSION] is fresh

    # An INCOMPLETE fresh scan (budget exhausted mid-session) must not replace
    # a complete disk entry: that would drop rows and lower sessions_scanned.
    partial = {"complete": False, "revision": 999.0}
    complete = {"complete": True, "revision": 1.0}
    assert index._merge_sessions({"s": complete}, {"s": partial}, keep={"s"})["s"] is complete

    # Retract the record: the parent revision drops below the disk entry's.
    (parent / "subagent-roster.v1.json").write_text(
        json.dumps({"version": 1, "generation": 2, "jobs": [], "accounting": {}, "records": []})
    )
    path = index.index_path(root)
    old = time.time() - 60
    os.utime(path, (old, old))

    refreshed = index.build_catalog(root, budget_s=1.5, refresh=True)
    assert "child.txt" not in {asset.derived_name for asset in refreshed.assets}
    saved = index._load(path)["sessions"][SESSION]
    assert saved["children"] == {}


def test_persist_rate_limits_revision_only_churn(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1, tmp_path / "a.txt"))
    index.build_catalog(root, budget_s=None)
    path = index.index_path(root)
    before = path.read_bytes()
    os.utime(path, None)
    entry = dict(index._load(path)["sessions"][SESSION])
    entry["revision"] = float(entry.get("revision") or 0.0) + 1000.0

    result = index._persist(path, {SESSION: entry}, visible={SESSION})

    assert result is not None and result[2] is False
    assert path.read_bytes() == before


def test_persist_writes_a_new_session_despite_the_rate_limit(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1, tmp_path / "a.txt"))
    index.build_catalog(root, budget_s=None)
    path = index.index_path(root)
    os.utime(path, None)
    entry = index._load(path)["sessions"][SESSION]

    result = index._persist(path, {SESSION: entry, OTHER: entry}, visible={SESSION, OTHER})

    assert result is not None and result[2] is True
    assert OTHER in index._load(path)["sessions"]


def test_persist_writes_a_first_completion(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1, tmp_path / "a.txt"))
    index.build_catalog(root, budget_s=None)
    path = index.index_path(root)
    payload = index._load(path)
    incomplete = {**payload["sessions"][SESSION], "complete": False}
    index._write(
        path,
        {
            "version": index.INDEX_SCHEMA,
            "built_at": 1.0,
            "revision": 1,
            "sessions": {SESSION: incomplete},
        },
    )
    os.utime(path, None)
    complete = {**incomplete, "complete": True}

    result = index._persist(path, {SESSION: complete}, visible={SESSION})

    assert result is not None and result[2] is True
    assert index._load(path)["sessions"][SESSION]["complete"] is True


def test_persist_writes_a_removal(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1, tmp_path / "a.txt"))
    session(root, OTHER, write_rows("w2", 2, tmp_path / "b.txt"))
    index.build_catalog(root, budget_s=None)
    path = index.index_path(root)
    os.utime(path, None)

    result = index._persist(
        path, {SESSION: index._load(path)["sessions"][SESSION]}, visible={SESSION}
    )

    assert result is not None and result[2] is True
    assert OTHER not in index._load(path)["sessions"]


def test_persist_returns_none_when_the_lock_is_held(tmp_path, monkeypatch) -> None:
    fcntl = pytest.importorskip("fcntl")
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1, tmp_path / "a.txt"))
    index.build_catalog(root, budget_s=None)
    path = index.index_path(root)
    entry = index._load(path)["sessions"][SESSION]
    handle = path.with_name("index.lock").open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    try:
        assert index._persist(path, {SESSION: entry}, visible={SESSION}) is None
    finally:
        handle.close()


def test_catalog_for_read_calls_a_partial_index_building(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1, tmp_path / "a.txt"))
    index.build_catalog(root, budget_s=None)
    path = index.index_path(root)
    payload = index._load(path)
    payload["sessions"][SESSION] = {**payload["sessions"][SESSION], "complete": False}
    index._write(path, payload)
    index._CACHE.clear()

    catalog = index.catalog_for_read(root)

    assert catalog.state == "building"


def test_catalog_for_read_reloads_after_another_process_writes(tmp_path, monkeypatch) -> None:
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1, tmp_path / "a.txt"))
    index.build_catalog(root, budget_s=None)
    extra = tmp_path / "extra.txt"
    extra.write_text("x")
    newer = _payload_with_extra(root, extra)
    newer["sessions"] = {
        **index._load(index.index_path(root))["sessions"],
        **newer["sessions"],
    }
    index._write(index.index_path(root), newer)

    catalog = index.catalog_for_read(root)

    asset_id = identity.asset_id(identity.key_for(str(extra), None))
    assert any(asset.id == asset_id for asset in catalog.assets)


def test_build_never_caches_assets_under_a_stat_from_a_racing_writer(tmp_path, monkeypatch) -> None:
    """A writer landing during ``_build_assets`` must not poison the cache.

    ``_persist`` releases the lock before the aggregate is built, so the stat
    captured under the lock is the only identity the assets can be trusted
    under. Reproduced: the newer file's stat was recorded with the older asset
    set, and ``refresh=True``/``get_asset`` kept serving the stale one.
    """
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1, tmp_path / "a.txt"))
    index.build_catalog(root, budget_s=None)
    index._CACHE.clear()

    raced = tmp_path / "raced.txt"
    raced.write_text("x")
    newer = _payload_with_extra(root, raced)
    newer["sessions"] = {
        **index._load(index.index_path(root))["sessions"],
        **newer["sessions"],
    }
    real_build_assets = index._build_assets
    fired: list[int] = []

    def racing_build_assets(sessions):
        assets = real_build_assets(sessions)
        if not fired:
            fired.append(1)
            index._write(index.index_path(root), newer)
        return assets

    monkeypatch.setattr(index, "_build_assets", racing_build_assets)
    index.build_catalog(root, budget_s=None)

    assert fired, "the racing writer must have fired"
    assert index._cached(index._cache_key(root)) is None, "stale assets must not be cached"
    raced_id = identity.asset_id(identity.key_for(str(raced), None))
    assert any(asset.id == raced_id for asset in index.catalog_for_read(root).assets)
    assert service.get_asset(root, raced_id)["name"] == "raced.txt"


def test_a_truncated_partial_index_is_not_ready_even_when_its_rows_are(
    tmp_path, monkeypatch
) -> None:
    """A budget-limited build can persist only complete entries and stop early.

    ``sessions_scanned == sessions_total`` is not proof of completeness; the
    top-level flag is. An index predating the flag is treated as building too,
    so it gets rebuilt rather than served as ready.
    """
    root = config_root(monkeypatch, tmp_path)
    session(root, SESSION, write_rows("w1", 1, tmp_path / "a.txt"))
    session(root, OTHER, write_rows("w2", 2, tmp_path / "b.txt"))
    index.build_catalog(root, budget_s=None)
    index._CACHE.clear()
    path = index.index_path(root)
    payload = index._load(path)
    payload["sessions"] = {SESSION: payload["sessions"][SESSION]}
    payload["complete"] = False
    index._write(path, payload)

    assert index.catalog_for_read(root).state == "building"

    payload.pop("complete")
    index._write(path, payload)
    index._CACHE.clear()
    assert index.catalog_for_read(root).state == "building"
