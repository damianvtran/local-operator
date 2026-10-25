"""Destination-side containment for peer-authored session sync plans.

An honest owner only carries relative links that resolve inside its session. The
destination still enforces that contract: its ``ask`` callable is an untrusted
transport, and the replica may already contain entries from an earlier plan.
These cells use the production ``pull`` seam with minimal wire fixtures; the
last cell also runs an honest owner through that seam.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import sync


def _plan(items: list[dict[str, Any]], attachments: list[dict[str, Any]] | None = None) -> Any:
    transcript = b"{}\n"

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        phase = frame["phase"]
        if phase == "plan":
            return {
                "plan_id": "plan",
                "transcript": {
                    "mode": "replace",
                    "region_bytes": len(transcript),
                    "total_bytes": len(transcript),
                    "digest": sync.sha256_bytes(transcript),
                },
                "items": items,
                "attachments": attachments or [],
            }
        if phase == "fetch":
            if frame["name"] == sync.TRANSCRIPT_NAME:
                payload = transcript
            else:
                member = next(
                    item for item in [*items, *(attachments or [])] if item["name"] == frame["name"]
                )
                payload = member["payload"]
            return {
                "data": base64.b64encode(payload).decode("ascii"),
                "chunk_digest": sync.sha256_bytes(payload),
            }
        raise AssertionError(f"unexpected sync phase: {frame}")

    return ask


def _link(name: str, target: str) -> dict[str, Any]:
    return {"name": name, "link": target, "digest": sync.sha256_bytes(target.encode())}


def _file(name: str, payload: bytes = b"payload") -> dict[str, Any]:
    return {
        "name": name,
        "bytes": len(payload),
        "digest": sync.sha256_bytes(payload),
        "payload": payload,
    }


def _pull(tmp_path: Path, ask: Any, *, attachments_root: Path | None = None) -> Path:
    dest = tmp_path / "replica"
    sync.pull(
        tmp_path / "config",
        "session",
        ask=ask,
        dest_dir=dest,
        attachments_root=attachments_root or dest,
    )
    return dest


def _assert_refused(tmp_path: Path, items: list[dict[str, Any]], name: str) -> None:
    with pytest.raises(sync.SyncRefused) as refusal:
        _pull(tmp_path, _plan(items))
    assert refusal.value.code in {"unknown_item", "no_replica"}
    assert name in refusal.value.message
    replica = tmp_path / "replica"
    if replica.exists():
        assert not list(replica.rglob("*.fetch"))
        assert not list(replica.rglob("*.link"))


@pytest.mark.parametrize("target", ["/tmp/outside", "../../outside", "inside/../../../outside"])
def test_pull_refuses_link_targets_that_resolve_outside(tmp_path: Path, target: str) -> None:
    name = "scratchpad/link"
    _assert_refused(tmp_path, [_link(name, target)], name)
    assert not (tmp_path / "replica" / "transcript.jsonl").exists()


def test_pull_refuses_a_nested_escaping_link_chain(tmp_path: Path) -> None:
    first = _link("scratchpad/first", "second")
    second = _link("scratchpad/second", "../../outside")
    with pytest.raises(sync.SyncRefused) as refusal:
        _pull(tmp_path, _plan([first, second]))
    assert "scratchpad/first" in refusal.value.message
    assert "outside" in refusal.value.message
    assert not (tmp_path / "replica" / "transcript.jsonl").exists()


@pytest.mark.parametrize("name", ["/tmp/outside", "title.json/../../outside"])
def test_pull_refuses_absolute_or_escaping_entry_names(tmp_path: Path, name: str) -> None:
    _assert_refused(tmp_path, [_file(name)], name)
    assert not Path("/tmp/outside").exists()


@pytest.mark.parametrize("leaf_is_link", [False, True])
def test_pull_refuses_preexisting_symlink_path_components(
    tmp_path: Path, leaf_is_link: bool
) -> None:
    dest = tmp_path / "replica"
    outside = tmp_path / "outside"
    outside.mkdir()
    (dest / "scratchpad").mkdir(parents=True)
    if leaf_is_link:
        (dest / "scratchpad" / "member").symlink_to(outside / "member")
    else:
        (dest / "scratchpad" / "nested").symlink_to(outside, target_is_directory=True)
    name = "scratchpad/member" if leaf_is_link else "scratchpad/nested/member"
    _assert_refused(tmp_path, [_file(name)], name)
    assert not (outside / "member").exists()


def test_pull_refuses_an_escaping_carried_link_before_a_later_file_write(
    tmp_path: Path,
) -> None:
    name = "scratchpad/link"
    _assert_refused(
        tmp_path,
        [_link(name, "../../outside"), _file("scratchpad/link/member")],
        name,
    )
    assert not (tmp_path / "outside" / "member").exists()


def test_pull_rejects_a_link_replaced_between_validation_and_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The final no-follow rename is the containment boundary, not a prior stat."""
    original_replace = sync._replace_at
    swapped = False

    def replace_with_link(stage: Path, destination: Path, scope: Any, **kwargs: Any) -> None:
        nonlocal swapped
        if destination.name == "honest":
            # Simulate a concurrent link substitution immediately BEFORE the write-time
            # containment check. The production dirfd rename is still the real primitive.
            destination.unlink(missing_ok=True)
            destination.symlink_to(tmp_path / "outside", target_is_directory=True)
            swapped = True
        original_replace(stage, destination, scope, **kwargs)

    monkeypatch.setattr(sync, "_replace_at", replace_with_link)
    _assert_refused(tmp_path, [_file("scratchpad/honest")], "scratchpad/honest")
    assert swapped


def test_attachment_store_symlink_cannot_redirect_a_pull(tmp_path: Path) -> None:
    store = tmp_path / "shared"
    (store / "attachments").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    name = "attachments/member.bin"
    (store / "attachments" / "member.bin").symlink_to(outside / "member.bin")
    with pytest.raises(sync.SyncRefused) as refusal:
        _pull(tmp_path, _plan([], [_file(name)]), attachments_root=store)
    assert Path(name).name in refusal.value.message
    assert not (outside / "member.bin").exists()


@pytest.mark.parametrize(
    "session_id",
    ["", ".", "..", "../escape", "/absolute", "a\\b", "D:escape", "a\x00b", "a\nb"],
)
def test_sync_from_refuses_session_ids_that_are_not_one_component(
    tmp_path: Path, session_id: str
) -> None:
    with pytest.raises(sync.SyncRefused) as refusal:
        sync.sync_from(tmp_path, session_id, ask=_plan([]))
    assert "session id" in refusal.value.message.lower()


def _recovered_replica(root: Path) -> None:
    owner = root / "owner"
    owner.mkdir(parents=True)
    session = owner / "sessions" / "abc123"
    (session / "scratchpad" / "nested").mkdir(parents=True)
    (session / "transcript.jsonl").write_text(
        '{"type":"image","attachment":"' + "a" * 32 + '"}\n', encoding="utf-8"
    )
    (session / "scratchpad" / "nested" / "notes.md").write_text("notes", encoding="utf-8")
    (session / "scratchpad" / "link").symlink_to("nested/notes.md")
    store = owner / "attachments"
    store.mkdir()
    (store / ("a" * 32 + ".bin")).write_bytes(b"blob")
    (store / ("a" * 32 + ".json")).write_text("{}", encoding="utf-8")

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        if frame["phase"] == "plan":
            return sync.build_manifest(owner, "abc123", have=frame.get("have") or {})
        if frame["phase"] == "fetch":
            return sync.serve_fetch(
                owner,
                "abc123",
                plan=frame["plan_id"],
                name=frame["name"],
                offset=frame.get("offset", 0),
                limit=frame.get("limit", sync.SYNC_CHUNK_BYTES),
            )
        return sync.serve_verify(
            owner,
            "abc123",
            plan=frame["plan_id"],
            name=frame["name"],
            prefix_bytes=frame["prefix_bytes"],
            prefix_digest=frame["prefix_digest"],
        )

    sync.sync_from(
        root,
        "abc123",
        ask=ask,
        into=sync.replica_dir(root, "abc123"),
        attachments_root=sync.replica_dir(root, "abc123"),
    )


@pytest.mark.parametrize("location", ["leaf", "directory"])
def test_recovery_refuses_shared_store_symlink_redirection(tmp_path: Path, location: str) -> None:
    root = tmp_path / "holder"
    _recovered_replica(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    if location == "leaf":
        (root / "attachments").mkdir()
        (root / "attachments" / ("a" * 32 + ".bin")).symlink_to(outside / "blob")
    else:
        (root / "attachments").symlink_to(outside, target_is_directory=True)
    with pytest.raises(sync.SyncRefused):
        sync.promote_replica(root, "abc123")
    assert not (outside / "blob").exists()


@pytest.mark.parametrize("location", ["blob", "directory"])
def test_recovery_refuses_replica_attachment_source_links(tmp_path: Path, location: str) -> None:
    root = tmp_path / "holder"
    _recovered_replica(root)
    replica = sync.replica_dir(root, "abc123")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "blob").write_bytes(b"not the replica blob")
    if location == "blob":
        (replica / "attachments" / ("a" * 32 + ".bin")).unlink()
        (replica / "attachments" / ("a" * 32 + ".bin")).symlink_to(outside / "blob")
    else:
        (replica / "attachments").rename(replica / "real-attachments")
        (replica / "attachments").symlink_to(outside, target_is_directory=True)
    with pytest.raises(sync.SyncRefused):
        sync.promote_replica(root, "abc123")
    assert not (root / "attachments").exists()


def test_recovery_digest_mismatch_preserves_shared_blob_and_cleans_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "holder"
    _recovered_replica(root)
    shared = root / "attachments" / ("a" * 32 + ".bin")
    shared.parent.mkdir()
    shared.write_bytes(b"previous good blob")
    original = sync._copy_contained

    def changed_during_copy(source: Path, destination: Path, scope: Any) -> None:
        def mutate_source() -> None:
            source.write_bytes(b"changed")

        monkeypatch.setattr(sync, "_stream_digest", lambda path: mutate_source() or "sha256:new")
        original(source, destination, scope)

    monkeypatch.setattr(sync, "_copy_contained", changed_during_copy)
    with pytest.raises(sync.SyncRefused) as refusal:
        sync.promote_replica(root, "abc123")
    assert "changed" in refusal.value.message
    assert shared.read_bytes() == b"previous good blob"
    assert not list(shared.parent.glob("*.copy"))


def test_replace_append_with_unchanged_link_recovers_immediately(tmp_path: Path) -> None:
    root = tmp_path / "holder"
    owner = root / "owner"
    session = owner / "sessions" / "abc123"
    (session / "scratchpad" / "nested").mkdir(parents=True)
    transcript = session / "transcript.jsonl"
    original = b'{"line":1}\n'
    transcript.write_bytes(original)
    (session / "scratchpad" / "nested" / "notes.md").write_text("notes", encoding="utf-8")
    (session / "scratchpad" / "nested" / "alternate.md").write_text(
        "alternate target with different bytes", encoding="utf-8"
    )
    (session / "scratchpad" / "link").symlink_to("nested/notes.md")

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        if frame["phase"] == "plan":
            return sync.build_manifest(owner, "abc123", have=frame.get("have") or {})
        if frame["phase"] == "fetch":
            return sync.serve_fetch(
                owner,
                "abc123",
                plan=frame["plan_id"],
                name=frame["name"],
                offset=frame.get("offset", 0),
                limit=frame.get("limit", sync.SYNC_CHUNK_BYTES),
            )
        return sync.serve_verify(
            owner,
            "abc123",
            plan=frame["plan_id"],
            name=frame["name"],
            prefix_bytes=frame["prefix_bytes"],
            prefix_digest=frame["prefix_digest"],
        )

    first = sync.sync_from(root, "abc123", ask=ask, into=sync.replica_dir(root, "abc123"))
    assert first["mode"] == "replace"
    (session / "scratchpad" / "link").unlink()
    (session / "scratchpad" / "link").symlink_to("nested/alternate.md")
    transcript.write_bytes(original + b'{"line":2}\n')
    second = sync.sync_from(root, "abc123", ask=ask, into=sync.replica_dir(root, "abc123"))
    assert second["mode"] == "append"
    assert second["items"] == 0
    cursor = sync.read_replica_cursor(root, "abc123")
    assert cursor["links"].count("scratchpad/link") == 1
    promoted = sync.promote_replica(root, "abc123", new_id="recovered")
    target = root / "sessions" / "recovered"
    assert promoted["session_id"] == "recovered"
    assert (target / "transcript.jsonl").read_bytes() == original + b'{"line":2}\n'
    assert (target / "scratchpad" / "link").read_text(encoding="utf-8") == (
        "alternate target with different bytes"
    )


@pytest.mark.parametrize("kind", ["link", "file"])
def test_source_deleted_member_does_not_survive_append_recovery(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "holder"
    owner = root / "owner"
    session = owner / "sessions" / "abc123"
    (session / "scratchpad").mkdir(parents=True)
    transcript = session / "transcript.jsonl"
    original = b'{"line":1}\n'
    transcript.write_bytes(original)
    (session / "title.json").write_text('{"title":"remove me"}', encoding="utf-8")
    (session / "scratchpad" / "notes.md").write_text("notes", encoding="utf-8")
    link = session / "scratchpad" / "link"
    link.symlink_to("notes.md")

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        if frame["phase"] == "plan":
            return sync.build_manifest(owner, "abc123", have=frame.get("have") or {})
        if frame["phase"] == "fetch":
            return sync.serve_fetch(
                owner,
                "abc123",
                plan=frame["plan_id"],
                name=frame["name"],
                offset=frame.get("offset", 0),
                limit=frame.get("limit", sync.SYNC_CHUNK_BYTES),
            )
        return sync.serve_verify(
            owner,
            "abc123",
            plan=frame["plan_id"],
            name=frame["name"],
            prefix_bytes=frame["prefix_bytes"],
            prefix_digest=frame["prefix_digest"],
        )

    sync.sync_from(root, "abc123", ask=ask, into=sync.replica_dir(root, "abc123"))
    name = "scratchpad/link" if kind == "link" else "title.json"
    (link if kind == "link" else session / "title.json").unlink()
    transcript.write_bytes(original + b'{"line":2}\n')
    second = sync.sync_from(root, "abc123", ask=ask, into=sync.replica_dir(root, "abc123"))
    assert second["mode"] == "append"
    replica = sync.replica_dir(root, "abc123")
    assert not (replica / name).exists()
    assert not (replica / name).is_symlink()
    cursor = sync.read_replica_cursor(root, "abc123")
    assert name not in cursor["files"]
    assert name not in cursor["links"]
    sync.promote_replica(root, "abc123", new_id="recovered")
    target = root / "sessions" / "recovered"
    assert not (target / name).exists()
    assert not (target / name).is_symlink()


def test_incomplete_plan_cannot_prune_held_member(tmp_path: Path) -> None:
    root = tmp_path / "holder"
    owner = root / "owner"
    session = owner / "sessions" / "abc123"
    (session / "scratchpad").mkdir(parents=True)
    (session / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    (session / "scratchpad" / "notes.md").write_text("keep", encoding="utf-8")

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        if frame["phase"] == "plan":
            return sync.build_manifest(owner, "abc123", have=frame.get("have") or {})
        return sync.serve_fetch(
            owner,
            "abc123",
            plan=frame["plan_id"],
            name=frame["name"],
            offset=frame.get("offset", 0),
            limit=frame.get("limit", sync.SYNC_CHUNK_BYTES),
        )

    sync.sync_from(root, "abc123", ask=ask, into=sync.replica_dir(root, "abc123"))
    replica = sync.replica_dir(root, "abc123")
    original = (replica / "scratchpad" / "notes.md").read_bytes()

    def partial(frame: dict[str, Any]) -> dict[str, Any]:
        answer = ask(frame)
        if frame["phase"] == "plan":
            answer.pop("members", None)
            answer["items"] = []
        return answer

    sync.sync_from(root, "abc123", ask=partial, into=replica)
    assert (replica / "scratchpad" / "notes.md").read_bytes() == original


def test_failed_transfer_does_not_prune_source_deleted_member(tmp_path: Path) -> None:
    root = tmp_path / "holder"
    owner = root / "owner"
    session = owner / "sessions" / "abc123"
    (session / "scratchpad").mkdir(parents=True)
    (session / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    (session / "title.json").write_text('{"title":"keep on failure"}', encoding="utf-8")

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        if frame["phase"] == "plan":
            return sync.build_manifest(owner, "abc123", have=frame.get("have") or {})
        return sync.serve_fetch(
            owner,
            "abc123",
            plan=frame["plan_id"],
            name=frame["name"],
            offset=frame.get("offset", 0),
            limit=frame.get("limit", sync.SYNC_CHUNK_BYTES),
        )

    sync.sync_from(root, "abc123", ask=ask, into=sync.replica_dir(root, "abc123"))
    (session / "title.json").unlink()
    (session / "transcript.jsonl").write_text('{}\n{"new":true}\n', encoding="utf-8")
    replica = sync.replica_dir(root, "abc123")
    old = (replica / "title.json").read_bytes()

    def failing(frame: dict[str, Any]) -> dict[str, Any]:
        if frame["phase"] == "fetch" and frame["name"] == sync.TRANSCRIPT_NAME:
            raise ConnectionError("transfer interrupted")
        return ask(frame)

    with pytest.raises(ConnectionError):
        sync.sync_from(root, "abc123", ask=failing, into=replica)
    assert (replica / "title.json").read_bytes() == old


def test_recovery_preserves_an_honest_in_tree_link(tmp_path: Path) -> None:
    root = tmp_path / "holder"
    _recovered_replica(root)
    promoted = sync.promote_replica(root, "abc123")
    target = root / "sessions" / promoted["session_id"]
    assert os.readlink(target / "scratchpad" / "link") == "nested/notes.md"
    assert (target / "scratchpad" / "link").read_text(encoding="utf-8") == "notes"
    assert (root / "attachments" / ("a" * 32 + ".bin")).read_bytes() == b"blob"


def test_existing_partial_transcript_resumes_without_truncation(tmp_path: Path) -> None:
    owner = tmp_path / "owner"
    session = owner / "sessions" / "abc123"
    session.mkdir(parents=True)
    rows = b'{"line":1}\n{"line":2}\n{"line":3}\n'
    (session / "transcript.jsonl").write_bytes(rows)

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        if frame["phase"] == "plan":
            return sync.build_manifest(owner, "abc123", have=frame.get("have") or {})
        if frame["phase"] == "fetch":
            return sync.serve_fetch(
                owner,
                "abc123",
                plan=frame["plan_id"],
                name=frame["name"],
                offset=frame.get("offset", 0),
                limit=frame.get("limit", sync.SYNC_CHUNK_BYTES),
            )
        return sync.serve_verify(
            owner,
            "abc123",
            plan=frame["plan_id"],
            name=frame["name"],
            prefix_bytes=frame["prefix_bytes"],
            prefix_digest=frame["prefix_digest"],
        )

    dest = tmp_path / "replica"
    dest.mkdir()
    (dest / "transcript.jsonl").write_bytes(rows[:12])
    result = sync.sync_from(owner, "abc123", ask=ask, into=dest)
    assert result["mode"] == "replace"
    assert result["bytes"] == len(rows) - 12
    assert (dest / "transcript.jsonl").read_bytes() == rows


def test_honest_in_tree_and_dangling_links_round_trip(tmp_path: Path) -> None:
    owner = tmp_path / "owner"
    session_id = "abc123"
    session = owner / "sessions" / session_id
    (session / "scratchpad" / "nested").mkdir(parents=True)
    (session / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    (session / "scratchpad" / "nested" / "notes.md").write_text("notes", encoding="utf-8")
    (session / "scratchpad" / "link").symlink_to("nested/notes.md")
    (session / "scratchpad" / "upward").symlink_to("nested/../nested/notes.md")
    (session / "scratchpad" / "dangling").symlink_to("nested/missing.md")

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        if frame["phase"] == "plan":
            return sync.build_manifest(owner, session_id, have=frame.get("have") or {})
        if frame["phase"] == "fetch":
            return sync.serve_fetch(
                owner,
                session_id,
                plan=frame["plan_id"],
                name=frame["name"],
                offset=frame.get("offset", 0),
                limit=frame.get("limit", sync.SYNC_CHUNK_BYTES),
            )
        return sync.serve_verify(
            owner,
            session_id,
            plan=frame["plan_id"],
            name=frame["name"],
            prefix_bytes=frame["prefix_bytes"],
            prefix_digest=frame["prefix_digest"],
        )

    dest = tmp_path / "replica"
    sync.pull(
        tmp_path / "config",
        session_id,
        ask=ask,
        dest_dir=dest,
        attachments_root=dest,
    )
    assert os.readlink(dest / "scratchpad" / "link") == "nested/notes.md"
    assert (dest / "scratchpad" / "link").read_text(encoding="utf-8") == "notes"
    assert (dest / "scratchpad" / "upward").read_text(encoding="utf-8") == "notes"
    assert os.readlink(dest / "scratchpad" / "dangling") == "nested/missing.md"
    assert not (dest / "scratchpad" / "dangling").exists()


@pytest.mark.parametrize("kind", ["replica", "staging"])
def test_sync_from_refuses_symlinked_network_store_parent(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "config"
    (root / "network").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "network" / kind).symlink_to(outside, target_is_directory=True)
    destination = root / "network" / kind / "abc123"
    with pytest.raises(sync.SyncRefused):
        sync.sync_from(
            root,
            "abc123",
            ask=_plan([]),
            into=destination,
            attachments_root=destination,
        )
    assert not (outside / "abc123").exists()


def test_sync_from_refuses_symlinked_attachment_store_root(tmp_path: Path) -> None:
    root = tmp_path / "config"
    root.mkdir()
    outside = tmp_path / "outside"
    (outside / "attachments").mkdir(parents=True)
    (root / "store").symlink_to(outside, target_is_directory=True)
    with pytest.raises(sync.SyncRefused):
        sync.sync_from(
            root,
            "abc123",
            ask=_plan([], [_file("attachments/member.bin")]),
            attachments_root=root / "store",
        )
    assert not (outside / "attachments" / "member.bin").exists()


def test_recovery_refuses_symlinked_sessions_root(tmp_path: Path) -> None:
    root = tmp_path / "holder"
    _recovered_replica(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "sessions").symlink_to(outside, target_is_directory=True)
    with pytest.raises(sync.SyncRefused):
        sync.promote_replica(root, "abc123", new_id="recovered")
    assert not (outside / "recovered").exists()


def test_refused_plan_preserves_compressed_transcript(tmp_path: Path) -> None:
    from local_operator.session import compression

    dest = tmp_path / "replica"
    dest.mkdir()
    (dest / "transcript.jsonl").write_bytes(b'{"old":true}\n')
    outcome = compression.compress_session(dest, codec="gzip")
    assert outcome.status == "compressed"
    archive = compression.archive_path(dest)
    assert archive is not None
    before = {path.name: path.read_bytes() for path in dest.iterdir()}
    with pytest.raises(sync.SyncRefused):
        _pull(tmp_path, _plan([_link("scratchpad/link", "../../outside")]))
    after = {path.name: path.read_bytes() for path in dest.iterdir()}
    assert after == before
    assert not (dest / "transcript.jsonl").exists()


def test_refused_link_destination_does_not_replace_existing_transcript(tmp_path: Path) -> None:
    dest = tmp_path / "replica"
    (dest / "scratchpad").mkdir(parents=True)
    original = b'{"old":true}\n'
    (dest / "transcript.jsonl").write_bytes(original)
    outside = tmp_path / "outside"
    outside.mkdir()
    (dest / "scratchpad" / "nested").symlink_to(outside, target_is_directory=True)
    with pytest.raises(sync.SyncRefused):
        _pull(tmp_path, _plan([_link("scratchpad/nested/link", "../honest")]))
    assert (dest / "transcript.jsonl").read_bytes() == original
    assert not (outside / "link").exists()


def test_stale_compressed_append_preserves_archive(tmp_path: Path) -> None:
    from local_operator.session import compression

    dest = tmp_path / "replica"
    dest.mkdir()
    (dest / "transcript.jsonl").write_bytes(b"{}\n")
    assert compression.compress_session(dest, codec="gzip").status == "compressed"
    archive = compression.archive_path(dest)
    assert archive is not None
    assert (dest / compression.PREVIEW_SIDECAR_NAME).is_file()
    before = {path.name: path.read_bytes() for path in dest.iterdir() if path.is_file()}
    plan = _plan([])

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        answer = plan(frame)
        if frame["phase"] == "plan":
            answer["transcript"] = {
                "mode": "append",
                "prefix_bytes": 4,
                "region_bytes": 3,
                "total_bytes": 7,
                "digest": sync.sha256_bytes(b"{}\n{}\n"),
            }
        return answer

    with pytest.raises(sync.SyncRefused) as refusal:
        _pull(tmp_path, ask)
    assert refusal.value.code == "stale_plan"
    after = {path.name: path.read_bytes() for path in dest.iterdir() if path.is_file()}
    assert after == before
    assert not (dest / "transcript.jsonl").exists()


def test_failed_compressed_append_preserves_full_replica_state(tmp_path: Path) -> None:
    from local_operator.session import compression

    dest = tmp_path / "replica"
    dest.mkdir()
    original = b"{}\n"
    (dest / "transcript.jsonl").write_bytes(original)
    assert compression.compress_session(dest, codec="gzip").status == "compressed"
    before = {path.name: path.read_bytes() for path in dest.iterdir() if path.is_file()}
    plan = _plan([])

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        answer = plan(frame)
        if frame["phase"] == "plan":
            answer["transcript"] = {
                "mode": "append",
                "prefix_bytes": len(original),
                "region_bytes": len(original),
                "total_bytes": len(original * 2),
                "digest": sync.sha256_bytes(b"not the appended transcript"),
            }
        return answer

    with pytest.raises(sync.SyncRefused) as refusal:
        _pull(tmp_path, ask)
    assert refusal.value.code == "digest_mismatch"
    after = {path.name: path.read_bytes() for path in dest.iterdir() if path.is_file()}
    assert after == before


def test_valid_compressed_append_restores_and_appends(tmp_path: Path) -> None:
    from local_operator.session import compression

    dest = tmp_path / "replica"
    dest.mkdir()
    original = b'{"line":1}\n'
    appended = b'{"line":2}\n'
    (dest / "transcript.jsonl").write_bytes(original)
    assert compression.compress_session(dest, codec="gzip").status == "compressed"
    plan = _plan([])

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        answer = plan(frame)
        if frame["phase"] == "plan":
            answer["transcript"] = {
                "mode": "append",
                "prefix_bytes": len(original),
                "region_bytes": len(appended),
                "total_bytes": len(original + appended),
                "digest": sync.sha256_bytes(original + appended),
            }
        if frame["phase"] == "fetch" and frame["name"] == sync.TRANSCRIPT_NAME:
            assert frame["offset"] == len(original)
            payload = (original + appended)[frame["offset"] :]
            return {
                "data": base64.b64encode(payload).decode("ascii"),
                "chunk_digest": sync.sha256_bytes(payload),
            }
        return answer

    result = sync.pull(
        tmp_path / "config", "session", ask=ask, dest_dir=dest, attachments_root=dest
    )
    assert result.transcript_mode == "append"
    assert result.bytes == len(appended)
    assert (dest / "transcript.jsonl").read_bytes() == original + appended
    assert compression.archive_path(dest) is None


def test_valid_plan_still_restores_compressed_transcript(tmp_path: Path) -> None:
    from local_operator.session import compression

    dest = tmp_path / "replica"
    dest.mkdir()
    (dest / "transcript.jsonl").write_bytes(b'{"old":true}\n')
    assert compression.compress_session(dest, codec="gzip").status == "compressed"
    _pull(tmp_path, _plan([]))
    assert (dest / "transcript.jsonl").read_bytes() == b"{}\n"
    assert compression.archive_path(dest) is None
