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


def _transcript_document(transcript: bytes) -> dict[str, Any]:
    return {
        "mode": "replace",
        "region_bytes": len(transcript),
        "total_bytes": len(transcript),
        "digest": sync.sha256_bytes(transcript),
    }


def _answering(
    items: list[dict[str, Any]],
    attachments: list[dict[str, Any]] | None,
    transcript: bytes,
    plan_document: dict[str, Any],
) -> Any:
    """One ``ask`` over a fixed plan document and the payloads that document names."""

    def payload_for(name: str) -> bytes:
        if name == sync.TRANSCRIPT_NAME:
            return transcript
        member = next(item for item in [*items, *(attachments or [])] if item["name"] == name)
        return member["payload"]

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        phase = frame["phase"]
        if phase == "plan":
            return dict(plan_document)
        if phase == "fetch":
            payload = payload_for(frame["name"])
            return {
                "data": base64.b64encode(payload).decode("ascii"),
                "chunk_digest": sync.sha256_bytes(payload),
            }
        if phase == "verify":
            # A retried pull resumes from the prefix it already holds, so the transport
            # has to answer the resumption check the owner would answer.
            prefix = payload_for(frame["name"])[: int(frame["prefix_bytes"])]
            return {"matches": sync.sha256_bytes(prefix) == frame["prefix_digest"]}
        raise AssertionError(f"unexpected sync phase: {frame}")

    return ask


def _plan(items: list[dict[str, Any]], attachments: list[dict[str, Any]] | None = None) -> Any:
    transcript = b"{}\n"
    return _answering(
        items,
        attachments,
        transcript,
        {
            "plan_id": "plan",
            "transcript": _transcript_document(transcript),
            "items": items,
            "attachments": attachments or [],
        },
    )


def _complete_plan(
    members: dict[str, str],
    *,
    items: list[dict[str, Any]] | None = None,
    links: tuple[str, ...] = (),
    transcript: bytes = b"{}\n",
    attachments: list[dict[str, Any]] | None = None,
) -> Any:
    """A plan that declares a COMPLETE source member set beside its transfer delta.

    ``members`` is the owner's WHOLE declaration and ``items`` only what it chose to
    send: a declared member with no item is the case the destination has to justify for
    itself (see ``_apply_validated_plan``). The transcript is added here because
    ``_complete_plan_members`` binds that member to the plan's own digest.
    """
    declared = {sync.TRANSCRIPT_NAME: sync.sha256_bytes(transcript), **members}
    return _answering(
        items or [],
        attachments,
        transcript,
        {
            "plan_id": sync._attested_digest(declared, links, "session"),
            "transcript": _transcript_document(transcript),
            "items": items or [],
            "attachments": attachments or [],
            "members": declared,
            "member_links": list(links),
        },
    )


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


# ---------------------------------------------------------------------------
# A configured root reached through a symbolic link
# ---------------------------------------------------------------------------


def _owner_ask(
    root: Path, session_id: str = "abc123", plans: list[dict[str, Any]] | None = None
) -> Any:
    """The production owner seam, with each plan it serves recorded when asked to."""

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        if frame["phase"] == "plan":
            plan = sync.build_manifest(root, session_id, have=frame.get("have") or {})
            if plans is not None:
                plans.append(plan)
            return plan
        if frame["phase"] == "fetch":
            return sync.serve_fetch(
                root,
                session_id,
                plan=frame["plan_id"],
                name=frame["name"],
                offset=frame.get("offset", 0),
                limit=frame.get("limit", sync.SYNC_CHUNK_BYTES),
            )
        return sync.serve_verify(
            root,
            session_id,
            plan=frame["plan_id"],
            name=frame["name"],
            prefix_bytes=frame["prefix_bytes"],
            prefix_digest=frame["prefix_digest"],
        )

    return ask


def test_pull_into_a_symlinked_configured_root_succeeds(tmp_path: Path) -> None:
    """The operator's configured root may itself be reached through a link.

    A macOS ``mktemp`` root is ``/var/...`` whose real path is ``/private/var/...``:
    resolving the root once must not turn that spelling into a refusal, while a link
    BELOW the root is still refused (the sibling cells above).
    """
    real = tmp_path / "root"
    real.mkdir()
    linked = tmp_path / "config"
    linked.symlink_to(real, target_is_directory=True)
    dest = linked / "network" / "replicas" / "session"

    sync.pull(linked, "session", ask=_plan([]), dest_dir=dest, attachments_root=linked)

    assert (real / "network" / "replicas" / "session" / "transcript.jsonl").read_bytes() == b"{}\n"


def test_sync_from_into_a_symlinked_configured_root_succeeds(tmp_path: Path) -> None:
    owner = tmp_path / "owner"
    session = owner / "sessions" / "abc123"
    session.mkdir(parents=True)
    (session / "transcript.jsonl").write_bytes(b"{}\n")
    real = tmp_path / "root"
    real.mkdir()
    linked = tmp_path / "config"
    linked.symlink_to(real, target_is_directory=True)

    result = sync.sync_from(linked, "abc123", ask=_owner_ask(owner))

    assert result["mode"] == "replace"
    assert (real / "network" / "replicas" / "abc123" / "transcript.jsonl").is_file()


def test_recovery_through_a_symlinked_configured_root_succeeds(tmp_path: Path) -> None:
    real = tmp_path / "holder"
    real.mkdir()
    _recovered_replica(real)
    linked = tmp_path / "config"
    linked.symlink_to(real, target_is_directory=True)

    promoted = sync.promote_replica(linked, "abc123", new_id="recovered")

    assert promoted["session_id"] == "recovered"
    assert (real / "sessions" / "recovered" / "transcript.jsonl").is_file()


def test_a_plain_pull_does_not_snapshot_the_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a compressed replica can be changed by the temporary restore."""
    calls: list[Path] = []
    original = sync._capture_transcript_state  # noqa: SLF001 — the call under test
    monkeypatch.setattr(
        sync,
        "_capture_transcript_state",
        lambda directory: calls.append(directory) or original(directory),
    )
    dest = tmp_path / "replica"
    dest.mkdir()
    (dest / "transcript.jsonl").write_bytes(b"{}\n")
    plan = _plan([])

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        if frame["phase"] == "verify":
            return {"matches": False}
        return plan(frame)

    sync.pull(tmp_path / "config", "session", ask=ask, dest_dir=dest, attachments_root=dest)

    assert calls == []


def test_a_compressed_pull_still_snapshots_before_restoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.session import compression

    calls: list[Path] = []
    original = sync._capture_transcript_state  # noqa: SLF001 — the call under test
    monkeypatch.setattr(
        sync,
        "_capture_transcript_state",
        lambda directory: calls.append(directory) or original(directory),
    )
    dest = tmp_path / "replica"
    dest.mkdir()
    (dest / "transcript.jsonl").write_bytes(b'{"old":true}\n')
    assert compression.compress_session(dest, codec="gzip").status == "compressed"

    _pull(tmp_path, _plan([]))

    assert calls == [dest]


# ---------------------------------------------------------------------------
# The attestation: a cursor is a claim about THIS destination, not about the plan
# ---------------------------------------------------------------------------


def test_a_declared_member_this_device_lacks_is_refused_not_attested(tmp_path: Path) -> None:
    """Every attested name must be true here, so an untransferred member is refused.

    The complete source set used to be copied into the cursor wholesale, so a name the
    owner declared and this destination never received was recorded as held: recovery
    then refused the copy and the session read as unavailable until a full re-pull.
    """
    name = "scratchpad/notes.md"
    ask = _complete_plan({name: sync.sha256_bytes(b"notes")})

    with pytest.raises(sync.SyncRefused) as refusal:
        sync.sync_from(tmp_path, "session", ask=ask, into=sync.replica_dir(tmp_path, "session"))

    assert refusal.value.code == "stale_plan"
    assert name in refusal.value.message
    assert sync.read_replica_cursor(tmp_path, "session") == {}


@pytest.mark.parametrize("reported", ["omitted", "stale_digest"])
def test_a_declared_member_with_stale_bytes_is_refused_and_left_alone(
    tmp_path: Path, reported: str
) -> None:
    """Neither the report nor a disk read may bless bytes the owner no longer declares."""
    name = "scratchpad/notes.md"
    dest = tmp_path / "replica"
    (dest / "scratchpad").mkdir(parents=True)
    (dest / "scratchpad" / "notes.md").write_bytes(b"stale")
    have = {"files": {name: sync.sha256_bytes(b"stale")}} if reported == "stale_digest" else {}

    with pytest.raises(sync.SyncRefused) as refusal:
        sync.pull(
            tmp_path,
            "session",
            ask=_complete_plan({name: sync.sha256_bytes(b"notes")}),
            dest_dir=dest,
            attachments_root=dest,
            have=have,
        )

    assert refusal.value.code == "stale_plan"
    assert name in refusal.value.message
    assert (dest / "scratchpad" / "notes.md").read_bytes() == b"stale"


def test_a_declared_link_this_device_lacks_is_refused(tmp_path: Path) -> None:
    name = "scratchpad/link"

    with pytest.raises(sync.SyncRefused) as refusal:
        sync.pull(
            tmp_path,
            "session",
            ask=_complete_plan({name: sync.sha256_bytes(b"notes.md")}, links=(name,)),
            dest_dir=tmp_path / "replica",
            attachments_root=tmp_path / "replica",
        )

    assert refusal.value.code == "stale_plan"
    assert name in refusal.value.message


def test_a_declared_link_present_here_is_attested(tmp_path: Path) -> None:
    """A link is a target TEXT, so that is what is measured when nothing else can be."""
    name = "scratchpad/link"
    dest = tmp_path / "replica"
    (dest / "scratchpad").mkdir(parents=True)
    (dest / "scratchpad" / "notes.md").write_text("notes", encoding="utf-8")
    (dest / "scratchpad" / "link").symlink_to("notes.md")
    digest = sync.sha256_bytes(b"notes.md")

    result = sync.pull(
        tmp_path,
        "session",
        ask=_complete_plan({name: digest}, links=(name,)),
        dest_dir=dest,
        attachments_root=dest,
    )

    assert result.files[name] == digest
    assert result.links == (name,)


def test_an_honest_append_attests_a_held_member_without_hashing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ordinary append omits unchanged members, and their own report must justify them.

    A destination that re-hashed the copy set to attest it would pay for every scratchpad
    on every turn, so the disk check must be unreachable on this path.
    """
    root = tmp_path / "holder"
    owner = root / "owner"
    session = owner / "sessions" / "abc123"
    (session / "scratchpad").mkdir(parents=True)
    transcript = session / "transcript.jsonl"
    original = b'{"line":1}\n'
    transcript.write_bytes(original)
    (session / "scratchpad" / "notes.md").write_bytes(b"notes")
    plans: list[dict[str, Any]] = []
    ask = _owner_ask(owner, "abc123", plans)
    sync.sync_from(root, "abc123", ask=ask, into=sync.replica_dir(root, "abc123"))
    transcript.write_bytes(original + b'{"line":2}\n')
    plans.clear()

    def refuse_to_hash(*args: Any, **kwargs: Any) -> str:
        raise AssertionError(f"the append hashed a member it already holds: {args[:3]}")

    monkeypatch.setattr(sync, "_held_member_digest_now", refuse_to_hash)
    second = sync.sync_from(root, "abc123", ask=ask, into=sync.replica_dir(root, "abc123"))

    assert second["mode"] == "append"
    assert second["items"] == 0
    assert plans[-1]["items"] == []
    cursor = sync.read_replica_cursor(root, "abc123")
    assert cursor["files"]["scratchpad/notes.md"] == sync.sha256_bytes(b"notes")


def test_an_owner_deletion_and_append_reconciles_through_the_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused plan is re-planned against what is REALLY here, and the cursor stays true.

    The owner deletes ``title.json`` and appends a turn. The first plan also declares a
    member this destination does not hold and no transfer carries — a plan assembled from
    stamps that no longer describe the source — and the copy loses a member it had
    reported in the same window. The refusal names a member this copy cannot justify,
    ``sync_from`` reports again, and the second plan re-sends the member that is gone, so
    the finished cursor describes the copy rather than the plan.
    """
    root = tmp_path / "holder"
    owner = root / "owner"
    session = owner / "sessions" / "abc123"
    (session / "scratchpad").mkdir(parents=True)
    transcript = session / "transcript.jsonl"
    original = b'{"line":1}\n'
    transcript.write_bytes(original)
    (session / "title.json").write_text('{"title":"removed"}', encoding="utf-8")
    (session / "scratchpad" / "notes.md").write_bytes(b"notes")
    replica = sync.replica_dir(root, "abc123")
    sync.sync_from(root, "abc123", ask=_owner_ask(owner, "abc123", []), into=replica)

    transcript.write_bytes(original + b'{"line":2}\n')
    (session / "title.json").unlink()
    plans: list[dict[str, Any]] = []
    honest = _owner_ask(owner, "abc123", plans)
    phantom = "scratchpad/phantom.md"
    refusals: list[sync.SyncRefused] = []

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        answer = honest(frame)
        if frame["phase"] == "plan" and len(plans) == 1:
            (replica / "scratchpad" / "notes.md").unlink()
            answer["members"] = {**answer["members"], phantom: sync.sha256_bytes(b"gone")}
            answer["plan_id"] = sync._attested_digest(
                answer["members"], tuple(answer["member_links"]), "abc123"
            )
        return answer

    original_pull = sync.pull

    def recording_pull(*args: Any, **kwargs: Any) -> Any:
        try:
            return original_pull(*args, **kwargs)
        except sync.SyncRefused as exc:
            refusals.append(exc)
            raise

    monkeypatch.setattr(sync, "pull", recording_pull)
    second = sync.sync_from(root, "abc123", ask=ask, into=replica)

    assert [exc.code for exc in refusals] == ["stale_plan"]
    # THE NAME IS DETERMINISTIC: the complete member set is walked in sorted order,
    # ``scratchpad/notes.md`` sorts before the phantom the plan also declares, and its
    # pin therefore runs first — so the refusal names the member this destination lost,
    # and nothing here rests on an accident of member order.
    assert "scratchpad/notes.md" in refusals[0].message
    assert [plan["transcript"]["mode"] for plan in plans] == ["append", "replace"]
    assert "scratchpad/notes.md" in [item["name"] for item in plans[1]["items"]]
    assert second["mode"] == "replace"
    cursor = sync.read_replica_cursor(root, "abc123")
    assert phantom not in cursor["files"]
    assert "title.json" not in cursor["files"]
    assert (replica / "scratchpad" / "notes.md").read_bytes() == b"notes"
    assert not (replica / "title.json").exists()
    for name, digest in cursor["files"].items():
        assert sync.sha256_bytes((replica / name).read_bytes()) == digest, name


# ---------------------------------------------------------------------------
# A report-only member is PINNED by its stat facts, not published on a digest alone
# ---------------------------------------------------------------------------


def _record_refused_pulls(monkeypatch: pytest.MonkeyPatch) -> list[sync.SyncRefused]:
    """Every refusal a pull raised, in order, with the real ``pull`` still running."""
    refusals: list[sync.SyncRefused] = []
    original_pull = sync.pull

    def recording_pull(*args: Any, **kwargs: Any) -> Any:
        try:
            return original_pull(*args, **kwargs)
        except sync.SyncRefused as exc:
            refusals.append(exc)
            raise

    monkeypatch.setattr(sync, "pull", recording_pull)
    return refusals


def _synced_replica_with_a_turn(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A holder whose replica is complete, and an owner that has appended one turn.

    The append is what makes the next plan an ``append``: every member the replica
    already holds at an unchanged digest is then OMITTED from the plan, so the pull has
    nothing but the destination's own report to justify it with — the case these cells
    are about.
    """
    root = tmp_path / "holder"
    owner = root / "owner"
    session = owner / "sessions" / "abc123"
    (session / "scratchpad").mkdir(parents=True)
    transcript = session / "transcript.jsonl"
    original = b'{"line":1}\n'
    transcript.write_bytes(original)
    (session / "scratchpad" / "notes.md").write_bytes(b"notes")
    replica = sync.replica_dir(root, "abc123")
    sync.sync_from(root, "abc123", ask=_owner_ask(owner), into=replica)
    transcript.write_bytes(original + b'{"line":2}\n')
    return root, owner, replica


def test_a_report_justified_member_deleted_during_the_plan_answer_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A member the report justified and the plan omitted must still be HERE when published.

    ``held_report`` digests the copy set once, before the owner is asked; the owner then
    omits what that report describes. A member deleted in that window used to be attested
    from the report anyway, so the cursor claimed a file the replica no longer held and
    ``promote_replica`` refused the copy as ``incomplete_replica``.
    """
    root, owner, replica = _synced_replica_with_a_turn(tmp_path)
    member = replica / "scratchpad" / "notes.md"
    plans: list[dict[str, Any]] = []
    honest = _owner_ask(owner, "abc123", plans)
    refusals = _record_refused_pulls(monkeypatch)

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        answer = honest(frame)
        if frame["phase"] == "plan" and len(plans) == 1:
            member.unlink()
        return answer

    second = sync.sync_from(root, "abc123", ask=ask, into=replica)

    assert [exc.code for exc in refusals] == ["stale_plan"]
    assert "scratchpad/notes.md" in refusals[0].message
    assert [item["name"] for item in plans[1]["items"]] == ["scratchpad/notes.md"]
    assert second["mode"] == "replace"
    assert (replica / "scratchpad" / "notes.md").read_bytes() == b"notes"
    promoted = sync.promote_replica(root, "abc123", new_id="recovered")
    assert promoted["session_id"] == "recovered"
    assert (root / "sessions" / "recovered" / "scratchpad" / "notes.md").read_bytes() == b"notes"


def test_a_report_justified_member_rewritten_during_the_plan_answer_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same window, different bytes: the report's digest may not outlive the bytes it saw."""
    root, owner, replica = _synced_replica_with_a_turn(tmp_path)
    member = replica / "scratchpad" / "notes.md"
    plans: list[dict[str, Any]] = []
    honest = _owner_ask(owner, "abc123", plans)
    refusals: list[sync.SyncRefused] = []
    at_refusal: list[bytes] = []
    original_pull = sync.pull

    def recording_pull(*args: Any, **kwargs: Any) -> Any:
        try:
            return original_pull(*args, **kwargs)
        except sync.SyncRefused as exc:
            refusals.append(exc)
            at_refusal.append(member.read_bytes())
            raise

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        answer = honest(frame)
        if frame["phase"] == "plan" and len(plans) == 1:
            member.write_bytes(b"tampered")
        return answer

    monkeypatch.setattr(sync, "pull", recording_pull)
    second = sync.sync_from(root, "abc123", ask=ask, into=replica)

    assert [exc.code for exc in refusals] == ["stale_plan"]
    assert "scratchpad/notes.md" in refusals[0].message
    # NOT DELETED OR REPAIRED BY THE REFUSAL: the bytes this device wrote stay in place
    # until the retry replaces them with the owner's.
    assert at_refusal == [b"tampered"]
    assert second["mode"] == "replace"
    assert member.read_bytes() == b"notes"
    cursor = sync.read_replica_cursor(root, "abc123")
    for name, digest in cursor["files"].items():
        assert sync.sha256_bytes((replica / name).read_bytes()) == digest, name


def test_an_unchanged_omitted_member_is_pinned_by_one_stat_not_a_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The honest append pays one ``lstat`` per omitted member, and reads no bytes of it.

    This is the cost the attestation may not regress: re-hashing the copy set on every
    turn is what the earlier cost fix removed, so the pin has to answer with stat facts.
    """
    root, owner, replica = _synced_replica_with_a_turn(tmp_path)
    member = replica / "scratchpad" / "notes.md"
    plans: list[dict[str, Any]] = []
    ask = _owner_ask(owner, "abc123", plans)
    opened: list[Path] = []
    original_open = sync._open_for_read

    def recording_open(path: Path, scope: Any) -> int:
        opened.append(Path(path))
        return original_open(path, scope)

    def refuse_to_hash(*args: Any, **kwargs: Any) -> str:
        raise AssertionError(f"the append hashed a member it already holds: {args[:3]}")

    monkeypatch.setattr(sync, "_open_for_read", recording_open)
    monkeypatch.setattr(sync, "_held_member_digest_now", refuse_to_hash)
    second = sync.sync_from(root, "abc123", ask=ask, into=replica)

    assert second["mode"] == "append"
    assert second["items"] == 0
    assert plans[-1]["items"] == []
    assert member.resolve() not in [path.resolve() for path in opened]
    assert "stats" in sync.held_report(replica, replica)
    cursor = sync.read_replica_cursor(root, "abc123")
    assert cursor["files"]["scratchpad/notes.md"] == sync.sha256_bytes(b"notes")


def test_a_report_justified_member_must_also_report_the_declared_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The report's stat facts and its digest are two facts; the plan needs BOTH.

    The facts prove the member the report hashed is still the one on disk; they say
    nothing about the digest the report claims those bytes hash to. Justified on the
    facts alone, a plan that declared a digest this device never held for an omitted
    member was published as the copy's own, so the cursor named bytes the replica did
    not have and a recovery refused it. A reported digest that disagrees with the
    declaration must fall through to the disk digest, which refuses ``stale_plan`` and
    names the member, so the existing retry repairs the copy.
    """
    root, owner, replica = _synced_replica_with_a_turn(tmp_path)
    name = "scratchpad/notes.md"
    member = replica / "scratchpad" / "notes.md"
    plans: list[dict[str, Any]] = []
    honest = _owner_ask(owner, "abc123", plans)
    forged = sync.sha256_bytes(b"a digest this device never held")
    refusals: list[sync.SyncRefused] = []
    at_refusal: list[tuple[bytes, str]] = []
    original_pull = sync.pull

    def recording_pull(*args: Any, **kwargs: Any) -> Any:
        try:
            return original_pull(*args, **kwargs)
        except sync.SyncRefused as exc:
            refusals.append(exc)
            at_refusal.append(
                (member.read_bytes(), sync.read_replica_cursor(root, "abc123")["files"][name])
            )
            raise

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        answer = honest(frame)
        if frame["phase"] == "plan" and len(plans) == 1:
            # STILL A WELL-FORMED PLAN: the member stays in the complete source set, is
            # not transferred, and the identity is recomputed over the forged digest.
            answer["members"][name] = forged
            answer["plan_id"] = sync._attested_digest(
                answer["members"], tuple(answer["member_links"]), "abc123"
            )
        return answer

    monkeypatch.setattr(sync, "pull", recording_pull)
    second = sync.sync_from(root, "abc123", ask=ask, into=replica)

    assert [exc.code for exc in refusals] == ["stale_plan"]
    assert name in refusals[0].message
    # NEITHER THE BYTES NOR THE CURSOR MOVED for the refused member.
    assert at_refusal == [(b"notes", sync.sha256_bytes(b"notes"))]
    assert second["mode"] == "replace"
    cursor = sync.read_replica_cursor(root, "abc123")
    assert cursor["files"][name] == sync.sha256_bytes(b"notes")
    assert cursor["files"][name] == sync.sha256_bytes(member.read_bytes())
    promoted = sync.promote_replica(root, "abc123", new_id="recovered")
    assert promoted["session_id"] == "recovered"
    assert (root / "sessions" / "recovered" / "scratchpad" / "notes.md").read_bytes() == b"notes"


def test_a_link_is_digested_after_the_stat_that_describes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``held_report``'s link branch must take its facts FIRST, as the other two do.

    The documented order is facts, then digest, so a write between the two leaves facts
    the live member no longer matches and the attestation refuses. The link branch read
    its target text first, so a link swapped between the two calls reported the OLD
    target's digest beside the NEW link's facts and the pin accepted that pairing: the
    cursor named a target the replica had moved past, which a recovery refuses. Here the
    walk reads ``notes.md`` and the link becomes ``other.md`` before the stat, so the
    report may only publish the digest of the target it actually measured facts for.
    """
    root = tmp_path / "holder"
    owner = root / "owner"
    session = owner / "sessions" / "abc123"
    (session / "scratchpad").mkdir(parents=True)
    transcript = session / "transcript.jsonl"
    original = b'{"line":1}\n'
    transcript.write_bytes(original)
    (session / "scratchpad" / "notes.md").write_bytes(b"notes")
    (session / "scratchpad" / "link").symlink_to("notes.md")
    replica = sync.replica_dir(root, "abc123")
    sync.sync_from(root, "abc123", ask=_owner_ask(owner), into=replica)
    transcript.write_bytes(original + b'{"line":2}\n')
    link = replica / "scratchpad" / "link"
    name = "scratchpad/link"
    honest_walk = sync.tree_link_entries
    swapped: list[int] = []

    def swapping_walk(directory: Path) -> list[tuple[str, str]]:
        entries = honest_walk(directory)
        if Path(directory) == replica and not swapped:
            # THE WINDOW: the target text has just been read by the walk, and the link
            # changes before ``held_report`` takes the facts that describe it.
            swapped.append(1)
            link.unlink()
            link.symlink_to("other.md")
        return entries

    monkeypatch.setattr(sync, "tree_link_entries", swapping_walk)
    sync.sync_from(root, "abc123", ask=_owner_ask(owner), into=replica)

    assert swapped == [1]
    cursor = sync.read_replica_cursor(root, "abc123")
    # THE CURSOR NAMES A TARGET THE REPLICA REALLY HOLDS: the pull either carried the
    # link from the plan (the honest repair) or refused — it may never publish the
    # digest of a target it read before the stat describing the link.
    assert cursor["files"][name] == sync.sha256_bytes(b"notes.md")
    assert sync.held_report(replica, replica)["files"][name] == cursor["files"][name]


@pytest.mark.parametrize("honest", [True, False])
def test_a_report_without_stat_facts_is_justified_by_a_digest_not_by_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, honest: bool
) -> None:
    """A ``have`` from a caller that carries no facts falls back to the disk digest.

    The bare digest is the same report and cannot justify itself: a member whose bytes
    disagree with it is refused rather than published on the report's word.
    """
    name = "scratchpad/notes.md"
    dest = tmp_path / "replica"
    (dest / "scratchpad").mkdir(parents=True)
    declared = sync.sha256_bytes(b"notes")
    (dest / "scratchpad" / "notes.md").write_bytes(b"notes" if honest else b"forged")
    measured: list[str] = []
    original_digest = sync._held_member_digest_now

    def recording_digest(*args: Any, **kwargs: Any) -> str:
        measured.append(str(args[2]))
        return original_digest(*args, **kwargs)

    monkeypatch.setattr(sync, "_held_member_digest_now", recording_digest)
    have = {"files": {name: declared}}

    if honest:
        result = sync.pull(
            tmp_path,
            "session",
            ask=_complete_plan({name: declared}),
            dest_dir=dest,
            attachments_root=dest,
            have=have,
        )
        assert result.files[name] == declared
    else:
        with pytest.raises(sync.SyncRefused) as refusal:
            sync.pull(
                tmp_path,
                "session",
                ask=_complete_plan({name: declared}),
                dest_dir=dest,
                attachments_root=dest,
                have=have,
            )
        assert refusal.value.code == "stale_plan"
        assert name in refusal.value.message
        assert (dest / "scratchpad" / "notes.md").read_bytes() == b"forged"
    assert measured == [name]


@pytest.mark.parametrize("name", ["scratchpad/../../escape.md", "attachments/../escape.bin"])
def test_an_unsafe_declared_member_is_unknown_not_a_stale_plan_retry(
    tmp_path: Path, name: str
) -> None:
    """An escaping member name is refused as one, and a retry is not what it needs.

    A GUARD rather than a red cell: the plan's own destination check already answers this
    before the attestation runs (``_validate_plan_destinations``). It is here because the
    attestation must not turn that refusal into an absence if the two ever drift apart,
    and the plan count pins that no retry was spent on it.
    """
    dest = sync.replica_dir(tmp_path, "session")
    dest.mkdir(parents=True)
    served: list[int] = []
    serving = _complete_plan({name: sync.sha256_bytes(b"payload")})

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        if frame["phase"] == "plan":
            served.append(1)
        return serving(frame)

    with pytest.raises(sync.SyncRefused) as refusal:
        sync.sync_from(tmp_path, "session", ask=ask, into=dest)

    assert refusal.value.code == "unknown_item"
    assert name in refusal.value.message
    assert len(served) == 1


def test_a_refused_member_read_is_unknown_not_a_stale_plan_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal from the scoped read is a refusal, not an absence.

    The unraced containment shapes are answered before the read itself (the containment
    walk, then the leaf checks), so this reaches the handler where it is still reachable:
    the read seam answers the ``unknown_item`` that ``_refuse_io`` produces, and that has
    to reach the caller unchanged. ``stale_plan`` is what ``sync_from`` retries, so
    swallowing the refusal as an absence reports the wrong problem for a member that is
    present and cannot be satisfied by asking again.
    """
    name = "scratchpad/notes.md"
    dest = tmp_path / "replica"
    (dest / "scratchpad").mkdir(parents=True)
    declared = sync.sha256_bytes(b"notes")
    (dest / "scratchpad" / "notes.md").write_bytes(b"notes")
    original_open = sync._open_for_read

    def refuse_the_member(path: Path, scope: Any) -> int:
        if Path(path).name == "notes.md":
            raise sync.SyncRefused(
                "unknown_item", "Sync member 'notes.md' was not written: it is unsafe."
            )
        return original_open(path, scope)

    monkeypatch.setattr(sync, "_open_for_read", refuse_the_member)

    with pytest.raises(sync.SyncRefused) as refusal:
        sync.pull(
            tmp_path,
            "session",
            ask=_complete_plan({name: declared}),
            dest_dir=dest,
            attachments_root=dest,
        )

    assert refusal.value.code == "unknown_item"
    assert refusal.value.message == "Sync member 'notes.md' was not written: it is unsafe."
