"""WHAT A SESSION DIRECTORY MAY HOLD, enumerated from the modules that write it.

THE TEST THE LAST DATA-LOSS BUG ARGUED FOR (review round 1, B-M2). ``scratchpad/``
and ``created_at.json`` were in neither the copy set nor the exclusion list, so a
move did not carry them and then deleted the source directory they were in: on the
operator's workstation that is 3,017 scratchpads and 10,840 birth-time sidecars, and
neither was recoverable. A copy set written as a list of files cannot notice a file
type nobody told it about, so the two guards here invert the question:

* :func:`test_every_entry_type_this_product_writes_is_copied_or_excluded` IMPORTS
  each name from the module that owns it, so a new sidecar lands in this test the
  moment its constant exists, and the failure says "add it to one list or the
  other, with a reason";
* :func:`test_a_directory_of_every_entry_type_round_trips` builds a directory
  holding ONE of every such entry and copies it for real, then asserts every
  classified entry either arrived byte-identical or did NOT arrive and is named in
  ``EXCLUDED_ENTRIES`` — which is what makes ``scratchpad/`` (a tree) and
  ``created_at.json`` (a file) both covered by one property rather than by two
  hand-written assertions;
* and ``sync.assert_complete`` is what makes the answer FAIL CLOSED in the product
  when the copy set is behind: a move refuses rather than deleting what it did not
  copy, so the next unlisted entry type is a refusal with a sentence, not another
  silent loss.

NOTHING HERE IS A GUESS ABOUT THE MISSING NAMES: the list below is measured, and
the comment on each entry says where the measurement comes from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import sync

# ---------------------------------------------------------------------------
# The entry types, imported from their owners
# ---------------------------------------------------------------------------


def _entry_owners() -> dict[str, str]:
    """``name -> the module that owns it`` for every entry a session may hold.

    Imported rather than spelled, deliberately: this is the list that has to grow
    when the product grows, and an import makes that automatic where a literal
    would not. The trees are included as directory names.
    """
    from local_operator.browser_bridge.resources import RESOURCE_NAME
    from local_operator.fork import FORK_BOUNDARY_NAME
    from local_operator.resume import (
        ATTACHMENT_SIDECAR_NAME,
        ORIGIN_CACHE_NAME,
        ORIGIN_NAME,
        TITLE_SIDECAR_NAME,
    )
    from local_operator.scratchpad import SCRATCHPAD_DIRNAME
    from local_operator.session.creation import CREATED_AT_NAME
    from local_operator.session.placement import MESH_STAMP_NAME
    from local_operator.session.retention import (
        ATTACHMENT_SIDECAR_FILENAME,
        DESKTOP_MARKER_NAME,
        LIVE_MARKER_NAME,
        TRANSCRIPT_FILENAME,
    )
    from local_operator.session.runtime.inbox import INBOX_NAME
    from local_operator.session.runtime.registry import (
        STOP_MARKER_NAME,
        TURN_JOURNAL_NAME,
    )
    from local_operator.session_lease import LEASE_NAME, MIRROR_NAME, RECOVERY_LOCK_NAME
    from local_operator.wakes.lock import WAKE_LOCK_NAME

    return {
        TRANSCRIPT_FILENAME: "session.transcript",
        TITLE_SIDECAR_NAME: "resume",
        ATTACHMENT_SIDECAR_NAME: "resume",
        ATTACHMENT_SIDECAR_FILENAME: "resume",
        ORIGIN_NAME: "resume",
        ORIGIN_CACHE_NAME: "resume",
        CREATED_AT_NAME: "session.creation",
        TURN_JOURNAL_NAME: "session.runtime.registry",
        STOP_MARKER_NAME: "session.runtime.registry",
        INBOX_NAME: "session.runtime.inbox",
        FORK_BOUNDARY_NAME: "fork",
        DESKTOP_MARKER_NAME: "session.retention",
        MESH_STAMP_NAME: "session.placement",
        LIVE_MARKER_NAME: "session.retention",
        LEASE_NAME: "session_lease",
        MIRROR_NAME: "session_lease",
        RECOVERY_LOCK_NAME: "session_lease",
        WAKE_LOCK_NAME: "wakes.lock",
        RESOURCE_NAME: "browser_bridge.resources",
        # The copy machinery's own two: a replica's cursor (never inside a session)
        # and the move's boot marker, which lives in a staging directory and is
        # deleted by the promote (a crash can leave one inside a promoted session,
        # which is why it is classified rather than ignored).
        sync.REPLICA_CURSOR_NAME: "network.sync",
        "ready.json": "network.mobility",
        # THE TREES. A directory, so it is classified as one.
        SCRATCHPAD_DIRNAME: "scratchpad",
    }


#: The names MEASURED in the operator's live store on 2026-09-24 (10,841 session
#: directories, ``ls | sort | uniq -c``, counts in the report). Kept separate from
#: the imported list because it is the EVIDENCE that the imported list is complete:
#: a name here that no constant owns is a writer nobody has found yet, and the test
#: below fails rather than letting it be dropped.
MEASURED_IN_THE_STORE: tuple[str, ...] = (
    "created_at.json",
    "transcript.jsonl",
    "title-scan.json",
    "origin.json",
    "scratchpad",
    ".browser-resource.json",
    "origin-scan.json",
    "title.json",
    "subagent-roster.v1.json",
    "attachment.json",
    "desktop.json",
    ".session.pid",
    "turn-journal.json",
    "inbox.jsonl",
    ".execution-lease.recovery",
    ".execution-lease",
    "runtime-stop.json",
    ".wake-write.lock",
    "mesh.json",
)


def _classified(name: str) -> bool:
    return name in sync.COPY_SET_NAMES or name in sync.COPY_SET_TREES or name in sync.NEVER_COPIED


# ---------------------------------------------------------------------------
# The two lists together have to cover everything
# ---------------------------------------------------------------------------


def test_every_entry_type_this_product_writes_is_copied_or_excluded() -> None:
    """Every entry type is copied, or excluded WITH ITS REASON. No third answer."""
    unclassified = sorted(name for name in _entry_owners() if not _classified(name))
    assert unclassified == [], (
        f"a session directory can hold {unclassified}, and the copy set neither "
        "carries it nor excludes it: a move would delete it. Add each name to "
        "sync.COPY_SET_NAMES (it travels), sync.COPY_SET_TREES (its files travel) "
        "or sync.EXCLUDED_ENTRIES (with the reason it must not)"
    )
    # Every excluded name states why, and the two exclusion views agree.
    assert set(sync.EXCLUDED_ENTRIES) == set(sync.NEVER_COPIED)
    for name in sorted(sync.NEVER_COPIED):
        assert sync.EXCLUDED_ENTRIES[name].strip(), f"{name} is excluded with no reason"
    # A name cannot be in two answers at once.
    assert not (set(sync.COPY_SET_NAMES) & set(sync.NEVER_COPIED))
    assert not (set(sync.COPY_SET_TREES) & set(sync.NEVER_COPIED))


def test_the_measured_store_entries_are_all_classified() -> None:
    """Every name the operator's real store holds is classified by this build.

    THE EVIDENCE HALF. The imported list above is only as complete as the constants
    it can find; this one is a census of the machine the feature runs on, so a
    writer that spells its name inline instead of exporting a constant still shows
    up instead of being lost in a move.
    """
    unclassified = sorted(name for name in MEASURED_IN_THE_STORE if not _classified(name))
    assert unclassified == [], (
        f"the live store holds {unclassified}, which this build's copy set does not "
        "classify; measure the writer and add it to sync.COPY_SET_NAMES, "
        "sync.COPY_SET_TREES or sync.EXCLUDED_ENTRIES"
    )


# ---------------------------------------------------------------------------
# The round trip: one of every entry type, copied for real
# ---------------------------------------------------------------------------


def _ask(root: Path) -> Any:
    """The owner's half, in process — the same shape ``test_sync.py`` uses."""

    def ask(frame: dict[str, Any]) -> dict[str, Any]:
        phase = str(frame.get("phase") or "")
        session_id = str(frame.get("session_id") or "")
        if phase == "plan":
            return sync.build_manifest(
                root, session_id, have=frame.get("have") if isinstance(frame, dict) else {}
            )
        if phase == "fetch":
            return sync.serve_fetch(
                root,
                session_id,
                plan=str(frame.get("plan_id") or ""),
                name=str(frame.get("name") or ""),
                offset=int(frame.get("offset") or 0),
                limit=int(frame.get("limit") or sync.SYNC_CHUNK_BYTES),
            )
        if phase == "verify":
            return sync.serve_verify(
                root,
                session_id,
                plan=str(frame.get("plan_id") or ""),
                name=str(frame.get("name") or ""),
                prefix_bytes=int(frame.get("prefix_bytes") or 0),
                prefix_digest=str(frame.get("prefix_digest") or ""),
            )
        raise AssertionError(f"unexpected phase {phase!r}")

    return ask


def _every_entry_directory(root: Path, session_id: str) -> Path:
    """One session directory holding ONE of every entry type, each with its own bytes.

    Distinct content per entry so "did it arrive" is a byte comparison rather than a
    presence check, and a tree with a nested directory so the tree half is real.
    """
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    for name in sorted(set(_entry_owners()) | set(MEASURED_IN_THE_STORE)):
        if name in sync.COPY_SET_TREES:
            continue
        (directory / name).write_text(f"content of {name}\n", encoding="utf-8")
    # The transcript is JSONL and (in the real product) references attachments, so it
    # gets a real row plus a referenced blob and its sidecar in the shared store.
    ref = "a" * 32
    (directory / "transcript.jsonl").write_text(
        json.dumps({"id": "e1", "type": "image", "attachment": ref}) + "\n", encoding="utf-8"
    )
    store = root / "attachments"
    store.mkdir(parents=True, exist_ok=True)
    (store / f"{ref}.bin").write_bytes(b"blob bytes")
    (store / f"{ref}.json").write_text(json.dumps({"mime_type": "image/png"}), encoding="utf-8")
    tree = directory / "scratchpad"
    (tree / "nested").mkdir(parents=True, exist_ok=True)
    (tree / "notes.md").write_text("notes\n", encoding="utf-8")
    (tree / "nested" / "run.txt").write_text("nested content\n", encoding="utf-8")
    return directory


def test_a_directory_of_every_entry_type_round_trips(tmp_path: Path) -> None:
    """Copy a directory holding every entry type: each one arrives or is excluded.

    THIS IS THE PROPERTY THAT REPLACES A LIST OF SPECIAL CASES. For every classified
    entry the outcome is asserted in the direction the classification promises —
    byte-identical at the destination for a copied name or a tree file, ABSENT for
    an excluded one — and a name that is neither would fail this test on its
    presence assertion (``copied`` is derived from the two lists, so a name added to
    one of them is exercised without editing this test).
    """
    source = tmp_path / "owner"
    source.mkdir()
    session_id = "b7d1c0ffee42"
    directory = _every_entry_directory(source, session_id)
    dest = tmp_path / "holder"

    sync.sync_from(source, session_id, ask=_ask(source), into=dest)

    for name in sorted(set(_entry_owners()) | set(MEASURED_IN_THE_STORE)):
        arrives = name in sync.COPY_SET_NAMES or name in sync.COPY_SET_TREES
        if name == "transcript.jsonl":
            assert (dest / name).read_bytes() == (directory / name).read_bytes()
            continue
        if name in sync.COPY_SET_TREES:
            # A tree is classified as a DIRECTORY: its own assertions are below, and
            # this is the one entry whose files (not itself) are the copy's members.
            assert (dest / name).is_dir(), f"{name} is a copy-set tree but did not arrive"
            continue
        if arrives:
            assert (dest / name).is_file(), f"{name} is in the copy set but did not arrive"
            assert (dest / name).read_bytes() == (directory / name).read_bytes(), name
        else:
            assert name in sync.EXCLUDED_ENTRIES, name
            assert not (
                dest / name
            ).exists(), f"{name} is excluded ({sync.EXCLUDED_ENTRIES[name]}) but arrived anyway"
    # The tree, including its nested directory, and the referenced blob's sidecar.
    assert (dest / "scratchpad" / "notes.md").read_text(encoding="utf-8") == "notes\n"
    assert (dest / "scratchpad" / "nested" / "run.txt").read_text(encoding="utf-8") == (
        "nested content\n"
    )
    assert (dest / "attachments" / f"{'a' * 32}.bin").is_file()
    assert (dest / "attachments" / f"{'a' * 32}.json").is_file()


# ---------------------------------------------------------------------------
# Fail closed: an entry nobody classified refuses a DELETING move
# ---------------------------------------------------------------------------


def test_an_unlisted_entry_refuses_a_deleting_move(tmp_path: Path) -> None:
    """``assert_complete`` names it, and the source is untouched.

    The product's answer to "the copy set is behind" has to be a refusal rather than
    a copy-it-and-delete-the-rest, because the alternative is the bug this file
    exists for. The sentence must name the entry, since the person reading it is the
    one who can move or remove it.
    """
    directory = tmp_path / "sessions" / "abc123"
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    (directory / "a-file-from-the-future.dat").write_bytes(b"\x00\x01")

    with pytest.raises(sync.SyncRefused) as refusal:
        sync.assert_complete(directory)
    assert refusal.value.code == "unlisted_content"
    assert "a-file-from-the-future.dat" in refusal.value.message
    assert (directory / "a-file-from-the-future.dat").exists()


def test_an_irregular_entry_in_a_tree_refuses_a_deleting_move(tmp_path: Path) -> None:
    """A symlink inside ``scratchpad/`` is not portable data, so a move refuses.

    A symlink's target names a path on the SOURCE device; writing the same text on
    the destination points an agent at a different file (or at nothing). Copying it
    would be a lie and skipping it silently would be a deletion, so the move refuses
    with the path named. Measured on the operator's workstation: 100 of 3,019
    scratchpads hold one.
    """
    directory = tmp_path / "sessions" / "abc123"
    (directory / "scratchpad").mkdir(parents=True)
    (directory / "transcript.jsonl").write_text("{}\n", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("elsewhere\n", encoding="utf-8")
    (directory / "scratchpad" / "link.txt").symlink_to(outside)

    with pytest.raises(sync.SyncRefused) as refusal:
        sync.assert_complete(directory)
    assert refusal.value.code == "unlisted_content"
    assert "scratchpad/link.txt" in refusal.value.message

    # The plan reports it too, without failing: a --keep copy carries what it can and
    # leaves the source alone, so it has nothing to lose by skipping it.
    from tests.unit.network.test_sync import seed  # noqa: PLC0415 — the shared fixture

    seed(tmp_path, "abc123")
    plan = sync.build_manifest(tmp_path, "abc123")
    assert "scratchpad/link.txt" in plan["trees_skipped"]
    assert "scratchpad/link.txt" not in plan["trees"]


def test_the_tree_walk_is_bounded_to_the_tree(tmp_path: Path) -> None:
    """A tree entry name cannot escape the tree, and a peer cannot ask for one.

    ``_tree_entry_path`` is the peer-facing guard: names arrive in fetch frames, so
    ``scratchpad/../../etc/passwd`` has to be refused rather than resolved. The
    attachment branch gets the same treatment (``_blob_file_name``), because a
    destination WRITES to the path that name resolves to.
    """
    from local_operator.network.sync import _item_path

    tmp_path.mkdir(parents=True, exist_ok=True)
    session = tmp_path / "sessions" / "abc123"
    session.mkdir(parents=True)
    (session / "scratchpad").mkdir()
    (session / "scratchpad" / "notes.md").write_text("ok\n", encoding="utf-8")

    assert _item_path(tmp_path, "abc123", "scratchpad/notes.md", None) == (
        session / "scratchpad" / "notes.md"
    )
    for escape in (
        "scratchpad/../transcript.jsonl",
        "scratchpad/../../outside.txt",
        "/etc/passwd",
        "scratchpad/",
        "scratchpad/./notes.md",
    ):
        assert _item_path(tmp_path, "abc123", escape, None) is None, escape
    for escape in ("attachments/../../etc/passwd", "attachments/", "attachments/a/b.bin"):
        assert _item_path(tmp_path, "abc123", escape, None) is None, escape


def test_a_copy_does_not_carry_the_store_of_an_unreferenced_blob(tmp_path: Path) -> None:
    """Only blobs the transcript references travel, with their sidecars.

    The store is never pruned, so it accumulates blobs no session uses. Copying the
    whole store would make every move cost the install's entire attachment history
    (and would put another session's content in this one's replica), so the copy set
    is the transcript's own references — and the sidecar travels with its blob, or
    the destination cannot resolve the mime type (review round 1, M-1).
    """
    source = tmp_path / "owner"
    source.mkdir()
    session_id = "c0ffee123456"
    directory = source / "sessions" / session_id
    directory.mkdir(parents=True)
    used, unused = "1" * 32, "2" * 32
    (directory / "transcript.jsonl").write_text(
        json.dumps({"id": "e1", "type": "image", "attachment": used}) + "\n", encoding="utf-8"
    )
    store = source / "attachments"
    store.mkdir()
    for ref in (used, unused):
        (store / f"{ref}.bin").write_bytes(b"bytes")
        (store / f"{ref}.json").write_text("{}", encoding="utf-8")

    dest = tmp_path / "holder"
    sync.sync_from(source, session_id, ask=_ask(source), into=dest)

    assert (dest / "attachments" / f"{used}.bin").is_file()
    assert (dest / "attachments" / f"{used}.json").is_file()
    assert not (dest / "attachments" / f"{unused}.bin").exists()
