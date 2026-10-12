"""Unpublished raw-offset proof: settled projection is a separate dependency seam."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from local_operator.harness.types import (
    CustomMessage,
    ImageContent,
    Message,
    TextContent,
    ToolCall,
    ToolResult,
)
from local_operator.session import transcript_index as ti
from local_operator.session.attachments import AttachmentStore
from local_operator.session.history_window import display_window
from local_operator.session.transcript import (
    ENTRY_CUSTOM,
    ENTRY_MESSAGE,
    ENTRY_PRUNE,
    Transcript,
    TranscriptEntry,
    _entry_to_message,
    encode_message_payload,
    read_indexed_raw_rows,
)

SID = "feedface0001"


def _write(root, entries):
    directory = root / "sessions" / SID
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("".join(e.to_json() + "\n" for e in entries))
    return directory


def _index(root):
    index = ti.refresh_index(root, SID)
    assert index is not None
    return index


def _window(transcript, last_id):
    return display_window(
        transcript,
        conversation_id=SID,
        owner_epoch="proof-epoch",
        through_id=last_id,
        max_messages=512,
    ).messages


def test_actual_owner_projection_equivalence_only_when_complete_closure_is_bounded(tmp_path):
    directory = tmp_path / "sessions" / SID
    attachments = AttachmentStore(directory)
    messages = [
        Message.user("fixture user"),
        Message.assistant(
            "fixture assistant", tool_calls=[ToolCall(id="call", name="read", arguments={})]
        ),
        Message.tool_result(
            ToolResult(
                tool_call_id="call", tool_name="read", content=[TextContent(text="fixture result")]
            )
        ),
    ]
    messages += [
        CustomMessage(custom_type="session_state", details={"text": "fixture injection"}),
        Message(role="user", content=[ImageContent(data="aGVsbG8=", mime_type="image/png")]),
    ]
    entries = [
        TranscriptEntry(m.id, 1.0, ENTRY_MESSAGE, encode_message_payload(m, attachments))
        for m in messages
    ]
    # Supplements are ENTRY_CUSTOM bookkeeping with an anchor, not arbitrary
    # CustomMessage conversation content. Raw presence cannot certify their
    # separate owner projection/anchor dependency as a display message body.
    entries.append(
        TranscriptEntry(
            "supplement-proof",
            1.5,
            ENTRY_CUSTOM,
            {
                "custom_type": "supplement_v1",
                "details": {"anchor": messages[1].id, "state": "done"},
            },
        )
    )
    entries.append(
        TranscriptEntry(
            "checkpoint-proof", 1.5, ENTRY_CUSTOM, {"custom_type": "checkpoint", "details": {}}
        )
    )
    target = messages[2].id
    entries.append(
        TranscriptEntry("prune-proof", 2.0, ENTRY_PRUNE, {"target": target, "notice": "[pruned]"})
    )
    directory = _write(tmp_path, entries)
    index = _index(tmp_path)
    ids = list(index.journal_rows)
    raw = read_indexed_raw_rows(directory, index, ids)
    assert raw is not None and not raw["remaining"] and not raw["missing"]
    assert set(index.journal_ranges) == set(index.journal_rows)
    persisted = ti.read_index(tmp_path, SID)
    assert persisted is not None and persisted.journal_ranges == index.journal_ranges
    owner = Transcript(directory)
    candidate = Transcript(tmp_path / "bounded-candidate")
    candidate._entries = raw["rows"]
    candidate._attachments = attachments
    actual, projected = _window(owner, entries[-1].id), _window(candidate, entries[-1].id)
    assert actual == projected
    partial = read_indexed_raw_rows(directory, index, [target])
    assert partial is not None
    unsafe = _entry_to_message(partial["rows"][0], attachments)
    settled = next(m for m in owner.build_llm_history() if m.id == target)
    assert (
        unsafe.model_dump() != settled.model_dump()
    ), "raw converter must not resurrect pruned content"
    # Offsets have no reverse target→later-prune edge. All source IDs is a safe
    # small-fixture closure, not a universal bounded projector for long journals.
    output = os.environ.get("PEER_PROOF_OUTPUT")
    if output:
        path = Path(output)
        path.mkdir(parents=True, exist_ok=True)
        body = json.dumps(actual, sort_keys=True, default=str).encode()
        private = path / "owner-projection.private.json"
        private.write_bytes(body)
        private.chmod(0o600)
        (path / "matrix-A.json").write_text(
            json.dumps(
                {
                    "actual_owner_projector": "history_window.display_window(Transcript)",
                    "full_closure_count": len(ids),
                    "full_closure_equivalent": True,
                    "owner_body_sha256": hashlib.sha256(body).hexdigest(),
                    "projection_shapes": (
                        "user/assistant/tool/custom/injection/media + real custom bookkeeping"
                    ),
                    "supplement": (
                        "raw present, not a display-window message; "
                        "separate anchor projection UNPROVEN"
                    ),
                    "bounded_projection": "same actual projector over bounded complete rows",
                    "partial_pruned_raw_converter": (
                        "FAIL: different from settled owner body; "
                        "raw presence is not payload readiness"
                    ),
                    "universal_dependency_closure": (
                        "UNAVAILABLE: index has no target/prune/call/compaction/"
                        "injection dependency edges"
                    ),
                },
                indent=2,
            )
        )


def test_append_ranges_and_same_id_rewrite_never_reattest_old_body(tmp_path):
    old = TranscriptEntry("same-id", 1, ENTRY_MESSAGE, encode_message_payload(Message.user("old")))
    directory = _write(tmp_path, [old])
    first = _index(tmp_path)
    extra = TranscriptEntry(
        "append-id", 2, ENTRY_MESSAGE, encode_message_payload(Message.user("append"))
    )
    with (directory / "transcript.jsonl").open("a") as handle:
        handle.write(extra.to_json() + "\n")
    grown = _index(tmp_path)
    assert grown.journal_generation == first.journal_generation
    assert grown.journal_ranges["same-id"] == first.journal_ranges["same-id"]
    new = TranscriptEntry("same-id", 3, ENTRY_MESSAGE, encode_message_payload(Message.user("new")))
    replacement = directory / "replacement"
    replacement.write_text(new.to_json() + "\n")
    replacement.replace(directory / "transcript.jsonl")
    rebuilt = _index(tmp_path)
    assert rebuilt.journal_generation != grown.journal_generation
    assert read_indexed_raw_rows(directory, grown, ["same-id"]) is None
    fresh = read_indexed_raw_rows(directory, rebuilt, ["same-id"])
    assert fresh is not None and fresh["rows"][0].payload == new.payload
    if os.environ.get("PEER_PROOF_OUTPUT"):
        out = Path(os.environ["PEER_PROOF_OUTPUT"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "matrix-A-rewrite.json").write_text(
            json.dumps(
                {
                    "original_generation": first.journal_generation,
                    "append_generation": grown.journal_generation,
                    "rewrite_generation": rebuilt.journal_generation,
                    "prefix_range_preserved": True,
                    "old_index_raw_read": "unavailable",
                    "same_id_new_body": "verified against actual rewritten row",
                    "rewritten_raw_rows": len(fresh["rows"]),
                },
                indent=2,
            )
        )


@pytest.mark.parametrize("bad", ["malformed", [""], ["x" * 129]])
def test_raw_lookup_rejects_malformed_shape(tmp_path, bad):
    directory = _write(
        tmp_path,
        [TranscriptEntry("row", 1, ENTRY_MESSAGE, encode_message_payload(Message.user("")))],
    )
    with pytest.raises(ValueError, match="bounded"):
        read_indexed_raw_rows(directory, _index(tmp_path), bad)


def test_persisted_ranges_fail_closed_on_bad_shape(tmp_path):
    _write(
        tmp_path,
        [TranscriptEntry("row", 1, ENTRY_MESSAGE, encode_message_payload(Message.user("")))],
    )
    _index(tmp_path)
    raw = json.loads(ti.index_path(tmp_path, SID).read_text())
    raw["journal_ranges"]["row"] = [-1, 0]
    assert ti._index_from_raw(raw) is None
    raw["journal_ranges"]["row"] = [0, raw["sig"]["size"]]
    raw["journal_rows"]["row"] = True
    assert ti._index_from_raw(raw) is None


def test_fd_path_replacement_and_budget_refuse_unverified_payload(tmp_path, monkeypatch):
    entry = TranscriptEntry(
        "row", 1, ENTRY_MESSAGE, encode_message_payload(Message.user("fixture"))
    )
    directory = _write(tmp_path, [entry])
    index = _index(tmp_path)
    limited = read_indexed_raw_rows(directory, index, ["row", "absent"], max_bytes=1)
    assert limited == {"rows": [], "missing": ["absent"], "remaining": ["row"], "bytes": 0}
    original = Path.stat
    path = directory / "transcript.jsonl"

    def replace_at_path_check(self, *args, **kwargs):
        if self == path:
            replacement = directory / "replacement"
            replacement.write_text(entry.to_json() + "\n")
            replacement.replace(path)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", replace_at_path_check)
    assert read_indexed_raw_rows(directory, index, ["row"]) is None


def test_large_dependency_closure_is_not_silently_truncated(tmp_path):
    entries = [
        TranscriptEntry(f"row-{i}", 1, ENTRY_MESSAGE, encode_message_payload(Message.user("")))
        for i in range(513)
    ]
    directory = _write(tmp_path, entries)
    index = _index(tmp_path)
    with pytest.raises(ValueError, match="bounded"):
        read_indexed_raw_rows(directory, index, list(index.journal_ranges))
