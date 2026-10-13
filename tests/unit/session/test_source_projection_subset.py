"""Arbitrary selected-range projection proof, not a faux whole-owner Transcript."""

import json
import os
from pathlib import Path

import pytest

from local_operator.session import transcript as tr
from local_operator.session import transcript_index as ti

SID = "feedface0007"


def row(id_, role="user", text="fixture", **payload):
    return tr.TranscriptEntry(
        id_,
        1,
        tr.ENTRY_MESSAGE,
        {"kind": "message", "role": role, "content": [{"text": text}], **payload},
    )


def prune(id_, target, notice):
    return tr.TranscriptEntry(id_, 2, tr.ENTRY_PRUNE, {"target": target, "notice": notice})


def write(root, entries):
    directory = root / "sessions" / SID
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        "".join(entry.to_json() + "\n" for entry in entries)
    )
    index = ti.refresh_index(root, SID)
    assert index is not None
    return directory, index


def get(directory, index, ids, **kwargs):
    return tr.project_indexed_source_subset(
        directory, index, ids, expected_generation=index.journal_generation, **kwargs
    )


def partition(result, requested):
    sets = [set(result[key]) for key in ("present", "missing", "unavailable", "remaining")]
    assert set.union(*sets) == set(requested)
    assert sum(len(s) for s in sets) == len(set(requested))


def test_long_warm_selected_ranges_and_far_later_last_prune(tmp_path, monkeypatch):
    entries = [
        row(
            "call-owner",
            "assistant",
            tool_calls=[{"id": "far-call", "name": "read", "arguments": {}}],
        ),
        row("target", "tool", "must not resurrect", tool_call_id="far-call", tool_name="read"),
    ]
    entries += [row(f"filler-{i}") for i in range(10001)]
    entries += [prune("prune-first", "target", "first"), prune("prune-last", "target", "last")]
    directory, index = write(tmp_path, entries)
    owner = tr.Transcript(directory)
    oracle = {message.id: message.model_dump(mode="json") for message in owner.build_llm_history()}
    cut_oracle = {
        message.id: message.model_dump(mode="json")
        for message in owner.build_llm_history(through_id="prune-first")
    }
    reads, seeks = [], []
    original_open = Path.open

    class ReadCounter:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def fileno(self):
            return self.handle.fileno()

        def seek(self, offset):
            seeks.append(offset)
            return self.handle.seek(offset)

        def read(self, size):
            data = self.handle.read(size)
            reads.append(len(data))
            return data

    def opened(self, mode="r", *args, **kwargs):
        handle = original_open(self, mode, *args, **kwargs)
        return (
            ReadCounter(handle)
            if self == directory / "transcript.jsonl" and mode == "rb"
            else handle
        )

    def forbidden(*args, **kwargs):
        pytest.fail("measured request attempted whole-file load/replay/page/cache derivation")

    monkeypatch.setattr(Path, "open", opened)
    for name in ("replay_entries", "read_replay_suffix", "read_transcript_page"):
        monkeypatch.setattr(tr, name, forbidden)
    monkeypatch.setattr(tr.Transcript, "__init__", forbidden)
    monkeypatch.setattr(ti.SourceProjectionViews, "build", forbidden)
    result = get(directory, index, ["target", "absent", "filler-9000"])
    assert result is not None
    partition(result, ["target", "absent", "filler-9000"])
    assert result["present"] == ["target", "filler-9000"] and result["missing"] == ["absent"]
    assert result["source_dependency_ids"] == ["call-owner", "prune-last"]
    assert set(result["read_ids"]) == {"call-owner", "target", "prune-last", "filler-9000"}
    assert sum(reads) == result["read_bytes"] < 512 * 1024
    expected_ranges = [index.journal_ranges[key] for key in result["read_ids"]]
    assert seeks == [start for start, _end in expected_ranges]
    for body in result["entries"]:
        assert body["message"] == oracle[body["id"]]
    assert "must not resurrect" not in json.dumps(result)
    cut = get(directory, index, ["target"], through_id="prune-first")
    assert cut["entries"][0]["message"] == cut_oracle["target"]
    assert cut["source_dependency_ids"] == ["call-owner", "prune-first"]
    output = os.environ.get("PEER_PROOF_OUTPUT")
    if output:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "subset-long-matrix.json").write_text(
            json.dumps(
                {
                    "journal_rows": len(entries),
                    "requested": 3,
                    "selected_raw_ids": result["read_ids"],
                    "selected_raw_bytes": result["read_bytes"],
                    "whole_helpers_fail_closed": True,
                    "actual_owner_body_equivalent": True,
                    "effective_prune": "prune-last",
                    "through_cut_prune": "prune-first",
                    "generation": index.journal_generation,
                },
                indent=2,
            )
        )


def test_nonadjacent_multicall_unit_and_ambiguous_ask_hidden_declines(tmp_path):
    entries = [
        row(
            "owner",
            "assistant",
            tool_calls=[
                {"id": "a", "name": "read", "arguments": {}},
                {"id": "b", "name": "bash", "arguments": {}},
            ],
        ),
        row("plain"),
        row("result-a", "tool", tool_call_id="a", tool_name="read"),
        row("between"),
        row("result-b", "tool", tool_call_id="b", tool_name="bash"),
        prune("pruned-b", "result-b", "blanked"),
    ]
    directory, index = write(tmp_path, entries)
    oracle = {m.id: m.model_dump(mode="json") for m in tr.Transcript(directory).build_llm_history()}
    result = get(directory, index, ["result-a"])
    assert result["present"] == ["result-a"]
    assert set(result["source_dependency_ids"]) == {"owner", "result-b", "pruned-b"}
    assert {e["id"] for e in result["dependency_entries"]} == {"owner", "result-b"}
    for body in result["entries"] + result["dependency_entries"]:
        assert body["message"] == oracle[body["id"]]
    assert "prune" not in [e["message"].get("type") for e in result["dependency_entries"]]
    for replacement in (
        row("second-owner", "assistant", tool_calls=[{"id": "a", "name": "read", "arguments": {}}]),
        row("owner", "assistant", tool_calls=[{"id": "a", "name": "ask", "arguments": {}}]),
        row("owner", "assistant", tool_calls=[{"id": "a", "name": "patience", "arguments": {}}]),
    ):
        selected = (
            entries + [replacement]
            if replacement.id == "second-owner"
            else [replacement] + entries[1:]
        )
        directory, changed = write(tmp_path, selected)
        rejected = get(directory, changed, ["result-a", "plain"])
        assert "result-a" in rejected["unavailable"] and "plain" in rejected["present"]


def test_compaction_candidates_crosscut_and_unknown_cut(tmp_path):
    owner = row(
        "call-owner", "assistant", tool_calls=[{"id": "a", "name": "read", "arguments": {}}]
    )
    marker = tr.TranscriptEntry(
        "compact",
        2,
        tr.ENTRY_COMPACTION,
        {
            "first_kept_entry_id": "kept",
            "preserved_user_turns": [{"id": "old", "text": "changed preserved body"}],
            "summary": "fixture",
        },
    )
    entries = [
        row("old"),
        owner,
        row("kept"),
        row("result", "tool", tool_call_id="a", tool_name="read"),
        marker,
    ]
    directory, index = write(tmp_path, entries)
    result = get(directory, index, ["old", "kept", "result"])
    assert set(result["unavailable"]) == {"old", "result"}
    assert result["present"] == ["kept"]
    marker.payload["first_kept_entry_id"] = "unknown-cut"
    directory, index = write(tmp_path, entries)
    result = get(directory, index, ["kept", "absent"])
    assert result["unavailable"] == ["kept"] and result["missing"] == ["absent"]


def test_custom_inline_artifact_and_external_media_supplement_declines(tmp_path):
    entries = [
        tr.TranscriptEntry(
            "inject",
            1,
            tr.ENTRY_MESSAGE,
            {"kind": "custom", "custom_type": "session_state", "details": {"text": "fixture"}},
        ),
        row("inline", content=[{"kind": "image", "data": "aGVsbG8=", "mime_type": "image/png"}]),
        row(
            "external",
            content=[{"type": "image", "attachment": "a" * 32, "mime_type": "image/png"}],
        ),
        tr.TranscriptEntry(
            "supplement",
            1,
            tr.ENTRY_CUSTOM,
            {"custom_type": "supplement_v1", "details": {"anchor": "inline", "state": "done"}},
        ),
        tr.TranscriptEntry(
            "hidden",
            1,
            tr.ENTRY_MESSAGE,
            {"kind": "custom", "custom_type": "wake_prompt", "details": {"hidden": True}},
        ),
    ]
    directory, index = write(tmp_path, entries)
    from local_operator.session.history_window import display_window

    owner = tr.Transcript(directory)
    window = display_window(
        owner,
        conversation_id=SID,
        owner_epoch="fixture-epoch",
        through_id=entries[-1].id,
        max_messages=512,
    )
    oracle = {message.id: message.model_dump(mode="json") for message in window.messages}
    result = get(directory, index, [e.id for e in entries])
    assert set(result["present"]) == {"inject", "inline"}
    assert next(body for body in result["entries"] if body["id"] == "inject")["kind"] == "custom"
    assert set(result["unavailable"]) == {"external", "supplement", "hidden"}
    for body in result["entries"]:
        assert body["message"] == oracle[body["id"]]
    assert result["entries"][1]["message"]["content"][0]["data"] == "aGVsbG8="


def test_huge_unit_fairness_aggregate_remaining_and_smaller_retry(tmp_path):
    entries = [row("huge", text="x" * (512 * 1024)), row("small")]
    directory, index = write(tmp_path, entries)
    result = get(directory, index, ["huge", "small"])
    assert result["unavailable"] == ["huge"] and result["present"] == ["small"]
    entries = [row(f"target-{i}", "assistant") for i in range(512)]
    entries += [prune(f"prune-{i}", f"target-{i}", "notice") for i in range(512)]
    directory, index = write(tmp_path, entries)
    result = get(directory, index, [e.id for e in entries[:512]])
    assert len(result["remaining"]) == 512 and result["retry_smaller_batch"]
    assert 0 < len(result["retry_ids"]) <= 256
    retried = get(directory, index, result["retry_ids"])
    assert set(retried["present"]) == set(result["retry_ids"])
    partition(result, [e.id for e in entries[:512]])


def test_cache_suffix_roundtrip_inverse_facts_and_duplicate_cross_boundary(tmp_path):
    entries = [row(f"user-{i}") for i in range(5)]
    directory, initial = write(tmp_path, entries)
    addition = [prune("late", "user-0", "notice"), row("user-0", text="duplicate")]
    with (directory / "transcript.jsonl").open("a") as handle:
        handle.write("".join(e.to_json() + "\n" for e in addition))
    appended = ti.refresh_index(tmp_path, SID)
    reloaded = ti.read_index(tmp_path, SID)
    fresh_root = tmp_path / "independent-full"
    _directory, rebuilt = write(fresh_root, entries + addition)
    assert (
        appended.source_projection_facts
        == rebuilt.source_projection_facts
        == reloaded.source_projection_facts
    )
    assert appended.source_views == rebuilt.source_views == reloaded.source_views
    assert appended.journal_generation == initial.journal_generation
    assert get(directory, appended, ["user-0", "user-3"])["unavailable"] == ["user-0"]
    raw = json.loads(ti.index_path(tmp_path, SID).read_text())
    assert "notice" not in json.dumps(raw["source_projection_facts"])
    raw["source_projection_facts"][0]["source_class"] = "made-up"
    assert ti._index_from_raw(raw) is None


def test_opaque_causal_scope_plain_progress_and_rewrite(tmp_path):
    entries = [
        row("plain"),
        row("owner", "assistant", tool_calls=[{"id": "a", "name": "read", "arguments": {}}]),
        row("opaque", "tool", text="x" * (3 * 1024 * 1024), tool_call_id="a", tool_name="read"),
    ]
    directory, index = write(tmp_path, entries)
    assert index.source_views.opaque_causal_seqs
    result = get(directory, index, ["plain", "owner", "opaque"])
    assert result["present"] == ["plain"] and set(result["unavailable"]) == {"owner", "opaque"}
    replacement = directory / "replacement"
    replacement.write_text(row("plain", text="rewritten").to_json() + "\n")
    replacement.replace(directory / "transcript.jsonl")
    assert get(directory, index, ["plain"]) is None
    rebuilt = ti.refresh_index(tmp_path, SID)
    assert rebuilt.journal_generation != index.journal_generation
    assert get(directory, rebuilt, ["plain"])["present"] == ["plain"]


@pytest.mark.parametrize("ids", [["x"] * 513, ["x" * 129], [""], "bad"])
def test_subset_lookup_bounds(tmp_path, ids):
    directory, index = write(tmp_path, [row("plain")])
    with pytest.raises(ValueError):
        get(directory, index, ids)


def test_intrinsic_large_call_unit_and_aggregate_raw_encoded_budgets(tmp_path):
    owner = row("owner", "assistant", tool_calls=[{"id": "a", "name": "read", "arguments": {}}])
    results = [row(f"r-{i}", "tool", tool_call_id="a", tool_name="read") for i in range(513)]
    directory, index = write(tmp_path, [owner, *results, row("plain")])
    result = get(directory, index, ["r-0", "plain"])
    assert result["unavailable"] == ["r-0"] and result["present"] == ["plain"]
    directory, index = write(
        tmp_path, [row("one", text="a" * 300000), row("two", text="b" * 300000)]
    )
    result = get(directory, index, ["one", "two"])
    assert len(result["present"]) == len(result["remaining"]) == 1
    assert get(directory, index, result["retry_ids"])["present"] == result["retry_ids"]
    directory, index = write(tmp_path, [row("one", text="a" * 500), row("two", text="b" * 500)])
    from local_operator.network.relay import session_history_reply_bytes

    single = get(directory, index, ["one"])
    budget = session_history_reply_bytes(0, single) + 100
    result = get(directory, index, ["one", "two"], encoded_budget=budget)
    assert result is not None and len(result["present"]) == len(result["remaining"]) == 1
    assert session_history_reply_bytes(0, result) <= budget
    tiny = get(directory, index, ["one"], encoded_budget=400)
    assert tiny is None or tiny["unavailable"] == ["one"]


def test_cross_cut_group_ask_divert_patience_and_plain_progress(tmp_path):
    owner = row("owner", "assistant", tool_calls=[{"id": "a", "name": "read", "arguments": {}}])
    marker = tr.TranscriptEntry(
        "cut", 1, tr.ENTRY_COMPACTION, {"first_kept_entry_id": "result", "summary": "fixture"}
    )
    result_row = row("result", "tool", tool_call_id="a", tool_name="read")
    directory, index = write(tmp_path, [row("before"), owner, marker, result_row, row("plain")])
    result = get(directory, index, ["result", "before", "plain"])
    assert result["unavailable"] == ["result"] and set(result["present"]) == {"before", "plain"}
    for name, details in [
        ("ask", {}),
        ("read", {"ask_gate": {"hidden": True, "verdict": "diverted"}}),
        ("patience", {}),
    ]:
        owner = row("owner", "assistant", tool_calls=[{"id": "a", "name": name, "arguments": {}}])
        target = row(
            "result",
            "tool",
            tool_call_id="a",
            tool_name=name,
            provider_payload={"details": details},
        )
        directory, index = write(tmp_path, [owner, target, row("plain")])
        result = get(directory, index, ["result", "plain"])
        assert result["unavailable"] == ["result"] and result["present"] == ["plain"]


def test_unknown_preserved_shape_and_global_unknown_scope(tmp_path):
    for payload in [
        {"first_kept_entry_id": "plain", "preserved_user_turns": [{"text": "no identity"}]},
        {"first_kept_entry_id": "absent"},
    ]:
        directory, index = write(
            tmp_path, [row("plain"), tr.TranscriptEntry("cut", 1, tr.ENTRY_COMPACTION, payload)]
        )
        result = get(directory, index, ["plain", "absent"])
        assert result["unavailable"] == ["plain"] and result["missing"] == ["absent"]
    directory, index = write(
        tmp_path, [row("plain"), tr.TranscriptEntry("unknown", 1, "unknown-type", {})]
    )
    assert get(directory, index, ["plain"])["unavailable"] == ["plain"]


def test_real_artifact_digest_survives_without_attachment_resolution(tmp_path, monkeypatch):
    artifact = row(
        "artifact", content=[{"kind": "image", "attachment": "a" * 32, "mime_type": "image/png"}]
    )
    directory, index = write(tmp_path, [artifact])
    owner = tr.Transcript(directory)
    expected = owner.build_llm_history()[0].model_dump(mode="json")

    def no_resolution(*args, **kwargs):
        pytest.fail("artifact unexpectedly attempted media expansion")

    monkeypatch.setattr(tr.AttachmentStore, "get", no_resolution)
    result = get(directory, index, ["artifact"])
    assert result["entries"][0]["message"] == expected
    assert result["entries"][0]["message"]["content"][0]["attachment"] == "a" * 32


def test_append_prune_into_prefix_keeps_domain_but_changes_body_proof(tmp_path):
    entries = [row("target", "assistant", "old body"), *[row(f"u-{i}") for i in range(8)]]
    directory, initial = write(tmp_path, entries)
    with (directory / "transcript.jsonl").open("a") as handle:
        handle.write(prune("late", "target", "new notice").to_json() + "\n")
    assert get(directory, initial, ["target"]) is None
    appended = ti.refresh_index(tmp_path, SID)
    assert appended.journal_generation == initial.journal_generation
    result = get(directory, appended, ["target"])
    expected = next(m for m in tr.Transcript(directory).build_llm_history() if m.id == "target")
    assert result["entries"][0]["message"] == expected.model_dump(mode="json")
    assert result["source_dependency_ids"] == ["late"]
    assert set(result["read_ids"]) == {"target", "late"}


def test_replace_after_projection_refuses_publication(tmp_path, monkeypatch):
    directory, index = write(tmp_path, [row("plain")])
    original = tr._replay_row

    def replaced(*args, **kwargs):
        message = original(*args, **kwargs)
        replacement = directory / "replacement"
        replacement.write_text(row("plain", text="changed").to_json() + "\n")
        replacement.replace(directory / "transcript.jsonl")
        return message

    monkeypatch.setattr(tr, "_replay_row", replaced)
    assert get(directory, index, ["plain"]) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_class", ["plain_message"]),
        ("flags", ["unknown-flag"]),
        ("call_ids", ["x" * 129]),
        ("seq", True),
        ("payload", {"text": "forbidden body"}),
    ],
)
def test_malformed_fact_cache_rejected_without_silent_repair(tmp_path, field, value):
    _directory, _index = write(tmp_path, [row("plain")])
    raw = json.loads(ti.index_path(tmp_path, SID).read_text())
    raw["source_projection_facts"][0][field] = value
    assert ti._index_from_raw(raw) is None


def test_fact_extraction_reuses_exactly_one_json_parse_per_source_row(tmp_path, monkeypatch):
    entries = [
        row("plain"),
        row("assistant", "assistant", tool_calls=[{"id": "a", "name": "read", "arguments": {}}]),
        row("result", "tool", tool_call_id="a", tool_name="read"),
        prune("prune", "result", "fixture"),
    ]
    original = ti.json.loads
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ti.json, "loads", counted)
    _directory, index = write(tmp_path, entries)
    assert len(calls) == len(entries)
    assert index.source_views.owners == {"a": ["assistant"]}
    assert index.source_views.results == {"a": ["result"]}


def test_complete_query_uri_fits_actual_serializer_not_literal_batch_count():
    import uuid
    from urllib.parse import urlencode

    path = "/v1/desktop/sessions/feedface0007/history"
    fixed = [
        ("journal_order", "1"),
        ("peer_journal_source", "1"),
        ("source_read", "payload"),
        ("expected_journal_generation", "a" * 32),
    ]
    ids = [uuid.uuid4().hex for _ in range(512)]

    def target(values):
        return (path + "?" + urlencode(fixed + [("order_ids", id_) for id_ in values])).encode()

    assert len(target(ids)) > 6144
    fitting = []
    for id_ in ids:
        if len(target(fitting + [id_])) > 6144:
            break
        fitting.append(id_)
    assert 0 < len(fitting) < 512 and len(target(fitting)) <= 6144
    assert len(target(fitting + [ids[len(fitting)]])) > 6144


def test_prune_original_request_remains_unavailable_but_named_dependency(tmp_path):
    directory, index = write(tmp_path, [row("target"), prune("prune", "target", "notice")])
    result = get(directory, index, ["target", "prune"])
    assert result["present"] == ["target"] and result["unavailable"] == ["prune"]
    assert result["source_dependency_ids"] == ["prune"]
    assert result["dependency_entries"] == []


def test_unproven_identity_scope_and_partial_index_never_fabricate_missing(tmp_path):
    from dataclasses import replace

    directory, index = write(tmp_path, [row("plain")])
    partial = replace(index, source_projection_facts=[])
    assert get(directory, partial, ["plain", "absent"]) is None
    with (directory / "transcript.jsonl").open("a") as handle:
        handle.write("malformed unknown identity\n")
    malformed = ti.refresh_index(tmp_path, SID)
    assert get(directory, malformed, ["plain", "absent"]) is None
