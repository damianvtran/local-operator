"""Selected-root discovery and cold attestation, separate from pinned #945 data."""
from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path

import pytest

from local_operator.harness.message_types import PEER_MESSAGE_MESSAGE_TYPE
from local_operator.harness.peer_cycle_display import (
    CUSTOM_TYPE, DisplayCycleAccumulator, discover_records, persist_record,
    validate_record, merge_page_records,
)
from local_operator.harness.types import (
    CustomMessage, Message, MessageEndEvent, PeerMessageDeliveredEvent, ToolCall,
    ToolExecutionStartEvent, ToolExecutionEndEvent, ToolResult, TextContent, QUIET_TURN_KEY,
)
from local_operator.session.transcript import Transcript


def make_cycle(index=0, *, wrapped_send=False, send_count=1, fold=None, root_output_lines=1, root_ids=None):
    fold = fold or DisplayCycleAccumulator("sess", "producer-epoch", lambda text: text.replace("SYNTHETIC_SECRET", "[redacted]"))
    key = lambda suffix: uuid.uuid5(uuid.NAMESPACE_URL, f"display-cycle/{index}/{suffix}").hex
    peer = CustomMessage(id=key("peer"), custom_type=PEER_MESSAGE_MESSAGE_TYPE,
                         attribution="user", details={"body": "original incoming", "sender": {}})
    fold.observe(PeerMessageDeliveredEvent(message_id=peer.id, body="original incoming", sender={}))
    calls = [ToolCall(id=root_ids[i] if root_ids is not None else key(f"send{i}"), name="eval" if wrapped_send else "send",
                     arguments={"code": "opaque worker program"} if wrapped_send else {"message": f"reply {i}"})
             for i in range(send_count)]
    calls.append(ToolCall(id=root_ids[-1] if root_ids is not None else key("quiet"), name="eval", arguments={"code": "opaque quiet program"}))
    parent = Message.assistant(id=key("parent"), tool_calls=calls)
    fold.observe(MessageEndEvent(message=parent))
    history = [peer, parent]
    for i, call in enumerate(calls):
        fold.observe(ToolExecutionStartEvent(tool_call_id=call.id, tool_name=call.name, args=call.arguments))
        if call.name == "eval":
            child_id = call.id + ":" + key(f"child{i}")
            child_name = "no_reply" if call is calls[-1] else "send"
            args = {} if child_name == "no_reply" else {"message": "SYNTHETIC_SECRET"}
            fold.observe(ToolExecutionStartEvent(tool_call_id=child_id, tool_name=child_name, args=args))
            child_result = ToolResult(tool_call_id=child_id, tool_name=child_name,
                content=[] if child_name == "no_reply" else [TextContent(text="SYNTHETIC_SECRET delivered")],
                details={QUIET_TURN_KEY: True} if child_name == "no_reply" else {"delivery": {"state": "delivered"}})
            fold.observe(ToolExecutionEndEvent(tool_call_id=child_id, tool_name=child_name, result=child_result))
        root_text = ("\n".join(f"Original root output line {line}" for line in range(root_output_lines))
                     if root_output_lines > 1 and call is calls[-1] else "root result unchanged")
        result = ToolResult(tool_call_id=call.id, tool_name=call.name,
            content=[TextContent(text=root_text)],
            details={"kernel_generation": "opaque"} if call.name == "eval" else {"delivery": {"state": "delivered"}})
        fold.observe(ToolExecutionEndEvent(tool_call_id=call.id, tool_name=call.name, result=result))
        history.append(Message.tool_result(result))
    assert fold.records, "actual typed fixture must produce proof before cold assertions"
    return fold, history, fold.records[-1]


def test_raw_parent_outcomes_cut_and_foreign_records_refuse_without_overlay():
    _, history, record = make_cycle()
    assert validate_record(record, history, "sess") == record
    assert validate_record(record, history[:-1], "sess") is None
    assert validate_record(record, history, "foreign") is None
    for field, value in [("version", 2), ("peer_id", "missing"), ("id", "collision")]:
        changed = dict(record, **{field: value})
        assert validate_record(changed, history, "sess") is None
    changed_history = copy.deepcopy(history)
    changed_history[-1].content = [TextContent(text="changed/pruned result")]
    assert validate_record(record, changed_history, "sess") is None
    for boundary in [Message.user("steer"), Message.assistant("real final"), Message.assistant()]:
        assert validate_record(record, [*history[:-1], boundary, history[-1]], "sess") is None
    forged = copy.deepcopy(history)
    forged[0].details["peer_cycle"] = record
    assert validate_record(None, forged, "sess") is None


def test_wrapped_zero_send_and_sixty_five_successful_sends_are_truthful():
    _, history, record = make_cycle(send_count=0)
    assert record["logical_actions"] == 0
    assert len(record["member_ids"]) == 2
    assert validate_record(record, history, "sess") is not None
    _, history, record = make_cycle(send_count=65)
    assert record["logical_actions"] == 65
    assert len(record["member_ids"]) == 67  # actual peer +65 sends +quiet eval carrier
    assert validate_record(record, history, "sess") is not None


def test_full_eval_keeps_only_missing_sanitized_send_payload_once():
    _, history, record = make_cycle(wrapped_send=True)
    assert len(record["member_ids"]) == 4 and record["logical_actions"] == 1
    assert len(record["child_rows"]) == 1
    assert "SYNTHETIC_SECRET" not in json.dumps(record)
    assert "[redacted]" in json.dumps(record["child_rows"])
    assert "original incoming" not in json.dumps(record)
    assert "opaque worker program" not in json.dumps(record)
    assert validate_record(record, history, "sess") is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped_send", [False, True])
async def test_sixty_five_cycles_cold_compact_first_and_last_do_not_depend_on_hot16(tmp_path, wrapped_send):
    transcript = Transcript(tmp_path / "sessions" / "sess")
    fold = DisplayCycleAccumulator("sess", "producer-epoch", lambda text: text.replace("SYNTHETIC_SECRET", "[redacted]"))
    cycles = []
    for index in range(65):
        fold, history, record = make_cycle(index, wrapped_send=wrapped_send, fold=fold)
        # Session publishes typed root END before its normal raw result append.
        # The archive therefore precedes that closing durable result, not a
        # fabricated later cut. Cold validation must wait for the real result.
        await transcript.append_messages(history[:-1])
        model_before = json.dumps([m.model_dump(mode="json") for m in transcript.build_llm_history()], sort_keys=True)
        assert await persist_record(transcript, record)
        assert await persist_record(transcript, record)  # O(1) identity +bounded equality
        model_after = json.dumps([m.model_dump(mode="json") for m in transcript.build_llm_history()], sort_keys=True)
        assert model_after == model_before
        assert validate_record(record, history[:-1], "sess") is None
        await transcript.append_message(history[-1])
        cycles.append((history, record))
    assert len(fold.records) == 16
    for index in range(2):
        await transcript.append_custom("frontend_state_checkpoint_v1", {
            "state": {"peer_cycle_display": {"refs": [r["id"] for r in fold.records]}}, "revision": index})
    assert await transcript.compact_file(min_reclaim_bytes=0) > 0
    cold = Transcript(transcript.directory)
    for index in (0, 64):
        history, expected = cycles[index]
        # Select only this page's raw roots; no full-history traversal is needed.
        response = discover_records(cold, history, "sess")
        assert response is not None and not response["remaining"] and not response["unavailable"]
        assert response["records"] == [expected]
        assert validate_record(response["records"][0], history, "sess") is not None
    archives = [row for row in cold._entries if row.payload.get("custom_type") == CUSTOM_TYPE]
    assert len(archives) == 65
    # A small read budget tells the truth instead of returning a truncated child.
    limited = discover_records(cold, cycles[0][0], "sess", max_bytes=1)
    assert limited is not None and not limited["records"] and limited["unavailable"]


@pytest.mark.asyncio
async def test_historical_reused_call_ids_select_exact_parent_on_each_separate_page(tmp_path):
    transcript = Transcript(tmp_path / "sessions" / "sess")
    cycles = []
    for index in (0, 1):
        _, history, record = make_cycle(index, root_ids=["reused-send", "reused-quiet"])
        await transcript.append_messages(history[:-1])
        assert await persist_record(transcript, record)
        await transcript.append_message(history[-1])
        cycles.append((history, record))
    assert cycles[0][1]["id"] != cycles[1][1]["id"]
    for history, expected in cycles:
        response = discover_records(transcript, history, "sess")
        assert response is not None and response["records"] == [expected]
    # Historical index disambiguation does NOT bless duplicate physical IDs in
    # one loaded view; that existing unique-source refusal remains intact.
    assert validate_record(cycles[1][1], [*cycles[0][0], *cycles[1][0]], "sess") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["prior-unresolved", "duplicate-current-result", "crossed-next-owner", "changed-hash", "cut"])
async def test_reused_root_ambiguity_stays_visible_and_never_borrows_another_result(tmp_path, mutation):
    from local_operator.harness.peer_cycle_display import validate_indexed_record
    from local_operator.session.transcript_index import refresh_index
    transcript = Transcript(tmp_path / "sessions" / "sess")
    _, first, first_record = make_cycle(0, root_ids=["reused-send", "reused-quiet"])
    _, second, second_record = make_cycle(1, root_ids=["reused-send", "reused-quiet"])
    if mutation == "prior-unresolved":
        first = [row for row in first if getattr(row, "tool_call_id", None) != "reused-send"]
    if mutation == "crossed-next-owner":
        delayed = first.pop(2)
        await transcript.append_messages(first)
        await transcript.append_messages(second[:2])
        await transcript.append_message(delayed)
        await transcript.append_messages(second[2:])
        target = first_record
    else:
        await transcript.append_messages(first)
        if mutation == "changed-hash":
            second[2].content = [TextContent(text="not the certified outcome")]
        await transcript.append_messages(second)
        if mutation == "duplicate-current-result":
            duplicate = second[2].model_copy(update={"id": uuid.uuid4().hex})
            await transcript.append_message(duplicate)
        target = second_record
    index = refresh_index(tmp_path, "sess")
    cut = index.journal_rows[second[1].id] if mutation == "cut" else None
    validated, _used = validate_indexed_record(transcript, target, "sess", index=index, cut=cut)
    assert validated is None


def test_index_reference_budget_refuses_before_iteration_or_sort():
    from types import SimpleNamespace
    from local_operator.harness.peer_cycle_display import _indexed_root_result

    class NoTraversal(list):
        def __iter__(self):
            raise AssertionError("over-budget metadata must not be traversed")

    owners = NoTraversal(["owner"] * 4097)
    index = SimpleNamespace(source_views=SimpleNamespace(owners={"call": owners}, results={"call": ["result"]}))
    selected, _used = _indexed_root_result(index, "call", "owner", None, 4096)
    assert selected is None


_CONTEXT_CASES = json.loads((Path(__file__).parents[2] / "fixtures" / "peer_cycle_display_v1.json").read_text())["cases"]


@pytest.mark.parametrize("case", _CONTEXT_CASES, ids=lambda case: case["mutation"])
def test_owned_context_corpus_refuses_incomplete_or_attention_graphs(case):
    _, history, record = make_cycle(wrapped_send=True)
    history, record = copy.deepcopy(history), copy.deepcopy(record)
    mutation = case["mutation"]
    if mutation == "wrong-version":
        record["version"] = 2
    elif mutation == "boolean-version":
        record["version"] = True
    elif mutation == "foreign-session":
        record["session_id"] = "foreign"
    elif mutation == "missing-result":
        history.pop()
    elif mutation == "changed-outcome":
        history[-1].content = [TextContent(text="changed")]
    elif mutation == "genuine-user":
        history.insert(-1, Message.user("real steering input"))
    elif mutation == "empty-final":
        history.insert(-1, Message.assistant())
    elif mutation == "held-peer":
        history[0].details["held"] = True
    elif mutation == "attachment-peer":
        history[0].details["attachments"] = ["typed attachment"]
    elif mutation == "bad-child-namespace":
        record["child_rows"][0]["tool_call_id"] = "foreign:child"
    elif mutation == "duplicate-member":
        record["member_ids"][-1] = record["member_ids"][0]
    elif mutation == "record-byte-overflow":
        record["extra"] = "x" * (64 * 1024 + 1)
    else:
        raise AssertionError(f"unknown owned corpus mutation: {mutation}")
    assert validate_record(record, history, "sess") is None


def test_compatible_page_union_preserves_old_proof_and_refuses_collisions():
    _, _, first = make_cycle(0)
    _, _, last = make_cycle(64)
    merged = merge_page_records({"records": [first]}, {"records": [last]})
    assert [r["id"] for r in merged["records"]] == [first["id"], last["id"]]
    conflicting = copy.deepcopy(first)
    conflicting["logical_actions"] = 99
    refused = merge_page_records(merged, {"records": [conflicting]})
    assert first["id"] in refused["unavailable"]
    assert all(r["id"] != first["id"] for r in refused["records"])
