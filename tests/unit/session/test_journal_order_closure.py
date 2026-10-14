"""Sparse journal-coordinate acquisition, not UI/private-parent heuristics.

These cases use the actual scanner/index and public producer helpers. They must
run under the repository's isolated HOME/config test prefix; no live session or
provider is needed. Cross-runtime UI proof is the separate combined postfix rig.
"""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from local_operator.session import transcript_index as ti

SID = "causalmeta01"


@pytest.fixture(autouse=True)
def _fresh_index_state():
    ti._reset_for_tests()
    yield
    ti._reset_for_tests()


def message(id_: str, role: str, text: str = "body", ts: float = 1) -> dict[str, Any]:
    return {
        "id": id_,
        "ts": ts,
        "type": "message",
        "payload": {"kind": "message", "role": role, "content": [{"text": text}]},
    }


def build_index(root: Path, rows: list[dict[str, Any]]):
    path = root / "sessions" / SID / "transcript.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    # refresh_index is the worker: it returns a cache but does not publish the
    # hot-path resident. Exercise the same public lifecycle as the real bridge,
    # including cache-mtime revalidation; never mock/bypass fresh_resident.
    async def ready():
        task, _started = ti.start_refresh(root, SID)
        return await task

    index = asyncio.run(ready())
    assert index is not None
    assert ti.fresh_resident(root, SID) is index
    return index


def sparse(id_: str, seq: int, role: str = "assistant") -> dict[str, Any]:
    return {"id": id_, "type": "message", "role": role, "journal_seq": seq}


def source(index):
    return index.sig["inode"], index.sig["size"], index.sig["mtime"]


def test_first_sparse_assistant_acquires_real_user_coordinate(tmp_path):
    index = build_index(tmp_path, [message("U", "user", ts=9), message("A", "assistant", ts=-9)])
    assert ti.journal_order_opener(index, "A", 1) == ("U", 0)
    batches = list(ti.journal_order_correction_batches(index, [sparse("A", 1)]))
    assert batches == [[sparse("U", 0, "user"), sparse("A", 1)]]
    assert not any("payload" in row or "text" in row for batch in batches for row in batch)
    assert index.journal_rows == {"U": 0, "A": 1}


def test_raw_complete_line_domain_includes_hidden_and_tool_rows(tmp_path):
    rows = [
        message("U", "user"),
        {"id": "silent", "ts": 1, "type": "prune", "payload": {}},
        message("T", "tool"),
        message("A", "assistant"),
    ]
    index = build_index(tmp_path, rows)
    assert index.journal_rows["A"] == 3
    assert next(doc for doc in index.messages if doc.id == "A").seq == 3
    assert index.runs[0].first_seq == 0
    assert index.runs[0].last_seq >= 3
    assert ti.journal_order_opener(index, "A", 3) == ("U", 0)
    assert ti.journal_order_opener(index, "A", 1) is None  # not display ordinal
    assert ti.journal_order_opener(index, "T", 2) is None  # run_key/role is not A


def test_run_key_can_be_tool_without_misidentifying_the_assistant(tmp_path):
    index = build_index(
        tmp_path, [message("U", "user"), message("A", "assistant"), message("T", "tool")]
    )
    index.runs[0] = replace(
        index.runs[0], key_id="tool:any-call"
    )  # key never grants the parent proof
    assert ti.journal_order_opener(index, "T", 2) is None
    assert ti.journal_order_opener(index, "A", 1) == ("U", 0)


def test_steer_is_not_guessed_as_a_new_opener(tmp_path):
    # Tool-only first assistant leaves this logical run open while U2 steers.
    first = message("A0", "assistant", "")
    first["payload"]["tool_calls"] = [{"id": "c", "name": "bash", "arguments": {}}]
    index = build_index(
        tmp_path, [message("U1", "user"), first, message("U2", "user"), message("A2", "assistant")]
    )
    assert index.runs[0].opening_user_id == "U1"
    assert ti.journal_order_opener(index, "A2", 3) == ("U1", 0)
    assert ti.journal_order_opener(index, "A0", 1) == ("U1", 0)
    assert list(ti.journal_order_correction_batches(index, [sparse("A0", 1)]))[0][0]["id"] == "U1"
    # U2's own requested coordinate remains truthful; never renamed into U1.
    answer = {"entries": []}
    assert ti.apply_journal_order(tmp_path, SID, answer, source(index), ["U2"])
    assert answer["journal_ordinals"] == {"U2": 2}


def test_closed_run_and_delayed_old_assistant_keep_exact_source_opener(tmp_path):
    index = build_index(
        tmp_path,
        [
            message("U0", "user"),
            message("A0", "assistant"),
            message("U2", "user"),
            message("A2", "assistant"),
        ],
    )
    assert ti.journal_order_opener(index, "A0", 1) == ("U0", 0)
    assert ti.journal_order_opener(index, "A2", 3) == ("U2", 2)
    batches = list(ti.journal_order_correction_batches(index, [sparse("A2", 3), sparse("A0", 1)]))
    assert batches == [
        [sparse("U2", 2, "user"), sparse("A2", 3), sparse("U0", 0, "user"), sparse("A0", 1)]
    ]
    assert index.journal_rows["A0"] < index.journal_rows["U2"]  # never old-A tail pinning


@pytest.mark.parametrize(
    "mutation", ["missing", "wrong_role", "injected", "wrong_slot", "range", "first", "domain"]
)
def test_missing_or_contradictory_source_proof_never_invents_extra(tmp_path, mutation):
    index = build_index(tmp_path, [message("U", "user"), message("A", "assistant")])
    if mutation == "missing":
        index.journal_rows.pop("U")
    elif mutation in ("wrong_role", "injected"):
        user = index.messages[0]
        index.messages[0] = ti.MessageDoc(
            user.id,
            user.ts,
            "assistant" if mutation == "wrong_role" else "user",
            user.text,
            mutation == "injected",
            user.seq,
        )
    elif mutation == "wrong_slot":
        index.journal_rows["U"] = 7
    elif mutation == "range":
        index.runs[0] = replace(index.runs[0], last_seq=0)
    elif mutation == "first":
        index.runs[0] = replace(index.runs[0], first_seq=-1)
    else:
        index.journal_generation = ""
    assert ti.journal_order_opener(index, "A", 1) is None
    assert list(ti.journal_order_correction_batches(index, [sparse("A", 1)])) == [[sparse("A", 1)]]


def test_head_cut_or_generated_input_does_not_become_genuine_user(tmp_path):
    index = build_index(tmp_path, [message("A", "assistant")])
    assert ti.journal_order_opener(index, "A", 0) is None
    index = build_index(
        tmp_path,
        [
            {
                "id": "peer",
                "ts": 1,
                "type": "message",
                "payload": {
                    "kind": "custom",
                    "custom_type": "peer_message",
                    "details": {"text": "wake"},
                },
            },
            message("A", "assistant"),
        ],
    )
    assert ti.journal_order_opener(index, "A", 1) is None


def test_wrong_assistant_id_role_or_stale_position_no_extra(tmp_path):
    index = build_index(
        tmp_path, [message("U", "user"), message("T", "tool"), message("A", "assistant")]
    )
    assert ti.journal_order_opener(index, "T", 1) is None
    assert ti.journal_order_opener(index, "A", 1) is None
    assert ti.journal_order_opener(index, "missing", 2) is None
    assert ti.journal_order_opener(index, "A", True) is None
    assert ti.journal_order_opener(index, "A", -1) is None


def test_pair_flushes_before_127_boundary_and_repeats_in_later_chunk(tmp_path):
    index = build_index(tmp_path, [message("U", "user"), message("A", "assistant")])
    filler = [{"id": f"silent-{n}", "journal_seq": n + 10} for n in range(127)]
    batches = list(ti.journal_order_correction_batches(index, filler + [sparse("A", 1)] * 140))
    assert len(batches[0]) == 127
    assert all(len(batch) <= 128 for batch in batches)
    assert len(batches) >= 3
    for batch in batches[1:]:
        assert batch[0] == sparse("U", 0, "user")
        assert all(row["id"] == "A" for row in batch[1:])
    # No global dedup: a later event alone still has U before every newly known A.
    assert sum(row["id"] == "A" for batch in batches for row in batch) == 140


def test_128_boundary_duplicate_and_empty_are_bounded_and_idempotent(tmp_path):
    index = build_index(tmp_path, [message("U", "user"), message("A", "assistant")])
    filler = [{"id": f"silent-{n}", "journal_seq": n + 10} for n in range(128)]
    batches = list(ti.journal_order_correction_batches(index, filler + [sparse("A", 1)]))
    assert len(batches[0]) == 128
    assert batches[1] == [sparse("U", 0, "user"), sparse("A", 1)]
    assert list(ti.journal_order_correction_batches(index, [])) == []


def test_ready_page_carries_opener_coordinate_without_user_body(tmp_path):
    index = build_index(tmp_path, [message("U", "user"), message("A", "assistant")])
    answer = {"entries": [{"id": "A", "payload": {"role": "assistant"}}]}
    assert ti.apply_journal_order(tmp_path, SID, answer, source(index), [])
    assert [row["id"] for row in answer["entries"]] == ["A"]
    assert answer["journal_ordinals"] == {"U": 0}
    assert answer["entries"][0]["journal_seq"] == 1
    assert answer["journal_missing_ids"] == []
    assert answer["journal_generation"] == index.journal_generation


def test_lookup_missing_partition_stays_requested_ids_only(tmp_path):
    index = build_index(tmp_path, [message("U", "user"), message("A", "assistant")])
    answer = {"entries": []}
    assert ti.apply_journal_order(tmp_path, SID, answer, source(index), ["A", "missing"])
    assert answer["journal_ordinals"] == {"A": 1, "U": 0}
    assert answer["journal_missing_ids"] == ["missing"]


@pytest.mark.parametrize("bad_source", [None, (0, 0, 0)])
def test_wrong_or_absent_source_signature_never_promotes_authority(tmp_path, bad_source):
    build_index(tmp_path, [message("U", "user"), message("A", "assistant")])
    answer = {"entries": [{"id": "A", "payload": {"role": "assistant"}}]}
    original = copy.deepcopy(answer)
    assert not ti.apply_journal_order(tmp_path, SID, answer, bad_source, ["A"])
    assert answer == original


def test_rewrite_remints_domain_and_old_signature_cannot_claim_new_user(tmp_path):
    index = build_index(tmp_path, [message("U0", "user"), message("A", "assistant")])
    old_signature = source(index)
    old_generation = index.journal_generation
    new = build_index(
        tmp_path, [message("U-new", "user", "different bytes"), message("A", "assistant")]
    )
    assert new.journal_generation != old_generation
    answer = {"entries": [{"id": "A"}]}
    assert not ti.apply_journal_order(tmp_path, SID, answer, old_signature, [])
    assert "journal_ordinals" not in answer
    assert ti.journal_order_opener(new, "A", 1) == ("U-new", 0)


def test_ordinal_bisects_do_not_walk_lazy_full_history(tmp_path):
    index = build_index(tmp_path, [message("U", "user"), message("A", "assistant")])

    class NoWalk(list[Any]):
        def __iter__(self):
            raise AssertionError("a metadata lookup must not walk full history")

    index.messages = NoWalk(index.messages)
    index.runs = NoWalk(index.runs)
    assert ti.journal_order_opener(index, "A", 1) == ("U", 0)


def test_requested_steer_facts_precede_replies_without_becoming_opener(tmp_path):
    first = message("A0", "assistant", "")
    first["payload"]["tool_calls"] = [{"id": "c", "name": "bash", "arguments": {}}]
    index = build_index(
        tmp_path, [message("U1", "user"), first, message("U2", "user"), message("A2", "assistant")]
    )
    batches = list(
        ti.journal_order_correction_batches(index, [sparse("A2", 3), sparse("U2", 2, "user")])
    )
    assert batches == [[sparse("U2", 2, "user"), sparse("U1", 0, "user"), sparse("A2", 3)]]
    assert ti.journal_order_opener(index, "A2", 3) == ("U1", 0)
    assert list(ti.journal_order_correction_batches(index, [sparse("A0", 1)])) == [
        [sparse("U1", 0, "user"), sparse("A0", 1)]
    ]  # late old A cannot borrow newer U2, even if U2 is now in the same run


def test_replay_duplicate_facts_preserve_native_ids_and_true_coordinates(tmp_path):
    index = build_index(
        tmp_path, [message("raw:user", "user", ts=5), message("raw:assistant", "assistant", ts=5)]
    )
    incoming = [
        sparse("raw:assistant", 1),
        sparse("raw:user", 0, "user"),
        sparse("raw:assistant", 1),
    ]
    original = copy.deepcopy(incoming)
    batches = list(ti.journal_order_correction_batches(index, incoming))
    assert incoming == original  # closure never modifies producer payload descriptors
    assert batches == [
        [sparse("raw:user", 0, "user"), sparse("raw:assistant", 1), sparse("raw:assistant", 1)]
    ]
    assert list(ti.journal_order_correction_batches(index, incoming)) == batches
