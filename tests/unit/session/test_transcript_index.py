"""The transcript index: derivation rules, cache invalidation, and the async view.

Every test drives SYNTHETIC journals written in the journal's own row format
(``{"id","ts","type","payload"}`` lines), so the derivation's rules are pinned
on shapes the S3 spike measured in the real store — the run/turn mismatch, steer
rows, hub runs with no user row, ``eligible: false`` settlements, torn tails —
without touching the operator's live sessions.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from local_operator.compaction.cutpoint import elision_notice_text
from local_operator.harness.types import Message, TextContent
from local_operator.session import transcript_index as ti
from local_operator.session.goal_judge import goal_continuation_prompt
from local_operator.session.transcript import Transcript, encode_message_payload

SID = "aabbccddee01"


@pytest.fixture(autouse=True)
def _clean_module_state():
    """The module keeps process-wide loop state; no test may inherit another's."""
    ti._reset_for_tests()
    yield
    ti._reset_for_tests()


def journal_path(root: Path) -> Path:
    return root / "sessions" / SID / "transcript.jsonl"


def write_rows(root: Path, rows: list[dict[str, Any]]) -> None:
    path = journal_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def user(id_: str, ts: float, text: str = "hello") -> dict[str, Any]:
    return {
        "id": id_,
        "ts": ts,
        "type": "message",
        "payload": {"kind": "message", "role": "user", "content": [{"text": text}]},
    }


def assistant(
    id_: str, ts: float, text: str = "answer", tool_calls: bool = False
) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": "message", "role": "assistant", "content": [{"text": text}]}
    if tool_calls:
        payload["tool_calls"] = [{"id": "c1", "name": "bash", "arguments": {}}]
    return {"id": id_, "ts": ts, "type": "message", "payload": payload}


def tool(id_: str, ts: float, text: str = "tool output") -> dict[str, Any]:
    return {
        "id": id_,
        "ts": ts,
        "type": "message",
        "payload": {"kind": "message", "role": "tool", "content": [{"text": text}]},
    }


def inject(
    id_: str, ts: float, custom_type: str = "hub_message", text: str = "peer says hi"
) -> dict[str, Any]:
    return {
        "id": id_,
        "ts": ts,
        "type": "message",
        "payload": {"kind": "custom", "custom_type": custom_type, "details": {"text": text}},
    }


#: The continuation text the goal judge persists every few minutes while a goal
#: runs — built through the producer so this fixture cannot drift from it.
GOAL_CONTINUATION = goal_continuation_prompt("make the fold hold")

#: The recogniser's deliberate edge (``is_goal_continuation_instruction``): a
#: message that merely OPENS with the template is the operator's own words.
QUOTE_NEGATIVE = (
    "Continue working toward this goal:\n\n"
    "I know the harness phrases it like this \u2014 this one is me telling you: keep going."
)


def stamped_user(id_: str, ts: float, text: str) -> dict[str, Any]:
    """A harness row as the renderer writes it: a ``Message`` carrying the stamp.

    Built through ``encode_message_payload``, the journal's own writer, so this
    is the row shape the scanner meets live (``provider_payload`` and all)
    rather than a hand-drawn approximation of it.
    """
    message = Message(
        role="user",
        content=[TextContent(text=text)],
        provider_payload={"harness_injected": True},
    )
    return {"id": id_, "ts": ts, "type": "message", "payload": encode_message_payload(message)}


def start(id_: str, ts: float, token: str) -> dict[str, Any]:
    return {
        "id": id_,
        "ts": ts,
        "type": "custom",
        "payload": {
            "custom_type": "attention_started",
            "details": {"conversation_id": f"session/{SID}", "token": token},
        },
    }


def marker(
    id_: str, ts: float, token: str, *, kind: str | None = "complete", eligible: bool = True
) -> dict[str, Any]:
    details: dict[str, Any] = {
        "conversation_id": f"session/{SID}",
        "token": token,
        "eligible": eligible,
    }
    if kind is not None and eligible:
        details["kind"] = kind
        details["anchor"] = "a-anchor"
    return {
        "id": id_,
        "ts": ts,
        "type": "custom",
        "payload": {"custom_type": "completion_attention", "details": details},
    }


def kinds(index: ti.TranscriptIndex) -> list[tuple[str, str]]:
    return [(c.kind, c.id) for c in index.checkpoints]


def outcomes(index: ti.TranscriptIndex) -> list[tuple[str, str | None]]:
    return [(c.id, c.outcome) for c in index.checkpoints]


def refreshed(root: Path) -> ti.TranscriptIndex:
    """``refresh_index`` for tests that hold a journal: it returns ``None``
    only for a session with no transcript, and every caller here has one."""
    index = ti.refresh_index(root, SID)
    assert index is not None
    return index


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def test_scan_derives_user_and_completion_checkpoints_with_outcomes(tmp_path):
    write_rows(
        tmp_path,
        [
            start("s1", 1.0, "t1"),
            user("u1", 1.1, "do the thing"),
            assistant("a1", 1.2, "working"),
            tool("x1", 1.3),
            assistant("a2", 1.4, "done"),
            marker("m1", 1.5, "t1"),
        ],
    )
    index = refreshed(tmp_path)
    assert index is not None
    assert kinds(index) == [("user", "u1"), ("completion", "a2")]
    assert outcomes(index) == [("u1", None), ("a2", "complete")]
    users = [c for c in index.checkpoints if c.kind == "user"]
    comps = [c for c in index.checkpoints if c.kind == "completion"]
    assert users[0].turn == 1 and comps[0].turn == 1
    assert users[0].seq == 1 and comps[0].seq == 4  # journal ordinals, 0-based
    assert comps[0].ts == 1.4
    # tool rows are never docs; user+assistant are.
    assert [(m.id, m.role) for m in index.messages] == [
        ("u1", "user"),
        ("a1", "assistant"),
        ("a2", "assistant"),
    ]


def test_injected_rows_are_docs_not_checkpoints(tmp_path):
    write_rows(
        tmp_path,
        [
            user("u1", 1.0),
            assistant("a1", 1.1),
            inject("i1", 1.2),
            assistant("a2", 1.3),
        ],
    )
    index = refreshed(tmp_path)
    assert kinds(index) == [("user", "u1"), ("completion", "a2")]
    docs = {m.id: m for m in index.messages}
    assert docs["i1"].injected is True and docs["i1"].role == "user"
    assert docs["i1"].text == "peer says hi"
    assert docs["u1"].injected is False and docs["a1"].injected is False


# ---------------------------------------------------------------------------
# Harness provenance (F1 of local-operator-ui#670)
# ---------------------------------------------------------------------------


def test_harness_rows_are_skipped_whole_not_filed_as_the_operator(tmp_path):
    """A continuation row is not the operator's message on ANY index product.

    Both legs, one decision \u2014 the folds' own (``harness/rows.py``): the
    structural stamp on rows this build minted, and the recognisers (chrome
    prompt families, notice heads) on rows written before the stamp existed.
    The row must vanish from BOTH products: no user checkpoint (the rail's
    "Your message" hover filed it as the operator's) and no find doc (a hit
    would reveal-jump to a row no human surface paints). And it must not open
    a phantom turn: u1's span carries on through both continuations to its
    real closing answer, a2.
    """
    write_rows(
        tmp_path,
        [
            user("u1", 1.0, "do the thing"),
            stamped_user("c1", 1.1, GOAL_CONTINUATION),
            assistant("a1", 1.2, "worked on it"),
            user("c2", 1.3, GOAL_CONTINUATION),  # a pre-stamp build's row: text-only evidence
            assistant("a2", 1.4, "carried on"),
            user("n1", 1.45, "[session warning] legacy notice row"),
            user("q1", 1.5, QUOTE_NEGATIVE),
            assistant("a3", 1.6, "understood"),
        ],
    )
    index = refreshed(tmp_path)
    # c1/c2/n1 are gone from checkpoints AND docs; the ordinary u1, the
    # recogniser-negative q1, and every assistant row stay.
    assert kinds(index) == [
        ("user", "u1"),
        ("completion", "a2"),
        ("user", "q1"),
        ("completion", "a3"),
    ]
    assert [(m.id, m.role, m.injected) for m in index.messages] == [
        ("u1", "user", False),
        ("a1", "assistant", False),
        ("a2", "assistant", False),
        ("q1", "user", False),
        ("a3", "assistant", False),
    ]


def test_a_user_message_merely_opening_with_the_template_stays_the_operators(tmp_path):
    """The negative control: the recogniser must not confiscate the operator's words."""
    write_rows(tmp_path, [user("q1", 1.0, QUOTE_NEGATIVE), assistant("a1", 1.1, "ok")])
    index = refreshed(tmp_path)
    assert kinds(index) == [("user", "q1"), ("completion", "a1")]
    assert [(m.id, m.injected) for m in index.messages] == [("q1", False), ("a1", False)]


def test_an_appended_harness_row_mints_no_checkpoint_on_the_incremental_path(tmp_path, monkeypatch):
    """The skip holds across the incremental seam, not just a full rescan."""
    write_rows(tmp_path, [user("u1", 1.0, "one"), assistant("a1", 1.1, "answer")])
    calls = _scanners(monkeypatch)
    base = refreshed(tmp_path)
    assert calls == {"full": 1, "incremental": 0}
    assert kinds(base) == [("user", "u1"), ("completion", "a1")]

    write_rows(tmp_path, [stamped_user("c1", 2.0, GOAL_CONTINUATION), assistant("a2", 2.1, "more")])
    grown = refreshed(tmp_path)
    assert calls == {"full": 1, "incremental": 1}
    assert kinds(grown) == [("user", "u1"), ("completion", "a2")]
    assert "c1" not in {m.id for m in grown.messages}


def test_a_legacy_elision_notice_row_is_not_the_operators_words(tmp_path):
    """A pre-#857 build journalled the elision notice as a turn; the index drops it.

    The notice is render-time only in this revision, but
    ``PRESERVED_TURN_ELISION_ID_PREFIX`` is retained precisely so transcripts
    written by the previous revision still parse — the read path recognises
    those rows by ID and drops them rather than replaying them as user turns.
    The index reaches that leg only if the entry id rides in beside the
    payload (QA round 1, Q-1); the notice's text matches no notice head, so
    the id is the only catch.
    """
    notice = elision_notice_text(2, 3)
    assert notice is not None
    write_rows(
        tmp_path,
        [
            user("u1", 1.0, "do the thing"),
            {
                "id": "compaction-elision-1",
                "ts": 1.1,
                "type": "message",
                "payload": {"kind": "message", "role": "user", "content": [{"text": notice}]},
            },
            assistant("a1", 1.2, "ok"),
        ],
    )
    index = refreshed(tmp_path)
    assert kinds(index) == [("user", "u1"), ("completion", "a1")]
    assert [m.id for m in index.messages] == ["u1", "a1"]


def test_the_stamp_leg_stands_on_its_own_with_neutral_words(tmp_path):
    """A stamped row whose words no recogniser claims must still be skipped.

    Every other stamped fixture in these cells carries the continuation's own
    text, which the chrome recogniser already claims — so this cell is the one
    that fails if the stamp leg is lost (review round 1, F1). The unstamped
    twin keeps it discriminating: the same neutral words on a row with no
    stamp stay the operator's, proving the text legs alone cannot claim them.
    """
    write_rows(
        tmp_path,
        [
            user("u1", 1.0, "do the thing"),
            stamped_user("c3", 1.1, "neutral harness words"),
            assistant("a1", 1.2, "ok"),
            user("u2", 1.3, "neutral harness words"),
            assistant("a2", 1.4, "done"),
        ],
    )
    index = refreshed(tmp_path)
    assert kinds(index) == [
        ("user", "u1"),
        ("completion", "a1"),
        ("user", "u2"),
        ("completion", "a2"),
    ]
    assert [(m.id, m.injected) for m in index.messages] == [
        ("u1", False),
        ("a1", False),
        ("u2", False),
        ("a2", False),
    ]


def test_completion_needs_span_content(tmp_path):
    write_rows(tmp_path, [user("u1", 1.0), user("u2", 1.1), assistant("a1", 1.2)])
    index = refreshed(tmp_path)
    # u1's span holds no non-user entry: no completion tick for turn 1.
    assert kinds(index) == [("user", "u1"), ("user", "u2"), ("completion", "a1")]
    assert index.checkpoints[-1].turn == 2


def test_marker_resolves_the_last_user_row_of_the_run(tmp_path):
    """A steer stays inside the run; the run's outcome lands on the LAST of its
    user rows (S3 note 1)."""
    write_rows(
        tmp_path,
        [
            start("s1", 1.0, "t1"),
            user("u1", 1.1, "start"),
            assistant("a1", 1.2),
            user("u2", 1.3, "steer"),
            assistant("a2", 1.4),
            marker("m1", 1.5, "t1"),
        ],
    )
    index = refreshed(tmp_path)
    assert outcomes(index) == [("u1", None), ("a1", None), ("u2", None), ("a2", "complete")]
    # Two completion ticks: one per turn — the mid-run steer closes turn 1.
    turns = [(c.kind, c.turn) for c in index.checkpoints]
    assert turns == [("user", 1), ("completion", 1), ("user", 2), ("completion", 2)]


def test_orphan_marker_is_ignored(tmp_path):
    write_rows(tmp_path, [user("u1", 1.0), assistant("a1", 1.1), marker("m1", 1.2, "ghost")])
    index = refreshed(tmp_path)
    assert kinds(index) == [("user", "u1"), ("completion", "a1")]
    assert index.checkpoints[-1].outcome == "open"  # nothing settled the tail


def test_tail_open_until_a_marker_lands(tmp_path):
    write_rows(tmp_path, [start("s1", 1.0, "t1"), user("u1", 1.1), assistant("a1", 1.2)])
    index = refreshed(tmp_path)
    assert index.checkpoints[-1].outcome == "open"
    write_rows(tmp_path, [marker("m1", 1.3, "t1")])
    index = refreshed(tmp_path)
    assert index.checkpoints[-1].outcome == "complete"


def test_tail_settles_on_a_later_run_with_no_user_row(tmp_path):
    """A hub/wake run after the tail user row settles the conversation without
    carrying a kind (S3 note 2); the tail is NOT left open forever."""
    write_rows(tmp_path, [start("s1", 1.0, "t1"), user("u1", 1.1), assistant("a1", 1.2)])
    index = refreshed(tmp_path)
    assert index.checkpoints[-1].outcome == "open"
    write_rows(
        tmp_path,
        [
            start("s2", 2.0, "h1"),
            inject("i1", 2.1, "wake_prompt"),
            assistant("a2", 2.2),
            marker("m2", 2.3, "h1"),
        ],
    )
    index = refreshed(tmp_path)
    # Settled by the later run: no kind to attach, and the hub run adds no
    # checkpoint of its own — the tail completion tracks the span's last row.
    assert index.checkpoints[-1].outcome is None
    assert [c.id for c in index.checkpoints if c.kind == "completion"] == ["a2"]
    assert len(index.checkpoints) == 2


def test_the_tail_keeps_its_own_outcome_when_later_runs_follow(tmp_path):
    write_rows(
        tmp_path,
        [
            start("s1", 1.0, "t1"),
            user("u1", 1.1),
            assistant("a1", 1.2),
            marker("m1", 1.3, "t1", kind="error"),
        ],
    )
    index = refreshed(tmp_path)
    assert index.checkpoints[-1].outcome == "error"
    write_rows(
        tmp_path,
        [
            start("s2", 2.0, "h1"),
            inject("i1", 2.1, "wake_prompt"),
            assistant("a2", 2.2),
            marker("m2", 2.3, "h1"),
        ],
    )
    index = refreshed(tmp_path)
    # The turn's OWN marker's kind survives; the completion still tracks the
    # span's last message row (the conversation's current end).
    assert index.checkpoints[-1].outcome == "error"
    assert [c.id for c in index.checkpoints if c.kind == "completion"] == ["a2"]


def test_a_closed_marker_never_claims_or_clears_a_turn_outcome(tmp_path):
    """THE v2 DIRECTIVE AT THE RAIL: a closure is inert; the completion stands.

    Session 23fc556c3799: the turn completed, then the zero-work disposal
    published on top. The attention store's newest row is the closure (rendered
    neutrally by every row surface), but the RAIL must keep the completed
    turn's own outcome: ``closed`` is not a rail outcome, and letting it attach
    would either claim the tick or erase it — the same masking, one derivation
    over. The second half is the control: a closure with no completion before
    it asserts nothing rather than inventing an outcome.
    """
    write_rows(
        tmp_path,
        [
            start("s1", 1.0, "t1"),
            user("u1", 1.1, "do the thing"),
            assistant("a1", 1.2, "done"),
            marker("m1", 1.3, "t1"),
            start("s2", 2.0, "t2"),
            inject("i1", 2.1, "hub_message"),
            marker("m2", 2.2, "t2", kind="closed"),
        ],
    )
    index = refreshed(tmp_path)
    assert outcomes(index) == [("u1", None), ("a1", "complete")]

    closed_only = tmp_path / "closed-only"
    closed_only.mkdir()
    write_rows(
        closed_only,
        [
            start("s1", 1.0, "t1"),
            user("u1", 1.1, "peer's ask"),
            inject("i1", 1.2, "hub_message"),
            marker("m1", 1.3, "t1", kind="closed"),
        ],
    )
    index = ti.refresh_index(closed_only, SID)
    assert index is not None
    assert [c.outcome for c in index.checkpoints if c.kind == "completion"] == [
        None
    ], "a closure alone must not claim the tick"


def test_a_retired_marker_reads_as_the_rails_existing_cut_treatment(tmp_path):
    """Round 2, NIT-2: the retired kind must never reach the frozen wire literal.

    ``CheckpointOutcome`` has no ``retired``, so the derivation normalizes it to
    ``interrupted`` — CircleSlash in warning ink, the rail's "cut short" mark:
    ``error`` would be the failure framing the arm exists to remove and
    ``complete`` would claim the cut turn finished. The control beside this is
    the cell above (``closed`` stays fully inert); this one pins that a marker
    which really did cut a turn still lands on an EXISTING outcome value.
    """
    write_rows(
        tmp_path,
        [
            start("s1", 1.0, "t1"),
            user("u1", 1.1, "verify the migration"),
            assistant("a1", 1.2, "halfway through"),
            marker("m1", 1.3, "t1", kind="retired"),
        ],
    )
    index = refreshed(tmp_path)
    assert outcomes(index) == [("u1", None), ("a1", "interrupted")]


def test_eligible_false_settles_without_an_outcome(tmp_path):
    write_rows(
        tmp_path,
        [
            start("s1", 1.0, "t1"),
            user("u1", 1.1),
            assistant("a1", 1.2),
            marker("m1", 1.3, "t1", kind=None, eligible=False),
        ],
    )
    index = refreshed(tmp_path)
    assert index.checkpoints[-1].outcome is None


def test_malformed_rows_are_dropped_and_ordinals_still_count(tmp_path):
    write_rows(tmp_path, [user("u1", 1.0), assistant("a1", 1.1), assistant("a2", 1.2)])
    path = journal_path(tmp_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    write_rows(tmp_path, [assistant("a3", 1.3)])
    index = refreshed(tmp_path)
    assert [c.id for c in index.checkpoints if c.kind == "completion"] == ["a3"]
    assert index.checkpoints[-1].seq == 4  # the malformed line still took an ordinal


def test_text_caps(tmp_path):
    write_rows(
        tmp_path,
        [
            user("u1", 1.0, "user words " * 200),
            assistant("a1", 1.1, "assistant words " * 4000),
        ],
    )
    index = refreshed(tmp_path)
    user_cp = next(c for c in index.checkpoints if c.kind == "user")
    assert len(user_cp.text) == ti.CHECKPOINT_TEXT_CAP
    doc = next(m for m in index.messages if m.role == "assistant")
    assert len(doc.text) == ti.DOC_TEXT_CAP


# ---------------------------------------------------------------------------
# Cache lifecycle: reuse, incremental append, rewrites, versions
# ---------------------------------------------------------------------------


def test_fresh_cache_is_reused_without_any_scan(tmp_path, monkeypatch):
    write_rows(tmp_path, [user("u1", 1.0), assistant("a1", 1.1)])
    first = refreshed(tmp_path)
    assert first is not None

    def explode(*args, **kwargs):
        raise AssertionError("a fresh cache must not be rescanned")

    monkeypatch.setattr(ti, "_scan_full", explode)
    monkeypatch.setattr(ti, "_scan_incremental", explode)
    second = refreshed(tmp_path)
    assert second is not None
    assert [(c.id, c.outcome) for c in second.checkpoints] == [
        (c.id, c.outcome) for c in first.checkpoints
    ]


def _scanners(monkeypatch) -> dict[str, int]:
    calls = {"full": 0, "incremental": 0}
    real_full, real_incremental = ti._scan_full, ti._scan_incremental

    def full(*args, **kwargs):
        calls["full"] += 1
        return real_full(*args, **kwargs)

    def incremental(*args, **kwargs):
        calls["incremental"] += 1
        return real_incremental(*args, **kwargs)

    monkeypatch.setattr(ti, "_scan_full", full)
    monkeypatch.setattr(ti, "_scan_incremental", incremental)
    return calls


def test_incremental_append_matches_a_full_rescan(tmp_path, tmp_path_factory, monkeypatch):
    write_rows(
        tmp_path,
        [
            start("s1", 1.0, "t1"),
            user("u1", 1.1),
            assistant("a1", 1.2, "one"),
            marker("m1", 1.3, "t1"),
        ],
    )
    calls = _scanners(monkeypatch)
    base = refreshed(tmp_path)
    assert calls == {"full": 1, "incremental": 0}

    write_rows(
        tmp_path,
        [
            start("s2", 2.0, "t2"),
            user("u2", 2.1, "two"),
            assistant("a2", 2.2, "answer two"),
            marker("m2", 2.3, "t2", kind="error"),
        ],
    )
    grown = refreshed(tmp_path)
    assert calls == {"full": 1, "incremental": 1}

    # The reference: the same bytes scanned fresh in another root.
    other = tmp_path_factory.mktemp("reference")
    write_rows(
        other, [json.loads(line) for line in journal_path(tmp_path).read_text().splitlines()]
    )
    reference = refreshed(other)

    assert [(c.id, c.kind, c.turn, c.seq, c.text, c.outcome) for c in grown.checkpoints] == [
        (c.id, c.kind, c.turn, c.seq, c.text, c.outcome) for c in reference.checkpoints
    ]
    assert [(m.id, m.role, m.text, m.injected, m.seq) for m in grown.messages] == [
        (m.id, m.role, m.text, m.injected, m.seq) for m in reference.messages
    ]
    assert base.scan.offset < grown.scan.offset == journal_path(tmp_path).stat().st_size


def test_incremental_replaces_the_open_tail_endpoint(tmp_path, monkeypatch):
    write_rows(tmp_path, [start("s1", 1.0, "t1"), user("u1", 1.1), assistant("a1", 1.2, "first")])
    calls = _scanners(monkeypatch)
    refreshed(tmp_path)
    write_rows(tmp_path, [assistant("a2", 1.3, "final answer")])
    index = refreshed(tmp_path)
    assert calls["incremental"] == 1
    completion = next(c for c in index.checkpoints if c.kind == "completion")
    assert completion.id == "a2" and completion.text == "final answer"


def test_a_rewrite_that_shrinks_rescans(tmp_path, monkeypatch):
    write_rows(tmp_path, [user("u1", 1.0, "hello " * 500), assistant("a1", 1.1, "answer " * 500)])
    refreshed(tmp_path)
    calls = _scanners(monkeypatch)
    # Simulate a compaction rewrite: same rows, smaller bytes, new mtime.
    rows = [json.loads(line) for line in journal_path(tmp_path).read_text().splitlines()]
    rows[0]["payload"]["content"] = [{"text": "hello"}]
    journal_path(tmp_path).write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    )
    index = refreshed(tmp_path)
    assert calls == {"full": 1, "incremental": 0}
    assert index.checkpoints[0].text == "hello"


@pytest.mark.asyncio
async def test_compact_file_rewrite_rescans(tmp_path, monkeypatch):
    """The real ``compact_file`` replaces the file; the next refresh must rescan
    and still produce the same checkpoints."""
    session = tmp_path / "sessions" / SID
    transcript = Transcript(session)
    await transcript.append_message(Message.user("do a thing", id="u1"))
    big = Message(role="tool", content=[TextContent(text="large " * 4000)], tool_call_id="c1")
    big.id = "x1"
    await transcript.append_message(big)
    await transcript.append_message(
        Message(role="assistant", content=[TextContent(text="done")], id="a1")
    )
    refreshed(tmp_path)
    calls = _scanners(monkeypatch)
    await transcript.append_prune("x1", "[pruned]")
    assert await transcript.compact_file(min_reclaim_bytes=0) > 0
    index = refreshed(tmp_path)
    assert calls == {"full": 1, "incremental": 0}
    assert [c.id for c in index.checkpoints] == ["u1", "a1"]
    assert index.scan.offset == journal_path(tmp_path).stat().st_size


@pytest.mark.asyncio
async def test_a_compact_that_left_the_file_longer_still_rescans(tmp_path, monkeypatch):
    """The shape a size+tail test cannot see through: a rewrite that NET GREW.

    ``compact_file`` only rewrites when it reclaims bytes against the CURRENT
    file, but that file already carried appends since the last scan — so the
    compacted result can still be LONGER than the recorded scan while the tail
    row parses unchanged. Folding is reserved for tool rows and dropped prune
    rows, so only the inode exposes the replacement; without it the incremental
    path keeps stale ordinals from before the drop (the drop shifts every
    later row) and the rail places ticks off the end of the journal.
    """
    session = tmp_path / "sessions" / SID
    transcript = Transcript(session)
    await transcript.append_message(Message.user("do a thing", id="u1"))
    small = Message(role="tool", content=[TextContent(text="x")], tool_call_id="c1")
    small.id = "x1"
    await transcript.append_message(small)
    await transcript.append_message(
        Message(role="assistant", content=[TextContent(text="done")], id="a1")
    )
    first = refreshed(tmp_path)
    assert first.scan.offset == journal_path(tmp_path).stat().st_size
    calls = _scanners(monkeypatch)
    # A notice much longer than the one-byte body it replaces, and a prune row
    # whose drop reclaims more than the fold spends — so the rewrite fires and
    # the file still ends up longer than the recorded scan.
    await transcript.append_prune(
        "x1", "[pruned: a notice long enough that the fold spends more than it reclaimed]"
    )
    assert await transcript.compact_file(min_reclaim_bytes=0) > 0
    assert journal_path(tmp_path).stat().st_size > first.scan.offset
    index = refreshed(tmp_path)
    assert calls == {"full": 1, "incremental": 0}
    assert [c.id for c in index.checkpoints] == ["u1", "a1"]


def test_version_bump_discards_scans_and_preserves_naming(tmp_path, monkeypatch):
    write_rows(tmp_path, [user("u1", 1.0), assistant("a1", 1.1)])
    refreshed(tmp_path)
    # Put a naming item under the live turn key, then bump the cache version.
    ti.patch_naming(tmp_path, SID, {"u1": {"name": "Do the thing", "summary": "It was done."}})
    bumped = ti.TRANSCRIPT_INDEX_VERSION + 1
    monkeypatch.setattr(ti, "TRANSCRIPT_INDEX_VERSION", bumped)
    rebuilt = refreshed(tmp_path)
    assert [c.id for c in rebuilt.checkpoints] == ["u1", "a1"]
    assert rebuilt.naming["items"]["u1"]["name"] == "Do the thing"
    # The stale-version cache was RESCANNED and rewritten under the new
    # version — not served as-is.
    assert json.loads(ti.index_path(tmp_path, SID).read_text())["version"] == bumped
    monkeypatch.undo()
    # The disk still carries the bumped version while the live build is back
    # at its own; bring the cache home FIRST, because ``patch_naming`` refuses
    # a version-mismatched document (that mismatch is the refresh's to settle)
    # and the tail would otherwise be absorbed by the version gate instead of
    # exercising the preservation filter.
    refreshed(tmp_path)
    assert json.loads(ti.index_path(tmp_path, SID).read_text())["version"] == (
        ti.TRANSCRIPT_INDEX_VERSION
    )
    # And an item whose turn key no longer exists is dropped at the next
    # RESCAN (a fresh cache is served as-is; the filter belongs to preservation).
    assert ti.patch_naming(tmp_path, SID, {"gone": {"name": "Stale"}}) is True
    write_rows(tmp_path, [assistant("a2", 1.2)])  # grown journal forces a rescan
    again = refreshed(tmp_path)
    assert set(again.naming["items"]) == {"u1"}


def test_a_name_landing_during_a_scan_survives_the_refresh_write(tmp_path, monkeypatch):
    """The refresh's whole-document write must not drop a mid-scan naming write.

    The two writers run in different processes by design — this scan in the
    daemon, the naming errand on the session's owner — and the scan carries
    its naming section from the document it read BEFORE it ran. The
    interleaving below is the deterministic shape of that race: a
    ``patch_naming`` lands while the refresh sits between the read and its
    write. Pre-fix the refresh's write dropped the item (agent review round
    1, MAJOR-1, reproduced the loss exactly here); the fix merges the on-disk
    section one last time before writing.
    """
    write_rows(tmp_path, [user("u1", 1.0), assistant("a1", 2.0)])
    refreshed(tmp_path)
    # A compact-style rewrite replaces the journal file (new inode), so the next
    # refresh takes the full-scan path this test intercepts. Unlike a version
    # bump, the cache stays readable under the live build, so the racing
    # ``patch_naming`` below really writes.
    journal_path(tmp_path).unlink()
    write_rows(tmp_path, [user("u1", 1.0), assistant("a1", 2.0)])
    real_scan = ti._scan_full

    def racing_scan(path, raw):
        index = real_scan(path, raw)
        ti.patch_naming(tmp_path, SID, {"u1": {"name": "Landed mid-scan", "summary": "S."}})
        return index

    monkeypatch.setattr(ti, "_scan_full", racing_scan)
    rebuilt = refreshed(tmp_path)
    assert rebuilt.naming["items"]["u1"]["name"] == "Landed mid-scan"
    on_disk = ti.read_index(tmp_path, SID)
    assert on_disk is not None
    assert on_disk.naming["items"]["u1"]["name"] == "Landed mid-scan"


def test_corrupt_cache_is_rebuilt_and_missing_journal_is_none(tmp_path):
    assert ti.refresh_index(tmp_path, SID) is None  # no journal
    write_rows(tmp_path, [user("u1", 1.0), assistant("a1", 1.1)])
    ti.index_path(tmp_path, SID).parent.mkdir(parents=True, exist_ok=True)
    ti.index_path(tmp_path, SID).write_text("{corrupt")
    index = refreshed(tmp_path)
    assert index is not None and len(index.checkpoints) == 2
    assert ti.read_index(tmp_path, SID) is not None


def test_torn_tail_is_not_scanned_until_completed(tmp_path):
    write_rows(tmp_path, [user("u1", 1.0), assistant("a1", 1.1)])
    path = journal_path(tmp_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            '{"id":"a2","ts":1.2,"type":"message","payload":{"kind":"message","role":"assistant"'
        )  # no newline
    index = refreshed(tmp_path)
    assert index.coverage["complete"] is False
    assert [c.id for c in index.checkpoints] == ["u1", "a1"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(',"content":[{"text":"late"}]}}\n')
    index = refreshed(tmp_path)
    assert index.coverage["complete"] is True
    assert [c.id for c in index.checkpoints] == ["u1", "a2"]


def test_probe_reports_missing_ready_and_stale(tmp_path):
    assert ti.probe_index(tmp_path, SID).state == "missing"
    write_rows(tmp_path, [user("u1", 1.0), assistant("a1", 1.1)])
    assert ti.probe_index(tmp_path, SID).state == "stale"
    refreshed(tmp_path)
    assert ti.probe_index(tmp_path, SID).state == "ready"
    write_rows(tmp_path, [assistant("a2", 1.2)])
    assert ti.probe_index(tmp_path, SID).state == "stale"


# ---------------------------------------------------------------------------
# The async view the route consumes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_checkpoints_view_missing_journal_is_ready_empty(tmp_path):
    view = await ti.checkpoints_view(tmp_path, SID)
    assert view["index"]["state"] == "ready"
    assert view["checkpoints"] == []


@pytest.mark.asyncio
async def test_checkpoints_view_builds_in_background_then_settles(tmp_path, monkeypatch):
    write_rows(
        tmp_path,
        [start("s1", 1.0, "t1"), user("u1", 1.1), assistant("a1", 1.2), marker("m1", 1.3, "t1")],
    )
    entered, release = threading.Event(), threading.Event()
    real = ti.refresh_index

    def slow(config_dir, session_id):
        entered.set()
        assert release.wait(30), "test never released the build"
        return real(config_dir, session_id)

    monkeypatch.setattr(ti, "refresh_index", slow)
    view = await ti.checkpoints_view(tmp_path, SID, wait_s=0.05)
    assert view["index"]["state"] == "building"
    assert await asyncio.to_thread(entered.wait, 30)
    release.set()
    deadline = asyncio.get_running_loop().time() + 30.0
    while True:
        view = await ti.checkpoints_view(tmp_path, SID, wait_s=0.05)
        if view["index"]["state"] == "ready":
            break
        assert asyncio.get_running_loop().time() < deadline, view
        await asyncio.sleep(0.02)
    assert [c["id"] for c in view["checkpoints"]] == ["u1", "a1"]
    assert view["checkpoints"][1]["outcome"] == "complete"
    assert view["checkpoints"][1]["naming"]["state"] == "pending"


@pytest.mark.asyncio
async def test_a_poll_joining_an_in_flight_build_skips_the_budget(tmp_path, monkeypatch):
    """The joined-poll rule: ONLY the call that starts a build waits for it.

    Every poll of a slow build used to re-await the in-flight task for the
    whole first-paint budget (measured: 6 polls over one 3.0 s scan, ~220 ms
    each — the rail polls while a build runs). Pinned STRUCTURALLY, not with a
    clock: a spy records the timeout ``checkpoints_view`` passes to
    ``asyncio.wait``, and the starting call must pass its budget while a poll
    that joins the same build passes NONE at all.
    """
    write_rows(
        tmp_path,
        [start("s1", 1.0, "t1"), user("u1", 1.1), assistant("a1", 1.2), marker("m1", 1.3, "t1")],
    )
    entered, release = threading.Event(), threading.Event()
    real = ti.refresh_index

    def slow(config_dir, session_id):
        entered.set()
        assert release.wait(30), "test never released the build"
        return real(config_dir, session_id)

    monkeypatch.setattr(ti, "refresh_index", slow)

    timeouts: list[float | None] = []
    real_wait = asyncio.wait

    async def spying_wait(fs, *, timeout=None, **kwargs):
        # Delegates to the real wait — recording only, so any other caller of
        # asyncio.wait in this test is unaffected by the spy.
        timeouts.append(timeout)
        return await real_wait(fs, timeout=timeout, **kwargs)

    monkeypatch.setattr(asyncio, "wait", spying_wait)
    try:
        first = await ti.checkpoints_view(tmp_path, SID, wait_s=0.05)
        assert first["index"]["state"] == "building"
        assert timeouts == [0.05], "the starting call must wait its first-paint budget"
        assert await asyncio.to_thread(entered.wait, 30)

        joined = await ti.checkpoints_view(tmp_path, SID, wait_s=0.05)
        assert joined["index"]["state"] == "building"
        assert timeouts == [0.05], "a joined poll re-awaited the in-flight build"
    finally:
        release.set()

    deadline = asyncio.get_running_loop().time() + 30.0
    while True:
        view = await ti.checkpoints_view(tmp_path, SID, wait_s=0.05)
        if view["index"]["state"] == "ready":
            break
        assert asyncio.get_running_loop().time() < deadline, view
        await asyncio.sleep(0.02)
    assert [c["id"] for c in view["checkpoints"]] == ["u1", "a1"]


@pytest.mark.asyncio
async def test_checkpoints_view_cooldown_after_failure(tmp_path, monkeypatch):
    write_rows(tmp_path, [user("u1", 1.0)])
    calls = {"n": 0}

    def broken(config_dir, session_id):
        calls["n"] += 1
        raise OSError("journal on fire")

    monkeypatch.setattr(ti, "refresh_index", broken)
    first = await ti.checkpoints_view(tmp_path, SID, wait_s=5)
    assert first["index"]["state"] == "error"
    second = await ti.checkpoints_view(tmp_path, SID, wait_s=5)
    assert second["index"]["state"] == "error"
    assert calls["n"] == 1  # the cooldown absorbed the second ask


# ---------------------------------------------------------------------------
# Remediation round 1: the incremental window, the closing answer, extraction,
# cache freshness, the inode term, and the build-pass sweep
# ---------------------------------------------------------------------------


def test_incremental_append_keeps_an_earlier_runs_outcome(tmp_path, tmp_path_factory, monkeypatch):
    """BLOCKER-1: the first re-derived turn's run start must ride in the region.

    Two settled runs, then ONE plain append. The re-derivation used to begin at
    u1 while u1's run start (s1) stayed outside the region, so m1 looked like an
    orphan and a1's ``complete`` vanished — while a full scan of the same bytes
    kept it. The window now backs up to the start row itself.
    """
    write_rows(
        tmp_path,
        [
            start("s1", 1.0, "t1"),
            user("u1", 1.1, "one"),
            assistant("a1", 1.2, "answer one"),
            marker("m1", 1.3, "t1"),
            start("s2", 2.0, "t2"),
            user("u2", 2.1, "two"),
            assistant("a2", 2.2, "answer two"),
            marker("m2", 2.3, "t2"),
        ],
    )
    calls = _scanners(monkeypatch)
    base = refreshed(tmp_path)
    assert calls == {"full": 1, "incremental": 0}
    assert [c.outcome for c in base.checkpoints if c.kind == "completion"] == [
        "complete",
        "complete",
    ]

    write_rows(tmp_path, [tool("x9", 3.0)])
    grown = refreshed(tmp_path)
    assert calls == {"full": 1, "incremental": 1}

    other = tmp_path_factory.mktemp("reference")
    write_rows(
        other, [json.loads(line) for line in journal_path(tmp_path).read_text().splitlines()]
    )
    reference = refreshed(other)
    assert [(c.id, c.kind, c.turn, c.seq, c.text, c.outcome) for c in grown.checkpoints] == [
        (c.id, c.kind, c.turn, c.seq, c.text, c.outcome) for c in reference.checkpoints
    ]
    assert [(c.id, c.outcome) for c in grown.checkpoints if c.kind == "completion"] == [
        ("a1", "complete"),
        ("a2", "complete"),
    ]


def test_a_turn_closes_on_its_closing_answer_not_its_last_row(tmp_path):
    """QA round 1's Q1 ruling: tool/inject rows no longer carry the tick.

    The old rule closed on whatever row was last — on real sessions 43/67
    closers were tool/inject rows and 24/67 hover texts were harness notices
    like ``[model switch]``. The tick now rides the turn's last assistant row
    with content; notices show through only via the no-answer fallback.
    """
    write_rows(
        tmp_path,
        [
            start("s1", 1.0, "t1"),
            user("u1", 1.1, "do the thing"),
            assistant("a1", 1.2, "the done answer"),
            tool("x1", 1.3),
            inject("i1", 1.4, text="[model switch] a notice"),
            marker("m1", 1.5, "t1"),
        ],
    )
    index = refreshed(tmp_path)
    completion = next(c for c in index.checkpoints if c.kind == "completion")
    assert completion.id == "a1"
    assert completion.text == "the done answer"
    assert completion.outcome == "complete"


def test_a_turn_with_no_answer_falls_back_to_its_last_row(tmp_path):
    """The fallback half of the ruling: a turn that never produced an answer
    still gets a tick, on its last message row."""
    write_rows(
        tmp_path,
        [
            start("s1", 1.0, "t1"),
            user("u1", 1.1, "question"),
            tool("x1", 1.2),
            marker("m1", 1.3, "t1"),
        ],
    )
    index = refreshed(tmp_path)
    completion = next(c for c in index.checkpoints if c.kind == "completion")
    assert completion.id == "x1"
    assert completion.text == ""
    assert completion.outcome == "complete"


def test_injected_rows_extract_text_beyond_the_text_key(tmp_path):
    """MAJOR-2: a peer's ``body`` and a marker's ``summary`` are text too.

    Indexing them as "" made a peer's message unfindable by BE-3's find — the
    empty doc was indistinguishable from a genuinely empty one.
    """
    write_rows(
        tmp_path,
        [
            user("u1", 1.0, "go"),
            {
                "id": "i1",
                "ts": 1.1,
                "type": "message",
                "payload": {
                    "kind": "custom",
                    "custom_type": "peer_message",
                    "details": {"body": "peer body here", "sender": "w1"},
                },
            },
            {
                "id": "i2",
                "ts": 1.2,
                "type": "message",
                "payload": {
                    "kind": "custom",
                    "custom_type": "compaction",
                    "details": {"summary": "folded summary"},
                },
            },
            {
                "id": "i3",
                "ts": 1.3,
                "type": "message",
                "payload": {"kind": "custom", "custom_type": "gate_timeout", "details": {}},
            },
            inject("i4", 1.4, text="plain text key"),
        ],
    )
    index = refreshed(tmp_path)
    texts = {m.id: m.text for m in index.messages if m.injected}
    assert texts == {
        "i1": "peer body here",
        "i2": "folded summary",
        "i3": "",
        "i4": "plain text key",
    }


def test_message_docs_carry_their_custom_type_and_round_trip(tmp_path):
    """``MessageDoc.custom_type``: the find filter's discriminator (design §5.3).

    Set from the inject row — the same payload field the injection rule
    already reads, so this is a field copy, not new scanning — kept through
    the cache round trip, and omitted from a genuine row's payload. A
    pre-bump payload without the key loads as ``None`` rather than dropping
    the doc: the VERSION GATE, not the reader, is what guarantees old peer
    docs are re-derived with the field populated (a v1 cache with
    ``custom_type=None`` peers would slip past the find filter — the leak the
    bump exists to close).
    """
    write_rows(
        tmp_path,
        [
            user("u1", 1.0, "go"),
            inject("i1", 1.1, custom_type="peer_message", text="peer body here"),
        ],
    )
    index = refreshed(tmp_path)
    docs = {m.id: m for m in index.messages}
    assert docs["u1"].custom_type is None
    assert docs["i1"].custom_type == "peer_message"

    payload = docs["i1"].to_payload()
    assert payload["custom_type"] == "peer_message"
    assert "custom_type" not in docs["u1"].to_payload()
    assert ti.MessageDoc.from_payload(payload) == docs["i1"]

    # A version-1 style payload (no key) loads as None rather than being
    # dropped; the version gate above, not the reader, keeps such docs out.
    legacy = {key: value for key, value in payload.items() if key != "custom_type"}
    restored = ti.MessageDoc.from_payload(legacy)
    assert restored is not None and restored.custom_type is None


@pytest.mark.asyncio
async def test_a_naming_write_is_seen_without_a_journal_change(tmp_path):
    """BE-2's freshness fix: the resident fast path must notice CACHE writes.

    ``patch_naming`` moves the cache file's mtime and leaves the journal alone;
    before the fix a resident index answered ``pending`` for as long as the
    journal stayed quiet (87 s measured in a live daemon, then a restart showed
    the name). The poll shape is deliberate: every subsequent view must agree.
    """
    write_rows(tmp_path, [user("u1", 1.0, "one"), assistant("a1", 1.1, "answer")])
    refreshed(tmp_path)
    first = await ti.checkpoints_view(tmp_path, SID)
    assert first["index"]["state"] == "ready"
    assert first["checkpoints"][1]["naming"] == {"state": "pending", "name": None, "summary": None}

    before = journal_path(tmp_path).stat()
    assert ti.patch_naming(tmp_path, SID, {"u1": {"name": "Do it", "summary": "Done."}}) is True
    after = journal_path(tmp_path).stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)

    for _ in range(3):  # the poll shape: ready stays ready, naming stays served
        view = await ti.checkpoints_view(tmp_path, SID)
        assert view["index"]["state"] == "ready"
        naming = view["checkpoints"][1]["naming"]
        assert naming["state"] == "ready"
        assert naming["name"] == "Do it"


def test_a_same_size_same_mtime_replacement_is_detected_by_inode(tmp_path, monkeypatch):
    """R3's discriminating shape for the freshness term (QA proved it bites).

    A byte-length-preserving rewrite under the SAME mtime: size and mtime both
    match the recorded signature, so only the inode exposes the replacement —
    drop that term and the stale text is served without a scan.
    """
    write_rows(tmp_path, [user("u1", 1.0, "first question"), assistant("a1", 1.1, "answer")])
    refreshed(tmp_path)
    stat = journal_path(tmp_path).stat()
    rows = [json.loads(line) for line in journal_path(tmp_path).read_text().splitlines()]
    assert rows[0]["payload"]["content"][0]["text"] == "first question"
    rows[0]["payload"]["content"][0]["text"] = "first questioN"  # same byte length
    swap = journal_path(tmp_path).with_suffix(".swap")
    swap.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    os.utime(swap, ns=(stat.st_mtime_ns, stat.st_mtime_ns))
    os.replace(swap, journal_path(tmp_path))
    replaced = journal_path(tmp_path).stat()
    assert replaced.st_ino != stat.st_ino  # the premise: a NEW inode ...
    assert replaced.st_size == stat.st_size
    assert replaced.st_mtime_ns == stat.st_mtime_ns  # ... at the SAME size and mtime

    calls = _scanners(monkeypatch)
    index = refreshed(tmp_path)
    assert calls == {"full": 1, "incremental": 0}
    assert index.checkpoints[0].text == "first questioN"


@pytest.mark.asyncio
async def test_an_unwritable_cache_still_serves_the_resident(tmp_path, monkeypatch):
    """MINOR-A: a cache that can never be written costs speed, not the read.

    With the stamp requirement and no cache file on disk, no stamp can ever
    match — before this clause every view fell through to probe -> stale -> a
    fresh full scan (measured 1,2,3 scans on successive calls; a writable root
    runs 1,1,1). An unwritable cache must not degrade into a rescan per poll.
    """
    write_rows(tmp_path, [user("u1", 1.0, "one"), assistant("a1", 1.1, "answer")])
    monkeypatch.setattr(ti, "write_index", lambda *args, **kwargs: None)
    calls = _scanners(monkeypatch)

    first = await ti.checkpoints_view(tmp_path, SID, wait_s=5.0)
    assert first["index"]["state"] == "ready"
    assert [c["id"] for c in first["checkpoints"]] == ["u1", "a1"]
    for _ in range(3):
        view = await ti.checkpoints_view(tmp_path, SID)
        assert view["index"]["state"] == "ready"
    assert calls == {"full": 1, "incremental": 0}


@pytest.mark.asyncio
async def test_a_fold_after_the_recorded_tail_still_rescans(tmp_path, monkeypatch):
    """R3: a fold strictly past the recorded tail is invisible to size+bytes.

    The tail row still verifies and the file NET GREW (the folded notice is
    longer than the one-byte tool body), so the same-inode ladder alone would
    take the incremental path; the inode is what selects the rescan. Asserted
    on the BRANCH taken, not only the output, because the branch is what the
    term exists for.
    """
    session = tmp_path / "sessions" / SID
    transcript = Transcript(session)
    await transcript.append_message(Message.user("one", id="u1"))
    await transcript.append_message(
        Message(role="assistant", content=[TextContent(text="answer one")], id="a1")
    )
    await transcript.append_message(Message.user("two", id="u2"))
    await transcript.append_message(
        Message(role="assistant", content=[TextContent(text="answer two")], id="a2")
    )
    first = refreshed(tmp_path)
    assert first.scan.offset == journal_path(tmp_path).stat().st_size

    calls = _scanners(monkeypatch)
    small = Message(role="tool", content=[TextContent(text="x")], tool_call_id="c1")
    small.id = "x9"
    await transcript.append_message(small)
    await transcript.append_prune(
        "x9", "[pruned: a notice clearly longer than the one-byte body it replaces]"
    )
    assert await transcript.compact_file(min_reclaim_bytes=0) > 0
    assert journal_path(tmp_path).stat().st_size > first.scan.offset  # net grew

    index = refreshed(tmp_path)
    assert calls == {"full": 1, "incremental": 0}
    assert index.scan.offset == journal_path(tmp_path).stat().st_size
    assert [c.id for c in index.checkpoints] == ["u1", "a1", "u2", "a2"]


def test_a_build_sweeps_the_cache_of_a_deleted_session(tmp_path, monkeypatch):
    """MINOR-4: D8's cleanup bullet, in the pass that builds (a full scan)."""
    write_rows(tmp_path, [user("u1", 1.0)])
    refreshed(tmp_path)
    gone = "beefbeef0001"
    other = tmp_path / "sessions" / gone
    other.mkdir(parents=True)
    (other / "transcript.jsonl").write_text("")
    assert ti.refresh_index(tmp_path, gone) is not None
    assert ti.index_path(tmp_path, gone).exists()

    (other / "transcript.jsonl").unlink()
    other.rmdir()
    # Force a BUILD pass on this root: a smaller journal means a full scan.
    rows = [json.loads(line) for line in journal_path(tmp_path).read_text().splitlines()]
    rows[0]["payload"]["content"] = [{"text": "h"}]
    journal_path(tmp_path).write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    )
    calls = _scanners(monkeypatch)
    refreshed(tmp_path)
    assert calls == {"full": 1, "incremental": 0}
    assert not ti.index_path(tmp_path, gone).exists()
    assert ti.index_path(tmp_path, SID).exists()  # a live session keeps its cache


# ---------------------------------------------------------------------------
# Runs: the open frame's per-run facts
# ---------------------------------------------------------------------------


def tool_row(
    id_: str,
    ts: float,
    *,
    text: str = "tool output",
    is_error: bool = False,
    duration_s: float | None = 2.0,
    fault: str | None = None,
    delivery: str | None = None,
) -> dict[str, Any]:
    """A tool row with the provider envelope the counters read."""
    details: dict[str, Any] = {}
    if fault is not None:
        details["__fault"] = fault
    if delivery is not None:
        details["delivery"] = delivery
    provider: dict[str, Any] = {"details": details, "useless": False}
    if duration_s is not None:
        provider["duration_s"] = duration_s
    return {
        "id": id_,
        "ts": ts,
        "type": "message",
        "payload": {
            "kind": "message",
            "role": "tool",
            "content": [{"text": text}],
            "is_error": is_error,
            "provider_payload": provider,
        },
    }


def incident(id_: str, ts: float, detail: str = "the turn died") -> dict[str, Any]:
    """The error-level custom the renderer paints as a terminal marker."""
    return {
        "id": id_,
        "ts": ts,
        "type": "message",
        "payload": {
            "kind": "custom",
            "custom_type": "session_incident",
            "details": {"text": detail},
        },
    }


def runs_of(index: ti.TranscriptIndex) -> list[tuple[str, str, int, int, float, bool]]:
    """Each run as (opening user, closing answer, actions, failed, worked, settled)."""
    return [
        (
            run.opening_user_id,
            run.closing_answer_id,
            run.action_count,
            run.failed_count,
            run.worked_seconds,
            run.settled,
        )
        for run in index.runs
    ]


def test_runs_partition_two_turns_and_count_their_tools(tmp_path):
    """The two-turn shape: counts, worked seconds and outcomes per run."""
    write_rows(
        tmp_path,
        [
            start("s1", 1.0, "t1"),
            user("u1", 1.1, "do the thing"),
            assistant("a1", 1.2, "working", tool_calls=True),
            tool_row("x1", 1.3, duration_s=3.0),
            tool_row("x2", 1.4, duration_s=1.5, is_error=True),
            assistant("a2", 1.5, "done"),
            marker("m1", 1.6, "t1"),
            start("s2", 2.0, "t2"),
            user("u2", 2.1, "again"),
            assistant("a3", 2.2, "working", tool_calls=True),
            tool_row("x3", 2.3, duration_s=0.5),
            assistant("a4", 2.4, "done"),
            marker("m2", 2.5, "t2"),
        ],
    )
    index = refreshed(tmp_path)
    assert runs_of(index) == [
        ("u1", "a2", 2, 1, 4.5, True),
        ("u2", "a4", 1, 0, 0.5, True),
    ]
    assert [run.outcome for run in index.runs] == ["complete", "complete"]
    # The span is the run's FIRST and LAST rows, on the journal's clock — the
    # last row, not the answer, because that is what the page's cut is measured
    # in (a trailing receipt belongs to the run it follows).
    assert index.runs[0].start_ts == 1.1 and index.runs[0].end_ts == 1.6


def test_a_steer_stays_inside_its_run(tmp_path):
    """The client's own rule, and the reason a count is a run's, not a turn's.

    A user row that arrives while the run is open and the run's last painting row
    is a TOOL row is a steer: the run keeps its identity — its FIRST user row —
    and its counts span both of the turn's cycles. A partition that cut here would
    report two half-runs where the bar draws one.
    """
    write_rows(
        tmp_path,
        [
            start("s1", 1.0, "t1"),
            user("u1", 1.1, "start"),
            assistant("a1", 1.2, "working", tool_calls=True),
            tool_row("x1", 1.3, duration_s=2.0),
            user("u2", 1.4, "actually, also this"),
            assistant("a2", 1.5, "working", tool_calls=True),
            tool_row("x2", 1.6, duration_s=3.0),
            assistant("a3", 1.7, "done"),
            marker("m1", 1.8, "t1"),
        ],
    )
    index = refreshed(tmp_path)
    assert runs_of(index) == [("u1", "a3", 2, 0, 5.0, True)]


def test_a_settled_answer_lets_the_next_user_row_open_a_run(tmp_path):
    """The closure's second arm: a run whose tail is a settled assistant row."""
    write_rows(
        tmp_path,
        [
            user("u1", 1.1, "one"),
            assistant("a1", 1.2, "answer one"),
            user("u2", 1.3, "two"),
            assistant("a2", 1.4, "answer two"),
        ],
    )
    index = refreshed(tmp_path)
    assert runs_of(index) == [
        ("u1", "a1", 0, 0, 0.0, True),
        ("u2", "a2", 0, 0, 0.0, False),
    ]
    # The tail run has no marker, so it is LIVE — not settled with an empty
    # outcome, and that difference is what the wire's counts are gated on.
    assert index.runs[-1].outcome == ti.OUTCOME_OPEN
    assert index.runs[0].outcome is None


def test_a_session_incident_closes_the_run_before_the_next_user_row(tmp_path):
    """The renderer's second terminal marker, honoured by the partition."""
    write_rows(
        tmp_path,
        [
            start("s1", 1.0, "t1"),
            user("u1", 1.1, "start"),
            assistant("a1", 1.2, "working", tool_calls=True),
            tool_row("x1", 1.3),
            incident("i1", 1.4),
            start("s2", 2.0, "t2"),
            user("u2", 2.1, "try again"),
            assistant("a2", 2.2, "done"),
            marker("m2", 2.3, "t2"),
        ],
    )
    index = refreshed(tmp_path)
    assert [run.opening_user_id for run in index.runs] == ["u1", "u2"]
    assert index.runs[0].action_count == 1


def test_the_failure_count_excludes_the_settled_non_failures(tmp_path):
    """A stopped call, a never-run call and a partial delivery are not failures.

    The count is only useful if it states what the client's own ``isFailedCall``
    would have derived from the same rows, so the three exclusions are pinned
    here: an ``is_error`` row that was aborted, one whose ``send`` delivery is
    partial, and one that never ran. A bar reading "3 failed" for that journal is
    a bar reporting work that did not happen.
    """
    write_rows(
        tmp_path,
        [
            user("u1", 1.1, "go"),
            assistant("a1", 1.2, "working", tool_calls=True),
            tool_row("x1", 1.3, is_error=True, fault="aborted"),
            tool_row("x2", 1.4, is_error=True, fault="skipped"),
            tool_row("x3", 1.5, is_error=True, delivery="mailbox"),
            tool_row("x4", 1.6, is_error=True, delivery="unconfirmed"),
            tool_row("x5", 1.7, is_error=True),
            assistant("a2", 1.8, "done"),
            marker("m1", 1.9, "t1"),
        ],
    )
    index = refreshed(tmp_path)
    run = index.runs[0]
    assert run.action_count == 5
    assert run.failed_count == 1


def test_a_dropped_row_body_marks_the_run_incomplete(tmp_path):
    """A row the scanner had to drop leaves a lower bound, and the run says so."""
    big = "z" * (3 << 20)
    write_rows(
        tmp_path,
        [
            user("u1", 1.1, "go"),
            tool_row("x1", 1.2, text=big, duration_s=4.0),
            tool_row("x2", 1.3, duration_s=1.0),
        ],
    )
    index = refreshed(tmp_path)
    run = index.runs[0]
    assert run.action_count == 2  # the head names the role, so it still counts
    assert run.complete is False
    assert run.worked_seconds == 1.0  # the dropped row's duration is unknowable


def test_incremental_appends_agree_with_a_full_rescan(tmp_path):
    """THE INVARIANT THE CARRIED RUN EXISTS FOR: append, refresh, rescan, equal.

    An incremental scan re-derives from a resume window that can sit INSIDE a
    run (a steer's own ``attention_started``), so the run straddling that window
    is carried rather than re-opened. Cutting the keeps at the window instead
    emitted the straddling run twice — once truncated, once head-cut — and the
    two counts reconciled with nothing. This asserts the only property that
    matters: the incremental answer equals the one a full scan of the same
    journal gives.
    """
    first = [
        start("s1", 1.0, "t1"),
        user("u1", 1.1, "start"),
        assistant("a1", 1.2, "working", tool_calls=True),
        tool_row("x1", 1.3, duration_s=2.0),
        user("u2", 1.4, "and also this"),
        assistant("a2", 1.5, "working", tool_calls=True),
        tool_row("x2", 1.6, duration_s=3.0),
    ]
    write_rows(tmp_path, first)
    initial = refreshed(tmp_path)
    assert len(initial.runs) == 1
    write_rows(
        tmp_path,
        [
            assistant("a3", 1.7, "done"),
            marker("m1", 1.8, "t1"),
            start("s2", 2.0, "t2"),
            user("u3", 2.1, "next"),
            assistant("a4", 2.2, "answer"),
        ],
    )
    incremental = refreshed(tmp_path)
    full = ti._scan_full(journal_path(tmp_path), None)
    assert runs_of(incremental) == runs_of(full)
    assert [run.opening_user_id for run in incremental.runs] == ["u1", "u3"]


def test_version_three_cache_is_discarded_so_runs_are_never_missing(tmp_path):
    """A cache without the runs section must not be served as "no runs here"."""
    write_rows(tmp_path, [user("u1", 1.1, "hi"), assistant("a1", 1.2, "hello")])
    index = refreshed(tmp_path)
    payload = json.loads(ti.index_path(tmp_path, SID).read_text())
    payload["version"] = 3
    payload.pop("runs", None)
    ti.index_path(tmp_path, SID).write_text(json.dumps(payload))
    raw = ti._read_raw(tmp_path, SID)
    assert raw is not None and raw["version"] == 3
    assert ti._index_from_raw(raw) is None
    assert len(index.runs) == 1
