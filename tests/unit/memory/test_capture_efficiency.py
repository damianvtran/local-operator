"""Capture efficiency (memory round 2026-10-27, Slice B2 + the B5 recall hook).

Each test pins one deterministic rule and FAILS on the pre-B2 code:

- debris admission: serialized transcript lines, truncation tails and (on the
  carry arm) sub-floor fragments never become facts, and are counted;
- state tags (goal/progress/next-step) are ONE live head per (session, tag),
  revised in place and rolled over with ``replaces`` at ``MAX_REVISIONS``;
- dedup is normalised (casefold + whitespace) within a session only;
- the per-session capture budget forgets the OLDEST captured candidates and
  never a deliberate fact (decision D3), and reports the evictions;
- ``recall_stats`` is a local side table, bumped once per entity per turn.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from local_operator import memory_capture
from local_operator.memory import MAX_REVISIONS, MemoryIdentity, MemoryStore
from local_operator.memory_capture import (
    CAPTURE_SESSION_TOKENS,
    capture_evicted,
    capture_status,
    carry_rows,
    extract,
    hint,
    persist,
)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(
        tmp_path / "cfg",
        MemoryIdentity("eff-session", "t", "a"),
        tmp_path / "sessions" / "eff-session",
        origin="eff-device",
    )


def _live(store: MemoryStore, tag: str | None = None) -> list[dict[str, Any]]:
    with store.connection() as db:
        rows = [json.loads(r[0]) for r in db.execute("SELECT fact FROM current")]
    return [f for f in rows if tag is None or f["tags"] == [tag]]


def _visible(store: MemoryStore, query: str) -> list[dict[str, Any]]:
    return store.search(query, "session", limit=20)


# --- debris admission ------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        '[Assistant tool calls] bash(command="make check", i="Running the gate now")',
        "[Tool result: bash] the gate is running and will finish in a few minutes",
        "[Injected] the release window has an owner and we should hand over",
        "the gate output was long [... 367 more characters truncated] and then ended",
    ],
)
def test_serialized_lines_and_truncation_tails_are_refused_and_counted(line):
    rows = carry_rows(f"- Asks: {line}\n")
    assert list(rows) == [] and rows.skipped == 1
    summary = extract("", f"## Key Decisions\n- {line}")
    assert list(summary) == [] and summary.skipped == 1


@pytest.mark.parametrize("fragment", ['ve", timeout=180)', ".... [100%]", "-ui/AGENTS.md."])
def test_carry_fragments_under_the_floor_are_refused(fragment):
    rows = carry_rows(f"- Asks: {fragment}\n- Asks: keep the staged rollout plan for the launch\n")
    assert [row["body"] for row in rows] == ["keep the staged rollout plan for the launch"]
    assert rows.skipped == 1


def test_short_curated_summary_constraints_still_count_as_knowledge():
    """The length floor is carry-only: a model-written short constraint is real."""
    rows = extract("", "## Constraints\n- Do not touch `/home/ec2-user/ui*`.")
    assert [row["body"] for row in rows] == ["Do not touch `/home/ec2-user/ui*`."]


def test_finish_sums_carry_skips_into_the_pass_attribution(store):
    import asyncio

    async def run():
        async def scan():
            return memory_capture.CaptureRows()

        return await memory_capture.finish(
            store,
            "carry-skip",
            asyncio.create_task(scan()),
            "",
            carry='- Asks: [Assistant tool calls] bash(command="x")\n'
            "- Asks: keep the staged rollout plan for the launch\n",
        )

    assert asyncio.run(run()) == "ok"
    detail = memory_capture.capture_detail(store, "carry-skip")
    assert detail is not None
    assert (detail.n, detail.skipped, detail.reason) == (1, 1, "admission-skip")


# --- state tags: one head per (session, tag) -------------------------------


def test_state_tags_keep_one_head_per_tag_across_five_passes(store):
    for n in range(5):
        rows = extract(
            "",
            f"## Goal\n- Ship the capture efficiency slice, pass {n}\n"
            f"## Progress\n- Finished step {n} of the rollout\n- Tests for step {n} pass\n"
            f"## Next Steps\n- Start step {n + 1} of the rollout\n"
            "## Key Decisions\n- Keep the stable prefix untouched\n",
        )
        persist(store, f"marker-{n}", rows, "ok", committed_at=1000.0 + n)
    for tag in ("goal", "progress", "next-step"):
        heads = _live(store, tag)
        assert len(heads) == 1, tag
    assert _live(store, "goal")[0]["body"].endswith("pass 4")
    # A pass's several bullets are ONE snapshot revision, newest pass wins.
    assert _live(store, "progress")[0]["body"] == (
        "Finished step 4 of the rollout; Tests for step 4 pass"
    )
    assert len(_live(store, "decision")) == 1
    assert len(_live(store)) == 4


def test_a_corrected_state_row_is_never_overwritten(store):
    persist(store, "m0", extract("", "## Goal\n- Ship the first version"), "ok", committed_at=1.0)
    [item] = store.search("first version", "session")
    store.put(
        "session",
        {
            "title": "Goal",
            "body": "Ship the corrected version",
            "tags": ["goal"],
            "status": "active",
        },
        entity=item["id"],
        expected_version=item["version"],
    )
    persist(store, "m1", extract("", "## Goal\n- Ship the second version"), "ok", committed_at=2.0)
    assert [f["body"] for f in _live(store, "goal")] == ["Ship the corrected version"]


def test_a_full_state_chain_rolls_over_with_replaces_not_a_freeze(store, monkeypatch):
    monkeypatch.setattr(memory_capture, "MAX_REVISIONS", 3)
    for n in range(5):
        rows = extract("", f"## Goal\n- Ship goal revision number {n}")
        persist(store, f"roll-{n}", rows, "ok", committed_at=10.0 + n)
    visible = [i for i in _visible(store, "goal revision") if i["status"] == "candidate"]
    assert [i["title"] for i in visible] == ["Ship goal revision number 4"]
    goals = _live(store, "goal")
    # Two generations exist; the newer one names the older through ``replaces``.
    assert len(goals) == 2
    assert any(f["replaces"] for f in goals)
    assert MAX_REVISIONS > 3  # the store's own cap was not touched


def test_eviction_never_kills_a_live_state_head(store, monkeypatch):
    """Review round 1 M3: a goal not restated by later passes is still live.

    It keeps its old ``source_time``, so it is the OLDEST captured candidate and
    was the first one evicted; the chain then resolved to the tombstoned entity
    and every later Goal section was skipped for the rest of the session.
    """
    monkeypatch.setattr(memory_capture, "CAPTURE_SESSION_TOKENS", 600)
    goal = "Ship the capture efficiency slice to production this week"
    persist(store, "m0", extract("", f"## Goal\n- {goal}"), "ok", committed_at=1.0)
    for n in range(6):
        persist(store, f"d{n}", _bulky(n), "ok", committed_at=10.0 + n)
    assert capture_status(store)["evicted"] > 0
    assert [f["body"] for f in _live(store, "goal")] == [goal]
    newer = "Ship the capture slice after the QA round passes"
    persist(store, "m1", extract("", f"## Goal\n- {newer}"), "ok", committed_at=100.0)
    assert [f["body"] for f in _live(store, "goal")] == [newer]


def test_a_forgotten_state_chain_starts_a_fresh_generation(store):
    """A forget stays a forget, and the tag keeps recording current state."""
    persist(store, "m0", extract("", "## Goal\n- Ship the first goal of the week"), "ok")
    [item] = store.search("first goal", "session")
    store.put("session", None, entity=item["id"], expected_version=item["version"], forget=True)
    assert _live(store, "goal") == []
    persist(store, "m1", extract("", "## Goal\n- Ship the second goal of the week"), "ok")
    [live] = store.search("second goal", "session")
    assert live["id"] != item["id"]
    assert store.show("session", item["id"])["status"] == "forgotten"


def test_a_restated_previous_section_never_joins_the_current_revision(store):
    """QA round 1 Q3: one section per state tag, the last one wins, never `4; 3`."""
    summary = (
        "## Goal\n- Ship the capture slice, revision 4\n"
        "## Key Decisions\n- Keep provenance per session\n"
        "## Goal\n- Ship the capture slice, revision 3\n"
    )
    persist(store, "m", extract("", summary), "ok")
    assert [f["body"] for f in _live(store, "goal")] == ["Ship the capture slice, revision 3"]


@pytest.mark.asyncio
async def test_the_summary_state_beats_the_transcripts_restated_state(store):
    """``finish``: raw history holds the previous pass's Goal; the summary's wins."""
    import asyncio

    async def scan():
        return extract("## Goal\n- Ship the capture slice, revision 3 (old)\n")

    await memory_capture.finish(
        store,
        "arms",
        asyncio.create_task(scan()),
        "## Goal\n- Ship the capture slice, revision 4 (current)\n",
    )
    assert [f["body"] for f in _live(store, "goal")] == [
        "Ship the capture slice, revision 4 (current)"
    ]


def test_a_shorter_bullet_contained_in_an_earlier_one_is_kept(store):
    """Review m4: per-bullet comparison, not substring containment."""
    summary = "## Progress\n- Tests pass for step 3 of the rollout\n- Tests pass\n"
    persist(store, "m", extract("", summary), "ok")
    assert [f["body"] for f in _live(store, "progress")] == [
        "Tests pass for step 3 of the rollout; Tests pass"
    ]


def test_an_unenforceable_budget_is_reported_not_silent(store, monkeypatch):
    """Review m2: at the tombstone floor the budget stops binding — say so."""
    monkeypatch.setattr(memory_capture, "CAPTURE_SESSION_TOKENS", 200)
    monkeypatch.setattr(memory_capture, "MAX_TOMBSTONES", 2)
    for n in range(4):
        persist(store, f"f{n}", _bulky(n), "ok", committed_at=1.0 + n)
    assert capture_status(store)["unenforced_sessions"] == 1
    assert "can no longer forget older candidates" in hint(store, "f3")


# --- normalised dedup ------------------------------------------------------


def test_case_and_whitespace_variants_refresh_one_fact(store):
    persist(store, "a", extract("", "## Key Decisions\n- Keep the  Stable prefix"), "ok")
    persist(store, "b", extract("", "## Key Decisions\n- keep the stable   prefix"), "ok")
    assert len(_live(store, "decision")) == 1


@pytest.mark.parametrize(
    "first,second",
    [
        ("Proof lives in src/Widget/Proof.py now", "Proof lives in src/widget/proof.py now"),
        ("Set the env var DEBUG_MODE before the run", "Set the env var debug_mode before the run"),
    ],
)
def test_identifiers_differing_only_in_case_stay_two_facts(store, first, second):
    """Review m3 / QA Q5: paths and env vars are case-sensitive here."""
    persist(store, "a", extract("", f"## Key Decisions\n- {first}\n- {second}"), "ok")
    assert sorted(f["body"] for f in _live(store, "decision")) == sorted([first, second])


def test_dedup_never_merges_across_sessions(tmp_path):
    rows = extract("", "## Key Decisions\n- Keep the stable prefix")
    for sid in ("s-one", "s-two"):
        persist(
            MemoryStore(tmp_path / "cfg", MemoryIdentity(sid, "t", "a"), None, origin="d"),
            "m",
            rows,
            "ok",
        )
    with MemoryStore(
        tmp_path / "cfg", MemoryIdentity("s-one", "t", "a"), None, origin="d"
    ).connection() as db:
        owners = sorted(r[0] for r in db.execute("SELECT owner FROM current"))
    assert owners == ["s-one", "s-two"]


def test_a_legacy_raw_body_entity_is_refreshed_not_duplicated(store):
    """A row captured before normalisation keeps its id on the next pass."""
    import hashlib

    from local_operator.memory import _dump

    body = "Keep the Stable prefix"
    legacy = hashlib.sha256(
        (store.identity.session_id + "|" + _dump([body, ["decision"]])).encode()
    ).hexdigest()
    # Simulate the pre-normalisation build: its key was the RAW body.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(memory_capture, "_normalized", lambda text: text)
        persist(store, "old", extract("", f"## Key Decisions\n- {body}"), "ok")
    assert [r["id"] for r in store.search("stable prefix", "session")] == [legacy]
    persist(store, "new", extract("", f"## Key Decisions\n- {body}"), "ok")
    assert [r["id"] for r in store.search("stable prefix", "session")] == [legacy]


# --- per-session capture budget (D3) ---------------------------------------


def _bulky(n: int) -> list[dict[str, Any]]:
    words = " ".join(f"word{n}x{k}" for k in range(28))
    body = f"Decision {n}: {words}"
    return [{"title": body[:100], "body": body, "tags": ["decision"]}]


def test_over_budget_evicts_oldest_captured_only_and_deliberate_survives(store, monkeypatch):
    monkeypatch.setattr(memory_capture, "CAPTURE_SESSION_TOKENS", 1000)
    deliberate = store.put(
        "session", {"title": "Operator rule", "body": "Always deploy on Tuesdays only."}
    )
    for n in range(12):
        persist(store, f"bulk-{n}", _bulky(n), "ok", committed_at=100.0 + n)
    with store.connection() as db:
        captured = db.execute(
            "SELECT json_extract(fact,'$.body'),tokens FROM current "
            "WHERE json_extract(fact,'$.source.author')='deterministic-capture'"
        ).fetchall()
        tombs = db.execute("SELECT count(*) FROM ops WHERE kind='tombstone'").fetchone()[0]
    assert sum(t for _, t in captured) <= 1000
    kept = sorted(int(b.split(":")[0].split()[1]) for b, _ in captured)
    # Only the OLDEST went, and the newest survived.
    assert kept == list(range(12 - len(kept), 12)) and 11 in kept
    assert tombs == 12 - len(kept) > 0
    assert store.show("session", deliberate["id"])["status"] == "active"
    total = sum(capture_evicted(store, f"bulk-{n}") for n in range(12))
    assert total == tombs == capture_status(store)["evicted"]
    last = next(n for n in range(12) if capture_evicted(store, f"bulk-{n}"))
    assert "forgotten to keep automatic capture" in hint(store, f"bulk-{last}")


def test_deliberate_tokens_are_not_charged_to_the_capture_budget(store, monkeypatch):
    monkeypatch.setattr(memory_capture, "CAPTURE_SESSION_TOKENS", 400)
    for n in range(8):
        store.put("session", {"title": f"Rule {n}", "body": " ".join(["stated"] * 30) + str(n)})
    persist(store, "one", _bulky(0), "ok", committed_at=1.0)
    assert capture_status(store)["evicted"] == 0
    assert len(_live(store, "decision")) == 1


def test_the_shipped_budget_is_sixteen_thousand():
    assert CAPTURE_SESSION_TOKENS == 16_000


def test_an_evicted_candidate_is_a_replicating_tombstone(store, monkeypatch):
    monkeypatch.setattr(memory_capture, "CAPTURE_SESSION_TOKENS", 200)
    persist(store, "e0", _bulky(0), "ok", committed_at=1.0)
    persist(store, "e1", _bulky(1), "ok", committed_at=2.0)
    with store.connection() as db:
        ops = [json.loads(r[0]) for r in db.execute("SELECT data FROM ops WHERE kind='tombstone'")]
    assert len(ops) == 1
    assert ops[0]["scope"] == "session" and ops[0]["fact"] is None and ops[0]["parents"]
    from local_operator.memory import SESSION_EXPORT_NAME

    # Exported to the portable session log like any forget, so a moved or
    # imported session cannot resurrect it.
    assert ops[0]["op_id"] in (store.session_dir / SESSION_EXPORT_NAME).read_text()


def test_the_carry_arm_reaches_new_material_across_passes(store):
    """QA round 2 Q7: a cumulative carry must not re-offer the same 12 asks forever.

    Each pass's carry holds every earlier ask, oldest first. Oldest-first selection
    re-picked the stored ones and never reached later turns; decisions never got a
    slot at all.
    """
    from local_operator.compaction.mine import Carry, render_carry

    asks: list[str] = []
    decisions: list[str] = []
    for n in range(4):
        asks += [f"please migrate billing table number {n}-{k} to the v2 schema" for k in range(8)]
        decisions += [
            f"we will keep the legacy reader for table {n} until friday" for _ in range(1)
        ]
        carry = render_carry(Carry(asks=list(asks), decisions=list(decisions)))
        persist(
            store, f"p{n}", memory_capture._carry_arm(store, carry, False), "ok", committed_at=n
        )
    bodies = {f["body"] for f in _live(store, "ask")}
    assert all(
        f"please migrate billing table number {n}-7 to the v2 schema" in bodies for n in range(4)
    )
    assert len(_live(store, "ask")) == 32  # every ask, across passes
    assert len(_live(store, "decision")) == 4  # decisions were never starved


def test_decisions_keep_a_share_of_a_full_pass():
    from local_operator.compaction.mine import Carry, render_carry

    carry = render_carry(
        Carry(
            asks=[f"please migrate billing table number {k} to the v2 schema" for k in range(20)],
            decisions=[
                f"we will keep the legacy reader for table {k} until friday" for k in range(20)
            ],
        )
    )
    rows = carry_rows(carry)
    tags = [r["tags"][0] for r in rows]
    assert (tags.count("decision"), tags.count("ask")) == (4, 8)
    # Newest first within each kind.
    assert rows[0]["body"].endswith("table 19 until friday")
    assert any(r["body"] == "please migrate billing table number 19 to the v2 schema" for r in rows)


# --- B5 recall telemetry ---------------------------------------------------


def _stats(store: MemoryStore) -> dict[str, int]:
    with store.connection() as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='recall_stats'").fetchone():
            return {}
        return {r[0]: r[1] for r in db.execute("SELECT entity,n FROM recall_stats")}


def test_recall_stats_costs_no_pages_until_a_read_is_recorded(store):
    """The byte cap is in pages; a store that never records a read pays nothing."""
    store.put("session", {"title": "Launch rule", "body": "Launch windows close at noon."})
    with store.connection() as db:
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='recall_stats'").fetchone()
    assert store.recall_summary() == {"live": 1, "recalled": 0, "reads": 0, "captured_recalled": 0}


def _tool_read(store: MemoryStore, path: str, turn: str) -> Any:
    """One agent ``read`` through the real tool executor, as the model issues it."""
    import asyncio

    from local_operator.harness.types import ToolContext
    from local_operator.tools.memory_tools import execute_memory

    context = ToolContext(
        session_id=store.identity.session_id,
        memory_config_dir=str(store.root.parent),
        send_turn_id=turn,
    )
    result = asyncio.run(execute_memory("call", "read", {"path": path}, context))
    assert not result.is_error, result
    return result


def test_agent_reads_bump_once_per_turn(store):
    fact = store.put("session", {"title": "Launch rule", "body": "Launch windows close at noon."})
    _tool_read(store, "memory://session?q=launch%20windows", "turn-1")
    _tool_read(store, f"memory://session/{fact['id']}", "turn-1")
    _tool_read(store, "memory://session?q=launch%20windows", "turn-1")
    assert _stats(store) == {fact["id"]: 1}
    _tool_read(store, f"memory://session/{fact['id']}", "turn-2")
    assert _stats(store) == {fact["id"]: 2}
    summary = store.recall_summary()
    assert (summary["live"], summary["recalled"], summary["reads"]) == (1, 1, 2)


def test_human_store_reads_are_not_recalls(store):
    """Review m1 / QA Q6: ``show``/``search`` also serve `lop memory`; only callers count."""
    fact = store.put("session", {"title": "Launch rule", "body": "Launch windows close at noon."})
    store.show("session", fact["id"])
    store.search("launch windows", "session")
    assert _stats(store) == {}


def test_a_recall_write_never_waits_on_a_held_write_lock(store, monkeypatch):
    """Review round 1 M4: telemetry must not stall a read behind a writer.

    Asserted structurally (AGENTS.md "Prefer a structural invariant"), not as a
    wall-clock bound that a loaded host or a cold import inflates (QA Q9): the
    connection is opened with the short telemetry timeout, the bump is dropped
    rather than queued, and a same-turn re-read does not try the write again.
    The only timing left is a CATASTROPHE bound, far under the 5 s store timeout
    the old write waited on.
    """
    import sqlite3
    import time

    from local_operator import memory as memory_module

    fact = store.put("session", {"title": "Launch rule", "body": "Launch windows close at noon."})
    _tool_read(store, f"memory://session/{fact['id']}", "warm-up")  # imports, schema
    opened: list[float] = []
    real_connect = sqlite3.connect

    def counting_connect(path, *args, **kwargs):
        opened.append(kwargs.get("timeout", args[0] if args else 5.0))
        return real_connect(path, *args, **kwargs)

    monkeypatch.setattr(memory_module.sqlite3, "connect", counting_connect)
    holder = real_connect(store.path, timeout=0)
    holder.execute("BEGIN IMMEDIATE")
    try:
        started = time.perf_counter()
        _tool_read(store, f"memory://session/{fact['id']}", "locked-turn")
        locked = time.perf_counter() - started
        telemetry = [t for t in opened if t < 1]
        _tool_read(store, f"memory://session/{fact['id']}", "locked-turn")
        retried = [t for t in opened if t < 1]
    finally:
        holder.rollback()
        holder.close()
    assert telemetry == [MemoryStore._RECALL_BUSY_MS / 1000]  # one short-timeout attempt
    assert retried == telemetry  # same turn: marked seen, not retried
    assert _stats(store) == {fact["id"]: 1}  # warm-up counted; the locked bump dropped
    assert locked < 2.0, locked  # catastrophe bound only; the old write waited 5 s


def test_a_forgotten_fact_leaves_no_recall_row(store):
    fact = store.put("session", {"title": "Launch rule", "body": "Launch windows close at noon."})
    _tool_read(store, f"memory://session/{fact['id']}", "t")
    assert _stats(store) == {fact["id"]: 1}
    store.put("session", None, entity=fact["id"], expected_version=fact["version"], forget=True)
    assert _stats(store) == {}


def test_a_suggestion_candidate_is_not_a_recall_until_rendered(store):
    store.put("session", {"title": "Launch rule", "body": "Launch windows close at noon."})
    store.begin_recall_turn("turn-s")
    store.search("launch windows noon", "session", tokens=180, limit=3, suggest=True)
    assert _stats(store) == {}


def test_recall_stats_is_never_journalled(store):
    fact = store.put("session", {"title": "Launch rule", "body": "Launch windows close at noon."})
    with store.connection() as db:
        before = db.execute("SELECT count(*) FROM ops").fetchone()[0]
    _tool_read(store, f"memory://session/{fact['id']}", "t")
    assert _stats(store) == {fact["id"]: 1}
    with store.connection() as db:
        assert db.execute("SELECT count(*) FROM ops").fetchone()[0] == before
