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
        "[peer_message] the release window has an owner and we should hand over",
        "[session-state] the operator detached from the session some minutes ago",
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


# --- normalised dedup ------------------------------------------------------


def test_case_and_whitespace_variants_refresh_one_fact(store):
    persist(store, "a", extract("", "## Key Decisions\n- Keep the  Stable prefix"), "ok")
    persist(store, "b", extract("", "## Key Decisions\n- keep the stable   PREFIX"), "ok")
    assert len(_live(store, "decision")) == 1


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


def test_search_and_show_bump_once_per_turn(store):
    fact = store.put("session", {"title": "Launch rule", "body": "Launch windows close at noon."})
    store.begin_recall_turn("turn-1")
    store.search("launch windows", "session")
    store.show("session", fact["id"])
    store.search("launch windows", "session")
    assert _stats(store) == {fact["id"]: 1}
    store.begin_recall_turn("turn-2")
    store.show("session", fact["id"])
    assert _stats(store) == {fact["id"]: 2}
    summary = store.recall_summary()
    assert (summary["live"], summary["recalled"], summary["reads"]) == (1, 1, 2)


def test_a_suggestion_candidate_is_not_a_recall_until_rendered(store):
    store.put("session", {"title": "Launch rule", "body": "Launch windows close at noon."})
    store.begin_recall_turn("turn-s")
    store.search("launch windows noon", "session", tokens=180, limit=3, suggest=True)
    assert _stats(store) == {}


def test_recall_stats_is_never_journalled(store):
    fact = store.put("session", {"title": "Launch rule", "body": "Launch windows close at noon."})
    with store.connection() as db:
        before = db.execute("SELECT count(*) FROM ops").fetchone()[0]
    store.begin_recall_turn("t")
    store.show("session", fact["id"])
    with store.connection() as db:
        assert db.execute("SELECT count(*) FROM ops").fetchone()[0] == before
