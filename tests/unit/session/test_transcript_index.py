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
import threading
from pathlib import Path

import pytest

from local_operator.harness.types import Message, TextContent
from local_operator.session import transcript_index as ti
from local_operator.session.transcript import Transcript

SID = "aabbccddee01"


@pytest.fixture(autouse=True)
def _clean_module_state():
    """The module keeps process-wide loop state; no test may inherit another's."""
    ti._reset_for_tests()
    yield
    ti._reset_for_tests()


def journal_path(root: Path) -> Path:
    return root / "sessions" / SID / "transcript.jsonl"


def write_rows(root: Path, rows: list[dict]) -> None:
    path = journal_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def user(id_: str, ts: float, text: str = "hello") -> dict:
    return {
        "id": id_,
        "ts": ts,
        "type": "message",
        "payload": {"kind": "message", "role": "user", "content": [{"text": text}]},
    }


def assistant(id_: str, ts: float, text: str = "answer", tool_calls: bool = False) -> dict:
    payload: dict = {"kind": "message", "role": "assistant", "content": [{"text": text}]}
    if tool_calls:
        payload["tool_calls"] = [{"id": "c1", "name": "bash", "arguments": {}}]
    return {"id": id_, "ts": ts, "type": "message", "payload": payload}


def tool(id_: str, ts: float, text: str = "tool output") -> dict:
    return {
        "id": id_,
        "ts": ts,
        "type": "message",
        "payload": {"kind": "message", "role": "tool", "content": [{"text": text}]},
    }


def inject(
    id_: str, ts: float, custom_type: str = "hub_message", text: str = "peer says hi"
) -> dict:
    return {
        "id": id_,
        "ts": ts,
        "type": "message",
        "payload": {"kind": "custom", "custom_type": custom_type, "details": {"text": text}},
    }


def start(id_: str, ts: float, token: str) -> dict:
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
) -> dict:
    details: dict = {
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
    index = ti.refresh_index(tmp_path, SID)
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
    index = ti.refresh_index(tmp_path, SID)
    assert kinds(index) == [("user", "u1"), ("completion", "a2")]
    docs = {m.id: m for m in index.messages}
    assert docs["i1"].injected is True and docs["i1"].role == "user"
    assert docs["i1"].text == "peer says hi"
    assert docs["u1"].injected is False and docs["a1"].injected is False


def test_completion_needs_span_content(tmp_path):
    write_rows(tmp_path, [user("u1", 1.0), user("u2", 1.1), assistant("a1", 1.2)])
    index = ti.refresh_index(tmp_path, SID)
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
    index = ti.refresh_index(tmp_path, SID)
    assert outcomes(index) == [("u1", None), ("a1", None), ("u2", None), ("a2", "complete")]
    # Two completion ticks: one per turn — the mid-run steer closes turn 1.
    turns = [(c.kind, c.turn) for c in index.checkpoints]
    assert turns == [("user", 1), ("completion", 1), ("user", 2), ("completion", 2)]


def test_orphan_marker_is_ignored(tmp_path):
    write_rows(tmp_path, [user("u1", 1.0), assistant("a1", 1.1), marker("m1", 1.2, "ghost")])
    index = ti.refresh_index(tmp_path, SID)
    assert kinds(index) == [("user", "u1"), ("completion", "a1")]
    assert index.checkpoints[-1].outcome == "open"  # nothing settled the tail


def test_tail_open_until_a_marker_lands(tmp_path):
    write_rows(tmp_path, [start("s1", 1.0, "t1"), user("u1", 1.1), assistant("a1", 1.2)])
    index = ti.refresh_index(tmp_path, SID)
    assert index.checkpoints[-1].outcome == "open"
    write_rows(tmp_path, [marker("m1", 1.3, "t1")])
    index = ti.refresh_index(tmp_path, SID)
    assert index.checkpoints[-1].outcome == "complete"


def test_tail_settles_on_a_later_run_with_no_user_row(tmp_path):
    """A hub/wake run after the tail user row settles the conversation without
    carrying a kind (S3 note 2); the tail is NOT left open forever."""
    write_rows(tmp_path, [start("s1", 1.0, "t1"), user("u1", 1.1), assistant("a1", 1.2)])
    index = ti.refresh_index(tmp_path, SID)
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
    index = ti.refresh_index(tmp_path, SID)
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
    index = ti.refresh_index(tmp_path, SID)
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
    index = ti.refresh_index(tmp_path, SID)
    # The turn's OWN marker's kind survives; the completion still tracks the
    # span's last message row (the conversation's current end).
    assert index.checkpoints[-1].outcome == "error"
    assert [c.id for c in index.checkpoints if c.kind == "completion"] == ["a2"]


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
    index = ti.refresh_index(tmp_path, SID)
    assert index.checkpoints[-1].outcome is None


def test_malformed_rows_are_dropped_and_ordinals_still_count(tmp_path):
    write_rows(tmp_path, [user("u1", 1.0), assistant("a1", 1.1), assistant("a2", 1.2)])
    path = journal_path(tmp_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    write_rows(tmp_path, [assistant("a3", 1.3)])
    index = ti.refresh_index(tmp_path, SID)
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
    index = ti.refresh_index(tmp_path, SID)
    user_cp = next(c for c in index.checkpoints if c.kind == "user")
    assert len(user_cp.text) == ti.CHECKPOINT_TEXT_CAP
    doc = next(m for m in index.messages if m.role == "assistant")
    assert len(doc.text) == ti.DOC_TEXT_CAP


# ---------------------------------------------------------------------------
# Cache lifecycle: reuse, incremental append, rewrites, versions
# ---------------------------------------------------------------------------


def test_fresh_cache_is_reused_without_any_scan(tmp_path, monkeypatch):
    write_rows(tmp_path, [user("u1", 1.0), assistant("a1", 1.1)])
    first = ti.refresh_index(tmp_path, SID)
    assert first is not None

    def explode(*args, **kwargs):
        raise AssertionError("a fresh cache must not be rescanned")

    monkeypatch.setattr(ti, "_scan_full", explode)
    monkeypatch.setattr(ti, "_scan_incremental", explode)
    second = ti.refresh_index(tmp_path, SID)
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
    base = ti.refresh_index(tmp_path, SID)
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
    grown = ti.refresh_index(tmp_path, SID)
    assert calls == {"full": 1, "incremental": 1}

    # The reference: the same bytes scanned fresh in another root.
    other = tmp_path_factory.mktemp("reference")
    write_rows(
        other, [json.loads(line) for line in journal_path(tmp_path).read_text().splitlines()]
    )
    reference = ti.refresh_index(other, SID)

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
    ti.refresh_index(tmp_path, SID)
    write_rows(tmp_path, [assistant("a2", 1.3, "final answer")])
    index = ti.refresh_index(tmp_path, SID)
    assert calls["incremental"] == 1
    completion = next(c for c in index.checkpoints if c.kind == "completion")
    assert completion.id == "a2" and completion.text == "final answer"


def test_a_rewrite_that_shrinks_rescans(tmp_path, monkeypatch):
    write_rows(tmp_path, [user("u1", 1.0, "hello " * 500), assistant("a1", 1.1, "answer " * 500)])
    ti.refresh_index(tmp_path, SID)
    calls = _scanners(monkeypatch)
    # Simulate a compaction rewrite: same rows, smaller bytes, new mtime.
    rows = [json.loads(line) for line in journal_path(tmp_path).read_text().splitlines()]
    rows[0]["payload"]["content"] = [{"text": "hello"}]
    journal_path(tmp_path).write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows)
    )
    index = ti.refresh_index(tmp_path, SID)
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
    ti.refresh_index(tmp_path, SID)
    calls = _scanners(monkeypatch)
    await transcript.append_prune("x1", "[pruned]")
    assert await transcript.compact_file(min_reclaim_bytes=0) > 0
    index = ti.refresh_index(tmp_path, SID)
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
    first = ti.refresh_index(tmp_path, SID)
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
    index = ti.refresh_index(tmp_path, SID)
    assert calls == {"full": 1, "incremental": 0}
    assert [c.id for c in index.checkpoints] == ["u1", "a1"]


def test_version_bump_discards_scans_and_preserves_naming(tmp_path, monkeypatch):
    write_rows(tmp_path, [user("u1", 1.0), assistant("a1", 1.1)])
    ti.refresh_index(tmp_path, SID)
    # Put a naming item under the live turn key, then bump the cache version.
    ti.patch_naming(tmp_path, SID, {"u1": {"name": "Do the thing", "summary": "It was done."}})
    monkeypatch.setattr(ti, "TRANSCRIPT_INDEX_VERSION", 2)
    refreshed = ti.refresh_index(tmp_path, SID)
    assert refreshed is not None
    assert [c.id for c in refreshed.checkpoints] == ["u1", "a1"]
    assert refreshed.naming["items"]["u1"]["name"] == "Do the thing"
    # The stale-version cache was RESCANNED and rewritten under the new
    # version — not served as-is.
    assert json.loads(ti.index_path(tmp_path, SID).read_text())["version"] == 2
    monkeypatch.undo()
    # And an item whose turn key no longer exists is dropped at the next
    # RESCAN (a fresh cache is served as-is; the filter belongs to preservation).
    ti.patch_naming(tmp_path, SID, {"gone": {"name": "Stale"}})
    write_rows(tmp_path, [assistant("a2", 1.2)])  # grown journal forces a rescan
    again = ti.refresh_index(tmp_path, SID)
    assert set(again.naming["items"]) == {"u1"}


def test_corrupt_cache_is_rebuilt_and_missing_journal_is_none(tmp_path):
    assert ti.refresh_index(tmp_path, SID) is None  # no journal
    write_rows(tmp_path, [user("u1", 1.0), assistant("a1", 1.1)])
    ti.index_path(tmp_path, SID).parent.mkdir(parents=True, exist_ok=True)
    ti.index_path(tmp_path, SID).write_text("{corrupt")
    index = ti.refresh_index(tmp_path, SID)
    assert index is not None and len(index.checkpoints) == 2
    assert ti.read_index(tmp_path, SID) is not None


def test_torn_tail_is_not_scanned_until_completed(tmp_path):
    write_rows(tmp_path, [user("u1", 1.0), assistant("a1", 1.1)])
    path = journal_path(tmp_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            '{"id":"a2","ts":1.2,"type":"message","payload":{"kind":"message","role":"assistant"'
        )  # no newline
    index = ti.refresh_index(tmp_path, SID)
    assert index.coverage["complete"] is False
    assert [c.id for c in index.checkpoints] == ["u1", "a1"]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(',"content":[{"text":"late"}]}}\n')
    index = ti.refresh_index(tmp_path, SID)
    assert index.coverage["complete"] is True
    assert [c.id for c in index.checkpoints] == ["u1", "a2"]


def test_probe_reports_missing_ready_and_stale(tmp_path):
    assert ti.probe_index(tmp_path, SID).state == "missing"
    write_rows(tmp_path, [user("u1", 1.0), assistant("a1", 1.1)])
    assert ti.probe_index(tmp_path, SID).state == "stale"
    ti.refresh_index(tmp_path, SID)
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
