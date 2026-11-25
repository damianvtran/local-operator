"""Suffix-only residency: a writer keeps only rows from the newest compaction's anchor.

WHY (fleet swap pressure, 2026-11): ``Transcript.__init__`` built a resident
``TranscriptEntry`` for every row, so a compaction never bounded a runtime's boot
memory even though large journals are only 0.2-6.6% post-compaction. These tests
pin the contract of the lazy load against an EAGER twin of the same directory
(``Transcript._force_eager``): every reader answers identically, ``entries()``
still returns every row (counted), and the two rewrite paths never write a
truncated history. Design brief: compaction-suffix-load.md (T1-T8, NC1-NC5).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.session import transcript as transcript_module
from local_operator.session.transcript import Transcript, TranscriptEntry


def _row(entry_id: str, entry_type: str, payload: dict[str, Any], ts: float = 1.0) -> str:
    return json.dumps(
        {"id": entry_id, "ts": ts, "type": entry_type, "payload": payload}, separators=(",", ":")
    )


def _user(i: str, text: str = "hello") -> str:
    return _row(i, "message", {"role": "user", "content": [{"type": "text", "text": text}]})


def _assistant(i: str, usage: int) -> str:
    return _row(
        i,
        "message",
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "ok " * 20}],
            "usage": {"input": usage, "output": 1, "context_tokens": usage},
        },
    )


def _tool(i: str, call: str, details: dict[str, Any], tool_name: str = "web_search") -> str:
    return _row(
        i,
        "message",
        {
            "role": "tool",
            "tool_call_id": call,
            "tool_name": tool_name,
            "content": [{"type": "text", "text": "result " * 30}],
            "provider_payload": {"details": details},
        },
    )


def _custom(i: str, custom_type: str, details: dict[str, Any]) -> str:
    return _row(i, "custom", {"custom_type": custom_type, "details": details})


def _compaction(i: str, first_kept: str | None, **extra: Any) -> str:
    payload: dict[str, Any] = {"summary": f"summary {i}", **extra}
    if first_kept is not None:
        payload["first_kept_entry_id"] = first_kept
    return _row(i, "compaction", payload)


def _money_prefix() -> list[str]:
    """A prefix holding every money shape, a prune, an admitted command and keep-set rows."""
    return [
        _user("u0"),
        _assistant("a0", 100),
        _tool("t0", "c0", {"search_cost": {"usd": 0.01}, "provider": "brave"}),
        _tool("t1", "c1", {"read_cost": {"usd": 0.02, "ledger_provider": "deepseek:read"}}),
        _tool(
            "t2",
            "c2",
            {"provider": "openai", "cost_usd": 0.05, "generation_id": "g1", "model": "m"},
            tool_name="generate_image",
        ),
        _row("p0", "prune", {"target": "t0", "notice": "[pruned]"}),
        _custom("cs0", "session_channel_spend.v1", {"kind": "start"}),
        _custom("cs1", "session_channel_spend.v1", {"record_id": "r1", "rev": 1, "amount": 3}),
        _custom("ss0", "session_state", {"blocks": {"a": "1"}}),
        _custom("sm0", "selected_model", {"selector": "x/y", "version": 2}),
        _row(
            "lost0",
            "message",
            {"role": "assistant", "content": [], "lost_usage_rows": 2},
        ),
        _row(
            "cmd0",
            "message",
            {
                "role": "user",
                "content": [{"type": "text", "text": "cmd"}],
                "provider_payload": {"producer_command_id": "command-1"},
            },
        ),
        _row(
            "inj0",
            "message",
            {"role": "user", "kind": "custom", "custom_type": "session_state", "content": []},
        ),
    ]


def _write(directory: Path, rows: list[str]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "transcript.jsonl"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _eager(directory: Path) -> Transcript:
    previous = Transcript._force_eager
    Transcript._force_eager = True
    try:
        return Transcript(directory, read_only=True)
    finally:
        Transcript._force_eager = previous


def _ids(messages: list[Any]) -> list[str]:
    return [str(getattr(message, "id", "")) for message in messages]


def _history_bytes(transcript: Transcript) -> list[str]:
    return [message.model_dump_json() for message in transcript.build_llm_history()]


def _filler(prefix: str, count: int) -> list[str]:
    rows: list[str] = []
    for k in range(count):
        rows.append(_user(f"{prefix}u{k}", "x" * 200))
        rows.append(_assistant(f"{prefix}a{k}", 10 + k))
    return rows


@pytest.fixture
def journal(tmp_path: Path) -> Path:
    """Money prefix, filler, one compaction anchored 4 rows BEFORE its own row."""
    rows = [*_money_prefix(), *_filler("f", 10)]
    rows += [_user("k0"), _assistant("k1", 500), _user("k2"), _assistant("k3", 600)]
    rows += [_compaction("c1", "k0", preserved_user_turns=[{"id": "inj0", "text": "x"}])]
    rows += [_user("after0"), _assistant("after1", 700)]
    directory = tmp_path / "sessions" / "s1"
    _write(directory, rows)
    return directory


# -- T1 equivalence -----------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    ["none", "one", "three", "anchor_is_next_row", "no_first_kept"],
)
def test_t1_lazy_history_equals_eager(tmp_path: Path, shape: str) -> None:
    rows = [*_money_prefix(), *_filler("a", 4)]
    if shape == "none":
        pass
    elif shape == "one":
        rows += [_user("k0"), _assistant("k1", 5), _compaction("c1", "k0")]
    elif shape == "three":
        rows += [_compaction("c1", "au1"), *_filler("b", 3)]
        rows += [_compaction("c2", "bu1"), *_filler("c", 3)]
        rows += [_user("k0"), _compaction("c3", "k0")]
    elif shape == "anchor_is_next_row":
        rows += [_compaction("c1", "n0"), _user("n0")]
    elif shape == "no_first_kept":
        rows += [_compaction("c1", None), _user("n0")]
    rows += [_user("tail"), _assistant("tail2", 9)]
    directory = tmp_path / shape
    _write(directory, rows)
    lazy = Transcript(directory)
    eager = _eager(directory)
    assert lazy.has_prefix is (shape not in ("none",))
    assert _history_bytes(lazy) == _history_bytes(eager)
    assert lazy.full_materialisations == 0
    lazy.close()


# -- T2 entries() contract ----------------------------------------------------


def test_t2_entries_returns_every_row_and_counts(journal: Path) -> None:
    lazy = Transcript(journal)
    eager = _eager(journal)
    assert lazy.has_prefix and lazy.prefix_rows > 0
    assert len(lazy.live_entries()) < len(eager.entries())
    assert [e.to_json() for e in lazy.entries()] == [e.to_json() for e in eager.entries()]
    assert lazy.full_materialisations == 1
    assert [e.id for e in lazy.iter_entries()] == [e.id for e in eager.entries()]
    assert lazy.full_materialisations == 2
    lazy.close()


# -- T3 projection parity -----------------------------------------------------


def _entry_id(entry: TranscriptEntry | None) -> str | None:
    """``entry.id``, or None when the slot is empty (parity compares both sides)."""
    return entry.id if entry is not None else None


def _projections(transcript: Transcript) -> dict[str, Any]:
    return {
        "all_usage_rows": transcript.all_usage_rows(),
        "lost_money_rows": transcript.lost_money_rows(),
        "search_spend_rows": transcript.search_spend_rows(),
        "channel_spend_rows": transcript.channel_spend_rows(),
        "channel_spend_tracked": transcript.channel_spend_tracked(),
        "channel_backfill_rows": transcript.channel_backfill_rows(),
        "pending_prunes": transcript.pending_prunes(),
        "entry_ids": transcript.entry_ids(),
        "admitted": transcript.has_admitted_command("command-1"),
        "has_entry_u0": transcript.has_entry("u0"),
        "latest_user": _entry_id(transcript.latest_user_entry()),
        "latest_custom": transcript.latest_custom("selected_model"),
        "latest_prune": _entry_id(transcript.latest_entry("prune")),
        "usages_since": transcript.usages_since_compaction(),
        "recent": [e.id for e in transcript.recent_entries(3)],
    }


def test_t3_projections_are_identical(journal: Path) -> None:
    lazy = Transcript(journal)
    eager = _eager(journal)
    assert lazy.has_prefix
    lazy_view, eager_view = _projections(lazy), _projections(eager)
    for key in eager_view:
        assert lazy_view[key] == eager_view[key], key
    # Non-vacuous: the prefix really carried each money shape.
    assert eager_view["all_usage_rows"] and eager_view["search_spend_rows"]
    assert eager_view["lost_money_rows"] and eager_view["channel_spend_tracked"]
    assert any(row["channel"] == "image" for row in eager_view["channel_backfill_rows"])
    assert lazy.full_materialisations == 0
    lazy.close()


def test_nc4_dropping_usage_from_skeleton_breaks_parity(
    journal: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: the parity test is sensitive to skeleton key drift."""
    original = transcript_module._skeleton

    def without_usage(entry: Any) -> Any:
        skeleton = original(entry)
        if skeleton is not None:
            skeleton.payload.pop("usage", None)
        return skeleton

    monkeypatch.setattr(transcript_module, "_skeleton", without_usage)
    lazy = Transcript(journal)
    eager = _eager(journal)
    assert lazy.all_usage_rows() != eager.all_usage_rows()
    lazy.close()


# -- T4 boot tripwire ---------------------------------------------------------


def _resume_args(resume: str) -> Any:
    import argparse

    return argparse.Namespace(
        hosting="test",
        model="test",
        agent_name=None,
        agent_id=None,
        yolo=True,
        train=False,
        resume=resume,
    )


@pytest.mark.asyncio
async def test_t4_resume_never_materialises_the_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.session.session import Session
    from local_operator.session_factory import create_session

    root = tmp_path / "config"
    home = tmp_path / "home"
    root.mkdir()
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("LOCAL_OPERATOR_NO_AIDA", "1")
    session_id = "5e55105e55aa"
    rows = [*_money_prefix(), *_filler("f", 20)]
    rows += [_user("k0"), _assistant("k1", 5), _compaction("c1", "k0"), _user("tail")]
    _write(root / "sessions" / session_id, rows)
    session = await create_session(
        _resume_args(session_id), ConfigManager(root), AgentRegistry(root)
    )
    # create_session returns the protocol; this test inspects the concrete
    # session's transcript, so narrow to it (a hosted session would fail here).
    assert isinstance(session, Session)
    try:
        transcript = session.transcript
        assert transcript.has_prefix
        assert transcript.full_materialisations == 0
        suffix_start = rows.index(_user("k0"))
        assert len(transcript.live_entries()) >= len(rows) - suffix_start
        assert transcript.prefix_rows == suffix_start
    finally:
        await session.dispose()
        session.transcript.close()
        # create_session opens the process-wide shared auth store on this
        # config root (auth.db + WAL/SHM); dispose() deliberately leaves it to
        # atexit, so close it here or the test leaks those fds (QA r1 Q3).
        from local_operator.config_watch import process_watcher
        from local_operator.providers.auth_store import close_shared_auth_stores

        close_shared_auth_stores()
        # The process-wide config watcher also outlives dispose() (its
        # kqueue + config-dir fd + wake pipes); stop it the way
        # test_runtime_locale does.
        await process_watcher(root).stop()


# -- T5 compact_file ----------------------------------------------------------


def _sha(path: Path, end: int) -> str:
    with path.open("rb") as handle:
        return hashlib.sha256(handle.read(end)).hexdigest()


async def _compactable(journal: Path) -> tuple[Transcript, int, str]:
    lazy = Transcript(journal)
    assert lazy.has_prefix and lazy._prefix is not None
    prefix_end = lazy._prefix.prefix_end
    before = _sha(journal / "transcript.jsonl", prefix_end)
    big = "z" * 4000
    await lazy.append_message(_message_tool("big", big))
    await lazy.append_prune("big", "[pruned]")
    # A suffix prune targeting a PREFIX row must survive the fold.
    await lazy.append_prune("t1", "[pruned read]")
    return lazy, prefix_end, before


def _message_tool(call: str, text: str) -> Any:
    from local_operator.harness.types import Message, TextContent, ToolResult

    message = Message.tool_result(
        ToolResult(tool_call_id=call, tool_name="web_read", content=[TextContent(text=text)])
    )
    message.id = call
    return message


@pytest.mark.asyncio
async def test_t5_compact_file_copies_the_prefix_verbatim(journal: Path) -> None:
    lazy, prefix_end, before = await _compactable(journal)
    reclaimed = await lazy.compact_file(min_reclaim_bytes=1)
    assert reclaimed > 0
    path = journal / "transcript.jsonl"
    assert _sha(path, prefix_end) == before
    assert lazy.full_materialisations == 0
    after_history = _history_bytes(lazy)
    lazy.close()
    reopened = Transcript(journal)
    eager = _eager(journal)
    assert _history_bytes(reopened) == _history_bytes(eager) == after_history
    assert eager.pending_prunes().get("t1") == "[pruned read]"
    assert [e.to_json() for e in reopened.entries()] == [e.to_json() for e in eager.entries()]
    reopened.close()


@pytest.mark.asyncio
async def test_nc3_rendering_only_resident_rows_truncates(
    journal: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: without the verbatim prefix copy the fold loses history."""
    lazy, prefix_end, before = await _compactable(journal)
    original = Transcript._replace_file

    def no_prefix(self: Transcript, payload: str, prefix_end: int = 0) -> None:
        original(self, payload, 0)

    monkeypatch.setattr(Transcript, "_replace_file", no_prefix)
    await lazy.compact_file(min_reclaim_bytes=1)
    lazy.close()
    assert _sha(journal / "transcript.jsonl", prefix_end) != before
    assert not _eager(journal).has_entry("u0")


# -- T6 fork ------------------------------------------------------------------


async def _fork_rows(journal: Path, transcript: Transcript, through: str | None) -> list[str]:
    fork_id, _omitted, _landed = await transcript.fork_snapshot(through_entry_id=through)
    child = journal.parent / fork_id / "transcript.jsonl"
    return [json.loads(line)["id"] for line in child.read_text().splitlines() if line]


@pytest.mark.parametrize("through", [None, "k2", "after0"])
@pytest.mark.asyncio
async def test_t6_fork_in_suffix_matches_eager(tmp_path: Path, through: str | None) -> None:
    rows = [*_filler("f", 6), _user("k0"), _assistant("k1", 5), _user("k2"), _assistant("k3", 6)]
    rows += [_compaction("c1", "k0"), _user("after0"), _assistant("after1", 7)]
    lazy_dir, eager_dir = tmp_path / "sessions" / "lz", tmp_path / "sessions" / "eg"
    _write(lazy_dir, rows)
    _write(eager_dir, rows)
    lazy = Transcript(lazy_dir)
    previous = Transcript._force_eager
    Transcript._force_eager = True
    try:
        eager = Transcript(eager_dir)
    finally:
        Transcript._force_eager = previous
    assert lazy.has_prefix and not eager.has_prefix
    assert await _fork_rows(lazy_dir, lazy, through) == await _fork_rows(eager_dir, eager, through)
    # k2 sits between the anchor (k0) and c1: replay slices c1 away and falls
    # back to row 0, so the prefix is read back; a cut after c1 never does.
    assert (lazy.full_materialisations > 0) is (through == "k2")
    lazy.close()
    eager.close()


@pytest.mark.asyncio
async def test_t6_fork_before_every_anchor_is_refused(journal: Path) -> None:
    from local_operator.session.errors import ForkRefused

    # Tool-free journal: the money prefix's unpaired tool rows would trip the
    # paired-prefix trim first (``unmatched_tool_result``, eager and lazy alike).
    directory = journal.parent / "s-anchor"
    rows = [*_filler("f", 10), _user("k0"), _assistant("k1", 500)]
    rows += [_compaction("c1", "k0"), _user("after0"), _assistant("after1", 700)]
    _write(directory, rows)
    eager = _eager(directory)
    with pytest.raises(ForkRefused) as eager_refused:
        await eager.fork_snapshot(through_entry_id="fu2")
    assert eager_refused.value.reason == "before_anchor"
    lazy = Transcript(directory)
    assert lazy.has_prefix
    with pytest.raises(ForkRefused) as refused:
        await lazy.fork_snapshot(through_entry_id="fu2")
    assert refused.value.reason == "before_anchor"
    # The cut sat in the dropped prefix, so the rule read it back once.
    assert lazy.full_materialisations >= 1
    lazy.close()


# -- T8 stale index / I5 ------------------------------------------------------


def test_t8_foreign_replace_is_detected_not_misread(journal: Path) -> None:
    lazy = Transcript(journal)
    path = journal / "transcript.jsonl"
    replacement = path.with_suffix(".other")
    replacement.write_text("\n".join(_filler("zz", 40)) + "\n")
    os.replace(replacement, path)
    with pytest.raises(OSError):
        lazy.entries()
    lazy.close()


def test_torn_last_row_stays_in_the_suffix(journal: Path) -> None:
    path = journal / "transcript.jsonl"
    with path.open("a") as handle:
        handle.write('{"id":"torn","ts":1.0,"type":"mess')
    lazy = Transcript(journal)
    eager = _eager(journal)
    assert lazy.has_prefix
    assert _history_bytes(lazy) == _history_bytes(eager)
    assert [e.id for e in lazy.entries()] == [e.id for e in eager.entries()]
    lazy.close()


@pytest.mark.asyncio
async def test_compaction_appended_while_running_keeps_answers(journal: Path) -> None:
    """I3: a mid-session compaction does not re-cut; the suffix just grows."""
    lazy = Transcript(journal)
    rows_before = lazy.prefix_rows
    await lazy.append_message(_message_tool("late", "late"))
    await lazy.append_compaction(summary="later", first_kept_entry_id="after0", tokens_before=1000)
    assert lazy.prefix_rows == rows_before
    eager = _eager(journal)
    assert _history_bytes(lazy) == _history_bytes(eager)
    assert lazy.all_usage_rows() == eager.all_usage_rows()
    lazy.close()


# -- negative controls --------------------------------------------------------


def test_nc1_unresolvable_anchor_loads_eagerly(tmp_path: Path) -> None:
    directory = tmp_path / "s"
    _write(directory, [*_filler("f", 4), _compaction("c1", "missing"), _user("t")])
    lazy = Transcript(directory)
    assert lazy.has_prefix is False
    assert _history_bytes(lazy) == _history_bytes(_eager(directory))
    lazy.close()


def test_nc2_second_writer_and_read_only_load_eagerly(journal: Path) -> None:
    owner = Transcript(journal)
    second = Transcript(journal)
    viewer = Transcript(journal, read_only=True)
    assert owner.has_prefix
    assert second.has_prefix is False
    assert viewer.has_prefix is False
    second.close()
    owner.close()


@pytest.mark.asyncio
async def test_nc5_vanished_journal_is_never_rebuilt_truncated(journal: Path) -> None:
    lazy = Transcript(journal)
    assert lazy.has_prefix
    (journal / "transcript.jsonl").unlink()
    with pytest.raises(OSError, match="refusing to rebuild"):
        await lazy.append_message(_message_tool("x", "x"))
    assert not (journal / "transcript.jsonl").exists()
    lazy.close()


@pytest.mark.asyncio
async def test_nc5_control_eager_transcript_still_rebuilds(journal: Path) -> None:
    """Control for NC5: the eager rebuild path is unchanged (not a blanket refusal)."""
    previous = Transcript._force_eager
    Transcript._force_eager = True
    try:
        eager = Transcript(journal)
    finally:
        Transcript._force_eager = previous
    (journal / "transcript.jsonl").unlink()
    await eager.append_message(_message_tool("x", "x"))
    assert (journal / "transcript.jsonl").exists()
    eager.close()


def test_cut_between_anchor_and_compaction_row_replays_like_eager(journal: Path) -> None:
    """Regression: a cut at k2 (after the anchor k0, before c1) slices c1 away,
    so eager replays from row 0; the lazy transcript must read the prefix back."""
    lazy = Transcript(journal)
    eager = _eager(journal)
    assert lazy.has_prefix
    for cut in ("k2", "k3", "after0"):
        assert _dump_history(lazy.build_llm_history(through_id=cut)) == _dump_history(
            eager.build_llm_history(through_id=cut)
        ), cut
    lazy.close()


def _dump_history(messages: list[Any]) -> list[Any]:
    return [message.model_dump() for message in messages]


# -- R1 display-page entry times for hoisted preserved turns -------------------


def _timed(rows: list[str]) -> list[str]:
    """``rows`` with a distinct ``ts`` each, so a missing join is visible."""
    out = []
    for k, row in enumerate(rows):
        data = json.loads(row)
        data["ts"] = 1000.0 + k
        out.append(json.dumps(data, separators=(",", ":")))
    return out


def test_display_entry_times_resolve_preserved_prefix_turns(tmp_path: Path) -> None:
    """A compaction's preserved turn sits BEFORE the anchor, so it is a prefix
    row; the page hoists it and its entry time must still join (R1)."""
    from local_operator.session.history_window import display_window

    rows = [*_money_prefix(), _user("pu", "keep me"), *_filler("f", 2)]
    rows += [_user("k0"), _assistant("k1", 5)]
    rows += [_compaction("c1", "k0", preserved_user_turns=[{"id": "pu", "text": "keep me"}])]
    rows += [_user("after0"), _assistant("after1", 7)]
    directory = tmp_path / "sessions" / "pt"
    _write(directory, _timed(rows))
    lazy = Transcript(directory)
    eager = _eager(directory)
    try:
        assert lazy.has_prefix

        def window(transcript: Transcript) -> Any:
            return display_window(
                transcript,
                conversation_id="pt",
                owner_epoch="e",
                through_id=transcript.live_entries()[-1].id,
            )

        lazy_page, eager_page = window(lazy), window(eager)
        assert "pu" in _ids(list(eager_page.messages))
        assert _ids(list(lazy_page.messages)) == _ids(list(eager_page.messages))
        assert eager_page.entry_times["pu"] == 1000.0 + rows.index(_user("pu", "keep me"))
        assert lazy_page.entry_times == eager_page.entry_times
        assert lazy.full_materialisations == 0
    finally:
        lazy.close()
        eager.close()


# -- R2 a read racing our own compact_file replace ------------------------------


def test_read_after_verbatim_replace_succeeds_before_identity_adopts(journal: Path) -> None:
    """``compact_file`` renames a verbatim-prefix copy over the journal in a
    worker thread and adopts the new inode only afterwards; an ``entries()``
    landing between must not raise (it made profile trust fail closed and
    cache "tainted"). Rows are still verified by id, so T8 keeps refusing a
    foreign replace (R2)."""
    lazy = Transcript(journal)
    eager = _eager(journal)
    try:
        assert lazy.has_prefix
        path = journal / "transcript.jsonl"
        copy = path.with_suffix(".compact")
        copy.write_bytes(path.read_bytes())
        before = os.stat(path).st_ino
        os.replace(copy, path)
        assert os.stat(path).st_ino != before
        assert [e.to_json() for e in lazy.entries()] == [e.to_json() for e in eager.entries()]
    finally:
        lazy.close()
        eager.close()
