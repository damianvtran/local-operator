"""Recovering names for ledger rows that were recorded before anyone mirrored one.

The sweep's value is entirely in what it produces for a REAL store shape: most
unnamed sessions are subagents, and a subagent's opening message is a shared
role prompt. Labelling those naively is worse than leaving the hex ids, because
two identical rows cannot be told apart at all — so the collision case is the
one these tests pin hardest.
"""

from __future__ import annotations

import dataclasses
import json
import time

from local_operator.analytics.backfill import backfill_analytics_session_names
from local_operator.analytics.model import CallSnapshot
from local_operator.analytics.store import (
    SESSION_NAME_RANK_BACKFILL,
    SESSION_NAME_RANK_TITLE,
    AnalyticsStore,
)


def _snap(session_id: str, parent: str = "") -> CallSnapshot:
    snap = CallSnapshot(
        ts_ms=int(time.time() * 1000),
        session_id=session_id,
        provider="anthropic",
        model_id="claude",
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        context_tokens=15,
        component_chars={"conversation": 40},
        ok=True,
    )
    return dataclasses.replace(snap, parent_session_id=parent) if parent else snap


def _session(config_dir, session_id: str, opener: str, *, origin: str = "") -> None:
    """A session directory as ``resume.session_name`` reads one."""
    directory = config_dir / "sessions" / session_id
    directory.mkdir(parents=True)
    # The real transcript shape ``resume.session_name`` scans: a ``message``
    # entry whose payload carries the role and the content blocks.
    entry = {
        "id": session_id,
        "ts": time.time(),
        "type": "message",
        "payload": {"kind": "message", "role": "user", "content": [{"text": opener}]},
    }
    (directory / "transcript.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
    if origin:
        (directory / "origin.json").write_text(json.dumps({"origin": origin}), encoding="utf-8")


def _names(store: AnalyticsStore) -> dict[str, str]:
    return getattr(store.aggregate(), "session_names", {})


def test_backfill_names_a_session_from_its_transcript(tmp_path):
    store = AnalyticsStore(tmp_path / "analytics.db")
    store.record_batch([_snap("aaa")])
    _session(tmp_path, "aaa", "Fix the analytics session naming bug")

    assert backfill_analytics_session_names(tmp_path, store=store) == 1
    assert _names(store)["aaa"] == "Fix the analytics session naming bug"
    store.close()


def test_backfill_never_overwrites_a_real_title(tmp_path):
    store = AnalyticsStore(tmp_path / "analytics.db")
    store.record_batch([_snap("aaa")])
    store.upsert_session_name("aaa", "The real title", rank=SESSION_NAME_RANK_TITLE)
    _session(tmp_path, "aaa", "some opening message")

    # Already named, so it is not even on the worklist.
    assert backfill_analytics_session_names(tmp_path, store=store) == 0
    assert _names(store)["aaa"] == "The real title"
    store.close()


def test_backfill_is_idempotent(tmp_path):
    store = AnalyticsStore(tmp_path / "analytics.db")
    store.record_batch([_snap("aaa")])
    _session(tmp_path, "aaa", "Fix the thing")

    assert backfill_analytics_session_names(tmp_path, store=store) == 1
    # Second pass: the row now has a name, so there is no work left.
    assert backfill_analytics_session_names(tmp_path, store=store) == 0
    store.close()


def test_backfill_skips_a_session_whose_directory_was_pruned(tmp_path):
    """The ledger outlives transcripts, so most unnamed history is unrecoverable.

    It must be skipped silently rather than raising or minting a placeholder.
    """
    store = AnalyticsStore(tmp_path / "analytics.db")
    store.record_batch([_snap("gone")])
    (tmp_path / "sessions").mkdir()

    assert backfill_analytics_session_names(tmp_path, store=store) == 0
    # ``aggregate`` emits a key for every session it has rows for; the name is
    # empty, which is what makes the table fall back to the id.
    assert _names(store)["gone"] == ""
    store.close()


def test_backfill_never_names_a_session_the_ledger_has_never_seen(tmp_path):
    """The worklist is the LEDGER, so a free session never gets a row."""
    store = AnalyticsStore(tmp_path / "analytics.db")
    _session(tmp_path, "unbilled", "a session that made no provider calls")

    assert backfill_analytics_session_names(tmp_path, store=store) == 0
    assert _names(store) == {}
    store.close()


def test_subagent_rows_are_distinguished_by_role_and_parent_title(tmp_path):
    """The collision case, which is most of a real store.

    Three reviewers delegated by two different conversations share a byte-
    identical opening message. Labelling from the opener alone gives three
    identical rows; the fix is to say which conversation each one served.
    """
    store = AnalyticsStore(tmp_path / "analytics.db")
    store.record_batch(
        [
            _snap("parent1"),
            _snap("parent2"),
            _snap("kid1", parent="parent1"),
            _snap("kid2", parent="parent2"),
        ]
    )
    store.upsert_session_name("parent1", "Fix usage data not updating")
    store.upsert_session_name("parent2", "Toggleable sidebar for sessions")
    role_prompt = "[role: reviewer]\nYou are an INDEPENDENT reviewer. You did not write this code."
    _session(tmp_path, "kid1", role_prompt, origin="subagent")
    _session(tmp_path, "kid2", role_prompt, origin="subagent")

    assert backfill_analytics_session_names(tmp_path, store=store) == 2
    names = _names(store)
    assert names["kid1"] == "reviewer · Fix usage data not updating"
    assert names["kid2"] == "reviewer · Toggleable sidebar for sessions"
    assert names["kid1"] != names["kid2"]


def test_a_team_brief_names_the_role_from_its_sentence_not_the_team(tmp_path):
    """``[team: lopdev]`` names the TEAM; the job is in the sentence below it."""
    store = AnalyticsStore(tmp_path / "analytics.db")
    store.record_batch([_snap("parent1"), _snap("kid", parent="parent1")])
    store.upsert_session_name("parent1", "Release the analytics fix")
    _session(
        tmp_path,
        "kid",
        "[team: lopdev]\n\nYou are qa-tester on this team. The manager is manager.",
        origin="subagent",
    )

    backfill_analytics_session_names(tmp_path, store=store)
    assert _names(store)["kid"] == "qa-tester · Release the analytics fix"
    store.close()


def test_a_subagent_whose_parent_has_no_title_keeps_its_role(tmp_path):
    store = AnalyticsStore(tmp_path / "analytics.db")
    store.record_batch([_snap("anon"), _snap("kid", parent="anon")])
    _session(tmp_path, "kid", "[role: coder] You implement one bounded slice.", origin="subagent")

    backfill_analytics_session_names(tmp_path, store=store)
    assert _names(store)["kid"] == "coder · anon"
    store.close()


def test_an_ordinary_session_keeps_its_own_opener(tmp_path):
    """Only a role prompt is rewritten; a human's own words are the name."""
    store = AnalyticsStore(tmp_path / "analytics.db")
    store.record_batch([_snap("parent1"), _snap("kid", parent="parent1")])
    store.upsert_session_name("parent1", "Some parent conversation")
    _session(tmp_path, "kid", "Please look at why the build is slow")

    backfill_analytics_session_names(tmp_path, store=store)
    assert _names(store)["kid"] == "subagent · Some parent conversation"
    store.close()


def test_backfilled_names_carry_backfill_rank_so_a_title_still_wins(tmp_path):
    store = AnalyticsStore(tmp_path / "analytics.db")
    store.record_batch([_snap("aaa")])
    _session(tmp_path, "aaa", "an opener")
    backfill_analytics_session_names(tmp_path, store=store)

    conn = store._connect()
    assert conn is not None
    row = conn.execute("SELECT rank FROM session_names WHERE session_id='aaa'").fetchone()
    assert row[0] == SESSION_NAME_RANK_BACKFILL

    # ...so the session's real title, whenever it lands, replaces it.
    store.upsert_session_name("aaa", "The generated title")
    assert _names(store)["aaa"] == "The generated title"
    store.close()


def test_backfill_honours_its_write_limit(tmp_path):
    store = AnalyticsStore(tmp_path / "analytics.db")
    for index in range(5):
        store.record_batch([_snap(f"s{index}")])
        _session(tmp_path, f"s{index}", f"Opener number {index}")

    assert backfill_analytics_session_names(tmp_path, store=store, limit=2) == 2
    # The rest are picked up by a later pass — no session is stranded.
    assert backfill_analytics_session_names(tmp_path, store=store, limit=10) == 3
    store.close()


def test_backfill_survives_an_unreadable_session_directory(tmp_path):
    """A corrupt transcript costs one row, never the sweep."""
    store = AnalyticsStore(tmp_path / "analytics.db")
    store.record_batch([_snap("bad"), _snap("good")])
    directory = tmp_path / "sessions" / "bad"
    directory.mkdir(parents=True)
    (directory / "transcript.jsonl").write_bytes(b"\x00not json at all")
    _session(tmp_path, "good", "A readable opener")

    assert backfill_analytics_session_names(tmp_path, store=store) == 1
    assert _names(store)["good"] == "A readable opener"
    store.close()
