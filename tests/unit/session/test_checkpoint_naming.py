"""Checkpoint naming: prompt, parser, cache keying and the warm op (D2/D9).

These pin everything a provider call is allowed to decide, and — as important —
everything it is not: the parser REJECTS over-long names instead of truncating
them, the digest is a pure function of the two stored strings so the cache's
hash comparison means the same thing in every process, a failure persists a
marker whose cooldown both warm and the manifest read from the same file, and
none of the failure paths raise into the caller.

Everything here runs against a seeded cache on a ``tmp_path``; no provider and
no journal are involved — ``complete_fn`` is a recorder.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.session import checkpoint_naming as cn
from local_operator.session import transcript_index as ti

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _clean_module_state():
    cn._reset_for_tests()
    yield
    cn._reset_for_tests()


def _seed(
    cfg: Path,
    sid: str,
    turns: list[tuple[str, str, str]],
    *,
    items: dict[str, Any] | None = None,
) -> None:
    """Write a transcript-index cache for ``turns`` — (user, answer, outcome)."""
    checkpoints: list[ti.Checkpoint] = []
    messages: list[ti.MessageDoc] = []
    seq = 0
    for turn, (user_text, answer_text, outcome) in enumerate(turns, start=1):
        uid, aid = f"u{turn}", f"a{turn}"
        checkpoints.append(
            ti.Checkpoint(
                id=uid,
                kind=ti.KIND_USER,
                turn=turn,
                ts=float(turn),
                seq=seq,
                text=user_text,
                outcome=None,
            )
        )
        messages.append(
            ti.MessageDoc(
                id=uid,
                ts=float(turn),
                role="user",
                text=user_text,
                injected=False,
                seq=seq,
            )
        )
        seq += 1
        checkpoints.append(
            ti.Checkpoint(
                id=aid,
                kind=ti.KIND_COMPLETION,
                turn=turn,
                ts=float(turn) + 0.5,
                seq=seq,
                text=answer_text,
                outcome=outcome,
            )
        )
        messages.append(
            ti.MessageDoc(
                id=aid,
                ts=float(turn) + 0.5,
                role="assistant",
                text=answer_text,
                injected=False,
                seq=seq,
            )
        )
        seq += 1
    index = ti.TranscriptIndex(
        checkpoints=checkpoints,
        messages=messages,
        sig={"size": seq, "mtime": 1.0, "last_id": checkpoints[-1].id},
        coverage={
            "first_id": checkpoints[0].id,
            "last_id": checkpoints[-1].id,
            "complete": True,
        },
        naming={
            "prompt_version": ti.NAMING_PROMPT_VERSION,
            "items": dict(items or {}),
        },
        scan=ti.ScanState(rows=seq, offset=0, window_offset=0, window_rows=0),
    )
    ti.write_index(cfg, sid, index)


def _items(cfg: Path, sid: str) -> dict[str, Any]:
    index = ti.read_index(cfg, sid)
    assert index is not None
    return dict(index.naming.get("items") or {})


async def _settle(cfg: Path, sid: str) -> None:
    """Wait for the module's background tasks for one session.

    The tasks are registered synchronously by ``warm_checkpoints`` before it
    returns, so this is a join, not a poll — the same reason the suite prefers
    event waits: there is no elapsed-time bound anywhere in it.
    """
    state = cn._STATE.get((str(cfg), sid))
    if state is not None and state.active:
        await asyncio.gather(*list(state.active.values()), return_exceptions=True)


class _Recorder:
    """A complete_fn double: records (system, prompt) and replays ``replies``."""

    def __init__(self, *replies: Any) -> None:
        self.calls: list[tuple[str, str]] = []
        self._replies = replies

    async def __call__(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        reply = self._replies[min(len(self.calls) - 1, len(self._replies) - 1)]
        if isinstance(reply, BaseException):
            raise reply
        return reply


def _ok(name: str = "Fix flaky registration test", summary: str = "Pinned the race.") -> str:
    return f"<name>{name}</name><summary>{summary}</summary>"


# ---------------------------------------------------------------------------
# the parser
# ---------------------------------------------------------------------------


async def test_parse_reads_name_and_summary() -> None:
    parsed = cn.parse_checkpoint(
        "<name>Fix flaky registration test</name>"
        "<summary>Pinned the race with a signal.</summary>"
    )
    assert parsed is not None
    assert parsed.name == "Fix flaky registration test"
    assert parsed.summary == "Pinned the race with a signal."


async def test_parse_sentinel_declines() -> None:
    assert cn.parse_checkpoint("<name/>") is None
    assert cn.parse_checkpoint("<name />") is None
    assert cn.parse_checkpoint("  <name/>\n") is None


async def test_parse_rejects_over_cap_names_never_truncates() -> None:
    # Nine words: over MAX_NAME_WORDS, REJECTED (naming.py's rule for titles).
    assert cn.parse_checkpoint("<name>one two three four five six seven eight nine</name>") is None
    # Over MAX_NAME_CHARS even at few words.
    assert cn.parse_checkpoint("<name>" + "x" * 81 + "</name>") is None
    # And the fit cases stay accepted.
    assert cn.parse_checkpoint("<name>six words make a fine name ok</name>") is not None


async def test_parse_last_visible_tag_wins() -> None:
    parsed = cn.parse_checkpoint("<name>draft</name> noise <name>Final Name</name>")
    assert parsed is not None and parsed.name == "Final Name"


async def test_parse_cuts_summary_but_keeps_the_name() -> None:
    long_summary = "word " * 60  # 300 chars, well over the 160 cap
    parsed = cn.parse_checkpoint(f"<name>Kept name</name><summary>{long_summary}</summary>")
    assert parsed is not None and parsed.name == "Kept name"
    assert len(parsed.summary) <= cn.SUMMARY_MAX_CHARS
    assert parsed.summary.endswith("…")


async def test_parse_missing_summary_is_empty_not_a_rejection() -> None:
    parsed = cn.parse_checkpoint("<name>Name only</name>")
    assert parsed is not None and parsed.summary == ""


async def test_parse_untagged_reply_yields_name_only() -> None:
    parsed = cn.parse_checkpoint("Fix the flaky registration test")
    assert parsed is not None
    assert parsed.name == "Fix the flaky registration test"
    assert parsed.summary == ""


async def test_parse_strips_thinking_envelopes() -> None:
    leaked = "<think>the user wants a name...</think><name>Real Name</name><summary>Done.</summary>"
    parsed = cn.parse_checkpoint(leaked)
    assert parsed is not None and parsed.name == "Real Name"
    # An unclosed envelope discards the reply: the rest may still be thinking.
    assert cn.parse_checkpoint("<think>still going<name>Not This</name>") is None
    # A bare "Thinking process:" preamble is not a name either.
    assert cn.parse_checkpoint("Thinking process:\nFix the parser") is None


async def test_parse_unwraps_a_fenced_json_reply() -> None:
    """The fenced arm is separate from plain JSON (agent review round 1, NIT-2)."""
    parsed = cn.parse_checkpoint('```json\n{"name": "Fix flaky registration test"}\n```')
    assert parsed is not None
    assert parsed.name == "Fix flaky registration test"


async def test_parse_unwraps_json_name() -> None:
    parsed = cn.parse_checkpoint('{"name": "JSON name here", "summary": "ignored"}')
    assert parsed is not None and parsed.name == "JSON name here"


async def test_parse_empty_inputs_are_none() -> None:
    assert cn.parse_checkpoint("") is None
    assert cn.parse_checkpoint("   \n ") is None
    assert cn.parse_checkpoint("<name></name>") is None


# ---------------------------------------------------------------------------
# the digest
# ---------------------------------------------------------------------------


async def test_digest_uses_user_text_and_final_answer(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    _seed(cfg, "s1", [("Ping the endpoint", "It answers 200 now.", "complete")])
    index = ti.read_index(cfg, "s1")
    assert index is not None
    digest = cn.build_turn_digest(index, 1)
    assert digest is not None
    assert "Ping the endpoint" in digest
    assert "It answers 200 now." in digest
    assert len(digest) <= cn.DIGEST_MAX_CHARS
    # Pure function of the stored strings: the hash means the same thing in
    # every process and across restarts.
    again = cn.build_turn_digest(index, 1)
    assert again is not None
    assert digest == again
    assert cn._digest_hash(digest) == cn._digest_hash(again)


async def test_digest_missing_pieces_are_none(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    _seed(cfg, "s1", [("Only this", "and this", "complete")])
    index = ti.read_index(cfg, "s1")
    assert index is not None
    assert cn.build_turn_digest(index, 99) is None


async def test_digest_empty_texts_are_not_spent_on(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    _seed(cfg, "s1", [("", "", "complete")])
    index = ti.read_index(cfg, "s1")
    assert index is not None
    assert cn.build_turn_digest(index, 1) is None


# ---------------------------------------------------------------------------
# the state derive (the manifest's single source of truth)
# ---------------------------------------------------------------------------


async def test_naming_state_covers_ready_pending_unavailable() -> None:
    now = time.time()
    assert cn.naming_state(None) == "pending"
    assert cn.naming_state({"name": "A name", "failed_ts": now}) == "ready"
    assert cn.naming_state({"state": "unavailable", "failed_ts": now}) == "unavailable"
    assert (
        cn.naming_state(
            {"state": "unavailable", "failed_ts": now - cn.NAMING_UNAVAILABLE_COOLDOWN_S - 1}
        )
        == "pending"
    )
    # The window is measured from the passed moment, so a caller reading a
    # historical stamp gets a deterministic answer.
    assert cn.naming_state({"failed_ts": now}, now=now + 1) == "unavailable"
    assert cn.naming_state({}, now=now) == "pending"


async def test_manifest_derives_all_three_states_from_the_section(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    now = time.time()
    _seed(
        cfg,
        "s1",
        [
            ("One", "First answer", "complete"),
            ("Two", "Second answer", "complete"),
            ("Three", "Third answer", "complete"),
        ],
        items={
            "u1": {"name": "Ready name", "summary": "Here.", "text_hash": "h1"},
            "u2": {"state": "unavailable", "failed_ts": now, "text_hash": "h2"},
        },
    )
    index = ti.read_index(cfg, "s1")
    assert index is not None
    view = ti._manifest_state("s1", "ready", index, None)
    states = {
        entry["turn"]: entry["naming"]["state"]
        for entry in view["checkpoints"]
        if entry["kind"] == ti.KIND_COMPLETION
    }
    assert states == {1: "ready", 2: "unavailable", 3: "pending"}
    ready = next(e for e in view["checkpoints"] if e["id"] == "a1")
    assert ready["naming"] == {"state": "ready", "name": "Ready name", "summary": "Here."}


# ---------------------------------------------------------------------------
# warm
# ---------------------------------------------------------------------------


async def test_warm_default_selects_recent_missing_and_generates(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    _seed(cfg, "s1", [("One", "First answer", "complete"), ("Two", "Second answer", "complete")])
    prompts: list[str] = []
    systems: list[str] = []

    async def naming(system: str, prompt: str) -> str:
        systems.append(system)
        prompts.append(prompt)
        # Keyed off the digest, not call order: the batch is scheduled newest
        # first, and asserting on that order would pin the scheduler rather
        # than the naming.
        return _ok("Second name" if "User: Two" in prompt else "First name")

    accepted = await cn.warm_checkpoints(cfg, "s1", complete_fn=naming)
    assert accepted == {"accepted": ["a2", "a1"], "pending": ["a2", "a1"]}
    await _settle(cfg, "s1")
    items = _items(cfg, "s1")
    assert items["u1"]["name"] == "First name"
    assert items["u2"]["name"] == "Second name"
    assert len(prompts) == 2
    assert set(systems) == {cn.CHECKPOINT_NAME_SYSTEM_PROMPT}
    assert prompts[0].startswith("User: ")  # the labelled digest, not raw text

    # Everything is named and current: a rail-open warm spends nothing.
    again = await cn.warm_checkpoints(cfg, "s1", complete_fn=naming)
    assert again == {"accepted": [], "pending": []}
    assert len(prompts) == 2


async def test_warm_limit_bounds_the_default_selection(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    _seed(
        cfg,
        "s1",
        [("One", "A1", "complete"), ("Two", "A2", "complete"), ("Three", "A3", "complete")],
    )
    rec = _Recorder(_ok(), _ok(), _ok())
    accepted = await cn.warm_checkpoints(cfg, "s1", limit=2, complete_fn=rec)
    assert accepted["accepted"] == ["a3", "a2"]
    await _settle(cfg, "s1")
    assert len(rec.calls) == 2


async def test_warm_default_skips_the_open_tail(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    _seed(cfg, "s1", [("One", "A1", "complete"), ("Two", "A2 (live)", "open")])
    rec = _Recorder(_ok())
    accepted = await cn.warm_checkpoints(cfg, "s1", complete_fn=rec)
    assert accepted == {"accepted": ["a1"], "pending": ["a1"]}
    await _settle(cfg, "s1")
    assert len(rec.calls) == 1


async def test_warm_explicit_open_tail_id_is_accepted(tmp_path: Path) -> None:
    # A hover names the live tail's turn too: the user asked about THAT tick,
    # and the hash comparison regenerates when the turn settles and moves.
    cfg = tmp_path / "cfg"
    _seed(cfg, "s1", [("One", "A1", "complete"), ("Two", "A2 (live)", "open")])
    rec = _Recorder(_ok("Live name"))
    accepted = await cn.warm_checkpoints(cfg, "s1", ids=["a2"], complete_fn=rec)
    assert accepted == {"accepted": ["a2"], "pending": ["a2"]}
    await _settle(cfg, "s1")
    assert _items(cfg, "s1")["u2"]["name"] == "Live name"


async def test_warm_user_id_echoes_and_names_its_turns_completion(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    _seed(cfg, "s1", [("One", "A1", "complete")])
    rec = _Recorder(_ok("Named via user tick"))
    accepted = await cn.warm_checkpoints(cfg, "s1", ids=["u1"], complete_fn=rec)
    assert accepted == {"accepted": ["u1"], "pending": ["u1"]}
    await _settle(cfg, "s1")
    items = _items(cfg, "s1")
    assert items["u1"]["name"] == "Named via user tick"


async def test_warm_unknown_and_duplicate_ids_are_dropped_not_refused(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    _seed(cfg, "s1", [("One", "A1", "complete")])
    rec = _Recorder(_ok())
    # Unknown id, then the same turn twice by its two ids: one target, first echo.
    accepted = await cn.warm_checkpoints(cfg, "s1", ids=["zzz", "a1", "u1"], complete_fn=rec)
    assert accepted == {"accepted": ["a1"], "pending": ["a1"]}
    await _settle(cfg, "s1")
    assert len(rec.calls) == 1


async def test_warm_empty_id_list_is_an_empty_receipt(tmp_path: Path) -> None:
    # An explicit empty list means "these zero ticks" — the default selection
    # is what sending NO ids means. The distinction is the caller's.
    cfg = tmp_path / "cfg"
    _seed(cfg, "s1", [("One", "A1", "complete")])
    rec = _Recorder(_ok())
    accepted = await cn.warm_checkpoints(cfg, "s1", ids=[], complete_fn=rec)
    assert accepted == {"accepted": [], "pending": []}
    await _settle(cfg, "s1")
    assert rec.calls == []


async def test_warm_caps_ids_at_the_module_bound(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    turns = [(f"Q{i}", f"A{i}", "complete") for i in range(1, 19)]
    _seed(cfg, "s1", turns)
    rec = _Recorder(_ok())
    ids = [f"a{i}" for i in range(1, 19)]
    accepted = await cn.warm_checkpoints(cfg, "s1", ids=ids, complete_fn=rec)
    assert len(accepted["accepted"]) == cn.MAX_WARM_IDS


async def test_warm_ready_current_name_is_a_no_op(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    _seed(cfg, "s1", [("One", "A1", "complete")])
    rec = _Recorder(_ok("Kept"), _ok("Unused"))
    first = await cn.warm_checkpoints(cfg, "s1", ids=["a1"], complete_fn=rec)
    assert first["pending"] == ["a1"]
    await _settle(cfg, "s1")
    second = await cn.warm_checkpoints(cfg, "s1", ids=["a1"], complete_fn=rec)
    assert second == {"accepted": ["a1"], "pending": []}
    assert len(rec.calls) == 1
    assert _items(cfg, "s1")["u1"]["name"] == "Kept"


async def test_warm_regenerates_when_the_turn_moved(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    _seed(cfg, "s1", [("One", "A1", "complete")])
    rec = _Recorder(_ok("First name"), _ok("Second name"))
    await cn.warm_checkpoints(cfg, "s1", ids=["a1"], complete_fn=rec)
    await _settle(cfg, "s1")
    before = _items(cfg, "s1")["u1"]

    # The turn grew: same key, different answer -> different digest hash.
    _seed(cfg, "s1", [("One", "A1 extended with the follow-up.", "complete")], items={"u1": before})
    accepted = await cn.warm_checkpoints(cfg, "s1", ids=["a1"], complete_fn=rec)
    assert accepted == {"accepted": ["a1"], "pending": ["a1"]}
    await _settle(cfg, "s1")
    after = _items(cfg, "s1")["u1"]
    assert after["name"] == "Second name"
    assert after["text_hash"] != before["text_hash"]
    assert len(rec.calls) == 2


async def test_a_failed_regeneration_keeps_the_last_good_name(tmp_path: Path) -> None:
    """A refresh's failure must not degrade what it was refreshing (MINOR-1).

    The kept pair is served by the manifest and guarded from re-spend by the
    same cooldown a never-named turn gets; once the window passes, the stale
    name is regenerated over — the same stale arm as ever, just not erasing
    the pair while it waits.
    """
    cfg = tmp_path / "cfg"
    _seed(cfg, "s1", [("One", "A1", "complete")])
    rec = _Recorder(_ok("Old name", "Old summary"))
    await cn.warm_checkpoints(cfg, "s1", ids=["a1"], complete_fn=rec)
    await _settle(cfg, "s1")
    before = _items(cfg, "s1")["u1"]

    # The turn grew, and the regeneration attempt FAILS.
    _seed(cfg, "s1", [("One", "A1 extended with the follow-up.", "complete")], items={"u1": before})
    failing = _Recorder(RuntimeError("provider exploded"))
    accepted = await cn.warm_checkpoints(cfg, "s1", ids=["a1"], complete_fn=failing)
    assert accepted == {"accepted": ["a1"], "pending": ["a1"]}
    await _settle(cfg, "s1")
    kept = _items(cfg, "s1")["u1"]
    assert kept["name"] == "Old name" and kept["summary"] == "Old summary"
    assert isinstance(kept["failed_ts"], float)
    assert kept["text_hash"] == before["text_hash"], "the pair still describes the old content"

    # The manifest keeps serving the pair: a present name reads ready.
    index = ti.read_index(cfg, "s1")
    assert index is not None
    view = ti._manifest_state("s1", "ready", index, None)
    ready = next(e for e in view["checkpoints"] if e["id"] == "a1")
    assert ready["naming"] == {"state": "ready", "name": "Old name", "summary": "Old summary"}

    # Inside the cooldown: accepted, never pending, not one more call.
    again = await cn.warm_checkpoints(cfg, "s1", ids=["a1"], complete_fn=failing)
    assert again == {"accepted": ["a1"], "pending": []}
    await _settle(cfg, "s1")
    assert len(failing.calls) == 1

    # Expired: the stale name is regenerated and the success replaces the pair.
    kept = _items(cfg, "s1")["u1"]
    kept["failed_ts"] = time.time() - cn.NAMING_UNAVAILABLE_COOLDOWN_S - 5
    ti.patch_naming(cfg, "s1", {"u1": kept})
    recovering = _Recorder(_ok("New name", "New summary"))
    third = await cn.warm_checkpoints(cfg, "s1", ids=["a1"], complete_fn=recovering)
    assert third == {"accepted": ["a1"], "pending": ["a1"]}
    await _settle(cfg, "s1")
    final = _items(cfg, "s1")["u1"]
    assert final["name"] == "New name" and "failed_ts" not in final


async def test_warm_failure_persists_unavailable_and_cooldown_skips(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    _seed(cfg, "s1", [("One", "A1", "complete")])
    failing = _Recorder(RuntimeError("provider exploded"))
    first = await cn.warm_checkpoints(cfg, "s1", ids=["a1"], complete_fn=failing)
    assert first == {"accepted": ["a1"], "pending": ["a1"]}
    await _settle(cfg, "s1")
    marker = _items(cfg, "s1")["u1"]
    assert marker["state"] == "unavailable"
    assert isinstance(marker["failed_ts"], float)
    assert "name" not in marker

    # Inside the cooldown: accepted, never pending, and not one more call.
    second = await cn.warm_checkpoints(cfg, "s1", ids=["a1"], complete_fn=failing)
    assert second == {"accepted": ["a1"], "pending": []}
    await _settle(cfg, "s1")
    assert len(failing.calls) == 1

    # Expired: the same turn is retried and the success replaces the marker.
    marker["failed_ts"] = time.time() - cn.NAMING_UNAVAILABLE_COOLDOWN_S - 5
    ti.patch_naming(cfg, "s1", {"u1": marker})
    recovering = _Recorder(_ok("Recovered name"))
    third = await cn.warm_checkpoints(cfg, "s1", ids=["a1"], complete_fn=recovering)
    assert third == {"accepted": ["a1"], "pending": ["a1"]}
    await _settle(cfg, "s1")
    assert _items(cfg, "s1")["u1"]["name"] == "Recovered name"


async def test_warm_timeout_writes_the_marker(tmp_path: Path, monkeypatch) -> None:
    """The ``wait_for`` bound itself is exercised, not just the failure shape.

    The marker is shared with the provider-error arm, so the risk this pins is
    the wiring: the timeout must still land a persisted item (agent review
    round 1, MINOR-2).
    """
    cfg = tmp_path / "cfg"
    _seed(cfg, "s1", [("One", "A1", "complete")])
    monkeypatch.setattr(cn, "CHECKPOINT_NAME_TIMEOUT_S", 0.05)

    async def hanging(system: str, prompt: str) -> str:
        await asyncio.sleep(30)
        return _ok("Never arrives")

    accepted = await cn.warm_checkpoints(cfg, "s1", ids=["a1"], complete_fn=hanging)
    assert accepted == {"accepted": ["a1"], "pending": ["a1"]}
    await _settle(cfg, "s1")
    marker = _items(cfg, "s1")["u1"]
    assert marker["state"] == "unavailable"
    assert isinstance(marker["failed_ts"], float)
    assert "name" not in marker


async def test_warm_decline_takes_the_same_unavailable_state(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    _seed(cfg, "s1", [("ok", "thanks", "complete")])
    declining = _Recorder("<name/>", "<name>Should not be asked</name>")
    await cn.warm_checkpoints(cfg, "s1", ids=["a1"], complete_fn=declining)
    await _settle(cfg, "s1")
    assert _items(cfg, "s1")["u1"]["state"] == "unavailable"
    again = await cn.warm_checkpoints(cfg, "s1", ids=["a1"], complete_fn=declining)
    assert again == {"accepted": ["a1"], "pending": []}
    await _settle(cfg, "s1")
    assert len(declining.calls) == 1


async def test_warm_in_flight_is_idempotent(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    _seed(cfg, "s1", [("One", "A1", "complete")])
    release = asyncio.Event()
    calls: list[str] = []

    async def held(system: str, prompt: str) -> str:
        calls.append(prompt)
        await release.wait()
        return _ok("Held")

    first = await cn.warm_checkpoints(cfg, "s1", complete_fn=held)
    assert first == {"accepted": ["a1"], "pending": ["a1"]}
    await asyncio.sleep(0)  # let the task start and block on the event
    second = await cn.warm_checkpoints(cfg, "s1", complete_fn=held)
    assert second == {"accepted": ["a1"], "pending": ["a1"]}
    assert len(calls) == 1, "a second warm re-spent on a turn already in flight"
    release.set()
    await _settle(cfg, "s1")
    assert _items(cfg, "s1")["u1"]["name"] == "Held"


async def test_warm_runs_one_call_at_a_time_per_session(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg"
    _seed(cfg, "s1", [("One", "A1", "complete"), ("Two", "A2", "complete")])
    in_flight = 0
    peak = 0

    async def overlapping(system: str, prompt: str) -> str:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        # Widen the overlap window; the ASSERTION is the structural peak below,
        # not this duration.
        await asyncio.sleep(0.02)
        in_flight -= 1
        return _ok()

    await cn.warm_checkpoints(cfg, "s1", complete_fn=overlapping)
    await _settle(cfg, "s1")
    assert peak == 1, "two naming errands for one session ran concurrently"


async def test_warm_without_an_index_is_an_empty_receipt(tmp_path: Path) -> None:
    # The cache is the manifest route's to build; a gesture must not scan.
    rec = _Recorder(_ok())
    accepted = await cn.warm_checkpoints(tmp_path / "cfg", "missing", complete_fn=rec)
    assert accepted == {"accepted": [], "pending": []}
    assert rec.calls == []
