"""``asks/dismiss.py`` — the COLD dismissal, the one ask op with no runtime.

Why this file exists (design ``docs/design/ask-nonblocking.md`` §2.2/§14, fork A).
The phone's dismiss used to reach a stopped conversation by ENGAGING it, which
boots a runtime, spends a turn and — before fork A — was the only thing that ever
delivered the deadline notice the dismissal then cancelled. The cold door writes
the same ``dismissed`` row a runtime writes, rewrites the derived index, and
DISARMS the deadline wake row so the supervisor cannot boot a runtime for a
dismissed ask. Four properties are worth pinning and none of them is visible from
a green queue suite:

* the refusal is the queue's own sentence, so a surface cannot tell which door
  answered it;
* the wake row goes away — the whole reason a dismissal can boot nothing;
* a session with a LIVE owner is refused rather than written behind (:data:`code`
  ``ask_owner_present``), because the owner republishes its in-memory wake list
  over an outside write and would never see the row;
* the happy path leaves the log and the index agreeing, and starts nothing.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, cast

from local_operator.asks import render, store
from local_operator.asks.dismiss import dismiss_cold
from local_operator.harness.wake import WakeSchedule
from local_operator.wakes import store as wakes_store
from local_operator.wakes.store import read_entry as read_wake_entry

SESSION = "colddismiss01"
ASK_ID = "a-colddismiss"
NOW = 1_700_000_000_000


def _run(coro):
    return asyncio.run(coro)


def _session(root: Path, *, queued_at: int = NOW - 3_600_000, timeout_s: int = 120) -> Path:
    """A session directory with ONE queued ask whose deadline has passed."""
    directory = store.session_dir(root, SESSION)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")
    store.append_event(
        directory,
        {
            "v": store.EVENT_SCHEMA,
            "kind": store.EVENT_QUEUED,
            "ask_id": ASK_ID,
            "at": queued_at,
            "created_at": queued_at,
            "expires_at": queued_at + timeout_s * 1000,
            "timeout_s": timeout_s,
            "urgent": False,
            "tool_call_id": "",
            "questions": [
                {
                    "id": "q1",
                    "question": "ship it?",
                    "options": [{"label": "yes", "description": ""}],
                    "multi": False,
                    "secret": False,
                    "persist": False,
                }
            ],
        },
    )
    return directory


def _wake_row(wake_id: str, **extra: Any) -> dict[str, Any]:
    return WakeSchedule(
        id=wake_id,
        message=f"message {wake_id}",
        next_due_at=NOW + 60_000,
        created_at=NOW,
        **extra,
    ).model_dump()


def _seed_wakes(directory: Path, rows: list[dict[str, Any]]) -> None:
    """The transcript's own ``wake_schedules`` entry — the arm module's base."""
    (directory / "transcript.jsonl").write_text(
        json.dumps(
            {
                "id": "wake-entry-1",
                "ts": 2.0,
                "type": "custom",
                "payload": {"custom_type": "wake_schedules", "details": {"schedules": rows}},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _wake_ids(directory: Path) -> list[str]:
    """The ids in the transcript's LATEST ``wake_schedules`` row."""
    latest: list[str] = []
    for line in (directory / "transcript.jsonl").read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        payload = entry.get("payload") or {}
        if payload.get("custom_type") == "wake_schedules":
            latest = [str(row.get("id")) for row in payload["details"]["schedules"]]
    return latest


def _kinds(directory: Path) -> list[str]:
    return [str(event.get("kind")) for event in store.read_events(directory)]


def test_a_cold_dismiss_accepts_an_open_ask_and_records_the_reason(tmp_path: Path):
    """D6 (design §15): a session with no runtime is "no agent work running", so an
    OPEN ask is dismissible there. The event carries who and why."""
    directory = _session(tmp_path, queued_at=NOW, timeout_s=3600)
    outcome = _run(
        dismiss_cold(
            tmp_path, SESSION, ASK_ID, by="phone", reason="answered_in_chat", now_ms=NOW + 1_000
        )
    )
    assert outcome == {"ok": True}
    assert _kinds(directory) == [store.EVENT_QUEUED, store.EVENT_DISMISSED]
    event = store.read_events(directory)[-1]
    assert event["by"] == {"surface": "phone"}
    assert event["reason"] == "answered_in_chat"


def test_a_cold_dismiss_without_a_reason_records_moot(tmp_path: Path):
    directory = _session(tmp_path, queued_at=NOW, timeout_s=3600)
    assert _run(dismiss_cold(tmp_path, SESSION, ASK_ID, by="phone", now_ms=NOW + 1))["ok"]
    assert store.read_events(directory)[-1]["reason"] == "moot"


def test_a_cold_dismiss_refuses_an_unknown_reason_and_writes_nothing(tmp_path: Path):
    directory = _session(tmp_path, queued_at=NOW, timeout_s=3600)
    outcome = _run(
        dismiss_cold(tmp_path, SESSION, ASK_ID, by="phone", reason="because", now_ms=NOW)
    )
    assert outcome == {"ok": False, "error": render.DISMISS_BAD_REASON}
    assert _kinds(directory) == [store.EVENT_QUEUED]


def test_a_cold_open_dismiss_removes_the_deadline_wake_row(tmp_path: Path):
    """D3: the row goes because a pre-deadline dismissal owes a RESPONSE row that any
    boot delivers, not because a deadline notice is owed."""
    directory = _session(tmp_path, queued_at=NOW, timeout_s=3600)
    _seed_wakes(
        directory,
        [
            _wake_row(store.timeout_row_id(ASK_ID), kind="ask_timeout", hidden=True),
            _wake_row("w-user"),
        ],
    )
    assert _run(dismiss_cold(tmp_path, SESSION, ASK_ID, by="phone", now_ms=NOW + 5))["ok"]
    assert _wake_ids(directory) == ["w-user"]


def test_the_next_boot_delivers_a_cold_open_dismissal_as_a_response_row(tmp_path: Path):
    """The cold dismissal boots nothing; the row is owed through the fold."""
    directory = _session(tmp_path, queued_at=NOW, timeout_s=3600)
    assert _run(
        dismiss_cold(
            tmp_path, SESSION, ASK_ID, by="phone", reason="answered_in_chat", now_ms=NOW + 5
        )
    )["ok"]
    (record,) = store.fold(store.read_events(directory), NOW + 10)
    assert record["response_owed"] is True and record["delivered"] is False
    assert store.expected_row_ids(record) == [store.response_row_id(ASK_ID)]


def test_a_cold_dismiss_refuses_an_ask_this_session_never_queued(tmp_path: Path):
    """An unknown id is its own answer, not an "already declined" that would be a
    lie about an id this process has never seen (:func:`render.refusal_copy`)."""
    directory = _session(tmp_path)
    outcome = _run(dismiss_cold(tmp_path, SESSION, "a-nowhere", by="phone", now_ms=NOW))
    assert outcome == {"ok": False, "error": render.refusal_copy(None)}
    assert _kinds(directory) == [store.EVENT_QUEUED]


def test_every_cold_refusal_carries_the_sentence_a_surface_renders(tmp_path: Path):
    """The route's last-resort string is never what a user reads (agent review
    round 1, R1-5): every refusal this door can produce already IS a sentence, so
    the desktop's fallback is a contract-break guard rather than user copy.

    Pinned here rather than in the route because the door is the producer: if a
    refusal ever grew an empty ``error``, the desktop would say "the dismissal was
    refused" in an untranslated English string instead of the ask's own words.
    """
    unknown_reason = _run(dismiss_cold(tmp_path, SESSION, ASK_ID, reason="because", now_ms=NOW))
    directory = _session(tmp_path)
    unknown_ask = _run(dismiss_cold(tmp_path, SESSION, "a-nowhere", by="phone", now_ms=NOW))
    already = _run(dismiss_cold(tmp_path, SESSION, ASK_ID, by="phone", now_ms=NOW))
    assert already.get("ok") is True, already
    twice = _run(dismiss_cold(tmp_path, SESSION, ASK_ID, by="phone", now_ms=NOW))
    for outcome in (unknown_reason, unknown_ask, twice):
        assert outcome.get("ok") is False, outcome
        assert str(outcome.get("error") or "").strip(), outcome
    assert _kinds(directory)[-1] == store.EVENT_DISMISSED


def test_a_cold_dismiss_drops_the_deadline_wake_row(tmp_path: Path):
    """THE WAKE ROW IS WHY THIS CAN BOOT NOTHING.

    ``ask-timeout-<id>`` is what makes a deadline survive a runtime that does not
    exist: the supervisor engages a runtime for it. A dismissed ask has nothing
    left to deliver by fire (fork A delivers the notice at the next real reopen),
    so the row must go with the dismissal — leaving it would boot the very
    runtime the op exists to avoid, for a notice the row's owner has put away.
    """
    directory = _session(tmp_path)
    _seed_wakes(
        directory,
        [
            _wake_row(store.timeout_row_id(ASK_ID), kind="ask_timeout", hidden=True),
            _wake_row("w-user"),
        ],
    )
    outcome = _run(dismiss_cold(tmp_path, SESSION, ASK_ID, by="phone", now_ms=NOW))
    assert outcome == {"ok": True}
    assert _wake_ids(directory) == ["w-user"]
    assert read_wake_entry(tmp_path, SESSION) is not None


def test_a_cold_dismiss_refuses_behind_a_live_owner(tmp_path: Path):
    """A live runtime owns the log AND its wake list, and republishes that list
    wholesale on its next persist — so an outside write would be lost, or would
    deliver the notice without the turn it owes. The relay falls back to the dial
    on THIS code alone, and nothing is written before it answers."""
    directory = _session(tmp_path)
    (directory / ".session.pid").write_text(str(os.getpid()), encoding="utf-8")
    outcome = _run(dismiss_cold(tmp_path, SESSION, ASK_ID, by="phone", now_ms=NOW))
    assert outcome == {"ok": False, "code": "ask_owner_present"}
    assert _kinds(directory) == [store.EVENT_QUEUED]


def test_a_cold_dismiss_records_the_event_and_the_index_and_starts_nothing(tmp_path: Path):
    """The happy path: the log is the truth, the index is rewritten from it, and
    the fold on the way out says the notice is still owed (the next reconcile
    delivers it). No runtime is started — nothing here can start one."""
    directory = _session(tmp_path)
    store.write_entry(
        tmp_path,
        SESSION,
        cwd="/tmp/colddismiss",
        asks=[store.pending_row(store.fold(store.read_events(directory), NOW)[0])],
    )

    outcome = _run(dismiss_cold(tmp_path, SESSION, ASK_ID, by="phone", now_ms=NOW))
    assert outcome == {"ok": True}

    events = store.read_events(directory)
    assert [event["kind"] for event in events] == [store.EVENT_QUEUED, store.EVENT_DISMISSED]
    dismissed = events[-1]
    assert dismissed["at"] == NOW
    assert dismissed["by"] == {"surface": "phone"}

    entry = store.read_entry(tmp_path, SESSION)
    assert entry is not None
    (row,) = entry["asks"]
    assert row["ask_id"] == ASK_ID
    assert row["status"] == store.STATUS_DISMISSED

    # The fold's own half of the contract: dismissed, and the notice is still owed
    # to the model (the queue's reconcile is what delivers it).
    (record,) = store.fold(events, NOW + 1)
    assert record["status"] == store.STATUS_DISMISSED
    assert record["notice_owed"] is True
    # Nothing claimed the session: this op boots nothing, and no runtime had it.
    from local_operator.resume import live_runtime_pid

    assert live_runtime_pid(tmp_path, SESSION) is None


def test_a_cold_dismiss_injects_nothing_of_its_own(tmp_path: Path):
    """The op writes ONE event and one index row: no TRANSCRIPT row is appended
    for it, so a caller that reads the transcript before and after sees no new
    entry — the notice (if owed) is the next reconcile's business, not this
    door's.

    Seeded against the REAL clock (``now_ms`` omitted) because that is the arm the
    relay and the TUI call: the default has to fold an ask whose deadline has
    passed exactly as an explicit ``now_ms`` does.
    """
    real = store.now_ms()
    directory = _session(tmp_path, queued_at=real - 3_600_000, timeout_s=120)
    before = (directory / "transcript.jsonl").read_text(encoding="utf-8")
    outcome = _run(dismiss_cold(tmp_path, SESSION, ASK_ID, by="phone"))
    assert outcome == {"ok": True}
    assert (directory / "transcript.jsonl").read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# round 1 remediation — the park marker, the divergent index, the surface word
# ---------------------------------------------------------------------------


class _StoppedProbe:
    """Exactly the fields the REAL ``AttachedSession.session_was_stopped`` reads.

    Called on the class with this as ``self`` so the cell below exercises the
    shipped implementation (the marker read plus its live-owner probe) rather
    than a copy of the rule: a marker no reader consults is not the property the
    review asked about.
    """

    _deliberate_stop = False

    def __init__(self, config_dir: Path) -> None:
        self._config_dir = config_dir
        self._session_id = SESSION


def _park(
    root: Path,
    directory: Path,
    *,
    stopped_at: int | None = None,
    held_at: int | None = None,
    rows: list[dict[str, Any]],
) -> None:
    """Park the session: an index entry carrying a park marker AND ``rows``.

    Both halves matter, and so does the shape they are in. ``stopped_at`` is
    stamped by the stop path (``session/runtime/control.py:_mark_wakes_dormant``),
    which counts the HIDDEN rows a session carries — so a stopped conversation
    whose only wake IS its ask's own deadline row is the reachable state, not a
    contrived one. ``held_at`` is Aida's pause marker
    (``aida/proactive.py``), which keeps the same schedules. The index entry is
    what every reader of a marker reads: ``attached.session_was_stopped``,
    ``attention._stopped_marker`` and the desktop warm guards for the first, and
    ``wakes.store.is_held`` for either.
    """
    wakes_store.write_entry(
        root,
        SESSION,
        cwd=str(directory),
        schedules=rows,
        preserve={"stopped_at": stopped_at} if stopped_at is not None else {"held_at": held_at},
    )


def test_a_cold_dismiss_leaves_a_parked_sessions_stop_marker_alone(tmp_path: Path):
    """A PARKED SESSION'S PARK MARKER OUTLIVES THE DISMISSAL (agent review round 1, MAJOR).

    ``cancel_wake`` REMOVES the index entry when the remainder is empty
    (``wakes/arm.py``), and that entry carried ``stopped_at`` — the durable
    marker that says this conversation was stopped on purpose. With the entry
    gone, ``session_was_stopped`` answers False, ``attention._stopped_marker``
    reads "not proven stopped", and the desktop warm guards
    (``server/utils/desktop_sessions.py``) may start the very runtime this PR
    promises not to start.

    The cancel buys nothing on a parked entry: every supervisor decision skips a
    held one (``wakes/supervisor.py``'s ``_is_held_entry``), so the row engages
    nothing while the session is parked — and it is stale by then, because only
    a terminal (timed-out) ask can be dismissed. So the row stays, the marker
    stays, and NO transcript write happens: the mtime that moves a conversation
    to the top of /resume and the phone's list (QA round 1, Q1) does not move.
    """
    directory = _session(tmp_path)
    hidden = _wake_row(store.timeout_row_id(ASK_ID), kind="ask_timeout", hidden=True)
    _seed_wakes(directory, [hidden])
    _park(tmp_path, directory, stopped_at=NOW - 60_000, rows=[hidden])
    before = (directory / "transcript.jsonl").read_bytes()
    mtime = (directory / "transcript.jsonl").stat().st_mtime_ns

    outcome = _run(dismiss_cold(tmp_path, SESSION, ASK_ID, by="phone", now_ms=NOW))
    assert outcome == {"ok": True}

    entry = read_wake_entry(tmp_path, SESSION)
    assert entry is not None, "the dismissal removed the entry that carried stopped_at"
    assert entry["stopped_at"] == NOW - 60_000
    assert wakes_store.is_held(entry) is True
    assert [row["id"] for row in entry["schedules"]] == [store.timeout_row_id(ASK_ID)]
    assert _wake_ids(directory) == [store.timeout_row_id(ASK_ID)]
    # No transcript write at all, so the conversation keeps its place in /resume.
    assert (directory / "transcript.jsonl").read_bytes() == before
    assert (directory / "transcript.jsonl").stat().st_mtime_ns == mtime

    # The two readers the review named still read "stopped".
    from local_operator.session.attention import _stopped_marker

    assert _stopped_marker(directory) is True
    from local_operator.session.attached import AttachedSession

    # The REAL method, on the smallest object that can carry it: a marker no
    # reader consults is not the property the review asked about.
    probe = cast(Any, _StoppedProbe(tmp_path))
    assert _run(AttachedSession.session_was_stopped(probe)) is True


def test_a_cold_dismiss_leaves_a_held_sessions_park_marker_alone(tmp_path: Path):
    """THE OTHER PARK MARKER, THE SAME SKIP (agent review round 2, M-5).

    ``is_held`` is ``bool(stopped_at or held_at)`` (``wakes/store.py``), so the
    cancel's skip covers both — but round 1 pinned only ``stopped_at``, which
    left the half Aida writes covered by reading the predicate rather than by a
    cell. ``held_at`` is the marker her pause lever stamps and her resume clears
    with no runtime involved, and the entry it lives in is the same entry
    ``cancel_wake``'s empty remainder would remove: the row stays, the marker
    stays, and no transcript write moves this conversation in /resume.
    """
    directory = _session(tmp_path)
    hidden = _wake_row(store.timeout_row_id(ASK_ID), kind="ask_timeout", hidden=True)
    _seed_wakes(directory, [hidden])
    _park(tmp_path, directory, held_at=NOW - 60_000, rows=[hidden])
    before = (directory / "transcript.jsonl").read_bytes()
    mtime = (directory / "transcript.jsonl").stat().st_mtime_ns

    outcome = _run(dismiss_cold(tmp_path, SESSION, ASK_ID, by="phone", now_ms=NOW))
    assert outcome == {"ok": True}

    entry = read_wake_entry(tmp_path, SESSION)
    assert entry is not None, "the dismissal removed the entry that carried held_at"
    assert entry["held_at"] == NOW - 60_000
    assert wakes_store.is_held(entry) is True
    assert [row["id"] for row in entry["schedules"]] == [store.timeout_row_id(ASK_ID)]
    assert _wake_ids(directory) == [store.timeout_row_id(ASK_ID)]
    assert (directory / "transcript.jsonl").read_bytes() == before
    assert (directory / "transcript.jsonl").stat().st_mtime_ns == mtime


def test_a_cold_dismiss_still_drops_the_wake_row_when_nothing_parks_it(tmp_path: Path):
    """THE OTHER SIDE OF THE PARKED RULE, and why the row must go there.

    With no ``stopped_at``/``held_at`` the entry is not parked, so nothing
    suppresses its fire: the supervisor engages a runtime for a due row, and a
    crashed conversation's stale deadline row would boot exactly the session the
    dismissal exists to avoid booting. So the row goes — and the transcript gains
    the empty ``wake_schedules`` bookkeeping row whose mtime moves the
    conversation in /resume. That reorder is accepted rather than overlooked: the
    wake list genuinely changed, and this session never carried a park marker for
    the entry removal to destroy.
    """
    directory = _session(tmp_path)
    hidden = _wake_row(store.timeout_row_id(ASK_ID), kind="ask_timeout", hidden=True)
    _seed_wakes(directory, [hidden, _wake_row("w-user")])
    wakes_store.write_entry(
        tmp_path, SESSION, cwd=str(directory), schedules=[hidden, _wake_row("w-user")]
    )

    outcome = _run(dismiss_cold(tmp_path, SESSION, ASK_ID, by="phone", now_ms=NOW))
    assert outcome == {"ok": True}
    assert _wake_ids(directory) == ["w-user"]
    entry = read_wake_entry(tmp_path, SESSION)
    assert entry is not None
    assert wakes_store.is_held(entry) is False
    assert [row["id"] for row in entry["schedules"]] == ["w-user"]


def test_a_cold_dismiss_repairs_an_index_row_the_transcript_never_carried(tmp_path: Path):
    """QA round 1, Q2: a divergent index keeps the row alive, so the dismissal heals it.

    ``cancel_wake`` reads its base from the TRANSCRIPT (``wakes/arm.py``'s own
    invariant) and refuses with ``wake_not_found`` when the id is not in it —
    which this door used to swallow, leaving an index row the supervisor reads
    and fires. The repair is the arm module's own: ``repair_index`` re-projects
    the transcript onto the index, so the stale row goes and the rows the
    transcript does carry (``w-user`` here) survive untouched. Best-effort, like
    every other index writer: an unreadable transcript leaves the index alone.
    """
    directory = _session(tmp_path)
    _seed_wakes(directory, [_wake_row("w-user")])
    wakes_store.write_entry(
        tmp_path,
        SESSION,
        cwd=str(directory),
        schedules=[
            _wake_row("w-user"),
            _wake_row(store.timeout_row_id(ASK_ID), kind="ask_timeout", hidden=True),
        ],
    )

    outcome = _run(dismiss_cold(tmp_path, SESSION, ASK_ID, by="phone", now_ms=NOW))
    assert outcome == {"ok": True}
    entry = read_wake_entry(tmp_path, SESSION)
    assert entry is not None
    assert [row["id"] for row in entry["schedules"]] == ["w-user"]
    assert _wake_ids(directory) == ["w-user"]


def test_a_cold_dismiss_records_the_surface_the_live_door_would(tmp_path: Path):
    """AGENT REVIEW ROUND 1, MINOR-3: the phone's dismissal names no surface.

    ``mobile/daemon.py`` passes ``by=str(body.get("by") or "")`` and the phone's
    ``ask_dismiss`` body has no ``by``, so the cold door recorded
    ``by: {"surface": ""}`` — an audit line that names nobody, where the live
    door records ``"remote"`` for the same tap (``serving.py``: ``by or
    "remote"``). The empty string is the one input that means "the remote door
    did not name itself", so the cold door defaults it the same way.
    """
    directory = _session(tmp_path)
    outcome = _run(dismiss_cold(tmp_path, SESSION, ASK_ID, by="", now_ms=NOW))
    assert outcome == {"ok": True}
    assert store.read_events(directory)[-1]["by"] == {"surface": "remote"}


def test_a_second_cold_dismiss_says_the_ask_is_already_dismissed(tmp_path: Path):
    """UX round 1, U3 = QA Q3 = agent review MINOR-4: the repeat tap.

    A second dismissal answered "only a timed-out ask can be dismissed; it is
    still open." — which is the opposite of what the user did, on the one row
    the surface had already struck out. The dismissed state gets its own
    sentence, shared with the live door (``AskQueue.dismiss``) because both doors
    answer this question with the same words.
    """
    directory = _session(tmp_path)
    assert _run(dismiss_cold(tmp_path, SESSION, ASK_ID, by="phone", now_ms=NOW)) == {"ok": True}
    again = _run(dismiss_cold(tmp_path, SESSION, ASK_ID, by="phone", now_ms=NOW + 1))
    assert again == {"ok": False, "error": "you already dismissed this."}
    assert _kinds(directory) == [store.EVENT_QUEUED, store.EVENT_DISMISSED]
