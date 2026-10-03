"""``local_operator.wakes.patience`` — cycle math, episode derivation, arm/cancel.

These pin the rules the design fixes in §8.2, in the order the module states
them: the wait progression and clamps; the WATERMARK (with the attribution trap
called out by its own test); the episode derivation from transcript facts + the
pending row; the arm path's gates (class, engine hold, caps, TTL); and the
fire-side helpers (stale / TTL / note / details). The session-side composition
of these — delivery, cancel-on-reply, the turn-end flush — lives in
``tests/unit/session/test_patience_session.py``.
"""

from __future__ import annotations

import time
from typing import Any

import pytest
import yaml

from local_operator.harness.wake_types import MAX_WAKE_SCHEDULES, WakeSchedule
from local_operator.resume import write_session_attachment
from local_operator.wakes import patience as P

#: A fixed "now" so every assertion is about arithmetic, not about when the
#: suite ran. Wall time is read nowhere in the pure layer.
NOW = 1_800_000_000_000


def policy(**overrides: Any):
    base: dict[str, Any] = dict(
        default_ms=300_000,
        backoff=3,
        max_attempts=3,
        ttl_ms=7_200_000,
        max_pending=4,
    )
    base.update(overrides)
    return P.PatiencePolicy(**base)


def patience_row(**overrides: Any) -> WakeSchedule:
    base: dict[str, Any] = dict(
        id="patience-1",
        message="",
        next_due_at=NOW + 300_000,
        created_at=NOW,
        kind="patience",
        hidden=True,
        episode_id="patience-1",
        attempt=1,
        armed_at=NOW,
        armed_after="",
        note="",
    )
    base.update(overrides)
    return WakeSchedule(**base)


def wake_entry(ts_s: float, *, kind: str | None = None, attempt: int = 1, started_at: int = NOW):
    details: dict[str, object] = {"text": "x"}
    if kind is not None:
        details.update(
            {
                "kind": kind,
                "hidden": True,
                "episode_id": "patience-1",
                "attempt": attempt,
                "episode_started_at": started_at,
            }
        )
    return {
        "ts": ts_s,
        "type": "message",
        "payload": {"custom_type": "wake_prompt", "attribution": "user", "details": details},
    }


def user_entry(ts_s: float, *, injected: bool = False):
    payload: dict[str, object] = {"role": "user", "content": []}
    if injected:
        payload["provider_payload"] = {"harness_injected": True}
    return {"ts": ts_s, "type": "message", "payload": payload}


def peer_entry(ts_s: float):
    return {
        "ts": ts_s,
        "type": "message",
        "payload": {"custom_type": "peer_message", "attribution": "user", "details": {}},
    }


def assistant_entry(ts_s: float):
    return {"ts": ts_s, "type": "message", "payload": {"role": "assistant", "content": []}}


class FakeScheduler:
    """The duck the arm/cancel paths use: ``schedules`` + ``await update()``."""

    def __init__(self, schedules=()):
        self._schedules = list(schedules)
        self.updates: list[list[WakeSchedule]] = []

    @property
    def schedules(self):
        return tuple(self._schedules)

    async def update(self, schedules):
        self._schedules = list(schedules)
        self.updates.append(list(schedules))


def aida_session_dir(tmp_path):
    """A session dir that RESOLVES proactive: the packaged aida seed's class."""
    session_dir = tmp_path / "sessions" / "sess00000001"
    session_dir.mkdir(parents=True)
    write_session_attachment(session_dir, team="", agent="aida", goal="")
    return session_dir


# ---------------------------------------------------------------------------
# Policy and cycle math
# ---------------------------------------------------------------------------


class TestPolicy:
    def test_defaults_match_the_shipped_constants(self, tmp_path) -> None:
        pol = P.policy(tmp_path)
        assert pol == P.PatiencePolicy(
            default_ms=P.DEFAULT_WAIT_MS,
            backoff=P.DEFAULT_BACKOFF,
            max_attempts=P.DEFAULT_MAX_ATTEMPTS,
            ttl_ms=P.DEFAULT_TTL_MS,
            max_pending=P.DEFAULT_MAX_PENDING,
        )
        assert P.DEFAULT_WAIT_MS == 300_000
        assert P.DEFAULT_MAX_ATTEMPTS == 3
        assert P.DEFAULT_TTL_MS == 7_200_000

    def test_config_overrides_are_read(self, tmp_path) -> None:
        (tmp_path / "config.yml").write_text(
            yaml.safe_dump(
                {
                    "values": {
                        "proactive": {
                            "patience": {
                                "default_ms": 120_000,
                                "backoff": 2,
                                "max_attempts": 4,
                                "episode_ttl_ms": 3_600_000,
                                "max_pending": 2,
                            }
                        }
                    }
                }
            )
        )
        pol = P.policy(tmp_path)
        assert (pol.default_ms, pol.backoff, pol.max_attempts) == (120_000, 2, 4)
        assert (pol.ttl_ms, pol.max_pending) == (3_600_000, 2)

    def test_garbage_values_degrade_to_the_defaults_without_raising(self, tmp_path) -> None:
        (tmp_path / "config.yml").write_text(
            yaml.safe_dump({"values": {"proactive": {"patience": {"default_ms": "soon"}}}})
        )
        assert P.policy(tmp_path).default_ms == P.DEFAULT_WAIT_MS


class TestCycleMath:
    def test_the_progression_is_five_fifteen_forty_five(self) -> None:
        pol = policy()
        waits = [P.patience_wait_ms(attempt, None, pol) for attempt in (1, 2, 3)]
        assert waits == [300_000, 900_000, 2_700_000]

    def test_attempt_one_honours_a_requested_wait_within_the_clamp(self) -> None:
        pol = policy()
        assert P.patience_wait_ms(1, 90_000, pol) == 90_000
        assert P.patience_wait_ms(1, 1, pol) == P.MIN_TIMEOUT_MS
        assert P.patience_wait_ms(1, P.MAX_TIMEOUT_MS * 10, pol) == P.MAX_TIMEOUT_MS

    def test_later_attempts_cannot_shorten_below_the_backoff_floor(self) -> None:
        pol = policy()
        # The backoff is a FLOOR, not a suggestion: an explicit short request on
        # attempt 2 is raised to the floor (over-long is the safe direction for
        # a mechanism whose failure mode is nagging).
        assert P.patience_wait_ms(2, 60_000, pol) == 900_000
        assert P.patience_wait_ms(2, 3_600_000, pol) == 3_600_000


# ---------------------------------------------------------------------------
# Transcript facts — the watermark, and the attribution trap
# ---------------------------------------------------------------------------


class TestScanEntries:
    def test_wake_and_patience_deliveries_are_not_replies(self) -> None:
        """THE TRAP (design §8.2.3): deliveries are ``attribution="user"`` too.

        A wake fire renders a user message; if the watermark counted role=user
        it would cancel its own patience wait the moment it fired. Only a real
        typed prompt counts as a reply.
        """
        entries = [
            user_entry(NOW / 1000 - 60),
            wake_entry(NOW / 1000 - 30),  # a scheduled fire, days into the session
            wake_entry(NOW / 1000 - 10, kind="patience"),  # the patience fire itself
        ]
        facts = P.scan_entries(entries)
        assert facts.last_inbound_ms == int((NOW / 1000 - 60) * 1000)
        assert facts.last_fire is not None and facts.last_fire.attempt == 1

    def test_peer_messages_count_as_inbound(self) -> None:
        entries = [peer_entry(NOW / 1000 - 5)]
        facts = P.scan_entries(entries)
        assert facts.last_inbound_ms == int((NOW / 1000 - 5) * 1000)

    def test_harness_injected_user_rows_do_not_count(self) -> None:
        entries = [user_entry(NOW / 1000 - 5, injected=True), user_entry(NOW / 1000 - 10)]
        facts = P.scan_entries(entries)
        assert facts.last_inbound_ms == int((NOW / 1000 - 10) * 1000)

    def test_assistant_rows_say_nothing(self) -> None:
        entries = [assistant_entry(NOW / 1000), assistant_entry(NOW / 1000 + 1)]
        assert P.scan_entries(entries) == P.EpisodeFacts(None, None, None)

    def test_only_non_patience_wakes_are_markers(self) -> None:
        entries = [wake_entry(NOW / 1000 - 1, kind="patience")]
        facts = P.scan_entries(entries)
        assert facts.last_marker_ms is None
        entries.append(wake_entry(NOW / 1000, kind=None))
        facts = P.scan_entries(entries)
        assert facts.last_marker_ms == int((NOW / 1000) * 1000)

    def test_cutoff_stops_the_walk_and_hides_older_facts(self) -> None:
        entries = [user_entry(NOW / 1000 - 10_000), assistant_entry(NOW / 1000 - 1)]
        facts = P.scan_entries(entries, cutoff_ms=int((NOW / 1000 - 100) * 1000))
        assert facts == P.EpisodeFacts(None, None, None)

    def test_only_the_newest_fact_of_each_kind_survives(self) -> None:
        entries = [
            user_entry(NOW / 1000 - 50),
            user_entry(NOW / 1000 - 40),
            wake_entry(NOW / 1000 - 30, kind="patience", attempt=1),
            wake_entry(NOW / 1000 - 20, kind="patience", attempt=2),
        ]
        facts = P.scan_entries(entries)
        assert facts.last_inbound_ms == int((NOW / 1000 - 40) * 1000)
        assert facts.last_fire is not None and facts.last_fire.attempt == 2


class TestScanTranscript:
    @pytest.mark.asyncio
    async def test_the_file_scan_agrees_with_the_live_scan(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The tool path reads the journal; the live path reads memory.

        One classification behind both, so the two can never disagree about
        whether a reply or a fire happened — this pins that by running them on
        the same rows.

        WHY THE CLOCK IS PINNED (the shard that came back red): the scan
        derives each fact's millisecond by truncating the entry's ``ts``, so
        two appends landing inside ONE millisecond made the ordering
        assertion below compare a value with itself (``assert X < X``; CI
        shard ``test (3.12, 1)`` failed exactly so). The transcript module's
        ``time`` is replaced with a strictly increasing stand-in — every read
        10 ms after the last — so the two appends cannot share a millisecond:
        determinism by construction, no sleep and no production change. The
        clock starts at the real time so the ``now_ms`` window given to the
        file scan still brackets the stamps.
        """
        from local_operator.harness.types import CustomMessage, Message
        from local_operator.harness.wake import WAKE_PROMPT_MESSAGE_TYPE
        from local_operator.session import transcript as transcript_mod
        from local_operator.session.transcript import Transcript

        class _AdvancingClock:
            """The transcript module's ``time``, advancing 10 ms per read."""

            def __init__(self) -> None:
                self._now = time.time()

            def time(self) -> float:
                self._now += 0.01
                return self._now

        monkeypatch.setattr(transcript_mod, "time", _AdvancingClock())

        directory = tmp_path / "sess"
        directory.mkdir(parents=True)
        transcript = Transcript(directory)
        now_s = time.time()
        await transcript.append_messages([Message.user("hello", id="m1")])
        await transcript.append_messages(
            [
                CustomMessage(
                    custom_type=WAKE_PROMPT_MESSAGE_TYPE,
                    attribution="user",
                    details={
                        "text": "fire",
                        "kind": "patience",
                        "hidden": True,
                        "attempt": 1,
                        "episode_id": "patience-1",
                        "episode_started_at": NOW,
                    },
                )
            ]
        )
        live = P.scan_entries(transcript.entries())
        disk = P.scan_transcript(
            directory, now_ms=int(now_s * 1000) + 1000, window_ms=P.DEFAULT_TTL_MS + 3_600_000
        )
        assert disk.last_inbound_ms == live.last_inbound_ms
        assert disk.last_fire is not None and live.last_fire is not None
        assert disk.last_fire.attempt == live.last_fire.attempt == 1
        # And the attribution trap holds on the file path too: the fire above
        # is an ``attribution="user"`` custom message and is NOT an inbound.
        assert disk.last_inbound_ms is not None
        assert disk.last_inbound_ms < disk.last_fire.ts_ms


# ---------------------------------------------------------------------------
# Episode derivation and the arm plan
# ---------------------------------------------------------------------------


class TestOpenEpisode:
    def test_no_pending_and_no_fire_means_no_episode(self) -> None:
        assert (
            P.open_episode([], P.EpisodeFacts(None, None, None), pol=policy(), now_ms=NOW) is None
        )

    def test_a_live_pending_row_is_the_episode(self) -> None:
        state = P.open_episode(
            [patience_row()], P.EpisodeFacts(None, None, None), pol=policy(), now_ms=NOW
        )
        assert state is not None
        assert (state.episode_id, state.attempt, state.started_at_ms) == ("patience-1", 1, NOW)
        assert state.pending_row is not None and state.terminal is False

    def test_a_pending_row_with_a_later_reply_is_dead(self) -> None:
        facts = P.EpisodeFacts(NOW + 1, None, NOW + 1)
        assert P.open_episode([patience_row()], facts, pol=policy(), now_ms=NOW + 2) is None

    def test_a_fire_reopens_the_episode_until_a_reply(self) -> None:
        fire = P.FireFact(
            ts_ms=NOW + 300_000, episode_id="patience-1", attempt=1, episode_started_at_ms=NOW
        )
        state = P.open_episode(
            [], P.EpisodeFacts(None, fire, None), pol=policy(), now_ms=NOW + 300_000
        )
        assert state is not None and state.attempt == 1 and state.pending_row is None

    def test_a_terminal_fire_closes_the_cycle(self) -> None:
        fire = P.FireFact(
            ts_ms=NOW, episode_id="patience-1", attempt=3, episode_started_at_ms=NOW - 3_600_000
        )
        state = P.open_episode([], P.EpisodeFacts(None, fire, None), pol=policy(), now_ms=NOW)
        assert state is not None and state.terminal is True

    def test_a_marker_after_the_terminal_fire_reopens_the_question(self) -> None:
        fire = P.FireFact(
            ts_ms=NOW, episode_id="patience-1", attempt=3, episode_started_at_ms=NOW - 3_600_000
        )
        facts = P.EpisodeFacts(None, fire, NOW + 10)
        assert P.open_episode([], facts, pol=policy(), now_ms=NOW + 10) is None

    def test_a_fire_past_its_ttl_is_closed_by_age(self) -> None:
        fire = P.FireFact(
            ts_ms=NOW, episode_id="patience-1", attempt=1, episode_started_at_ms=NOW - 8_000_000
        )
        assert (
            P.open_episode([], P.EpisodeFacts(None, fire, None), pol=policy(), now_ms=NOW) is None
        )


class TestPlanArm:
    def test_a_fresh_arm_starts_attempt_one(self) -> None:
        out = P.plan_arm([], P.EpisodeFacts(None, None, None), policy(), now_ms=NOW)
        assert out.ok and out.row.attempt == 1 and out.row.kind == "patience"
        assert out.row.hidden is True
        assert out.row.created_at == NOW and out.row.armed_at == NOW
        assert out.row.next_due_at == NOW + 300_000

    def test_a_continuation_while_pending_bumps_the_attempt_in_place(self) -> None:
        out = P.plan_arm(
            [patience_row()], P.EpisodeFacts(None, None, None), policy(), now_ms=NOW + 1000
        )
        assert out.ok and out.replaced_id == "patience-1"
        assert out.row.id == "patience-1" and out.row.attempt == 2
        # The TTL anchor is the EPISODE start, preserved across the re-arm.
        assert out.row.created_at == NOW
        assert out.row.next_due_at == NOW + 1000 + 900_000

    def test_a_continuation_after_a_fire_resumes_the_same_episode(self) -> None:
        fire = P.FireFact(
            ts_ms=NOW + 300_000, episode_id="patience-1", attempt=2, episode_started_at_ms=NOW
        )
        out = P.plan_arm([], P.EpisodeFacts(None, fire, None), policy(), now_ms=NOW + 300_000)
        assert out.ok
        assert out.row.id == "patience-1" and out.row.attempt == 3
        assert out.row.created_at == NOW  # the original episode start rides along
        assert out.replaced_id == ""

    def test_the_attempt_bound_refuses_with_a_sentence(self) -> None:
        fire = P.FireFact(
            ts_ms=NOW, episode_id="patience-1", attempt=3, episode_started_at_ms=NOW - 3_600_000
        )
        out = P.plan_arm([], P.EpisodeFacts(None, fire, None), policy(), now_ms=NOW)
        assert not out.ok
        assert "closed" in out.error

    def test_the_pending_cap_refuses(self) -> None:
        # Four pending rows whose episodes are CLOSED by a reply (the watermark
        # would retire them at delivery; cancel-on-reply just has not landed
        # yet). None is continuable, so this arm would be a FIFTH pending row —
        # refused, with the remedy named.
        rows = [
            patience_row(id=f"patience-{i}", episode_id=f"patience-{i}", armed_at=NOW - 10_000)
            for i in range(1, 5)
        ]
        facts = P.EpisodeFacts(NOW - 5_000, None, NOW - 5_000)
        out = P.plan_arm(rows, facts, policy(), now_ms=NOW)
        assert not out.ok and "at most 4 pending" in out.error

    def test_the_schedule_cap_refuses(self) -> None:
        rows = [
            WakeSchedule(id=f"w{i}", message="m", next_due_at=NOW + i)
            for i in range(1, MAX_WAKE_SCHEDULES + 1)
        ]
        out = P.plan_arm(rows, P.EpisodeFacts(None, None, None), policy(), now_ms=NOW)
        assert not out.ok and "16" in out.error

    def test_a_reused_episode_id_cannot_drop_someone_elses_row(self) -> None:
        # A fire whose episode id collides with a live foreign row: the plan
        # mints a fresh id rather than replacing the row.
        foreign = patience_row(id="patience-1", episode_id="patience-1", armed_after="peer:x")
        fire = P.FireFact(
            ts_ms=NOW, episode_id="patience-1", attempt=1, episode_started_at_ms=NOW - 1000
        )
        rows = [foreign, WakeSchedule(id="w1", message="m", next_due_at=NOW + 10)]
        out = P.plan_arm(rows, P.EpisodeFacts(None, fire, None), policy(), now_ms=NOW)
        # The pending foreign row wins the classification (it is newer than the
        # fire), so this arms a continuation of IT rather than dropping it.
        assert out.ok and out.row.id == "patience-1"

    def test_the_ttl_leaves_no_room_refuses(self) -> None:
        pending = patience_row(created_at=NOW - P.DEFAULT_TTL_MS + 30_000)
        out = P.plan_arm([pending], P.EpisodeFacts(None, None, None), policy(), now_ms=NOW)
        assert not out.ok and "bound" in out.error


# ---------------------------------------------------------------------------
# The arm path (gates + persistence)
# ---------------------------------------------------------------------------


class TestArmPatience:
    @pytest.mark.asyncio
    async def test_a_reactive_session_is_refused(self, tmp_path) -> None:
        scheduler = FakeScheduler()
        outcome = await P.arm_patience(
            scheduler,
            session_dir=tmp_path / "sessions" / "plain",
            config_dir=tmp_path,
            action_class="reactive",
            now_ms=NOW,
        )
        assert not outcome.ok and "proactive class" in outcome.error
        assert scheduler.updates == []

    @pytest.mark.asyncio
    async def test_an_engine_hold_is_refused(self, tmp_path) -> None:
        scheduler = FakeScheduler()
        outcome = await P.arm_patience(
            scheduler,
            session_dir=aida_session_dir(tmp_path),
            config_dir=tmp_path,
            action_class="proactive",
            now_ms=NOW,
            suppressed=True,
        )
        assert not outcome.ok and "paused" in outcome.error
        assert scheduler.updates == []

    @pytest.mark.asyncio
    async def test_an_arm_writes_one_hidden_row_and_persists_once(self, tmp_path) -> None:
        scheduler = FakeScheduler()
        outcome = await P.arm_patience(
            scheduler,
            session_dir=aida_session_dir(tmp_path),
            config_dir=tmp_path,
            action_class="proactive",
            now_ms=NOW,
            note="still waiting on the deploy",
            after="message:m1",
        )
        assert outcome.ok and outcome.row is not None
        assert len(scheduler.updates) == 1
        (written,) = scheduler.schedules
        assert written.kind == "patience" and written.hidden is True
        assert written.attempt == 1 and written.note == "still waiting on the deploy"
        assert written.armed_after == "message:m1"

    @pytest.mark.asyncio
    async def test_a_continuation_reads_the_transcript_for_the_episode(self, tmp_path) -> None:
        from local_operator.harness.types import CustomMessage
        from local_operator.harness.wake import WAKE_PROMPT_MESSAGE_TYPE
        from local_operator.session.transcript import Transcript

        # Real wall time on purpose: the scan is bounded in TIME, so the
        # fixture and the request must share one clock base (a fixed 2027
        # ``now`` would put a freshly appended row outside the window).
        now_ms = int(time.time() * 1000)
        session_dir = aida_session_dir(tmp_path)
        transcript = Transcript(session_dir)
        await transcript.append_messages(
            [
                CustomMessage(
                    custom_type=WAKE_PROMPT_MESSAGE_TYPE,
                    attribution="user",
                    details={
                        "text": "no reply yet",
                        "kind": "patience",
                        "hidden": True,
                        "episode_id": "patience-4",
                        "attempt": 1,
                        "episode_started_at": now_ms - 300_000,
                    },
                )
            ]
        )
        scheduler = FakeScheduler()
        outcome = await P.arm_patience(
            scheduler,
            session_dir=session_dir,
            config_dir=tmp_path,
            action_class="proactive",
            now_ms=now_ms,
        )
        assert outcome.ok and outcome.row is not None
        assert outcome.row.id == "patience-4" and outcome.row.attempt == 2
        assert outcome.row.created_at == now_ms - 300_000


class TestCancelPatience:
    @pytest.mark.asyncio
    async def test_cancel_all_retires_every_pending_wait(self) -> None:
        scheduler = FakeScheduler(
            [patience_row(), WakeSchedule(id="w1", message="m", next_due_at=NOW)]
        )
        cancelled, error = await P.cancel_patience(scheduler)
        assert cancelled == ["patience-1"] and error == ""
        assert [r.id for r in scheduler.schedules] == ["w1"]

    @pytest.mark.asyncio
    async def test_cancel_by_id_can_address_one_wait(self) -> None:
        scheduler = FakeScheduler(
            [patience_row(id="patience-1"), patience_row(id="patience-2", episode_id="patience-2")]
        )
        cancelled, error = await P.cancel_patience(scheduler, row_id="patience-2")
        assert cancelled == ["patience-2"] and error == ""
        assert [r.id for r in scheduler.schedules] == ["patience-1"]

    @pytest.mark.asyncio
    async def test_an_unknown_id_reports_the_known_ids(self) -> None:
        scheduler = FakeScheduler([patience_row()])
        cancelled, error = await P.cancel_patience(scheduler, row_id="patience-9")
        assert cancelled == [] and "patience-1" in error
        assert scheduler.updates == []


class TestRetireAll:
    def test_retire_all_keeps_scheduled_rows(self) -> None:
        kept, cancelled = P.retire_all(
            [patience_row(), WakeSchedule(id="w2", message="m", next_due_at=NOW)]
        )
        assert [r.id for r in kept] == ["w2"] and cancelled == ["patience-1"]


# ---------------------------------------------------------------------------
# Fire-side helpers
# ---------------------------------------------------------------------------


class TestFireHelpers:
    def test_the_watermark_retires_a_fire_that_predates_the_reply(self) -> None:
        facts = P.EpisodeFacts(NOW + 1, None, NOW + 1)
        assert P.fire_is_stale(patience_row(armed_at=NOW), facts) is True

    def test_the_watermark_is_strict_at_the_same_millisecond(self) -> None:
        facts = P.EpisodeFacts(NOW, None, NOW)
        assert P.fire_is_stale(patience_row(armed_at=NOW), facts) is False

    def test_no_armed_at_never_reads_stale(self) -> None:
        facts = P.EpisodeFacts(NOW + 1, None, NOW + 1)
        assert P.fire_is_stale(patience_row(armed_at=0), facts) is False

    def test_ttl_is_measured_from_the_episode_start(self) -> None:
        old = patience_row(created_at=NOW - P.DEFAULT_TTL_MS - 1)
        assert P.fire_past_ttl(old, policy(), now_ms=NOW) is True
        assert P.fire_past_ttl(patience_row(), policy(), now_ms=NOW) is False

    def test_the_note_names_the_peer_target_and_the_elapsed_wait(self) -> None:
        note = P.fire_note(patience_row(armed_after="peer:builder"), policy(), now_ms=NOW + 300_000)
        assert "a reply from builder" in note
        assert "5m" in note
        assert "attempt 1 of 3" in note
        assert "internal timer note" in note
        assert "FINAL attempt" not in note

    def test_the_final_note_instructs_one_terminal_acknowledgement(self) -> None:
        note = P.fire_note(patience_row(attempt=3), policy(), now_ms=NOW + 2_700_000)
        assert "FINAL attempt" in note
        assert "ONE terminal acknowledgement" in note

    def test_fire_details_carry_the_episode_state(self) -> None:
        details = P.fire_details(patience_row(attempt=2))
        assert details["kind"] == "patience" and details["hidden"] is True
        assert details["episode_id"] == "patience-1"
        assert details["attempt"] == 2
        assert details["episode_started_at"] == NOW


def test_the_patience_predicates_are_one_implementation() -> None:
    """The predicate collapse review round 1 asked for (R3), pinned.

    Four call sites used to compare the ``kind`` literal themselves and two
    modules each owned a filter; both now route through ``wakes.store`` (with
    ``wakes.patience`` re-exporting it for its internal callers). This asserts
    the two spellings answer IDENTICALLY in both row shapes, so a future edit
    to one of them fails here rather than in whichever surface counts wakes.
    """
    from local_operator.wakes import store

    model_scheduled = WakeSchedule(id="w1", message="m", next_due_at=1)
    model_wait = WakeSchedule(id="p1", message="m", next_due_at=1, kind="patience", hidden=True)
    dict_scheduled = {"id": "w1", "kind": "scheduled"}
    dict_wait = {"id": "p1", "kind": "patience"}

    for row, expected in [
        (model_scheduled, False),
        (model_wait, True),
        (dict_scheduled, False),
        (dict_wait, True),
    ]:
        assert store.is_patience_row(row) is expected, row
        assert P.is_patience_row(row) is expected, row

    rows = [model_scheduled, model_wait, dict_scheduled, dict_wait]
    assert (
        P.scheduled_rows(rows)
        == store.scheduled_rows(rows)
        == [
            model_scheduled,
            dict_scheduled,
        ]
    )
