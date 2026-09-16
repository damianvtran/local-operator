"""The desktop notification frame: when it is emitted, and when it must not be.

The defect this contract closes is a false statement on a lock screen. The app
toasted "Turn complete — The agent finished its turn." on every `turn_end` (one
MODEL CALL, of which an agentic turn has many) and on every `agent_end` (the
parent's own end, which routinely arrives while `task` children are still
working). Both assert something untrue, and the second one is the case the TUI
has always deliberately suppressed.

The fix is a change of AUTHORITY, not a better heuristic: the bridge emits a
notification if and only if it observes a newly published, unseen `completions`
row. That row exists only because `Session._publish_attention_outcome` decided
the turn produced a notifiable outcome, inside the process that owns the job
manager, using the same delegated-children check the TUI uses. So the bridge
cannot disagree with the TUI about whether a turn finished — it is reading a
projection of one decision rather than making the same decision twice.

Two further properties are pinned here because each has a failure mode that is
invisible until it hits a user:

- **A notification is not replayed.** An edge whose value is timeliness must
  not arrive hours later on a reconnect, and it must not consume one of the 256
  replay slots a real transcript event needs.
- **Claiming a banner is not marking it read.** `POST /notified` writes the
  DELIVERY watermark only. Routing it through the read watermark would clear
  the sidebar's unseen mark for a conversation the user never opened.

Cold-path style throughout, mirroring `test_desktop_attention.py`: no runtime
is started and no bridge is acquired by anything that does not need one.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import uuid
from pathlib import Path
from typing import Any

import pytest

from local_operator.notifications import NOTIFICATION_CONTRACT_VERSION
from local_operator.server.utils.desktop_sessions import (
    BRIDGE_NOTIFIABLE_KINDS,
    DesktopSessionBridge,
    DesktopSessions,
)
from local_operator.session.attention import AttentionStore
from local_operator.tui.notify import BODIES


def _publish(
    root: Path,
    session_id: str,
    anchor: str,
    kind: str = "complete",
    *,
    baseline_seen: bool | None = None,
) -> str:
    token = str(uuid.uuid4())
    AttentionStore(root / "attention.db").publish(
        f"session/{session_id}", token, anchor, kind, baseline_seen=baseline_seen
    )
    return token


class _Bridge(DesktopSessionBridge):
    """A bridge with a recording publisher and no runtime behind it.

    The notification edge lives in `refresh_attention`, which reads the store
    and publishes — neither of which needs an attached session. Driving the
    real method over a real store while recording frames is what makes the
    ORDER (`attention` then `notification`) and the COUNT observable, and both
    are properties no assertion on a single call could see.
    """

    def __init__(self, root: Path, session_id: str) -> None:
        super().__init__(root, session_id, str(root))
        self.frames: list[tuple[dict[str, Any], bool]] = []

    def publish(self, kind: str, payload: dict[str, Any], *, replay: bool = True) -> None:
        super().publish(kind, payload, replay=replay)
        self.frames.append(
            (
                {
                    "session_id": self.session_id,
                    "epoch": self.epoch,
                    "seq": self.sequence,
                    "type": kind,
                    "payload": payload,
                },
                replay,
            )
        )

    @property
    def kinds(self) -> list[str]:
        return [frame["type"] for frame, _ in self.frames]

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [frame for frame, _ in self.frames if frame["type"] == kind]


def _session_dir(root: Path, session_id: str, *, assistant: str = "", title: str = "") -> None:
    """Give a created session a transcript the composer can read."""
    import json

    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": "user-1",
            "ts": 1.0,
            "type": "message",
            "payload": {"kind": "message", "role": "user", "content": [{"text": "go"}]},
        }
    ]
    if assistant:
        rows.append(
            {
                "id": "assistant-1",
                "ts": 2.0,
                "type": "message",
                "payload": {
                    "kind": "message",
                    "role": "assistant",
                    "content": [{"text": assistant}],
                },
            }
        )
    with (directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    if title:
        (directory / "title.json").write_text(
            json.dumps({"text": title, "user_set": True, "names": [title]}), encoding="utf-8"
        )


@pytest.fixture(autouse=True)
def names_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the privacy flag ON, so a developer's own config cannot decide a test."""
    monkeypatch.setattr("local_operator.tui.notify.settings_get", lambda key, default=None: True)


async def _baselined(root: Path, session_id: str) -> _Bridge:
    """A bridge that has taken its first attention reading, with frames cleared.

    Every test past T-B2 starts here, because the baseline read is what makes
    a later change NEWS rather than history.
    """
    bridge = _Bridge(root, session_id)
    await bridge.refresh_attention()
    bridge.frames.clear()
    return bridge


@pytest.mark.asyncio
async def test_a_published_completion_emits_one_notification_after_the_attention_frame(
    tmp_path: Path,
) -> None:
    """T-B1. The frame, its content, and its position relative to `attention`.

    Ordering is not cosmetic: a reader that toasts should already hold the
    receipt state that explains the toast, so a renderer never shows a banner
    for a completion its own sidebar has not yet marked unseen.

    This also pins the ORDERING RISK named in the design (§11.2 item 5): the
    composer reads the transcript while the session writes it, and the row it
    fires on is published AFTER message persistence — so the snippet must be
    the real last line, not a degraded "Task complete".
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    _session_dir(tmp_path, sid, assistant="Rebuilt the index.", title="Index work")
    bridge = await _baselined(tmp_path, sid)

    _publish(tmp_path, sid, "result-1")
    await bridge.refresh_attention()

    assert bridge.kinds == ["attention", "notification"]
    payload = bridge.of("notification")[0]["payload"]
    assert payload["contract"] == NOTIFICATION_CONTRACT_VERSION
    assert payload["kind"] == "complete"
    assert payload["title"] == "Index work"
    assert payload["status"] == "Complete"
    assert payload["body"] == "Rebuilt the index."
    assert payload["body_is_snippet"] is True
    assert payload["title_is_session_name"] is True
    assert payload["session_name"] == "Index work"
    assert payload["focus_policy"] == "when_unfocused"
    # The dedupe key is bound to the DURABLE token, not to this bridge's
    # sequence: `acquire()` resets the sequence after a detached interval, so a
    # seq-keyed client would re-toast the same completion on every reconnect.
    # Literal `complete:` on this side, kind-relative on the error side: the
    # prefix is the frame's own kind, never a hardcoded row-class label.
    assert payload["dedupe_key"] == f"complete:{sid}:{payload['completion_token']}"
    assert payload["completion_token"] == bridge.attention["completion_token"]


@pytest.mark.asyncio
async def test_the_first_read_of_a_bridges_life_announces_nothing(tmp_path: Path) -> None:
    """T-B2. Opening a session must not toast the completion it ended on.

    The baseline rule is the same one the `attention` frame already uses: a
    bridge's first read is the session's HISTORY, not news. Without it, opening
    the app would raise a banner for every session that finished while it was
    closed — which is the flood the whole feature is shaped to avoid.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    _session_dir(tmp_path, sid, assistant="Finished last week.")
    _publish(tmp_path, sid, "already-there")

    bridge = _Bridge(tmp_path, sid)
    await bridge.refresh_attention()

    assert bridge.attention["unseen"] is True, "the completion IS outstanding"
    assert bridge.frames == [], "and it is still not news"


@pytest.mark.asyncio
async def test_the_same_completion_seen_twice_emits_one_frame(tmp_path: Path) -> None:
    """T-B3. Revision churn must not re-announce a completion.

    `refresh_attention` publishes on full-state inequality, and a heal
    deliberately republishes a corrected state — so a state change with an
    unchanged token is routine and must not reach the wire as a second banner.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    _session_dir(tmp_path, sid, assistant="Done.")
    bridge = await _baselined(tmp_path, sid)

    token = _publish(tmp_path, sid, "result-1")
    await bridge.refresh_attention()
    assert len(bridge.of("notification")) == 1

    # Force a state change that keeps the token: the runtime going live flips
    # `supported`, which is exactly the churn the emitter must ignore.
    bridge.attention = dict(bridge.attention, supported=not bridge.attention["supported"])
    await bridge.refresh_attention()
    # And an idempotent re-read of the identical state.
    await bridge.refresh_attention()

    assert len(bridge.of("notification")) == 1
    assert bridge.of("notification")[0]["payload"]["completion_token"] == token


@pytest.mark.asyncio
async def test_an_interruption_is_not_announced_on_the_desktop(tmp_path: Path) -> None:
    """T-B4. The user pressed the key themselves a moment ago.

    Telling somebody their own Ctrl+C worked is the definition of a
    notification nobody wants, and it is the call the TUI already makes. The
    `attention` frame still goes out, because the SIDEBAR should still show the
    outcome — the suppression is of the interruption, not of the fact.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    _session_dir(tmp_path, sid, assistant="Half-finished.")
    bridge = await _baselined(tmp_path, sid)

    _publish(tmp_path, sid, "result-1", kind="interrupted")
    await bridge.refresh_attention()

    assert bridge.kinds == ["attention"]
    assert "interrupted" not in BRIDGE_NOTIFIABLE_KINDS
    assert BRIDGE_NOTIFIABLE_KINDS == frozenset({"complete", "error"})


@pytest.mark.asyncio
async def test_an_error_is_announced_with_the_house_sentence(tmp_path: Path) -> None:
    """`error` IS notifiable, and never borrows the last assistant line.

    The pair with T-B4: the two non-success kinds are treated differently on
    purpose, because a failure is something the user must act on and an
    interruption is something they just did.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    _session_dir(tmp_path, sid, assistant="All 412 tests pass.", title="Release prep")
    bridge = await _baselined(tmp_path, sid)

    _publish(tmp_path, sid, "completion-abc", kind="error")
    await bridge.refresh_attention()

    payload = bridge.of("notification")[0]["payload"]
    assert payload["kind"] == "error"
    assert payload["status"] == "Needs attention"
    assert payload["body"] == BODIES["error"]
    assert payload["body_is_snippet"] is False
    # The dedupe prefix is the frame's own kind (round 1, n1), not a hardcoded
    # `complete:` — paired with T-B1's literal, the two reachable notifiable
    # kinds pin the property from both sides.
    assert payload["dedupe_key"].startswith(f"error:{sid}:")
    assert "412 tests pass" not in str(payload)


@pytest.mark.asyncio
async def test_an_agent_end_event_never_produces_a_notification(tmp_path: Path) -> None:
    """T-B5. The anti-regression test for the reported defect's second half.

    `agent_end` still reaches the wire as an `event` frame — several renderer
    paths settle the transcript on it, so suppressing it to fix a toast would
    break painting on every UI version. What changes is that it is no longer a
    NOTIFICATION trigger: the parent's own end routinely lands while `task`
    children are still working.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    bridge = await _baselined(tmp_path, sid)

    from local_operator.harness.types import AgentEndEvent

    bridge._event(AgentEndEvent(aborted=False))

    assert bridge.kinds == ["event"]
    assert bridge.of("event")[0]["payload"]["type"] == "agent_end"
    assert bridge.of("notification") == []


@pytest.mark.asyncio
async def test_a_turn_end_event_never_produces_a_notification(tmp_path: Path) -> None:
    """T-B6. The loudest half of the defect: one model call is not a turn.

    An agentic turn is many `turn_end`s, so this fired a banner per STEP. The
    frame still travels as an `event` for the transcript; it simply has nothing
    to do with whether the user owes the session anything.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    bridge = await _baselined(tmp_path, sid)

    from local_operator.harness.types import TurnEndEvent

    for _ in range(5):
        bridge._event(TurnEndEvent(message=None, tool_results=[]))

    assert bridge.kinds == ["event"] * 5
    assert bridge.of("notification") == []


@pytest.mark.asyncio
async def test_a_notification_is_never_retained_for_replay(tmp_path: Path) -> None:
    """T-B7. An edge whose value is timeliness must not arrive hours late.

    Replaying it would toast the user about a turn that finished while their
    laptop lid was shut. The durable signal is not lost: `attention` and the
    sidebar's unseen mark both survive a reconnect and are the right surface
    for "you missed something".

    It also costs nothing from the 256-frame replay budget, where every slot it
    took would push out a real transcript event.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    _session_dir(tmp_path, sid, assistant="Done.")
    bridge = await _baselined(tmp_path, sid)

    from local_operator.harness.types import TurnEndEvent

    bridge._event(TurnEndEvent(message=None, tool_results=[]))
    _publish(tmp_path, sid, "result-1")
    await bridge.refresh_attention()
    bridge._event(TurnEndEvent(message=None, tool_results=[]))

    retained = [frame["type"] for frame, _ in bridge.replay]
    assert "notification" not in retained
    assert retained == ["event", "attention", "event"]
    # The transcript events AROUND it are retained, which is the half that
    # would break if `replay=False` were applied too broadly.
    assert len(bridge.of("notification")) == 1


@pytest.mark.asyncio
async def test_seq_still_advances_and_a_reconnect_at_it_is_not_gapped(tmp_path: Path) -> None:
    """T-B8. The gap arithmetic survives a seq that was never retained.

    `events()` computes `gap` from `after_seq < first - 1`, where `first` is
    the oldest RETAINED frame's seq. A skipped seq therefore never becomes
    `first`, and a client reconnecting at exactly the notification's cursor
    still satisfies the test against the next retained frame. Not incrementing
    would instead make `seq` non-monotonic across the two publish paths and
    break every renderer's receipt cursor.

    If this is wrong the symptom is a spurious full-snapshot refetch after
    every completion, which a user sees as a transcript flash.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    _session_dir(tmp_path, sid, assistant="Done.")
    bridge = await _baselined(tmp_path, sid)

    from local_operator.harness.types import TurnEndEvent

    bridge._event(TurnEndEvent(message=None, tool_results=[]))
    _publish(tmp_path, sid, "result-1")
    await bridge.refresh_attention()
    bridge._event(TurnEndEvent(message=None, tool_results=[]))

    seqs = [frame["seq"] for frame, _ in bridge.frames]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs), "seq must stay monotonic"
    notification_seq = bridge.of("notification")[0]["seq"]
    assert notification_seq not in [frame["seq"] for frame, _ in bridge.replay]

    # Reconnecting AT the notification's cursor is not a gap, and the frames
    # after it are replayed.
    first = bridge.replay[0][0]["seq"]
    cutoff = bridge.sequence
    assert not (notification_seq < first - 1 or notification_seq > cutoff)
    replayed = [f["seq"] for f, _ in bridge.replay if notification_seq < f["seq"] <= cutoff]
    assert replayed == [cutoff]


@pytest.mark.asyncio
async def test_a_claim_is_taken_once_and_the_second_caller_loses(tmp_path: Path) -> None:
    """T-B9. `claim_delivery` is the arbiter; exactly one surface ever wins.

    A losing claim is the FEATURE, not a bug: when both a TUI observer and the
    desktop are eligible for the same completion, one of them silently drops
    its banner and the user is told once.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    token = _publish(tmp_path, sid, "result-1")

    assert await pool.claim_notification(sid, token) is True
    assert await pool.claim_notification(sid, token) is False

    # A NEWER completion is a separate claim, so a delivered conversation is
    # not permanently silenced.
    newer = _publish(tmp_path, sid, "result-2")
    assert await pool.claim_notification(sid, newer) is True


@pytest.mark.asyncio
async def test_claiming_a_banner_never_marks_anything_read(tmp_path: Path) -> None:
    """T-B10. The two watermarks, kept apart on a real store.

    Notifying is cheap and reversible; marking-read is destructive. Routing
    this through `acknowledge` would clear the sidebar's checkmark for a
    conversation the user never opened — which `docs/ATTENTION.md` forbids
    outright and `docs/SESSION_SIDEBAR.md` restates.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    token = _publish(tmp_path, sid, "result-1")
    store = AttentionStore(tmp_path / "attention.db")

    before = store.state(f"session/{sid}")
    assert await pool.claim_notification(sid, token) is True
    after = store.state(f"session/{sid}")

    assert after == before, "the claim must not move ANY field of the read state"
    assert after["unseen"] is True
    assert after["revision"] == before["revision"]

    # And the read watermark still works afterwards, so the claim did not
    # merely fail to write: it wrote somewhere else.
    assert (await pool.acknowledge_attention(sid, token))["unseen"] is False


@pytest.mark.asyncio
async def test_a_claim_never_acquires_a_bridge_or_starts_a_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-B11. Cold path, proven by making the warm path explode.

    A completion worth announcing is usually one whose owner has already
    exited. Spawning a process to decide whether to show a banner would be a
    remarkable cost for chrome — and `DesktopSessions.session()` is the only
    other way in, so an exploding `session()` is what keeps this true if
    someone later "simplifies" the implementation.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    token = _publish(tmp_path, sid, "result-1")

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("a notification claim must not acquire a session bridge")

    monkeypatch.setattr(DesktopSessions, "session", forbidden)
    assert await pool.claim_notification(sid, token) is True
    assert not (tmp_path / "sessions" / sid / ".session.pid").exists()
    assert not pool.bridges


@pytest.mark.asyncio
async def test_an_unknown_or_foreign_token_claims_nothing(tmp_path: Path) -> None:
    """T-B12. A claim must not invent a watermark it had no completion for.

    Rejecting a well-formed token belonging to ANOTHER session matters more
    than rejecting garbage: the ids are uniform, so a mixed-up client would
    otherwise silence a conversation the user never opened.
    """
    pool = DesktopSessions(tmp_path)
    mine = await pool.create(str(tmp_path))
    other = tmp_path / "other"
    other.mkdir()
    theirs = await pool.create(str(other))
    foreign = _publish(tmp_path, theirs, "their-result")
    _publish(tmp_path, mine, "my-result")

    for bad in (foreign, str(uuid.uuid4()), "not-a-uuid"):
        assert await pool.claim_notification(mine, bad) is False

    store = AttentionStore(tmp_path / "attention.db")
    assert store.state(f"session/{mine}")["unseen"] is True
    assert store.state(f"session/{theirs}")["unseen"] is True
    # The other session's own completion is still claimable, so nothing above
    # consumed it.
    assert await pool.claim_notification(theirs, foreign) is True

    # An unknown SESSION is a 404 in the same shape the read receipt uses.
    for bogus in ("../../etc", "not-hex", "a" * 12, mine.upper(), "", "0123456789ab"):
        with pytest.raises(KeyError):
            await pool.claim_notification(bogus, foreign)


@pytest.mark.asyncio
async def test_a_compose_failure_costs_the_banner_not_the_attention_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T-B13. Chrome must not take out the receipt sync it rides on.

    `refresh_attention` is how the desktop, the phone and the TUI stay in step
    about what has been read. A composer that raised into it would stop that
    sync for the life of the bridge — a new way to hide an unread result, which
    is the failure the attention feature exists to prevent.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    _session_dir(tmp_path, sid, assistant="Done.")
    bridge = await _baselined(tmp_path, sid)

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("transcript unreadable")

    monkeypatch.setattr("local_operator.notifications.compose", explode)
    _publish(tmp_path, sid, "result-1")
    state = await bridge.refresh_attention()

    assert bridge.kinds == ["attention"], "the receipt frame still went out"
    assert state["unseen"] is True
    assert bridge.attention["completion_token"] is not None

    # And the bridge recovers on the next completion once compose works again.
    monkeypatch.undo()
    monkeypatch.setattr("local_operator.tui.notify.settings_get", lambda key, default=None: True)
    _publish(tmp_path, sid, "result-2")
    await bridge.refresh_attention()
    assert len(bridge.of("notification")) == 1


@pytest.mark.asyncio
async def test_a_new_token_arriving_already_acknowledged_is_not_news(tmp_path: Path) -> None:
    """T-B14. A token the bridge has NEVER seen can still arrive pre-read.

    The resume-adoption path publishes with ``baseline_seen=True``
    (``attention.py:243``), which also writes the receipts row at the
    completion's own sequence: a brand-new token lands with ``unseen: false``.
    Every OTHER guard clause passes here — the token is new and the kind
    notifiable — so without the ``unseen`` clause adoption would announce a
    completion the store already considers read, the T-B2 flood by another
    route. Pinned as a test because deleting that clause left this whole suite
    green (review round 1, m1).
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    _session_dir(tmp_path, sid, assistant="Ran while the desktop was detached.")
    bridge = await _baselined(tmp_path, sid)

    token = _publish(tmp_path, sid, "result-1", baseline_seen=True)
    state = await bridge.refresh_attention()

    # The guard's premises, proven rather than assumed: only the `unseen`
    # clause can be what held the banner back.
    assert state["completion_token"] == token
    assert state["unseen"] is False
    assert bridge.kinds == ["attention"], "the state frame went out, the banner did not"
    assert bridge.of("notification") == []


@pytest.mark.asyncio
async def test_the_live_name_beats_the_sidecar_on_the_wire(tmp_path: Path) -> None:
    """A rename reaches frontend state before it reaches `title.json`.

    The bridge passes the live `conversation_title` when a runtime is attached
    and falls back to the stored title for a cold bridge. A banner naming a
    session by the title it had ten seconds ago sends the user looking for a
    conversation that no longer exists under that name.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    _session_dir(tmp_path, sid, assistant="Done.", title="Stale sidecar name")
    bridge = await _baselined(tmp_path, sid)

    class _Remote:
        """The two attributes `refresh_attention` reads off an attached session."""

        is_cold = False
        supports_completion_ack = True
        frontend_state = type("_State", (), {"conversation_title": "Renamed live"})()

    bridge.remote = _Remote()  # type: ignore[assignment]
    _publish(tmp_path, sid, "result-1")
    await bridge.refresh_attention()

    assert bridge.of("notification")[0]["payload"]["title"] == "Renamed live"


@pytest.mark.asyncio
async def test_the_privacy_flag_is_honoured_on_the_wire_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The frame carries neither the name nor the snippet when the flag is off.

    The wire is the surface where a leak is least recoverable: once a
    conversation's content has left the backend it is in a renderer's memory,
    its logs and possibly its crash reports. The gate therefore runs before the
    payload is built, not in the renderer.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    _session_dir(tmp_path, sid, assistant="Merged the acquisition docs.", title="Project Atlas")
    bridge = await _baselined(tmp_path, sid)

    monkeypatch.setattr("local_operator.tui.notify.settings_get", lambda key, default=None: False)
    _publish(tmp_path, sid, "result-1")
    await bridge.refresh_attention()

    payload = bridge.of("notification")[0]["payload"]
    assert payload["title"] == "Local Operator"
    assert payload["body"] == BODIES["complete"]
    assert payload["session_name"] is None
    assert payload["title_is_session_name"] is False
    assert payload["body_is_snippet"] is False
    assert "Atlas" not in str(payload) and "acquisition" not in str(payload)


@pytest.mark.asyncio
async def test_the_poll_loop_delivers_the_notification_end_to_end(tmp_path: Path) -> None:
    """The 1 s poll is the real trigger; the hook must actually be reached by it.

    Every test above drives `refresh_attention` directly, which would pass with
    the hook connected to a method nothing calls. This one publishes into the
    store and waits for the frame to arrive through the production path: the
    revision-gated poll a live bridge runs.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    _session_dir(tmp_path, sid, assistant="The report is ready.", title="Weekly report")

    bridge = _Bridge(tmp_path, sid)
    poll = asyncio.create_task(bridge._poll_attention())
    try:
        for _ in range(400):
            if bridge.attention:
                break
            await asyncio.sleep(0.01)
        assert bridge.attention, "the poll never took its baseline reading"
        bridge.frames.clear()

        _publish(tmp_path, sid, "result-1")
        for _ in range(400):
            if bridge.of("notification"):
                break
            await asyncio.sleep(0.01)
    finally:
        poll.cancel()
        with contextlib.suppress(BaseException):
            await poll

    assert bridge.kinds == ["attention", "notification"]
    assert bridge.of("notification")[0]["payload"]["body"] == "The report is ready."


@pytest.mark.asyncio
async def test_a_store_error_during_the_edge_does_not_end_the_poll(tmp_path: Path) -> None:
    """A broken store costs one tick, with the notification hook installed.

    `test_desktop_attention.py` pins this for the attention read; repeated here
    because the hook added a second await inside the same try, and an exception
    escaping IT would end the loop just as effectively.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    _session_dir(tmp_path, sid, assistant="Done.")
    bridge = _Bridge(tmp_path, sid)
    poll = asyncio.create_task(bridge._poll_attention())
    try:
        _publish(tmp_path, sid, "result-1")
        for _ in range(400):
            if bridge.attention.get("completion_token"):
                break
            await asyncio.sleep(0.01)

        with contextlib.closing(sqlite3.connect(tmp_path / "attention.db")) as conn:
            conn.execute("DROP TABLE completions")
            conn.commit()
        bridge.attention_poll_key = None
        await asyncio.sleep(2.5)
        assert not poll.done(), "the poll loop died on a transient store error"
    finally:
        poll.cancel()
        with contextlib.suppress(BaseException):
            await poll


@pytest.mark.asyncio
async def test_an_error_frame_carries_the_failure_text(tmp_path: Path) -> None:
    """D4 on the wire: the banner names the cause, not just the state.

    `test_compose.py` pins the composition; this pins that the bridge puts the
    new field on the frame, because a renderer reads the payload and not the
    dataclass.
    """
    import json as _json

    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    _session_dir(tmp_path, sid, assistant="All 412 tests pass.", title="Release prep")
    directory = tmp_path / "sessions" / sid
    with (directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            _json.dumps(
                {
                    "id": "incident-1",
                    "ts": 3.0,
                    "type": "message",
                    "payload": {
                        "kind": "custom",
                        "custom_type": "session_incident",
                        "details": {
                            "text": "[session incident] rate-limit: ...",
                            "raw": "rate limit: quota exhausted for anthropic/claude-opus-5",
                        },
                    },
                }
            )
            + "\n"
        )
    bridge = await _baselined(tmp_path, sid)

    _publish(tmp_path, sid, "completion-abc", kind="error")
    await bridge.refresh_attention()

    payload = bridge.of("notification")[0]["payload"]
    assert payload["body"] == "rate limit: quota exhausted for anthropic/claude-opus-5"
    assert payload["body_is_failure"] is True
    assert payload["body_is_snippet"] is False
    assert "412 tests pass" not in str(payload)


def test_a_parked_gate_carries_the_session_name_for_a_banner() -> None:
    """D3: an anonymous gate banner cannot be triaged.

    A toast saying only "Waiting for approval" with three sessions open names
    none of them, so the user must open each one to find the run being held
    hostage — on the surface whose entire job is to save them that trip.

    Gates deliberately stay on the EXISTING `pending_gate` path rather than
    gaining a `notification` frame: the desktop already receives this card in
    the snapshot and update frames, and a second channel for one question is
    how one gate becomes two banners. So the fix is a field on the card, not a
    new wire.

    Driven through `ServingSessionHandle._publish_pending_gate` rather than by
    calling the helper, because what is under test is that the PUBLICATION
    stamps it — a helper nothing calls would leave every real gate anonymous.
    """
    from local_operator.mobile.types import PendingRequest
    from local_operator.session.frontend_state import PendingGateState

    published: dict[str, Any] = {}

    class _Store:
        def mutate(self, **changes: Any) -> None:
            published.update(changes)

    class _Session:
        conversation_name = "Quota reporting fix"
        _frontend_state_store = _Store()

    class _Projection:
        pending = PendingRequest(
            request_id="gate-1", kind="approval", title="write", detail="write: /etc/hosts"
        )

    from local_operator.session.runtime.serving import ServingSessionHandle

    handle = ServingSessionHandle.__new__(ServingSessionHandle)
    handle._session = _Session()  # type: ignore[attr-defined]
    handle._projection = _Projection()  # type: ignore[attr-defined]

    handle._publish_pending_gate()

    gate = published["pending_gate"]
    assert gate["session_name"] == "Quota reporting fix"
    # The card still validates as the canonical model, which is what makes the
    # field additive rather than a shape a strict consumer would reject.
    assert PendingGateState(**gate).session_name == "Quota reporting fix"
    # And the rest of the card is untouched.
    assert gate["kind"] == "approval" and gate["detail"] == "write: /etc/hosts"


def test_the_gates_session_name_obeys_the_privacy_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The privacy decision stays backend-owned on this path too.

    Only the backend can read `display.notification_session_name`, and a
    renderer re-deriving it is one that can get it wrong inside a signed binary
    the user updates on their own schedule. Empty is also the OLD shape, so the
    opt-out degrades to exactly the anonymous card every existing viewer draws.
    """
    from local_operator.mobile.types import PendingRequest
    from local_operator.session.frontend_state import PendingGateState
    from local_operator.session.runtime.serving import ServingSessionHandle

    published: dict[str, Any] = {}

    class _Store:
        def mutate(self, **changes: Any) -> None:
            published.update(changes)

    class _Session:
        conversation_name = "Secret client migration"
        _frontend_state_store = _Store()

    class _Projection:
        pending = PendingRequest(request_id="gate-1", kind="ask", title="ask", detail="which env?")

    handle = ServingSessionHandle.__new__(ServingSessionHandle)
    handle._session = _Session()  # type: ignore[attr-defined]
    handle._projection = _Projection()  # type: ignore[attr-defined]

    monkeypatch.setattr("local_operator.tui.notify.settings_get", lambda key, default=None: False)
    handle._publish_pending_gate()

    gate = published["pending_gate"]
    assert gate["session_name"] == ""
    assert "Secret client" not in str(gate)
    # An OLD viewer sees a card it fully understands; a NEW viewer reading an
    # OLD payload (no key at all) gets the same empty default. Additive in both
    # skew directions, which is the property that lets this ship before the UI.
    assert PendingGateState(**gate).session_name == ""
    assert PendingGateState(request_id="g", kind="ask", title="t").session_name == ""
