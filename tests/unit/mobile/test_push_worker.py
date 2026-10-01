"""The push worker: the cursors, the gates, the queue and the two keys.

Push/ack-sync S5 of ADR 0006. The exit criteria this file is written against are
the slice row's own: an ``unseen`` + ``notify`` + no-presence completion emits
exactly once; a suppressed one is deferred and then emitted (or terminated by an
ack) and never dropped silently; a restart neither re-pushes the backlog nor
drops what landed while down; a catch-up above the burst limit emits ONE digest;
a cloud 5xx neither stalls a turn nor grows unbounded; and ``deliveries`` is
untouched before and after.

The store is REAL (a real ``attention.db`` under a tmp root), the clock is
injected, and the control plane is a recorder — which is the whole point of the
transport seam: the design is exercised with no cloud in existence.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from local_operator.mobile import daemon as daemon_module
from local_operator.mobile import push_worker as worker_module
from local_operator.mobile.daemon import MobileDaemon
from local_operator.mobile.push_payload import (
    ALERT_BODY_FIELD,
    ALERT_FIELD,
    ALERT_TITLE_FIELD,
    TYPE_DIGEST,
    attention_emit_key,
    completion_emit_key,
    digest_emit_key,
)
from local_operator.mobile.push_worker import (
    DEFERRAL_WINDOW_S,
    EMIT_RETRY_ATTEMPTS,
    EMIT_RETRY_INTERVAL_S,
    PRESENCE_RECHECK_S,
    PUSH_WORKER_STATE_NAME,
    EmitAccepted,
    EmitRefused,
    PushWorker,
    state_path,
)
from local_operator.session.attention import AttentionStore, provisional_anchor
from local_operator.tui.notify import APP_NAME, BODIES

COMPUTER = "computer-handle"

#: The daemon's scan cadence, for the passes a test counts rather than waits for
#: (``daemon.SCAN_INTERVAL_S``, spelled here so this file's arithmetic does not
#: import the loop's own constant).
TICK_S = 2.0


class Clock:
    """A hand-driven clock, because every bound here is a duration."""

    def __init__(self, at: float = 1000.0) -> None:
        self.at = at

    def __call__(self) -> float:
        return self.at

    def advance(self, seconds: float) -> None:
        self.at += seconds


class ControlPlane:
    """The stub control plane: records every emit, answers what the test says.

    ``verdicts`` is consumed one per call; the last one repeats. The default is
    the ADR's accept, ``202 {emit_id, accepted_at}``.
    """

    def __init__(self, *, verdicts: Sequence[EmitAccepted | EmitRefused] | None = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._verdicts: list[EmitAccepted | EmitRefused] = list(verdicts or [])

    def __call__(
        self, body: Mapping[str, Any], *, idempotency_key: str
    ) -> EmitAccepted | EmitRefused:
        self.calls.append((idempotency_key, dict(body)))
        if self._verdicts:
            verdict = self._verdicts.pop(0) if len(self._verdicts) > 1 else self._verdicts[0]
        else:
            verdict = EmitAccepted(emit_id=uuid.uuid4().hex, accepted_at=1)
        assert isinstance(verdict, (EmitAccepted, EmitRefused))
        return verdict

    @property
    def keys(self) -> list[str]:
        return [key for key, _body in self.calls]


class Presence:
    """The gate's input, so a test can sit the user at the computer."""

    def __init__(self, attended: bool = False) -> None:
        self.attended = attended

    def __call__(self) -> bool:
        return self.attended


class Devices:
    """§2.3 gate 4: whether any device of this computer may be delivered to."""

    def __init__(self, live: bool = True) -> None:
        self.live = live

    def __call__(self) -> bool:
        return self.live


class Harness:
    """A worker over a real store, with every input the tests need to steer."""

    def __init__(
        self,
        root: Path,
        *,
        attended: bool = False,
        live: bool = True,
        arm: bool = True,
        verdicts: Sequence[EmitAccepted | EmitRefused] | None = None,
    ) -> None:
        self.root = root
        (root / "sessions").mkdir(parents=True, exist_ok=True)
        self.store = AttentionStore(root / "attention.db")
        self.plane = ControlPlane(verdicts=verdicts)
        self.clock = Clock()
        self.presence = Presence(attended)
        self.devices = Devices(live)
        self.worker = self.build()
        if arm:
            # CONSTRUCTING THE HARNESS IS ENABLEMENT. Push is on from here, so
            # the baseline is taken now and a test's ``publish`` lands in an
            # armed worker; the pre-enablement test turns it off and arms by
            # hand, which is the only difference between the two cases.
            self.arm()

    def arm(self) -> None:
        """The enabling pass: the first tick, which takes the baseline."""
        assert self.worker.tick() == []

    def build(self) -> PushWorker:
        """A worker for this root — a new instance IS a daemon restart."""
        return PushWorker(
            store=self.store,
            transport=self.plane,
            config_dir=self.root,
            computer=COMPUTER,
            clock=self.clock,
            presence=self.presence,
            live_device=self.devices,
        )

    def restart(self) -> PushWorker:
        """Replace the worker with a fresh one over the same durable state."""
        self.worker = self.build()
        return self.worker

    def cursor(self) -> dict[str, Any]:
        import json

        return json.loads(state_path(self.root).read_text())

    def publish(
        self,
        session_id: str,
        *,
        kind: str = "complete",
        notify: bool = True,
        token: str | None = None,
        anchor: str | None = None,
    ) -> str:
        token = token or str(uuid.uuid4())
        self.store.publish(
            f"session/{session_id}",
            token,
            anchor or f"entry-{token[:8]}",
            kind,
            notify=notify,
        )
        return token


@pytest.fixture()
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


def _types(calls: Sequence[tuple[str, dict[str, Any]]]) -> list[str]:
    return [str(body["type"]) for _key, body in calls]


# -- the eligibility gate and the cursor --------------------------------------


def test_a_completion_emits_exactly_once_with_the_content_key(harness: Harness) -> None:
    """The first exit criterion, and the §3.4 key recipe on the real body."""
    token = harness.publish("alpha")
    records = harness.worker.tick()

    assert len(records) == 1
    assert records[0].accepted is True
    assert records[0].status == 202
    assert len(harness.plane.calls) == 1
    key, body = harness.plane.calls[0]
    anchor = str(harness.store.state("session/alpha")["anchor_id"])
    assert key == completion_emit_key(token, anchor, "complete")
    assert completion_emit_key(token, anchor, "complete") != completion_emit_key(
        token, anchor, "interrupted"
    ), "a heal must mint a different key"
    assert body["type"] == "completion"
    assert body["v"] == 1
    assert body["completion_token"] == token
    assert body["kind"] == "complete"
    assert body["computer"] == COMPUTER
    # The handle, never the session id.
    assert body["conversation"] != "alpha"
    assert len(str(body["conversation"])) == 22

    # The cursor moved to the position the accept covered...
    assert harness.cursor()["publication_cursor"] == 1
    # ...so the next tick has nothing to do.
    assert harness.worker.tick() == []
    assert len(harness.plane.calls) == 1


def test_nothing_published_before_enablement_is_ever_pushed(tmp_path: Path) -> None:
    """The baseline. ``unseen`` is a LEVEL, so a worker with no baseline would
    push this machine's whole history on its first tick."""
    harness = Harness(tmp_path, arm=False)
    harness.publish("old-one")
    harness.publish("old-two")

    harness.arm()
    assert harness.plane.calls == []
    assert harness.cursor()["publication_cursor"] == 2

    harness.publish("new-one")
    assert len(harness.worker.tick()) == 1
    assert harness.plane.calls[0][1]["count"] == 3


def test_a_completion_that_is_not_notify_resolves_without_an_emit(harness: Harness) -> None:
    """Gate 2. The row is passed over — not deferred, because no push describes
    it — and the cursor must not sit behind it forever."""
    harness.publish("quiet", notify=False)

    assert harness.worker.tick() == []
    assert harness.plane.calls == []
    assert harness.cursor()["publication_cursor"] == 1


def test_a_completion_already_read_elsewhere_resolves_without_an_emit(harness: Harness) -> None:
    """Gate 1, and the reason §2.3 checks ``unseen`` at send time.

    The ack is not silent, and must not be: it is also the attention emit's
    trigger (§2.1's third read), so the phone's badge gets corrected. What it
    must not produce is a completion push for something already read.
    """
    token = harness.publish("read-already")
    harness.store.acknowledge("session/read-already", token)

    records = harness.worker.tick()
    assert "completion" not in _types(harness.plane.calls)
    assert [record.kind for record in records] == ["attention"]
    assert harness.cursor()["publication_cursor"] == 1


def test_a_computer_with_no_live_device_emits_nothing(harness: Harness) -> None:
    """Gate 4 is a SKIP, not a deferral: there is no device the event could
    reach, and holding the cursor for a device that may never pair would both
    block the position and push a backlog the ADR forbids."""
    harness.devices.live = False
    harness.publish("nobody-to-tell")

    assert harness.worker.tick() == []
    assert harness.plane.calls == []
    assert harness.cursor()["publication_cursor"] == 1


# -- the presence deferral ----------------------------------------------------


def test_presence_defers_then_emits_when_the_presence_goes_away(harness: Harness) -> None:
    """§2.3 decision 2: a suppressed push is DEFERRED, never dropped silently."""
    harness.presence.attended = True
    harness.publish("suppressed")

    assert harness.worker.tick() == []
    assert harness.plane.calls == []
    # Nothing has moved: the position is still ahead of the cursor, which is
    # what makes a restart re-consider the item.
    assert harness.cursor()["publication_cursor"] == 0

    harness.clock.advance(PRESENCE_RECHECK_S - 1)
    assert harness.worker.tick() == []

    harness.clock.advance(1)
    harness.presence.attended = False
    assert len(harness.worker.tick()) == 1
    assert _types(harness.plane.calls) == ["completion"]
    assert harness.cursor()["publication_cursor"] == 1


def test_presence_defers_to_the_window_bound_then_emits_anyway(harness: Harness) -> None:
    """The 5-minute bound: the deferral is a delay, not a suppression.

    The window is measured from the pass that SAW the completion, not from the
    publish: the worker is the observer, and it notices within one scan
    interval.
    """
    harness.presence.attended = True
    harness.publish("still-here")
    assert harness.worker.tick() == []

    harness.clock.advance(DEFERRAL_WINDOW_S - 1)
    assert harness.worker.tick() == []

    harness.clock.advance(1)
    assert len(harness.worker.tick()) == 1, "the window bound must release the push"


def test_an_ack_terminates_the_deferral_without_an_emit(harness: Harness) -> None:
    """§2.3: reading the conversation in the app is exactly what makes
    ``unseen`` false, so the deferral ends and the completion is never sent."""
    harness.presence.attended = True
    token = harness.publish("read-in-the-app")
    assert harness.worker.tick() == []

    harness.store.acknowledge("session/read-in-the-app", token)
    harness.clock.advance(PRESENCE_RECHECK_S)

    # The ack is not silent — it is the attention emit's own trigger — but no
    # completion push for a completion the user has already read.
    records = harness.worker.tick()
    assert [record.kind for record in records] == ["attention"]
    assert _types(harness.plane.calls) == ["attention"]
    assert harness.cursor()["publication_cursor"] == 1


def test_a_deferred_completion_survives_a_restart(harness: Harness) -> None:
    """A daemon restart interrupts the deferral, and the item is re-considered
    from the cursor rather than lost (ADR §2.1, round 2 M2)."""
    harness.presence.attended = True
    harness.publish("deferred")

    assert harness.worker.tick() == []
    harness.restart()
    harness.presence.attended = False

    assert len(harness.worker.tick()) == 1
    assert _types(harness.plane.calls) == ["completion"]


# -- restart, catch-up and the burst ceiling ----------------------------------


def test_a_restart_neither_repushes_nor_drops_what_landed_while_down(harness: Harness) -> None:
    """The cursor test, driven with a real store and a simulated down-time."""
    harness.publish("before")
    assert len(harness.worker.tick()) == 1
    assert harness.cursor()["publication_cursor"] == 1

    # The daemon goes away; a completion lands while it is down.
    harness.restart()
    harness.publish("while-down")

    records = harness.worker.tick()
    assert len(records) == 1, "exactly the completion that landed while down"
    assert len(harness.plane.calls) == 2, "the accepted one is not re-emitted"
    assert (
        harness.plane.calls[1][1]["completion_token"]
        == harness.store.state("session/while-down")["completion_token"]
    )
    assert harness.cursor()["publication_cursor"] == 2


def test_a_catch_up_above_the_burst_limit_is_one_digest(harness: Harness) -> None:
    """> BURST_LIMIT eligible rows in one pass is ONE emit naming the count —
    and, since the lane's ruling, a VISIBLE one: its own type, not the attention
    form's silent wake."""
    for index in range(worker_module.BURST_LIMIT + 1):
        harness.publish(f"burst-{index}")

    records = harness.worker.tick()

    assert len(records) == 1, "the whole catch-up is exactly one emit"
    key, body = harness.plane.calls[0]
    assert body["type"] == worker_module.EVENT_DIGEST == TYPE_DIGEST
    assert body["count"] == worker_module.BURST_LIMIT + 1
    assert key == digest_emit_key(body["emit_id"], COMPUTER)
    assert "exclude" not in body, "a tick-detected batch excludes nobody"
    assert harness.cursor()["publication_cursor"] == worker_module.BURST_LIMIT + 1
    assert (
        harness.cursor()["attention_sequence"] == 0
    ), "a digest is not sequence-keyed and must not burn the attention counter"
    assert body[ALERT_FIELD] == {
        ALERT_TITLE_FIELD: APP_NAME,
        ALERT_BODY_FIELD: "Complete · 4 conversations need you",
    }, "a uniform batch says its real state, and the count is the batch's"
    for name in ("conversation", "completion_token", "kind"):
        assert name not in body, f"a digest spans conversations and names none: {name}"


def test_the_alert_rides_the_visible_types_and_never_the_silent_one(harness: Harness) -> None:
    """The lane's ruling: the machine writes the text for completion and digest,
    and an attention push carries none at all — it is a silent badge correction.
    """
    token = harness.publish("alert-me")
    assert [record.kind for record in harness.worker.tick()] == ["completion"]
    alert = harness.plane.calls[0][1][ALERT_FIELD]
    assert alert[ALERT_TITLE_FIELD] == APP_NAME
    assert alert[ALERT_BODY_FIELD].endswith("1 conversation needs you")
    assert alert[ALERT_BODY_FIELD].startswith(BODIES["complete"])

    harness.store.acknowledge("session/alert-me", token)
    assert [record.kind for record in harness.worker.tick()] == ["attention"]
    assert ALERT_FIELD not in harness.plane.calls[1][1], "the silent form has no text"


def test_a_restart_mid_window_reuses_the_same_digest_key(tmp_path: Path) -> None:
    """§3.4 as merged (``cc2569a4``): the window's ``emit_id`` is DURABLE.

    The daemon dies with a digest in flight and its ``202`` lost, which is the
    case the cloud's dedupe exists for: the retry after the restart must wear
    the SAME ``Idempotency-Key``. A digest is the one VISIBLE type, so a second
    key would put the same banner on the user's lock screen twice.
    """
    harness = Harness(tmp_path, verdicts=[EmitRefused(status=503)])
    for index in range(worker_module.BURST_LIMIT + 2):
        harness.publish(f"window-{index}")

    assert [record.kind for record in harness.worker.tick()] == ["digest"]
    window = harness.cursor()["digest_window"]
    assert isinstance(window, dict), "the closed window is persisted as it is minted"
    assert window["publications"] == [1, 2, 3, 4, 5], "the positions ARE the window"

    restarted = harness.restart()
    assert [record.kind for record in restarted.tick()] == ["digest"]

    assert (
        harness.plane.keys[0] == harness.plane.keys[1]
    ), "a restart mid-window re-emits under the SAME key"
    assert harness.plane.calls[0][1]["emit_id"] == harness.plane.calls[1][1]["emit_id"]


def test_a_new_burst_mints_a_new_window_and_key(tmp_path: Path) -> None:
    """The other direction (§3.4): the id is per WINDOW, not per machine — a
    genuine second burst must never be deduped away by the cloud."""
    harness = Harness(tmp_path)
    for index in range(worker_module.BURST_LIMIT + 1):
        harness.publish(f"burst-one-{index}")
    harness.worker.tick()
    assert harness.cursor()["digest_window"] is None, "the 202 closes the window"

    for index in range(worker_module.BURST_LIMIT + 1):
        harness.publish(f"burst-two-{index}")
    harness.worker.tick()

    assert len(harness.plane.calls) == 2
    assert harness.plane.keys[0] != harness.plane.keys[1]
    assert harness.plane.calls[0][1]["emit_id"] != harness.plane.calls[1][1]["emit_id"]


def test_a_window_that_grew_is_not_the_same_window(tmp_path: Path) -> None:
    """Sameness is the POSITION SET, not merely "a window is pending".

    Rows that arrive while the window is in flight are part of the batch a
    restart re-folds, so the batch a restart re-reads has grown — and the cloud
    must see that as a new delivery rather than dedupe away the rows the first
    frame never named.
    """
    harness = Harness(tmp_path, verdicts=[EmitRefused(status=503)])
    for index in range(worker_module.BURST_LIMIT + 2):
        harness.publish(f"grown-{index}")
    harness.worker.tick()
    assert harness.cursor()["digest_window"]["publications"] == [1, 2, 3, 4, 5]

    for index in range(2):
        harness.publish(f"grown-later-{index}")

    restarted = harness.restart()
    assert [record.kind for record in restarted.tick()] == ["digest"]
    assert harness.plane.calls[1][1]["count"] == 7, "the grown batch is what went out"
    assert harness.plane.keys[1] != harness.plane.keys[0], "a grown batch mints its own id"


def test_a_dropped_window_is_cleared_so_the_next_burst_mints_again(tmp_path: Path) -> None:
    """The third exit of the three (§3.4): the drop-with-log clears the window
    as surely as the ``202`` does, so the next batch cannot wear a key the cloud
    has already seen."""
    harness = Harness(tmp_path, verdicts=[EmitRefused(status=503)])
    for index in range(worker_module.BURST_LIMIT + 2):
        harness.publish(f"dropped-{index}")

    for _ in range(worker_module.EMIT_RETRY_ATTEMPTS):
        harness.worker.tick()
        harness.clock.advance(worker_module.EMIT_RETRY_INTERVAL_S)

    assert harness.worker.pending() == 0, "the batch was dropped at the bound"
    assert harness.cursor()["digest_window"] is None

    for index in range(worker_module.BURST_LIMIT + 2):
        harness.publish(f"after-the-drop-{index}")
    harness.worker.tick()
    assert harness.plane.keys[-1] not in harness.plane.keys[:-1], "a fresh window, a fresh key"


def test_a_refused_digest_retries_its_own_key_then_drops_at_the_bound(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """B1's first arm: the digest is a first-class retryable item.

    Before the fix the item stranded in the queue (no later pass could see it),
    so it was never retried, never dropped, and the cursor pinned for good.
    """
    harness = Harness(tmp_path, verdicts=[EmitRefused(status=503)])
    for index in range(worker_module.BURST_LIMIT + 2):
        harness.publish(f"refused-{index}")

    with caplog.at_level(logging.WARNING, logger=worker_module.__name__):
        first = harness.worker.tick()
        assert [record.status for record in first] == [503]
        assert harness.worker.pending() == 1, "one digest, and nothing else queued"
        assert harness.cursor()["publication_cursor"] == 0, "the cursor has not moved"

        # Too soon to retry: the wire interval is the ADR's, not the tick's.
        harness.clock.advance(worker_module.EMIT_RETRY_INTERVAL_S - 1)
        assert harness.worker.tick() == []

        for _ in range(worker_module.EMIT_RETRY_ATTEMPTS - 1):
            harness.clock.advance(1)
            assert len(harness.worker.tick()) == 1
            harness.clock.advance(worker_module.EMIT_RETRY_INTERVAL_S)

    assert len(harness.plane.calls) == worker_module.EMIT_RETRY_ATTEMPTS
    assert set(harness.plane.keys) == {
        harness.plane.keys[0]
    }, "every attempt wears the SAME key: a re-minted key cannot be deduped"
    drops = [record for record in caplog.records if "dropped" in record.message]
    assert len(drops) == 1, f"ONE drop line, not {len(drops)}"
    assert harness.worker.pending() == 0, "the queue is empty after the drop"
    assert (
        harness.cursor()["publication_cursor"] == worker_module.BURST_LIMIT + 2
    ), "the cursor advances once the batch is dropped"


def test_a_sustained_refusal_neither_grows_the_queue_nor_remints(tmp_path: Path) -> None:
    """B1's second arm, at the repro's scale: 60 passes under a permanent 503.

    The measured defect was the queue and the minted key space each growing by
    one per pass while the cursor pinned. Both must be flat now.
    """
    harness = Harness(tmp_path, verdicts=[EmitRefused(status=503)])
    for index in range(worker_module.BURST_LIMIT + 2):
        harness.publish(f"leak-{index}")

    for _ in range(75):
        harness.worker.tick()
        harness.clock.advance(TICK_S)

    assert (
        len(set(harness.plane.keys)) == 1
    ), f"one batch, one key: {sorted(set(harness.plane.keys))}"
    assert len(harness.plane.calls) == worker_module.EMIT_RETRY_ATTEMPTS
    assert harness.worker.pending() == 0
    assert harness.cursor()["publication_cursor"] == worker_module.BURST_LIMIT + 2


def test_a_pending_digest_keeps_its_rows_out_of_the_next_pass(tmp_path: Path) -> None:
    """The other half of B1: a pending digest COVERS its rows, so a later pass
    cannot re-derive the same batch as fresh completions — which would put one
    event on the wire under two identities while the first is still queued."""
    harness = Harness(tmp_path, verdicts=[EmitRefused(status=503)])
    for index in range(worker_module.BURST_LIMIT + 3):
        harness.publish(f"mixed-{index}")

    for _ in range(6):
        harness.worker.tick()
        harness.clock.advance(TICK_S)

    assert harness.worker.pending() == 1, "the digest is the only item in the queue"
    assert len(set(harness.plane.keys)) == 1


def test_a_catch_up_at_the_limit_is_not_coalesced(harness: Harness) -> None:
    """The boundary's other side, so the digest rule cannot drift to ``>=``."""
    for index in range(worker_module.BURST_LIMIT):
        harness.publish(f"exact-{index}")

    records = harness.worker.tick()

    assert len(records) == worker_module.BURST_LIMIT
    assert _types(harness.plane.calls) == ["completion"] * worker_module.BURST_LIMIT


def test_the_burst_limit_is_the_shared_one() -> None:
    """The ADR mirrors the TUI/desktop burst rule, and the two must not drift."""
    from local_operator.server.utils.desktop_feed import BURST_LIMIT

    assert worker_module.BURST_LIMIT == BURST_LIMIT


# -- the cloud's answer -------------------------------------------------------


def test_a_refusal_holds_the_cursor_and_retries_the_same_key(harness: Harness) -> None:
    """§2.1: the cursor advances ONLY on the cloud's accept, so a refused push
    survives a restart and is retried with the same key."""
    harness.plane = ControlPlane(verdicts=[EmitRefused(status=503)])
    harness.worker = harness.build()
    harness.publish("refused-once")

    records = harness.worker.tick()
    assert [record.accepted for record in records] == [False]
    assert harness.plane.keys == harness.plane.keys[:1]
    first_key = harness.plane.keys[0]
    # NOT advanced: the item is still ahead of the cursor.
    assert harness.cursor()["publication_cursor"] == 0
    assert [row["sequence"] for row in harness.store.published_since(0)] == [1]

    harness.clock.advance(EMIT_RETRY_INTERVAL_S)
    harness.worker.tick()
    assert harness.plane.keys == [first_key, first_key]
    assert harness.cursor()["publication_cursor"] == 0

    # The accept is what moves it.
    harness.plane._verdicts = [EmitAccepted(emit_id="e", accepted_at=2)]  # noqa: SLF001
    harness.clock.advance(EMIT_RETRY_INTERVAL_S)
    assert [record.accepted for record in harness.worker.tick()] == [True]
    assert harness.cursor()["publication_cursor"] == 1


def test_a_permanent_refusal_drops_the_item_with_one_log_line(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    """The bound that keeps one undeliverable item from blocking the position:
    3 attempts over ~2 minutes, then a drop, and the queue does not grow."""
    harness.plane = ControlPlane(verdicts=[EmitRefused(status=500)])
    harness.worker = harness.build()
    harness.publish("never-accepted")

    with caplog.at_level(logging.WARNING, logger=worker_module.__name__):
        for _ in range(EMIT_RETRY_ATTEMPTS):
            harness.worker.tick()
            harness.clock.advance(EMIT_RETRY_INTERVAL_S)

    assert len(harness.plane.calls) == EMIT_RETRY_ATTEMPTS
    assert len({key for key in harness.plane.keys}) == 1, "every retry keeps the key"
    assert len([r for r in caplog.records if "dropped a completion emit" in r.message]) == 1
    # Dropped, so the position moved and the item is not retried again.
    assert harness.cursor()["publication_cursor"] == 1
    assert harness.worker.tick() == []
    assert len(harness.plane.calls) == EMIT_RETRY_ATTEMPTS


def test_a_refused_emit_does_not_stall_the_next_one(harness: Harness) -> None:
    """A 5xx costs its own emit and nothing else: one item's refusal must not
    hold the rest of the tick."""
    harness.plane = ControlPlane(verdicts=[EmitRefused(status=500)])
    harness.worker = harness.build()
    harness.publish("first")
    harness.publish("second")

    records = harness.worker.tick()

    assert len(records) == 2
    assert harness.plane.keys[0] != harness.plane.keys[1]
    # The first is still pending and the second resolved, so the cursor sits
    # behind the first and both are re-read next tick — the blocking item is
    # bounded, not skipped.
    assert harness.cursor()["publication_cursor"] == 0


def test_deliveries_are_untouched_by_a_tick(harness: Harness) -> None:
    """§2.3 decision 1: the phone is not a rung, so the worker never claims."""
    harness.publish("no-claim")
    before = _deliveries(harness.store.path)

    assert len(harness.worker.tick()) == 1

    assert _deliveries(harness.store.path) == before


def _deliveries(path: Path) -> list[tuple[Any, ...]]:
    with sqlite3.connect(path) as conn:
        return [tuple(row) for row in conn.execute("SELECT * FROM deliveries ORDER BY 1")]


# -- heals, and the cursors the publication read cannot see --------------------


def test_a_heal_mints_a_new_key_on_the_supersede_cursor(harness: Harness) -> None:
    """§3.4: the completion key is the record's content, so a heal is a NEW
    delivery — and §2.1's supersede cursor is the read that sees it."""
    token = str(uuid.uuid4())
    harness.store.publish("session/healed", token, provisional_anchor(token), "interrupted")
    assert len(harness.worker.tick()) == 1
    first_key = harness.plane.keys[0]

    harness.store.publish("session/healed", token, "entry-9", "complete")
    records = harness.worker.tick()

    assert len(records) == 1, "the correction is a second delivery"
    second_key, body = harness.plane.calls[1]
    assert second_key != first_key
    assert body["kind"] == "complete"
    assert second_key == completion_emit_key(token, "entry-9", "complete")
    # The supersede cursor carried it, and the publication cursor never moved:
    # the heal rewrites the row in place.
    assert harness.cursor()["supersede_cursor"] == 1
    assert harness.cursor()["publication_cursor"] == 1


def test_a_heal_of_a_conversation_already_read_emits_nothing(harness: Harness) -> None:
    """A correction to something the user has already read is not a reason to
    buzz, and the cursor still moves past it."""
    token = str(uuid.uuid4())
    harness.store.publish("session/healed", token, provisional_anchor(token), "interrupted")
    harness.worker.tick()
    harness.store.acknowledge("session/healed", token)

    harness.store.publish("session/healed", token, "entry-9", "complete")
    records = harness.worker.tick()
    # No SECOND completion emit: the heal's own position resolved without one.
    assert _types(harness.plane.calls) == ["completion", "attention"]
    assert [record.kind for record in records] == ["attention"]
    assert harness.cursor()["supersede_cursor"] == 1


def test_a_pruned_heal_log_rebaselines_instead_of_flooding(
    harness: Harness, caplog: pytest.LogCaptureFixture
) -> None:
    """The store keeps 256 heals; a downtime longer than that loses their
    identity, and the ADR's answer is to re-baseline and say so — never to
    sweep."""
    heals = 300
    for index in range(heals):
        token = str(uuid.uuid4())
        conversation = f"session/heal-{index}"
        harness.store.publish(conversation, token, provisional_anchor(token), "interrupted")
        harness.store.publish(conversation, token, f"entry-{index}", "complete")

    with caplog.at_level(logging.WARNING, logger=worker_module.__name__):
        records = harness.worker.tick()

    assert harness.cursor()["supersede_cursor"] == heals
    assert [r for r in caplog.records if "pruned" in r.message], caplog.text
    # 300 new conversations in one pass is ONE digest, not 300 pushes.
    assert len(records) == 1


# -- the attention emit -------------------------------------------------------


def test_an_ack_emits_one_attention_push_keyed_on_the_sequence(harness: Harness) -> None:
    """§2.1's third read: an ack is neither a publication nor a heal, and the
    phone has to be told to re-read."""
    token = harness.publish("acked-elsewhere")
    harness.worker.tick()
    assert len(harness.plane.calls) == 1

    harness.store.acknowledge("session/acked-elsewhere", token)
    records = harness.worker.tick()

    assert len(records) == 1
    key, body = harness.plane.calls[1]
    assert body["type"] == "attention"
    assert key == attention_emit_key(1)
    assert "exclude" not in body, "a tick-detected change never knows who acked"
    assert body["count"] == 0


def test_a_restart_does_not_re_emit_a_past_ack(harness: Harness) -> None:
    """The acknowledgement map is baselined in memory at start, so a restart
    does not turn this machine's read history into a stream of badge pushes."""
    token = harness.publish("acked")
    harness.worker.tick()
    harness.store.acknowledge("session/acked", token)

    harness.restart()
    assert harness.worker.tick() == []
    assert len(harness.plane.calls) == 1


# -- the durable position -----------------------------------------------------


def test_the_cursor_file_is_private_and_survives_a_restart(harness: Harness) -> None:
    harness.publish("one")
    harness.worker.tick()

    path = state_path(harness.root)
    assert path.name == PUSH_WORKER_STATE_NAME
    assert (path.stat().st_mode & 0o777) == 0o600
    before = harness.cursor()

    harness.restart()
    harness.worker.tick()
    assert harness.cursor() == before


def test_a_corrupt_cursor_file_rebaselines_rather_than_repushing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unreadable position is not a licence to push this machine's history:
    the safe failure is to start from now, and to say so once."""
    harness = Harness(tmp_path, arm=False)
    state_path(tmp_path).write_text("{not json")
    harness.publish("long-ago")

    with caplog.at_level(logging.WARNING, logger=worker_module.__name__):
        assert harness.worker.tick() == []

    assert harness.plane.calls == []
    assert [r for r in caplog.records if "re-baselined" in r.message]
    assert harness.cursor()["publication_cursor"] == 1

    harness.publish("after-the-fault")
    assert len(harness.worker.tick()) == 1


def test_the_worker_never_writes_the_registry_or_the_handle_key_unasked(harness: Harness) -> None:
    """The worker's only writes are its own cursor file and the handle key it
    mints through S2's own function."""
    harness.publish("no-devices")
    harness.worker.tick()

    names = {path.name for path in harness.root.iterdir()}
    assert "mobile-push-devices.json" not in names
    assert {"mobile-push-worker.json", "push-handle.key"} <= names


# -- the daemon owns the loop -------------------------------------------------


def test_an_attention_emit_retries_on_the_wire_interval(tmp_path: Path) -> None:
    """m1: the badge correction obeys the same 60 s cadence as everything else.

    It used to be returned unconditionally, so a refused correction spent its
    three attempts in ~6 s — twenty times the frozen budget, on the one emit
    whose key cannot be re-derived from a record.
    """
    harness = Harness(tmp_path, verdicts=[EmitAccepted(emit_id="e", accepted_at=1)])
    token = harness.publish("acked-elsewhere")
    harness.worker.tick()  # the completion emit, accepted
    harness.store.acknowledge("session/acked-elsewhere", token)

    harness.plane._verdicts = [EmitRefused(status=500)]
    first = harness.worker.tick()
    assert [record.kind for record in first] == ["attention"]
    assert len(harness.plane.calls) == 2

    # Any number of ticks inside the interval must leave the wire alone.
    for _ in range(10):
        harness.clock.advance(TICK_S)
        assert harness.worker.tick() == []
    assert len(harness.plane.calls) == 2

    harness.clock.advance(EMIT_RETRY_INTERVAL_S)
    second = harness.worker.tick()
    assert [record.kind for record in second] == ["attention"]
    assert harness.plane.keys[1] == harness.plane.keys[2], "the same key again"


def test_a_tick_cannot_re_enter_a_pass_in_flight(harness: Harness) -> None:
    """The daemon cuts the scan loose on ``PUSH_TICK_TIMEOUT_S`` and cannot kill
    the thread, so a second pass must be refused rather than interleaved: the
    queue, the cursors and the key space are single-claimant state.

    THE CELL HAS TO GIVE THE NESTED PASS SOMETHING TO DO (review round 2, R2-1).
    It published nothing extra, so the nested tick was a no-op for the wrong
    reason — ``_due`` refused the in-flight item because :meth:`_attempt` stamps
    ``last_attempt_at`` before it calls the transport, which is round 1's m1 fix
    doing its job. The row published from INSIDE the transport is eligible on its
    own (no attempt, no deferral), so a nested pass that ran would emit it; and
    the lock itself is observed from inside the pass, which is the assertion the
    guard actually owns.
    """
    inner: list[list[worker_module.EmitRecord]] = []
    held: list[bool] = []
    nested: list[bool] = []
    plane = harness.plane

    def reentrant(body: Mapping[str, Any], *, idempotency_key: str):
        harness.publish("reentrant-extra")
        held.append(harness.worker.pass_held())
        if not nested:
            # ONE nested pass, and the flag is set BEFORE the call so a transport
            # that is itself driving a pass cannot recurse (which would report a
            # RecursionError instead of the assertion this cell is about).
            nested.append(True)
            inner.append(harness.worker.tick())
        return plane(body, idempotency_key=idempotency_key)

    harness.worker.transport = reentrant
    harness.publish("reentrant")

    records = harness.worker.tick()

    assert len(records) == 1, "the outer pass still emits"
    assert held == [True], "the guard is HELD while the pass runs"
    assert inner == [[]], "the inner pass did nothing at all"
    assert harness.worker.pending() == 0
    assert len(harness.plane.calls) == 1, "and it did not put a second emit on the wire"

    # The row the nested pass would have taken is still ahead of the cursor, so
    # the refusal held it back rather than losing it.
    assert [record.kind for record in harness.worker.tick()] == ["completion"]
    assert len(harness.plane.calls) == 2


def test_the_pass_guard_is_held_only_while_a_pass_runs(harness: Harness) -> None:
    """The discrimination for the cell above.

    ``pass_held()`` has to be a real observation: if the lock were taken for the
    worker's whole life the assertion inside the transport would hold for the
    wrong reason, and the mutant that removes the guard would keep it green.
    """
    assert harness.worker.pass_held() is False, "no pass is running here"
    harness.publish("guard-observation")
    harness.worker.tick()
    assert harness.worker.pass_held() is False, "and none is running afterwards"


def test_the_cursor_file_is_not_rewritten_when_nothing_moved(harness: Harness) -> None:
    """n1: a pass that changes no position must not rename an identical file.

    Measured across a RESTART, which is where the optimisation is armed: a fresh
    worker that reads the file back must know what is already on disk, or its
    first tick rewrites an identical file.
    """
    harness.publish("moves-the-position")
    harness.worker.tick()
    path = state_path(harness.root)

    restarted = harness.restart()
    before = path.stat().st_ino
    assert restarted.tick() == []
    assert path.stat().st_ino == before, "the first pass after a restart rewrote the file"

    harness.publish("moves-it-again")
    assert len(restarted.tick()) == 1
    assert path.stat().st_ino != before, "a real move still lands on disk"


def test_the_daemon_builds_one_session_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Q-2: ``__init__`` carried the same three assignments twice (a rebase
    artifact), so two ``SessionTable``s were built per daemon."""
    built: list[object] = []
    original = daemon_module.SessionTable

    class Counting(original):  # type: ignore[misc,valid-type]
        def __init__(self) -> None:
            super().__init__()
            built.append(self)

    monkeypatch.setattr(daemon_module, "SessionTable", Counting)
    daemon_module.MobileDaemon(port=0, password="pw")
    assert len(built) == 1, f"one table per daemon, built {len(built)} times"


def test_the_daemon_scan_loop_ticks_the_worker(harness: Harness, tmp_path: Path) -> None:
    """The slice is "in the mobile daemon": the 2 s loop is what runs it."""
    daemon = MobileDaemon(port=0, password="pw", push_worker=harness.worker)
    harness.publish("published-under-the-daemon")

    asyncio.run(daemon._scan_once())

    assert len(harness.plane.calls) == 1


def test_a_worker_fault_costs_one_tick_and_not_the_loop(tmp_path: Path) -> None:
    """A push fault must not take the daemon's scan with it."""

    class Exploding:
        def tick(self) -> list[object]:
            raise RuntimeError("the cloud is a smoking hole")

    daemon = MobileDaemon(port=0, password="pw", push_worker=Exploding())  # type: ignore[arg-type]
    asyncio.run(daemon._scan_once())
