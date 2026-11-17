"""ADR 0008 slice 1: the push worker holds and releases on DESK presence.

Before this, "attended" was window focus only, which produced the operator's
complaint (G1: at the desk in another app, the phone buzzed with every desktop
banner) and its mirror (G2: window left focused, user walked away, a banner to
an empty desk dropped the phone push for good). The rule pinned here is ADR
0008 §3.3:

1. hold at arrival only if ``desk_active(now)`` AND this desk can banner the kind;
2. otherwise push now (lock, stale, untrusted, cannot-banner);
3. read anywhere while held → drop;
4. seen at the desk → drop, and "seen" is POSITIVE evidence (a desktop banner
   claim or the card on screen) PLUS input after it;
5. desk went away → release, through the same burst/digest rules.

A report with no desk fields keeps today's focus rule for one release.

Every desk report here is materialised exactly as the server does it
(``materialise_desk_report``) on the harness's WALL clock, which is also the
worker's clock: one machine, two processes, one ``time.time``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.mobile import push_worker as push_worker_module
from local_operator.mobile.push_worker import SEEN_GRACE_S, PushWorker
from local_operator.session.runtime.presence import (
    PRESENCE_TTL_S,
    DeskReport,
    DesktopPresence,
    materialise_desk_report,
)
from tests.unit.mobile.test_push_worker import COMPUTER, TICK_S, Harness, _types

#: The app's beat cadence (``presence.PRESENCE_BEAT_S``), spelled here so a test
#: reads as "one beat later".
BEAT_S = 15.0


@pytest.fixture(autouse=True)
def _desktop_notifications_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same opt-in as ``test_push_worker``: the transport here is in-memory."""
    monkeypatch.delenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", raising=False)


class Desk:
    """The desktop app's reports, as the server would materialise them.

    ``beat`` is one ``POST /v1/desktop/presence`` with desk fields, stamped on
    the harness clock at receipt. ``legacy`` sits an OLD app at the desk: window
    state only, no desk fields.
    """

    def __init__(self, harness: "DeskHarness") -> None:
        self.harness = harness
        self.reports: tuple[DeskReport, ...] = ()
        self.kinds = frozenset({"complete", "error"})
        self.focused = False
        self.viewing = ""

    def beat(
        self,
        *,
        idle_s: int = 0,
        locked: bool = False,
        idle_trusted: bool = True,
        away_after_s: int = 120,
    ) -> None:
        self.reports = (
            materialise_desk_report(
                received_at=self.harness.clock(),
                idle_s=idle_s,
                locked=locked,
                idle_trusted=idle_trusted,
                away_after_s=away_after_s,
            ),
        )

    def legacy(self, *, focused: bool) -> None:
        self.reports = ()
        self.focused = focused

    def __call__(self) -> DesktopPresence:
        return DesktopPresence(
            present=True,
            deliverable=True,
            kinds=self.kinds,
            session_id=self.viewing,
            attended=self.focused,
            has_window=True,
            desk=self.reports,
        )

    def attending(self, session_id: str) -> bool:
        return self().attending(session_id)


class DeskHarness(Harness):
    """``Harness`` with the desk seam instead of the legacy bool presence."""

    def __init__(self, root: Path) -> None:
        self.desk = Desk(self)
        super().__init__(root)

    def build(self) -> PushWorker:
        return PushWorker(
            store=self.store,
            transport=self.plane,
            config_dir=self.root,
            computer=COMPUTER,
            clock=self.clock,
            desk=self.desk,
            live_devices=self.devices,
            attending=self.desk.attending,
            preferences=self.preferences,
        )

    def idle_beats(self, seconds: float, *, since_input: float = 0.0) -> None:
        """Beat every 15 s for ``seconds`` with NO input; tick after each beat."""
        elapsed = 0.0
        while elapsed < seconds:
            self.clock.advance(BEAT_S)
            elapsed += BEAT_S
            self.desk.beat(idle_s=int(since_input + elapsed))
            self.worker.tick()


@pytest.fixture()
def desk(tmp_path: Path) -> DeskHarness:
    return DeskHarness(tmp_path)


def _claim_banner(harness: DeskHarness, session_id: str, token: str) -> None:
    """The desktop banner's own claim path (``claim_notification``)."""
    assert harness.store.claim_delivery(f"session/{session_id}", token, "desktop")


# -- G1: at the desk, in another app ---------------------------------------------


def test_g1_at_the_desk_in_another_app_holds_then_drops_after_banner_and_input(
    desk: DeskHarness,
) -> None:
    desk.desk.beat(idle_s=0)  # typing in the editor; lop window NOT focused
    token = desk.publish("g1")
    assert desk.worker.tick() == [], "held: the desktop will banner it"

    _claim_banner(desk, "g1", token)
    desk.clock.advance(TICK_S)
    assert desk.worker.tick() == [], "surfaced, but no input after it yet"

    desk.clock.advance(BEAT_S)
    desk.desk.beat(idle_s=0)  # still typing: input after the banner
    assert desk.worker.tick() == []
    assert desk.cursor()["publication_cursor"] == 1, "seen at the desk: resolved"

    desk.idle_beats(10 * 60)
    assert desk.plane.calls == [], "walking away later never pushes a seen item"


def test_g1_input_within_the_seen_grace_is_not_evidence(desk: DeskHarness) -> None:
    """``last_input_at`` must be STRICTLY past ``surfaced_at + SEEN_GRACE_S``."""
    desk.desk.beat(idle_s=0)
    token = desk.publish("grace")
    desk.worker.tick()
    _claim_banner(desk, "grace", token)
    desk.worker.tick()  # surfaced_at = now
    desk.desk.beat(idle_s=0)  # input AT surfaced_at, inside the grace
    desk.worker.tick()
    assert desk.cursor()["publication_cursor"] == 0, "still held"

    desk.clock.advance(SEEN_GRACE_S + 1)
    desk.desk.beat(idle_s=0)
    desk.worker.tick()
    assert desk.cursor()["publication_cursor"] == 1


def test_g1_the_card_on_screen_plus_input_is_seen(desk: DeskHarness) -> None:
    """Rung 1 is positive evidence too, but it no longer drops on its own."""
    desk.desk.beat(idle_s=0)
    desk.desk.focused = True
    desk.desk.viewing = "onscreen"
    desk.publish("onscreen")
    assert desk.worker.tick() == []
    assert desk.cursor()["publication_cursor"] == 0, "evidence, not yet seen"

    desk.clock.advance(BEAT_S)
    desk.desk.beat(idle_s=0)
    desk.worker.tick()
    assert desk.cursor()["publication_cursor"] == 1


# -- G2: window left focused, user walked away -----------------------------------


def test_g2_a_banner_to_an_empty_desk_does_not_drop_the_push(desk: DeskHarness) -> None:
    desk.desk.focused = True
    desk.desk.beat(idle_s=0)
    token = desk.publish("g2")
    assert desk.worker.tick() == []
    _claim_banner(desk, "g2", token)  # bannered... to nobody
    desk.worker.tick()
    assert desk.plane.calls == []

    # No input from here on. away_at = last input + 120 s.
    desk.idle_beats(105)
    assert desk.plane.calls == [], "inside the threshold: still held"
    desk.idle_beats(BEAT_S, since_input=105)
    assert _types(desk.plane.calls) == ["completion"], "released once away"


def test_g2_the_operators_threshold_moves_the_release(desk: DeskHarness) -> None:
    desk.desk.focused = True
    desk.desk.beat(idle_s=0, away_after_s=300)
    desk.publish("slow")
    desk.worker.tick()
    for _ in range(19):  # 285 s of beats, idle growing, threshold 300 s
        desk.clock.advance(BEAT_S)
        desk.desk.beat(idle_s=int(desk.clock() - 1000.0), away_after_s=300)
        desk.worker.tick()
    assert desk.plane.calls == []
    desk.clock.advance(BEAT_S)
    desk.desk.beat(idle_s=int(desk.clock() - 1000.0), away_after_s=300)
    desk.worker.tick()
    assert _types(desk.plane.calls) == ["completion"]


# -- G3, G10, stale, untrusted: push now ---------------------------------------


def test_g3_a_locked_screen_pushes_at_once(desk: DeskHarness) -> None:
    desk.desk.beat(idle_s=0, locked=True)
    desk.publish("g3")
    assert [record.kind for record in desk.worker.tick()] == ["completion"]


def test_g3_locking_while_held_releases(desk: DeskHarness) -> None:
    desk.desk.beat(idle_s=0)
    desk.publish("lock-later")
    assert desk.worker.tick() == []
    desk.clock.advance(TICK_S)
    desk.desk.beat(idle_s=0, locked=True)  # the app sends at once on lock
    assert [record.kind for record in desk.worker.tick()] == ["completion"]


def test_g10_a_desk_that_cannot_banner_the_kind_pushes_now(desk: DeskHarness) -> None:
    desk.desk.kinds = frozenset({"error"})
    desk.desk.beat(idle_s=0)
    desk.publish("g10", kind="complete")
    assert [record.kind for record in desk.worker.tick()] == ["completion"]


def test_a_stale_report_pushes_now(desk: DeskHarness) -> None:
    desk.desk.beat(idle_s=0)
    desk.clock.advance(PRESENCE_TTL_S + 1)  # three missed beats
    desk.publish("stale")
    assert [record.kind for record in desk.worker.tick()] == ["completion"]


def test_a_held_item_is_released_when_the_reports_go_stale(desk: DeskHarness) -> None:
    desk.desk.beat(idle_s=0)
    desk.publish("app-quit")
    assert desk.worker.tick() == []
    desk.clock.advance(PRESENCE_TTL_S + 1)
    assert [record.kind for record in desk.worker.tick()] == ["completion"]


def test_an_untrusted_report_pushes_now(desk: DeskHarness) -> None:
    desk.desk.beat(idle_s=0, idle_trusted=False)
    desk.publish("wayland")
    assert [record.kind for record in desk.worker.tick()] == ["completion"]


def test_an_untrusted_record_beside_an_idle_trusted_one_is_not_active(desk: DeskHarness) -> None:
    now = desk.clock()
    desk.desk.reports = (
        materialise_desk_report(
            received_at=now, idle_s=0, locked=False, idle_trusted=False, away_after_s=120
        ),
        materialise_desk_report(
            received_at=now, idle_s=600, locked=False, idle_trusted=True, away_after_s=120
        ),
    )
    desk.publish("m3")
    assert [record.kind for record in desk.worker.tick()] == ["completion"]


# -- rule 3 and 5 neighbours: read, burst ----------------------------------------


def test_a_read_during_the_hold_drops(desk: DeskHarness) -> None:
    desk.desk.beat(idle_s=0)
    token = desk.publish("read-on-phone")
    desk.worker.tick()
    desk.route_ack("session/read-on-phone", token)
    desk.worker.tick()
    desk.idle_beats(5 * 60)
    assert "completion" not in _types(desk.plane.calls)


def test_released_items_still_fold_into_one_digest(desk: DeskHarness) -> None:
    desk.desk.beat(idle_s=0)
    for index in range(5):
        desk.publish(f"burst-{index}")
    assert desk.worker.tick() == []
    desk.clock.advance(TICK_S)
    desk.desk.beat(idle_s=0, locked=True)
    records = desk.worker.tick()
    assert [record.kind for record in records] == ["digest"]
    assert desk.plane.calls[0][1]["count"] == 5


# -- version skew: no desk fields keeps today's focus rule ----------------------


def test_a_legacy_report_keeps_the_focus_rule(desk: DeskHarness) -> None:
    """Row "old app, new core": focus holds, a banner claim drops (as #4101)."""
    desk.desk.legacy(focused=True)
    token = desk.publish("legacy")
    assert desk.worker.tick() == []
    _claim_banner(desk, "legacy", token)
    desk.worker.tick()
    assert desk.cursor()["publication_cursor"] == 1, "legacy: the claim alone drops"


def test_a_legacy_unfocused_report_pushes_now(desk: DeskHarness) -> None:
    desk.desk.legacy(focused=False)
    desk.publish("legacy-unfocused")
    assert [record.kind for record in desk.worker.tick()] == ["completion"]


def test_the_production_default_reads_the_desk_presence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No seam injected: the worker reads ``desktop_presence(cached=False)``."""
    harness = Harness(tmp_path, arm=False)
    calls: list[bool] = []

    def fake_presence(root=None, *, cached: bool = True) -> DesktopPresence:
        calls.append(cached)
        return DesktopPresence(
            present=True,
            deliverable=True,
            kinds=frozenset({"complete"}),
            desk=(
                materialise_desk_report(
                    received_at=harness.clock(),
                    idle_s=0,
                    locked=False,
                    idle_trusted=True,
                    away_after_s=120,
                ),
            ),
        )

    monkeypatch.setattr(push_worker_module, "desktop_presence", fake_presence)
    harness.worker = PushWorker(
        store=harness.store,
        transport=harness.plane,
        config_dir=harness.root,
        computer=COMPUTER,
        clock=harness.clock,
        live_devices=harness.devices,
    )
    harness.arm()
    harness.publish("default-path")
    assert harness.worker.tick() == [], "held on the desk report, not on focus"
    assert calls and not any(calls), "uncached: the decision is terminal"
