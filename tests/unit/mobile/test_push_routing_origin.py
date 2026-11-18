"""ADR 0009 slice 1b: the push worker routes by ORIGIN first, presence second.

WHY (operator, 2026-11-17): "if started on mobile it should notify on mobile,
if started on other channels/desktop/remote it should notify on the desktop
unless we can tell the user is not there and there's an ask or something that
requires the user's attention, then notify on mobile".

The matrix pinned here (mobile repo ``docs/adr/0009-*`` §2), under the shipped
defaults ``completions_scope: phone`` and ``attention: away``:

- a mobile-origin completion or error pushes, is never held, and is resolved
  only when its card is on an attended desktop window (rung 1); it still
  pushes with the desktop's own banners switched off;
- a non-mobile completion never pushes under ``phone``; under ``all`` it keeps
  ADR 0008's hold;
- a non-mobile error is held while the desk is active and released when away;
- a run that ended on an ask it opened (``needs_user``) pushes whatever its
  ``notify`` and whatever "Agent finished" says: held at an active desk under
  ``away``, at once under ``always``, never under ``never``; the next run with
  the same ask still open carries ``needs_user = 0`` and stays silent.

The rig is ``test_push_worker_desk``'s: a real store, the in-memory transport,
desk reports materialised exactly as the server does.
"""

from __future__ import annotations

import pytest

from local_operator.notifications import routing as push_routing
from local_operator.server.utils.desktop_sessions import BRIDGE_NOTIFIABLE_KINDS
from tests.unit.mobile.test_push_worker import _types
from tests.unit.mobile.test_push_worker_desk import BEAT_S, DeskHarness


@pytest.fixture(autouse=True)
def _desktop_notifications_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same opt-in as ``test_push_worker``: the transport here is in-memory."""
    monkeypatch.delenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", raising=False)


@pytest.fixture()
def desk(tmp_path) -> DeskHarness:  # noqa: ANN001
    harness = DeskHarness(tmp_path)
    harness.scope = "phone"  # the shipped default
    return harness


def _away(desk: DeskHarness) -> None:
    """No input for longer than the 120 s threshold: the desk goes away."""
    desk.idle_beats(120 + BEAT_S)


# -- mobile origin -----------------------------------------------------------------


def test_a_mobile_completion_pushes_while_the_desk_is_active(desk: DeskHarness) -> None:
    desk.desk.beat(idle_s=0)
    desk.publish("phone-run", origin="mobile")
    assert [record.kind for record in desk.worker.tick()] == ["completion"]


def test_a_mobile_error_pushes_while_the_desk_is_active(desk: DeskHarness) -> None:
    desk.desk.beat(idle_s=0)
    desk.publish("phone-fail", kind="error", origin="mobile")
    assert _types(desk.plane.calls) == [] and desk.worker.tick()
    assert [body["kind"] for _key, body in desk.plane.calls] == ["error"]


def test_a_mobile_completion_on_an_attended_window_is_resolved(desk: DeskHarness) -> None:
    desk.desk.beat(idle_s=0)
    desk.desk.focused = True
    desk.desk.viewing = "on-screen"
    desk.publish("on-screen", origin="mobile")
    assert desk.worker.tick() == []
    assert desk.cursor()["publication_cursor"] == 1, "resolved, not held"
    _away(desk)
    assert desk.plane.calls == [], "never pushed later either"


def test_a_mobile_completion_pushes_with_desktop_banners_off(
    desk: DeskHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``notifications_enabled()`` is the desktop's own kill switch; a user who
    turned desktop banners off still wants the phone to hear about phone work.
    The negative control: a non-mobile row under ``all`` is still silenced."""
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    desk.scope = "all"
    desk.publish("desk-run", origin="desktop")
    desk.publish("phone-run", origin="mobile")
    desk.worker.tick()
    assert [body["kind"] for _key, body in desk.plane.calls] == ["complete"]
    assert len(desk.plane.calls) == 1


# -- non-mobile completions ------------------------------------------------------------


@pytest.mark.parametrize("origin", ["desktop", "tui", "remote", "unknown", ""])
def test_a_non_mobile_completion_never_pushes_under_phone(desk: DeskHarness, origin: str) -> None:
    desk.publish("desk-run", origin=origin)
    desk.worker.tick()
    _away(desk)
    assert desk.plane.calls == []
    assert desk.cursor()["publication_cursor"] == 1, "resolved without an emit"


def test_a_desktop_completion_is_held_under_all_and_released_on_away(
    desk: DeskHarness,
) -> None:
    desk.scope = "all"
    desk.desk.beat(idle_s=0)
    desk.publish("desk-run", origin="desktop")
    assert desk.worker.tick() == [], "ADR 0008's hold"
    _away(desk)
    assert _types(desk.plane.calls) == ["completion"]


# -- non-mobile errors -------------------------------------------------------------


def test_a_desktop_error_is_held_and_released_on_away(desk: DeskHarness) -> None:
    desk.desk.beat(idle_s=0)
    desk.publish("desk-fail", kind="error", origin="desktop")
    assert desk.worker.tick() == []
    assert desk.plane.calls == []
    _away(desk)
    assert [body["kind"] for _key, body in desk.plane.calls] == ["error"]


# -- needs you -----------------------------------------------------------------------


def test_a_wake_run_ending_on_an_open_ask_pushes_when_away_with_completions_off(
    desk: DeskHarness,
) -> None:
    """The R2-1 headline row: ``notify = false`` (a quiet wake), "Agent
    finished: Off", desk away (no report at all reads as not active)."""
    desk.overrides = {"completions": False}
    desk.publish("wake", notify=False, needs_user=True, origin="unknown")
    desk.worker.tick()
    assert _types(desk.plane.calls) == ["completion"]


def test_needs_you_is_held_at_an_active_desk_under_away_and_released(
    desk: DeskHarness,
) -> None:
    desk.desk.beat(idle_s=0)
    desk.publish("wake", notify=False, needs_user=True, origin="desktop")
    assert desk.worker.tick() == []
    _away(desk)
    assert _types(desk.plane.calls) == ["completion"]


def test_needs_you_pushes_at_once_under_always(desk: DeskHarness) -> None:
    desk.overrides = {"attention": "always"}
    desk.desk.beat(idle_s=0)
    desk.publish("wake", notify=False, needs_user=True, origin="desktop")
    desk.worker.tick()
    assert _types(desk.plane.calls) == ["completion"]


def test_needs_you_never_pushes_under_never(desk: DeskHarness) -> None:
    desk.overrides = {"attention": "never"}
    desk.publish("wake", notify=False, needs_user=True, origin="desktop")
    desk.worker.tick()
    _away(desk)
    assert desk.plane.calls == []


def test_the_same_ask_still_open_on_a_later_run_does_not_push_again(
    desk: DeskHarness,
) -> None:
    """The session stamps ``needs_user`` only for an ask opened in the run's
    window, so a later wake run with the same ask open publishes
    ``needs_user = 0``, ``notify = 0``: the worker adds no dedup and the row
    falls through to ``notify`` like any row. A run opening a NEW ask pushes."""
    desk.publish("wake", notify=False, needs_user=True, origin="unknown")
    desk.worker.tick()
    assert len(desk.plane.calls) == 1
    desk.publish("wake", notify=False, needs_user=False, origin="unknown")
    desk.worker.tick()
    assert len(desk.plane.calls) == 1, "same ask, later run: silent"
    desk.publish("wake", notify=False, needs_user=True, origin="unknown")
    desk.worker.tick()
    assert len(desk.plane.calls) == 2, "a new ask pushes"


def test_needs_you_answers_to_attention_not_completions_in_the_preference_gate(
    desk: DeskHarness,
) -> None:
    """Edit 4: "Agent finished: Off" must never mute a question, and the master
    switch still does."""
    desk.overrides = {"completions": False}
    desk.publish("ask", needs_user=True, origin="mobile")
    desk.worker.tick()
    assert len(desk.plane.calls) == 1
    desk.overrides = {"enabled": False}
    desk.publish("ask-2", needs_user=True, origin="mobile")
    desk.worker.tick()
    assert len(desk.plane.calls) == 1


# -- the shared predicate ----------------------------------------------------------


def test_the_routing_kinds_are_the_desktops() -> None:
    assert push_routing.NOTIFIABLE_KINDS == BRIDGE_NOTIFIABLE_KINDS


def _row(**fields: object) -> dict[str, object]:
    return {"unseen": True, "kind": "complete", "notify": True, "origin": "mobile", **fields}


def test_phone_would_emit_follows_the_devices(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    """One predicate for the worker and the desktop: no live device, or a
    category off on every live device, means the phone would not emit, and
    then the desktop must banner (the item is never silent everywhere)."""
    from local_operator.mobile import push_devices

    monkeypatch.setattr(push_routing, "live_device_ids", lambda _root: ["d1"])
    prefs = {"d1": dict(push_devices.PREFERENCE_DEFAULTS)}
    monkeypatch.setattr(push_devices, "device_preferences", lambda _root: prefs)
    assert push_routing.phone_would_emit(_row(), tmp_path)
    assert push_routing.desktop_defers_to_phone(_row(), tmp_path)
    assert not push_routing.desktop_defers_to_phone(_row(origin="desktop"), tmp_path)
    assert not push_routing.desktop_defers_to_phone(
        _row(needs_user=True), tmp_path
    ), "needs-you is never suppressed on the desktop"
    prefs["d1"]["completions"] = False
    assert not push_routing.desktop_defers_to_phone(_row(), tmp_path)
    prefs["d1"]["completions"] = True
    monkeypatch.setattr(push_routing, "live_device_ids", lambda _root: [])
    assert not push_routing.desktop_defers_to_phone(_row(), tmp_path), "no device: desktop banners"
