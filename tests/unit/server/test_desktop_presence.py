"""The machine-wide delivery presence: written, renewed, revoked, reaped.

The presence exists because every signal before it was per SESSION, so a
background session's completion raised a backend OS toast on whichever machine
the backend runs on — even while the desktop app was up and focused on a
different conversation.

Two properties decide whether it is safe to act on, and both are pinned here:

- **A claim is believed only while it is ANSWERING.** ``can_notify`` means "can
  attempt delivery", the same standard the per-session watch lease is held to,
  so a dropped socket revokes the claim and three missed beats expire it. A
  lease that outlives its app is the failure mode that would silence every
  runtime on the machine for a banner nobody can raise.
- **It is a machine-wide REACHABILITY answer, never a visibility one.** Its
  window state is read as "somebody is looking at this window", which is the
  only sound way to answer rung 1 for a surface the backend cannot see, and its
  ``session_id`` is discarded the moment no window exists.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import pytest

from local_operator.server.utils.desktop_presence import (
    DesktopDeliveryPublisher,
    PresenceClaim,
)
from local_operator.session.runtime.presence import (
    PRESENCE_TTL_S,
    delivery_path,
    desktop_delivery_present,
    desktop_presence,
    desktop_viewing_session,
    read_delivery,
    reset_cache,
)


@pytest.fixture(autouse=True)
def _clear_presence_cache():
    reset_cache()
    yield
    reset_cache()


def _beat(publisher: DesktopDeliveryPublisher, **overrides):
    window = {"exists": True, "focused": True, "visible": True, "minimized": False}
    window.update(overrides.pop("window", {}))
    publisher.update(
        "sub-1",
        can_notify=overrides.pop("can_notify", True),
        can_notify_kinds=overrides.pop("can_notify_kinds", ["complete", "error"]),
        session_id=overrides.pop("session_id", ""),
        window=window,
    )
    reset_cache()


def test_a_subscribed_app_publishes_an_aggregate_other_processes_can_read(tmp_path):
    """The route's whole purpose: a sibling process must be able to see it."""
    publisher = DesktopDeliveryPublisher(tmp_path)
    _beat(publisher, session_id="a" * 12)

    raw = json.loads(delivery_path(tmp_path).read_text())
    assert raw["pid"] == os.getpid()
    assert raw["can_notify"] is True
    assert raw["can_notify_kinds"] == ["complete", "error"]
    assert raw["subscribers"] == 1
    assert raw["session_id"] == "a" * 12
    assert raw["heartbeat_at"] > 0
    assert desktop_delivery_present(tmp_path, "complete") is True
    publisher.close()


def test_the_file_is_private_and_staged(tmp_path):
    """The permissions ARE the authorization story, copied from `viewers`."""
    publisher = DesktopDeliveryPublisher(tmp_path)
    _beat(publisher)
    directory = delivery_path(tmp_path).parent
    assert os.stat(directory).st_mode & 0o777 == 0o700
    assert os.stat(delivery_path(tmp_path)).st_mode & 0o777 == 0o600
    assert not [name for name in os.listdir(directory) if name.endswith(".tmp")]
    publisher.close()


def test_disconnect_revokes_the_lease_immediately(tmp_path):
    """THE LOAD-BEARING REVOCATION.

    Withdrawing only on a missed heartbeat would leave a 45 s window in which
    every runtime on the machine stays silent for a banner nobody can raise.
    The SSE socket is the liveness signal, so its teardown takes the claim.
    """
    publisher = DesktopDeliveryPublisher(tmp_path)
    _beat(publisher)
    assert desktop_delivery_present(tmp_path, "complete") is True

    publisher.drop("sub-1")
    reset_cache()
    assert desktop_delivery_present(tmp_path, "complete") is False
    assert not delivery_path(tmp_path).exists()
    publisher.close()


def test_can_notify_false_keeps_rung_two_ineligible(tmp_path):
    """A connected app that cannot notify must not silence the runtime."""
    publisher = DesktopDeliveryPublisher(tmp_path)
    _beat(publisher, can_notify=False)

    assert desktop_delivery_present(tmp_path, "complete") is False
    # It is still a LIVE claim, which is a different question rung 1 falls back
    # on; the two must not be conflated in either direction.
    assert desktop_presence(tmp_path).present is True
    publisher.close()


def test_the_presence_is_narrowed_by_kind(tmp_path):
    """The gate kind must keep its per-session lease and its per-session toast.

    The machine-wide feed carries completions only. A presence that claimed
    every kind would silence a background session's parked `ask` with nothing to
    replace it — a regression against today — so the wire carries the kinds and
    a reader asks for the one it is about to route.
    """
    publisher = DesktopDeliveryPublisher(tmp_path)
    _beat(publisher, can_notify_kinds=["complete", "error"])
    assert desktop_delivery_present(tmp_path, kind="complete") is True
    assert desktop_delivery_present(tmp_path, kind="error") is True
    assert desktop_delivery_present(tmp_path, kind="ask") is False
    assert desktop_delivery_present(tmp_path, kind="approval") is False
    publisher.close()


def test_a_claim_with_no_kinds_claims_nothing(tmp_path):
    """An app that forgot to advertise must not win rung 2 for an unchecked kind."""
    publisher = DesktopDeliveryPublisher(tmp_path)
    _beat(publisher, can_notify_kinds=[])
    assert desktop_delivery_present(tmp_path, kind="complete") is False
    publisher.close()


def test_a_dead_pid_is_reaped(tmp_path):
    """A `kill -9`ed server must not keep suppressing banners."""
    payload = {
        "pid": 999_999_999,
        "can_notify": True,
        "can_notify_kinds": ["complete"],
        "subscribers": 1,
        "window": {"exists": True, "focused": True, "visible": True, "minimized": False},
        "session_id": "b" * 12,
        "heartbeat_at": time.time(),
    }
    delivery_path(tmp_path).write_text(json.dumps(payload))
    reset_cache()

    assert read_delivery(tmp_path).present is False
    assert desktop_delivery_present(tmp_path, "complete") is False


def test_a_stale_heartbeat_is_reaped(tmp_path):
    """Three missed beats is a dead app, whatever its pid says."""
    payload = {
        "pid": os.getpid(),
        "can_notify": True,
        "can_notify_kinds": ["complete"],
        "subscribers": 1,
        "window": {"exists": True, "focused": True, "visible": True, "minimized": False},
        "session_id": "c" * 12,
        "heartbeat_at": time.time() - PRESENCE_TTL_S - 1.0,
    }
    delivery_path(tmp_path).write_text(json.dumps(payload))
    reset_cache()

    assert desktop_delivery_present(tmp_path, "complete") is False


def test_a_corrupt_file_is_the_absent_answer_not_an_exception(tmp_path):
    """This is read on a turn's announce path; a raise would cost a turn."""
    delivery_path(tmp_path).write_text("{not json")
    reset_cache()
    assert read_delivery(tmp_path).present is False
    delivery_path(tmp_path).write_text("[]")
    reset_cache()
    assert read_delivery(tmp_path).present is False


def test_the_window_state_is_what_rung_one_reads(tmp_path):
    """Attended means focused AND visible AND not minimised, and nothing less.

    A window behind another app is *reachable* (it can raise a banner) and is
    not *attended* (nobody is reading it). Deriving rung 1 from reachability is
    the defect this field exists to prevent.
    """
    publisher = DesktopDeliveryPublisher(tmp_path)
    _beat(publisher, session_id="d" * 12, window={"focused": False})
    assert desktop_viewing_session(tmp_path) == ""
    _beat(publisher, session_id="d" * 12, window={"visible": False})
    assert desktop_viewing_session(tmp_path) == ""
    _beat(publisher, session_id="d" * 12, window={"minimized": True})
    assert desktop_viewing_session(tmp_path) == ""
    _beat(publisher, session_id="d" * 12)
    assert desktop_viewing_session(tmp_path) == "d" * 12
    publisher.close()


def test_a_stale_session_id_on_a_windowless_app_is_ignored(tmp_path):
    """DESIGN REVIEW m2: a closed window cannot be displaying anything.

    The app survives in the macOS dock, and its record may still name the
    conversation it was showing. Honouring that id would make the backend treat
    a conversation the user cannot see as "the card is on screen".
    """
    publisher = DesktopDeliveryPublisher(tmp_path)
    _beat(publisher, session_id="e" * 12, window={"exists": False})

    assert desktop_viewing_session(tmp_path) == ""
    raw = json.loads(delivery_path(tmp_path).read_text())
    assert raw["session_id"] == ""
    # ...and it can still raise a banner, which is what keeps rung 2 eligible.
    assert desktop_delivery_present(tmp_path, "complete") is True
    publisher.close()


def test_the_beat_renews_and_the_reaper_expires(tmp_path):
    """A claim that stops arriving must expire; one that keeps arriving must not.

    Asserted directly on the publisher's own bookkeeping rather than by sleeping
    out a 15 s beat: what matters is that a claim older than the TTL is dropped
    BEFORE the aggregate is rewritten, so the writer never freshens a heartbeat
    on behalf of an app it has not heard from.
    """
    publisher = DesktopDeliveryPublisher(tmp_path)
    _beat(publisher)
    assert publisher.present() is True

    # Age the claim past the TTL without a beat, then run one reap.
    publisher.claims["sub-1"].seen_at = time.monotonic() - PRESENCE_TTL_S - 1.0
    publisher._reap()
    assert publisher.present() is False
    publisher._write()
    assert not delivery_path(tmp_path).exists()
    publisher.close()


def test_the_aggregate_takes_the_union_of_the_kinds_on_offer(tmp_path):
    """Two windows are two claims; a kind either can deliver IS deliverable."""
    publisher = DesktopDeliveryPublisher(tmp_path)
    publisher.update(
        "sub-1",
        can_notify=True,
        can_notify_kinds=["complete"],
        window={"exists": True, "focused": False, "visible": True, "minimized": False},
    )
    publisher.update(
        "sub-2",
        can_notify=True,
        can_notify_kinds=["error"],
        window={"exists": True, "focused": True, "visible": True, "minimized": False},
    )
    reset_cache()
    assert publisher.kinds() == frozenset({"complete", "error"})
    assert desktop_delivery_present(tmp_path, "error") is True
    publisher.close()


def test_close_is_idempotent_and_withdraws(tmp_path):
    publisher = DesktopDeliveryPublisher(tmp_path)
    _beat(publisher)
    publisher.close()
    publisher.close()
    assert not delivery_path(tmp_path).exists()


def test_the_claim_dataclass_defaults_to_the_quiet_answer():
    """A default-constructed claim asserts nothing."""
    claim = PresenceClaim()
    assert claim.can_notify is False
    assert claim.kinds == frozenset()
    assert claim.has_window is False
    assert claim.session_id == ""


@pytest.mark.asyncio
async def test_the_beat_loop_writes_and_then_stops_when_nothing_is_claimed(tmp_path):
    """The timer half: it renews while a claim lives and stops when none does."""
    publisher = DesktopDeliveryPublisher(tmp_path)
    _beat(publisher)
    publisher._ensure_beat()
    assert publisher._beat_task is not None
    publisher.drop("sub-1")
    await asyncio.sleep(0)
    publisher.close()
    assert publisher._beat_task is None
