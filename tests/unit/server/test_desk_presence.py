"""ADR 0008 slice 1: the desk report is accepted, materialised and read.

What is pinned here, and why each is load-bearing:

- **Version skew.** ``PresenceBeat`` is ``extra="forbid"``, so a field the core
  does not declare is a 422 the app swallows, and the lease then lapses. The
  four desk fields are declared and optional, and the feature key that tells an
  app it may send them is advertised. A beat WITHOUT them stays valid and
  publishes the same record it always did (no ``desk`` key at all).
- **Clock domain.** The server stamps ``received_at`` with WALL-clock
  ``time.time``; the push daemon is another process and compares with its own
  ``time.time``. A monotonic value must never cross (ADR m2).
- **Any-aggregation over trusted reports only.** An untrusted ``idle_s = 0``
  beside a trusted idle report is not active (ADR m3).
- **The shared conformance table** that agent-server copies byte-identical.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from local_operator.server.routes.desktop_sessions import PresenceBeat
from local_operator.server.utils import desktop_presence as publisher_module
from local_operator.server.utils.desktop_presence import DesktopDeliveryPublisher
from local_operator.session.runtime.presence import (
    AWAY_AFTER_CHOICES_S,
    DEFAULT_AWAY_AFTER_S,
    PRESENCE_TTL_S,
    DeskReport,
    DesktopPresence,
    delivery_record_path,
    materialise_desk_report,
    read_delivery,
    reset_cache,
)

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "desk_presence_cases.json"

#: The sha256 of the shared conformance table. agent-server asserts the same
#: digest over its copy (ADR 0008 §4 slice 1), so changing a case is a
#: deliberate two-repo change rather than silent drift between Python and Go.
FIXTURE_SHA256 = "9d54779b395cc286fa8d6c95f971fbf1e3b374d5769fc748500d0218b116e3e3"


@pytest.fixture(autouse=True)
def _clear_presence_cache():
    reset_cache()
    yield
    reset_cache()


def _beat(publisher: DesktopDeliveryPublisher, sub: str = "sub-1", **desk) -> None:
    publisher.update(
        sub,
        can_notify=True,
        can_notify_kinds=["complete", "error"],
        session_id="",
        window={"exists": True, "focused": False, "visible": True, "minimized": False},
        **desk,
    )


# -- version skew: the route model ---------------------------------------------


def test_todays_beat_without_desk_fields_is_still_valid() -> None:
    """Row "old app, new core": the beat an old app sends must keep validating."""
    beat = PresenceBeat.model_validate(
        {
            "subscription_id": "s",
            "can_notify": True,
            "can_notify_kinds": ["complete", "error"],
            "window": {"exists": True, "focused": True, "visible": True, "minimized": False},
        }
    )
    assert beat.idle_s is None
    assert beat.locked is None
    assert beat.idle_trusted is None
    assert beat.away_after_s is None


def test_a_beat_with_the_four_desk_fields_is_valid() -> None:
    beat = PresenceBeat.model_validate(
        {
            "subscription_id": "s",
            "can_notify": True,
            "idle_s": 12,
            "locked": False,
            "idle_trusted": True,
            "away_after_s": 300,
        }
    )
    assert (beat.idle_s, beat.locked, beat.idle_trusted, beat.away_after_s) == (
        12,
        False,
        True,
        300,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("idle_s", -1),
        ("idle_s", "5"),
        ("locked", "yes"),
        ("idle_trusted", 1),
        ("away_after_s", 90),
        ("away_after_s", 0),
    ],
)
def test_malformed_desk_fields_are_refused(field: str, value: object) -> None:
    """Strict types: a coerced ``"yes"`` or an off-list threshold is a 422, never a guess."""
    with pytest.raises(ValidationError):
        PresenceBeat.model_validate({"subscription_id": "s", "can_notify": True, field: value})


def test_unknown_fields_are_still_forbidden() -> None:
    """``extra="forbid"`` stays: it is why the feature key, not optionality, gates the app."""
    with pytest.raises(ValidationError):
        PresenceBeat.model_validate({"subscription_id": "s", "can_notify": True, "invented": 1})


def test_the_feature_key_is_advertised() -> None:
    from local_operator.server.features import static_feature_flags

    assert static_feature_flags()["desk_presence"] == 1


# -- the publisher: byte-identical legacy record, wall-clock stamping ----------


def test_a_legacy_beat_publishes_no_desk_key(tmp_path: Path) -> None:
    """Without desk fields the record is exactly today's shape (no ``desk``)."""
    publisher = DesktopDeliveryPublisher(tmp_path)
    _beat(publisher)
    raw = json.loads(delivery_record_path(publisher.instance_id, tmp_path).read_text())
    assert "desk" not in raw
    assert read_delivery(tmp_path).desk_reported is False
    publisher.close()


def test_the_receiver_stamps_wall_clock_and_no_monotonic_value_crosses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ADR m2: the server's ``time.time`` stamps; the daemon's ``now`` reads.

    ``time.monotonic`` is pinned to an absurd value: if it leaked into the
    record, ``received_at`` would equal it and the daemon's wall-clock ``now``
    would read the report as stale (or eternally fresh).
    """
    server_wall = 1_800_000_000.0
    monkeypatch.setattr(publisher_module.time, "time", lambda: server_wall)
    monkeypatch.setattr(publisher_module.time, "monotonic", lambda: 42.0)
    publisher = DesktopDeliveryPublisher(tmp_path)
    _beat(publisher, idle_s=30, locked=False, idle_trusted=True, away_after_s=120)

    raw = json.loads(delivery_record_path(publisher.instance_id, tmp_path).read_text())
    (entry,) = raw["desk"]
    assert entry["received_at"] == server_wall
    assert entry["last_input_at"] == server_wall - 30
    assert entry["away_at"] == server_wall - 30 + 120
    assert 42.0 not in entry.values()

    # The DAEMON's clock: another process, same machine, its own time.time.
    from local_operator.session.runtime import presence as presence_module

    monkeypatch.setattr(presence_module.time, "time", lambda: server_wall + 5)
    presence = read_delivery(tmp_path)
    assert presence.desk_active(server_wall + 5) is True
    assert presence.desk_active(server_wall + 90) is False, "idle 30 + 90 >= 120"
    assert presence.last_input_at(server_wall + 5) == server_wall - 30
    publisher.close()


def test_a_later_legacy_beat_drops_the_desk_report(tmp_path: Path) -> None:
    """Missing fields are dropped rather than retained (the claim's own rule)."""
    publisher = DesktopDeliveryPublisher(tmp_path)
    _beat(publisher, idle_s=0, locked=False, idle_trusted=True, away_after_s=120)
    assert read_delivery(tmp_path).desk_reported is True
    _beat(publisher)
    assert read_delivery(tmp_path).desk_reported is False
    publisher.close()


def test_two_subscriptions_publish_two_reports(tmp_path: Path) -> None:
    publisher = DesktopDeliveryPublisher(tmp_path)
    _beat(publisher, "sub-1", idle_s=0, locked=False, idle_trusted=True, away_after_s=120)
    _beat(publisher, "sub-2", idle_s=500, locked=False, idle_trusted=True, away_after_s=120)
    presence = read_delivery(tmp_path)
    assert len(presence.desk) == 2
    assert presence.desk_active(time.time()) is True, "any desk active is active"
    publisher.close()


# -- the predicate ---------------------------------------------------------------


def _report(**overrides) -> DeskReport:
    fields = {"idle_s": 0, "locked": False, "idle_trusted": True, "away_after_s": 120}
    fields.update(overrides)
    received_at = fields.pop("received_at", 1000.0)
    return materialise_desk_report(received_at=received_at, **fields)


def test_an_untrusted_zero_idle_cannot_keep_the_desk_active() -> None:
    """ADR m3: per-report evaluation, so the untrusted record contributes nothing."""
    presence = DesktopPresence(
        present=True,
        desk=(_report(idle_s=0, idle_trusted=False), _report(idle_s=200, idle_trusted=True)),
    )
    assert presence.desk_active(1001.0) is False
    assert presence.last_input_at(1001.0) == 800.0, "trusted records only"


def test_only_untrusted_reports_are_unknown() -> None:
    presence = DesktopPresence(present=True, desk=(_report(idle_trusted=False),))
    assert presence.desk_reported is True
    assert presence.desk_known(1001.0) is False
    assert presence.desk_active(1001.0) is False
    assert presence.last_input_at(1001.0) is None


def test_a_stale_report_is_unknown() -> None:
    presence = DesktopPresence(present=True, desk=(_report(away_after_s=600),))
    assert presence.desk_active(1000.0 + PRESENCE_TTL_S) is True
    assert presence.desk_active(1000.0 + PRESENCE_TTL_S + 1) is False
    assert presence.desk_known(1000.0 + PRESENCE_TTL_S + 1) is False


def test_a_partial_report_leans_towards_notifying() -> None:
    """``idle_trusted`` without ``idle_s`` is no reading at all: untrusted."""
    report = materialise_desk_report(
        received_at=1000.0, idle_s=None, locked=None, idle_trusted=True, away_after_s=None
    )
    assert report.idle_trusted is False
    assert report.away_at == 1000.0 + DEFAULT_AWAY_AFTER_S


def test_a_malformed_published_entry_is_ignored() -> None:
    assert DeskReport.from_record({"received_at": "x"}) is None
    assert DeskReport.from_record([]) is None
    parsed = DeskReport.from_record(
        {"received_at": 1, "last_input_at": 1, "away_at": 2, "idle_trusted": 1, "locked": "true"}
    )
    assert parsed is not None
    assert parsed.idle_trusted is False and parsed.locked is False, "only literal True counts"


def test_the_threshold_choices_are_the_adr_allow_list() -> None:
    assert AWAY_AFTER_CHOICES_S == (60, 120, 300, 600)
    assert DEFAULT_AWAY_AFTER_S == 120


# -- the shared conformance table -----------------------------------------------


def _cases() -> list[dict[str, Any]]:
    return json.loads(FIXTURE.read_text())["cases"]


def test_the_conformance_fixture_is_the_pinned_copy() -> None:
    """agent-server holds a byte-identical copy and asserts the same digest."""
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["name"])
def test_the_conformance_table(case: dict[str, Any]) -> None:
    report = materialise_desk_report(received_at=case["received_at"], **case["report"])
    presence = DesktopPresence(present=True, desk=(report,))
    now = case["now"]
    if presence.desk_active(now):
        verdict = "active"
    elif presence.desk_known(now):
        verdict = "inactive"
    else:
        verdict = "unknown"
    assert verdict == case["expect"]
