"""Numbers that arrived from a PEER, validated at the boundary that reads them.

WHY THIS FILE EXISTS (QA round 2, one slice over from the credentials fix). The credentials
slice closed this bug class in ten places and missed the federated listing, which reads the
SAME peer documents through the same kind of bare ``int(...)``/``float(...)``. Measured
in-process before this: ``{"pid": "abc"}`` raised ``ValueError`` out of ``PeerRow.from_json``,
which the listing calls inside its own ``_load()`` — so ONE listed peer could break the
listing for every other peer in it — and ``{"pid": 10**400}`` was ACCEPTED as a pid, which is
a value no device can dial.

THE VALIDATOR IS NOT WRITTEN HERE. ``types.peer_number``/``types.peer_int`` are lifted from
the credentials slice (that branch's commit ``8d001f254``, review round 4) so the mesh has
ONE spelling of the rule: an ``int`` is compared as an int rather than through ``float()``
(``float(10**400)`` raises ``OverflowError``, which is NOT a ``ValueError`` and so escaped
every caller), anything above ``2**53`` is over the cap, and an integer string parses as an
int first so a 401-digit string cannot become ``inf``.

THE DIRECTION OF EVERY FALLBACK IS ASSERTED, not just the absence of a crash, because a
fallback that makes the bad input WIN is worse than the exception it replaces:

* a garbled ``pid`` must not look like a live one;
* a garbled ``started``/``age_s`` must not make a stale peer look fresh;
* a garbled ``protocol`` must not select the newest path;
* a garbled ``stamp_revision`` must not win a "which stamp is newer" comparison.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from local_operator.network import projection, types
from local_operator.session.placement import SessionPlacement

#: Every spelling QA asked for, plus a bool and a container: the shapes a peer can send.
#: ``1.5`` is listed separately: it is garbage for a field that NAMES something (a pid, a
#: protocol revision) and a perfectly ordinary value for a duration (an age, a start time).
BAD_VALUES: tuple[Any, ...] = ("abc", -5, 10**400, None, "", [1], True)
#: The fields where a fractional value must be refused rather than rounded.
NON_WHOLE_VALUES: tuple[Any, ...] = BAD_VALUES + (1.5,)


# ---------------------------------------------------------------------------
# The helper itself
# ---------------------------------------------------------------------------


def test_peer_number_is_total_and_capped() -> None:
    """No input raises, and nothing above the ceiling is a value."""
    for value in BAD_VALUES:
        assert types.peer_number(value, default=3.0) >= 0, value
    # The 401-digit cases, as a JSON integer and as a STRING: ``float()`` would raise on the
    # first and produce ``inf`` for the second.
    assert types.peer_number(10**400, default=7.0) == 7.0
    assert types.peer_number("1" + "0" * 400, default=7.0) == 7.0
    assert types.peer_number("1e400", default=7.0) == 7.0
    assert types.peer_number(float("inf"), default=7.0) == 7.0
    assert types.peer_number(float("nan"), default=7.0) == 7.0
    # A cap CLAMPS rather than defaulting: an over-long retry becomes the longest allowed,
    # never the shorter default (the credentials slice's R4-m2).
    assert types.peer_number(10**9, default=7.0, maximum=300.0) == 300.0
    assert types.peer_number("abc", default=7.0, maximum=300.0) == 7.0
    # Exactness: an int is never round-tripped through a float, so 2**53 + 1 is not 2**53.
    assert types.peer_number(2**53, default=0.0) == 2**53
    assert types.peer_number(2**53 + 1, default=0.0) == 0.0
    assert types.peer_int("42", default=0) == 42
    assert types.peer_int(2.9, default=0) == 2


# ---------------------------------------------------------------------------
# The peer's row document
# ---------------------------------------------------------------------------


def _row(**fields: Any) -> projection.PeerRow:
    return projection.PeerRow.from_json(fields, device_id="d_peer", device_name="pixel")


@pytest.mark.parametrize("value", NON_WHOLE_VALUES)
def test_a_bad_pid_is_not_a_pid(value: Any) -> None:
    """A pid that could not be read is 0 — a value this device cannot dial.

    ``1.5`` is in this list on purpose: ``int(1.5)`` is 1, and 1 is a real pid, so flooring
    a fractional pid would FABRICATE a plausible one instead of refusing it.
    """
    row = _row(pid=value, started=1.0)
    assert isinstance(row.pid, int)
    assert row.pid == 0, row.pid
    # And the record facade keeps that reading rather than inventing a live owner.
    assert row.to_record().pid == 0


@pytest.mark.parametrize("value", BAD_VALUES)
def test_a_bad_age_never_looks_fresh(value: Any) -> None:
    """An unreadable age is the STALEST value the protocol carries, not 0 seconds."""
    row = _row(age_s=value)
    assert row.age_s >= projection.AGE_UNKNOWN_S, row.age_s
    assert row.age_s > 3600, "an unreadable age must not read as seconds ago"


@pytest.mark.parametrize("value", BAD_VALUES)
def test_a_bad_started_never_reads_as_just_now(value: Any) -> None:
    """``started`` is rendered through ``to_record``, which spells 0 as ``time.time()``."""
    row = _row(started=value)
    assert row.started == projection.STARTED_UNKNOWN_S, row.started
    assert row.started, "a falsy started is rendered as 'just now' by to_record"
    assert row.to_record().started_at == projection.STARTED_UNKNOWN_S


def test_a_fractional_age_and_start_are_ordinary_values() -> None:
    """A duration NAMES nothing: 1.5 s old is 1.5 s old, and must not be discarded.

    The one place a fractional value is legitimate, stated so "refuse the fractional one"
    cannot be over-applied to the fields where it would throw away a real reading.
    """
    assert _row(age_s=1.5).age_s == 1.5
    assert _row(started=1.5).started == 1.5
    assert _row(started=1.5).to_record().started_at == 1.5


def test_the_peer_facts_age_is_the_same_rule() -> None:
    """The listing's per-device block is peer input too (``peers()``)."""
    for value in BAD_VALUES:
        assert (
            types.peer_number(
                value, default=projection.AGE_UNKNOWN_S, maximum=projection.AGE_UNKNOWN_S
            )
            == projection.AGE_UNKNOWN_S
        ), value


# ---------------------------------------------------------------------------
# The refresh: absent and unreadable are different answers
# ---------------------------------------------------------------------------


def _facts(**overrides: Any) -> projection.RemoteSessionFacts:
    fields: dict[str, Any] = {
        "session_id": "s1",
        "device_id": "d_peer",
        "pid": 4242,
        "protocol": 3,
        "conversation_name": "c",
        "cwd": "/tmp",
        "model_label": "m",
        "capabilities": (),
        "state": "idle",
        "reachable": True,
    }
    fields.update(overrides)
    return projection.RemoteSessionFacts(**fields)


@pytest.mark.parametrize("value", NON_WHOLE_VALUES)
def test_a_bad_protocol_never_selects_the_newest_path(value: Any) -> None:
    """A revision that could not be read is 0: the OLDEST thing this protocol can be."""
    refreshed = projection._refresh_facts(_facts(), {"protocol": value})
    assert refreshed.protocol == 0, refreshed.protocol


def test_an_absent_field_keeps_the_value_the_client_was_built_with() -> None:
    """ABSENT IS NOT THE SAME AS UNREADABLE, and the difference is asserted both ways.

    A key the row does not carry is not a claim at all, so the value the client was built
    with survives. A key the row DOES carry is the peer's claim, however it is spelled —
    including ``null``, which means "this session has no pid on my device" and must not
    resurrect the pid from the previous refresh.
    """
    assert projection._refresh_facts(_facts(), {}).protocol == 3
    assert projection._refresh_facts(_facts(), {}).pid == 4242
    assert projection._refresh_facts(_facts(), {"protocol": None}).protocol == 0
    assert projection._refresh_facts(_facts(), {"pid": None}).pid == 0


@pytest.mark.parametrize("value", BAD_VALUES)
def test_a_bad_refreshed_pid_clears_rather_than_resurrects(value: Any) -> None:
    """The previous pid is the peer's word too: an unreadable one leaves nothing live."""
    assert projection._refresh_facts(_facts(), {"pid": value}).pid == 0


# ---------------------------------------------------------------------------
# The stamp a moved session carries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", NON_WHOLE_VALUES)
def test_a_bad_stamp_revision_cannot_win(value: Any) -> None:
    """A stamp arrives on a peer's bytes, so its numbers are peer input."""
    placement = SessionPlacement.from_json({"stamp_revision": value})
    assert isinstance(placement.stamp_revision, int)
    assert placement.stamp_revision == 0, placement.stamp_revision
    assert SessionPlacement.from_json({"stamp_revision": 2**53 + 1}).stamp_revision == 0


# ---------------------------------------------------------------------------
# The listing's own reader: one bad peer must not break the others
# ---------------------------------------------------------------------------


def test_one_peer_s_bad_numbers_do_not_break_the_listing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """THE QA REPRO, at the layer that failed: ``_load`` over a mixed reply.

    Before this, ``{"pid": "abc"}`` raised out of ``_load`` itself, so a single listed peer
    removed EVERY peer's rows from the sidebar — the failure QA measured. The row for the
    bad peer is now honest about what could not be read, and the good peer's row is
    unaffected.
    """
    catalog = projection.RelayPeerCatalog(tmp_path)
    reply = {
        "sessions": [
            {"session_id": "bad", "peer": {"device_id": "d_bad"}, "pid": "abc", "age_s": "x"},
            {"session_id": "good", "peer": {"device_id": "d_good"}, "pid": 99, "age_s": 5.0},
        ]
    }
    monkeypatch.setattr(catalog, "_call", lambda op, **fields: reply)

    rows = catalog.rows()

    assert [row.session_id for row in rows] == ["bad", "good"]
    assert rows[0].pid == 0 and rows[0].age_s == projection.AGE_UNKNOWN_S
    assert rows[1].pid == 99 and rows[1].age_s == 5.0
    assert math.isfinite(rows[1].age_s)
