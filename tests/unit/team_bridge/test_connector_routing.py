"""Which local session a DM reply wakes: the pure decision and its two inputs.

The replay table is the 2026-11 incident: Bill (a task session) DMed two
colleagues' agents, the answers landed in the designated session, and the asker
never saw them. Each row is run through the REAL ledger and the real decision,
with liveness supplied as the registry would answer.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.team_bridge import reply_guard
from local_operator.team_bridge.connector import routing
from local_operator.team_bridge.connector.routing import (
    DESIGNATED_AMBIGUOUS,
    DESIGNATED_IS_ASKER,
    DESIGNATED_NO_LEDGER,
    DESIGNATED_NO_POSTER,
    DESIGNATED_NOT_DM,
    DESIGNATED_NOT_LIVE,
    DESIGNATED_OLD_BUILD,
    DESIGNATED_SENDER,
    POSTER_WINDOW_S,
    ROUTED_SOLE_POSTER,
    ROUTED_THREAD,
    decide_route,
)

ENDPOINT = "bep_1"
DM = "bcv_aaaaaaaa"
OTHER_DM = "bcv_bbbbbbbb"
OWN = "acc_owner"
PEER = "acc_peer"
BILL = "188646efd00e"
B = "5a1b2c3d4e5f"
DESIGNATED = "sess_cos"
NOW = 1_800_000_000.0


def _post(tmp: Path, session: str, at: float, *, message_id: str, conversation: str = DM) -> None:
    record = reply_guard.record_outbound(
        tmp,
        endpoint_id=ENDPOINT,
        conversation_id=conversation,
        session_id=session,
        client_ref=f"c-{message_id}",
        kind="dm",
        ask=True,
        now=at,
    )
    reply_guard.confirm_outbound(tmp, record, message_id=message_id, seq=1)


def _route(
    tmp: Path,
    *,
    live: frozenset[str] = frozenset({BILL, B, DESIGNATED}),
    designated_unroutable: bool = False,
    kind: str | None = "dm",
    sender_class: str | None = "agent",
    sender: str = PEER,
    thread_root: str | None = None,
    now: float = NOW,
    conversation: str = DM,
    ledger: Any = "real",
) -> routing.RouteDecision:
    if ledger == "real":
        ledger = reply_guard.routing_facts(
            tmp,
            endpoint_id=ENDPOINT,
            conversation_id=conversation,
            thread_root_id=thread_root,
            since=routing.window_start(now),
        )
    return decide_route(
        kind=kind,
        sender_class=sender_class,
        sender_account_id=sender,
        own_account_id=OWN,
        designated_session=DESIGNATED,
        thread_root_id=thread_root,
        ledger=ledger,
        live_capable=live,
        designated_unroutable=designated_unroutable,
    )


# -- the incident replay -------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "gap_s"),
    [("Ola seq 81", 31), ("Ola seq 83", 31 + 120), ("Damian seq 329", 8 * 60 + 9)],
)
def test_the_incident_replies_route_to_the_session_that_asked(
    tmp_path: Path, label: str, gap_s: int
) -> None:
    _post(tmp_path, BILL, NOW - gap_s, message_id="bcmsg_ask00001")

    decision = _route(tmp_path)

    assert decision.target == BILL, label
    assert decision.reason == ROUTED_SOLE_POSTER


def test_counterfactual_two_posters_go_to_the_designated_session_with_candidates(
    tmp_path: Path,
) -> None:
    _post(tmp_path, BILL, NOW - 300, message_id="bcmsg_ask00001")
    _post(tmp_path, B, NOW - 60, message_id="bcmsg_ask00002")

    decision = _route(tmp_path)

    assert decision.target is None and decision.reason == DESIGNATED_AMBIGUOUS
    assert [session for session, _ in decision.candidates] == [B, BILL]  # newest first


# -- who may route ----------------------------------------------------------------


@pytest.mark.parametrize("kind", ["group", "channel", None])
@pytest.mark.parametrize("thread_root", [None, "bcmsg_ask00001"])
def test_only_a_dm_ever_routes(tmp_path: Path, kind: str | None, thread_root: str | None) -> None:
    """The kind gate outranks a thread hit, even one naming our own post.

    PR #4177 threads a channel/group answer under the message it answers BY
    DEFAULT, so the next inbound message there now carries a ``thread_root_id``
    that is one of our own posts — exactly the fact that routes a DM. A group or
    channel must still never route: its members have no "the asker" to reach.
    """

    _post(tmp_path, BILL, NOW - 10, message_id="bcmsg_ask00001")
    decision = _route(tmp_path, kind=kind, thread_root=thread_root)
    assert decision.target is None and decision.reason == DESIGNATED_NOT_DM


@pytest.mark.parametrize(
    ("sender_class", "sender"),
    [("app", PEER), ("unknown", PEER), (None, PEER), ("agent", OWN), ("human", OWN), ("agent", "")],
)
def test_only_another_accounts_human_or_agent_routes(
    tmp_path: Path, sender_class: str | None, sender: str
) -> None:
    _post(tmp_path, BILL, NOW - 10, message_id="bcmsg_ask00001")
    decision = _route(tmp_path, sender_class=sender_class, sender=sender)
    assert decision.target is None and decision.reason == DESIGNATED_SENDER


def test_no_poster_in_the_window_goes_to_the_designated_session(tmp_path: Path) -> None:
    _post(tmp_path, BILL, NOW - POSTER_WINDOW_S - 1, message_id="bcmsg_ask00001")
    decision = _route(tmp_path)
    assert decision.target is None and decision.reason == DESIGNATED_NO_POSTER


def test_the_window_edge_counts_a_post_exactly_at_the_limit(tmp_path: Path) -> None:
    _post(tmp_path, BILL, NOW - POSTER_WINDOW_S, message_id="bcmsg_ask00001")
    assert _route(tmp_path).target == BILL


def test_the_designated_session_as_the_only_poster_is_unchanged(tmp_path: Path) -> None:
    _post(tmp_path, DESIGNATED, NOW - 10, message_id="bcmsg_ask00001")
    decision = _route(tmp_path)
    assert decision.target is None and decision.reason == DESIGNATED_IS_ASKER


@pytest.mark.parametrize(
    "live", [frozenset(), frozenset({DESIGNATED}), frozenset({DESIGNATED, "someone_else"})]
)
def test_a_poster_that_is_not_live_or_lacks_the_capability_is_not_routed_to(
    tmp_path: Path, live: frozenset[str]
) -> None:
    _post(tmp_path, BILL, NOW - 10, message_id="bcmsg_ask00001")
    decision = _route(tmp_path, live=live)
    assert decision.target is None and decision.reason == DESIGNATED_NOT_LIVE


def test_a_ledger_that_cannot_be_read_never_routes(tmp_path: Path) -> None:
    decision = _route(tmp_path, ledger=None)
    assert decision.target is None and decision.reason == DESIGNATED_NO_LEDGER


def test_a_ledger_that_is_down_on_disk_fails_open(tmp_path: Path) -> None:
    reply_guard.guard_path(tmp_path).mkdir(parents=True)  # a directory is not a database
    assert _route(tmp_path).target is None


# -- the thread root -----------------------------------------------------------------


def test_a_thread_root_posted_by_a_session_decides_on_its_own_without_a_window(
    tmp_path: Path,
) -> None:
    # Bill posted two days ago, B posted a minute ago: the thread names Bill.
    _post(tmp_path, BILL, NOW - 2 * 86_400, message_id="bcmsg_ask00001")
    _post(tmp_path, B, NOW - 60, message_id="bcmsg_ask00002")

    decision = _route(tmp_path, thread_root="bcmsg_ask00001")

    assert decision.target == BILL and decision.reason == ROUTED_THREAD


def test_a_root_from_another_conversation_picks_nothing(tmp_path: Path) -> None:
    """The ledger query is scoped to THIS conversation, so naming a message one
    of our sessions posted somewhere else cannot steer this DM."""

    _post(tmp_path, B, NOW - 5, message_id="bcmsg_elsewhere", conversation=OTHER_DM)
    _post(tmp_path, BILL, NOW - 5, message_id="bcmsg_ask00001")

    decision = _route(tmp_path, thread_root="bcmsg_elsewhere")

    # No ledger hit in this DM: falls through to the window rule (Bill alone).
    assert decision.target == BILL and decision.reason == ROUTED_SOLE_POSTER


def test_an_unknown_root_is_ignored(tmp_path: Path) -> None:
    _post(tmp_path, BILL, NOW - 5, message_id="bcmsg_ask00001")
    decision = _route(tmp_path, thread_root="bcmsg_unknown1")
    assert decision.target == BILL and decision.reason == ROUTED_SOLE_POSTER


def test_a_thread_hit_on_a_dead_session_falls_back(tmp_path: Path) -> None:
    _post(tmp_path, BILL, NOW - 5, message_id="bcmsg_ask00001")
    decision = _route(tmp_path, thread_root="bcmsg_ask00001", live=frozenset({DESIGNATED}))
    assert decision.target is None and decision.reason == DESIGNATED_NOT_LIVE


# -- nothing reads text ---------------------------------------------------------------


def test_the_decision_takes_no_message_text() -> None:
    """A forged "— ref 7KQ2M" or "reply to Bill" in a body selects nothing,
    because the function cannot be given one."""

    names = set(inspect.signature(decide_route).parameters)
    assert not {"body", "text", "ref", "content", "message"} & names


# -- liveness ------------------------------------------------------------------------------


def _record(session: str, caps: list[str]) -> Any:
    return SimpleNamespace(session_id=session, capabilities=caps)


def test_only_live_sessions_with_the_capability_count_and_the_scan_does_not_reap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.session.runtime import registry
    from local_operator.session.runtime.types import (
        TEAM_BRIDGE_ROUTED_INBOX_CAPABILITY as CAP,
    )

    seen: list[dict[str, Any]] = []

    def scan(root: Any = None, *args: Any, **kwargs: Any) -> list[tuple[Any, str]]:
        seen.append(kwargs)
        return [
            (_record("live_ok", [CAP]), "live"),
            (_record("live_old_build", []), "live"),
            (_record("wedged", [CAP]), "wedged"),
            (_record("stale", [CAP]), "stale"),
        ]

    monkeypatch.setattr(registry, "scan", scan)

    assert routing.live_capable_sessions(None) == frozenset({"live_ok"})
    assert seen == [{"reap": False}]


def test_liveness_is_scanned_once_per_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.session.runtime import registry

    calls: list[int] = []
    monkeypatch.setattr(registry, "scan", lambda *a, **k: calls.append(1) or [])

    with routing.iteration_scope():
        routing.live_capable_sessions(None)
        routing.live_capable_sessions(None)
        routing.live_capable_sessions(None, fresh=True)  # the pre-delivery recheck
    assert len(calls) == 2
    routing.live_capable_sessions(None)
    routing.live_capable_sessions(None)
    assert len(calls) == 4  # no scope, no cache


def test_an_unreadable_registry_is_an_empty_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.session.runtime import registry

    def boom(*a: Any, **k: Any) -> Any:
        raise OSError("registry unreadable")

    monkeypatch.setattr(registry, "scan", boom)
    assert routing.live_capable_sessions(None) == frozenset()


# -- a designated session on an old build ---------------------------------------------


def test_an_old_designated_build_blocks_routing_for_every_route(tmp_path: Path) -> None:
    """An old build reads ``designated_session = me`` and would claim a row routed
    away from it, serving a stranger's reply to the wrong session."""

    _post(tmp_path, BILL, NOW - 40, message_id="bcmsg_ask00001")
    decision = _route(tmp_path, designated_unroutable=True)
    assert (decision.target, decision.reason) == (None, DESIGNATED_OLD_BUILD)

    _post(tmp_path, BILL, NOW - 10, message_id="bcmsg_root0001")
    threaded = _route(tmp_path, thread_root="bcmsg_root0001", designated_unroutable=True)
    assert (threaded.target, threaded.reason) == (None, DESIGNATED_OLD_BUILD)

    # Negative control: the same facts route when the designated build is new.
    assert _route(tmp_path, designated_unroutable=False).target == BILL


def test_only_a_live_designated_session_without_the_capability_blocks_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.session.runtime import registry
    from local_operator.session.runtime.types import (
        TEAM_BRIDGE_ROUTED_INBOX_CAPABILITY as CAP,
    )

    def run(rows: list[tuple[Any, str]]) -> bool:
        monkeypatch.setattr(registry, "scan", lambda *a, **k: rows)
        return routing.designated_blocks_routing(None, DESIGNATED)

    assert run([(_record(DESIGNATED, []), "live")]) is True
    assert run([(_record(DESIGNATED, [CAP]), "live")]) is False
    # Not live (stopped, wedged, unknown): it reads nothing, so it is no obstacle.
    assert run([(_record(DESIGNATED, []), "wedged")]) is False
    assert run([(_record(B, [CAP]), "live")]) is False
    assert run([]) is False


def test_the_registry_record_for_the_recheck_is_live_capable_and_never_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator.session.runtime import registry
    from local_operator.session.runtime.types import (
        TEAM_BRIDGE_ROUTED_INBOX_CAPABILITY as CAP,
    )

    rows = [(_record(B, [CAP]), "live"), (_record(BILL, []), "live")]
    monkeypatch.setattr(registry, "scan", lambda *a, **k: rows)
    with routing.iteration_scope():
        assert routing.live_capable_record(None, B) is rows[0][0]
        assert routing.live_capable_record(None, BILL) is None
        rows.clear()  # the session died after the iteration's scan
        assert routing.live_capable_record(None, B) is None
