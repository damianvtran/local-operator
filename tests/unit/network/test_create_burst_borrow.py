"""A credential borrow during a burst of slow creates, over two REAL relays.

WHY THIS FILE EXISTS (#3547 QA round 2, Q3 = agent review MINOR-1). #3547 moved
``net_session_create`` off the link's reader onto the slow-op lane, whose four
workers it then SHARED with ``net_broker``. A create holds its worker for its
whole first-turn wait, so four or more slow first turns on the relay that also
serves a credential pinned every worker, and a borrow arriving then queued past
its borrower's ``CLIENT_DEADLINE_S`` (8 s) and failed as ``owner_offline`` —
"reconnect that device" about a device that was busy creating the sessions being
offloaded to it. QA measured that on head and on main alike (8.01 s, every borrow).

THE RIG is QA's burst-borrow row, scaled: the credential-owner fixture
(``mesh``: A owns an expired stub-provider login, B borrows it over both relay
legs) with A ALSO serving sessions whose first turn outlives the borrower's
deadline. B opens a link to A, fires a burst of creates with a first prompt, and
borrows while every create is inside its turn wait. The borrower's 8 s deadline
is the shipped one, untouched: the creates' wait (``FIRST_TURN_WAIT_S``) is
shortened to 15 s, so a run costs ~20-35 s rather than minutes. 15 and not 10:
the borrow is issued 3-4 s into the burst, and with 10 s waits a borrow queued
behind them on ``c0fb45f6`` was served in 6.9-7.1 s — inside its deadline, so
the test could not tell the broken lane from the fixed one. At 15 s the queued
borrow outwaits its 8 s deadline, exactly as QA's 45 s waits did.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from local_operator.network import relay
from local_operator.network.credentials.client import CLIENT_DEADLINE_S
from local_operator.network.credentials.types import Grant
from tests.unit.network import test_session_plane as plane
from tests.unit.network.test_create_first_turn_budget import _TimedHandle
from tests.unit.network.test_credentials_owner import (  # noqa: F401 — fixtures by import
    STUB_PROVIDER,
    idp,
    stub_provider,
)
from tests.unit.network.test_credentials_real_link import (  # noqa: F401 — fixtures by import
    _borrower_client,
    _pull,
    _share,
    mesh,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)

#: The creates' first-turn wait here: far enough past the borrower's deadline
#: that a borrow queued behind the creates' workers cannot be served in time.
_WAIT_S = 15.0


class _LongTurn(_TimedHandle):
    """A first turn that outlives the wait, so every create holds its worker."""

    delay_s = _WAIT_S + 3.0


@pytest.mark.parametrize("creates", [4, 6])
def test_a_borrow_during_a_burst_of_slow_creates_is_served(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, creates: int
) -> None:
    rig: Any = request.getfixturevalue("mesh")
    _server_a, _server_b, host, port = request.getfixturevalue("devices")
    # The borrow goes out ~4 s into the burst; a queued one must outwait 8 s.
    assert _WAIT_S - 5.0 > CLIENT_DEADLINE_S, "the rig must hold creates past the deadline"
    _share(rig)
    assert STUB_PROVIDER in _pull(rig).get("changed", [])
    monkeypatch.setattr(relay, "FIRST_TURN_WAIT_S", _WAIT_S)
    monkeypatch.setattr(plane, "_Handle", _LongTurn)
    served = plane._serve(monkeypatch, rig.a.root)  # noqa: SLF001
    link, reason = rig.b.dial(rig.network_id, host=f"{host}:{port}")
    assert link is not None, reason
    hop = relay.engage_hop_bound_s()
    replies: dict[int, Any] = {}
    finished: dict[int, float] = {}
    began = time.monotonic()

    def _create(req: int) -> None:
        replies[req] = link.request(
            {
                "op": "net_session_create",
                "req": req,
                "locality": "remote",
                "cwd": str(rig.a.root),
                "name": "",
                "prompt": f"burst {req}",
            },
            timeout=hop,
        )
        finished[req] = time.monotonic() - began

    workers = [threading.Thread(target=_create, args=(700 + i,)) for i in range(creates)]
    try:
        # STAGGERED, as in the budget file: this rig's in-process stand-in for a
        # runtime spawn can wedge when two publish at the same instant (a limit of
        # the fake, keyed by pid; production spawns processes). Every create is
        # still inside its turn wait when the borrow below is issued.
        for worker in workers:
            worker.start()
            time.sleep(0.4)
        time.sleep(1.0)
        asked = time.monotonic()
        grant = _borrower_client(rig).request_grant_sync(STUB_PROVIDER, session_id="burst-borrow")
        latency = time.monotonic() - asked
        for worker in workers:
            worker.join(timeout=hop + 5.0)
    finally:
        link.close("test")
        plane._stop_all(served)  # noqa: SLF001
    kind = "Grant(<redacted>)" if isinstance(grant, Grant) else repr(grant)
    print(f"\n{creates} creates; borrow at t+{asked - began:.1f}s: {latency:.2f} s -> {kind}")
    for req in sorted(replies):
        reply = replies[req]
        turn = (reply or {}).get("detail", {}) if reply and reply.get("op") == "ack" else {}
        shown = turn.get("turn") if isinstance(turn, dict) else None
        print(f"  create {req}: {finished.get(req, -1):.1f}s {reply and reply.get('op')} {shown}")
    assert isinstance(grant, Grant), grant
    assert latency < CLIENT_DEADLINE_S, latency
    # EVERY CREATE GOT A RECEIPT: an ack with its turn's state, or the lane's own
    # sentence — never ``None``, which the CLI prints as "stopped answering".
    assert sorted(replies) == [700 + i for i in range(creates)], replies
    for req, reply in replies.items():
        assert reply is not None and reply.get("op") in ("ack", "error"), (req, reply)
        assert finished[req] <= hop, (req, finished[req])
