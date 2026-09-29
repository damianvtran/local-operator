"""The promotion proof is an identity, not an existence (review round 2, MAJOR).

A tombstone is the ONE fact §6.5's last row turns on: the owner handed the id away
and named the device that may adopt it. Its PRESENCE, though, only says the id left
its owner — and a receiver whose handoff was rolled back keeps its verified
``network/staging/<id>/`` bytes on purpose, so the owner can hand the same id to a
different device later and the older copy is then the copy of a handoff that no
longer describes it. Reading "a record exists" as "the record is mine" made a
promote out of that, which is a second owner for one id.

Three cells, one per layer the answer passes through:

1. the owner's own handler answers its record only to the device the record names,
   and prefer the link's admitted identity over the ``to_device`` an asker writes;
2. the device that promotes checks the record itself, so a peer of ANY build —
   including one that answers the way this head used to — cannot talk it into a
   second owner;
3. the state the review describes, driven end to end on the wire, which refuses and
   leaves the only copy where it is.

And a fourth, added by review round 3 (MINOR 1): the SECOND consumer of a peer's answer,
``_destination_move``, has its own cell — its check could be reverted with the whole
network suite still green, which is an evidence gap rather than a code one.

Cells 1, 2 and 4 are red on the pre-fix code; cell 3 is a state assertion (the wire's
own gate, ``authorizer._move_scope``, refuses a third device a layer earlier — see the
cell's own note). The legitimate direction is pinned by
``test_mobility_stranded_commit.py::test_the_device_holding_the_staged_copy_can_adopt_it``
and by cell 1's first half here.
"""

from __future__ import annotations

import json
import shutil
import time
from typing import Any, cast

import pytest

from local_operator.network import mobility, sync
from local_operator.network.projection import write_tombstone
from tests.unit.network.test_mobility import (  # noqa: F401 — fixture `pair` reaches for these
    SESSION,
    Devices,
    _move,
    _owned_session,
    pair,
)
from tests.unit.network.test_relay_e2e import _pair_settled, devices  # noqa: F401

#: A device that is in neither the pair nor the record's history: the one an id was
#: handed ON to, which is the whole point of the check.
THIRD = "d_" + "c" * 16


def _link_for(device_id: str) -> Any:
    """A link stand-in carrying ONE thing these handlers read: the admitted device id.

    Not a relay link: the authorisation that would have produced one is the subject of
    the relay's own cells, and the handler's gate has to hold for a caller that already
    got past it (a unit test, or a future op that reaches this handler by another door).
    """
    return type("Link", (), {"device_id": device_id, "network_id": "n_test", "context": None})()


def _staged_copy_of_the_receivers_move(
    server_a: Any, server_b: Any, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """Drive a real A -> B move, then put B's promoted directory back into staging.

    The crash window constructed rather than raced, exactly as the adoption cell does:
    the bytes and the ``ready.json`` are the product's own (the digest is computed by
    ``_staging_content_digest``), so a cell that made either up would test the refusal
    instead of the thing under test.
    """
    moved = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)
    assert moved["ok"] is True, moved
    staged = sync.staging_dir(server_b.root, SESSION)
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(server_b.root / "sessions" / SESSION), str(staged))
    (staged / "ready.json").write_text(
        json.dumps(
            {
                "version": 1,
                "lease_epoch": "e_constructed",
                "content_digest": mobility._staging_content_digest(  # noqa: SLF001
                    server_b, SESSION, staged
                ),
                "plan_id": "",
                "mode": "move",
                "owner_device": server_a.identity.device_id,
                "source_session_id": SESSION,
                "archived": False,
                "promoted": False,
                "at": time.time(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return staged


def test_the_record_answers_only_the_device_it_names(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cell 1: the owner's handler, called with the record already naming somebody else.

    Driven at the handler rather than over the wire on purpose — upstream of it,
    ``authorizer._move_scope`` scopes a ``status`` frame to the device the record names,
    and a cell that only drove the wire could not tell the two gates apart. What is
    asserted here is that THIS gate holds on its own.
    """
    both: Devices = request.getfixturevalue("pair")
    server_a, server_b, _host, _port = both
    _pair_settled(both, monkeypatch, role="admin", settings=server_b.settings)
    _owned_session(server_a)
    moved = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)
    assert moved["ok"] is True, moved

    handler = server_a._handlers["net_session_move"]  # noqa: SLF001 — the relay's own table
    frame = {
        "op": "net_session_move",
        "phase": "status",
        "session_id": SESSION,
        "to_device": server_b.identity.device_id,
    }
    link_b = _link_for(server_b.identity.device_id)

    # THE LEGITIMATE DIRECTION first, so the gate is not a blanket refusal: the device
    # the record names still reads its own commit out of it.
    answered = handler(link_b, dict(frame))
    assert isinstance(answered, dict), answered
    assert answered["result"] == "tombstone", answered
    assert answered["tombstone"]["device_id"] == server_b.identity.device_id, answered

    # THE ID IS HANDED ON, so this device's copy is no longer the copy the record
    # describes. The answer stops being a commit for it.
    write_tombstone(SESSION, device_id=THIRD, device_name="peer-c", config_dir=server_a.root)
    refused = handler(link_b, dict(frame))
    assert isinstance(refused, dict), refused
    assert refused["result"] == "refused", refused
    assert refused.get("tombstone") is None, "a record for another device is not this one's proof"
    assert "peer-c" in refused["message"], refused["message"]

    # AND THE FRAME CANNOT PROMOTE ITSELF: ``to_device`` is what the asker writes, so a
    # frame claiming the id the record names is still refused on the LINK's identity.
    claimed = handler(link_b, {**frame, "to_device": THIRD})
    assert isinstance(claimed, dict), claimed
    assert claimed["result"] == "refused", claimed


def test_a_peer_that_answers_the_old_way_cannot_make_this_device_promote(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cell 2: the promote itself, against a peer that answers existence-only.

    ``_source_status`` used to return the record to whoever asked, so a device holding
    a verified copy of an older attempt was told "it is yours" and promoted it with a
    second owner the result. This cell does not depend on the owner being fixed: the
    peer's answer here is the OLD shape (the record, no identity check), and what must
    stop the promote is this device's own reading of the record it received.
    """
    both: Devices = request.getfixturevalue("pair")
    server_a, server_b, _host, _port = both
    _pair_settled(both, monkeypatch, role="admin", settings=server_b.settings)
    _owned_session(server_a)
    staged = _staged_copy_of_the_receivers_move(server_a, server_b, monkeypatch)
    assert not (server_b.root / "sessions" / SESSION).is_dir()

    def _answer_the_old_way(
        self: Any, frame: dict[str, Any], *, timeout: float | None = None
    ) -> Any:
        assert frame.get("phase") == "status", frame
        return {
            "result": "tombstone",
            "owner": False,
            "tombstone": {"device_id": THIRD, "device_name": "peer-c", "moved_at": time.time()},
            "session_id": SESSION,
        }

    monkeypatch.setattr(mobility.LinkTransport, "ask", _answer_the_old_way)

    refused = _move(server_b, SESSION, to="local", monkeypatch=monkeypatch)

    assert refused["ok"] is False, refused
    assert refused["code"] == "third_device", refused
    # THE REFUSAL NAMES THE CONDITION, not just the outcome: what would have to be true
    # for these bytes to be adoptable is that the record name THIS device, and the
    # device it does name is where the adopting verb has to run.
    assert "peer-c" in refused["message"], refused["message"]
    assert "--to local" in refused["message"], refused["message"]
    assert refused["changed"] is False, refused
    assert not (
        server_b.root / "sessions" / SESSION
    ).is_dir(), "a record naming another device was read as permission to promote"
    assert staged.is_dir(), "the refusal must leave the only copy of the conversation alone"


def test_the_wire_refuses_a_third_device_and_leaves_the_copy(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cell 3: the state itself, driven end to end through this device's own relay.

    The record names a third device, the receiver holds the only verified copy, and the
    user asks this device for the id. What is pinned is the OUTCOME — refused, nothing
    promoted, the bytes left alone — and NOT a sentence: in this state the ask is
    refused a layer earlier, by ``authorizer._move_scope`` (the wire's own identity
    gate), so the answer the user reads here is ``resolve_remote_owner``'s "no device in
    this network holds <id>" rather than cell 2's. A cell that asserted the sentence
    here would be asserting which layer spoke first.
    """
    both: Devices = request.getfixturevalue("pair")
    server_a, server_b, _host, _port = both
    _pair_settled(both, monkeypatch, role="admin", settings=server_b.settings)
    _owned_session(server_a)
    staged = _staged_copy_of_the_receivers_move(server_a, server_b, monkeypatch)
    write_tombstone(SESSION, device_id=THIRD, device_name="peer-c", config_dir=server_a.root)

    refused = _move(server_b, SESSION, to="local", monkeypatch=monkeypatch)

    assert refused["ok"] is False, refused
    assert refused["changed"] is False, refused
    assert not (
        server_b.root / "sessions" / SESSION
    ).is_dir(), "a copy the record does not name was promoted"
    assert staged.is_dir(), "the refusal must leave the only copy of the conversation alone"
    # AND THE COPY IS STILL THE PRODUCT'S OWN VERIFIED ONE, not a directory the refusal
    # half-emptied: the promote is the only thing that consumes a staging directory.
    assert (staged / "ready.json").is_file()


def test_the_pull_route_reads_the_record_before_it_promotes(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cell 4: the OTHER consumer — `_destination_move` — with its own discriminating cell.

    The check there could be reverted with the whole 858-test network suite still green
    (review round 3, MINOR 1): every other cell reaches a promote through
    ``_reconcile_destination``, which is the ``--to local`` route, so nothing exercised
    the branch that decides in the PULL route. Here the peer's answer is handed to that
    function directly, the way the invite path hands it one.
    """
    both: Devices = request.getfixturevalue("pair")
    server_a, server_b, _host, _port = both
    _pair_settled(both, monkeypatch, role="admin", settings=server_b.settings)
    _owned_session(server_a)
    staged = _staged_copy_of_the_receivers_move(server_a, server_b, monkeypatch)
    assert not (server_b.root / "sessions" / SESSION).is_dir()

    class _AnswersTheRecordOnly:
        """A peer answering the existence-only way — what the owner did before this fix."""

        link = None

        def ask(self, frame: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
            assert frame.get("phase") == "status", frame
            return {
                "result": "tombstone",
                "owner": False,
                "tombstone": {
                    "device_id": THIRD,
                    "device_name": "peer-c",
                    "moved_at": time.time(),
                },
                "session_id": SESSION,
            }

    result, refusal, _target = mobility._destination_move(  # noqa: SLF001 — the pull route
        server_b,
        SESSION,
        transport=cast(Any, _AnswersTheRecordOnly()),
        keep=False,
        wait_s=0.0,
        owner_device=server_a.identity.device_id,
        owner_name="device-a",
    )

    assert result is None, result
    assert refusal is not None, refusal
    assert refusal["code"] == "third_device", refusal
    assert "peer-c" in refusal["message"], refusal
    assert not (
        server_b.root / "sessions" / SESSION
    ).is_dir(), "the pull route promoted a copy the record does not name"
    assert staged.is_dir(), "the refusal must leave the only copy of the conversation alone"
