"""The commit that outran the copy: what a move says when the receiver never confirmed.

QA round 1, Q1 and Q2 (2026-09-27). Driven on the two-device rig first, then pinned
here.

THE FINDING, IN THE PRODUCT'S OWN WORDS. With the receiving device SIGKILLed at the
owner's ``handing_off`` point, the owner answered ``rc 0`` with ``phase: done`` and
audited ``session.handoff.committed``, then retired its own copy behind a tombstone and
hid the id from its listing — while the only copy of the conversation was in the
receiver's ``network/staging/<id>/``. A retry from the owner said ``third_device``:
"run this from the device that holds it", naming the device that held nothing; and the
receiver could not adopt its own staged bytes, because ``move <id> --to local`` answered
"no device in this network holds <id>" — the destination's journal entry is written by
that device's LIVE relay, and every reconcile skips an entry its own live instance wrote
(the ``include_own`` guard that protects a genuinely in-flight handoff).

Cells, one per half of the repair:

1. the wait separates "committed here" from "confirmed there", and holds the
   confirmation window the bound already carried;
2. the receipt for an unconfirmed commit is a refusal that names the route home;
3. the device holding the staged copy can adopt it when the user asks it for the id;
4. an invite names the endpoint the relay actually listens on (Q2);
5. the same receipt is TRUE in the other state a missing confirmation can mean — the
   receiver promoted and the frame was lost (QA round 2, Q1).
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from typing import Any, NoReturn

import pytest

from local_operator.network import mobility, store, sync
from local_operator.network.projection import write_tombstone
from tests.unit.network.test_mobility import (  # noqa: F401 — fixtures and helpers
    SESSION,
    Devices,
    _move,
    _owned_session,
    pair,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — the fixture `pair` reaches for
    _pair_settled,
    devices,
)


def test_the_wait_separates_a_commit_here_from_a_confirmation_there(
    request: pytest.FixtureRequest,
) -> None:
    """``committed`` is this device's fact; ``done`` is the destination's.

    The wait used to return the tombstone's ``True`` — so a commit whose destination
    never promoted was indistinguishable from a finished move, which is how the
    receipt came to claim one.
    """
    server_a = request.getfixturevalue("pair")[0]
    _owned_session(server_a)
    progress = mobility.progress_for(server_a)
    try:
        outcome, refusal = mobility._await_own_progress(server_a, SESSION, budget=0.2)
        assert (outcome, refusal) == ("none", None), (outcome, refusal)

        write_tombstone(
            SESSION,
            device_id="d_" + "b" * 16,
            device_name="peer-b",
            config_dir=server_a.root,
        )
        outcome, refusal = mobility._await_own_progress(server_a, SESSION, budget=0.5)
        assert outcome == "committed", outcome
        assert refusal is None, refusal

        # AND THE CONFIRMATION WINDOW IS REAL: a `done` that arrives after the commit
        # still ends the wait as a confirmation. Under the old code the tombstone had
        # already returned, so this signal could never have been observed.
        progress.forget(SESSION)
        threading.Timer(0.3, lambda: progress.signal(SESSION, "done")).start()
        outcome, refusal = mobility._await_own_progress(server_a, SESSION, budget=2.0)
        assert outcome == "done", outcome
        assert refusal is None, refusal
    finally:
        progress.forget(SESSION)


def test_an_unconfirmed_commit_is_refused_with_the_route_home(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The receipt for a commit no destination confirmed, driven end to end.

    The receiver dies where it really dies — after ``ready`` (which is what lets the
    owner commit) and before its promote. What comes back must not be a success line
    for an id that is then on no device's listing.
    """
    both: Devices = request.getfixturevalue("pair")
    server_a, server_b, _host, _port = both
    _pair_settled(both, monkeypatch, role="admin", settings=server_b.settings)
    _owned_session(server_a)
    # ONLY THE TIMEOUT IS SHORTENED, and it is a timeout rather than a tolerance: the
    # destination dies either way, so a smaller window cannot make a wrong answer
    # right — it only stops the cell sitting out the full thirty seconds.
    monkeypatch.setattr(mobility, "OFFLOAD_CONFIRM_WAIT_S", 0.5)

    def _die(*args: Any, **kwargs: Any) -> NoReturn:
        # NoReturn, not the old ``-> bool``: this stub stands in for ``_promote``
        # (now ``mobility._PromoteOutcome``) and always raises.
        raise RuntimeError("the receiver died before its promote (the crash window)")

    monkeypatch.setattr(mobility, "_promote", _die)
    result = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)

    assert result["ok"] is False, result
    assert result["code"] == "unconfirmed", result
    assert result["changed"] is True, "this device did change: it retired its own copy"
    message = str(result["message"])
    assert "--to local" in message, message
    assert f"network/staging/{SESSION}" in message, message
    assert result["phase_reached"] == "committed", result["phase_reached"]

    # AND THE SENTENCE DESCRIBES A REAL STATE: the only copy is staged on B.
    staged = sync.staging_dir(server_b.root, SESSION)
    assert (staged / "ready.json").is_file(), "the receiver's verified copy must survive"
    assert not (server_b.root / "sessions" / SESSION).is_dir()


def test_the_receipt_is_true_when_the_receiver_DID_promote(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QA round 2, Q1: the same receipt, in the state it used to misdescribe.

    A missing ``done`` frame has two causes, and this is the other one: the receiver ran
    its promote (one ``os.replace``) and the frame was lost. The sentence that asserted
    "its verified copy is in that device's ``network/staging/<id>``" then named an EMPTY
    directory while the conversation sat on the destination, listed, with the verb it
    named answering ``already_local`` — measured on the two-device rig by freezing the
    promote rename. What is pinned here is the STATE and the sentence's SHAPE: the check
    that decides between the two states comes before the staging path.
    """
    both: Devices = request.getfixturevalue("pair")
    server_a, server_b, _host, _port = both
    _pair_settled(both, monkeypatch, role="admin", settings=server_b.settings)
    _owned_session(server_a)
    monkeypatch.setattr(mobility, "OFFLOAD_CONFIRM_WAIT_S", 0.5)
    original_ask = mobility.LinkTransport.ask

    def _lose_the_done_frame(
        self: Any, frame: dict[str, Any], *, timeout: float | None = None
    ) -> dict[str, Any]:
        if frame.get("phase") == "done":
            # DROPPED HERE RATHER THAN IN THE HANDLER, so the promote and everything
            # before it run for real: the destination is told its acknowledgement
            # landed, and this device's own wait never learns about it.
            return {"result": "done", "session_id": SESSION}
        return original_ask(self, frame, timeout=timeout)

    monkeypatch.setattr(mobility.LinkTransport, "ask", _lose_the_done_frame)
    result = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)

    assert result["ok"] is False, result
    assert result["code"] == "unconfirmed", result
    assert result["changed"] is True, result
    assert result["phase_reached"] == "committed", result["phase_reached"]

    # AND THE STATE IS THE ONE THE OLD SENTENCE GOT WRONG: the conversation is ON the
    # destination, with its transcript, and no staging directory is left at all.
    assert (server_b.root / "sessions" / SESSION / "transcript.jsonl").is_file()
    assert not sync.staging_dir(server_b.root, SESSION).exists(), (
        "the promote consumes the staging directory, which is what makes the old "
        "sentence name a path that does not exist in this state"
    )

    # THE SENTENCE IS TRUE HERE: it hands over the check that decides the state — the
    # destination's own listing — BEFORE the staging path, which is empty in this one.
    # Both halves matter, and each is a way a wrong wording fails: one that names
    # neither fails the first assertion, and one that names the staging path first
    # fails the second.
    message = str(result["message"])
    assert "listing" in message, message
    assert message.index("listing") < message.index(f"network/staging/{SESSION}"), message


def test_the_device_holding_the_staged_copy_can_adopt_it(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """§6.5 row 5, asked for by the user on the device that holds the bytes.

    The state is CONSTRUCTED rather than raced — the window between the receiver's
    ``ready`` and its one ``os.replace`` is milliseconds here, and the rig audit found
    the receiver usually wins it. Constructing it is faithful: a real move is driven
    first (so the owner's tombstone is the product's own), then the promoted directory
    goes back into staging with a ``ready.json`` of the shape the receiver writes, which
    is exactly what the crash window leaves behind.
    """
    both: Devices = request.getfixturevalue("pair")
    server_a, server_b, _host, _port = both
    _pair_settled(both, monkeypatch, role="admin", settings=server_b.settings)
    _owned_session(server_a)
    moved = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)
    assert moved["ok"] is True, moved
    assert (server_b.root / "sessions" / SESSION).is_dir()

    staged = sync.staging_dir(server_b.root, SESSION)
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(server_b.root / "sessions" / SESSION), str(staged))
    (staged / "ready.json").write_text(
        json.dumps(
            {
                "version": 1,
                "lease_epoch": "e_constructed",
                # THE REAL DIGEST, computed by the product's own function: the adopt
                # re-verifies the staged bytes against this field, so a cell that made
                # one up would be testing the refusal rather than the adoption (measured
                # while writing this cell: a fabricated digest answered "the staged copy
                # of that conversation no longer matches the bytes that were verified").
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
    assert not (server_b.root / "sessions" / SESSION).is_dir()

    adopted = _move(server_b, SESSION, to="local", monkeypatch=monkeypatch)

    assert adopted["ok"] is True, adopted
    assert adopted.get("recovered") is True, adopted
    assert (server_b.root / "sessions" / SESSION).is_dir(), "the staged copy must be promoted"
    assert not staged.exists(), "the staging directory is consumed by the promote"

    # AND THE ADOPTION IS DIGEST-GATED, so this route cannot become a way to promote
    # bytes that never verified: the same construction with one byte changed refuses,
    # promotes nothing, and leaves the copy where it is.
    damaged = sync.staging_dir(server_b.root, SESSION)
    damaged.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(server_b.root / "sessions" / SESSION), str(damaged))
    (damaged / "ready.json").write_text(
        json.dumps(
            {
                "version": 1,
                "lease_epoch": "e_constructed",
                "content_digest": "sha256:not-the-bytes-on-disk",
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

    refused = _move(server_b, SESSION, to="local", monkeypatch=monkeypatch)

    assert refused["ok"] is False, refused
    assert not (
        server_b.root / "sessions" / SESSION
    ).is_dir(), "a copy whose bytes do not match what was verified was promoted"
    assert damaged.is_dir(), "the refusal must leave the only copy of the conversation alone"


def test_an_invite_names_the_endpoint_the_relay_actually_listens_on(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q2: a token that told the joiner nothing about where to dial.

    The record's ``listen.advertised`` is written by the process that ran
    ``init``/``join`` from ITS config, so a relay started with ``serve --port`` on an
    address the record never learned left the mint with ``"hosts": []`` — and then
    ``join @file`` refused ``no_host`` while the inviter's guidance printed no ``--host``
    either.
    """
    both: Devices = request.getfixturevalue("pair")
    server_a, _server_b, _host, _port = both
    _pair_settled(both, monkeypatch, role="admin", settings=both[1].settings)
    record = server_a._require_network("")  # noqa: SLF001 — the relay's own resolver
    live = server_a.advertised_endpoints()

    # THE STATE QA MEASURED: the record learns nothing, the process knows better.
    with store.mutate(record.network_id, server_a.root) as current:
        current.listen["advertised"] = []
    assert store.load(record.network_id, server_a.root).listen["advertised"] == []

    minted = server_a._ctl_invite(  # noqa: SLF001 — the control op the CLI calls
        {
            "network": record.network_id,
            "role": "read",
            "ttl_s": 600.0,
            "hosts": [],
            "device_id": "",
        }
    )

    assert minted["hosts"] == live, (minted["hosts"], live)
    assert minted["hosts"], (
        "the invite named no endpoint at all, so `join @file` refuses `no_host` and the "
        f"operator has to pass --host by hand (this device advertises {live})"
    )
