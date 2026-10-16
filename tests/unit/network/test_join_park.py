"""The joining device's two-phase pair: ``--park`` opens it, ``--confirm`` answers it.

WHY THIS IS ITS OWN FILE. ``test_pairing_confirm.py`` covers the INVITER's parked
question — the relay parks it in a file because a launchd daemon has no terminal, and
``lop network confirm`` answers it. This is the mirror on the joining device, and it is
the half an AGENT drives, so the properties under test are the ones the human gate
rests on:

* a parked pairing sends NOTHING to the peer until a person's transcription arrives
  (the socket is held by the process that dialled, and the frame that tells the inviter
  a human is present is the frame the answer releases);
* the confirmation is checked against THIS device's own derivation, and a mistyped
  digit leaves the ceremony open rather than spending the invite;
* the invocation that answers is a different process from the one that dialled, which
  is the whole shape — and the AGENT TOOL has no way to be that invocation: it parks a
  ceremony and verifies the result, but the second phase is the person's own command
  (agent review round 1, semantic finding 1; the operator's ruling was to remove the
  field rather than guard it, because the code the tool prints IS the code the
  comparison expects).

The humans are supplied by the test, which is what a test is allowed to do: the
inviter's person is the real parked question answered through the relay's own control
op, and the joiner's person is a second invocation of the CLI — the command the guide
hands the user, run here by the test because a test cannot be a person. What that
invocation SO RECORDS is asserted: a machine-supplied code is written as ``harness``,
not dressed up as a human answer (semantic finding 3).
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import ToolContext
from local_operator.network import cli as net_cli
from local_operator.network import invite as invite_mod
from local_operator.network import relay, store
from local_operator.network import tool as net_tool
from local_operator.network import types, wire
from local_operator.network.handshake import (
    Credential,
    Handshake,
    pair_abort_frame,
    pair_timeout_seconds,
    sas_matches,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _answer_confirmation,
    _init_network,
    devices,
)


def _park_args(**overrides: Any) -> Namespace:
    """The Namespace argparse builds for ``join @<token> --park --json``."""
    base: dict[str, Any] = {
        "confirm": "",
        "park": True,
        "sas_stdin": False,
        "verify": False,
        "emit_sas": False,
        "name": "",
        "host": "",
        "json": True,
        "token": "",
    }
    base.update(overrides)
    return Namespace(**base)


def _confirm_args(code: str) -> Namespace:
    """The Namespace argparse builds for ``join --confirm <code> --json``."""
    return _park_args(confirm=code, park=False)


def _minted(server: relay.RelayServer, *, role: str = "drive", ttl_s: float = 600.0) -> Any:
    """A network on ``server`` with one invite, the way ``lop network invite`` does it.

    The hosts come from ``--hosts``' own argument, so the envelope names an endpoint a
    JOINEER can dial without a ``--host`` override: the agent path has no override, and
    an invite that did not carry its endpoint would make the tool unusable against a
    loopback test relay.
    """
    record = _init_network(server)
    state = store.load_secrets(record.network_id, server.root)
    minted = invite_mod.mint(
        record,
        state.secret,
        role=role,
        ttl_s=ttl_s,
        hosts=[f"127.0.0.1:{server.settings.port}"],
    )
    record.invites.append(minted.record)
    store.save(record, server.root)
    store.save_invite_token(minted.record.invite_id, minted.token, server.root)
    return record, minted


def _park(server_b: relay.RelayServer, *, host: str, port: int, minted: Any) -> Any:
    """Phase one, driven exactly as ``lop network join @token --park`` drives it."""
    return net_cli._join_one(  # noqa: SLF001 — the CLI's own driver, run as the CLI runs it
        host=f"{host}:{port}",
        token=minted.token,
        envelope=minted.envelope,
        identity=server_b.identity,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        args=_park_args(name=server_b.identity.name),
        wire=wire,
        Handshake=Handshake,
        Credential=Credential,
        pair_abort_frame=pair_abort_frame,
        pair_timeout_seconds=pair_timeout_seconds,
        sas_matches=sas_matches,
        invite_mod=invite_mod,
        store=store,
        relay_mod=relay,
    )


def _wait_for_parked(root: Any, *, timeout: float = 15.0) -> types.PendingJoin:
    """The parked ceremony phase one writes, waited for by the second invocation."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        rows = store.pending_joins(root)
        if rows:
            return rows[-1]
        time.sleep(0.05)
    raise AssertionError("the parking invocation never wrote a pending-join record")


def test_a_parked_pairing_joins_only_after_the_code_arrives(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
) -> None:
    """The ceremony, with the parker in a thread because it WAITS for its answer.

    The order of the assertions is the property: nothing has been sent while the
    ceremony is parked, a wrong code changes nothing at all, and the right one — the
    transcription this device derived, which is what a person types after comparing the
    two screens — completes a pair over real sockets.
    """
    server_a, server_b, host, port = devices
    record, minted = _minted(server_a)

    parked: dict[str, Any] = {}
    failures: list[BaseException] = []

    def _park_it() -> None:
        try:
            parked["result"] = _park(server_b, host=host, port=port, minted=minted)
        except BaseException as exc:  # noqa: BLE001 — reported below, not swallowed
            failures.append(exc)

    thread = threading.Thread(target=_park_it, daemon=True)
    thread.start()
    row = _wait_for_parked(server_b.root)
    try:
        assert row.status == "awaiting_confirmation"
        assert len(row.sas) == 6 and row.sas.isdigit()
        assert row.invite_id == minted.record.invite_id
        assert row.fingerprint
        assert net_cli._process_alive(row.pid)  # noqa: SLF001 — the ceremony's holder
        assert thread.is_alive(), "a parked pairing must WAIT rather than finish"

        # NOTHING HAS BEEN SENT: no transcription reached the inviter, so its own
        # parked question does not exist, the joiner holds no network, and the
        # network still has exactly one member.
        assert server_a._ctl_pair_pending({}) == []  # noqa: SLF001 — the CLI's control op
        assert store.list_networks(server_b.root) == []
        assert len(store.load(record.network_id, server_a.root).active_members()) == 1

        # A SECOND PARK ON THE SAME INVITE IS REFUSED, and before it dials: two sockets
        # for one invite would race for the single admission the inviter grants, and the
        # first parker would be left waiting on an answer written for the second.
        with pytest.raises(types.MeshRefusal) as duplicate:
            net_cli._cmd_join(_park_args(token=minted.token))  # noqa: SLF001
        assert duplicate.value.code == "pairing_already_parked"
        assert store.pending_joins(server_b.root)[-1].pid == row.pid

        # A MISTYPED DIGIT IS NOT A SPENT INVITE: it is refused by name, the ceremony
        # stays open, and the peer is still untouched.
        wrong = f"{(int(row.sas) + 1) % 1000000:06d}"
        with pytest.raises(types.MeshRefusal) as refused:
            net_cli._cmd_join(_confirm_args(wrong))  # noqa: SLF001
        assert refused.value.code == "sas_mismatch"
        assert store.pending_joins(server_b.root)[-1].status == "awaiting_confirmation"
        assert server_a._ctl_pair_pending({}) == []  # noqa: SLF001

        answered: dict[str, Any] = {}

        def _inviter_person() -> None:
            answered.update(_answer_confirmation(server_a) or {})

        answer_thread = threading.Thread(target=_inviter_person, daemon=True)
        answer_thread.start()
        try:
            # THE JOINER'S PERSON: the code this device printed, read on its own
            # screen and typed into a SECOND invocation — the second process is the
            # shape, and it is what makes this the two-phase pair rather than a prompt.
            code = net_cli._cmd_join(_confirm_args(row.sas))  # noqa: SLF001
        finally:
            answer_thread.join(20)
            thread.join(20)

        assert not failures, f"the parking side raised: {failures[0]!r}"
        assert code == 0
        assert answered, "the inviter's person never saw the parked question"
        lines, payload = parked["result"]
        assert payload["network_id"] == record.network_id
        assert payload["device_id"] == server_b.identity.device_id
        assert any("joined" in line for line in lines)
        # The joining device now holds the network, the inviter counts two members, and
        # the parked record is gone rather than lingering as a live ceremony.
        assert [item.network_id for item in store.list_networks(server_b.root)] == [
            record.network_id
        ]
        assert len(store.load(record.network_id, server_a.root).active_members()) == 2
        assert store.pending_joins(server_b.root) == []
    finally:
        thread.join(5)


def test_an_unanswered_park_gives_up_and_joins_nothing(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
) -> None:
    """``JoinParkUnanswered`` is the design's exit ``3``: a human was required, and
    nobody came. The window is ``min(remaining, 180)``, so a short invite is what makes
    this test seconds rather than minutes.
    """
    server_a, server_b, host, port = devices
    record, minted = _minted(server_a, ttl_s=2.0)

    with pytest.raises(types.JoinParkUnanswered) as refused:
        _park(server_b, host=host, port=port, minted=minted)
    assert refused.value.code == "pairing_unanswered"
    assert "read back" in refused.value.sentence or "confirm" in refused.value.sentence
    rows = store.pending_joins(server_b.root)
    assert rows and rows[-1].status == "unanswered"
    assert rows[-1].error_code == "pairing_unanswered"
    assert store.list_networks(server_b.root) == []
    assert len(store.load(record.network_id, server_a.root).active_members()) == 1


def test_the_join_exit_code_three_reaches_the_caller(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The exception-to-exit-code mapping, asserted on the dispatcher itself: ``3`` for
    a missing human, ``1`` for everything else, and the same ``--json`` body either way.
    """

    def _unanswered(args: Namespace) -> int:
        del args
        raise types.JoinParkUnanswered("pairing_unanswered", "nobody answered in time")

    def _refused(args: Namespace) -> int:
        del args
        raise types.MeshRefusal("declined", "this device declined the pairing")

    monkeypatch.setitem(net_cli._HANDLERS, "join", _unanswered)  # noqa: SLF001
    assert net_cli.main(Namespace(network_command="join", json=True)) == 3
    body = json.loads(capsys.readouterr().out)
    assert body == {
        "ok": False,
        "code": "pairing_unanswered",
        "message": "nobody answered in time",
    }

    monkeypatch.setitem(net_cli._HANDLERS, "join", _refused)  # noqa: SLF001
    assert net_cli.main(Namespace(network_command="join", json=True)) == 1


def test_the_cli_parses_the_two_phases_as_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flags exist on the REAL parser, so the guide's command lines are commands."""
    import argparse

    parser = argparse.ArgumentParser(prog="lop")
    subparsers = parser.add_subparsers(dest="subcommand")
    net_cli.add_parser(subparsers)
    park = parser.parse_args(["network", "join", "@invite.token", "--park", "--json"])
    assert park.park is True and park.confirm == "" and park.token == "@invite.token"
    answer = parser.parse_args(["network", "join", "--confirm", "481926", "--json"])
    assert answer.confirm == "481926" and answer.park is False
    assert answer.token == ""


def test_the_tool_parks_and_the_person_runs_phase_two(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The agent path end to end: the tool PARKS, the person's own command finishes it.

    This is the transcript the brief asks for, in the shape agent review round 1 and
    the operator's ruling settled. The first tool call starts the pairing and hands back
    the code; nothing is joined. Phase two is then the PERSON's command — the CLI's own
    ``lop network join --confirm <code>``, which is exactly what the guide hands over —
    and the tool is used again only to VERIFY the membership it can no longer create.
    That is still "fully handle the connection once approved": the approval IS the
    human's confirmation.
    """
    server_a, server_b, host, port = devices
    record, minted = _minted(server_a)

    def _call(args: dict[str, Any]) -> Any:
        async def _run() -> Any:
            return await net_tool.execute_network("call-1", args, None, None, ToolContext(cwd="."))

        return asyncio.run(_run())

    started = _call({"action": "join", "token": minted.token})
    assert not started.is_error, _digest(started)
    digest = _digest(started)
    row = _wait_for_parked(server_b.root)
    # The code the agent is told to show the user is THIS device's derivation, rendered
    # the way a person reads it.
    assert row.sas in digest.replace(" ", "")
    assert f"{row.sas[:3]} {row.sas[3:]}" in digest
    assert "pairing started" in digest
    assert "nothing is joined until the person confirms" in digest

    # A PARKED PAIRING HAS SENT NOTHING, and the ceremony is held by a REAL process —
    # the one the second call has to reach. The inviter's own parked question is the
    # sharpest form of "nothing was sent": the relay only parks one when a
    # transcription has ARRIVED, so an empty queue is proof that the frame carrying the
    # human step is still unsent.
    assert store.list_networks(server_b.root) == []
    assert len(store.load(record.network_id, server_a.root).active_members()) == 1
    assert server_a._ctl_pair_pending({}) == []  # noqa: SLF001 — the CLI's own control op
    assert row.pid != os.getpid()
    assert net_cli._process_alive(row.pid)  # noqa: SLF001

    # THE TOOL CANNOT ANSWER IT. Not "will not": there is no field, so the second
    # phase is a command only a person can run, and an agent that wanted to skip it
    # would have nothing to call.
    assert "confirm" not in net_tool.NetworkParams.model_fields

    answered: dict[str, Any] = {}
    recorded: list[Any] = []
    real_save = store.save_join_answer

    def _capture(answer: Any) -> Any:
        recorded.append(answer)
        return real_save(answer)

    monkeypatch.setattr(store, "save_join_answer", _capture)

    answer_thread = threading.Thread(
        target=lambda: answered.update(_answer_confirmation(server_a) or {}), daemon=True
    )
    answer_thread.start()
    try:
        # THE PERSON'S STEP, run where they run it: the CLI's own phase two, with the
        # code the user read off the other device's screen.
        receipt = net_cli._cmd_join(_confirm_args(row.sas))  # noqa: SLF001 — the CLI's own
    finally:
        answer_thread.join(20)

    assert receipt == 0
    assert answered, "the inviter's person never saw the parked question"
    # WHO ANSWERED, in the record's own words. The code arrived from a machine here (a
    # test cannot be a person: no TTY), so that is what the record says — the field is
    # for "did a person read this back?", and a machine-supplied value is no longer
    # written as a human one (semantic finding 3).
    assert [item.answered_by for item in recorded] == ["harness"], recorded

    # The record goes with the ceremony, and that is the whole of what is asserted about
    # its END here. The parked process's EXIT is deliberately not: the parker is our own
    # child, and once its stdout pipe is gone it can sit as a zombie until we reap it,
    # where a liveness probe SUCCEEDS for a corpse — so an "it is gone" assertion is a
    # statement about this process's reaping, not about the product, and it measured the
    # difference on CI (passed on macOS, failed on Linux for exactly that reason). What
    # IS asserted is the set of product properties around it: the holder is alive while
    # parked (above), a second park is refused as `pairing_already_parked` (in
    # `test_a_parked_pairing_joins_only_after_the_code_arrives`), and the record and the
    # network agree once the person has answered.
    assert store.pending_joins(server_b.root) == []
    assert [item.network_id for item in store.list_networks(server_b.root)] == [record.network_id]
    # Which is what the agent's own verification step then reports: the tool cannot
    # create the pairing, and it CAN check it.
    verified = _call({"action": "ls"})
    assert not verified.is_error
    assert record.name in _digest(verified)


def _digest(result: Any) -> str:
    return "\n".join(part.text for part in result.content)


def _pending_result_fixture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> types.PendingJoin:
    """A real stored record, without starting a pairing process or spending an invite."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    pending = types.PendingJoin(
        invite_id="synthetic-result-race",
        network_id="synthetic-network",
        network_name="result-race",
        inviter_device_id="synthetic-inviter",
        inviter_name="inviter",
        sas="123456",
        fingerprint="synthetic-fingerprint",
        pid=424242,
        expires_at=time.time() + 30,
    )
    store.save_pending_join(pending)
    return pending


@pytest.mark.parametrize("status", ["joined", "refused"])
def test_a_pairing_outcome_published_between_read_and_exit_probe_is_preserved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str
) -> None:
    pending = _pending_result_fixture(monkeypatch, tmp_path)

    def finished_between_read_and_probe(pid: int) -> bool:
        assert pid == pending.pid
        # The real parker publishes its outcome, then exits. Force that exact
        # interleaving after the reader took its awaiting-confirmation snapshot;
        # disk serialization makes this a distinct row, not a mutated test double.
        net_cli._parked_outcome(
            pending,
            status=status,
            store=store,
            result={"ok": status == "joined"},
            error_code="declined" if status == "refused" else "",
        )
        return False

    monkeypatch.setattr(net_cli, "_process_alive", finished_between_read_and_probe)
    outcome = net_cli._await_join_result(pending, store)
    assert outcome is not None, "a published outcome was mistaken for an abandoned ceremony"
    assert outcome.status == status
    assert outcome.pid == 0
    assert outcome.result == {"ok": status == "joined"}
    assert outcome.error_code == ("declined" if status == "refused" else "")


def test_a_dead_pairing_without_a_published_outcome_is_still_abandoned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The record GONE is the true abandonment: nothing is left to read or publish.

    The briefer, single-parameter form of this case (a constant ``removed=True``
    parametrization) was folded into the body; the ``removed=False`` arm it once had
    asserted ``None`` for a live row, which was the silent-death bug itself. That case
    is now covered from the truthful side by
    ``test_a_parker_that_died_unpublished_leaves_a_terminal_outcome`` below.
    """
    pending = _pending_result_fixture(monkeypatch, tmp_path)

    def dead_without_outcome(_pid: int) -> bool:
        store.clear_pending_join(pending.invite_id)
        return False

    monkeypatch.setattr(net_cli, "_process_alive", dead_without_outcome)
    assert net_cli._await_join_result(pending, store) is None


def test_a_parker_that_died_unpublished_leaves_a_terminal_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """RED-first: a dead parker with a still-awaiting row must publish the TRUTH.

    THE FLAKE (main run 38841686162, 1 of ~3-in-45): a parked ceremony whose child
    died during ``result_read`` without publishing a terminal row left the record
    ``awaiting_confirmation`` with a dead pid, so ``_await_join_result`` returned
    ``None`` and the answering invocation reported "nothing was joined" over a
    ceremony that had in fact stopped. A SIGKILLed parker cannot run its own writer,
    so the fix has to be READER-side: the observer publishes the terminal outcome.

    A synthetic record is the controlled boundary here (as in the sibling race test):
    a real ``SIGKILL`` of a live parker mid-``result_read`` is not reproducible on
    demand, and ``_parked_outcome`` / ``_await_join_result`` are the real functions
    against the real disk store.
    """
    pending = _pending_result_fixture(monkeypatch, tmp_path)

    monkeypatch.setattr(net_cli, "_process_alive", lambda _pid: False)
    outcome = net_cli._await_join_result(pending, store)

    # NOT ``None``: the ceremony is reported, not invented as abandoned.
    assert outcome is not None, "a dead parker with a live row was read as abandonment"
    assert outcome.status == "refused"
    assert outcome.error_code == "pairing_stopped"
    assert outcome.message == net_cli._JOIN_STOPPED_SENTENCE  # noqa: SLF001
    assert outcome.pid == 0, "a terminal outcome must zero the pid it reports"
    # DURABLE: the row on disk is terminal too, so no later reader finds it lingering.
    stored = store.pending_join(pending.invite_id)
    assert stored is not None and stored.status == "refused"
    assert stored.error_code == "pairing_stopped"


def test_answering_a_vanished_parker_says_what_happened(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The answering invocation must carry the stopped sentence, not a bare abandonment.

    This is the phase-two half of the flake: when the parked record is gone entirely
    by the time the reader looks, ``_answer_parked_join`` is the surface the person
    sees, and it has to name the stop rather than the generic "nothing was joined".
    """
    pending = _pending_result_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(net_cli, "_live_parked_joins", lambda _store: [pending])
    monkeypatch.setattr(net_cli, "_await_join_result", lambda _pending, _store: None)

    with pytest.raises(types.MeshRefusal) as refused:
        net_cli._answer_parked_join(_confirm_args(pending.sas), pending.sas)  # noqa: SLF001
    assert refused.value.code == "pairing_stopped"
    assert refused.value.sentence == net_cli._JOIN_STOPPED_SENTENCE  # noqa: SLF001


def test_the_parker_publishes_when_its_own_finish_is_cut_by_the_link(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The WRITER half: a parker whose own ``result_read`` dies publishes the truth.

    The reader-side fix cannot cover a death the parker LIVES through and could have
    reported — that is this path. ``_finish_pairing`` raises the transport error the
    flake names (``wire.LinkCryptoError``), and the parked row must go TERMINAL with
    ``pairing_stopped`` rather than linger ``awaiting_confirmation``. ``_park_join`` is
    called directly with the real disk store and a pre-written answer (so it proceeds
    straight to the finish); the socket/codec/reader are controlled boundaries, which
    is the qualification: the subject is the parker's own catch-and-publish, not the
    wire.
    """
    server_a, server_b, host, port = devices
    _record, minted = _minted(server_a)

    # A REAL attempt record, left at ``result_read`` as the real parker leaves it before
    # the wire read: the fix is gated on that stage, so a synthetic attempt at the wrong
    # stage would not exercise it.
    attempt = net_cli._JoinAttempt(host=f"{host}:{port}")  # noqa: SLF001
    attempt.stage = "result_read"

    def _cut_by_the_link(**_kwargs: Any) -> Any:
        raise wire.LinkCryptoError("the result read failed authentication", kind="auth")

    monkeypatch.setattr(net_cli, "_finish_pairing", _cut_by_the_link)
    # The park CLEARS any pre-written answer before it waits, so the answer is supplied
    # through the wait itself: a controlled boundary for the human step, which is not
    # what this test is about. It returns the admit answer that reaches the finish.
    monkeypatch.setattr(
        net_cli,
        "_await_join_answer",
        lambda invite_id, _window, _store: types.JoinAnswer(
            invite_id=invite_id, decision="admit", code="123456", answered_by="harness"
        ),
    )

    with pytest.raises(wire.LinkCryptoError):
        net_cli._park_join(  # noqa: SLF001 — the CLI's own parker
            args=_park_args(name=server_b.identity.name),
            store=store,
            envelope=minted.envelope,
            result=Namespace(sas="123456"),
            fingerprint="synthetic-fingerprint",
            sock=None,
            codec=None,
            reader=None,
            host=f"{host}:{port}",
            identity=server_b.identity,
            advertised=[],
            handshake=None,
            offer_view={"state": "absent", "items": []},
            attempt=attempt,
            helpers={
                "invite_mod": invite_mod,
                "pair_timeout_seconds": pair_timeout_seconds,
                "pair_abort_frame": pair_abort_frame,
            },
        )

    row = store.pending_join(minted.envelope.invite_id)
    assert row is not None, "the parked record went with the ceremony without an outcome"
    assert row.status == "refused"
    assert row.error_code == "pairing_stopped"
    assert row.message == net_cli._JOIN_STOPPED_SENTENCE  # noqa: SLF001
    assert row.pid == 0, "a terminal outcome must zero the pid it reports"


def test_a_local_disk_fault_is_not_relabelled_a_peer_side_stop(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F2: the writer clause must NOT claim "invite is untouched" over a local fault.

    ``_finish_pairing`` advances the attempt to ``persist`` once the admission is read,
    so an ``OSError`` from the LOCAL ``_persist_join`` write reaches the same ``try`` but
    a different stage. The parker must leave that row alone (the outer handler records
    the attempt) rather than publish ``pairing_stopped`` with the wire-death sentence.
    """
    server_a, server_b, host, port = devices
    _record, minted = _minted(server_a)

    attempt = net_cli._JoinAttempt(host=f"{host}:{port}")  # noqa: SLF001
    # ``persist``: the read succeeded and the LOCAL write is what failed.
    attempt.stage = "persist"

    def _disk_full(**_kwargs: Any) -> Any:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(net_cli, "_finish_pairing", _disk_full)
    monkeypatch.setattr(
        net_cli,
        "_await_join_answer",
        lambda invite_id, _window, _store: types.JoinAnswer(
            invite_id=invite_id, decision="admit", code="123456", answered_by="harness"
        ),
    )

    with pytest.raises(OSError):
        net_cli._park_join(  # noqa: SLF001 — the CLI's own parker
            args=_park_args(name=server_b.identity.name),
            store=store,
            envelope=minted.envelope,
            result=Namespace(sas="123456"),
            fingerprint="synthetic-fingerprint",
            sock=None,
            codec=None,
            reader=None,
            host=f"{host}:{port}",
            identity=server_b.identity,
            advertised=[],
            handshake=None,
            offer_view={"state": "absent", "items": []},
            attempt=attempt,
            helpers={
                "invite_mod": invite_mod,
                "pair_timeout_seconds": pair_timeout_seconds,
                "pair_abort_frame": pair_abort_frame,
            },
        )

    row = store.pending_join(minted.envelope.invite_id)
    assert row is not None
    assert row.status == "awaiting_confirmation", (
        "a local disk fault was reported as a peer-side stop: "
        f"status={row.status!r} code={row.error_code!r}"
    )
    assert row.error_code == ""
