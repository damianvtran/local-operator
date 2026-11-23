"""The joiner's shared answer module, and the agent-facing copy after the move to the app.

``network.pairing_answers`` is the ONE checker and writer behind both doors that
answer a parked join: ``lop network join --confirm <code>`` (terminal fallback) and the
Mesh approvals card route. Parity is pinned by driving the CLI's own phase two and
the card with the same code against the same parked record and comparing the answer
each writes. The copy cells pin the operator's request — pairing must not require
terminal commands — on the surfaces an agent reads: the ``network`` tool's receipt and
hint, and the park JSON's ``sentence``, which now point at the app and name no
``lop network join --confirm`` command.
"""

from __future__ import annotations

import os
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from local_operator.network import cli as net_cli
from local_operator.network import pairing_answers, store
from local_operator.network import tool as net_tool
from local_operator.network import types

CODE = "481926"


def _park(root: Path, *, invite_id: str = "i_parity", verify: bool = False) -> types.PendingJoin:
    store.ensure_pending_dir(root)
    pending = types.PendingJoin(
        invite_id=invite_id,
        network_id="n_" + "3" * 24,
        network_name="home-net",
        inviter_device_id="d_" + "a" * 32,
        inviter_name="desk",
        sas=CODE,
        fingerprint="K7QM-3XPD-4WZ9-8NRB",
        verify=verify,
        pid=os.getpid(),
        expires_at=time.time() + 300,
        nonce=types.new_pairing_nonce(),
    )
    store.save_pending_join(pending, root)
    return pending


def test_the_cli_and_the_card_write_the_same_answer(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    pending = _park(root)
    written: list[types.JoinAnswer] = []
    real_submit = pairing_answers.submit

    def spy(store_: Any, row: Any, **kwargs: Any) -> types.JoinAnswer:
        answer = real_submit(store_, row, **kwargs)
        written.append(answer)
        return answer

    monkeypatch.setattr(pairing_answers, "submit", spy)
    # The CLI's phase two, with the parker's outcome supplied (no real socket here).
    monkeypatch.setattr(
        net_cli,
        "_await_join_result",
        lambda _pending, _store: SimpleNamespace(
            status="joined", result={"ok": True}, result_lines=[]
        ),
    )
    monkeypatch.setattr(net_cli, "_emit", lambda _args, payload, _lines: 0)
    args = Namespace(json=True, sas_stdin=False)
    assert net_cli._answer_parked_join(args, CODE) == 0  # noqa: SLF001
    cli_answer = written[-1]

    pending = _park(root)
    outcome = pairing_answers.answer_card(
        SimpleNamespace(
            pending_joins=lambda: store.pending_joins(root),
            join_answer=lambda i: store.join_answer(i, root),
            save_join_answer=lambda a: store.save_join_answer(a, root),
        ),
        pending.invite_id,
        decision="admit",
        code=CODE,
        nonce=pending.nonce,
        answered_by="desktop",
    )
    assert outcome.status == "waiting_for_other_device"
    card_answer = written[-1]
    strip = {"answered_by", "answered_at"}
    assert {k: v for k, v in cli_answer.to_json().items() if k not in strip} == {
        k: v for k, v in card_answer.to_json().items() if k not in strip
    }
    assert pairing_answers.parked_code_matches(pending, CODE)
    assert net_cli._parked_code_matches(pending, CODE)  # noqa: SLF001 — the CLI's alias
    assert not net_cli._parked_code_matches(pending, "000000")  # noqa: SLF001


def test_a_stale_nonce_or_a_dead_parker_answers_nothing(root: Path) -> None:
    pending = _park(root)
    bound = SimpleNamespace(
        pending_joins=lambda: store.pending_joins(root),
        join_answer=lambda i: store.join_answer(i, root),
        save_join_answer=lambda a: store.save_join_answer(a, root),
    )
    cases: list[tuple[str, Callable[[int], bool]]] = [
        ("00" * 16, pairing_answers.process_alive),  # a stale card
        (pending.nonce, lambda _pid: False),  # the parker is gone
    ]
    for nonce, alive in cases:
        with pytest.raises(types.MeshRefusal) as excinfo:
            pairing_answers.answer_card(
                bound,
                pending.invite_id,
                decision="admit",
                code=CODE,
                nonce=nonce,
                answered_by="desktop",
                alive=alive,
            )
        assert excinfo.value.code == "pairing_expired"
    assert store.join_answer(pending.invite_id, root) is None


def test_the_park_sentence_points_at_the_app_not_a_command() -> None:
    pending = types.PendingJoin(
        invite_id="i_x",
        network_id="n_x",
        network_name="home-net",
        inviter_device_id="d_x",
        inviter_name="desk",
        sas=CODE,
        fingerprint="K7QM",
        expires_at=time.time() + 60,
    )
    sentence = net_cli._awaiting_payload(pending)["sentence"]  # noqa: SLF001
    assert "Mesh approvals" in sentence and "481 926" in sentence
    assert "lop network join --confirm" not in sentence
    # ``--verify`` (fingerprint) cannot be typed into a six-digit card: terminal-only.
    pending.verify = True
    assert (
        "lop network join --confirm" in net_cli._awaiting_payload(pending)["sentence"]
    )  # noqa: SLF001


def test_the_tool_copy_names_the_app_and_no_confirm_command() -> None:
    receipt = net_tool._render(  # noqa: SLF001
        "join",
        {
            "status": "awaiting_confirmation",
            "sas": CODE,
            "name": "home-net",
            "inviter": "desk",
            "sentence": "s",
            "seconds_left": 512.0,
        },
    )
    text = "\n".join(receipt) + "\n" + net_tool._hint("join")  # noqa: SLF001
    assert "Mesh approvals" in text and "typing the code the other device shows" in text
    assert "lop network join --confirm" not in text
    assert "512s left" in text
    assert "lop network join --confirm" not in (net_tool.__doc__ or "")


def _bound(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        pending_joins=lambda: store.pending_joins(root),
        join_answer=lambda i: store.join_answer(i, root),
        save_join_answer=lambda a: store.save_join_answer(a, root),
    )


def test_card_strikes_are_forgotten_when_the_ceremony_ends(root: Path) -> None:
    """N4: the strikes map shrinks. An answered ceremony's entry goes with the answer,
    and an entry for a ceremony that is no longer parked goes on the next answer."""
    pairing_answers._strikes.clear()  # noqa: SLF001 — process-wide; start from empty
    first = _park(root, invite_id="i_first")
    outcome = pairing_answers.answer_card(
        _bound(root),
        first.invite_id,
        decision="admit",
        code="000000",
        nonce=first.nonce,
        answered_by="desktop",
    )
    assert outcome.status == "sas_mismatch"
    assert (first.invite_id, first.nonce) in pairing_answers._strikes  # noqa: SLF001
    outcome = pairing_answers.answer_card(
        _bound(root),
        first.invite_id,
        decision="admit",
        code=CODE,
        nonce=first.nonce,
        answered_by="desktop",
    )
    assert outcome.status == "waiting_for_other_device"
    assert pairing_answers._strikes == {}, "an answered ceremony kept its strikes"  # noqa: SLF001

    gone = _park(root, invite_id="i_gone")
    pairing_answers.answer_card(
        _bound(root),
        gone.invite_id,
        decision="admit",
        code="000000",
        nonce=gone.nonce,
        answered_by="desktop",
    )
    store.clear_pending_join(gone.invite_id, root)  # its window closed / parker ended
    other = _park(root, invite_id="i_other")
    pairing_answers.answer_card(
        _bound(root),
        other.invite_id,
        decision="admit",
        code="000000",
        nonce=other.nonce,
        answered_by="desktop",
    )
    assert set(pairing_answers._strikes) == {(other.invite_id, other.nonce)}  # noqa: SLF001
    pairing_answers._strikes.clear()  # noqa: SLF001


def test_a_spent_invite_refusal_drops_the_same_token_advice() -> None:
    """Q3: when the inviting device consumed the invite (``spent`` on the abort frame)
    the remedy is a fresh invite, not "run it again with the SAME token"."""
    from local_operator.network import handshake as hs_mod

    frame = hs_mod.pair_abort_frame(req=0, reason="sas_mismatch", detail="attempts spent")
    assert "spent" not in frame, "the key is additive and only present when true"
    assert hs_mod.pair_abort_frame(req=0, reason="sas_mismatch", spent=True)["spent"] is True
    retry = str(hs_mod.refusal_from_pairing("sas_mismatch", detail="x"))
    used = str(hs_mod.refusal_from_pairing("sas_mismatch", detail="x", spent=True))
    assert "SAME token" in retry
    assert "SAME token" not in used and "mint a fresh invite" in used


def test_a_minted_invite_names_the_inviter_by_its_device_name() -> None:
    """Q1: the envelope carries the inviter's own member-row name when no listener name
    was set, so the joiner's card shows a name rather than a device id."""
    from local_operator.network import invite as invite_mod
    from local_operator.network import wire

    record = types.NetworkRecord(network_id="n_" + "5" * 24, name="home-net")
    record.self_device_id = "d_" + "e" * 32
    record.members.append(types.MemberRecord(device_id=record.self_device_id, name="studio-mac"))
    material = wire.b64u(b"\x05" * 32)
    minted = invite_mod.mint(record, material, role="drive")
    assert minted.envelope.inviter_name == "studio-mac"
    record.members.clear()
    assert invite_mod.mint(record, material, role="drive").envelope.inviter_name == (
        record.self_device_id
    )
