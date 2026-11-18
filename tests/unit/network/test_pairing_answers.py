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
