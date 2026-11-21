"""The inviter's half of the human step, on the CLI the operator actually has (R4).

``mesh-transport-identity.md`` §5.3 is "B transcribes, A compares". The joiner's half
(the transcription) was always there; the inviter's half was not — nothing on the
inviting device showed a code, so there was nothing for its human to compare against
and the check was decorative.

The relay is a launchd daemon on the device that invites, so it cannot prompt. It
parks the pairing in a 0600 record carrying BOTH codes, and ``lop network confirm``
asks the person. This file pins that flow, and pins that an agent cannot answer it.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.network import cli as net_cli
from local_operator.network import relay, store, types
from local_operator.network.audit import AuditLog

NETWORK = "n_0123456789abcdef01234567"


def _pending(
    root: Path, *, sas: str = "481926", transcribed: str = "481926"
) -> types.PendingPairing:
    pending = types.PendingPairing(
        invite_id="i_abc123",
        network_id=NETWORK,
        network_name="home-net",
        joiner_device_id="d_" + "b" * 32,
        joiner_name="laptop",
        sas=sas,
        fingerprint="K7QM-3XPD-4WZ9-8NRB",
        transcribed=transcribed,
        peer_addr="127.0.0.1:4097",
        expires_at=time.time() + 120,
        prompt=(
            f'd_{"b" * 32} ("laptop", new device) transcribed 481 926 to join home-net as '
            "drive.\nYOUR screen shows 481 926.\nDo they match?"
        ),
        # Every relay park stamps a fresh ceremony (``new_pairing_nonce``), and the
        # answer latch only recognises decisions stamped with it (S-R3-1), so the
        # fixture parks the way production does.
        ceremony=types.new_pairing_nonce(),
    )
    store.save_pending_pairing(pending, root)
    return pending


@contextmanager
def _live_relay(root: Path) -> Iterator[relay.RelayServer]:
    """A started relay on loopback whose control socket the CLI really dials.

    ``_cmd_confirm`` finds it through ``store.find_own_relay`` and talks to it over
    ``relay.control_request`` — the same hop ``lop network confirm`` takes — so the
    cells below prove the CLI's outcome against what the relay actually recorded.
    """
    server = relay.RelayServer(
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        audit=AuditLog(root),
    )
    assert server.root == root, f"the serve-shaped relay resolved {server.root}"
    server.start()
    try:
        assert store.find_own_relay() is not None, "the started relay published no record"
        yield server
    finally:
        server.stop()


def _pairing_events(root: Path) -> list[str]:
    return [
        str(row.get("event"))
        for row in AuditLog(root).tail(500)
        if str(row.get("event") or "").startswith("pairing_")
    ]


def _args(**fields: object) -> Namespace:
    base: dict[str, object] = {
        "json": True,
        "invite_id": "",
        "list_pending": False,
        "decline": False,
        "sas_stdin": False,
    }
    base.update(fields)
    return Namespace(**base)


def test_confirm_list_shows_both_codes_to_the_person(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--list` is how an operator SEES the question: the prompt, the joiner, and the
    code this device derived. It answers nothing."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _pending(root)
    assert net_cli._cmd_confirm(_args(list_pending=True)) == 0  # noqa: SLF001
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    rows = payload["pending"]
    assert [row["invite_id"] for row in rows] == ["i_abc123"]
    assert rows[0]["sas"] == "481926"
    assert "YOUR screen shows 481 926" in rows[0]["prompt"]
    # Listing is not answering: the pairing is still parked.
    assert store.pending_pairing("i_abc123", root) is not None
    assert store.pair_decision("i_abc123", root) is None


def test_confirm_refuses_without_a_terminal_and_answers_nothing(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE PROPERTY: an agent cannot complete a pairing.

    This test process has no TTY, which is exactly the case a scripted caller is in,
    so the refusal here is the production behaviour rather than a simulation of it.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _pending(root)
    assert net_cli._has_terminal() is False  # noqa: SLF001 — the guard itself
    with pytest.raises(types.MeshRefusal) as excinfo:
        net_cli._cmd_confirm(_args())  # noqa: SLF001
    assert excinfo.value.code == "confirm_needs_tty"
    assert "terminal" in excinfo.value.sentence
    assert "serve" in excinfo.value.sentence
    assert store.pair_decision("i_abc123", root) is None, "a refused prompt answered anyway"


def test_confirm_with_no_parked_pairing_names_the_next_step(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    with pytest.raises(types.MeshRefusal) as excinfo:
        net_cli._cmd_confirm(_args(list_pending=False))  # noqa: SLF001
    assert excinfo.value.code == "no_pending_pairing"
    # §2.9's repave (design round 1, D2): the sentence names the product actions
    # — check the queue, mint an invite — and no longer pastes a command.
    assert "Check the pairing queue, or mint an invite." in excinfo.value.sentence
    # And an unknown invite id says so instead of answering somebody else's pairing.
    _pending(root)
    with pytest.raises(types.MeshRefusal) as excinfo:
        net_cli._cmd_confirm(_args(invite_id="i_someone_else"))  # noqa: SLF001
    assert excinfo.value.code == "no_pending_pairing"
    assert "i_someone_else" in excinfo.value.sentence


def test_the_harness_seam_is_closed_unless_the_test_mode_variable_is_set(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--sas-stdin` is the e2e harness's seam. Setting it by accident (or on purpose)
    in a real pairing turns the human check into a formality, so it is refused
    outside `LOP_NETWORK_TEST_MODE=1`."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.delenv("LOP_NETWORK_TEST_MODE", raising=False)
    _pending(root)
    with pytest.raises(types.MeshRefusal) as excinfo:
        net_cli._cmd_confirm(_args(sas_stdin=True))  # noqa: SLF001
    assert excinfo.value.code == "test_seam_closed"
    assert "LOP_NETWORK_TEST_MODE" in excinfo.value.sentence


def test_a_decline_through_the_relay_never_admits(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The relay refuses to record an admission that was not a match, even when asked
    politely: the decision file is self-describing, and `matched` is the truth of it."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _pending(root)
    # Serve-shaped: the decision file this cell reads is resolved from the relay's OWN
    # root, so building it the way ``lop network serve`` does is what proves the two
    # agree (see ``test_relay_e2e.serve_shaped_relay``).
    server = relay.RelayServer(settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"))
    assert server.root == root, f"the serve-shaped relay resolved {server.root}"
    try:
        answered = server._ctl_pair_confirm(  # noqa: SLF001
            {"invite_id": "i_abc123", "decision": "admit", "matched": False}
        )
        assert answered["decision"] == "decline"
        assert answered["matched"] is False
        decision = store.pair_decision("i_abc123", root)
        assert decision is not None and decision.matched is False

        # THE DECISION IS TERMINAL (security review round 1, S-R1-2): a second answer
        # to the same ceremony is refused and the decline stands.
        with pytest.raises(types.MeshRefusal) as excinfo:
            server._ctl_pair_confirm(  # noqa: SLF001
                {"invite_id": "i_abc123", "decision": "admit", "matched": True}
            )
        # Outcome-neutral code (S-R2-2): the first answer could have been an admit.
        assert excinfo.value.code == "pairing_answered"
        kept = store.pair_decision("i_abc123", root)
        assert kept is not None and kept.decision == "decline"

        # And an "admit that did not match" is impossible by construction (a fresh
        # ceremony: the listener clears the decision when one ends).
        store.clear_pair_decision("i_abc123", root)
        server._ctl_pair_confirm(  # noqa: SLF001
            {"invite_id": "i_abc123", "decision": "admit", "matched": True, "answered_by": "human"}
        )
        matched = store.pair_decision("i_abc123", root)
        assert matched is not None and matched.decision == "admit" and matched.matched
    finally:
        server.stop()


def test_confirm_on_a_named_invite_answers_that_one(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two parked pairings at once is a real state (two people, one device), so the
    command must answer the one it was asked about."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _pending(root)
    second = types.PendingPairing(
        invite_id="i_def456",
        network_id=NETWORK,
        network_name="home-net",
        joiner_device_id="d_" + "c" * 32,
        joiner_name="phone",
        sas="731502",
        fingerprint="AAAA-BBBB",
        expires_at=time.time() + 120,
        prompt="second question",
    )
    store.save_pending_pairing(second, root)

    # Through a live relay: an answer is only ever recorded by the relay (S-R3-1).
    with _live_relay(root):
        assert net_cli._cmd_confirm(_args(decline=True, invite_id="i_def456")) == 0  # noqa: SLF001
    assert store.pair_decision("i_def456", root) is not None
    assert store.pair_decision("i_abc123", root) is None, "the wrong pairing was answered"
    assert store.pending_pairing("i_abc123", root) is not None


# ---------------------------------------------------------------------------
# Only a live relay acknowledgment is a terminal answer (security review round 3,
# S-R3-1 / agent review n-R3-1 / QA Q-R3-1)
# ---------------------------------------------------------------------------


def test_confirm_with_no_relay_records_nothing_and_says_so(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) No relay: the request never left this device, so nothing CAN have been
    recorded. The CLI used to write an unstamped decision file plus a
    ``pairing_refused`` audit row and exit 0 — a success nobody was watching for,
    and a file that later blocked the live answer."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _pending(root)
    assert store.find_own_relay() is None
    with pytest.raises(types.MeshRefusal) as excinfo:
        net_cli._cmd_confirm(_args(decline=True))  # noqa: SLF001
    assert excinfo.value.code == net_cli.CODE_RELAY_UNAVAILABLE
    assert excinfo.value.sentence == (
        "No relay is running on this device, so nothing was recorded. " "Start it and answer again."
    )
    assert store.pair_decision("i_abc123", root) is None, "an offline answer was persisted"
    assert _pairing_events(root) == [], "an offline answer was audited as a success"
    # The real CLI entry point turns that refusal into a non-zero exit.
    args = _args(decline=True)
    args.network_command = "confirm"
    assert net_cli.main(args) == 1


def test_confirm_reply_lost_before_the_relay_handled_it_leaves_the_ceremony_open(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(b) The frame went out but no reply came back, and the relay never handled it.
    The CLI must not claim either outcome, must write nothing itself, and a valid
    live answer afterwards must still land."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    pending = _pending(root)
    real_control = relay.control_request
    with _live_relay(root):
        # Sent-but-unanswered, as ``report_unsent`` callers see it: no
        # ``ControlRequestUnsent`` (the frame left) and no reply (``None``); the
        # relay never handled it.
        monkeypatch.setattr(relay, "control_request", lambda *a, **k: None)
        with pytest.raises(types.MeshRefusal) as excinfo:
            net_cli._cmd_confirm(_args(decline=True))  # noqa: SLF001
        assert excinfo.value.code == "pairing_answer_unconfirmed"
        assert excinfo.value.sentence == (
            "Could not confirm whether the relay recorded this answer."
        )
        assert "nothing was recorded" not in excinfo.value.sentence
        assert store.pair_decision("i_abc123", root) is None
        assert _pairing_events(root) == []

        monkeypatch.setattr(relay, "control_request", real_control)
        assert net_cli._cmd_confirm(_args(decline=True)) == 0  # noqa: SLF001
    decided = store.pair_decision("i_abc123", root)
    assert decided is not None and decided.decision == "decline"
    assert decided.ceremony == pending.ceremony, "the live answer was not stamped"


def test_confirm_reply_lost_after_the_relay_recorded_never_claims_nothing_happened(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(c) The relay recorded the answer and only the reply was lost. The CLI says it
    could not confirm (never "nothing was recorded"), the first decision stands, and
    a second answer is refused by the terminal latch."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    pending = _pending(root)
    real_control = relay.control_request

    def _reply_lost(*args: Any, **kwargs: Any) -> None:
        real_control(*args, **kwargs)  # the relay really handles and records it
        return None

    with _live_relay(root) as server:
        monkeypatch.setattr(relay, "control_request", _reply_lost)
        with pytest.raises(types.MeshRefusal) as excinfo:
            net_cli._cmd_confirm(_args(decline=True))  # noqa: SLF001
        assert excinfo.value.code == "pairing_answer_unconfirmed"
        assert "Could not confirm" in excinfo.value.sentence
        assert "nothing was recorded" not in excinfo.value.sentence
        first = store.pair_decision("i_abc123", root)
        assert first is not None and first.decision == "decline"
        assert first.ceremony == pending.ceremony

        # The second answer is an ADMIT at a terminal: it must not replace the
        # recorded decline.
        monkeypatch.setattr(relay, "control_request", real_control)
        monkeypatch.setattr(net_cli, "_has_terminal", lambda: True)
        monkeypatch.setattr("builtins.input", lambda prompt="": "y")
        with pytest.raises(types.MeshRefusal) as second:
            net_cli._cmd_confirm(_args())  # noqa: SLF001
        assert second.value.code == "pairing_answered"
        # The desktop card's route maps the same latch refusal to HTTP 409.
        from local_operator.server.routes.desktop_mesh_pairings import _http

        assert _http(second.value).status_code == 409
        assert server is not None
    kept = store.pair_decision("i_abc123", root)
    assert kept is not None and kept.decision == "decline" and kept.matched is False
    assert kept.ceremony == first.ceremony


def test_a_stray_unstamped_decision_does_not_block_a_live_answer(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(d) Q-R3-1: an unstamped decision file (out-of-band, or left by the old CLI
    fallback) is never an answer to the poll, so it must not latch the ceremony shut
    either — the live answer replaces it, stamped with the current ceremony."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    pending = _pending(root)
    store.save_pair_decision(
        types.PairDecision(
            invite_id="i_abc123", decision="decline", matched=False, reason="declined"
        ),
        root,
    )
    stray = store.pair_decision("i_abc123", root)
    assert stray is not None and stray.ceremony == ""
    with _live_relay(root):
        assert net_cli._cmd_confirm(_args(decline=True)) == 0  # noqa: SLF001
    decided = store.pair_decision("i_abc123", root)
    assert decided is not None and decided.ceremony == pending.ceremony
    # The poll and the latch read ONE rule: an unstamped file answers no ceremony
    # (not even an unstamped one), and only the live stamp does.
    assert relay._answers_ceremony(stray, pending.ceremony) is False  # noqa: SLF001
    assert relay._answers_ceremony(stray, "") is False  # noqa: SLF001
    assert relay._answers_ceremony(decided, pending.ceremony) is True  # noqa: SLF001
    assert relay._answers_ceremony(decided, "another-ceremony") is False  # noqa: SLF001


def test_control_request_tells_an_unsent_request_from_a_lost_reply(root: Path) -> None:
    """The client half of the split, on real loopback sockets: a refused connect
    raises ``ControlRequestUnsent`` (the op cannot have run), a connection that took
    the frame and closed without replying is ``None`` (it may have run). Without the
    opt-in both stay the plain ``None`` every other caller reads."""
    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    dead_port = closed.getsockname()[1]
    closed.close()
    dead = cast(Any, type("Rec", (), {"control_port": dead_port, "control_key": "k"})())
    assert relay.control_request(dead, "net_pair_confirm", timeout=1.0) is None
    with pytest.raises(types.ControlRequestUnsent):
        relay.control_request(dead, "net_pair_confirm", timeout=1.0, report_unsent=True)

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    received: list[bytes] = []

    def _swallow() -> None:
        conn, _ = listener.accept()
        with conn:
            conn.settimeout(2.0)
            data = b""
            while data.count(b"\n") < 2:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk
            received.append(data)

    thread = threading.Thread(target=_swallow, daemon=True)
    thread.start()
    live = cast(
        Any, type("Rec", (), {"control_port": listener.getsockname()[1], "control_key": "k"})()
    )
    try:
        assert (
            relay.control_request(live, "net_pair_confirm", timeout=2.0, report_unsent=True) is None
        )
    finally:
        thread.join(5.0)
        listener.close()
    assert received and b"net_pair_confirm" in received[0], "the op frame never left"
