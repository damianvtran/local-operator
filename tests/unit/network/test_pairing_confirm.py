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
import time
from argparse import Namespace
from pathlib import Path

import pytest

from local_operator.network import cli as net_cli
from local_operator.network import relay, store, types

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
    )
    store.save_pending_pairing(pending, root)
    return pending


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
    assert "lop network invite" in excinfo.value.sentence
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

        # And an "admit that did not match" is impossible by construction.
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

    assert net_cli._cmd_confirm(_args(decline=True, invite_id="i_def456")) == 0  # noqa: SLF001
    assert store.pair_decision("i_def456", root) is not None
    assert store.pair_decision("i_abc123", root) is None, "the wrong pairing was answered"
    assert store.pending_pairing("i_abc123", root) is not None
