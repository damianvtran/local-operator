"""``join --automated`` — the no-human-at-the-joining-end ceremony (design §2.6).

WHAT THESE CELLS PIN, and why they are the whole security story of the flag:

* the automated join SENDS THIS DEVICE'S OWN DERIVED CODE as the transcription
  (a prompt is never read — asserted by making ``_read_code`` fatal), and the
  INVITER still compares it with its own derivation: admission happens only
  after the inviting side's confirm, so there is no bless anywhere;
* WITHOUT a confirm nothing is admitted — the pre-answered decision is the
  operator's gesture, not a bypass, and the compare runs before any decision is
  read;
* the flag exists on the real parser, and the combinations that cannot mean one
  thing (``--automated --park``, ``--automated --verify`` …) are usage refusals.

The two relays are real sockets and the real ceremony; only the two humans are
supplied — the same discipline ``test_join_park`` and ``test_relay_e2e`` state.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from typing import Any

import pytest

from local_operator.network import cli as net_cli
from local_operator.network import invite as invite_mod
from local_operator.network import relay, store, types, wire
from local_operator.network.handshake import (
    Credential,
    Handshake,
    pair_abort_frame,
    pair_timeout_seconds,
    sas_matches,
)
from tests.unit.network.test_join_park import _minted
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _await_event,
    _events,
    _init_network,
    devices,
)


def _automated_args(**overrides: Any) -> Namespace:
    """The Namespace argparse builds for ``join @<token> --automated --json``."""
    base: dict[str, Any] = {
        "confirm": "",
        "park": False,
        "automated": True,
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


def _automated_join(
    server_b: relay.RelayServer,
    *,
    host: str,
    port: int,
    minted: Any,
    args: Namespace | None = None,
) -> Any:
    """Phase-one-and-only, driven exactly as ``lop network join --automated``."""
    return net_cli._join_one(  # noqa: SLF001 — the CLI's own driver, run as the CLI runs it
        host=f"{host}:{port}",
        token=minted.token,
        envelope=minted.envelope,
        identity=server_b.identity,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        args=args or _automated_args(name=server_b.identity.name),
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


def _pre_answer(server_a: relay.RelayServer, invite_id: str) -> None:
    """The operator's approval, pre-answering the relay's parked confirm (§3.3)."""
    store.save_pair_decision(
        types.PairDecision(
            invite_id=invite_id,
            decision="admit",
            matched=True,
            reason="",
            answered_by="approval:ap_test0001",
            shares=[],
        ),
        server_a.root,
    )


def test_an_automated_join_admits_only_with_the_pre_answered_confirm(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The happy half: no prompt is read, the confirm admits, the codes compare.

    ``_read_code`` is made FATAL rather than stubbed: if the automated path ever
    grew a prompt, this cell fails instead of quietly asking a human that CI does
    not have. Admission itself is the confirmation half — the decision file is
    what the relay reads, and the wire compare ran first (a mismatch would have
    refused before the decision was ever polled).
    """
    server_a, server_b, host, port = devices
    record, minted = _minted(server_a)
    _pre_answer(server_a, minted.record.invite_id)

    def _never_prompt(args: Any, derived: str, fingerprint: str) -> str:
        raise AssertionError("--automated must never read a code from a prompt")

    monkeypatch.setattr(net_cli, "_read_code", _never_prompt)

    lines, payload = _automated_join(server_b, host=host, port=port, minted=minted)

    assert payload["network_id"] == record.network_id
    assert payload["device_id"] == server_b.identity.device_id
    # THE VALUE THE COMPARE SAW, on the join payload: this device's own derivation.
    assert isinstance(payload.get("sas"), str)
    assert len(payload["sas"]) == 6 and payload["sas"].isdigit()
    assert any("joined" in line for line in lines)
    # Admitted, and admitted by the CONFIRM: the relay recorded the confirmation
    # from the pre-answered decision and never a refusal.
    refreshed = store.load(record.network_id, server_a.root)
    assert refreshed.member(server_b.identity.device_id) is not None
    events = _await_event(server_a, "pairing_confirmed")
    assert "pairing_confirmed" in events
    assert "pairing_refused" not in events


def test_an_automated_join_nobody_confirms_admits_nothing(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
) -> None:
    """The refusal half — the shape that proves no bless was added.

    No decision file, so nobody has confirmed: the ceremony runs out its window
    and the inviter refuses. THE ASSERTION IS THE OUTCOME, NOT THE EXCEPTION
    TYPE, because the failure arrives in one of two legitimate shapes: a refusal
    frame raises ``MeshRefusal``, while a window that runs out mid-read comes
    back as the per-host sentence ``_cmd_join`` loops over. Both mean the same
    thing, and an admission in either case would fail here. A short invite TTL is
    what makes the wait seconds rather than minutes (``min(remaining, 180)``),
    exactly as ``test_join_park``'s unanswered cell bounds its own wait.
    """
    server_a, server_b, host, port = devices
    record, minted = _minted(server_a, ttl_s=2.0)

    outcome: Any = None
    try:
        outcome = _automated_join(server_b, host=host, port=port, minted=minted)
    except (types.MeshRefusal, OSError):
        outcome = None
    # A FAILURE ARRIVES AS None (raised), A STRING (the per-host sentence
    # ``_join_one`` returns), or a sentence the caller loops over — never as an
    # admission tuple, which is the only outcome the assertion forbids.
    assert not isinstance(outcome, tuple), f"an unconfirmed automated join admitted: {outcome!r}"

    assert store.load(record.network_id, server_a.root).member(server_b.identity.device_id) is None
    assert store.list_networks(server_b.root) == []
    # The negative that matters: nothing was ever confirmed for this pairing.
    assert "pairing_confirmed" not in _events(server_a)


def test_a_doctored_transcription_is_refused_through_the_automated_path(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """M4's no-bless property, pinned FOR ``--automated`` (agent review round 1,
    Finding 4).

    The automated path's transcription is this device's own derivation, so the
    only way to doctor it is to swap the value the join SENDS — done here by
    wrapping ``_finish_pairing``, i.e. the framed ``net_pair_ready`` value, with
    a wrong-but-well-formed code. The relay's compare runs on what crossed the
    wire, before any decision is read, so the ceremony must refuse and nothing
    may be admitted; the invite records the spent attempt. If anyone ever adds
    an automated shortcut that treats "automated" as locally trusted, or moves
    the compare behind the decision gate, this cell fails.
    """
    server_a, server_b, host, port = devices
    record, minted = _minted(server_a)

    real = net_cli._finish_pairing  # noqa: SLF001 — the frame-building seam itself

    def _doctored(*args: Any, **kwargs: Any) -> Any:
        kwargs["typed"] = "000001" if kwargs.get("typed") != "000001" else "000002"
        return real(*args, **kwargs)

    monkeypatch.setattr(net_cli, "_finish_pairing", _doctored)

    with pytest.raises(types.MeshRefusal) as refused:
        _automated_join(server_b, host=host, port=port, minted=minted)

    assert refused.value.code == "sas_mismatch"
    refreshed = store.load(record.network_id, server_a.root)
    assert refreshed.member(server_b.identity.device_id) is None
    assert store.list_networks(server_b.root) == []
    # The refusal is audited, the admission is not, and the attempt is spent.
    _await_event(server_a, "pairing_refused")
    assert "pairing_confirmed" not in _events(server_a)
    assert refreshed.invites[0].attempts >= 1


def test_the_cli_parses_the_automated_flag() -> None:
    """The flag exists on the REAL parser, so the guide's command line is a command."""
    parser = argparse.ArgumentParser(prog="lop")
    subparsers = parser.add_subparsers(dest="subcommand")
    net_cli.add_parser(subparsers)
    automated = parser.parse_args(["network", "join", "@invite.token", "--automated", "--json"])
    assert automated.automated is True
    assert automated.park is False and automated.verify is False
    only_automated = parser.parse_args(["network", "join", "--automated"])
    assert only_automated.automated is True and only_automated.json is False


def test_automated_refuses_flag_pairs_it_cannot_honour(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each conflict is a usage error (rc 2), before anything dials."""

    def _args(**overrides: Any) -> Namespace:
        base: dict[str, Any] = {
            "confirm": "",
            "park": False,
            "automated": True,
            "sas_stdin": False,
            "verify": False,
            "emit_sas": False,
            "json": False,
        }
        base.update(overrides)
        return Namespace(**base)

    for conflict, _words in (
        ("park", "--park"),
        ("emit_sas", "--emit-sas"),
        ("sas_stdin", "--sas-stdin"),
        ("verify", "--verify"),
    ):
        assert net_cli._cmd_join(_args(**{conflict: True})) == 2  # noqa: SLF001
        assert "--automated" in capsys.readouterr().err
