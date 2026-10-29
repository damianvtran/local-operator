"""The two production seats that run the invite prune (review round 1 on #2752).

``store.prune_invites`` / ``store.sweep_invites`` are tested directly in
``test_store.py``; these cells pin the CALLERS, because a prune nobody calls is
the defect the PR fixed (``store.prune`` had no production caller, so dead
``minted`` rows piled up forever):

* both MINT paths — the relay's ``net_invite`` op (``_ctl_invite``) and the CLI's
  no-relay fallback (``_invite_locally``) — drop dead rows and their token files
  in the same write that appends the new one (R2);
* the RELAY START seat runs one sweep, and a sweep that raises is reported and
  swallowed rather than failing the relay (R3).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pytest

from local_operator.network import cli as net_cli
from local_operator.network import identity, relay, store, types, wire
from tests.unit.network.conftest import wait_for

HOUR = 60 * 60.0


def _seed(root: Path) -> types.NetworkRecord:
    """A real network holding one DEAD invite (400 h old, 10 min TTL) and one live one."""
    own = identity.mint(root, name="seat-device")
    record = types.NetworkRecord(
        network_id=store.new_network_id(),
        name="seat-net",
        created_by=own.device_id,
        self_device_id=own.device_id,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
        listen={"address": "127.0.0.1", "port": 0, "advertised": []},
    )
    record.members.append(
        types.MemberRecord(
            device_id=own.device_id,
            public_key=own.public_key,
            name=own.name,
            role="admin",
            capabilities=sorted(types.capabilities_for_role("admin")),
            added_via="self",
        )
    )
    now = time.time()
    record.invites = [
        types.InviteRecord(invite_id="deadinvite", minted_at=now - 400 * HOUR, ttl_s=600.0),
        types.InviteRecord(invite_id="liveinvite", minted_at=now - 60.0, ttl_s=600.0),
    ]
    store.save(record, root)
    from secrets import token_bytes

    store.save_secrets(
        types.SecretState(network_id=record.network_id, epoch=1, secret=wire.b64u(token_bytes(32))),
        root,
    )
    store.save_invite_token("deadinvite", "tok-dead", root)
    store.save_invite_token("liveinvite", "tok-live", root)
    return record


def _assert_pruned_and_appended(root: Path, record: types.NetworkRecord, new_id: str) -> None:
    ids = [invite.invite_id for invite in store.load(record.network_id, root).invites]
    assert ids == ["liveinvite", new_id], ids
    assert not store.invite_path("deadinvite", root).exists(), "the dead token file survived"
    assert store.invite_path("liveinvite", root).exists(), "a LIVE token file was deleted"
    assert store.invite_path(new_id, root).exists()


def _relay(root: Path, monkeypatch: pytest.MonkeyPatch) -> relay.RelayServer:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    return relay.RelayServer(
        root=root, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )


def test_the_relay_mint_drops_dead_rows_and_their_tokens(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = _seed(root)
    server = _relay(root, monkeypatch)
    minted = server._ctl_invite({"network": record.network_id})  # noqa: SLF001
    _assert_pruned_and_appended(root, record, minted["invite_id"])


def test_the_cli_local_mint_drops_dead_rows_and_their_tokens(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_invite_locally`` is the no-relay path; it resolves the AMBIENT config dir."""
    record = _seed(root)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    args = argparse.Namespace(network=record.network_id, role="drive", expires=600.0, device="")
    payload, _path = net_cli._invite_locally(args, ["127.0.0.1:4097"])  # noqa: SLF001
    _assert_pruned_and_appended(root, record, payload["invite_id"])


def test_a_relay_start_sweeps_the_backlog_without_a_mint(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The motivating case: a device that stops minting still clears dead rows."""
    record = _seed(root)
    server = _relay(root, monkeypatch)
    server.start()
    try:
        assert wait_for(
            lambda: [i.invite_id for i in store.load(record.network_id, root).invites]
            == ["liveinvite"]
        ), "the start sweep did not drop the dead row"
        assert wait_for(lambda: not store.invite_path("deadinvite", root).exists())
        assert store.invite_path("liveinvite", root).exists()
    finally:
        server.stop()


def test_a_failing_start_sweep_is_reported_and_swallowed(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed(root)
    called: list[Path | None] = []

    def boom(root_arg: Path | None = None, **_kwargs: object) -> int:
        called.append(root_arg)
        raise OSError("disk on fire")

    monkeypatch.setattr(store, "sweep_invites", boom)
    server = _relay(root, monkeypatch)
    server.start()
    try:
        assert wait_for(lambda: bool(called)), "start() never reached the invite sweep"
        # Accumulated: each ``readouterr`` drains the buffer, so a poll that read and
        # discarded the line on an earlier tick would never see it again.
        seen: list[str] = []
        assert wait_for(
            lambda: seen.append(capsys.readouterr().err) is None
            and "invite sweep failed (disk on fire)" in "".join(seen)
        )
        # The relay is still serving: its control op answers after the failed sweep.
        assert server._ctl_invite({"network": _only_network(root)})["invite_id"]  # noqa: SLF001
    finally:
        server.stop()


def _only_network(root: Path) -> str:
    (record,) = store.list_networks(root)
    return record.network_id


# ---------------------------------------------------------------------------
# The wire split: `state` stays the stored ledger, `listed_state` is the display
# word (review round 1 R5, pinned in round 2 R6).
# ---------------------------------------------------------------------------


def test_the_show_json_carries_stored_state_and_listed_state_apart(
    root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No relay answers, so this is the CLI's own `show` payload, the offline twin of `net_show`."""
    import json

    record = _seed(root)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setattr(net_cli, "_relay_call", lambda *a, **k: None)
    args = argparse.Namespace(network=record.network_id, json=True)
    assert net_cli._cmd_show(args) == 0  # noqa: SLF001
    invites = {i["invite_id"]: i for i in json.loads(capsys.readouterr().out)["invites"]}
    # The dead row is STILL `minted` in the ledger; only the display word says expired.
    assert invites["deadinvite"]["state"] == "minted"
    assert invites["deadinvite"]["listed_state"] == "expired"
    assert invites["liveinvite"]["state"] == "minted"
    assert invites["liveinvite"]["listed_state"] == "minted"


def test_the_text_listings_print_listed_state_and_fall_back_to_state() -> None:
    """An older relay sends no `listed_state`; both renderers must still print a word."""
    from local_operator.network import tool as net_tool

    base = {"name": "n", "network_id": "n_x", "epoch": 1, "trust": "active", "role": "admin"}
    new = {
        **base,
        "invites": [
            {"invite_id": "a", "role": "drive", "state": "minted", "listed_state": "expired"}
        ],
    }
    old = {**base, "invites": [{"invite_id": "a", "role": "drive", "state": "minted"}]}
    assert "invite a  drive  expired" in "\n".join(net_cli._show_lines(new))  # noqa: SLF001
    assert "invite a  drive  minted" in "\n".join(net_cli._show_lines(old))  # noqa: SLF001
    assert "invite a  drive  expired" in "\n".join(net_tool._render("show", new))  # noqa: SLF001
    assert "invite a  drive  minted" in "\n".join(net_tool._render("show", old))  # noqa: SLF001
