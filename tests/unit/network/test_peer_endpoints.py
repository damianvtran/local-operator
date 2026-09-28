"""Where a device says it can be reached, and what an admission may write down.

WHY THIS FILE EXISTS (session-mobility audit, 2026-09-26). Two endpoints bugs sat
between a paired device and every verb that has to dial it — a session move, a
remote listing, a doctor probe — and both were measured on a real two-device
loopback rig before they were fixed:

* **The running relay's listen block was never consulted.** A device whose relay
  runs `lop network serve --port 41902` over a config that never learned that port
  — the SHIPPED config carries no ``network`` section at all — advertised
  ``advertise_endpoints() == []``. Nothing else in the ceremony republishes the
  truth later, so the peer's row kept whatever the admission wrote.
* **The admission wrote the joiner's ephemeral client port.** With the joiner
  declaring nothing, ``endpoints`` fell back to the observed source address of the
  pairing connection. That address is a client-side socket that is closed by the
  time anybody dials it: ``lop network peers`` reported
  ``connect_failed:ConnectionRefusedError`` against a port that never listened for
  the mesh, and ``lop sessions move <id> --to <that device>`` answered
  ``unreachable`` — the peer was permanently undialable, which is exactly the
  F-2 failure the fallback was believed to have fixed.

Both cells below fail on the pre-fix code for the reason their docstring names:
the first returns ``[]`` where it must name the live relay, the second records the
observed address where it must record nothing.
"""

from __future__ import annotations

import os
import socket
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import addresses, relay, store, types
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures and ceremony
    _pair,
    devices,
)

Devices = tuple[relay.RelayServer, relay.RelayServer, str, int]
__all__ = ["Devices"]


def _publish_listen(root: Path, listen: dict[str, Any], *, pid: int | None = None) -> None:
    """A live relay record on this process's pid, exactly as `serve` publishes it.

    The pid is the READER's liveness test (``store.session.runtime.registry``
    classifies a record by whether its pid is alive), so a record planted with a
    dead pid would be invisible to ``find_own_relay`` and the test would pass
    without exercising the code it is about.
    """
    store.publish_peer_record(
        types.PeerRecord(
            pid=pid if pid is not None else os.getpid(),
            listen=listen,
        ),
        root,
    )


def test_a_live_relay_supplies_the_endpoints_a_silent_config_cannot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config that never learned the port must not hide the listener.

    Measured on the rig: ``NetworkSettings.from_config()`` answered
    ``0.0.0.0:4097`` (the shipped defaults — there is no ``network`` section in a
    generated config), the relay really listened on ``127.0.0.1:41902``, and the
    join advertised NOTHING. The relay's own published record is the answer,
    read from the root the settings were READ FROM.

    ``getaddrinfo`` is stubbed to the empty answer because that is what the rig's
    isolated HOME produced for a ``0.0.0.0`` listener; leaving it real would make
    this cell assert about the CI host's LAN address.
    """
    root = tmp_path / "device"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [])
    _publish_listen(
        root,
        {"address": "127.0.0.1", "port": 41902, "advertised": ["127.0.0.1:41902"]},
    )

    settings = relay.NetworkSettings.from_config()
    assert settings.root == root, "from_config must record the store it read"
    assert relay.advertise_endpoints(settings) == ["127.0.0.1:41902"]


def test_settings_without_a_store_make_no_claim_about_a_relay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hand-built settings object must not borrow the AMBIENT install's relay.

    This is the cross-wiring this guard exists for, and it was measured as 29 red
    cells in ``tests/unit/network``: two devices in one process, the ambient config
    dir pointing at device A while device B's join asks to advertise itself, and a
    live-relay lookup against the ambient root answering with A's endpoint. B then
    told A that B was reachable at A's own address, so A dialled ITSELF for B —
    ``handshake_refused``, then ``unreachable`` for every move.

    ``root=None`` is the honest answer for settings nobody read from a store, so
    the lookup must be skipped rather than resolved ambiently.
    """
    root = tmp_path / "ambient"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _publish_listen(
        root,
        {"address": "127.0.0.1", "port": 41902, "advertised": ["127.0.0.1:41902"]},
    )

    settings = relay.NetworkSettings(port=4097, listen_address="127.0.0.1")
    assert settings.root is None, "a hand-built settings object claims no store"
    assert relay.advertise_endpoints(settings) == ["127.0.0.1:4097"]


def test_the_configured_addresses_still_answer_when_no_relay_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The escape hatch must stay open: nobody running is not nobody declared.

    ``network.advertise_hosts`` is what a device behind a tunnel declares, and it is
    an answer the live relay cannot give. Pinned beside the cell above so "ask the
    relay first" cannot become "ask the relay only".
    """
    root = tmp_path / "device"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))

    settings = relay.NetworkSettings.from_config()
    settings = replace(settings, listen_address="127.0.0.1", advertise_hosts=("mesh.example:4200",))
    assert relay.advertise_endpoints(settings) == ["mesh.example:4200", "127.0.0.1:4097"]


def test_a_detected_address_is_published_with_the_live_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The port in a detected entry is the LISTENER's, not the config's.

    The config's port beside a detected LAN address is a claim about a socket that
    is not there — the same "nothing can dial it" outcome one layer down.

    The detection seam is ``network/addresses.py``: what used to be a
    ``getaddrinfo(gethostname())`` lookup is now the interface table, so patching the
    resolver faked nothing and the runner's REAL addresses leaked into the answer.
    """
    root = tmp_path / "device"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    _publish_listen(
        root,
        {"address": "0.0.0.0", "port": 41902, "advertised": []},
    )
    monkeypatch.setattr(addresses, "local_ipv4_addresses", lambda: ["10.1.2.3"])

    settings = relay.NetworkSettings.from_config()
    assert relay.advertise_endpoints(settings) == ["10.1.2.3:41902"]


def test_an_admission_records_no_endpoint_when_the_joiner_declared_none(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A joiner that declared nothing must leave the row EMPTY.

    The observed source address is a client-side ephemeral port, closed by the
    time anything dials it, and writing it made the member permanently
    undialable while every surface reported a concrete endpoint. The honest
    answer is the documented ``no_endpoint`` state, which is what an empty list
    means (``relay.peer_reason_words``).

    ``advertise_endpoints`` is stubbed to ``[]`` because that is a REAL state a
    joiner can be in — it is the state the pre-fix code produced on its own for a
    device whose config is silent and whose host resolves to loopback only — and
    the admission is what this cell is about. The relay is live and dialable
    throughout: what is asserted is that the inviter does not INVENT an address.
    """
    both: Devices = request.getfixturevalue("devices")
    server_a, server_b, _host, _port = both

    monkeypatch.setattr(relay, "advertise_endpoints", lambda *args, **kwargs: [])

    record, _host, _port = _pair(both, monkeypatch)

    loaded = store.load(record.network_id, server_a.root)
    joiner_id = server_b.identity.device_id
    row = next(member for member in loaded.members if member.device_id == joiner_id)
    assert row.endpoints == [], (
        f"the inviter recorded {row.endpoints!r} for a joiner that declared nothing — "
        "an observed client port is not an endpoint any peer can dial"
    )
