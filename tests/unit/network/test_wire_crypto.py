"""The wire: the key schedule, the AEAD codec, framing and version gating."""

from __future__ import annotations

import json

import pytest

from local_operator.network import wire
from local_operator.network.types import HandshakeRefusal


def _shared() -> bytes:
    """A real X25519 shared secret, so the schedule is exercised as it is used."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

    a = X25519PrivateKey.generate()
    b = X25519PrivateKey.generate()
    a_public = a.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    b_public = b.public_key().public_bytes(
        encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
    )
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey

    assert a.exchange(X25519PublicKey.from_public_bytes(b_public)) == b.exchange(
        X25519PublicKey.from_public_bytes(a_public)
    )
    return a.exchange(X25519PublicKey.from_public_bytes(b_public))


# ---------------------------------------------------------------------------
# The key schedule
# ---------------------------------------------------------------------------


def test_epoch_key_is_domain_separated_by_network_and_epoch() -> None:
    material = wire.b64u(b"0" * 32)
    key = wire.epoch_key(material, "n_one", 7)
    assert len(key) == 32
    assert key == wire.epoch_key(material, "n_one", 7)
    assert key != wire.epoch_key(material, "n_two", 7)
    assert key != wire.epoch_key(material, "n_one", 8)


def test_invite_key_is_per_token() -> None:
    material = wire.b64u(b"1" * 32)
    first = wire.invite_key(material, "n_one", "invite-a")
    second = wire.invite_key(material, "n_one", "invite-b")
    assert first != second
    # A token cannot be re-labelled with another invite's id: the key moves with it.
    assert first != wire.invite_key(material, "n_one", "invite-a2")


def test_invite_mac_binds_the_payload_bytes() -> None:
    key = b"k" * 32
    assert wire.invite_mac(key, b"payload") != wire.invite_mac(key, b"payload ")
    assert wire.invite_mac(key, b"payload") == wire.invite_mac(key, b"payload")


def test_link_keys_split_into_two_directions() -> None:
    keys = wire.link_keys(_shared(), b"t" * 32, b"l" * 16)
    assert keys.k_d2l != keys.k_l2d
    assert len(keys.iv_d) == 4 and len(keys.iv_l) == 4
    assert keys.iv_d != keys.iv_l
    d_key, _d_iv, d_direction = keys.send_params("dialer")
    l_key, _l_iv, l_direction = keys.receive_params("dialer")
    assert (d_key, d_direction) == (keys.k_d2l, wire.ROLE_DIALER)
    assert (l_key, l_direction) == (keys.k_l2d, wire.ROLE_LISTENER)


def test_sas_is_six_digits_and_moves_with_the_transcript() -> None:
    shared = _shared()
    first = wire.sas_code(shared, b"a" * 32)
    second = wire.sas_code(shared, b"b" * 32)
    assert len(first) == 6 and first.isdigit()
    assert first != second
    assert wire.sas_display(first) == f"{first[:3]} {first[3:]}"
    assert wire.normalize_sas("481 926") == "481926"
    assert wire.normalize_sas("481-926") == "481926"
    assert wire.normalize_sas("48192") == ""
    assert wire.normalize_sas("not a code") == ""


def test_fingerprint_is_160_bits_in_eight_groups() -> None:
    text = wire.transcript_fingerprint(b"\x01" * 32)
    assert text.count("-") == 7
    assert len(text.replace("-", "")) == 32
    assert set(text) <= set("0123456789ABCDEFGHJKMNPQRSTVWXYZ-")


# ---------------------------------------------------------------------------
# The record codec
# ---------------------------------------------------------------------------


def test_records_round_trip_and_the_counter_advances() -> None:
    keys = wire.link_keys(_shared(), b"t" * 32, b"l" * 16)
    dialer = wire.LinkCrypto(keys, role="dialer")
    listener = wire.LinkCrypto(keys, role="listener")
    for index in range(200):
        payload = dialer.seal({"op": "ping", "req": index})
        assert listener.open(payload[4:]) == {"op": "ping", "req": index}
    assert dialer.sent == 200
    assert listener.received == 200


def test_ten_thousand_records_use_distinct_counters() -> None:
    """Nonces are DERIVED from the counter, so a reused counter would be a reused
    nonce — the one failure mode a hand-rolled AEAD must not have."""
    keys = wire.link_keys(_shared(), b"t" * 32, b"l" * 16)
    dialer = wire.LinkCrypto(keys, role="dialer")
    listener = wire.LinkCrypto(keys, role="listener")
    seen: set[int] = set()
    for _ in range(10_000):
        payload = dialer.seal({"op": "ping", "req": 1})
        listener.open(payload[4:])
        seen.add(dialer.sent)
    assert len(seen) == 10_000


def test_a_replayed_record_is_refused() -> None:
    """The negative case: replaying the same record fails, because the receiver's
    counter has moved on and the AAD binds the sequence."""
    keys = wire.link_keys(_shared(), b"t" * 32, b"l" * 16)
    dialer = wire.LinkCrypto(keys, role="dialer")
    listener = wire.LinkCrypto(keys, role="listener")
    payload = dialer.seal({"op": "prompt", "req": 1})[4:]
    assert listener.open(payload)["op"] == "prompt"
    with pytest.raises(wire.LinkCryptoError):
        listener.open(payload)


def test_a_tampered_record_is_refused_and_nothing_is_repaired() -> None:
    keys = wire.link_keys(_shared(), b"t" * 32, b"l" * 16)
    listener = wire.LinkCrypto(keys, role="listener")
    for index in range(4):
        # A fresh sealer per attempt: the receiver's counter is the thing under test,
        # so the sender must be positioned at the same index each round.
        drafter = wire.LinkCrypto(keys, role="dialer")
        for _ in range(index):
            drafter.seal({"op": "ping", "req": 0})
        payload = bytearray(drafter.seal({"op": "prompt", "req": 9})[4:])
        payload[len(payload) // 2] ^= 0x01
        with pytest.raises(wire.LinkCryptoError):
            listener.open(bytes(payload))


def test_a_link_id_mismatch_in_the_aad_is_refused() -> None:
    """``link_id`` never crosses the wire, so two links between the same pair in the
    same second cannot have a record transplanted between them."""
    shared = _shared()
    digest = b"t" * 32
    first_keys = wire.link_keys(shared, digest, b"a" * 16)
    second_keys = wire.link_keys(shared, digest, b"b" * 16)
    dialer = wire.LinkCrypto(first_keys, role="dialer")
    listener = wire.LinkCrypto(second_keys, role="listener")
    with pytest.raises(wire.LinkCryptoError):
        listener.open(dialer.seal({"op": "ping", "req": 1})[4:])


def test_reflection_is_impossible_through_the_direction_split() -> None:
    """A peer that echoes another's records back at it hands them a record
    encrypted under the key for the OPPOSITE direction, which fails the tag."""
    keys = wire.link_keys(_shared(), b"t" * 32, b"l" * 16)
    same_side = wire.LinkCrypto(keys, role="dialer")
    with pytest.raises(wire.LinkCryptoError):
        same_side.open(same_side.seal({"op": "ping", "req": 1})[4:])


def test_oversized_record_is_refused_before_allocating(socketpair: object) -> None:
    """The length prefix is checked FIRST: a peer cannot ask for 8 MiB of buffer by
    writing four bytes."""
    client, server = socketpair  # type: ignore[misc]
    # Written by the ACCEPTED end and read by the connecting end: sending on the same
    # socket the reader is reading from would time out rather than exercise the bound.
    server.sendall((wire.MAX_RECORD_BYTES + 1).to_bytes(4, "big"))
    with pytest.raises(wire.LinkCryptoError) as excinfo:
        wire.FrameReader(client).read_record_payload(wire.deadline_in(2.0))
    assert "limit" in str(excinfo.value)


def test_frame_reader_pipelines_without_losing_bytes(socketpair: object) -> None:
    """Byte-oriented sockets have no message boundaries, so a reader that assumed
    one-read-one-frame would lose the tail of a pipelined write."""
    client, server = socketpair  # type: ignore[misc]
    server.sendall(
        wire.encode_line({"op": "hello", "n": 1}) + wire.encode_line({"op": "hello", "n": 2})
    )
    reader = wire.FrameReader(client)
    assert reader.read_line(wire.deadline_in(2.0))["n"] == 1
    assert reader.read_line(wire.deadline_in(2.0))["n"] == 2


def test_oversized_handshake_line_is_refused(socketpair: object) -> None:
    client, server = socketpair  # type: ignore[misc]
    server.sendall(b"{" + b"x" * (wire.MAX_HANDSHAKE_LINE + 10))
    with pytest.raises(wire.LinkCryptoError):
        wire.FrameReader(client).read_line(wire.deadline_in(2.0))


def test_a_closed_socket_reads_as_a_connection_error(socketpair: object) -> None:
    client, server = socketpair  # type: ignore[misc]
    server.close()
    with pytest.raises((ConnectionError, OSError)):
        wire.FrameReader(client).read_line(wire.deadline_in(2.0))


# ---------------------------------------------------------------------------
# Version and capability negotiation
# ---------------------------------------------------------------------------


def test_unknown_link_version_is_refused_naming_both_numbers() -> None:
    wire.check_link_version(1, mine=1)
    with pytest.raises(HandshakeRefusal) as excinfo:
        wire.check_link_version(2, mine=1)
    assert "1" in excinfo.value.sentence and "2" in excinfo.value.sentence


def test_negotiated_capabilities_are_the_intersection() -> None:
    mine = list(wire.LINK_CAPABILITIES)
    assert wire.negotiate_capabilities(mine, ["mesh-net-v1", "unknown-thing"]) == ["mesh-net-v1"]
    # An old peer advertising nothing gets nothing — never a default.
    assert wire.negotiate_capabilities(mine, []) == []


def test_keepalive_is_a_reused_control_op() -> None:
    frame = wire.keepalive_frame(7)
    assert frame["op"] == "ping"
    assert wire.is_keepalive(frame)
    assert not wire.is_bye(frame)
    assert wire.is_bye({"op": "net_bye"})


def test_canonical_json_is_stable_across_key_order() -> None:
    assert wire.canonical_json({"b": 1, "a": 2}) == wire.canonical_json({"a": 2, "b": 1})
    assert json.loads(wire.canonical_json({"a": "é"}))["a"] == "é"
