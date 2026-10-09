"""The wire: the key schedule, the AEAD codec, framing and version gating."""

from __future__ import annotations

import json
import zlib
from typing import Any

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


def test_the_codec_kinds_for_limit_and_sequence_are_the_payloads_own_tokens() -> None:
    """F4 (QA round 1 note): ``auth`` and ``parse`` are pinned by the failure
    cells; these are the two kinds a cell can force with no wire at all.

    A payload past the record-size limit and a sequence number at the
    ``MAX_SEQ`` ceiling — past it the derived nonce stops being safe, so the
    codec refuses to keep counting — must carry their own kinds, because the
    joiner's ``join`` block discriminates exactly these tokens.
    """
    keys = wire.link_keys(_shared(), b"t" * 32, b"l" * 16)
    dialer = wire.LinkCrypto(keys, role="dialer")
    with pytest.raises(wire.LinkCryptoError) as excinfo:
        dialer.seal({"op": "x", "blob": "a" * (wire.MAX_RECORD_BYTES + 1)})
    assert excinfo.value.kind == "limit"

    exhausted = wire.LinkCrypto(keys, role="listener")
    # The RECEIVING side's counter IS the thing under test (its ceiling), and
    # reaching it honestly would mean 2**40 records; the poke is the mechanism.
    # The check runs before decryption, so a junk payload proves the kind.
    exhausted._recv_seq = wire.LinkCrypto.MAX_SEQ  # noqa: SLF001
    with pytest.raises(wire.LinkCryptoError) as excinfo:
        exhausted.open(b"whatever")
    assert excinfo.value.kind == "sequence"


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


def test_the_handshake_reader_hands_its_buffer_to_the_record_reader(socketpair: object) -> None:
    """THE HANDOVER BETWEEN A SOCKET'S TWO PHASES, which is where a frame is lost.

    A socket carries JSON-line handshake frames and then sealed records. A reader
    reads in 64 KiB chunks, so the reader that finishes the handshake may already
    be holding the first record — and the code that abandons it (a fresh reader is
    created for the record phase) drops those bytes. The record is then decrypted
    out of sequence, which surfaces as ``LinkCryptoError`` on a link whose
    handshake was perfect, seconds after it was established.

    That is not hypothetical: it is what a peer that speaks IMMEDIATELY after its
    handshake hits — the membership pull this build makes at every link
    establishment — and it is why the reader carries its buffer across instead of a
    new one starting empty.
    """
    client, server = socketpair  # type: ignore[misc]
    record = (4).to_bytes(wire.LOCAL_PREFIX_LEN, "big") + b"wxyz"
    # One write: the last handshake frame and the first record in the same segment,
    # which is exactly what the receiving socket sees when a peer pipelines.
    server.sendall(wire.encode_line({"op": "welcome"}) + record)
    handshake_reader = wire.FrameReader(client)
    assert handshake_reader.read_line(wire.deadline_in(2.0))["op"] == "welcome"
    carried = wire.FrameReader(client, buffered=handshake_reader.pending())
    assert carried.read_record_payload(wire.deadline_in(2.0)) == b"wxyz"


def test_a_reader_that_started_empty_does_not_invent_bytes(socketpair: object) -> None:
    """The other half of the handover contract: an empty handover is an empty
    buffer, not a licence to read whatever is next on the socket."""
    client, server = socketpair  # type: ignore[misc]
    server.sendall(wire.encode_line({"op": "welcome"}))
    handshake_reader = wire.FrameReader(client)
    assert handshake_reader.read_line(wire.deadline_in(2.0))["op"] == "welcome"
    assert handshake_reader.pending() == b""
    carried = wire.FrameReader(client, buffered=handshake_reader.pending())
    with pytest.raises((TimeoutError, OSError)):
        carried.read_record_payload(wire.deadline_in(0.2))


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


# ---------------------------------------------------------------------------
# Record compression (``zlib-records-v1``)
# ---------------------------------------------------------------------------
#
# The matrix below pins the MIXED-VERSION RULE (wire.py "Record compression"):
# compression is on only when BOTH ends advertised the capability, and a link
# where it is off is byte-identical to the codec before the capability existed.


def _page(rows: int = 100) -> dict[str, Any]:
    """A ``net_session_history``-shaped reply: repetitive journal JSON, ~100 KB+."""
    entries = [
        {
            "id": index,
            "ts": 1_760_000_000.0 + index,
            "type": "assistant_message",
            "payload": {"text": f"row {index} " + "lorem ipsum dolor sit amet " * 40},
        }
        for index in range(rows)
    ]
    return {"op": "ack", "req": 7, "detail": {"entries": entries, "has_more": True}}


def _pair(*, dialer_zlib: bool, listener_zlib: bool) -> tuple[wire.LinkCrypto, wire.LinkCrypto]:
    keys = wire.link_keys(_shared(), b"t" * 32, b"l" * 16)
    return (
        wire.LinkCrypto(keys, role="dialer", compression=dialer_zlib),
        wire.LinkCrypto(keys, role="listener", compression=listener_zlib),
    )


def _plaintext_of(frame: dict[str, Any]) -> bytes:
    return json.dumps(frame, sort_keys=False, separators=(",", ":"), ensure_ascii=False).encode()


def _decrypt_raw(codec: wire.LinkCrypto, record: bytes, sequence: int = 0) -> bytes:
    """The AEAD plaintext of a sealed record, to inspect what is ON the wire."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    return AESGCM(codec._recv_key).decrypt(  # noqa: SLF001
        codec._nonce(codec._recv_iv, sequence),  # noqa: SLF001
        record[4:],
        codec._aad(codec._recv_direction, sequence),  # noqa: SLF001
    )


def _seal_raw(sender: wire.LinkCrypto, plaintext: bytes) -> bytes:
    """A record with an arbitrary plaintext, authenticated like a real one.

    The malformed-input cells need a PEER that authenticates garbage (a bad tag is
    a different, already-tested failure), so this encrypts under the sender's key
    with the next counter exactly as ``seal`` does.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    sequence = sender._send_seq  # noqa: SLF001
    payload = AESGCM(sender._key).encrypt(  # noqa: SLF001
        sender._nonce(sender._iv, sequence),  # noqa: SLF001
        plaintext,
        sender._aad(sender._direction, sequence),  # noqa: SLF001
    )
    sender._send_seq += 1  # noqa: SLF001
    return len(payload).to_bytes(4, "big") + payload


def test_the_capability_is_advertised_and_is_the_codecs_only_gate() -> None:
    assert wire.ZLIB_RECORDS_V1 in wire.LINK_CAPABILITIES
    assert wire.ZLIB_RECORDS_V1 == "zlib-records-v1"


@pytest.mark.parametrize(
    ("mine", "theirs", "expected"),
    [
        # new <-> new: compress.
        (list(wire.LINK_CAPABILITIES), list(wire.LINK_CAPABILITIES), True),
        # new <-> old (the old peer has every capability EXCEPT this one): plaintext.
        (
            list(wire.LINK_CAPABILITIES),
            [c for c in wire.LINK_CAPABILITIES if c != wire.ZLIB_RECORDS_V1],
            False,
        ),
        # new <-> a peer that advertised nothing at all (pre-capability build).
        (list(wire.LINK_CAPABILITIES), [], False),
        # old <-> old.
        (
            [c for c in wire.LINK_CAPABILITIES if c != wire.ZLIB_RECORDS_V1],
            [c for c in wire.LINK_CAPABILITIES if c != wire.ZLIB_RECORDS_V1],
            False,
        ),
        # The mirror of new <-> old: the OTHER end is new.
        (
            [c for c in wire.LINK_CAPABILITIES if c != wire.ZLIB_RECORDS_V1],
            list(wire.LINK_CAPABILITIES),
            False,
        ),
    ],
)
def test_handshake_codec_compresses_only_on_the_intersection(
    mine: list[str], theirs: list[str], expected: bool
) -> None:
    """``Handshake.codec`` is the ONE place the decision is made, on both roles."""
    from local_operator.network import handshake as handshake_mod

    # Drive the real method over a stand-in carrying just what it reads; the full
    # handshake is exercised by tests/unit/network/test_handshake.py and the real
    # two-ended cell below.
    class _Stub:
        capabilities = mine
        peer_capabilities = theirs
        role = "dialer"

        @staticmethod
        def establish() -> object:
            class _Result:
                keys = wire.link_keys(_shared(), b"t" * 32, b"l" * 16)

            return _Result()

    codec = handshake_mod.Handshake.codec(_Stub())  # type: ignore[arg-type]
    assert codec.compression is expected


# The real two-ended handshake cells live in test_handshake.py, which owns the
# socket-and-thread harness (``run_handshake``).


def test_a_frame_over_the_threshold_round_trips_compressed() -> None:
    sender, receiver = _pair(dialer_zlib=True, listener_zlib=True)
    frame = _page()
    plaintext = _plaintext_of(frame)
    assert len(plaintext) > wire.COMPRESS_MIN_BYTES
    record = sender.seal(frame)
    # Fewer bytes cross; the marker is inside the AEAD, never in the prefix.
    assert len(record) < len(plaintext) // 3
    assert int.from_bytes(record[:4], "big") == len(record) - 4
    assert _decrypt_raw(receiver, record).startswith(wire.COMPRESSED_MARKER)
    assert receiver.open(record[4:]) == frame


def test_compressed_and_plain_records_interleave_in_one_reader() -> None:
    """The marker makes each record self-describing, so one reader decodes a mix."""
    sender, receiver = _pair(dialer_zlib=True, listener_zlib=True)
    small = {"op": "ping", "req": 1}
    big = _page(20)
    for frame in (small, big, small, big, big, small):
        assert receiver.open(sender.seal(frame)[4:]) == frame


def test_below_the_threshold_is_byte_identical_to_the_uncompressed_codec() -> None:
    keys = wire.link_keys(_shared(), b"t" * 32, b"l" * 16)
    compressing = wire.LinkCrypto(keys, role="dialer", compression=True)
    plain = wire.LinkCrypto(keys, role="dialer")
    # Just under the threshold, so the boundary — not just "tiny" — is what is pinned.
    filler = "a" * (wire.COMPRESS_MIN_BYTES - len('{"op":"x","pad":""}') - 1)
    frame = {"op": "x", "pad": filler}
    assert len(_plaintext_of(frame)) == wire.COMPRESS_MIN_BYTES - 1
    # AES-GCM with a derived nonce is deterministic, so equal bytes is equal bytes.
    assert compressing.seal(frame) == plain.seal(frame)


def test_the_threshold_itself_compresses_when_it_shrinks() -> None:
    sender, receiver = _pair(dialer_zlib=True, listener_zlib=True)
    filler = "a" * (wire.COMPRESS_MIN_BYTES - len('{"op":"x","pad":""}'))
    frame = {"op": "x", "pad": filler}
    assert len(_plaintext_of(frame)) == wire.COMPRESS_MIN_BYTES
    record = sender.seal(frame)
    assert _decrypt_raw(receiver, record).startswith(wire.COMPRESSED_MARKER)


def test_a_frame_zlib_cannot_shrink_is_sent_as_plain_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frame the compressor does not make smaller is never made bigger: it goes
    out exactly as the uncompressed codec would have sent it.

    Real JSON cannot be made incompressible (base64 of random bytes still shrinks
    ~25 %, because a character carries 6 bits), so the compressor is made to lose.
    """
    keys = wire.link_keys(_shared(), b"t" * 32, b"l" * 16)
    compressing = wire.LinkCrypto(keys, role="dialer", compression=True)
    plain = wire.LinkCrypto(keys, role="dialer")
    frame = _page(10)
    monkeypatch.setattr(wire.zlib, "compress", lambda data, level=-1: data + b"\x00" * 16)
    assert compressing.seal(frame) == plain.seal(frame)


def test_without_the_capability_a_big_frame_is_byte_identical_and_old_readable() -> None:
    """THE OLD PEER'S VIEW. The receiver here is a codec built exactly as the
    previous release built it (no ``compression`` argument): it must decode what a
    new build sends it, and the bytes must be the ones the previous build sent."""
    keys = wire.link_keys(_shared(), b"t" * 32, b"l" * 16)
    new_build = wire.LinkCrypto(keys, role="dialer", compression=False)
    old_build = wire.LinkCrypto(keys, role="dialer")
    old_reader = wire.LinkCrypto(keys, role="listener")
    frame = _page()
    record = new_build.seal(frame)
    assert record == old_build.seal(frame)
    assert _decrypt_raw(old_reader, record) == _plaintext_of(frame)
    assert old_reader.open(record[4:]) == frame


def test_a_new_build_decodes_a_plaintext_record_from_an_old_peer() -> None:
    keys = wire.link_keys(_shared(), b"t" * 32, b"l" * 16)
    old_writer = wire.LinkCrypto(keys, role="dialer")
    new_reader = wire.LinkCrypto(keys, role="listener", compression=True)
    frame = _page()
    assert new_reader.open(old_writer.seal(frame)[4:]) == frame


def test_a_compressed_record_on_a_link_that_did_not_negotiate_it_is_refused() -> None:
    """The receive side of the rule: a peer that compresses without having
    negotiated it is the same fatal parse error a bad plaintext always was."""
    sender, receiver = _pair(dialer_zlib=True, listener_zlib=False)
    with pytest.raises(wire.LinkCryptoError) as excinfo:
        receiver.open(sender.seal(_page())[4:])
    assert excinfo.value.kind == "parse"


def test_credential_broker_frames_are_never_compressed() -> None:
    keys = wire.link_keys(_shared(), b"t" * 32, b"l" * 16)
    compressing = wire.LinkCrypto(keys, role="dialer", compression=True)
    plain = wire.LinkCrypto(keys, role="dialer")
    request = {"op": "net_broker", "kind": "copy", "pad": "a" * 5000}
    reply = wire.UncompressedFrame({"op": "ack", "req": 3, "detail": {"value": "a" * 5000}})
    assert compressing.seal(request) == plain.seal(request)
    assert compressing.seal(reply) == plain.seal(dict(reply))


@pytest.mark.parametrize(
    "body",
    [
        b"",  # marker and nothing after it
        b"not a zlib stream",
        zlib.compress(b'{"op":"ping"}')[:-4],  # truncated: no end-of-stream
        zlib.compress(b'{"op":"ping"}') + b"trailing",  # bytes after the stream's end
        zlib.compress(b"[1,2,3]"),  # inflates to JSON that is not an object
        zlib.compress(b"\xff\xfe not utf-8"),
    ],
)
def test_a_malformed_compressed_record_is_fatal_and_yields_no_frame(body: bytes) -> None:
    sender, receiver = _pair(dialer_zlib=True, listener_zlib=True)
    record = _seal_raw(sender, wire.COMPRESSED_MARKER + body)
    with pytest.raises(wire.LinkCryptoError) as excinfo:
        receiver.open(record[4:])
    assert excinfo.value.kind in {"parse", "limit"}


def test_a_decompression_bomb_is_refused_at_the_record_ceiling() -> None:
    """A few KB that inflate past ``MAX_RECORD_BYTES`` must stop at the ceiling —
    the bound applies as the stream inflates, not after."""
    sender, receiver = _pair(dialer_zlib=True, listener_zlib=True)
    bomb = zlib.compress(b"{" + b"a" * (wire.MAX_RECORD_BYTES + 1024), 9)
    assert len(bomb) < 64 * 1024
    record = _seal_raw(sender, wire.COMPRESSED_MARKER + bomb)
    with pytest.raises(wire.LinkCryptoError) as excinfo:
        receiver.open(record[4:])
    assert excinfo.value.kind == "limit"


def test_the_ceiling_is_on_the_compressed_size_that_crosses(socketpair: object) -> None:
    """The length prefix is still checked against ``MAX_RECORD_BYTES`` and it is the
    COMPRESSED length that is in it: a frame whose plaintext is over the ceiling is
    still refused by ``seal`` (the plaintext bound is unchanged), and an oversized
    prefix is still refused before a byte is allocated."""
    sender, _receiver = _pair(dialer_zlib=True, listener_zlib=True)
    with pytest.raises(wire.LinkCryptoError) as excinfo:
        sender.seal({"op": "x", "blob": "a" * (wire.MAX_RECORD_BYTES + 1)})
    assert excinfo.value.kind == "limit"
    client, server = socketpair  # type: ignore[misc]
    server.sendall((wire.MAX_RECORD_BYTES + 1).to_bytes(4, "big"))
    with pytest.raises(wire.LinkCryptoError):
        wire.FrameReader(client).read_record_payload(wire.deadline_in(2.0))


def test_a_tampered_compressed_record_fails_authentication_before_inflation() -> None:
    sender, receiver = _pair(dialer_zlib=True, listener_zlib=True)
    record = bytearray(sender.seal(_page()))
    record[-1] ^= 0x01
    with pytest.raises(wire.LinkCryptoError) as excinfo:
        receiver.open(bytes(record[4:]))
    assert excinfo.value.kind == "auth"


def test_compressed_records_cross_a_real_socket_pipelined(socketpair: object) -> None:
    """The bounded in-process link: real loopback sockets, the production
    ``FrameReader``, a mix of compressed and plain records back to back."""
    client, server = socketpair  # type: ignore[misc]
    sender, receiver = _pair(dialer_zlib=True, listener_zlib=True)
    frames = [{"op": "ping", "req": 1}, _page(30), {"op": "ping", "req": 2}, _page(60)]
    server.sendall(b"".join(sender.seal(frame) for frame in frames))
    reader = wire.FrameReader(client)
    for frame in frames:
        assert receiver.open(reader.read_record_payload(wire.deadline_in(5.0))) == frame
