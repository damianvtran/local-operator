"""Bytes written through ``os.open`` + ``os.write`` must be the bytes we meant.

THE MEASURED FAILURE THIS PINS. ``xplat-probe-windows`` reported:

    FAIL  secret.roundtrip  secret set failed: ...\\config\\secrets\\master.key is
                            33 bytes; a master key is 32.

``generate_master_key()`` is ``os.urandom(32)`` -- it cannot return 33 bytes.
The extra byte is the CRT's: on Windows ``os.open`` opens in TEXT mode unless
``O_BINARY`` is passed, and text mode TRANSLATES, so a ``0x0A`` in the buffer
handed to ``os.write`` is written as ``0x0D 0x0A``. A 32-byte key therefore
lands as 33 whenever the key happens to contain a newline.

WHY IT HID FOR SO LONG: the corruption is INTERMITTENT. It needs one specific
byte value among 32 random ones -- about one Windows run in eight -- so the
probe's ``secret.roundtrip`` passed on most runs and failed on one, which reads
exactly like a flake and is not one. A test that waits for the random byte would
be just as unreliable, so these tests FORCE it: every payload here contains
``0x0A`` by construction, which makes the failure deterministic on any platform
that translates and a no-op everywhere else.

Off Windows ``paths.O_BINARY`` is ``0``, so ``flags | O_BINARY`` is
bit-identical to ``flags`` -- these tests cannot change POSIX behaviour, they
only stop the Windows arm from corrupting what it writes.
"""

from __future__ import annotations

import os
from pathlib import Path

from local_operator import paths
from local_operator.secrets import keys

#: 32 bytes of the one value that text mode rewrites. Same length as a master
#: key, so a translated write is one byte longer than what was handed in.
NEWLINE_PAYLOAD = b"\n" * 32


def test_o_binary_is_a_no_op_off_windows() -> None:
    """The premise every other assertion rests on, checked rather than assumed."""
    if os.name == "nt":
        assert paths.O_BINARY != 0, "Windows must have a real O_BINARY"
    else:
        assert paths.O_BINARY == 0, "POSIX has no such mode, so the flag must vanish"


def test_write_private_file_preserves_a_newline_byte(tmp_path: Path) -> None:
    """The primitive the master key is written through."""
    target = tmp_path / "payload.bin"

    keys.write_private_file(target, NEWLINE_PAYLOAD)

    written = target.read_bytes()
    assert written == NEWLINE_PAYLOAD, (
        f"wrote {len(NEWLINE_PAYLOAD)} bytes and read back {len(written)}: "
        "the descriptor was in text mode and translated the payload"
    )


def test_create_private_file_preserves_a_newline_byte(tmp_path: Path) -> None:
    """The exclusive variant, which is what first-use master keys go through."""
    target = tmp_path / "master.key"

    assert keys.create_private_file(target, NEWLINE_PAYLOAD) is True

    assert target.read_bytes() == NEWLINE_PAYLOAD


def test_the_master_key_round_trips_when_it_contains_a_newline(tmp_path: Path, monkeypatch) -> None:
    """The end-to-end shape of the reported failure, with the byte forced.

    ``generate_master_key`` is replaced with a key that contains the byte text
    mode rewrites, so this fails on a platform that translates regardless of
    what the random source produces — which is the whole point: the real bug
    only appeared on the runs where the key happened to hold it.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(keys, "generate_master_key", lambda: NEWLINE_PAYLOAD)

    created = keys.load_master_key(create=True)

    assert (
        created == NEWLINE_PAYLOAD
    ), f"read back {len(created)} bytes where {len(NEWLINE_PAYLOAD)} were written"
    assert keys.load_master_key() == NEWLINE_PAYLOAD, "and it must survive a re-read"


def test_the_stored_secret_is_not_translated(tmp_path: Path, monkeypatch) -> None:
    """The sibling site: the secret VALUE goes through its own ``os.open``.

    A translated VALUE is the worse half of the same bug — the key file size is
    checked and refuses, while a value that gains a ``0x0D`` decrypts to
    something subtly different from what was stored.
    """
    from local_operator.secrets import handlers

    source = Path(handlers.__file__).read_text(encoding="utf-8")
    opened = [line for line in source.splitlines() if "os.open(" in line and "O_WRONLY" in line]
    assert opened, "the value write moved; this test needs pointing at its new site"
    for line in opened:
        assert "O_BINARY" in line, (
            "the secret VALUE is written through an os.open without O_BINARY: " f"{line.strip()}"
        )
