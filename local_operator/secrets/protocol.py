"""The broker wire format, and where the socket lives.

Stdlib-only and free of the crypto stack, for the same reason
:mod:`local_operator.secrets.cli` is: a client that only needs to discover
"is there a broker?" must not pay for an OpenSSL binding to ask, and
``tests/unit/secrets/test_startup_cost.py`` pins that.

**The frame.** One JSON object per message, length-prefixed with a 4-byte
big-endian header. Newline-delimited JSON would be cheaper to read in a log but
wrong here: a value is arbitrary BYTES (a PEM, a binary token) and any
delimiter-scanning reader has to define what happens when the payload contains
the delimiter. A length prefix has no such case. Byte-valued fields are
base64'd inside the JSON rather than sent raw, so the frame is always valid
UTF-8 JSON and a malformed value fails at parse rather than at use.

**Why a size ceiling exists at all.** The length prefix is read from an
unauthenticated peer — authentication happens after the connection is
established — so a 4-byte length is an invitation to ask the broker to allocate
4 GiB. :data:`MAX_FRAME_BYTES` bounds that at a size no legitimate secret
approaches.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import stat
import struct
import tempfile
from pathlib import Path
from typing import Any

#: Wire protocol version. Sent on every request and checked by the broker: the
#: operator updates the runtime underneath live sessions (``AGENTS.md``), so a
#: newer client meeting an older broker is routine, not exceptional, and must
#: produce a clear message instead of a confusing parse error.
PROTOCOL_VERSION = 1

#: Ceiling on a single frame. Comfortably above any credential — a 4096-bit RSA
#: key in PEM is ~3 KiB — and small enough that a hostile length prefix cannot
#: exhaust memory.
MAX_FRAME_BYTES = 4 * 1024 * 1024

_HEADER = struct.Struct(">I")

#: Filename of the socket inside the secrets directory.
SOCKET_FILENAME = "broker.sock"
#: Filename of the lock that serialises lazy broker startup.
LOCK_FILENAME = "broker.lock"

#: macOS ``sockaddr_un.sun_path`` is 104 bytes and ``bind()`` fails silently
#: past it. Measured on this machine: a 103-byte path binds, 104 fails. The
#: config dir is operator-controlled (``LOCAL_OPERATOR_CONFIG_DIR``, and tests
#: use deep ``tmp_path`` trees), so this is a real limit rather than a
#: theoretical one, and :func:`socket_path` reports it as such.
MAX_SOCKET_PATH = 103


class ProtocolError(Exception):
    """A frame could not be read or was not well-formed."""


def _runtime_fallback_dir(secrets_directory: Path) -> Path:
    """A short, private directory to hold the socket when the real one is too deep.

    Named from a hash of the secrets directory so that two stores never collide
    on one socket — which would be a correctness AND a security problem, since
    a caller would reach a broker holding a different store's key.

    The uid is in the name, and :func:`ensure_runtime_dir` verifies ownership
    and mode before anything binds or connects: on Linux ``TMPDIR`` is commonly
    the shared ``/tmp``, where another user could pre-create the directory and
    wait for us to place a socket inside it. macOS gives each user a private
    ``/var/folders`` root, but the check is unconditional rather than
    platform-dependent, because the cost is one ``stat`` and the failure it
    prevents is silent.
    """
    digest = hashlib.sha256(str(secrets_directory).encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"lop-secrets-{os.getuid()}-{digest}"


def ensure_runtime_dir(directory: Path) -> Path:
    """Create ``directory`` 0700 and refuse it if someone else owns it.

    ``mkdir`` with ``exist_ok`` would happily adopt a directory an attacker
    created first. The explicit checks below are what make adopting one safe:
    it must be a real directory (not a symlink into somewhere else), owned by
    this uid, and inaccessible to anyone else.
    """
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise ProtocolError(f"{directory} is not a directory")
    if info.st_uid != os.getuid():
        raise ProtocolError(
            f"{directory} is owned by uid {info.st_uid}, not {os.getuid()}; refusing to "
            "put a secret-broker socket inside a directory another account controls"
        )
    # Re-assert the mode: exist_ok=True does not apply the mode to a directory
    # that already existed, and the umask masks it on creation.
    os.chmod(directory, 0o700)
    return directory


def socket_path(base: Path | None = None) -> Path:
    """Where the broker listens.

    Normally inside the 0700 secrets directory, so the socket inherits a
    directory that already excludes other users; the socket itself is
    additionally chmod'd 0600 by the broker. macOS enforces both (spike 7:
    ``chmod 000`` gave even the owner ``EACCES`` on ``connect()``), which makes
    the mode bits a real barrier rather than a decoration.

    **The length fallback is not cosmetic.** ``sockaddr_un.sun_path`` is 104
    bytes on macOS and ``bind()`` fails past it — measured on this machine: a
    103-byte path binds, 104 fails with a bare ``EINVAL``. The default config
    dir yields a 49-byte path and is never affected, but
    ``LOCAL_OPERATOR_CONFIG_DIR`` is operator-controlled and pytest's
    ``tmp_path`` is routinely ~145 bytes, so a deep config dir would otherwise
    make the store simply unusable with an error naming nothing actionable.
    When the natural path does not fit, the socket moves to a short private
    directory under ``TMPDIR`` whose name is derived from the secrets directory.
    Only the RENDEZVOUS POINT moves; the key, the database and the audit log
    stay where they were.
    """
    from local_operator.secrets.keys import secrets_dir

    directory = secrets_dir(base)
    candidate = directory / SOCKET_FILENAME
    if len(str(candidate)) <= MAX_SOCKET_PATH:
        return candidate
    return _runtime_fallback_dir(directory) / SOCKET_FILENAME


def lock_path(base: Path | None = None) -> Path:
    """The lazy-start lock file. See :func:`local_operator.secrets.client.ensure_broker`.

    Deliberately placed BESIDE the socket, following it into the fallback
    directory when the path length forces one. The lock exists to make ten
    racing sessions produce one broker *at a given rendezvous point*; a lock in
    a different directory than the socket it guards would serialise the wrong
    thing.
    """
    return socket_path(base).parent / LOCK_FILENAME


def encode_bytes(raw: bytes) -> str:
    """Bytes as base64 text, for embedding in a JSON frame."""
    return base64.b64encode(raw).decode("ascii")


def decode_bytes(text: str) -> bytes:
    """Inverse of :func:`encode_bytes`; raises :class:`ProtocolError` on junk."""
    try:
        return base64.b64decode(text.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise ProtocolError(f"malformed base64 field: {exc}") from exc


def send_frame(connection: socket.socket, payload: dict[str, Any]) -> None:
    """Write one length-prefixed JSON frame."""
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(body) > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame of {len(body)} bytes exceeds the {MAX_FRAME_BYTES} limit")
    connection.sendall(_HEADER.pack(len(body)) + body)


def _recv_exactly(connection: socket.socket, count: int) -> bytes:
    """Read exactly ``count`` bytes or raise.

    ``recv`` on a stream socket may return short for any reason; a reader that
    treats one ``recv`` as one message works for small frames on an idle
    machine and corrupts under load, which is the worst possible failure
    distribution for a security boundary.
    """
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = connection.recv(remaining)
        if not chunk:
            raise ProtocolError("connection closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(connection: socket.socket) -> dict[str, Any]:
    """Read one length-prefixed JSON frame."""
    length = _HEADER.unpack(_recv_exactly(connection, _HEADER.size))[0]
    if length > MAX_FRAME_BYTES:
        raise ProtocolError(f"peer announced a {length}-byte frame; the limit is {MAX_FRAME_BYTES}")
    try:
        payload = json.loads(_recv_exactly(connection, length).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"malformed frame: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("frame was not a JSON object")
    return payload
