"""Conversation handles for push deep links: mint, resolve, rotate.

Push/ack-sync S2 of ADR 0006 (``damianvtran/local-operator-mobile`` PR #14 @
22e2cce2, §3.1/§4). The aggregate read (S1) carries a ``push_handle`` on each
of its ``conversations[]`` rows, a push's ``thread-id``/collapse identity is
that same handle, and ``GET /api/push/conversation/{handle}`` is how a cold
tap -- a conversation that is no longer unread, or one the app has not listed
yet -- gets back to a session id. This module owns the whole mechanism: the
key, the mint recipe, and the lookup.

Four properties are the design, and each one is easy to lose by simplifying:

- **No stored mapping.** The handle is a pure function --
  ``base64url(HMAC-SHA256(key, conversation_identity))[:22]`` -- so there is
  nothing to migrate, nothing to prune when a conversation is deleted, and no
  second table that can disagree with the attention store.
  :func:`local_operator.session.attention.conversation_identity` is already
  the stable input; the mint adds no identity of its own.
- **Stability is the whole point.** A daemon restart, a re-register, or a
  later completion on the same conversation must NOT remint: the key persists
  at ``<config root>/push-handle.key`` (32 random bytes, 0600, minted on
  first use), the mint is deterministic, and nothing here ever replaces an
  existing key. A handle that moved between pushes would split one
  conversation into several threads on the phone, and every pending push
  would fail to resolve.
- **Rotation is an operation, not a side effect.** Deleting the key file is
  the documented way to break every handle at once; pending pushes then
  resolve to 404 and the app falls back to its list [ADR §4]. A key file that
  cannot be read as 32 bytes is refused (:class:`PushHandleKeyCorrupt`)
  rather than silently replaced, because replacing it would invalidate every
  outstanding handle as a side effect of a fault.
- **Per machine, not per account.** The key lives under the config root, so
  two machines mint different handles for what a user thinks of as the same
  conversation -- correct, because on this design they are different
  conversations [ADR §1.6/§4].

The key is deliberately write-lazy: it appears when the first handle is
actually minted (the aggregate's first build that serves a row), never at
startup, and never from a resolve -- a GET that finds no key answers "this
machine mints nothing" instead of minting one.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import threading
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

#: The key file, directly under ``config_dir()`` beside the daemon's other
#: owner-private state (``mobile-seen.json``, ``mobile-push-devices.json``).
PUSH_HANDLE_KEY_NAME = "push-handle.key"

#: The key is 32 random bytes -- the width the ADR states, and one SHA-256
#: block's worth of key material.
_KEY_BYTES = 32

#: 22 base64url characters = 132 bits of the 256-bit HMAC. Long enough that
#: two conversations on one machine colliding is unreachable, short enough
#: that the handle stays cheap on every wire it rides (pushes, deep links).
_HANDLE_CHARS = 22

#: The base64url alphabet, for the cheap shape screen in
#: :func:`resolve_conversation` -- a string that cannot be a handle need not
#: reach the filesystem.
_HANDLE_ALPHABET = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_")

#: One lock over the read-or-mint cycle. The daemon runs the mint from
#: ``asyncio.to_thread`` workers, and two overlapping builds must not both
#: believe they minted the key -- whoever loses that race would mint handles
#: nobody else agrees with. In-process is sufficient and deliberate, the same
#: rule the device registry states: exactly one daemon process per config root
#: mints this key (a restart's predecessor is gone before its successor
#: serves), and the ``O_EXCL`` arm below covers what the lock cannot.
_LOCK = threading.Lock()


class PushHandleKeyCorrupt(RuntimeError):
    """The stored key exists but is not the 32 bytes the mint needs.

    Refused rather than replaced: see the module docstring's rotation bullet.
    :func:`resolve_conversation` treats it as "this machine mints no handle
    for anything" (its caller answers 404); the aggregate read serves its rows
    without ``push_handle``. The operator's repair is the documented rotation
    -- delete the file; the next mint creates a fresh one.
    """


def key_path(config_dir: Path) -> Path:
    """The key file's path under one config root.

    Public because the daemon's refusal log names it: "which file" is the
    first question a reader of that log line asks -- the same rule the device
    registry states for its ``store_path``.
    """
    return config_dir / PUSH_HANDLE_KEY_NAME


def handle_for(key: bytes, identity: str) -> str:
    """The mint recipe, pure: ``base64url(HMAC-SHA256(key, identity))[:22]``.

    Public because tests pin the exact output bytes here, and because the
    push worker (S5) needs the same recipe for the payload's ``conversation``
    field: one spelling of the mint, never a second.
    """
    digest = hmac.new(key, identity.encode("utf-8"), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")[:_HANDLE_CHARS]


def conversation_handles(config_dir: Path, session_ids: Sequence[str]) -> list[str]:
    """The handle per session id, in order -- reading (or minting) the key ONCE.

    ``session_ids`` are session directory names under ``config_dir()/sessions``;
    each is minted over ``conversation_identity`` of its directory, the same
    input the attention store keys the conversation by. One key read feeds the
    whole batch, so a build with n unread rows pays one read, not n.
    """
    from local_operator.session.attention import conversation_identity

    key = _load_or_mint_key(config_dir)
    root = config_dir / "sessions"
    return [handle_for(key, conversation_identity(root / session_id)) for session_id in session_ids]


def resolve_conversation(config_dir: Path, handle: str) -> str | None:
    """The session id ``handle`` names, or ``None`` when nothing here mints it.

    The lookup is a scan over the conversation directories (the handle is not
    invertible, and there is deliberately no stored mapping): mint each
    directory's handle and compare. A conversation that no longer exists is
    simply not among the candidates, and a rotated key simply changes which
    handles the scan mints -- both end in the caller's 404, which is why this
    function answers ``None`` rather than raising for either. The
    never-a-500 contract the route states rests on that.

    This function never mints: a root with no key file resolves nothing, so a
    GET cannot be what brings a key into being (see the module docstring).
    """
    if len(handle) != _HANDLE_CHARS or not set(handle) <= _HANDLE_ALPHABET:
        # Cannot be a handle this machine mints; skip the scan entirely.
        return None
    try:
        key = _read_key(key_path(config_dir))
    except (PushHandleKeyCorrupt, OSError) as exc:
        # A key this machine cannot use resolves nothing. One bounded line,
        # no ``exc_info`` -- the log-hygiene rule the device registry states
        # at ``_push_call`` (QA round 1 Q-1): the sentence
        # IS the diagnosis, and a retrying client must not add a traceback
        # per attempt. Warning, not an error: the route's contract is that an
        # unresolvable handle is cleanly unknown, and the aggregate read says
        # the same thing by omitting the field.
        logger.warning("push handle resolution refused at %s: %s", key_path(config_dir), exc)
        return None
    if key is None:
        return None
    from local_operator.session.attention import conversation_identity

    root = config_dir / "sessions"
    try:
        with os.scandir(root) as entries:
            names = [entry.name for entry in entries if entry.is_dir()]
    except FileNotFoundError:
        # No conversations on this machine at all: the honest answer is that
        # nothing here mints this handle.
        return None
    except OSError as exc:
        # One line, no ``exc_info``: the same rule the key refusal states.
        logger.warning("push handle resolution could not scan %s: %s", root, exc)
        return None
    for name in names:
        if hmac.compare_digest(handle_for(key, conversation_identity(root / name)), handle):
            return name
    return None


def _read_key(path: Path) -> bytes | None:
    """The stored key, ``None`` when no key exists yet -- or the refusal.

    A key file with any other width is refused rather than truncated or
    padded: either would silently change every handle it mints.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    if len(raw) != _KEY_BYTES:
        raise PushHandleKeyCorrupt(f"{path} holds {len(raw)} bytes, not {_KEY_BYTES}")
    return raw


def _load_or_mint_key(config_dir: Path) -> bytes:
    """Read the key, minting one on first use -- never replacing an existing one.

    The mint is ``O_CREAT|O_EXCL`` so two racers cannot both become "the" key:
    the loser reads the winner's file on its next pass, and both sides then
    mint identical handles. The lock makes the common case (two builds inside
    this process) a single serialized read-or-mint; the ``O_EXCL`` arm covers
    the cross-process case the lock cannot.
    """
    path = key_path(config_dir)
    with _LOCK:
        for _ in range(2):
            key = _read_key(path)
            if key is not None:
                return key
            try:
                return _mint_key(path)
            except FileExistsError:
                # Another process won the mint between the read and the
                # create; its key is the key -- read it on the next pass.
                continue
        key = _read_key(path)
        if key is None:
            raise PushHandleKeyCorrupt(f"{path} could not be read after a concurrent mint")
        return key


def _mint_key(path: Path) -> bytes:
    """Create the key 0600, first-writer-wins, or raise ``FileExistsError``.

    One ``os.write`` of 32 bytes under an ``O_EXCL`` create. The mode is set
    with ``chmod`` after the write so a restrictive umask cannot make the file
    anything other than 0600 (the create itself asks for 0600, so there is no
    window in which it is more permissive). A crash between create and write
    leaves an empty file -- refused as corrupt rather than silently replaced,
    so the operator's repair stays the documented one: delete it, and the
    next use mints.
    """
    key = os.urandom(_KEY_BYTES)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        if os.write(descriptor, key) != _KEY_BYTES:
            raise OSError(f"short write minting {path}")
        os.chmod(path, 0o600)
    except BaseException:
        os.close(descriptor)
        try:
            path.unlink()
        except OSError:
            pass
        raise
    os.close(descriptor)
    return key
