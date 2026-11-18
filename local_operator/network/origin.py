"""Mesh origin stamp (G5) and relay→runtime proof (G6).

Design: ``docs/design/mesh-origin-stamp.md`` (security review 60334, round 3
APPROVED). This module owns the shared primitives; the call sites are the relay
(``network/dial.py``, ``network/relay.py``), the runtime control server
(``session/runtime/server.py``), the spool and the ask store.

WHAT IT IS. A remote prompt, steer or ask answer used to carry one boolean
("remote") and nothing about WHICH device sent it. The receiving runtime now
writes one object — the *stamp* — beside it:

* on a transcript row: ``message.provider_payload["lop_mesh_origin"]``;
* on an ask answer: ``record["answered_by"]["mesh_origin"]``.

Its fields are ``origin_device_id``, ``network_id``, ``link_id``,
``capabilities`` (sorted), ``verified`` and ``stamped_by``; ``operator_origin``
is a RESERVED slot for G7 that nothing here ever writes.

THE TRUST RULES, each one a finding the review closed:

1. **Never from a frame.** The stamp is built only by the RECEIVING runtime,
   from its own connection facts. A frame-supplied ``lop_mesh_origin`` /
   ``mesh_origin`` is stripped wherever a frame is read (:data:`FRAME_ORIGIN_KEYS`).
2. **``verified`` = proof_ok AND block.peer_verified.** ``proof_ok`` is this
   module's G6 proof: the owner's relay proves, per connection, that it holds
   ``K_relay`` (HKDF of this device's Ed25519 seed) over the runtime's salt, the
   relay's nonce and the canonical identity block it declared. ``peer_verified``
   is the per-link handshake fact (``PeerLink.peer_verified``) carried INSIDE the
   MAC'd block, so a version-skewed relay can only ever lower it.
3. **Evidence, never a verdict.** A stored ``verified`` is re-judged at consent
   time by :func:`mesh_origin_counts` against live membership and THIS device's
   current identity; anything imported by a move or replica promote is
   downgraded to ``verified=false`` (:func:`downgrade_stamps`).
4. **"verified" means a DEVICE over an authenticated link, never a human.**
   Separating a human from an agent on the peer is G7's job (``operator_origin``).

INERT IN THIS RELEASE. No guard or guardrail reads the stamp or calls
:func:`mesh_origin_counts`; ``tests/unit/network/test_mesh_origin.py`` pins that.

RESIDUAL RISK (design §3.3), stated so no reader over-trusts this: the proof
raises "forge a mesh-verified connection" from "read the 0600 runtime record" to
"read the device private key". It is NOT a boundary against an unconfined
same-uid process, and nothing at rest is integrity protected.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import stat
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from local_operator.session.compression import ARCHIVE_FILENAMES

logger = logging.getLogger(__name__)

#: ``provider_payload`` key on a transcript row (prompt or steer). Invisible to the
#: model and to providers, like ``lop_remote_prompt`` beside it.
MESH_ORIGIN_PAYLOAD_KEY = "lop_mesh_origin"
#: Key inside ``answered_by`` on an ask answer, and the keyword every layer of the
#: carriage (server → handle → session → queue) uses.
MESH_ORIGIN_ANSWER_KEY = "mesh_origin"
#: Every spelling a FRAME could use to smuggle a stamp. Stripped on read, always:
#: rule 1 says a stamp is built by the receiver, so a frame copy is at best noise
#: and at worst a forgery.
FRAME_ORIGIN_KEYS: tuple[str, ...] = (MESH_ORIGIN_PAYLOAD_KEY, MESH_ORIGIN_ANSWER_KEY)

#: The op the owner's relay sends as its FIRST post-welcome frame (design §3.1).
RELAY_ATTEST_OP = "relay_attest"
#: Domain separation for the proof, distinct from the operator-capability labels
#: (``harness/approval._PROOF_LABEL_*``) so no proof is replayable as another.
RELAY_ATTEST_LABEL = b"lop-relay-attest-v1"
#: HKDF ``info`` for ``K_relay``: binds the derived key to this one purpose, so the
#: same seed cannot yield a key some other construction also uses.
RELAY_KEY_INFO = b"lop-relay-runtime-v1"
RELAY_KEY_BYTES = 32

#: The string keys of the proved block (M-11); with ``capabilities`` and
#: ``peer_verified`` they are the six keys the proof binds. ``name`` is display data
#: but IS bound, so a relay cannot relabel a proved device. Any key a relay sends
#: outside these six is unbound and never reaches the stamp.
BLOCK_STR_KEYS: tuple[str, ...] = ("device_id", "name", "network_id", "link_id")


# ---------------------------------------------------------------------------
# K_relay
# ---------------------------------------------------------------------------


def derive_relay_key(seed: bytes) -> bytes:
    """``HKDF-SHA256(ikm=seed, salt=0^32, info=RELAY_KEY_INFO, L=32)`` (RFC 5869).

    Stdlib HMAC rather than ``cryptography``'s HKDF so this module stays importable
    by the runtime without pulling the crypto stack onto its startup path; the
    construction is the two-line RFC one and ``L=32`` needs a single expand block.
    No new secret file (design §3.2): the device key already IS the mesh trust
    root, and deriving from it inherits its confinement deny rule.
    """
    if not isinstance(seed, (bytes, bytearray)) or len(seed) != 32:
        raise ValueError("the relay key derives from a 32-byte Ed25519 seed")
    prk = hmac.new(b"\x00" * hashlib.sha256().digest_size, bytes(seed), hashlib.sha256).digest()
    return hmac.new(prk, RELAY_KEY_INFO + b"\x01", hashlib.sha256).digest()[:RELAY_KEY_BYTES]


def relay_key_for(root: Path | None = None) -> tuple[bytes, str] | None:
    """``(K_relay, device_id)`` for this install, or ``None`` with no identity.

    The SEED never leaves this function: it is decoded, derived from and dropped,
    so a caller (the runtime caches the result for its lifetime, M-10) holds only
    the purpose-bound key. The relay calls this per dial instead, so an identity
    rotation is picked up there and a runtime still holding the old key simply
    fails the proof — fail-closed, never a stale "verified".
    """
    from local_operator.network import identity as identity_module
    from local_operator.network.wire import unb64u

    loaded = identity_module.load(root)
    if loaded is None or not loaded.private_key:
        return None
    seed = unb64u(loaded.private_key)
    try:
        return derive_relay_key(seed), str(loaded.device_id)
    finally:
        del seed


def relay_nonce() -> str:
    """A fresh per-dial nonce (the relay's) or salt (the runtime's), hex.

    Reuses the operator handshake's generator and wire shape so the one
    ``is_wire_hex`` check validates both directions.
    """
    from local_operator.harness.approval import operator_nonce

    return operator_nonce()


# ---------------------------------------------------------------------------
# The proved block and the proof
# ---------------------------------------------------------------------------


def canonical_block(peer: Mapping[str, Any] | None, capabilities: Iterable[Any]) -> dict[str, Any]:
    """The exact six-key block the proof covers (design §3.1, M-11).

    Both ends compute it independently — the relay from the ``peer`` block and the
    capability list it is about to send, the runtime from what it received — so the
    form is pinned: four ``str`` keys, ``peer_verified`` a ``bool`` that is True
    only for a literal ``True`` (absent, ``"true"`` or ``1`` read as False: the
    skew-safe direction), and ``capabilities`` as ``sorted(set(str(c)))``, which is
    what ``dial.py`` sends and what the runtime's frozenset re-sorts to.
    """
    source = peer if isinstance(peer, Mapping) else {}
    block: dict[str, Any] = {key: str(source.get(key) or "") for key in BLOCK_STR_KEYS}
    block["peer_verified"] = source.get("peer_verified") is True
    block["capabilities"] = sorted({str(item) for item in (capabilities or ())})
    return block


def _canonical_bytes(block: Mapping[str, Any]) -> bytes:
    from local_operator.network.wire import canonical_json

    return canonical_json(dict(block))


def relay_proof(key: bytes, *, relay_nonce: str, relay_salt: str, block: Mapping[str, Any]) -> str:
    """``hmac_sha256(K_relay, label || nonce || ":" || salt || ":" || canonical(block))``.

    The ``request_proof_ok`` shape (``harness/approval._proof``) plus one bound
    input: the block, so a proof minted for one identity cannot be presented with
    another (test 4). The runtime's salt makes a captured auth frame + proof
    worthless on any other connection.
    """
    message = (
        RELAY_ATTEST_LABEL
        + relay_nonce.encode("ascii")
        + b":"
        + relay_salt.encode("ascii")
        + b":"
        + _canonical_bytes(block)
    )
    return hmac.new(bytes(key), message, hashlib.sha256).hexdigest()


def relay_proof_ok(
    supplied: object,
    *,
    key: bytes | None,
    relay_nonce: str,
    relay_salt: str,
    block: Mapping[str, Any],
) -> bool:
    """Constant-time check of an offered attest proof. FAIL-CLOSED on every gap.

    No key (this install has no identity, or it could not be read), no nonce or
    salt (an old relay, or a connection that never offered one), a non-hex or
    wrong-length proof: each is ``False``, never an exception, mirroring
    ``harness/approval._proof_ok``.
    """
    from local_operator.harness.approval import is_wire_hex

    if not isinstance(key, (bytes, bytearray)) or len(key) != RELAY_KEY_BYTES:
        return False
    if not is_wire_hex(relay_nonce) or not is_wire_hex(relay_salt):
        return False
    if not isinstance(supplied, str) or not is_wire_hex(supplied):
        return False
    try:
        expected = relay_proof(key, relay_nonce=relay_nonce, relay_salt=relay_salt, block=block)
    except (TypeError, ValueError, UnicodeError):
        return False
    return hmac.compare_digest(supplied.lower().encode("ascii"), expected.encode("ascii"))


# ---------------------------------------------------------------------------
# The stamp
# ---------------------------------------------------------------------------


def build_stamp(
    peer: Mapping[str, Any] | None,
    capabilities: Iterable[Any],
    *,
    verified: bool,
    stamped_by: str,
) -> dict[str, Any]:
    """The stamp a RECEIVING runtime writes for one non-local delivery.

    ``verified`` must already be ``proof_ok and peer_verified`` (the caller holds
    the connection); this function only shapes it. An unproved connection still
    gets a stamp with the CLAIMED identity and ``verified=False`` (design §2 rule
    2): audit evidence of who said they were writing, never trust.
    ``operator_origin`` is deliberately absent (G7's slot).
    """
    block = canonical_block(peer, capabilities)
    return {
        "origin_device_id": block["device_id"],
        "network_id": block["network_id"],
        "link_id": block["link_id"],
        "capabilities": block["capabilities"],
        "verified": verified is True,
        "stamped_by": str(stamped_by or ""),
    }


def fold_answer_origin(
    first: Mapping[str, Any] | None, revisions: Sequence[Mapping[str, Any] | None]
) -> dict[str, Any] | None:
    """The stamp an ask record publishes across its first answer and every revision.

    Design §2 rule 4 (M-8): the record keeps the FIRST answer's attribution, so
    without a fold a verified "No" revised to "Yes" by an unproven dial would keep
    ``verified=true`` over words the device never wrote. ``verified`` is therefore
    the AND over every write, where a revision with NO stamp (a local or unproven
    dial) or ``verified:false`` counts as false, and a revision from another device
    or minted by another runtime (``origin_device_id``/``stamped_by`` differ) is
    false too. ``None`` when the first answer carried no stamp: absent is already
    unverified (rule 5) and inventing one would misattribute the first write.
    """
    if not isinstance(first, Mapping):
        return None
    folded = dict(first)
    verified = first.get("verified") is True
    for revision in revisions:
        if not isinstance(revision, Mapping) or revision.get("verified") is not True:
            verified = False
            continue
        if str(revision.get("origin_device_id") or "") != str(
            first.get("origin_device_id") or ""
        ) or str(revision.get("stamped_by") or "") != str(first.get("stamped_by") or ""):
            verified = False
    folded["verified"] = verified
    return folded


# ---------------------------------------------------------------------------
# Imported sessions: downgrade every stamp (M-4, R2-1)
# ---------------------------------------------------------------------------

#: Byte needle that every stamp key contains once JSON-encoded. A file (or a line)
#: without it carries no stamp, so the downgrade never parses or rewrites it — the
#: common case for a session no remote device ever wrote to.
_STAMP_NEEDLE = b'mesh_origin"'

#: A ``\u00XX`` escape of an ASCII lower-case letter or ``_`` — the only way a
#: JSON key can DECODE to ``mesh_origin`` without containing :data:`_STAMP_NEEDLE`
#: as bytes (``"mesh\u005forigin"``). ``json.dumps`` never writes one (it escapes
#: only non-ASCII and control characters), so an honest row never matches; a
#: line that does is treated as possibly stamped (review R3-1).
_KEY_LETTER_ESCAPE = re.compile(rb"\\u00(?:5[fF]|[67][0-9a-fA-F])")


class StampLineRejected(ValueError):
    """A possibly stamped line that the ONE strict stamp parser will not read."""


def line_may_carry_stamp(line: bytes) -> bool:
    """Whether ``line`` (raw bytes, never decoded) could hold a stamp key.

    Matched on BYTES before any decoding so no decoder choice (strict,
    ``"replace"``, surrogateescape) can hide a key from it. False means no JSON
    parser can find a :data:`FRAME_ORIGIN_KEYS` key on the line.
    """
    return _STAMP_NEEDLE in line or _KEY_LETTER_ESCAPE.search(line) is not None


def parse_stamp_line(line: bytes) -> Any:
    """Parse ONE ``\\n``-delimited JSONL line under the strict stamp rules.

    THE single parser that decides whether a stamp at rest is read at all (review
    R3-1). The import downgrade and every reader that interprets a stamp
    (:func:`read_jsonl_rows`) go through it, so they cannot disagree on line
    boundaries or decoding again. Before it, the downgrade split on ``\\n`` and
    decoded strictly while ``asks.store.read_events`` split with
    ``bytes.splitlines()`` (also on ``\\r``) and decoded with ``"replace"``: a
    line ``b"garbage\\r" + <verified answer>`` or one invalid UTF-8 byte made the
    downgrade skip a line the ask fold then read as ``verified: true``.

    Strict means: the writer's own framing (``\\n`` only — a ``\\r`` anywhere is
    refused, since a ``splitlines`` reader would cut the line there), strict
    UTF-8, valid JSON, and decodable at this depth. Raises
    :class:`StampLineRejected` otherwise; the message names the reason.

    Not covered, deliberately: ``str.splitlines`` also cuts on U+0085/U+2028/
    U+2029 (legal raw inside a JSON string), but no stamp reader splits that way —
    the ones that do (census, peek, preview) never interpret a stamp.
    """
    if b"\r" in line:
        raise StampLineRejected("carries a carriage return")
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise StampLineRejected("is not strict UTF-8") from exc
    try:
        return json.loads(text)
    except RecursionError as exc:
        # The decoder's depth limit follows the CALLER's stack depth (a C
        # recursion counter on 3.12, the stack pointer on 3.14; review R2-1), so a
        # line too deep here may still decode for a shallower reader.
        raise StampLineRejected("is too deep to lower") from exc
    except ValueError as exc:
        raise StampLineRejected("is not valid JSON") from exc


def read_jsonl_rows(raw: bytes) -> list[Any]:
    """Every JSON value in a JSONL file, for a reader that INTERPRETS stamps.

    Tolerant for ordinary content exactly as the readers were before (rows split
    with ``bytes.splitlines()``, decoded with ``"replace"``, torn rows skipped),
    but a line that may carry a stamp is read through :func:`parse_stamp_line`.
    If the strict parser rejects it, its rows are still read tolerantly — so a
    torn line costs no ordinary content — with every stamp in them forced to
    ``verified: false``: a stamp from a line the import door would have refused
    is never verified. Users: ``asks.store.read_events``,
    ``session.runtime.inbox._parse``.
    """
    rows: list[Any] = []
    for line in raw.split(b"\n"):
        lowered = False
        if line_may_carry_stamp(line):
            try:
                rows.append(parse_stamp_line(line))
                continue
            except StampLineRejected:
                lowered = True
        for piece in line.splitlines():
            if not piece.strip():
                continue
            try:
                payload = json.loads(piece.decode("utf-8", "replace"))
            except (ValueError, RecursionError):
                continue  # a torn line (killed mid-write) is skipped, never fatal
            if lowered:
                _downgrade_value(payload)
            rows.append(payload)
    return rows


#: The transcript's ARCHIVE forms, imported from the module that owns them rather
#: than spelled here. A local spelling put the literal transcript name into a
#: ``network/`` module, which ``test_relay_e2e.test_the_relay_writes_no_session_state``
#: forbids outside the move's own modules; importing it is also the only spelling
#: that cannot drift. The import is cheap and safe for the relay process:
#: ``session.compression`` imports only the stdlib at module scope (its
#: ``local_operator`` imports are all function-local), measured at ~3 ms over an
#: already-imported ``network.origin``. Only the NAME list is used here — nothing
#: in this module decompresses (see :func:`downgrade_stamps`).
TRANSCRIPT_ARCHIVE_NAMES: tuple[str, ...] = ARCHIVE_FILENAMES


class ImportStampsRefused(ValueError):
    """An imported session carries something the downgrade cannot lower.

    Raised instead of skipping it, so both import doors refuse the session rather
    than land a stamp that still reads ``verified: true`` (security review S-2).
    """


def _downgrade_value(value: Any) -> bool:
    """Set ``verified=False`` on every stamp nested in ``value``; True if changed.

    ITERATIVE ON PURPOSE (security review S-1). ``json.loads``'s C decoder accepts
    nesting far deeper than Python's recursion limit (1,500 and 3,000 levels both
    parse at the default limit of 1,000), so a recursive walk raised
    ``RecursionError`` on a row a hostile member wrote, and the import door
    aborted with the stamp still verified. An explicit stack has no such ceiling;
    its size is bounded by the decoded value, which is already in memory.
    """
    changed = False
    stack: list[Any] = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, item in current.items():
                if key in FRAME_ORIGIN_KEYS and isinstance(item, dict):
                    if item.get("verified") is not False:
                        item["verified"] = False
                        changed = True
                if isinstance(item, (dict, list)):
                    stack.append(item)
        elif isinstance(current, list):
            stack.extend(item for item in current if isinstance(item, (dict, list)))
    return changed


def _open_regular_file(path: Path) -> tuple[bytes, os.stat_result]:
    """Read ``path`` only if it is a regular file reached without a symlink.

    WHY (guardrails bar, PR #4169): ``_downgrade_file`` ends in an ``os.replace``
    onto ``path``, and the session-deletion guard allow-lists that call only
    because its target is provably the same regular FILE this call read, never a
    directory and never something a symlink points at. ``O_NOFOLLOW`` refuses a
    symlink at open where the platform has it; the ``lstat`` covers the rest; the
    ``fstat`` of the open descriptor is the identity the replace is checked
    against. Any mismatch raises :class:`ImportStampsRefused`, which every import
    door treats as a refusal.
    """
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise ImportStampsRefused(f"{path.name}: cannot be read ({exc})") from exc
    if not stat.S_ISREG(before.st_mode):
        raise ImportStampsRefused(f"{path.name}: not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ImportStampsRefused(f"{path.name}: cannot be opened ({exc})") from exc
    with os.fdopen(fd, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise ImportStampsRefused(f"{path.name}: changed while it was being opened")
        return handle.read(), opened


def _downgrade_file(path: Path) -> int:
    raw, opened = _open_regular_file(path)
    if not line_may_carry_stamp(raw):
        return 0
    lines = raw.split(b"\n")
    changed = 0
    for index, line in enumerate(lines):
        if not line_may_carry_stamp(line):
            # No stamp key can be in it, whatever a reader's decoder: left
            # byte-identical, torn or not.
            continue
        try:
            payload = parse_stamp_line(line)
        except StampLineRejected as exc:
            # FAIL CLOSED (reviews R2-1, R3-1). Any reason the ONE strict parser
            # gives (depth, ``\r``, invalid UTF-8, bad JSON) refuses the import:
            # skipping it is what let a lenient reader find ``verified: true``.
            # Readers lower such a line's stamps too (``read_jsonl_rows``), but
            # the door does not lean on that.
            raise ImportStampsRefused(f"{path.name}: a stamped line {exc}") from exc
        if _downgrade_value(payload):
            try:
                lines[index] = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            except RecursionError as exc:
                # The encoder's depth limit is close to, but not provably at or
                # above, the decoder's. A stamp we lowered but cannot write back
                # must refuse the import, never leave the original line in place.
                raise ImportStampsRefused(f"{path.name}: a stamped line is too deep") from exc
            changed += 1
    if changed:
        fd, staged = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".mo")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(b"\n".join(lines))
            os.chmod(staged, opened.st_mode & 0o777)
            # Re-check, immediately before the swap, that ``path`` is still the
            # very regular file read above (same device and inode, not a symlink):
            # the guard's allow-list row rests on this. ``staged`` is a sibling
            # (``mkstemp(dir=path.parent)``), so the swap never crosses a directory.
            current = os.lstat(path)
            if not stat.S_ISREG(current.st_mode) or (current.st_dev, current.st_ino) != (
                opened.st_dev,
                opened.st_ino,
            ):
                raise ImportStampsRefused(f"{path.name}: replaced while it was being rewritten")
            os.replace(staged, path)
        except BaseException:
            Path(staged).unlink(missing_ok=True)
            raise
    return changed


def downgrade_stamps(directory: Path) -> int:
    """Set EVERY stamp in an imported session directory to ``verified=false``.

    Called by the two import doors — the move's ``_promote`` (on the staging copy,
    before its one ``os.replace``) and ``sync.promote_replica`` — WHATEVER the
    stamp's ``stamped_by`` (R2-1): a session shed A→D and recalled D→A can carry
    rows D wrote naming ``stamped_by=A``, and nothing at rest is authenticated, so
    anything that arrived from another device is evidence only. Original fields
    are kept for audit; only ``verified`` changes.

    Every ``*.jsonl`` under the directory (transcript, ask log, inbox spool,
    child journals) is scanned; symlinks are never followed. Raises ``OSError``
    on a write failure so the caller can refuse the import rather than land a
    session whose stamps were not lowered (fail-closed). Returns the number of
    lines rewritten.

    A transcript ARCHIVE (any name in :data:`TRANSCRIPT_ARCHIVE_NAMES`, any depth) raises
    :class:`ImportStampsRefused` before anything is rewritten (security review
    S-2): readers resolve the archive when the plain file is absent
    (``compression.transcript_source``), so skipping it would land its stamps
    verified. Refusing is the smaller safe option than decompressing here: an
    honest sender restores the archive before it plans the copy (see the note on
    ``sync.COPY_SET_NAMES``), so only a hand-crafted plan ever carries one, and
    in-place decompression would pull the zstd codec and the compression
    module's write protocol into both import doors for a case no honest peer
    produces. Callers must treat ANY exception as a refusal.
    """
    for current, dirnames, filenames in os.walk(directory, followlinks=False):
        dirnames[:] = [name for name in dirnames if not Path(current, name).is_symlink()]
        for name in filenames:
            if name in TRANSCRIPT_ARCHIVE_NAMES:
                raise ImportStampsRefused(
                    f"{Path(current, name).relative_to(directory)}: a compressed transcript "
                    "cannot have its origin stamps lowered"
                )
    total = 0
    for current, dirnames, filenames in os.walk(directory, followlinks=False):
        dirnames[:] = [name for name in dirnames if not Path(current, name).is_symlink()]
        for name in filenames:
            if not name.endswith(".jsonl"):
                continue
            path = Path(current, name)
            if path.is_symlink() or not path.is_file():
                continue
            total += _downgrade_file(path)
    if total:
        logger.info("mesh origin: downgraded %d imported stamp line(s) in %s", total, directory)
    return total


# ---------------------------------------------------------------------------
# Consent-time re-derivation (design §2.1)
# ---------------------------------------------------------------------------


def mesh_origin_counts(
    stamp: Mapping[str, Any] | None,
    *,
    _root: Path | None = None,
    _identity: Any = None,
) -> bool:
    """True iff a delivery stamp still counts as an authenticated mesh device NOW.

    The stored ``verified`` is evidence, not a verdict: this re-judges it against
    THIS runtime's own config root and identity (both internal — never from the
    stamp, the session or moved data, R2-2/R2-3) and the live network record.
    Any exception reads ``False`` (M-7). True only if: ``network_id`` is one safe
    path component (checked BEFORE any path is built, R2-2); the record loads;
    ``verified is True``; ``stamped_by`` is this device (identity rotation therefore
    voids every own stamp, fail-closed); the network is trusted; the member row's
    OWN ``device_id`` equals the origin (``member()`` also matches
    ``previous_ids``, M-7) and is active; and the id is not in ``removed_ids``.
    ``record.epoch`` is deliberately not gated: an unrelated rotation must not void
    an active member's rows.

    ``_root``/``_identity`` are TEST-ONLY seams (deferred NIT R3-1): private,
    keyword-only, and never supplied by production code — a test greps for it.
    Consumer: the guard, in a later change; this release has no caller.
    """
    try:
        if not isinstance(stamp, Mapping):
            return False
        if stamp.get("verified") is not True:
            return False
        from local_operator.network import store

        network_id = stamp.get("network_id")
        store.safe_component(network_id)
        origin_device_id = stamp.get("origin_device_id")
        if not isinstance(origin_device_id, str) or not origin_device_id:
            return False
        root = _root
        if root is None:
            from local_operator.paths import config_dir

            root = config_dir()
        identity = _identity
        if identity is None:
            from local_operator.network import identity as identity_module

            identity = identity_module.load(root)
        self_device_id = str(getattr(identity, "device_id", "") or "")
        if not self_device_id or stamp.get("stamped_by") != self_device_id:
            return False
        record = store.load(str(network_id), root)
        if record.trust != "active":
            return False
        row = record.member(origin_device_id)
        if row is None or row.device_id != origin_device_id or not row.active:
            return False
        if origin_device_id in (record.removed_ids or ()):
            return False
        return True
    except Exception:  # noqa: BLE001 — fail-closed by contract (M-7)
        logger.debug("mesh origin: a stamp did not count", exc_info=True)
        return False
