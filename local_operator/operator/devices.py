"""Paired devices: the operator-signed certificate that makes a phone a signer.

WHY THIS IS A FILE AND NOT A KEYCHAIN ITEM, and why it is NOT the anchor. The
anchor (:mod:`local_operator.operator.trust`) is root-owned because its JOB is to
be un-substitutable — a same-uid process that could point the runtime at its own
key would make every claim in this package false. A device certificate has the
opposite requirement: the runtime has to be able to read it per session, the
relay has to be able to read it per dial, and a paired phone must be able to
revoke/refresh without a privileged step. So it lives under the operator's own
config root, where the same-uid subject CAN write it.

THE SUBSTITUTION IS DEFEATED BY THE SIGNATURE, NOT BY THE PATH. A certificate is
an operator-signed statement over ``(device_id, spki, label, issued_at,
not_after)``; :func:`local_operator.operator.verify.verify_device_cert` checks
that signature against the ANCHORED operator key. A file written by a same-uid
attacker therefore verifies only if they can forge an ES256 signature under the
operator's key — at which point they did not need the file. That is the whole
argument, and it is pinned by a test
(``tests/unit/operator/test_operator_devices.py::test_a_substituted_certificate_fails_verification``).

WHAT IS *NOT* DEFENDED, stated rather than implied (design §4 residual): this is
an AVAILABILITY surface, not a boundary. A same-uid subject can DELETE a device
certificate (the phone stops being able to sign until the operator pairs it
again) and can write as many garbage ones as it likes (each is refused, but the
directory grows). Both are denials, not escalations: neither produces a
signature the runtime will accept. The revocation list has the opposite
requirement from the certificate for exactly this reason — it lives in the
ROOT-OWNED anchor (``OperatorAnchor.devices``, written by
``lop operator revoke`` through the same privileged install step as the anchor
itself), so a device the operator has revoked cannot be un-revoked by the
subject it was revoked from.

THE PAIRING HANDSHAKE lives here too, because every step of it is a fact about
this directory:

1. ``lop pair`` mints a short-lived CODE (0600, config root) and prints it;
2. the phone generates its own ES256 key in WebCrypto with
   ``extractable: false`` — the private half is never exported, so nothing on
   this machine can sign as the phone — and POSTs ``{code, spki, name}`` to the
   relay;
3. the relay checks the code against the live one and drops a PENDING request
   (0600) naming the device's public point. It cannot do more than that: it has
   no operator key and no way to obtain a signature;
4. ``lop pair`` (still watching) sees the pending request, runs the ONE
   presence-gated signing entry point, and installs the certificate;
5. the phone polls, reads its certificate, and stores it locally.

Step 4 is the only step that needs the operator key, and it is reachable ONLY
from a local process with the presence store — which is why the relay in step 3
is a courier rather than an authority.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import secrets
import stat
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_operator.operator.verify import DeviceCert, key_id_for

#: The certificate record's own format version, mirroring ``ANCHOR_VERSION``.
DEVICE_RECORD_VERSION = 1

#: What a paired device may do. Both directions of authority-increasing request,
#: and nothing else: reading and the ordinary/tightening ops need no certificate
#: at all (they ride the record key), so a scope beyond these two would be a
#: capability nobody asked for.
DEVICE_SCOPES = ("loosen", "approve")

#: How long a pairing code stays live. Short on purpose: the code is the only
#: thing standing between a stranger on the tunnel and a request the operator
#: might approve by reflex, and the honest operator is holding the phone while
#: ``lop pair`` is running.
PAIRING_TTL_S = 600

# The certificate's LIFETIME is deliberately not a constant here. It belongs to
# ``sign.issue_device_cert``, which is the one place that builds the statement,
# and a second default in this module would be a second answer to "how long is a
# paired phone good for". The store reads ``exp`` off the certificate it was
# handed instead.


def operator_root(config_root: Path) -> Path:
    """The operator's private working area under the config root."""
    return Path(config_root) / "operator"


def devices_dir(config_root: Path) -> Path:
    """Where paired devices' certificates live. 0700, created on demand."""
    return operator_root(config_root) / "devices"


def device_path(config_root: Path, device_id: str) -> Path:
    """One file per device, named by the DERIVED id (see :func:`new_device_id`).

    The name is validated rather than used raw: ``device_id`` arrives from a
    pairing request, and a value containing ``..`` or a separator would let a
    request write outside this directory.
    """
    return devices_dir(config_root) / f"{_safe_name(device_id)}.json"


def pairing_path(config_root: Path) -> Path:
    """The live pairing code, 0600 inside the operator root."""
    return operator_root(config_root) / "pairing.json"


def pending_dir(config_root: Path) -> Path:
    """Pairing requests a device has posted and the operator has not answered."""
    return operator_root(config_root) / "pending"


def pending_path(config_root: Path, device_id: str) -> Path:
    return pending_dir(config_root) / f"{_safe_name(device_id)}.json"


def new_device_id(spki: bytes) -> str:
    """A device's id IS the id of the key it proves. Derived, never chosen.

    Chosen ids would need a registry and a uniqueness rule; a derived id needs
    neither, and it makes the revocation list self-describing — an operator
    reading ``lop operator devices`` sees the same string the certificate carries.
    """
    return key_id_for(spki)


def encode_spki(spki: bytes) -> str:
    """The wire form of a public point: unpadded url-safe base64.

    Shared by the store and the relay so the pairing request the phone POSTs and
    the record written for it cannot disagree about the encoding of the same key.
    """
    return base64.urlsafe_b64encode(spki).decode("ascii").rstrip("=")


def decode_spki(encoded: object) -> bytes | None:
    """A public point from its wire form, or ``None``. P-256 uncompressed only.

    Bounded and validated HERE, at the first boundary the value crosses, rather
    than at the point it is signed: a malformed key must fail before the operator
    is asked for a presence gesture, because spending that gesture on a request
    this build cannot read is a gesture wasted on a refusal.
    """
    if not isinstance(encoded, str) or not encoded or len(encoded) > 512:
        return None
    padded = encoded + "=" * (-len(encoded) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
    except (ValueError, binascii.Error):
        return None
    from local_operator.operator.verify import decode_point

    return raw if decode_point(raw) is not None else None


def _safe_name(value: str) -> str:
    """A filesystem-safe stem for an id that arrived over the wire."""
    cleaned = "".join(
        character for character in str(value) if character.isalnum() or character in "-_"
    )
    if not cleaned or cleaned != str(value) or len(cleaned) > 128:
        raise ValueError("device id must be a short alphanumeric string")
    return cleaned


def _write_private(path: Path, data: bytes) -> None:
    """Write 0600, creating the parent 0700. Used for every request-side file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    # O_NOFOLLOW where the platform has it: a symlink planted in the operator
    # root would otherwise redirect this write anywhere the uid can reach.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, data)
    finally:
        os.close(descriptor)
    os.chmod(path, 0o600)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return body if isinstance(body, dict) else None


# ---------------------------------------------------------------------------
# The pairing handshake
# ---------------------------------------------------------------------------


def begin_pairing(
    config_root: Path, *, ttl_s: int = PAIRING_TTL_S, now: float | None = None
) -> str:
    """Mint a fresh pairing code and publish it. Returns the code.

    ``token_hex(16)`` — 32 characters the operator reads off the screen. Eight
    bytes would be 16 characters and still far beyond guessing inside a ten
    minute window, but the code is typed by a human, so the length is chosen for
    transcription rather than for entropy: it is upper- and lower-case hex, which
    has no visually ambiguous pair.
    """
    stamp = time.time() if now is None else now
    code = secrets.token_hex(16)
    _write_private(
        pairing_path(config_root),
        json.dumps(
            {"v": 1, "code": code, "issued_at": int(stamp), "expires_at": int(stamp + ttl_s)}
        ).encode("utf-8"),
    )
    return code


def read_pairing(config_root: Path, *, now: float | None = None) -> str | None:
    """The live pairing code, or ``None`` when there is none or it has expired.

    Expiry is checked on read rather than by deleting on a timer: nothing here
    runs in the background, and a stale file that answers ``None`` is the same
    outcome as no file.
    """
    body = _read_json(pairing_path(config_root))
    if body is None:
        return None
    code = body.get("code")
    expires_at = body.get("expires_at")
    if not isinstance(code, str) or not code:
        return None
    if not isinstance(expires_at, int):
        return None
    stamp = time.time() if now is None else now
    if stamp > expires_at:
        return None
    return code


def clear_pairing(config_root: Path) -> None:
    """Burn the code. Called as soon as a request is accepted, so one code is
    one device — a code that stayed live would let a second phone pair on a
    gesture the operator already made for the first."""
    try:
        pairing_path(config_root).unlink()
    except OSError:
        pass


def write_pending(
    config_root: Path,
    *,
    device_id: str,
    name: str,
    spki: str,
    code: str,
    now: float | None = None,
) -> Path:
    """Record a device's pairing request for the operator to approve.

    The CODE is not stored here: the caller has already checked it against
    :func:`read_pairing`, and writing a second copy would be a second thing to
    expire and a second thing to read out of the file.
    """
    stamp = int(time.time() if now is None else now)
    path = pending_path(config_root, device_id)
    _write_private(
        path,
        json.dumps(
            {"v": 1, "device_id": device_id, "name": name, "spki": spki, "requested_at": stamp}
        ).encode("utf-8"),
    )
    return path


def read_pending(config_root: Path, device_id: str) -> dict[str, Any] | None:
    return _read_json(pending_path(config_root, device_id))


def list_pending(config_root: Path) -> list[dict[str, Any]]:
    """Every unanswered request, oldest first.

    Names come from the filesystem and are re-validated on the way out: a file
    planted directly in the directory is not a request that was posted through
    :func:`write_pending`, and must not become one by being listed.
    """
    rows: list[dict[str, Any]] = []
    try:
        entries = sorted(pending_dir(config_root).iterdir())
    except OSError:
        return rows
    for entry in entries:
        if entry.suffix != ".json":
            continue
        body = _read_json(entry)
        if body is None:
            continue
        device_id = body.get("device_id")
        spki = body.get("spki")
        name = body.get("name")
        if not isinstance(device_id, str) or not isinstance(spki, str) or not isinstance(name, str):
            continue
        if entry.stem != device_id:
            # The filename and the statement disagree. Refused rather than
            # reconciled: the operator is about to sign the statement, and
            # guessing which half is the device's intent is not this code's job.
            continue
        rows.append(body)
    rows.sort(key=lambda row: int(row.get("requested_at") or 0))
    return rows


def drop_pending(config_root: Path, device_id: str) -> None:
    try:
        pending_path(config_root, device_id).unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# The certificate store
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StoredDevice:
    """One row of the store: the wire certificate plus the operator's metadata."""

    device_id: str
    name: str
    spki: bytes
    key_id: str
    scope: tuple[str, ...]
    issued_at: int
    not_after: int
    operator_key_id: str
    certificate: str

    def as_json(self) -> dict[str, Any]:
        """The on-disk record. ``certificate`` is the exact string the phone
        presents, kept whole so the machine and the device cannot disagree
        about what was signed — a re-encoding from fields would be a second,
        subtly different statement."""
        return {
            "v": DEVICE_RECORD_VERSION,
            "kind": "device",
            "device_id": self.device_id,
            "name": self.name,
            "spki": base64.urlsafe_b64encode(self.spki).decode("ascii").rstrip("="),
            "key_id": self.key_id,
            "scope": list(self.scope),
            "iat": self.issued_at,
            "exp": self.not_after,
            "operator_key_id": self.operator_key_id,
            "certificate": self.certificate,
        }


def write_device_cert(
    config_root: Path,
    *,
    certificate: str,
    parsed: DeviceCert,
    operator_key_id: str,
    name: str = "",
    scope: tuple[str, ...] = DEVICE_SCOPES,
) -> StoredDevice:
    """Install a paired device's certificate. 0644 under a 0700 directory.

    Readable by anything running as the operator ON PURPOSE, and the mode says
    so: the relay and every runtime on this machine read it, and it holds
    PUBLIC data only — the point is that reading it is worthless without the
    operator's signature on it and the phone's private half, neither of which is
    in this file.
    """
    device_id = new_device_id(parsed.spki)
    stored = StoredDevice(
        device_id=device_id,
        name=name or parsed.label,
        spki=parsed.spki,
        key_id=key_id_for(parsed.spki),
        scope=tuple(scope),
        issued_at=parsed.issued_at,
        not_after=parsed.not_after,
        operator_key_id=operator_key_id,
        certificate=certificate,
    )
    path = device_path(config_root, device_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    path.write_text(json.dumps(stored.as_json(), indent=2) + "\n", encoding="utf-8")
    os.chmod(path, 0o644)
    return stored


def read_device(config_root: Path, device_id: str) -> StoredDevice | None:
    """One stored device, or ``None`` when absent or malformed."""
    body = _read_json(device_path(config_root, device_id))
    if body is None or body.get("kind") != "device" or body.get("v") != DEVICE_RECORD_VERSION:
        return None
    certificate = body.get("certificate")
    if not isinstance(certificate, str) or not certificate:
        return None
    raw = body.get("spki")
    if not isinstance(raw, str):
        return None
    padded = raw + "=" * (-len(raw) % 4)
    try:
        spki = base64.urlsafe_b64decode(padded.encode("ascii"))
    except (ValueError, binascii.Error):
        return None
    scope = body.get("scope")
    return StoredDevice(
        device_id=str(body.get("device_id") or ""),
        name=str(body.get("name") or ""),
        spki=spki,
        key_id=str(body.get("key_id") or ""),
        scope=tuple(str(item) for item in scope) if isinstance(scope, list) else (),
        issued_at=int(body.get("iat") or 0),
        not_after=int(body.get("exp") or 0),
        operator_key_id=str(body.get("operator_key_id") or ""),
        certificate=certificate,
    )


def list_devices(config_root: Path) -> list[StoredDevice]:
    """Every stored device, sorted by id so the listing is stable."""
    found: list[StoredDevice] = []
    try:
        entries = sorted(devices_dir(config_root).iterdir())
    except OSError:
        return found
    for entry in entries:
        if entry.suffix != ".json":
            continue
        device = read_device(config_root, entry.stem)
        if device is not None:
            found.append(device)
    return found


def paired_certificate(config_root: Path) -> str | None:
    """A certificate this machine can present, or ``None`` when nothing is paired.

    Used by the RELAY, and the reason it takes the first rather than a specific
    device is that the relay does not know which phone is on the other end of
    the tunnel: it is a courier, and any certificate it forwards is checked
    against the anchor by the runtime and against the phone's own key by the
    signature. Picking one is a routing choice, not an authority decision —
    which is exactly the property that lets the relay make it.
    """
    for device in list_devices(config_root):
        return device.certificate
    return None


def is_revoked(config_root: Path, device_id: str) -> bool:
    """Whether this device is revoked ANYWHERE — the record, or the anchor.

    THE TWO LISTS DISAGREEING WAS A REAL GAP (UX round 6, U5). ``revoked.json``
    under the config root is the relay's own belt-and-braces copy, written by
    ``lop operator devices --revoke``; the ROOT-OWNED ANCHOR is the authoritative
    list the runtime consults, and an operator who edits the anchor directly (the
    documented way to revoke a device whose certificate the relay still holds)
    leaves the local copy saying nothing. Measured: in exactly that state a
    revoked device re-paired on a fresh code (HTTP 200). So the guard asks BOTH,
    in the order that costs the least when the answer is yes.

    The anchor read is a plain file read here rather than the runtime's cached
    view: this is the setup path, on a human's action, not a frame.
    """
    if is_revoked_here(config_root, device_id):
        return True
    from local_operator.operator.trust import device_is_revoked, load_anchor

    loaded = load_anchor()
    if not loaded.usable or loaded.anchor is None:
        return False
    return device_is_revoked(loaded.anchor, device_id)


def is_revoked_here(config_root: Path, device_id: str) -> bool:
    """Whether a LOCALLY RECORDED revocation names this device.

    One half of :func:`is_revoked`, which is what a caller deciding whether to
    accept a device should use: the authoritative list is the root-owned anchor's
    (:func:`local_operator.operator.trust.device_is_revoked`, which the runtime
    consults), and this copy exists so the common case does not need a privileged
    read. Asking only this one was the gap ``is_revoked`` closes.
    """
    body = _read_json(operator_root(config_root) / "revoked.json")
    if body is None:
        return False
    listed = body.get("devices")
    return isinstance(listed, list) and device_id in listed


def record_revocation(config_root: Path, device_id: str) -> None:
    """Note a revocation in the operator root, and drop the certificate.

    Belt to the anchor's braces: the anchor is what the runtime checks, but the
    relay is what decides whether to forward a signature, and it can answer that
    question without a privileged read. Removing the certificate is what makes
    the refusal true even on a host whose anchor has not caught up.
    """
    path = operator_root(config_root) / "revoked.json"
    body = _read_json(path) or {"v": 1, "devices": []}
    listed = body.get("devices")
    if not isinstance(listed, list):
        listed = []
    if device_id not in listed:
        listed.append(device_id)
    body["devices"] = listed
    _write_private(path, json.dumps(body).encode("utf-8"))
    try:
        device_path(config_root, device_id).unlink()
    except OSError:
        pass


def device_certificate_mode(path: Path) -> int:
    """The stored certificate's mode, for the tests that pin 0644-under-0700."""
    return stat.S_IMODE(path.stat().st_mode)


__all__ = [
    "DEVICE_RECORD_VERSION",
    "DEVICE_SCOPES",
    "PAIRING_TTL_S",
    "StoredDevice",
    "begin_pairing",
    "clear_pairing",
    "decode_spki",
    "device_certificate_mode",
    "device_path",
    "devices_dir",
    "drop_pending",
    "encode_spki",
    "is_revoked",
    "is_revoked_here",
    "list_devices",
    "list_pending",
    "new_device_id",
    "operator_root",
    "paired_certificate",
    "pairing_path",
    "pending_dir",
    "pending_path",
    "read_device",
    "read_pairing",
    "read_pending",
    "record_revocation",
    "write_device_cert",
    "write_pending",
]
