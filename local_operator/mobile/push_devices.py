"""The push device registry: the phones this computer knows to push to.

Push/ack-sync S4 of ADR 0006 (``damianvtran/local-operator-mobile`` PR #14 @
22e2cce2, §3.1/§4). Three relay routes read and write this store — ``POST
/api/push/register``, ``GET /api/push/devices``, ``DELETE
/api/push/devices/{device_id}`` — and nothing else does; the store's own rules
(field validation, the idempotent upsert, atomic private writes) live here so
the daemon's handlers stay request-shaped, the way ``mobile_projects`` splits
it.

Two decisions look odd until the reasons are on the table:

- **No push token is ever stored.** The register request carries one because
  the request contract is the app's, but the machine keeps only what a Settings
  list renders. ADR §4's recommended split is device registry in the cloud —
  the machine holds no token at all — and §3.1 puts registration on the relay
  because the relay is the only party with a credential on both sides. The
  forward step (S7) does not exist yet; until it does, the token is validated
  and dropped, and ``token`` must appear NOWHERE in a stored record (a test
  asserts the string never reaches the file).
- **The register payload is DECLARATIVE.** Re-registering for the same
  ``(install_id, platform)`` REPLACES the record's metadata — ``environment``,
  ``app_version``, and the optional ``name`` label (an omitted ``name`` clears
  a stored one, so the app owns its label's whole lifecycle) — and keeps the
  ``device_id``, so a device that rotates its push token nightly cannot
  accumulate rows.

The store is one JSON object under the config root, beside the daemon's other
owner-private state (``mobile-seen.json``), written 0600 and atomically: temp
file in the same directory, chmod before the replace, the same discipline as
``SeenStore._persist_locked``.

A store that cannot be parsed or fails validation is REFUSED, not repaired:
every operation raises :class:`PushRegistryCorrupt` and the file is left
byte-for-byte alone. The tempting alternatives — starting fresh, dropping the
bad record, rewriting it into shape — are all silent data loss for the devices
it names, and a caller has to be able to tell "no devices" from "cannot read
the devices". The registry is self-contained: it never reads or writes the
attention store (``attention.db``) and is never an authority for unread state.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

#: The store file, directly under ``config_dir()`` beside the other
#: owner-private state (``mobile-seen.json``).
PUSH_DEVICES_STORE_NAME = "mobile-push-devices.json"

#: The enums the register contract allows (ADR §3.1).
PLATFORMS = ("ios", "android")
ENVIRONMENTS = ("sandbox", "production")

#: Bound on every free-text field, the token included. Generous on purpose:
#: the token is opaque, and an over-tight bound would refuse a future
#: platform's longer token. The point is only that a malformed caller cannot
#: grow the store without limit.
MAX_FIELD_CHARS = 1024

#: One lock over the store's read-modify-write cycle. The routes run their
#: store calls on worker threads (``asyncio.to_thread``), so two concurrent
#: registrations could otherwise interleave load/save and lose one. In-process
#: is sufficient and deliberate: exactly one daemon process writes this store
#: (a restart's predecessor is gone before its successor serves requests).
_LOCK = threading.Lock()

#: The record's canonical shape. ``name`` is the only optional field (the
#: register body does not define it — see the module docstring and the PR
#: notes — so pre-name records and nameless devices both round-trip).
_REQUIRED_RECORD_FIELDS = frozenset(
    {
        "device_id",
        "platform",
        "environment",
        "app_version",
        "install_id",
        "registered_at",
        "last_seen_at",
    }
)
_OPTIONAL_RECORD_FIELDS = frozenset({"name"})
_RECORD_FIELDS = _REQUIRED_RECORD_FIELDS | _OPTIONAL_RECORD_FIELDS


class PushDeviceRefusal(Exception):
    """One register payload this build cannot accept, written for the reader.

    ``message`` is the sentence the daemon's 422 JSON body carries — the same
    refusal shape ``api_session_seen`` answers a bad ``completion_token`` with.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class PushRegistryCorrupt(RuntimeError):
    """The stored registry cannot be used, and must not be rewritten.

    Raised for an unreadable file, invalid JSON, or a record this build cannot
    round-trip. Every registry operation refuses while it stands and the file
    is left untouched by the raiser; the daemon answers it as an internal fault
    (500) whose message names the problem.
    """


def register(config_dir: Path, body: object, *, now: float | None = None) -> dict[str, Any]:
    """``POST /api/push/register`` — record one device, idempotent on identity.

    The upsert IS the idempotency (module docstring): a re-register with a
    rotated token keeps ``device_id`` and ``registered_at`` — that is what "the
    same device" means — replaces the declarative metadata, and bumps
    ``last_seen_at`` to now. The response carries the RECORD's ``registered_at``
    rather than the request's clock: the app re-registers on every launch and a
    value that moved under it would make "registered" un-anchorable.

    ``token`` is validated and deliberately dropped. No cloud call happens here
    — the forward is future work (S7) — and when it lands the machine still
    stores nothing, per ADR §4: the interface is the register contract, and its
    custody is the cloud's.
    """
    fields = _checked_registration(body)
    with _LOCK:
        records = _load(config_dir)
        stamp = int(time.time() if now is None else now)
        record = _find(records, fields["install_id"], fields["platform"])
        if record is None:
            record = {
                "device_id": uuid.uuid4().hex,
                "platform": fields["platform"],
                "environment": fields["environment"],
                "app_version": fields["app_version"],
                "install_id": fields["install_id"],
                "registered_at": stamp,
                "last_seen_at": stamp,
            }
            records.append(record)
        else:
            record["environment"] = fields["environment"]
            record["app_version"] = fields["app_version"]
            record["last_seen_at"] = stamp
        if fields["name"] is not None:
            record["name"] = fields["name"]
        else:
            record.pop("name", None)
        _save(config_dir, records)
        return {
            "ok": True,
            "device_id": record["device_id"],
            "registered_at": record["registered_at"],
        }


def list_devices(config_dir: Path) -> dict[str, Any]:
    """``GET /api/push/devices`` — the Settings list.

    Exactly the fields the ADR's shape names, ``environment`` and
    ``install_id`` excluded on purpose: the phone renders them nowhere, and the
    wire is a contract, not the store's dump. ``name`` is omitted when unknown
    (the repo's absence rule — a null would be a client-visible claim it must
    special-case; absence is the truth).

    Registration order, and a re-register updates in place, so a device's
    position does not move under a token rotation. READ-ONLY: nothing is
    written, so opening Settings cannot bump anything.
    """
    with _LOCK:
        records = _load(config_dir)
    devices: list[dict[str, Any]] = []
    for record in records:
        entry: dict[str, Any] = {
            "device_id": record["device_id"],
            "platform": record["platform"],
            "app_version": record["app_version"],
            "registered_at": record["registered_at"],
            "last_seen_at": record["last_seen_at"],
        }
        if "name" in record:
            entry["name"] = record["name"]
        devices.append(entry)
    return {"devices": devices}


def deregister(config_dir: Path, device_id: str) -> dict[str, Any]:
    """``DELETE /api/push/devices/{device_id}`` — remove one device, idempotently.

    An id the registry does not hold (or no longer holds) stays ``{"ok": true}``
    and nothing is written: the app retries this on sign-out, and a retry after
    a successful delete must not read as a failure. Deliberately NOT scoped to
    "the caller's own device" — the relay's cookie is one operator for the
    whole computer, and the stolen-phone case (ADR §4) needs one device to be
    able to revoke another.

    Refuses (raises) on a store it cannot read, even when the id is absent:
    "not in a store I cannot read" is not an answer this store is willing to
    give.
    """
    with _LOCK:
        records = _load(config_dir)
        remaining = [record for record in records if record["device_id"] != device_id]
        if len(remaining) != len(records):
            _save(config_dir, remaining)
    return {"ok": True}


# -- payload validation ---------------------------------------------------------


def _checked_registration(body: object) -> dict[str, Any]:
    """One register body, field by field, strictly — or a refusal sentence."""
    if not isinstance(body, dict):
        raise PushDeviceRefusal("a JSON object body is required")
    platform = body.get("platform")
    if platform not in PLATFORMS:
        raise PushDeviceRefusal('platform must be "ios" or "android"')
    environment = body.get("environment")
    if environment not in ENVIRONMENTS:
        raise PushDeviceRefusal('environment must be "sandbox" or "production"')
    # Validated, then dropped — never persisted (module docstring, ADR §4).
    token = body.get("token")
    if not _is_text(token):
        raise PushDeviceRefusal("token is required")
    app_version = _checked_text(body.get("app_version"), "app_version")
    install_id = _checked_text(body.get("install_id"), "install_id")
    # ``name`` is device-local metadata only (a user-editable label): never used
    # for routing or matching, and it must never drift into carrying
    # conversation/machine session content. Blank means "no label given", which
    # the store records as absent.
    name = body.get("name")
    if name is None or (isinstance(name, str) and not name.strip()):
        name = None
    elif isinstance(name, str):
        name = _bounded(name.strip(), "name")
    else:
        raise PushDeviceRefusal("name must be a string when provided")
    return {
        "platform": platform,
        "environment": environment,
        "app_version": app_version,
        "install_id": install_id,
        "name": name,
    }


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= MAX_FIELD_CHARS


def _checked_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PushDeviceRefusal(f"{field} is required")
    return _bounded(value.strip(), field)


def _bounded(value: str, field: str) -> str:
    if len(value) > MAX_FIELD_CHARS:
        raise PushDeviceRefusal(f"{field} is too long")
    return value


def _find(records: list[dict[str, Any]], install_id: str, platform: str) -> dict[str, Any] | None:
    """The record this registration identifies, if any — the idempotency key."""
    for record in records:
        if record["install_id"] == install_id and record["platform"] == platform:
            return record
    return None


# -- the file -------------------------------------------------------------------


def _load(config_dir: Path) -> list[dict[str, Any]]:
    """The stored records, validated; ``[]`` when the store does not exist yet.

    Everything else — an unreadable file, JSON that does not parse, a shape
    this build does not know — refuses with :class:`PushRegistryCorrupt` rather
    than degrading, per the module docstring. There is no "best effort" read:
    an empty answer for a file that exists and cannot be read is the one answer
    that silently loses devices.
    """
    path = config_dir / PUSH_DEVICES_STORE_NAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise PushRegistryCorrupt(
            f"push device registry cannot be read ({exc}); refusing to treat it as empty"
        ) from exc
    if not isinstance(raw, dict) or set(raw) != {"devices"} or not isinstance(raw["devices"], list):
        raise PushRegistryCorrupt(
            "push device registry has an unexpected shape; refusing to rewrite it"
        )
    return [_validated_record(entry, index) for index, entry in enumerate(raw["devices"])]


def _validated_record(entry: object, index: int) -> dict[str, Any]:
    """One stored record, validated strictly in both directions.

    A record MISSING a required field cannot answer idempotency; a record
    carrying a field this build does not know would be silently DROPPED by the
    next save — the silent repair this store refuses. So the canonical shape is
    the only shape that round-trips, and anything else refuses the store.
    """
    if not isinstance(entry, dict):
        raise PushRegistryCorrupt(
            f"push device registry record {index} is not an object; refusing to rewrite it"
        )
    unknown = sorted(set(entry) - _RECORD_FIELDS)
    missing = sorted(_REQUIRED_RECORD_FIELDS - set(entry))
    if unknown or missing:
        raise PushRegistryCorrupt(
            f"push device registry record {index} is not a record this build wrote"
            f" (missing {missing}, unknown {unknown}); refusing to rewrite it"
        )
    if entry["platform"] not in PLATFORMS:
        raise PushRegistryCorrupt(
            f"push device registry record {index} has an invalid platform; refusing to rewrite it"
        )
    if entry["environment"] not in ENVIRONMENTS:
        raise PushRegistryCorrupt(
            f"push device registry record {index} has an invalid environment;"
            " refusing to rewrite it"
        )
    for field in ("device_id", "app_version", "install_id", "name"):
        if field not in entry:
            continue
        value = entry[field]
        if not isinstance(value, str) or not value or len(value) > MAX_FIELD_CHARS:
            raise PushRegistryCorrupt(
                f"push device registry record {index} has an invalid {field};"
                " refusing to rewrite it"
            )
    for field in ("registered_at", "last_seen_at"):
        value = entry[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise PushRegistryCorrupt(
                f"push device registry record {index} has an invalid {field};"
                " refusing to rewrite it"
            )
    return dict(entry)


def _save(config_dir: Path, records: list[dict[str, Any]]) -> None:
    """Atomic 0600 write: temp file in the same directory, then replace.

    The replace guarantees a reader sees the old file or the new one, never a
    half-written one, and the chmod lands BEFORE the replace so the store is
    never briefly world-readable. Failures propagate: unlike the seen store,
    whose verdicts stay correct in memory, a register/deregister verdict must
    not be reported as accepted when the disk never got it.
    """
    path = config_dir / PUSH_DEVICES_STORE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{PUSH_DEVICES_STORE_NAME}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"devices": records}, handle, separators=(",", ":"))
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
