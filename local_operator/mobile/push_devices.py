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
  accumulate rows. ``install_id`` is validated as a UUID (ADR §3.1) and stored
  in canonical form; matching is by PARSED identity, because a record may hold
  any spelling a writer stored (this store never rewrites on read) and one
  phone must not fork into two rows. An unknown body key refuses the request —
  the same strictness the stored records get.

The store is one JSON object under the config root, beside the daemon's other
owner-private state (``mobile-seen.json``), written 0600 and atomically: temp
file in the same directory, chmod before the replace, the same discipline as
``SeenStore._persist_locked``.

The registry is BOUNDED (``MAX_DEVICE_ENTRIES``, the same discipline as
``SeenStore._bound_locked``): overflow drops the least-recently-seen records,
so a caller-side bug that mints a fresh ``install_id`` per launch cannot grow
the store or the Settings payload without limit.

``device_id`` is machine-minted and **provisional**. ADR §3.1/§4 assign the id
to the cloud's registry ("the relay keeps the cloud's ``device_id``"), but the
S7 forward that would create that registry does not exist yet, so this build
mints a local ``uuid4``. DELETE acts on whatever id the registry currently
holds; when S7 lands, the cloud's id becomes authoritative and supersedes the
local one inside this module, and an app still holding a pre-S7 id keeps
getting the documented idempotent ``{"ok": true}``.

A store that cannot be parsed or fails validation is REFUSED, not repaired:
every operation raises :class:`PushRegistryCorrupt` and the file is left
byte-for-byte alone. The tempting alternatives — starting fresh, dropping the
bad record, rewriting it into shape — are all silent data loss for the devices
it names, and a caller has to be able to tell "no devices" from "cannot read
the devices". The registry is self-contained: it never reads or writes the
attention store (``attention.db``) and is never an authority for unread state.

The device LIFECYCLE (push/ack-sync S4a) lives here too, and it is one
vocabulary with one resolver. ADR §4 gives a device exactly one of five states —
**live**, **expired** (``expired_at``), **unpaired** (``unpaired_at``),
**revoked** (``revoked_at``) and **absent** (no row) — and the precedence
``revoked > unpaired > expired`` decides which one a row carrying several markers
is in. :func:`device_state` is that precedence, and the register route, ``list``
and the delivery gate the emit worker will read (S5/S4c) all call it, so no two
surfaces can read one row as two different states. ``PRECEDENCE`` is rendered
from the same table the resolver walks, so the sentence the wire carries cannot
drift away from the code that answers. **The emit worker reads a state and never
writes a marker** (ADR §4 rule 2): the relay's routes are the only writers. A
re-register is the one act that clears ``expired_at`` — and it is refused on a
``revoked_at`` or ``unpaired_at`` row, which is what makes those two stick
against the device's own next launch.

Two things this build mints, and they are deliberately different:

- **``device_key``** — one secret per device, minted and returned by every
  ``register`` call and stored machine-side. It is what a later request moves its
  own device's state with (ADR §4 rule 2's ``m3``: ``X-Lop-Device`` alone is a
  claim, the key is the proof); :func:`device_key_matches` is the comparison and
  :func:`note_credential` is its one caller. It is returned ONCE per call, is
  never rendered by ``list``, and never reaches a log.
- **``operator_key``** — one per machine, stored in this file beside the records
  and never given to any device. It is how the daemon's operators-only route
  (``.../unrevoke``) knows a caller is the machine's own surface rather than a
  phone: the phone's requests arrive through the tunnel gateway, which rebuilds
  the request headers from a fixed allowlist of presentation headers
  (``local_operator/tunnels/gateway.py``) and so cannot carry a header of our
  choosing at all — and a phone that somehow did carry one still would not have
  the value. **The honest limit, stated rather than implied:** this is not a
  boundary against another process on THIS machine. Such a process can read the
  key, and it could equally rewrite the store file directly; what the key adds is
  that the one direction a device must never be able to take — restoring its own
  revoked state — is closed by a secret the device does not have, instead of by
  a claim about who the caller is.

A record may carry the markers and the credential fields ABSENT: an earlier
build wrote neither, and an upgrade must not refuse a store the previous version
produced. Absence is the truth for those rows — ``list`` omits a credential
field it does not have rather than inventing a value for it (the repo's absence
rule) — and the next ``register`` fills them in.

**The credential fields, and who may write them** (push/ack-sync S4c; ADR §4
rule 2). Three of the record's optional fields exist for one question — *may the
cloud deliver to this device?* — and the answer is the relay's to compute,
because the relay is the only component that sees the ``lop_mobile`` cookie:

- ``credential_expires_at`` — the instant the cookie the device last PRESENTED
dies (``auth.cookie_expiry``). Written at ``register`` time, because the
registering row is the device that presented it and the relay's ``/login`` route
knows no device (round 7 Q-F15 / R8-m1). This is the field the lapse is DERIVED
from; on the Radient route the gateway mints a fresh cookie per request, so no
cookie there is ever old and that route's lock is the cloud's grant, not this
clock.
- ``credential_live`` and ``last_authenticated_at`` — the flag and the stamp.
Written ONLY by :func:`note_credential`, on an authenticated request that names
its device with the key minted for it, which is what makes the write
attributable (a register on its own cannot attribute the computer's cookie to one
device: review round 1 R4).
- :func:`credential_live_at` is the ONE derivation — the stored flag, the
Settings list and the credential report all answer "live?" through it, so no two
readers can disagree about what the word means.

**Who writes a MARKER, and the direction that matters.** The relay's routes write
markers: ``register`` clears ``expired_at`` (the restore path), :func:`rotate_credentials`
sets ``expired_at`` on every row when the relay password rotates, and
:func:`note_credential` sets it when a request arrives carrying a stale cookie.
The EMIT side writes nothing at all: :func:`credential_facts` is its only input
and is read-only, which is what makes "the emit worker writes no marker" a
structural fact rather than a promise. And the asymmetry the whole design rests
on is that a device's own report can only PAUSE its delivery, never restore it —
:func:`note_credential` never clears a marker, so accepting a self-reported
lapse is safe (§4 rule 2).
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import tempfile
import threading
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: The store file, directly under ``config_dir()`` beside the other
#: owner-private state (``mobile-seen.json``).
PUSH_DEVICES_STORE_NAME = "mobile-push-devices.json"

#: The enums the register contract allows (ADR §3.1).
PLATFORMS = ("ios", "android")
ENVIRONMENTS = ("sandbox", "production")

#: The device states (ADR §4), spelled once. The fifth state, ABSENT, is the
#: absence of a row rather than a value a row can hold, which is why it is not
#: in this tuple — a reader asking "which state is this device in" has already
#: answered ABSENT by finding no row.
STATE_LIVE = "live"
STATE_EXPIRED = "expired"
STATE_UNPAIRED = "unpaired"
STATE_REVOKED = "revoked"
DEVICE_STATES = (STATE_LIVE, STATE_EXPIRED, STATE_UNPAIRED, STATE_REVOKED)

#: State -> the marker field that puts a row in it, **highest precedence first**.
#: This tuple is the ONE definition of ADR §4's ``revoked > unpaired > expired``:
#: :func:`device_state` walks it, and ``PRECEDENCE`` — the string the wire carries
#: — is rendered from it, so the sentence and the resolver cannot drift apart.
_STATE_MARKERS: tuple[tuple[str, str], ...] = (
    (STATE_REVOKED, "revoked_at"),
    (STATE_UNPAIRED, "unpaired_at"),
    (STATE_EXPIRED, "expired_at"),
)

#: The precedence, rendered: ``"revoked > unpaired > expired"``.
PRECEDENCE = " > ".join(state for state, _marker in _STATE_MARKERS)

#: The state VOCABULARY, described once. The descriptions are the product's own
#: words for each state — the CLI's legend renders them and the app's Settings
#: will mirror them — so they live here rather than in a renderer, for the same
#: reason the refusal sentences do: two surfaces that spell the same state two
#: ways is the defect the single resolver exists to prevent, one layer up.
#:
#: ``absent`` is in the vocabulary but not in ``DEVICE_STATES``: no row can hold
#: it (it IS the absence of a row), and ``list`` never renders it as a ``state``.
#: It is here because the CLI shows it — a verb whose read-back finds the row
#: gone, or a row an operator asks about after a 60-day drop — and the one word
#: that needs explaining must not be the one word that is not explained.
STATE_ABSENT = "absent"
STATE_DESCRIPTIONS: dict[str, str] = {
    STATE_LIVE: "registered, and push resumes on its next authenticated read",
    STATE_EXPIRED: "notifications are paused for this device until you sign in again",
    STATE_UNPAIRED: "this computer is no longer paired",
    STATE_REVOKED: "this device was revoked on this computer",
    STATE_ABSENT: "not in this computer's registry; it may register again",
}
#: The order the descriptions are rendered in: the four row states by precedence
#: (weakest first, so the legend reads bottom-up as the escalation it is), then
#: ``absent``, which is not a row state at all.
DESCRIBED_STATES = (STATE_LIVE, STATE_EXPIRED, STATE_UNPAIRED, STATE_REVOKED, STATE_ABSENT)

#: What the precedence MEANS, in one sentence, rendered from ``PRECEDENCE`` so
#: the explanation and the rule cannot drift. A bare ``revoked > unpaired >
#: expired`` is the wire's notation for a rule about one row carrying two
#: markers; this is the same fact said to a person.
PRECEDENCE_SENTENCE = (
    "a device can carry more than one marker; the strongest is shown " f"({PRECEDENCE})"
)

#: The refusal codes and sentences of the register route (ADR §3.1). Spelled as
#: constants because the same three sentences are the CLI's and, later, the
#: app's copy: a router and a renderer that spell them separately will drift.
DEVICE_REVOKED_CODE = "device_revoked"
DEVICE_REVOKED_MESSAGE = "this device was revoked on this computer"
DEVICE_UNPAIRED_CODE = "device_unpaired"
DEVICE_UNPAIRED_MESSAGE = "this computer is no longer paired"
DEVICE_ABSENT_CODE = "device_absent"
MACHINE_ONLY_CODE = "machine_only"
MACHINE_ONLY_MESSAGE = "a device cannot restore itself — use the computer or your account"

#: The header an operator surface presents to the unrevoke route, and the store's
#: field holding the key it must match. Both spelled here so the daemon, the CLI
#: and the tests name one string each.
OPERATOR_KEY_HEADER = "X-Lop-Operator-Key"
OPERATOR_KEY_FIELD = "operator_key"

#: The two headers a relay request presents to move ITS OWN device's credential
#: state (ADR §4 rule 2's ``m3``; the names are S3's settled ones). ``X-Lop-Device``
#: names the ``install_id`` — attribution only, a CLAIM — and ``X-Lop-Device-Key``
#: carries the per-device key minted at registration, which is what makes the
#: claim a proof. A request without the key moves NO device's state: the key is
#: IDENTITY, never permission, which is why a valid key for a tombstoned row is
#: still refused by the state rules (:func:`note_credential`).
#:
#: Both are HEADERS and never query parameters, and both are spelled here rather
#: than in the daemon for the same reason the refusal sentences are: the header
#: the phone sends and the header the relay reads must be one string each.
DEVICE_HEADER = "X-Lop-Device"
DEVICE_KEY_HEADER = "X-Lop-Device-Key"

#: The CLOUD's idle-drop threshold, in days (ADR §2.2): a device with no
#: authenticated request for this long has its row deleted cloud-side, with no
#: marker. It is defined here, once, because the CLI renders a sentence that
#: quotes it — and it is deliberately NOT a local state: **the machine cannot see
#: that drop**, so no row ever reads as "dropped" and no local timer derives from
#: this number. A local lapse is ``expired``; the drop is the cloud's own row
#: deletion, and conflating the two would tell the user their device was removed
#: when the machine has no way to know that.
CLOUD_IDLE_DROP_DAYS = 60

#: Bound on every free-text field, the token included. Generous on purpose:
#: the token is opaque, and an over-tight bound would refuse a future
#: platform's longer token. The point is only that a malformed caller cannot
#: grow the store without limit.
MAX_FIELD_CHARS = 1024

#: Bound on registered devices (the ``SeenStore._bound_locked`` discipline). A
#: computer's devices are physical — a handful for any real user — so 256 is an
#: order of magnitude past anything a healthy app produces, while still bounding
#: both the file and the Settings payload. Overflow drops the least-recently-seen
#: records: firing at all means a caller is minting identities it should not
#: (the bug review round 1's M1 names), and the bound is a backstop, not a limit
#: a healthy fleet approaches.
MAX_DEVICE_ENTRIES = 256

#: One lock over the store's read-modify-write cycle. The routes run their
#: store calls on worker threads (``asyncio.to_thread``), so two concurrent
#: registrations could otherwise interleave load/save and lose one. In-process
#: is sufficient and deliberate: exactly one daemon process writes this store
#: (a restart's predecessor is gone before its successor serves requests).
_LOCK = threading.Lock()

#: The record's canonical shape. ``name`` is optional because the register body
#: does not define it — see the module docstring and the PR notes — so pre-name
#: records and nameless devices both round-trip.
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
#: Written by THIS build, and ABSENT in a record an earlier build wrote (the
#: module docstring's upgrade rule) — so absence is accepted on read and every
#: one of these refuses only when present-but-invalid. ``device_key``,
#: ``credential_live`` and ``last_authenticated_at`` are ADR §2.2's per-device
#: **Credential record**, kept in the device's own row because this store's
#: reader is one route and one CLI; the three ``*_at`` markers are §4's states.
_OPTIONAL_RECORD_FIELDS = frozenset(
    {
        "name",
        "device_key",
        "credential_live",
        "credential_expires_at",
        "last_authenticated_at",
        "expired_at",
        "unpaired_at",
        "revoked_at",
    }
)
_RECORD_FIELDS = _REQUIRED_RECORD_FIELDS | _OPTIONAL_RECORD_FIELDS

#: The store's own two top-level keys. Anything else refuses (the store is
#: written by this module or not at all), which is what keeps a hand-edited or
#: future-version file from being silently rewritten into this build's shape.
_STORE_FIELDS = frozenset({"devices", OPERATOR_KEY_FIELD})

#: The register body's canonical keys; an unknown key refuses (below).
_REGISTER_FIELDS = frozenset(
    {"platform", "token", "environment", "app_version", "install_id", "name"}
)


class PushDeviceRefusal(Exception):
    """One register payload this build cannot accept, written for the reader.

    ``message`` is the sentence the daemon's 422 JSON body carries — the same
    refusal shape ``api_session_seen`` answers a bad ``completion_token`` with.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class PushDeviceStateRefusal(PushDeviceRefusal):
    """A refusal that carries a machine-readable code and its own status.

    ADR §3.1 spells three of these for the register route (``device_revoked``,
    ``device_unpaired``, both 403) and one for the operators-only route
    (``machine_only``). They are a subclass rather than a sibling so a caller
    that only knows the payload refusal still refuses these correctly; the
    daemon's handler catches this class FIRST, because that is what chooses the
    status and adds the ``code`` field.
    """

    def __init__(self, code: str, message: str, *, status_code: int = 403) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class PushDeviceAbsent(PushDeviceStateRefusal):
    """No row for the ``device_id`` an operator asked about.

    Only the operator's ``unrevoke`` raises this, and only there: ``revoke`` is
    idempotent by contract (the app retries it on sign-out, and a retry after a
    successful revoke must not read as a failure — ADR §3.1), while "there is no
    such device" is information the operator's command needs, not a retry.
    """

    def __init__(self, device_id: str) -> None:
        super().__init__(
            DEVICE_ABSENT_CODE,
            f"no device {device_id} in this computer's push registry",
            status_code=404,
        )
        self.device_id = device_id


class PushRegistryCorrupt(RuntimeError):
    """The stored registry cannot be used, and must not be rewritten.

    Raised for an unreadable file, invalid JSON, or a record this build cannot
    round-trip. Every registry operation refuses while it stands and the file
    is left untouched by the raiser; the daemon answers it as an internal fault
    (500) whose message names the problem.
    """


def store_path(config_dir: Path) -> Path:
    """The registry file's path under one config root.

    Public because the daemon's refusal log names it (``_push_call``): "which
    file" is the first question a reader of that log line asks.
    """
    return config_dir / PUSH_DEVICES_STORE_NAME


def device_state(record: Mapping[str, Any]) -> str:
    """THE resolver of ADR §4's precedence: ``revoked > unpaired > expired``.

    One row can carry several markers (a device revoked, and then its computer
    unpaired), and this is the only place that decides which one wins. The
    register route calls it before it writes, ``list`` calls it to render, and the
    emit side reads it through :func:`credential_facts` — so no two consumers can
    read one row as two different states, which is the invariant ADR §4 states and
    this function IS.

    A row with no marker is ``live`` — including a row an earlier build wrote,
    which is the truth for it rather than a fallback: nothing marked it, so
    nothing has stopped it. ``absent`` is not a return value; a caller asking
    about a device that has no row at all has already answered that by looking.

    Where its consumers are, for the next reader who greps for them: all of them
    are inside this module (the register route, the per-device evaluation, ``list``
    and the emit-side read, :func:`credential_facts`), and nothing outside
    ``local_operator/mobile`` calls it yet — the daemon and the CLI read the
    rendered ``state`` from those answers rather than resolving a row themselves.
    """
    for state, marker in _STATE_MARKERS:
        if record.get(marker):
            return state
    return STATE_LIVE


def credential_live_at(record: Mapping[str, Any], now: float) -> bool:
    """THE lapse rule for one device at one instant (ADR §4 rule 2).

    A device's credential is live while the cookie it last presented has not
    died: ``credential_expires_at > now``. The instant is the COOKIE'S OWN and
    not a timer and not a last-request stamp (round 7 Q-F15: the cookie is signed
    once at login and never renewed, so ``last_authenticated_at`` + TTL would
    over-report — a phone silent since day 29 would look alive until day 59 while
    its cookie really died on day 30).

    Two inputs, in this order, and the order is the design:

    1. an ``expired_at`` marker answers NOT live, outright. It is the one fact a
       rotation writes that no clock can infer (a rotation kills every cookie
       while each `credential_expires_at` is still days in the future), so the
       marker has to outrank the arithmetic or a rotated device would report
       live until its dead cookie's nominal expiry.
    2. ``credential_expires_at`` when the row carries one — the derivation, read
       fresh. This is what makes a lapse observable with NO request at all (the
       heartbeat case an app that was shut all week produces), which is why the
       report recomputes through here rather than reading the stored flag.
    3. the stored ``credential_live`` otherwise, for a row an earlier build wrote
       before the expiry was recorded. Absent on all three is NOT live: a device
       no request has ever named has no credential fact, and inventing one would
       tell the cloud it may deliver on evidence the machine does not hold.

    It is one function and not a rule restated per caller so that every reader —
    the Settings list, the credential report, the recompute itself — answers
    "live?" the same way, and the only thing that can differ between them is the
    instant they ask about. The stored flag is an INPUT to it (point 3), never a
    second answer beside it (review round 1, AR-1).
    """
    if "expired_at" in record:
        return False
    expires = record.get("credential_expires_at")
    if isinstance(expires, int) and not isinstance(expires, bool):
        return expires > now
    return bool(record.get("credential_live"))


def device_key_matches(record: Mapping[str, Any], presented: object) -> bool:
    """Whether ``presented`` is this device's key (ADR §4 rule 2's ``m3``).

    Constant-time, and False for every absence: a record with no key, a key that
    is not a string, an empty presentation. This is the check that makes
    ``X-Lop-Device: <install_id>`` a proof rather than a claim — a request that
    moves a device's state must present the key minted for that device at
    registration, so a stolen cookie cannot vouch for a sibling device. Its one
    caller is :func:`note_credential`, which is why the comparison lives here
    rather than on that route.
    """
    stored = record.get("device_key")
    if not isinstance(stored, str) or not stored:
        return False
    if not isinstance(presented, str) or not presented:
        return False
    return hmac.compare_digest(stored.encode(), presented.encode())


def operator_key(config_dir: Path) -> str | None:
    """This machine's operator key, or ``None`` when the store holds none yet.

    The key is the second half of "only the operator surface may unrevoke"
    (module docstring): the tunnel gateway cannot forward a header of our
    choosing, and a device that somehow did would still not have this value.

    READ-ONLY, AND THAT IS THE POINT (review round 1, R5). Exactly one process
    writes this store — the daemon — and the key is minted inside
    :func:`register`, under that same lock, as part of a write the daemon was
    making anyway. An earlier revision let the CLI mint here when the store had
    none, which made a second process a writer for exactly the case the branch
    existed for (a store written before this build): the CLI's read-modify-write
    could interleave with a registration and lose one of the two, since
    ``_LOCK`` and the atomic replace protect the file's integrity, not the
    update. A caller with no key gets ``None`` and says so — an honest "not yet"
    beats a write that can drop a device.

    The consequence, stated rather than discovered: on a store written before
    this build, the key does not exist until some device registers again, so
    ``unrevoke`` is unavailable until then. That is the trade R5 asked for, and
    it is one app launch wide.
    """
    with _LOCK:
        store = _load_store(config_dir)
    key = store.get(OPERATOR_KEY_FIELD)
    return key if isinstance(key, str) and key else None


def verify_operator_key(config_dir: Path, presented: object) -> bool:
    """Whether a request presented THIS machine's operator key.

    Never mints (a verifier that minted would hand the first caller the key it
    just failed to present) and never swallows a store it cannot read: a
    corrupt store raises, and the daemon answers its usual 500 — "the device may
    not restore itself" would be a false sentence for a machine whose registry
    is unreadable.
    """
    if not isinstance(presented, str) or not presented:
        return False
    with _LOCK:
        store = _load_store(config_dir)
        stored = store.get(OPERATOR_KEY_FIELD)
    if not isinstance(stored, str) or not stored:
        return False
    return hmac.compare_digest(stored.encode(), presented.encode())


def register(
    config_dir: Path,
    body: object,
    *,
    credential_expires_at: int | None = None,
    now: float | None = None,
) -> dict[str, Any]:
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

    REFUSED before anything is written when this ``install_id``'s row is
    ``revoked`` or ``unpaired`` (ADR §3.1): silence would be worse than either
    sentence, because the app would show a registered device that never receives
    anything. The two refusals are the states' own sentences and codes, resolved
    by :func:`device_state` — the same resolver ``list`` renders from, so a row
    carrying both markers refuses as the same state ``list`` reports. An
    ``expired`` row is NOT refused: authenticating and re-registering IS the act
    that clears ``expired_at`` (that is why it is a different marker from
    ``revoked_at``), and a fresh ``install_id`` is refused on nothing at all.

    A successful call also mints this device's per-device key (ADR §3.1) —
    returned ONCE, in this response only — and stores it in the device's own row.

    It does NOT write ``credential_live`` or ``last_authenticated_at``, and that
    is a correction rather than an omission (review round 1, R4 / ADR §4 rule 2,
    QA round 4 Q-F2). The flag is a PER-DEVICE fact, and this route cannot
    attribute the request to the device: on the direct route the app holds the
    machine's cookie (one cookie for every device of this computer), and on the
    Radient route the edge STRIPS it and the gateway injects it on every request,
    so "cannot tell device A from device B". A successful register therefore
    proves an authenticated connection, not this device's live credential — and
    the relay's per-device reading is where §4 rule 2 puts it: a request that
    NAMES its device (``X-Lop-Device`` + the key minted here) is
    :func:`note_credential`'s route, and that is where those two fields are
    written. Until then ``list`` omits them rather than over-claiming a flag the
    ADR has the cloud enforce against.

    ``credential_expires_at`` IS written here, and the difference from the two
    fields above is the whole of ADR §4 rule 2's round 7 Q-F15 / R8-m1: the
    expiry is not a claim about this device, it is the instant the cookie the
    request PRESENTED dies, and the registering row is the device that presented
    it — so the caller reads it off the cookie (``auth.cookie_expiry``) and hands
    it in. The relay's ``/login`` route cannot write it: it knows no device, and
    on a first install there is no row to write it to. The value is the DIRECT
    route's lapse rule (``credential_live_at`` compares it against now); on the
    RADIENT route the gateway mints a fresh cookie on every request
    (``tunnels/gateway.py``'s ``sign_cookie`` call), so no cookie there is ever
    old and a device's liveness on that route is the CLOUD's grant, not this
    clock (ADR §4 rule 2).
    """
    fields = _checked_registration(body)
    with _LOCK:
        store = _load_store(config_dir)
        records = store["devices"]
        stamp = int(time.time() if now is None else now)
        record = _find(records, fields["install_id"], fields["platform"])
        if record is not None:
            state = device_state(record)
            if state == STATE_REVOKED:
                raise PushDeviceStateRefusal(DEVICE_REVOKED_CODE, DEVICE_REVOKED_MESSAGE)
            if state == STATE_UNPAIRED:
                raise PushDeviceStateRefusal(DEVICE_UNPAIRED_CODE, DEVICE_UNPAIRED_MESSAGE)
        # Minted per call (ADR §3.1: "the relay mints that device's per-device
        # key here"), so a re-register replaces the key as it replaces the
        # metadata — the app that just registered is the only holder of the
        # current one, and a stale key from a previous installation buys nothing.
        device_key = secrets.token_urlsafe(32)
        if record is None:
            record = {
                # Machine-minted and provisional: the cloud's id supersedes it
                # once the S7 forward exists (module docstring).
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
        # The re-register clears the credential LAPSE and nothing else: this is
        # ADR §3.1's "expired_at does NOT refuse", implemented where it matters.
        # The two refusal markers are untouched — a row that reaches here has
        # neither, and clearing one would undo a decision no request made.
        record.pop("expired_at", None)
        # The presented cookie's own expiry (the parameter's docstring above):
        # refreshed on every register, so the direct route's lapse rule tracks
        # the cookie the app is actually holding. A caller that presents no
        # cookie fact (a non-browser client, a probe) leaves whatever the row
        # already carries alone rather than clearing a real one.
        if credential_expires_at is not None:
            record["credential_expires_at"] = int(credential_expires_at)
        record["device_key"] = device_key
        # The machine's operator key is minted WITH the first device, inside this
        # daemon-side write rather than lazily by the CLI: exactly one process
        # writes this file, so the normal path never has two writers (R5), and
        # the conditional means a re-register costs a branch rather than a
        # CSPRNG call (N3 — ``dict.setdefault`` would evaluate its default
        # eagerly, minting 32 random bytes per registration and discarding
        # them).
        if not store.get(OPERATOR_KEY_FIELD):
            store[OPERATOR_KEY_FIELD] = secrets.token_urlsafe(32)
        store["devices"] = _prune_locked(records)
        _save(config_dir, store)
        return {
            "ok": True,
            "device_id": record["device_id"],
            "device_key": device_key,
            "registered_at": record["registered_at"],
        }


def note_credential(
    config_dir: Path,
    *,
    install_id: object,
    device_key: object,
    credential_expires_at: object,
    now: float | None = None,
) -> dict[str, Any] | None:
    """Apply ONE authenticated request's credential evidence to ITS device.

    This is ADR §4 rule 2's evaluation, and it is the relay's because the relay
    is the only component that sees the ``lop_mobile`` cookie. It runs on an
    authenticated request that NAMES a device — ``X-Lop-Device: <install_id>``
    plus ``X-Lop-Device-Key: <device_key>`` — and recomputes that device's
    ``credential_live`` and ``last_authenticated_at`` from the expiry the request
    presented (``auth.cookie_expiry`` on the cookie, which is what makes the
    instant the COOKIE'S and not ``now``).

    The three refusals that make it safe, each of which is a test:

    * **no key, or a wrong one, moves NO device's state.** Attribution is bound,
      not asserted (``m3``): ``X-Lop-Device`` is a claim anyone holding the
      computer's cookie can make, so without the per-device key there is no
      proof and this function returns ``None`` without so much as reading the
      store's answer into a write. A request naming a device that is not in the
      registry is the same case — there is nothing to attribute to.
    * **a valid key for a TOMBSTONED row moves nothing.** The key proves
      IDENTITY, never PERMISSION (``push_devices``' header constants say so):
      the state rules still win, so a revoked or unpaired device cannot write a
      credential fact that would restore it. Checked BEFORE the recompute, so
      there is no intermediate state to observe.
    * **it can only PAUSE, never restore.** A lapsed cookie writes ``expired_at``;
      nothing here ever CLEARS a marker. That asymmetry is the whole reason a
      device's own report is safe to accept (ADR §4 rule 2: "every self-reported
      action here can only *pause* its own delivery, never restore it"). The way
      back is the register that follows an authenticated login, which clears
      ``expired_at`` because the credential that earned it was live.

    **What it writes, exactly — and a claim this docstring used to get wrong**
    (review round 1, AR-5). It rewrites the row once per ATTRIBUTABLE request,
    and that is the semantics rather than an accident: ``last_authenticated_at``
    IS the fact such a request updates, so the row changes whenever the request
    crosses an integer second. The ``before == after`` guard below therefore
    suppresses a rewrite only inside the same second (a retry, a burst); an
    earlier draft claimed it left a phone polling every 2 s as a non-writer,
    which was false of the code. The fix is the claim, not the write — "when did
    this device last authenticate" is a question whose whole point is that it
    moves, and the ADR's list shape renders it. What the guard does buy is that a
    same-second burst does not touch the file, so the store's mtime stays a signal
    a reader can use rather than noise.

    The return value is ``None`` for "no state moved" and otherwise the moved
    facts, which is what lets a caller (or a test) tell an evaluation apart from
    a no-op without reading the store behind it.
    """
    stamp = int(time.time() if now is None else now)
    if not isinstance(credential_expires_at, int) or isinstance(credential_expires_at, bool):
        return None
    if not isinstance(install_id, str) or not install_id:
        return None
    with _LOCK:
        store = _load_store(config_dir)
        record = next(
            (
                candidate
                for candidate in store["devices"]
                if candidate.get("install_id") == install_id
                and device_key_matches(candidate, device_key)
            ),
            None,
        )
        if record is None:
            return None
        if device_state(record) in (STATE_REVOKED, STATE_UNPAIRED):
            # Refused rather than written-and-ignored: the marker is the machine's
            # answer, and a credential fact written underneath it would be a
            # second one.
            return None
        # The recompute. ``credential_live_at`` is asked about the row with the
        # presented expiry in place, so a marker a rotation wrote still outranks
        # the arithmetic — this request may not be the first evidence since it.
        presented_live = credential_expires_at > stamp
        live = credential_live_at({**record, "credential_expires_at": credential_expires_at}, stamp)
        before = {
            "credential_live": record.get("credential_live"),
            "credential_expires_at": record.get("credential_expires_at"),
            "last_authenticated_at": record.get("last_authenticated_at"),
            "expired_at": record.get("expired_at"),
        }
        record["credential_expires_at"] = credential_expires_at
        record["credential_live"] = live
        record["last_authenticated_at"] = stamp
        if not presented_live:
            # Written from the COOKIE's own death, not this request's clock: the
            # phone may arrive up to the cookie skew late, and the marker is meant
            # to say when the credential lapsed. Guarded on the presented cookie
            # rather than on the derived flag because a live cookie must never
            # stamp a FUTURE instant into a marker — a row a rotation expired, and
            # whose device has since logged in again, keeps the rotation's
            # instant until the register that clears it (this function may not
            # restore delivery; ADR §4 rule 2's "one writer" says the register is
            # where ``expired_at`` goes).
            record["expired_at"] = credential_expires_at
        after = {
            "credential_live": record.get("credential_live"),
            "credential_expires_at": record.get("credential_expires_at"),
            "last_authenticated_at": record.get("last_authenticated_at"),
            "expired_at": record.get("expired_at"),
        }
        if after == before:
            return None
        _save(config_dir, store)
        return {
            "device_id": record["device_id"],
            "state": device_state(record),
            **after,
        }


def credential_facts(config_dir: Path, *, now: float | None = None) -> list[dict[str, Any]]:
    """READ-ONLY per-device credential facts — the emit path's only input.

    The counterweight to :func:`note_credential` and the reason the "one writer"
    rule is structural rather than a convention: this function cannot write,
    because the only mutation in this module runs under ``_LOCK`` inside the
    functions that spell ``_save``, and an emitter that went through here
    physically has no call to make (ADR §4 rule 2: "the emit worker writes no
    marker at all: it reads the flag and the markers, skips a device, and emits
    the event"). A test asserts the read leaves the store's bytes and mtime
    untouched, so the invariant fails loudly if a future edit reaches for a write.

    ``credential_live`` is the derived value at ``now`` (:func:`credential_live_at`)
    rather than the stored flag — that is what lets an app that has been shut for
    a week be reported not-live at the next heartbeat with no request involved.
    ``expired_at``/``unpaired_at``/``revoked_at`` ride along because the emitting
    side must skip a device the machine has marked; the wire block built from
    this keeps only the four fields the ADR names.
    """
    stamp = float(time.time() if now is None else now)
    facts: list[dict[str, Any]] = []
    with _LOCK:
        records = _load(config_dir)
    for record in records:
        fact: dict[str, Any] = {
            "device_id": record["device_id"],
            "state": device_state(record),
            "credential_live": credential_live_at(record, stamp),
        }
        for field in ("credential_expires_at", "last_authenticated_at", "expired_at"):
            if field in record:
                fact[field] = record[field]
        facts.append(fact)
    return facts


def rotate_credentials(config_dir: Path, *, now: float | None = None) -> list[str]:
    """A relay-password rotation: every device's cookie dies at once.

    The machine-side half of ADR §4 path 3, and it is ONE action for the whole
    registry because the cookie key is derived from the password — rotating it
    invalidates every device's ``lop_mobile`` cookie simultaneously
    (``auth``'s module docstring). It writes ``expired_at`` and NEVER
    ``revoked_at``, which is the distinction §4 turns on: a rotation is not a
    decision about any device, the tokens are kept, and every device is welcome
    back the moment it authenticates (the register that follows a fresh login
    clears the marker).

    Every row gets the marker, including one already carrying a stronger one: the
    truth it records — this cookie died here — is true of all of them, and
    precedence (``revoked`` > ``unpaired`` > ``expired``) is what keeps each row's
    ANSWER stable. The returned ids are what the caller hands to the credential
    report, which is where §4 rule 2's "one credential-change event per rotation"
    is enforced — one event naming every device, never one per device.

    On the RADIENT route this is an outage rather than a lever (§4 path 3): the
    gateway signs with the password it read at construction, so until the
    connector restarts it signs with a STALE one and every device of that
    computer stops, a restart healing all of them. The account side is the lever
    there; this marker is still the honest machine-side record of what happened.
    """
    stamp = int(time.time() if now is None else now)
    with _LOCK:
        store = _load_store(config_dir)
        rotated: list[str] = []
        changed = False
        for record in store["devices"]:
            rotated.append(record["device_id"])
            if record.get("expired_at") == stamp:
                # Idempotent: a second rotation at the same instant writes nothing.
                continue
            # The marker ALONE, and no ``credential_live`` beside it. An earlier
            # revision also wrote the flag here, to stop the Settings list
            # rendering "live" beside an ``expired`` state — which was treating a
            # derived answer as a second stored fact, the exact drift review
            # round 1's AR-1 found between two renderers. ``credential_live_at``
            # answers False for a row carrying this marker (it outranks the
            # arithmetic), so the marker is the whole write and the stored flag
            # keeps meaning one thing: what the relay computed when it last saw
            # this device's cookie.
            record["expired_at"] = stamp
            changed = True
        if changed:
            _save(config_dir, store)
    return rotated


def list_devices(config_dir: Path, *, now: float | None = None) -> dict[str, Any]:
    """``GET /api/push/devices`` — the Settings list.

    Exactly the fields the ADR's shape names, ``environment`` and
    ``install_id`` excluded on purpose: the phone renders them nowhere, and the
    wire is a contract, not the store's dump. ``name`` is omitted when unknown
    (the repo's absence rule — a null would be a client-visible claim it must
    special-case; absence is the truth), and so are ``credential_live`` and
    ``last_authenticated_at`` on a row an earlier build wrote: this store will
    not invent a credential fact it does not hold.

    ``credential_live`` is the DERIVED answer (:func:`credential_live_at`), not
    the flag the record happens to hold (review round 1, AR-1 / QA Q-F1). The
    stored flag is what the relay computed when it last SAW the cookie, and the
    two part company exactly when it matters — a cookie that died with no keyed
    request since leaves ``credential_live: true`` on the row while the machine
    has stopped delivering, which is the case the ADR's heartbeat exists for. One
    reader rendering the stored value while the credential report derived would be
    one row read two ways, so both go through the one rule. The presence guard is
    widened to either credential field rather than to the flag alone: a row with an
    expiry and no stored flag still has a fact to report, and a row with NEITHER is
    an earlier build's and keeps omitting the key.

    That makes this function's answer a function of the CLOCK, which is the
    intent rather than a side effect: a lapse is derived from the cookie's own
    death, so the same store can render differently across a cookie's lifetime
    without anything having written to it. ``now`` is injectable for the same
    reason every other function here takes one.

    ``state`` and the response's ``precedence`` are ADR §4's one vocabulary: the
    state is :func:`device_state`'s answer on this row, and the precedence string
    is rendered from the same table the resolver walks. **No secret leaves here**
    — not the push token (never stored), not the per-device ``device_key`` (the
    response that mints it is the only place it appears), not the machine's
    operator key.

    Registration order, and a re-register updates in place, so a device's
    position does not move under a token rotation. READ-ONLY: nothing is
    written, so opening Settings cannot bump anything.
    """
    stamp = float(time.time() if now is None else now)
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
            "state": device_state(record),
        }
        if "name" in record:
            entry["name"] = record["name"]
        if "credential_live" in record or "credential_expires_at" in record:
            entry["credential_live"] = credential_live_at(record, stamp)
        if "last_authenticated_at" in record:
            entry["last_authenticated_at"] = record["last_authenticated_at"]
        devices.append(entry)
    return {"devices": devices, "precedence": PRECEDENCE}


def revoke(config_dir: Path, device_id: str, *, now: float | None = None) -> dict[str, Any]:
    """``DELETE /api/push/devices/{device_id}`` — REVOKE one device.

    A TOMBSTONE, not a row removal (ADR §4 rule 1): ``revoked_at`` is set and the
    row stays, which is what makes the revoke stick for the same ``install_id``
    instead of being undone by the app's next launch — the register route refuses
    a row carrying it. The cloud drops the token; **no token is stored here to
    drop** (module docstring), and nothing else in the record is touched, because
    the metadata is what the operator's Settings list renders while deciding
    whether to unrevoke.

    Idempotent in both directions an idempotent route needs to be: an id the
    registry does not hold stays ``{"ok": true}`` and nothing is written (the app
    retries this on sign-out, and a retry after a successful revoke must not read
    as a failure), and a second call for a row already tombstoned leaves the
    first tombstone's timestamp alone — the revoke happened once.

    Deliberately NOT scoped to "the caller's own device" — the relay's cookie is
    one operator for the whole computer, and the stolen-phone case (ADR §4) needs
    one device to be able to revoke another. SELF-TARGETING IS SAFE ON THIS ROUTE
    AND ONLY ON THIS ONE: a device that revokes itself can only reduce its own
    access, which is why the way BACK lives on a different route that this one
    cannot reach (:func:`unrevoke`).

    Refuses (raises) on a store it cannot read, even when the id is absent:
    "not in a store I cannot read" is not an answer this store is willing to
    give.
    """
    with _LOCK:
        store = _load_store(config_dir)
        record = _by_id(store["devices"], device_id)
        if record is not None and "revoked_at" not in record:
            record["revoked_at"] = int(time.time() if now is None else now)
            _save(config_dir, store)
    return {"ok": True}


def unrevoke(config_dir: Path, device_id: str, *, now: float | None = None) -> dict[str, Any]:
    """``POST /api/push/devices/{device_id}/unrevoke`` — the way back, operator only.

    THE ROUTE THAT CONSUMES THIS FUNCTION IS THE MACHINE'S, AND THAT IS THE
    POINT: a revoked phone holding a live cookie must not be able to clear its
    own tombstone (ADR §4's named worst failure), so the daemon refuses any
    caller that cannot present this machine's operator key before it gets here.
    This function is what the operator's surface calls once that gate has passed.

    It clears every RESTORABLE marker the row carries — ``revoked_at`` and/or
    ``unpaired_at``, the two different states with the two different refusals —
    and **restores no token and no credential**: ``credential_live`` and
    ``last_authenticated_at`` are left exactly as the last authenticated request
    wrote them, because the device must register again, and registering needs a
    live credential. A row carrying BOTH is restored in one act rather than left
    in a state nobody asked for: the operator's intent is "this device is welcome
    back", and stopping at the stronger marker would refuse its next
    registration under the weaker one, for a reason no one chose. A row with no
    restorable marker is a no-op that still answers ``ok``: there is nothing to
    restore, and the caller can read the state back from ``list``.

    ``expired_at`` is deliberately NOT among them, and the word "restorable" is
    doing real work above: a lapse is not a decision about this device, and its
    way back is the one every lapse has — sign in and register again, which is
    what clears that marker (ADR §4 rule 2). This route is the operator's lever
    over the two markers that ARE decisions, and widening it to the third would
    make it a lever over authentication it cannot perform.

    The CLI's result line names the STRONGEST marker the pre-verb state implies
    (``device_state``) — ``revoked`` → ``unrevoked …``, ``unpaired`` → ``cleared
    the unpaired marker on …``, ``expired`` → ``nothing to clear on …`` plus "it
    is expired, not revoked — signing in again is what resumes push", and no
    marker → ``nothing to clear on …`` — and its ``state:`` clause reports the
    outcome the read-back found. The line deliberately does NOT enumerate every
    marker it cleared: the strongest marker is what the row *was*, one row is one
    line, and a sentence listing two markers would describe it in two
    vocabularies at once (the mobile lane's round-1 note, decided as "the loop is
    right, the prose was not").

    Raises :class:`PushDeviceAbsent` for an id the registry does not hold. That is
    deliberately the opposite of :func:`revoke`'s leniency: "no such device" is
    something an operator command must be able to report, while a retried revoke
    must not report a failure it did not have.
    """
    with _LOCK:
        store = _load_store(config_dir)
        record = _by_id(store["devices"], device_id)
        if record is None:
            raise PushDeviceAbsent(device_id)
        cleared = False
        for marker in ("revoked_at", "unpaired_at"):
            if marker in record:
                del record[marker]
                cleared = True
        if cleared:
            _save(config_dir, store)
    return {"ok": True, "device_id": device_id}


# -- payload validation ---------------------------------------------------------


def _checked_registration(body: object) -> dict[str, Any]:
    """One register body, field by field, strictly — or a refusal sentence."""
    if not isinstance(body, dict):
        raise PushDeviceRefusal("a JSON object body is required")
    # Unknown keys refuse in BOTH directions — the register body like a stored
    # record, matching the projects routes' ``_selected`` (whose models are
    # ``extra="forbid"``): a typing slip in an identity field ("installID")
    # must read as a refusal, not as "admitted".
    unknown = sorted(set(body) - _REGISTER_FIELDS)
    if unknown:
        # The echo is BOUNDED (review round 2's N1, QA's Q2): the names are
        # caller-chosen, and an unbounded refusal body is a caller-controlled
        # surface — capped at the same width as every field on this route.
        names = ", ".join(unknown)[:MAX_FIELD_CHARS]
        raise PushDeviceRefusal(f"unknown field(s): {names}")
    platform = body.get("platform")
    if platform not in PLATFORMS:
        raise PushDeviceRefusal('platform must be "ios" or "android"')
    environment = body.get("environment")
    if environment not in ENVIRONMENTS:
        raise PushDeviceRefusal('environment must be "sandbox" or "production"')
    # Validated, then dropped — never persisted (module docstring, ADR §4). The
    # two failure modes stay distinct: an over-long token is not a missing one.
    token = body.get("token")
    if not isinstance(token, str) or not token.strip():
        raise PushDeviceRefusal("token is required")
    if len(token) > MAX_FIELD_CHARS:
        raise PushDeviceRefusal("token is too long")
    app_version = _checked_text(body.get("app_version"), "app_version")
    install_id = _install_id(body.get("install_id"))
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


def _install_id(value: object) -> str:
    """The register body's ``install_id``: a UUID, canonicalised.

    ADR §3.1 spells the field "<uuid, minted once and kept in the keystore>".
    Parsed rather than pattern-matched, and re-spelled canonically (lowercase,
    hyphenated) so a case variant of the same UUID — iOS spells them uppercase
    by default — cannot fork one device into two ``(install_id, platform)``
    identities; such a fork would defeat the idempotency this slice rests on.
    """
    if not isinstance(value, str) or not value.strip():
        raise PushDeviceRefusal("install_id is required")
    if len(value.strip()) > MAX_FIELD_CHARS:
        raise PushDeviceRefusal("install_id is too long")
    try:
        return str(uuid.UUID(value.strip()))
    except ValueError as exc:
        raise PushDeviceRefusal("install_id must be a UUID") from exc


def _checked_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PushDeviceRefusal(f"{field} is required")
    return _bounded(value.strip(), field)


def _bounded(value: str, field: str) -> str:
    if len(value) > MAX_FIELD_CHARS:
        raise PushDeviceRefusal(f"{field} is too long")
    return value


def _find(records: list[dict[str, Any]], install_id: str, platform: str) -> dict[str, Any] | None:
    """The record this registration identifies, if any — the idempotency key.

    Matched on the PARSED UUID, not the stored spelling (review round 2's M1,
    QA's Q1): ``_validated_record`` accepts any spelling a writer could have
    stored — this store never rewrites on read — and a build before the
    canonicalisation landed stored the caller's spelling verbatim, so an
    uppercase record is a shape that exists on disk. Comparing strings there
    forked one phone into two rows on the next launch; identities cannot fork.
    Both sides are guaranteed parseable: the incoming value by ``_install_id``,
    the stored one by ``_validated_record``.
    """
    identity = uuid.UUID(install_id)
    for record in records:
        if record["platform"] == platform and uuid.UUID(record["install_id"]) == identity:
            return record
    return None


def _by_id(records: list[dict[str, Any]], device_id: str) -> dict[str, Any] | None:
    """The row this ``device_id`` names, or None. First match wins.

    A duplicate id is a shape only a foreign or hand-edited file can produce
    (this store never mints one twice), and the operators' routes act on the row
    they can see first rather than refusing the whole command over it.
    """
    for record in records:
        if record["device_id"] == device_id:
            return record
    return None


def _prune_locked(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the least-recently-seen records past ``MAX_DEVICE_ENTRIES``.

    ``SeenStore._bound_locked``'s discipline: prune on the write path, so a
    caller-side bug that mints a fresh ``install_id`` per launch cannot grow
    the store or the Settings payload without limit. The ranking is
    ``last_seen_at`` (``registered_at`` breaks ties): every healthy
    re-register refreshes its record, so a recently-active device is the LAST
    thing this can touch, and ghosts from a broken caller age out as newer
    registrations push in. Firing at all already means something upstream is
    wrong — no bound can drop rows otherwise — so least-harm is the most this
    can do, and it beats growth. The caller holds ``_LOCK``.
    """
    excess = len(records) - MAX_DEVICE_ENTRIES
    if excess <= 0:
        return records
    ranked = sorted(records, key=lambda record: (record["last_seen_at"], record["registered_at"]))
    # Exactly `excess` RECORDS leave, selected by position — never by
    # `device_id` VALUE: duplicate ids (only a foreign or hand-edited file can
    # produce them; this store never mints one twice) would otherwise cost
    # BOTH twin rows for one slot, over-pruning past the bound. `SeenStore`'s
    # bound pops `ranked[:excess]` off unique dict keys; a list has no such
    # key, so object identity is the selector that cannot collapse
    # equal-looking records, and the survivors keep their original order.
    doomed = {id(record) for record in ranked[:excess]}
    return [record for record in records if id(record) not in doomed]


# -- the file -------------------------------------------------------------------


def _load(config_dir: Path) -> list[dict[str, Any]]:
    """The stored records, validated; ``[]`` when the store does not exist yet.

    The record list alone, for a reader that has no business with the store's
    other key — :func:`_load_store` is the whole file, and it owns the refusal
    rule both callers share.
    """
    return _load_store(config_dir)["devices"]


def _load_store(config_dir: Path) -> dict[str, Any]:
    """The whole store, validated: ``{"devices": [...], "operator_key": ...}``.

    Everything else — an unreadable file, JSON that does not parse, a shape
    this build does not know — refuses with :class:`PushRegistryCorrupt` rather
    than degrading, per the module docstring. There is no "best effort" read:
    an empty answer for a file that exists and cannot be read is the one answer
    that silently loses devices.

    ``operator_key`` is present only when the store holds one (a store this
    build wrote, or one it has since minted for; an older store simply has
    none). The returned dict is what a writer mutates and hands to :func:`_save`,
    so the key survives every rewrite — which is what makes "minted once" true.
    """
    path = store_path(config_dir)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"devices": []}
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise PushRegistryCorrupt(
            f"push device registry cannot be read ({exc}); refusing to treat it as empty"
        ) from exc
    if (
        not isinstance(raw, dict)
        or "devices" not in raw
        or not set(raw) <= _STORE_FIELDS
        or not isinstance(raw["devices"], list)
    ):
        raise PushRegistryCorrupt(
            "push device registry has an unexpected shape; refusing to rewrite it"
        )
    store: dict[str, Any] = {
        "devices": [_validated_record(entry, index) for index, entry in enumerate(raw["devices"])]
    }
    if OPERATOR_KEY_FIELD in raw:
        key = raw[OPERATOR_KEY_FIELD]
        if not isinstance(key, str) or not key or len(key) > MAX_FIELD_CHARS:
            raise PushRegistryCorrupt(
                "push device registry has an invalid operator key; refusing to rewrite it"
            )
        store[OPERATOR_KEY_FIELD] = key
    return store


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
    for field in ("device_id", "app_version", "install_id", "name", "device_key"):
        if field not in entry:
            continue
        value = entry[field]
        if not isinstance(value, str) or not value or len(value) > MAX_FIELD_CHARS:
            raise PushRegistryCorrupt(
                f"push device registry record {index} has an invalid {field};"
                " refusing to rewrite it"
            )
    # The identity is a UUID by contract (ADR §3.1). Any stored spelling is
    # accepted — normalising on read would be a rewrite, which this store never
    # does — but a non-UUID install_id is not a record this build can produce.
    try:
        uuid.UUID(entry["install_id"])
    except ValueError as exc:
        raise PushRegistryCorrupt(
            f"push device registry record {index} has an invalid install_id;"
            " refusing to rewrite it"
        ) from exc
    for field in ("registered_at", "last_seen_at"):
        value = entry[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise PushRegistryCorrupt(
                f"push device registry record {index} has an invalid {field};"
                " refusing to rewrite it"
            )
    # The three state markers and the credential's own timestamp are written by
    # THIS build, so they are optional (an earlier build's record has none) and
    # validated only when present. A bool is refused alongside a non-int: it is
    # an ``int`` to Python and it is not a timestamp anybody wrote.
    for field in (
        "expired_at",
        "unpaired_at",
        "revoked_at",
        "last_authenticated_at",
        "credential_expires_at",
    ):
        if field not in entry:
            continue
        value = entry[field]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise PushRegistryCorrupt(
                f"push device registry record {index} has an invalid {field};"
                " refusing to rewrite it"
            )
    if "credential_live" in entry and not isinstance(entry["credential_live"], bool):
        raise PushRegistryCorrupt(
            f"push device registry record {index} has an invalid credential_live;"
            " refusing to rewrite it"
        )
    return dict(entry)


def _save(config_dir: Path, store: dict[str, Any]) -> None:
    """Atomic 0600 write of the whole store: temp file in the same directory, then replace.

    The replace guarantees a reader sees the old file or the new one, never a
    half-written one, and the chmod lands BEFORE the replace so the store is
    never briefly world-readable. Failures propagate: unlike the seen store,
    whose verdicts stay correct in memory, a register/revoke verdict must
    not be reported as accepted when the disk never got it.

    The WHOLE store is written, not just the records, on purpose: the machine's
    operator key lives in this file, so a rewrite that took only the device list
    would drop the key the operators-only route checks.
    """
    path = store_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{PUSH_DEVICES_STORE_NAME}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(store, handle, separators=(",", ":"))
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
