"""The push payload the machine sends, and the two keys that dedupe its emits.

Push/ack-sync S3 of ADR 0006 (``damianvtran/local-operator-mobile`` @
``b03aeb15f``, §3.2's machine-side literals and §3.4's key table, **plus the
lane's ruling of 2026-10-01 adding the third type** — the coalesced catch-up,
which the freeze had left unstated and which was settled as a VISIBLE alert
rather than the silent attention form). S3 is the
FREEZE: this module is the one place the core spells the payloads and the keys,
and ``fixtures/push/`` carries the same shapes as data, so the mobile app
and the cloud build against one written contract instead of against prose.

Three decisions shape this module, and each avoids a named trap:

- **The payload is composed here, never by the desktop composer.** The desktop
  notifier's body is a last-assistant-line snippet when the privacy flag allows,
  and its failure text may name a provider, a model or a quota
  (``notifications/compose.py``); reusing that path would put model-written text
  into a push. §3.2 names this builder as a slice deliverable for exactly that
  reason. What the phone renders as the banner's title and body is composed
  cloud-side from the ``kind`` — which is what the ``kind`` field is FOR.

- **``kind`` is the STORE's vocabulary, not the composer's.** The composer's set
  lists ``retired`` and the gate kinds but NOT ``closed``, so a builder keyed to
  it would silently drop a real outcome. The store's five —
  ``complete``, ``error``, ``interrupted``, ``closed``, ``retired`` — are the same
  five the mobile client's ``types.ts`` renders, and :func:`completion_payload`
  refuses anything outside them rather than emitting a kind the app cannot show.

- **The attention form is not the completion form with a flag.** A tap on an
  attention push must not deep-link anywhere: it is a badge correction, so it
  carries no conversation handle, no completion token and no kind. What it does
  carry is the devices to exclude — and ``exclude`` is OPTIONAL, because a
  tick-detected change never knows who acknowledged ("absent means exclude
  nobody").

- **The digest is a THIRD form, and it is visible on purpose.** A coalesced
  catch-up spans conversations, so it can carry none of the completion form's
  record fields and cannot be keyed on any one record's content; it is a "look at
  the app" nudge whose body is a house constant plus the count, and the type is
  what tells the cloud to compose an ALERT from it rather than the silent
  ``content-available`` wake the attention form means. Collapsing it per COMPUTER
  (rather than per conversation) is what keeps a burst to one banner on a phone
  that already has several queued.

What this module deliberately does NOT build is the APNs/FCM envelope
(``aps``). The machine sends the ``alert`` the user reads and the ``data``
object, and stops: the cloud delivers that text verbatim (it never renders,
edits or rewords it for any type) and wraps it in the envelope its fan-out owns.
The machine also never sends a badge — ``aps.badge`` is rejected by §1.5 — and
``fixtures/push/payload-forbidden-fields.json`` names it as forbidden so a later edit
cannot add it here without failing a test.

**The alert text is composed HERE, from the house constants, and that is a
privacy rule rather than a formatting one** (the lane's ruling of 2026-10-01).
The only strings a push may carry are the product's own: ``APP_NAME`` for the
title, and a house constant plus the count for the body. A builder that handed
the cloud a conversation name, a transcript snippet or a provider's error line
would put model-written text on a lock screen, which is why the alert is built
by this module's own functions and why the fixtures pin its literals.

The keys are frozen here too, and they are deliberately different in kind
(§3.4). A **completion** emit is keyed on the record's CONTENT, so a heal mints a
different key and idempotency cannot swallow the correction; an **attention** emit
is keyed on the machine's monotone emit sequence, because an acknowledgement is
not a heal and there is nothing to correct. A **digest** is keyed on the id the
machine minted for it WHEN THE WINDOW CLOSED, because a set has no content to
re-derive from and a key minted at retry time would give one batch a new identity
per attempt — the cloud could not dedupe a delivery whose ``202`` was lost. All
three ride one ``Idempotency-Key`` header on one route.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

#: The credential report block's row key (ADR §3.2 #2), IMPORTED from the module
#: that owns the block rather than spelled a second time here — one wire key, one
#: home (review round 1, M3). S4c's ``push_credentials`` mints the rows and names
#: this key; :func:`emit_body` only attaches the finished block to a payload, so
#: naming the block's key is the one thing this module should not do for itself.
#:
#: Re-exported deliberately: a caller building an emit body should not have to
#: reach into the block's producer to name the block's key, and this module's tests
#: assert the imported value against the filed fixtures' literal.
from local_operator.mobile.push_credentials import REPORT_DEVICES_FIELD
from local_operator.tui.notify import APP_NAME, BODIES, BODY_COMPLETE, digest_subtitle

#: The alert object: the user-visible text the machine composes and the cloud
#: delivers verbatim. Its two keys are named here because the deny list is
#: scanned by key name and ``title``/``body`` are forbidden as PAYLOAD fields —
#: this object is the one place they may appear, holding the product's own words
#: (:func:`completion_alert` / :func:`digest_alert`) and nothing else.
ALERT_FIELD = "alert"
ALERT_TITLE_FIELD = "title"
ALERT_BODY_FIELD = "body"

#: The payload version every emit body carries (ADR §3.2). It is a wire field
#: rather than a constant on the client because the cloud's contract is
#: ``extra="forbid"``: a reader that does not know ``v`` must refuse the body
#: rather than guess at its shape.
PAYLOAD_VERSION = 1

#: The two emit types. ``type`` exists so a tap can tell a completion (deep-link
#: to the conversation) from an attention push (correct the badge, go nowhere)
#: and from a digest (a whole burst, so it names no conversation either — but it
#: is a VISIBLE alert, which is what its own type carries).
TYPE_COMPLETION = "completion"
TYPE_ATTENTION = "attention"
TYPE_DIGEST = "digest"

#: The kinds a completion emit may carry: the STORE's vocabulary (ADR §3.2).
#: Deliberately not the composer's set — see the module docstring — and spelled
#: once here so a caller cannot pass a kind the app has no rendering for.
PUSH_KINDS = ("complete", "error", "interrupted", "closed", "retired")

#: The one machine→cloud emit route (ADR §3.2 #2 — there is no second
#: spelling, and the ``devices`` report block rides this body rather than a call
#: of its own). ``{tunnel_id}`` is the connector's tunnel identity.
EMIT_ROUTE = "/v1/tunnels/{tunnel_id}/push/events"

#: The header carrying :func:`completion_emit_key` / :func:`attention_emit_key`
#: on that route. Named here so the emitter and its tests cannot diverge on the
#: casing of a header the cloud dedupes by.
IDEMPOTENCY_HEADER = "Idempotency-Key"

#: The prefix that keeps the two key spaces disjoint by SHAPE. A completion key
#: is 64 lowercase hex characters, so ``attention-<n>`` can never be mistaken for
#: one however the counters move (the S6 exit criterion: the attention key never
#: collides with a completion key).
ATTENTION_KEY_PREFIX = "attention-"


def completion_emit_key(completion_token: str, anchor_id: str, kind: str) -> str:
    """§3.4's completion key: ``sha256(completion_token ‖ anchor_id ‖ kind)``.

    ``‖`` is read as plain CONCATENATION — the three parts in that order, no
    separator and no salt — which is what the recipe notation means. The reading
    is recorded with worked digests in
    ``fixtures/push/emit-idempotency-keys.json``, because the cloud must
    derive byte-identical keys and a separator invented on one side is the one
    way this contract can split without either side noticing.

    The key is the record's CONTENT, which is exactly what the store's supersede
    rewrites, so a heal (``interrupted`` corrected to ``complete``) mints a
    DIFFERENT key: the correction is a new delivery, and idempotency cannot
    swallow it (§3.4, §3.3). That is also why the anchor is part of the recipe —
    a token alone would key two different outcomes of one turn alike.
    """
    recipe = f"{completion_token}{anchor_id}{kind}"
    return hashlib.sha256(recipe.encode("utf-8")).hexdigest()


#: The literal the digest key recipe starts with (ADR §3.4 as ruled
#: 2026-10-01): ``sha256("digest" ‖ emit_id ‖ computer)``. Spelled once, here,
#: because the cloud must derive a byte-identical key and a second spelling
#: is how the two sides split without noticing.
DIGEST_KEY_LITERAL = "digest"


def digest_emit_key(emit_id: str, computer: str) -> str:
    """A digest emit's key: ``sha256("digest" ‖ emit_id ‖ computer)``.

    ``‖`` is concatenation again — the literal, then the id the machine minted
    when the coalescing window closed, then the computer handle, with no
    separator and no salt — and the reading is recorded with a worked digest in
    ``fixtures/push/emit-idempotency-keys.json`` so both sides derive it the same
    way.

    THE ID IS THE IDENTITY, which is why the recipe uses it rather than the
    ``count`` or the member set: a set has no stable content to re-derive from
    (the members are resolved when the batch is accepted, and the count moves
    with every other conversation on the machine), while an id minted once at
    window-close is exactly what makes a retry the same emit. Minting at retry
    time instead would hand one batch a new key per attempt — the defect review
    round 1 found.
    """
    recipe = f"{DIGEST_KEY_LITERAL}{emit_id}{computer}"
    return hashlib.sha256(recipe.encode("utf-8")).hexdigest()


def attention_emit_key(sequence: int) -> str:
    """§3.4's attention key: the machine's monotone emit SEQUENCE, not a digest.

    An acknowledgement is not a heal, so there is nothing to correct and nothing
    to key on beyond "this emit, once": the sequence is the machine's own counter,
    persisted with the cursor that produced it, and the cloud dedupes on it. The
    prefix is what makes the two key spaces disjoint by shape — see
    :data:`ATTENTION_KEY_PREFIX`.
    """
    return f"{ATTENTION_KEY_PREFIX}{sequence}"


def count_phrase(count: int) -> str:
    """The body's count term: ``1 conversation needs you`` / ``5 conversations need you``.

    THE NOUN MATCHES WHAT THE COUNT IS. ``count`` is the machine's unread
    CONVERSATION count at composition time (§3.2's field table), the same number
    S1's badge is built from, so the sentence says "conversations" and never
    "updates" — a word that would invite a reader to guess at what changed.

    The singular is spelled rather than left to the template: a count of one is
    the COMMON case for a completion push, and "1 conversations need you" is
    prose no surface should ship. The lane's ruling fixes the noun and the count;
    this is the one shape of it that reads correctly at both ends.
    """
    noun = "conversation needs" if count == 1 else "conversations need"
    return f"{count} {noun} you"


def completion_alert(kind: str, count: int) -> dict[str, str]:
    """The alert for one completion: the product name, then the outcome + count.

    An unknown kind falls back to ``BODY_COMPLETE`` exactly as the composer's own
    ``BODIES.get(kind, BODY_COMPLETE)`` does — one default, and the same one the
    local banner takes, so a push and a toast never disagree about an outcome the
    vocabulary does not carry.
    """
    body = BODIES.get(kind, BODY_COMPLETE)
    return {ALERT_TITLE_FIELD: APP_NAME, ALERT_BODY_FIELD: f"{body} · {count_phrase(count)}"}


def digest_alert(kinds: Sequence[str], count: int) -> dict[str, str]:
    """The alert for a coalesced catch-up: the SET's honest state, then the count.

    The leading phrase is ``digest_subtitle``'s answer, not a fixed "Task
    complete": the batch is whatever the burst rule held back, so a set that is
    entirely failures must not be announced as complete (the TUI's own design
    round 2, D8). A uniform set gets its real category out of the house
    vocabulary, a mixed one says so, and a batch whose kinds could not be read
    falls back to the same default a single unknown kind takes.
    """
    state = digest_subtitle(kinds) or BODY_COMPLETE
    return {ALERT_TITLE_FIELD: APP_NAME, ALERT_BODY_FIELD: f"{state} · {count_phrase(count)}"}


def completion_payload(
    *,
    computer: str,
    conversation: str,
    completion_token: str,
    kind: str,
    emit_id: str,
    count: int,
) -> dict[str, Any]:
    """One completion emit's payload — the ``data`` object of ADR §3.2 #2.

    ``computer`` and ``conversation`` are the machine's OPAQUE handles (a
    per-account computer handle and the per-machine conversation handle S2
    mints), never an account id and never a session id: the handle is what makes
    the push's collapse key per conversation without naming the conversation to
    the cloud. ``count`` is the machine's unread count at composition time, the
    one aggregate the ADR discloses on purpose.

    ``kind`` is validated against the store's vocabulary rather than passed
    through: the composer's set is missing ``closed``, so an unvalidated
    pass-through would let a caller emit an outcome the app cannot render — and
    the failure would be a notification that silently says nothing.
    """
    if kind not in PUSH_KINDS:
        raise ValueError(f"unknown completion kind: {kind!r}")
    return {
        "v": PAYLOAD_VERSION,
        "type": TYPE_COMPLETION,
        "computer": computer,
        "conversation": conversation,
        "completion_token": completion_token,
        "kind": kind,
        "emit_id": emit_id,
        "count": count,
        ALERT_FIELD: completion_alert(kind, count),
    }


def attention_payload(
    *,
    computer: str,
    count: int,
    emit_id: str,
    exclude: Sequence[str] | None = None,
) -> dict[str, Any]:
    """One attention emit's payload — the badge correction of ADR §3.2.

    No ``conversation``, no ``completion_token`` and no ``kind``: the type exists
    precisely so a tap goes nowhere, and a field the app must not read is a field
    that should not be sent.

    ``exclude`` is the devices that must NOT be woken by this correction — the
    device that just acknowledged, on the nudge path — and it is OMITTED when the
    caller does not know ("absent means exclude nobody"), rather than sent empty:
    the repo's absence rule, so the cloud reads one spelling of "no exclusions"
    and not two.
    """
    payload: dict[str, Any] = {
        "v": PAYLOAD_VERSION,
        "type": TYPE_ATTENTION,
        "computer": computer,
        "count": count,
        "emit_id": emit_id,
    }
    if exclude is not None:
        payload["exclude"] = list(exclude)
    return payload


def digest_payload(
    *,
    computer: str,
    count: int,
    emit_id: str,
    kinds: Sequence[str],
    exclude: Sequence[str] | None = None,
) -> dict[str, Any]:
    """One digest emit's payload — the coalesced catch-up of ADR §2.1's burst rule.

    The same field set as the attention form, and that is not a coincidence: both
    stand for a set rather than for a record, so both carry the machine's count
    and nothing a tap could deep-link to. What differs is the TYPE, and it is
    load-bearing — §3.2 fixes the attention form's envelope as a silent,
    best-effort wake, so a burst that rode it would arrive with no banner at all.
    ``type: "digest"`` is what tells the cloud to compose an alert: a house
    constant plus the count, never a name, a snippet or an error line.

    ``exclude`` means what it means on the attention form, and it is REQUIRED when
    the digest follows an ack nudge (the device that just acted must not be told
    about its own action) — which is a rule about that caller, not about this
    builder: the tick-detected path has no device to exclude and omits the field,
    which is the one spelling of "exclude nobody".
    """
    payload: dict[str, Any] = {
        "v": PAYLOAD_VERSION,
        "type": TYPE_DIGEST,
        "computer": computer,
        "count": count,
        "emit_id": emit_id,
        ALERT_FIELD: digest_alert(kinds, count),
    }
    if exclude is not None:
        payload["exclude"] = list(exclude)
    return payload


def emit_body(payload: Mapping[str, Any], devices: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The §3.2 emit body: one payload plus the credential report block.

    The block is NEW on a call that already exists — the emit — so this helper
    adds it and nothing else. Its rows are built where their facts are (S4c's
    coalescer), never re-spelled here: two spellings of one wire block is the
    drift a freeze exists to prevent.
    """
    return {**payload, REPORT_DEVICES_FIELD: [dict(row) for row in devices]}
