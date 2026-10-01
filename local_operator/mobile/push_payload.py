"""The push payload the machine sends, and the two keys that dedupe its emits.

Push/ack-sync S3 of ADR 0006 (``damianvtran/local-operator-mobile`` @
``b03aeb15f``, §3.2's machine-side literals and §3.4's key table). S3 is the
FREEZE: this module is the one place the core spells the two payloads and the two
keys, and ``fixtures/push/`` carries the same shapes as data, so the mobile app
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

What this module deliberately does NOT build is the APNs/FCM envelope
(``aps``/``data``) a phone receives. That object belongs to the cloud's fan-out
and is a proposal; the machine sends the ``data`` object and stops. The machine
also never sends a badge — ``aps.badge`` is rejected by §1.5 — and
``fixtures/push/payload-forbidden-fields.json`` names it as forbidden so a later edit
cannot add it here without failing a test.

The keys are frozen here too, and they are deliberately different in kind
(§3.4). A **completion** emit is keyed on the record's CONTENT, so a heal mints a
different key and idempotency cannot swallow the correction; an **attention** emit
is keyed on the machine's monotone emit sequence, because an acknowledgement is
not a heal and there is nothing to correct. Both ride one ``Idempotency-Key``
header on one route.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

#: The payload version every emit body carries (ADR §3.2). It is a wire field
#: rather than a constant on the client because the cloud's contract is
#: ``extra="forbid"``: a reader that does not know ``v`` must refuse the body
#: rather than guess at its shape.
PAYLOAD_VERSION = 1

#: The two emit types. ``type`` exists so a tap can tell a completion (deep-link
#: to the conversation) from an attention push (correct the badge, go nowhere).
TYPE_COMPLETION = "completion"
TYPE_ATTENTION = "attention"

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

#: The credential report block's row key (ADR §3.2 #2). It is S4c's shape, sent
#: as a whole block on this body; :func:`emit_body` adds it and never builds a
#: row, so the block has one builder and this module has none.
#:
#: THE NAME IS S4C'S ON PURPOSE (review round 1, M3). S4c part 1 (PR #1881) spells
#: this key as ``REPORT_DEVICES_FIELD`` in ``push_credentials`` — which owns the
#: block's rows — and one wire key with two names in two modules of one package is
#: the drift a freeze exists to prevent, so this constant takes that module's NAME
#: rather than inventing a second. It cannot be an import yet: #1881 is open, not
#: merged, and importing an unmerged module would put this branch's tests on
#: someone else's uncommitted tree. The one-line follow-up when #1881 merges is to
#: replace this assignment with ``from ...push_credentials import
#: REPORT_DEVICES_FIELD`` — the name and the value stay, so nothing else moves.
REPORT_DEVICES_FIELD = "devices"

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


def attention_emit_key(sequence: int) -> str:
    """§3.4's attention key: the machine's monotone emit SEQUENCE, not a digest.

    An acknowledgement is not a heal, so there is nothing to correct and nothing
    to key on beyond "this emit, once": the sequence is the machine's own counter,
    persisted with the cursor that produced it, and the cloud dedupes on it. The
    prefix is what makes the two key spaces disjoint by shape — see
    :data:`ATTENTION_KEY_PREFIX`.
    """
    return f"{ATTENTION_KEY_PREFIX}{sequence}"


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


def emit_body(payload: Mapping[str, Any], devices: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The §3.2 emit body: one payload plus the credential report block.

    The block is NEW on a call that already exists — the emit — so this helper
    adds it and nothing else. Its rows are built where their facts are (S4c's
    coalescer), never re-spelled here: two spellings of one wire block is the
    drift a freeze exists to prevent.
    """
    return {**payload, REPORT_DEVICES_FIELD: [dict(row) for row in devices]}
