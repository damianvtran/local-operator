"""Recording and scoping ONE display-name resolution (the delivery-side half).

WHY THIS MODULE, and what it is NOT. The resolver itself lives in
``team_bridge.types`` (:func:`~local_operator.team_bridge.types.resolve_member_by_name`)
beside ``validate_send_recipient``, because the shape rule it answers must be the
same object the tool and the wire client read. What lives HERE is everything the
resolver must not do: fetch or shape its candidate set, and write the audit row.
Neither may live inside the matcher — a matcher that recorded its own attempts
would be trusted to record them, and one that fetched its own candidates could
resolve outside the scope it was handed.

TWO FUNCTIONS, deliberately separate:

* :func:`members_from_conversation` turns ONE conversation read into the
  candidate set ``channel_post`` resolves against. The scope is the conversation
  the message lands in — the least-privilege rule of this slice. A name that
  exists in the organization but is not an active member of THAT conversation is
  not a candidate, so it cannot resolve; that is the difference between "I know
  who you mean" and "I am allowed to deliver to them".
* :func:`resolve_for_delivery` records the attempt and THEN hands back the
  member. The ordering is the whole point: the audit row is written BEFORE the
  caller can post anything, so a delivery that cannot be accounted for cannot
  happen. The refusal arm writes the same row (``resolved_account_id`` null), so
  a name that did not resolve is reconstructable too — the case a later
  misdelivery claim actually turns on.

THE AUDIT FILE follows the repo's existing append-only audit convention
(``local_operator.browser_files.audit``): one JSON line, ``O_APPEND``, ``0600``
on the file and ``0700`` on its directory, ``ts_ms`` first. It is NOT a new
store: no schema ladder, no migrations, nothing that has to be repaired — the
native store's versioned, damage-guarded schema is a deliberate and expensive
guarantee, and an accountability log is not what it is for. One divergence from
that precedent, and it is the point of this file: ``browser_files.audit`` may
never break a turn because a missing download record costs the user a keystroke,
whereas an unwritten DELIVERY record is the exact claim this feature exists to
make answerable. So a failed write raises and the directed post refuses.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .types import (
    MEMBER_KINDS,
    MemberCandidate,
    NameResolution,
    RecipientNameRefused,
    ResolutionScope,
    TeamBridgeTypeError,
    resolve_member_by_name,
)

#: One JSONL file beside the store and the outbox, in the owner's own config dir.
RESOLUTION_AUDIT_FILENAME = "recipient_resolutions.jsonl"

#: Membership states that mean "not in the room yet" (or no longer in it). The
#: hub's own vocabulary for a human invitation is ``invited``; an agent stays
#: ``invited`` until its owner consents. A candidate must be past those, because
#: resolving a name to somebody who cannot receive is a silent non-delivery —
#: the class this feature exists to make impossible. A row carrying NO state
#: field is a current member (the conversation read lists members, not invites
#: under a separate key), so absence is not treated as inactive.
INACTIVE_MEMBERSHIP_STATES = ("invited", "pending", "declined", "left", "removed", "revoked")

#: The state keys a conversation member row may carry its membership in. Kept as
#: a tuple, not a single name, so a server-side rename between ``state`` (the
#: frozen participant shape's spelling) and ``status`` does not silently turn
#: every member into a candidate. ALL present keys are consulted and ANY inactive
#: one excludes the member: a row cannot be active in one spelling and left in
#: another, and the fail-closed reading is the only one that cannot deliver to
#: somebody who is not there.
_MEMBERSHIP_STATE_KEYS = ("state", "status", "membership")

#: How many ambiguous candidates are named in one audit row: the count is always
#: exact, the list is bounded so one pathological conversation cannot grow a row
#: without limit. The refusal a model reads is bounded too, at
#: ``types.NAMED_CANDIDATES_MAX`` (tighter, because that one is billed on every
#: ambiguous call, while the row is read only when somebody asks what happened).
_AUDIT_CANDIDATE_MAX = 20


class ResolutionAuditUnavailable(TeamBridgeTypeError):
    """The resolution could not be recorded, so nothing may be delivered.

    A directed post whose recipient we cannot account for is refused rather than
    sent: the audit row is what makes a later "it went to the wrong person" claim
    answerable, and a delivery we cannot answer for is the failure mode this
    module exists to prevent. ``code`` is its own verb so the caller reads WHY the
    post did not happen instead of a transport sentence that would send it
    retrying.
    """

    code = "audit_unavailable"


def resolution_audit_path(config_dir: Path | str) -> Path:
    """The audit log's one path: ``<config_dir>/team_bridge/recipient_resolutions.jsonl``."""

    return Path(config_dir) / "team_bridge" / RESOLUTION_AUDIT_FILENAME


def _membership_is_active(member: Mapping[str, Any]) -> bool:
    """Whether a member row is past invitation. ANY present inactive state wins."""

    for key in _MEMBERSHIP_STATE_KEYS:
        state = member.get(key)
        if isinstance(state, str) and state.strip():
            if state.strip().casefold() in INACTIVE_MEMBERSHIP_STATES:
                return False
    return True


def members_from_conversation(payload: Any) -> tuple[MemberCandidate, ...]:
    """Project one conversation read into the members a name may resolve to.

    Reads ONLY what the resolver needs: ``account_id``, ``display_label``,
    ``kind`` and (for an agent) ``endpoint_id``. A row missing any part we cannot
    deliver with is skipped rather than repaired — inventing a kind or an id here
    would be the guess the resolver refuses to make one step later, and skipping
    is visible as a zero-match refusal rather than as a wrong delivery.

    The payload is the desktop/client ``conversation`` read (``members[]``, or a
    ``{"conversation": {...}}`` wrapper) — the same shape the create path already
    tolerates. Anything unreadable yields NO candidates, never a directory-wide
    fallback: a resolver handed an empty set refuses every name, which is the
    safe direction.
    """

    if not isinstance(payload, Mapping):
        return ()
    view: Mapping[str, Any] = payload
    nested = payload.get("conversation")
    if isinstance(nested, Mapping):
        view = nested
    rows = view.get("members")
    if not isinstance(rows, Iterable) or isinstance(rows, (str, bytes)):
        return ()
    candidates: list[MemberCandidate] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        account_id = row.get("account_id")
        display_label = row.get("display_label")
        kind = row.get("kind")
        if not isinstance(account_id, str) or not account_id.strip():
            continue
        if not isinstance(display_label, str) or not display_label.strip():
            continue
        if kind not in MEMBER_KINDS:
            continue
        if not _membership_is_active(row):
            continue
        endpoint_id = row.get("endpoint_id")
        candidates.append(
            MemberCandidate(
                account_id=account_id,
                display_label=display_label,
                kind=str(kind),
                endpoint_id=endpoint_id if isinstance(endpoint_id, str) else "",
            )
        )
    return tuple(candidates)


def record_resolution(
    *,
    config_dir: Path | str,
    scope: ResolutionScope,
    scope_id: str,
    resolution: NameResolution,
    outcome: str | None = None,
) -> None:
    """Append ONE resolution row, before the caller may deliver. Raises on failure.

    ``outcome`` is a CLOSED word (:data:`OUTCOMES`) and is checked here rather
    than trusted: this row is read months later by somebody reconstructing a
    misdelivery, so a sixth word invented at a call site would quietly become a
    state nobody queries for. An unknown word is a caller bug and refuses.

    The row holds what a reconstruction needs and nothing that belongs to a body:
    the requested display name (the only way to answer "who did they ask for"),
    the resolved account id or null, the matched count, the scope word and the
    scope's own id. A person's display name is personal data, which is why the
    file is created ``0600`` in a ``0700`` directory and never leaves this machine.

    ``outcome`` overrides the derived word for the ONE attempt that never got to
    classify anything: a candidate set that could not be read. Deriving an
    outcome there would have written ``no_match``, which says the members were
    consulted and nobody matched — the opposite of what happened.
    """

    word = outcome or _outcome(resolution)
    if word not in OUTCOMES:
        raise TeamBridgeTypeError(f"unknown resolution outcome {word!r}; nothing was recorded")
    row = {
        "ts_ms": int(time.time() * 1000),
        "scope": scope.name,
        "scope_id": scope_id,
        "requested": resolution.requested,
        "resolved_account_id": (
            resolution.resolved.account_id if resolution.resolved is not None else None
        ),
        "matched_count": resolution.matched_count,
        "outcome": word,
        "candidates": [
            {
                "display_label": member.display_label,
                "account_id": member.account_id,
                "kind": member.kind,
            }
            for member in resolution.matches[:_AUDIT_CANDIDATE_MAX]
        ],
    }
    path = resolution_audit_path(config_dir)
    directory = path.parent
    try:
        # Only a directory THIS writer creates is tightened to 0700: the
        # ``team_bridge`` directory is shared with the store and the outbox, and
        # re-moding a directory an earlier component created is not this log's
        # business. The ROW is what must be private, and it is created 0600
        # below whether the directory was new or not.
        created = not directory.exists()
        directory.mkdir(parents=True, exist_ok=True)
        if created:
            os.chmod(directory, 0o700)
        line = json.dumps(row, ensure_ascii=False, default=str)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            os.close(descriptor)
            raise
    except OSError as error:
        raise ResolutionAuditUnavailable(
            "the recipient resolution could not be recorded, so nothing was posted: "
            "a directed delivery that cannot be accounted for is refused "
            f"({error.__class__.__name__})"
        ) from error


#: The outcome words, closed. ``resolved``/``no_match``/``ambiguous`` are the
#: three answers the matcher can give; ``unavailable`` is the fourth state, for an
#: attempt whose candidate set could not be read at all (recorded BEFORE the
#: refusal, so "I asked for Ada and nothing happened" is on the record too).
OUTCOMES = ("resolved", "no_match", "ambiguous", "unavailable")


def _outcome(resolution: NameResolution) -> str:
    if resolution.matched_count == 1:
        return "resolved"
    return "no_match" if resolution.matched_count == 0 else "ambiguous"


def resolve_for_delivery(
    *,
    config_dir: Path | str,
    scope: ResolutionScope,
    scope_id: str,
    requested_name: Any,
    candidates: Iterable[MemberCandidate],
) -> MemberCandidate:
    """Record the attempt, then return the exactly-one member (or refuse).

    THE one call site for a delivery: it is the resolver, the audit row and the
    ordering between them, so no caller can resolve a name, post something and
    only afterwards discover it has no record. A refusal writes its row first
    too — :class:`~local_operator.team_bridge.types.RecipientNameRefused` carries
    the :class:`~local_operator.team_bridge.types.NameResolution` the row needs.

    Wiring ``send``/``reply`` to the owner's allowlist is therefore this same
    call with :data:`~local_operator.team_bridge.types.SCOPE_ALLOWLIST` and that
    scope's candidates — the sentence, the audit word and the matching rule
    arrive with it rather than being re-derived.
    """

    try:
        member = resolve_member_by_name(requested_name, candidates, scope=scope)
    except RecipientNameRefused as refusal:
        record_resolution(
            config_dir=config_dir,
            scope=scope,
            scope_id=scope_id,
            resolution=refusal.resolution,
        )
        raise
    # The row records WHAT WAS ASKED (trimmed, the form the matcher compared),
    # never the member's own label: a reconstruction has to show the caller's
    # words next to the ids, or "they asked for A and it went to B" is
    # indistinguishable from a correct resolution.
    record_resolution(
        config_dir=config_dir,
        scope=scope,
        scope_id=scope_id,
        resolution=NameResolution(requested=requested_name.strip(), matches=(member,)),
    )
    return member


__all__ = [
    "INACTIVE_MEMBERSHIP_STATES",
    "OUTCOMES",
    "RESOLUTION_AUDIT_FILENAME",
    "ResolutionAuditUnavailable",
    "members_from_conversation",
    "record_resolution",
    "resolve_for_delivery",
    "resolution_audit_path",
]
