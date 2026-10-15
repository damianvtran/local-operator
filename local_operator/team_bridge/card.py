"""The HUMAN-facing view of a bridge carriage (D1).

A bridge row carries two different things and they must not be confused:

- ``details["text"]`` / ``details["body"]`` — what the MODEL receives. That is
  the wire payload: the §11.1 untrusted-content clause, the body hash, the
  locator. It is deliberately explicit because a model must be told the bytes
  are third-party.
- ``details["bridge"]`` — the ``bridge_meta`` the receiver VALIDATED against its
  own stored provenance (sender, organization, excerpt, thread). It is a hint by
  contract, never authority — but it is the only thing a human surface may show.

Contract §13.9.10/§12.1.3: no ledger row carries third-party body text, and the
render must not turn a provenance view into a body view. So the card is built
from the validated meta only, and a row whose meta lacks those fields degrades
to ONE generic sentence — never the wire text (which would put the clause and
the body hash in front of a reader), and never the wake's own sender account id,
which names this side's owner rather than the member asking (RD-1).
"""

from __future__ import annotations

from typing import Any, Mapping

#: What the card says when the validated meta named nobody. One generic sentence:
#: never the wire payload, and never a raw account id (see the module docstring).
GENERIC_CARD_SUBJECT = "an organization member"


def _first_line(value: Any) -> str:
    """One bounded line: a carriage's fields are single-line by contract, but a
    rendered frame must not be reflowed by a newline that slipped through."""
    return " ".join(str(value or "").split())


def bridge_card_subject(details: Mapping[str, Any] | None) -> str:
    """Who a reader is looking at: the VALIDATED member, else the generic line.

    Only ``details["bridge"]`` — the meta the receiver checked against its own
    stored provenance — can name a member here. It used to fall back to
    ``details["sender"]["account_id"]``, and that dict is the WAKE's own sender,
    stamped from the RECEIVING side's owner (``connector/wake.py``); a carriage
    whose display meta was absent (a shape ``screening`` explicitly tolerates as
    "unknown provenance") therefore named the reader's OWN account as the human
    asking, on the one card whose whole subject is that somebody else is. The
    sibling's docstring already said it: that dict "names the receiving side,
    never the human asking" (RD-1). A raw account id identifies nobody the
    reader can place anyway, so the absent-meta carriage degrades straight to
    :data:`GENERIC_CARD_SUBJECT`.
    """
    meta = (details or {}).get("bridge") or {}
    name = _first_line(meta.get("sender_name"))
    organization = _first_line(meta.get("organization"))
    if name and organization:
        return f"{name} · {organization}"
    return name or GENERIC_CARD_SUBJECT


def bridge_card_text(details: Mapping[str, Any] | None) -> str:
    """The card's BODY line: the screened excerpt ALONE when the identity column
    resolved a member, else the subject-prefixed form. Never the model's payload
    (see the module docstring).

    D8 (design round 2, PR #2378). The identity column sits immediately to this
    row's left and paints ``<name> (<organization>)`` from exactly the fields
    the old ``<subject> — <excerpt>`` prefix restated. On one collapsed row that
    spent the budget twice on the same two validated fields: measured 114 cells
    at 100 columns against a budget of 82, and at 60 columns the reader got the
    member twice (``Priya Raman (Acme Research) · Priya Raman…``) and no request
    at all. So the composition rule is: the identity slot carries identity, the
    body carries the excerpt, and never the reverse.

    The predicate is the identity column's OWN — ``bridge_card_sender``, which
    ``TeamBridgeBlock._sender_name`` reads — so the two surfaces cannot disagree
    about whether a member was named. When a name resolved, this returns the
    excerpt alone (the row composes ``identity · body`` in
    ``PeerMessageBlock._build_row``; an enriched carriage with no excerpt is
    therefore the identity alone, the shape every body-less peer receipt takes).

    When NO name resolved the subject is the only place a human is named at all
    (the generic sentence, or a name that arrived without an organization), so
    the old subject-prefixed form is kept byte-for-byte for that carriage — its
    pins stay true.
    """
    meta = (details or {}).get("bridge") or {}
    excerpt = _first_line(meta.get("excerpt"))
    if bridge_card_sender(details).get("name"):
        return excerpt
    subject = bridge_card_subject(details)
    return f"{subject} — {excerpt}" if excerpt else f"{subject} — (no preview available)"


def bridge_card_sender(details: Mapping[str, Any] | None) -> dict[str, Any]:
    """The identity column's fields, enriched from the validated meta.

    The wake's own sender carries the LOCAL owner account (the receiving side)
    plus the provenance kind; the member a reader needs to see is the producer's
    screened sender, which is what ``bridge_meta`` validated. The wire sender
    wins where it already names someone, so a transport that stamps a display
    name keeps it.
    """
    sender = dict((details or {}).get("sender") or {})
    meta = (details or {}).get("bridge") or {}
    for wire_key, meta_key in (("name", "sender_name"), ("organization", "organization")):
        if not sender.get(wire_key) and meta.get(meta_key):
            sender[wire_key] = _first_line(meta.get(meta_key))
    return sender
