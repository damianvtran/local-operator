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
to the account id (or a generic line) — it NEVER falls back to the wire text,
which would put the clause and the body hash in front of a reader.
"""

from __future__ import annotations

from typing import Any, Mapping

#: What the card says when even the account id is missing. A generic sentence,
#: never the wire payload.
GENERIC_CARD_SUBJECT = "an organization member"


def _first_line(value: Any) -> str:
    """One bounded line: a carriage's fields are single-line by contract, but a
    rendered frame must not be reflowed by a newline that slipped through."""
    return " ".join(str(value or "").split())


def bridge_card_subject(details: Mapping[str, Any] | None) -> str:
    """Who a reader is looking at: validated meta first, wire sender second."""
    meta = (details or {}).get("bridge") or {}
    wire_sender = (details or {}).get("sender") or {}
    name = _first_line(meta.get("sender_name")) or _first_line(wire_sender.get("account_id"))
    organization = _first_line(meta.get("organization"))
    if name and organization:
        return f"{name} · {organization}"
    return name or GENERIC_CARD_SUBJECT


def bridge_card_text(details: Mapping[str, Any] | None) -> str:
    """The card's body line: the subject plus the screened preview, or a stated
    absence. Never the model's payload (see the module docstring)."""
    meta = (details or {}).get("bridge") or {}
    excerpt = _first_line(meta.get("excerpt"))
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
