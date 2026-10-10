"""The forge-adapter interface: one small protocol, host-independent payloads.

WHAT AN ADAPTER IS. One module per forge family, each exposing a singleton:
``kind``/``full``/``pieces`` (identity and the endpoints it reads), the async
``fetch`` (the one I/O method), and the pure combiners ``state``/``ci``/
``comments`` that the rest of the package reads from stored pieces.

WHY PIECES. A code request is fetched from several endpoints (a GitHub PR is
one endpoint for the body, one for comments, one for reviews, one for check
runs; a GitLab MR is one endpoint that carries its pipeline). Each endpoint has
its own ETag, and a refresh can have any subset answer ``304 Not Modified`` —
the measured asymmetry (design §D.1: a GitHub 304 is free, a GitLab 304 still
costs quota) makes conditional requests the right default everywhere, but the
304 handling is PER ENDPOINT, so the adapter returns a :class:`FetchOutcome`
that names which pieces were revalidated instead of one tri-state for the whole
ref. The service combines the outcome with the stored entry: a ``304`` piece
keeps its stored data, a ``200`` piece replaces it.

NORMALISED AT FETCH TIME. Pieces are stored normalised — the field names here
are the ones the cache, the route and the tool read, so nothing downstream
needs to know which forge answered. The two exceptions are deliberate:
``raw_status`` and the CI counters are kept as the host reported them, because
"unknown" must never be invented (the design's rule) and a host that does not
carry a counter says so with ``None``.

COMMENT AUTHOR IS NOT STORED. The account is shared, so the comment's author
proves nothing (operator rule, design §B.1). ``reviewer`` text lives in the
comment BODY and is parsed by ``rounds.py``; nothing here reads or records a
login.

NOTHING HERE KNOWS ABOUT SESSIONS, CACHING POLICY OR TTLs. An adapter fetches
one ref and maps it; the service owns eligibility, cooling, backoff and
persistence. That is the boundary that lets a later PR add Gitea in one file
plus fixtures.

PINNED CONSTANTS ARE THE CAPS. Every pagination loop is bounded
(:data:`MAX_PAGES`) and every comment body is truncated only at STORAGE time
(the cache's 4 KB rule), never here: the parser reads bodies whole.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping, Protocol, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.code_requests.refs import Ref

#: How many pages of one list endpoint a fetch will walk (100 items per page on
#: both hosts when asked). Four pages is 400 comments — far beyond any review
#: thread measured here — and the bound exists so one pathological PR cannot
#: make a background refresh unbounded.
MAX_PAGES = 4

#: The normalised states, and the ONE home of their spelling. ``draft`` is a
#: state rather than a flag beside ``open`` because every reader branches on it
#: (a draft is not "awaiting review" yet), and the route's ``state`` field has
#: exactly one value at a time.
STATES: tuple[str, ...] = ("open", "draft", "merged", "closed")


@dataclass
class FetchOutcome:
    """What one ref's fetch produced: replaced pieces, 304s, and validators.

    ``pieces`` holds ONLY the endpoints whose fresh content should REPLACE the
    stored piece; ``not_modified`` names the endpoints whose stored content
    remains valid. An endpoint that answered 200 with no ETag gets an EMPTY
    validator mapping, clearing any stale one: a host that stops sending ETags
    must not make the next request claim a validator it no longer honours.
    Mutable on purpose — the adapter's private methods fill it as pieces land,
    so a partial failure can still return what did land.
    """

    pieces: dict[str, Any] = field(default_factory=dict)
    not_modified: set[str] = field(default_factory=set)
    validators: dict[str, dict[str, str]] = field(default_factory=dict)


class ForgeHTTPError(Exception):
    """One failed HTTP exchange, classified so the service can react.

    ``kind`` drives the policy: ``unauthorized`` triggers one fresh re-resolve;
    ``rate_limited`` cools the HOST until ``reset_at`` (epoch seconds) or
    ``retry_after`` seconds from now; ``server``/``network`` raise the per-KEY
    backoff; ``not_found`` is a row-level condition (keep the last known data,
    say so); ``forbidden``/``http`` are everything else. ``piece`` names which
    endpoint failed, so a store can render "comments could not be refreshed"
    rather than blaming the whole row.
    """

    def __init__(
        self,
        kind: str,
        piece: str,
        status: int | None,
        message: str = "",
        *,
        reset_at: float | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message or f"{kind} on {piece}" + (f" (HTTP {status})" if status else ""))
        self.kind = kind
        self.piece = piece
        self.status = status
        self.reset_at = reset_at
        self.retry_after = retry_after


@dataclass(frozen=True)
class Comment:
    """One normalised comment: id, when it landed, its body, and its link.

    A dataclass rather than a bare dict because the parser and the excerpt
    renderer both read it by name, and the storage round-trip (dict on disk,
    this on the wire to the parser) is the one place a typo would otherwise
    ride silently. ``to_payload``/``from_payload`` ARE that round-trip.
    """

    id: str
    created_at: float
    body: str
    url: str
    #: Which endpoint it came from (``issue``/``review`` on GitHub, ``note`` on
    #: GitLab). Kept for the excerpt renderer's provenance line only.
    source: str = "comment"

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "body": self.body,
            "url": self.url,
            "source": self.source,
        }

    @staticmethod
    def from_payload(raw: Mapping[str, Any]) -> "Comment | None":
        try:
            return Comment(
                id=str(raw["id"]),
                created_at=float(raw["created_at"]),
                body=str(raw["body"]),
                url=str(raw.get("url") or ""),
                source=str(raw.get("source") or "comment"),
            )
        except (KeyError, TypeError, ValueError):
            return None


def epoch(value: Any, *, fallback: float = 0.0) -> float:
    """An ISO-8601 timestamp as epoch seconds, or ``fallback`` when unusable.

    Both hosts return ISO strings; a missing/garbled one is not worth a failed
    fetch, and it must not read as 1970 (a "0.0" would make freshness maths
    wrong rather than absent), so the caller passes its own fallback.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return fallback
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return fallback
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


class Forge(Protocol):
    """The interface a forge adapter implements. One singleton per forge."""

    #: ``github`` | ``gitlab`` (the keys ``refs.py`` produces).
    kind: str
    #: False would mean detect-and-link only — the registry never holds one, but
    #: the field is part of the protocol so a future partial adapter can say so.
    full: bool
    #: The piece names this adapter fetches, in fetch order.
    pieces: tuple[str, ...]

    async def fetch(
        self,
        ref: Ref,
        validators: Mapping[str, Mapping[str, str]],
        token: str,
        *,
        stored: Mapping[str, Any] | None = None,
        client: Any = None,
    ) -> FetchOutcome:
        """Read one ref conditionally; raise :class:`ForgeHTTPError` on failure.

        ``validators`` maps piece name -> stored validators (``etag`` /
        ``last_modified``); a piece with stored validators is requested with
        ``If-None-Match``/``If-Modified-Since`` and its 304 lands in
        ``not_modified``. ``stored`` is the STORED pieces mapping, needed
        because a revalidated summary is not re-sent and the CI piece's URL
        depends on the stored ``head_sha``. ``client`` is an
        ``httpx.AsyncClient`` override for tests (a ``MockTransport``);
        production passes ``None`` and the adapter builds its own short-lived
        client.
        """
        ...

    def state(self, pieces: Mapping[str, Any]) -> str:
        """The row's state from stored pieces: one of :data:`STATES`."""
        ...

    def ci(self, pieces: Mapping[str, Any]) -> dict[str, Any]:
        """The combined CI payload: ``{status, passed, failed, pending, total, url}``.

        Counters are ``None`` where the host does not carry them — never a
        guessed number (the design's "unknown is never a guess" rule).
        """
        ...

    def comments(self, pieces: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Every comment piece merged, oldest first, as :meth:`Comment.to_payload`."""
        ...


def merge_comments(
    stored: Sequence[Mapping[str, Any]], fresh: Sequence[Sequence[Mapping[str, Any]]]
) -> list[dict[str, Any]]:
    """Merge comment sources into one oldest-first list, deduped by id.

    Sources order later entries first on the same id (a review comment that also
    appears as an issue comment is one fact), and ``created_at`` breaks ties so
    the parser always sees one deterministic order.
    """
    seen: dict[str, dict[str, Any]] = {}
    for group in (stored, *fresh):
        for raw in group:
            comment = Comment.from_payload(raw)
            if comment is None:
                continue
            seen[comment.id] = comment.to_payload()
    out = list(seen.values())
    out.sort(key=lambda item: (float(item.get("created_at") or 0.0), str(item.get("id") or "")))
    return out


__all__ = [
    "Comment",
    "FetchOutcome",
    "Forge",
    "ForgeHTTPError",
    "MAX_PAGES",
    "STATES",
    "epoch",
    "merge_comments",
]
