"""GitLab (gitlab.com and self-hosted) — the full read adapter.

TWO PIECES, and the second one is why the GitLab TTL is longer (design §D.1,
measured): a GitLab ``304`` still costs a unit of the API quota, unlike
GitHub's. Fewer endpoints and a slower refresh cadence is the response.

* ``summary`` — ``GET /projects/{id}/merge_requests/{iid}``. Carries the whole
  normalised summary including ``diff_refs.head_sha`` (what the review rounds
  are checked against) and ``head_pipeline`` — so CI is derived from THIS
  piece, not a third call. Verified live: ``head_pipeline.status`` and
  ``head_pipeline.web_url`` arrive on the MR payload.
* ``notes`` — ``/merge_requests/{iid}/notes``. System notes (``system: true``:
  "added 1 commit", "assigned to @x") are skipped at fetch time — they are
  events, not prose, and nothing in the convention parser reads them.

THE PROJECT IS URL-ENCODED. GitLab keys projects by a path that contains ``/``
(subgroups), and the REST API wants it percent-encoded
(``minervaai%2Fminerva-skills``); :func:`project_id` is the one spelling of
that conversion.

SELF-HOSTED. ``api_base(host)`` returns ``https://{host}/api/v4`` for any host
— gitlab.com is simply ``host == "gitlab.com"`` — because the REST surface is
the same on both and ``refs.py`` has already confirmed the host (the
``/-/merge_requests/N`` path shape is unique to GitLab, and for qualified refs
the host came from a remote or the configured-host list).

CI COUNTERS ARE ``None``. GitLab's MR payload carries the pipeline's status
and URL but not pass/fail counts; the counts live on the pipeline detail
endpoint, which this slice deliberately does not call (the design names
``head_pipeline.status`` on the MR as the CI source). ``None`` means "this
host did not carry it" — never a fabricated zero.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping
from urllib.parse import quote

import httpx

from local_operator.code_requests.adapters.base import (
    MAX_PAGES,
    Comment,
    FetchOutcome,
    ForgeHTTPError,
    epoch,
)

logger = logging.getLogger(__name__)

#: How long one HTTP exchange may take.
_TIMEOUT_S = 15.0

#: ``head_pipeline.status`` -> the normalised status. Anything unrecognised maps
#: to ``pending`` (a pipeline in a state we do not know is not "done"), and the
#: raw value rides along on the CI payload for a reader that wants it.
_SUCCESS_STATUSES = frozenset({"success"})
_FAILURE_STATUSES = frozenset({"failed", "canceled"})
_PENDING_STATUSES = frozenset(
    {
        "created",
        "waiting_for_resource",
        "preparing",
        "pending",
        "running",
        "scheduled",
        "manual",
        "blocked",
        "canceling",
    }
)


def api_base(host: str) -> str:
    """The REST origin for ``host``; gitlab.com included (its REST is identical)."""
    return f"https://{host}/api/v4"


def project_id(project: str) -> str:
    """GitLab's percent-encoded project path (``group/sub/proj`` -> ``group%2Fsub%2Fproj``)."""
    return quote(project, safe="")


def _headers(token: str, validators: Mapping[str, str] | None = None) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "local-operator-code-requests",
        "Authorization": f"Bearer {token}",
    }
    if validators:
        etag = validators.get("etag")
        last_modified = validators.get("last_modified")
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
    return headers


def _validators(response: httpx.Response) -> dict[str, str]:
    out: dict[str, str] = {}
    etag = response.headers.get("etag")
    last_modified = response.headers.get("last-modified")
    if etag:
        out["etag"] = etag
    if last_modified:
        out["last_modified"] = last_modified
    return out


def _raise_for_status(response: httpx.Response, piece: str) -> None:
    code = response.status_code
    if 200 <= code < 300:
        return
    headers = response.headers
    if code == 401:
        raise ForgeHTTPError("unauthorized", piece, code)
    if code == 404:
        # A missing MR or a project this token cannot see. Row-level: keep the
        # last known data, say so on the row.
        raise ForgeHTTPError("not_found", piece, code)
    if code == 429 or (
        code == 403
        and (
            str(headers.get("ratelimit-remaining") or "") == "0"
            or "rate limit" in response.text[:2048].lower()
        )
    ):
        raise ForgeHTTPError(
            "rate_limited",
            piece,
            code,
            reset_at=_reset_at(headers),
            retry_after=_retry_after(headers),
        )
    if code == 403:
        raise ForgeHTTPError("forbidden", piece, code)
    if code >= 500:
        raise ForgeHTTPError("server", piece, code)
    raise ForgeHTTPError("http", piece, code)


def _reset_at(headers: Mapping[str, str]) -> float | None:
    """Epoch seconds from ``RateLimit-Reset``/``Ratelimit-Reset`` when present."""
    for name in ("ratelimit-reset", "x-ratelimit-reset"):
        raw = headers.get(name)
        if not raw:
            continue
        try:
            return float(str(raw).strip())
        except ValueError:
            continue
    return None


def _retry_after(headers: Mapping[str, str]) -> float | None:
    """Seconds from ``Retry-After`` (delta-seconds only; an HTTP-date is rare here)."""
    raw = headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def _origin_of(url: httpx.URL) -> tuple[str, str, int]:
    """``(scheme, host, port)`` — the pin a followed ``Link`` must match."""
    return (url.scheme, str(url.host or ""), int(url.port or 0))


def _next_link(response: httpx.Response) -> str | None:
    """The ``rel="next"`` URL from a ``Link`` header, or ``None``.

    Pinned to the response's own origin, like the GitHub twin: a followed Link
    is a fresh request that carries the bearer, so a cross-origin one must
    stop pagination rather than leak the token (review round 1, F8).
    """
    link = response.headers.get("link")
    if not link:
        return None
    origin = _origin_of(response.request.url) if response.request is not None else None
    for part in link.split(","):
        if 'rel="next"' in part.replace(" ", ""):
            target = part.split(";")[0].strip().strip("<>")
            if not target.startswith("http"):
                continue
            if origin is None or _origin_of(httpx.URL(target)) != origin:
                logger.warning("code-requests: ignored a rel=next page outside the API origin")
                return None
            return target
    return None


class GitLabAdapter:
    """The GitLab singleton: two pieces, MR-derived CI, system notes skipped."""

    kind = "gitlab"
    full = True
    pieces: tuple[str, ...] = ("summary", "notes")

    def state(self, pieces: Mapping[str, Any]) -> str:
        summary = pieces.get("summary")
        if not isinstance(summary, Mapping):
            # Only reachable for a stored entry with no summary (a partial
            # fetch); the route renders link-only rows for those, never a state.
            return "open"
        if str(summary.get("raw_state") or "") == "merged" or summary.get("merged"):
            return "merged"
        if str(summary.get("raw_state") or "opened") in ("closed", "locked"):
            return "closed"
        if summary.get("draft"):
            return "draft"
        return "open"

    def ci(self, pieces: Mapping[str, Any]) -> dict[str, Any]:
        summary = pieces.get("summary")
        pipeline = summary.get("head_pipeline") if isinstance(summary, Mapping) else None
        if not isinstance(pipeline, Mapping) or not pipeline.get("status"):
            return {
                "status": "none",
                "passed": None,
                "failed": None,
                "pending": None,
                "total": None,
                "url": None,
            }
        raw = str(pipeline.get("status") or "unknown")
        if raw in _SUCCESS_STATUSES:
            status = "success"
        elif raw in _FAILURE_STATUSES:
            status = "failure"
        elif raw in _PENDING_STATUSES:
            status = "pending"
        else:
            status = "unknown"
        return {
            "status": status,
            "passed": None,
            "failed": None,
            "pending": None,
            "total": None,
            "url": str(pipeline.get("web_url") or "") or None,
            "raw_status": raw,
        }

    def comments(self, pieces: Mapping[str, Any]) -> list[dict[str, Any]]:
        notes = pieces.get("notes")
        if not isinstance(notes, list):
            return []
        return [dict(item) for item in notes if isinstance(item, Mapping)]

    async def fetch(
        self,
        ref: Any,
        validators: Mapping[str, Mapping[str, str]],
        token: str,
        *,
        stored: Mapping[str, Any] | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> FetchOutcome:
        own_client = client is None
        http = client or httpx.AsyncClient(timeout=_TIMEOUT_S, follow_redirects=True)
        outcome = FetchOutcome()
        base = api_base(ref.host)
        try:
            await self._fetch_summary(http, base, ref, validators, token, outcome)
            await self._fetch_notes(http, base, ref, validators, token, outcome)
        finally:
            if own_client:
                await http.aclose()
        return outcome

    async def _get(
        self,
        http: httpx.AsyncClient,
        url: str,
        token: str,
        validators: Mapping[str, str] | None,
    ) -> httpx.Response:
        try:
            return await http.get(url, headers=_headers(token, validators))
        except httpx.HTTPError as exc:
            raise ForgeHTTPError("network", url, None) from exc

    async def _fetch_summary(
        self,
        http: httpx.AsyncClient,
        base: str,
        ref: Any,
        validators: Mapping[str, Mapping[str, str]],
        token: str,
        outcome: FetchOutcome,
    ) -> None:
        url = f"{base}/projects/{project_id(ref.project)}/merge_requests/{ref.number}"
        response = await self._get(http, url, token, validators.get("summary"))
        if response.status_code == 304:
            outcome.not_modified.add("summary")
            return
        _raise_for_status(response, "summary")
        raw = response.json()
        if not isinstance(raw, Mapping):
            raise ForgeHTTPError("http", "summary", response.status_code, "unexpected body")
        diff_refs = raw.get("diff_refs")
        head_sha = ""
        if isinstance(diff_refs, Mapping):
            head_sha = str(diff_refs.get("head_sha") or "")
        if not head_sha:
            head_sha = str(raw.get("sha") or "")
        pipeline = raw.get("head_pipeline")
        summary: dict[str, Any] = {
            "state": None,  # filled by the service from ``state()``
            "raw_state": str(raw.get("state") or ""),
            "draft": bool(raw.get("draft") or raw.get("work_in_progress")),
            "title": str(raw.get("title") or ""),
            "body": str(raw.get("description") or ""),
            "head_sha": head_sha,
            "head_ref": str(raw.get("source_branch") or ""),
            "base_ref": str(raw.get("target_branch") or ""),
            "author": (
                str((raw.get("author") or {}).get("username") or "")
                if isinstance(raw.get("author"), Mapping)
                else ""
            ),
            "created_at": epoch(raw.get("created_at")),
            "updated_at": epoch(raw.get("updated_at")),
            "merged_at": epoch(raw.get("merged_at")),
            "head_pipeline": {
                "status": (
                    str((pipeline or {}).get("status") or "")
                    if isinstance(pipeline, Mapping)
                    else ""
                ),
                "web_url": (
                    str((pipeline or {}).get("web_url") or "")
                    if isinstance(pipeline, Mapping)
                    else ""
                ),
            },
        }
        outcome.pieces["summary"] = summary
        outcome.validators["summary"] = _validators(response)

    async def _fetch_notes(
        self,
        http: httpx.AsyncClient,
        base: str,
        ref: Any,
        validators: Mapping[str, Mapping[str, str]],
        token: str,
        outcome: FetchOutcome,
    ) -> None:
        url: str | None = (
            f"{base}/projects/{project_id(ref.project)}/merge_requests/{ref.number}"
            f"/notes?per_page=100&sort=asc&order_by=created_at"
        )
        items: list[dict[str, Any]] = []
        first = True
        pages = 0
        while url and pages < MAX_PAGES:
            response = await self._get(http, url, token, validators.get("notes") if first else None)
            if response.status_code == 304 and first:
                outcome.not_modified.add("notes")
                return
            _raise_for_status(response, "notes")
            payload = response.json()
            if not isinstance(payload, list):
                raise ForgeHTTPError("http", "notes", response.status_code, "unexpected body")
            for item in payload:
                if not isinstance(item, Mapping) or item.get("system"):
                    # System notes are events ("added 1 commit"), not prose.
                    continue
                note = Comment(
                    id=f"note:{item.get('id')}",
                    created_at=epoch(item.get("created_at")),
                    body=str(item.get("body") or ""),
                    url=(
                        str((item.get("_links") or {}).get("note") or "")
                        if isinstance(item.get("_links"), Mapping)
                        else ""
                    ),
                    source="note",
                )
                items.append(note.to_payload())
            if first:
                outcome.validators["notes"] = _validators(response)
            url = _next_link(response) if len(payload) >= 100 else None
            first = False
            pages += 1
        outcome.pieces["notes"] = items


#: The registry's singleton.
ADAPTER = GitLabAdapter()
