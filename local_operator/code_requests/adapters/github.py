"""GitHub (github.com and GHES) — the full read adapter.

FOUR PIECES, each conditional:

* ``summary`` — ``GET /repos/{project}/pulls/{n}``: state, draft, title, body,
  head/base refs and sha, author, timestamps.
* ``comments`` — ``.../issues/{n}/comments``: the issue-thread comments (where
  the round convention lives for most PRs).
* ``reviews`` — ``.../pulls/{n}/reviews``: review submissions whose bodies carry
  the convention (bodies only; a bodyless approval says nothing a parser reads).
* ``ci`` — ``.../commits/{head_sha}/check-runs``: the check runs for the PR's
  head commit.

WHY CHECK-RUNS AND NOT STATUSES. The convention's own PRs are all checks-based
(GitHub Actions); the legacy combined-status endpoint answers a different
question (commit contexts, not workflow jobs) and reading both would count one
CI twice. The design fixes check-runs, and the counters below come straight off
its ``total_count`` and per-run ``status``/``conclusion`` fields.

THE HEAD SHA IS LOAD-BEARING for the CI piece, so a ``304`` on ``summary``
still yields one: the check-runs URL is built from the STORED summary's
``head_sha`` when the fresh response was revalidated — which is why ``fetch``
takes the stored pieces, not just the validators.

GHES. :func:`api_base` returns ``https://api.github.com`` for github.com and
``https://{host}/api/v3`` for a self-hosted instance — one code path, two
origins, because the REST surface is the same.

API-VERSION PIN. ``X-GitHub-Api-Version`` is pinned to the version the shapes
below were verified against (2022-11-28); when the pin moves, the fields read
below are what to re-verify against a live response.

SOFT PIECES. CI is the row's garnish, not its identity: a 404 (checks disabled)
or a refused call degrades the CI piece to "not fetched" rather than failing a
refresh that already has the summary and comments. The summary and comments
piece failures are hard — the caller keeps the last known data and says so.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import httpx

from local_operator.code_requests.adapters.base import (
    MAX_PAGES,
    Comment,
    FetchOutcome,
    ForgeHTTPError,
    epoch,
    merge_comments,
)

logger = logging.getLogger(__name__)

#: How long one HTTP exchange may take. A read-only background fetch; the
#: service serializes attempts, so a slow host costs its own attempt only.
_TIMEOUT_S = 15.0

#: Check-run conclusions that end red in GitHub's own UI.
_FAILURE_CONCLUSIONS = frozenset({"failure", "timed_out", "startup_failure", "action_required"})
#: Conclusions that mean "completed and not failing".
_PASS_CONCLUSIONS = frozenset({"success", "skipped", "neutral"})
#: Statuses that mean "not finished yet".
_PENDING_STATUSES = frozenset({"queued", "in_progress", "waiting", "pending", "requested"})


def api_base(host: str) -> str:
    """The REST origin for ``host``: public API, or GHES's ``/api/v3``."""
    return "https://api.github.com" if host == "github.com" else f"https://{host}/api/v3"


def _headers(token: str, validators: Mapping[str, str] | None = None) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
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
    """The validators a response offered, for the next conditional request."""
    out: dict[str, str] = {}
    etag = response.headers.get("etag")
    last_modified = response.headers.get("last-modified")
    if etag:
        out["etag"] = etag
    if last_modified:
        out["last_modified"] = last_modified
    return out


def _rate_pause(headers: Mapping[str, str]) -> tuple[float | None, float | None]:
    """``(reset_at, retry_after)`` from a rate-limit response, either optional.

    ``X-RateLimit-Reset`` is an epoch-seconds instant; ``Retry-After`` is
    seconds from now. The service stores cooling as an INSTANT, so the epoch
    form is passed through untouched and the seconds form is left for the
    service to add to its own clock — one conversion at the store, not two.
    """
    reset_at: float | None = None
    retry_after: float | None = None
    try:
        reset_at = float(headers.get("x-ratelimit-reset") or "")
    except ValueError:
        reset_at = None
    try:
        retry_after = max(0.0, float(headers.get("retry-after") or ""))
    except ValueError:
        retry_after = None
    return reset_at, retry_after


def _raise_for_status(response: httpx.Response, piece: str) -> None:
    code = response.status_code
    if 200 <= code < 300:
        return
    if code == 401:
        raise ForgeHTTPError("unauthorized", piece, code)
    if code == 429 or (
        code == 403
        and (
            response.headers.get("x-ratelimit-remaining") == "0"
            or "rate limit" in response.text[:2048].lower()
        )
    ):
        reset_at, retry_after = _rate_pause(response.headers)
        raise ForgeHTTPError(
            "rate_limited", piece, code, reset_at=reset_at, retry_after=retry_after
        )
    if code == 403:
        raise ForgeHTTPError("forbidden", piece, code)
    if code == 404:
        raise ForgeHTTPError("not_found", piece, code)
    if code >= 500:
        raise ForgeHTTPError("server", piece, code)
    raise ForgeHTTPError("http", piece, code)


def _origin_of(url: httpx.URL) -> tuple[str, str, int]:
    """``(scheme, host, port)`` — the pin a followed ``Link`` must match.

    ``httpx.URL.port`` resolves the scheme's default, so ``https://x/y`` and
    ``https://x:443/y`` are one origin.
    """
    return (url.scheme, str(url.host or ""), int(url.port or 0))


def _next_link(response: httpx.Response) -> str | None:
    """The ``rel="next"`` URL from a ``Link`` header, or ``None``.

    The next page is followed WITH the same bearer token, and ``httpx`` strips
    auth on cross-origin REDIRECTS only — a followed Link is a fresh request.
    A forge (or a hostile proxy in front of one) answering ``Link: …,
    <https://evil/x>`` would therefore hand our credential to another host
    (review round 1, F8). Pin the candidate to the origin of the response it
    arrived on; anything else ends pagination here, with the pages already
    read kept.
    """
    link = response.headers.get("link")
    if not link:
        return None
    origin = _origin_of(response.request.url) if response.request is not None else None
    for part in link.split(","):
        if 'rel="next"' not in part.replace(" ", ""):
            continue
        target = part.split(";")[0].strip().strip("<>")
        if not target.startswith("http"):
            continue
        if origin is None or _origin_of(httpx.URL(target)) != origin:
            logger.warning("code-requests: ignored a rel=next page outside the API origin")
            return None
        return target
    return None


class GitHubAdapter:
    """The registry's ``github`` singleton. Stateless; one instance per process."""

    kind: str = "github"
    full: bool = True
    pieces: tuple[str, ...] = ("summary", "comments", "reviews", "ci")

    # -- pure combiners (stored pieces -> normalised values) ---------------

    def state(self, pieces: Mapping[str, Any]) -> str:
        """``merged``/``closed``/``draft``/``open`` from the stored summary.

        Only ever called with a stored summary present; the fallback is
        ``open`` because a pieces-less row is link-only and never draws a
        state, and inventing ``closed`` would be the more ominous guess.
        """
        summary = pieces.get("summary")
        if not isinstance(summary, Mapping):
            return "open"
        if summary.get("merged"):
            return "merged"
        if str(summary.get("raw_state") or "open") == "closed":
            return "closed"
        if summary.get("draft"):
            return "draft"
        return "open"

    def ci(self, pieces: Mapping[str, Any]) -> dict[str, Any]:
        """The CI payload: ``{status, passed, failed, pending, total, url}``.

        ``status`` is ``none`` when checks exist but none configured (a
        definitive zero), ``unknown`` when the piece was never fetched, could
        not be read, or carried a shape this version does not classify — the
        design's "unknown is never a guess". Counters stay ``None`` when the
        host did not carry them.
        """
        raw = pieces.get("ci")
        base: dict[str, Any] = {
            "status": "unknown",
            "passed": None,
            "failed": None,
            "pending": None,
            "total": None,
            "url": None,
        }
        if not isinstance(raw, Mapping):
            return base
        base["url"] = raw.get("url")
        runs = raw.get("runs")
        if not isinstance(runs, list) or not runs:
            if raw.get("fetched"):
                # A definitive "no checks ran": red would be wrong and grey
                # would say "cannot tell", so it gets its own word.
                base.update(status="none", passed=0, failed=0, pending=0, total=0)
            return base
        passed = failed = pending = 0
        unknown = False
        for run in runs:
            if not isinstance(run, Mapping):
                unknown = True
                continue
            status = str(run.get("status") or "")
            conclusion = str(run.get("conclusion") or "")
            if status != "completed" or status in _PENDING_STATUSES:
                pending += 1
            elif conclusion in _FAILURE_CONCLUSIONS:
                failed += 1
            elif conclusion in _PASS_CONCLUSIONS:
                passed += 1
            else:
                unknown = True
        if failed:
            status = "failure"
        elif pending:
            status = "pending"
        elif unknown and not passed:
            status = "unknown"
        else:
            status = "success"
        total = raw.get("total")
        return {
            "status": status,
            "passed": passed,
            "failed": failed,
            "pending": pending,
            "total": total if isinstance(total, int) else len(runs),
            "url": base["url"],
        }

    def comments(self, pieces: Mapping[str, Any]) -> list[dict[str, Any]]:
        """Issue comments and review bodies merged, oldest first, deduped by id."""
        groups: list[list[Mapping[str, Any]]] = []
        for name in ("comments", "reviews"):
            raw = pieces.get(name)
            if isinstance(raw, list):
                groups.append([item for item in raw if isinstance(item, Mapping)])
        return merge_comments([], groups)

    # -- fetch --------------------------------------------------------------

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
        stored = stored or {}
        try:
            head_sha = ""
            stored_summary = stored.get("summary")
            if isinstance(stored_summary, Mapping):
                head_sha = str(stored_summary.get("head_sha") or "")
            summary = await self._fetch_summary(http, base, ref, validators, token, outcome)
            if summary is not None:
                head_sha = str(summary.get("head_sha") or head_sha)
            await self._fetch_list(
                http,
                base,
                f"/repos/{ref.project}/issues/{ref.number}/comments",
                "comments",
                "comment",
                validators,
                token,
                outcome,
            )
            await self._fetch_list(
                http,
                base,
                f"/repos/{ref.project}/pulls/{ref.number}/reviews",
                "reviews",
                "review",
                validators,
                token,
                outcome,
            )
            if head_sha:
                await self._fetch_ci(http, base, ref, head_sha, validators, token, outcome)
            return outcome
        finally:
            if own_client:
                await http.aclose()

    async def probe_pull(
        self, ref: Any, token: str, *, client: httpx.AsyncClient | None = None
    ) -> str:
        """Whether a qualified ref's number is a pull request: one unconditional GET.

        ``"pull"`` (200), ``"issue"`` (404 — GitHub numbers issues and pull
        requests from one sequence, so a missing pull request at N means the
        number is an issue, or invisible to this login), ``"unknown"`` for
        anything else. Rate-limit and auth outcomes raise
        :class:`ForgeHTTPError` so the caller can cool the host or degrade;
        they are not silently flattened into ``"unknown"``.

        The credential gate ran in the caller before this is reached, so the
        token only ever travels to an authenticated host.
        """
        url = f"{api_base(ref.host)}/repos/{ref.project}/pulls/{ref.number}"
        own_client = client is None
        http = client or httpx.AsyncClient(timeout=_TIMEOUT_S, follow_redirects=True)
        try:
            response = await self._get(http, url, token, None)
            if response.status_code == 200:
                return "pull"
            if response.status_code == 404:
                return "issue"
            _raise_for_status(response, "shorthand")
            return "unknown"
        finally:
            if own_client:
                await http.aclose()

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
    ) -> dict[str, Any] | None:
        """The pulls/{n} piece. Returns the fresh summary, or ``None`` on 304."""
        url = f"{base}/repos/{ref.project}/pulls/{ref.number}"
        response = await self._get(http, url, token, validators.get("summary"))
        if response.status_code == 304:
            outcome.not_modified.add("summary")
            return None
        _raise_for_status(response, "summary")
        data = response.json()
        if not isinstance(data, Mapping):
            raise ForgeHTTPError("http", "summary", response.status_code, "unexpected body")
        raw_head = data.get("head")
        raw_base = data.get("base")
        raw_user = data.get("user")
        head: Mapping[str, Any] = raw_head if isinstance(raw_head, Mapping) else {}
        base_branch: Mapping[str, Any] = raw_base if isinstance(raw_base, Mapping) else {}
        user: Mapping[str, Any] = raw_user if isinstance(raw_user, Mapping) else {}
        summary = {
            "raw_state": str(data.get("state") or "open"),
            "merged": bool(data.get("merged")),
            "draft": bool(data.get("draft")),
            "title": str(data.get("title") or ""),
            "body": str(data.get("body") or ""),
            "head_sha": str(head.get("sha") or ""),
            "head_ref": str(head.get("ref") or ""),
            "base_ref": str(base_branch.get("ref") or ""),
            "author": str(user.get("login") or ""),
            "created_at": epoch(data.get("created_at")),
            "updated_at": epoch(data.get("updated_at")),
        }
        outcome.pieces["summary"] = summary
        outcome.validators["summary"] = _validators(response)
        return summary

    async def _fetch_list(
        self,
        http: httpx.AsyncClient,
        base: str,
        path: str,
        piece: str,
        kind: str,
        validators: Mapping[str, Mapping[str, str]],
        token: str,
        outcome: FetchOutcome,
    ) -> None:
        """One paged comment list (issue comments or review bodies)."""
        separator = "&" if "?" in path else "?"
        url: str | None = f"{base}{path}{separator}per_page=100"
        items: list[dict[str, Any]] = []
        first = True
        pages = 0
        while url and pages < MAX_PAGES:
            response = await self._get(http, url, token, validators.get(piece) if first else None)
            if response.status_code == 304 and first:
                outcome.not_modified.add(piece)
                return
            _raise_for_status(response, piece)
            payload = response.json()
            if not isinstance(payload, list):
                raise ForgeHTTPError("http", piece, response.status_code, "unexpected body")
            for item in payload:
                comment = self._one_comment(item, kind)
                if comment is not None:
                    items.append(comment)
            if first:
                outcome.validators[piece] = _validators(response)
            url = _next_link(response) if len(payload) >= 100 else None
            first = False
            pages += 1
        outcome.pieces[piece] = items

    def _one_comment(self, item: Any, kind: str) -> dict[str, Any] | None:
        if not isinstance(item, Mapping):
            return None
        body = str(item.get("body") or "")
        if kind == "review" and not body.strip():
            # A bodyless approval carries nothing the round parser reads.
            return None
        created = item.get("submitted_at") if kind == "review" else item.get("created_at")
        comment = Comment(
            id=f"{kind}:{item.get('id')}",
            created_at=epoch(created),
            body=body,
            url=str(item.get("html_url") or ""),
            source=kind,
        )
        return comment.to_payload()

    async def _fetch_ci(
        self,
        http: httpx.AsyncClient,
        base: str,
        ref: Any,
        head_sha: str,
        validators: Mapping[str, Mapping[str, str]],
        token: str,
        outcome: FetchOutcome,
    ) -> None:
        url = f"{base}/repos/{ref.project}/commits/{head_sha}/check-runs?per_page=100"
        try:
            response = await self._get(http, url, token, validators.get("ci"))
        except ForgeHTTPError:
            # A NETWORK failure is not evidence about CI. Propagate it: the
            # service keeps the stored "ci" piece AND its validator, so the
            # next pass revalidates against them (review round 1, F4b/c —
            # writing `{"fetched": False}` here used to wipe a real CI and
            # leave the stale validator paired with empty data, so every
            # later 304 kept the row stuck on "not fetched"). `_fetch_one`
            # treats the propagated error as the pass's failure, which is the
            # same posture as any other piece's network error.
            raise
        if response.status_code == 304:
            outcome.not_modified.add("ci")
            return
        if response.status_code == 404:
            # Checks disabled, or a head commit this token cannot read. A
            # definitive "not fetched" beats failing the whole refresh for the
            # garnish — and it clears the old validator (the FetchOutcome
            # convention for "replaced, nothing conditional left"): without
            # that, the next pass would send the stale ETag, take a 304 for
            # the emptied piece and stay stuck on "not fetched" (review
            # round 1, F4c).
            outcome.pieces["ci"] = {"fetched": False, "runs": [], "total": None, "url": None}
            outcome.validators["ci"] = {}
            return
        _raise_for_status(response, "ci")
        data = response.json()
        runs_raw = data.get("check_runs") if isinstance(data, Mapping) else None
        runs: list[dict[str, Any]] = []
        for item in runs_raw if isinstance(runs_raw, list) else []:
            if not isinstance(item, Mapping):
                continue
            runs.append(
                {
                    "name": str(item.get("name") or ""),
                    "status": str(item.get("status") or ""),
                    "conclusion": str(item.get("conclusion") or ""),
                }
            )
        total = data.get("total_count") if isinstance(data, Mapping) else None
        outcome.pieces["ci"] = {
            "fetched": True,
            "runs": runs,
            "total": int(total) if isinstance(total, int) else len(runs),
            "url": f"https://{ref.host}/{ref.project}/pull/{ref.number}/checks",
        }
        outcome.validators["ci"] = _validators(response)


#: The registry's singleton (``adapters/__init__.py`` imports this name).
ADAPTER = GitHubAdapter()
