#!/usr/bin/env python3
"""Cost gate for the full suite: is fast CI green, and is this run still needed?

WHY THIS EXISTS (GH #4074). `full-suite.yml` costs ~380 runner-minutes per run
and ``ci.yml`` ~15. Before this helper a label armed on a head whose fast CI
was pending or red still paid the full price, and every merge to ``main``
started a run that the next merge made redundant. This script is the single
decision point for both, run by the ``precondition`` job BEFORE any heavy job.

WHAT IT DECIDES, by event (the job passes ``--event``):

* ``pull_request`` (a ``labeled`` arm): the newest ``ci.yml`` ``pull_request``
  run for the PR head SHA must have a ``CI Gate`` job whose latest attempt
  concluded ``success``. Pending / red / missing / ambiguous -> exit 1.
* ``workflow_dispatch`` (``train/*`` and WIDE-route branches): the newest
  ``ci.yml`` ``push``/``workflow_dispatch`` run on EXACTLY the dispatched SHA,
  branch and repository must be green. A carrier PR's ``pull_request`` run
  tests ``refs/pull/<n>/merge`` -- a different tree -- so it never counts.
* ``push`` to ``main``: no fast-CI wait. After a bounded settle, if a newer
  non-release, untagged ``main`` tip exists (every tag ref is resolved,
  annotated ones peeled) that descends from this SHA AND
  that tip has its own full-suite run, this run is ``superseded`` (heavy jobs
  skip, the aggregate reports grey). Anything unknown -> run in full.
* ``schedule`` and anything else: never gated, always full (the nightly is the
  coverage backstop).

FAIL CLOSED. Any API error, unexpected payload, pagination overflow or unbound
fork/PR association is "not green" (exit 1) for the PR/dispatch gates, and
"run in full" for the push coalescer. There is deliberately no bypass input:
the only way through is a green ``CI Gate`` on that exact head.

TRUST. The workflow downloads THIS file from `gate_base` (the live main tip the
run tests against; PR events) or the run's own SHA (other events) through the
contents API, never from the PR checkout, so a pull request cannot rewrite its
own cost gate while holding the token. The token is the job's read-only
``GITHUB_TOKEN``; this module only
issues GET requests and needs nothing beyond the standard library, so a broken
install can never decide the gate.

The runner never waits for CI: a red gate fails in seconds and waiting belongs
on the client (``--local`` mode, run by ``make fast-gate``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

CI_WORKFLOW = "ci.yml"
CI_WORKFLOW_PATH = ".github/workflows/ci.yml"
FULL_WORKFLOW = "full-suite.yml"
GATE_JOB = "CI Gate"
#: Bounded pagination: 100 items per page, never an open-ended crawl.
MAX_PAGES = 3
PER_PAGE = 100
#: Total attempts per request (1 try + retries on 5xx / connection errors).
ATTEMPTS = 3
#: Settle before the push tip comparison. SAMPLE (not fleet statistics) from
#: 2026-11 main metadata, last 100 commits / 99 gaps: 61 < 5 min, 78 < 10 min,
#: median 2 min. The settle costs ~5 runner-min on EVERY main run, including
#: one that proves to be the final tip; no net-savings figure is claimed.
DEFAULT_SETTLE_SECONDS = 300
#: Tag resolution bounds. ``matching-refs`` returns every tag in one JSON body
#: (~900 today, no ``Link``), so the page cap exists only for the day GitHub
#: starts paginating it; hitting any cap means UNKNOWN -> run in full, never a
#: half-resolved "untagged". These are API-resource bounds, not latency claims.
MAX_TAG_PAGES = 30
MAX_ANNOTATED_RESOLUTIONS = 200
MAX_TAG_CHAIN = 5
#: Wall-clock budget for ALL tag lookups, inside the job's 10 min timeout after
#: the 300 s settle. Exceeding it is UNKNOWN -> full, not a partial verdict.
TAG_BUDGET_SECONDS = 120
#: Tip-run conclusions that do not count as "the tip will be tested".
_DEAD_CONCLUSIONS = frozenset({"cancelled", "skipped", "stale", "startup_failure"})
RELEASE_SUBJECT_PREFIX = "chore(release)"

#: ``fetch(path, headers_out=None)``; ``headers_out`` receives the ``link`` header.
Fetch = Callable[..., Any]


class ApiError(RuntimeError):
    """A GitHub API call that could not be completed or understood."""


def make_fetch(api_url: str, token: str) -> Fetch:
    """GET ``path`` as JSON with bounded retries; raise ``ApiError`` otherwise."""

    def fetch(path: str, headers_out: dict[str, str] | None = None) -> Any:
        request = urllib.request.Request(
            api_url.rstrip("/") + path,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        last = "no attempt"
        for attempt in range(ATTEMPTS):
            try:
                with urllib.request.urlopen(request, timeout=20) as response:  # noqa: S310
                    if headers_out is not None:
                        headers_out["link"] = response.headers.get("Link") or ""
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code} for {path}"
                # 4xx (auth, missing, rate limit) will not heal in seconds.
                if exc.code < 500:
                    break
            except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
                last = f"{type(exc).__name__} for {path}: {exc}"
            if attempt + 1 < ATTEMPTS:
                time.sleep(1 + attempt)
        raise ApiError(last)

    return fetch


def _paged(fetch: Fetch, path: str, key: str) -> list[dict[str, Any]]:
    """Collect ``key`` items over at most ``MAX_PAGES`` pages; overflow is an error."""
    sep = "&" if "?" in path else "?"
    items: list[dict[str, Any]] = []
    for page in range(1, MAX_PAGES + 1):
        payload = fetch(f"{path}{sep}per_page={PER_PAGE}&page={page}")
        batch = payload.get(key) if isinstance(payload, dict) else None
        if not isinstance(batch, list):
            raise ApiError(f"unexpected payload for {path}")
        items.extend(b for b in batch if isinstance(b, dict))
        if len(batch) < PER_PAGE:
            return items
    raise ApiError(f"more than {MAX_PAGES * PER_PAGE} entries for {path}; refusing to guess")


def _next_link_path(link: str, api_url: str) -> str | None:
    """The ``rel="next"`` target as a path+query, or None when there is no next page.

    Only a URL on the SAME origin as the API we were configured with is
    followed: ``fetch`` sends the job token on every request, so a ``Link`` to
    any other host would be a token exfiltration primitive.
    """
    for part in link.split(","):
        pieces = part.split(";")
        if not any(p.strip().replace(" ", "") == 'rel="next"' for p in pieces[1:]):
            continue
        target = pieces[0].strip()
        if not (target.startswith("<") and target.endswith(">")):
            raise ApiError("malformed Link header")
        url, base = urllib.parse.urlsplit(target[1:-1]), urllib.parse.urlsplit(api_url)
        if (url.scheme, url.netloc) != (base.scheme, base.netloc):
            raise ApiError("Link next points off the GitHub API origin; refusing to follow")
        prefix = base.path.rstrip("/")
        path = url.path[len(prefix) :] if prefix and url.path.startswith(prefix) else url.path
        return path + (f"?{url.query}" if url.query else "")
    return None


def tags_point_at(
    fetch: Fetch, api_url: str, repo: str, targets: frozenset[str], deadline: float
) -> bool:
    """True when ANY tag (lightweight or annotated, any name) resolves to a ``targets`` commit.

    WHY NOT ``/tags?per_page=100``: that lists the newest 100 tags by version
    sort, so an old or non-semver tag on a commit was invisible and the commit
    was wrongly coalesced (repro: first 100 tags missed it). ``matching-refs``
    lists every tag ref. A lightweight tag's ref object IS the commit; an
    annotated one points at a tag object that must be peeled via
    ``/git/tags/<sha>`` until a commit appears. Each distinct tag object is
    resolved at most once for both targets. Any cap, cycle, deadline, malformed
    payload or HTTP error raises ``ApiError`` -> caller runs in full.
    """

    def check_time() -> None:
        if time.monotonic() > deadline:
            raise ApiError("tag lookup exceeded its time budget")

    path: str | None = f"/repos/{repo}/git/matching-refs/tags/"
    seen_pages: set[str] = set()
    annotated: set[str] = set()
    for _ in range(MAX_TAG_PAGES + 1):
        if path is None:
            break
        if path in seen_pages:
            raise ApiError("repeated Link page in tag listing")
        if len(seen_pages) >= MAX_TAG_PAGES:
            raise ApiError(f"tag listing exceeds {MAX_TAG_PAGES} pages")
        seen_pages.add(path)
        check_time()
        headers: dict[str, str] = {}
        refs = fetch(path, headers)
        if not isinstance(refs, list):
            raise ApiError("unexpected matching-refs payload")
        for ref in refs:
            obj = ref.get("object") if isinstance(ref, dict) else None
            if not isinstance(obj, dict) or not isinstance(obj.get("sha"), str):
                raise ApiError("malformed tag ref")
            if obj.get("type") == "commit":
                if obj["sha"] in targets:
                    return True
            elif obj.get("type") == "tag":
                annotated.add(obj["sha"])
            else:
                raise ApiError(f"tag ref of unknown type {obj.get('type')!r}")
        path = _next_link_path(headers.get("link", ""), api_url)
    else:
        raise ApiError("tag listing did not terminate")
    if len(annotated) > MAX_ANNOTATED_RESOLUTIONS:
        raise ApiError(f"more than {MAX_ANNOTATED_RESOLUTIONS} annotated tags")
    resolved: dict[str, str] = {}  # tag-object sha -> commit sha, shared by all chains
    for start in sorted(annotated):
        chain: list[str] = []
        cur = start
        while cur not in resolved:
            if cur in chain or len(chain) >= MAX_TAG_CHAIN:
                raise ApiError("annotated tag cycle or chain too deep")
            chain.append(cur)
            check_time()
            node = fetch(f"/repos/{repo}/git/tags/{cur}")
            obj = node.get("object") if isinstance(node, dict) else None
            if not isinstance(obj, dict) or not isinstance(obj.get("sha"), str):
                raise ApiError("malformed annotated tag")
            if obj.get("type") == "commit":
                resolved[cur] = obj["sha"]
            elif obj.get("type") == "tag":
                cur = obj["sha"]
            else:
                raise ApiError(f"annotated tag targets unknown type {obj.get('type')!r}")
        for link in chain:
            resolved[link] = resolved[cur]
        if resolved[cur] in targets:
            return True
    return False


def _newest(runs: list[dict[str, Any]]) -> dict[str, Any]:
    return max(runs, key=lambda r: (str(r.get("created_at")), int(r.get("id") or 0)))


def _gate_verdict(fetch: Fetch, repo: str, run: dict[str, Any]) -> tuple[bool, str]:
    """Require a finished run whose LATEST attempt has exactly one green ``CI Gate``."""
    rid, attempt = run.get("id"), run.get("run_attempt")
    if not isinstance(rid, int) or not isinstance(attempt, int):
        return False, "ci.yml run has no id/attempt"
    if run.get("status") != "completed":
        return False, f"ci.yml run {rid} is {run.get('status')}, not finished"
    # The explicit attempt endpoint: a rerun's jobs must never be mixed with
    # an older attempt's green.
    jobs = _paged(fetch, f"/repos/{repo}/actions/runs/{rid}/attempts/{attempt}/jobs", "jobs")
    gates = [j for j in jobs if j.get("name") == GATE_JOB]
    if len(gates) != 1:
        return False, f"ci.yml run {rid} attempt {attempt} has {len(gates)} `{GATE_JOB}` jobs"
    conclusion = gates[0].get("conclusion")
    if conclusion != "success":
        return (
            False,
            f"`{GATE_JOB}` on ci.yml run {rid} attempt {attempt} is {conclusion or 'pending'}",
        )
    return True, f"`{GATE_JOB}` green on ci.yml run {rid} attempt {attempt}"


def _full_name(run: dict[str, Any], field: str) -> str:
    value = run.get(field)
    return str(value.get("full_name")) if isinstance(value, dict) else ""


def _is_ci_path(run: dict[str, Any]) -> bool:
    path = str(run.get("path") or "")
    return path == CI_WORKFLOW_PATH or path.startswith(CI_WORKFLOW_PATH + "@")


def check_pull_request(
    fetch: Fetch, repo: str, pr: int, head_sha: str, head_repo: str
) -> tuple[bool, str]:
    """Fast CI must be green for THIS PR head. Unbound runs (forks) are ignored."""
    runs = _paged(
        fetch,
        f"/repos/{repo}/actions/workflows/{CI_WORKFLOW}/runs?"
        f"head_sha={head_sha}&event=pull_request",
        "workflow_runs",
    )
    bound = [
        r
        for r in runs
        if r.get("head_sha") == head_sha
        and r.get("event") == "pull_request"
        and _is_ci_path(r)
        and _full_name(r, "repository") == repo
        and _full_name(r, "head_repository") == head_repo
        # Fork runs carry an empty pull_requests list: unbound == unknown.
        and any(isinstance(p, dict) and p.get("number") == pr for p in r.get("pull_requests") or [])
    ]
    if not bound:
        return False, f"no ci.yml pull_request run bound to PR #{pr} at {head_sha}"
    return _gate_verdict(fetch, repo, _newest(bound))


def check_dispatch(fetch: Fetch, repo: str, ref_name: str, sha: str) -> tuple[bool, str]:
    """A dispatched full run needs a green ci.yml push/dispatch run on the SAME raw head."""
    runs = _paged(
        fetch,
        f"/repos/{repo}/actions/workflows/{CI_WORKFLOW}/runs?head_sha={sha}",
        "workflow_runs",
    )
    bound = [
        r
        for r in runs
        if r.get("head_sha") == sha
        and r.get("head_branch") == ref_name
        and r.get("event") in ("workflow_dispatch", "push")
        and _is_ci_path(r)
        and _full_name(r, "repository") == repo
        and _full_name(r, "head_repository") == repo
    ]
    if not bound:
        return False, f"no ci.yml push/dispatch run for {ref_name} at {sha}"
    return _gate_verdict(fetch, repo, _newest(bound))


def check_superseded(
    fetch: Fetch,
    repo: str,
    sha: str,
    settle: int,
    sleep: Callable[[float], None],
    api_url: str = "https://api.github.com",
) -> tuple[bool, str]:
    """True only when a newer, tested, non-release main tip provably replaces ``sha``.

    Every uncertainty returns ``(False, ...)`` -- run in full. Tag/release
    commits always run in full: they are what ships.
    """
    try:
        subject = str(fetch(f"/repos/{repo}/commits/{sha}")["commit"]["message"]).splitlines()[0]
        if subject.startswith(RELEASE_SUBJECT_PREFIX):
            return False, "release commit: runs in full"
        sleep(settle)
        tip = str(fetch(f"/repos/{repo}/git/ref/heads/main")["object"]["sha"])
        if tip == sha:
            return False, "still the main tip: runs in full"
        # Release tags are cut on `chore(release)` commits (checked above), but
        # any other commit may carry a tag too, and that one must run in full.
        if tags_point_at(
            fetch, api_url, repo, frozenset({sha, tip}), time.monotonic() + TAG_BUDGET_SECONDS
        ):
            return False, "tagged commit: runs in full"
        if fetch(f"/repos/{repo}/compare/{sha}...{tip}").get("status") != "ahead":
            return False, "main tip does not descend from this SHA: runs in full"
        tip_runs = _paged(
            fetch,
            f"/repos/{repo}/actions/workflows/{FULL_WORKFLOW}/runs?head_sha={tip}&event=push",
            "workflow_runs",
        )
        live = [
            r
            for r in tip_runs
            if r.get("head_sha") == tip and r.get("conclusion") not in _DEAD_CONCLUSIONS
        ]
        if not live:
            return False, f"main tip {tip} has no live full-suite run: runs in full"
        return True, f"SUPERSEDED by main tip {tip}; no full verdict for {sha}"
    except (ApiError, KeyError, IndexError, TypeError, ValueError) as exc:
        return False, f"cannot prove supersession ({exc}): running in full"


def _write(path: str | None, text: str) -> None:
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text)


def _local_token() -> str:
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    return subprocess.run(  # noqa: S603
        ["gh", "auth", "token"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    ).stdout.strip()


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast-CI precondition for the full suite")
    parser.add_argument("--event", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument(
        "--api-url", default=os.environ.get("GITHUB_API_URL", "https://api.github.com")
    )
    parser.add_argument("--pr", type=int)
    parser.add_argument("--head-sha", default="")
    parser.add_argument("--head-repo", default="")
    parser.add_argument("--ref-name", default="")
    parser.add_argument("--sha", default="")
    parser.add_argument("--ref", default="", help="full ref, e.g. refs/heads/main")
    parser.add_argument("--settle", type=int, default=DEFAULT_SETTLE_SECONDS)
    parser.add_argument("--github-output")
    parser.add_argument("--summary")
    parser.add_argument("--local", action="store_true", help="token from env or `gh auth token`")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, sleep: Callable[[float], None] = time.sleep) -> int:
    args = _parse(argv)
    try:
        token = _local_token() if args.local else os.environ.get("GITHUB_TOKEN", "")
    except (subprocess.SubprocessError, OSError):
        token = ""
    if not token:
        print("::error title=Fast-CI precondition::no GitHub token available", file=sys.stderr)
        return 1
    fetch = make_fetch(args.api_url, token)
    superseded = False
    rerun = ""
    try:
        if args.event == "pull_request":
            if not (args.pr and args.head_sha and args.head_repo):
                ok, why = False, "pull_request event without PR number/head SHA/head repo"
            else:
                ok, why = check_pull_request(
                    fetch, args.repo, args.pr, args.head_sha, args.head_repo
                )
            rerun = f"after CI Gate is green: make merge-gate PR={args.pr}"
        elif args.event == "workflow_dispatch":
            if not (args.ref_name and args.sha):
                ok, why = False, "dispatch without ref name/SHA"
            else:
                ok, why = check_dispatch(fetch, args.repo, args.ref_name, args.sha)
            rerun = (
                f"dispatch ci.yml on {args.ref_name} first: gh workflow run ci.yml "
                f"--ref {args.ref_name}; wait for CI Gate; then dispatch full-suite"
            )
        elif args.event == "push" and args.ref == "refs/heads/main" and args.sha:
            superseded, why = check_superseded(
                fetch, args.repo, args.sha, args.settle, sleep, args.api_url
            )
            ok = not superseded
        else:  # schedule, tag pushes, anything else: never gated, always full.
            ok, why = True, f"{args.event}: not gated, runs in full"
    except ApiError as exc:
        ok, why = False, f"GitHub API unavailable ({exc}); unknown is not green"
    _write(
        args.github_output,
        f"ok={'true' if ok else 'false'}\nsuperseded={'true' if superseded else 'false'}\n",
    )
    if superseded:
        line = why
    elif ok:
        line = f"FAST-CI OK: {why}"
    else:
        line = f"FAST-CI NOT GREEN: {why}. Full suite not started." + (
            f" Next: {rerun}." if rerun else ""
        )
    print(line)
    _write(args.summary, f"### Fast-CI precondition\n\n{line}\n")
    if not ok and not superseded:
        print(f"::error title=Fast-CI precondition::{line}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
