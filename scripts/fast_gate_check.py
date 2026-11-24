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
* ``push`` to ``main`` (BATCHED, GH #4074 cut 1): an ordinary merge no longer
  pays for the full matrix. Measured 2026-11-17..23, main pushes ran the full
  suite 129 times a week (~7.5k runner-min/day) and each answer was subsumed
  by the next merge's. So an ordinary push is ``superseded`` at once (no
  settle): heavy jobs skip, the aggregate reports grey with ``BATCHED``, and
  the HOURLY schedule tests the newest tip instead. A RELEASE commit (a
  ``chore(release)`` subject, or a ``Merge pull request`` commit whose PR
  title is ``chore(release)``) and any TAGGED commit (every tag ref resolved,
  annotated ones peeled) still run in full on their exact SHA: they are what
  ships, and ``--release-check`` refuses a tag without that green run. A tag
  lookup that cannot finish is UNKNOWN -> run in full.
* ``schedule`` with ``--schedule`` equal to ``HOURLY_CRON``: the batch run. It
  tests the main tip in full UNLESS that exact SHA already has a full-suite
  run that carries a full verdict (green or red), has heavy jobs in flight, or
  is an OLDER scheduled/release run still deciding -- quiet hours then cost
  two tiny jobs, not ~380 runner-minutes. Anything unknown -> run in full.
* ``schedule`` with any other cron (the 07:17 nightly) and anything else:
  never gated, always full (the nightly is the coverage backstop).

TWO CLIENT MODES that gate nothing in CI themselves:

* ``--release-check --sha <SHA>`` (``make release-check SHA=<sha>``): exit 0
  only when a full-suite run on EXACTLY that main SHA concluded ``success``
  with the FULL matrix (both Python legs of ``test`` and ``coverage-report``,
  linux and macOS ``tui-e2e``, a green ``Merge gate (full suite)``). The
  release owner runs it before ``gh release create``. A batched/superseded
  run is ``success`` at the run level with every heavy job skipped, which is
  why the jobs are read rather than the run conclusion.
* ``--red-range``: run by the aggregate of a RED main/hourly run. Names
  ``<last green full main SHA>..<this SHA>`` and the merged PRs in it, so an
  owner can fix forward from a batch that mixed many merges. Fail-soft: it
  always exits 0 and the workflow step is ``continue-on-error``.

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
on the client (``--local`` mode, run by ``make fast-gate``). A runner-side poll
was rejected (GH #4194/#4215/#4145): every observed early arm found ci.yml
``queued`` because the shared runner pool was saturated, and ci.yml pull_request
runs took median 46 / p90 137 / max 210 min created->completed (n=140,
2026-11-17..18), so a waiting precondition would hold a runner for up to hours
while competing with the very run it waits for. Instead ``make merge-gate``
runs this check BEFORE touching the label and refuses on exit 1.

EXIT CODES: 0 green; 1 decided NOT green (pending, red, absent, ambiguous,
unbound); 3 UNKNOWN (API or token unavailable, unexpected payload). The runner
treats every non-zero as fail-closed; only the client distinguishes 3, so a
local network/API outage does not block an arm the runner will judge anyway.
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
#: The hourly batch cron. MUST equal the schedule entry in full-suite.yml (a
#: hygiene test compares them): ``github.event.schedule`` is the only thing
#: that tells the hourly batch from the ungated nightly, and a mismatch would
#: silently make the hourly ungated (full every hour) -- costly, never unsafe.
HOURLY_CRON = "41 * * * *"
#: The aggregate's check name; a run's verdict is read from this job.
MERGE_GATE_JOB = "Merge gate (full suite)"
#: The non-PR matrix axes a release-grade run must show (full-suite.yml's
#: ``python-version`` and tui-e2e ``os`` off pull_request; pinned by a test).
RELEASE_PYTHONS = ("3.12", "3.13")
RELEASE_OSES = ("ubuntu-latest", "macos-latest")
#: Events whose run tests the raw main SHA (a pull_request run tests a merge ref).
_MAIN_EVENTS = frozenset({"push", "schedule", "workflow_dispatch"})
_LIVE_STATUSES = frozenset({"queued", "in_progress", "waiting", "pending", "requested"})
#: Red-range bounds: candidate runs inspected for the last green full main run.
MAX_GREEN_CANDIDATES = 25
#: Tag resolution bounds. ``matching-refs`` returns every tag in one JSON body
#: (~900 today, no ``Link``), so the page cap exists only for the day GitHub
#: starts paginating it; hitting any cap means UNKNOWN -> run in full, never a
#: half-resolved "untagged". These are API-resource bounds, not latency claims.
MAX_TAG_PAGES = 30
MAX_ANNOTATED_RESOLUTIONS = 200
MAX_TAG_CHAIN = 5
#: Wall-clock budget for ALL tag lookups, inside the job's 10 min timeout after
#: the precondition's other work. Exceeding it is UNKNOWN -> full, not a partial verdict.
TAG_BUDGET_SECONDS = 120
RELEASE_SUBJECT_PREFIX = "chore(release)"
#: See the module docstring: 1 = decided not green, 3 = could not decide.
EXIT_NOT_GREEN = 1
EXIT_UNKNOWN = 3

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


def is_release_message(message: str) -> bool:
    """A ``chore(release)`` commit, squash-merged OR merged with a merge commit.

    WHY THE MERGE-COMMIT ARM: release bumps land both ways on main -- 0.112.6
    (23fb2955) as ``chore(release): bump ...`` but 0.112.1-0.112.4 (e.g.
    23ff4323) as ``Merge pull request #4460 from ...`` with the PR title on the
    first body line. A subject-only test misses the latter, and a missed
    release commit would be BATCHED instead of getting its own full run.
    """
    lines = [line.strip() for line in str(message).splitlines()]
    if not lines:
        return False
    if lines[0].startswith(RELEASE_SUBJECT_PREFIX):
        return True
    if lines[0].startswith("Merge pull request #"):
        body = [line for line in lines[1:] if line]
        return bool(body) and body[0].startswith(RELEASE_SUBJECT_PREFIX)
    return False


def check_main_push(
    fetch: Fetch, repo: str, sha: str, api_url: str = "https://api.github.com"
) -> tuple[bool, str]:
    """``(batched, why)``: True when an ordinary main push defers to the hourly batch.

    A release or tagged commit is NOT batched (it ships; ``--release-check``
    needs a full run on its exact SHA). If the commit cannot be read the push
    runs in full -- the batch is a saving, never a reason to lose a verdict.
    """
    try:
        message = str(fetch(f"/repos/{repo}/commits/{sha}")["commit"]["message"])
        if is_release_message(message):
            return False, "release commit: runs in full on its exact SHA"
        if tags_point_at(
            fetch, api_url, repo, frozenset({sha}), time.monotonic() + TAG_BUDGET_SECONDS
        ):
            return False, "tagged commit: runs in full"
    except (ApiError, KeyError, IndexError, TypeError, ValueError) as exc:
        return False, f"cannot classify this push ({exc}): running in full"
    return True, (
        f"BATCHED: ordinary push {sha}; the hourly batch (cron '{HOURLY_CRON}') "
        "tests the newest main tip in full, so this SHA gets no verdict of its own"
    )


def _main_runs(fetch: Fetch, repo: str, sha: str) -> list[dict[str, Any]]:
    """Full-suite runs that tested EXACTLY ``sha`` on main (never a PR merge ref)."""
    runs = _paged(
        fetch,
        f"/repos/{repo}/actions/workflows/{FULL_WORKFLOW}/runs?head_sha={sha}",
        "workflow_runs",
    )
    return [
        r
        for r in runs
        if r.get("head_sha") == sha
        and r.get("head_branch") == "main"
        and r.get("event") in _MAIN_EVENTS
        and _is_full_path(r)
        and _full_name(r, "repository") == repo
    ]


def _is_full_path(run: dict[str, Any]) -> bool:
    path = str(run.get("path") or "")
    full = f".github/workflows/{FULL_WORKFLOW}"
    return path == full or path.startswith(full + "@")


def _attempt_jobs(fetch: Fetch, repo: str, run: dict[str, Any]) -> list[dict[str, Any]]:
    """Jobs of the run's LATEST attempt only: an old attempt's green never counts."""
    rid, attempt = run.get("id"), run.get("run_attempt")
    if not isinstance(rid, int) or not isinstance(attempt, int):
        raise ApiError("full-suite run has no id/attempt")
    return _paged(fetch, f"/repos/{repo}/actions/runs/{rid}/attempts/{attempt}/jobs", "jobs")


def _matrix_jobs(jobs: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    # A SKIPPED matrix job is listed once under its bare id (`test`); an
    # expanded one as `test (3.12, 0)`. Only expanded jobs are real work.
    return [j for j in jobs if str(j.get("name") or "").startswith(prefix + " (")]


def _has_full_verdict(jobs: list[dict[str, Any]]) -> bool:
    """The suite RAN and the aggregate judged it, green or red (not cancelled/skipped)."""
    gates = [j for j in jobs if j.get("name") == MERGE_GATE_JOB]
    tests = _matrix_jobs(jobs, "test")
    return (
        len(gates) == 1
        and gates[0].get("conclusion") in ("success", "failure")
        and any(t.get("conclusion") in ("success", "failure") for t in tests)
    )


def check_hourly(fetch: Fetch, repo: str, sha: str, run_id: int) -> tuple[bool, str]:
    """``(skip, why)`` for the hourly batch on main tip ``sha``.

    Skip ONLY when another run already covers this exact SHA:
      * a completed run with a full verdict. RED counts too, deliberately: an
        hourly re-run of an unchanged red tip would spend ~380 runner-minutes an
        hour re-proving a verdict owners already have (fix forward moves the
        tip, which the next hour tests; the nightly re-runs it once a day);
      * a live run whose matrix jobs are already expanded (heavy work in flight);
      * an OLDER (lower id) live schedule or release-push run still deciding --
        each runs in full or itself defers to a real verdict. Only OLDER ones:
        two runs deciding at once must not each wait for the other.
    Anything else, including every API doubt, runs in full.
    """
    try:
        for run in sorted(_main_runs(fetch, repo, sha), key=lambda r: -int(r.get("id") or 0)):
            rid = run.get("id")
            if rid == run_id:
                continue
            status = run.get("status")
            if status == "completed":
                # A cancelled/timed-out run is no verdict even if some shards finished.
                if run.get("conclusion") in ("success", "failure") and _has_full_verdict(
                    _attempt_jobs(fetch, repo, run)
                ):
                    return True, (
                        f"HOURLY SKIP: {sha} already has a full verdict "
                        f"({run.get('conclusion')}) in full-suite run {rid}"
                    )
                continue
            if status not in _LIVE_STATUSES:
                continue
            jobs = _attempt_jobs(fetch, repo, run)
            if _matrix_jobs(jobs, "test") or _matrix_jobs(jobs, "tui-e2e"):
                return True, f"HOURLY SKIP: full-suite run {rid} is already testing {sha}"
            message = str((run.get("head_commit") or {}).get("message") or "")
            deciding = run.get("event") == "schedule" or (
                run.get("event") == "push" and is_release_message(message)
            )
            if deciding and isinstance(rid, int) and rid < run_id:
                return (
                    True,
                    f"HOURLY SKIP: older full-suite run {rid} ({run.get('event')}) covers {sha}",
                )
    except (ApiError, KeyError, IndexError, TypeError, ValueError) as exc:
        return False, f"cannot prove {sha} is covered ({exc}): running in full"
    return False, f"hourly batch: {sha} has no full run yet: runs in full"


def full_matrix_problems(jobs: list[dict[str, Any]]) -> list[str]:
    """Why ``jobs`` (one attempt) is NOT a release-grade full run; empty when it is."""
    problems: list[str] = []
    gates = [j for j in jobs if j.get("name") == MERGE_GATE_JOB]
    if len(gates) != 1 or gates[0].get("conclusion") != "success":
        problems.append(f"`{MERGE_GATE_JOB}` is not exactly one green job")
    for prefix, axis in (
        ("test", RELEASE_PYTHONS),
        ("coverage-report", RELEASE_PYTHONS),
        ("tui-e2e", RELEASE_OSES),
    ):
        expanded = _matrix_jobs(jobs, prefix)
        red = [str(j.get("name")) for j in expanded if j.get("conclusion") != "success"]
        if red:
            problems.append(f"{prefix} not green: {', '.join(sorted(red))}")
        # The first matrix value in the job name, e.g. `test (3.12, 0)` -> 3.12.
        seen = {
            str(j.get("name"))[len(prefix) + 2 :].split(",")[0].rstrip(")").strip()
            for j in expanded
        }
        missing = [value for value in axis if value not in seen]
        if missing:
            problems.append(f"{prefix} never ran for {', '.join(missing)} (skipped or batched)")
    return problems


def check_release(fetch: Fetch, repo: str, sha: str) -> tuple[bool, str]:
    """A release target needs a green FULL-matrix full-suite run on exactly ``sha``."""
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        return False, f"release target must be a full 40-hex SHA, got {sha!r}"
    runs = sorted(_main_runs(fetch, repo, sha), key=lambda r: -int(r.get("id") or 0))
    if not runs:
        return False, f"no full-suite run on main tested {sha}"
    seen: list[str] = []
    for run in runs:
        rid = run.get("id")
        if run.get("status") != "completed" or run.get("conclusion") != "success":
            seen.append(f"run {rid} {run.get('status')}/{run.get('conclusion')}")
            continue
        problems = full_matrix_problems(_attempt_jobs(fetch, repo, run))
        if not problems:
            return (
                True,
                f"full-suite run {rid} ({run.get('event')}) on {sha} is green with the full matrix",
            )
        seen.append(f"run {rid}: {'; '.join(problems)}")
    return False, f"no green full-matrix run on {sha} ({' | '.join(seen)})"


_PR_MERGE = re.compile(r"^Merge pull request #(\d+)")
_PR_SQUASH = re.compile(r"\(#(\d+)\)\s*$")


def _pr_line(commit: dict[str, Any]) -> str:
    """``#N title (sha7)`` for a first-parent commit; the raw subject when no PR is named."""
    sha7 = str(commit.get("sha") or "")[:7]
    lines = [
        line.strip() for line in str((commit.get("commit") or {}).get("message") or "").splitlines()
    ]
    subject = lines[0] if lines else ""
    merge = _PR_MERGE.match(subject)
    if merge:
        body = [line for line in lines[1:] if line]
        return f"#{merge.group(1)} {body[0] if body else subject} ({sha7})"
    squash = _PR_SQUASH.search(subject)
    if squash:
        return f"#{squash.group(1)} {_PR_SQUASH.sub('', subject).strip()} ({sha7})"
    return f"{subject} ({sha7}, no PR number)"


def red_range(fetch: Fetch, repo: str, sha: str, run_id: int) -> str:
    """The text naming ``<last green full main SHA>..<sha>`` and its merged PRs.

    Raises on any API doubt; ``main`` turns that into a fail-soft message.
    """
    runs = _paged(
        fetch,
        f"/repos/{repo}/actions/workflows/{FULL_WORKFLOW}/runs?branch=main&status=success",
        "workflow_runs",
    )
    candidates = sorted(
        (
            r
            for r in runs
            if r.get("head_branch") == "main"
            and r.get("event") in _MAIN_EVENTS
            and r.get("conclusion") == "success"
            and r.get("head_sha") != sha
            and r.get("id") != run_id
            and _full_name(r, "repository") == repo
        ),
        key=lambda r: -int(r.get("id") or 0),
    )[:MAX_GREEN_CANDIDATES]
    for run in candidates:
        # Run-level success includes BATCHED runs whose heavy jobs all skipped;
        # only a run whose suite actually ran green is a "last green".
        jobs = _attempt_jobs(fetch, repo, run)
        if not (_has_full_verdict(jobs) and not full_matrix_problems(jobs)):
            continue
        green = str(run.get("head_sha"))
        compare = fetch(f"/repos/{repo}/compare/{green}...{sha}")
        if compare.get("status") != "ahead":
            continue  # not an ancestor of this SHA (e.g. a newer green): keep looking
        by_sha = {c.get("sha"): c for c in compare.get("commits") or [] if isinstance(c, dict)}
        lines: list[str] = []
        cur: Any = sha
        while cur in by_sha and cur != green:
            commit = by_sha[cur]
            lines.append(f"- {_pr_line(commit)}")
            parents = commit.get("parents") or []
            cur = parents[0].get("sha") if parents and isinstance(parents[0], dict) else None
        partial = (
            ""
            if cur == green
            else (
                f"\n(partial: the compare API returned {len(by_sha)} of "
                f"{compare.get('total_commits')} commits; run the git command for the rest)"
            )
        )
        return (
            f"RED BATCH RANGE: {green}..{sha} ({len(lines)} first-parent commits since the "
            f"last green full main run {run.get('id')})\n"
            + "\n".join(lines)
            + partial
            + f"\nreproduce: git log --first-parent --oneline {green}..{sha}"
        )
    raise ApiError(f"no green full-matrix main run among the newest {len(candidates)} candidates")


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
    parser.add_argument(
        "--schedule",
        default="",
        help="github.event.schedule: tells the hourly batch from the nightly",
    )
    parser.add_argument("--run-id", type=int, default=0, help="this run's id (never counts itself)")
    parser.add_argument(
        "--release-check", action="store_true", help="verify a green full-matrix run on --sha"
    )
    parser.add_argument(
        "--red-range",
        action="store_true",
        help="name the merge range of a red main run (fail-soft)",
    )
    parser.add_argument("--github-output")
    parser.add_argument("--summary")
    parser.add_argument("--local", action="store_true", help="token from env or `gh auth token`")
    return parser.parse_args(argv)


def _release_main(fetch: Fetch, args: argparse.Namespace) -> int:
    """``make release-check``: fail closed; exit 0 only on a verified full green run."""
    try:
        ok, why = check_release(fetch, args.repo, args.sha)
    except (ApiError, KeyError, IndexError, TypeError, ValueError) as exc:
        print(f"RELEASE CHECK UNKNOWN: GitHub API unavailable ({exc}); do not cut the release")
        return EXIT_UNKNOWN
    if ok:
        print(f"RELEASE CHECK OK: {why}")
        return 0
    print(
        f"RELEASE CHECK FAILED: {why}. Do not run `gh release create`. Next: wait for the "
        "release commit's own full-suite push run (gh run list --workflow full-suite.yml "
        f"--commit {args.sha}); if it is red, fix forward and cut on the new bump."
    )
    return EXIT_NOT_GREEN


def _red_range_main(fetch: Fetch, args: argparse.Namespace) -> int:
    """Fail-SOFT by contract: always exit 0, so it can never change a run's colour."""
    if args.event == "pull_request" or args.ref != "refs/heads/main" or not args.sha:
        text = "red range: not a main run (a PR gate names its own head); nothing to report"
    else:
        try:
            text = red_range(fetch, args.repo, args.sha, args.run_id)
        except Exception as exc:  # noqa: BLE001 -- fail-soft is the contract
            text = (
                f"RED BATCH RANGE unavailable ({type(exc).__name__}: {exc}). Find the last "
                "green full main run with: gh run list --workflow full-suite.yml --branch main "
                f"--status success, then git log --first-parent --oneline <green>..{args.sha}"
            )
    print(text)
    try:
        _write(args.summary, f"### Red main run: merge range\n\n```\n{text}\n```\n")
    except OSError:
        pass
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    try:
        token = _local_token() if args.local else os.environ.get("GITHUB_TOKEN", "")
    except (subprocess.SubprocessError, OSError):
        token = ""
    if not token:
        if args.red_range:  # fail-soft: never colour a run over a missing token
            print("RED BATCH RANGE unavailable: no GitHub token")
            return 0
        print("::error title=Fast-CI precondition::no GitHub token available", file=sys.stderr)
        return EXIT_UNKNOWN
    fetch = make_fetch(args.api_url, token)
    if args.release_check:
        return _release_main(fetch, args)
    if args.red_range:
        return _red_range_main(fetch, args)
    superseded = False
    unknown = False
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
            superseded, why = check_main_push(fetch, args.repo, args.sha, args.api_url)
            ok = not superseded
        elif args.event == "schedule" and args.schedule == HOURLY_CRON and args.sha:
            # The hourly batch shares the `superseded` output with the push
            # path: heavy jobs skip and the aggregate is grey, never green.
            superseded, why = check_hourly(fetch, args.repo, args.sha, args.run_id)
            ok = not superseded
        else:  # the nightly, tag pushes, anything else: never gated, always full.
            ok, why = True, f"{args.event}: not gated, runs in full"
    except ApiError as exc:
        ok, why = False, f"GitHub API unavailable ({exc}); unknown is not green"
        unknown = True
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
        return EXIT_UNKNOWN if unknown else EXIT_NOT_GREEN
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
