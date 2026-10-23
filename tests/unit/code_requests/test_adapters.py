"""The forge adapters against mock transports: mapping, conditionals, errors.

Every payload here is a SYNTHESIZED shape — field subsets of the REST
responses the design's spike fetched from public repositories — so no private
comment body is committed with the tests. The live round-trips (real GitHub PR
and GitLab MR, real ETag/304 behaviour) are recorded in the PR body instead.
"""

from __future__ import annotations

import httpx
import pytest

from local_operator.code_requests.adapters import adapter_for, base, github, gitlab
from local_operator.code_requests.adapters.base import ForgeHTTPError
from local_operator.code_requests.refs import Ref, parse_any


def _ref(url: str) -> Ref:
    """A parsed ref, typed: ``parse_any`` returns ``Ref | None`` and a module
    constant narrowed by an ``assert`` does not stay narrowed inside functions,
    which is exactly where these tests consume it."""
    ref = parse_any(url)
    assert ref is not None, url
    return ref


GH = _ref("https://github.com/o/r/pull/7")
GL = _ref("https://gitlab.com/g/s/p/-/merge_requests/4")

CWD = "https://api.github.com"


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5)


def _gh_summary_payload(**overrides):
    payload = {
        "number": 7,
        "state": "closed",
        "draft": False,
        "title": "feat: a change",
        "body": "the description",
        "merged": True,
        "user": {"login": "someone"},
        "created_at": "2026-10-01T00:00:00Z",
        "updated_at": "2026-10-02T00:00:00Z",
        "head": {"ref": "feat/x", "sha": "a3ffd2b38a1c4f9d0e2b"},
        "base": {"ref": "main"},
    }
    payload.update(overrides)
    return payload


def _gh_handler(calls: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        # Match on the PATH: the adapter appends ``?per_page=100`` to every
        # list endpoint, so endswith-on-the-full-URL would never match.
        path = request.url.path
        if path.endswith("/pulls/7"):
            return httpx.Response(
                200,
                json=_gh_summary_payload(),
                headers={"ETag": 'W/"sum-1"', "X-RateLimit-Remaining": "4999"},
            )
        if path.endswith("/issues/7/comments"):
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 11,
                        "body": (
                            "### Agent review — round 2\n**Reviewer:** r\n"
                            "**Scope:** `main..a3ffd2b`\n**Verdict:** clean."
                        ),
                        "created_at": "2026-10-02T01:00:00Z",
                        "html_url": "https://x/11",
                    },
                    {
                        "id": 12,
                        "body": "just chatting",
                        "created_at": "2026-10-02T02:00:00Z",
                        "html_url": "https://x/12",
                    },
                ],
                headers={"ETag": 'W/"cmt-1"'},
            )
        if path.endswith("/pulls/7/reviews"):
            return httpx.Response(200, json=[], headers={"ETag": 'W/"rev-1"'})
        if path.endswith("/check-runs"):
            return httpx.Response(
                200,
                json={
                    "total_count": 2,
                    "check_runs": [
                        {"conclusion": "success", "name": "test", "status": "completed"},
                        {"conclusion": "failure", "name": "lint", "status": "completed"},
                    ],
                },
                headers={"ETag": 'W/"ci-1"'},
            )
        raise AssertionError(f"unexpected GitHub request {path}")

    return handler


@pytest.mark.asyncio
async def test_github_fetch_maps_summary_comments_and_ci() -> None:
    calls: list[httpx.Request] = []
    adapter = adapter_for(GH)
    assert adapter is not None
    async with _client(_gh_handler(calls)) as client:
        outcome = await adapter.fetch(GH, {}, "tok", client=client)
    pieces = outcome.pieces
    # The stored summary keeps the FORGE's own fields (raw_state/merged/draft);
    # the single ``state`` word is the service's derivation via ``state()``.
    assert pieces["summary"]["raw_state"] == "closed"
    assert pieces["summary"]["merged"] is True
    assert pieces["summary"]["head_sha"] == "a3ffd2b38a1c4f9d0e2b"
    assert adapter.state(pieces) == "merged"
    # BOTH comments are fetched (filtering to the convention is the round
    # parser's job, and it needs the non-convention ones to leave them out of
    # the lanes honestly); the adapter only normalises the shape.
    bodies = [item["body"] for item in pieces["comments"]]
    assert len(bodies) == 2
    assert adapter.comments(pieces)[1]["body"] == "just chatting"
    ci = adapter.ci(pieces)
    assert ci["status"] == "failure" and ci["total"] == 2 and ci["passed"] == 1
    assert sorted(outcome.validators) == ["ci", "comments", "reviews", "summary"]
    assert outcome.validators["summary"]["etag"] == 'W/"sum-1"'


# ---------------------------------------------------------------------------
# CI ACCOUNTING. These four shapes are the ones that made a merge gate lie:
# ``skipped`` and ``neutral`` were in _PASS_CONCLUSIONS, so a run with zero
# successes published a pass count and (with nothing pending) the word
# "success". Each shape below is a field-for-field subset of a LIVE GitHub
# check-runs response; the run ids are named where the shape was measured.
# ---------------------------------------------------------------------------


def _github_ci(*runs: tuple[str, str], total: int | None = None) -> dict:
    """The ``pieces["ci"]`` shape ``_fetch_ci`` stores, from (status, conclusion)."""
    items = [
        {"name": f"job-{i}", "status": run_status, "conclusion": conclusion}
        for i, (run_status, conclusion) in enumerate(runs)
    ]
    return {
        "ci": {
            "fetched": True,
            "runs": items,
            "total": total if total is not None else len(items),
            "url": None,
        }
    }


def test_skipped_checks_are_never_counted_as_passes() -> None:
    """(a) skipped-only -> not passed. Measured shape: radient-ml/agent-server
    run 40002268730 (2 SKIPPED, 7 queued/in_progress, ZERO success) reported
    "2 passed" from this exact payload."""
    ci = github.ADAPTER.ci(
        _github_ci(
            ("completed", "skipped"),
            ("completed", "skipped"),
            ("queued", ""),
            ("queued", ""),
            ("queued", ""),
            ("in_progress", ""),
            ("in_progress", ""),
            ("in_progress", ""),
            ("in_progress", ""),
        )
    )
    assert ci["passed"] == 0, "a skipped check is not a pass"
    assert ci["skipped"] == 2
    assert ci["pending"] == 7
    assert ci["status"] == "pending"
    assert ci["status"] != "success"


def test_queued_and_in_progress_checks_are_pending_not_passed() -> None:
    """(b) queued/in_progress -> not passed, and the payload distinguishes the
    states. The counters must PARTITION the run set (``skipped``/``neutral``
    exist as their own keys); before the fix there was no ``skipped`` key at
    all, which is how a skip could vanish into the pass count."""
    ci = github.ADAPTER.ci(
        _github_ci(
            ("queued", ""),
            ("queued", ""),
            ("queued", ""),
            ("in_progress", ""),
            ("in_progress", ""),
            ("in_progress", ""),
            ("in_progress", ""),
        )
    )
    assert ci["passed"] == 0
    assert ci["pending"] == 7
    assert ci["failed"] == 0
    assert ci["status"] == "pending"
    assert ci["skipped"] == 0 and ci["neutral"] == 0
    assert (
        ci["passed"] + ci["failed"] + ci["skipped"] + ci["neutral"] + ci["pending"]
        == ci["total"]
    )


def test_success_count_excludes_skips_in_a_mixed_run() -> None:
    """(c) mixed success+skipped -> the pass count is successes only.

    Measured shape: a 49-check run in damianvtran/local-operator with 5
    success and 39 skipped, which used to report ``44 passed``.
    """
    ci = github.ADAPTER.ci(
        _github_ci(
            *([("completed", "success")] * 5),
            *([("completed", "skipped")] * 39),
            *([("queued", "")] * 5),
        )
    )
    assert ci["passed"] == 5, "only successes are passes"
    assert ci["skipped"] == 39
    assert ci["pending"] == 5
    assert ci["status"] == "pending"
    assert (
        ci["passed"] + ci["failed"] + ci["skipped"] + ci["neutral"] + ci["pending"]
        == ci["total"]
    )

    # The settled version of the same mix: no work in flight, yet not green.
    settled = github.ADAPTER.ci(
        _github_ci(
            ("completed", "success"),
            ("completed", "success"),
            ("completed", "skipped"),
        )
    )
    assert settled["passed"] == 2
    assert settled["skipped"] == 1
    assert settled["status"] == "skipped", "a settled non-pass is not `success`"


def test_a_run_with_zero_successes_never_reaches_success() -> None:
    """(d) zero successes can never buy the success headline.

    The dangerous case is not the one with pending work (that already said
    ``pending``): it is a FULLY SETTLED run where every check was skipped,
    which used to publish ``status: success, passed: N`` with no success at
    all. Neutral and unclassifiable conclusions get the same treatment, and
    the one genuinely all-success run still passes.
    """
    all_skipped = github.ADAPTER.ci(
        _github_ci(("completed", "skipped"), ("completed", "skipped"))
    )
    assert all_skipped["passed"] == 0
    assert all_skipped["status"] == "skipped"

    all_neutral = github.ADAPTER.ci(
        _github_ci(("completed", "neutral"), ("completed", "neutral"))
    )
    assert all_neutral["passed"] == 0
    assert all_neutral["status"] == "neutral"

    # A conclusion this version does not classify is never a pass either.
    unclassified = github.ADAPTER.ci(
        _github_ci(("completed", "success"), ("completed", "cancelled"))
    )
    assert unclassified["passed"] == 1
    assert unclassified["status"] == "unknown"

    # The control: every check concluded success -> the only path to `success`.
    green = github.ADAPTER.ci(
        _github_ci(("completed", "success"), ("completed", "success"))
    )
    assert green["passed"] == 2 and green["status"] == "success"


def test_a_partial_check_run_read_is_never_a_pass() -> None:
    """``total_count`` above the one page read means the unread tail could hold
    the failure that decides the gate, so a partial read cannot be green."""
    ci = github.ADAPTER.ci(
        _github_ci(*([("completed", "success")] * 100), total=150)
    )
    assert ci["passed"] == 100
    assert ci["status"] == "unknown"


def test_a_partial_read_stays_unknown_when_page_one_carries_a_skip() -> None:
    """Review round 1, F1: ``skipped`` used to outrank the partial-read flag, so
    a page that happened to carry one skip reported a settled non-pass word and
    dropped the incomplete-read signal entirely -- permanent for any repo with
    more than 100 checks."""
    ci = github.ADAPTER.ci(
        _github_ci(
            ("completed", "skipped"),
            *([("completed", "success")] * 99),
            total=150,
        )
    )
    assert ci["passed"] == 99 and ci["skipped"] == 1
    assert ci["status"] == "unknown", "an incomplete read is not a settled state"


def test_a_fetched_but_empty_set_with_a_positive_total_is_unknown() -> None:
    """Review round 1, F3: ``runs: []`` with ``total_count: 9`` is an
    inconsistent read, not the definitive \"no checks ran\" the ``none`` word
    claims."""
    ci = github.ADAPTER.ci({"ci": {"fetched": True, "runs": [], "total": 9, "url": None}})
    assert ci["status"] == "unknown"
    assert ci["total"] == 9
    assert ci["passed"] is None, "no runs were read, so no counter may be invented"

    # The control: a genuinely empty set still gets its own word.
    empty = github.ADAPTER.ci(
        {"ci": {"fetched": True, "runs": [], "total": 0, "url": None}}
    )
    assert empty["status"] == "none" and empty["total"] == 0


@pytest.mark.asyncio
async def test_github_conditional_request_revalidates_with_304() -> None:
    calls: list[httpx.Request] = []
    adapter = adapter_for(GH)
    assert adapter is not None
    validators = {"summary": {"etag": 'W/"sum-1"'}}
    async with _client(_gh_handler(calls)) as client:
        await adapter.fetch(GH, validators, "tok", client=client)
    summary_request = next(c for c in calls if str(c.url).endswith("/pulls/7"))
    assert summary_request.headers.get("if-none-match") == 'W/"sum-1"'


@pytest.mark.asyncio
async def test_github_304_lands_in_not_modified_not_pieces() -> None:
    """A whole-pass 304 (every piece conditional): nothing replaced, all kept."""
    seen: list[str] = []
    validators = {
        "summary": {"etag": 'W/"sum-1"'},
        "comments": {"etag": 'W/"cmt-1"'},
        "reviews": {"etag": 'W/"rev-1"'},
        "ci": {"etag": 'W/"ci-1"'},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        header = request.headers.get("if-none-match", "")
        seen.append(header)
        if header:
            return httpx.Response(304)
        raise AssertionError("a stored validator must produce a conditional request")

    adapter = adapter_for(GH)
    assert adapter is not None
    async with _client(handler) as client:
        outcome = await adapter.fetch(
            GH,
            validators,
            "tok",
            # The CI piece addresses ``commits/{head_sha}/check-runs``, so a
            # stored summary must supply the sha; without it the CI leg is
            # skipped entirely (nothing to ask about) — which is itself why
            # this pass would otherwise send three conditional requests.
            stored={"summary": {"head_sha": "a3ffd2b38a1c4f9d0e2b"}},
            client=client,
        )
    assert set(outcome.not_modified) == {"summary", "comments", "reviews", "ci"}
    assert outcome.pieces == {}
    assert len(seen) == 4
    # A 304 produces NO validator update here: the stored validators survive
    # through ``cache.merge_pieces`` (tested beside it), so re-echoing them in
    # the outcome would be a second copy of the same fact.
    assert outcome.validators == {}


@pytest.mark.asyncio
async def test_github_401_is_the_unauthorized_kind() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Bad credentials"})

    adapter = adapter_for(GH)
    assert adapter is not None
    async with _client(handler) as client:
        with pytest.raises(ForgeHTTPError) as caught:
            await adapter.fetch(GH, {}, "bad", client=client)
    assert caught.value.kind == "unauthorized"


@pytest.mark.asyncio
async def test_github_403_with_zero_remaining_is_rate_limited_with_reset() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"message": "API rate limit exceeded"},
            headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "2000000000"},
        )

    adapter = adapter_for(GH)
    assert adapter is not None
    async with _client(handler) as client:
        with pytest.raises(ForgeHTTPError) as caught:
            await adapter.fetch(GH, {}, "tok", client=client)
    assert caught.value.kind == "rate_limited"
    assert caught.value.reset_at == 2000000000.0


@pytest.mark.asyncio
async def test_github_429_honours_retry_after() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "30"})

    adapter = adapter_for(GH)
    assert adapter is not None
    async with _client(handler) as client:
        with pytest.raises(ForgeHTTPError) as caught:
            await adapter.fetch(GH, {}, "tok", client=client)
    assert caught.value.kind == "rate_limited"
    assert caught.value.retry_after == 30.0


def _gl_handler(calls: list[httpx.Request]):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        path = request.url.path
        if path.endswith("/merge_requests/4"):
            return httpx.Response(
                200,
                json={
                    "id": 41,
                    "iid": 4,
                    "state": "merged",
                    "draft": False,
                    "title": "docs: a change",
                    "description": "the description",
                    "source_branch": "docs/x",
                    "target_branch": "main",
                    "sha": "c21d9a88a1c4f9d0e2b11111111111111111111",
                    "diff_refs": {"head_sha": "c21d9a88a1c4f9d0e2b11111111111111111111"},
                    "author": {"username": "someone"},
                    "created_at": "2026-10-01T00:00:00Z",
                    "updated_at": "2026-10-02T00:00:00Z",
                    "head_pipeline": {"status": "success", "web_url": "https://gl/p/1"},
                },
                headers={"ETag": 'W/"mr-1"', "RateLimit-Remaining": "1999"},
            )
        if path.endswith("/merge_requests/4/notes"):
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 21,
                        "system": True,
                        "body": "added 1 commit",
                        "created_at": "2026-10-02T01:00:00Z",
                    },
                    {
                        "id": 22,
                        "system": False,
                        "body": "### Agent review — round 3\n**Verdict:** APPROVE — TERMINAL.",
                        "created_at": "2026-10-02T02:00:00Z",
                    },
                ],
                headers={"ETag": 'W/"notes-1"'},
            )
        raise AssertionError(f"unexpected GitLab request {path}")

    return handler


@pytest.mark.asyncio
async def test_gitlab_fetch_skips_system_notes_and_reads_pipeline() -> None:
    calls: list[httpx.Request] = []
    adapter = adapter_for(GL)
    assert adapter is not None
    async with _client(_gl_handler(calls)) as client:
        outcome = await adapter.fetch(GL, {}, "tok", client=client)
    pieces = outcome.pieces
    assert pieces["summary"]["head_sha"].startswith("c21d9a88")
    assert adapter.state(pieces) == "merged"
    notes = pieces["notes"]
    # Ids are namespaced per surface (``note:22``) so a comment and a note can
    # never collide in one merged list, and the system note is dropped at
    # fetch time — it says nothing a reader needs and would only add noise to
    # the round parser's input.
    assert [item["id"] for item in notes] == ["note:22"]
    ci = adapter.ci(pieces)
    assert ci["status"] == "success" and ci["total"] is None
    assert outcome.validators["summary"]["etag"] == 'W/"mr-1"'


@pytest.mark.asyncio
async def test_gitlab_sends_conditional_and_maps_401() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(401, json={"message": "401 Unauthorized"})

    adapter = adapter_for(GL)
    assert adapter is not None
    async with _client(handler) as client:
        with pytest.raises(ForgeHTTPError) as caught:
            await adapter.fetch(GL, {"summary": {"etag": 'W/"mr-1"'}}, "tok", client=client)
    assert caught.value.kind == "unauthorized"
    assert seen[0].headers.get("if-none-match") == 'W/"mr-1"'


def test_detect_and_link_forges_have_no_adapter() -> None:
    ref = _ref("https://bitbucket.org/team/repo/pull-requests/9")
    assert ref.full is False
    assert adapter_for(ref) is None


# ---------------------------------------------------------------------------
# review round 1: F8 (rel=next stays on the API origin), F4b (a CI error is not
# flattened), QA Q3 (the pull-request probe)
# ---------------------------------------------------------------------------


def _paged_gh_handler(calls: list[httpx.Request], next_url: str):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        path = request.url.path
        if path.endswith("/pulls/7"):
            return httpx.Response(200, json=_gh_summary_payload())
        if path.endswith("/issues/7/comments"):
            return httpx.Response(
                200,
                json=[
                    {"id": i, "body": "c", "created_at": "2026-10-02T01:00:00Z", "html_url": "u"}
                    for i in range(100)
                ],
                headers={"Link": f'<{next_url}>; rel="next"'},
            )
        if path.endswith("/pulls/7/reviews"):
            return httpx.Response(200, json=[])
        if "/commits/" in path:
            return httpx.Response(200, json={"check_runs": [], "total_count": 0})
        raise AssertionError(f"unexpected path {path}")

    return handler


@pytest.mark.asyncio
async def test_a_cross_origin_next_link_is_never_followed_github() -> None:
    calls: list[httpx.Request] = []
    handler = _paged_gh_handler(calls, "https://evil.example.invalid/issues/7/comments")
    async with _client(handler) as client:
        outcome = await github.ADAPTER.fetch(GH, {}, "tok", client=client)
    assert all(
        request.url.host == "api.github.com" for request in calls
    ), "the bearer must not follow a rel=next page to another origin"
    assert len(outcome.pieces.get("comments") or []) == 100


@pytest.mark.asyncio
async def test_a_same_origin_next_link_still_paginates_github() -> None:
    calls: list[httpx.Request] = []
    handler = _paged_gh_handler(
        calls, "https://api.github.com/repos/o/r/issues/7/comments?per_page=100&page=2"
    )
    async with _client(handler) as client:
        await github.ADAPTER.fetch(GH, {}, "tok", client=client)
    assert any(request.url.params.get("page") == "2" for request in calls)


@pytest.mark.asyncio
async def test_a_cross_origin_next_link_is_never_followed_gitlab() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        path = request.url.path
        if path.endswith("/merge_requests/4"):
            return httpx.Response(
                200,
                json={
                    "iid": 4,
                    "state": "opened",
                    "title": "t",
                    "sha": "c" * 12,
                    "source_branch": "b",
                    "target_branch": "main",
                    "author": {"username": "u"},
                    "web_url": "https://gitlab.com/g/s/p/-/merge_requests/4",
                },
            )
        if path.endswith("/notes"):
            return httpx.Response(
                200,
                json=[
                    {
                        "id": i,
                        "body": "n",
                        "created_at": "2026-10-02T01:00:00Z",
                        "system": False,
                    }
                    for i in range(100)
                ],
                headers={"Link": '<https://evil.example.invalid/notes>; rel="next"'},
            )
        raise AssertionError(f"unexpected path {path}")

    async with _client(handler) as client:
        await gitlab.ADAPTER.fetch(GL, {}, "tok", client=client)
    assert all(request.url.host == "gitlab.com" for request in calls)


@pytest.mark.asyncio
async def test_a_network_error_on_ci_is_not_flattened_into_a_piece() -> None:
    """F4b: the CI failure PROPAGATES (so the pass can back off), and nothing
    in the outcome claims the CI is empty."""

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/commits/" in path:
            raise httpx.ConnectError("boom")
        if path.endswith("/pulls/7"):
            return httpx.Response(200, json=_gh_summary_payload())
        return httpx.Response(200, json=[])

    async with _client(handler) as client:
        with pytest.raises(base.ForgeHTTPError) as caught:
            await github.ADAPTER.fetch(GH, {}, "tok", client=client)
    assert caught.value.kind == "network"
    assert "check-runs" in str(caught.value.piece), "the failing endpoint is named"


@pytest.mark.asyncio
async def test_probe_pull_reads_the_verdict() -> None:
    def make(status: int):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path.endswith("/pulls/7")
            return httpx.Response(status, json={} if status == 200 else {"message": "x"})

        return handler

    async with _client(make(200)) as client:
        assert await github.ADAPTER.probe_pull(GH, "tok", client=client) == "pull"
    async with _client(make(404)) as client:
        assert await github.ADAPTER.probe_pull(GH, "tok", client=client) == "issue"
    async with _client(make(429)) as client:
        with pytest.raises(base.ForgeHTTPError) as caught:
            await github.ADAPTER.probe_pull(GH, "tok", client=client)
    assert caught.value.kind == "rate_limited"
