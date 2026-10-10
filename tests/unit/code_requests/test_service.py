"""The fetch service: eligibility, the pass, failure posture, the read merge.

The adapters are replaced with fakes here — their network behaviour is tested
in ``test_adapters`` — so every case is deterministic and offline. The live
round-trips are in the PR body.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.code_requests import cache
from local_operator.code_requests import credentials as code_credentials
from local_operator.code_requests import ledger, service
from local_operator.code_requests.adapters import base, github
from local_operator.code_requests.refs import Ref, parse_any


def _ref(url: str) -> Ref:
    """A parsed ref, typed: ``parse_any`` returns ``Ref | None`` and a module
    constant narrowed by an ``assert`` does not stay narrowed inside functions,
    which is exactly where these tests consume it."""
    ref = parse_any(url)
    assert ref is not None, url
    return ref


REF = _ref("https://github.com/o/r/pull/7")

ROW = {"key": REF.key, "ref": REF.to_payload(), "relation": "acted", "relations": ["acted"]}
LINK_ONLY_ROW = {
    "key": "bitbucket.org/t/r#9",
    "ref": {
        "key": "bitbucket.org/t/r#9",
        "forge": "bitbucket",
        "host": "bitbucket.org",
        "project": "t/r",
        "number": 9,
        "url": "https://bitbucket.org/t/r/pull-requests/9",
        "full": False,
        "reason": "Bitbucket is detect-and-link only in this slice.",
    },
    "relation": "mentioned",
    "relations": ["mentioned"],
}


@pytest.fixture(autouse=True)
def _clean():
    cache._reset_for_tests()
    yield
    cache._reset_for_tests()


@pytest.fixture(autouse=True)
def _fake_credentials(monkeypatch: pytest.MonkeyPatch):
    """A dummy token for every host, so no test can reach a real login.

    ``resolve`` is patched at the SERVICE's namespace (it imported the function
    by name). Without this, an isolated HOME has no gh/glab config and the
    service degrades to link-only before any fake fetch runs — which is the
    right PRODUCT behaviour, but not what these tests are exercising. The 401
    test overrides this patch itself, on purpose.
    """

    def fake(host, forge, *, home=None, config_dir=None, fresh=False):
        return code_credentials.Token(host=host, forge=forge, value="tok", source="test")

    monkeypatch.setattr(service, "resolve", fake)
    return fake


def _entry(checked_ago: float = 0.0, **extra: Any) -> dict[str, Any]:
    entry = {
        "pieces": {"summary": {"head_sha": "a" * 12, "title": "t"}},
        "validators": {},
        "summary": {"state": "open", "head_sha": "a" * 12},
        "state": "open",
        "ci": {"status": "pending"},
        "lanes": [],
        "checked_at": time.time() - checked_ago,
        "fetched_at": time.time() - checked_ago,
        "stale": False,
        "refresh_error": None,
    }
    entry.update(extra)
    return entry


def _outcome(pieces: dict[str, Any] | None = None, not_modified: set[str] | None = None):
    outcome = base.FetchOutcome()
    if pieces:
        outcome.pieces.update(pieces)
    if not_modified:
        outcome.not_modified.update(not_modified)
    return outcome


def _patch_fetch(monkeypatch: pytest.MonkeyPatch, fake) -> None:
    monkeypatch.setattr(github.ADAPTER, "fetch", fake)


# ---------------------------------------------------------------------------
# plan()
# ---------------------------------------------------------------------------


def test_plan_picks_missing_dirty_and_expired(tmp_path: Path) -> None:
    rows = [ROW, LINK_ONLY_ROW]
    planned = service.plan(tmp_path, "s1", rows)
    assert [ref.key for ref in planned.refs] == [REF.key]
    assert planned.reasons[REF.key] == "missing"
    # A detect-and-link row never fetches, whatever else is true.
    assert LINK_ONLY_ROW["key"] not in planned.reasons

    cache.write_entry(tmp_path, _entry(), ref=REF)
    assert service.plan(tmp_path, "s1", rows).refs == []
    assert service.plan(tmp_path, "s1", rows, force=True).reasons[REF.key] == "force"

    cache.mark_dirty(tmp_path, "s1", keys=[REF.key])
    planned = service.plan(tmp_path, "s1", rows)
    assert planned.reasons[REF.key] == "dirty"

    cache.clear_dirty(tmp_path, "s1")
    cache.write_entry(tmp_path, _entry(checked_ago=10_000), ref=REF)
    assert service.plan(tmp_path, "s1", rows).reasons[REF.key] == "expired"


def test_plan_respects_cooling_and_backoff_even_under_force(tmp_path: Path) -> None:
    rows = [ROW]
    cache.write_entry(tmp_path, _entry(checked_ago=10_000), ref=REF)
    cache.note_rate_limited(REF.host)
    planned = service.plan(tmp_path, "s1", rows, force=True)
    assert planned.refs == []
    assert REF.host in planned.cooling

    cache.note_host_success(REF.host)
    cache.note_key_failure(REF.key)
    planned = service.plan(tmp_path, "s1", rows)
    assert planned.refs == []
    assert REF.key in planned.backing_off


def test_plan_keys_filter(tmp_path: Path) -> None:
    other = parse_any("https://github.com/o/r/pull/8")
    assert other is not None
    rows = [ROW, {"key": other.key, "ref": other.to_payload()}]
    planned = service.plan(tmp_path, "s1", rows, keys=[other.key])
    assert [ref.key for ref in planned.refs] == [other.key]


# ---------------------------------------------------------------------------
# refresh_keys / refresh_session
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refresh_writes_the_derived_entry(tmp_path: Path, monkeypatch) -> None:
    async def fake_fetch(ref, validators, token, *, stored=None, client=None):
        return _outcome(
            {
                "summary": {
                    "raw_state": "closed",
                    "merged": True,
                    "draft": False,
                    "title": "t",
                    "head_sha": "a3ffd2b38a1c4f9d0e2b",
                },
                "comments": [
                    {
                        "id": "comment:1",
                        "body": (
                            "### Agent review — round 2\n"
                            "**Scope:** `main..a3ffd2b`\n**Verdict:** clean."
                        ),
                        # epoch seconds: the adapters normalise before storage
                        "created_at": 1759366800.0,
                    }
                ],
                "reviews": [],
                "ci": {"fetched": True, "runs": [], "total": 0, "url": None},
            }
        )

    _patch_fetch(monkeypatch, fake_fetch)
    report = await service.refresh_keys(tmp_path, [REF])
    assert report.changed == [REF.key]
    entry = cache.read_entry(tmp_path, REF)
    assert entry is not None and entry["state"] == "merged"
    lanes = entry["lanes"]
    assert lanes and lanes[0]["lane"] == "agent" and lanes[0]["round"] == 2
    # Freshness is prefix-based against the stored head sha.
    assert lanes[0]["freshness"] == "fresh"
    assert entry["comments_total"] == 1
    assert entry["stale"] is False and entry["refresh_error"] is None


@pytest.mark.asyncio
async def test_refresh_session_touches_the_index_and_consumes_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A NON-force refresh_session must touch the index only on change, and must
    # consume the dirty mark it acted on.
    index_path = ledger.index_path(tmp_path, "s1")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    old = time.time() - 500
    index_path.write_text(
        json.dumps(
            {
                "schema": ledger.INDEX_SCHEMA,
                "session_id": "s1",
                "updated_at": old,
                "rows": [ROW],
                "tool_output_only": 0,
                "hints": [],
                "events": 0,
            }
        ),
        encoding="utf-8",
    )
    cache.mark_dirty(tmp_path, "s1", keys=[REF.key])

    async def fake_fetch(ref, validators, token, *, stored=None, client=None):
        return _outcome({"summary": {"head_sha": "a" * 12, "title": "t"}})

    _patch_fetch(monkeypatch, fake_fetch)
    report = await service.refresh_session(tmp_path, "s1", [ROW])
    assert report.changed == [REF.key]
    after = json.loads(index_path.read_text(encoding="utf-8"))
    assert after["updated_at"] > old  # the feed's completion signal moved
    assert after["rows"][0]["key"] == REF.key  # rows preserved verbatim
    assert cache.read_dirty(tmp_path, "s1")["keys"] == []  # mark consumed


@pytest.mark.asyncio
async def test_concurrent_refreshes_collapse_to_one_fetch(tmp_path: Path, monkeypatch) -> None:
    calls = 0
    release = asyncio.Event()

    async def fake_fetch(ref, validators, token, *, stored=None, client=None):
        nonlocal calls
        calls += 1
        await release.wait()
        return _outcome({"summary": {"head_sha": "b" * 12, "title": "t"}})

    _patch_fetch(monkeypatch, fake_fetch)
    first = asyncio.create_task(service.refresh_keys(tmp_path, [REF]))
    await asyncio.sleep(0.05)  # let the first task take the key lock
    second = asyncio.create_task(service.refresh_keys(tmp_path, [REF]))
    # Let the second task run too: it must take its ``started`` stamp BEFORE
    # the first pass writes, which is the overlap the collapse rule is about.
    await asyncio.sleep(0.05)
    release.set()
    await asyncio.gather(first, second)
    assert calls == 1


@pytest.mark.asyncio
async def test_failure_keeps_the_last_known_row(tmp_path: Path, monkeypatch) -> None:
    cache.write_entry(tmp_path, _entry(), ref=REF)
    calls = 0

    async def failing_fetch(ref, validators, token, *, stored=None, client=None):
        nonlocal calls
        calls += 1
        raise base.ForgeHTTPError("server", "summary", 503)

    _patch_fetch(monkeypatch, failing_fetch)
    report = await service.refresh_keys(tmp_path, [REF])
    # The FIRST failure is user-visible — the stale marker and the reason
    # appear where the row had none — so it reports changed exactly once
    # (review round 1, F3: a REPEAT of the same error must not, see the
    # five-pass test below).
    assert report.changed == [REF.key]
    entry = cache.read_entry(tmp_path, REF)
    assert entry is not None
    assert entry["stale"] is True and "503" in entry["refresh_error"]
    assert entry["pieces"]  # last known data survives
    assert isinstance(entry.get("checked_at"), (int, float))  # the TTL can pace it
    assert cache.key_backoff_until(REF.key) is not None

    # A second pass inside the backoff makes no call and reports nothing.
    again = await service.refresh_keys(tmp_path, [REF])
    assert again.changed == [] and calls == 1


@pytest.mark.asyncio
async def test_401_re_resolves_once_then_degrades_to_link_only(tmp_path: Path, monkeypatch) -> None:
    calls: list[bool] = []

    def fake_resolve(host, forge, *, home=None, config_dir=None, fresh=False):
        calls.append(fresh)
        if fresh:
            raise code_credentials.CredentialError("absent", "sign in with gh")
        return code_credentials.Token(host=host, forge=forge, value="tok", source="gh")

    async def unauthorized(ref, validators, token, *, stored=None, client=None):
        raise base.ForgeHTTPError("unauthorized", "summary", 401)

    monkeypatch.setattr(service, "resolve", fake_resolve)
    _patch_fetch(monkeypatch, unauthorized)
    await service.refresh_keys(tmp_path, [REF])
    assert calls == [False, True]  # exactly one fresh re-resolve
    view = service.view_for_ref(tmp_path, REF)
    assert view is not None and view["link_only"] is True
    assert "credential rejected" in view["link_only_reason"]


# ---------------------------------------------------------------------------
# the read side
# ---------------------------------------------------------------------------


def test_view_row_merges_and_reports_staleness(tmp_path: Path) -> None:
    cache.write_entry(tmp_path, _entry(stale=True, refresh_error="refresh failed (503)"), ref=REF)
    merged = service.view_rows(tmp_path, [ROW])[0]
    assert merged["link_only"] is False
    assert merged["url"] == REF.url and merged["forge"] == "github"
    assert merged["summary"]["state"] == "open"
    assert merged["stale"] is True and merged["refresh_error"] == "refresh failed (503)"
    # A link-only row keeps its reason and never draws a summary.
    other = service.view_rows(tmp_path, [LINK_ONLY_ROW])[0]
    assert other["link_only"] is True and other["summary"] is None
    assert "detect-and-link" in other["reason"]


def test_lane_freshness_note_reads_the_parsed_state() -> None:
    entry = _entry()
    entry["lanes"] = [
        {
            "lane": "agent",
            "round": 2,
            "state": "clean",
            "state_copy": "clean",
            "freshness": "fresh",
            "reviewed_head": "9d29452",
        }
    ]
    note = service.lane_freshness_note(entry, "9d29452abcdef")
    assert note == "agent review r2 clean on 9d29452"
    entry["lanes"][0]["freshness"] = "stale"
    assert service.lane_freshness_note(entry, "9d29452abcdef") == (
        "agent review r2 clean, head 9d29452"
    )
    assert service.lane_freshness_note(None, "x") == ""


@pytest.mark.asyncio
async def test_show_fetches_when_missing_and_respects_cooling(tmp_path: Path, monkeypatch) -> None:
    calls = []

    async def fake_fetch(ref, validators, token, *, stored=None, client=None):
        calls.append(ref.key)
        return _outcome({"summary": {"head_sha": "c" * 12, "title": "t"}})

    _patch_fetch(monkeypatch, fake_fetch)
    view = await service.show(tmp_path, REF)
    assert calls == [REF.key]
    assert view["link_only"] is False

    # Cooling refuses even an on-demand show: a rate-limited host must not be
    # hammered by a model's curiosity, and the stale answer is the honest one.
    cache.drop_entry(tmp_path, REF)
    cache.note_rate_limited(REF.host)
    calls.clear()
    view = await service.show(tmp_path, REF)
    assert calls == []
    assert view["link_only"] is True


# ---------------------------------------------------------------------------
# review round 1: F1/F2/F3/F4a/F6 — and the token-on-disk invariant (F5)
# ---------------------------------------------------------------------------


def _seed_index(tmp_path: Path, session_id: str = "s1") -> None:
    """An index with one row: ``mark_dirty(all_rows=True)`` is a no-op without rows."""
    from local_operator.code_requests.scan import Row, ScanResult

    assert ledger.write_index(tmp_path, session_id, ScanResult(rows=[Row(ref=REF)]))


@pytest.mark.asyncio
async def test_a_turn_end_all_mark_is_consumed_by_one_pass(tmp_path: Path, monkeypatch) -> None:
    """One mark, N passes: exactly one fetch — the QA round 1, Q2 loop, closed."""
    calls: list[str] = []

    async def fake_fetch(ref, validators, token, *, stored=None, client=None):
        calls.append(ref.key)
        return _outcome(
            pieces={"summary": {"title": "t", "head_sha": "a" * 12}},
            not_modified={"comments", "ci"},
        )

    _patch_fetch(monkeypatch, fake_fetch)
    cache.write_entry(tmp_path, _entry(checked_ago=1.0), ref=REF)  # fresh within its TTL
    _seed_index(tmp_path)
    cache.mark_dirty(tmp_path, "s1", all_rows=True)
    for _ in range(3):
        await service.refresh_session(tmp_path, "s1", [ROW])
    assert calls == [REF.key], "only the first pass may refetch; the mark is consumed"
    assert not cache.dirty_path(tmp_path, "s1").exists()


@pytest.mark.asyncio
async def test_a_404_is_throttled_and_the_revision_does_not_move(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[str] = []

    async def not_found(ref, validators, token, *, stored=None, client=None):
        calls.append(ref.key)
        raise base.ForgeHTTPError("not_found", "summary", 404)

    _patch_fetch(monkeypatch, not_found)
    changed: list[bool] = []
    for _ in range(5):
        report = await service.refresh_session(tmp_path, "s1", [ROW])
        changed.append(bool(report.changed))
    assert calls == [REF.key], "five passes against a 404 make one call"
    assert changed == [True, False, False, False, False], (
        "the first failure is user-visible; identical repeats must not report changed "
        "(the feed revision is what a client's refetch loop keys on)"
    )
    entry = cache.read_entry(tmp_path, REF)
    assert entry is not None and entry["refresh_error"] == "not found at the forge"
    assert isinstance(entry.get("checked_at"), (int, float))


@pytest.mark.asyncio
async def test_a_429_stops_the_rest_of_the_pass_and_holds_cooling(
    tmp_path: Path, monkeypatch
) -> None:
    refs = [_ref(f"https://github.com/o/r/pull/{n}") for n in (7, 8, 9)]
    calls: list[str] = []

    async def fetch(ref, validators, token, *, stored=None, client=None):
        calls.append(ref.key)
        if len(calls) == 1:
            raise base.ForgeHTTPError("rate_limited", "summary", 429, retry_after=120.0)
        return _outcome(pieces={"summary": {"title": "t", "head_sha": "a" * 12}})

    _patch_fetch(monkeypatch, fetch)
    report = await service.refresh_keys(tmp_path, refs)
    assert calls == [refs[0].key], "after the 429 the same-host refs make zero calls"
    assert report.cooling.get("github.com") is not None
    assert cache.cooling_until("github.com") is not None


@pytest.mark.asyncio
async def test_show_honours_the_sessions_dirty_mark(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []

    async def fetch(ref, validators, token, *, stored=None, client=None):
        calls.append(ref.key)
        return _outcome(pieces={"summary": {"title": "new", "head_sha": "b" * 12}})

    _patch_fetch(monkeypatch, fetch)
    cache.write_entry(tmp_path, _entry(), ref=REF)  # fresh within its TTL
    await service.show(tmp_path, REF, session_id="s1")
    assert calls == [], "a fresh entry is served from cache"
    cache.mark_dirty(tmp_path, "s1", keys=[REF.key])
    await service.show(tmp_path, REF, session_id="s1")
    assert calls == [REF.key], "the session's own act marks the row it must re-read"
    assert not cache.dirty_path(tmp_path, "s1").exists(), "show consumed the key's mark"
    await service.show(tmp_path, REF, session_id="s1")
    assert calls == [REF.key], "and the mark does not keep re-firing"


@pytest.mark.asyncio
async def test_an_unknown_host_gets_no_token_and_makes_no_request(
    tmp_path: Path, monkeypatch
) -> None:
    """F1's repro: a faked store secret + an env token still resolve NOTHING for
    a host nobody is signed in to — and no adapter call is ever made."""
    from local_operator.code_requests import credentials as creds
    from local_operator.code_requests.adapters import gitlab as gitlab_module

    monkeypatch.setattr(service, "resolve", creds.resolve)  # undo the autouse dummy
    creds._reset_for_tests()
    monkeypatch.setattr(creds, "_find_glab", lambda home: None)
    monkeypatch.setattr(creds, "_store_secret", lambda config_dir: "store-secret-must-not-travel")
    monkeypatch.setenv("GITLAB_TOKEN", "env-token-must-not-travel")
    calls: list[str] = []

    async def fetch(ref, validators, token, *, stored=None, client=None):
        calls.append(ref.key)
        return _outcome()

    monkeypatch.setattr(gitlab_module.ADAPTER, "fetch", fetch)
    evil = _ref("https://evil.example.invalid/g/p/-/merge_requests/1")
    await service.refresh_keys(tmp_path, [evil])
    assert calls == [], "no request may be sent to a host with no usable login"
    entry = cache.read_entry(tmp_path, evil)
    assert entry is not None
    assert "sign in with the glab CLI" in str(entry.get("refresh_error") or "")


@pytest.mark.asyncio
async def test_no_token_reaches_the_disk_cache(tmp_path: Path, monkeypatch) -> None:
    """The token rides request headers, never the stored entry (F5's invariant)."""
    import httpx

    from local_operator.code_requests.adapters import github as gh_module

    real = gh_module.ADAPTER
    token = "tok-DISK-SECRET-123456"

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/commits/" in path:
            return httpx.Response(200, json={"check_runs": [], "total_count": 0})
        if "/comments" in path or "/reviews" in path:
            return httpx.Response(200, json=[])
        return httpx.Response(
            200,
            json={
                "number": 7,
                "state": "open",
                "title": "t",
                "head": {"sha": "a" * 12, "ref": "x"},
                "base": {"ref": "main"},
                "user": {"login": "u"},
            },
        )

    transport = httpx.MockTransport(handler)

    class Proxy:
        kind = real.kind
        full = real.full
        pieces = real.pieces
        state = staticmethod(real.state)
        comments = staticmethod(real.comments)
        ci = staticmethod(real.ci)

        async def fetch(self, ref, validators, tok, *, stored=None, client=None):
            async with httpx.AsyncClient(transport=transport, timeout=5) as http:
                return await real.fetch(ref, validators, tok, stored=stored, client=http)

    monkeypatch.setattr(service, "adapter_for", lambda ref: Proxy())
    monkeypatch.setattr(
        service,
        "resolve",
        lambda host, forge, **kw: code_credentials.Token(
            host=host, forge=forge, value=token, source="test"
        ),
    )
    report = await service.refresh_keys(tmp_path, [REF])
    assert report.changed, "the fetch landed"
    blob = b"".join(path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())
    assert token.encode() not in blob
    assert b"authorization" not in blob.lower()


@pytest.mark.asyncio
async def test_a_transient_ci_error_keeps_the_last_known_ci_and_its_validator(
    tmp_path: Path, monkeypatch
) -> None:
    """F4c: after a CI blip the stored CI SURVIVES, pairs with its old ETag, and
    a later 304 leaves it in place — the stuck-unknown state cannot recur."""
    import httpx

    from local_operator.code_requests.adapters import github as gh_module

    real = gh_module.ADAPTER
    phase = {"fail_ci": False, "ci": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/commits/" in path:
            if phase["fail_ci"]:
                raise httpx.ConnectError("boom")
            phase["ci"] += 1
            if request.headers.get("if-none-match") == 'W/"ci-1"':
                return httpx.Response(304)
            return httpx.Response(
                200,
                json={
                    "check_runs": [{"name": "t", "status": "completed", "conclusion": "failure"}],
                    "total_count": 1,
                },
                headers={"ETag": 'W/"ci-1"'},
            )
        if "/comments" in path or "/reviews" in path:
            return httpx.Response(200, json=[])
        return httpx.Response(
            200,
            json={
                "number": 7,
                "state": "open",
                "title": "t",
                "head": {"sha": "a" * 12, "ref": "x"},
                "base": {"ref": "main"},
                "user": {"login": "u"},
            },
        )

    transport = httpx.MockTransport(handler)

    class Proxy:
        kind = real.kind
        full = real.full
        pieces = real.pieces
        state = staticmethod(real.state)
        comments = staticmethod(real.comments)
        ci = staticmethod(real.ci)

        async def fetch(self, ref, validators, tok, *, stored=None, client=None):
            async with httpx.AsyncClient(transport=transport, timeout=5) as http:
                return await real.fetch(ref, validators, tok, stored=stored, client=http)

    monkeypatch.setattr(service, "adapter_for", lambda ref: Proxy())
    await service.refresh_keys(tmp_path, [REF])
    entry = cache.read_entry(tmp_path, REF)
    assert entry is not None
    assert entry["ci"]["status"] == "failure"
    assert entry["validators"]["ci"]["etag"] == 'W/"ci-1"'

    phase["fail_ci"] = True
    cache._reset_for_tests()  # force the next read from disk; memory tiers cleared
    cache.clear_key_backoff(REF.key)
    await service.refresh_keys(tmp_path, [REF], force=True)
    entry = cache.read_entry(tmp_path, REF)
    assert entry is not None
    assert entry["stale"] is True
    assert entry["pieces"]["ci"], "the last known CI survived the transient failure"
    assert entry["validators"]["ci"]["etag"] == 'W/"ci-1"', "and its validator still pairs"

    phase["fail_ci"] = False
    cache.clear_key_backoff(REF.key)
    await service.refresh_keys(tmp_path, [REF], force=True)
    entry = cache.read_entry(tmp_path, REF)
    assert entry is not None
    assert entry["ci"]["status"] == "failure", "the 304 left the stored CI in place"


def test_the_row_carries_the_comment_count_under_the_ui_field_name() -> None:
    """QA round 1, Q7: the desktop row contract reads ``summary.comments``
    (``DesktopCodeRequestSummary.comments``); null, never a rendered 0."""
    raw = dict(ROW)
    entry = _entry()
    entry["comments_total"] = 6
    row = service.view_row(raw, entry)
    assert row["summary"]["comments"] == 6
    entry.pop("comments_total")
    assert service.view_row(raw, entry)["summary"]["comments"] is None


# ---------------------------------------------------------------------------
# cross-round findings X1 (fetch-time parse survives truncation), X3 (per-forge
# remedy copy)
# ---------------------------------------------------------------------------


def _long_convention_body() -> str:
    """#2106's real round-2 shape: header first, verdict at char ~6708 of ~6979."""
    header = (
        "### Agent review — round 2\n\n"
        "**Reviewer:** reviewer on a-model\n"
        "**Scope:** `main..9d29452abc123`\n\n"
    )
    filler = ("A remediation paragraph that speaks about many small details. " * 100) + "\n\n"
    tail = "**Verdict:** clean — merge-ready. All majors closed on the head above.\n"
    body = header + filler + tail
    assert len(body) > cache.COMMENT_BODY_MAX + 512, len(body)
    assert body.index("**Verdict:**") > cache.COMMENT_BODY_MAX, "the verdict sits past the cap"
    return body


@pytest.mark.asyncio
async def test_a_long_comments_verdict_survives_a_304_rebuild(tmp_path: Path, monkeypatch) -> None:
    """X1: parse at fetch time on the FULL body, replay on a non-refetching pass.

    Before the fix the second (all-304) pass re-parsed the stored, capped copy:
    the header survived but the verdict past 4 KiB was cut, so a lane the fetch
    read as ``clean`` degraded to ``unstated`` without any forge change.
    """
    import httpx

    from local_operator.code_requests.adapters import github as gh_module

    real = gh_module.ADAPTER
    body = _long_convention_body()
    count = {"fetches": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("if-none-match"):
            return httpx.Response(304)
        count["fetches"] += 1
        path = request.url.path
        if "/commits/" in path:
            return httpx.Response(200, json={"check_runs": [], "total_count": 0})
        if path.endswith("/issues/7/comments"):
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 42,
                        "body": body,
                        "created_at": "2026-10-02T01:00:00Z",
                        "html_url": "https://x/42",
                    }
                ],
                headers={"ETag": 'W/"c-1"'},
            )
        if path.endswith("/pulls/7/reviews"):
            return httpx.Response(200, json=[], headers={"ETag": 'W/"r-1"'})
        return httpx.Response(
            200,
            json={
                "number": 7,
                "state": "open",
                "title": "t",
                "head": {"sha": "9d29452abc123", "ref": "x"},
                "base": {"ref": "main"},
                "user": {"login": "u"},
            },
            headers={"ETag": 'W/"s-1"'},
        )

    transport = httpx.MockTransport(handler)

    class Proxy:
        kind = real.kind
        full = real.full
        pieces = real.pieces
        state = staticmethod(real.state)
        comments = staticmethod(real.comments)
        ci = staticmethod(real.ci)

        async def fetch(self, ref, validators, tok, *, stored=None, client=None):
            async with httpx.AsyncClient(transport=transport, timeout=5) as http:
                return await real.fetch(ref, validators, tok, stored=stored, client=http)

    monkeypatch.setattr(service, "adapter_for", lambda ref: Proxy())
    await service.refresh_keys(tmp_path, [REF])
    entry = cache.read_entry(tmp_path, REF)
    assert entry is not None
    lane = next(item for item in entry["lanes"] if item["lane"] == "agent")
    assert lane["state"] == "clean", lane
    # The stored body is still bounded; the parse is what survives whole.
    stored_body = str(entry["pieces"]["comments"][0]["body"])
    assert len(stored_body) <= cache.COMMENT_BODY_MAX + 32
    assert "[truncated]" in stored_body

    # Second pass: every piece answers 304 → the stored parse is replayed.
    cache._reset_for_tests()
    cache.clear_key_backoff(REF.key)
    await service.refresh_keys(tmp_path, [REF], force=True)
    entry = cache.read_entry(tmp_path, REF)
    assert entry is not None
    lane = next(item for item in entry["lanes"] if item["lane"] == "agent")
    assert (
        lane["state"] == "clean"
    ), "the verdict past the 4 KiB cap must not degrade on a 304 rebuild: " + str(lane)


def test_link_only_hints_name_the_right_cli_per_forge() -> None:
    """X3: gh for GitHub, glab for GitLab, and nothing CLI-specific for
    detect-and-link-only forges (Codeberg used to say 'sign in with gh')."""
    assert "gh CLI" in service.link_only_hint({"forge": "github", "host": "github.com"})
    assert "gh auth login --hostname ghe.corp" in service.link_only_hint(
        {"forge": "github", "host": "ghe.corp"}
    )
    assert "glab CLI" in service.link_only_hint({"forge": "gitlab", "host": "gitlab.com"})
    codeberg = service.link_only_hint({"forge": "gitea", "host": "codeberg.org"})
    assert "gh" not in codeberg and "glab" not in codeberg
    assert "isn't tracked yet" in codeberg
    unknown = service.link_only_hint(None)
    assert "isn't tracked yet" in unknown


def test_the_wire_row_carries_the_hint() -> None:
    raw = {
        "key": "[redacted]",
        "ref": {
            "key": "[redacted]",
            "forge": "gitea",
            "host": "codeberg.org",
            "project": "o/r",
            "number": 3,
            "url": "https://codeberg.org/o/r/pulls/3",
            "full": False,
            "reason": "Gitea/Forgejo is detect-and-link: the link opens, no state is fetched",
        },
        "relation": "mentioned",
        "relations": ["mentioned"],
    }
    row = service.view_row(raw, None)
    assert row["link_only"] is True
    assert row["link_only_hint"] == "Link only — this host isn't tracked yet."


# ---------------------------------------------------------------------------
# review round 2: N2 (mixed refetch keeps full-body parses), N3 (show vs the
# all-mark), N4 (all-mark survives unserviced rows), Q10 (the probe is
# reachable), Q13 (cooling state per row)
# ---------------------------------------------------------------------------


def _mixed_handler(calls: list[str], body: str, *, changing: str):
    """Phase 1 primes both pieces; phase 2 refetches ONLY ``changing``.

    The long convention comment (verdict near the end, past the 4 KiB cap)
    lives on the OTHER piece, which answers 304 in phase 2; the changing piece
    gains a body the parser ignores. The lane must stay decided by the long
    comment — the exact N2 shape, in both directions (reviews change while
    comments 304, and the reverse).
    """
    import httpx

    new_comment = {
        "id": 43,
        "body": "LGTM!",
        "created_at": "2026-10-03T00:00:00Z",
        "html_url": "u",
    }
    long_comment = {
        "id": 42,
        "body": body,
        "created_at": "2026-10-02T01:00:00Z",
        "html_url": "u",
    }
    piece_paths = {
        "comments": "/issues/7/comments",
        "reviews": "/pulls/7/reviews",
    }

    def piece_response(request: httpx.Request, piece: str, etag_in: str, etag_out: str):
        inm = request.headers.get("if-none-match")
        if piece == changing:
            if inm == etag_in:
                return httpx.Response(200, json=[new_comment], headers={"ETag": etag_out})
            if inm == etag_out:
                return httpx.Response(304)
            return httpx.Response(200, json=[], headers={"ETag": etag_in})
        if inm == etag_in:
            return httpx.Response(304)
        return httpx.Response(200, json=[long_comment], headers={"ETag": etag_in})

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        calls.append(path)
        if "/commits/" in path:
            return httpx.Response(200, json={"check_runs": [], "total_count": 0})
        for piece, suffix in piece_paths.items():
            if path.endswith(suffix):
                return piece_response(request, piece, f'W/"{piece[0]}-1"', f'W/"{piece[0]}-2"')
        return httpx.Response(
            200,
            json={
                "number": 7,
                "state": "open",
                "title": "t",
                "head": {"sha": "9d29452abc123", "ref": "x"},
                "base": {"ref": "main"},
                "user": {"login": "u"},
            },
            headers={"ETag": 'W/"s-1"'},
        )

    return handler


def _install_proxy(monkeypatch: pytest.MonkeyPatch, handler_factory) -> None:
    """Serve the REAL adapter through a MockTransport, nothing else faked."""
    import httpx

    from local_operator.code_requests.adapters import github as gh_module

    real = gh_module.ADAPTER

    class Proxy:
        kind = real.kind
        full = real.full
        pieces = real.pieces
        state = staticmethod(real.state)
        comments = staticmethod(real.comments)
        ci = staticmethod(real.ci)

        async def fetch(self, ref, validators, tok, *, stored=None, client=None):
            transport = httpx.MockTransport(handler_factory())
            async with httpx.AsyncClient(transport=transport, timeout=5) as http:
                return await real.fetch(ref, validators, tok, stored=stored, client=http)

    monkeypatch.setattr(service, "adapter_for", lambda ref: Proxy())


async def _mixed_lane(
    tmp_path: Path,
    monkeypatch,
    *,
    changing: str,
    body: str | None = None,
    expected: str = "clean",
) -> dict[str, Any]:
    body = body if body is not None else _long_convention_body()
    calls: list[str] = []
    _install_proxy(monkeypatch, lambda: _mixed_handler(calls, body, changing=changing))
    await service.refresh_keys(tmp_path, [REF])
    entry = cache.read_entry(tmp_path, REF)
    assert entry is not None
    lane = next(item for item in entry["lanes"] if item["lane"] == "agent")
    assert lane["state"] == expected, lane
    # Phase 2: only the ``changing`` piece refetches (the other answers 304).
    cache._reset_for_tests()
    cache.clear_key_backoff(REF.key)
    await service.refresh_keys(tmp_path, [REF], force=True)
    entry = cache.read_entry(tmp_path, REF)
    assert entry is not None
    return next(item for item in entry["lanes"] if item["lane"] == "agent")


@pytest.mark.asyncio
async def test_a_review_mixed_refresh_keeps_the_long_comments_verdict(
    tmp_path: Path, monkeypatch
) -> None:
    lane = await _mixed_lane(tmp_path, monkeypatch, changing="reviews")
    assert lane["state"] == "clean", "reviews changed while comments 304'd: " + str(lane)


@pytest.mark.asyncio
async def test_a_comments_mixed_refresh_keeps_the_long_reviews_verdict(
    tmp_path: Path, monkeypatch
) -> None:
    lane = await _mixed_lane(tmp_path, monkeypatch, changing="comments")
    assert lane["state"] == "clean", "comments changed while reviews 304'd: " + str(lane)


@pytest.mark.asyncio
async def test_show_ignores_the_sessions_all_mark(tmp_path: Path, monkeypatch) -> None:
    """N3: an unconsumed turn-end mark must not make every ``show`` refetch."""
    calls: list[str] = []

    async def fetch(ref, validators, token, *, stored=None, client=None):
        calls.append(ref.key)
        return _outcome(pieces={"summary": {"title": "new", "head_sha": "b" * 12}})

    _patch_fetch(monkeypatch, fetch)
    cache.write_entry(tmp_path, _entry(), ref=REF)  # fresh within its TTL
    _seed_index(tmp_path)
    cache.mark_dirty(tmp_path, "s1", all_rows=True)
    for _ in range(3):
        await service.show(tmp_path, REF, session_id="s1")
    assert calls == [], "the all mark is not the ref's own mark"
    assert cache.read_dirty(tmp_path, "s1")["all"] is True
    # The key's own mark still revalidates (F6, unchanged) and consumes itself.
    cache.mark_dirty(tmp_path, "s1", keys=[REF.key])
    await service.show(tmp_path, REF, session_id="s1")
    assert calls == [REF.key]
    assert cache.read_dirty(tmp_path, "s1")["all"] is True


@pytest.mark.asyncio
async def test_an_all_mark_survives_a_pass_that_skipped_a_cooling_host(
    tmp_path: Path, monkeypatch
) -> None:
    """N4: clearing ``all`` for rows the pass never serviced loses their revalidation."""
    calls: list[str] = []

    async def fetch(ref, validators, token, *, stored=None, client=None):
        calls.append(ref.key)
        return _outcome(pieces={"summary": {"title": "t", "head_sha": "a" * 12}})

    _patch_fetch(monkeypatch, fetch)
    _seed_index(tmp_path)
    cache.mark_dirty(tmp_path, "s1", all_rows=True)
    cache.note_rate_limited("github.com", retry_after=120.0)
    await service.refresh_session(tmp_path, "s1", [ROW])
    assert calls == [], "the cooling host is not called"
    assert cache.read_dirty(tmp_path, "s1")["all"] is True, "the mark was NOT serviced"
    # Once the window ends, the next pass services the row and consumes the mark.
    cache._reset_for_tests()
    await service.refresh_session(tmp_path, "s1", [ROW])
    assert calls == [REF.key]
    assert cache.read_dirty(tmp_path, "s1")["all"] is False


@pytest.mark.asyncio
async def test_probe_shorthand_reaches_the_adapter_without_a_full_ref(
    tmp_path: Path, monkeypatch
) -> None:
    """Q10: the advertised resolve works through the REAL lookup chain.

    No ``adapter_for`` patch anywhere: the probe takes its adapter from the
    forge registry (``forge_adapter``), because ``adapter_for`` refuses every
    unconfirmed ref by design and used to make this path unreachable.
    """
    import httpx

    from local_operator.code_requests.adapters import adapter_for
    from local_operator.code_requests.adapters import github as gh_module

    ref = parse_any("o/r#7")
    assert ref is not None and not ref.full
    assert adapter_for(ref) is None, "the fetch gate still refuses an unconfirmed ref"

    original = gh_module.GitHubAdapter.probe_pull
    seen: list[str] = []
    status = {"code": 200}

    async def probe(self, probe_ref, token, *, client=None):
        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(f"{request.method} {request.url.path}")
            return httpx.Response(status["code"], json={})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=5) as http:
            return await original(gh_module.ADAPTER, probe_ref, token, client=http)

    monkeypatch.setattr(gh_module.GitHubAdapter, "probe_pull", probe)
    try:
        verdict, promoted = await service.probe_shorthand(tmp_path, ref)
        assert verdict == "pull" and promoted is not None and promoted.full is True
        assert seen == ["GET /repos/o/r/pulls/7"]
        cache._reset_for_tests()
        credentials_reset()
        status["code"] = 404
        verdict, promoted = await service.probe_shorthand(tmp_path, ref)
        assert verdict == "issue" and promoted is None
        assert seen[-1] == "GET /repos/o/r/pulls/7"
    finally:
        monkeypatch.undo()


def credentials_reset() -> None:
    from local_operator.code_requests import credentials as creds

    creds._reset_for_tests()


def test_a_never_fetched_row_states_cooling_not_plain_link_only() -> None:
    """Q13: a tracked row on a cooling host must show the WAIT, not link-only."""
    raw = dict(ROW)
    cache.note_rate_limited("github.com", retry_after=120.0)
    try:
        row = service.view_row(raw, None)
        assert row["link_only"] is True
        assert isinstance(row.get("cooling_until"), float)
        assert "cooling" in str(row.get("reason") or "")
        assert "sign in" not in str(row.get("reason") or "")
    finally:
        cache._reset_for_tests()
    row = service.view_row(raw, None)
    assert "cooling_until" not in row
    assert "reason" not in row


# ---------------------------------------------------------------------------
# final round: Q15 (tool surface parity), N5 (keys-scoped all-mark), N6
# (chained kick), N7 (ignored survives replays)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_show_never_fetched_carries_cooling_and_hint(tmp_path: Path) -> None:
    """Q15: the entry-less fallback is what a cooling/detect-and-link show hits."""
    cache.note_rate_limited("github.com", retry_after=120.0)
    try:
        view = await service.show(tmp_path, REF)
    finally:
        cache._reset_for_tests()
    assert view["link_only"] is True
    assert isinstance(view["cooling_until"], float)
    assert view["link_only_reason"] == f"cooling — {service.cooling_copy(view['cooling_until'])}"
    assert view["link_only_hint"] == "Link only — sign in with the gh CLI to track this one."

    gitea = parse_any("https://codeberg.org/o/r/pulls/3")
    assert gitea is not None
    plain = await service.show(tmp_path, gitea)
    assert plain["link_only"] is True
    assert "cooling_until" not in plain
    assert plain["link_only_hint"] == "Link only — this host isn't tracked yet."
    assert "detect-and-link" in str(plain["link_only_reason"])


@pytest.mark.asyncio
async def test_an_all_mark_survives_a_keys_scoped_pass(tmp_path: Path, monkeypatch) -> None:
    """N5: the acted kick's keys-scoped pass must not consume the turn-end mark."""
    calls: list[str] = []

    async def fetch(ref, validators, token, *, stored=None, client=None):
        calls.append(ref.key)
        return _outcome(pieces={"summary": {"title": "t", "head_sha": "a" * 12}})

    _patch_fetch(monkeypatch, fetch)
    _seed_index(tmp_path)
    cache.mark_dirty(tmp_path, "s1", all_rows=True)
    await service.refresh_session(tmp_path, "s1", [ROW], keys=[REF.key])
    assert calls == [REF.key]
    assert (
        cache.read_dirty(tmp_path, "s1")["all"] is True
    ), "the filtered-out rows were not serviced"
    # The unfiltered pass covers whatever is left and consumes the mark.
    await service.refresh_session(tmp_path, "s1", [ROW])
    assert cache.read_dirty(tmp_path, "s1")["all"] is False


@pytest.mark.asyncio
async def test_a_joined_kick_is_chained_not_lost(tmp_path: Path, monkeypatch) -> None:
    """N6: a kick that joins an in-flight pass schedules a follow-on for its keys."""
    runs: list[list[str]] = []
    gate = asyncio.Event()

    async def slow(config_dir, session_id, rows, *, keys=None, force=False):
        runs.append(sorted(str(k) for k in (keys or ())))
        if len(runs) == 1:
            await gate.wait()
        return service.RefreshReport()

    monkeypatch.setattr(service, "refresh_session", slow)
    assert service.schedule_session_refresh(tmp_path, "s1", [], keys=["a"]) is True
    await asyncio.sleep(0)
    # Two kicked events join while the first pass is in flight: one follow-on
    # carries the UNION of their keys, and runs without any read.
    assert service.schedule_session_refresh(tmp_path, "s1", [], keys=["b"], chain=True) is True
    assert service.schedule_session_refresh(tmp_path, "s1", [], keys=["c"], chain=True) is True
    gate.set()
    for _ in range(200):
        if len(runs) == 2:
            break
        await asyncio.sleep(0.01)
    assert runs == [["a"], ["b", "c"]], runs


@pytest.mark.asyncio
async def test_the_ignored_list_survives_a_replay_only_rebuild(tmp_path: Path, monkeypatch) -> None:
    """N7: a 304-replayed piece contributes no fresh parse, so its ignored ids replay."""
    import httpx

    from local_operator.code_requests.adapters import github as gh_module

    real = gh_module.ADAPTER

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "/commits/" in path:
            return httpx.Response(200, json={"check_runs": [], "total_count": 0})
        if path.endswith("/issues/7/comments"):
            return httpx.Response(200, json=[], headers={"ETag": 'W/"c-1"'})
        if path.endswith("/pulls/7/reviews"):
            if request.headers.get("if-none-match") == 'W/"r-1"':
                return httpx.Response(304)
            return httpx.Response(
                200,
                json=[
                    {
                        "id": 43,
                        "body": "LGTM!",
                        "submitted_at": "2026-10-03T00:00:00Z",
                        "html_url": "u",
                    }
                ],
                headers={"ETag": 'W/"r-1"'},
            )
        return httpx.Response(
            200,
            json={
                "number": 7,
                "state": "open",
                "title": "t",
                "head": {"sha": "a" * 12, "ref": "x"},
                "base": {"ref": "main"},
                "user": {"login": "u"},
            },
        )

    transport = httpx.MockTransport(handler)

    class Proxy:
        kind = real.kind
        full = real.full
        pieces = real.pieces
        state = staticmethod(real.state)
        comments = staticmethod(real.comments)
        ci = staticmethod(real.ci)

        async def fetch(self, ref, validators, tok, *, stored=None, client=None):
            async with httpx.AsyncClient(transport=transport, timeout=5) as http:
                return await real.fetch(ref, validators, tok, stored=stored, client=http)

    monkeypatch.setattr(service, "adapter_for", lambda ref: Proxy())
    cache.clear_key_backoff(REF.key)
    await service.refresh_keys(tmp_path, [REF])
    entry = cache.read_entry(tmp_path, REF)
    assert entry is not None
    assert entry["convention"]["ignored"]["reviews"] == ["review:43"]
    # A replay-only rebuild (everything 304s) keeps it.
    cache._reset_for_tests()
    cache.clear_key_backoff(REF.key)
    await service.refresh_keys(tmp_path, [REF], force=True)
    entry = cache.read_entry(tmp_path, REF)
    assert entry is not None
    assert entry["convention"]["ignored"]["reviews"] == ["review:43"]


# ---------------------------------------------------------------------------
# a verdict the parser must see WHOLE: past the 4 KiB display cap AND past the
# parser's top-of-body field window — the shape of #2112's round-4 review
# ---------------------------------------------------------------------------


def _deep_verdict_body() -> str:
    """#2112's round-4 agent review, reshaped: 45 lines, verdict at char ~4450.

    Two independent bounds used to hide this verdict: the 4 KiB body cap on the
    stored copy (a replay that re-parsed the stored body lost it) and the
    parser's 40-line field window (the FULL body lost it too, so no amount of
    parsing-before-trimming alone made the lane read ``clean``). The helper
    asserts both preconditions so the tests below cannot silently weaken.
    """
    header = (
        "### Agent review — round 4\n\n"
        "Reviewer: reviewer on a-model\n"
        "Scope: `46b12d2ff9..039476dff3` (one commit)\n\n"
    )
    filler = "".join(
        f"- Finding {index}: a verification note that is long enough to matter. " + "x" * 40 + "\n"
        for index in range(42)
    )
    tail = "\n**Verdict: `clean` — no BLOCKER, no MAJOR — round 4 is TERMINAL on `039476dff3`.**\n"
    body = header + filler + tail
    verdict_at = body.index("**Verdict")
    assert verdict_at > cache.COMMENT_BODY_MAX, verdict_at
    verdict_line = len(body[:verdict_at].splitlines())
    assert verdict_line > 40, verdict_line  # rounds.FIELD_SCAN_LINES
    return body


async def _prime_deep_verdict(tmp_path: Path, monkeypatch) -> dict[str, Any]:
    """One FIRST fetch through the real GitHub adapter; returns the agent lane."""
    body = _deep_verdict_body()
    _install_proxy(monkeypatch, lambda: _mixed_handler([], body, changing="reviews"))
    await service.refresh_keys(tmp_path, [REF])
    return _agent_lane(tmp_path)


def _agent_lane(tmp_path: Path) -> dict[str, Any]:
    entry = cache.read_entry(tmp_path, REF)
    assert entry is not None
    return next(item for item in entry["lanes"] if item["lane"] == "agent")


@pytest.mark.asyncio
async def test_a_deep_verdict_is_read_on_the_first_fetch(tmp_path: Path, monkeypatch) -> None:
    lane = await _prime_deep_verdict(tmp_path, monkeypatch)
    assert lane["state"] == "terminal", lane
    assert "TERMINAL" in lane["verdict"]
    entry = cache.read_entry(tmp_path, REF)
    assert entry is not None
    stored = str(entry["pieces"]["comments"][0]["body"])
    assert "[truncated]" in stored, "the stored copy IS capped; the parse must not depend on it"


@pytest.mark.asyncio
async def test_a_deep_verdict_survives_a_restart_disk_replay(tmp_path: Path, monkeypatch) -> None:
    """Process restart = empty memory tier, entry read back from disk."""
    await _prime_deep_verdict(tmp_path, monkeypatch)
    cache._reset_for_tests()
    assert _agent_lane(tmp_path)["state"] == "terminal"
    # ... and the row the UI draws is built from that same stored parse.
    entry = cache.read_entry(tmp_path, REF)
    row = service.view_row(ROW, entry)
    agent = next(item for item in row["lanes"] if item["lane"] == "agent")
    assert agent["state"] == "terminal", agent


@pytest.mark.asyncio
async def test_a_deep_verdict_survives_an_all_304_refresh(tmp_path: Path, monkeypatch) -> None:
    await _prime_deep_verdict(tmp_path, monkeypatch)
    cache._reset_for_tests()
    cache.clear_key_backoff(REF.key)
    await service.refresh_keys(tmp_path, [REF], force=True)
    assert _agent_lane(tmp_path)["state"] == "terminal"


@pytest.mark.asyncio
@pytest.mark.parametrize("changing", ["reviews", "comments"])
async def test_a_deep_verdict_survives_a_mixed_200_304_refresh(
    tmp_path: Path, monkeypatch, changing: str
) -> None:
    lane = await _mixed_lane(
        tmp_path, monkeypatch, changing=changing, body=_deep_verdict_body(), expected="terminal"
    )
    assert lane["state"] == "terminal", lane


@pytest.mark.asyncio
async def test_a_piece_with_no_stored_parse_refetches_instead_of_reading_the_capped_copy(
    tmp_path: Path, monkeypatch
) -> None:
    """The one rebuild path that DID parse a trimmed body.

    An entry with stored pieces but no ``convention`` (written before the
    fetch-time parse existed, or edited) used to answer 304 for a comment piece
    and then parse the capped copy it kept. The validators of such a piece must
    be dropped so the pass refetches it whole, ONCE: the refetch writes the parse,
    after which the pieces revalidate normally (no refetch loop; review F3).
    """
    import httpx

    body = _deep_verdict_body()
    sent: list[tuple[str, bool]] = []  # (comment-piece path, carried If-None-Match)
    _COMMENTS_PATH, _REVIEWS_PATH = "/repos/o/r/issues/7/comments", "/repos/o/r/pulls/7/reviews"

    def factory():
        inner = _mixed_handler([], body, changing="reviews")

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith(("/issues/7/comments", "/pulls/7/reviews")):
                sent.append((request.url.path, bool(request.headers.get("if-none-match"))))
            return inner(request)

        return handler

    _install_proxy(monkeypatch, factory)
    await service.refresh_keys(tmp_path, [REF])
    entry = cache.read_entry(tmp_path, REF)
    assert entry is not None
    legacy = {k: v for k, v in entry.items() if k != "convention"}
    legacy["lanes"] = []
    cache.write_entry(tmp_path, legacy, ref=REF)
    cache._reset_for_tests()

    async def forced_pass() -> list[tuple[str, bool]]:
        cache.clear_key_backoff(REF.key)
        sent.clear()
        await service.refresh_keys(tmp_path, [REF], force=True)
        return list(sent)

    # Pass 1: the legacy entry's validators are dropped -> unconditional GETs.
    first = await forced_pass()
    assert sorted(first) == [(_COMMENTS_PATH, False), (_REVIEWS_PATH, False)], first
    assert _agent_lane(tmp_path)["state"] == "terminal"
    # Passes 2 and 3: the parse is stored, so both pieces revalidate (304) again.
    for _ in range(2):
        later = await forced_pass()
        assert sorted(later) == [(_COMMENTS_PATH, True), (_REVIEWS_PATH, True)], later
        assert _agent_lane(tmp_path)["state"] == "terminal"
