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

    async def failing_fetch(ref, validators, token, *, stored=None, client=None):
        raise base.ForgeHTTPError("server", "summary", 503)

    _patch_fetch(monkeypatch, failing_fetch)
    report = await service.refresh_keys(tmp_path, [REF])
    assert report.changed == []
    entry = cache.read_entry(tmp_path, REF)
    assert entry is not None
    assert entry["stale"] is True and "503" in entry["refresh_error"]
    assert entry["pieces"]  # last known data survives
    assert cache.key_backoff_until(REF.key) is not None


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
