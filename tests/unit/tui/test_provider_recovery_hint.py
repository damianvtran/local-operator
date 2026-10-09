"""The TUI's recovery hint wiring: ONE pipeline, two access patterns.

``OperatorApp._provider_recovery_hints`` is the shared front every surface of
the app routes a turn error through; its two thin twins differ ONLY in how the
Radient usage arm probes. ``_with_recovery_hint`` (sync, used by the
``on_turn_ended`` message handler) renders CACHE-ONLY and never blocks — a sync
handler runs on the app's event loop, and review round 1 (R1) measured its
previous inline probe freezing input/repaint for 5.04s — while
``_with_recovery_hint_async`` (used by the three worker sites) awaits the
bounded probe. These tests pin what both return for the cases the Radient
workstream added or could disturb: a Radient quota error gains the usage-limit
remedy on the awaited arm; the sync arm serves only what the process already
knows; a Radient AUTH error keeps its existing ``/login`` hint and does not
gain the quota one (no double-fire); and any other provider's quota error
stays byte-identical.

The store and the probe seams are injected — no test reads the operator's
credentials or the network — and the app object is built the same way
``test_usage_continuity`` builds it, without a running Textual app.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from local_operator.providers import radient_recovery as rr
from local_operator.providers.auth_store import AuthStore
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.transcript import NoticeBlock

RENDERED_QUOTA = "out of credits (HTTP 402): insufficient credits"
RENDERED_AUTH = "authentication failed (HTTP 401): bad key"


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch):
    """No test may read the operator's store or environment for a key."""
    import local_operator.providers.registry as registry

    monkeypatch.delenv("RADIENT_API_KEY", raising=False)
    monkeypatch.setattr(registry, "provider_secret_value", lambda *args, **kwargs: None)
    rr.reset_recovery_cache()
    yield
    rr.reset_recovery_cache()


def _app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, arm_probe: bool, calls: list[str]):
    """``OperatorApp`` with an injected store; probes record into ``calls``.

    Both probe seams are armed when ``arm_probe`` is set, and the SYNC one is a
    tripwire: the sync arm must never reach it (that is the R1 regression), so
    any call it records fails whichever test asserted a clean list.
    """
    import local_operator.providers.auth_store as auth_store_module

    store = AuthStore(tmp_path / "auth.db")
    if arm_probe:
        store.upsert_credential("radient", {"type": "oauth", "access": "tok-1", "refresh": "r"})

        async def probe_async(token: str):
            calls.append(token)
            return rr.parse_verification({"signup_grant": "pending", "grant_amount": 5})

        def probe_sync(token: str):
            calls.append(f"SYNC:{token}")
            return rr.parse_verification({"signup_grant": "pending", "grant_amount": 5})

        monkeypatch.setattr(rr, "_probe_verification_async", probe_async)
        monkeypatch.setattr(rr, "_probe_verification_sync", probe_sync)
    monkeypatch.setattr(auth_store_module, "shared_auth_store", lambda *args, **kwargs: store)

    app = OperatorApp(lambda: None)  # type: ignore[arg-type]
    app._session = MagicMock(model_label="radient/auto")
    return app


@pytest.mark.asyncio
async def test_the_awaited_arm_gains_the_usage_limit_remedy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    app = _app(monkeypatch, tmp_path, arm_probe=True, calls=calls)

    out = await app._with_recovery_hint_async(RENDERED_QUOTA)

    assert out.startswith(RENDERED_QUOTA)
    assert "Check your inbox" in out and "$5 in free credits" in out
    assert "/login" not in out, "the quota remedy must not borrow the auth remedy"
    assert calls == ["tok-1"]


@pytest.mark.asyncio
async def test_the_sync_arm_is_cache_only_and_never_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 1, R1: the sync path must not wait on a probe.

    Cold: the text renders unchanged, promptly, with the (armed) sync tripwire
    untouched. Warm: the shared cache serves the sentence with no new probe.
    """
    calls: list[str] = []
    app = _app(monkeypatch, tmp_path, arm_probe=True, calls=calls)

    started = time.monotonic()
    cold = app._with_recovery_hint(RENDERED_QUOTA)
    elapsed = time.monotonic() - started
    assert cold == RENDERED_QUOTA
    assert elapsed < 0.5, f"the sync arm took {elapsed:.2f}s — it must not wait"
    assert calls == [], "the sync arm must not spend a probe"

    await app._with_recovery_hint_async(RENDERED_QUOTA)  # warms the shared cache

    warm = app._with_recovery_hint(RENDERED_QUOTA)
    assert "Check your inbox" in warm and "$5 in free credits" in warm
    assert calls == ["tok-1"], "the warm render must come from the cache"


def test_a_deliberately_slow_probe_does_not_delay_the_sync_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1's regression shape: arm probes that would burn five seconds and
    assert the sync arm returns promptly without touching either of them."""
    calls: list[str] = []
    app = _app(monkeypatch, tmp_path, arm_probe=False, calls=calls)
    store = AuthStore(tmp_path / "slow.db")
    store.upsert_credential("radient", {"type": "oauth", "access": "tok-1", "refresh": "r"})

    def slow_sync(token: str):
        calls.append(f"SYNC:{token}")
        time.sleep(5)
        return None

    async def slow_async(token: str):
        calls.append(token)
        await asyncio.sleep(5)
        return None

    monkeypatch.setattr(rr, "_probe_verification_sync", slow_sync)
    monkeypatch.setattr(rr, "_probe_verification_async", slow_async)
    import local_operator.providers.auth_store as auth_store_module

    monkeypatch.setattr(auth_store_module, "shared_auth_store", lambda *a, **k: store)
    rr.reset_recovery_cache()

    started = time.monotonic()
    out = app._with_recovery_hint(RENDERED_QUOTA)
    elapsed = time.monotonic() - started

    assert out == RENDERED_QUOTA
    assert elapsed < 0.5, f"the sync arm took {elapsed:.2f}s with a slow probe armed"
    assert calls == [], "neither probe twin may run on the sync path"


def test_the_sync_arm_answers_the_no_credential_case_without_a_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    app = _app(monkeypatch, tmp_path, arm_probe=False, calls=calls)

    out = app._with_recovery_hint(RENDERED_QUOTA)

    assert out == f"{RENDERED_QUOTA}\n{rr._neutral_text()}"
    assert calls == []


@pytest.mark.asyncio
async def test_a_radient_auth_error_keeps_its_login_hint_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The disjointness the display sites depend on: one kind, one remedy."""
    calls: list[str] = []
    app = _app(monkeypatch, tmp_path, arm_probe=True, calls=calls)

    out = app._with_recovery_hint(RENDERED_AUTH)
    awaited = await app._with_recovery_hint_async(RENDERED_AUTH)

    assert "/login radient" in out and "/login radient" in awaited
    assert "Check your inbox" not in out and "Check your inbox" not in awaited
    assert calls == [], "an auth error must not spend a Radient usage probe"


@pytest.mark.asyncio
async def test_another_providers_quota_error_is_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    app = _app(monkeypatch, tmp_path, arm_probe=True, calls=calls)
    app._session = MagicMock(model_label="openai/gpt-5")

    assert app._with_recovery_hint(RENDERED_QUOTA) == RENDERED_QUOTA
    assert await app._with_recovery_hint_async(RENDERED_QUOTA) == RENDERED_QUOTA
    assert calls == []


@pytest.mark.asyncio
async def test_the_notice_gains_the_sentence_when_the_probe_lands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker half: the settled notice is EXTENDED in place, not replaced."""
    calls: list[str] = []
    app = _app(monkeypatch, tmp_path, arm_probe=True, calls=calls)
    notice = NoticeBlock(RENDERED_QUOTA, "error")

    await app._finish_recovery_notice(notice, RENDERED_QUOTA)

    text = notice.text()
    assert text.startswith(RENDERED_QUOTA)
    assert "Check your inbox" in text and "$5 in free credits" in text
    assert calls == ["tok-1"]


@pytest.mark.asyncio
async def test_the_scheduler_kicks_only_when_a_probe_could_decide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_schedule_recovery_notice``'s gate, and its worker wiring, in one pass."""
    calls: list[str] = []
    app = _app(monkeypatch, tmp_path, arm_probe=True, calls=calls)
    tasks: list[asyncio.Task[None]] = []

    def fake_run_worker(coro, **kwargs):  # noqa: ANN001 — Textual's shape
        assert kwargs.get("thread") is False
        tasks.append(asyncio.create_task(coro))
        return MagicMock()

    app.run_worker = fake_run_worker  # type: ignore[method-assign]
    notice = NoticeBlock(RENDERED_QUOTA, "error")

    # Cold: a probe could decide, so a worker is scheduled — and its completion
    # extends the notice.
    app._schedule_recovery_notice(notice, RENDERED_QUOTA)
    assert len(tasks) == 1
    await tasks[0]
    assert "Check your inbox" in notice.text()

    # Warm cache, non-quota error, non-Radient provider: nothing more scheduled.
    app._schedule_recovery_notice(notice, RENDERED_QUOTA)
    app._schedule_recovery_notice(notice, RENDERED_AUTH)
    app._session = MagicMock(model_label="openai/gpt-5")
    app._schedule_recovery_notice(notice, RENDERED_QUOTA)
    assert len(tasks) == 1
