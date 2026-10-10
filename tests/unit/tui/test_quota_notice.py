"""The splash's pre-emptive no-quota line: cached-only, verdict-fed, one row.

WHY THIS FILE EXISTS. ``tui/quota_notice`` is the seam between the one verdict
(``providers.quota_notice``, merged in #2128) and the splash's status stack,
and each half has a failure mode the other's tests cannot see:

- the gathering path must NEVER fetch. The desktop route may spend one bounded
  live refresh; a splash paint may not — and a regression that wired
  ``fetch_usage`` into this read would still pass every route test. So the
  fetch layer is armed AFTER a real warm-up, and any touch fails the test.
- the copy must be the VERDICT's, never a second string set (the whole point
  of the merged verdict is ONE decision and one set of sentences), so the
  verdict is spied and the line must follow it — including states this branch
  does not know about yet (Radient's ``unverified`` arriving with PR2).
- the row must fit, shed and never double with the credential warning, which
  is geometry rather than logic and is pinned on ``build_welcome_lines`` —
  the same builder the view measures itself with.

The frame states the PR's evidence quotes live here too (``QUOTA_STATES`` and
``frame_app``), because the capture script (``scripts/quota_notice_shot.py``)
must render the SAME fixtures these tests pin rather than a drifting copy.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from rich.cells import cell_len
from rich.text import Text

from local_operator.harness.types import ModelSpec
from local_operator.providers import radient_recovery as rr
from local_operator.providers import usage as usage_module
from local_operator.providers.billing_links import BILLING_LINKS
from local_operator.providers.controller import (
    CatalogueEntry,
    ControllerAuthStore,
    ProviderController,
)
from local_operator.providers.quota_notice import (
    QuotaAction,
    QuotaVerdict,
    evaluate_quota_notice,
)
from local_operator.providers.radient_recovery import (
    CLAIM_URL,
    RecoveryFacts,
    VerificationFacts,
    recovery_line,
)
from local_operator.providers.usage import UsageAmount, UsageLimit, UsageReport
from local_operator.providers.usage_cache import USAGE_REPORT_TTL_MS, UsageCacheStore
from local_operator.tui import quota_notice as tui_quota
from local_operator.tui import theme as theme_mod
from local_operator.tui.quota_notice import QuotaNoticeLine, quota_notice_line
from local_operator.tui.widgets.welcome import (
    _PRIORITY_NOTICE,
    _PRIORITY_QUOTA,
    _PRIORITY_WARNING,
    WelcomeInfo,
    _status_rows,
    build_welcome_lines,
    session_welcome_info,
)
from tests.unit.providers.test_controller import FakeAuthStore
from tests.unit.tui.test_welcome import (
    ROOMY_H,
    ROOMY_W,
    FakeProviders,
    FakeSession,
    _info,
    _make_app,
    _settled_welcome,
    plain,
)

#: Frozen "now" for the pure matrix; frame stamps are wall-clock instead
#: (``quota_state``), because the app calls the verdict with its own clock.
NOW_MS = 1_800_000_000_000


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch):
    """No test may read the operator's own store, env or Radient cache."""
    for name in (
        "DEEPSEEK_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "RADIENT_API_KEY",
        "KIMI_API_KEY",
        "MOONSHOT_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    rr.reset_recovery_cache()
    yield
    rr.reset_recovery_cache()


# ---------------------------------------------------------------------------
# Fixtures and doubles
# ---------------------------------------------------------------------------


def _balance_limit(remaining: float, *, resets_at_ms: int | None = None) -> UsageLimit:
    """A denominator-less balance row, the shape DeepSeek/Kimi/Radient emit."""
    return UsageLimit(
        id="probe:balance",
        label="Credit balance (USD)",
        amount=UsageAmount(remaining=remaining),
        window="lifetime",
        resets_at_ms=resets_at_ms,
        shared=True,
    )


def _window_limit(
    *, used: float, total: float, resets_at_ms: int | None, label: str = "5 hour"
) -> UsageLimit:
    return UsageLimit(
        id="probe:window",
        label=label,
        amount=UsageAmount(used=used, limit=total, used_fraction=used / total),
        window=label,
        resets_at_ms=resets_at_ms,
        shared=True,
    )


def _report(
    provider: str,
    *rows: UsageLimit,
    fetched_at: int = NOW_MS,
    identity: str | None = None,
) -> UsageReport:
    return UsageReport(
        provider=provider,
        fetched_at=fetched_at,
        limits=list(rows),
        identity=identity,
    )


def _catalogue_entry(
    provider: str, model_id: str, *, input_price: float, output_price: float, routed: bool = False
) -> CatalogueEntry:
    return CatalogueEntry(
        provider=provider,
        model_id=model_id,
        label=model_id,
        context_window=64_000,
        input_price=input_price,
        output_price=output_price,
        connected=True,
        routed=routed,
    )


def _api_key_row() -> SimpleNamespace:
    return SimpleNamespace(credential_type="api_key", data={"key": "sk-test"})


class _RowsStore:
    """The credential-row surface ``_api_key_present`` reads."""

    def __init__(self, rows: list[Any]) -> None:
        self._rows = list(rows)

    def list_credentials(self, provider: str | None = None) -> list[Any]:
        return list(self._rows)


class QuotaController:
    """The controller facade ``quota_notice_line`` may read, in memory.

    Only the reads the module is allowed to make exist here, and the
    network-capable member (``fetch_usage``) is a RECORDER rather than an
    absence: a regression that starts fetching should fail an assertion, not
    be swallowed by the module's own degrade guard.
    """

    def __init__(
        self,
        *,
        reports: list[UsageReport] | None = None,
        entry: CatalogueEntry | None = None,
        expected: list[str] | None = None,
        api_key_rows: bool = False,
        missing: dict[str, bool] | None = None,
        boom: bool = False,
    ) -> None:
        self.reports = list(reports or [])
        self.entry = entry
        self.expected = list(expected or [])
        self.missing = dict(missing or {})
        self.boom = boom
        self._auth_store = _RowsStore([_api_key_row()] if api_key_rows else [])
        self.fetch_calls: list[list[str]] = []

    def is_usable(self, provider_id: str) -> bool:
        return not self.missing.get(provider_id, False)

    def cached_usage_reports(self, provider: str | None = None) -> list[UsageReport]:
        if self.boom:
            raise RuntimeError("usage cache unavailable")
        return list(self.reports)

    def expected_oauth_identities(self, provider: str) -> list[str]:
        return list(self.expected)

    def initial_catalogue(self, **_kwargs: Any) -> list[CatalogueEntry]:
        return [self.entry] if self.entry is not None else []

    def entry_for(self, provider: str, model_id: str, *, spec: Any = None) -> CatalogueEntry | None:
        return self.entry

    @property
    def auth_store(self) -> _RowsStore:
        return self._auth_store

    def can_report_usage(self, provider: str) -> bool:
        return False

    def usage_cache_age_ms(self, provider: str) -> int | None:
        return None

    async def fetch_usage(
        self, provider_ids: list[str] | None = None, *, force_refresh: bool = False
    ) -> list[UsageReport]:
        self.fetch_calls.append(list(provider_ids or []))
        return []


class QuotaSession(FakeSession):
    """The welcome tests' session, carrying the model pair the notice reads."""

    def __init__(self, provider: str = "deepseek", model_id: str = "deepseek-chat") -> None:
        super().__init__(f"{provider}/{model_id}")
        self._model = ModelSpec(provider=provider, model_id=model_id)

    @property
    def model(self) -> Any:
        return self._model


# ---------------------------------------------------------------------------
# The cached-only read: a real controller, a real warm row, an armed fetch layer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_read_answers_from_the_warm_row_and_never_fetches(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The warmer's row is the whole data source; the network layer stays untouched.

    Seeded exactly the way the app's warmer seeds it — one real
    ``fetch_usage`` with the provider endpoint stubbed — so the cache under
    test is the production cache. The fetch layer is then replaced: any
    further fetch (or a probe through the fetcher) is a regression this test
    exists to catch.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    store = FakeAuthStore()
    cache = UsageCacheStore(tmp_path / "usage_cache.db")
    # ``cast``: the double is the same deliberately-partial stand-in
    # ``test_controller`` uses behind unannotated fixtures; pyright cannot see
    # a duck as the protocol it stands in for.
    controller = ProviderController(
        cast(ControllerAuthStore, store), login_callbacks=None, usage_cache=cache
    )
    try:
        # ``api_keys`` is the surface ``get_api_key`` answers from — an
        # ``upsert_credential`` row alone never feeds the fetch path's
        # credential cascade (see ``FakeAuthStore`` in test_controller).
        store.api_keys["deepseek"] = "sk-deepseek-test"
        calls = {"n": 0}

        async def fetch_deepseek(client: Any, api_key: str) -> UsageReport:
            calls["n"] += 1
            return _report("deepseek", _balance_limit(0.0), fetched_at=int(time.time() * 1000))

        monkeypatch.setattr(usage_module, "fetch_deepseek_balance", fetch_deepseek)
        await controller.fetch_usage(["deepseek"])
        assert calls["n"] == 1

        # Armed AFTER the warm-up: the row is on disk, the network must not be.
        fetches: list[list[str]] = []

        async def recorded_fetch(*args: Any, **kwargs: Any) -> list[UsageReport]:
            fetches.append(list(args[0] if args else kwargs.get("provider_ids") or []))
            return []

        monkeypatch.setattr(ProviderController, "fetch_usage", recorded_fetch)
        monkeypatch.setattr(usage_module, "fetch_deepseek_balance", recorded_fetch)

        line = quota_notice_line(QuotaSession(), controller)
        assert line == QuotaNoticeLine(
            text="No balance on DeepSeek — top up at the DeepSeek platform.",
            url=BILLING_LINKS["deepseek"].url,
        )
        assert fetches == [], "the cached read must not fetch"
        assert calls["n"] == 1, "and it must not probe the provider endpoint either"
    finally:
        cache.close()


# ---------------------------------------------------------------------------
# Verdict reuse: the spied verdict is the only author
# ---------------------------------------------------------------------------


def test_the_line_is_the_verdicts_copy_from_the_verdicts_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every input is handed to the ONE verdict; every output comes back from it."""
    captured: dict[str, Any] = {}
    sentinel = QuotaVerdict(
        state="depleted",
        kind="balance",
        model_free=False,
        title="SENTINEL TITLE",
        body="SENTINEL BODY\nsecond line",
        actions=(
            QuotaAction("open_url", "Top up", url="https://example.test/up"),
            QuotaAction("refresh", "I topped up"),
        ),
    )

    def spy(**kwargs: Any) -> QuotaVerdict:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(tui_quota, "evaluate_quota_notice", spy)
    reports = [_report("deepseek", _balance_limit(0.0))]
    controller = QuotaController(reports=reports)

    line = quota_notice_line(QuotaSession(), controller, now_ms=NOW_MS)

    assert captured["provider"] == "deepseek"
    assert captured["model"] == "deepseek-chat"
    assert captured["reports"] == reports
    assert captured["expected_identities"] == []
    assert captured["api_key_present"] is False
    assert captured["now_ms"] == NOW_MS
    assert captured["radient_facts"] is None
    assert captured["resend_available"] is False
    # The copy is the sentinel's, newline-folded for one splash row; the URL
    # is the first open_url action's target.
    assert line == QuotaNoticeLine(text="SENTINEL BODY second line", url="https://example.test/up")


def test_an_empty_copy_verdict_shows_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ok``/``unknown``/``not_applicable`` are the verdict's own "say nothing"."""
    monkeypatch.setattr(
        tui_quota, "evaluate_quota_notice", lambda **kwargs: QuotaVerdict("ok", "balance", False)
    )
    controller = QuotaController(reports=[_report("deepseek", _balance_limit(0.0))])
    assert quota_notice_line(QuotaSession(), controller, now_ms=NOW_MS) is None


def test_a_url_already_in_the_sentence_is_not_printed_twice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Radient's sentences carry their own links; a trailing duplicate helps nobody."""
    url = BILLING_LINKS["radient"].url
    assert url is not None  # the dashboard table always carries Radient's link
    monkeypatch.setattr(
        tui_quota,
        "evaluate_quota_notice",
        lambda **kwargs: QuotaVerdict(
            "depleted",
            "radient",
            False,
            title="No credit left on Radient",
            body=f"You're out of credits. Top up in the Radient console: {url}",
            actions=(QuotaAction("open_url", "Top up at Radient", url=url),),
        ),
    )
    controller = QuotaController(reports=[_report("radient", _balance_limit(0.0))])
    line = quota_notice_line(QuotaSession("radient", "auto"), controller, now_ms=NOW_MS)
    assert line is not None
    assert line.url is None
    assert line.text.endswith(url)


@pytest.mark.asyncio
async def test_the_radient_sentence_is_the_cached_builders_and_the_probe_never_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The verify-to-claim copy is ``radient_recovery``'s, served from its cache.

    The runtime's own warm-up is exercised (a real store row, the module's own
    probe entry point), then the notice reads the facts-level cached arm: the
    verdict must classify ``unverified`` from those facts (never from sentence
    text), the sentence must be the builder's, and the probe must not run
    again on this path.
    """
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("radient", {"type": "oauth", "access": "tok-1", "refresh": "r"})
    probes: list[str] = []

    async def probe(token: str) -> VerificationFacts:
        probes.append(token)
        return VerificationFacts(signup_grant="pending", email_verified=False, grant_amount=5.0)

    monkeypatch.setattr(rr, "_probe_verification_async", probe)
    facts = await rr.get_recovery_facts(store=store)
    expected = recovery_line(facts)
    assert "Check your inbox" in expected
    assert probes == ["tok-1"]

    controller = QuotaController(reports=[_report("radient", _balance_limit(0.0))])
    line = quota_notice_line(QuotaSession("radient", "auto"), controller, now_ms=NOW_MS)

    assert line is not None
    assert line.text == " ".join(expected.split()), "the builder's sentence, unfolded once"
    assert "\n" not in line.text
    assert probes == ["tok-1"], "the notice must read the cached arm, never probe"
    # The claim URL already lives inside the sentence, so nothing is appended.
    assert line.url is None
    # Parroting the same inputs through the verdict names the state the row is
    # rendering — the one this branch does not spell itself.
    direct = evaluate_quota_notice(
        provider="radient",
        model="auto",
        reports=[_report("radient", _balance_limit(0.0))],
        now_ms=NOW_MS,
        radient_facts=facts,
    )
    assert direct.state == "unverified"
    assert line.text == " ".join(direct.body.split())


def test_no_cached_radient_answer_degrades_to_the_neutral_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No probe has answered yet: the verdict's neutral text stands in.

    The facts-level cached arm is empty (or its read fails); the notice must
    still render the neutral sentence rather than sending anything, probing,
    or guessing a verification state from nothing.
    """
    monkeypatch.setattr(rr, "recovery_facts_cached", lambda: None)
    controller = QuotaController(reports=[_report("radient", _balance_limit(0.0))])
    line = quota_notice_line(QuotaSession("radient", "auto"), controller, now_ms=NOW_MS)
    assert line is not None
    assert line.text == " ".join(recovery_line(RecoveryFacts(signed_in=None)).split())
    # The neutral sentence carries both of its URLs itself; none is appended.
    assert line.url is None


def test_a_cold_cache_with_a_stored_credential_still_never_probes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review round 1 (m1): credential present, cache cold — no probe, no block.

    The warm-cache tests can be served by either twin from the same cache, and
    the empty-store test gives a probing twin no token to probe with — so
    neither pins "this path never probes". Here a real store holds a Radient
    login while the process cache is cold (the autouse fixture resets it), and
    BOTH probe legs are recorders: swapping ``recovery_facts_cached()`` for
    ``get_recovery_facts_sync()`` would block the splash paint on the wire and
    land here as a recorded probe.
    """
    from local_operator.providers import auth_store as auth_store_mod
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("radient", {"type": "oauth", "access": "tok-cold", "refresh": "r"})
    monkeypatch.setattr(auth_store_mod, "shared_auth_store", lambda: store)

    sync_probes: list[str] = []
    async_probes: list[str] = []

    def probe_sync(token: str) -> VerificationFacts:
        sync_probes.append(token)
        return VerificationFacts(signup_grant="pending", email_verified=False)

    async def probe_async(token: str) -> VerificationFacts:
        async_probes.append(token)
        return VerificationFacts(signup_grant="pending", email_verified=False)

    monkeypatch.setattr(rr, "_probe_verification_sync", probe_sync)
    monkeypatch.setattr(rr, "_probe_verification_async", probe_async)

    controller = QuotaController(reports=[_report("radient", _balance_limit(0.0))])
    line = quota_notice_line(QuotaSession("radient", "auto"), controller, now_ms=NOW_MS)

    assert line is not None
    assert line.text == " ".join(recovery_line(RecoveryFacts(signed_in=None)).split())
    assert sync_probes == [], "the cached arm must never probe"
    assert async_probes == [], "and neither may the async twin"


@pytest.mark.asyncio
async def test_verified_radient_facts_shift_the_copy_to_the_topup_sentence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cached arm feeds the classifier both ways: claimed facts = top up."""
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore(tmp_path / "auth.db")
    store.upsert_credential("radient", {"type": "oauth", "access": "tok-1", "refresh": "r"})

    async def probe(token: str) -> VerificationFacts:
        return VerificationFacts(signup_grant="claimed", email_verified=True)

    monkeypatch.setattr(rr, "_probe_verification_async", probe)
    facts = await rr.get_recovery_facts(store=store)
    assert facts.verification is not None

    controller = QuotaController(reports=[_report("radient", _balance_limit(0.0))])
    line = quota_notice_line(QuotaSession("radient", "auto"), controller, now_ms=NOW_MS)
    assert line is not None
    assert "Top up" in line.text
    assert line.text == " ".join(recovery_line(facts).split())
    assert line.url is None


# ---------------------------------------------------------------------------
# The state matrix: what shows and what stays silent
# ---------------------------------------------------------------------------


def test_depleted_balance_shows_the_topup_sentence_and_link() -> None:
    controller = QuotaController(reports=[_report("deepseek", _balance_limit(0.0))])
    line = quota_notice_line(QuotaSession(), controller, now_ms=NOW_MS)
    assert line == QuotaNoticeLine(
        text="No balance on DeepSeek — top up at the DeepSeek platform.",
        url=BILLING_LINKS["deepseek"].url,
    )


def test_spent_plan_window_shows_the_reset_sentence_and_link() -> None:
    controller = QuotaController(
        reports=[
            _report(
                "anthropic",
                _window_limit(used=100, total=100, resets_at_ms=NOW_MS + 7_200_000),
            )
        ]
    )
    line = quota_notice_line(QuotaSession("anthropic", "claude-opus-5"), controller, now_ms=NOW_MS)
    assert line == QuotaNoticeLine(
        text="Anthropic limit reached — 5 hour resets in 2h.",
        url=BILLING_LINKS["anthropic"].url,
    )


def test_the_free_model_rule_comes_from_the_listing_row() -> None:
    """A quoted 0.0/0.0 listing row suppresses even a depleted account."""
    model_id = "deepseek/deepseek-chat-v3.1:free"
    reports = [_report("openrouter", _window_limit(used=100, total=100, resets_at_ms=None))]
    session = QuotaSession("openrouter", model_id)

    free = QuotaController(
        reports=reports,
        entry=_catalogue_entry("openrouter", model_id, input_price=0.0, output_price=0.0),
    )
    assert quota_notice_line(session, free, now_ms=NOW_MS) is None

    # The same account on a PAID route of the same provider is a notice: the
    # suppression must be the row's prices, not the provider.
    paid = QuotaController(
        reports=reports,
        entry=_catalogue_entry("openrouter", model_id, input_price=0.000001, output_price=0.0),
    )
    assert quota_notice_line(session, paid, now_ms=NOW_MS) is not None


def test_a_stale_row_is_not_evidence() -> None:
    aged = _report("deepseek", _balance_limit(0.0), fetched_at=NOW_MS - USAGE_REPORT_TTL_MS - 1)
    controller = QuotaController(reports=[aged])
    assert quota_notice_line(QuotaSession(), controller, now_ms=NOW_MS) is None


def test_no_reports_shows_nothing() -> None:
    assert quota_notice_line(QuotaSession(), QuotaController(), now_ms=NOW_MS) is None


def test_an_unavailable_report_shows_nothing() -> None:
    report = _report("deepseek", _balance_limit(0.0))
    report = dataclasses.replace(report, usage_unavailable=True)
    assert (
        quota_notice_line(QuotaSession(), QuotaController(reports=[report]), now_ms=NOW_MS) is None
    )


def test_a_positive_balance_shows_nothing() -> None:
    controller = QuotaController(reports=[_report("deepseek", _balance_limit(12.5))])
    assert quota_notice_line(QuotaSession(), controller, now_ms=NOW_MS) is None


def test_one_live_account_suppresses_the_notice() -> None:
    controller = QuotaController(
        reports=[
            _report(
                "anthropic",
                _window_limit(used=100, total=100, resets_at_ms=NOW_MS + 3_600_000),
                identity="a@example.com",
            ),
            _report(
                "anthropic",
                _window_limit(used=10, total=100, resets_at_ms=NOW_MS + 3_600_000),
                identity="b@example.com",
            ),
        ],
        expected=["a@example.com", "b@example.com"],
    )
    assert (
        quota_notice_line(QuotaSession("anthropic", "claude-opus-5"), controller, now_ms=NOW_MS)
        is None
    )


def test_an_uncovered_account_suppresses_the_notice() -> None:
    controller = QuotaController(
        reports=[
            _report(
                "anthropic",
                _window_limit(used=100, total=100, resets_at_ms=NOW_MS + 3_600_000),
                identity="a@example.com",
            )
        ],
        expected=["a@example.com", "b@example.com"],
    )
    assert (
        quota_notice_line(QuotaSession("anthropic", "claude-opus-5"), controller, now_ms=NOW_MS)
        is None
    )


def test_mixed_credentials_suppress_the_notice() -> None:
    """An OAuth login plus a live API key: the skipped route is unproven."""
    reports = [
        _report(
            "kimi",
            _window_limit(used=100, total=100, resets_at_ms=NOW_MS + 3_600_000),
            identity="user@example.com",
        )
    ]
    session = QuotaSession("kimi", "kimi-k2")
    mixed = QuotaController(reports=reports, expected=["user@example.com"], api_key_rows=True)
    assert quota_notice_line(session, mixed, now_ms=NOW_MS) is None

    # The same depleted account WITHOUT the second route is a notice.
    single = QuotaController(reports=reports, expected=["user@example.com"])
    assert quota_notice_line(session, single, now_ms=NOW_MS) is not None


def test_a_provider_without_a_fetcher_shows_nothing() -> None:
    """No endpoint means no evidence, ever: the notice must stay silent.

    The fixture feeds a depleted-LOOKING report anyway: the point is that the
    verdict's ``usage_supported`` gate refuses to look at it. ``openai`` is
    deliberately NOT the example — its OAuth route has a fetcher, so a
    ChatGPT-login user in a spent window DOES earn a notice; the no-route
    examples are API-key-only providers like ``google``.
    """
    assert usage_module.usage_supported("google") is False
    controller = QuotaController(reports=[_report("google", _balance_limit(0.0))])
    assert (
        quota_notice_line(QuotaSession("google", "gemini-3-pro"), controller, now_ms=NOW_MS) is None
    )


def test_a_broken_read_degrades_to_no_line() -> None:
    assert quota_notice_line(QuotaSession(), QuotaController(boom=True), now_ms=NOW_MS) is None


def test_no_spec_or_no_facade_shows_nothing() -> None:
    controller = QuotaController(reports=[_report("deepseek", _balance_limit(0.0))])
    assert quota_notice_line(FakeSession(), controller, now_ms=NOW_MS) is None
    assert quota_notice_line(None, controller, now_ms=NOW_MS) is None
    assert quota_notice_line(QuotaSession(), None, now_ms=NOW_MS) is None


# ---------------------------------------------------------------------------
# The row: geometry, ink, shedding, co-existence
# ---------------------------------------------------------------------------


def _quota_info(**overrides: Any) -> WelcomeInfo:
    fields: dict[str, Any] = dict(
        quota_notice=QuotaNoticeLine(
            text="No balance on DeepSeek — top up at the DeepSeek platform.",
            url=BILLING_LINKS["deepseek"].url,
        )
    )
    fields.update(overrides)
    return dataclasses.replace(_info("deepseek"), **fields)


def _row_with(lines: list[Any], needle: str) -> Any:
    matches = [line for line in lines if needle in line.plain]
    assert matches, [line.plain for line in lines]
    return matches[0]


def test_the_row_paints_beside_the_credential_warning_in_the_warning_ink() -> None:
    lines = build_welcome_lines(_quota_info(), ROOMY_W, ROOMY_H)
    body = plain(lines)
    assert body.count("not logged in") == 1
    assert body.count("No balance on DeepSeek") == 1

    # One row above the login warning: both amber `!` facts, the warning keeps
    # its place as the last status row.
    quota_index = next(i for i, line in enumerate(lines) if "No balance on DeepSeek" in line.plain)
    warning_index = next(i for i, line in enumerate(lines) if "not logged in" in line.plain)
    assert quota_index == warning_index - 1

    warning = theme_mod.semantic_color("warning")
    row = _row_with(lines, "No balance on DeepSeek")
    styles = [str(span.style) for span in row.spans] or [str(row.style)]
    assert warning in " ".join(styles), styles
    assert "!" in row.plain


def test_the_url_is_dropped_whole_when_the_rows_width_cannot_hold_it() -> None:
    url = BILLING_LINKS["deepseek"].url
    assert url is not None  # the dashboard table always carries DeepSeek's link
    info = _quota_info()
    full = f"! No balance on DeepSeek — top up at the DeepSeek platform. {url}"
    exact = cell_len(full)

    kept = plain(build_welcome_lines(info, exact, ROOMY_H))
    assert url in kept, "at exactly the row's width the link is kept"

    dropped = plain(build_welcome_lines(info, exact - 1, ROOMY_H))
    assert "No balance on DeepSeek — top up at the DeepSeek platform." in dropped
    assert url not in dropped, "one cell short: the link goes WHOLE, never half-printed"
    assert "platform.deepseek" not in dropped


def test_a_sentence_wider_than_the_row_truncates_from_the_tail() -> None:
    """The condition stays readable; only the tail goes (it is not a command)."""
    lines = build_welcome_lines(_quota_info(), 40, ROOMY_H)
    row = _row_with(lines, "No balance")
    assert row.plain.rstrip().endswith("…")
    assert "! No balance on DeepSeek" in row.plain


def test_the_notice_row_never_moves_the_shared_pad() -> None:
    """Design round 1 (D1): the stack's left edge is identical with the row on/off.

    The row is a sentence centred on its own width, like the tip; the shared
    pad is computed without it. Before the fix the pad collapsed to zero the
    moment the row landed — the whole stack re-anchored left by half the row
    (280 px at 96 cells) on the cold-boot path the warmer repaints.
    """
    width = ROOMY_W
    with_row = build_welcome_lines(_quota_info(), width, ROOMY_H)
    without = build_welcome_lines(_quota_info(quota_notice=None), width, ROOMY_H)
    assert len(with_row) == len(without) + 1, "exactly the one measured row is added"

    quota = _row_with(with_row, "No balance on DeepSeek")
    assert quota.plain.strip().startswith("! "), "the row keeps its warning glyph"
    pad = len(quota.plain) - len(quota.plain.lstrip(" "))
    assert pad == (width - cell_len(quota.plain.strip())) // 2, "centred on its own width"

    def pads(lines: list[Any]) -> list[int]:
        return [
            len(line.plain) - len(line.plain.lstrip(" "))
            for line in lines
            if line.plain.strip() and "No balance on DeepSeek" not in line.plain
        ]

    assert pads(with_row) == pads(without), "the shared pad must not see the notice row"


def test_no_url_is_half_printed_at_any_width() -> None:
    """Design round 1 (D2) / QA Q2: a URL shows whole or not at all, every width.

    The appended-URL rule only governs a URL appended at the tail; Radient's
    rides INSIDE its sentence, where the final truncation used to cut
    mid-address across content widths 181–227. ``_status_rows`` is swept with
    the same truncation the builder applies, so the band cannot regress
    quietly.
    """
    pending = RecoveryFacts(
        signed_in=True,
        verification=VerificationFacts(
            signup_grant="pending", email_verified=False, grant_amount=5.0
        ),
    )
    radient = dataclasses.replace(
        _info("radient"),
        quota_notice=QuotaNoticeLine(text=" ".join(recovery_line(pending).split()), url=None),
    )
    for info, url in ((_quota_info(), BILLING_LINKS["deepseek"].url), (radient, CLAIM_URL)):
        assert url is not None
        for width in range(24, 260):
            rows = _status_rows(info, width)
            row = next(line for priority, line in rows if priority == _PRIORITY_QUOTA)
            rendered = Text(row.plain)
            rendered.truncate(width, overflow="ellipsis")
            text = rendered.plain
            if url in text:
                continue
            assert "https://" not in text, (width, text)

    # The band's middle: at 200 the address is gone WHOLE — with the clause
    # that introduced it — rather than losing its tail mid-name.
    rows = _status_rows(radient, 200)
    row = next(line for priority, line in rows if priority == _PRIORITY_QUOTA)
    rendered = Text(row.plain)
    rendered.truncate(200, overflow="ellipsis")
    assert rendered.plain.rstrip().endswith("verification email…"), rendered.plain


def test_the_stack_orders_and_sheds_the_quota_row_before_the_warning() -> None:
    """Priorities put the row between the notice and the login warning."""
    assert _PRIORITY_NOTICE < _PRIORITY_QUOTA < _PRIORITY_WARNING

    info = dataclasses.replace(
        _quota_info(),
        notice="anthropic quota low — falling back to zai/glm-5.3",
    )
    texts = [text.plain for _, text in _status_rows(info, ROOMY_W)]
    quota_index = next(i for i, t in enumerate(texts) if "No balance on DeepSeek" in t)
    warning_index = next(i for i, t in enumerate(texts) if "not logged in" in t)
    notice_index = next(i for i, t in enumerate(texts) if "anthropic quota low" in t)
    assert notice_index < quota_index < warning_index

    # Under height pressure the quota row goes first, the warning stays: it is
    # the one row that changes what the user must DO.
    two = build_welcome_lines(info, ROOMY_W, 2)
    assert len(two) == 2
    assert "not logged in" in plain(two) and "No balance on DeepSeek" in plain(two)
    one = build_welcome_lines(info, ROOMY_W, 1)
    assert len(one) == 1
    assert "not logged in" in plain(one)
    assert "No balance on DeepSeek" not in plain(one)


# ---------------------------------------------------------------------------
# Wiring: session_welcome_info and the splash through a real pilot
# ---------------------------------------------------------------------------


def test_session_welcome_info_reads_the_quota_line() -> None:
    controller = QuotaController(
        reports=[_report("deepseek", _balance_limit(0.0), fetched_at=int(time.time() * 1000))]
    )
    info = session_welcome_info(QuotaSession(), controller)
    assert info.quota_notice == QuotaNoticeLine(
        text="No balance on DeepSeek — top up at the DeepSeek platform.",
        url=BILLING_LINKS["deepseek"].url,
    )


def test_session_welcome_info_degrades_when_the_quota_read_fails() -> None:
    assert session_welcome_info(QuotaSession(), QuotaController(boom=True)).quota_notice is None
    # A facade with no quota surface at all (an embedding host) is silence too.
    assert session_welcome_info(QuotaSession(), FakeProviders(missing={})).quota_notice is None
    assert session_welcome_info(None, QuotaController()).quota_notice is None


@pytest.mark.asyncio
async def test_the_splash_shows_the_row_and_adds_exactly_one_measured_row() -> None:
    """Through the real app: the row renders, measures, and does not scroll."""
    controller = QuotaController(
        reports=[_report("deepseek", _balance_limit(0.0), fetched_at=int(time.time() * 1000))]
    )
    app = _make_app(QuotaSession(), controller)
    async with app.run_test(size=(100, 30)) as pilot:
        welcome = await _settled_welcome(pilot)
        await pilot.pause()
        info = welcome._info
        assert info.quota_notice is not None
        lines = build_welcome_lines(info, welcome.size.width, welcome.size.height)
        assert "No balance on DeepSeek" in plain(lines)

        # The widget is content-sized (pinned by `test_visible_on_boot`), so the
        # row is MEASURED: removing the notice removes exactly one row.
        assert len(lines) == welcome.size.height
        without = dataclasses.replace(info, quota_notice=None)
        assert len(build_welcome_lines(without, welcome.size.width, welcome.size.height)) == (
            len(lines) - 1
        )
        # No scrollbar: virtual equals actual on the boot screen (the playbook's
        # tall-overlay check — the notice must not push the screen scrollable).
        assert app.screen.virtual_size.height == app.screen.size.height


@pytest.mark.asyncio
async def test_the_warm_worker_refreshes_the_splash_when_the_row_lands() -> None:
    """A cold splash picks the notice up from the warm-up worker, not luck.

    The app.py hunk under test: the warmer's row is what the notice reads, so
    the worker must re-snapshot the splash when it lands — otherwise a cold
    boot would show the notice only after some unrelated refresh event.
    """
    controller = QuotaController(reports=[])
    session = QuotaSession()
    app = _make_app(session, controller)
    async with app.run_test(size=(100, 30)) as pilot:
        welcome = await _settled_welcome(pilot)
        await pilot.pause()
        assert welcome._info.quota_notice is None

        warmed = [_report("deepseek", _balance_limit(0.0), fetched_at=int(time.time() * 1000))]

        async def fetch_usage(
            provider_ids: list[str] | None = None, *, force_refresh: bool = False
        ) -> list[UsageReport]:
            controller.reports = list(warmed)
            return []

        controller.fetch_usage = fetch_usage  # type: ignore[method-assign]
        controller.can_report_usage = lambda provider: True  # type: ignore[method-assign]
        controller.usage_cache_age_ms = lambda provider: None  # type: ignore[method-assign]

        app._warm_usage_background()
        for _ in range(50):
            await pilot.pause()
            if welcome._info.quota_notice is not None:
                break
        assert welcome._info.quota_notice is not None
        assert "No balance on DeepSeek" in welcome._info.quota_notice.text


# ---------------------------------------------------------------------------
# Frame states: shared with scripts/quota_notice_shot.py
# ---------------------------------------------------------------------------

#: Every state the capture script renders. ``narrow`` is a size of ``depleted``
#: rather than a state of its own — the script maps it.
QUOTA_STATES = (
    "depleted",
    "plan-window",
    "radient-unverified",
    "free-model",
    "unknown",
    "connected",
    "coexist",
)


def quota_state(name: str, *, now_ms: int | None = None) -> tuple[QuotaSession, QuotaController]:
    """One frame state as ``(session, controller)``.

    Wall-clock stamps by default, because the app passes its own clock to the
    verdict; ``now_ms`` is for tests that need a frozen instant. Every report
    the frame shows is FRESH on purpose — a frame of a stale state would show
    no row at all, which is the ``unknown`` state's job.
    """
    now = int(time.time() * 1000) if now_ms is None else now_ms
    if name == "depleted":
        return QuotaSession("deepseek", "deepseek-chat"), QuotaController(
            reports=[_report("deepseek", _balance_limit(0.0), fetched_at=now)]
        )
    if name == "plan-window":
        return QuotaSession("anthropic", "claude-opus-5"), QuotaController(
            reports=[
                _report(
                    "anthropic",
                    _window_limit(used=100, total=100, resets_at_ms=now + 59 * 60_000),
                    fetched_at=now,
                )
            ]
        )
    if name == "radient-unverified":
        # The sentence comes from the seeded process cache (see
        # ``install_radient_pending_fixture``), like a runtime whose 402 path
        # already probed; the frame itself must not probe.
        return QuotaSession("radient", "auto"), QuotaController(
            reports=[_report("radient", _balance_limit(0.0), fetched_at=now)]
        )
    if name == "free-model":
        model_id = "deepseek/deepseek-chat-v3.1:free"
        return QuotaSession("openrouter", model_id), QuotaController(
            reports=[
                _report(
                    "openrouter",
                    _window_limit(used=100, total=100, resets_at_ms=None),
                    fetched_at=now,
                )
            ],
            entry=_catalogue_entry("openrouter", model_id, input_price=0.0, output_price=0.0),
        )
    if name == "unknown":
        return QuotaSession("deepseek", "deepseek-chat"), QuotaController(
            reports=[
                _report(
                    "deepseek",
                    _balance_limit(0.0),
                    fetched_at=now - USAGE_REPORT_TTL_MS - 60_000,
                )
            ]
        )
    if name == "connected":
        return QuotaSession("anthropic", "claude-opus-5"), QuotaController(
            reports=[
                _report(
                    "anthropic",
                    _window_limit(used=20, total=100, resets_at_ms=now + 3_600_000),
                    fetched_at=now,
                )
            ]
        )
    if name == "coexist":
        # Both rows at once: a render-level fixture (in production a depleted
        # report and a missing credential for the same provider rarely line up
        # — the report would be stale — but the layout must hold when they do).
        return QuotaSession("deepseek", "deepseek-chat"), QuotaController(
            reports=[_report("deepseek", _balance_limit(0.0), fetched_at=now)],
            missing={"deepseek": True},
        )
    raise ValueError(f"unknown quota frame state: {name}")


def install_radient_pending_fixture() -> None:
    """Seed the Radient process cache with a pending-verification result.

    The sentence is the REAL ``recovery_line`` output for the facts a pending
    signup produces (the verify-to-claim branch); only the ``/me`` probe that
    normally feeds the cache is skipped, because a frame must not cross the
    network. The unit tests exercise the real cached arm instead, through
    ``get_recovery_facts``.
    """
    facts = RecoveryFacts(
        signed_in=True,
        verification=VerificationFacts(
            signup_grant="pending", email_verified=False, grant_amount=5.0
        ),
    )
    rr.reset_recovery_cache()
    rr._remember(facts, time.monotonic())


def frame_app(state: str):
    """The real ``OperatorApp`` for one capture state.

    Fixtures live here and rendering does not: the script (and any reviewer)
    drives the same app the product ships, over a controller facade that only
    carries the reads the notice is allowed to make.
    """
    if state == "radient-unverified":
        install_radient_pending_fixture()
    session, controller = quota_state(state)
    return _make_app(session, controller)
