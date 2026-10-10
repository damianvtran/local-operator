"""The welcome splash's pre-emptive "no quota" line: one CACHED-ONLY verdict read.

WHY THIS EXISTS. A session's first send can be refused because the account
behind the selected model is empty, and every surface that can say so now asks
one function (``providers.quota_notice.evaluate_quota_notice``). The desktop
renderer reads it over ``GET /v1/desktop/quota-notice``. The TUI cannot: a
splash paint runs on the app's event loop, and the desktop route's bounded live
refresh would open the composer's first frame on a network call. This module is
therefore the TUI's gathering step — it feeds the SAME verdict from the usage
row the 60 s warmer (``OperatorApp._warm_usage_background``) already keeps
warm, and authors nothing itself: every sentence comes back from the verdict,
which in turn delegates Radient's copy to ``radient_recovery``.

NO NETWORK, BY CONSTRUCTION. Every read below is a local one:
``ProviderController.cached_usage_reports`` (the shared cache; the warmer runs
the fetches), ``initial_catalogue``/``entry_for`` (the same disk peeks the
``/model`` picker makes per keystroke), ``expected_oauth_identities`` and the
auth-store rows (SQLite), and ``radient_recovery``'s cached arm, which never
probes. A stale or missing row is not refreshed here — the verdict's own
freshness rule refuses to speak on it, and the warmer's next tick is what
repairs it. That is also why there is no ``refresh`` trigger in this module:
the one cadence that matters is the worker's, and the app pokes the splash
(the ``refresh_info`` call in ``_warm_usage_worker``) when that worker lands.

THE TWO RULES THAT ARE EASY TO GET WRONG HERE, and where they come from:
- The free-model suppression needs the CACHED LISTING first, ``entry_for``
  second — aggregators ship no static rows, so the quoted ``0.0/0.0`` of a
  listing-derived ``:free`` route exists only in the listing document. This is
  the desktop route's ``_quota_model_entry`` rule (round-1 M1 there) mirrored
  rather than re-derived; skipping it would re-show the false notice that
  round fixed.
- Mixed credentials suppress the notice: an OAuth login plus a live API key
  means the fetcher only ran the OAuth route, so the API-key half is unproven.
  The route's ``_api_key_present`` is mirrored for the same reason — including
  its bias: an unreadable store answers ``True`` (absence is the half the
  notice needs; an unreadable store cannot vouch for it).

Porting the two helpers rather than importing the route is deliberate: the
server layer is not importable from ``tui``, and the rules they encode are
verdict INPUTS, not copy. The route's docstrings remain the authoritative
statement of each.

NOTHING HERE RAISES. A quota hint is a decoration on the first frame; a
reduced host, a mid-teardown session or an unreadable store degrades to "show
nothing" rather than taking the splash down — the same contract
``session_welcome_info`` applies to its own reads.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from local_operator.providers.quota_notice import evaluate_quota_notice

if TYPE_CHECKING:  # pragma: no cover - import weight guard, not a behaviour
    from local_operator.providers.controller import CatalogueEntry
    from local_operator.providers.radient_recovery import RecoveryFacts


@dataclass(frozen=True)
class QuotaNoticeLine:
    """One advisory row for the splash, as the renderer needs it.

    ``text`` is the verdict's body verbatim except for newline folding — the
    splash row is a single line, and the Radient sentences may carry two.
    ``url`` is the first ``open_url`` action's destination, kept separate so a
    width too narrow to hold both can drop the link WHOLE (the same rule the
    login warning's ``/login <provider>`` tail follows) instead of clipping
    half of an address nobody can open. It is ``None`` when the sentence
    already carries its URL (Radient's every branch does) — printing it twice
    would help nobody.
    """

    text: str
    url: str | None = None


def quota_notice_line(
    session: Any | None,
    providers: Any | None,
    *,
    now_ms: int | None = None,
) -> QuotaNoticeLine | None:
    """The advisory line for the session's model, or ``None`` for "show nothing".

    ``session`` supplies the model pair the notice is about (the SELECTED
    spec — the same one the warmer keeps warm) and ``providers`` is the
    ``ProviderController`` facade. Both are read attribute-tolerantly because
    embedding hosts and the pilot fakes pass reduced objects; anything that
    cannot be answered degrades to ``None``.

    ``now_ms`` is the verdict's clock, injectable for tests. The verdict is
    the authority on every decision — freshness, every-account coverage, the
    free model, the mixed-credential refusal — and this function adds exactly
    the two transforms a one-row terminal rendering needs (newline folding,
    URL extraction above).
    """
    try:
        spec = getattr(session, "model", None)
        provider = str(getattr(spec, "provider", "") or "")
        model_id = str(getattr(spec, "model_id", "") or "")
        if not provider or not model_id or providers is None:
            return None

        reports = providers.cached_usage_reports(provider)
        expected = providers.expected_oauth_identities(provider)
        api_key_present = bool(expected) and _api_key_present(providers, provider)
        entry = _model_entry(providers, provider, model_id)
        verdict = evaluate_quota_notice(
            provider=provider,
            model=model_id,
            reports=reports,
            expected_identities=expected,
            api_key_present=api_key_present,
            entry=entry,
            now_ms=now_ms if now_ms is not None else int(time.time() * 1000),
            radient_facts=_radient_facts(provider),
            # ``resend_available`` gates only the resend ACTION, which v1
            # (text and URL, no press) never renders; its honest answer needs
            # the async store cascade that a splash paint must not run. False
            # is inert here, not a claim about the account.
            resend_available=False,
        )
    except Exception:  # noqa: BLE001 — a decoration must never take the first frame down
        return None

    # "Empty copy = show nothing" is the verdict's own contract for
    # ok/unknown/not_applicable; this reads the copy rather than a state list,
    # so a state added later (the Radient ``unverified`` of PR2) renders
    # without a change here.
    text = verdict.body or verdict.title
    if not text:
        return None
    text = " ".join(text.split())
    url = next((a.url for a in verdict.actions if a.id == "open_url" and a.url), None)
    if url and url in text:
        url = None
    return QuotaNoticeLine(text=text, url=url)


def _model_entry(providers: Any, provider: str, model_id: str) -> "CatalogueEntry | None":
    """The pair's catalogue row for the free-model rule, CACHED LISTING first.

    Mirrors ``server/routes/desktop_quota.py::_quota_model_entry`` (round-1 M1
    there): ``entry_for`` resolves ``static_models()`` plus the session spec,
    and an aggregator has neither — every listing-derived ``:free`` route
    would answer ``None`` and a spent account would be warned about a model it
    can send on. The listing rows are the ones the picker paints and the only
    place a quoted ``0.0`` exists for those providers. Both branches price a
    stated zero to exactly ``0.0`` and a silence to ``-1``; the read is
    guarded so a broken frame falls back to the registry row.
    """
    try:
        rows = providers.initial_catalogue()
    except Exception:  # noqa: BLE001 — a broken frame falls back to the registry row
        rows = []
    for row in rows:
        if row.provider == provider and row.model_id == model_id:
            return row
    return providers.entry_for(provider, model_id)


def _api_key_present(providers: Any, provider: str) -> bool:
    """Whether a NON-OAuth API-key credential exists for ``provider``.

    Mirrors the route's async twin; see its docstring for the full argument.
    ``AuthStore.get_api_key`` is deliberately NOT the probe (its cascade
    returns an OAuth row's own token), and an unreadable answer is ``True``:
    the mix rule needs "the skipped route had nothing to spend", and absence
    is the half no unreadable store can vouch for.
    """
    from local_operator.providers.registry import (
        credential_provider_id,
        resolve_env_key,
    )

    try:
        if resolve_env_key(provider):
            return True
    except Exception:  # noqa: BLE001 — an unreadable env answer cannot prove absence
        return True
    try:
        rows = providers.auth_store.list_credentials(credential_provider_id(provider))
    except Exception:  # noqa: BLE001 — unreadable store cannot prove the absence
        return True
    return any(getattr(row, "credential_type", None) == "api_key" for row in rows)


def _radient_facts(provider: str) -> "RecoveryFacts | None":
    """Radient's cached verification facts, or ``None`` when only a probe could say.

    The verdict classifies ``unverified``/``verified`` from FACTS — never from
    sentence text (``account_state``'s own docstring) — so this reads
    ``radient_recovery``'s facts-level cached arm, which never probes: a warm
    probe cache answers, and ``None`` (a credential exists but no probe has
    run) lets the verdict's own neutral rendering stand in. A live probe is
    the one thing this path must not do (see the module docstring), so the
    awaited arms are intentionally unused.
    """
    try:
        from local_operator.providers.registry import credential_provider_id

        if credential_provider_id(provider) != "radient":
            return None
        from local_operator.providers.radient_recovery import recovery_facts_cached

        return recovery_facts_cached()
    except Exception:  # noqa: BLE001 — an unreadable Radient read degrades to neutral
        return None
