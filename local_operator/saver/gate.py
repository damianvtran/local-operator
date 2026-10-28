"""The Radient Saver gate and its one resolver (contract §9).

WHY ONE RESOLVER. Saver can be turned on or off from several places over its
slices — the setting, an environment override, a session's own ``/radient``
choice (LO2), an agent's pin (LO3), a parent's value at spawn — and two readers
deciding the question differently is how a session ends up uploading history
the operator switched off. :func:`resolve_saver` is the only answer, first match
wins, and it returns WHY so a surface can name the remedy.

LO1 SCOPE. The gate (step 1) and the default (step 5) are wired. Steps 2–4 have
their parameters in place and are honoured when a caller passes them, but no
caller does yet: LO2 wires the session choice and LO3 the agent pins.

THE GATE IS OFF BY DEFAULT and the availability half is read rarely: a fresh
owner-pinned ``GET /v1/tunnels/billing`` through ``tunnels.cli._positive_balance``
— at enable, on resume and on a 402 only, never per prefetch (§8, S-31) — and an
unknown answer counts as no.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

#: Registry key of the mode setting (``settings_io.py``).
SETTING_KEY = "radient.saver.mode"
#: The env override, the ``LOP_ASK_GATE`` precedent (``asks/policy.py``).
ENV_OVERRIDE = "LOP_RADIENT_SAVER"
MODES = ("off", "opt_in", "auto")
#: The shipped default, supplied by :func:`configured_mode` when the key is
#: ABSENT. The "later flip" (§9) is this one value (and the registry row's
#: display default, pinned to it by a test). It is never in
#: ``config.DEFAULT_CONFIG``: that would persist an explicit ``off`` on any
#: unrelated settings write, and explicit settings are untouched by the flip.
DEFAULT_MODE = "off"

Source = Literal[
    "gate",
    "signed_out",
    "no_credit",
    "session",
    "agent",
    "parent",
    "auto_unavailable",
    "default",
]


@dataclass(frozen=True)
class SaverDecision:
    on: bool
    source: Source


def configured_mode(settings: Mapping[str, Any] | None) -> str:
    """``radient.saver.mode`` with the env override applied.

    The env value wins when it names a mode; a typo or an empty value falls back
    to the setting, and an unrecognised setting falls back to ``off`` — the
    fail-closed direction for a feature that uploads history.
    """
    override = os.environ.get(ENV_OVERRIDE, "").strip().lower()
    if override in MODES:
        return override
    radient = settings.get("radient") if isinstance(settings, Mapping) else None
    saver = radient.get("saver") if isinstance(radient, Mapping) else None
    mode = saver.get("mode") if isinstance(saver, Mapping) else None
    if isinstance(mode, str) and mode.strip().lower() in MODES:
        return mode.strip().lower()
    return DEFAULT_MODE


@dataclass(frozen=True)
class Availability:
    """The gate's account half: signed in, and a positive fresh balance."""

    signed_in: bool
    positive_balance: bool


def resolve_saver(
    settings: Mapping[str, Any] | None,
    availability: Availability | None,
    *,
    session_choice: bool | None = None,
    agent_pin: bool | None = None,
    parent_value: bool | None = None,
) -> SaverDecision:
    """``resolve_saver(session) -> (on, source)``, first match wins (§9).

    1. Gate: mode ``off`` ⇒ off (``gate``); not signed in ⇒ off
       (``signed_out``); no positive fresh balance ⇒ off (``no_credit``).
       ``availability=None`` (not read yet, or unreadable) is "unknown", which
       counts as no.
    2. The session's explicit choice (``/radient``, ``--radient``,
       ``SessionSpec.radient_saver``) — LO2.
    3. Agent pin (team slot, then profile) — LO3.
    4. A child's parent value at spawn — LO3.
    5. Default: ``auto`` ⇒ **OFF, always, in LO1** (``auto_unavailable``);
       ``opt_in`` ⇒ off.

    WHY ``auto`` IS UNREACHABLE. It turns Saver on for a user who never asked,
    which is only safe once the verifier can PROVE an epoch cheaper than the
    history it replaces. Check 6 is a provenance rule in rev7 and the
    non-amplification clause lands with the token comparison (PR #3078), so an
    accepted epoch could otherwise cost several times more than the unreduced
    history while §5a still bills the user for "removed tokens". That is the
    same shape as the signup-grant floor: an unattended default plus an
    unenforced economic bound. ``opt_in`` and an explicit session choice are
    unaffected — a user who asks is a different case from one who never did,
    and LO2's session choice still wins at step 2.
    """
    mode = configured_mode(settings)
    if mode == "off":
        return SaverDecision(False, "gate")
    if availability is None or not availability.signed_in:
        return SaverDecision(False, "signed_out")
    if not availability.positive_balance:
        return SaverDecision(False, "no_credit")
    if session_choice is not None:
        return SaverDecision(session_choice, "session")
    if agent_pin is not None:
        return SaverDecision(agent_pin, "agent")
    if parent_value is not None:
        return SaverDecision(parent_value, "parent")
    if mode == "auto":
        return SaverDecision(False, "auto_unavailable")
    return SaverDecision(False, "default")


#: The availability read's TOTAL deadline (S-4): a refresh, the request and the
#: body together. httpx's own ``timeout=30`` is per read, so a 4 B/s drip held a
#: turn 9.2 s and one byte every 29 s could hold it indefinitely.
AVAILABILITY_DEADLINE_S = 5.0
#: After a read that could not answer (timeout, transport, non-200), the next
#: one is started no sooner than this: the prepare path's transport back-off.
AVAILABILITY_RETRY_S = 60.0
#: A billing quote is a few hundred bytes; anything past this is not one.
MAX_BILLING_BYTES = 16 * 1024


async def read_availability() -> Availability | None:
    """Signed in, and a positive balance from a FRESH owner-pinned billing read.

    Reuses the tunnel plane's credential selection and its ``_positive_balance``
    predicate (missing, boolean or non-finite amounts are unavailable). Answers
    ``None`` when the read could not complete (timeout, transport, non-200,
    oversized or encoded body), which the caller retries after a back-off;
    never raises.

    The request follows the prepare path's transport rules, not
    ``RadientTunnels.request``'s, because this read happens on every
    Saver-enabled session (S-4):

    - ``Accept-Encoding: identity`` and a ``Content-Encoding`` answer refused
      unread: no decoder is built, so a 61 KB gzip can no longer inflate to
      187 MiB (the S3-7 surface ``/saver/prepare`` already closed);
    - the raw body capped at :data:`MAX_BILLING_BYTES` while it streams;
    - the host is the configured Radient API base (``RADIENT_API_BASE_URL``,
      the same base the prepare path uses), under the same exact-host rule, so
      a staging setup no longer sends the read to production (QA, other
      observations);
    - ``trust_env=False`` and ``follow_redirects=False``.

    The total deadline is applied by :func:`start_availability_read`.
    """
    import json

    import httpx

    from local_operator.env import get_env_config, normalize_radient_api_base_url
    from local_operator.providers.auth_store import AuthStore
    from local_operator.saver.client import _read_capped, destination_allowed
    from local_operator.tunnels.api import credential_id
    from local_operator.tunnels.cli import _positive_balance

    try:
        selected = credential_id(None)
    except Exception:  # noqa: BLE001 — no single Radient login: signed out
        return Availability(signed_in=False, positive_balance=False)
    base = normalize_radient_api_base_url(get_env_config().radient_api_base_url)
    if not destination_allowed(base):
        return Availability(signed_in=True, positive_balance=False)
    try:
        from contextlib import closing

        with closing(AuthStore()) as store:
            credentials = await store.ensure_oauth_fresh_or_raise(selected)
        token = credentials.get("access") if credentials else None
        if not isinstance(token, str) or not token:
            return Availability(signed_in=False, positive_balance=False)
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
            request = client.build_request(
                "GET",
                base + "/tunnels/billing",
                headers={"Authorization": f"Bearer {token}", "Accept-Encoding": "identity"},
            )
            response = await client.send(request, stream=True)
            try:
                if response.status_code != 200:
                    return None
                raw = await _read_capped(response, MAX_BILLING_BYTES)
            finally:
                await response.aclose()
        envelope = json.loads(raw)
    except Exception:  # noqa: BLE001 — could not answer: retried after the back-off
        return None
    quote = envelope.get("result") if isinstance(envelope, dict) else None
    return Availability(
        signed_in=True,
        positive_balance=isinstance(quote, dict) and _positive_balance(quote),
    )


class AvailabilityRead:
    """One background availability read, never awaited by a turn (R-1/S-4/Q1).

    ``result()`` is ``None`` ("not known yet", which the resolver treats as
    off) until the read answers. A read that could not answer, or ran past
    :data:`AVAILABILITY_DEADLINE_S`, ``expired()`` after
    :data:`AVAILABILITY_RETRY_S`, and the caller starts a new one then.
    """

    def __init__(self) -> None:
        import asyncio

        self._started = _monotonic()
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> Availability | None:
        import asyncio

        try:
            async with asyncio.timeout(AVAILABILITY_DEADLINE_S):
                return await read_availability()
        except Exception:  # noqa: BLE001 — a deadline is "could not answer"
            return None

    def result(self) -> Availability | None:
        if not self._task.done() or self._task.cancelled():
            return None
        return self._task.result()

    def expired(self) -> bool:
        if not self._task.done():
            return False
        if self._task.cancelled() or self._task.result() is None:
            return _monotonic() - self._started >= AVAILABILITY_RETRY_S
        return False


def start_availability_read() -> AvailabilityRead:
    """Start the gate's account half in the background (must be on a running loop)."""
    return AvailabilityRead()


def _monotonic() -> float:
    import time

    return time.monotonic()
