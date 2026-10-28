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
#: The shipped default. The "later flip" (§9) is this one value, here and in
#: ``config.DEFAULT_CONFIG``; explicit settings are untouched by it.
DEFAULT_MODE = "off"

Source = Literal[
    "gate",
    "signed_out",
    "no_credit",
    "session",
    "agent",
    "parent",
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
    5. Default: ``auto`` ⇒ on, ``opt_in`` ⇒ off.
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
    return SaverDecision(mode == "auto", "default")


async def read_availability() -> Availability:
    """Signed in, and a positive balance from a FRESH owner-pinned billing read.

    Reuses the tunnel plane's owner-pinned client and its ``_positive_balance``
    predicate (missing, boolean or non-finite amounts are unavailable). Any
    failure — no login, several logins and none selected, a network error, a
    non-200 — answers "not available"; it never raises.
    """
    import httpx

    from local_operator.tunnels.api import RadientTunnels, credential_id
    from local_operator.tunnels.cli import _positive_balance

    try:
        selected = credential_id(None)
    except Exception:  # noqa: BLE001 — no single Radient login: signed out
        return Availability(signed_in=False, positive_balance=False)
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            quote = await RadientTunnels(selected, client).request("GET", "/billing")
    except Exception:  # noqa: BLE001 — unknown counts as no (S-31)
        return Availability(signed_in=True, positive_balance=False)
    return Availability(
        signed_in=True,
        positive_balance=isinstance(quote, dict) and _positive_balance(quote),
    )
