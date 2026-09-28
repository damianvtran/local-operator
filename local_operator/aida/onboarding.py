"""Aida's first-run greeting — the onboarding baseline slice A ships.

The frozen desktop contract (design §4) has a ``greet`` op that must deliver
ONE onboarding turn and be idempotent. Slice A ships the mechanism; slice B
(the onboarding PR) owns the fresh-install predicate, the TUI/UI routing, the
profile-capture guidance and the integration-nudge ledger that extends
``onboarding.json``.

WHY THE GREETING RIDES THE WAKE PATH. There is no turn-injection API on the
messages route (it admits user turns only), and adding one for a single
greeting would be a new write surface every session would then be equal to. A
one-shot wake due now is the existing "start a turn in this conversation"
primitive: it writes the greeting as a user-attributed message and runs her
turn, and it works with no live runtime at all (the supervisor starts her).
The operator accepted the trade (design §7 item 1): the transcript shows the
greeting as a wake line until a later polish suppresses the artifact.

IDEMPOTENCE. ``onboarding.json``'s ``greeted_at`` is the authority, so a
cleared localStorage or a second renderer cannot re-fire the greeting; it is
stamped only AFTER the arm lands, so a refusal (no provider, paused, a
contended lock) leaves the greeting still owed and the next call retries.

NO-PROVIDER REFUSAL. ``greet`` refuses when no provider is configured — the
frozen contract's edge case — and deliberately does not stamp, so the greeting
fires after the user completes setup. The predicate is the boot path's own
(``session_factory``'s hosting resolution), asked in a try/except so this
module stays import-light and unknown failure degrades to "not yet" rather
than burning the one greeting into a turn that cannot run.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from local_operator.aida import state

logger = logging.getLogger(__name__)

#: Stable id for the greeting row, so a double-call race lands as a conflict
#: (treated as already-armed) rather than as two greetings.
GREETING_WAKE_ID = "aida-greeting"

#: How rarely she may nudge about setting up an integration (``aida.onboarding.
#: nudge_days``); consumed by slice B, pinned here so the registry row and its
#: consumer cannot drift.
DEFAULT_NUDGE_DAYS = 14

#: The greeting wake's self-prompt. Short: the persona and the detail live in
#: the packaged seed (``agent_seeds/aida.md``), which is loaded as her
#: instructions; this message only says what THIS turn is for.
GREETING_MESSAGE = (
    "First-run greeting. The operator has just finished setting up Local Operator and "
    "this is your first conversation with them. Introduce yourself as Aida, their chief "
    "of staff, describe briefly what you can do for them, and ask the few details about "
    "them that would help you be useful (their name, what they do, and how they would "
    "like to be addressed). Keep it warm and brief — a short message, not a manual."
)


def greeted_at(config_dir: Path | str) -> int | None:
    """When the greeting was delivered, or ``None`` while it is still owed."""
    data = state.read_json(state.onboarding_path(config_dir), what="onboarding")
    if data is None:
        return None
    stamp = data.get("greeted_at")
    return int(stamp) if isinstance(stamp, int) else None


def _mark_greeted(config_dir: Path | str, now_ms: int) -> None:
    data = state.read_json(state.onboarding_path(config_dir), what="onboarding") or {}
    data["greeted_at"] = now_ms
    state.write_json(state.onboarding_path(config_dir), data)


def provider_configured(config_dir: Path | str) -> bool:
    """Whether a provider/model pair is resolvable — the boot path's predicate.

    Deliberately the SAME resolution ``create_session`` performs
    (``resolve_hosting_model_with_source`` with no agent and no explicit
    selection), because "the greeting can run" must mean exactly "a session
    could boot here". Import is lazy: ``session_factory`` is heavy and this
    module is imported by boot paths. Unknown failures answer False: the
    refusal is retryable (nothing is stamped), so the safe direction of an
    unreadable config is "greet later", not "greet into a turn that cannot run".
    """
    import argparse

    try:
        from local_operator.config import ConfigManager
        from local_operator.session_factory import resolve_hosting_model_with_source

        args = argparse.Namespace(hosting=None, model=None, agent_name=None, agent_id=None)
        resolve_hosting_model_with_source(None, args, ConfigManager(config_dir=Path(config_dir)))
        return True
    except Exception:  # noqa: BLE001 — unconfigured or unreadable => not yet
        return False


async def greet(config_dir: Path | str, session_id: str, *, now_ms: int | None = None) -> str:
    """Arm the one-time onboarding greeting. Returns one word for the receipt.

    ``"greeted"`` (armed now), ``"already"`` (stamped earlier / row survived),
    ``"disabled"``, ``"paused"``, ``"no-provider"``, ``"owner"`` (a live
    runtime holds the session; its reconcile will pick the row up — the arm
    refusal is retried by the caller if it matters) or ``"failed"``.
    """
    from local_operator.aida import proactive

    root = Path(config_dir)
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    try:
        if state.env_disabled():
            return "disabled"
        pol = proactive.policy(root)
        if not pol.enabled:
            return "disabled"
        if pol.paused:
            return "paused"
        if greeted_at(root) is not None:
            return "already"
        if not provider_configured(root):
            return "no-provider"

        from local_operator.wakes.arm import WakeWriteError, arm_wake

        try:
            await arm_wake(
                root,
                session_id,
                {"message": GREETING_MESSAGE, "in": "1s"},
                wake_id=GREETING_WAKE_ID,
                now_ms=now,
            )
        except WakeWriteError as exc:
            # 409 = the row already exists (a racing second greet): the
            # greeting IS armed, which is the fact this call exists to
            # establish. 503 = a live owner; leave it unstamped so the caller
            # can retry after the owner reconciles.
            if exc.status == 409:
                _mark_greeted(root, now)
                return "greeted"
            if exc.status == 503:
                return "owner"
            logger.warning("aida: greeting arm refused: %s", exc)
            return "failed"
        _mark_greeted(root, now)
        return "greeted"
    except Exception:  # noqa: BLE001 — a greeting must never fail its caller
        logger.warning("aida: greet failed", exc_info=True)
        return "failed"
