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

from local_operator.aida import naming, state

logger = logging.getLogger(__name__)

#: Stable id for the greeting row, so a double-call race lands as a conflict
#: (treated as already-armed) rather than as two greetings.
GREETING_WAKE_ID = "aida-greeting"

#: How rarely she may nudge about setting up an integration (``aida.onboarding.
#: nudge_days``). The registry row's help text already points here; the engine
#: reads it through :func:`nudge_offer`.
DEFAULT_NUDGE_DAYS = 14

#: The permission clause the engine appends to the cadence message when the
#: nudge window is open (R25). The OFFER is the ledger event
#: (:func:`nudge_offer` stamps it), and her instructions admit a nudge only
#: when this sentence is present — so the bound is engine-enforced rather than
#: honour-system, and the ledger keeps its single writer.
NUDGE_CLAUSE = (
    "Integration nudge window: open. If ONE missing integration (an MCP server "
    "such as Google Workspace, Linear or Slack, or a similar set-up step) would be "
    "concretely useful to something in flight, you may suggest that single "
    "integration in this check-in and offer to set it up. Never more than one, and "
    "if nothing in flight needs one, do not bring integrations up."
)


def greeting_message(config_dir: Path | str) -> str:
    """The greeting wake's self-prompt, addressed by the CONFIGURED name.

    A function, not a constant: the greeting is a first-run turn that can fire
    long after the operator renamed her (``aida.name``), and the instruction
    the user watches her follow must say the name she is actually called —
    "Introduce yourself as Aida" after a rename to "Sovereign" is exactly the
    stale reference this feature removes. Short: the persona, the recording
    rules and the integration help live in the packaged seed
    (``agent_seeds/aida.md``), which is loaded as her instructions; this
    message only says what THIS turn is for — introduce, describe, ask the
    details, and point at the next steps.
    """
    return (
        "First-run greeting. The operator has just finished setting up Local Operator and "
        "this is your first conversation with them. Introduce yourself as "
        f"{naming.display_name(config_dir)}, their chief of staff in Local Operator: you "
        "take their requests and route them to the right team, specialist agent or new "
        "parallel session, keep an eye on everything in flight, and help them get set up. "
        "Ask the few details about them that would make you useful — their name, how they "
        "would like to be addressed, what they work on, and an email if they want it on "
        "file — and mention that you can help connect their tools (Google Workspace, "
        "Linear, Slack and other integrations) whenever they are ready. When they answer, "
        "record what they agree to keep, following your instructions for the recording. "
        "Keep it warm and brief — a short message, not a manual."
    )


def greeted_at(config_dir: Path | str) -> int | None:
    """When the greeting was armed and not cancelled undelivered, else ``None``.

    ``None`` means owed: never armed, refused (no provider / paused / owner),
    or armed and then dropped by a pause before it could fire — the clear in
    :func:`clear_greeted`. The distinction matters because "owed" is what
    re-arms it (``proactive.resume``, and the engine's reconcile for a live
    owner), so a stamp that survived a cancelled row would be a greeting lost
    for good with the ledger claiming it was delivered (review round 1, m1).
    """
    data = state.read_json(state.onboarding_path(config_dir), what="onboarding")
    if data is None:
        return None
    stamp = data.get("greeted_at")
    return int(stamp) if isinstance(stamp, int) else None


def mark_greeted(config_dir: Path | str, now_ms: int) -> None:
    """Stamp the greeting as armed-and-owed (the ledger the receipts read).

    Public because the engine's own live-owner ensure arms the same row from
    :func:`proactive.reconcile` and owes the ledger the same fact: without the
    stamp that path would find no row on a later reconcile and arm a SECOND
    greeting (review round 1, m1). The clear that undoes it when a row is
    cancelled undelivered is :func:`clear_greeted`.
    """
    data = state.read_json(state.onboarding_path(config_dir), what="onboarding") or {}
    data["greeted_at"] = now_ms
    state.write_json(state.onboarding_path(config_dir), data)


def clear_greeted(config_dir: Path | str) -> None:
    """Un-stamp the greeting: it was armed and will never be delivered.

    Called by the two places that can drop the ``aida-greeting`` row before it
    fires — ``proactive.pause`` for the external cancel, and the engine's
    paused branch for a live owner's own reconcile — so the next resume arms
    it again instead of the ledger reporting a greeting the user never saw.
    Best-effort like its sibling: an unreadable file is left as the reader's
    ``None`` already treats it.
    """
    path = state.onboarding_path(config_dir)
    data = state.read_json(path, what="onboarding")
    if data is None or "greeted_at" not in data:
        return
    data["greeted_at"] = None
    try:
        state.write_json(path, data)
    except OSError:
        logger.warning("aida: could not clear greeted_at", exc_info=True)


def _sessions_root(config_dir: Path | str) -> Path:
    return Path(config_dir) / "sessions"


def _counts_as_operator_conversation(entry: Path, root: Path) -> bool:
    """Whether a user-classified session directory is a conversation HAD.

    Existence alone cannot answer it: the TUI boot materialises a session
    directory for every normal launch — the viewer's own id, holding only
    ``.execution-lease`` + ``.session.pid``, no transcript — so on a fresh
    install's FIRST boot the scan saw "a user session" and the predicate read
    false, and a provider-present first contact never greeted (UX review
    round 2, U4; the setup-exit path was unaffected because a setup-state boot
    creates no conversation directory at all).

    The discriminator is the engagement signal the runtime already answers
    this question with (``session_has_durable_history``: a real ``Message``
    row, never a custom one — wake prompts and quiet-dial peer notes persist
    without a turn), so "never engaged" reads as "no conversation had".
    Deliberately fail-closed around it, in R22's direction ("existing users
    are never re-routed"): a transcript with ANY bytes the reader will not
    call history — torn rows, a custom-only journal, a file it cannot open —
    counts as the operator's. Only the two shapes that positively say nothing
    was ever said here are discounted: no transcript at all, and an empty one.
    """
    from local_operator.session.runtime.engagement import (
        TRANSCRIPT_FILENAME,
        session_has_durable_history,
    )

    if session_has_durable_history(entry.name, root=root):
        return True
    try:
        return (entry / TRANSCRIPT_FILENAME).stat().st_size > 0
    except FileNotFoundError:
        return False
    except OSError:
        return True


def other_user_sessions(config_dir: Path | str) -> list[str] | None:
    """Ids of the operator's conversations besides hers, or ``None`` when unknown.

    The scan behind the first-run predicate (design §2.9): "no user sessions
    exist" means no ``is_user_session`` directory under ``sessions/`` other
    than the one ``aida/state.json`` names. Non-user origins (subagent runs,
    agent-shell sessions) deliberately do not count — they are machines'
    conversations, and an install whose store holds only those is still a
    fresh one as far as a human's first-run experience goes.

    Existence alone is not a conversation either: a directory counts only
    once it shows engagement (:func:`_counts_as_operator_conversation`) — the
    boot materialises one for every launch, and counting that made a fresh
    install look used the moment it started once (UX review round 2, U4).

    FAIL-CLOSED, in both ways a read can fail: an unreadable store answers
    ``None`` and an entry that cannot be classified counts as the operator's.
    The cost of the conservative direction is a first-run experience shown a
    moment later or not at all; the cost of the other is re-routing somebody
    with conversations (R22's "existing users are never re-routed"), which is
    the one outcome this predicate exists to prevent.
    """
    from local_operator.resume import is_user_session  # lazy: resume is heavy

    root = Path(config_dir)
    hers = state.session_id_of(root)
    try:
        entries = sorted(_sessions_root(root).iterdir())
    except FileNotFoundError:
        return []
    except OSError:
        logger.warning("aida: could not list the session store", exc_info=True)
        return None
    found: list[str] = []
    for entry in entries:
        if not entry.is_dir() or (hers and entry.name == hers):
            continue
        try:
            is_users = is_user_session(entry)
        except Exception:  # noqa: BLE001 — unclassifiable counts as the user's
            logger.warning("aida: could not classify session %s", entry.name, exc_info=True)
            is_users = True
        if is_users and _counts_as_operator_conversation(entry, root):
            found.append(entry.name)
    return found


def fresh_install(config_dir: Path | str) -> bool:
    """The first-run predicate: no human conversations ever, provider resolves.

    Both halves are load-bearing and both come from the design (§2.9): the
    sessions half keeps existing installs from being re-routed, and the
    provider half keeps the greeting from arming into a turn that cannot run
    (the same predicate :func:`greet` refuses on). Never raises: an unreadable
    store or config answers False, because "not fresh" is the direction that
    cannot re-route somebody.
    """
    try:
        others = other_user_sessions(config_dir)
    except Exception:  # noqa: BLE001 — a predicate, never a boot dependency
        logger.warning("aida: could not scan for user sessions", exc_info=True)
        return False
    if others is None or others:
        return False
    return provider_configured(config_dir)


def first_run_pending(config_dir: Path | str) -> bool:
    """Whether the first-run experience is still OWED to this install.

    ``fresh_install`` and the greeting ledger together, which is the gate the
    desktop contract states (design §4: "greeted=false and no user
    conversations other than hers"): once ``greeted_at`` is stamped, the
    routing does not fire again even if the store is wiped later — the ledger
    is what makes "once per config root" true (design §7 item 4's proposal).
    """
    try:
        return fresh_install(config_dir) and greeted_at(config_dir) is None
    except Exception:  # noqa: BLE001 — same posture as :func:`fresh_install`
        logger.warning("aida: first-run predicate failed", exc_info=True)
        return False


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
                {"message": greeting_message(root), "in": "1s"},
                wake_id=GREETING_WAKE_ID,
                now_ms=now,
            )
        except WakeWriteError as exc:
            # 409 = the row already exists (a racing second greet): the
            # greeting IS armed, which is the fact this call exists to
            # establish. 503 = a live owner; leave it unstamped so the caller
            # can retry after the owner reconciles.
            if exc.status == 409:
                mark_greeted(root, now)
                return "greeted"
            if exc.status == 503:
                return "owner"
            logger.warning("aida: greeting arm refused: %s", exc)
            return "failed"
        mark_greeted(root, now)
        return "greeted"
    except Exception:  # noqa: BLE001 — a greeting must never fail its caller
        logger.warning("aida: greet failed", exc_info=True)
        return "failed"


# --------------------------------------------------------------------------- #
# The integration-nudge ledger (R25) — the engine's half, read open/closed
# --------------------------------------------------------------------------- #


def _nudge_days(config_dir: Path | str) -> int:
    """``aida.onboarding.nudge_days``, best-effort, floored at one day.

    Read through the settings facade (the registry row's own reader), so the
    CLI, ``/settings`` and this bound cannot disagree about the value; an
    unreadable config answers the default rather than costing the bound, and
    a value below the registry's minimum answers it too (a hand-edited 0 must
    not turn the nudge into every-day spam).
    """
    try:
        from local_operator import settings_io
        from local_operator.config import ConfigManager

        setting = settings_io.BY_KEY.get("aida.onboarding.nudge_days")
        if setting is None:  # pragma: no cover - registry guarantee
            return DEFAULT_NUDGE_DAYS
        days = int(settings_io.read_setting(ConfigManager(config_dir=Path(config_dir)), setting))
    except Exception:  # noqa: BLE001 — the default IS the bound of last resort
        logger.debug("aida: could not read nudge_days", exc_info=True)
        return DEFAULT_NUDGE_DAYS
    return days if days >= 1 else DEFAULT_NUDGE_DAYS


def nudge_offer(config_dir: Path | str, *, now_ms: int | None = None) -> str | None:
    """The nudge clause for a cadence row being ARMED now, or ``None`` if not due.

    One function, two effects, and they are deliberately inseparable: when the
    window is open it stamps the ledger (``nudge_offered_at``,
    ``nudge_offers``) and returns :data:`NUDGE_CLAUSE` for the row's message;
    when it is closed — or the stamp cannot be recorded — it returns ``None``.
    That ordering is what makes the bound real rather than advisory: the
    permission and its ledger move together, and a clause can never exist
    without the stamp that spends the window.

    WHO SENDS THE NUDGE. The offer rides the CADENCE ROW (its message is
    persisted in the transcript), so the moment of truth is the fire, not this
    call — and the engine, which cannot read her prose, records the offer
    rather than an actual suggestion. An offer that never turns into a nudge
    is therefore spent conservatively (the next window opens in
    ``nudge_days``); the alternative — an honour-system ledger she writes —
    loses a racing read-modify-write of ``onboarding.json`` against the
    greeting's own writers, and R25's "bounded" would stop being enforceable.

    The read-modify-write runs under the aida lock so two processes arming the
    cadence at once cannot both spend one window. Never raises: a contended
    lock answers ``None`` (no offer without a stamp — the safe direction).
    """
    root = Path(config_dir)
    now = int(time.time() * 1000) if now_ms is None else int(now_ms)
    window_ms = _nudge_days(root) * 86_400_000
    try:
        with state.locked(root):
            path = state.onboarding_path(root)
            data = state.read_json(path, what="onboarding") or {}
            last = data.get("nudge_offered_at")
            last_ms: int | None = None
            if isinstance(last, (int, float)) and not isinstance(last, bool):
                last_ms = int(last)
            if last_ms is not None and now - last_ms < window_ms:
                return None
            data["nudge_offered_at"] = now
            data["nudge_offers"] = int(data.get("nudge_offers") or 0) + 1
            state.write_json(path, data)
    except Exception:  # noqa: BLE001 — a lock refusal must not cost the cadence
        logger.warning("aida: could not record the nudge window", exc_info=True)
        return None
    return NUDGE_CLAUSE
