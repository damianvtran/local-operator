"""Post-sign-in re-home: repair live sessions stranded on an unreachable provider.

Why this module exists. ``plan_login_defaults`` fixes the CONFIG default, but a
conversation's ``selected_model`` outranks config (the journalled selection
``session_factory``/``cold_model`` honour), and its owner is the only writer that
may move it. So a user who signs in to a working provider can still have every
open session pinned to a provider whose credential is gone — the reported bug:
"the default and open sessions stay on a stale radient/auto". This module is the
desktop's half of the repair: it decides WHOM to ask (the pool's bound, idle,
stranded sessions), and the owner applies a compare-and-set switch through
``rehome_if_current`` beside ``set_model_effort``.

It is deliberately separate from ``desktop_auth.apply_desktop_login_defaults``:
that function is synchronous, shared with the CLI/TUI policy and unit-tested as a
pure write, while this one is async (it awaits every owner) and needs the app's
session pool — a request-shaped concern, not a config-policy one. Both are called
from the same two places (a completed sign-in and a saved API key), because both
paths store a credential and both must repair what that credential strands.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from local_operator.providers.model_access import (
    credentialed_chat_providers_here,
    is_first_provider_login,
    rehome_deferred_notice,
    rehome_notice,
)

logger = logging.getLogger(__name__)

#: The shape ``defaults_applied`` gains, additively: how many open sessions the
#: sign-in moved. The renderer may ignore it; a receipt composed from it must
#: name both providers (see :func:`with_rehome_count`).
REHOMED_KEY = "rehomed_sessions"

#: Additive sibling of ``REHOMED_KEY``: how many conversations were BUSY at the
#: sign-in and therefore did not move. They are not silent failures — each was
#: told by its own owner — but a sign-in whose every session was busy must say so
#: in its receipt too, or the login reads as "fixed" over a conversation that
#: still cannot run a turn (UX review U1).
DEFERRED_KEY = "deferred_sessions"


async def rehome_after_login(
    app: Any, login_provider: str
) -> tuple[list[tuple[str, str]], list[str]]:
    """Move the sessions this sign-in stranded, or explain nothing and move none.

    Takes the STARLETTE APP rather than the request: the hook it is installed as
    (``DesktopAuth.rehome``) outlives any single request, and reading ``app.state``
    keeps the per-request object out of the closure.

    THE FIRST-LOGIN RULE (operator refinement, round 1) is the second thing this
    checks, after the target: sessions move ONLY on the user's first provider
    login — that is, when the credentialed chat providers other than
    ``login_provider`` are empty (``is_first_provider_login``; Radient counts, so
    a prior Radient/web sign-in makes later logins non-first). The rule exists
    for the state a session can be in BEFORE any provider login: started by some
    earlier build, pinned to ``radient/auto``, nothing behind it — the reported
    bug. That first OpenAI or Anthropic login should move it, and NO later
    provider login should ever re-home a conversation: adding a second provider
    to switch models with must not silently re-point open chats. The config half
    (``plan_login_defaults``) is not scoped by this rule — it repairs an
    unrunnable default on any login.

    Three reads, all after the credential write so the new provider is visible:
    this app's config manager (the default the user just ended up with), the
    credential store (which providers are actually reachable), and the session
    pool (which conversations this process is bound to). Any of them missing means
    this host cannot do the job — an embedded app with no manager, a test client
    with no pool — and the answer is "nothing moved", never an error: the sign-in
    itself has already succeeded and must not be reported as failed.

    Returns ``(moved, deferred)``: the sessions that switched, and the old labels
    of the busy ones the owner refused — the deferrals the receipt must count so
    a sign-in where every conversation was busy is not silent (UX review U1).
    Each deferred conversation is told by the owner itself, when it refuses.

    The store read runs OFF the loop (SQLite, and the pool's own reads treat a
    controller as thread-affine), and the config values ride along so the local
    providers' ``base_url`` check reads the file once.
    """
    state = getattr(app, "state", None)
    manager = getattr(state, "config_manager", None)
    pool = getattr(state, "desktop_sessions", None)
    if manager is None or pool is None:
        return [], []
    values = manager.get_config().values
    provider = str(values.get("hosting", "") or "").strip().lower()
    model_id = str(values.get("model_name", "") or "").strip()
    if not provider or not model_id:
        # Nothing to move them TO. The planner leaves this state alone as well
        # (its case 1), so it is not a failure: a default with no model is its
        # own setup state, and re-homing onto a half-configured pair would trade
        # one stranded session for another.
        return [], []
    accessible = await asyncio.to_thread(
        credentialed_chat_providers_here,
        config_dir=Path(pool.root),
        config_values=values,
    )
    if not is_first_provider_login(accessible, login_provider):
        return [], []
    return await pool.rehome_stranded_sessions(accessible, provider, model_id)


def with_rehome_count(
    applied: dict[str, Any] | None,
    moved: list[tuple[str, str]],
    deferred: list[str],
) -> dict[str, Any] | None:
    """Fold ``moved`` and ``deferred`` into the ``defaults_applied`` receipt.

    ``None`` from ``apply_desktop_login_defaults`` means "the config default was
    left alone and there is nothing to say" — but a login that moved live sessions
    (or left busy ones where they were) DID do something the user must be told
    about, so in that case a receipt is composed here: the planner has no
    sentence for it, because it never touched a session.

    When the config default WAS replaced, the planner's receipt stands untouched
    — it already names both providers, and the design's frozen sentences are
    asserted verbatim in tests — and the counts join it as additive fields. The
    mixed case (some moved, some busy) says the move in the receipt and carries
    both counts; each busy conversation has already been told by its own owner.
    """
    if not moved and not deferred:
        return applied
    if applied is None:
        if moved:
            count = len(moved)
            if count == 1:
                receipt = rehome_notice(moved[0][0], moved[0][1])
            else:
                # The old provider is NOT named: the moved sessions can come off
                # two different stranded providers, and quoting the first one's
                # let a two-provider move read as if one provider had been
                # involved (review NIT-1).
                receipt = f"Moved {count} open sessions to {moved[0][1]}."
        else:
            receipt = rehome_deferred_notice(deferred[0], count=len(deferred))
        applied = {
            "hosting": None,
            "model": None,
            "model_name": None,
            "receipt": receipt,
        }
    result = dict(applied)
    if moved:
        result[REHOMED_KEY] = len(moved)
    if deferred:
        result[DEFERRED_KEY] = len(deferred)
    return result
