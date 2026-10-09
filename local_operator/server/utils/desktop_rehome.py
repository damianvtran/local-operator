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
    rehome_notice,
)

logger = logging.getLogger(__name__)

#: The shape ``defaults_applied`` gains, additively: how many open sessions the
#: sign-in moved. The renderer may ignore it; a receipt composed from it must
#: name both providers (see :func:`with_rehome_count`).
REHOMED_KEY = "rehomed_sessions"


async def rehome_after_login(app: Any) -> list[tuple[str, str]]:
    """Move the sessions this sign-in stranded, or explain nothing and move none.

    Takes the STARLETTE APP rather than the request: the hook it is installed as
    (``DesktopAuth.rehome``) outlives any single request, and reading ``app.state``
    keeps the per-request object out of the closure.

    Three reads, all after the credential write so the new provider is visible:
    this app's config manager (the default the user just ended up with), the
    credential store (which providers are actually reachable), and the session
    pool (which conversations this process is bound to). Any of them missing means
    this host cannot do the job — an embedded app with no manager, a test client
    with no pool — and the answer is "nothing moved", never an error: the sign-in
    itself has already succeeded and must not be reported as failed.

    The store read runs OFF the loop (SQLite, and the pool's own reads treat a
    controller as thread-affine), and the config values ride along so the local
    providers' ``base_url`` check reads the file once.
    """
    state = getattr(app, "state", None)
    manager = getattr(state, "config_manager", None)
    pool = getattr(state, "desktop_sessions", None)
    if manager is None or pool is None:
        return []
    values = manager.get_config().values
    provider = str(values.get("hosting", "") or "").strip().lower()
    model_id = str(values.get("model_name", "") or "").strip()
    if not provider or not model_id:
        # Nothing to move them TO. The planner leaves this state alone as well
        # (its case 1), so it is not a failure: a default with no model is its
        # own setup state, and re-homing onto a half-configured pair would trade
        # one stranded session for another.
        return []
    accessible = await asyncio.to_thread(
        credentialed_chat_providers_here,
        config_dir=Path(pool.root),
        config_values=values,
    )
    return await pool.rehome_stranded_sessions(accessible, provider, model_id)


def with_rehome_count(
    applied: dict[str, Any] | None, moved: list[tuple[str, str]]
) -> dict[str, Any] | None:
    """Fold ``moved`` into the ``defaults_applied`` receipt the routes return.

    ``None`` from ``apply_desktop_login_defaults`` means "the config default was
    left alone and there is nothing to say" — but a login that moved live sessions
    DID do something the user must be told about, so in that case a receipt is
    composed here (the planner has no sentence for it; it never touched a session).

    When the config default WAS replaced, the planner's receipt stands untouched
    — it already names both providers, and the design's frozen sentences are
    asserted verbatim in tests — and the count joins it as the additive field.
    """
    if not moved:
        return applied
    count = len(moved)
    if applied is None:
        old, new = moved[0]
        if count == 1:
            receipt = rehome_notice(old, new)
        else:
            old_provider = old.partition("/")[0] or old
            receipt = f"Moved {count} open sessions to {new} — not signed in to {old_provider}."
        return {
            "hosting": None,
            "model": None,
            "model_name": None,
            "receipt": receipt,
            REHOMED_KEY: count,
        }
    return {**applied, REHOMED_KEY: count}
