"""Promote a session credential into the long-term store (design §5.4).

The session path is unchanged: `/credential` and `ask secret=true` still store
into `VariableStore` memory, still inject into bash, still cannot be read back.
What is added is a ROUTE — the operator (or the agent, per §5.2) can decide
that a secret handed over for this session should outlive it.

**Why this is its own module and not a method on either store.** The two stores
are deliberately separate types with separate lifetimes (`variables.py` holds
process memory; `secrets/store.py` holds encrypted disk). Putting the promotion
on `VariableStore` would give the session store an import of the crypto stack
that every session pays for and only this route uses; putting it on
`SecretStore` would give the disk store knowledge of session objects it has no
other reason to know. A function that takes both is the seam with the smallest
footprint.

**It reads the session value and writes it; it does not expose it.** The value
crosses this function in memory and is never returned, logged, or put in a
result — the callers get a name and an outcome. That is what lets the TUI and
the `ask` path call it on a value the MODEL must never see.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PromotionResult:
    """Outcome of one promotion. Carries a NAME and a reason, never a value."""

    ok: bool
    name: str
    #: Operator/model-facing sentence. Safe to display: no secret bytes.
    message: str
    #: True when the secret was already in the long-term store under this name
    #: and was left untouched. Not a failure, and reported distinctly so the
    #: caller does not tell the operator it stored something it did not.
    already_present: bool = False


def promote_session_credential(
    store: Any,
    key: str,
    *,
    description: str = "",
    base: Path | None = None,
) -> PromotionResult:
    """Copy session credential ``key`` into the encrypted long-term store.

    ``store`` is the session :class:`~local_operator.variables.VariableStore`,
    consumed through ``credential_env()`` because that is the only accessor
    that yields values — deliberately, so a promotion is visibly a read of the
    injectable set rather than a new private door into it.

    The session copy is KEPT. Promotion is additive: the credential stays
    injected into this session's bash children (which is what the agent is
    already using) and additionally survives the session. Removing it would
    break every command in flight that expects the env var.

    ``base`` is for tests only; production always resolves the real config dir.
    """
    from local_operator.secrets.access import open_store, session_id
    from local_operator.secrets.errors import SecretExists, SecretStoreError

    reader = getattr(store, "credential_env", None)
    if not callable(reader):
        return PromotionResult(False, key, "This session cannot hold credentials.")
    try:
        values = reader()
    except Exception:
        logger.warning("could not read the session credential store", exc_info=True)
        return PromotionResult(False, key, "This session's credential store could not be read.")
    # `store` is deliberately typed `Any` (a duck-typed VariableStore, which is
    # also how `builtin.py` consumes it), so the shape of what came back is
    # checked rather than assumed: a third-party store returning something
    # else must not raise out of a promotion.
    if not isinstance(values, dict):
        return PromotionResult(False, key, "This session's credential store could not be read.")
    value = values.get(key)
    if not value:
        # NAMES THE GESTURE THAT STILL WORKS. This used to advise
        # `/credential {key}`, which the typed capture retired: the space after
        # the token opens a masked span, so typing a key name mints it as a short
        # secret instead of reaching the `<KEY>` prompt (QA round 1, Q1). Advice
        # an operator cannot follow is worse than none — they would have typed it
        # and watched their key name become a credential. The inline gesture
        # generates the name, so the way to persist is to hand the secret over
        # and then persist the name the chip reports.
        return PromotionResult(
            False,
            key,
            f"No session credential named {key}. Hand one over with /credential "
            "followed by a space and the secret, then --persist the name its chip reports.",
        )
    try:
        target = open_store(base, create=True)
        target.set(
            key,
            value.encode("utf-8"),
            description=description.strip(),
            session_id=session_id(),
        )
    except SecretExists:
        # NOT an overwrite. `set` refuses an existing name on purpose (the
        # previous value is retained nowhere), and silently calling `update`
        # here would turn "promote this" into "replace whatever is already
        # under that name" — destroying a long-term secret because a session
        # happened to reuse its name. The operator is told and chooses.
        return PromotionResult(
            False,
            key,
            f"{key} already exists in the long-term store; it was NOT replaced. "
            # R8: this message rides a model-visible channel (the ask-persist
            # path reports it to the model), so it must not coach an
            # irreversible CLI delete the model could attempt. "Ask the
            # operator" is the safe wording — `lop secret rm` prompts without
            # `--yes` and cannot be done silently, so nothing here is a thing
            # the model can act on.
            f"Ask the operator to remove or rename it, or promote it under another name.",
            already_present=True,
        )
    except SecretStoreError as exc:
        return PromotionResult(False, key, str(exc))
    return PromotionResult(
        True,
        key,
        f"Promoted {key} to the encrypted long-term store; it now survives this session. "
        f"Reachable as $(lop secret get {key}).",
    )


def promote_session_credential_guarded(
    store: Any,
    key: str,
    *,
    description: str = "",
    base: Path | None = None,
) -> PromotionResult:
    """``promote_session_credential`` that never raises (R4).

    The ``/credential --persist`` owner path, the viewer route and the ask
    path all promote the same session credential, but only the ask path used
    to absorb a non-``SecretStoreError`` from the store stack (an unwrapped
    ``sqlite3.OperationalError``, an ``OSError`` on the key file). The same
    promotion must not have two robustness levels, so the guard lives here and
    every caller takes it. The session copy is already in session memory by
    the time this runs, so a failure degrades to "session only" rather than
    losing the secret — and says so, because the operator's (or model's) plan
    depends on which it was. A traceback carries names only, never the value.
    """
    try:
        return promote_session_credential(store, key, description=description, base=base)
    except Exception:
        logger.warning("could not promote %s to the long-term store", key, exc_info=True)
        return PromotionResult(
            False,
            key,
            "Kept for this session only — saving it to the long-term store failed.",
        )
