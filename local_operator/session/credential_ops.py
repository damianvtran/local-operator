"""The one ``/credential`` verb table, shared by every session shape.

Storing a credential is a SESSION capability, not a viewer one: the store
exists so the environment of the ``bash`` commands the agent runs can carry
the secret, and those commands run wherever the session's turn loop runs. A
session that runs its tools in this process therefore executes the verbs
against its own store (``Session.credential_op``); a session that is a window
onto a runtime routes them there
(``RemoteSession.credential_op`` → the runtime's
``OwnedSessionHandle.credential_op`` → this same table).

Before this module existed, the table lived only on the runtime's handle, and
the capability was declared on the viewer protocol alone — so the codebase
asserted "storing a credential is a viewer capability", the in-process session
never implemented it, and the TUI's submit seam (which cannot know which shape
it holds) degraded on the only branch the shipping product can reach. Keeping
ONE table is what prevents that drift from coming back: the verbs cannot
diverge between the two shapes if there is only one copy.

The value crosses into this function and never leaves it: it is written to the
store, never logged, never journalled, never echoed in the answer. Only the
key name and the outcome cross back.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

#: The journal half of the table's contract: ``(key, action=..., replaced=...)``.
#: Named so the parameter reads as behaviour rather than as a bare callable.
CredentialJournal = Callable[..., None]


async def run_credential_verb(
    store: Any,
    journal: CredentialJournal | None,
    action: str,
    key: str,
    value: str,
) -> dict[str, Any]:
    """Run one ``/credential`` verb against ``store``; announce through ``journal``.

    ``store`` is a :class:`~local_operator.variables.VariableStore`; ``journal``
    is the session's credential-change announcer (a callable with the
    ``journal_credential_change`` shape) or ``None`` to skip announcing — the
    announcement is best-effort by design, so a missing or raising journal must
    never fail a store that already succeeded.

    Returns plain data rather than a receipt object because the callers need
    the FACTS (did it replace, what was removed) to build their own notices,
    and because the store verb's receipt has to name the key that was actually
    normalized and stored, not the one that was typed.
    """
    if store is None or not hasattr(store, "store_credential"):
        return {"ok": False, "reason": "unavailable"}
    if action == "list":
        return {
            "ok": True,
            "credentials": [
                {"key": item.key, "source": item.source} for item in store.list_credentials()
            ],
        }
    if action == "names":
        return {"ok": True, "names": list(store.credential_names())}
    if action == "forget":
        removed = bool(store.forget_credential(key))
        if removed:
            _announce(journal, key, action="forgot")
        return {"ok": True, "removed": removed, "key": key}
    if action == "forget-all":
        # Names BEFORE the clear: the store is empty afterwards, so reading
        # them after would announce nothing to the model.
        names = list(store.credential_names())
        count = int(store.clear_credentials())
        for name in names:
            _announce(journal, name, action="forgot")
        return {"ok": True, "count": count, "names": names}
    if action == "persist":
        # The promotion route: both stores that matter are the session's own —
        # the in-memory store holding the value and the config dir holding the
        # encrypted long-term one. Only a name and an outcome sentence cross
        # back; never the value.
        from local_operator.secrets.promote import promote_session_credential_guarded

        outcome = promote_session_credential_guarded(store, key)
        return {"ok": True, "promoted": outcome.ok, "key": key, "message": outcome.message}
    if action == "store":
        result = store.store_credential(key, value, "command")
        credential = getattr(result, "credential", None)
        if not getattr(result, "ok", False) or credential is None:
            return {"ok": False, "reason": getattr(result, "reason", "") or "empty-value"}
        replaced = bool(getattr(result, "replaced", False))
        # The announcement is what makes the key findable on LATER turns, so it
        # belongs on the side that owns the context, immediately after the
        # write it reports — never before it, and never on a refusal.
        _announce(journal, credential.key, replaced=replaced)
        return {"ok": True, "key": credential.key, "replaced": replaced}
    return {"ok": False, "reason": "unknown-action"}


def _announce(
    journal: CredentialJournal | None, key: str, *, action: str = "stored", replaced: bool = False
) -> None:
    """Best-effort announcement; a failed one must not fail the store."""
    if journal is None:
        return
    try:
        journal(key, action=action, replaced=replaced)
    except Exception:  # noqa: BLE001 — the credential is already stored
        logger.warning("could not announce credential change", exc_info=True)
