"""Receive-side core for a peer switching THIS session's model (``peer_set_model``).

Both owner hosts answer the op — the TUI (``mobile/tui_handle.py``) and the
serving/exec runtime (``session/runtime/serving.py``) — and they must agree on
what counts as a servable pair and on every sentence a sender reads back. Those
sentences are a user-visible contract (design D2), so they are composed once,
here, and each host only supplies its own way of APPLYING the switch: the TUI
runs its own ``/model`` on the Textual thread, the runtime calls its
``set_model_effort``. Neither host mutates ``Session`` from the control
server's loop.

Import-light on purpose, like ``peer_send``: the validator and the credential
store are imported inside the functions that need them, so a handle module can
import this without dragging the model graph into its own import time.
"""

from __future__ import annotations

from contextlib import closing
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from local_operator.harness.types import ModelSpec

#: The prefix of the audit card the TARGET's transcript records. A record-only
#: ``peer_message`` card rather than a new card type (design D6): it is the one
#: card every front end already renders with the sender's name, pid and model,
#: so the target's human and model both see who switched them, from what, to
#: what — and it survives resume.
AUDIT_PREFIX = "[remote model switch]"


def provider_usable_here(provider: str) -> bool:
    """Whether THIS process's config has a credential ``provider`` can run on.

    ``ProviderController.is_usable`` is the predicate (store rows, an env key,
    or a provider that needs none); a bare ``has_any_credential`` would refuse a
    working env-key setup. Neither owner host holds a controller of its own, so
    the store is opened, read and CLOSED inside this one call — the same
    short-lived read ``mobile/daemon.py``'s model sheet already makes — rather
    than reaching through the session's private stream wiring for the store it
    runs on. Runs on a worker thread (``asyncio.to_thread`` at both call sites):
    it is a SQLite read.

    An unreadable store raises: the caller cannot tell "no credential" from "the
    store is locked", and a switch refused as unconfirmed is retryable where a
    switch onto a provider with no credential dies on the next turn.
    """
    from local_operator.paths import config_dir
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.controller import ProviderController

    with closing(AuthStore()) as store:
        return ProviderController(store, config_dir()).is_usable(provider)


def validate_peer_selection(provider: str, model_id: str) -> "ModelSpec":
    """The spec a peer's pick resolves to HERE, or ``ModelSelectionRefused``.

    Target-side only (design D3): this session's config dir, credential store
    and catalogue cache are the authoritative ones, and a sender running under a
    different ``LOCAL_OPERATOR_CONFIG_DIR`` would approve a pair this session
    cannot run. Runs before anything is mutated, so a refusal cannot
    half-switch.
    """
    from local_operator.model.configure import (
        ModelSelectionRefused,
        validate_model_selection,
    )

    def usable(name: str) -> bool:
        try:
            return provider_usable_here(name)
        except Exception as error:  # noqa: BLE001 — any store fault is "cannot confirm"
            raise ModelSelectionRefused(
                "credentials_unreadable",
                f"could not read that session's credential store ({type(error).__name__})",
            ) from error

    return validate_model_selection(provider, model_id, usable=usable)


def normalise_pair(provider: str, model_id: str) -> tuple[str, str]:
    """Trim both halves and lower-case the provider, exactly as ``/model`` does."""
    return provider.strip().lower(), model_id.strip()


def running_subagent_count(session: Any) -> int:
    """How many subagents keep their current model through this switch (D9b).

    ``Session.running_subagents`` is the one predicate the stop ladder and the
    reaper already use. A switch is never propagated to running children, so
    the sender is told how many there are instead of being left to assume they
    moved too. Zero on any failure: the count only qualifies a receipt.
    """
    probe = getattr(session, "running_subagents", None)
    if not callable(probe):
        return 0
    try:
        count = probe()
        return max(0, int(count)) if isinstance(count, int) else 0
    except Exception:  # noqa: BLE001 — a qualifier must never fail a switch
        return 0


def refusal_detail(reason: str, current: str) -> str:
    """``refused: <reason>; still on <current>`` — the error frame's message."""
    return f"refused: {reason.rstrip('.')}; still on {current}"


def already_on_detail(label: str) -> str:
    return f"already on {label}; nothing changed"


def accepted_detail(label: str) -> str:
    """A TUI local-setup provider activates asynchronously (a capacity probe)."""
    return f"accepted: checking local capacity for {label}; the switch applies when that finishes"


def switched_detail(old: str, new: str, *, busy: bool, running_subagents: int) -> str:
    """The success receipt, per design §2.

    ``busy`` names the ``/model`` semantics the switch actually has: it lands at
    the next PROVIDER CALL, so a call already in flight finishes on the old
    model — the one moment "starting when" is a live question.
    """
    if busy:
        text = (
            f"switched to {new} (was {old}) mid-turn; the call in flight finishes on "
            f"{old}, every later call uses {new}"
        )
    else:
        text = f"switched to {new} (was {old}); its next turn runs on it"
    if running_subagents > 0:
        if running_subagents == 1:
            kept = "1 running subagent keeps its current model"
        else:
            kept = f"{running_subagents} running subagents keep their current model"
        text += f"; {kept} — new and resumed ones use {new}"
    return text


def audit_body(old: str, new: str) -> str:
    """The body of the record-only peer card written on the target."""
    return f"{AUDIT_PREFIX} switched this session from {old} to {new}"
