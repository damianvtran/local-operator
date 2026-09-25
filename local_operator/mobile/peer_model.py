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


def already_selected(session: Any, label: str) -> bool:
    """Is ``label`` the SELECTED model, with no provider fallback serving instead?

    The "already on" test, and it must read the selection, never the effective
    route (review round 1, M1). While a fallback is pinned, the effective label
    is the fallback's; comparing against it made a switch ONTO the fallback's
    model a no-op, so the selection stayed on the model being moved away from
    and the session returned to it when the route settled — the motivating
    incident exactly. ``/model`` compares against the selection too, and an
    explicit re-selection withdraws the pin (``Session.set_model``), so a pinned
    session always goes through the switch.
    """
    selected = str(getattr(session, "model_label", "") or "")
    return selected == label and getattr(session, "active_fallback", None) is None


def selected_label(session: Any) -> str:
    """``provider/model`` the session has SELECTED (the "was" in a receipt)."""
    return str(getattr(session, "model_label", "") or "")


def provider_call_in_flight(session: Any) -> bool:
    """Is the running turn waiting on a PROVIDER call (not a tool) right now?

    Decides the busy receipt's wording only (UX round 1, U7). A turn's live
    context ends in unanswered tool calls for the whole of a tool batch
    (``protocol.unanswered_tail_call_ids`` is that one rule), so an open batch
    means the turn is waiting on a tool or on an approval, and no call is in
    flight. Any read failure answers True, the historical wording: this only
    chooses between two true-enough sentences.
    """
    messages = getattr(getattr(session, "_context", None), "messages", None)
    if not messages:
        return True
    try:
        from local_operator.session.protocol import unanswered_tail_call_ids

        return not unanswered_tail_call_ids(list(messages))
    except Exception:  # noqa: BLE001 — wording only; never fail a switch over it
        return True


def refusal_detail(reason: str, current: str) -> str:
    """``refused: <reason>; still on <current>`` — the error frame's message."""
    return f"refused: {reason.rstrip('.')}; still on {current}"


# THE RESULT STRINGS ARE SHORT LINES, OUTCOME FIRST (design round 1, D1/D6).
# The sender's TUI card clips a success body per LINE rather than wrapping it,
# so a one-line receipt lost its busy semantics and its subagent count at every
# width (the busy form measured 298 cells). Each line below stays well under the
# ~94-cell budget an expanded card has at 100 columns once the models are
# long, and the FIRST word differs per outcome — `switched`, `already on`,
# `pending` — so the collapsed row and a skimming reader tell them apart.


def already_on_detail(label: str) -> str:
    return f"already on {label}\nnothing changed"


def accepted_detail(label: str) -> str:
    """A TUI local-setup provider activates asynchronously (a capacity probe)."""
    return (
        f"pending: switch to {label} accepted\n"
        "it applies once that session's local capacity check finishes"
    )


def switched_detail(
    old: str, new: str, *, busy: bool, running_subagents: int, calling: bool = True
) -> str:
    """The success receipt, per design §2, as outcome-first short lines.

    ``busy`` names the ``/model`` semantics the switch actually has: it lands at
    the next PROVIDER CALL. ``calling`` says whether a provider call is what the
    target is in the middle of: when it is parked on a tool (or on an approval)
    no call is in flight, so the line speaks of the current STEP instead, which
    is true in both states (UX round 1, U7).
    """
    lines = [f"switched to {new} (was {old})"]
    if busy:
        in_flight = "the call in flight" if calling else "the current step"
        # "the old/new model", not the ids again: the first line names both,
        # and two long ids here pushed this line past a card's width (D1).
        lines.append(
            f"mid-turn: {in_flight} finishes on the old model; later calls use the new one"
        )
    else:
        lines.append("its next turn runs on it")
    if running_subagents > 0:
        if running_subagents == 1:
            kept = "1 running subagent keeps its model"
        else:
            kept = f"{running_subagents} running subagents keep their model"
        lines.append(f"{kept}; new and resumed ones use the new one")
    return "\n".join(lines)


def partial_switch_detail(old: str, in_force: str, error: BaseException) -> str:
    """The apply raised, but the read-back shows the switch took (review N1).

    ``Session.set_model`` assigns the spec before its journal writes and its
    stream notify, so an exception from a later step can leave the new model
    in force. Reporting that as a refusal would tell the sender nothing
    changed; reporting it as a clean switch would hide the fault.
    """
    return (
        f"switched to {in_force} (was {old})\n"
        f"with an error after the switch: {type(error).__name__}: {error}"
    )


def audit_body(old: str, new: str, sender: dict[str, Any] | None = None) -> str:
    """The body of the record-only peer card written on the target.

    NEW MODEL FIRST (design round 1, D3; UX U3): on resume this card is the only
    trace of the switch — the live notice is a harness row replay skips — and
    its collapsed row is clipped from the right, so what the session is on NOW
    has to survive a narrow terminal. The sender is named last because the card
    header already names it.
    """
    who = sender_label(sender or {})
    tail = f" — switched by {who}" if who else ""
    return f"{AUDIT_PREFIX} now on {new} (was {old}){tail}"


def pending_audit_body(old: str, new: str, sender: dict[str, Any] | None = None) -> str:
    """The card for an ACCEPTED switch whose outcome is still being decided.

    A TUI local-setup provider activates after a capacity probe, so at the
    moment the card is written the switch may still fail. "requested" says
    exactly that; the switch notice that follows (or a refusal notice) says
    how it ended.
    """
    who = sender_label(sender or {})
    tail = f" — requested by {who}" if who else ""
    return f"{AUDIT_PREFIX} switch to {new} requested (on {old} until it applies){tail}"


def sender_label(sender: dict[str, Any]) -> str:
    """A readable name for whoever asked: the conversation, else ``pid N``."""
    name = str(sender.get("conversation_name") or "").strip()
    if name:
        return name
    pid = sender.get("pid")
    return f"pid {pid}" if isinstance(pid, int) and not isinstance(pid, bool) else ""
