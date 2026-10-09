"""Aida's desktop surface: her state, and the five verbs that move it.

The frozen contract (design §4), which both repos were written against:

    GET  /v1/desktop/aida
        200 { enabled, session_id, paused, greeted, name,
              greeting: {state, surface, requested_at, armed_at, delivered_at},
              first_run_pending, operator: {name, email, source} | null }
    POST /v1/desktop/aida   body {"op": open|pause|resume|greet|status}
        200 { session_id, paused, greeted, greeting_state }
        409 when disabled (``aida_disabled``)

``name`` is her configured display name (``aida.name``, default "Aida"): the
renderer labels her row with it, and it is read live on every GET, so a rename
landed anywhere (``/aida rename``, renaming her conversation, a /settings edit)
changes what every renderer shows with no restart. ADDITIVE to the frozen
shape — a client written before it ignores the field and defaults to "Aida"
(the UI's ``aida.data?.name ?? "Aida"``), so the freeze holds.

``greeting``/``first_run_pending``/``operator`` (first-run onboarding, Lane B)
and ``greeting_state`` on the op shape are additive the same way. ``greeted``
now means DELIVERED (it used to flip at arm time); ``greet`` is the ATTENDED
request — it is the only way the desktop can start the greeting, and a headless
runtime never can (``aida/onboarding.py``'s state machine).

``GET`` never creates: ``session_id`` is ``null`` until something ensures her
(``open``/``greet``, the TUI's ``/aida``, either boot hook). ``open`` and
``greet`` ensure the session; ``greet`` is idempotent through the
``onboarding.json`` ledger; ``pause``/``resume`` flip ``aida.cadence.paused``
and the supervisor's hold marker.

WHY EVERY OP IS AVAILABLE WITH NO LIVE RUNTIME. Nothing here writes session or
wake files directly: each op delegates to ``local_operator.aida``, whose writers
already own the one-writer invariants (``ensure_session`` for her session;
``proactive`` for the cadence, its holds and the escalation budget;
``onboarding.greet`` for the greeting). With no runtime open, those writers use
``wakes/arm.py`` — the documented external arm path, transcript first, install
hook last — and with a runtime open they refuse the file write and the live
session reconciles through the config watcher instead. A route that wrote files
itself would be the second writer this subsystem keeps removing.

WHY 409 RATHER THAN A SILENT 200 FOR DISABLED. ``aida.enabled = false`` (or
``LOCAL_OPERATOR_NO_AIDA``) is a supported steady state, and the UI reads
``GET`` first — a disabled install answers ``enabled: false`` and the renderer
hides the row. The 409 is for the OTHER ordering: a client that cached a
previous ``true``, or that raced the switch, must be told its op did not run
rather than receive a session id that will never exist. The code is part of the
contract because a renderer can only tell "disabled" from "your request was
malformed" by the code, not the sentence.

WHY A HELD STORE LOCK IS A 200 RECEIPT, NOT A 500. ``pause``/``resume`` take the
aida store lock as their first act (``aida.state.locked``), so a peer that holds
it for the whole wait REFUSES the op with ``WakeLockBusy`` — the lock module's
own documented answer for that case, retryable, and normal on a seam with several
attended writers. The refusal used to escape this module as a 500, which told the
operator to read the logs for a miss that fixes itself on the next tick; it now
answers the receipt the TUI's ``/aida`` handler renders for the identical refusal,
and the two surfaces are pinned to that ONE sentence by
``tests/unit/server/test_desktop_aida.py`` (QA-O1). A genuine failure from the op
still surfaces as it did: this converts the lock refusal only, never a defect.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Aida"], dependencies=[Depends(require_desktop)])

#: The five verbs the frozen contract names. A ``Literal`` rather than a free
#: string so an unknown op is a 422 from FastAPI's own validation (the shape
#: this file never has to invent sentences for) and the generated schema
#: publishes the vocabulary to the renderer.
AidaOpName = Literal["open", "pause", "resume", "greet", "status"]


class AidaOp(BaseModel):
    """The POST body. ``extra="forbid"`` so a misspelled field is a 422 rather
    than a silently ignored no-op — the same rule the wake bodies use."""

    model_config = {"extra": "forbid"}

    op: AidaOpName


class AidaState(BaseModel):
    """GET's answer: the full state, including the enable bit the discovery
    loop reads to decide whether to render anything at all."""

    enabled: bool
    session_id: str | None = None
    paused: bool
    greeted: bool
    #: Her configured display name (``aida.name``; see the module docstring).
    #: A plain string with no null state: the config layer's reader always
    #: answers SOMETHING (the default stands in for unset/invalid), so a
    #: renderer never needs a fallback branch — the field is exactly what the
    #: surfaces should say.
    name: str
    #: The greeting ledger (``onboarding.greeting_record``): ``state`` is one of
    #: owed / requested / armed / delivered / skipped, plus the surface that
    #: requested it and the three stamps. ADDITIVE, like ``name``: ``greeted``
    #: stays and now means "delivered". The renderer needs the distinction —
    #: requested/armed means "open her conversation, she is about to speak",
    #: which a boolean cannot say.
    greeting: dict[str, Any] = {}
    #: Whether this install is still owed the first-run experience (no human
    #: conversations besides hers, a provider resolves, greeting not settled).
    #: The desktop's onboarding ``finish()`` reads it to decide between
    #: ``greet`` + her conversation and a plain ``/chat``.
    first_run_pending: bool = False
    #: Who the operator is, when the install knows it WITHOUT asking: the
    #: Radient sign-in's ``id_token`` claims (``{"name", "email", "source":
    #: "radient"}``), else ``null`` — in which case she asks for it herself.
    operator: dict[str, str] | None = None
    #: The greeting ledger's STATE WORD, mirrored onto the READ (``owed`` /
    #: ``requested`` / ``armed`` / ``delivered`` / ``skipped``). ADDITIVE. The
    #: read already carries the whole ``greeting`` record, but a renderer that
    #: has to promise "she will say hello first" on the step BEFORE the press
    #: needs the same word the POST answers with, without walking a nested
    #: object to find it — and ``greeted`` alone cannot say it: a pending
    #: greeting and one that will never come are both ``false`` there.
    greeting_state: str = "owed"
    #: The live-owner flag (see ``AidaOpState.held``), present on the READ so the
    #: two shapes carry one field set and a client binding them shares a
    #: decoder. A GET performs no operation, so a live owner is simply acting on
    #: the next POST — false here means "no operation is being carried by
    #: someone else right now", which is exactly the fact it reports.
    held: bool = False


class AidaOpState(BaseModel):
    """Every successful POST's answer — WITHOUT ``enabled``.

    The freeze (design §4) spells the two shapes separately and the difference
    is load-bearing: a POST is only reachable when the feature is enabled (a
    disabled backend answers 409 ``aida_disabled``), so an ``enabled`` field on
    this answer can never be ``false`` and only invites a client to branch on a
    constant. GET keeps it, because GET is the call a client makes BEFORE it
    knows whether the feature is there at all.
    """

    session_id: str | None = None
    paused: bool
    greeted: bool
    #: The greeting ledger's state word (see ``AidaState.greeting``). ADDITIVE
    #: to the frozen op shape: a ``greet`` answers ``requested``/``armed`` when
    #: she is about to speak, ``delivered``/``skipped`` when she never will
    #: again — what the desktop needs to decide whether to navigate to her.
    greeting_state: str = "owed"
    #: Whether a LIVE session on this machine owns her rows, so this call\'s
    #: effect is carried out by that owner rather than here: the greeting will
    #: arrive in the other window, the pause lands on its next tick, the resume
    #: arms there. ADDITIVE, and deliberately one field across all three ops —
    #: a client that had to read this out of the message prose could only
    #: branch on our wording. ``held`` is a fact about WHO acts next, not an
    #: error: every answer carrying it is a 200 and the operation is in effect.
    held: bool = False


def _config_dir(request: Request):
    return request.app.state.config_manager.config_dir


def _refuse_disabled(op: str, name: str) -> HTTPException:
    return HTTPException(
        409,
        {
            "code": "aida_disabled",
            "message": (
                f"{name} is disabled on this backend, so {op!r} did nothing. "
                "Enable her with aida.enabled (or unset LOCAL_OPERATOR_NO_AIDA) "
                "and try again."
            ),
        },
    )


def _state(root: Any) -> AidaState:
    from local_operator.aida import enabled, naming, onboarding, proactive
    from local_operator.aida import state as aida_state

    policy = proactive.policy(root)
    return AidaState(
        enabled=enabled(root),
        session_id=aida_state.session_id_of(root),
        paused=policy.paused,
        greeted=onboarding.greeted_at(root) is not None,
        # Read live (never raises; the default stands in for unset/invalid),
        # so a rename needs no invalidation step on this side.
        name=naming.display_name(root),
        greeting=onboarding.greeting_record(root),
        first_run_pending=onboarding.first_run_pending(root),
        operator=_operator(root),
        # The read carries the same word the POST answers with: the desktop's
        # step 3 promises "she will say hello first" BEFORE the press, and it
        # reads this document to decide.
        greeting_state=onboarding.greeting_state(root),
    )


def _operator(root: Any) -> dict[str, str] | None:
    from local_operator.aida import onboarding

    identity = onboarding.radient_identity(root)
    if not identity:
        return None
    return {
        "name": identity.get("name", ""),
        "email": identity.get("email", ""),
        "source": "radient",
    }


def _reply(state: AidaState, message: str) -> CRUDResponse[AidaState]:
    return CRUDResponse(status=200, message=message, result=state)


def _op_reply(state: AidaState, message: str, *, held: bool = False) -> CRUDResponse[AidaOpState]:
    """A POST's answer, in the frozen op shape (no ``enabled``; see `AidaOpState`)."""
    return CRUDResponse(
        status=200,
        message=message,
        result=AidaOpState(
            session_id=state.session_id,
            paused=state.paused,
            greeted=state.greeted,
            greeting_state=str(state.greeting.get("state") or "owed"),
            held=held,
        ),
    )


def _refuse_lock(root: Any, verb: str, name: str, exc: BaseException) -> CRUDResponse[AidaOpState]:
    """A store-lock refusal as a receipt — the SAME sentence the TUI answers with.

    The op did not run: a peer held the aida store lock for the whole wait, so
    ``locked`` raised before anything was unpaused or cancelled. That refusal is
    retryable and normal on this seam, which is why it answers 200 with the
    sentence the TUI's ``/aida`` handler renders for the identical refusal
    (``tui/app.py::_aida_control``) — and why the sentence is built here in the
    same shape rather than re-worded: the two surfaces are pinned to each other
    by ``tests/unit/server/test_desktop_aida.py``, so one cannot drift into a
    second account of the refusal without a red cell.

    NOT the ``busy`` word's receipt below: that word is ``resume``'s arm attempt
    AFTER the unpause landed (round 3d, N2), while this one is the op never
    running at all — so the ``result`` here is the state read NOW, still
    ``paused`` after a refused resume, and ``held`` stays false because no live
    owner is carrying anything out.
    """
    from local_operator.aida import state as aida_state

    aida_state.note_lock_refusal(f"the {verb} command", exc)
    return _op_reply(_state(root), f"could not {verb} {name}: {exc}")


@router.get("/v1/desktop/aida", response_model=CRUDResponse[AidaState])
async def get_aida(request: Request) -> CRUDResponse[AidaState]:
    """Her state. Never creates: a caller that wants her to exist POSTs ``open``."""
    from local_operator import aida

    root = _config_dir(request)
    if not aida.enabled(root):
        # NOT a 409 here: the GET is how a renderer DISCOVERS the switch, so it
        # must answer. \u00a73.4 of the design: new UI x disabled backend renders
        # the row hidden and answers a typed /aida with a receipt.
        policy_state = _state(root)
        return _reply(policy_state, f"{policy_state.name} is disabled on this backend.")
    state = _state(root)
    return _reply(state, f"{state.name} state retrieved.")


@router.post("/v1/desktop/aida", response_model=CRUDResponse[AidaOpState])
async def post_aida(body: AidaOp, request: Request) -> CRUDResponse[AidaOpState]:
    """Run one op. Refuses with 409 ``aida_disabled`` on a disabled install."""
    from local_operator import aida
    from local_operator.aida import naming, onboarding, proactive
    from local_operator.aida import state as aida_state
    from local_operator.wakes.lock import WakeLockBusy, WakeLockUnavailable

    root = _config_dir(request)
    name = naming.display_name(root)
    if not aida.enabled(root):
        raise _refuse_disabled(body.op, name)

    if body.op == "status":
        return _op_reply(_state(root), f"{name} state retrieved.")

    if body.op in ("open", "greet"):
        # BOTH ENSURE, per the contract: greet is a first-run conversation, and
        # a client that greets without opening first must not need a second
        # call. Disabled was refused above; a bootstrap failure answers 500 so
        # the client can retry rather than silently getting a session id that
        # does not exist.
        session_id = await aida.ensure_session(root)
        if session_id is None:
            raise HTTPException(
                500,
                {
                    "code": "aida_bootstrap_failed",
                    "message": (
                        f"{name}'s session could not be created; try again (see the logs)."
                    ),
                },
            )
        if body.op == "greet":
            # ``surface=desktop``: the route is called by the window the person
            # is looking at, which is what makes it one of the two attended
            # entries into the greeting ledger (audit A1).
            outcome = await onboarding.greet(root, session_id, surface=onboarding.SURFACE_DESKTOP)
            state = _state(root)
            if outcome == "no-provider":
                # THE ONE REFUSAL THE CONTRACT NAMES: nothing is stamped, so
                # the greeting fires after setup completes.
                raise HTTPException(
                    409,
                    {
                        "code": "aida_no_provider",
                        "message": (
                            f"No provider is configured yet, so {name} has nothing to "
                            "greet you with. Connect a provider, then try again."
                        ),
                    },
                )
            if outcome == "paused":
                return _op_reply(
                    state,
                    f"{name} is paused, so the greeting is being held; /aida resume "
                    "delivers it.",
                )
            if outcome == "owner":
                return _op_reply(
                    state,
                    f"{name} is open in another window; it will say hello there.",
                    held=True,
                )
            if outcome == "failed":
                raise HTTPException(
                    500,
                    {
                        "code": "aida_greet_failed",
                        "message": "The greeting could not be scheduled; try again (see the logs).",
                    },
                )
            if outcome == "already":
                # ``already`` covers delivered and a racing second arm; the
                # ledger says which, so the sentence matches the fact.
                if onboarding.greeting_state(root) == onboarding.GREETING_DELIVERED:
                    return _op_reply(state, f"{name} has already introduced herself.")
                return _op_reply(state, f"{name} is already on her way to introducing herself.")
            if outcome == "skipped":
                # R-5: NOT the same fact as ``already``. This install already has
                # conversations, so she will never introduce herself here —
                # correct behaviour, and saying "she already has" would be false.
                return _op_reply(
                    state,
                    f"{name} does not introduce herself on an install that already "
                    "has conversations.",
                )
            return _op_reply(state, f"{name} is introducing herself in her conversation.")
        return _op_reply(_state(root), f"{name}'s conversation is ready.")

    session_id = aida_state.session_id_of(root) or ""
    if body.op == "pause":
        try:
            outcome = await proactive.pause(root, session_id)
        except (WakeLockBusy, WakeLockUnavailable) as exc:
            # A PEER HELD THE STORE LOCK FOR THE WHOLE WAIT, so nothing was
            # paused: the refusal is the receipt (see `_refuse_lock`).
            return _refuse_lock(root, body.op, name, exc)
        state = _state(root)
        message = f"{name} is paused; she will not check in proactively."
        if outcome.owner_blocked:
            message += " Her open session applies the hold within a moment."
        return _op_reply(state, message, held=bool(outcome.owner_blocked))

    # resume
    try:
        arm = await proactive.resume(root, session_id)
    except (WakeLockBusy, WakeLockUnavailable) as exc:
        # ``resume`` takes the store lock before it unpauses anything, so a
        # refusal here means the op did not run — NOT the ``busy`` word below,
        # which describes a resume whose unpause landed (see `_refuse_lock`).
        return _refuse_lock(root, body.op, name, exc)
    state = _state(root)
    if arm == "owner":
        # Correct and expected, not a failure: a live session owns its rows and
        # arms the next occurrence on its own watcher tick.
        return _op_reply(
            state,
            f"{name} is active again; her open session will arm the next check-in.",
            held=True,
        )
    if arm == "no-session":
        return _op_reply(state, f"{name} is active again; her next conversation arms the check-in.")
    if arm == "busy":
        # The store lock was held for the whole wait, so NOTHING was armed on this
        # call (round 3d, N2): the cadence word is new, and the generic sentence
        # below would tell the operator a check-in exists when it does not. The
        # sentence stays calm because the refusal is retryable — the next boot or
        # tick arms it — which is what the TUI's fallback copy says too.
        return _op_reply(
            state,
            f"{name} is active again, but her files were busy just now; "
            "the next check-in arms on her next boot.",
        )
    return _op_reply(state, f"{name} is active again; the next check-in is armed.")
