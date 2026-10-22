"""H5 host seam: attach the bridge runtime to the account-bound designated session.

WHY THIS MODULE EXISTS. The bridge tool must exist in exactly ONE session per
machine — the persisted, account-bound CoS session the wake path already names
(the native enrollment record, ``store.designated_session_for``) — and nowhere
else. That is a HOST decision, taken where every writer session is built
(``session_factory._prepare``): a session is handed the runtime at
construction or never, and the tool's ``createIf`` reports absence for every
other session without any per-surface choice.

CONSTRUCTION IS NETWORK-FREE AND GATED ON PERSISTED FACTS ONLY (any miss =>
``None`` => no tool):
- ``team_bridge.enabled`` is ON (the shipped default is OFF during the
  consent hold; an explicit ``true`` is this machine's opt-in);
- a committed owner exists (``store.active_owner``: zero OR ambiguous rows
  read ``None`` — never a guess);
- the owner's enrollment generation is still live (``current_generation_is_live``);
- the NATIVE designation equals THIS session id (``designated_session_for``);
- a consent receipt exists (``owner_consent_receipt``).
The availability/UI record is deliberately NOT a gate: it is per-process UI
state written only when a UI boundary read ran here, so gating on it made a
cold host (or a read-only record) lose the tool even though every persisted
native fact held — and it let a cached UI belief decide a model-tool
admission. Eligibility is ADMISSION's business, below, and a cached record is
never authority for it either.

ADMISSION IS A BOUNDARY READ, NEVER A POLL. ``admit_bridge_runtime`` is
single-flight per runtime (one status fetch in flight, ever) and settles
``pending``/``admitted``/``paused``/``disabled``/``signin`` from ONE
authenticated ``GET /messaging/status`` on the runtime's own pinned client
(the exact credential row). Its ONLY callers are boundaries: the host boot
(drain/reconcile site) and an op-edge that landed while not admitted. There is
deliberately NO timer, NO backoff loop, NO process-life retry and NO passive
status/inbox poll — a transient failure stays visibly ``pending`` until the
next boundary or an explicit trusted human action, and nothing is ever
promoted from a cache. Zero traffic exists while the feature is off or native
authority is missing (no runtime is built then).

EVERY OP, STILL FRESH. Admission proves WHO could act; each operation still
re-checks the live authority: a cheap LOCAL fence (``owner_fence_refusal``)
refuses ``owner_changed`` when the committed owner/generation/designation
changed under a running session (no silent rebind — re-admission requires a
new construction), and the client runs the fresh server auth/kill verdict on
every call. A server ``bridge_disabled``/kill observed at a STATUS boundary
settles ``runtime.admission`` to ``disabled`` — that field is written only
there (see the boundary logic below); an OP-TIME kill is enforced on the call
itself and does not rewrite the cached admission. Per §36 the state is not
process-permanent: a later trusted boundary may re-admit once the local gates
hold.

TYPED MUTATING REFUSALS. ``admission_refusal`` maps the unsettled states to
closed codes: ``bridge_not_ready`` (pending), ``bridge_signin_required``,
``bridge_paused``, ``bridge_disabled``, and ``bridge_read_only`` for the F
posture (retained history: the flow cannot resume — a typed refusal, never a
claim that the server is unreachable; server read history stays meaningful for
the read surfaces).

ERROR POLICY. Store damage/ambiguity reads as "no authority" (``StoreRefused``
/ ``sqlite3.Error`` => ``None`` / typed refusal). Everything else propagates:
a blanket ``except`` would swallow programmer errors and mint a quieter
failure mode than the crash it hides.

SCOPE. Subagent children never receive the runtime: the child-building code
paths do not pass ``bridge_runtime`` (see the parameter's comment on
``Session``), and this helper is called only for the writer sessions the
factory builds.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import Mapping
from typing import Any

from .tool import BridgeRuntime

logger = logging.getLogger(__name__)

__all__ = [
    "admission_refusal",
    "admit_bridge_runtime",
    "admit_soon",
    "bridge_runtime_for_session",
    "owner_fence_refusal",
]


def bridge_runtime_for_session(*, config_manager: Any, session_id: str) -> BridgeRuntime | None:
    """Build the session's bridge runtime, or ``None`` — a gate miss is not an error.

    No network, no fetch, no spawn: the client is cheap until its first
    request (its own contract), and every input here is local state the
    factory already has in hand. The constructed state is ALWAYS ``pending``
    — the persisted gates prove *who*, never *current eligibility*.
    """

    from .client import TeamBridgeClient
    from .settings import TeamBridgeSettings
    from .store import Store, StoreRefused

    try:
        settings = TeamBridgeSettings.from_config(config_manager)
        if not settings.enabled:
            return None
        config_dir = config_manager.config_dir
        store = Store(config_dir)
        owner = store.active_owner()
        if owner is None:
            return None
        if not store.current_generation_is_live(owner):
            return None
        if store.designated_session_for(owner) != session_id:
            return None
        if store.owner_consent_receipt(owner) is None:
            return None
        client = TeamBridgeClient(
            config_dir=config_dir,
            # The owner pin: this client half is bound to the EXACT credential
            # row the enrollment committed — a same-account re-login is a
            # different row and needs explicit rebinding (store docstring).
            credential_id=owner.credential_id,
        )
        return BridgeRuntime(
            client=client,
            config_dir=config_dir,
            tenant_id=owner.tenant_id,
            owner=owner,
            session_id=session_id,
        )
    except (StoreRefused, sqlite3.Error):
        return None


def _admission_from_payload(payload: Any, owner: Any) -> tuple[str, str]:
    """The admission reading of one status payload — ENROLLED ROW ONLY (S-4).

    The payload is scanned for the COMMITTED tenant id directly and ONLY that
    row's eligibility/endpoint/state are read. The default selection
    (``client.select_tenant``: first eligible row) is the ONBOARDING choice
    and must never decide an existing binding's admission: with another
    eligible org earlier in the list it misread a valid enrolled row as
    ``owner_changed``. Missing/ambiguous rows are typed refusals — no org is
    ever substituted.
    """

    from .client import _api_supported

    if not isinstance(payload, Mapping) or not _api_supported(payload):
        return "disabled", "update_required"
    wanted = str(getattr(owner, "tenant_id", ""))
    rows = [
        t
        for t in (payload.get("tenants") or [])
        if isinstance(t, Mapping) and str(t.get("tenant_id") or "") == wanted
    ]
    if not rows:
        return "disabled", "not_enabled"
    if len(rows) > 1:
        return "disabled", "ambiguous_tenant_row"
    selected = rows[0]
    if not bool(selected.get("eligible")):
        # F: a lapse/kill/scope exclusion, byte-identical by design (S-1).
        # Retained history stays READABLE (fetch ops are listed in tool.py);
        # mutating ops get the typed read-only refusal.
        return "disabled", "read_only"
    endpoint = selected.get("endpoint")
    if not isinstance(endpoint, Mapping):
        return "disabled", "not_enabled"
    if str(endpoint.get("id") or "") != str(getattr(owner, "endpoint_id", "")):
        return "disabled", "owner_changed"
    state = endpoint.get("state")
    if state == "active":
        return "admitted", ""
    if state == "paused":
        return "paused", ""
    return "disabled", "opted_out"


async def admit_bridge_runtime(runtime: BridgeRuntime) -> str:
    """ONE boundary-triggered admission, COALESCED across callers (H5-1/S-2).

    At most one fetch is in flight per runtime — the bound authority epoch:
    a caller with a live task JOINS it (N op edges never become N fetches),
    and the slot clears after settlement so a later genuine boundary may try
    again. Authority is rechecked at entry, immediately before the GET, and
    after the await before the result may settle state: a revoked epoch
    cancels queued work and DISCARDS its result — never a stale re-admit,
    never a silent rebind. No timer, no backoff, no poll.
    """

    from .authority import authority_fence

    if authority_fence(runtime) is not None:
        _cancel_inflight_admission(runtime)  # revoked: invalidate queued work
        return runtime.admission
    task = _ensure_admission_task(runtime)
    if task is None:
        return runtime.admission
    try:
        await asyncio.shield(task)
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 — a settle failure must not crash the boundary
        # _admission_fetch settles Bridge-typed outcomes itself; anything left
        # is a programmer error and is logged, never laundered into "pending".
        logger.warning("bridge admission task failed", exc_info=True)
    return runtime.admission


async def _admission_fetch(runtime: BridgeRuntime) -> None:
    """The ONE in-flight fetch: fence → GET → fence → settle."""

    from .authority import authority_fence
    from .client import BridgeRefusal, BridgeSignInRequired, BridgeUnreachable

    if authority_fence(runtime) is not None:
        return  # revoked between scheduling and the GET: zero traffic
    try:
        payload = await runtime.client.status()
    except BridgeSignInRequired:
        runtime.admission = "signin"
        runtime.admission_note = "signin_required"
        return
    except BridgeUnreachable:
        # Transport only: nothing was answered. Stays VISIBLY pending
        # until the next boundary (or an explicit human action) — no timer.
        runtime.admission = "pending"
        runtime.admission_note = "unreachable"
        return
    except BridgeRefusal as refusal:
        runtime.admission = "disabled"
        runtime.admission_note = str(getattr(refusal, "code", "refused"))
        return
    if authority_fence(runtime) is not None:
        # The answer describes an epoch that is no longer bound: DISCARD it —
        # a stale result must never re-admit or rebind.
        return
    state, note = _admission_from_payload(payload, runtime.owner)
    runtime.admission = state
    runtime.admission_note = note


def _ensure_admission_task(runtime: BridgeRuntime) -> "asyncio.Task[None] | None":
    """The ONE shared task: return a live one, else create it (or nothing)."""

    task = runtime._admission_task
    if task is not None and not task.done():
        return task
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None  # no running loop: nothing to schedule here
    runtime._admission_task = task = loop.create_task(_admission_fetch(runtime))
    task.add_done_callback(_swallow_admission_result)
    return task


def _cancel_inflight_admission(runtime: BridgeRuntime) -> None:
    """Revocation invalidates queued work: cancel the in-flight task, no slot."""

    task = runtime._admission_task
    runtime._admission_task = None
    if task is not None and not task.done():
        task.cancel()


def admit_soon(runtime: BridgeRuntime) -> None:
    """Schedule the single-flight admission from a boundary; never waits.

    The ONLY triggers are boundaries — the boot caller and an op-edge that
    landed while not admitted. Deliberately NO timer, NO backoff loop and NO
    process-life retry (§34): a transient failure remains ``pending`` until
    the next boundary or an explicit trusted human action.
    """

    from .authority import authority_fence

    if authority_fence(runtime) is not None:
        _cancel_inflight_admission(runtime)  # off/drifted: zero traffic
        return
    _ensure_admission_task(runtime)


def _swallow_admission_result(task: "asyncio.Task[Any]") -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.warning("bridge admission attempt failed", exc_info=exc)


def admission_refusal(runtime: BridgeRuntime, *, is_read: bool = False) -> dict[str, str] | None:
    """The typed refusal for an unsettled admission, or ``None`` when admitted.

    ``is_read`` marks FETCH ops (``take_up`` — a body fetch, not a mutation):
    the F posture keeps retained history readable and D stops FLOW, not the
    view (§5.6), so those two states admit reads and refuse every mutation
    with the typed refusal. Disabled/signin/pending refuse both directions.
    """

    state = runtime.admission
    if state == "admitted":
        return None
    if state == "paused":
        if is_read:
            return None
        return {
            "code": "bridge_paused",
            "error": (
                "agent messages are paused for this organization (§5.6 D) — "
                "nothing is sent until it is resumed"
            ),
        }
    if state == "signin":
        return {
            "code": "bridge_signin_required",
            "error": "sign in to Radient to use agent messages",
        }
    if state == "disabled":
        if runtime.admission_note == "read_only":
            if is_read:
                return None
            return {
                "code": "bridge_read_only",
                "error": (
                    "this organization's agent messages are read-only right now "
                    "(§5.6 F) — no mutating operation is accepted"
                ),
            }
        return {
            "code": "bridge_disabled",
            "error": "agent messages are turned off for this endpoint/organization right now",
        }
    return {
        "code": "bridge_not_ready",
        "error": (
            "agent messages eligibility has not been confirmed yet — " "try again in a moment"
        ),
    }


def owner_fence_refusal(runtime: BridgeRuntime) -> dict[str, str] | None:
    """The LOCAL per-op fence — now the FULL shared one (``authority.py``).

    Kept name for existing callers (tool.py): the implementation moved so the
    decide/retry/recovery doors check the SAME facts (enabled, unique owner,
    live non-empty generation, consent, base/account/tenant/endpoint/credential
    row, tenant consistency, designation) as construction and admission.
    """

    from .authority import authority_fence

    return authority_fence(runtime)


def _owner_changed() -> dict[str, str]:
    return {
        "code": "owner_changed",
        "error": (
            "the Radient account binding for this session changed — reopen the "
            "session to bind the current enrollment (no silent rebind)"
        ),
    }
