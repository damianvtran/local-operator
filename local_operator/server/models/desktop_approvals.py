"""Wire shapes for the desktop's approval surfaces (``features.approvals``).

The frozen shapes are the design note's §3.5 / the desktop contract's: the list
answer is ``{approvals: [{approval_id, state, what, requested_by, withdrawn_by,
expires_at, device}]}``, and a decision answers ``{approval_id, state,
signature:{key_id}}``.
The dict-typed halves are deliberate rather than lazy: ``what`` and the
where-block are the record's OWN vocabulary (the design grows scopes there
without a wire change), and pinning them field-by-field here would make the
backend the second place a new scope must be declared.

REQUEST MODELS FORBID EXTRA KEYS on this plane (``routes/desktop_sessions.Input``);
the two decision routes take NO body at all — approving is the gesture, and a
body would only invite a caller to think it could carry one.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ApprovalRow(BaseModel):
    """One record in the badge/list shape (frozen keys, §3.5)."""

    approval_id: str
    state: str
    what: dict[str, Any]
    requested_by: dict[str, Any]
    #: The requester's own settle, once one happened (design review round 1, D2
    #: — decision: fix the shared row, so a panel can name the withdrawer).
    withdrawn_by: dict[str, Any] | None = None
    expires_at: float | None = None
    #: The where-block, under the key its KIND spells: ``device`` for
    #: ``device_onboard``, ``machine`` for ``local_authority``. Exactly one is
    #: present; the record's own kind is what tells a reader which.
    device: dict[str, Any] | None = None
    machine: dict[str, Any] | None = None
    #: OPTIONAL extras, present ONLY on a credential-permission row (never stored: derived
    #: from the signed record by the backend at read time). An older UI ignores them; an
    #: older backend omits them. ``can_approve_here`` is False until the app can state what
    #: is signed, and ``approve_remedy`` names the terminal command instead. ``exclude_if``
    #: (the ``desktop_sessions`` policy for additive keys) drops them from every other row,
    #: so the frozen §3.5 key set is byte-identical for onboarding and local-authority rows.
    summary: str | None = Field(default=None, exclude_if=lambda value: value is None)
    details: list[str] | None = Field(default=None, exclude_if=lambda value: value is None)
    can_approve_here: bool | None = Field(default=None, exclude_if=lambda value: value is None)
    approve_remedy: str | None = Field(default=None, exclude_if=lambda value: value is None)


class ApprovalList(BaseModel):
    approvals: list[ApprovalRow]


class ApprovalDecision(BaseModel):
    """The answer to approve/deny: the frozen decision shape."""

    approval_id: str
    state: str
    #: ``{"key_id": ...}``; ``""`` when the decision carries no signature (a deny).
    signature: dict[str, Any]
