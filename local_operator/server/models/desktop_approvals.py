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

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


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
    typed_approval: dict[str, Any] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class ApprovalList(BaseModel):
    approvals: list[ApprovalRow]


class ApprovalDecision(BaseModel):
    """The answer to approve/deny: the frozen decision shape."""

    approval_id: str
    state: str
    #: ``{"key_id": ...}``; ``""`` when the decision carries no signature (a deny).
    signature: dict[str, Any]


class OnboardProbe(BaseModel):
    """Public SSH identity observation only; never credential resolution or login."""

    model_config = {"extra": "forbid", "strict": True}
    host: str = Field(min_length=1, max_length=253, pattern=r"^[A-Za-z0-9_.:][A-Za-z0-9_.:-]*$")
    user: str = Field(default="", max_length=64, pattern=r"^(?:[A-Za-z0-9_.][A-Za-z0-9_.-]*)?$")
    port: int | None = Field(default=None, ge=1, le=65535)

    @field_validator("host")
    @classmethod
    def existing_host_parser(cls, value: str) -> str:
        from local_operator.network.enrollment import ssh_target
        from local_operator.network.types import MeshRefusal

        try:
            return ssh_target(value)
        except MeshRefusal:
            raise ValueError("Use an unambiguous SSH host and optional port") from None


class OnboardRequest(BaseModel):
    """An existing signed card, not commands or a caller-authored authority blob.

    The surface chooses explicit scopes but cannot supply requester provenance,
    admin role, root, shell args, private key bytes, or install/build overrides.
    """

    model_config = {"extra": "forbid", "strict": True}
    request_id: str = Field(min_length=5, max_length=128, pattern=r"^req_[A-Za-z0-9_-]+$")
    host: str = Field(min_length=1, max_length=253, pattern=r"^[A-Za-z0-9_.:][A-Za-z0-9_.:-]*$")
    user: str = Field(default="", max_length=64, pattern=r"^(?:[A-Za-z0-9_.][A-Za-z0-9_.-]*)?$")
    name: str = Field(default="", max_length=128)
    network_id: str = Field(min_length=1, max_length=128)
    credential_ref: str = Field(min_length=1, max_length=512, pattern=r"^[^\r\n]+$")
    host_key_fp: str = Field(min_length=1, max_length=128, pattern=r"^SHA256:[A-Za-z0-9+/=]+$")
    unattended: bool = True
    grant: list[str] = Field(default_factory=lambda: ["approve"], max_length=32)

    @field_validator("host")
    @classmethod
    def existing_host_parser(cls, value: str) -> str:
        return OnboardProbe.existing_host_parser(value)

    @field_validator("credential_ref")
    @classmethod
    def reference_not_material(cls, value: str) -> str:
        from local_operator.network.enrollment import credential_reference
        from local_operator.network.types import MeshRefusal

        try:
            return credential_reference(value)
        except MeshRefusal:
            # The app's protected422 handler strips rejected input entirely.
            raise ValueError("Use a credential reference, not private key material") from None

    @field_validator("grant")
    @classmethod
    def existing_grants(cls, value: list[str]) -> list[str]:
        from local_operator.network.types import GRANTABLE_CAPABILITIES

        if any(item != "approve" or item not in GRANTABLE_CAPABILITIES for item in value):
            raise ValueError("Choose an existing mesh capability")
        return list(dict.fromkeys(value))


class NetworkEnrollmentAcknowledgment(BaseModel):
    """Closed compatibility admission; booleans must never coerce to version one."""

    model_config = {"extra": "forbid", "strict": True}
    authority_version: int
    scope: Literal["network_only"]
    request_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("authority_version", mode="before")
    @classmethod
    def exact_version(cls, value: Any) -> int:
        if type(value) is not int or value != 1:
            raise ValueError("Unknown network enrollment authority version")
        return value

    @field_validator("request_digest")
    @classmethod
    def exact_digest(cls, value: str) -> str:
        from local_operator.network.enrollment import DIGEST

        if DIGEST.fullmatch(value) is None:
            raise ValueError("Use the exact displayed enrollment request commitment")
        return value
