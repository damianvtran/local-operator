"""Typed refusals for the community client, and their one wire projection.

WHY CODES, NOT SENTENCES. Every surface (desktop renderer, relay web, later the
TUI) owns its own localized copy for these states (RFC §5 copy tables), so the
client and the local routes carry a closed machine-readable ``code`` plus the
structured facts the copy needs (category, reference, retry_after, scope, the
new guidelines version). Nothing here quotes the hub's ``message`` field: the
server's text is English, may change, and an upstream echo is how a body or an
internal detail would leak through a local route.

The mapping follows ``docs/community-api.yaml`` (agent-server
feat/community-registry): 401 Unauthorized, 403 Forbidden, 404 NotFound (one
body for unknown/draft/hidden/unjoined), 409 Conflict (GuidelinesChanged or an
a state precondition), 413 PayloadTooLarge, 422 Refused, 429 RateLimited,
503 ModerationUnavailable, 500 Unavailable.

IMAGES UNAVAILABLE is not yet a named response in that contract. The RFC keeps
images disabled until their gates pass (canon §7; probation is "no images"),
and the kill switch has an ``images`` switch, so the hub must refuse an image
submission distinctly from a moderation outage. This client recognises the
``code: images_unavailable`` body on ANY status as that state; the contract
lane should name it (reported with this change).
"""

from __future__ import annotations

from typing import Any


class CommunityError(Exception):
    """A community operation did not succeed. ``code`` is the closed wire code.

    ``http_status`` is the status a LOCAL route answers with: the hub's own for
    hub refusals, a fixed one for client-side conditions.
    """

    code: str = "community_error"
    http_status: int = 502
    #: The Idempotency-Key of the submit/edit attempt this refusal ended, set by
    #: ``member_ops.run`` so a surface can replay or read that exact attempt.
    attempt_key: str | None = None

    def __init__(self, code: str | None = None, *, http_status: int | None = None) -> None:
        if code is not None:
            self.code = code
        if http_status is not None:
            self.http_status = http_status
        # ``str(exc)`` is the code: a traceback or log line never carries more.
        super().__init__(self.code)

    def details(self) -> dict[str, Any]:
        """Structured facts the surface's copy needs, beyond the code."""

        return {}

    def wire(self) -> dict[str, Any]:
        """The local routes' refusal body: ``{"code": ..., **details}``."""

        return {"code": self.code, **self.details()}


class CommunityDisabled(CommunityError):
    """The pilot switch is off (``community_enabled()``)."""

    code = "community_disabled"
    http_status = 404


class CommunitySignInRequired(CommunityError):
    """No usable pinned Radient OAuth row, or a 401 survived one forced refresh."""

    code = "community_signin_required"
    http_status = 401


class CommunityAccountMismatch(CommunityError):
    """The request was captured for another account than the one signed in now."""

    code = "account_mismatch"
    http_status = 409


class CommunityDestinationRefused(CommunityError):
    """The configured Radient host may not receive the account's bearer."""

    code = "community_destination_refused"
    http_status = 409


class CommunityUnreachable(CommunityError):
    """The hub could not be reached (transport failure or timeout)."""

    code = "community_unreachable"
    http_status = 502


class CommunityInvalidResponse(CommunityError):
    """The hub answered in a shape this build cannot read (or over the size cap)."""

    code = "community_invalid_response"
    http_status = 502


class CommunityInvalidRequest(CommunityError):
    """A local precondition failed before anything left (malformed id, bad image)."""

    code = "invalid_request"
    http_status = 422


class CommunityBadRequest(CommunityError):
    """The hub's 400: malformed or ambiguous request."""

    code = "bad_request"
    http_status = 400


class CommunityForbidden(CommunityError):
    """The hub's 403: missing role, seat, or a sanction applies."""

    code = "forbidden"
    http_status = 403


class CommunityNotFound(CommunityError):
    """The hub's 404. Unknown, draft, hidden and unjoined are deliberately one state."""

    code = "community_not_found"
    http_status = 404


class CommunityConflict(CommunityError):
    """A 409 that is not a guidelines change: a state precondition failed.

    NOT an idempotency-key reuse: the hub answers a reused key with the first
    attempt's stored outcome, never a 409 (agent-server #414 review R1-M2).
    """

    code = "conflict"
    http_status = 409


class CommunityGuidelinesChanged(CommunityConflict):
    """The guidelines changed: re-read bootstrap, re-accept, then retry."""

    code = "guidelines_changed"

    def __init__(self, guidelines_version: str | None) -> None:
        super().__init__()
        self.guidelines_version = guidelines_version

    def details(self) -> dict[str, Any]:
        return {"guidelines_version": self.guidelines_version}


class CommunityPayloadTooLarge(CommunityError):
    """The hub's 413 (images ≤ 5 MiB, ≤ 4096 px, 4 per message)."""

    code = "payload_too_large"
    http_status = 413


class CommunityRefused(CommunityError):
    """Moderation refused the submission. Shown only to the sender.

    ``category`` and ``reference`` are the only details the contract allows
    (no model output, scores or matched text).
    """

    code = "refused"
    http_status = 422

    def __init__(self, category: str, reference: str) -> None:
        super().__init__()
        self.category = category
        self.reference = reference

    def details(self) -> dict[str, Any]:
        return {"category": self.category, "reference": self.reference}


class CommunityRateLimited(CommunityError):
    """A per-sender window refused; nothing was screened or stored."""

    code = "rate_limited"
    http_status = 429

    def __init__(self, retry_after: int, scope: str | None) -> None:
        super().__init__()
        self.retry_after = retry_after
        self.scope = scope

    def details(self) -> dict[str, Any]:
        return {"retry_after": self.retry_after, "scope": self.scope}


class CommunityModerationUnavailable(CommunityError):
    """No verdict could be obtained; the submission failed closed, nothing stored."""

    code = "moderation_unavailable"
    http_status = 503

    def __init__(self, retry_after: int | None) -> None:
        super().__init__()
        self.retry_after = retry_after

    def details(self) -> dict[str, Any]:
        return {"retry_after": self.retry_after}


class CommunityImagesUnavailable(CommunityError):
    """Images are switched off (gates, kill switch or probation). Text still works."""

    code = "images_unavailable"
    http_status = 409


class CommunityUnavailable(CommunityError):
    """The hub's 500/other 5xx: community storage unavailable, no detail."""

    code = "community_unavailable"
    http_status = 503


__all__ = [
    "CommunityAccountMismatch",
    "CommunityBadRequest",
    "CommunityConflict",
    "CommunityDestinationRefused",
    "CommunityDisabled",
    "CommunityError",
    "CommunityForbidden",
    "CommunityGuidelinesChanged",
    "CommunityImagesUnavailable",
    "CommunityInvalidRequest",
    "CommunityInvalidResponse",
    "CommunityModerationUnavailable",
    "CommunityNotFound",
    "CommunityPayloadTooLarge",
    "CommunityRateLimited",
    "CommunityRefused",
    "CommunitySignInRequired",
    "CommunityUnavailable",
    "CommunityUnreachable",
]
