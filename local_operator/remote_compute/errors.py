"""Why a remote-compute call failed, as types rather than prose.

SAME REDACTION RULE AS THE TUNNELS CLIENT. A Radient response body can echo a
bearer, an account identifier or a pricing field, so nothing this package raises
carries one: every sentence here is a fixed literal and the only variable part
is the HTTP status. That rule is what makes the CLASS the signal — a 402 and a
503 both read as "the request was not admitted", and only a caller that can tell
the types apart knows whether waiting (capacity) or paying (credit) is the fix.

``LoginRequired`` IS REUSED, NOT RESPELLED. A 401 here and a 401 there mean the
same thing to the operator (log in again), and ``tunnels.errors.LoginRequired``
is already the type every surface on that path knows. A second class with the
same meaning would be the kind of duplicate that gets caught one handler at a
time.
"""

from __future__ import annotations


class RemoteComputeError(ValueError):
    """Base for every remote-compute failure that is not a login problem.

    A ``ValueError`` subclass on purpose, exactly as ``tunnels.errors`` is: the
    CLI and every other handler on the Radient path already catch
    ``ValueError``, and none of them should have to learn a new base class to
    keep reporting a failed request.
    """


class InvalidRequest(RemoteComputeError):
    """400 — the server refused the body's shape or a query parameter.

    A LOCAL bug or a contract drift, never something the operator can retype
    their way out of: the message says so rather than asking them to check a
    field, because the client is the one that composed it.
    """


class InsufficientCredit(RemoteComputeError):
    """402 — admission refused for credit, so no retry changes the answer."""


class RunNotFound(RemoteComputeError):
    """404 — no such run under the selected login, or the path is wrong."""


class IdempotencyConflict(RemoteComputeError):
    """409 — this idempotency key was already used with a DIFFERENT body.

    Distinct from a replay (same key, same body, same run): a conflict means the
    caller reused a key for new work, and retrying it unchanged can never
    succeed.
    """


class CapacityUnavailable(RemoteComputeError):
    """503 — capacity or substrate is unavailable; the same call may work later."""


class Unreachable(RemoteComputeError):
    """The request never produced a response (DNS, TLS, timeout, refused).

    Typed rather than left as the raw ``httpx`` error so a caller can report
    "could not reach Radient" without importing the transport's exception tree,
    and so the chain (``__cause__``) stays available for anyone who needs the
    underlying fault.
    """


#: Status → type, one row per status the public contract names EXCEPT 401,
#: whose type is the shared :class:`~local_operator.tunnels.errors.LoginRequired`
#: and is therefore not a ``RemoteComputeError`` (see ``client._failed``).
ERROR_BY_STATUS: dict[int, type[RemoteComputeError]] = {
    400: InvalidRequest,
    402: InsufficientCredit,
    404: RunNotFound,
    409: IdempotencyConflict,
    503: CapacityUnavailable,
}

__all__ = [
    "ERROR_BY_STATUS",
    "CapacityUnavailable",
    "IdempotencyConflict",
    "InsufficientCredit",
    "InvalidRequest",
    "RemoteComputeError",
    "RunNotFound",
    "Unreachable",
]
