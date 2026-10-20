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


class NetworkMismatch(RemoteComputeError):
    """The run Radient answered with is not the network's run.

    A run is keyed by the network it was submitted through, so a response that
    names ANOTHER network — or names none at all — is not evidence the work
    landed where it was addressed. There is no retry that repairs that: the run
    exists, it is just not this tenant's, and treating it as one would report
    another network's work as this network's receipt.

    A MISSING ``network_id`` IS A MISMATCH, deliberately. The field is in the
    public contract, so its absence is a response this client cannot verify;
    accepting it would mean a build that silently stops checking the tenant.
    """


class InvalidIdempotencyKey(RemoteComputeError):
    """A key this client will not put in the header, and WHY, as a code.

    The reason is a machine code rather than a sentence because each reason has
    its own remedy and the copy is keyed, so the wording lives in the locale
    catalogues and the client (which has no locale) only classifies. Refused
    here rather than left to the server: a 400 after submission cannot tell the
    operator which of their keys was wrong.

    ``reason`` is ``empty``, ``too_long`` or ``not_printable``; ``max_bytes`` is
    the contract bound, carried so the surface can name it without importing the
    client's constant into its copy.
    """

    def __init__(self, reason: str, *, max_bytes: int = 0) -> None:
        super().__init__(f"the idempotency key is {reason}")
        self.reason = reason
        self.max_bytes = max_bytes


class Unreachable(RemoteComputeError):
    """The request never produced a response (DNS, TLS, timeout, refused).

    Typed rather than left as the raw ``httpx`` error so a caller can report
    "could not reach Radient" without importing the transport's exception tree.

    THE CHAIN IS CUT ON PURPOSE (``raise ... from None`` at the raise site). An
    ``httpx`` transport error carries the REQUEST it failed on, and that request
    carries the ``Authorization`` header — so keeping ``__cause__`` available, as
    this class used to argue for, put a bearer one ``traceback.format_exception``
    away from any log or support paste. The redaction rule at the top of this
    module wins over the debugging convenience: the fault is reported by its
    TYPE in this sentence, and the raw chain is never re-attached.
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
    "InvalidIdempotencyKey",
    "InvalidRequest",
    "NetworkMismatch",
    "RemoteComputeError",
    "RunNotFound",
    "Unreachable",
]
