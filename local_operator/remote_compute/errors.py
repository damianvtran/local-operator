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


class PreviewNotEnabled(RemoteComputeError):
    """403 — remote compute is not on this account's preview allowlist yet.

    The server-side preview is admitted per account, so a login the allowlist
    does not carry is refused before any run is composed. It is NOT a login
    problem (401 answers the same status for an expired grant) and not a credit
    problem (402 names that), which is why it gets its own type: the remedy is
    "wait for the preview to reach this account", and no retry or new key
    changes it.
    """


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


class RunIdentityMismatch(RemoteComputeError):
    """The run Radient answered with is not the run that was asked for.

    Every read is addressed by id (``GET /runs/:id``, ``POST /runs/:id/cancel``),
    so a body naming another run is a response this client cannot attribute: a
    cached page, a proxy that answered the wrong path, or a server bug. Printing
    it would show a state that belongs to work the operator did not ask about,
    and a cancel would look applied to a run nobody cancelled.

    There is nothing to retry: the id is either the one sent or it is not.

    BOTH IDS ARE CARRIED, as fields rather than only inside the sentence. The
    surface that reports this has keyed copy, so it has to build the sentence
    itself and name both runs; parsing them back out of an English message would
    make the copy a wire format.
    """

    def __init__(self, *, asked_for: str, answered: str) -> None:
        super().__init__(
            f"Radient answered for run {answered!r}, not the run {asked_for!r} that was asked for."
        )
        self.asked_for = asked_for
        self.answered = answered


class InvalidApiOverride(RemoteComputeError):
    """``LOP_REMOTE_COMPUTE_API_URL`` names a base this client will not use.

    The bearer goes to whatever host is named, so the override is held to the
    rules :func:`local_operator.remote_compute.config.resolve_api_url` states
    and an unusable value is a REFUSAL, never a silent fall back to production:
    a run sent to the wrong host is work nobody can account for, and the
    operator set the variable on purpose.
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


class UnreadableEnvelope(RemoteComputeError):
    """A success body that looks like an envelope this build cannot parse.

    The envelope's key set is exact (``msg``/``result``/``error``), and a body
    that names ``result`` BESIDE keys outside that set is neither a true envelope
    nor the payload itself. The strict test is kept on purpose — loosening it to
    "carries a ``result``" would truncate any object that grows such a field —
    so the unreadable shape is surfaced as its own failure instead of being
    guessed at.

    It means Radient's response format changed after this build was cut, and the
    remedy is a newer lop, not a retry.
    """


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
    403: PreviewNotEnabled,
    404: RunNotFound,
    409: IdempotencyConflict,
    503: CapacityUnavailable,
}

__all__ = [
    "ERROR_BY_STATUS",
    "CapacityUnavailable",
    "IdempotencyConflict",
    "InsufficientCredit",
    "InvalidApiOverride",
    "InvalidIdempotencyKey",
    "InvalidRequest",
    "NetworkMismatch",
    "PreviewNotEnabled",
    "RemoteComputeError",
    "RunIdentityMismatch",
    "RunNotFound",
    "Unreachable",
    "UnreadableEnvelope",
]
