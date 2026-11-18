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

import datetime
import email.utils

# MODULE-LEVEL, and that is safe: ``tunnels.errors`` is a family of ``ValueError``
# subclasses with no imports of its own, and the shared login type has to be
# constructible here for a coded ``revoked`` refusal. The tunnels PACKAGE is a
# five-line docstring, so this costs an import of two small modules, not a walk
# into the mesh or session layers.
from local_operator.tunnels.errors import LoginRequired


class RemoteComputeError(ValueError):
    """Base for every remote-compute failure that is not a login problem.

    A ``ValueError`` subclass on purpose, exactly as ``tunnels.errors`` is: the
    CLI and every other handler on the Radient path already catch
    ``ValueError``, and none of them should have to learn a new base class to
    keep reporting a failed request.

    EVERY INSTANCE MAY CARRY THE SERVER'S OWN ACCOUNT OF THE REFUSAL, and does
    whenever the client raised it from an HTTP answer: ``code`` is the contract's
    machine code (``run_limit``, ``preview_not_enabled``, …) when that answer
    named one, ``status`` the HTTP status, and ``retry_after`` the parsed
    ``Retry-After`` when the server sent one. They are FIELDS rather than part of
    the sentence because the sentence is an English literal this module owns for
    logs and tracebacks, while a surface with keyed copy needs the code to pick
    the right translated sentence, and a caller that may retry needs the delay as
    a number.

    THE CLASS IS COARSER THAN THE CODE, on purpose (see :data:`ERROR_BY_CODE`):
    the class names the REMEDY — pay, wait, log in, pick a new name — and that is
    the distinction an ``isinstance`` handler can act on without knowing every
    code. A surface that needs to be more precise than the remedy reads ``code``.
    """

    #: The contract's refusal code, when the answer carried one this build knows.
    code: str | None = None
    #: The HTTP status that refused the request, when there was one.
    status: int | None = None
    #: Seconds the server asked the caller to wait, parsed from ``Retry-After``.
    retry_after: float | None = None

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        status: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.retry_after = retry_after


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
    """409 ``idempotency_conflict`` — the key was already used with a DIFFERENT body.

    Distinct from a replay (same key, same body, same run): a conflict means the
    caller reused a key for new work, and retrying it unchanged can never
    succeed.

    A 409 IS NOT ALWAYS THIS, which is why the class is no longer the 409's only
    answer (see :data:`ERROR_BY_CODE`): the server also refuses a run at its
    concurrent limit, a lease that is still busy, a name another run claimed and
    a run that already finished with 409, and reporting any of those as an
    idempotency conflict told the operator to change a key when the fix was to
    wait or to free a slot.
    """


class RunLimitReached(RemoteComputeError):
    """``run_limit``/``workload_limit`` — the account is at a concurrency limit.

    The remedy is to finish or cancel something and try again; a new key, more
    credit or a re-login changes nothing. The distinction from
    :class:`InsufficientCredit` is the whole point: an account inside its plan
    that is merely busy was being told, via the old status table, that its
    idempotency key had conflicted.

    The CODE says which limit, and both are one class because both are one
    remedy; the copy therefore names a concurrent limit rather than a "run" one.
    """


class RunBusy(RemoteComputeError):
    """``run_busy``/``workload_busy`` — that run or workload is already occupied.

    The server answers 409 while another operation holds the same run or
    workload, so the request may succeed once it is free. Waiting is the remedy;
    retrying immediately is the same request and the same refusal.
    """


class Conflict(RemoteComputeError):
    """A 409 (or a coded conflict) whose precise cause this build cannot name.

    THE GENERIC ARM OF THE 409, kept so an unrecognised conflict is honest: an
    answer that carries no code and no message this build can attribute to one
    cause is reported as a conflict, never as an idempotency problem. ``code``
    still carries whatever the answer did name (``name_claimed``,
    ``run_finished``), so a surface can refine the sentence without a new class.

    It is also the answer for a coded conflict whose remedy is not covered by a
    class of its own: report the conflict, do not invent a remedy for it.
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


class UnusableRepo(RemoteComputeError):
    """A directory whose repository cannot be handed to a remote run, and WHY.

    The reason is a machine code rather than a sentence for the same reason
    :class:`InvalidIdempotencyKey`'s is: each reason has its own remedy and its
    own keyed copy, and this module has no locale. ``reason`` is
    ``not_https_remote`` (the origin is not a credential-free ``https://`` URL —
    an ssh or scp-like form needs a key and an agent the container does not
    have), ``unpushed_commit`` (HEAD is on no ``origin`` tracking ref, so the
    container would clone a repository that does not contain the commit) or
    ``no_remote`` (no origin, no commit, or not a repository at all).

    REFUSED LOCALLY RATHER THAN SUBMITTED: a run whose commit the container
    cannot fetch is work that fails minutes later at the far end, with a message
    about a checkout the operator cannot see.
    """

    def __init__(self, reason: str, *, detail: str = "") -> None:
        super().__init__(f"this repository cannot be used for a remote run: {reason}")
        self.reason = reason
        #: The remote URL, branch or path the refusal is about, already scrubbed
        #: of userinfo by :mod:`local_operator.remote_compute.repo`; empty when
        #: there is nothing useful to name.
        self.detail = detail


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


#: The refusal codes the server's contract names, as this build knows them.
#:
#: A CLOSED SET, and that is load-bearing twice. It is what
#: :func:`refusal_code` will accept out of a response body — an unknown string is
#: dropped rather than carried into an error or a log — and it is what a test can
#: enumerate so a code the contract gains later fails here instead of falling
#: through to the status table.
REFUSAL_CODES: frozenset[str] = frozenset(
    {
        "preview_not_enabled",
        "insufficient_credits",
        "run_limit",
        "workload_limit",
        "idempotency_conflict",
        "run_busy",
        "workload_busy",
        "name_claimed",
        "not_found",
        "run_finished",
        "revoked",
        "invalid_request",
        "invalid_idempotency_key",
        "capacity_unavailable",
        "admission_unavailable",
        "not_configured",
    }
)

#: The prefix every attestation refusal shares (``attestation_*``).
#:
#: A prefix rather than an enumeration: the family is the far end's to extend
#: (nonce, image, measurement, …), and a new member means "this run was not
#: admitted because the machine could not be vouched for", which is one remedy
#: this build already reports. A code that is not in :data:`REFUSAL_CODES` and
#: does not carry this prefix is dropped.
ATTESTATION_CODE_PREFIX = "attestation_"


class _LoginRequiredSentinel:
    """The table's value for "this refusal's type is the SHARED login error".

    A distinct TYPE rather than a marker string, so the tables stay honestly
    typed (``dict[str, type[RemoteComputeError] | _LoginRequiredSentinel]``) and
    a reader cannot pass a stack of strings to a branch that expects a class.
    :data:`~local_operator.tunnels.errors.LoginRequired` itself cannot sit in the
    table because it is deliberately NOT a ``RemoteComputeError``.
    """


#: The single instance :data:`ERROR_BY_CODE` uses for ``revoked``.
LOGIN_REQUIRED = _LoginRequiredSentinel()

#: Code → type: the server's own reason, when the answer carried one.
#:
#: READ BEFORE THE STATUS, ALWAYS. A status is the transport's word for a class
#: of refusals (409 covers four of them), so the status table below is only the
#: fallback for a server that does not send a code yet — the whole point of this
#: table is that a 409 says WHICH 409 it is.
#:
#: WHERE TWO CODES SHARE A CLASS, they share a REMEDY and nothing else: the code
#: still travels on the error, so a surface can be more precise than the class.
#: ``revoked`` maps to the shared login error because the remedy is a re-login,
#: and a second type for "log in again" is exactly the duplicate this package's
#: docstring refuses.
ERROR_BY_CODE: dict[str, type[RemoteComputeError] | _LoginRequiredSentinel] = {
    "preview_not_enabled": PreviewNotEnabled,
    "insufficient_credits": InsufficientCredit,
    "run_limit": RunLimitReached,
    "workload_limit": RunLimitReached,
    "idempotency_conflict": IdempotencyConflict,
    "run_busy": RunBusy,
    "workload_busy": RunBusy,
    "name_claimed": Conflict,
    "not_found": RunNotFound,
    "run_finished": Conflict,
    "revoked": LOGIN_REQUIRED,
    "invalid_request": InvalidRequest,
    # A key the SERVER calls invalid is a contract disagreement, not a typo: this
    # client checks shape before sending (``check_idempotency_key``), so a key it
    # sent and the server refused is a bug on one of the two sides — the same
    # reading ``InvalidRequest`` already carries.
    "invalid_idempotency_key": InvalidRequest,
    "capacity_unavailable": CapacityUnavailable,
    "admission_unavailable": CapacityUnavailable,
    "not_configured": Conflict,
}

#: Status → type, one row per status the public contract names EXCEPT 401,
#: whose type is the shared :class:`~local_operator.tunnels.errors.LoginRequired`
#: and is therefore not a ``RemoteComputeError`` (see ``client._failed``).
#:
#: THE FALLBACK, for a server that sends no code. It is unchanged from before the
#: codes existed so an older server maps exactly as it always did, with ONE
#: exception: 409's row is the generic :class:`Conflict`, because a codeless 409
#: is not evidence of an idempotency problem. :func:`refusal_for` is what
#: recovers the idempotency reading when the message says so.
ERROR_BY_STATUS: dict[int, type[RemoteComputeError]] = {
    400: InvalidRequest,
    402: InsufficientCredit,
    403: PreviewNotEnabled,
    404: RunNotFound,
    409: Conflict,
    429: CapacityUnavailable,
    503: CapacityUnavailable,
}

#: Message fragments that identify an IDEMPOTENCY conflict in a codeless 409.
#:
#: A codeless 409 is ambiguous by construction, and the alternative to this table
#: is telling an operator at their run limit to change a key. So the message is
#: consulted, but ONLY for the one reading that changes the remedy: the phrase
#: must name idempotency AND a reuse/conflict of it, and anything else stays the
#: generic conflict. Matching is case-folded and deliberately narrow — a message
#: this build does not recognise is a conflict it does not name, never a guess.
_IDEMPOTENCY_HINTS: tuple[str, ...] = (
    "idempotency key",
    "idempotency-key",
)
_IDEMPOTENCY_CONFLICT_HINTS: tuple[str, ...] = (
    "already used",
    "already been used",
    "different request",
    "different body",
    "conflict",
)


def _sentence(status: int) -> str:
    """The fixed sentence an error carries; the status is the only variable part."""
    return f"Radient refused the remote-compute request (HTTP {status})."


def refusal_code(body: object) -> str | None:
    """The refusal code in a response body, or ``None`` when there is none here.

    THE ONLY PLACE A BODY IS READ FOR AN ERROR, and it reads a CLOSED set of
    strings: the field is looked for at the envelope's own level, inside an
    ``error`` object and inside a ``detail`` object (the three shapes a refusal
    envelope has been seen in), and a value outside :data:`REFUSAL_CODES` that
    does not carry :data:`ATTESTATION_CODE_PREFIX` is DROPPED. So a body cannot
    smuggle prose, an account id or a token into a log through this door — what
    comes out is one enum member or nothing.
    """
    if not isinstance(body, dict):
        return None
    for candidate in _code_candidates(body):
        if candidate in REFUSAL_CODES or candidate.startswith(ATTESTATION_CODE_PREFIX):
            return candidate
    return None


def _code_candidates(body: dict[str, object]) -> list[str]:
    """Every plausible ``code`` string in a refusal body, in reading order."""
    values: list[str] = []
    candidates = (
        body.get("code"),
        _nested_code(body.get("error")),
        _nested_code(body.get("detail")),
    )
    for value in candidates:
        if isinstance(value, str) and value:
            values.append(value)
    return values


def _nested_code(container: object) -> object:
    """``container["code"]`` when ``container`` is an object, else ``None``."""
    return container.get("code") if isinstance(container, dict) else None


def refusal_message(body: object) -> str | None:
    """The server's own sentence for a refusal, or ``None``.

    USED FOR DISAMBIGUATION ONLY, never reported: the 409 fallback needs to know
    whether a codeless conflict names idempotency. Capped so a pathological body
    cannot turn into a large string here, and never stored on the error.
    """
    if not isinstance(body, dict):
        return None
    for key in ("msg", "message", "detail", "error"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value[:500]
    return None


def retry_after_seconds(value: str | None, *, now: datetime.datetime | None = None) -> float | None:
    """``Retry-After`` as seconds from now, or ``None`` when it is not usable.

    THE HEADER HAS TWO LEGAL SHAPES — delta-seconds (``120``) and an HTTP-date
    (``Wed, 21 Oct 2015 07:28:00 GMT``) — and both are parsed here so a caller
    never has to know which one the server chose. A date is converted against
    ``now`` (UTC when it carries no offset); a date in the past is ``0.0`` rather
    than a negative delay, because "negative seconds from now" is a number no
    caller can sleep on. Anything unparsable is ``None``: an unreadable header is
    the absence of a hint, not a zero-second one.
    """
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.isdigit():
        return float(int(raw))
    try:
        parsed = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    current = now if now is not None else datetime.datetime.now(datetime.timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=datetime.timezone.utc)
    return max(0.0, (parsed - current).total_seconds())


def refusal_for(
    status: int,
    *,
    code: str | None = None,
    message: str | None = None,
    retry_after: float | None = None,
) -> RemoteComputeError | LoginRequired:
    """The typed failure for an HTTP answer, by CODE first and status second.

    THE ORDER IS THE FIX. A status is a class of refusals: 409 alone covers a run
    at its limit, a busy lease, a claimed name and an idempotency conflict, so
    mapping on it told a user at their run limit that their idempotency key had
    conflicted. The code says which one, and only a server that does not send one
    falls back to the status table.

    WHAT IS NOT KNOWN IS NOT INVENTED. An unknown code is treated exactly like no
    code (the status table answers), and an ambiguous 409 with no code is only
    attributed to idempotency when ``message`` says so — see
    :data:`_IDEMPOTENCY_HINTS`. Everything else is :class:`Conflict`.

    ``retry_after`` is carried, never applied: this module knows no policy for
    how long a caller should wait, and the one caller that sleeps owns it.
    """
    if status == 401:
        return LoginRequired(
            "The Radient login was refused for this remote-compute request (HTTP 401); log in "
            "again with `lop login radient`."
        )
    error_type = _error_for(status, code=code, message=message)
    if isinstance(error_type, _LoginRequiredSentinel):
        return LoginRequired(
            "The Radient login for this remote-compute request is no longer valid; log in "
            "again with `lop login radient`."
        )
    if error_type is None:
        return RemoteComputeError(
            _sentence(status), code=code, status=status, retry_after=retry_after
        )
    return error_type(_sentence(status), code=code, status=status, retry_after=retry_after)


def _error_for(
    status: int, *, code: str | None, message: str | None
) -> type[RemoteComputeError] | _LoginRequiredSentinel | None:
    """The class for this answer, by code first; ``None`` means the base class."""
    if code is not None:
        mapped = ERROR_BY_CODE.get(code)
        if mapped is not None:
            return mapped
        # An attestation refusal is the far end saying it could not vouch for the
        # machine. It is a conflict this build cannot remedy, and it must not be
        # reported as one it can (a retry, a payment, a new key).
        if code.startswith(ATTESTATION_CODE_PREFIX):
            return Conflict
        return ERROR_BY_STATUS.get(status)
    mapped_status = ERROR_BY_STATUS.get(status)
    if status == 409 and mapped_status is Conflict and _names_idempotency(message):
        return IdempotencyConflict
    return mapped_status


def _names_idempotency(message: str | None) -> bool:
    """Whether a codeless 409's message precisely names an idempotency conflict."""
    if not message:
        return False
    folded = message.casefold()
    if not any(hint in folded for hint in _IDEMPOTENCY_HINTS):
        return False
    return any(hint in folded for hint in _IDEMPOTENCY_CONFLICT_HINTS)


__all__ = [
    "ATTESTATION_CODE_PREFIX",
    "ERROR_BY_CODE",
    "ERROR_BY_STATUS",
    "LOGIN_REQUIRED",
    "REFUSAL_CODES",
    "CapacityUnavailable",
    "Conflict",
    "IdempotencyConflict",
    "InsufficientCredit",
    "InvalidApiOverride",
    "InvalidIdempotencyKey",
    "InvalidRequest",
    "NetworkMismatch",
    "PreviewNotEnabled",
    "RemoteComputeError",
    "RunBusy",
    "RunIdentityMismatch",
    "RunLimitReached",
    "RunNotFound",
    "Unreachable",
    "UnreadableEnvelope",
    "UnusableRepo",
    "refusal_code",
    "refusal_for",
    "refusal_message",
    "retry_after_seconds",
]
