"""The sentence each typed remote-compute refusal carries, in one mapping.

WHY THIS IS NOT IN ``cli.py``. The client is a transport and raises TYPES; which
sentence a person reads for one is a presentation decision, and there are now TWO
surfaces that make it — the CLI prints it to stderr, and the desktop server route
returns it in a JSON body. Two ladders would word one refusal twice, and the first
reviewer of either would have to diff them: so the ladder is here, and each surface
keeps only what is its own (an exit code, an HTTP status).

THE ORDER OF THE ARMS IS LOAD-BEARING. Each arm is an ``isinstance`` test over one
class family, so a subclass must be named before its base — ``IdempotencyConflict``
before ``Conflict``, ``RunLimitReached`` before the generic ``RemoteComputeError``
tail — or the more specific remedy's sentence is unreachable. The final arm covers
everything else in the family with the failure's own text, which is a fixed literal
per :mod:`local_operator.remote_compute.errors` and never a provider body.

NOTHING HERE INVENTS COPY. Every sentence is a key of
``wire.remote_compute`` (the library's own namespace), and ``LoginRequired``'s is
the tunnels family's long-standing prose, which already names its remedy. A reason
the library supplies as a MACHINE code (a refused idempotency key, an unusable
repository) is mapped to that reason's key here, because this is the surface that
owns wording and ``errors.py`` has no locale.
"""

from __future__ import annotations

from local_operator.i18n.keys import wire_remote_compute as _wr
from local_operator.i18n.text import tr
from local_operator.remote_compute import repo as repo_module
from local_operator.remote_compute.errors import (
    CapacityUnavailable,
    Conflict,
    IdempotencyConflict,
    InsufficientCredit,
    InvalidApiOverride,
    InvalidIdempotencyKey,
    InvalidRequest,
    NetworkMismatch,
    PreviewNotEnabled,
    RunBusy,
    RunIdentityMismatch,
    RunLimitReached,
    RunNotFound,
    Unreachable,
    UnreadableEnvelope,
    UnusableRepo,
)
from local_operator.tunnels.errors import LoginRequired

_t = tr("wire.remote_compute")


def refusal_message(failure: BaseException) -> str:
    """The localized sentence for ``failure``, in the locale active right now.

    Resolved per call rather than captured: the server has ONE process locale, and
    it moves when the ``language`` setting does, so a resolver built at import time
    would serve the boot language until a restart (see
    :func:`local_operator.i18n.text.tr`).

    A failure this module does not know is reported with its own sentence rather
    than as an unknown type: every class in the family carries a fixed English
    literal, and hiding that behind a generic message would drop the one detail
    that survives the redaction rule.
    """
    if isinstance(failure, LoginRequired):
        # The tunnels family's own prose, which names `lop login radient`. Reusing
        # the instance's text keeps ONE sentence for one remedy across both
        # products; the transport already wrote it as a literal.
        return str(failure)
    if isinstance(failure, InvalidIdempotencyKey):
        return _key_message(failure)
    if isinstance(failure, NetworkMismatch):
        return _t(_wr.network_mismatch())
    if isinstance(failure, RunIdentityMismatch):
        return _t(_wr.run_identity_mismatch(asked_for=failure.asked_for, answered=failure.answered))
    if isinstance(failure, UnreadableEnvelope):
        return _t(_wr.unreadable_envelope())
    if isinstance(failure, Unreachable):
        return _t(_wr.unreachable())
    if isinstance(failure, InvalidApiOverride):
        return _t(_wr.api_override_invalid())
    if isinstance(failure, PreviewNotEnabled):
        return _t(_wr.preview_not_enabled())
    if isinstance(failure, InsufficientCredit):
        return _t(_wr.no_credit())
    if isinstance(failure, CapacityUnavailable):
        return _t(_wr.no_capacity())
    if isinstance(failure, RunNotFound):
        return _t(_wr.unknown_run())
    if isinstance(failure, IdempotencyConflict):
        return _t(_wr.conflict())
    if isinstance(failure, RunLimitReached):
        return _t(_wr.run_limit())
    if isinstance(failure, RunBusy):
        return _t(_wr.run_busy())
    if isinstance(failure, Conflict):
        return _t(_wr.refusal_conflict())
    if isinstance(failure, UnusableRepo):
        return _repo_message(failure)
    if isinstance(failure, InvalidRequest):
        return _t(_wr.invalid_request())
    return _t(_wr.failed(error=str(failure)))


def _key_message(failure: InvalidIdempotencyKey) -> str:
    """One sentence per reason a key was refused, all of them local to this device.

    The client supplies the REASON rather than a sentence because these are keyed
    messages (see ``errors.InvalidIdempotencyKey``); this is the surface that owns
    their wording.
    """
    if failure.reason == "empty":
        return _t(_wr.key_empty())
    if failure.reason == "too_long":
        return _t(_wr.key_too_long(max_bytes=str(failure.max_bytes)))
    return _t(_wr.key_not_printable())


def _repo_message(failure: UnusableRepo) -> str:
    """One sentence per reason a directory cannot be used for a remote run."""
    if failure.reason == repo_module.REASON_UNPUSHED:
        return _t(_wr.repo_unpushed_commit(detail=failure.detail))
    if failure.reason == repo_module.REASON_NOT_HTTPS:
        return _t(_wr.repo_not_https_remote())
    return _t(_wr.repo_no_remote())


__all__ = ["refusal_message"]
