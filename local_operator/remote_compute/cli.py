"""``lop exec --remote``: submit a short-lived run to Radient compute (tier 1).

WHAT IS AND IS NOT REMOTE. Tier 1 sends a prompt plus a git repository and
commit; the far end checks the repository out and runs the task there. Nothing
from this machine is uploaded — that is why ``--repo`` and ``--sha`` are
required rather than defaulting to the working directory, and why the refusal
says so instead of quietly copying anything. Workloads (tier 2), billing
surfaces and attestation belong to other slices and are not reachable from here.

REFUSALS, NOT SILENT IGNORES. Every ``exec`` flag that only means something to
a local session is refused by name when it meets ``--remote`` rather than
parsed and dropped: a run that ignores ``--tools`` or ``--background`` is a run
the operator believes was configured one way and was configured another. The
per-flag decision is in :data:`LOCAL_ONLY_FLAGS`, which is also the list the
tests iterate.

WHY THE RUN IS ADDRESSED BEFORE THE REQUEST IS BUILT. The order here is the
order of the things the operator can FIX: the preview switch, the flags, the
repository, the prompt, the login, the network, the model. Each refusal names
one remedy and exits non-zero, so a bad invocation costs a sentence rather than
a request that was addressed to a host before it was refused.

CTRL-C DOES NOT CANCEL. Interrupting a watch stops this process watching; it
does not ask, and it does not cancel the run, because a run id may be burning
hours on the far end and killing it from a keyboard interrupt is not something
this command was asked to do. The goodbye sentence names ``--remote-cancel``
instead.

COPY LIVES IN A NAMESPACE OF ITS OWN (``wire.remote_compute`` /
``cli.exec_remote_help``) rather than in ``wire.exec`` / ``cli.exec_help``, and
that is deliberate: an English edit to a namespace a locale already CARRIES
invalidates that locale's ledger row, which unships the whole locale. A new
namespace is English-only by construction and cannot.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from typing import Any

import httpx

from local_operator.exec_mode import ExecArgs, resolve_hosting_model_dry
from local_operator.i18n.keys import wire_remote_compute as _wr
from local_operator.i18n.text import tr
from local_operator.network.types import MeshRefusal
from local_operator.remote_compute import RunSpec, is_enabled, resolve_network
from local_operator.remote_compute.arguments import (
    LOCAL_ONLY_FLAGS,
    REMOTE_ONLY_FLAGS,
    first_set,
    is_set,
)
from local_operator.remote_compute.client import (
    KNOWN_RUN_STATUSES,
    TERMINAL_RUN_STATUSES,
    RemoteComputeClient,
    check_idempotency_key,
    new_idempotency_key,
    run_id,
    run_network,
    run_status,
)
from local_operator.remote_compute.errors import (
    CapacityUnavailable,
    IdempotencyConflict,
    InsufficientCredit,
    InvalidIdempotencyKey,
    InvalidRequest,
    NetworkMismatch,
    RunNotFound,
    Unreachable,
)
from local_operator.remote_compute.request import (
    RunCreateRequest,
    is_commit_sha,
    is_repo_url,
)
from local_operator.tunnels import api as tunnels_api
from local_operator.tunnels.errors import LoginRequired

_t = tr("wire.remote_compute")

#: The first wait between two ``GET /runs/:id`` reads while watching a run.
POLL_INTERVAL_SECONDS = 2.0

#: The longest wait the backoff will reach, so a slow run is still noticed.
POLL_MAX_INTERVAL_SECONDS = 30.0

#: How long the watch will wait at most before it gives up on the bound.
#:
#: ``max_hours`` is the RUN's own bound, enforced by the server; this is the
#: margin past it after which a watch can no longer learn anything the run's
#: deadline did not already decide.
WATCH_MARGIN_SECONDS = 300.0

#: ``time.monotonic``, named so a test can move the watch's clock without sleeping.
_monotonic = time.monotonic

#: The defaults the flags fall back to when the operator does not pass them.
#:
#: Held as ``None`` in the parser rather than as these values, so "not passed"
#: stays distinguishable from "passed with the default" — which is what lets
#: ``--size`` without ``--remote`` be refused instead of silently ignored.
DEFAULT_SIZE = "m"
DEFAULT_MAX_HOURS = 2.0


def _refuse(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _http_client() -> httpx.AsyncClient:
    """The transport this command uses; the seam the tests replace with a fake.

    One place, so no test needs to reach into a request to prove it is mocked,
    and so the CLI never constructs a second kind of client.
    """
    return httpx.AsyncClient(follow_redirects=False)


#: ``asyncio.sleep``, named so a test can raise ``KeyboardInterrupt`` from the
#: watch loop exactly where a Ctrl-C would land.
_sleep = asyncio.sleep


def _poll_delay(interval: float) -> float:
    """The next wait: ×1.5, jittered ±10%, never past the cap.

    The jitter is not decoration: several sessions watching runs that were
    submitted together would otherwise poll on the same second forever. The cap
    is applied AFTER the jitter so the bound is a real bound.
    """
    return min(POLL_MAX_INTERVAL_SECONDS, interval * 1.5 * random.uniform(0.9, 1.1))


def _hours(value: float) -> str:
    """``2.0`` → ``"2"``: the copy's unit is hours, so a trailing ``.0`` is noise."""
    return str(int(value)) if float(value).is_integer() else str(value)


def _key_refusal(failure: InvalidIdempotencyKey) -> int:
    """One sentence per reason the key was refused, all of them local.

    The client supplies the REASON rather than a sentence because these are
    keyed messages (see ``errors.InvalidIdempotencyKey``); this is the surface
    that owns their wording.
    """
    if failure.reason == "empty":
        return _refuse(_t(_wr.key_empty()))
    if failure.reason == "too_long":
        return _refuse(_t(_wr.key_too_long(max_bytes=str(failure.max_bytes))))
    return _refuse(_t(_wr.key_not_printable()))


def _report(failure: BaseException) -> int:
    """One sentence per typed failure — and never a word of a provider body.

    The mapping lives here rather than in the client so the client stays a
    transport: which failures deserve their own remedy is a presentation
    decision, and the sentences it picks are the ones an operator can act on.
    """
    if isinstance(failure, LoginRequired):
        return _refuse(str(failure))
    if isinstance(failure, InvalidIdempotencyKey):
        return _key_refusal(failure)
    if isinstance(failure, NetworkMismatch):
        return _refuse(_t(_wr.network_mismatch()))
    if isinstance(failure, Unreachable):
        return _refuse(_t(_wr.unreachable()))
    if isinstance(failure, InsufficientCredit):
        return _refuse(_t(_wr.no_credit()))
    if isinstance(failure, CapacityUnavailable):
        return _refuse(_t(_wr.no_capacity()))
    if isinstance(failure, RunNotFound):
        return _refuse(_t(_wr.unknown_run()))
    if isinstance(failure, IdempotencyConflict):
        return _refuse(_t(_wr.conflict()))
    if isinstance(failure, InvalidRequest):
        return _refuse(_t(_wr.invalid_request()))
    return _refuse(_t(_wr.failed(error=str(failure))))


def _credential() -> int:
    """This device's signed-in Radient account, or the refusal that names login.

    Resolved through the tunnels client's own selector, so "which Radient login
    is this?" has ONE answer on this machine — and one sentence, already
    pointing at ``lop login radient``, for the case where the answer is none.
    """
    return tunnels_api.credential_id(None)


async def _create_run(
    credential: int, request: RunCreateRequest, idempotency_key: str
) -> dict[str, Any]:
    async with _http_client() as http:
        client = RemoteComputeClient(http, credential_id=credential)
        return await client.create_run(request, idempotency_key=idempotency_key)


async def _get_run(credential: int, identifier: str) -> dict[str, Any]:
    async with _http_client() as http:
        return await RemoteComputeClient(http, credential_id=credential).get_run(identifier)


async def _cancel_run(credential: int, identifier: str) -> dict[str, Any]:
    async with _http_client() as http:
        return await RemoteComputeClient(http, credential_id=credential).cancel_run(identifier)


async def _watch(
    credential: int, submitted: dict[str, Any], *, max_hours: float
) -> dict[str, Any] | None:
    """Poll until a terminal status; ``None`` when the watch's bound is reached.

    Prints transitions rather than every read: a run that sits in ``running``
    for forty minutes would otherwise write a line every two seconds into a log
    nobody reads, and the transitions are what a person is waiting for.

    THE WAIT BACKS OFF and the watch is BOUNDED by the run's own ``max_hours``
    plus a small margin. Two separate reasons: fixed-rate polling for hours is
    load nobody asked for, and a run past its server-side deadline can no longer
    change, so a watch that kept asking would never end. Reaching the bound is
    REPORTED by the caller (the run is still running and it says how to check
    it) rather than looking like a finished run.

    A status this build cannot name is NOT terminal — stopping early would report
    a still-burning run as finished — and it is announced once, as unknown, so a
    run stuck in a vocabulary this client does not share is visible.
    """
    identifier = run_id(submitted)
    network = run_network(submitted)
    current_status = run_status(submitted)
    print(_t(_wr.status_line(status=current_status)))
    announced_unknown: set[str] = set()
    if current_status not in KNOWN_RUN_STATUSES:
        print(_t(_wr.status_unknown(status=current_status)))
        announced_unknown.add(current_status)
    deadline = _monotonic() + max_hours * 3600.0 + WATCH_MARGIN_SECONDS
    interval = POLL_INTERVAL_SECONDS
    payload = submitted
    async with _http_client() as http:
        client = RemoteComputeClient(http, credential_id=credential)
        while current_status not in TERMINAL_RUN_STATUSES:
            if _monotonic() >= deadline:
                return None
            await _sleep(interval)
            interval = _poll_delay(interval)
            payload = await client.get_run(identifier, expected_network=network)
            updated = run_status(payload)
            if updated != current_status:
                current_status = updated
                if updated in KNOWN_RUN_STATUSES:
                    print(_t(_wr.status_line(status=updated)))
                elif updated not in announced_unknown:
                    print(_t(_wr.status_unknown(status=updated)))
                    announced_unknown.add(updated)
    return payload


def _run_options_set(args: argparse.Namespace) -> bool:
    """Whether a run-submitting flag was passed alongside a remote command."""
    return any(is_set(getattr(args, name, None)) for name, _ in REMOTE_ONLY_FLAGS) or bool(
        getattr(args, "command", None)
    )


def _remote_status(identifier: str | None) -> int:
    """``--remote-status``: the run as JSON, or a typed refusal.

    Does NOT require the preview switch: reading a run this account already
    owns is not sending work, and a switch turned off after submission must not
    hide the state of work that is still running.
    """
    if not identifier:
        return _refuse(_t(_wr.alone()))
    try:
        payload = asyncio.run(_get_run(_credential(), identifier))
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError) as failure:
        return _report(failure)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


def _remote_cancel(identifier: str | None) -> int:
    """``--remote-cancel``: idempotent, and reachable while the switch is off."""
    if not identifier:
        return _refuse(_t(_wr.alone()))
    try:
        asyncio.run(_cancel_run(_credential(), identifier))
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError) as failure:
        return _report(failure)
    print(_t(_wr.cancelled(run_id=identifier)))
    return 0


def _submit(args: argparse.Namespace) -> int:
    """``--remote``: refuse what cannot work, then submit and watch (or detach)."""
    if not is_enabled():
        return _refuse(_t(_wr.disabled()))
    local_only = first_set(args, LOCAL_ONLY_FLAGS)
    if local_only is not None:
        return _refuse(_t(_wr.local_only(flag=local_only)))
    repo = getattr(args, "repo", None)
    sha = getattr(args, "sha", None)
    if not repo or not sha:
        return _refuse(_t(_wr.needs_repo()))
    # SHAPE, not just presence: a branch name or a local path reads as a valid
    # argument and then fails at the far end, after submission.
    if not is_repo_url(repo):
        return _refuse(_t(_wr.bad_repo()))
    if not is_commit_sha(sha):
        return _refuse(_t(_wr.bad_sha()))
    prompt = getattr(args, "command", None)
    if not prompt or not str(prompt).strip():
        return _refuse(_t(_wr.needs_prompt()))
    # BEFORE THE CREDENTIAL READ (E8), because these key refusals are local: an
    # operator whose invocation is already wrong should not first be told the
    # token endpoint could not be reached. An EMPTY `--idempotency-key` is
    # refused rather than replaced with a fresh key: the operator asked for a
    # retry to collide with an earlier submission, and a minted key would submit
    # the work a second time.
    key_argument = getattr(args, "idempotency_key", None)
    try:
        key = (
            check_idempotency_key(key_argument)
            if key_argument is not None
            else new_idempotency_key()
        )
    except InvalidIdempotencyKey as failure:
        # Through ``_report`` so there is ONE mapping from a typed failure to a
        # sentence, whether it was refused here or inside the client.
        return _report(failure)
    try:
        credential = _credential()
    except ValueError as failure:
        return _refuse(str(failure))
    try:
        binding = resolve_network(getattr(args, "network", None))
    except MeshRefusal as refusal:
        return _refuse(refusal.sentence)
    exec_args = ExecArgs(
        hosting=getattr(args, "hosting", None),
        model=getattr(args, "model", None),
        agent_name=getattr(args, "agent_name", None),
        agent_id=getattr(args, "agent_id", None),
        team=getattr(args, "team", None),
    )
    try:
        hosting, model = resolve_hosting_model_dry(exec_args)
    except ValueError as failure:
        return _refuse(_t(_wr.failed(error=str(failure))))
    max_hours = (
        DEFAULT_MAX_HOURS if getattr(args, "max_hours", None) is None else float(args.max_hours)
    )
    try:
        spec = RunSpec(
            prompt=str(prompt),
            repo_url=str(repo),
            repo_sha=str(sha),
            hosting=hosting,
            model=model,
            size=getattr(args, "size", None) or DEFAULT_SIZE,
            max_hours=max_hours,
            budget_usd=getattr(args, "budget_usd", None),
            name=getattr(args, "name", None),
        )
    except ValueError as failure:
        return _refuse(_t(_wr.failed(error=str(failure))))
    request = spec.bind(binding)
    try:
        submitted = asyncio.run(_create_run(credential, request, key))
        identifier = run_id(submitted)
        # AS THE SERVER RETURNED IT: the receipt may only claim what the response
        # confirmed, never what this client asked for.
        network = run_network(submitted)
    except KeyboardInterrupt:
        # A Ctrl-C here may land after the request was sent, so the run may
        # exist: saying nothing is right, sending a cancel is not.
        return 130
    except (ValueError, OSError) as failure:
        return _report(failure)
    if getattr(args, "detach", False):
        # The run id ALONE on stdout, so `id=$(lop exec --remote --detach ...)`
        # captures exactly the id. The receipt — key included — goes to stderr.
        print(identifier)
        print(
            _t(
                _wr.detached(
                    run_id=identifier,
                    network_id=network,
                    idempotency_key=key,
                )
            ),
            file=sys.stderr,
        )
        return 0
    print(_t(_wr.submitted(run_id=identifier, network_id=network)))
    try:
        final = asyncio.run(_watch(credential, submitted, max_hours=max_hours))
    except KeyboardInterrupt:
        print(_t(_wr.interrupted(run_id=identifier)))
        # 130 is the SIGINT convention this CLI already uses for a quiet cancel,
        # and it is the honest code here: the watch stopped, the run did not.
        return 130
    except (ValueError, OSError) as failure:
        return _report(failure)
    if final is None:
        # Never silent: the run has outlived this command's watch, and the
        # operator is told so AND how to ask again.
        print(_t(_wr.still_running(run_id=identifier, max_hours=_hours(max_hours))))
        return 0
    status = run_status(final)
    print(_t(_wr.finished(run_id=identifier, status=status)))
    return 0 if status == "succeeded" else 1


def remote_command(args: argparse.Namespace) -> int:
    """The single entry point ``cli.main`` calls for every remote invocation."""
    status_id = getattr(args, "remote_status", None)
    cancel_id = getattr(args, "remote_cancel", None)
    if status_id or cancel_id:
        # The two commands are READ-ONLY or single-purpose: a prompt or any run
        # flag beside one of them is a composed command that would silently drop
        # half of what was typed, so it is refused instead.
        if (status_id and cancel_id) or is_set(getattr(args, "remote", None)):
            return _refuse(_t(_wr.alone()))
        if _run_options_set(args):
            return _refuse(_t(_wr.alone()))
        return _remote_status(status_id) if status_id else _remote_cancel(cancel_id)
    if not is_set(getattr(args, "remote", None)):
        # A remote-only flag without --remote: name the flag that was typed.
        spelling = first_set(args, REMOTE_ONLY_FLAGS) or "--detach"
        return _refuse(_t(_wr.only_with(flag=spelling)))
    return _submit(args)


__all__ = [
    "DEFAULT_MAX_HOURS",
    "DEFAULT_SIZE",
    "LOCAL_ONLY_FLAGS",
    "POLL_INTERVAL_SECONDS",
    "POLL_MAX_INTERVAL_SECONDS",
    "REMOTE_ONLY_FLAGS",
    "WATCH_MARGIN_SECONDS",
    "remote_command",
]
