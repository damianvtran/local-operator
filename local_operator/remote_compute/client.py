"""The remote-compute HTTP client: owner-pinned Radient calls, never a key cascade.

REUSE, BY COMPOSITION. This client makes the same requests the tunnels client
makes, against the same OAuth grant, so it reuses that client's parts rather
than re-deriving them: ``tunnels.api.credential_id`` for "which signed-in
Radient account is this device's", and ``tunnels.errors.LoginRequired`` for the
one failure a retry cannot fix. The HOST comes from
:func:`local_operator.remote_compute.config.resolve_api_url`, whose default is
``tunnels.config.DEFAULT_API_URL``. That is the one place the two products
differ on purpose: remote compute may be pointed at a non-production Radient
for testing, while a tunnel's credential stays pinned to the host its origin
proof is issued for. What it does NOT do is share the request method: the tunnels
one prefixes ``/v1/tunnels``, unwraps the tunnel envelope and maps 404 to a
tunnel-specific type, so a shared method would have to be parameterised by path,
envelope and error table — three knobs for two callers, which is the shape where
the next divergent detail gets special-cased. The reasoning the two share is
the ORDER (resolve owner → refresh → fixed headers → one request → typed error,
never a body) and that order is written down in both places, twice, on purpose:
it is nine lines, and the alternative is a shared helper whose every sentence
would have to be parameterised by which product is failing.

WHOLE-BODY REDACTION, EXACTLY AS THERE. No failure this module raises carries a
word of the provider's response: status codes and fixed sentences only. A body
can hold a bearer, an account id or a credit figure, and none of those belong in
a log, a transcript, or a support paste.

A REMOTE COMPUTE REQUEST IS ALWAYS BUILT FROM A :class:`RunCreateRequest`. The
create method's parameter is that type and not a mapping, so the tenant field
can only come from a binding ``resolve_network`` issued — the guarantee is a
type fact at the network boundary, not a rule a caller is asked to follow.

THE ENVELOPE AND THE NESTED RUN. Every 2xx body is ``{"msg": <sentence>,
"result": <body>}``, and a create's ``result`` is a WRAPPER around the run:
``{"run": <Run>, "run_token"?: <secret>, "tunnel"?: <secret>}``. So a create
reads one level deeper than a read does, and :func:`_decoded` stays a plain
envelope unwrapper rather than learning about runs.

THE CREATE WRAPPER CARRIES SECRETS, AND THIS MODULE BINDS NONE OF THEM.
``run_token`` authenticates the pod to the server and ``tunnel`` carries a
cloudflared token and an origin credential; both are returned once, on the
201, meant for the pod and for nobody on this machine. :meth:`create_run`
reads ``run`` and lets the wrapper go out of scope, so there is no value here
to log, print, cache or serialise by accident — the rule is structural, not a
redaction step someone has to remember to run.
"""

from __future__ import annotations

import dataclasses
import json
import urllib.parse
import uuid
from contextlib import closing
from typing import Any

import httpx

from local_operator.providers.auth_store import (
    AuthStore,
    AuthStoreError,
    CredentialInvalidError,
)
from local_operator.remote_compute.config import resolve_api_url
from local_operator.remote_compute.errors import (
    ERROR_BY_STATUS,
    InvalidIdempotencyKey,
    NetworkMismatch,
    RemoteComputeError,
    RunIdentityMismatch,
    Unreachable,
    UnreadableEnvelope,
)
from local_operator.remote_compute.request import RunCreateRequest
from local_operator.tunnels import api as tunnels_api
from local_operator.tunnels.errors import LoginRequired

#: Where the run endpoints live, relative to the API host.
RUNS_PATH = "/v1/remote-compute/runs"

#: The keys the v1 response envelope may carry, and nothing else.
#:
#: The envelope is ``{"msg": "<sentence>", "result": <body>}``. Membership is
#: checked as a WHOLE-KEY-SET test rather than "does a ``result`` key exist",
#: because the run object has twenty-odd top-level fields of its own and a run
#: that ever carried one called ``result`` would be truncated to that field by
#: the narrower rule. ``error`` is listed because an error envelope names it —
#: an error body is never surfaced, but an error-shaped body with a ``result``
#: beside it is still an envelope, not a run. A body that names ``result``
#: BESIDE a key outside this set is refused as :class:`UnreadableEnvelope`
#: rather than quietly treated as the payload (see ``_decoded``).
ENVELOPE_KEYS: frozenset[str] = frozenset({"msg", "result", "error"})

#: The contract's bound on ``Idempotency-Key``: 1 to 128 BYTES.
MAX_IDEMPOTENCY_KEY_BYTES = 128

#: Statuses after which a run will not change again, so polling can stop.
#:
#: ``stopped`` covers every way a run can end (the OUTCOME says which), and
#: ``deleted`` is the state after the record is gone. ``error`` is deliberately
#: NOT here: the server retries an errored run, so a watch that stopped on it
#: would report work as finished that is about to run again.
TERMINAL_RUN_STATUSES: frozenset[str] = frozenset({"stopped", "deleted"})

#: Every status the contract's closed set names.
#:
#: Anything outside it is a status this client cannot read, so it is neither
#: terminal nor announced as understood — the watch says so once rather than
#: pretending it is one of these.
KNOWN_RUN_STATUSES: frozenset[str] = frozenset(
    {
        "pending",
        "reconciling",
        "running",
        "stopping",
        "stopped",
        "revoking",
        "deleted",
        "error",
    }
)

#: The outcomes a stopped run can carry, as the contract spells them.
#:
#: ``succeeded`` is exit 0, ``failed`` any other exit, ``expired`` a deadline and
#: ``cancelled`` a user cancel. Anything else (including absent) is reported as
#: unknown rather than guessed from ``status``: two ways of spelling a run's end
#: is the disagreement this vocabulary exists to prevent.
KNOWN_RUN_OUTCOMES: frozenset[str] = frozenset({"succeeded", "failed", "expired", "cancelled"})

#: The status a live HTTP answer uses to say "this was a replay, not a new run".
REPLAY_STATUS = 200


def new_idempotency_key() -> str:
    """A fresh key per invocation; a retry reuses it through ``--idempotency-key``."""
    return str(uuid.uuid4())


def check_idempotency_key(key: str) -> str:
    """The contract's header rule, checked before a request is composed.

    BYTES, not characters: the header is bytes on the wire, and a 100-character
    key of non-ASCII script is over the limit the server enforces. PRINTABLE
    ASCII only, for the same reason at the other end of the range — a key with a
    newline or a control byte is not a header value, and a server that accepts
    it would be storing a key no retry can reproduce byte-for-byte.

    A refusal carries a REASON CODE (see :class:`InvalidIdempotencyKey`) rather
    than a sentence: the caller is a CLI whose copy is keyed per locale.
    """
    if not isinstance(key, str) or not key:
        raise InvalidIdempotencyKey("empty")
    if len(key.encode("utf-8")) > MAX_IDEMPOTENCY_KEY_BYTES:
        raise InvalidIdempotencyKey("too_long", max_bytes=MAX_IDEMPOTENCY_KEY_BYTES)
    if any(not 0x20 <= ord(character) <= 0x7E for character in key):
        raise InvalidIdempotencyKey("not_printable")
    return key


def _run_path(run_id: str) -> str:
    """``/<id>``, quoted so a caller's string cannot address another endpoint."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("a remote run id is required")
    return "/" + urllib.parse.quote(run_id, safe="")


def _decoded(response: httpx.Response) -> Any:
    """The result payload, unwrapped from the envelope.

    ONLY AN ENVELOPE IS UNWRAPPED, and the test is exact: the payload's whole key
    set is a subset of :data:`ENVELOPE_KEYS` and names ``result``. The narrower
    "a ``result`` key exists" rule truncated any object that happened to have a
    top-level field of that name, and the run object is exactly the kind of thing
    that grows fields.

    A BODY THAT NAMES ``result`` BESIDE OTHER KEYS IS NOT GUESSED AT. It is
    neither an envelope (the keys are outside the set) nor the payload itself
    (the payload of a per-run endpoint is a run, and a run's ``result`` field
    arrives here only after unwrapping, never at this level) — so it is the
    response format having moved on without this build. Refusing it by type is
    the honest answer; accepting it would either truncate a run to one field or
    hand the CLI a body it would report as a run.

    AN UNPARSEABLE BODY IS RAISED OUTSIDE THE HANDLER. ``json.JSONDecodeError``
    keeps the whole document on ``.doc``, so raising it (or chaining it) would
    park a provider body — and whatever the server echoed in it — on an
    exception the CLI prints. Leaving the ``except`` block first means neither
    ``__cause__`` nor ``__context__`` exists to be walked to it.
    """
    unparseable = False
    payload: Any = None
    try:
        payload = response.json()
    except ValueError:
        unparseable = True
    if unparseable:
        raise RemoteComputeError("Radient returned an invalid remote-compute response.")
    if isinstance(payload, dict) and "result" in payload:
        if payload.keys() <= ENVELOPE_KEYS:
            return payload["result"]
        raise UnreadableEnvelope(
            "Radient returned a body that looks like an envelope with keys this build does "
            "not know, so its response format has changed."
        )
    return payload


def _confirm_network(run: dict[str, Any], *, expected: str) -> str:
    """The run's network, refused as a typed mismatch when it is not ``expected``.

    Takes the RUN, never the response wrapper: the tenant field is a run field,
    so a create has to reach through ``result["run"]`` to be checked at all. A
    client that checked the wrapper would find no ``network_id`` there and
    either miss the field entirely or read the wrong level.

    Returns the value the SERVER sent, so a caller that reports a network is
    reporting what Radient confirmed rather than what it asked for; the two are
    equal here by construction, but only this one is evidence.
    """
    returned = run.get("network_id")
    if not isinstance(returned, str) or not returned:
        raise NetworkMismatch(
            "Radient returned a remote run without a network id, so it cannot be "
            "confirmed as this network's run."
        )
    if returned != expected:
        raise NetworkMismatch(
            "Radient returned a remote run from a different network than the one it "
            "was submitted to."
        )
    return returned


def _failed(status: int) -> RemoteComputeError | LoginRequired:
    """The typed error for ``status``, with a sentence that carries no body.

    ``401`` is answered with the SHARED ``LoginRequired`` rather than a row in
    :data:`ERROR_BY_STATUS`: its remedy is the one a tunnel request's is (run
    ``lop login radient``), and a second class for the same meaning would make
    every handler learn the difference.
    """
    if status == 401:
        return LoginRequired(
            "The Radient login was refused for this remote-compute request (HTTP 401); log in "
            "again with `lop login radient`."
        )
    error = ERROR_BY_STATUS.get(status)
    if error is not None:
        return error(f"Radient refused the remote-compute request (HTTP {status}).")
    return RemoteComputeError(f"Radient remote-compute request failed (HTTP {status}).")


@dataclasses.dataclass(frozen=True)
class CreatedRun:
    """A submitted run, and whether the server replayed an earlier submission.

    ``replayed`` is the 200-vs-201 distinction: a 201 created this run, a 200
    answered an idempotent replay with the run an earlier POST created. The run
    is the same either way, which is exactly why the status has to be carried
    beside it rather than read back out of it.

    THE POD'S SECRETS ARE NOT HERE. The create wrapper's ``run_token`` and
    ``tunnel`` are for the far end; this type is the whole of what a create
    hands back, so no caller can reach them from what it is given.
    """

    run: dict[str, Any]
    replayed: bool


@dataclasses.dataclass(frozen=True)
class _Reply:
    """A decoded payload and the status that carried it."""

    payload: Any
    status: int


class RemoteComputeClient:
    """Owner-pinned calls against ``/v1/remote-compute``.

    ``credential_id`` pins one Radient OAuth row. ``None`` resolves the device's
    current login on every call — the right default for a CLI invocation, which
    is one call in one process, and the reason a re-login mid-session is picked
    up without rebuilding the client.
    """

    def __init__(self, client: httpx.AsyncClient, *, credential_id: int | None = None) -> None:
        self.client = client
        self.selected = credential_id
        # RESOLVED ONCE PER CLIENT, not per request: an unusable override has to
        # be refused BEFORE a request is composed, and the destination warning
        # then lands once rather than ahead of every poll of a watch.
        self.api_url = resolve_api_url()

    async def _headers(self) -> dict[str, str]:
        """A fresh bearer, or a typed refusal — never a fallback to an API key.

        The inference resolver rotates credentials and may fall back to an
        environment key. That policy is unusable here: a remote run is billed to,
        owned by and visible to ONE account, so a quota event or an expired login
        may not silently switch which account the work belongs to.
        """
        selected = self.selected if self.selected is not None else tunnels_api.credential_id(None)
        with closing(AuthStore()) as store:
            row = store.get_credential(selected)
            if row is None or row.provider != "radient" or row.credential_type != "oauth":
                # LoginRequired, not the generic error: no retry can conjure the
                # credential row back, so the caller must be told to stop asking
                # and log in again.
                raise LoginRequired(
                    "The remote-compute Radient login is unavailable; log in again with "
                    "`lop login radient`."
                )
            try:
                credentials = await store.ensure_oauth_fresh_or_raise(selected)
            except AuthStoreError as failure:
                # Chained, never flattened: a token endpoint that could not be
                # reached and a grant the IdP revoked arrive here as one
                # sentence, and only the chain can tell an operator which it was.
                # The revoked case gets the class a caller can act on; the
                # reachability case stays a fixed sentence with its cause.
                if isinstance(failure, CredentialInvalidError):
                    raise LoginRequired(
                        "The remote-compute Radient login could not be refreshed; log in again "
                        "with `lop login radient`."
                    ) from failure
                raise RemoteComputeError(
                    "The remote-compute Radient login could not be refreshed."
                ) from failure
        token = credentials.get("access") if credentials else None
        if not isinstance(token, str) or not token:
            raise LoginRequired(
                "The remote-compute Radient login expired; log in again with `lop login radient`."
            )
        return {"Authorization": f"Bearer {token}"}

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        empty_body_ok: bool = False,
    ) -> _Reply:
        """One call, as the payload plus the status that answered it.

        The STATUS comes back beside the payload because 200 and 201 mean
        different things on a create (replay versus created) and the body is
        identical either way, so the status is not recoverable from it.
        """
        headers = await self._headers()
        if idempotency_key is not None:
            headers["Idempotency-Key"] = check_idempotency_key(idempotency_key)
        # THE FAILURE IS RAISED OUTSIDE THE HANDLER, on purpose. ``from None``
        # only suppresses the traceback's "during handling" banner; the httpx
        # error is still reachable as ``__context__``, and it carries the
        # request, which carries the Authorization header. Leaving the
        # ``except`` block first means neither ``__cause__`` nor
        # ``__context__`` exists, so there is no object to walk to a bearer.
        response: httpx.Response | None = None
        try:
            response = await self.client.request(
                method,
                self.api_url + RUNS_PATH + path,
                json=body,
                params=params,
                headers=headers,
                timeout=30,
                follow_redirects=False,
            )
        except httpx.HTTPError:
            pass
        if response is None:
            raise Unreachable("Radient could not be reached for this remote-compute request.")
        # Never surface a provider body — see the module docstring.
        if response.status_code >= 400:
            raise _failed(response.status_code)
        if empty_body_ok and not response.content.strip():
            # A cancel answers 204 (or a bodiless 200) on success, and a bodiless
            # success is not a malformed response.
            return _Reply(payload={}, status=response.status_code)
        return _Reply(payload=_decoded(response), status=response.status_code)

    async def create_run(self, request: RunCreateRequest, *, idempotency_key: str) -> CreatedRun:
        """Submit one run. A replay with the same key AND body returns the same run.

        ``idempotency_key`` is a required keyword with no default: a retry that
        silently minted a new key would submit the work twice, and the whole
        point of the header is that the caller decided what "the same request"
        means.

        ONLY ``result["run"]`` IS READ. The same wrapper carries ``run_token``
        and ``tunnel`` — the pod's secrets, returned once — and this method
        never names either: there is no binding to leak into a return value, a
        log line or a cache, and none is created here for a later edit to
        discover it can read.
        """
        if not isinstance(request, RunCreateRequest):
            raise TypeError(
                "create_run takes a RunCreateRequest built from a resolved network "
                "binding; a mapping could name a network this device is not in"
            )
        reply = await self._request(
            "POST", "", body=request.to_json(), idempotency_key=idempotency_key
        )
        wrapper = reply.payload
        run = wrapper.get("run") if isinstance(wrapper, dict) else None
        if not isinstance(run, dict):
            raise RemoteComputeError("Radient returned a remote-compute response without a run.")
        # The tenant is CONFIRMED off the nested RUN, not assumed from the request
        # and not read off the wrapper (which has no such field): a server that
        # answered with another network's run would otherwise be reported as this
        # network's receipt.
        _confirm_network(run, expected=request.network_id)
        return CreatedRun(run=run, replayed=reply.status == REPLAY_STATUS)

    async def get_run(
        self, identifier: str, *, expected_network: str | None = None
    ) -> dict[str, Any]:
        """One run by id, and — when the caller names the network — its tenant confirmed.

        ``expected_network`` exists for the watch loop, which is following a run
        it already submitted: a poll that starts answering about a different
        network's run must stop the watch with a typed failure rather than
        reporting that run's status as the one being watched.
        """
        reply = await self._request("GET", _run_path(identifier))
        run = _single_run(reply.payload, identifier)
        if expected_network is not None:
            _confirm_network(run, expected=expected_network)
        return run

    async def list_runs(
        self, *, network_id: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """One page of runs, newest first by the server's ordering.

        ``network_id`` narrows the listing. An EMPTY string is refused here
        rather than sent: the endpoint answers 400 for it, and an empty value in
        a query string is almost always an unset variable the caller meant to
        leave out.

        The page is ``result["runs"]``, with the page's own ``limit``/``offset``
        beside it; only the runs are returned, because a caller that read the
        echoed paging back would be trusting the server's copy of its own request.
        """
        if network_id is not None and not network_id.strip():
            raise ValueError("network_id must be a network id or omitted, not an empty string")
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if network_id is not None:
            params["network_id"] = network_id
        reply = await self._request("GET", "", params=params)
        page = reply.payload
        runs = page.get("runs") if isinstance(page, dict) else None
        if not isinstance(runs, list) or not all(isinstance(run, dict) for run in runs):
            raise RemoteComputeError("Radient returned an invalid remote-compute run list.")
        return runs

    async def cancel_run(self, identifier: str) -> dict[str, Any]:
        """Cancel a run. Idempotent: cancelling a finished run is not an error.

        A bodiless 2xx (204, or a 200 with no body) is SUCCESS: the cancel
        endpoint documents no response body, and reading the absence as a
        malformed response turned a cancel that worked into an error report.

        The answer, when there is one, must name the run that was cancelled: a
        body about another run would make this command report work cancelled
        that nobody cancelled.
        """
        reply = await self._request("POST", _run_path(identifier) + "/cancel", empty_body_ok=True)
        if not reply.payload:
            return {}
        return _single_run(reply.payload, identifier)


def _single_run(payload: Any, asked_for: str) -> dict[str, Any]:
    """The run in a per-run response, refused unless it is the run asked for.

    Every per-run endpoint is addressed by id, so the id in the body is the one
    check that ties an answer to the question. A missing ``id`` is refused by
    :func:`run_id`; a different one is refused here.
    """
    if not isinstance(payload, dict):
        raise RemoteComputeError("Radient returned an invalid remote-compute response.")
    answered = run_id(payload)
    if answered != asked_for:
        raise RunIdentityMismatch(asked_for=asked_for, answered=answered)
    return payload


def run_status(payload: dict[str, Any]) -> str:
    """The run's status, or a typed refusal — never an empty string to compare."""
    status = payload.get("status")
    if not isinstance(status, str) or not status:
        raise RemoteComputeError("Radient returned a remote run without a status.")
    return status


def run_id(payload: dict[str, Any]) -> str:
    """The run's id, or a typed refusal."""
    identifier = payload.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise RemoteComputeError("Radient returned a remote run without an id.")
    return identifier


def run_network(payload: dict[str, Any]) -> str:
    """The run's network AS THE SERVER RETURNED IT, or a typed refusal.

    Read after :meth:`RemoteComputeClient.create_run` has confirmed it, so this
    is the value a receipt may call confirmed; it is never the caller's
    expectation.
    """
    network = payload.get("network_id")
    if not isinstance(network, str) or not network:
        raise NetworkMismatch("Radient returned a remote run without a network id.")
    return network


def is_terminal(payload: dict[str, Any]) -> bool:
    """Whether polling may stop for this run."""
    return run_status(payload) in TERMINAL_RUN_STATUSES


def run_outcome(payload: dict[str, Any]) -> str:
    """The run's OUTCOME, or ``"unknown"`` when the server named none.

    An outcome is only meaningful once a run has stopped, and the contract says
    so; a run still running has none. ``unknown`` rather than an empty string
    because every caller renders it: the finished line needs a word to print,
    and the exit code needs a case it can map to failure without inventing
    success.

    A value outside the four known outcomes is returned AS GIVEN, and
    ``unknown`` only covers absence — a build that mapped a new outcome to
    ``failed`` would be claiming a result the server did not report.
    """
    outcome = payload.get("outcome")
    if not isinstance(outcome, str) or not outcome:
        return "unknown"
    return outcome


def run_exit_code(payload: dict[str, Any]) -> int | None:
    """The run's exit code when the server reported one, else ``None``.

    ``None`` and 0 are different answers and are never collapsed: a run with no
    exit code (expired, cancelled, never started) did not exit 0. ``bool`` is
    excluded because it is an ``int`` to ``isinstance`` and ``True`` is not an
    exit code.
    """
    code = payload.get("exit_code")
    if isinstance(code, bool) or not isinstance(code, int):
        return None
    return code


def charged_amount(payload: dict[str, Any]) -> str | None:
    """What the run was charged, as the server reported it, or ``None``.

    DISPLAYED, NEVER COMPUTED. ``charged_usd`` is the settled amount and wins
    when present; ``cost`` is the breakdown the server also sends, shown as its
    JSON when there is no settled amount. This client knows no prices, so it
    quotes whatever the run carries rather than doing arithmetic on someone
    else's rate card.
    """
    charged = payload.get("charged_usd")
    if isinstance(charged, bool) or not isinstance(charged, (int, float, str)):
        cost = payload.get("cost")
        if isinstance(cost, dict) and cost:
            return json.dumps(cost, ensure_ascii=False, sort_keys=True)
        return None
    return str(charged)


def run_max_hours(payload: dict[str, Any]) -> float | None:
    """The run's ``max_hours`` as the server admitted it, or ``None``.

    The server may cap what was asked for, so this is the value a watch must
    bound itself by; ``None`` (absent, or not a number) is the caller's cue to
    fall back to the hours it requested. ``bool`` is excluded for the same
    reason as in :func:`run_exit_code`.
    """
    hours = payload.get("max_hours")
    if isinstance(hours, bool) or not isinstance(hours, (int, float)):
        return None
    return float(hours)


__all__ = [
    "CreatedRun",
    "ENVELOPE_KEYS",
    "KNOWN_RUN_OUTCOMES",
    "KNOWN_RUN_STATUSES",
    "MAX_IDEMPOTENCY_KEY_BYTES",
    "REPLAY_STATUS",
    "RUNS_PATH",
    "TERMINAL_RUN_STATUSES",
    "RemoteComputeClient",
    "charged_amount",
    "check_idempotency_key",
    "is_terminal",
    "new_idempotency_key",
    "run_exit_code",
    "run_id",
    "run_max_hours",
    "run_network",
    "run_outcome",
    "run_status",
]
