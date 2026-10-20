"""The remote-compute HTTP client: owner-pinned Radient calls, never a key cascade.

REUSE, BY COMPOSITION. This client makes the same requests the tunnels client
makes, against the same host and the same OAuth grant, so it reuses that
client's parts rather than re-deriving them: ``tunnels.config.DEFAULT_API_URL``
for the host, ``tunnels.api.credential_id`` for "which signed-in Radient account
is this device's", and ``tunnels.errors.LoginRequired`` for the one failure a
retry cannot fix. What it does NOT do is share the request method: the tunnels
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
"""

from __future__ import annotations

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
from local_operator.remote_compute.errors import (
    ERROR_BY_STATUS,
    RemoteComputeError,
    Unreachable,
)
from local_operator.remote_compute.request import RunCreateRequest
from local_operator.tunnels import api as tunnels_api
from local_operator.tunnels.config import DEFAULT_API_URL
from local_operator.tunnels.errors import LoginRequired

#: Where the run endpoints live, relative to the API host.
RUNS_PATH = "/v1/remote-compute/runs"

#: The contract's bound on ``Idempotency-Key``: 1 to 128 BYTES.
MAX_IDEMPOTENCY_KEY_BYTES = 128

#: Statuses after which a run will not change again, so polling can stop.
#:
#: The public contract names the request/response FIELDS but not the status
#: vocabulary, so this is the client's one reading of it, in one place. An
#: unknown status is deliberately NOT terminal: a run whose state this build
#: cannot name is still running as far as anyone here can tell, and stopping a
#: watch early would report a run as finished while it is burning hours.
TERMINAL_RUN_STATUSES: frozenset[str] = frozenset(
    {"succeeded", "failed", "cancelled", "canceled", "expired"}
)


def new_idempotency_key() -> str:
    """A fresh key per invocation; a retry reuses it through ``--idempotency-key``."""
    return str(uuid.uuid4())


def check_idempotency_key(key: str) -> str:
    """The contract's 1..128-BYTE rule, checked before a request is composed.

    BYTES, not characters: the header is bytes on the wire, and a 100-character
    key of non-ASCII script is over the limit the server enforces. Refused here
    so the caller learns which key was wrong, rather than getting a bare 400
    after the work was addressed.
    """
    if not isinstance(key, str) or not key:
        raise ValueError("an idempotency key is required")
    if len(key.encode("utf-8")) > MAX_IDEMPOTENCY_KEY_BYTES:
        raise ValueError(f"the idempotency key is longer than {MAX_IDEMPOTENCY_KEY_BYTES} bytes")
    return key


def _run_path(run_id: str) -> str:
    """``/<id>``, quoted so a caller's string cannot address another endpoint."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("a remote run id is required")
    return "/" + urllib.parse.quote(run_id, safe="")


def _decoded(response: httpx.Response) -> Any:
    """The result payload, unwrapped from the ``{"result": ...}`` envelope.

    The API's envelope is shared with every other v1 endpoint on this host; a
    bare payload is accepted too because the run endpoints document the RUN as
    the response body and an envelope is a transport detail this client should
    not make a caller think about.
    """
    try:
        payload = response.json()
    except ValueError as failure:
        raise RemoteComputeError(
            "Radient returned an invalid remote-compute response."
        ) from failure
    if isinstance(payload, dict) and "result" in payload:
        return payload["result"]
    return payload


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
    ) -> Any:
        headers = await self._headers()
        if idempotency_key is not None:
            headers["Idempotency-Key"] = check_idempotency_key(idempotency_key)
        try:
            response = await self.client.request(
                method,
                DEFAULT_API_URL + RUNS_PATH + path,
                json=body,
                params=params,
                headers=headers,
                timeout=30,
                follow_redirects=False,
            )
        except httpx.HTTPError as failure:
            # Re-raised as a typed error with the cause kept: `httpx`'s own
            # hierarchy is a transport detail, and `__cause__` is where the real
            # fault (a timeout, a refused connection) still lives.
            raise Unreachable(
                "Radient could not be reached for this remote-compute request."
            ) from failure
        # Never surface a provider body — see the module docstring.
        if response.status_code >= 400:
            raise _failed(response.status_code)
        return _decoded(response)

    async def create_run(
        self, request: RunCreateRequest, *, idempotency_key: str
    ) -> dict[str, Any]:
        """Submit one run. A replay with the same key AND body returns the same run.

        ``idempotency_key`` is a required keyword with no default: a retry that
        silently minted a new key would submit the work twice, and the whole
        point of the header is that the caller decided what "the same request"
        means.
        """
        if not isinstance(request, RunCreateRequest):
            raise TypeError(
                "create_run takes a RunCreateRequest built from a resolved network "
                "binding; a mapping could name a network this device is not in"
            )
        payload = await self._request(
            "POST", "", body=request.to_json(), idempotency_key=idempotency_key
        )
        if not isinstance(payload, dict):
            raise RemoteComputeError("Radient returned an invalid remote-compute response.")
        return payload

    async def get_run(self, run_id: str) -> dict[str, Any]:
        payload = await self._request("GET", _run_path(run_id))
        if not isinstance(payload, dict):
            raise RemoteComputeError("Radient returned an invalid remote-compute response.")
        return payload

    async def list_runs(
        self, *, network_id: str | None = None, limit: int = 50, offset: int = 0
    ) -> Any:
        """One page of runs, newest first by the server's ordering.

        ``network_id`` narrows the listing. An EMPTY string is refused here
        rather than sent: the endpoint answers 400 for it, and an empty value in
        a query string is almost always an unset variable the caller meant to
        leave out.
        """
        if network_id is not None and not network_id.strip():
            raise ValueError("network_id must be a network id or omitted, not an empty string")
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if network_id is not None:
            params["network_id"] = network_id
        return await self._request("GET", "", params=params)

    async def cancel_run(self, run_id: str) -> dict[str, Any]:
        """Cancel a run. Idempotent: cancelling a finished run is not an error."""
        payload = await self._request("POST", _run_path(run_id) + "/cancel")
        if not isinstance(payload, dict):
            raise RemoteComputeError("Radient returned an invalid remote-compute response.")
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


def is_terminal(payload: dict[str, Any]) -> bool:
    """Whether polling may stop for this run."""
    return run_status(payload) in TERMINAL_RUN_STATUSES


__all__ = [
    "MAX_IDEMPOTENCY_KEY_BYTES",
    "RUNS_PATH",
    "TERMINAL_RUN_STATUSES",
    "RemoteComputeClient",
    "check_idempotency_key",
    "is_terminal",
    "new_idempotency_key",
    "run_id",
    "run_status",
]
