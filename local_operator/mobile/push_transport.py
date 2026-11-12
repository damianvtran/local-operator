"""The Radient cloud transport for mobile push — push/ack-sync S7.

WHY THIS EXISTS. Core already builds and queues the three machine→cloud frames
(``push_worker`` emits, ``push_credentials`` report blocks, the register
forward), but until now nothing bound them to a wire: ``PushWorker`` and
``CredentialReport`` take an injected transport and the product built neither,
and ``POST /api/push/register`` validated the phone's token and dropped it. This
module is that binding, per agent-server ``docs/SESSION-PUSH.md`` §4 and ADR
0006 §3.2 (local-operator-mobile).

THE CREDENTIAL IS THE TUNNEL'S RADIENT LOGIN. Every call goes through
:class:`~local_operator.tunnels.api.RadientTunnels`, pinned to the tunnel's
``credential_id`` — the same owner bearer, refresh and response redaction the
tunnel commands use — and only ever to ``DEFAULT_API_URL``. The "computer"
handle the cloud scopes devices by is the ``tunnel_id`` (SESSION-PUSH §1): it is
already account-scoped and opaque, so nothing new is minted. No tunnel, no lane
(ADR §2.4): :func:`build_push_lane` answers ``None`` and the daemon behaves
exactly as before.

WHAT IT MUST NEVER DO.
* Persist the push token. ``register_forward`` holds it only for the life of
  the forward task; the machine's registry still stores nothing (ADR §4).
* Retry an emit. The worker owns the wire retry (3 attempts over ~2 min, same
  ``Idempotency-Key``, ``push_worker.EMIT_RETRY_*``); a second retry layer here
  would multiply attempts against a cloud that is already refusing.
* Block the daemon's event loop. ``emit`` and ``credentials`` are synchronous
  because their callers run on a worker thread (``asyncio.to_thread`` in
  ``MobileDaemon._scan_once``); each call runs its own short event loop on that
  thread. ``register_forward`` is a coroutine scheduled on the daemon's loop.

DISARM. ``LoginRequired`` (no usable login) and ``RecordNotFound`` (404: this
login does not own the tunnel — or the cloud predates the push routes) disarm
the transport until the daemon restarts, with ONE log line: neither heals on a
retry, and a lane that kept knocking would spend the worker's retries and the
log on a certainty. This is also what makes a core release safe against a cloud
that has not shipped the routes yet (SESSION-PUSH §5, "inert … 404 → disarm").
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from local_operator.mobile.push_credentials import (
    CARRIER_EMIT,
    CARRIER_HEARTBEAT,
    CARRIER_REGISTER,
    REPORT_HEARTBEAT_ROUTE,
    REPORT_REGISTER_ROUTE,
    CredentialReport,
    block_devices,
    registry_loader,
)
from local_operator.mobile.push_payload import EMIT_ROUTE
from local_operator.mobile.push_worker import (
    EMIT_RETRY_ATTEMPTS,
    EMIT_RETRY_INTERVAL_S,
    EmitAccepted,
    EmitRefused,
    PushWorker,
)
from local_operator.tunnels.api import RadientTunnels
from local_operator.tunnels.errors import LoginRequired, RecordNotFound, RequestFailed

logger = logging.getLogger(__name__)

#: ``RadientTunnels.request`` takes paths relative to this family; the emit
#: route lives inside it, so it keeps that client's strict envelope check (the
#: 202 is ``{msg, result: {emit_id, accepted_at}}``, SESSION-PUSH §0 item 2).
_TUNNELS_PREFIX = "/v1/tunnels"

#: Status reported for a refusal that never got an HTTP answer (timeout,
#: connection error, disarmed lane). ``EmitRefused`` documents 0 as "no status".
NO_STATUS = 0


class PushTransportDisarmed(RuntimeError):
    """Raised by ``credentials`` once the lane is disarmed.

    A raise rather than a silent return because ``CredentialReport._flush``
    reads "the transport returned" as "the block went out" and advances its
    bookkeeping; a disarmed lane sent nothing and must not say otherwise.
    """


def _default_client() -> httpx.AsyncClient:
    # trust_env=False for the reason the tunnel connector gives
    # (``tunnels/service.py``): proxy configuration and environment credentials
    # are never forwarded to Radient. Per-request timeout=30 is set by
    # ``RadientTunnels.request`` itself.
    return httpx.AsyncClient(trust_env=False)


class RadientPushTransport:
    """One computer's three cloud calls, bound to its tunnel and its login."""

    def __init__(
        self,
        *,
        tunnel_id: str,
        credential_id: int,
        client_factory: Callable[[], httpx.AsyncClient] = _default_client,
        sleep: Callable[[float], Any] = asyncio.sleep,
        retry_interval_s: float = EMIT_RETRY_INTERVAL_S,
    ) -> None:
        """``client_factory`` and ``sleep`` are injected for tests only.

        The factory is how a test (and the loopback evidence rig) substitutes an
        ``httpx.MockTransport`` or a local stub: the HOST is never configurable
        — ``RadientTunnels`` always targets ``DEFAULT_API_URL``.
        """
        self.tunnel_id = tunnel_id
        self.credential_id = credential_id
        self._client_factory = client_factory
        self._sleep = sleep
        self._retry_interval_s = retry_interval_s
        self._disarmed = False
        self._disarm_lock = threading.Lock()
        #: Strong references to in-flight register forwards: the event loop
        #: holds tasks weakly, and a forward collected mid-retry would vanish
        #: without its log line.
        self._forwards: set[asyncio.Task[None]] = set()

    @property
    def computer(self) -> str:
        """The §3.2 ``computer`` handle — the tunnel id (SESSION-PUSH §1)."""
        return self.tunnel_id

    @property
    def disarmed(self) -> bool:
        return self._disarmed

    def _disarm(self, failure: Exception) -> None:
        with self._disarm_lock:
            if self._disarmed:
                return
            self._disarmed = True
        # The class name only: the sentences are fixed literals already, but
        # nothing about the login or the response body belongs in this log.
        logger.warning(
            "mobile push disarmed until restart (%s): the tunnel's Radient login "
            "cannot reach the push routes for tunnel %s",
            type(failure).__name__,
            self.tunnel_id,
        )

    async def _call(
        self,
        *,
        path: str | None = None,
        api_path: str | None = None,
        body: dict[str, Any],
        idempotency_key: str | None = None,
    ) -> Any:
        async with self._client_factory() as client:
            api = RadientTunnels(self.credential_id, client)
            if api_path is not None:
                return await api.request("POST", body=body, api_path=api_path)
            return await api.request("POST", path or "", body=body, idempotency_key=idempotency_key)

    # -- the PushEmitTransport -------------------------------------------------

    def emit(self, body: Mapping[str, Any], *, idempotency_key: str) -> EmitAccepted | EmitRefused:
        """POST one emit; 2xx is an accept, anything else one refusal. Never raises.

        The worker acts on the verdict's TYPE only (``push_worker._attempt``):
        the cloud's ``emit_id``/``accepted_at`` are carried for the record and
        fall back to the body's own ``emit_id`` and the local clock when a 2xx
        answer omits them, because the accept is the status, not the fields.
        """
        if self._disarmed:
            return EmitRefused(NO_STATUS)
        route = EMIT_ROUTE.format(tunnel_id=quote(self.tunnel_id, safe=""))
        try:
            result = asyncio.run(
                self._call(
                    path=route.removeprefix(_TUNNELS_PREFIX),
                    body=dict(body),
                    idempotency_key=idempotency_key,
                )
            )
        except (LoginRequired, RecordNotFound) as failure:
            self._disarm(failure)
            return EmitRefused(404 if isinstance(failure, RecordNotFound) else NO_STATUS)
        except RequestFailed as failure:
            return EmitRefused(failure.status)
        except (httpx.HTTPError, ValueError, OSError):
            # Timeout, connection refused, an invalid envelope, a refresh that
            # could not reach the IdP: all one refusal (push_worker's
            # ``EmitRefused`` says why one shape serves every case).
            return EmitRefused(NO_STATUS)
        emit_id = result.get("emit_id") if isinstance(result, dict) else None
        accepted_at = result.get("accepted_at") if isinstance(result, dict) else None
        return EmitAccepted(
            emit_id=emit_id if isinstance(emit_id, str) else str(body.get("emit_id", "")),
            accepted_at=(
                accepted_at
                if isinstance(accepted_at, int) and not isinstance(accepted_at, bool)
                else int(time.time())
            ),
        )

    # -- the CredentialTransport -----------------------------------------------

    def credentials(self, block: dict[str, Any], *, carrier: str) -> None:
        """Send the block ONLY for the heartbeat carrier.

        ``CredentialReport._flush`` calls its transport for every carrier, but
        on ``register`` and ``emit`` the block RIDES that call's body (the
        worker's ``report_block`` and :meth:`register_forward` put it there), so
        sending it here too would double-report. Raising on failure is the
        contract: ``_flush`` leaves its bookkeeping untouched so the next
        heartbeat re-sends the same facts.
        """
        if carrier != CARRIER_HEARTBEAT:
            return
        if self._disarmed:
            raise PushTransportDisarmed("mobile push is disarmed until restart")
        try:
            asyncio.run(self._call(api_path=REPORT_HEARTBEAT_ROUTE, body=dict(block)))
        except (LoginRequired, RecordNotFound) as failure:
            self._disarm(failure)
            raise

    # -- the registration forward ----------------------------------------------

    def schedule_register_forward(
        self, device: Mapping[str, Any], devices: Sequence[Mapping[str, Any]]
    ) -> asyncio.Task[None] | None:
        """Start :meth:`register_forward` on the running loop; never awaited by the route.

        The app's register response must not wait on the cloud (SESSION-PUSH
        §4.5): a slow or absent cloud would otherwise turn every app launch into
        a 30 s spinner. Returns the task for tests; ``None`` when disarmed.
        """
        if self._disarmed:
            return None
        task = asyncio.get_running_loop().create_task(self.register_forward(device, devices))
        self._forwards.add(task)
        task.add_done_callback(self._forwards.discard)
        return task

    async def register_forward(
        self, device: Mapping[str, Any], devices: Sequence[Mapping[str, Any]]
    ) -> None:
        """POST ``/v1/push/register``: 3 attempts over ~2 min, then one log line.

        The SAME attempt budget as the emit (``push_worker.EMIT_RETRY_*``), so the
        two cloud calls fail on one clock. The token lives in ``device`` for the
        life of this coroutine and nowhere else; no log line here ever formats
        ``device``. A permanent refusal (4xx other than 429) stops early: the
        same body cannot succeed on a retry.
        """
        body = {
            "computer": self.tunnel_id,
            "device": dict(device),
            "devices": [dict(row) for row in devices],
        }
        status = NO_STATUS
        for attempt in range(1, EMIT_RETRY_ATTEMPTS + 1):
            if self._disarmed:
                return
            try:
                await self._call(api_path=REPORT_REGISTER_ROUTE, body=body)
                return
            except (LoginRequired, RecordNotFound) as failure:
                self._disarm(failure)
                return
            except RequestFailed as failure:
                status = failure.status
                if 400 <= status < 500 and status != 429:
                    break
            except (httpx.HTTPError, ValueError, OSError):
                status = NO_STATUS
            if attempt < EMIT_RETRY_ATTEMPTS:
                await self._sleep(self._retry_interval_s)
        logger.warning(
            "mobile push register forward for device %s gave up (last status %s)",
            device.get("device_id"),
            status,
        )


class PushLane:
    """The armed lane: worker, report and transport, plus the report's lock.

    ``CredentialReport`` is a single-threaded state machine, and three callers
    reach it from different threads — the worker's emit composition and the
    heartbeat (both on the scan's ``to_thread``) and the register route (on the
    event loop). One lock serialises them; every hold is a local registry read
    plus, for the heartbeat only, one bounded HTTP call.
    """

    def __init__(self, transport: RadientPushTransport, config_dir: Path) -> None:
        self.transport = transport
        self.lock = threading.Lock()
        self.report = CredentialReport(
            transport.credentials,
            loader=registry_loader(config_dir),
            computer=transport.computer,
        )
        from local_operator.session.attention import AttentionStore

        self.worker = PushWorker(
            store=AttentionStore(),
            transport=transport.emit,
            config_dir=config_dir,
            computer=transport.computer,
            report_block=self._emit_rows,
        )
        #: When the last heartbeat attempt FAILED, so the scan does not retry it
        #: every 2 s: ``heartbeat_due`` stays true after a failure (the report's
        #: bookkeeping is honest), and the retry spacing reuses the emit's
        #: interval rather than inventing a new one.
        self._heartbeat_failed_at: float | None = None

    def _emit_rows(self) -> list[dict[str, Any]]:
        # piggyback() returns None when nothing is due — the common case — and
        # the worker then sends the bare payload plus any sound-readiness rows.
        with self.lock:
            block = self.report.piggyback(CARRIER_EMIT)
        return block_devices(block) if block is not None else []

    def register_rows(self) -> list[dict[str, Any]]:
        """The report block for a register forward (``CARRIER_REGISTER``)."""
        with self.lock:
            block = self.report.piggyback(CARRIER_REGISTER)
        return block_devices(block) if block is not None else []

    def heartbeat(self, *, now: float | None = None) -> dict[str, Any] | None:
        """One heartbeat when due; failures are logged and spaced, never raised."""
        stamp = time.time() if now is None else now
        if self.transport.disarmed:
            return None
        if (
            self._heartbeat_failed_at is not None
            and stamp - self._heartbeat_failed_at < EMIT_RETRY_INTERVAL_S
        ):
            return None
        try:
            with self.lock:
                if not self.report.heartbeat_due(now=stamp):
                    return None
                sent = self.report.heartbeat(now=stamp)
        except (RequestFailed, LoginRequired, RecordNotFound, PushTransportDisarmed) as failure:
            self._heartbeat_failed_at = stamp
            logger.info(
                "mobile push heartbeat failed (status %s)", getattr(failure, "status", None)
            )
            return None
        except (httpx.HTTPError, ValueError, OSError) as failure:
            self._heartbeat_failed_at = stamp
            logger.info("mobile push heartbeat failed (%s)", type(failure).__name__)
            return None
        self._heartbeat_failed_at = None
        return sent


def build_push_lane(config_dir: Path) -> PushLane | None:
    """The lane for this machine, or ``None`` when it has no live tunnel.

    ``None`` is today's behaviour, deliberately (ADR §2.4): push rides the
    tunnel's Radient login, so a machine that is not signed in to Radient with a
    tunnel has nothing to authenticate with and no ``computer`` handle. A revoked
    tunnel deletes ``config.json`` (``lop tunnel revoke``), so an absent file is
    the revoked case too; a record still carrying ``revoked_at`` is refused for
    the same reason.
    """
    from local_operator.tunnels import config as tunnel_config

    try:
        value = tunnel_config.load()
    except (ValueError, OSError):
        return None
    tunnel_id = value.get("tunnel_id")
    credential_id = value.get("credential_id")
    record = value.get("record")
    if not isinstance(tunnel_id, str) or not tunnel_id:
        return None
    if not isinstance(credential_id, int) or isinstance(credential_id, bool):
        return None
    if isinstance(record, dict) and record.get("revoked_at"):
        return None
    return PushLane(
        RadientPushTransport(tunnel_id=tunnel_id, credential_id=credential_id), config_dir
    )
