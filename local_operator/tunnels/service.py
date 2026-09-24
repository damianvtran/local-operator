"""Supervise the loopback gateway and one cloudflared connector as one unit.

A terminal session never owns their lifetime. launchd/systemd supervises this
process; stopping it withdraws remote access without terminating agent sessions.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import os
import shutil
import signal
import socket
import subprocess
from collections.abc import Iterator
from typing import Any, Literal, NamedTuple
from urllib.parse import quote

import httpx

from local_operator.mobile.auth import load_password
from local_operator.procstate import install_loop_signal_handlers
from local_operator.providers.auth_store import (
    CredentialInvalidError,
    RefreshUnconfirmedError,
)
from local_operator.tunnels import config, state
from local_operator.tunnels.api import RadientTunnels
from local_operator.tunnels.errors import (
    LocalPrerequisite,
    LoginRequired,
    ReenrolmentRequired,
)
from local_operator.tunnels.gateway import (
    AUTHORIZATION_DEFERRED,
    LOCAL_PREREQUISITE,
    LOGIN_REQUIRED,
    REENROLMENT_REQUIRED,
    REFUSED,
    TERMINAL_DETAIL,
    UNREACHABLE,
    Gateway,
)

POLL_SECONDS = 10

#: The kinds a supervisor must never retry, because nothing the process can do
#: changes them: each one needs an act by the operator.
PARKING_KINDS = frozenset({"terminal_login", "terminal_remote", "terminal_config"})


class Failure(NamedTuple):
    """What a failure means for the supervisor, not what it said.

    ``reason`` and ``detail`` come from `gateway`'s vocabulary wherever it has
    an entry, because the same cause reaches a phone's 503, `lop tunnel
    status` and this service log, and one cause must not become three wordings.

    Four kinds, not the five the design sketched: its ``crash`` and this
    module's ``transient`` default are the same outcome (retry, exit 1), and a
    literal member nothing can return is a claim the code does not make.
    """

    kind: Literal["transient", "terminal_login", "terminal_remote", "terminal_config"]
    reason: str
    detail: str


def _announce(message: str) -> None:
    """One timestamped, self-describing line: this log's only shape.

    launchd and systemd capture this process's stdout verbatim, so these lines
    are read beside every other unit's and by an operator who was not watching
    when they were written. Until this existed they carried neither a time nor
    which process authored them: the incident's 870 restarts left 870 identical
    sentences with no way to tell a dead grant from a first attempt.

    Local time with an explicit offset, not UTC: the question a line answers is
    "when did this happen on this machine".
    """
    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"{stamp} local_operator.tunnels.service: {message}", flush=True)


def cloudflared_binary(configured: str | None = None) -> str:
    binary = configured or shutil.which("cloudflared")
    if not binary:
        # LocalPrerequisite, not ValueError: the supervisor must stop retrying
        # (see `classify_failure`), and the remedy is a local install that
        # re-arms the connector.
        raise LocalPrerequisite(
            "Install cloudflared 2025.4.0 or newer, then run lop tunnel install."
        )
    result = subprocess.run(
        [binary, "tunnel", "run", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode or "--token-file" not in result.stdout:
        raise LocalPrerequisite(
            "Update cloudflared to 2025.4.0 or newer (token-file support required)."
        )
    return binary


def tunnel_path(value: dict[str, Any]) -> str:
    identifier = value.get("tunnel_id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("Missing configured tunnel ID.")
    return "/" + quote(identifier, safe="")


def pinned_harness_ports(value: dict[str, Any]) -> tuple[dict[str, int] | None, str]:
    """The harness ports the operator last approved from this device.

    `lop tunnel create/connect/configure` persist the whole cloud record
    locally, so `record.harnesses` is the device's own copy of the port map an
    operator ran a local command to accept. Returns `(None, reason)` — meaning
    "no usable pin" — for a record written before the field existed or
    hand-edited into a shape we cannot trust, so callers can fall back rather
    than strand a device whose config predates this check.

    One unusable entry disables the pin for the whole record, so the reason
    names that entry: without it an operator reading the fallback warning knows
    the pin is off but not which line of config.json to repair.
    """
    record = value.get("record")
    harnesses = record.get("harnesses") if isinstance(record, dict) else None
    if not isinstance(harnesses, list) or not harnesses:
        return None, "this tunnel configuration predates it"
    pinned: dict[str, int] = {}
    for position, harness in enumerate(harnesses, start=1):
        if not isinstance(harness, dict) or not isinstance(harness.get("id"), str):
            return None, f"stored harness entry {position} has no usable id"
        if harness["id"] in pinned:
            # The same silent last-wins ambiguity the JWKS loader refuses on a
            # repeated kid (gateway.OriginVerifier): two entries claiming one
            # id leave the approved port decided by serialization order rather
            # than by the operator, and this file is hand-editable. Fall back
            # and name it instead of picking a winner. Falling back rather than
            # raising keeps the untrustworthy-shape contract above: a bad local
            # record must never strand a device that the cloud still serves.
            return None, f"stored harness {harness['id']} is listed twice"
        try:
            pinned[harness["id"]] = config.port(harness.get("port"))
        except ValueError:
            return None, f"stored harness {harness['id']} has an unusable port"
    return pinned, ""


def enforce_harness_ports(connection: dict[str, Any], value: dict[str, Any]) -> None:
    """Refuse a /connect that repoints a harness at an unapproved local port.

    The cloud validates a harness port only as 1024-65535 and != gateway_port,
    and replaces the harness list wholesale on PATCH, so anyone holding the
    owner's console session can aim a harness at any other loopback service.
    The gateway then attaches this device's mobile-relay cookie or OpenCode
    Basic credential to whatever answers there (gateway.headers), handing a
    live relay credential to an unrelated local service. Pinning against the
    locally stored record makes changing a harness port require an act on this
    machine, exactly as the cloud already treats gateway_port.

    Fail-closed but recoverable: the console PATCH that legitimately changes a
    port bumps the tunnel version, the poller restarts, and this refusal then
    names the local command that re-approves it. Refreshing the pin is left to
    those commands (they are what write `record`); re-pinning from the polled
    record here would restore exactly the silent repointing this prevents.
    """
    pinned, unpinnable = pinned_harness_ports(value)
    if pinned is None:
        print(
            f"Warning: harness port pinning is inactive ({unpinnable}), so console port "
            "changes are not verified locally. Run lop tunnel connect to pin them.",
            flush=True,
        )
        return
    for harness in connection["tunnel"]["harnesses"]:
        # A disabled harness is never dialed (Gateway.harness filters on it),
        # so its port cannot carry a credential and is not worth an outage.
        if not harness["enabled"]:
            continue
        approved = pinned.get(harness["id"])
        if approved is None:
            # A harness the console ADDED since the last local command. The
            # remedy is the same, but reporting it as a port change describes
            # an event that did not happen and sends the operator hunting for
            # a port they never set.
            raise ReenrolmentRequired(
                f"Harness {harness['id']} is not approved on this device. "
                "Run lop tunnel connect again."
            )
        if approved != harness["port"]:
            raise ReenrolmentRequired(
                f"Harness port for {harness['id']} changed in the console. "
                "Run lop tunnel connect again."
            )


def _failure_chain(failure: BaseException) -> Iterator[BaseException]:
    """The failure and everything it was chained from, each visited once.

    Both links are followed, deliberately, including a context Python would
    otherwise suppress: the questions asked of the chain (did we reach the
    control plane at all? did the store judge the grant dead?) are about what
    happened, not about which link the raise site chose to publish. The walk is
    depth-first and the order does not matter, because every question is a
    boolean; the `seen` set is what makes a self-referential chain terminate.
    """
    seen: set[int] = set()
    pending: list[BaseException] = [failure]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend(link for link in (current.__cause__, current.__context__) if link)


def _could_not_reach_control_plane(failure: BaseException) -> bool:
    """True when the failure, or anything it was chained from, is a transport error.

    Reading the chain is load-bearing, not defensive: `RadientTunnels.request`
    raises httpx errors directly for its own request, but a failure to reach the
    *token endpoint* while refreshing the login arrives wrapped — AuthStore wraps
    the refresh exception in an `AuthStoreError` with the transport error on
    `__cause__`. A long outage reaches that path by definition, because the
    access token always falls inside its refresh skew eventually, and flattening
    it to "the login expired" is exactly the misdirection this classification
    exists to remove.
    """
    return any(isinstance(link, httpx.TransportError) for link in _failure_chain(failure))


def _is_a_dead_login(failure: BaseException) -> bool:
    """Whether the credential store judged the tunnel's grant unusable.

    Two signals, and the chain is what unifies them. `LoginRequired` is what the
    tunnel client raises for the login cases it can name without echoing a
    provider body; `CredentialInvalidError` is the store's own verdict, the same
    one `/usage` and the routing cascade read. Reading the chain rather than the
    outermost type is what keeps this correct if either wrapper changes: the
    verdict belongs to the store, and the client is only its reporter.
    """
    return any(
        isinstance(link, (LoginRequired, CredentialInvalidError))
        for link in _failure_chain(failure)
    )


def _is_a_deferred_refresh(failure: BaseException) -> bool:
    """Whether the credential store DEFERRED the refresh instead of failing it.

    Read off the chain, like `_is_a_dead_login` and for the same reason: the
    store's verdict belongs to the store, and the client is only its reporter.
    `RadientTunnels.request` raises one `ValueError` for every refresh failure it
    cannot name itself, with the store's exception as its cause, so the
    distinction survives only in the chain.

    The state is TRANSIENT and the connector must keep retrying it: the store is
    holding the refresh off because the token's last exchange is unsettled, which
    clears by itself (120 s at the longest), and parking the connector for it
    would withdraw remote access until a person acted on a state that needed
    nobody. It is still an `AuthStoreError`, so without this arm it would reach
    `authorization_failure_reason`'s `REFUSED` fallback and every surface would
    blame the login — which is exactly the misreport the phone read as "the
    authentication expired".
    """
    return any(isinstance(link, RefreshUnconfirmedError) for link in _failure_chain(failure))


def authorization_failure_reason(failure: BaseException) -> str:
    """Name why the poller could not renew the relay authorization lease.

    Only the poller sees the control plane's answer, and swallowing it is what
    made every cause indistinguishable: the gateway serves one flat 503 once the
    lease lapses, so a computer that had merely lost its network looked exactly
    like a revoked tunnel or a lapsed plan and sent the operator to re-enroll a
    tunnel that was healthy.

    So the question is only ever *did we reach Radient?*. `RadientTunnels.request`
    raises httpx errors for transport trouble and a ValueError for everything the
    control plane answered — a >=400 status, an unusable login, a malformed
    envelope — and AuthStore, per `_could_not_reach_control_plane`, preserves the
    transport error it hit while refreshing. The wording each reason produces
    lives with the gateway, so one cause cannot drift into two sentences.
    """
    if _could_not_reach_control_plane(failure):
        return UNREACHABLE
    if isinstance(failure, httpx.HTTPError):
        # An httpx failure that is not a transport error still means no answer
        # came back from Radient.
        return UNREACHABLE
    if _is_a_deferred_refresh(failure):
        # The store answered; it simply will not present the token yet. AFTER the
        # transport arms, because a failure that never reached Radient is a network
        # fault whatever it is chained to, and BEFORE the `REFUSED` fallback, which
        # would blame the login for a wait.
        return AUTHORIZATION_DEFERRED
    return REFUSED


def classify_failure(failure: BaseException) -> Failure:
    """Decide whether the supervisor should retry this failure, or stop.

    `run()` used to end every cold-start failure the same way: one sentence, exit
    1, and a supervisor that restarts the unit forever (launchd
    `KeepAlive{SuccessfulExit:false}` + `ThrottleInterval 10`). That is correct
    for a fault that clears itself and catastrophic for one that cannot: the
    incident this exists for logged 870 identical sentences over a dead login,
    and the phone was unusable throughout.

    So each failure is placed on one side of a single question — *can a retry
    change the outcome?* — and an unclear answer stays on the retrying side. That
    default is deliberate and asymmetric: parking WITHDRAWS remote access until a
    person acts, so only a failure this function can name earns it, and a bug or a
    cloud-side change we do not recognise keeps today's behaviour.

    ``transient`` — a transport fault, or anything unclassified. Unchanged: exit
    1, and the supervisor's 10-second floor is the documented backoff.

    ``terminal_login`` — the store judged the grant dead (a token-endpoint
    refusal, prose or code, or a row that no longer holds a usable credential).
    Parked, and re-armed by the login that fixes it.

    ``terminal_remote`` — the cloud's answer invalidates this device's enrolment
    (a console harness or gateway-port change): the same local command that
    renews the record is what clears the park.

    ``terminal_config`` — a local prerequisite is missing. Parked; installing it
    re-arms the connector.

    Anything else — including a failure this module did not author — is
    ``transient``: `main` deliberately keeps propagating exceptions outside the
    two families it catches (fail closed, with a traceback) rather than parking
    on a shape it cannot describe.
    """
    if _could_not_reach_control_plane(failure):
        return Failure("transient", UNREACHABLE, TERMINAL_DETAIL[UNREACHABLE])
    if _is_a_dead_login(failure):
        return Failure("terminal_login", LOGIN_REQUIRED, TERMINAL_DETAIL[LOGIN_REQUIRED])
    if _is_a_deferred_refresh(failure):
        # TRANSIENT, deliberately: the store is waiting out an exchange whose
        # outcome is unsettled and the state clears by itself, so a park here would
        # withdraw remote access until a person acted — against this function's own
        # asymmetry. It is named all the same, because "transient" without a cause
        # is the flat refusal that sent a working machine to a sign-in page.
        return Failure("transient", AUTHORIZATION_DEFERRED, TERMINAL_DETAIL[AUTHORIZATION_DEFERRED])
    if isinstance(failure, LocalPrerequisite):
        # The sentence is the failure's own, and that is safe: every ValueError
        # this package raises is a fixed module literal (see `main`), and these
        # two kinds describe local state rather than anything a provider said.
        return Failure("terminal_config", LOCAL_PREREQUISITE, str(failure))
    if isinstance(failure, ReenrolmentRequired):
        return Failure("terminal_remote", REENROLMENT_REQUIRED, str(failure))
    # Everything else keeps today's behaviour, including a tunnel the cloud says
    # is suspended or disabled: that one RESUMES by itself at the 10-second floor
    # once the operator tops up credit or reactivates it (see the poller below),
    # so parking it would delete a self-heal rather than remove a retry loop.
    return Failure("transient", REFUSED, TERMINAL_DETAIL[REFUSED])


def _park(verdict: Failure) -> None:
    """Record the park and say so once, on a transition.

    Exit 0, which is this module's documented "the supervisor must not retry"
    idiom (`run`'s stopped branch) and the only one that works on both launchd
    and systemd: neither can exempt a code under a successful-exit rule.
    """
    announce = state.mark_parked(
        reason=verdict.reason, detail=verdict.detail, credential_id=_parked_credential_id()
    )
    if not announce:
        # Rate-limited while unchanged, and the limit lives in the state file
        # because the process is what a restart loop throws away. Silence here is
        # the point: a second identical park is not news, and 870 repetitions of
        # one sentence is what made the incident's log unreadable.
        return
    record = state.parked() or {}
    remedy = record.get("remedy", {}).get("command", "")
    _announce(
        f"connector parked reason={verdict.reason} attempts={record.get('attempts', 1)}"
        f"{f' — run {remedy}' if remedy else ''}: {verdict.detail}"
    )


def _parked_credential_id() -> int | None:
    """The credential this device's tunnel owns, or None if that is unreadable.

    Read here rather than threaded through `run` because the park is written by
    `main`, which sees the failure and not the configuration, and because a
    tunnel whose config cannot be read is exactly one of the failures that must
    still record a reason.
    """
    try:
        value = config.load()
    except (OSError, ValueError):
        return None
    identifier = value.get("credential_id")
    return identifier if isinstance(identifier, int) and not isinstance(identifier, bool) else None


def _withdraw_park(*, because: str) -> None:
    """Withdraw a park, saying once why it no longer describes this connector.

    Every surface reads the state file as the truth about a process none of them
    can see, so a park that outlived its condition would have the terminal, `lop
    tunnel status` and the desktop route all describing a connector that is
    running without it. Both callers pass the fact that made it false: the
    connector is serving, or the connector is retrying (a transient fault means
    it is NOT waiting for a person, whatever it was parked for before).
    """
    previous = state.parked()
    state.clear()
    if previous is not None:
        _announce(f"park withdrawn ({because}); was reason={previous.get('reason')}")


def active(record: Any) -> bool:
    return (
        isinstance(record, dict)
        and record.get("enabled") is True
        and record.get("status") == "active"
        and (
            "billing" not in record
            or (isinstance(record["billing"], dict) and record["billing"].get("eligible") is True)
        )
    )


async def run() -> int:
    import uvicorn

    try:
        value = config.load()
    except ValueError as failure:
        # No tunnel configured on this device, or a config.json nothing can
        # parse. No retry produces either, and every remedy (`lop tunnel create`,
        # repairing the file) runs through `lop tunnel install`, which re-arms a
        # parked connector.
        raise LocalPrerequisite(str(failure)) from failure
    if value.get("stopped"):
        # A user-level service may run again at the next login. A persisted
        # explicit stop is successful, so supervisors must not keep retrying it.
        #
        # A park is withdrawn here too: a connector the operator stopped is not
        # "waiting for a person" behind that stop, and a stale park would have
        # the TUI and `lop tunnel status` describe a tunnel this file says is not
        # in use at all.
        state.clear()
        return 0
    binary = cloudflared_binary(value.get("cloudflared_path"))
    # Bind before contacting the cloud or starting cloudflared: a conflicting
    # listener must never receive an enabled public route even momentarily.
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        listener.bind(("127.0.0.1", config.port(value["gateway_port"])))
        listener.listen(128)
        listener.setblocking(False)
    except BaseException:
        listener.close()
        raise
    token_file = config.directory() / "cloudflared.token"
    ready_file = config.directory() / "cloudflared.pid"
    child: asyncio.subprocess.Process | None = None
    scanner: asyncio.Task[None] | None = None
    server: Any = None
    serve_task: asyncio.Task[Any] | None = None
    gateway: Gateway | None = None
    restart = False
    try:
        # Proxy configuration and environment credentials are not forwarded to
        # either Radient or loopback harnesses. Both use explicit destinations.
        async with httpx.AsyncClient(
            trust_env=False, timeout=httpx.Timeout(15, read=None)
        ) as client:
            api = RadientTunnels(int(value["credential_id"]), client)
            connection = config.validate_connection(
                await api.request("POST", tunnel_path(value) + "/connect")
            )
            if not active(connection["tunnel"]):
                # Deliberately NOT parked. A suspended or disabled tunnel resumes
                # by itself — the operator tops up credit or reactivates it in the
                # console and the next attempt (the supervisor's 10-second floor)
                # picks it up, which is what this same branch documents in the
                # poller below. Parking would trade retry noise for a device that
                # does not come back until someone runs a local command, and no
                # local command is the remedy here.
                raise ValueError(
                    "Tunnel disabled or billing suspended. Review it in the Radient console."
                )
            if connection["gateway_port"] != value["gateway_port"]:
                # ReenrolmentRequired: retrying re-sends a /connect the cloud will
                # keep answering the same way, and the remedy (lop tunnel connect)
                # re-arms a parked connector.
                raise ReenrolmentRequired(
                    "Gateway port changed in the console. Run lop tunnel connect again."
                )
            enforce_harness_ports(connection, value)
            needs_mobile = any(
                h["enabled"] and h["id"] == "local-operator"
                for h in connection["tunnel"]["harnesses"]
            )
            password = await asyncio.to_thread(load_password) if needs_mobile else None
            if needs_mobile and password is None:
                password = value.get("mobile_password")
            if needs_mobile and not password:
                raise LocalPrerequisite("Install the mobile relay with lop mobile install first.")
            gateway = Gateway(
                connection,
                client,
                mobile_password=password,
                opencode_basic=value.get("opencode_basic"),
                connector_ready=lambda: (
                    ready_file.exists() and child is not None and child.returncode is None
                ),
            )
            token = connection.get("cloudflared_token")
            if not isinstance(token, str) or not token or "\n" in token:
                raise ValueError("Radient did not return a connector token.")
            config.private_write(token_file, token)
            ready_file.unlink(missing_ok=True)
            # No credentials in argv, inherited environment, or logs. A
            # per-tunnel token authorizes only its connector, never Cloudflare
            # account administration. cloudflared debug logging is forbidden.
            child_env = {
                k: v
                for k, v in os.environ.items()
                if k in {"PATH", "HOME", "SYSTEMROOT", "SSL_CERT_FILE", "SSL_CERT_DIR"}
            }
            child = await asyncio.create_subprocess_exec(
                binary,
                "tunnel",
                "--no-autoupdate",
                "--loglevel",
                "error",
                "--grace-period",
                "1s",
                "--pidfile",
                str(ready_file),
                "run",
                "--token-file",
                str(token_file),
                env=child_env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            server = uvicorn.Server(
                uvicorn.Config(
                    gateway.app(),
                    host="127.0.0.1",
                    port=value["gateway_port"],
                    log_level="warning",
                    access_log=False,
                    proxy_headers=False,
                    timeout_graceful_shutdown=2,
                )
            )
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            # ONE helper for every server boot path: `add_signal_handler` is
            # Unix-only and the Windows Proactor loop's inherited stub raises
            # NotImplementedError, so this block existed here, in the mobile
            # relay and in the session runtime — three copies of one platform
            # fact, two of which were missing the guard. See
            # procstate.install_loop_signal_handlers.
            install_loop_signal_handlers(loop, {signal.SIGTERM: stop.set, signal.SIGINT: stop.set})

            async def poll() -> None:
                nonlocal restart
                assert gateway is not None
                while not stop.is_set():
                    # Read outside the try: this is this device's own file, not
                    # Radient's answer, and an unreadable one must not be
                    # recorded as a refusal. It ends the poll task, which the
                    # supervisor treats as a restartable crash below.
                    if config.load().get("stopped"):
                        gateway.revoked = True
                        stop.set()
                        return
                    try:
                        record = await api.request("GET", tunnel_path(value))
                    except (ValueError, httpx.HTTPError) as failure:
                        # The gateway's short authorization lease closes even
                        # when the control-plane network is unavailable, so the
                        # failure is recorded rather than discarded: the phone's
                        # 503 and `lop tunnel status` both report which cause it
                        # was. authorize() clears it on the next success.
                        gateway.note_authorization_failure(authorization_failure_reason(failure))
                    else:
                        # Configuration changes invalidate all signed requests
                        # from the old version. Restart for a new proof context,
                        # without ever guessing how to merge trust boundaries.
                        if (
                            not active(record)
                            or record.get("version") != connection["tunnel"]["version"]
                        ):
                            # Resume after console reactivation or a credit
                            # top-up. User services retry at their 10s floor;
                            # /connect refuses publication while suspended.
                            #
                            # Deliberately still a restart rather than a park, for
                            # both conditions: a suspended tunnel clears at that
                            # floor once the operator tops up, and a console change
                            # makes the next /connect mint a fresh proof context. The
                            # cold-start re-enrolment case is the one that cannot,
                            # and that is where the park belongs.
                            restart = True
                            gateway.revoked = True
                            stop.set()
                            return
                        gateway.authorize()
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=POLL_SECONDS)
                    except TimeoutError:
                        pass

            scanner = asyncio.create_task(poll())
            serve_task = asyncio.create_task(server.serve(sockets=[listener]))
            # Serving, so the park is over: this is the only point at which the
            # connector is genuinely back, and every surface reads the state file
            # as the truth about a process none of them can see.
            _withdraw_park(because="the connector is serving")
            stopping = asyncio.create_task(stop.wait())
            exited = asyncio.create_task(child.wait())
            try:
                # `scanner` belongs in this set: a poller that dies of a failure
                # this loop does not classify must stop the unit rather than
                # leave a gateway serving with a lapsed lease and nothing left to
                # renew it. Fail closed, then restart.
                await asyncio.wait(
                    {scanner, serve_task, stopping, exited},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            finally:
                stopping.cancel()
                exited.cancel()
                await asyncio.gather(stopping, exited, return_exceptions=True)
                # Consume the poller's own failure when that is what ended the
                # unit, and bound it: teardown must not wait on a poller parked
                # inside a control-plane request.
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(
                        asyncio.gather(scanner, return_exceptions=True), timeout=5
                    )
            # Crashes, a dead poller and remote suspension are restartable. Only
            # an explicit local stop exits successfully and stays stopped under
            # supervision.
            return 1 if restart or not stop.is_set() else 0
    finally:
        if gateway is not None:
            gateway.revoked = True
        if scanner is not None:
            scanner.cancel()
            await asyncio.gather(scanner, return_exceptions=True)
        if server is not None:
            server.should_exit = True
        if serve_task is not None:
            await serve_task
        if child is not None and child.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                child.terminate()
            try:
                await asyncio.wait_for(child.wait(), timeout=5)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    child.kill()
                await child.wait()
        listener.close()
        token_file.unlink(missing_ok=True)
        ready_file.unlink(missing_ok=True)


def main() -> int:
    try:
        return asyncio.run(run())
    except ValueError as failure:
        verdict = classify_failure(failure)
        # Service logs contain operational state, never upstream bodies,
        # request URLs, or a traceback containing credential arguments, and
        # printing a ValueError's text keeps that property: every ValueError
        # this package raises is a fixed literal carrying at most a harness id,
        # a tunnel status, or an HTTP status code (api.py deliberately refuses
        # to echo provider bodies). The only ValueErrors run() does not author
        # are json.JSONDecodeError on a corrupt config.json or /connect body,
        # whose message is a byte offset and never the document, and int() on a
        # hand-edited credential_id, which is a local row id and not a secret.
        # This is the only place the remedy is written down: suppressing it is
        # what left the harness-port refusal telling the operator to check a
        # Radient login that is fine, with nothing naming lop tunnel connect.
        if verdict.kind in PARKING_KINDS:
            # Exit 0 IS the report: this module's contract is that launchd and
            # systemd supervise it, and a successful exit is the one idiom both
            # honour as "do not restart me". Parking is therefore how the crash
            # loop ends -- not a longer sleep, not a distinct exit code, both of
            # which were evaluated and neither of which can exempt a code under
            # launchd's `KeepAlive{SuccessfulExit:false}`.
            _park(verdict)
            return 0
        # RETRYING, which is not parked. A park says "stopped, waiting for a
        # person"; exiting 1 says the supervisor should try again in 10 seconds,
        # and a stale park left beside a retrying connector would have the
        # terminal nagging about a login for a machine already trying on its own.
        _withdraw_park(because="the connector is retrying")
        _announce(f"connector stopped reason={verdict.reason}: {failure}")
        return 1
    except (OSError, httpx.HTTPError) as failure:
        # These carry text this module did not author: httpx echoes the full
        # request URL (query string included) and OSError echoes filesystem
        # paths, so only the generic line is safe here. The reason code is this
        # module's own vocabulary and stays: without it an outage and a bug read
        # as the same line, which is the property this log was rebuilt for.
        _withdraw_park(because="the connector is retrying")
        _announce(
            f"connector stopped reason={classify_failure(failure).reason}; "
            "check lop tunnel status and your Radient login."
        )
        return 1


if __name__ == "__main__":
    # Linux comm axis: this service's unit names the IMAGE on macOS, and where
    # there is no such image the process names itself (see
    # :func:`procname.brand_this_process`; a no-op on macOS).
    from local_operator import procname

    procname.brand_this_process()
    raise SystemExit(main())
