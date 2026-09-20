"""One answer to "is remote access working here", shared by every surface.

`lop tunnel status`, the desktop route and (through the park file) the TUI all
describe the same machine, and the incident this module exists for was a status
command that answered the question from the CLOUD's cached view while the local
connector was dead. So the assembly lives here, once: these are the shapes a
caller renders or forwards, and a second implementation is how two surfaces come
to disagree about what "parked" means — with the one nobody was looking at
being the wrong one.

Two rules the code below keeps, both of them lessons from that incident:

* The connector's own state is read on THIS device. A park file outranks a
  health probe, because a parked connector is not running and the probe can
  only ever answer "there is no gateway here".
* A check that cannot run reports that it could not run. Neither a lost network
  nor an unreadable store is allowed to read as "your login is dead", which is
  what sent an offline machine to a login it did not need.
"""

from __future__ import annotations

from contextlib import closing
from typing import Any

import httpx

from local_operator.providers.auth_store import (
    AuthStore,
    AuthStoreError,
    CredentialInvalidError,
)
from local_operator.tunnels import config, gateway, state


async def probe(value: dict[str, Any]) -> dict[str, Any]:
    """Ask the local gateway what it is doing, for a connector with no park."""
    healthy = False
    connected = False
    served = False
    reason = ""
    detail = ""
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            reply = await client.get(
                f"http://127.0.0.1:{value['gateway_port']}/_lop_tunnel/health", timeout=2
            )
            served = reply.status_code == 200
            payload = reply.json() if served else {}
            if not isinstance(payload, dict):
                # A stale or foreign listener on this port can answer 200 with
                # any JSON at all. Anything but an object is not a health
                # payload, and a status surface must not raise over it.
                payload = {}
            healthy = served and payload.get("ok") is True
            connected = healthy and payload.get("connected") is True
            if not healthy:
                # The gateway names why it is refusing relayed requests, and
                # this is the surface where a command can be offered at all.
                # A reason this build does not know falls back to the relay's
                # own sentence rather than saying nothing.
                reason = str(payload.get("reason") or "")
                detail = gateway.terminal_detail(reason, str(payload.get("detail") or ""))
    except (httpx.HTTPError, ValueError):
        # A stopped connector and a gateway that is not there are also
        # different jobs: the first is this process, the second is the unit.
        detail = (
            "The local relay gateway is not answering on "
            f"127.0.0.1:{value['gateway_port']}; run lop tunnel install to restore it"
        )
    if connected:
        word = "connected"
    elif healthy:
        word = "connecting"
    elif served:
        # The gateway answered and is refusing to serve. That is not a stopped
        # connector — cloudflared may still hold the edge connection — and
        # "stopped" beside a sentence promising it clears itself would
        # contradict the payload this just read.
        word = "not serving"
    else:
        word = "stopped"
    return {"state": word, "reason": reason, "detail": detail, "since": None, "remedy": None}


async def connector_state(value: dict[str, Any], *, reachable: bool = True) -> dict[str, Any]:
    """The connector's real state, and why it is in it.

    The park file outranks the health probe, and it has to: a parked connector
    is not running, so the probe can only ever answer "there is no gateway
    here" — reporting that as the state is the lie this surface is being fixed
    for. A deliberate local stop outranks both: a tunnel the operator stopped is
    not waiting on anyone.

    ``reachable=False`` skips the loopback probe for callers that must not
    block on a socket they do not need — `GET /v1/auth/status` is polled beside
    an interactive login form, where a 2-second connect timeout would be felt.
    Those callers get the parked/stopped facts, which is the whole of what they
    are asking about, and ``unknown`` otherwise.
    """
    if value.get("stopped"):
        return {"state": "stopped", "reason": "", "detail": "", "since": None, "remedy": None}
    parked = state.parked()
    if parked is not None:
        return {
            "state": "parked",
            "reason": str(parked.get("reason") or ""),
            "detail": str(parked.get("detail") or ""),
            "since": parked.get("first_at"),
            "remedy": parked.get("remedy") if isinstance(parked.get("remedy"), dict) else None,
        }
    if not reachable:
        return {"state": "unknown", "reason": "", "detail": "", "since": None, "remedy": None}
    return await probe(value)


async def login_verdict(value: dict[str, Any]) -> dict[str, Any]:
    """Whether the login that owns this tunnel still works, decided locally.

    Decided from this device's own credential store rather than from the cloud
    read, because when the login is dead the cloud read is precisely what cannot
    answer. A refresh that could not REACH the token endpoint reports `unknown`
    rather than `login_required`: sending an offline machine to a login it does
    not need is the misdirection this surface exists to remove, and
    `CredentialInvalidError` is the one signal that separates the two.
    """
    selected = value.get("credential_id")
    dead: dict[str, Any] = {"credential_id": selected, "state": "login_required"}
    if not isinstance(selected, int) or isinstance(selected, bool):
        return dead
    with closing(AuthStore()) as store:
        row = store.get_credential(selected)
        if row is None or row.provider != "radient" or row.credential_type != "oauth":
            return dead
        try:
            await store.ensure_oauth_fresh_or_raise(selected)
        except CredentialInvalidError:
            return dead
        except AuthStoreError:
            # Reachable row, unusable answer: the network, not the login.
            return {"credential_id": selected, "state": "unknown"}
    return {"credential_id": selected, "state": "ok"}


def remedy(
    value: dict[str, Any], connector: dict[str, Any], login: dict[str, Any]
) -> dict[str, str] | None:
    """The one command that clears what was just found, when one exists.

    The connector's own park carries its remedy (only it knows whether the fix is
    a login, an install or a re-enrolment); nothing else has one to offer, and
    `null` says so rather than inventing a command.

    A deliberately stopped tunnel has NO remedy even when the credential behind
    it is dead: `stopped` means the operator is not using the tunnel, and
    handing them a sign-in command for remote access they turned off is the nag
    the design rules out. The state word cannot carry this — a gateway that is
    simply not answering also reports `stopped` — so the decision is taken from
    the configuration, which is where the operator's intent lives.
    """
    if value.get("stopped"):
        return None
    parked = connector.get("remedy")
    if isinstance(parked, dict) and parked.get("command"):
        return {
            "command": str(parked["command"]),
            "url": str(parked.get("url") or gateway.CONSOLE_URL),
        }
    if login.get("state") == "login_required":
        return {
            "command": gateway.TERMINAL_REMEDY[gateway.LOGIN_REQUIRED],
            "url": gateway.CONSOLE_URL,
        }
    return None


def payload(
    value: dict[str, Any],
    record: dict[str, Any],
    *,
    source: str,
    cloud_reason: str,
    connector: dict[str, Any],
    login: dict[str, Any],
) -> dict[str, Any]:
    """The machine-readable state: the shape every consumer reads.

    READ-ONLY BY CONSTRUCTION: there is no field here a caller could use to
    change anything, which is what lets the desktop route expose it without
    inventing an operation. ``cloud.source`` is the provenance of the tunnel
    record — `live` when the cloud answered this request, `cached` when the copy
    stored at the last `create`/`connect`/`configure` had to stand in — and it
    exists because reading `status: active` off a cached copy is exactly how a
    withdrawn tunnel looked healthy.
    """
    return {
        "tunnel_id": record.get("id"),
        "cloud": {
            "status": record.get("status", "configured"),
            "source": source,
            "reason": cloud_reason,
        },
        "connector": connector,
        "login": login,
        "remedy": remedy(value, connector, login),
    }


async def local_payload(*, reachable: bool = True) -> dict[str, Any]:
    """The payload for a surface with no cloud read of its own.

    ``cloud.source`` is honestly `cached`: this reports what is stored on this
    device plus what the connector is doing, which is the part a cloud read
    cannot answer when the connection or the login is what is broken.
    """
    try:
        value = config.load()
    except ValueError:
        return {
            "configured": False,
            "tunnel_id": None,
            "cloud": {"status": "not configured", "source": "cached", "reason": ""},
            # Not "stopped": nothing was stopped here, and a surface that said so
            # would have an operator looking for a connector this machine has
            # never enrolled.
            "connector": {
                "state": "not configured",
                "reason": "",
                "detail": "",
                "since": None,
                "remedy": None,
            },
            "login": {"credential_id": None, "state": "unknown"},
            "remedy": None,
        }
    report = payload(
        value,
        value.get("record") if isinstance(value.get("record"), dict) else {},
        source="cached",
        cloud_reason="",
        connector=await connector_state(value, reachable=reachable),
        login=await login_verdict(value),
    )
    report["configured"] = True
    return report
