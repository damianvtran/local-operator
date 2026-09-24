"""Credential brokering across the mesh (mesh-credentials.md; build plan §2).

The package's public surface, and the P0 seam it fills in: :func:`install`
registers ``net_broker`` SLOW — a grant can sit through a provider refresh
(bounded at ``providers.auth_store.PROVIDER_REFRESH_TOTAL_BUDGET_S``) — plus the
three LEG-1 control ops, whose names P0 declared in ``types.LOCAL_OPS``.

``broker_credential`` stays ADMIN-ONLY: it is in no role but ``admin``
(``types.ROLE_CAPABILITIES``), and only an admin device may grant it to a peer's
local row (``relay.set_member_capabilities``). ``credential share`` therefore needs
an admin on THIS device, and the broker checks the holder list on top of it.

Stdlib only at import: the relay imports this package at construction, and nothing
on that path may pull the provider auth store. ``owner`` and ``client`` are light
(they hold plain data and reach for the store lazily, inside the methods that need
it); ``store`` is the heavy one — it imports ``providers.auth_store`` for
``OAuthAccess``/``StoredCredential`` — and is imported ONLY from the session
construction path via :func:`build_auth_store`.

Two facts about a refusal worth keeping beside the code that raises one:

* **A refusal is not a transport failure.** It travels inside an ``ack``'s
  ``detail`` with a machine ``code`` (§3.2), because the transport's own codes
  (``not_authorised``, ``not_a_member``, ``unknown_op``) share one namespace with
  the broker's and mean something else.
* **A refused borrower makes ZERO token POSTs.** It holds no refresh token, so
  there is nothing here to POST with; the owner's stored grant stops working at its
  own expiry and the existing no-credential path takes over (§2.4).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from local_operator.network.relay import RelayServer

#: ``network.credentials.grant_ttl_s``: the longest a lent access token lives on a
#: borrower before it must ask the owner again. The grant also never outlives the
#: token's own expiry (``min(token_exp, now + ttl)``). 15 minutes bounds how long a
#: revoked borrower keeps working and how long a borrower rides out an offline
#: owner (build plan §2.4); the Mac↔EC2 grant latency measured in rollout settles
#: whether it moves (design Q3).
GRANT_TTL_S = 900.0

#: The owner-side deadline for one ``net_broker`` request (seconds): the provider
#: refresh's own total budget (``providers.auth_store.PROVIDER_REFRESH_TOTAL_BUDGET_S``,
#: 60 s) plus a margin for the lease wait and the reply. Restated rather than
#: imported because that module is far too heavy for the relay's construction
#: path; ``tests/unit/network/test_slow_ops.py`` pins that it stays above it.
BROKER_OP_DEADLINE_S = 75.0


def grant_ttl_s(root: Path | None = None) -> float:
    """``network.credentials.grant_ttl_s``, through the package's ONE config reader."""
    from local_operator.network import store

    return float(store.read_config(("network", "credentials", "grant_ttl_s"), GRANT_TTL_S, root))


def build_auth_store(config_dir: Path | None = None) -> Any:
    """The session's credential store: plain, or mesh-aware when this device borrows.

    Imported lazily through this module so ``network.credentials.store`` — and with
    it ``providers.auth_store`` — is reached only from the session construction
    path, never from the relay's.
    """
    from local_operator.network.credentials.store import build_auth_store as _build

    return _build(config_dir)


class _LocalOps:
    """LEG 1: the control-socket ops a runtime uses to ask its own relay.

    One instance per relay, holding the relay-side client (``None`` on a device with
    no placement, in which case every op refuses in operator language rather than
    with the generic unknown-op sentence — the slice IS in this build, so telling
    the operator ``mesh-credentials.md owns that slice`` would be wrong).
    """

    def __init__(self, client: Any = None) -> None:
        self._client = client

    def grant(self, frame: dict[str, Any]) -> dict[str, Any]:
        key = str(frame.get("credential_key") or "")
        provider = str(frame.get("provider") or key)
        if self._client is None:
            return _detail("not_a_holder", key, _no_placement_sentence(provider))
        return self._client.serve_grant(
            key,
            provider=provider,
            session_id=str(frame.get("session_id") or ""),
            model_id=str(frame.get("model_id") or ""),
            force_refresh=bool(frame.get("force_refresh")),
        )

    def report(self, frame: dict[str, Any]) -> dict[str, Any]:
        key = str(frame.get("credential_key") or "")
        if self._client is None:
            return _detail("not_a_holder", key, _no_placement_sentence(key))
        return self._client.serve_report(
            key,
            kind=str(frame.get("kind") or "noted"),
            session_id=str(frame.get("session_id") or ""),
            model_id=str(frame.get("model_id") or ""),
            for_session=str(frame.get("session_id") or ""),
            retry_after_ms=int(frame.get("retry_after_ms") or 0),
        )

    def placement(self, frame: dict[str, Any]) -> dict[str, Any]:
        """Pull the placement document from each owner this device knows of.

        The PULL half of the sync, and the reason a ``credential share`` reaches a
        running peer without a proactive fan-out the relay would have to own: one
        bounded round trip, issued by an explicit act rather than on every provider
        call.
        """
        if self._client is None:
            return {"kind": "ack", "key": "", "changed": [], "owners": 0}
        return self._client.pull_placement()


def _detail(code: str, key: str, message: str) -> dict[str, Any]:
    return {"kind": "error", "code": code, "key": key, "message": message}


def _no_placement_sentence(provider: str) -> str:
    return (
        f"this device has no shared credential for {provider!r}: nothing here is lent to "
        "this device and nothing here lends it. Ask the device that signed in to it to "
        "share it, or sign in here with 'lop login'."
    )


def install(server: RelayServer) -> None:
    """Register this slice's ops on ``server``.

    ``net_broker`` is registered SLOW unconditionally, as P0 did, so the deadline and
    the off-reader dispatch are the same whether or not this device lends anything —
    a relay that owns nothing keeps answering with the by-name refusal it answered
    with before this slice existed, and its peer learns that in one frame rather than
    after a reader stalls.

    When this device DOES own something to lend, ``MeshCredentialBroker.for_relay``
    builds the broker and the real handler replaces the refusal.
    """
    from local_operator.network.credentials import client as client_mod
    from local_operator.network.credentials import owner as owner_mod
    from local_operator.network.relay import not_implemented_peer_op

    peer_client = client_mod.MeshCredentialClient.for_relay(server)
    broker = owner_mod.MeshCredentialBroker.for_relay(server)
    handler = broker.on_broker if broker is not None else not_implemented_peer_op("net_broker")
    ops = _LocalOps(peer_client)
    server.register_ops(
        {"net_broker": handler},
        local_handlers={
            "credential_grant": ops.grant,
            "credential_report": ops.report,
            "credential_placement": ops.placement,
        },
        slow={"net_broker": BROKER_OP_DEADLINE_S},
    )
