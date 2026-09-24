"""Credential brokering across the mesh (mesh-credentials.md; build plan §2).

THIS IS THE P0 SEAM. The credentials slice (C) adds the modules beside this one
(``types``, ``placement``, ``owner``, ``client``, ``store``, ``messages``). Fixed
here: the grant-lifetime setting and its default, and :func:`install`, which
registers ``net_broker`` SLOW — a grant can sit through a provider refresh — with
the by-name refusal until the slice lands.

``broker_credential`` stays ADMIN-ONLY: it is in no role but ``admin``
(``types.ROLE_CAPABILITIES``), and only an admin device may grant it to a peer's
local row (``relay.set_member_capabilities``).

Stdlib only at import: the relay imports this package at construction, and
nothing on that path may pull the provider auth store.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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


def install(server: RelayServer) -> None:
    """Register ``net_broker`` on ``server``: the by-name refusal, on the slow path."""
    from local_operator.network.relay import not_implemented_peer_op

    server.register_ops(
        {"net_broker": not_implemented_peer_op("net_broker")},
        slow={"net_broker": BROKER_OP_DEADLINE_S},
    )
