"""The mesh network package: one relay per install, and what it is allowed to do.

CONTRACT, IN ONE PARAGRAPH. A ``lop`` install can join a **network** — a named set
of paired devices that each hold a long-lived Ed25519 device key and a shared
network secret with an **epoch**. Joining is explicit and human-confirmed on both
ends (a single-use invite token carries the secret out of band, and both humans
compare a code derived from the handshake transcript). Every peer link is mutually
authenticated at the transport layer and authorised at the application layer
against that link's current membership and the exact capability each op needs, at
the one chokepoint :meth:`~local_operator.network.authorizer.Authorizer.check`.
Membership is revocable without visiting the other devices — a removal rotates the
secret and bumps the epoch, so a removed device is refused by every remaining
member — and every semantic event is appended to a bounded, append-only audit log
that never carries key material. This package is the **transport and membership**
core: it owns device identity, the records, pairing, the peer listener, the
authoriser, the audit log and the ``lop network`` CLI group. It owns no
transcript, holds no session lease, runs no turn, and never receives a session's
control key; the session plane, the credential broker, the catalogue fan-out and
the UIs are separate slices that build on these seams.

WHY THIS MODULE IMPORTS ALMOST NOTHING. ``local_operator/cli.py`` imports it to
register the ``lop network`` group, so every ``lop`` invocation — including
``--version`` — pays for whatever it pulls in. Only the vocabulary is re-exported
here; the crypto, the relay and the CLI verbs are imported by the code that uses
them, and ``tests/unit/network/test_imports.py`` pins that importing this package
loads neither ``cryptography`` nor ``asyncio``.
"""

from __future__ import annotations

from local_operator.network.types import (
    CAPABILITIES,
    MESH_PROTOCOL_VERSION,
    NET_OPS,
    NET_PAIR_OPS,
    PEERS_RUN_DIRNAME,
    ROLE_CAPABILITIES,
    HandshakeResult,
    InviteRecord,
    MemberRecord,
    MeshRefusal,
    NetworkRecord,
    PeerRecord,
    SecretState,
)

__all__ = [
    "CAPABILITIES",
    "MESH_PROTOCOL_VERSION",
    "NET_OPS",
    "NET_PAIR_OPS",
    "PEERS_RUN_DIRNAME",
    "ROLE_CAPABILITIES",
    "HandshakeResult",
    "InviteRecord",
    "MemberRecord",
    "MeshRefusal",
    "NetworkRecord",
    "PeerRecord",
    "SecretState",
]
