"""How a caller gets an open store — the seam the broker slides into.

Every surface (the CLI today; the agent tool and the eval library later) goes
through :func:`open_store` rather than constructing a
:class:`~local_operator.secrets.store.SecretStore` itself. That indirection is
the whole point of this module: today it reads the master key from the key file
in the same process, and the broker PR replaces the *body* of this function
with a unix-socket round trip to a daemon that holds the key in RAM and
authenticates the peer by ancestry. Nothing else has to change.

The seam is deliberately drawn at the KEY, not at the store: the broker serves
the master key material to an authenticated peer, and the peer then talks to
SQLite directly. Putting the whole store behind the socket would mean
reimplementing every verb as a wire protocol, and would put the database's
concurrency story inside a single-threaded daemon rather than in WAL, which
§4 chose precisely because it already works under ~10 sessions.

**The broker is now behind this seam.** :func:`open_store` asks the broker for
the master key first and falls back to the key file when no broker can be
reached. The fallback is honest rather than silent-and-equivalent, and which
tier is live decides everything:

- ``keyfile`` mode: the key is on disk beside the store, so a caller the broker
  refuses can read it directly anyway (design §8). The broker's value here is
  the audit trail and the §6 redaction notice, not access control, and a
  denial or an outage therefore falls back rather than failing — refusing
  would break the operator's own terminal while stopping no attacker.
- ``passphrase`` mode: there is no unwrapped key on disk to fall back TO. A
  denial is enforced, and an unreachable broker is a clearly-worded failure
  rather than a degraded success (design §13).
"""

from __future__ import annotations

import os
from pathlib import Path

from local_operator.secrets.errors import BrokerUnavailable, SecretStoreError
from local_operator.secrets.keys import key_mode, load_master_key
from local_operator.secrets.store import SecretStore

#: Environment variable naming the session a CLI invocation belongs to, if it
#: was launched by one. Recorded in the audit trail so a retrieval can be
#: attributed. Read-only here and never trusted for authorization — a same-uid
#: process can forge any environment variable it likes (design §2.2, spike 2),
#: which is exactly why the BROKER authenticates by ancestry instead.
SESSION_ID_ENV = "LOCAL_OPERATOR_SESSION_ID"


def session_id() -> str | None:
    """The session id to attribute an audit entry to, if one is advertised."""
    value = os.environ.get(SESSION_ID_ENV, "").strip()
    return value or None


def master_key_for(base: Path | None = None, *, create: bool = False) -> bytes:
    """The master key, from the broker when there is one, else from disk.

    **Why the broker is tried first even in ``keyfile`` mode, where it grants
    nothing extra.** Going through it means retrievals are attributed to a peer
    in the audit chain and, once PR 3 lands, that the §6 redaction notice
    fires. Reading the key file directly is correct but unobserved.

    **Why the fallback is safe to take.** In ``keyfile`` mode the key file IS
    the at-rest story; a caller that can read it is a caller the broker would
    have served, so falling back changes observability, not authority. In
    ``passphrase`` mode the fallback does not exist — :func:`load_master_key`
    raises, because the unwrapped key lives only in broker memory.

    ``create`` short-circuits the broker: initialising a brand-new store has no
    key to fetch yet, and starting a daemon to be told so is pure latency on
    the one path where the operator is waiting.
    """
    if create:
        return load_master_key(base, create=True)

    # Imported here, not at module scope: this module is on the path of every
    # `lop secret` verb, and the client drags in fcntl/socket machinery that
    # the create path above never needs.
    from local_operator.secrets.client import (
        BrokerDenied,
        BrokerLocked,
        ensure_broker,
        fetch_master_key,
    )

    hardened = key_mode(base) == "passphrase"
    try:
        if ensure_broker(base):
            return fetch_master_key(base)
        if hardened:
            raise BrokerUnavailable(
                "This store is hardened with a passphrase and the secret broker could not "
                "be started, so there is no key to decrypt it with. Start one with "
                "`lop secret broker start`, then `lop secret unlock`."
            )
    except BrokerLocked:
        # A locked store has no unwrapped key ANYWHERE, so there is nothing to
        # fall back to and pretending otherwise would just fail later and less
        # clearly.
        raise
    except BrokerDenied:
        # **Why a denial is not fatal in keyfile mode, which looks alarming and
        # is not.** In that mode the master key sits on disk beside the store,
        # so a caller the broker just refused can read it directly with no
        # broker involved at all — design §8 states this outright ("Script
        # reading a known key file path: NOT stopped"). Failing here would
        # therefore add no security whatsoever while breaking the store's
        # PRIMARY surface: the operator's own `lop secret get` in their own
        # terminal has no lop session among its ancestors and is denied by
        # construction.
        #
        # The ancestry check earns its keep in the tier where it is backed by
        # something: in passphrase mode the key is not on disk unwrapped, the
        # broker is the only holder, and a denial is enforced below. That is
        # the same boundary §8's table draws, implemented rather than widened.
        if hardened:
            # Re-worded rather than re-raised. The broker's own reason is about
            # ANCESTRY ("no lop session is registered with the broker"), which
            # is accurate and useless to the person reading it: in this tier a
            # denial almost always means the store has not been unlocked since
            # the last reboot, and the ancestry wording sends the operator
            # hunting a session problem they cannot act on. Say what to do.
            raise BrokerDenied(
                "This store is hardened with a passphrase and this caller is not authorized "
                "to use the unlocked key. If you have not unlocked it since the last reboot, "
                "run `lop secret unlock`."
            ) from None
    except SecretStoreError:
        # Anything else — no broker, a wedged one, a version mismatch — is an
        # availability problem, and §13 requires the store to keep working in
        # keyfile mode rather than becoming unusable because a daemon died.
        if hardened:
            raise
    return load_master_key(base)


def open_store(base: Path | None = None, *, create: bool = False) -> SecretStore:
    """Return a store ready to use, obtaining the master key for it.

    ``create`` is passed by the write verbs only: a ``get`` against a store
    that does not exist must say so rather than silently initialising an empty
    one and then reporting the secret missing, which are different problems
    with different fixes.
    """
    return SecretStore(master_key_for(base, create=create), base=base)
