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

Nothing here stubs behaviour that lies. ``lop secret harden`` and
``lop secret unlock`` report that they need the broker, because they do; they
do not pretend to succeed.
"""

from __future__ import annotations

import os
from pathlib import Path

from local_operator.secrets.keys import load_master_key
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


def open_store(base: Path | None = None, *, create: bool = False) -> SecretStore:
    """Return a store ready to use, obtaining the master key for it.

    ``create`` is passed by the write verbs only: a ``get`` against a store
    that does not exist must say so rather than silently initialising an empty
    one and then reporting the secret missing, which are different problems
    with different fixes.

    **Broker seam.** PR 2 replaces the ``load_master_key`` call below with a
    broker request, falling back to the key file when the broker cannot start
    (design §13). The signature and every caller stay as they are.
    """
    return SecretStore(load_master_key(base, create=create), base=base)
