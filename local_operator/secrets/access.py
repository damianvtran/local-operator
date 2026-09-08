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

from local_operator.secrets.crypto import key_fingerprint
from local_operator.secrets.errors import SecretStoreError
from local_operator.secrets.keys import (
    load_master_key,
    replace_master_key,
    staged_key_path,
)
from local_operator.secrets.store import SecretStore, recorded_key_fingerprint

#: How many times :func:`resolve_master_key` re-reads the key state before
#: giving up. Each attempt loses only to a rotation completing concurrently,
#: and a rotation is a rare, operator-initiated act — several in a row while
#: one process tries to open the store means something is genuinely wrong, and
#: an error naming that beats an unbounded spin.
_RESOLVE_ATTEMPTS = 8

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


def resolve_master_key(base: Path | None = None, *, create: bool = False) -> bytes:
    """The key that actually opens this store, completing a broken rotation.

    ``rotate`` stages the new key, commits the re-sealed database, then
    installs the staged key as ``master.key``. A crash in the middle step
    leaves the database sealed under the staged key while ``master.key`` still
    holds the superseded one — every secret present and none of them readable,
    from an ordinary power cut.

    This function closes that window. When a staged key exists and its
    fingerprint matches the one the DATABASE records, the rotation committed
    and only the install is outstanding, so it is finished here and the store
    opens normally. The fingerprint comparison is what makes this safe to do
    automatically: adopting a staged key on its mere presence would let a
    leftover file from an abandoned rotation replace a perfectly good key.

    Any other combination falls through to the installed key untouched — a
    staged key that does not match is inert, and a store with no fingerprint
    row predates the mechanism and is opened exactly as before.

    **Written as a retry around an invariant, not as an ordering.** The rule is
    "return a key whose fingerprint is the one the database records", and every
    input to that decision — the key file, the staging file, the fingerprint
    row — can be changed by another session between any two reads. An earlier
    version read the key file once at the top and could then return that stale
    key after a concurrent rotation had replaced it, handing the caller a key
    that no longer opens the store. So each attempt re-reads what it needs and
    the result is checked against the invariant before it is returned.
    """
    for _ in range(_RESOLVE_ATTEMPTS):
        installed = load_master_key(base, create=create)
        expected = recorded_key_fingerprint(base)
        if expected is None or key_fingerprint(installed) == expected:
            # No store, a store predating the fingerprint row, or the ordinary
            # healthy case: the installed key is the store's key.
            return installed

        # The installed key does not open this database. Either a rotation
        # committed and died before installing its key, or one is completing
        # right now in another session.
        try:
            candidate = staged_key_path(base).read_bytes()
        except OSError:
            # No staged key to adopt. If another session is mid-install the
            # next attempt sees the new key file; if nothing is in flight the
            # loop ends and the mismatch is reported below rather than
            # returning a key that cannot decrypt anything.
            continue

        if key_fingerprint(candidate) != expected:
            continue

        # The re-seal committed under this key; only the install is missing.
        # Completing it here rather than leaving the store readable-but-
        # unrepaired means the next crash does not find the same half-done
        # state.
        replace_master_key(base, candidate)
        return candidate

    raise SecretStoreError(
        "This secret store is sealed under a master key that is not on disk. A key "
        "rotation appears to have been interrupted; the key that opens this store was "
        f"neither installed at {staged_key_path(base).parent / 'master.key'} nor left "
        "staged beside it."
    )


def open_store(base: Path | None = None, *, create: bool = False) -> SecretStore:
    """Return a store ready to use, obtaining the master key for it.

    ``create`` is passed by the write verbs only: a ``get`` against a store
    that does not exist must say so rather than silently initialising an empty
    one and then reporting the secret missing, which are different problems
    with different fixes.

    **Broker seam.** PR 2 replaces the key lookup below with a broker request,
    falling back to the key file when the broker cannot start (design §13). The
    signature and every caller stay as they are.
    """
    return SecretStore(resolve_master_key(base, create=create), base=base)
