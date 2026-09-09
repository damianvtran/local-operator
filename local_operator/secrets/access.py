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

from local_operator.secrets.crypto import key_fingerprint
from local_operator.secrets.errors import BrokerUnavailable, SecretStoreError
from local_operator.secrets.keys import (
    key_mode,
    load_master_key,
    secrets_dir,
    staged_key_paths,
)
from local_operator.secrets.store import (
    SecretStore,
    install_master_key_if_current,
    recorded_key_fingerprint,
)

#: How many times :func:`resolve_master_key` re-reads the key state before
#: giving up. Each attempt loses only to a rotation completing concurrently,
#: and a rotation is a rare, operator-initiated act â several in a row while
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

    **All staged keys are considered, not one well-known file.** Rotation is
    concurrent, so several rotations can sit between their COMMIT and their
    install at once and each stages under its own name; exactly one of them
    holds the key this database is now sealed under, and it is found by
    fingerprint. Scanning only a single path meant a second rotator's staging
    could displace the copy this store needed.

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
        matched: bytes | None = None
        for path in staged_key_paths(base):
            try:
                candidate = path.read_bytes()
            except OSError:
                # Raced with the owning rotation installing and removing it.
                continue
            if key_fingerprint(candidate) == expected:
                matched = candidate
                break

        if matched is None:
            # No staged key opens this database. If another session is
            # mid-install the next attempt sees the new key file; if nothing is
            # in flight the loop ends and the mismatch is reported below rather
            # than returning a key that cannot decrypt anything.
            continue

        # The re-seal committed under this key; only the install is missing.
        # Completing it here rather than leaving the store readable-but-
        # unrepaired means the next crash does not find the same half-done
        # state.
        #
        # Through the SAME compare-and-swap a rotator's own install uses, and
        # for the same reason: this is an install outside the re-seal
        # transaction, so without the guard a rotation committing between the
        # fingerprint read above and this write would have its key clobbered by
        # the stale one adopted here — the identical permanent-brick shape, just
        # reached from the repair path instead of from `rotate`. A refusal means
        # the store moved on under this call's feet, so the loop re-reads rather
        # than returning a key that provably no longer opens the database.
        if not install_master_key_if_current(matched, base):
            continue
        return matched

    raise SecretStoreError(
        "This secret store is sealed under a master key that is not on disk. A key "
        "rotation appears to have been interrupted; the key that opens this store was "
        f"neither installed at {secrets_dir(base) / 'master.key'} nor left "
        "staged beside it."
    )


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

    ``create`` short-circuits the broker only when there is genuinely no key to
    fetch: initialising a brand-new store has none yet, and starting a daemon
    to be told so is pure latency on the one path where the operator is
    waiting. **It must not short-circuit a store that already exists in
    passphrase mode** — see below.
    """
    hardened = key_mode(base) == "passphrase"

    # **Why `create` is not honoured in the hardened tier (QA Q9).** `set` is
    # the only verb passing `create=True`, and taking the short-circuit above
    # sent it straight to `load_master_key(create=True)`, which raises in
    # passphrase mode because there is deliberately no key on disk. The result
    # was that a hardened store could never accept a NEW secret even while
    # unlocked, with no workaround (`update` refuses unknown names) and an
    # error telling the operator to run the `unlock` they had just run
    # successfully. A hardened store was therefore frozen at whatever it held
    # when it was hardened — which breaks the harden-then-migrate sequence this
    # whole tier exists for.
    #
    # The short-circuit was always about latency on a store that does not yet
    # exist. A hardened store BY DEFINITION exists (its wrapped key is on
    # disk), so there is nothing to create and the broker is the only holder of
    # the key: fall through to the normal path, which serves an authorized
    # caller and gives a locked store the actionable "run `lop secret unlock`"
    # instead of the key-file message.
    if create and not hardened:
        return resolve_master_key(base, create=True)

    # Imported here, not at module scope: this module is on the path of every
    # `lop secret` verb, and the client drags in fcntl/socket machinery that
    # the create path above never needs.
    from local_operator.secrets.client import (
        BrokerDenied,
        BrokerLocked,
        ensure_broker,
        fetch_master_key,
    )

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
    return resolve_master_key(base)


def open_store(base: Path | None = None, *, create: bool = False) -> SecretStore:
    """Return a store ready to use, obtaining the master key for it.

    ``create`` is passed by the write verbs only: a ``get`` against a store
    that does not exist must say so rather than silently initialising an empty
    one and then reporting the secret missing, which are different problems
    with different fixes.

    **The broker seam is now closed.** :func:`master_key_for` asks the broker
    first and falls back to the key file when none can be reached (design
    §13); the disk side of that fallback goes through
    :func:`resolve_master_key`, so an interrupted rotation is still repaired on
    every path that reads a key from disk. The two compose in one direction
    only, and deliberately: the broker holds an already-resolved key in memory,
    so a caller it serves needs no repair, while a caller it refuses or cannot
    reach lands on the repairing path rather than on a raw
    :func:`load_master_key`.
    """
    return SecretStore(master_key_for(base, create=create), base=base)
