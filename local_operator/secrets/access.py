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
import time
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


def forget_inherited_session_id() -> None:
    """Drop an inherited :data:`SESSION_ID_ENV` from a process that hosts its OWN session.

    The value names the session whose TOOL ran this process. That is right for a
    command a tool runs (``lop network approvals request`` files as that session),
    and wrong for a process that hosts a session itself (a runtime, a `lop exec`
    run started from a tool's shell): its in-process :func:`session_id` readers
    would audit as the PARENT. Called at those hosts' entry. Clearing, not
    replacing: the host does not know its own id yet, so unattributed is the honest
    answer until it does, which is what such a process recorded before
    ``LOCAL_OPERATOR_SESSION_ID`` was exported at all. Its own tool children get
    their context's id from the child-env injection either way.
    """
    os.environ.pop(SESSION_ID_ENV, None)


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


def master_key_for(
    base: Path | None = None, *, create: bool = False, deadline: float | None = None
) -> bytes:
    """The master key, from the broker when there is one, else from disk.

    ``deadline`` (``time.monotonic``) bounds the broker leg — pings, spawn poll
    and the ``key`` request — for a caller that is already spending one budget
    (#4389 Q4: ``retrieve_secret``'s local fallback used to re-ping a silent
    broker twice with fresh 10 s timeouts, ~29 s in all). Once it has passed,
    ``ensure_broker`` does no further broker work and this goes straight to the
    key file. ``None`` is the ordinary, unbudgeted call.

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
    from local_operator.secrets.errors import BrokerIncompatible

    # Passed only when there is one, so the many doubles standing in for these
    # two calls with `(base)` keep working.
    bounded = {"deadline": deadline} if deadline is not None else {}
    try:
        if ensure_broker(base, **bounded):
            return fetch_master_key(base, **bounded)
        if hardened:
            raise BrokerUnavailable(
                "This store is hardened with a passphrase and the secret broker could not "
                "be started, so there is no key to decrypt it with. Start one with "
                "`lop secret broker start`, then `lop secret unlock`."
            )
    except BrokerIncompatible:
        # **A stale broker is fatal for VALUES, not for METADATA (Q4).**
        # `retrieve_secret` lets this propagate, because serving a value from
        # the local decrypt while a live broker sits there unable to publish
        # the §6 notice is the Q3 defect. This function is the other kind of
        # caller: `__contains__`, `__iter__` and `lop secret list` open the
        # store to read RECORD METADATA, hand out no bytes, and owe no notice —
        # so failing them would take the store's read-only surfaces down over a
        # skew that costs them nothing. In the hardened tier there is no key on
        # disk to fall back to, so it stays fatal there.
        if hardened:
            raise
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


def open_store(
    base: Path | None = None, *, create: bool = False, deadline: float | None = None
) -> SecretStore:
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

    ``deadline`` is threaded to :func:`master_key_for` (#4389 Q4).
    """
    if deadline is None:
        return SecretStore(master_key_for(base, create=create), base=base)
    return SecretStore(master_key_for(base, create=create, deadline=deadline), base=base)


def retrieve_secret(name: str, base: Path | None = None, *, role: str = "agent") -> bytes:
    """One secret's VALUE, announced to the owning session first (design §6).

    **Why this exists rather than a bare ``open_store().get()`` (QA Q3).** The
    broker's ``retrieve`` op is what makes the §6 redaction notice fire: it
    tells the session that owns this peer to scrub the value and WAITS for the
    ack before answering, so a value taken through ``$(lop secret get NAME)``
    cannot reach a model-visible channel ahead of the filter that redacts it.
    Decrypting locally is functionally identical to the caller and silently
    skips that notification — which is exactly what happened: the documented
    ``$(lop secret get NAME)`` path painted the credential raw in the live
    stream, in ``jobs(op='peek')`` and in the tool result, because the notifier
    and its consumer were never connected. Every value-returning surface routes
    through HERE so there is ONE announcement path rather than one per caller.

    ``role`` is the NAMESPACE this reader claims, forwarded to the store's own
    check. Agent-context surfaces (the CLI's ``get``, the eval ``secrets``
    mapping, the ``secret_ref`` resolver) are the default and therefore refuse a
    ``LOP_PROVIDER_*`` name; a provider-context reader passes ``role="provider"``
    explicitly. Defaulting to the restrictive side is deliberate — a caller that
    forgets the argument is denied rather than served.

    **The fallback is the same shape, and the same trade, as
    :func:`master_key_for`'s (see below).** A retrieval must not become
    impossible because no daemon is running: ``lop secret get`` typed in a
    plain terminal, from a script, or on the keyfile tier has no session to
    notify and frequently no broker at all. An availability failure therefore
    degrades to the local decrypt — UNNOTIFIED, never blocked. That is the
    round-2 invariant applied to this path: degrade to unnotified, never to a
    blocked or failed retrieval.

    **Except inside a live session's lineage, which fails closed.** That
    invariant holds for callers with no session to notify. A session's tool
    child (:func:`in_live_session_lineage`) HAS one, and its stdout is the
    transcript, so "unnotified" there means the value lands in the transcript
    unredacted — the very leak §6 exists to stop, triggered by nothing rarer
    than a broker crash mid-session (a respawned daemon has an empty session
    table, so it denies; the old ``BrokerDenied`` clause then decrypted
    locally). Such a caller waits a bounded :data:`SESSION_BROKER_WAIT_S` for a
    broker to serve it and otherwise raises
    :class:`~local_operator.secrets.errors.SessionBrokerLost`; it never reaches
    the local decrypt. Plain ``lop secret get`` in an operator's terminal keeps
    the fallback described above.

    **The fallback is keyed on REACHABILITY, not on the refusal.** This is the
    one place the shape deliberately departs from :func:`master_key_for`, and
    the asymmetry is the whole safety property. Falling back after a *live*
    broker refused would serve, unnotified, precisely the value that broker had
    just decided could not be kept out of the operator's transcript — the
    fail-closed rule (review R3) inverted. So only an UNREACHABLE broker
    (:class:`BrokerUnavailable`, or one that will not start) drops to the local
    decrypt; a broker that answered and said no is believed. That covers the
    ``unredactable`` denial, which is exactly "the owning session never
    acknowledged", without needing a new exception class on the wire.

    A broker left running across a runtime update answers with a
    protocol-version refusal, and that raises (:class:`BrokerIncompatible`)
    instead of quietly reading the key file — it is LIVE, so the reachability
    rule above puts it on the believed side of the line. Round 4 found this
    docstring describing behaviour the code did not have: the refusal reached
    ``is_running`` as a bare ``SecretStoreError``, which reported ``False``,
    and control fell to the unnotified local decrypt — a full reproduction of
    Q3, arming itself at the first protocol bump, i.e. exactly at a runtime
    update. The class exists so the two cases stay distinguishable.

    **How the ``unredactable`` denial is actually covered.** An earlier version
    of this docstring claimed the ``BrokerDenied`` clause below believed it.
    The property holds — the reviewer drove all five refusal codes and no
    ``unredactable`` reply serves a value — but by a different route:
    ``unredactable`` is raised STRUCTURALLY inside the broker's retrieve
    handler and never reaches that clause at all, arriving instead as a bare
    :class:`SecretStoreError` that this function does not catch, so it
    propagates. Right property, wrong mechanism, now stated as it is.

    **``BrokerDenied`` in the keyfile tier is the documented exception.** §8
    concedes the broker is not the boundary there — the master key is on disk
    beside the store, so a refused caller reads it directly anyway, and the
    operator's own terminal has no lop session among its ancestors and is
    denied by construction. Making that fatal would break the store's primary
    surface while buying nothing. In the hardened tier the denial stands, and
    so it does in ANY tier once the read observed a broker roll (#4389 S8): it
    then ends in a served read or :class:`BrokerRolling`, never a local decrypt.

    **The error taxonomy is identical on both paths (R11/Q5).** A store failure
    raises the same class whichever side served it — the broker names the class
    on the wire and :func:`local_operator.secrets.errors.error_for_kind`
    rebuilds it — so a caller translating :class:`SecretNotFound` into a
    ``KeyError`` behaves the same with a broker live and with none running.
    Routing this through the broker while the classes still collapsed to
    ``SecretStoreError`` is what broke ``secrets.get(name, default)``.
    """
    # Imported inside the function, matching `master_key_for` above: this
    # module is on the path of every `lop secret` verb, and the client drags in
    # socket/fcntl machinery the write verbs never need.
    from local_operator.secrets.client import (
        ROLL_WAIT_S,
        BrokerDenied,
        ensure_broker,
        retrieve,
        rolling_in_progress,
    )

    hardened = key_mode(base) == "passphrase"
    owner = None if hardened else in_live_session_lineage(base)
    if owner is not None:
        # The keyfile-tier fallback below is for callers with no session to
        # notify. A registered session's tool child has one, so it never takes
        # it; see `_retrieve_inside_session` for the bounded wait and refusal.
        return _retrieve_inside_session(name, base, role=role, owner=owner)
    # **One budget for the whole call, fallback included (QA Q2, #4389 Q4).**
    # The liveness ping, any spawn, the retrieval AND the local fallback's own
    # broker leg (`open_store` → `master_key_for` → `ensure_broker`) share one
    # deadline of the client's roll budget. Before, the ping alone could wait a
    # full request timeout against a silent broker (measured 14 s), and with
    # no roll in sight the fallback re-pinged it twice more (QA measured ~29 s
    # before the local decrypt). A healthy broker answers in well under it: the
    # one slow leg it has, the redaction notice, is capped at 2 s.
    deadline = time.monotonic() + ROLL_WAIT_S
    # **The roll observation is held for the whole call (#4389 S8).** It is
    # read ONCE, before anything can spend the budget. A sentinel that expires
    # while the liveness ping or the retrieval waits must not turn "a roll was
    # under way and the broker never answered" back into an ordinary outage —
    # that is the unnotified local decrypt the roll guard exists to prevent,
    # and lineages that predate the session marker rely on this guard alone.
    observed_roll = rolling_in_progress(base)
    try:
        if ensure_broker(base, deadline=deadline):
            # `observed_roll` is passed only when set, so doubles standing in for
            # `retrieve` with the pre-#4389 keywords keep working.
            seen = {"observed_roll": True} if observed_roll else {}
            return retrieve(name, base, role=role, deadline=deadline, **seen)
        # No broker could be started. In the hardened tier there is no
        # unwrapped key on disk to fall back to, so let the local path raise
        # its own accurate message rather than inventing one here.
        # Re-sampled, not just the entry reading (#4389 M4-1): a roll that
        # began after entry is just as much a reason this broker is missing.
        observed_roll = observed_roll or rolling_in_progress(base)
        if observed_roll and not hardened:
            raise _roll_outlasted(ROLL_WAIT_S)
    except BrokerUnavailable as exc:
        # Reached one moment, gone the next. Availability, not policy: §13
        # keeps the store usable rather than making a daemon a hard dependency
        # — unless a roll was observed, when silence is the roll not finishing.
        if hardened:
            raise
        if observed_roll or rolling_in_progress(base):
            raise _roll_outlasted(ROLL_WAIT_S) from exc
    except BrokerDenied as exc:
        # With NO roll observed a denial is policy, not silence: it keeps the
        # keyfile tier's documented fallback (§8). Once a roll was observed it
        # does not (#4389 S8, security r4): the successor's hold covers only
        # lineages it expects, so a denial here may be a late lineage it never
        # heard of, and the local decrypt would skip the redaction notice. The
        # client already refuses a denial it saw during a roll
        # (`_retrieve_through_roll`); this clause covers the entry observation
        # and a roll that started after it (M4-1).
        if hardened:
            raise
        if observed_roll or rolling_in_progress(base):
            raise _roll_denied() from exc
    return open_store(base, deadline=deadline).get(name, role=role, session_id=session_id())


def _roll_outlasted(budget: float) -> Exception:
    """The refusal for a broker that stayed unreachable after a roll was observed."""
    from local_operator.secrets.client import BrokerRolling

    return BrokerRolling(
        "the secret broker is being replaced by an update and did not come back "
        f"within {budget:g}s; retry in a moment"
    )


def _roll_denied() -> Exception:
    """The refusal for a broker denial met by a read that observed a roll."""
    from local_operator.secrets.client import BrokerRolling

    return BrokerRolling(
        "the secret broker refused this read while it was being replaced by an "
        "update; the value was not read locally, so retry in a moment"
    )


# `BrokerIncompatible` is deliberately absent from the clauses above: it is a
# LIVE broker, so the reachability rule believes it and it propagates to the
# caller carrying the daemon's own restart advice.


#: Upper bound on how long a retrieval inside a session waits for a broker to
#: serve it before failing closed — the WHOLE wait, every ping, spawn poll and
#: request clipped to it (#4388 R1 / #4389 S5: the two PRs' budgets used to
#: stack to ~16 s). Equal to ``client.ROLL_WAIT_S`` (8 s, the ceiling the
#: security owner set for a `$(lop secret get …)` to stall) on purpose: a read
#: that meets a broker ROLL is held by the successor for ``broker.ROLL_GRACE_S``
#: plus the redaction notice, which is exactly that budget, so a shorter wait
#: here would refuse reads an update roll was about to serve. It still covers a
#: crash: ``client.STARTUP_TIMEOUT_S`` (5 s) to respawn plus the session's
#: re-registration (≤ ~2 s, ``session._REREGISTER_DELAYS_S``). Pinned by
#: ``test_session_fail_closed.py::test_session_wait_matches_the_roll_budget``.
#: Module-level so tests can shrink it rather than sleeping through it.
SESSION_BROKER_WAIT_S = 8.0

#: Pause between attempts inside that window. Each attempt is a socket
#: connect plus, when nothing listens, a non-blocking spawn-lock probe, so a
#: quarter second keeps the loop to ~24 attempts rather than a busy spin.
_SESSION_BROKER_RETRY_S = 0.25


def in_live_session_lineage(base: Path | None = None) -> tuple[int, str] | None:
    """The registered session whose redactor this process's output reaches, if any.

    Returns ``(pid, session_id)`` of the nearest ANCESTOR holding a live
    registration with the broker for this store, else ``None``. The answer
    comes from the lineage markers a registration publishes once the broker
    has accepted it (``secrets.session.registered_session_ancestor``, which
    documents why the record lives on disk): the broker's own session table is
    the authority while it runs (broker.py ``_session_for``), but it dies with
    the broker — the one case this has to decide.

    Deliberately NOT ``LOCAL_OPERATOR_SESSION_ID``. Every session's tool child
    carries that (``tools.builtin.session_id_env_injection``) whether or not the
    session ever reached a broker; a session that booted before the store
    existed, or whose registration was refused, has no redactor to notify, and
    its children keep the §13 fallback exactly as before. The env var is also
    attribution only (builtin.py ``session_id_env_injection`` docstring).
    """
    from local_operator.secrets.session import registered_session_ancestor

    try:
        return registered_session_ancestor(base)
    except Exception:  # noqa: BLE001 — an unreadable marker must not break a retrieval
        return None


def _retrieve_inside_session(
    name: str, base: Path | None, *, role: str, owner: tuple[int, str]
) -> bytes:
    """Serve a session descendant through the broker, or refuse; never decrypt locally.

    **Why a bounded wait, and why it also retries a denial.** A broker that
    crashed or rolled leaves two observable states: nothing listening
    (:class:`BrokerUnavailable`, or ``ensure_broker`` cannot start one) and a
    fresh daemon that has not yet seen the session re-register, which denies the
    descendant for want of a registered ancestor (``BrokerDenied``). Both can
    clear by themselves within a few seconds, so both are retried until
    :data:`SESSION_BROKER_WAIT_S` elapses. A denial is still never a reason to
    fall back — the retry only asks the same broker again — so a genuine policy
    refusal costs a bounded delay and then the same refusal.

    **A roll is part of the same wait.** ``BrokerRolling`` (the client's roll
    budget ran out) is folded into this refusal rather than escaping as its own
    class, and the client gets this function's deadline, so a roll, a crash and
    a late re-registration (security S6) all end in ONE bounded refusal.

    Everything else the broker answers propagates on the first attempt, as on
    the out-of-session path: a served value, ``SecretNotFound`` (so
    ``secrets.get(name, default)`` still sees its miss, R11/Q5), the structural
    ``unredactable`` refusal and ``BrokerIncompatible``.

    The error names the remedy rather than the mechanism: the session has to
    re-register with a running broker, which today means restarting it.
    """

    from local_operator.secrets.client import (
        STARTUP_TIMEOUT_S,
        BrokerDenied,
        BrokerRolling,
        ensure_broker,
        retrieve,
    )
    from local_operator.secrets.errors import SessionBrokerLost

    deadline = time.monotonic() + SESSION_BROKER_WAIT_S
    last: str = "no secret broker could be started"
    while True:
        remaining = deadline - time.monotonic()
        # No attempt starts after the window closes (#4389 S5/CR9): the old
        # 100 ms floor bought one more ping-and-spawn past the bound.
        if remaining <= 0:
            break
        try:
            # `ensure_broker`'s own spawn poll is clipped to the window so one
            # slow start cannot carry the total past the bound.
            if ensure_broker(
                base,
                timeout=min(STARTUP_TIMEOUT_S, remaining),
                deadline=deadline,
            ):
                return retrieve(name, base, role=role, deadline=deadline)
            last = "no secret broker could be started"
        except (BrokerUnavailable, BrokerDenied, BrokerRolling) as exc:
            last = str(exc)
        if time.monotonic() + _SESSION_BROKER_RETRY_S > deadline:
            break
        time.sleep(_SESSION_BROKER_RETRY_S)
    # Plain English, like every message in `secrets.errors`: the CLI prints
    # `str(exc)` verbatim and this module's errors are not catalogue-keyed.
    owner_pid, owner_id = owner
    label = f"{owner_id!r} (pid {owner_pid})" if owner_id else f"pid {owner_pid}"
    raise SessionBrokerLost(
        f"refusing to read secret {name!r}: this command runs inside lop session "
        f"{label}, whose redactor is registered with the secret broker, but no "
        f"broker is serving that session ({last}). The value was not read, so it "
        "cannot reach the transcript unredacted. Restart the session so it "
        "registers with a running broker (`lop secret broker status` shows one), "
        "then retry."
    )
