"""Error types for the encrypted secret store.

Its own stdlib-only module so that callers which must not pull the crypto
stack — argument registration on the CLI startup path, and the broker client
that lands in a follow-up — can still catch and classify a failure. Every one
of these carries a message written for the operator standing at a terminal,
because the CLI prints ``str(exc)`` verbatim to stderr.
"""

from __future__ import annotations

#: Valid values of a record's ``kind``. ``file`` marks a secret whose plaintext
#: is a file's contents, materialised to a private temporary path for the
#: duration of one command by ``lop secret file`` (design §7).
#:
#: It lives in this stdlib-only module, rather than beside the schema in
#: ``store.py``, because ``cli.add_parser`` needs it to build ``--kind``'s
#: choices on EVERY ``lop`` invocation. Importing it from ``store`` there would
#: drag SQLite and the whole crypto stack onto the startup path that
#: ``tests/unit/secrets/test_startup_cost.py`` pins as clean.
KINDS = ("string", "file")


class SecretStoreError(Exception):
    """Base class for every failure raised by the secret store."""


class InvalidSecretName(SecretStoreError):
    """The name is empty, over-long, or carries control characters.

    Names are bound into the AEAD's associated data with ``\\x00`` separators
    (see :mod:`local_operator.secrets.crypto`), so a name containing a NUL
    could make two different records produce the same AAD encoding. Rejecting
    the input is the only way to keep that encoding unambiguous.
    """


class SecretNotFound(SecretStoreError):
    """No record matches the requested name."""


class SecretExists(SecretStoreError):
    """A record with that name is already present.

    ``set`` refuses rather than replacing: silently overwriting a live
    credential from a typo'd name is unrecoverable, since the previous value
    is not kept anywhere.
    """


class SecretCorrupt(SecretStoreError):
    """A record failed to authenticate — it will not be returned.

    Raised when AES-GCM reports ``InvalidTag``, which happens when the
    ciphertext, the nonce or any AAD-bound metadata (record id, name,
    description, kind, key generation) has been altered since the record was
    sealed. This is the fail-closed path: a tampered record is an error, never
    a plausible-looking wrong answer.
    """


class IncompatibleStore(SecretStoreError):
    """The store on disk was written by a runtime newer than this one.

    The operator runs many sessions and updates the runtime underneath them,
    so version skew is routine. An older runtime must never migrate a newer
    store downward or guess at a record format it does not know.
    """


class StaleKeyEpoch(SecretStoreError):
    """The master key this caller holds is no longer the store's key.

    Raised when a write's key does not match the fingerprint the store records,
    which happens when another session completed a ``rotate`` after this one
    loaded the key — the routine case on a machine running ~10 sessions at
    once. The write is refused rather than committed, because a record sealed
    under a superseded key is indexed under a superseded index key too: it
    would be unreachable by name AND undecryptable, while ``set`` reported
    success. Failing closed costs the operator one retry; succeeding costs them
    the secret.
    """


class InsecurePermissions(SecretStoreError):
    """A store file is readable by someone other than its owner.

    The whole at-rest story in ``keyfile`` mode rests on the mode bits, so a
    key file that another account can read is a condition to stop on, not one
    to quietly repair — the exposure already happened and the operator needs
    to know.
    """


class BrokerUnavailable(SecretStoreError):
    """The key broker could not be reached.

    Strictly "nothing answered": the socket is absent, refused the connection,
    or the daemon accepted and then never replied. A broker that ANSWERED and
    said no is a different condition and must not be collapsed into this one —
    see :class:`BrokerIncompatible` and the round-4 note on it.
    """


class BrokerIncompatible(SecretStoreError):
    """A broker is listening, but it speaks a different protocol version.

    **Its own class because "live but unusable" is not "unreachable", and
    conflating them silently disarmed a safety property (round-4 Q4).** The
    broker refuses a version-mismatched request, and every such refusal used to
    reach callers as a bare :class:`SecretStoreError` that
    ``client.is_running`` swallowed into ``False``. Every caller then read that
    as "no broker", and :func:`local_operator.secrets.access.retrieve_secret`
    degraded to the UNNOTIFIED local decrypt — serving the value with no §6
    redaction notice, which is the exact defect (Q3) the broker seam exists to
    close. Because a stale daemon is most likely precisely at a runtime update,
    the laundering armed itself at the worst moment.

    ``PROTOCOL_VERSION`` has only ever been 1, so no shipped runtime can reach
    this yet; it is a latent hole being closed before the first bump makes it
    live rather than after.

    It carries the daemon's ``pid`` and the ``protocol`` it speaks because the
    version refusal is the one message that MUST be actionable: the advice is
    to restart the broker, and stopping it needs a pid that no other request
    can obtain — ``status`` fails the same version gate.
    """

    def __init__(self, message: str, *, pid: int | None = None, protocol: object = None) -> None:
        super().__init__(message)
        self.pid = pid
        self.protocol = protocol


#: Store-error classes that survive a broker round trip, by class name.
#:
#: **Why a name-keyed allowlist and not ``getattr`` on this module.** The class
#: name arrives from the broker socket, and resolving an arbitrary attribute
#: name off a module because a peer asked for it is how a wire format turns
#: into a lookup primitive. An explicit table can only ever yield one of these
#: classes, and an unknown name degrades to :class:`SecretStoreError` rather
#: than raising something unexpected.
#:
#: **Why the taxonomy has to cross the seam at all (round-4 R11/Q5).** The
#: broker flattened every store failure into one ``"store"`` reply code, so a
#: caller could not tell a missing secret from a corrupt record — and
#: :class:`~local_operator.secrets.runtime.SecretsMapping` translates exactly
#: one of them (:class:`SecretNotFound`) into the ``KeyError`` its ``Mapping``
#: contract owes callers. With the class erased, ``secrets.get("ABSENT",
#: "dflt")`` RAISED instead of returning the default whenever a broker happened
#: to be live. Preserving the class here fixes it for every consumer at once,
#: rather than in the one that noticed.
WIRE_ERROR_KINDS: dict[str, type[SecretStoreError]] = {
    cls.__name__: cls
    for cls in (
        InvalidSecretName,
        SecretNotFound,
        SecretExists,
        SecretCorrupt,
        IncompatibleStore,
        StaleKeyEpoch,
        InsecurePermissions,
    )
}


def error_for_kind(kind: object, message: str) -> SecretStoreError:
    """Rebuild a store error the broker named, or the base class.

    An unknown or absent ``kind`` is the OLD-broker case and must not be an
    error of its own: a daemon from before this field existed passes the
    version gate (the wire version did not change) and simply says nothing
    about the class, which is exactly the pre-round-4 behaviour.
    """
    if isinstance(kind, str):
        cls = WIRE_ERROR_KINDS.get(kind)
        if cls is not None:
            return cls(message)
    return SecretStoreError(message)
