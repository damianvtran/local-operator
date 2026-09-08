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

    Defined here, and not raised anywhere yet, because the CLI's ``harden`` and
    ``unlock`` verbs report it as a not-yet-shipped capability. The broker
    daemon and peer authentication land in the follow-up PR; that PR raises
    this from the broker client in ``access.open_store``.
    """
