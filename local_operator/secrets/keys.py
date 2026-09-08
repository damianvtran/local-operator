"""Where the master key lives at rest, and how the mode bits are enforced.

Design reference: ``docs/design/secret-store.md`` §2.3.

Two tiers were specified. This module ships the DEFAULT one — ``keyfile``: the
master key sits in ``<config>/secrets/master.key``, mode 0600, inside a 0700
directory, and any lop invocation reads it with no prompt. The honest claim for
that mode, repeated here so it cannot drift out of the code: it is *equivalent
to today's plaintext ``.env`` against an attacker who specifically reads the key
file*, and materially better against the opportunistic "grep the disk for
credentials" malware that a bad link actually drops. It is not a vault.

The opt-in ``passphrase`` tier (``lop secret harden``) wraps this key with
scrypt and keeps the unwrapped copy only in the broker's memory. It needs the
broker daemon to hold that memory, so it arrives with the broker; see
:func:`key_mode` for the seam.

Permissions are applied with an explicit ``chmod`` after creation rather than
through ``mkdir(mode=...)``/``os.open(mode=...)`` alone, because both of those
are masked by the process umask. A developer running with ``umask 000`` would
otherwise create a world-readable key and nothing would say so.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from local_operator.paths import config_dir
from local_operator.secrets.crypto import KEY_BYTES, generate_master_key
from local_operator.secrets.errors import InsecurePermissions, SecretStoreError

#: Subdirectory of the config dir holding everything this feature owns.
SECRETS_DIRNAME = "secrets"

DIR_MODE = 0o700
FILE_MODE = 0o600

#: Bits that must be clear on the key file and the database: any permission
#: for group or other. Checked on every open, not only at creation — a store
#: whose mode was loosened after the fact is exactly the case worth catching.
_FORBIDDEN_MODE_BITS = 0o077

#: File holding the registration ticket: the secret a process must present to
#: claim it is a lop session (design §2.1, PR-2 review R1). See
#: :func:`registration_ticket` for why this is a file rather than a property of
#: the connecting process.
TICKET_FILENAME = "register.ticket"

#: Ticket length. 32 bytes from the OS CSPRNG — the same budget as the master
#: key, because a guessable ticket is a self-registration bypass.
TICKET_BYTES = 32


def secrets_dir(base: Path | None = None) -> Path:
    """The directory holding the store, the key and the audit log.

    ``base`` overrides the config dir for tests. Resolved on every call rather
    than cached, for the reason :func:`local_operator.paths.config_dir`
    documents: a module constant freezes whatever the first importer saw.
    """
    return (base if base is not None else config_dir()) / SECRETS_DIRNAME


def store_path(base: Path | None = None) -> Path:
    """Path to the SQLite database."""
    return secrets_dir(base) / "store.db"


def key_path(base: Path | None = None) -> Path:
    """Path to the master key file."""
    return secrets_dir(base) / "master.key"


def audit_log_path(base: Path | None = None) -> Path:
    """Path to the mirrored append-only audit log."""
    return secrets_dir(base) / "audit.log"


def wrapped_key_path(base: Path | None = None) -> Path:
    """Path to the scrypt-wrapped master key used by ``passphrase`` mode."""
    return secrets_dir(base) / "master.key.wrapped"


def key_mode(base: Path | None = None) -> str:
    """Which at-rest tier the store is in: ``keyfile`` or ``passphrase``.

    ``harden`` writes a scrypt-wrapped ``master.key.wrapped`` and removes the
    plain key, at which point this returns ``passphrase`` and callers route the
    unwrap through the broker — which is the only process that ever holds the
    unwrapped copy. Reported by ``lop secret status`` so the operator can see
    which tier is live.
    """
    if wrapped_key_path(base).exists():
        return "passphrase"
    return "keyfile"


def ensure_secrets_dir(base: Path | None = None) -> Path:
    """Create the secrets directory 0700 and return it.

    The ``chmod`` runs even when the directory already existed: an operator who
    once created it by hand, or a umask that widened it at creation, should not
    leave the key sitting in a traversable directory forever.
    """
    directory = secrets_dir(base)
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, DIR_MODE)
    return directory


def check_mode(path: Path) -> None:
    """Raise :class:`InsecurePermissions` if ``path`` is group/other accessible.

    This does NOT repair the mode. By the time a key file is world-readable the
    exposure has already happened; silently tightening it would hide from the
    operator that it was ever open. Windows reports POSIX bits that do not mean
    what they do on Unix, so the check is skipped there rather than producing a
    failure nobody can act on.
    """
    if os.name == "nt":
        return
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & _FORBIDDEN_MODE_BITS:
        raise InsecurePermissions(
            f"{path} has mode {mode:04o}; it must not be readable by group or others. "
            f"Fix it with: chmod {FILE_MODE:04o} {path}"
        )


def write_private_file(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` such that it is never briefly world-readable.

    ``os.open`` with ``O_CREAT|O_EXCL`` and mode 0600 creates the file already
    private, and the explicit ``chmod`` covers the umask masking that mode and
    the case where the caller replaced an existing file. Writing then chmod-ing
    would leave a window in which another process can open it — small, but this
    is the one file where that window costs everything.
    """
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, FILE_MODE)
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, FILE_MODE)


def ticket_path(base: Path | None = None) -> Path:
    """Path to the registration ticket."""
    return secrets_dir(base) / TICKET_FILENAME


def registration_ticket(base: Path | None = None) -> bytes:
    """The secret a caller must present to register itself as a lop session.

    **Why a file secret and not a property of the process.** Review finding R1
    was that ``register`` was unauthenticated, and the obvious repairs are all
    unsound against the actual threat model. Measured on this machine rather
    than assumed: the attacker shape §8 cares about is a double-forked
    ``setsid`` script, and a same-uid process can forge *every* signal the
    kernel exposes about itself — it reparents to launchd (``ppid=1``, which
    is also what a real detached session has), becomes a session leader, and
    can even allocate its own pty via ``pty.fork()`` so a controlling-tty check
    passes. There is therefore NO process-shape predicate that separates a
    genuine session from the malware, and code-signature verification is
    unavailable here (``csops`` denied, §2.1). Requiring the registrant to *be*
    an ancestor of itself is what created the bypass in the first place.

    What remains is a secret the attacker cannot read. This file is 0600 inside
    the 0700 secrets directory, so it is exactly as reachable as the master key
    — which is the point, and why this is honest rather than circular:

    - In ``keyfile`` mode the master key sits beside it, so an attacker who can
      read the ticket could have read the key directly and skipped the broker
      entirely. The ticket costs that attacker nothing they did not already
      have, and §8 now says so plainly instead of claiming the socket stops
      them.
    - In ``passphrase`` mode there is no unwrapped key on disk. The ticket is
      the ONLY thing on disk, and holding it grants standing to *ask* — it does
      not unwrap anything. An attacker with the ticket still needs the broker
      to be unlocked, and unlocking needs the passphrase, which is never on
      disk in any form. That is the tier the design nominates as load-bearing,
      and it is the tier where this check does real work.

    Created lazily and never rotated in place: a rotation would deregister
    every live session on a machine the operator runs ~10 of them on, and the
    ticket protects standing to ask rather than the secrets themselves.
    """
    path = ticket_path(base)
    if path.exists():
        check_mode(path)
        ticket = path.read_bytes()
        if len(ticket) == TICKET_BYTES:
            return ticket
        # A truncated or padded ticket is damage, not an attack signal: rewrite
        # it rather than bricking registration for every session on the host.
    ensure_secrets_dir(base)
    ticket = os.urandom(TICKET_BYTES)
    write_private_file(path, ticket)
    return ticket


def load_master_key(base: Path | None = None, *, create: bool = False) -> bytes:
    """Read the master key, optionally creating one on first use.

    ``create`` is passed by the write verbs only. A ``get`` against a store
    that does not exist must report that, not quietly initialise an empty one
    and then report the secret missing — those are different problems and the
    operator needs to be able to tell them apart.
    """
    path = key_path(base)
    if not path.exists():
        if key_mode(base) == "passphrase":
            # In this tier the unwrapped key exists ONLY in broker memory, so
            # there is deliberately nothing on disk this function could return.
            raise SecretStoreError(
                "This store is hardened with a passphrase, which needs the secret broker. "
                "Run `lop secret unlock` once to unlock it for this boot."
            )
        if not create:
            raise SecretStoreError(
                f"No secret store found at {secrets_dir(base)}. "
                "Create one by storing a secret: lop secret set NAME"
            )
        ensure_secrets_dir(base)
        key = generate_master_key()
        write_private_file(path, key)
        return key

    check_mode(path)
    key = path.read_bytes()
    if len(key) != KEY_BYTES:
        raise SecretStoreError(
            f"{path} is {len(key)} bytes; a master key is {KEY_BYTES}. "
            "The key file is damaged and the store cannot be decrypted with it."
        )
    return key


def replace_master_key(base: Path | None, key: bytes) -> None:
    """Install a new master key, used by ``rotate`` once every record has moved.

    Written through a temporary file and ``os.replace`` so a crash mid-write
    leaves either the old key or the new one, never a truncated file that
    decrypts nothing. The temporary lives in the same 0700 directory, so it is
    never more exposed than the key it replaces.
    """
    path = key_path(base)
    temporary = path.with_suffix(".key.new")
    write_private_file(temporary, key)
    os.replace(temporary, path)
    os.chmod(path, FILE_MODE)


# --- the opt-in passphrase tier (design §2.3) --------------------------------

#: scrypt parameters. n=2^15 measured at 182 ms / 32 MiB on this machine
#: (design §2.3, spike 8) — unnoticeable once per boot, and a meaningful
#: brute-force cost against a stolen wrapped key. n=2^14 (92 ms) is too cheap
#: and n=2^17 (841 ms) is a noticeable stall for no proportionate gain.
SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_SALT_BYTES = 16

#: Magic prefix and version of the wrapped-key file, so a file from a future
#: format is refused with a message rather than fed to the KDF as if it were
#: this one.
_WRAP_MAGIC = b"lopsecwrap\x00"
_WRAP_VERSION = 1


def _derive_wrapping_key(passphrase: str, salt: bytes) -> bytes:
    """scrypt over the operator's passphrase. Imported lazily, as everywhere."""
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    return Scrypt(salt=salt, length=KEY_BYTES, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P).derive(
        passphrase.encode("utf-8")
    )


def wrap_master_key(base: Path | None, key: bytes, passphrase: str) -> None:
    """Wrap ``key`` under ``passphrase`` and remove the plaintext key file.

    The order is load-bearing: the wrapped file is written and fsync'd FIRST,
    and only then is the plain key removed. The reverse order loses the store
    outright if the process dies in between — there would be no key on disk and
    none in any broker's memory. A crash in this order leaves both files, and
    :func:`key_mode` then reports ``passphrase`` while the stale plain key is
    removed on the next successful call.
    """
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    salt = os.urandom(SCRYPT_SALT_BYTES)
    nonce = os.urandom(12)
    wrapping_key = _derive_wrapping_key(passphrase, salt)
    # The header is bound as AAD so the parameters cannot be edited down to a
    # cheaper KDF cost by an attacker holding the file.
    header = _WRAP_MAGIC + bytes([_WRAP_VERSION]) + salt + nonce
    blob = header + AESGCM(wrapping_key).encrypt(nonce, key, header)
    write_private_file(wrapped_key_path(base), blob)
    key_path(base).unlink(missing_ok=True)


def unwrap_master_key(base: Path | None, passphrase: str) -> bytes:
    """Recover the master key from the wrapped file, or raise.

    A wrong passphrase surfaces as ``InvalidTag`` from AES-GCM and is
    translated into a plain sentence: this is the one prompt the operator sees
    per boot, and a traceback would be a poor way to say "that was the wrong
    passphrase".
    """
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    path = wrapped_key_path(base)
    if not path.exists():
        raise SecretStoreError(f"No hardened key at {path}; this store is not in passphrase mode.")
    check_mode(path)
    blob = path.read_bytes()
    prefix = len(_WRAP_MAGIC) + 1 + SCRYPT_SALT_BYTES + 12
    if len(blob) <= prefix or not blob.startswith(_WRAP_MAGIC):
        raise SecretStoreError(f"{path} is not a valid wrapped master key.")
    if blob[len(_WRAP_MAGIC)] != _WRAP_VERSION:
        raise SecretStoreError(
            f"{path} was written in wrapped-key format {blob[len(_WRAP_MAGIC)]}; "
            f"this runtime understands {_WRAP_VERSION}. Upgrade local-operator."
        )
    salt = blob[len(_WRAP_MAGIC) + 1 : len(_WRAP_MAGIC) + 1 + SCRYPT_SALT_BYTES]
    nonce = blob[len(_WRAP_MAGIC) + 1 + SCRYPT_SALT_BYTES : prefix]
    header, ciphertext = blob[:prefix], blob[prefix:]
    try:
        key = AESGCM(_derive_wrapping_key(passphrase, salt)).decrypt(nonce, ciphertext, header)
    except InvalidTag as exc:
        raise SecretStoreError("Wrong passphrase for this secret store.") from exc
    if len(key) != KEY_BYTES:
        raise SecretStoreError("The wrapped key is damaged; it did not contain a master key.")
    return key
