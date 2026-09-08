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


def key_mode(base: Path | None = None) -> str:
    """Which at-rest tier the store is in: ``keyfile`` or ``passphrase``.

    Always ``keyfile`` today. The seam the broker PR fills: ``harden`` writes a
    scrypt-wrapped ``master.key.wrapped`` beside the plain key and removes the
    plain one, at which point this returns ``passphrase`` and callers route the
    unwrap through the broker instead of reading the file. Reported by
    ``lop secret status`` so the operator can see which tier is live.
    """
    if (secrets_dir(base) / "master.key.wrapped").exists():
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
            # Unreachable until the broker PR writes the wrapped key; stated
            # here so the failure is a clear message rather than "no such file".
            raise SecretStoreError(
                "This store is hardened with a passphrase, which needs the secret broker. "
                "Run `lop secret unlock` once the broker is available."
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
