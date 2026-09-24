"""Read the retired plaintext ``credentials.env`` — the migration's own reader.

This module exists for exactly ONE consumer: ``lop secret migrate-env``
(:func:`local_operator.secrets.handlers._migrate_env`). The plaintext credential
file is no longer a credential SOURCE — every reader resolves the encrypted
store first and the process environment second, and the transition leg that fell
back to the file was removed in PR2a — but an operator who has not yet migrated
still has real keys sitting in it, greppable, and this is the reader that moves
them into the store.

It was extracted from the deleted ``local_operator.credentials`` module rather
than left inside it for one reason: for a module to be DELETED without the
filename it is named after surviving inside it. Its name states what it is — a
legacy-format reader on its way out — where a method on a class called
``CredentialManager`` would read as a live credential mechanism.

**The policies here were each paid for in review rounds and are not re-derivable
from the code** (see :func:`_open_regular_store` and :func:`read_credentials`):

* a migration MUST NOT create the file it is emptying — hence ``ENOENT`` is
  ``{}`` and nothing here writes;
* a migration MUST NOT print a value — hence the values come back as
  ``SecretStr``, whose repr masks, rather than as ``str``;
* a blank value is NOT a credential, so a blank line does not get migrated as
  though it were a secret.
"""

import errno
import os
import stat
from pathlib import Path
from typing import Dict

from pydantic import SecretStr

#: Name of the file used to store credentials in .env format. The file is
#: retired (PR2c deletes it on this host); the NAME is still needed to locate it
#: for the migration.
CREDENTIALS_FILE_NAME: str = "credentials.env"


def _open_regular_store(path: str, flags: int) -> int:
    """Validate the opened object before a buffered reader can consume it.

    A path probe races replacement and hides diagnostic errnos on newer Python.
    Opening nonblocking first also avoids waiting for a FIFO writer; checking
    that SAME descriptor rejects devices such as /dev/zero before an unbounded
    read. Regular-file symlinks remain supported. The opener owns the descriptor
    until it returns it to ``open``, including every validation failure.
    """
    fd = os.open(path, flags | getattr(os, "O_NONBLOCK", 0))
    try:
        mode = os.fstat(fd).st_mode
        if not stat.S_ISREG(mode):
            code = errno.EISDIR if stat.S_ISDIR(mode) else errno.EINVAL
            raise OSError(code, "Credential store must be a regular file", path)
        if hasattr(os, "O_NONBLOCK"):
            os.set_blocking(fd, True)
        return fd
    except BaseException:
        os.close(fd)
        raise


def read_credentials(config_dir: Path, *, non_empty: bool = True) -> Dict[str, SecretStr]:
    """Credential KEY→VALUE pairs from ``<config_dir>/credentials.env``.

    Read WITHOUT creating anything. A migration that ran the file's own
    constructor would CREATE an empty ``credentials.env`` as a side effect of
    reading it, silently undoing itself on a host that had already been cleaned
    up — the defect PR2a closed for every construction site, and the one thing
    this reader must keep true now that it is the last thing to touch the file.

    An ABSENT file is ``{}``: that is the expected END state of the migration,
    not an error. Every other errno is raised for the caller to report as
    degraded, because "no credentials recorded" and "could not look" are
    different answers: ``EACCES`` on the file or a directory on the way to it,
    ``ELOOP`` from a file symlinked to itself, ``ENOTDIR`` when the config root
    is not a directory at all, ``EISDIR`` for a directory in the file's place.

    **UTF-8 explicitly, with ``surrogateescape``.** Text mode without an
    ``encoding`` uses the platform default, which on Windows is the ANSI code
    page (cp1252 on an en-US install) — so a file this project's own POSIX
    builds write as UTF-8 was decoded as cp1252 there, silently turning a
    non-ASCII credential into mojibake. Silently wrong is the worst outcome for
    a credential read, so the encoding is named rather than inherited.
    ``surrogateescape`` keeps the read lossless for bytes that are not valid
    UTF-8 (a file written by an older build, or hand-edited).

    ``non_empty`` drops a key recorded with an empty value: a blank string is not
    a credential, and migrating one would move nothing across while looking like
    it had moved a secret. The comparison is on the REVEALED string, not on the
    ``SecretStr`` object — pydantic defines no ``__bool__`` on ``SecretStr``, so
    an object's default truthiness would make every blank-valued key look real.
    """
    values: Dict[str, SecretStr] = {}
    try:
        with open(
            config_dir / CREDENTIALS_FILE_NAME,
            "r",
            encoding="utf-8",
            errors="surrogateescape",
            opener=_open_regular_store,
        ) as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    values[key] = SecretStr(value)
    except OSError as exc:
        # ABSENT is the expected END state of the migration, so it is `{}` rather
        # than an error — ``lop secret migrate-env`` runs routinely on a host that
        # has already been cleaned up, and must report "no credentials to migrate"
        # instead of raising. Every OTHER errno is re-raised for the caller to
        # report as degraded: EACCES on the file or a directory on the way to it,
        # ELOOP from a file symlinked to itself, ENOTDIR when the config root is
        # not a directory at all, EISDIR for a directory in the file's place.
        # That split is copied verbatim from the classmethod this was extracted
        # from (``CredentialManager.read_credentials``); dropping it here would
        # silently turn a permissions problem into "nothing to migrate".
        if exc.errno == errno.ENOENT:
            return {}
        raise

    if not non_empty:
        return values
    return {key: value for key, value in values.items() if value.get_secret_value()}
