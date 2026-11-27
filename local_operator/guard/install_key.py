"""The Radient Guard install key: created lazily, only inside an admitted accept.

WHY (Guard RFC v0.9 §5.1). A Guard consent grant is signed by a per-install
Ed25519 key so the server can refuse a JWT-only, cookie or Console request
(§5.2.7). The CTO decision of 2026-11-27 rules out hardware, attestation and
per-action ceremonies, so this is a SOFTWARE key in the OS credential store.

THE ONE RULE THIS MODULE EXISTS TO KEEP. Nothing here runs at sign-in, at
start-up, on idle or for an account that is not on Radient Team. The only
caller of :func:`get_or_create` is ``guard.consent.GuardConsentHost._commit``,
which runs after ``admit_increasing`` admitted a person's accept. A non-Team
install therefore never writes the credential store (the L2a QA row "zero
keychain writes"), and :func:`load` alone never creates anything.

WHERE THE KEY LIVES, in order (§5.1):

* macOS: the login Keychain, service :data:`KEYCHAIN_SERVICE`, account = the
  install id, written through ``security -i`` on STDIN so the key is never on
  a command line (the ``mobile/auth.py`` pattern);
* Linux with libsecret's ``secret-tool``: the Secret Service, same service;
* otherwise (a container, a server, Windows today): a 0600 file at
  ``<config_dir>/guard/install.key`` in a 0700 directory. DPAPI on Windows is
  NOT implemented in this slice; Windows takes the file. Stated, not hidden:
  the RFC lists DPAPI before the file, and it is deferred to the Windows
  key-store follow-up that must land before A9 activation (RFC §17').

WHAT PROTECTS THE FILE ON WINDOWS (security round 1, S-5). ``os.chmod`` there
only toggles the read-only attribute, so the "0600" modes above mean nothing.
What actually guards the file is the ACL the config directory inherits from
the user profile (the user, SYSTEM and Administrators). Against residual (i) --
a same-user process -- that is equivalent to DPAPI, which is also same-user;
DPAPI would add only protection of an offline disk image.

A STORE FAILURE IS NEVER "NO KEY" (security S-4 / review MINOR-2). Only the
store's own not-found answer reads as absent (Keychain exit 44,
``errSecItemNotFound``; ``secret-tool`` exit 1 with nothing on stderr). A locked
Keychain, a cancelled access prompt or a D-Bus failure raises
:class:`GuardKeyStoreError`; minting a fresh key there would orphan the enrolled
device and spend the server's enrol quota.

CREATE IS CREATE-IF-ABSENT (security S-3). Two first accepts can race; the
file store links its temp file into place (fails if present), the Keychain
adds without ``-U`` (fails if present), and :func:`get_or_create` serializes on
a lock file and re-reads after its write, so both callers end on ONE key.
Linux is the exception to "the store refuses a duplicate": libsecret's
``secret-tool store`` REPLACES an existing item for the same attributes, so
there create-if-absent rests entirely on the ``guard/key.lock`` ``flock`` plus
the re-read. That is sufficient because one install is one config dir is one
lock file; a second writer outside that lock (another config dir pointing at
the same Secret Service item) is not a shape lop creates (L2a security nit).

The model never receives the key: it is not in env, tool results, logs, Team
Bridge or the decision journal, and the built-in ``guard-credential``
guardrail (``guardrails/check.py``) refuses a model tool call that names the
service or the file. That guardrail is defence in depth only (§5.7a); a
same-user process can still read the store, which is the RFC's stated
residual (i).

Rigs and tests set ``LOCAL_OPERATOR_GUARD_KEY_STORE=file`` (or pass a backend)
so no test can ever raise a Keychain prompt on the operator's screen.
"""

from __future__ import annotations

import contextlib
import os
import re
import secrets
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final, Protocol

#: The OS credential-store service name the RFC fixes (§5.1). The guardrail
#: keys on this exact string.
KEYCHAIN_SERVICE: Final = "com.radient.guard"

#: The fallback key file's name under ``<config_dir>/guard/``. Distinctive on
#: purpose: the guardrail refuses any model tool call naming it.
KEY_FILENAME: Final = "install.key"

#: Rigs force the file backend with this variable (never the Keychain).
BACKEND_ENV: Final = "LOCAL_OPERATOR_GUARD_KEY_STORE"

_TIMEOUT_S: Final = 10.0

#: Absolute, so a PATH-shadowed ``security`` never receives the key on stdin.
SECURITY_BIN: Final = "/usr/bin/security"

#: ``security`` exits 44 (errSecItemNotFound) when no such item exists.
_KEYCHAIN_NOT_FOUND: Final = 44

#: ``guard.consent`` mints install ids as ``uuid4().hex``; anything else is
#: refused before it can reach a ``security -i`` script line.
_INSTALL_ID: Final = re.compile(r"[0-9a-f]{32}")


def _check_install_id(install_id: str) -> str:
    if not isinstance(install_id, str) or _INSTALL_ID.fullmatch(install_id) is None:
        raise GuardKeyStoreError("the Guard install id is malformed")
    return install_id


class GuardKeyStoreError(Exception):
    """The credential store could not be read or written."""


class KeyStore(Protocol):
    """Where one install's private key bytes live. Hex text in, hex text out."""

    name: str

    def load(self, install_id: str) -> str | None: ...

    def store(self, install_id: str, secret_hex: str) -> None: ...

    def delete(self, install_id: str) -> None: ...


def key_file(config_dir: Path | str) -> Path:
    return Path(config_dir) / "guard" / KEY_FILENAME


def lock_file(config_dir: Path | str) -> Path:
    """The lock :func:`get_or_create` serializes key creation on (no secret in it)."""

    return Path(config_dir) / "guard" / "key.lock"


class FileKeyStore:
    """A 0600 file in a 0700 directory: the last-resort store (§5.1)."""

    name = "file"

    def __init__(self, config_dir: Path | str) -> None:
        self._path = key_file(config_dir)

    def load(self, install_id: str) -> str | None:
        try:
            text = self._path.read_text(encoding="utf-8")
        except (FileNotFoundError, NotADirectoryError):
            return None
        except OSError as exc:
            raise GuardKeyStoreError("the Guard install key file is unreadable") from exc
        owner, _, value = text.strip().partition(":")
        return value or None if owner == install_id else None

    def store(self, install_id: str, secret_hex: str) -> None:
        directory = self._path.parent
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        # A per-writer temp (two racing creators never share one), O_EXCL at
        # 0600: the key is never readable at a wider mode, not even briefly.
        tmp = directory / f"{KEY_FILENAME}.{secrets.token_hex(8)}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            try:
                os.write(fd, f"{install_id}:{secret_hex}\n".encode("ascii"))
                os.fsync(fd)
            finally:
                os.close(fd)
            try:
                # Create-if-absent: ``link`` fails when the final file exists,
                # so a racing creator for THIS install never overwrites the key
                # the other one may already have enrolled.
                os.link(tmp, self._path)
            except FileExistsError:
                if self.load(install_id) is None:
                    # A key file left by a previous install id (its store was
                    # reset): nothing enrolled can still use it; replace it.
                    os.replace(tmp, self._path)
            except OSError:
                # A filesystem with no hard links (some FUSE/FAT mounts):
                # ``get_or_create``'s lock and re-read are the race guard there.
                os.replace(tmp, self._path)
        finally:
            tmp.unlink(missing_ok=True)

    def delete(self, install_id: str) -> None:
        if self.load(install_id) is not None:
            self._path.unlink(missing_ok=True)


class KeychainKeyStore:
    """The macOS login Keychain through ``/usr/bin/security``."""

    name = "keychain"

    def load(self, install_id: str) -> str | None:
        _check_install_id(install_id)
        try:
            result = subprocess.run(
                [
                    SECURITY_BIN,
                    "find-generic-password",
                    "-s",
                    KEYCHAIN_SERVICE,
                    "-a",
                    install_id,
                    "-w",
                ],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GuardKeyStoreError("the Keychain could not be reached") from exc
        if result.returncode == _KEYCHAIN_NOT_FOUND:
            return None
        if result.returncode != 0:
            # Locked, ACL prompt cancelled, no GUI session: NOT "absent".
            raise GuardKeyStoreError("the Keychain could not be read")
        return result.stdout.strip() or None

    def store(self, install_id: str, secret_hex: str) -> None:
        # STDIN, never argv: ``ps`` must not show the key. No ``-U``: an
        # existing item is never overwritten (``get_or_create`` re-reads it).
        _check_install_id(install_id)
        script = f'add-generic-password -s {KEYCHAIN_SERVICE} -a "{install_id}" -w "{secret_hex}"\n'
        try:
            proc = subprocess.run(
                [SECURITY_BIN, "-i"],
                input=script,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GuardKeyStoreError("the Keychain could not be reached") from exc
        if proc.returncode != 0:
            raise GuardKeyStoreError("the Keychain refused the Guard install key")

    def delete(self, install_id: str) -> None:
        try:
            subprocess.run(
                [SECURITY_BIN, "delete-generic-password", "-s", KEYCHAIN_SERVICE, "-a", install_id],
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GuardKeyStoreError("the Keychain could not be reached") from exc


class SecretToolKeyStore:
    """The Secret Service through libsecret's ``secret-tool`` (value on STDIN)."""

    name = "secret-service"

    def __init__(self, binary: str) -> None:
        self._binary = binary

    def _run(self, args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                [self._binary, *args],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_S,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GuardKeyStoreError("the Secret Service could not be reached") from exc

    def load(self, install_id: str) -> str | None:
        result = self._run(["lookup", "service", KEYCHAIN_SERVICE, "account", install_id])
        if result.returncode == 0:
            return result.stdout.strip() or None
        if result.returncode == 1 and not (result.stderr or "").strip():
            # libsecret's not-found: exit 1, silent. A D-Bus or unlock failure
            # also exits 1 but says why on stderr -- that is an error, not absent.
            return None
        raise GuardKeyStoreError("the Secret Service could not be read")

    def store(self, install_id: str, secret_hex: str) -> None:
        result = self._run(
            [
                "store",
                "--label",
                "Radient Guard install key",
                "service",
                KEYCHAIN_SERVICE,
                "account",
                install_id,
            ],
            stdin=secret_hex,
        )
        if result.returncode != 0:
            raise GuardKeyStoreError("the Secret Service refused the Guard install key")

    def delete(self, install_id: str) -> None:
        self._run(["clear", "service", KEYCHAIN_SERVICE, "account", install_id])


def default_store(config_dir: Path | str) -> KeyStore:
    """The store for THIS machine; see the module docstring for the order."""

    if os.environ.get(BACKEND_ENV, "").strip().lower() == "file":
        return FileKeyStore(config_dir)
    if sys.platform == "darwin":
        return KeychainKeyStore()
    if not sys.platform.startswith("win"):
        binary = shutil.which("secret-tool")
        if binary:
            return SecretToolKeyStore(binary)
    return FileKeyStore(config_dir)


def _private_from_hex(secret_hex: str):  # -> Ed25519PrivateKey
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    try:
        raw = bytes.fromhex(secret_hex)
    except ValueError as exc:
        raise GuardKeyStoreError("the stored Guard install key is malformed") from exc
    if len(raw) != 32:
        raise GuardKeyStoreError("the stored Guard install key is malformed")
    return Ed25519PrivateKey.from_private_bytes(raw)


def load(store: KeyStore, install_id: str):  # -> Ed25519PrivateKey | None
    """The existing key, or ``None``. NEVER creates one."""

    secret_hex = store.load(install_id)
    return None if secret_hex is None else _private_from_hex(secret_hex)


def get_or_create(  # -> tuple[Ed25519PrivateKey, bool]
    store: KeyStore, install_id: str, *, lock_path: Path | None = None
):
    """The install key, generating and storing it on first use; ``(key, created)``.

    Call ONLY from inside an admitted accept (module docstring). The private
    bytes go straight from the generator into the store and are never logged.

    ``lock_path`` serializes creators across processes (POSIX ``flock``; a
    no-op where ``fcntl`` is absent, which is the file backend's Windows case
    and that backend's ``link`` is already create-if-absent). After writing,
    the key is RE-READ: whichever key the store holds is the one returned, so
    a creator that lost a race signs with the winner's key, never its own.
    """

    with _creation_lock(lock_path):
        existing = load(store, install_id)
        if existing is not None:
            return existing, False
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        key = Ed25519PrivateKey.generate()
        raw = key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        try:
            store.store(install_id, raw.hex())
        except GuardKeyStoreError:
            # The add may have failed BECAUSE a racing creator's item exists.
            winner = load(store, install_id)
            if winner is None:
                raise
            return winner, False
        stored = load(store, install_id)
        if stored is None:
            raise GuardKeyStoreError("the Guard install key did not persist")
        if public_raw(stored) != public_raw(key):
            return stored, False
        return key, True


@contextlib.contextmanager
def _creation_lock(lock_path: Path | None) -> Iterator[None]:
    if lock_path is None:
        yield
        return
    try:
        import fcntl
    except ImportError:  # Windows: see get_or_create
        yield
        return
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(lock_path.parent, 0o700)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def public_raw(key) -> bytes:
    from cryptography.hazmat.primitives import serialization

    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)


__all__ = [
    "BACKEND_ENV",
    "KEYCHAIN_SERVICE",
    "KEY_FILENAME",
    "FileKeyStore",
    "GuardKeyStoreError",
    "KeyStore",
    "KeychainKeyStore",
    "SecretToolKeyStore",
    "default_store",
    "get_or_create",
    "key_file",
    "load",
    "lock_file",
    "public_raw",
]
