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
  NOT implemented in this slice; Windows takes the 0600 file. Stated, not
  hidden: the RFC lists DPAPI before the file and a later slice can add it.

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

import os
import shutil
import subprocess
import sys
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
        tmp = directory / (KEY_FILENAME + ".tmp")
        # O_EXCL on a fresh temp, then rename: the key is never readable at a
        # wider mode, not even for the instant between create and chmod.
        tmp.unlink(missing_ok=True)
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, f"{install_id}:{secret_hex}\n".encode("ascii"))
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, self._path)

    def delete(self, install_id: str) -> None:
        if self.load(install_id) is not None:
            self._path.unlink(missing_ok=True)


class KeychainKeyStore:
    """The macOS login Keychain through ``/usr/bin/security``."""

    name = "keychain"

    def load(self, install_id: str) -> str | None:
        try:
            result = subprocess.run(
                [
                    "security",
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
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    def store(self, install_id: str, secret_hex: str) -> None:
        # STDIN, never argv: ``ps`` must not show the key. ``-U`` updates in place.
        script = (
            f'add-generic-password -U -s {KEYCHAIN_SERVICE} -a "{install_id}" -w "{secret_hex}"\n'
        )
        try:
            proc = subprocess.run(
                ["security", "-i"], input=script, capture_output=True, text=True, timeout=_TIMEOUT_S
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GuardKeyStoreError("the Keychain could not be reached") from exc
        if proc.returncode != 0:
            raise GuardKeyStoreError("the Keychain refused the Guard install key")

    def delete(self, install_id: str) -> None:
        try:
            subprocess.run(
                ["security", "delete-generic-password", "-s", KEYCHAIN_SERVICE, "-a", install_id],
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
        return (result.stdout.strip() or None) if result.returncode == 0 else None

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


def get_or_create(store: KeyStore, install_id: str):  # -> tuple[Ed25519PrivateKey, bool]
    """The install key, generating and storing it on first use; ``(key, created)``.

    Call ONLY from inside an admitted accept (module docstring). The private
    bytes go straight from the generator into the store and are never logged.
    """

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    existing = load(store, install_id)
    if existing is not None:
        return existing, False
    key = Ed25519PrivateKey.generate()
    raw = key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    store.store(install_id, raw.hex())
    return key, True


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
    "public_raw",
]
