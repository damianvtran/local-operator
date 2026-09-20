"""Password and cookie auth for the mobile daemon.

One owner, one password, stateless signed cookies — the omp mobile model,
which fits a personal control plane exactly:

- **The password lives in the platform's own secret store**, never on a command
  line or in a log, and ``LOP_MOBILE_PASSWORD`` overrides for containers and
  foreground dev runs. An existing password is kept on reinstall so rotation
  never happens behind the user's back; ``lop mobile password`` rotates it
  deliberately.

  Where it lives, per platform — one line each, because the reason matters
  more than the table's shape:

  * **macOS** — the login Keychain via ``security``. What the Keychain is for,
    and the original path.
  * **Linux** — the Secret Service via ``secret-tool``; a ``0600`` file when
    libsecret is absent. libsecret is the desktop's keyring, and a bare server
    has no keyring daemon at all, where a file the user owns is the honest
    fallback (the answer ``tunnels.config.private_write`` already gives for the
    connector token).
  * **Windows** — DPAPI (``CryptProtectData``), keyed to the user: the
    platform's equivalent of the Keychain, with no new dependency, no prompt,
    and a blob that is useless to any other user on the box.

  Before this, all three read the Keychain: on Linux and Windows
  ``load_password`` silently answered ``None`` (so ``lop mobile serve`` exited 2
  and told the user to run ``lop mobile install``, which cannot work there) and
  ``store_password`` raised an uncaught ``FileNotFoundError`` at the CLI.
- **Cookies are HMAC-signed with a key derived from the password**, so
  rotation invalidates every live session for free and a daemon restart
  logs nobody out. The value is just an expiry timestamp — there is no
  session table to leak or clean up.
- **The response contract splits browser and API**: a browser GET with no
  valid cookie gets a 303 to the login page (so an installed PWA lands
  somewhere sensible), an ``/api`` call gets a 401 (so the client can react
  instead of parsing HTML). Health checks assert this gate, not mere
  liveness.

Stdlib only: hashlib, hmac, os, secrets, shutil, subprocess, sys.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

from local_operator.paths import config_dir
from local_operator.tunnels.config import private_write

#: The platform, read ONCE into module constants rather than inline in the
#: branches below — and the reason is not tidiness.
#:
#: A test can only flip an INLINE ``sys.platform``/``os.name`` read by patching it
#: process-wide (``auth.sys`` IS the ``sys`` module, ``auth.os`` IS ``os``), and
#: that window is destructive. ``pathlib`` picks ``WindowsPath``/``PosixPath``
#: from ``os.name`` at CALL time, so every ``Path(...)`` built while the patch is
#: live raises ``UnsupportedOperation: cannot instantiate 'WindowsPath' on your
#: system`` — measured 2026-09-18: importing each of the package's modules under
#: ``os.name = "nt"`` on this host failed, 150 of 150 scanned. A module imported
#: inside such a window is therefore either a hard error somewhere unrelated or,
#: when whatever triggered the import swallows it, a HALF-INITIALISED module
#: whose later use raises ``NameError`` in whichever test touches it next. That
#: is the shape of the order-dependent failures the auth platform tests produced
#: in a batch and never standalone.
#:
#: :mod:`local_operator.secrets.keys`, :mod:`local_operator.tools.group_reaper`
#: and :mod:`local_operator.procstate` all hold their platform this way, and
#: their tests patch the constant rather than the module — one convention, not
#: two. Nothing else in this module may read ``sys.platform``/``os.name``
#: directly: it is the ONE place the question is answered.
_PLATFORM = sys.platform
_IS_WINDOWS = os.name == "nt"

#: Keychain coordinates. The service name is stable across reinstalls; the
#: account is the local username, matching how the login Keychain scopes
#: generic passwords.
KEYCHAIN_SERVICE = "lop-mobile"

#: The documented portable override: first class on every platform, not a
#: workaround. It is also the answer for a Linux with no keyring daemon at all.
PASSWORD_ENV = "LOP_MOBILE_PASSWORD"

#: What the Secret Service item is labelled as, so it is findable in a keyring
#: browser. The lookup attributes (below) are what the code keys on.
SECRET_TOOL_LABEL = "Local Operator mobile portal"

#: Cookie name and lifetime. Thirty days matches the omp mobile finding: an
#: installed PWA re-prompting daily is the reason people uninstall control
#: planes.
COOKIE_NAME = "lop_mobile"
COOKIE_TTL_S = 30 * 24 * 3600

#: Accept clock skew on cookie expiry so a phone with a drifting clock does
#: not bounce in and out of auth.
_SKEW_S = 60


class PasswordStoreUnavailable(RuntimeError):
    """This platform has no secret store the daemon can write to.

    Raised by the WRITE paths only. ``load_password`` keeps its "absence is
    ``None``" contract, because every status surface calls it and a store that
    cannot be read is indistinguishable from an unconfigured one — the refusal
    belongs at the point the user asked for a change, not on a status read.

    The message names the OS, the command that works there, and the portable
    override, in that order, because the override is the answer on a machine
    where the store genuinely cannot exist (a container, a headless box with no
    keyring daemon).
    """


def _account() -> str:
    """The store's account key: the login name, spelled as each OS spells it.

    macOS keeps the historical spelling (``USER``, empty when unset) because the
    Keychain items already stored under it must keep resolving. Windows spells
    the same fact ``USERNAME``; reading ``USER`` there would produce an empty
    account and a second, invisible secret.
    """
    if _PLATFORM == "darwin":
        return os.environ.get("USER", "")
    if _IS_WINDOWS:
        return os.environ.get("USERNAME") or os.environ.get("USER") or ""
    return os.environ.get("USER") or ""


def password_file() -> Path:
    """The Linux fallback store: a ``0600`` file under this config root."""
    return config_dir() / "mobile" / "password"


def dpapi_file() -> Path:
    """The Windows store: a DPAPI blob keyed to this user."""
    return config_dir() / "mobile" / "password.dpapi"


def _secret_tool() -> str | None:
    """``secret-tool`` when this Linux has libsecret's CLI, else ``None``.

    Guarding on the BINARY and not on the platform, for the reason
    :mod:`local_operator.supervisors` records: a platform is not a capability.
    A container or a server has no Secret Service *and* no ``secret-tool``, and
    that machine gets the file fallback instead of a failed write.
    """
    if _IS_WINDOWS or _PLATFORM == "darwin":
        return None
    return shutil.which("secret-tool")


def store_description() -> str:
    """Where the password lives on THIS machine, for messages and status lines.

    One function rather than a sentence per caller: the install steps, the
    status surface and the refusal messages all have to agree about where the
    secret is, and they disagreed the moment the store stopped being macOS-only.
    """
    if _PLATFORM == "darwin":
        return f"the login Keychain (service {KEYCHAIN_SERVICE})"
    if _secret_tool():
        return f"the Secret Service keyring (secret-tool, service {KEYCHAIN_SERVICE})"
    if _IS_WINDOWS:
        return f"a DPAPI-protected file ({dpapi_file()})"
    return f"a 0600 file ({password_file()})"


def _load_keychain() -> str | None:
    """The macOS Keychain read. Never raises — absence is ``None``."""
    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                _account(),
                "-w",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    password = result.stdout.strip()
    return password or None


def _load_secret_tool(binary: str) -> str | None:
    """Read the Secret Service item, or ``None``.

    A machine with a keyring daemon that is not reachable from this session (no
    D-Bus, a bare SSH login) exits non-zero here and is reported as "no
    password", which is the truth from this process's point of view.
    """
    try:
        result = subprocess.run(
            [binary, "lookup", "service", KEYCHAIN_SERVICE, "account", _account()],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    password = result.stdout.strip()
    return password or None


def _load_file(path: Path) -> str | None:
    """Read a stored password from ``path``, or ``None``."""
    try:
        password = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return password or None


def _dpapi(protect: bool, payload: bytes) -> bytes:
    """``CryptProtectData``/``CryptUnprotectData`` over ``payload``.

    DPAPI and not a plaintext file: this is the Windows equivalent of the
    Keychain, it needs no dependency beyond ``crypt32``, and the blob is
    decryptable only by this user on this machine (and, per the API, by a
    domain admin). ``use_last_error`` is set so a refusal can be reported with
    Windows' own status code rather than a bare ``False``.
    """
    import ctypes
    from ctypes import wintypes

    class DataBlob(ctypes.Structure):
        _fields_ = (("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char)))

    # ``getattr`` rather than the attributes directly: ``WinDLL`` and
    # ``get_last_error`` exist only on Windows, so a direct reference is a
    # type error on the platforms this file is also imported on. The same
    # spelling as ``tools/eval.py`` and ``procstate.py`` — one way of reaching a
    # Win32 API from a portable module, not three.
    crypt32 = getattr(ctypes, "WinDLL")("crypt32", use_last_error=True)
    kernel32 = getattr(ctypes, "WinDLL")("kernel32", use_last_error=True)
    last_error = getattr(ctypes, "get_last_error")
    buffer = ctypes.create_string_buffer(payload, len(payload))
    source = DataBlob(len(payload), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char)))
    result = DataBlob()
    call = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    if not call(
        ctypes.byref(source), None, None, None, None, 0, ctypes.byref(result)
    ):  # pragma: no cover - Windows only
        raise OSError(last_error(), "the DPAPI call failed")
    try:
        return ctypes.string_at(result.pbData, result.cbData)
    finally:
        kernel32.LocalFree(result.pbData)


def load_password() -> str | None:
    """Resolve the portal password: env override first, then this platform's store.

    Never raises — absence is a ``None`` the caller turns into a first-run flow.
    Order matters: ``LOP_MOBILE_PASSWORD`` is the documented container/dev path
    and must win, and on Linux the keyring is preferred over the file so a user
    who upgraded from the file fallback to libsecret reads ONE answer rather
    than a stale copy.
    """
    env = os.environ.get(PASSWORD_ENV)
    if env:
        return env
    if _PLATFORM == "darwin":
        return _load_keychain()
    binary = _secret_tool()
    if binary:
        stored = _load_secret_tool(binary)
        if stored:
            return stored
    if _IS_WINDOWS:
        blob = dpapi_file()
        try:
            raw = blob.read_bytes()
        except OSError:
            return None
        try:
            decrypted = _dpapi(False, raw).decode("utf-8").strip()
        except Exception:  # noqa: BLE001 — an unreadable store is "no password"
            return None
        return decrypted or None
    return _load_file(password_file())


def store_password(password: str) -> None:
    """Write the password to this platform's store. Replaces any existing one.

    macOS: ``security -i`` over stdin, so the value never touches argv where
    ``ps`` could read it. Linux: ``secret-tool`` over stdin, same reason, else a
    ``0600`` file. Windows: a DPAPI blob under the config root.

    Raises :class:`PasswordStoreUnavailable` when this platform has no store, so
    ``lop mobile install`` fails with a sentence naming the alternative instead
    of a traceback.
    """
    if "\n" in password or "\r" in password:
        # A newline would split a mini-shell command into two (macOS) or write a
        # truncated secret and a second line of junk. Generated passwords can't
        # contain one; env-supplied ones can.
        raise ValueError("password must not contain newlines")
    if _PLATFORM == "darwin":
        _store_keychain(password)
        return
    binary = _secret_tool()
    if binary:
        _store_secret_tool(binary, password)
        return
    if _IS_WINDOWS:
        _store_dpapi(password)
        return
    if not private_write(password_file(), password + "\n"):
        raise PasswordStoreUnavailable(
            "could not write the mobile password to "
            f"{password_file()}; check the directory's permissions"
        )


def _store_keychain(password: str) -> None:
    """The macOS Keychain write, unchanged: ``security -i`` on stdin, ``-U`` semantics."""
    escaped = password.replace("\\", "\\\\").replace('"', '\\"')
    try:
        proc = subprocess.run(
            ["security", "-i"],
            input=(
                f'delete-generic-password -s {KEYCHAIN_SERVICE} -a "{_account()}"\n'
                f'add-generic-password -s {KEYCHAIN_SERVICE} -a "{_account()}" '
                f'-w "{escaped}"\n'
            ),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PasswordStoreUnavailable(
            f"could not reach the Keychain (`security`): {exc}. Set {PASSWORD_ENV} instead."
        ) from exc
    if proc.returncode != 0:
        raise RuntimeError(f"keychain write failed: {proc.stderr.strip()[:200]}")


def _store_secret_tool(binary: str, password: str) -> None:
    """Store through ``secret-tool``; the value goes on STDIN, never argv."""
    try:
        proc = subprocess.run(
            [
                binary,
                "store",
                "--label",
                SECRET_TOOL_LABEL,
                "service",
                KEYCHAIN_SERVICE,
                "account",
                _account(),
            ],
            input=password,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PasswordStoreUnavailable(
            f"could not reach the Secret Service keyring (secret-tool): {exc}. "
            f"Set {PASSWORD_ENV} instead."
        ) from exc
    if proc.returncode != 0:
        # A keyring daemon that cannot be autolaunched (no D-Bus session, a bare
        # server) lands here — name the working alternative rather than the
        # D-Bus error alone.
        raise PasswordStoreUnavailable(
            f"the Secret Service keyring refused the write: {proc.stderr.strip()[:200]}. "
            f"Set {PASSWORD_ENV}, or install a keyring daemon and retry."
        )
    # A plaintext copy beside the keyring would be strictly worse than the
    # keyring; drop one left by an earlier file-backed install.
    password_file().unlink(missing_ok=True)


def _store_dpapi(password: str) -> None:
    """Write a DPAPI blob for this user."""
    try:
        blob = _dpapi(True, password.encode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — no crypt32 on this box means no store
        raise PasswordStoreUnavailable(
            f"the DPAPI call failed on this Windows ({exc}). Set {PASSWORD_ENV} instead."
        ) from exc
    path = dpapi_file()
    try:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(blob)
    except OSError as exc:
        raise PasswordStoreUnavailable(
            f"could not write the DPAPI store at {path}: {exc}. Set {PASSWORD_ENV} instead."
        ) from exc


def generate_password() -> str:
    """A fresh password: URL-safe, 32 bytes of entropy, no shell-hostile
    characters — it will be typed on a phone exactly once per rotation."""
    return secrets.token_urlsafe(24)


def delete_password() -> None:
    """Remove the stored password (uninstall --purge). Absent is success."""
    if _PLATFORM == "darwin":
        subprocess.run(
            [
                "security",
                "delete-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                _account(),
            ],
            capture_output=True,
            timeout=5,
        )
        return
    binary = _secret_tool()
    if binary:
        subprocess.run(
            [binary, "clear", "service", KEYCHAIN_SERVICE, "account", _account()],
            capture_output=True,
            timeout=10,
        )
    for path in (password_file(), dpapi_file()):
        try:
            path.unlink(missing_ok=True)
        except OSError:  # pragma: no cover - a purge that cannot unlink still reports
            continue


def check_password(candidate: str, actual: str) -> bool:
    """Constant-time compare; a login endpoint is the one place a timing
    oracle is worth caring about because it is remote by design."""
    return hmac.compare_digest(candidate.encode(), actual.encode())


# ---------------------------------------------------------------------------
# Signed cookies
# ---------------------------------------------------------------------------


def _cookie_key(password: str) -> bytes:
    # Domain-separated derivation so the password itself is never the HMAC
    # key and a future second use cannot cross-sign.
    return hashlib.sha256(b"lop-mobile-cookie\0" + password.encode()).digest()


def sign_cookie(password: str, now: float | None = None) -> str:
    """Value = ``<expiry>.<hmac-hex>``; expiry is when the cookie DIES."""
    expiry = int((now or time.time()) + COOKIE_TTL_S)
    sig = hmac.new(_cookie_key(password), str(expiry).encode(), hashlib.sha256).hexdigest()
    return f"{expiry}.{sig}"


def verify_cookie(value: str | None, password: str, now: float | None = None) -> bool:
    if not value or "." not in value:
        return False
    expiry_text, sig = value.rsplit(".", 1)
    try:
        expiry = int(expiry_text)
    except ValueError:
        return False
    expected = hmac.new(_cookie_key(password), expiry_text.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return False
    return expiry > (now or time.time()) - _SKEW_S


def basic_auth_header_user() -> str:
    """The login's fixed username, shown on the login page and used by tools
    that speak basic auth to the API. Fixed because there is exactly one
    account: the owner."""
    return "lop"
