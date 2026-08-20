"""Password and cookie auth for the mobile daemon.

One owner, one password, stateless signed cookies — the omp mobile model,
which fits a personal control plane exactly:

- **The password lives in the macOS Keychain** (service ``lop-mobile``,
  account = local username) via the ``security`` CLI, never on a command line
  or in a log. ``LOP_MOBILE_PASSWORD`` overrides for containers and
  foreground dev runs. An existing password is kept on reinstall so rotation
  never happens behind the user's back; ``lop mobile password`` rotates
  deliberately.
- **Cookies are HMAC-signed with a key derived from the password**, so
  rotation invalidates every live session for free and a daemon restart
  logs nobody out. The value is just an expiry timestamp — there is no
  session table to leak or clean up.
- **The response contract splits browser and API**: a browser GET with no
  valid cookie gets a 303 to the login page (so an installed PWA lands
  somewhere sensible), an ``/api`` call gets a 401 (so the client can react
  instead of parsing HTML). Health checks assert this gate, not mere
  liveness.

Stdlib only: hmac, hashlib, secrets, subprocess.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import subprocess
import time

#: Keychain coordinates. The service name is stable across reinstalls; the
#: account is the local username, matching how the login Keychain scopes
#: generic passwords.
KEYCHAIN_SERVICE = "lop-mobile"

#: Cookie name and lifetime. Thirty days matches the omp mobile finding: an
#: installed PWA re-prompting daily is the reason people uninstall control
#: planes.
COOKIE_NAME = "lop_mobile"
COOKIE_TTL_S = 30 * 24 * 3600

#: Accept clock skew on cookie expiry so a phone with a drifting clock does
#: not bounce in and out of auth.
_SKEW_S = 60


def load_password() -> str | None:
    """Resolve the portal password: env override first (containers, dev),
    then the Keychain. Never raises — absence is a ``None`` the caller turns
    into a first-run flow."""
    env = os.environ.get("LOP_MOBILE_PASSWORD")
    if env:
        return env
    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                KEYCHAIN_SERVICE,
                "-a",
                os.environ.get("USER", ""),
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


def store_password(password: str) -> None:
    """Write the password to the Keychain via ``security -i`` over stdin —
    the value never touches argv, where ``ps`` could read it. Replaces any
    existing item (``-U``) so rotation is one call."""
    # ``security -i`` runs a tiny command REPL on stdin; quoting the value for
    # that mini-shell (never for /bin/sh) keeps the password off argv while
    # surviving spaces and quotes in generated passwords.
    escaped = password.replace("\\", "\\\\").replace('"', '\\"')
    proc = subprocess.run(
        ["security", "-i"],
        input=(
            f'delete-generic-password -s {KEYCHAIN_SERVICE} -a "{os.environ.get("USER", "")}"\n'
            f'add-generic-password -s {KEYCHAIN_SERVICE} -a "{os.environ.get("USER", "")}" '
            f'-w "{escaped}"\n'
        ),
        capture_output=True,
        text=True,
        timeout=10,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"keychain write failed: {proc.stderr.strip()[:200]}")


def generate_password() -> str:
    """A fresh password: URL-safe, 32 bytes of entropy, no shell-hostile
    characters — it will be typed on a phone exactly once per rotation."""
    return secrets.token_urlsafe(24)


def delete_password() -> None:
    """Remove the Keychain item (uninstall --purge). Absent is success."""
    subprocess.run(
        [
            "security",
            "delete-generic-password",
            "-s",
            KEYCHAIN_SERVICE,
            "-a",
            os.environ.get("USER", ""),
        ],
        capture_output=True,
        timeout=5,
    )


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
