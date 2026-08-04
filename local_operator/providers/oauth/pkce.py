"""PKCE helpers (RFC 7636).

omp uses 96 random bytes → base64url verifier, SHA-256 challenge (S256).
The verifier entropy far exceeds the RFC minimum; keep it identical to omp so
tokens minted by either harness are interchangeable.
"""

from __future__ import annotations

import base64
import hashlib
import secrets

VERIFIER_RANDOM_BYTES = 96


def _b64url(data: bytes) -> str:
    """base64url without padding, as RFC 7636 requires."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def create_pkce_verifier() -> str:
    """Generate a fresh code verifier: 96 random bytes, base64url-encoded."""
    return _b64url(secrets.token_bytes(VERIFIER_RANDOM_BYTES))


def create_pkce_challenge(verifier: str) -> str:
    """S256 challenge for ``verifier``: base64url(SHA-256(verifier))."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return _b64url(digest)


def create_pkce_pair() -> tuple[str, str]:
    """Return ``(verifier, challenge)``."""
    verifier = create_pkce_verifier()
    return verifier, create_pkce_challenge(verifier)
