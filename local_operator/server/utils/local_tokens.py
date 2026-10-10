"""This process's local surface keys: the Phase A access token and the signing key.

WHY THIS EXISTS. Loopback is not a boundary between local accounts (see the
trust model in ``docs/API_FILESYSTEM_BOUNDARIES.md``): on a shared host, any
local account that can reach the daemon's port can call it. The file-serving
RFC's answer is phased (`docs/design/file-serving-and-surface-convergence.md`
§4): a per-boot credential that every surface reads the way the desktop app
already discovers the daemon -- from the serve record -- with **no user
action**, shipped PERMISSIVE first so no mixed-version combination breaks.

PHASE A IS THE CONTRACT THIS MODULE IMPLEMENTS. Nothing here refuses a
request. A presented token is validated and, when valid, counted (so the code
path is genuinely exercised); an invalid or STALE one is treated as absent --
never a refusal in Phase A. Every classification is TOTAL: attacker-shaped
values classify, they never raise -- a raise here would be a 500 on a route
whose contract is "served regardless", and the shape of a raise must not
depend on the serving generation. The one observable side effect is a single
WARNING line per distinct stale generation (WARNING because the daemon's
console logging runs at the default level; a line nobody sees is not a
notice), so the compat window shows stale surfaces without a log sink (see
:func:`observe_credentials`). Phases B/C (settings-gated, later releases)
reuse the same primitives to refuse.

THE KEY SPLIT. One boot draws a 256-bit seed from the OS CSPRNG
(``secrets.token_bytes(32)``) and derives two values with HKDF-SHA256
(RFC 5869 over stdlib ``hmac``/``hashlib``, no new dependency):

* the **access token** surfaces present -- published in the serve record, and
* the **signing key** that mints signed query URLs -- server-side only, never
  published, never logged.

The seed itself is never stored. The record therefore carries exactly what a
local surface needs to read files, and nothing that grants more.

TOKEN FORMAT: ``"<generation>.<secret>"``, where the secret is the unpadded
urlsafe base64 of 32 HKDF bytes. The generation PREFIX is what makes a stale
token classifiable from an invalid one -- a restart both re-derives the secret
(so the old one cannot match) and bumps the generation (so the mismatch is
nameable) -- and staleness is the one distinction Phase A writes to the log.
The format is part of the documented contract (``docs/DESKTOP_API.md``), not
an internal detail: surfaces compare the generation they hold against the
record's ``boot_generation``, and the server classifies without a lookup table.

SIGNED URLS. For ``src=``-loads, which cannot carry a header, the sign
endpoint mints ``/v1/static/<route>?path=...&exp=...&sig=...`` where

    sig = HMAC-SHA256(signing_key, f"{route}\\n{path}\\n{exp}")

with ``exp`` in unix seconds and ``path`` the DECODED query value exactly as
the request carries it (:func:`canonical_signed_bytes` is the one spelling
both minting and verification use). The signature is path-scoped by
construction: swapping the path, the route or ``exp`` changes the signed bytes.
Verification runs BEFORE any filesystem resolution (:func:`verify_signed_url`
is pure string/HMAC work), so an invalid signature can never become a probe of
what exists. Phase A validation only classifies; it does not refuse.

GENERATION COUNTER. ``<serve run dir>/generation`` is a small persisted
counter, bumped once per ANNOUNCED boot (a boot that publishes a record --
see ``server/app.py``; an unannounced embed mints lazily at generation 0 and
publishes nothing, so no surface can hold a token for it). The write keeps the
staged shape the record family uses -- tempfile in the same directory, 0600,
``os.replace`` -- because that private writer belongs to the session runtime
and this artifact is an integer, not a record; the SHAPE is the shared thing.
Two daemons in one config root may collide on a generation (both read the same
last value); the contract needs INEQUALITY after a restart, not strict
monotonicity, and a collision merely classifies a stale token as invalid.

No FastAPI import here: this module is imported by the app, the registry and
their tests, and it stays stdlib-only on purpose.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import re
import secrets
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode

logger = logging.getLogger("local_operator.server.utils.local_tokens")

#: The header a surface presents its access token in. Part of the documented
#: contract (docs/DESKTOP_API.md); namespaced like every other header here.
ACCESS_TOKEN_HEADER = "X-Local-Token"

#: Signed-URL lifetimes: default for previews (Manager ruling, RFC §4.1), and
#: the minting cap. Verification enforces only the expiry; the cap is a
#: MINTING limit, since a signature cannot be forged but a bug in a caller
#: could request forever-urls.
TTL_DEFAULT_S = 600
TTL_MAX_S = 3600

#: The counter file's name inside the serve run directory (``run/serve``).
GENERATION_FILENAME = "generation"

#: RFC 5869's default salt: HashLen zero bytes (an absent salt pads to the
#: same HMAC key, so this is explicit rather than different).
_HKDF_SALT = bytes(32)

_SEED_BYTES = 32
_OKM_BYTES = 32
_TOKEN_INFO = b"lop-serve-access-token-v1"
_SIGNING_INFO = b"lop-serve-signing-key-v1"

#: Strict spelling of the signed query values: lowercase hex, 64 chars (SHA-256
#: digest size), and a digit-only ``exp`` of at most 10 digits. ``bytes.fromhex``
#: alone accepts embedded whitespace and ``str.isdigit`` accepts non-ASCII
#: digits, so the regex is the strictness (a malformed value is classified, it
#: must not raise).
_HEX_RE = re.compile(r"[0-9a-f]{64}")
_EXP_RE = re.compile(r"[0-9]{1,10}")

#: The token generation prefix's strict spelling: digit-only, at most 10 digits
#: (the same bound as ``exp``, far above any real boot count). The bound is
#: TOTALITY, not taste: the value is attacker-supplied on every static GET, and
#: ``int()`` on an unbounded digit string is O(n^2) and raises above 4300 digits
#: (the interpreter's str-digit safety wall) -- a crafted prefix must therefore
#: be rejected BEFORE any conversion, not guarded around it.
_GENERATION_RE = re.compile(r"[0-9]{1,10}")

#: The token secret's strict spelling: the url-safe base64 alphabet (a minted
#: secret is 43 unpadded chars of it; the cap is a bound, not the format).
#: Anything outside this set -- whitespace, controls, non-ASCII, punctuation --
#: was never a real token, so it classifies invalid BEFORE the generation
#: comparison: that keeps malformed shapes generation-independent (no oracle)
#: and keeps ``compare_digest`` from ever seeing a non-ASCII str.
_SECRET_RE = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _parse_token(presented: str) -> tuple[int, str] | None:
    """``(generation, secret)`` for a well-formed token, ``None`` otherwise.

    TOTAL by construction: every ``str`` returns, none raises. Both callers
    (``token_status`` and the observation log) share this one parser so the
    bound cannot be fixed in one place and left open in the other.
    """
    prefix, sep, secret = presented.partition(".")
    if not sep or not _GENERATION_RE.fullmatch(prefix):
        return None
    return int(prefix), secret


def _hkdf_sha256(*, ikm: bytes, info: bytes) -> bytes:
    """RFC 5869 HKDF-SHA256 (extract, expand) over stdlib ``hmac``/``hashlib``."""
    prk = hmac.new(_HKDF_SALT, ikm, hashlib.sha256).digest()
    okm = b""
    block = b""
    counter = 1
    while len(okm) < _OKM_BYTES:
        block = hmac.new(prk, block + info + bytes([counter]), hashlib.sha256).digest()
        okm += block
        counter += 1
    return okm[:_OKM_BYTES]


@dataclass(frozen=True)
class SurfaceKeys:
    """One boot's derived surface credentials. Frozen, and the seed is not kept.

    ``access_token`` is the record's published value; ``signing_key`` never
    leaves the process. The type carries its own validation so the sign route,
    the acceptance hook and the tests all speak one spelling of the contract.
    """

    generation: int
    access_token: str
    signing_key: bytes

    def token_status(self, presented: str) -> str:
        """``"valid"`` / ``"stale"`` / ``"invalid"`` for a presented token.

        TOTAL: every ``str`` classifies and none raises. The value is
        attacker-supplied on every static GET and at the sign door, so a raise
        here is a 500 where the contract says "treated as absent" -- and it was
        also a generation oracle: a non-ASCII secret used to raise only when
        its prefix MATCHED the serving generation. Malformed SHAPES are
        therefore classified before any generation comparison, making their
        outcome independent of the serving generation by construction.

        ``stale`` is a well-formed token whose generation is not the serving
        one -- the class Phase A logs once; ``invalid`` is everything else,
        including a well-formed token with the right generation but the wrong
        secret. Constant-time comparison for the secret (a token guess must
        not be tunable by timing).
        """
        parsed = _parse_token(presented)
        if parsed is None:
            return "invalid"
        generation, secret = parsed
        if not _SECRET_RE.fullmatch(secret):
            # Outside the alphabet a minted secret can have: never a real token,
            # so invalid regardless of generation (and ``compare_digest`` can
            # never see the non-ASCII str it would raise on).
            return "invalid"
        if generation != self.generation:
            return "stale"
        expected_secret = self.access_token.partition(".")[2]
        if secrets.compare_digest(secret, expected_secret):
            return "valid"
        return "invalid"

    def sign(self, route: str, path: str, exp: int) -> str:
        """The hex HMAC for one route + path + expiry (see :func:`canonical_signed_bytes`)."""
        return hmac.new(
            self.signing_key, canonical_signed_bytes(route, path, exp), hashlib.sha256
        ).hexdigest()

    def signature_matches(self, route: str, path: str, exp: int, signature_hex: str) -> bool:
        """Constant-time check of a presented signature for the given triple."""
        return secrets.compare_digest(signature_hex, self.sign(route, path, exp))


def canonical_signed_bytes(route: str, path: str, exp: int) -> bytes:
    """The exact byte string both ends sign and verify (documented; one spelling).

    ``route`` is the route id (``images``/``videos``/``audio``/``html``),
    ``path`` is the decoded query value exactly as the request carries it, and
    ``exp`` is canonicalized through ``int`` so equivalent digit spellings
    (``"0123"`` vs ``"123"``) sign and verify identically.
    """
    return f"{route}\n{path}\n{int(exp)}".encode("utf-8")


def sign_url(
    route: str, path: str, *, ttl_s: int | None = None, now: float | None = None
) -> tuple[str, int]:
    """Mint ``(url, exp)`` for one route + path; ``ttl_s`` defaults to 600.

    The URL is ORIGIN-RELATIVE (``/v1/static/<route>?path=...&exp=...&sig=...``)
    so it can go straight into an ``src``; the surface knows the host it dials.
    ``urlencode`` percent-encodes the path for the wire; the SIGNED bytes are
    always the decoded form, and the server decodes before verifying, so the
    two sides agree without a canonicalisation dialect.
    """
    keys = process_keys()
    exp = int(now if now is not None else time.time()) + (TTL_DEFAULT_S if ttl_s is None else ttl_s)
    signature = keys.sign(route, path, exp)
    query = urlencode({"path": path, "exp": exp, "sig": signature})
    return f"/v1/static/{route}?{query}", exp


def verify_signed_url(
    *,
    route: str,
    path: str,
    exp_raw: str | None,
    sig_raw: str | None,
    now: float | None = None,
) -> bool:
    """Whether a signed URL's ``exp``/``sig`` are valid for this route + path.

    Never raises: every malformed shape is ``False`` (which Phase A treats as
    absent). Pure string/HMAC work by design -- no filesystem, no resolution --
    so calling this before any path work keeps it from becoming an oracle.
    Expiry: ``exp`` earlier than now is invalid; the future is not bounded here
    because minting caps the TTL and a forged ``exp`` cannot carry a signature.
    """
    if exp_raw is None or sig_raw is None:
        return False
    if not _EXP_RE.fullmatch(exp_raw) or not _HEX_RE.fullmatch(sig_raw):
        return False
    exp = int(exp_raw)
    if exp < int(now if now is not None else time.time()):
        return False
    return process_keys().signature_matches(route, path, exp, sig_raw)


_KEYS: SurfaceKeys | None = None
_KEYS_LOCK = threading.Lock()


def mint_process_keys(*, generation: int) -> SurfaceKeys:
    """Draw this process's keys ONCE; later calls return the same object.

    First call wins, so the boot sequence (``server/app.py``, before the record
    is assembled) and any lazy caller cannot disagree about which token this
    process serves -- the record and the validator are always the same object.
    """
    global _KEYS
    with _KEYS_LOCK:
        if _KEYS is None:
            seed = secrets.token_bytes(_SEED_BYTES)
            secret_bytes = _hkdf_sha256(ikm=seed, info=_TOKEN_INFO)
            secret = base64.urlsafe_b64encode(secret_bytes).rstrip(b"=").decode("ascii")
            _KEYS = SurfaceKeys(
                generation=generation,
                access_token=f"{generation}.{secret}",
                signing_key=_hkdf_sha256(ikm=seed, info=_SIGNING_INFO),
            )
        return _KEYS


def process_keys() -> SurfaceKeys:
    """The keys for THIS process; mints lazily (generation 0) if the boot did not.

    A boot that publishes no record (a bare embed, a test app) has no
    generation counter and nothing external holding its token, so generation 0
    is an honest label; the announced boot path always mints first with the
    real counter (see ``mint_process_keys`` callers).
    """
    return mint_process_keys(generation=0)


def bump_generation(path: Path) -> int:
    """Read the serve namespace's boot counter, +1, persist, and return the new value.

    The counter lives beside the records (``run/serve/generation``); a missing
    or unreadable file reads as 0 so a fresh root starts at 1 and a torn file
    cannot wedge a boot. The directory is created 0700 here when the boot gets
    this far first, matching the namespace's permissions story.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    try:
        previous = int(path.read_text(encoding="ascii").strip() or "0")
    except (OSError, ValueError):
        previous = 0
    generation = previous + 1
    _write_staged(path, f"{generation}\n")
    return generation


def _write_staged(target: Path, text: str) -> None:
    """Staged single-file write, 0600 -- the shape the record family uses.

    Kept here rather than imported from ``session/runtime/registry.py`` because
    that writer is private to the session runtime and this artifact is an
    integer, not a record; the shape (tempfile in the SAME directory, explicit
    chmod, ``os.replace``) is what must not diverge, and it does not.
    """
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.chmod(tmp, 0o600)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


#: Phase A audit counters (in-process; no route exposes them). "The path is
#: exercised" made countable: a compat window can assert that surfaces really
#: do present valid credentials before Phase B can be argued for.
_STATS: dict[str, int] = {
    "tokens_valid": 0,
    "tokens_stale": 0,
    "tokens_invalid": 0,
    "signatures_valid": 0,
}

#: Distinct stale generations already logged this process. A cap keeps a caller
#: fabricating generations from turning the log into an unbounded sink; the
#: first few name the real stale surfaces, which is the compat signal.
_STALE_LOGGED: set[int] = set()
_STALE_LOG_CAP = 8


def stats() -> dict[str, int]:
    """A snapshot of the Phase A audit counters (tests and diagnostics)."""
    return dict(_STATS)


def observe_credentials(
    *,
    route: str,
    path: str,
    token: str | None,
    exp_raw: str | None,
    sig_raw: str | None,
) -> None:
    """Classify this request's local credentials for the Phase A audit.

    NEVER raises and never influences the response -- permissiveness is the
    caller's contract (the static routes serve regardless in Phase A; Phases
    B/C will read a hardened version of this result). The token is validated
    and counted; a stale token additionally logs one WARNING per distinct
    generation (WARNING so a daemon without ``LOG_LEVEL`` still shows it).
    Signed-URL params are verified (before any resolution, which is why this
    is called BEFORE ``resolve_servable``) and counted when valid; an invalid
    or expired signature is silently absent in Phase A -- it is not a
    generation mismatch, so the one-log rule does not apply to it.
    """
    keys = process_keys()
    if token:
        status = keys.token_status(token)
        if status == "valid":
            _STATS["tokens_valid"] += 1
        elif status == "stale":
            _STATS["tokens_stale"] += 1
            parsed = _parse_token(token)
            if parsed is not None:  # stale implies it; stated for the reader
                presented_generation = parsed[0]
                if (
                    presented_generation not in _STALE_LOGGED
                    and len(_STALE_LOGGED) < _STALE_LOG_CAP
                ):
                    _STALE_LOGGED.add(presented_generation)
                    # WARNING, not INFO, because the daemon's console logging runs
                    # at LOG_LEVEL's default WARNING -- a line nobody sees is not a
                    # notice, and this line IS the Phase A compat signal (the same
                    # reasoning as server/retire.py's notices to the operator).
                    logger.warning(
                        "local surface token from generation %s presented; this daemon serves "
                        "generation %s (Phase A: treated as absent)",
                        presented_generation,
                        keys.generation,
                    )
        else:
            _STATS["tokens_invalid"] += 1
    if exp_raw is not None or sig_raw is not None:
        if verify_signed_url(route=route, path=path, exp_raw=exp_raw, sig_raw=sig_raw):
            _STATS["signatures_valid"] += 1
