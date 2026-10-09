"""Resolve this device's forge logins for READ-ONLY fetches, without ever leaking one.

WHY THIS EXISTS, AND WHY IT IS NOT ``network/credentials``. The mesh adapters there
resolve credentials to LEND them to other devices; this resolver only needs its own
device's login for a read on the operator's behalf, so it borrows their discovery
patterns without their delivery machinery (no broker, no env construction, no grant
narrowing). The design (§B.4) names the two ladders:

* **GitHub** — the same ladder ``network/credentials/github.py`` walks, reusing its
  ``read_gh_token`` for ``github.com`` and its off-PATH ``find_gh`` probe for a
  self-hosted (GHES) host, where the token is asked of gh itself with
  ``--hostname <host>``.
* **GitLab** — ``network/credentials/gitlab.py`` is a stub (it pins the names and
  contains no behaviour), so the sibling is built here: ``glab config get token
  --host <host>``, then the ``GITLAB_TOKEN`` store secret. Read-only and
  fetch-time only; the stub's eventual delivery slice can grow its own ladder
  without inheriting this one.

THE RULES THIS MODULE KEEPS, because it is the one place a token exists as a value:

1. **Never printed, logged, persisted, or put in an error.** A token lives in
   process memory (this module's cache) and in the HTTP request header the
   adapter builds. Errors carry a NAME and a remedy — the shape
   ``network/credentials/github.py`` already keeps (''STRUCTURAL only'').
2. **Registered for scrubbing before first use.** Every resolved value is
   registered with the process-wide redaction store
   (``local_operator.mcp.redaction``, which attaches a logging filter to every
   root handler and scrubs its registered values out of records the module does
   not own). That is the AGENTS.md rule — a value must be registered before
   output carrying it can be read — applied to the only sensor that exists
   outside a session: process diagnostics.
3. **Re-resolved on a 401, once.** ``resolve(..., fresh=True)`` bypasses the
   cache; the service calls it when the forge rejects a token, and a second
   rejection degrades the row to ``link_only`` with "credential rejected" —
   never an error wall.
4. **Only an AUTHENTICATED host gets a token.** A URL can name any host; a
   token may only travel to one this device is actually signed in to. The
   per-host CLI probes ARE the membership test — each runs with every
   environment variable whose name ends in ``TOKEN`` stripped
   (:data:`_ENV_TOKEN_NAME`), so a ``gh``/``glab`` answer can only come from
   its own stored login, never from an env var echoing itself back for an
   arbitrary ``--host`` (the vectors review round 1, F1, measured; the family
   deny-list that missed ``OAUTH_TOKEN`` is review round 2, N1 — the rule is
   name-ending, not family-name, because the next variable is the one a
   deny-list misses). The one env exception is by construction:
   ``GH_TOKEN``/``GITHUB_TOKEN`` are github.com's own token and ``GITLAB_TOKEN``
   is gitlab.com's (QA round 1, Q5 — a headless device authenticated by env);
   neither vouches for any other host. The ``GITLAB_TOKEN`` STORE secret is
   likewise canonical-host only. A host that resolves to no credential makes
   ZERO requests and degrades to ``link_only`` with the sign-in remedy.

NO NETWORK HERE. Both reference CLIs are asked for a token only — a local
keychain/config read — so no network child process is spawned and the
``XPC_FLAGS`` DNS trap that bites network-spawning children does not apply.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

#: The resolution outcomes. ``absent`` is the zero-setup state (no login found);
#: ``unusable`` means something exists but cannot be read (no binary AND no file
#: token, a malformed login store); ``rejected`` is set by the CALLER after a
#: 401 that survived a fresh re-resolve. Every one degrades to ``link_only``.
CredentialKind = Literal["absent", "unusable", "rejected"]

#: How long a ``gh``/``glab`` token query may take. The same 10 s the mesh
#: adapter allows its own CLI leg.
_CLI_TIMEOUT_S = 10.0

#: Fallback bin directories for ``glab`` discovery, mirroring
#: ``github.GH_FALLBACK_BIN_DIRS`` for the same reason: a launchd-spawned
#: process gets a minimal PATH where a Homebrew glab is invisible.
_GLAB_FALLBACK_BIN_DIRS: tuple[str, ...] = ("/opt/homebrew/bin", "/usr/local/bin")

#: Environment variables stripped from a CLI probe child: EVERY name ending
#: in ``TOKEN`` (case-insensitive). WHY: ``gh auth token --hostname H`` and
#: ``glab config get token --host H`` echo the corresponding env variable for
#: ANY H, so an unstripped child would make an unknown host look signed-in
#: (measured: ``GITLAB_TOKEN=[redacted] glab config get token --host
#: evil.example`` prints the token). A deny-list of known family names is how
#: the round-1 list missed ``OAUTH_TOKEN`` — one of glab's documented env
#: precedence names — and leaked a token to any ``--host`` a URL named (review
#: round 2, N1). The membership rule is the suffix, not the family. With the
#: strip in place, only the CLI's own stored, per-host login can answer —
#: which is exactly the membership check the fetch needs.
#: ``.*`` is required: the single call site matches with ``.match()`` (an
#: anchored match), so the pattern must span from the START to the suffix —
#: bare ``TOKEN$`` matched nothing at all under ``.match()``.
_ENV_TOKEN_NAME = re.compile(r".*TOKEN$", re.IGNORECASE)

#: Which host each family's canonical env token may speak for. ``GH_TOKEN`` is
#: github.com's own token; it must never vouch for a GHES host, and
#: ``GITLAB_TOKEN`` likewise means gitlab.com only.
_CANONICAL_HOSTS = {"github": "github.com", "gitlab": "gitlab.com"}


class CredentialError(Exception):
    """No usable credential, with a message safe to show (never a value)."""

    def __init__(self, kind: CredentialKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


@dataclass(frozen=True)
class Token:
    """One resolved forge token. ``value`` never appears in a ``repr``/log."""

    host: str
    forge: str
    value: str = field(repr=False)
    #: Which arm served it — ``gh``, ``glab`` or ``secret-store``. Metadata only:
    #: it never reaches a model-visible surface (the design's "no boolean about
    #: whether a credential exists" is about the WIRE; this is for logs that
    #: name a remedy).
    source: str = ""


_CACHE: dict[tuple[str, str], Token] = {}
_CACHE_LOCK = threading.Lock()


def _remember(token: Token) -> Token:
    """Cache AND register the token for scrubbing, before any caller can use it."""
    with _CACHE_LOCK:
        _CACHE[(token.host, token.forge)] = token
    _register_for_scrubbing(token.value)
    return token


def _register_for_scrubbing(value: str) -> None:
    """Register a resolved value with the process-wide redaction store.

    Imported here (not at module top): the store pulls ``variables``/
    ``redaction_shapes`` in, and a caller that never resolves a credential —
    every test that only parses refs, most sessions — should not pay for it.
    Best-effort by contract: scrubbing is defense in depth, and a store that
    refuses to register must not refuse the fetch (the value is still never
    printed by this module's own paths).
    """
    try:
        from local_operator.mcp import redaction

        redaction.register(value)
    except Exception:  # noqa: BLE001 - registration never gates a read
        logger.debug("could not register a forge token for scrubbing", exc_info=True)


def resolve(
    host: str,
    forge: str,
    *,
    home: Path | None = None,
    config_dir: Path | None = None,
    fresh: bool = False,
) -> Token:
    """Resolve one host's token, or raise :class:`CredentialError`.

    ``fresh=True`` skips the in-memory cache — the 401 path, so a re-login or a
    rotation reaches the NEXT attempt rather than the next process.
    """
    key = (host, forge)
    if fresh:
        # A rejected token must not be re-served from memory: the 401 path
        # cleared it here, so the NEXT attempt (not just this one) resolves anew.
        with _CACHE_LOCK:
            _CACHE.pop(key, None)
    else:
        with _CACHE_LOCK:
            cached = _CACHE.get(key)
        if cached is not None:
            return cached
    if forge == "github":
        token = _resolve_github(host, home)
    elif forge == "gitlab":
        token = _resolve_gitlab(host, home, config_dir)
    else:
        raise CredentialError(
            "absent", f"no credential path exists for {forge} (detect-and-link host)"
        )
    return _remember(token)


def _resolve_github(host: str, home: Path | None) -> Token:
    """GitHub's ladder — and the host-authentication gate for GHES.

    ``github.com`` goes through ``read_gh_token`` — its two-step resolution
    (hosts file, then ``gh auth token``) is the tested one, and duplicating it
    here would be a second implementation to drift. When that reports
    ``absent``, an env-only ``GH_TOKEN``/``GITHUB_TOKEN`` still counts: a
    headless device authenticated by environment IS signed in to github.com,
    and only to github.com (QA round 1, Q5).

    A GHES host is served ONLY by the gh CLI's own stored login for that exact
    host: the probe runs with every ``*_TOKEN`` environment variable stripped
    (:func:`_cli_env`), so ``GH_ENTERPRISE_TOKEN`` cannot vouch for a host
    nobody signed in to (review round 1, F1). No login for the host, no
    token — and the caller then makes no request at all.
    """
    from local_operator.network.credentials import github as github_credentials

    if host == "github.com":
        try:
            value = github_credentials.read_gh_token(home)
            return Token(host=host, forge="github", value=value, source="gh")
        except github_credentials.GithubGhError as exc:
            if exc.kind != "absent":
                raise _github_refusal(exc.kind) from exc
        value = _env_token("github")
        if value:
            return Token(host=host, forge="github", value=value, source="env")
        # ASK GH ITSELF when its DEFAULT hosts file has nothing (cross-round
        # finding X2): gh honours ``GH_CONFIG_DIR``, the OS keyring and its own
        # defaults, and its answer can only come from THIS host's stored login
        # — the probe env strips every ``*_TOKEN`` variable, so an env token
        # cannot echo back (the explicit arm above already covers that case,
        # deliberately, for github.com only).
        exe = github_credentials.find_gh(home)
        if exe is not None:
            try:
                value = _cli_token(
                    [exe, "auth", "token", "--hostname", "github.com"],
                    remedy="sign in with the gh CLI (`gh auth login`)",
                    forge="github",
                    allow_empty=True,
                )
            except CredentialError:
                # A gh that cannot produce a github.com token here means the
                # same thing as no login: degrade to the sign-in remedy (the
                # F1 contract), never an error wall.
                value = ""
            if value:
                return Token(host=host, forge="github", value=value, source="gh")
        raise _github_refusal("absent")

    exe = github_credentials.find_gh(home)
    if exe is None:
        raise _github_refusal("absent")
    value = _cli_token(
        [exe, "auth", "token", "--hostname", host],
        remedy=f"sign in with the gh CLI (`gh auth login --hostname {host}`)",
        forge="github",
        # An empty answer for THIS host is the absent signal (the glab
        # probe's contract); a refusal with output is still "unusable".
        allow_empty=True,
    )
    if not value:
        raise _github_refusal("absent")
    return Token(host=host, forge="github", value=value, source="gh")


def _env_token(forge: str) -> str:
    """The CANONICAL host's env token, or ``""``.

    ``GH_TOKEN``/``GITHUB_TOKEN`` for github.com, ``GITLAB_TOKEN`` for
    gitlab.com — consulted only for those hosts, so an env variable can never
    vouch for another (review round 1, F1; QA round 1, Q5).
    """
    names = ("GH_TOKEN", "GITHUB_TOKEN") if forge == "github" else ("GITLAB_TOKEN",)
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _github_refusal(kind: str) -> CredentialError:
    if kind == "absent":
        return CredentialError(
            "absent",
            "Can't refresh — sign in with the gh CLI (`gh auth login`), then refresh.",
        )
    return CredentialError(
        "unusable",
        "Can't refresh — the gh CLI login could not be read on this device; "
        "sign in again with `gh auth login`, then refresh.",
    )


def _resolve_gitlab(host: str, home: Path | None, config_dir: Path | None) -> Token:
    """GitLab's ladder — glab's own per-host login first, canonical env and
    store secret LAST and ONLY for gitlab.com.

    Order is the design's (``glab config get token --host H`` is named as the
    sibling of ``gh auth token``): the CLI login needs no setup beyond what a
    user of ``glab`` already did, and the probe runs with every ``*_TOKEN``
    environment variable stripped (:func:`_cli_env`), so glab can only answer
    from its own stored config — a stored login for ``H`` IS the proof that
    ``H`` is authenticated (review round 1, F1: before the strip,
    ``$GITLAB_TOKEN`` came back for ANY ``--host``). The store secret and the
    env arm are canonical-host only, so a self-hosted instance with no glab
    login degrades to link_only with the sign-in remedy rather than borrowing
    gitlab.com's credential.
    """
    exe = _find_glab(home)
    cli_error: CredentialError | None = None
    if exe is not None:
        try:
            value = _cli_token(
                [exe, "config", "get", "token", "--host", host],
                remedy=(f"sign in with the glab CLI (`glab auth login --hostname {host}`)"),
                forge="gitlab",
                # glab exits 0 with empty output for an unknown host: empty stdout
                # IS the absent signal here, not a failure.
                allow_empty=True,
            )
        except CredentialError as exc:
            cli_error = exc
            value = ""
        if value:
            return Token(host=host, forge="gitlab", value=value, source="glab")
    if host == _CANONICAL_HOSTS["gitlab"]:
        value = _env_token("gitlab")
        if value:
            return Token(host=host, forge="gitlab", value=value, source="env")
        value = _store_secret(config_dir)
        if value:
            return Token(host=host, forge="gitlab", value=value, source="secret-store")
    if cli_error is not None:
        # The CLI was present and refused: report the real diagnosis rather than
        # the vaguer ''absent'' that only the no-login path should carry.
        raise cli_error
    raise CredentialError(
        "absent",
        "Can't refresh — sign in with the glab CLI (`glab auth login`), then refresh.",
    )


def _find_glab(home: Path | None) -> str | None:
    """``glab`` off-PATH too, the ``find_gh`` probe's shape at a smaller scale."""
    found = shutil.which("glab")
    if found:
        return found
    root = Path.home() if home is None else home
    for directory in (root / ".local" / "bin", *(Path(p) for p in _GLAB_FALLBACK_BIN_DIRS)):
        found = shutil.which("glab", path=str(directory))
        if found:
            return found
    return None


def _cli_token(
    argv: list[str],
    *,
    remedy: str,
    forge: str,
    allow_empty: bool = False,
) -> str:
    """Run one token query and return its single-line stdout, or raise.

    STRUCTURAL only: stdout is captured and never logged; the refusal names the
    remedy and the forge, never the output. A multi-line or oversized answer is
    refused exactly as the mesh adapter refuses one for ``gh``.
    """
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_CLI_TIMEOUT_S,
            env=_cli_env(),
        )
    except Exception as exc:  # noqa: BLE001 - a timeout or spawn failure is "unusable"
        raise CredentialError(
            "unusable",
            f"Can't refresh — the {forge} CLI could not be asked ({exc.__class__.__name__}).",
        ) from exc
    value = proc.stdout.strip()
    if proc.returncode != 0 or "\n" in value or len(value) > 512:
        raise CredentialError("unusable", f"Can't refresh — {remedy}, then refresh.")
    if not value and not allow_empty:
        raise CredentialError("absent", f"Can't refresh — {remedy}, then refresh.")
    return value


def _cli_env() -> dict[str, str]:
    """The child environment for a token query: prompts off, colour off, and
    EVERY token-valued variable stripped.

    ``gh auth token --hostname H`` and ``glab config get token --host H``
    print the corresponding environment variable for ANY ``H`` (measured), so
    an unstripped child would hand this module a token for a host the operator
    never signed in to — the F1 vector. With the strip, the child can only
    answer from its own stored, per-host login, which is the membership test
    itself. The canonical env tokens are read by :func:`_env_token` in THIS
    process, never through a child.
    """
    env = {name: value for name, value in os.environ.items() if not _ENV_TOKEN_NAME.match(name)}
    env.update(
        {
            "GH_PROMPT_DISABLED": "1",
            "GLAB_PROMPT_DISABLED": "1",
            "NO_COLOR": "1",
        }
    )
    return env


def _store_secret(config_dir: Path | None) -> str:
    """The ``GITLAB_TOKEN`` store secret's value, or ``""`` when not configured.

    Mirrors ``github.read_token_secret``: the announced retrieval path
    (``secrets.access.retrieve_secret``), a name pinned by the gitlab stub, and
    nothing that could print the value. Any failure — no store, no secret, an
    unreadable store — is "not configured here" for THIS read-only resolver:
    unlike a lending ladder there is no security boundary to protect by
    refusing, only a fetch that degrades to link-only.
    """
    try:
        # Resolved OUTSIDE the store try-block: ``SecretStoreError`` is caught
        # below, and an import that lands inside the try leaves the name
        # unbound at the ``except`` (pyright's reportPossiblyUnboundVariable —
        # a real hazard, not a style note).
        from local_operator.secrets import access
        from local_operator.secrets.errors import SecretStoreError
        from local_operator.secrets.keys import store_path
    except Exception:  # noqa: BLE001 - no secrets package, nothing configured here
        return ""
    try:
        base = config_dir
        if not store_path(base).exists():
            return ""
        raw = access.retrieve_secret("GITLAB_TOKEN", base)
    except SecretStoreError:
        return ""
    except Exception:  # noqa: BLE001 - any store failure is "not configured" here
        logger.debug("GITLAB_TOKEN could not be read", exc_info=True)
        return ""
    value = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, bytes) else ""
    if not value or "\n" in value or len(value) > 512:
        return ""
    return value


def _reset_for_tests() -> None:
    """Drop the memoized tokens. Tests that fake resolution call this."""
    with _CACHE_LOCK:
        _CACHE.clear()


__all__ = ["CredentialError", "CredentialKind", "Token", "resolve"]
