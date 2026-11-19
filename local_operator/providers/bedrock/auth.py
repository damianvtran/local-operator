"""Resolve a Bedrock connection into request headers (bearer or SigV4).

All five connection methods reduce to ONE of two wire schemes (design D2):

* **Bearer** — ``Authorization: Bearer <Bedrock API key>`` for ``api-key``
  (sealed in the encrypted store, or a raw key) and ``env-bearer``
  (``AWS_BEARER_TOKEN_BEDROCK``).
* **SigV4** (service ``bedrock``) over botocore credentials for ``profile``
  (static, SSO, ``aws login``, assume-role, credential_process),
  ``access-key`` (sealed pair) and ``ambient`` (the default chain: env,
  container/AgentCore, web identity, IMDSv2 ...).

Hard rules, each a constraint learned from the design review:

* botocore is imported LAZILY, here only — never from ``providers.registry``
  or ``providers.clients`` (a subprocess test pins it).
* Never shell out and never prompt. ``aws sso login`` is an explicit user
  action (Slice A2); a headless worker fails fast with a remedy instead.
* Credential resolution can hit the network (SSO refresh, IMDS, the container
  endpoint, STS), so the async path runs it in a worker thread under a
  timeout. botocore gives ``credential_process`` no timeout of its own, so a
  hung helper leaks one thread — bounded, and no hang reaches the turn.
* The SAME bytes that are signed are the bytes sent (callers pass
  ``content=``), otherwise SigV4's payload hash would not match.
* Sessions are cached per connection so ``RefreshableCredentials`` keep their
  refresh state across turns instead of re-reading SSO caches every request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from local_operator.providers.bedrock import connection as bconn
from local_operator.providers.bedrock.connection import Connection

if TYPE_CHECKING:
    from local_operator.providers.failover import ProviderError

#: Upper bound on one credential resolution (SSO refresh / IMDS / STS).
CREDENTIAL_TIMEOUT_S = 15.0

#: SigV4 signing name for both ``bedrock`` and ``bedrock-runtime`` endpoints.
SIGNING_NAME = "bedrock"

_SESSION_CACHE_MAX = 16
_session_lock = threading.Lock()
_sessions: "OrderedDict[str, Any]" = OrderedDict()

_log = logging.getLogger(__name__)
_botocore_quieted = False


class _BotocoreToDebug(logging.Handler):
    """Re-log botocore records on our logger at DEBUG, message only.

    The traceback is dropped on purpose: botocore's mandatory-refresh failure
    logs ``exc_info=True`` (~40 lines), and the SAME failure already reaches
    the user as one classified ``ProviderError`` sentence.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            _log.debug("botocore %s: %s", record.levelname, record.getMessage())
        except Exception:  # noqa: BLE001 - logging must never break resolution
            pass


def _quiet_botocore_logging() -> None:
    """Keep botocore's own log lines off the user's terminal (QA #4270 Q3).

    botocore logs at INFO on every resolution ("Found credentials in shared
    credentials file: ...") and at WARNING with a traceback when a refresh
    fails. Under ``lop login`` / ``lop exec`` the root handler writes those to
    stderr between our own lines. Bedrock is the only botocore user in this
    process, so the ``botocore`` logger tree is detached from the root
    (``propagate=False``), raised to WARNING (no INFO record is even built)
    and forwarded to our DEBUG log for diagnosis. Idempotent; it runs before
    the first botocore session is created.
    """
    global _botocore_quieted
    if _botocore_quieted:
        return
    logger = logging.getLogger("botocore")
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    if not any(isinstance(h, _BotocoreToDebug) for h in logger.handlers):
        logger.addHandler(_BotocoreToDebug())
    _botocore_quieted = True


@dataclass(frozen=True)
class ResolvedAuth:
    """What a request needs: a bearer token XOR frozen SigV4 credentials."""

    region: str
    bearer: str | None = None
    #: ``botocore.credentials.ReadOnlyCredentials`` when SigV4.
    credentials: Any = None


class BedrockCredentialError(Exception):
    """A connection that cannot produce credentials (pre-wire, never retried)."""

    def __init__(self, provider_error: "ProviderError") -> None:
        super().__init__(str(provider_error))
        self.provider_error = provider_error


def _secret_value(ref: str) -> str | None:
    """Read a sealed ref from the encrypted store (namespace-pinned)."""
    if not bconn.REF_PATTERN.fullmatch(ref):
        return None
    from local_operator.providers.registry import provider_secret_value

    return provider_secret_value(ref)


def _session_for(conn: Connection) -> Any:
    """A cached ``botocore.session.Session`` for a profile/ambient connection.

    ``Session(profile=X)`` makes botocore drop the env credential provider, so
    an explicit profile ignores an ambient ``AWS_ACCESS_KEY_ID`` (verified,
    design §2.8) — exactly the isolation an operator with production keys in
    their shell needs.
    """
    _quiet_botocore_logging()
    import botocore.session

    key = f"{conn.method}:{conn.profile or ''}"
    with _session_lock:
        cached = _sessions.get(key)
        if cached is not None:
            _sessions.move_to_end(key)
            return cached
        session = (
            botocore.session.Session(profile=conn.profile)
            if conn.profile
            else (botocore.session.Session())
        )
        _sessions[key] = session
        while len(_sessions) > _SESSION_CACHE_MAX:
            _sessions.popitem(last=False)
        return session


def reset_session_cache() -> None:
    """Drop cached sessions (tests; a re-login that changed ~/.aws)."""
    with _session_lock:
        _sessions.clear()


def _profile_region(conn: Connection) -> str | None:
    if conn.method not in ("profile", "ambient"):
        return None
    try:
        region = _session_for(conn).get_config_variable("region")
    except Exception:  # noqa: BLE001 - a broken profile surfaces at credential time
        return None
    return region if isinstance(region, str) and region else None


def region_for(conn: Connection, values: Mapping[str, Any] | None = None) -> str:
    """The connection's Region (design §5.2 precedence)."""
    profile_region = None if conn.region else _profile_region(conn)
    return bconn.resolve_region(conn, profile_region=profile_region, values=values)


def resolve_sync(key: str, values: Mapping[str, Any] | None = None) -> ResolvedAuth:
    """Blocking resolution. Raises :class:`BedrockCredentialError`.

    Used directly by sync callers (discovery runs in a worker thread already)
    and through :func:`resolve` (thread + timeout) by the async ones.
    """
    from local_operator.providers.bedrock import errors

    try:
        conn = bconn.parse(key)
    except bconn.DescriptorError as exc:
        raise BedrockCredentialError(errors.invalid_descriptor_error(str(exc))) from exc
    region = region_for(conn, values)
    if conn.raw_key:
        return ResolvedAuth(region=region, bearer=conn.raw_key)
    if conn.method == "env-bearer":
        token = os.environ.get(bconn.BEARER_ENV, "").strip()
        if not token:
            raise BedrockCredentialError(errors.no_credentials_error())
        return ResolvedAuth(region=region, bearer=token)
    if conn.method == "api-key":
        assert conn.ref is not None  # parse() guarantees it
        token = _secret_value(conn.ref)
        if not token:
            raise BedrockCredentialError(errors.secret_missing_error())
        return ResolvedAuth(region=region, bearer=token.strip())
    if conn.method == "access-key":
        assert conn.ref is not None
        raw = _secret_value(conn.ref)
        try:
            pair = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            pair = None
        if not isinstance(pair, dict) or not pair.get("access_key_id"):
            raise BedrockCredentialError(errors.secret_missing_error())
        from botocore.credentials import Credentials

        creds = Credentials(
            pair["access_key_id"], pair.get("secret_access_key") or "", pair.get("session_token")
        )
        return ResolvedAuth(region=region, credentials=creds.get_frozen_credentials())
    # profile / ambient: the botocore chain. Every botocore failure here is a
    # pre-wire credential problem with a method-specific remedy.
    try:
        session = _session_for(conn)
        creds = session.get_credentials()
        if creds is None:
            raise BedrockCredentialError(errors.no_credentials_error(conn))
        frozen = creds.get_frozen_credentials()
    except BedrockCredentialError:
        raise
    except Exception as exc:  # noqa: BLE001 - classified, never swallowed
        raise BedrockCredentialError(errors.classify_credential_error(exc, conn)) from exc
    if not frozen.access_key:
        raise BedrockCredentialError(errors.no_credentials_error(conn))
    return ResolvedAuth(region=region, credentials=frozen)


async def resolve(
    key: str, *, timeout: float = CREDENTIAL_TIMEOUT_S, values: Mapping[str, Any] | None = None
) -> ResolvedAuth:
    """:func:`resolve_sync` off the event loop, bounded by ``timeout``."""
    from local_operator.providers.bedrock import errors

    try:
        return await asyncio.wait_for(asyncio.to_thread(resolve_sync, key, values), timeout)
    except (TimeoutError, asyncio.TimeoutError) as exc:
        raise BedrockCredentialError(errors.credential_timeout_error(timeout)) from exc


def sign_headers(
    auth: ResolvedAuth,
    method: str,
    url: str,
    body: bytes,
    headers: Mapping[str, str],
) -> dict[str, str]:
    """``headers`` plus the auth headers for this exact request.

    Bearer: one ``Authorization`` header. SigV4: botocore's ``SigV4Auth`` over
    an ``AWSRequest`` built from the SAME method/url/body/headers, then the
    ``Authorization``/``X-Amz-Date``/``X-Amz-Security-Token`` it produced are
    copied out (httpx sends the rest unchanged).
    """
    out = dict(headers)
    if auth.bearer is not None:
        out["Authorization"] = f"Bearer {auth.bearer}"
        return out
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest

    request = AWSRequest(method=method, url=url, data=body, headers=dict(headers))
    SigV4Auth(auth.credentials, SIGNING_NAME, auth.region).add_auth(request)
    for name in ("Authorization", "X-Amz-Date", "X-Amz-Security-Token"):
        value = request.headers.get(name)
        if value is not None:
            out[name] = value
    return out
