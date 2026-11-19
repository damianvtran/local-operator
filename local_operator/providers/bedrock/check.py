"""Validate a Bedrock connection with one free, signed control-plane read.

``key_check.check_api_key`` would send the key as ``x-api-key``/bearer to a
``base_url`` Bedrock does not have, so Bedrock branches here (design §5.5).

Probe: ``GET https://bedrock.{region}.amazonaws.com/inference-profiles
?typeEquals=SYSTEM_DEFINED&maxResults=1`` — no inference, no cost, and it
exercises exactly the credential + Region the runtime will use. CountTokens is
deliberately NOT used: some cross-Region-only Claude models do not support it,
so it would reject good credentials.

Verdicts keep ``key_check``'s three values: ``True`` accepted; ``False``
definitively rejected (bad signature/key, expired token, unresolvable
credentials such as an expired SSO session); ``None`` unverified (the identity
works but may not list models — common for least-privilege invoke-only roles —
or a timeout/network error). ``None`` never blocks saving.

As in ``key_check``, nothing from the response BODY is returned except the
classified error type; AWS bodies do not echo keys, but the rule is uniform.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from local_operator.providers.bedrock import auth as bauth
from local_operator.providers.bedrock import connection as bconn

if TYPE_CHECKING:
    from local_operator.providers.key_check import KeyCheck

logger = logging.getLogger(__name__)

_REJECTED_TYPES = {
    "unrecognizedclientexception",
    "invalidsignatureexception",
    "expiredtokenexception",
    "incompletesignature",
    "incompletesignatureexception",
    "invalidclienttokenid",
    "missingauthenticationtokenexception",
}


def _keys():  # noqa: ANN202 - generated module
    from local_operator.i18n.keys import wire_errors_bedrock

    return wire_errors_bedrock


def _text(message) -> str:  # noqa: ANN001
    from local_operator.i18n.messages import render_resolved

    return render_resolved(message)


#: Machine codes for a REJECTED connection (``KeyCheck.code``), the vocabulary
#: the desktop AWS panel switches on (local-operator-ui ``refusalTitle``). An
#: unverified verdict (``valid is None``) carries NO code by contract: the
#: renderer distinguishes those by ``reason`` alone (contract delta 6).
CODE_REJECTED = "bedrock_credentials_rejected"
CODE_SSO_EXPIRED = "bedrock_sso_expired"
CODE_PROFILE_NOT_FOUND = "bedrock_profile_not_found"
CODE_NO_CREDENTIALS = "bedrock_no_credentials"


def credential_error_code(exc: BaseException | None) -> str:
    """The rejection code for a credential-resolution failure.

    Classified from the ORIGINAL botocore exception (``__cause__`` of the
    :class:`~local_operator.providers.bedrock.auth.BedrockCredentialError`), by
    class name like ``errors.classify_credential_error``, so the code and the
    sentence can never disagree about what went wrong.
    """
    names = {cls.__name__ for cls in type(exc).__mro__} if exc is not None else set()
    if names & {
        "SSOTokenLoadError",
        "UnauthorizedSSOTokenError",
        "SSOError",
        "LoginTokenLoadError",
        "LoginError",
    } or ("TokenRetrievalError" in names and "sso" in str(exc).lower()):
        return CODE_SSO_EXPIRED
    if "ProfileNotFound" in names:
        return CODE_PROFILE_NOT_FOUND
    if exc is None or names & {"NoCredentialsError", "PartialCredentialsError"}:
        return CODE_NO_CREDENTIALS
    return CODE_REJECTED


async def check_connection(
    key: str, timeout: float, *, client: httpx.AsyncClient | None = None
) -> "KeyCheck":
    """Probe one connection. Never raises (the save route has no guard)."""
    from local_operator.providers.key_check import KeyCheck

    try:
        resolved = await bauth.resolve(key, timeout=timeout)
    except bauth.BedrockCredentialError as exc:
        error = exc.provider_error
        # A credential timeout (IMDS/SSO endpoint slow) is "could not tell".
        if error.retryable:
            return KeyCheck(None, str(error))
        return KeyCheck(False, str(error), credential_error_code(exc.__cause__))
    return await probe(resolved, timeout, client=client)


async def probe(
    resolved: "bauth.ResolvedAuth", timeout: float, *, client: httpx.AsyncClient | None = None
) -> "KeyCheck":
    """The signed control-plane probe for credentials already in hand.

    Split from :func:`check_connection` so a login can validate secret
    material BEFORE it is sealed (an access-key pair or API key typed into the
    connect form has no ``ref`` yet, and storing an unchecked secret is the
    exact thing ``save_key`` exists to avoid). Never raises.
    """
    from local_operator.providers.key_check import KeyCheck

    url = str(
        httpx.URL(
            f"{bconn.control_base(resolved.region)}/inference-profiles",
            params={"typeEquals": "SYSTEM_DEFINED", "maxResults": "1"},
        )
    )
    headers = bauth.sign_headers(resolved, "GET", url, b"", {"Accept": "application/json"})
    owned = client is None
    http = client or httpx.AsyncClient(timeout=timeout)
    try:
        response = await http.get(url, headers=headers, timeout=timeout)
    except Exception as error:  # noqa: BLE001 - "never raises" is the contract
        logger.debug("Bedrock connection check could not complete: %s", type(error))
        return KeyCheck(None, _text(_keys().check_unverified(region=resolved.region)))
    finally:
        if owned:
            await http.aclose()
    status = response.status_code
    if 200 <= status < 300:
        return KeyCheck(True, None)
    error_type = (response.headers.get("x-amzn-ErrorType") or "").split(":", 1)[0].lower()
    if status == 401 or error_type in _REJECTED_TYPES:
        return KeyCheck(
            False, _text(_keys().credentials_rejected(region=resolved.region)), CODE_REJECTED
        )
    if status == 403:
        # Credentials resolved and signed fine, but this identity may not list
        # profiles (invoke-only role): it may still run models.
        return KeyCheck(None, _text(_keys().check_list_denied(region=resolved.region)))
    return KeyCheck(None, _text(_keys().check_unverified(region=resolved.region)))
