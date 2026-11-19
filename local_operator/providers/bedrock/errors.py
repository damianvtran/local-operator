"""Bedrock failures -> ``ProviderError`` with the right failover/incident flags.

Three sources, one table (design §6.5):

* HTTP error responses: AWS names the error in ``x-amzn-ErrorType``
  (``ThrottlingException:http://internal.amazon.com/...``) and the JSON body's
  ``message``/``Message``.
* Mid-stream exception frames of ``vnd.amazon.eventstream``
  (``:message-type=exception``, ``:exception-type=throttlingException``).
* botocore credential-resolution exceptions (pre-wire, ``request_sent=False``).

Why a table and not the generic ``raise_for_status``: a 403 on Bedrock is two
opposite things. "You don't have access to the model" is a per-model console
setting and must NOT mark the credential bad (``auth_error=False``) — the same
connection runs every other model — while an IAM/signature 403 is a real auth
failure. Context overflow ("input is too long", "too many tokens") stays a
``request`` error whose message ``incidents.py`` already classifies as
overflow, so compaction kicks in with no Bedrock-specific code there.

User-facing remedies are keyed (``wire.errors.bedrock``) and rendered in the
resolved language; AWS's own ``message`` text is appended verbatim (it is the
provider's diagnostic, like every other provider's error text).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from local_operator.providers.bedrock.connection import Connection

if TYPE_CHECKING:
    import httpx

    from local_operator.providers.failover import ProviderError


def _text(message: Any) -> str:
    from local_operator.i18n.messages import render_resolved

    return render_resolved(message)


def _keys() -> Any:
    from local_operator.i18n.keys import wire_errors_bedrock

    return wire_errors_bedrock


def _provider_error(*args: Any, **kwargs: Any) -> "ProviderError":
    from local_operator.providers.failover import ProviderError

    return ProviderError(*args, **kwargs)


# -- pre-wire credential errors ----------------------------------------------


def _prewire(text: str) -> "ProviderError":
    return _provider_error(
        None, text, retryable=False, auth_error=True, kind="auth", request_sent=False
    )


def no_credentials_error(conn: Connection | None = None) -> "ProviderError":
    if conn is not None and conn.method == "profile" and conn.profile:
        return _prewire(_text(_keys().profile_no_credentials(profile=conn.profile)))
    return _prewire(
        _text(
            _keys().no_credentials(
                bearer_env="AWS_BEARER_TOKEN_BEDROCK", auth_env="LOP_BEDROCK_AUTH=ambient"
            )
        )
    )


def secret_missing_error() -> "ProviderError":
    return _prewire(_text(_keys().secret_missing(command="lop login bedrock")))


def invalid_descriptor_error(detail: str) -> "ProviderError":
    return _prewire(_text(_keys().invalid_connection(detail=detail)))


def credential_timeout_error(seconds: float) -> "ProviderError":
    # Retryable: an IMDS/SSO endpoint that did not answer in time is a blip,
    # not a verdict on the credential.
    from local_operator.providers.failover import ProviderError

    return ProviderError(
        None,
        _text(_keys().credential_timeout(seconds=int(seconds))),
        retryable=True,
        kind="timeout",
        request_sent=False,
    )


def _login_command(conn: Connection | None) -> str:
    profile = (conn.profile if conn else None) or "default"
    return f"aws sso login --profile {profile}"


def classify_credential_error(exc: BaseException, conn: Connection | None) -> "ProviderError":
    """botocore resolution exception -> pre-wire auth ``ProviderError``.

    Matched by class NAME (not isinstance) so this module never imports
    botocore itself and stays importable on the cold path.
    """
    names = {cls.__name__ for cls in type(exc).__mro__}
    profile = (conn.profile if conn else None) or "default"
    if names & {"LoginTokenLoadError", "LoginError"}:
        # ``aws login`` (login_session) profiles: same shape as SSO, other command.
        return _prewire(
            _text(
                _keys().session_expired(profile=profile, command=f"aws login --profile {profile}")
            )
        )
    if names & {"SSOTokenLoadError", "UnauthorizedSSOTokenError", "SSOError"} or (
        "TokenRetrievalError" in names and "sso" in str(exc).lower()
    ):
        return _prewire(
            _text(_keys().session_expired(profile=profile, command=_login_command(conn)))
        )
    if "ProfileNotFound" in names:
        return _prewire(_text(_keys().profile_not_found(profile=profile)))
    if "NoCredentialsError" in names or "PartialCredentialsError" in names:
        return no_credentials_error(conn)
    detail = str(exc).strip() or type(exc).__name__
    return _prewire(_text(_keys().credential_resolution_failed(detail=detail)))


# -- wire errors ---------------------------------------------------------------

_THROTTLE = {"throttlingexception", "servicequotaexceededexception", "toomanyrequestsexception"}
_TRANSIENT = {
    "modelnotreadyexception",
    "serviceunavailableexception",
    "internalserverexception",
    "modelstreamerrorexception",
    "internalfailure",
}
_TIMEOUT = {"modeltimeoutexception"}
_AUTH = {
    "unrecognizedclientexception",
    "invalidsignatureexception",
    "expiredtokenexception",
    "incompletesignature",
    "incompletesignatureexception",
    "missingauthenticationtokenexception",
    "invalidclienttokenid",
}

_DEFAULT_STATUS = {
    "throttlingexception": 429,
    "servicequotaexceededexception": 429,
    "modelnotreadyexception": 503,
    "serviceunavailableexception": 503,
    "internalserverexception": 500,
    "modelstreamerrorexception": 500,
    "modeltimeoutexception": 408,
    "validationexception": 400,
    "resourcenotfoundexception": 404,
    "accessdeniedexception": 403,
}


def _model_access_denied(message: str) -> bool:
    lowered = message.lower()
    return (
        "access to the model" in lowered
        or "model access" in lowered
        or ("not authorized to invoke this model" in lowered)
    )


def provider_error_for(
    error_type: str,
    message: str,
    status: int | None,
    *,
    model: str = "",
    region: str = "",
    retry_after_ms: int | None = None,
) -> "ProviderError":
    """The ONE mapping for both HTTP error responses and exception frames."""
    from local_operator.providers.failover import ProviderError

    kind_key = error_type.split(":", 1)[0].strip().lower()
    if status is None:
        status = _DEFAULT_STATUS.get(kind_key)
    shown = f"{error_type.split(':', 1)[0] or 'Bedrock error'}: {message}".strip()
    if kind_key in _THROTTLE:
        return ProviderError(
            status or 429, shown, retryable=True, kind="quota", retry_after_ms=retry_after_ms
        )
    if kind_key in _TRANSIENT:
        return ProviderError(status or 503, shown, retryable=True, kind="transient")
    if kind_key in _TIMEOUT:
        return ProviderError(status or 408, shown, retryable=True, kind="timeout")
    if kind_key in _AUTH:
        return ProviderError(
            status or 403,
            f"{_text(_keys().credentials_rejected(region=region))} ({shown})",
            auth_error=True,
            kind="auth",
        )
    if kind_key == "accessdeniedexception":
        if _model_access_denied(message):
            # Per-model console setting: the credential is FINE for other
            # models, so never block it (auth_error=False).
            return ProviderError(
                status or 403,
                f"{_text(_keys().model_access(model=model, region=region))} ({shown})",
                auth_error=False,
                kind="request",
            )
        return ProviderError(
            status or 403,
            f"{_text(_keys().access_denied(region=region))} ({shown})",
            auth_error=True,
            kind="auth",
        )
    if kind_key == "resourcenotfoundexception":
        return ProviderError(
            status or 404,
            f"{_text(_keys().model_unavailable(model=model, region=region))} ({shown})",
            kind="request",
        )
    if kind_key == "validationexception":
        # Overflow wording passes through verbatim so incidents.py classifies
        # it; every other validation names model + Region for the reader.
        lowered = message.lower()
        if "too long" in lowered or "too many tokens" in lowered:
            return ProviderError(status or 400, shown, kind="request")
        suffix = f" [{model} @ {region}]" if model or region else ""
        return ProviderError(status or 400, f"{shown}{suffix}", kind="request")
    # Unknown error type: fall back to status semantics like raise_for_status.
    retryable = status is not None and (status == 429 or status >= 500 or status in (408, 504))
    return ProviderError(status, shown, retryable=retryable, auth_error=status in (401, 403))


def _header_error_type(response: "httpx.Response") -> str:
    return response.headers.get("x-amzn-ErrorType") or response.headers.get("x-amzn-errortype", "")


def _body_message(response: "httpx.Response") -> tuple[str, str]:
    """``(error_type_from_body, message)`` from an AWS JSON error body."""
    try:
        payload = json.loads(response.content or b"{}")
    except (ValueError, UnicodeDecodeError):
        return "", (response.text or "").strip()[:2000]
    if not isinstance(payload, dict):
        return "", str(payload)[:2000]
    message = payload.get("message") or payload.get("Message") or ""
    error_type = payload.get("__type") or ""
    return str(error_type).rsplit("#", 1)[-1], str(message)


def raise_for_status(response: "httpx.Response", *, model: str = "", region: str = "") -> None:
    """``bedrock_raise_for_status``: AWS-aware, else the generic mapping."""
    if response.status_code < 400:
        return
    header_type = _header_error_type(response)
    body_type, message = _body_message(response)
    error_type = header_type or body_type
    if not error_type:
        from local_operator.providers.clients import raise_for_status as generic

        generic(response)
        return
    retry_after_ms: int | None = None
    retry_after = response.headers.get("retry-after")
    if retry_after and retry_after.strip().isdigit():
        retry_after_ms = int(retry_after.strip()) * 1000
    raise provider_error_for(
        error_type,
        message,
        response.status_code,
        model=model,
        region=region,
        retry_after_ms=retry_after_ms,
    )
