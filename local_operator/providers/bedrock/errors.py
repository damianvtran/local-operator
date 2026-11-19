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
import re
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


def _unresolved(text: str) -> "ProviderError":
    """A pre-wire failure of the botocore CHAIN (SSO/``aws login`` cache, profile,
    ambient lookup): labelled ``auth`` for the reader, but NOT rotation-eligible.

    ``auth_error=True`` would make the failover rotate, and rotation writes a
    60 s ``auth_credential_blocks`` row on the stored connection. Nothing about
    the row is wrong here -- the chain is re-read on every attempt -- so after
    the user follows the remedy (``aws sso login``) the very next turn must
    resolve again instead of hitting the block and reading "rate limit or
    quota exceeded" (QA round 2, Q8). ``auth_error=False`` with no status and
    ``retryable=False`` is exactly ``is_rotation_eligible() is False``; the
    failover then hands over to the fallback chain like any other
    non-retryable provider failure.
    """
    return _provider_error(
        None, text, retryable=False, auth_error=False, kind="auth", request_sent=False
    )


def no_credentials_text() -> str:
    """How to turn Bedrock on: the bearer env var, the ambient switch, a stored row.

    Shared by the runtime pre-wire error and the CLI's "no credentials"
    preflight, so a D6 user whose ``AWS_PROFILE`` was deliberately ignored is
    pointed at the explicit ``LOP_BEDROCK_AUTH=ambient`` switch on both paths.
    """
    return _text(
        _keys().no_credentials(
            bearer_env="AWS_BEARER_TOKEN_BEDROCK", auth_env="LOP_BEDROCK_AUTH=ambient"
        )
    )


def no_credentials_error(conn: Connection | None = None) -> "ProviderError":
    """No credentials. With ``conn`` this is a botocore-chain lookup that came
    back empty (re-read every attempt, so no rotation -- see :func:`_unresolved`);
    without one (no key, an unparseable stored key, an unset bearer env var) the
    old rotating shape is kept so a broken stored row still yields to a sibling.
    """
    if conn is None:
        return _prewire(no_credentials_text())
    if conn.method == "profile" and conn.profile:
        return _unresolved(_text(_keys().profile_no_credentials(profile=conn.profile)))
    return _unresolved(no_credentials_text())


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
    """botocore resolution exception -> pre-wire ``ProviderError``.

    Matched by class NAME (not isinstance) so this module never imports
    botocore itself and stays importable on the cold path.

    None of these rotate (see :func:`_unresolved`). A NETWORK failure while
    resolving (SSO portal, STS, IMDS or a proxy unreachable) is not a verdict
    on the credential at all: design §5.5 makes it transient/unverified, so it
    is retryable -- ``check.check_connection`` turns that into ``valid=None``
    and the connect flow stores the row as unverified instead of refusing it.
    """
    names = _exception_names(exc)
    profile = (conn.profile if conn else None) or "default"
    if names & _NETWORK_ERRORS:
        detail = str(exc).strip() or type(exc).__name__
        return _provider_error(
            None,
            _text(_keys().credential_resolution_failed(detail=detail)),
            retryable=True,
            kind="transient",
            request_sent=False,
        )
    if names & {"LoginTokenLoadError", "LoginError"}:
        # ``aws login`` (login_session) profiles: same shape as SSO, other command.
        return _unresolved(
            _text(
                _keys().session_expired(profile=profile, command=f"aws login --profile {profile}")
            )
        )
    if names & {"SSOTokenLoadError", "UnauthorizedSSOTokenError", "SSOError"} or (
        "TokenRetrievalError" in names and "sso" in str(exc).lower()
    ):
        return _unresolved(
            _text(_keys().session_expired(profile=profile, command=_login_command(conn)))
        )
    if "ProfileNotFound" in names:
        return _unresolved(_text(_keys().profile_not_found(profile=profile)))
    if "NoCredentialsError" in names or "PartialCredentialsError" in names:
        return no_credentials_error(conn)
    detail = str(exc).strip() or type(exc).__name__
    return _unresolved(_text(_keys().credential_resolution_failed(detail=detail)))


# botocore's transport failures: ``ConnectionError`` (EndpointConnectionError,
# ConnectTimeoutError, ProxyConnectionError, SSLError) and ``HTTPClientError``
# (ReadTimeoutError, ConnectionClosedError). The builtin ``ConnectionError``
# shares the name and is a network failure too. ``TimeoutError`` covers a raw
# socket timeout surfacing from a credential_process or container endpoint.
_NETWORK_ERRORS = frozenset({"ConnectionError", "HTTPClientError", "TimeoutError"})


def _exception_names(exc: BaseException) -> set[str]:
    """Class names across ``exc`` and its cause/context chain.

    botocore's refresh path can re-raise a transport failure wrapped in its own
    error; the chain is where the network cause survives. Bounded and
    cycle-safe.
    """
    names: set[str] = set()
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen and len(seen) < 8:
        seen.add(id(current))
        names.update(cls.__name__ for cls in type(current).__mro__)
        current = current.__cause__ or current.__context__
    return names


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


# An IAM deny that names one model or inference-profile ARN ("... is not
# authorized to perform: bedrock:InvokeModelWithResponseStream on resource:
# arn:aws:bedrock:us-west-2::foundation-model/...") is scoped to that resource:
# the usual cause is a cross-Region profile routing to a destination Region the
# policy does not list, or a worker role allowed only some models. The same
# credential still works for other models, so it must not be treated as a
# credential failure (which would block the whole connection for every model).
# A bare "not authorized to perform" without such a resource stays auth.
_MODEL_RESOURCE = re.compile(
    r"on resource:\s*arn:[^\s]*:(?:foundation-model|inference-profile|"
    r"application-inference-profile)/",
    re.IGNORECASE,
)

# The action and resource AWS names in an IAM deny, so the remedy can say
# exactly what to grant: "... is not authorized to perform:
# bedrock:InvokeModelWithResponseStream on resource: arn:...:inference-profile/x
# because no identity-based policy allows ...". The ARN ends at whitespace;
# trailing punctuation is trimmed by the caller.
_IAM_DENY_PARTS = re.compile(
    r"not authorized to perform:\s*(?P<action>\S+)\s+on resource:\s*(?P<resource>arn:\S+)",
    re.IGNORECASE,
)


def _console_model_access(message: str) -> bool:
    """The per-model console setting ("You don't have access to the model ...").

    Deliberately NOT the IAM shape: AWS names the model ARN in every IAM
    invoke deny, including a principal with no Bedrock permission at all, so
    an ARN alone does not mean "enable model access in the console" (review
    round 2, R2-1). The remedy for an IAM deny is an IAM grant.
    """
    lowered = message.lower()
    return (
        "access to the model" in lowered
        or "model access" in lowered
        or "not authorized to invoke this model" in lowered
    )


def _iam_model_deny(message: str, model: str) -> tuple[str, str] | None:
    """``(action, resource)`` for an IAM deny scoped to a model/profile ARN.

    When AWS reworded the sentence but still named a model ARN, the deny stays
    model-scoped (no credential block) and the copy falls back to the invoke
    action family and the requested model.
    """
    if not _MODEL_RESOURCE.search(message):
        return None
    match = _IAM_DENY_PARTS.search(message)
    if match is None:
        return "bedrock:InvokeModel*", model or "this model"
    return match.group("action"), match.group("resource").rstrip(".,;:)")


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
        iam_scoped = _iam_model_deny(message, model)
        if iam_scoped is not None:
            # Scoped to one model/profile ARN: the same identity may still be
            # allowed other models (or other destination Regions), so no
            # credential block (auth_error=False, kind=request). The words
            # name the IAM action and resource to grant.
            action, resource = iam_scoped
            return ProviderError(
                status or 403,
                f"{_text(_keys().iam_denied(action=action, resource=resource))} ({shown})",
                auth_error=False,
                kind="request",
            )
        if _console_model_access(message):
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
