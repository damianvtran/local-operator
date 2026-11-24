"""The responder run's AWS credentials: one role, pinned in code (S-2a, S-13).

UNWIRED BY DESIGN (see ``jail.py``): slice C2's spawner calls
:func:`resolve_responder_credentials`; nothing does today.

WHY PINNED. ``responders.yaml`` is same-uid writable and the guardrail fence is
not a boundary (PR #4383 round 3), so nothing the policy file says may choose
which credentials a run holds. The role is therefore a module constant:
``credential_profile`` in the policy selects NOTHING (C2 refuses any value but
the role's own name at parse). The lop credential store is never consulted --
this module imports nothing from ``local_operator.secrets`` or the credential
store, and a test pins that by AST -- because a store entry is exactly the
kind of thing a planter swaps behind a stable name.

WHY THE STS ENDPOINT IS PINNED TOO (security design round 2, S-13). Both STS
calls would otherwise resolve their endpoint through the owner's AWS config
and environment, which are same-uid writable: an ``endpoint_url`` in
``~/.aws/config`` or ``AWS_ENDPOINT_URL_STS`` points STS at a responder the
planter controls, which hands back write-capable keys from ``AssumeRole`` and
then answers ``GetCallerIdentity`` with the expected ARN. So:

* both clients are built with an explicit ``endpoint_url`` (an explicit
  argument outranks every configured endpoint in botocore), ``verify=True``
  (botocore's own CA bundle, never a configured ``ca_bundle``) and
  ``proxies={}`` (no environment proxy can sit in the middle);
* the identity check runs on a session that reads NO config or credentials
  file and holds only the returned triple;
* the ARN must match the FULL prefix including the account, and the access key
  must be temporary (``ASIA…``): a long-lived ``AKIA`` key is never the output
  of ``AssumeRole``.

RESIDUAL (accepted in the design, §4): an actor holding the owner's live admin
SSO session can edit the role's own policy. D's explicit Deny list and the
session policy below bound silent widening; neither stops a deliberate admin.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

#: The ONLY principal a responder run may hold (radient-infra PR D).
RESPONDER_ACCOUNT_ID = "951603962783"
RESPONDER_ROLE_NAME = "radient-responder-readonly"
RESPONDER_ROLE_ARN = f"arn:aws:iam::{RESPONDER_ACCOUNT_ID}:role/{RESPONDER_ROLE_NAME}"

#: What ``GetCallerIdentity`` must answer for the returned triple: the full
#: assumed-role prefix, account included (S-13), never a suffix match.
EXPECTED_ASSUMED_ROLE_PREFIX = (
    f"arn:aws:sts::{RESPONDER_ACCOUNT_ID}:assumed-role/{RESPONDER_ROLE_NAME}/"
)

#: The regional STS endpoint, pinned. Radient production lives in ca-central-1.
STS_REGION = "ca-central-1"
STS_ENDPOINT = f"https://sts.{STS_REGION}.amazonaws.com"

#: The owner's SSO profile the host assumes FROM. Its configuration may be
#: tampered with; that is why the RESULT is verified on a config-free session.
SOURCE_PROFILE = "radient-prod-admin"

#: One hour: the role's ``MaxSessionDuration`` (design §4, D) and a bound on an
#: exfiltrated triple's lifetime.
SESSION_DURATION_SECONDS = 3600

#: Session policy: intersects with the role's own policy, so it can only
#: narrow. Re-states D's explicit Deny so a managed-policy drift on the role
#: cannot widen a run silently.
SESSION_POLICY = json.dumps(
    {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": "*", "Resource": "*"},
            {
                "Effect": "Deny",
                "Action": [
                    "secretsmanager:*",
                    "ssm:GetParameter*",
                    "kms:Decrypt",
                    "s3:GetObject",
                    "lambda:GetFunction*",
                    "lambda:ListFunctions",
                    "ecs:DescribeTaskDefinition",
                    "sts:AssumeRole",
                    "iam:PassRole",
                ],
                "Resource": "*",
            },
        ],
    },
    separators=(",", ":"),
    sort_keys=True,
)

_TEMPORARY_KEY = re.compile(r"^ASIA[A-Z0-9]{12,}$")
_SESSION_NAME = re.compile(r"^[A-Za-z0-9+=,.@-]{2,64}$")

#: ``(client, operation, params) -> response``. The default calls botocore;
#: tests pass a recorder so no network is touched and the CLIENT the call was
#: made on (its endpoint, its credentials) can be asserted.
Caller = Callable[[Any, str, Mapping[str, Any]], Mapping[str, Any]]


class CredentialPinError(RuntimeError):
    """The pinned role could not be proven; the spawner answers ``spawn_failed``."""


@dataclass(frozen=True)
class PinnedCredentials:
    """A verified STS triple for :data:`RESPONDER_ROLE_ARN`. Secrets never print."""

    access_key_id: str
    secret_access_key: str = field(repr=False)
    session_token: str = field(repr=False)
    expiration: str
    assumed_role_arn: str

    def env(self) -> dict[str, str]:
        """The child's AWS keys (all on ``jail.ENV_ALLOWLIST``)."""

        return {
            "AWS_ACCESS_KEY_ID": self.access_key_id,
            "AWS_SECRET_ACCESS_KEY": self.secret_access_key,
            "AWS_SESSION_TOKEN": self.session_token,
            "AWS_REGION": STS_REGION,
            "AWS_DEFAULT_REGION": STS_REGION,
        }


def _call(client: Any, operation: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
    return getattr(client, operation)(**params)


def _client_config() -> Any:
    from botocore.config import Config

    # ``proxies={}`` (not None): botocore reads environment proxies only when
    # this is None (botocore/endpoint.py ``if proxies is None``).
    return Config(proxies={})


#: Profile keys that make botocore RUN something or chain to another principal
#: when it resolves the source credentials (security round 1, S-19). The owner's
#: ``~/.aws/config`` is same-uid writable, so a planted ``credential_process``
#: on the source profile would execute unjailed on every alert wake. Static
#: keys are refused too: the source must be the operator's SSO session.
_FORBIDDEN_SOURCE_KEYS: frozenset[str] = frozenset(
    {
        "credential_process",
        "credential_source",
        "source_profile",
        "role_arn",
        "web_identity_token_file",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
    }
)


def _require_pure_sso_profile(session: Any) -> None:
    """Refuse a source profile that is not a plain SSO profile (S-19).

    Read from the SAME parse botocore will use (``full_config``), before any
    client or credential provider exists, so nothing in the profile has run.
    """

    profiles = session.full_config.get("profiles", {})
    profile = profiles.get(SOURCE_PROFILE)
    if not isinstance(profile, Mapping):
        raise CredentialPinError(f"no AWS profile {SOURCE_PROFILE!r} is configured")
    planted = sorted(_FORBIDDEN_SOURCE_KEYS & set(profile))
    if planted:
        raise CredentialPinError(
            f"AWS profile {SOURCE_PROFILE!r} is not a plain SSO profile ({', '.join(planted)})"
        )
    if not ({"sso_session", "sso_start_url"} & set(profile)):
        raise CredentialPinError(f"AWS profile {SOURCE_PROFILE!r} is not an SSO profile")


def _source_client() -> Any:
    """STS on the owner's SSO profile, endpoint pinned regardless of config."""

    import botocore.session

    session = botocore.session.Session(profile=SOURCE_PROFILE)
    _require_pure_sso_profile(session)
    return session.create_client(
        "sts",
        region_name=STS_REGION,
        endpoint_url=STS_ENDPOINT,
        verify=True,
        config=_client_config(),
    )


def _identity_client(access_key_id: str, secret_access_key: str, session_token: str) -> Any:
    """STS holding ONLY the returned triple, on a session that reads no files."""

    import botocore.session

    session = botocore.session.Session()
    # No config or shared-credentials file: nothing the owner's ~/.aws says
    # (endpoint, ca_bundle, credential_process) reaches this client.
    session.set_config_variable("config_file", os.devnull)
    session.set_config_variable("credentials_file", os.devnull)
    return session.create_client(
        "sts",
        region_name=STS_REGION,
        endpoint_url=STS_ENDPOINT,
        verify=True,
        config=_client_config(),
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        aws_session_token=session_token,
    )


def _require_pinned_endpoint(client: Any) -> None:
    endpoint = getattr(getattr(client, "meta", None), "endpoint_url", None)
    if endpoint != STS_ENDPOINT:
        raise CredentialPinError(f"STS client endpoint {endpoint!r} is not {STS_ENDPOINT}")


def resolve_responder_credentials(*, session_name: str, call: Caller = _call) -> PinnedCredentials:
    """Assume :data:`RESPONDER_ROLE_ARN` and prove the result, or raise.

    Takes no role, profile or store key: there is nothing for a caller (or a
    policy file) to choose. ``session_name`` is the CloudTrail label (the
    run's alert id), validated to STS's own grammar.
    """

    if _SESSION_NAME.fullmatch(session_name) is None:
        raise CredentialPinError("role session name is not a valid STS session name")
    try:
        # Client construction is INSIDE the guard (review R1-1): botocore
        # resolves the profile while building the client, so a member machine
        # without it raises ProfileNotFound -- which must reach the spawner as
        # CredentialPinError (``spawn_failed``), like every other failure here.
        source = _source_client()
        _require_pinned_endpoint(source)
        assumed = call(
            source,
            "assume_role",
            {
                "RoleArn": RESPONDER_ROLE_ARN,
                "RoleSessionName": session_name,
                "DurationSeconds": SESSION_DURATION_SECONDS,
                "Policy": SESSION_POLICY,
            },
        )
        creds = assumed["Credentials"]
        key, secret, token = creds["AccessKeyId"], creds["SecretAccessKey"], creds["SessionToken"]
        expiration = str(creds.get("Expiration", ""))
    except CredentialPinError:
        raise
    except Exception as exc:  # botocore raises a wide family; all mean "no spawn"
        raise CredentialPinError(f"AssumeRole failed ({type(exc).__name__})") from None
    if not isinstance(key, str) or _TEMPORARY_KEY.fullmatch(key) is None:
        raise CredentialPinError("AssumeRole returned a non-temporary access key")

    try:
        identity_client = _identity_client(key, secret, token)
        _require_pinned_endpoint(identity_client)
        identity = call(identity_client, "get_caller_identity", {})
    except CredentialPinError:
        raise
    except Exception as exc:
        raise CredentialPinError(f"GetCallerIdentity failed ({type(exc).__name__})") from None
    arn = identity.get("Arn")
    if identity.get("Account") != RESPONDER_ACCOUNT_ID:
        raise CredentialPinError("caller identity is not in the responder account")
    if not isinstance(arn, str) or not arn.startswith(EXPECTED_ASSUMED_ROLE_PREFIX):
        raise CredentialPinError("caller identity is not the pinned responder role")
    reported = assumed.get("AssumedRoleUser", {}).get("Arn")
    if reported is not None and reported != arn:
        raise CredentialPinError("AssumeRole and GetCallerIdentity disagree on the principal")
    return PinnedCredentials(
        access_key_id=key,
        secret_access_key=secret,
        session_token=token,
        expiration=expiration,
        assumed_role_arn=arn,
    )


__all__ = [
    "EXPECTED_ASSUMED_ROLE_PREFIX",
    "RESPONDER_ACCOUNT_ID",
    "RESPONDER_ROLE_ARN",
    "RESPONDER_ROLE_NAME",
    "STS_ENDPOINT",
    "CredentialPinError",
    "PinnedCredentials",
    "resolve_responder_credentials",
]
