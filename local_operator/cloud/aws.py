"""The AWS side of tier 1: resolved credentials, clients, and the account guard.

WHY THE SDK IS IMPORTED INSIDE A FUNCTION, NEVER AT IMPORT TIME. ``boto3`` is an
OPTIONAL dependency (§3.7): it lives in the ``cloud`` extra and deliberately NOT
in ``all``, because an end user who never provisions cloud compute should not
install an AWS SDK. If any module in this package imported it at import time, a
default install's failure would be a bare ``ModuleNotFoundError`` from deep
inside a package it never asked for — so every AWS import here is lazy and behind
``optional.require_extra``, which is what turns it into a sentence naming the
extra and the exact command that fixes it.

THE ACCOUNT GUARD IS HERE RATHER THAN IN A SCRIPT. ``verify_account`` is a
property of the provider (§4.1), not of the driver script that first needed it:
the POC's own rule is that the check runs FIRST, before a single mutating call,
and a check that lives in a script is one a second caller can skip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from local_operator.cloud.provider import CloudRefused
from local_operator.optional import require_extra

#: The POC's sandbox account (``minerva_sandbox``). Everything tier 1 can touch
#: lives here, and the guard below refuses to operate anywhere else.
EXPECTED_POC_ACCOUNT = "325492156725"

#: The region those resources are in. Named rather than ambient: every client is
#: built with it explicitly, so a stray ``AWS_REGION`` cannot retarget a run.
REGION = "ca-central-1"

#: One assumed controller session has to outlive a whole ``--runs N`` sweep, and a
#: single run's own client deadline is 2 h + 10 min.
CONTROLLER_SESSION_SECONDS = 10800

#: The presigned PUT's life: long enough to outlast a run plus its upload.
PRESIGN_EXPIRES_SECONDS = 10800

#: The POC's driver-role session name. A constant because the role's CloudTrail
#: entries are read by name during an incident.
CONTROLLER_SESSION_NAME = "lop-poc-driver"


@dataclass(frozen=True)
class AwsCredentials:
    """Resolved AWS secrets. Never logged, never printed, never in the environment.

    ``__repr__`` is redacted for the same reason the OSWorld adapter's
    ``AwsCredentials`` redacts its own: the values reach a dataclass that a log
    line, a traceback or an ``f-string`` will happily render, and only one of those
    has a reviewer watching. ``dataclasses`` does not overwrite a ``__repr__``
    defined in the class body, so this one is the one that runs.
    """

    access_key_id: str
    secret_access_key: str
    session_token: str | None = None

    def __repr__(self) -> str:  # pragma: no cover - defensive against accidental echo
        return "AwsCredentials(<redacted>)"


@dataclass(frozen=True)
class FargateClients:
    """The injectable I/O surface, so tests use botocore's Stubber.

    Built by the CLI, from whichever identities its profile resolution produced:
    in the POC's deployment ``sts`` is the OPERATOR's client (the account
    assertion runs as the operator) while ``ecs`` and ``s3`` are the assumed
    controller's, so the two are not interchangeable.

    ``definitions`` is that split made explicit, and it is the one field §4.2's
    sketch does not name. The extraction map sends
    ``active_task_definition(operator, outputs)`` into ``describe()``, and the
    controller role DELIBERATELY holds no ``ecs:DescribeTaskDefinition`` (a fact
    the POC records where it reads the definition with the operator session), so a
    provider holding only the controller's client could not make the call the map
    assigns it. It defaults to ``None``, meaning "``ecs`` can read definitions
    too", so a caller that has one identity needs one client.
    """

    ecs: Any
    sts: Any
    s3: Any
    definitions: Any = None


def sdk_exceptions() -> Any:
    """``botocore.exceptions``, imported lazily and behind the ``cloud`` extra.

    A module-level import would break a default install (§3.7), so the exception
    TYPE a caller has to catch has to be reached the same way as the client it
    comes from. It is a function rather than a module constant for exactly that
    reason.
    """
    return cast(Any, require_extra("botocore.exceptions", "cloud", "Cloud task provisioning"))


def s3_client(session: Any, region: str = REGION) -> Any:
    """An S3 client whose PRESIGNED URLs address the regional endpoint.

    botocore signs ``generate_presigned_url`` for S3 against the LEGACY global
    host (``<bucket>.s3.amazonaws.com``) even when the client's own endpoint is
    regional — measured by the POC on botocore 1.43.109 — and S3 answers a request
    for a ca-central-1 bucket that way with ``307 TemporaryRedirect``. Naming the
    signature version and the addressing style is what puts the bucket's own
    region back in the Host header, which is the only form the signature is good
    for.
    """
    config = sdk_config()
    return session.client(
        "s3",
        endpoint_url=f"https://s3.{region}.amazonaws.com",
        config=config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )


def sdk_config() -> Any:
    """``botocore.config.Config``, lazily — same reason as :func:`sdk_exceptions`."""
    return cast(Any, require_extra("botocore.config", "cloud", "Cloud task provisioning")).Config


def build_clients(
    credentials: AwsCredentials,
    region: str = REGION,
    *,
    definitions: Any = None,
    sts: Any = None,
) -> FargateClients:
    """Construct the real clients from resolved secrets.

    The session is built from the credential VALUES, so boto3's own provider
    chain (environment variables, ``~/.aws``, instance metadata) is never
    consulted — which is what keeps the caller's profile resolution
    authoritative. ``region_name`` is passed explicitly on every client for the
    same reason.

    ``sts`` and ``definitions`` are injectable so that the operator/controller
    identity split above can be expressed by the caller; both default to clients
    built from the same credentials.
    """
    boto3 = cast(Any, require_extra("boto3", "cloud", "Cloud task provisioning"))
    session = boto3.Session(
        aws_access_key_id=credentials.access_key_id,
        aws_secret_access_key=credentials.secret_access_key,
        aws_session_token=credentials.session_token,
        region_name=region,
    )
    return FargateClients(
        ecs=session.client("ecs"),
        sts=sts if sts is not None else session.client("sts"),
        s3=s3_client(session, region),
        definitions=definitions,
    )


def verify_account(clients: FargateClients, expected: str = EXPECTED_POC_ACCOUNT) -> str:
    """Refuse to operate in any account but ``expected``, and name the caller.

    Returns the caller's ARN (useful for a log line) and raises
    :class:`CloudRefused` with ``code="wrong_account"`` otherwise. The refusal
    sentence names the variable that usually caused it, because "which account am
    I in" is the question every other guard on this path depends on and a stray
    ``AWS_PROFILE`` silently retargets everything downstream of it.
    """
    identity = clients.sts.get_caller_identity()
    account = str(identity["Account"])
    if account != expected:
        raise CloudRefused(
            "wrong_account",
            f"refusing to run: caller account is {account}, expected {expected} "
            "(check AWS_PROFILE; the POC lives in minerva_sandbox)",
        )
    return str(identity["Arn"])


def assume_role(
    clients: FargateClients,
    role_arn: str,
    *,
    session_name: str = CONTROLLER_SESSION_NAME,
    duration_s: int = CONTROLLER_SESSION_SECONDS,
) -> AwsCredentials:
    """Assume a scoped role and return credentials for it — nothing more.

    The CLI does this once and hands the provider the result; the provider never
    reads an ``argparse`` namespace (§4.1). The operator's own identity is used for
    exactly two things: this AssumeRole (through the same ``sts`` client) and
    whatever the caller does with its own clients.
    """
    response = clients.sts.assume_role(
        RoleArn=role_arn,
        RoleSessionName=session_name,
        DurationSeconds=duration_s,
    )
    credentials = response["Credentials"]
    return AwsCredentials(
        access_key_id=str(credentials["AccessKeyId"]),
        secret_access_key=str(credentials["SecretAccessKey"]),
        session_token=str(credentials["SessionToken"]),
    )


@dataclass(frozen=True)
class CloudBlobs:
    """The object-store side of a run: presign for the container, probe for drain.

    The container never holds an S3 credential: it is handed a presigned PUT that
    can write exactly one key and dies with the session that signed it. So the
    two things a provider needs are a URL to hand out and a way to see whether the
    container used it — and nothing else here reaches the bucket.

    (Result HAND-OFF — downloading and unpacking the tarball — is not the
    provider's; §4.1 puts it on this same injected seam with the CLI.)
    """

    s3: Any
    bucket: str
    prefix: str = "runs"
    presign_expires_s: int = PRESIGN_EXPIRES_SECONDS

    def key(self, run_id: str, name: str) -> str:
        """The object key for one of a run's artifacts."""
        return f"{self.prefix}/{run_id}/{name}"

    def presign_put(self, key: str) -> str:
        """A presigned PUT for exactly one object key.

        Signed with the caller's credentials, so the URL can write nothing but its
        own key.
        """
        return str(
            self.s3.generate_presigned_url(
                "put_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=self.presign_expires_s,
                HttpMethod="PUT",
            )
        )

    def object_present(self, key: str) -> bool:
        """Whether one object exists. This is the drain's checkpoint marker.

        A ``head_object`` and not a ``list_objects``: the policy grants
        ``s3:GetObject`` on the run prefix and deliberately not ``s3:ListBucket``,
        so a listing would be refused. The two codes read as ABSENT are the POC's
        own measured ones (``download_artifacts``): 403 comes back for an object
        the caller may not even list, which is what S3 answers here for an upload
        that never happened, and reading it as "still missing" is exactly what the
        drain needs. Every other failure propagates: a credential that expired is
        not an empty bucket.
        """
        errors = sdk_exceptions()
        try:
            self.s3.head_object(Bucket=self.bucket, Key=key)
        except errors.ClientError as error:
            code = str((getattr(error, "response", {}) or {}).get("Error", {}).get("Code", ""))
            if code in ("404", "403", "NoSuchKey", "NotFound"):
                return False
            raise
        return True
