"""Amazon Bedrock connection descriptors, Region precedence and the env tier.

WHY A DESCRIPTOR. Every credential seam in the harness — failover's account
walk, discovery's ``api_key=``, ``key_check.check_api_key``, multi-account
rotation — passes ONE opaque string per credential. A Bedrock "credential" is
really a *connection*: an auth method (Bedrock API key, AWS profile incl.
SSO/``aws login``, IAM access keys, the ambient default chain) plus a Region.
Encoding that connection as a non-secret URI and storing it as the row's
``key`` means none of those seams change, and two Regions become two accounts
(so throttling rotation doubles as cross-Region resilience). Design:
``docs/providers/BEDROCK.md`` and the A1 design §5.

    bedrock://api-key?region=us-east-1&ref=BEDROCK_SECRET_<32hex>
    bedrock://profile?name=<profile>&region=us-east-1
    bedrock://access-key?region=us-east-1&ref=BEDROCK_SECRET_<32hex>
    bedrock://ambient[?region=...]
    bedrock://env-bearer[?region=...]
    <anything without the bedrock:// prefix>   -> a raw Bedrock API key

Secret material (a Bedrock API key, an access-key pair) is NEVER in the
descriptor: it is sealed in the encrypted store under ``ref`` by the
``AuthStore`` seal hook (``providers/bedrock/store.py``). ``ref`` is regex
pinned so a descriptor can never point the resolver at an arbitrary operator
secret.

PURE: this module imports neither botocore nor the secret store, so the
registry's env-tier callable can run on every provider census for free.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, TypeGuard
from urllib.parse import parse_qsl, urlencode, urlsplit

#: Descriptor scheme. A key without it is a raw Bedrock API key (the generic
#: ``PUT /v1/auth/providers/bedrock/key`` and paste-a-key login path).
SCHEME = "bedrock://"

Method = Literal["api-key", "profile", "access-key", "ambient", "env-bearer"]
METHODS: tuple[str, ...] = ("api-key", "profile", "access-key", "ambient", "env-bearer")

#: Encrypted-store namespace for sealed Bedrock secrets; mirrors
#: ``meta_muse_credentials.REF_PATTERN`` (fresh uuid4 hex per write).
REF_PREFIX = "BEDROCK_SECRET_"
REF_PATTERN = re.compile(r"BEDROCK_SECRET_[a-f0-9]{32}")

#: ``providers.bedrock.region`` default: empty means "inherit" — fall through
#: to ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` / the profile's region / fallback.
DEFAULT_BEDROCK_REGION_SETTING = ""
#: ``providers.bedrock.auth`` default: empty means "only a stored connection or
#: an explicit env signal enables Bedrock". ``"ambient"`` opts into the default
#: AWS credential chain with no stored row (headless workers).
DEFAULT_BEDROCK_AUTH_SETTING = ""
#: Last-resort Region when nothing names one. us-east-1 carries every Claude
#: model and every ``us.``/``global.`` inference profile.
FALLBACK_REGION = "us-east-1"

#: Testing/QA hook, NOT a supported production setting (VPC endpoints are a v1
#: non-goal): when set, every Bedrock request (runtime and control plane) goes
#: to this base URL instead of ``https://bedrock[-runtime].{region}.amazonaws.com``.
#: It exists so ``scripts/bedrock_fake_server.py`` can exercise the real
#: ``lop exec``/SDK stack without AWS. Signing still happens (over the override
#: host), so the auth code path is the production one.
ENDPOINT_OVERRIDE_ENV = "LOP_BEDROCK_ENDPOINT_URL"

#: Explicit env switch for the ambient chain (headless workers). Only the
#: literal ``ambient`` (case-insensitive) turns it on.
AUTH_ENV = "LOP_BEDROCK_AUTH"
BEARER_ENV = "AWS_BEARER_TOKEN_BEDROCK"

#: Env vars that only AWS compute sets: ECS/AgentCore container credentials,
#: EKS IRSA / web identity, Lambda. Their presence means "this process runs ON
#: AWS with a role", which is the only case the ambient chain may self-enable.
#: Deliberately NOT ``AWS_PROFILE`` / ``AWS_ACCESS_KEY_ID``: an operator's
#: laptop exports those for unrelated production work, and auto-enabling would
#: mark Bedrock usable and feed login-default planning (design D6).
COMPUTE_MARKERS: tuple[str, ...] = (
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_LAMBDA_FUNCTION_NAME",
    "ECS_CONTAINER_METADATA_URI_V4",
)

_REGION_RE = re.compile(r"[a-z]{2}(-[a-z]+)+-\d+")


class DescriptorError(ValueError):
    """A ``bedrock://`` string that does not describe a usable connection."""


@dataclass(frozen=True)
class Connection:
    """One parsed Bedrock connection. Carries no secret value, only ``ref``."""

    method: Method
    region: str | None = None
    profile: str | None = None
    ref: str | None = None
    #: Only for a raw (non-descriptor) key: the Bedrock API key itself. Never
    #: produced by :func:`parse` for a ``bedrock://`` string.
    raw_key: str | None = None

    @property
    def uses_bearer(self) -> bool:
        return self.method in ("api-key", "env-bearer")

    def descriptor(self) -> str:
        """The canonical ``bedrock://`` string (never contains ``raw_key``)."""
        return format_descriptor(
            self.method, region=self.region, profile=self.profile, ref=self.ref
        )

    def scope_key(self) -> str:
        """Non-secret identity for caches: method plus profile or key fingerprint.

        The Region is NOT included; callers (``discovery.catalogue_scope``)
        prepend the resolved Region. Never contains a secret or a raw ref:
        a raw key and a sealed ref each contribute only a short SHA-256
        fingerprint, so two different keys in one Region (possibly in two
        AWS accounts) never share a discovery listing.
        """
        import hashlib

        if self.raw_key:
            fp = hashlib.sha256(self.raw_key.encode("utf-8")).hexdigest()[:12]
            return f"api-key:{fp}"
        if self.ref:
            fp = hashlib.sha256(self.ref.encode("utf-8")).hexdigest()[:12]
            return f"{self.method}:{self.profile or ''}:{fp}"
        return f"{self.method}:{self.profile or ''}"


def is_descriptor(key: object) -> TypeGuard[str]:
    return isinstance(key, str) and key.startswith(SCHEME)


def format_descriptor(
    method: str, *, region: str | None = None, profile: str | None = None, ref: str | None = None
) -> str:
    if method not in METHODS:
        raise DescriptorError(f"unknown Bedrock connection method {method!r}")
    query: list[tuple[str, str]] = []
    if method == "profile":
        if not profile:
            raise DescriptorError("a profile connection needs a profile name")
        query.append(("name", profile))
    if region:
        query.append(("region", region))
    if method in ("api-key", "access-key"):
        if not ref or not REF_PATTERN.fullmatch(ref):
            raise DescriptorError("a sealed Bedrock connection needs a BEDROCK_SECRET_ ref")
        query.append(("ref", ref))
    suffix = f"?{urlencode(query)}" if query else ""
    return f"{SCHEME}{method}{suffix}"


def parse(key: str, *, require_ref: bool = True) -> Connection:
    """Parse a stored/env key into a :class:`Connection`. Raises :class:`DescriptorError`.

    A string without the scheme is a raw Bedrock API key (bearer). Unknown
    query keys are refused rather than ignored, so a typo cannot silently
    drop the Region and send a request to the fallback one.

    ``require_ref=False`` is ONLY for the seal hook (``store.seal_credential``),
    which receives a not-yet-sealed ``api-key``/``access-key`` descriptor plus
    the secret and mints the ref itself; a ref that IS present must still
    match ``REF_PATTERN``. Every resolver/delete path keeps the default, so a
    persisted sealed descriptor without a valid ref is always refused.
    """
    if not isinstance(key, str) or not key.strip():
        raise DescriptorError("empty Bedrock credential")
    key = key.strip()
    if not key.startswith(SCHEME):
        return Connection(method="api-key", raw_key=key)
    parts = urlsplit(key)
    method = parts.netloc or parts.path.lstrip("/")
    if method not in METHODS:
        raise DescriptorError(f"unknown Bedrock connection method {method!r}")
    params = dict(parse_qsl(parts.query, keep_blank_values=True))
    allowed = {"region", "name", "ref"}
    unknown = set(params) - allowed
    if unknown:
        raise DescriptorError(f"unknown Bedrock descriptor field(s): {', '.join(sorted(unknown))}")
    region = params.get("region") or None
    if region is not None and not valid_region(region):
        raise DescriptorError(f"invalid AWS Region {region!r}")
    ref = params.get("ref") or None
    if method in ("api-key", "access-key"):
        if (ref is None and require_ref) or (ref is not None and not REF_PATTERN.fullmatch(ref)):
            raise DescriptorError("Bedrock descriptor ref is missing or outside BEDROCK_SECRET_*")
    elif ref is not None:
        raise DescriptorError(f"a {method} connection carries no secret ref")
    profile = params.get("name") or None
    if method == "profile" and not profile:
        raise DescriptorError("a profile connection needs a profile name")
    if method != "profile" and profile is not None:
        raise DescriptorError(f"a {method} connection carries no profile name")
    return Connection(
        method=method,  # type: ignore[arg-type]  # validated against METHODS above
        region=region,
        profile=profile,
        ref=ref,
    )


def valid_region(region: str) -> bool:
    """Shape check only (``us-east-1``, ``us-gov-west-1``); no network, no botocore."""
    return bool(_REGION_RE.fullmatch(region))


def _bedrock_settings(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if values is None:
        try:
            from local_operator.config import read_config_values
            from local_operator.paths import config_dir

            values = read_config_values(config_dir()) or {}
        except Exception:  # noqa: BLE001 - an unreadable config means "unset"
            values = {}
    providers = values.get("providers") if isinstance(values, Mapping) else None
    entry = providers.get("bedrock") if isinstance(providers, Mapping) else None
    return entry if isinstance(entry, Mapping) else {}


def configured_region(values: Mapping[str, Any] | None = None) -> str:
    raw = _bedrock_settings(values).get("region", DEFAULT_BEDROCK_REGION_SETTING)
    return raw.strip() if isinstance(raw, str) else ""


def configured_auth(values: Mapping[str, Any] | None = None) -> str:
    raw = _bedrock_settings(values).get("auth", DEFAULT_BEDROCK_AUTH_SETTING)
    return raw.strip().lower() if isinstance(raw, str) else ""


def resolve_region(
    conn: Connection,
    *,
    profile_region: str | None = None,
    values: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """The ONE Region function inference, discovery and validation share.

    A stored connection's own Region always wins (it was chosen at login).
    Otherwise: ``providers.bedrock.region`` > ``AWS_REGION`` >
    ``AWS_DEFAULT_REGION`` > the resolved profile's ``region`` > us-east-1.
    ``profile_region`` is supplied by the botocore-side caller, so this stays
    pure.
    """
    if conn.region:
        return conn.region
    env = os.environ if environ is None else environ
    for candidate in (
        configured_region(values),
        env.get("AWS_REGION", ""),
        env.get("AWS_DEFAULT_REGION", ""),
        profile_region or "",
    ):
        candidate = candidate.strip()
        if candidate and valid_region(candidate):
            return candidate
    return FALLBACK_REGION


def bedrock_env_key(
    environ: Mapping[str, str] | None = None, values: Mapping[str, Any] | None = None
) -> str | None:
    """The registry env tier (``ProviderDefinition.env_keys``) for Bedrock.

    First match wins: (1) ``AWS_BEARER_TOKEN_BEDROCK`` -> ``env-bearer``;
    (2) ``LOP_BEDROCK_AUTH=ambient`` or ``providers.bedrock.auth: ambient`` ->
    ``ambient``; (3) an AWS-compute marker -> ``ambient``; else ``None``.
    A plain ``AWS_PROFILE`` / ``AWS_ACCESS_KEY_ID`` never enables Bedrock (D6).
    The returned descriptor carries no Region: the Region is resolved at use
    time so a worker's ``AWS_REGION`` change is honoured without a restart.
    """
    env = os.environ if environ is None else environ
    if env.get(BEARER_ENV, "").strip():
        return format_descriptor("env-bearer")
    if env.get(AUTH_ENV, "").strip().lower() == "ambient":
        return format_descriptor("ambient")
    if any(env.get(name, "").strip() for name in COMPUTE_MARKERS):
        return format_descriptor("ambient")
    # Config last: it is a file read, and every env answer above is free.
    if configured_auth(values) == "ambient":
        return format_descriptor("ambient")
    return None


def endpoint_override(environ: Mapping[str, str] | None = None) -> str | None:
    env = os.environ if environ is None else environ
    value = env.get(ENDPOINT_OVERRIDE_ENV, "").strip().rstrip("/")
    return value or None


def runtime_base(region: str, environ: Mapping[str, str] | None = None) -> str:
    return endpoint_override(environ) or f"https://bedrock-runtime.{region}.amazonaws.com"


def control_base(region: str, environ: Mapping[str, str] | None = None) -> str:
    return endpoint_override(environ) or f"https://bedrock.{region}.amazonaws.com"


#: Geo prefix of a Region, for picking the matching cross-Region inference
#: profile (``us.``, ``eu.``, ``apac.`` ...). Only a hint for suggestions:
#: routing is never inferred from the prefix alone (profiles depend on the
#: SOURCE Region, see design §2.5).
def geo_prefix(region: str) -> str:
    if region.startswith("us-gov-"):
        return "us-gov"
    if region.startswith(("us-", "ca-")):
        return "us"
    if region.startswith("eu-"):
        return "eu"
    if region.startswith("ap-northeast-1"):
        return "jp"
    if region.startswith("ap-southeast-2") or region.startswith("ap-southeast-4"):
        return "au"
    if region.startswith("ap-"):
        return "apac"
    return "us"


def account_label(conn: Connection, region: str) -> str:
    """Human label for a row (``dev · us-east-1``); never a secret."""
    who = {
        "profile": conn.profile or "profile",
        "api-key": "API key",  # i18n: ignore technical method label, mirrors AWS console wording
        "access-key": "access key",  # i18n: ignore technical method label
        "ambient": "default chain",  # i18n: ignore technical method label
        "env-bearer": "AWS_BEARER_TOKEN_BEDROCK",
    }[conn.method]
    return f"{who} · {region}"
