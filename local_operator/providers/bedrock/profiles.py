"""Offline AWS context for the Bedrock login surfaces (design §5.4 / §7 route 3).

Everything here answers "what could the operator connect with?" WITHOUT
resolving a single credential. That is a hard rule, not a nicety:

* ``credential_process`` profiles run an arbitrary helper on resolution, which
  can hang (no timeout in botocore) or pop a hardware-key/keychain prompt
  (aws-vault, granted, 1Password). A profile picker must never trigger that.
* SSO / ``aws login`` profiles would hit the OIDC endpoint to refresh, i.e. the
  network, on every census render.

So profiles are classified from the *keys* in the merged config
(``Session().full_config``, which already folds ``~/.aws/credentials`` in), and
SSO / ``aws login`` session state is read from the on-disk token caches
(``~/.aws/sso/cache`` via botocore's ``SSOTokenLoader``; ``~/.aws/login/cache``
via ``LoginTokenLoader``) and compared against the wall clock. No secret value
ever leaves this module: the context carries names, kinds, Regions and session
*states* only.

Paths honour ``AWS_CONFIG_FILE`` / ``AWS_SHARED_CREDENTIALS_FILE`` (botocore's
own resolution) and ``AWS_LOGIN_CACHE_DIRECTORY``; tests point them at tmp
fixtures so the operator's real ``~/.aws`` is never read.
"""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from local_operator.providers.bedrock import connection as bconn

ProfileKind = Literal["static", "sso", "login", "assume_role", "process", "web_identity", "other"]
SessionState = Literal["valid", "expired_refreshable", "expired", "missing", "not_applicable"]

#: Input bounds shared with the desktop renderer (local-operator-ui
#: ``desktop-contract.ts`` ``awsRegion``/``awsProfileName``/``access_key_id``).
#: The Region becomes part of an AWS hostname and the profile name a
#: ``[profile …]`` lookup / an ``aws --profile`` argv element, so both are held
#: to the shapes AWS itself uses; the server validates again whatever the
#: renderer already checked (a compromised render path is the threat).
REGION_INPUT_RE = re.compile(r"^[a-z]{2}(-[a-z]+)+-\d{1,2}$")
PROFILE_INPUT_RE = re.compile(r"^[\w .@+/:=,-]{1,128}$")
ACCESS_KEY_ID_RE = re.compile(r"^[A-Z0-9]{16,128}$")

#: Regions where Bedrock runtime serves Anthropic models (directly or as the
#: source of a ``us.``/``eu.``/``apac.``/``global.`` inference profile), as of
#: 2026-11. botocore's offline endpoint data no longer lists ``bedrock-runtime``
#: (it moved to endpoint rulesets), so the list is curated here; "Other…" in
#: every picker accepts any shape-valid Region, so a Region AWS adds later is
#: never blocked by this table, only not suggested.
BEDROCK_REGIONS: tuple[str, ...] = (
    "us-east-1",
    "us-east-2",
    "us-west-2",
    "us-gov-west-1",
    "ca-central-1",
    "sa-east-1",
    "eu-central-1",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "eu-north-1",
    "eu-south-1",
    "eu-south-2",
    "eu-central-2",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    "ap-south-1",
    "ap-south-2",
    "ap-southeast-1",
    "ap-southeast-2",
    "ap-southeast-4",
)


def valid_region_input(region: object) -> bool:
    return isinstance(region, str) and len(region) <= 32 and bool(REGION_INPUT_RE.fullmatch(region))


def valid_profile_input(name: object) -> bool:
    return isinstance(name, str) and bool(PROFILE_INPUT_RE.fullmatch(name))


@dataclass(frozen=True)
class ProfileSession:
    state: SessionState
    expires_at: str | None = None
    login_command: str | None = None


@dataclass(frozen=True)
class AwsProfile:
    name: str
    kind: ProfileKind
    region: str | None
    session: ProfileSession = field(default_factory=lambda: ProfileSession("not_applicable"))

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _session(profile: str | None = None) -> Any:
    import botocore.session

    # A fresh Session per call, never auth.py's cached one: the picker must see
    # an ``~/.aws/config`` the operator edited a second ago, and a Session reads
    # its config files once.
    return botocore.session.Session(profile=profile) if profile else botocore.session.Session()


def _classify(config: Mapping[str, Any]) -> ProfileKind:
    # Order matters: an SSO profile may also carry ``role_arn``-free leftovers,
    # and an assume-role profile may chain to a static source. The FIRST
    # credential source botocore would try for that profile decides the kind
    # (botocore's provider chain checks these before static keys for a named
    # profile only via ``source_profile``; static keys in the profile itself
    # win, which is why "static" is checked after the explicit mechanisms).
    if config.get("sso_session") or config.get("sso_start_url"):
        return "sso"
    if config.get("login_session"):
        return "login"
    if config.get("role_arn"):
        return "web_identity" if config.get("web_identity_token_file") else "assume_role"
    if config.get("credential_process"):
        return "process"
    if config.get("web_identity_token_file"):
        return "web_identity"
    if config.get("aws_access_key_id"):
        return "static"
    return "other"


def login_command(name: str, kind: ProfileKind) -> str | None:
    """The command that refreshes this profile's session (shown, never run here).

    Quoted with ``shlex`` because profile names may contain spaces or ``/``
    and the operator is expected to paste this into a shell.
    """
    import shlex

    if kind == "sso":
        return f"aws sso login --profile {shlex.quote(name)}"
    if kind == "login":
        return f"aws login --profile {shlex.quote(name)}"
    return None


def _parse_expiry(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    text = raw.strip().replace("Z", "+00:00").replace("UTC", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _iso(when: datetime) -> str:
    return when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sso_cache_dir() -> str:
    # botocore's own default (credentials.py SSOProvider._SSO_TOKEN_CACHE_DIR).
    return os.path.expanduser(os.path.join("~", ".aws", "sso", "cache"))


def sso_status(
    name: str,
    *,
    full_config: Mapping[str, Any] | None = None,
    now: datetime | None = None,
    sso_cache_dir: str | None = None,
) -> ProfileSession:
    """Offline session state for one profile. Never touches the network.

    ``valid`` — the cached access token is unexpired. ``expired_refreshable`` —
    it expired but a refresh token is cached (botocore will refresh at use time,
    so this still works without a sign-in). ``expired`` — expired and not
    refreshable: a sign-in is needed. ``missing`` — no token cached at all.
    ``not_applicable`` — the profile is not SSO / ``aws login``.
    """
    from botocore.utils import (
        JSONFileCache,
        LoginTokenLoader,
        SSOTokenLoader,
        get_login_token_cache_directory,
    )

    config = full_config if full_config is not None else _session().full_config
    profiles = config.get("profiles", {}) if isinstance(config, Mapping) else {}
    profile = profiles.get(name) if isinstance(profiles, Mapping) else None
    if not isinstance(profile, Mapping):
        return ProfileSession("missing")
    kind = _classify(profile)
    command = login_command(name, kind)
    clock = now or datetime.now(UTC)
    if kind == "sso":
        session_name = profile.get("sso_session")
        start_url = profile.get("sso_start_url")
        if session_name and not start_url:
            sessions = config.get("sso_sessions", {})
            entry = sessions.get(session_name) if isinstance(sessions, Mapping) else None
            start_url = entry.get("sso_start_url") if isinstance(entry, Mapping) else None
        loader = SSOTokenLoader(cache=JSONFileCache(sso_cache_dir or _sso_cache_dir()))
        try:
            token = loader(start_url or "", session_name=session_name or None)
        except Exception:  # noqa: BLE001 - SSOTokenLoadError / unreadable cache file
            return ProfileSession("missing", None, command)
        expiry = _parse_expiry(token.get("expiresAt"))
        refreshable = bool(token.get("refreshToken"))
        registration = _parse_expiry(token.get("registrationExpiresAt"))
        if registration is not None and registration <= clock:
            refreshable = False
    elif kind == "login":
        loader = LoginTokenLoader(JSONFileCache(get_login_token_cache_directory()))
        try:
            token = loader.load_token(str(profile.get("login_session")))
        except Exception:  # noqa: BLE001 - unreadable cache file
            token = None
        if not isinstance(token, Mapping):
            return ProfileSession("missing", None, command)
        access = token.get("accessToken")
        expiry = _parse_expiry(access.get("expiresAt") if isinstance(access, Mapping) else None)
        refreshable = bool(token.get("refreshToken"))
    else:
        return ProfileSession("not_applicable")
    if expiry is None:
        return ProfileSession("missing", None, command)
    if expiry > clock:
        state: SessionState = "valid"
    else:
        state = "expired_refreshable" if refreshable else "expired"
    return ProfileSession(state, _iso(expiry), command)


def list_profiles(
    *, now: datetime | None = None, sso_cache_dir: str | None = None
) -> list[AwsProfile]:
    """Every configured profile with its kind, Region and session state.

    Reads config only (``full_config``); NEVER ``get_credentials()``. A config
    file botocore cannot parse raises its own ``ConfigParseError`` — the route
    turns that into ``bedrock_context_failed`` with the parser's text, because
    "your ~/.aws/config line 14 is broken" is the actionable fact.
    """
    config = _session().full_config
    profiles = config.get("profiles", {}) if isinstance(config, Mapping) else {}
    result: list[AwsProfile] = []
    for name in sorted(profiles):
        entry = profiles.get(name)
        if not isinstance(entry, Mapping):
            continue
        kind = _classify(entry)
        region = entry.get("region")
        region = region if isinstance(region, str) and bconn.valid_region(region) else None
        session = (
            sso_status(name, full_config=config, now=now, sso_cache_dir=sso_cache_dir)
            if kind in ("sso", "login")
            else ProfileSession("not_applicable")
        )
        result.append(AwsProfile(name=name, kind=kind, region=region, session=session))
    return result


def profile_region(name: str) -> str | None:
    """A profile's configured Region, from config only."""
    for profile in list_profiles():
        if profile.name == name:
            return profile.region
    return None


def region_labels() -> dict[str, str]:
    """Region id -> AWS's own display name, from botocore's offline partition data.

    AWS publishes these names ("US East (N. Virginia)") in English only in its
    SDK data; they are proper names of AWS infrastructure, shown beside the
    untranslated Region id.
    """
    try:
        # The bare data loader, not ``Session().get_data``: a Session binds to
        # ``AWS_PROFILE``, and a shell exporting a profile this config does not
        # define makes every Session data read raise ``ProfileNotFound`` —
        # which would blank the Region labels for a reason unrelated to them.
        from botocore.loaders import create_loader

        data = create_loader().load_data("endpoints")
    except Exception:  # noqa: BLE001 - a broken botocore install still lists ids
        return {}
    labels: dict[str, str] = {}
    for partition in data.get("partitions", []) if isinstance(data, Mapping) else []:
        for region_id, meta in (partition.get("regions") or {}).items():
            description = meta.get("description") if isinstance(meta, Mapping) else None
            if isinstance(description, str) and description:
                labels[region_id] = description
    return labels


def regions() -> list[dict[str, str]]:
    labels = region_labels()
    return [
        {"id": region, "label": labels.get(region, region), "geo_prefix": bconn.geo_prefix(region)}
        for region in BEDROCK_REGIONS
    ]


DefaultRegionSource = Literal["config", "AWS_REGION", "AWS_DEFAULT_REGION", "profile", "fallback"]


def default_region(
    *,
    profile_region_value: str | None = None,
    values: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, DefaultRegionSource]:
    """The §5.2 precedence with its SOURCE, so the UI can say where it came from.

    Mirrors ``connection.resolve_region`` for a connection with no Region of
    its own; kept beside it rather than changing that pure function's return
    type, which inference and discovery already depend on.
    """
    env = os.environ if environ is None else environ
    candidates: tuple[tuple[str, DefaultRegionSource], ...] = (
        (bconn.configured_region(values), "config"),
        (env.get("AWS_REGION", ""), "AWS_REGION"),
        (env.get("AWS_DEFAULT_REGION", ""), "AWS_DEFAULT_REGION"),
        (profile_region_value or "", "profile"),
    )
    for candidate, source in candidates:
        candidate = candidate.strip()
        if candidate and bconn.valid_region(candidate):
            return candidate, source
    return bconn.FALLBACK_REGION, "fallback"


def ambient_sources(
    environ: Mapping[str, str] | None = None, values: Mapping[str, Any] | None = None
) -> tuple[bool, list[str]]:
    """Whether the env tier enables Bedrock, and which ambient signals exist.

    ``sources`` lists what the default chain WOULD find (names only), so the
    "Use this environment" choice can say what it is about to use;
    ``enabled`` is the env tier's own verdict (design D6: ``AWS_PROFILE`` /
    ``AWS_ACCESS_KEY_ID`` alone never enable it).
    """
    env = os.environ if environ is None else environ
    sources: list[str] = []
    if env.get(bconn.BEARER_ENV):
        sources.append(f"env:{bconn.BEARER_ENV}")
    if env.get("AWS_PROFILE"):
        sources.append("env:AWS_PROFILE")
    if env.get("AWS_ACCESS_KEY_ID"):
        sources.append("env:AWS_ACCESS_KEY_ID")
    if env.get("AWS_CONTAINER_CREDENTIALS_FULL_URI") or env.get(
        "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"
    ):
        sources.append("container")
    if env.get("AWS_WEB_IDENTITY_TOKEN_FILE"):
        sources.append("web_identity")
    if env.get("AWS_LAMBDA_FUNCTION_NAME"):
        sources.append("lambda")
    if (env.get(bconn.AUTH_ENV) or "").strip().lower() == "ambient":
        sources.append(f"env:{bconn.AUTH_ENV}")
    if bconn.configured_auth(values) == "ambient":
        sources.append("config")
    enabled = bconn.bedrock_env_key(env, values) is not None
    return enabled, sources


def aws_cli_path() -> str | None:
    return shutil.which("aws")
