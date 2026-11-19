"""Interactive Amazon Bedrock connect: the shared core and the terminal login.

Design §7/§8. Bedrock is the one provider whose "login" is a STRUCTURED
choice — a method, a Region, maybe a profile, maybe two secrets — so it is
neither a paste-a-key nor an OAuth flow. Two hosts drive the same core:

* :func:`login_bedrock` — the registry login (``__lo_login_kind__ = "aws"``)
  that ``lop login bedrock`` and the TUI's ``/login`` run. It asks through
  ``LoginCallbacks.on_choice`` (a picker) and ``on_setup_input`` (masked
  fields), validates, and returns the row dict ``ProviderController.login``
  upserts — the AuthStore seal hook moves any secret into the encrypted store
  before auth.db sees the row.
* ``PUT /v1/auth/providers/bedrock/connection`` (``server/routes/auth.py``) —
  the desktop panel, which posts a :class:`ConnectionInput` it already
  collected and stores the row this module builds.

Both go through :func:`validate_input` -> :func:`check_input` ->
:func:`build_row`, so a connection is checked the same way whichever surface
made it, and secret material is checked BEFORE it is stored (``save_key``'s
rule: an unchecked key is never written).

Secrets are never logged, echoed or put in a message: inputs arrive through
masked prompts / ``SecretStr``, live in local variables only until the row is
handed to the store, and every user-facing sentence names the connection by
its non-secret label (``dev · us-east-1``).
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from local_operator.providers.bedrock import connection as bconn
from local_operator.providers.bedrock import profiles as bprofiles

if TYPE_CHECKING:
    from local_operator.harness.types import AbortSignal
    from local_operator.providers.key_check import KeyCheck
    from local_operator.providers.oauth.callback_server import LoginCallbacks

logger = logging.getLogger(__name__)

#: The registry reads this off the login callable to derive
#: ``ProviderDefinition.login_kind`` ("aws"), the same way paste-a-key logins
#: carry ``__lo_api_key_login__``. Hosts branch on the kind, never on the id.
LOGIN_KIND_ATTR = "__lo_login_kind__"
LOGIN_KIND = "aws"

#: The default-chain method's descriptor spelling, taken from the descriptor
#: vocabulary itself so the two can never drift (wire and descriptor agree on
#: this one; the others differ by ``_`` vs ``-``).
DEFAULT_CHAIN: Final[str] = bconn.METHODS[3]
#: Wire method (desktop contract, underscores) -> descriptor method (hyphens).
DESCRIPTOR_METHOD: Final[dict[str, str]] = {
    "api_key": "api-key",
    "profile": "profile",
    "access_key": "access-key",
    DEFAULT_CHAIN: DEFAULT_CHAIN,
}
INPUT_METHODS: Final[tuple[str, ...]] = tuple(DESCRIPTOR_METHOD)
#: The reverse, for the context's ``connections`` list. ``env-bearer`` keeps
#: its hyphen: the contract spells it that way (contract delta 4).
WIRE_METHOD: Final[dict[str, str]] = {
    **{descriptor: wire for wire, descriptor in DESCRIPTOR_METHOD.items()},
    "env-bearer": "env-bearer",
}
#: Bound for one connection check, the order of ``key_check``'s own probes.
CHECK_TIMEOUT_S = 15.0
#: Re-asks for one malformed field before the flow gives up (typos happen; an
#: endless loop on a host that keeps returning junk does not).
_MAX_ATTEMPTS = 3
#: Picker value for "type a Region"; not a shape-valid Region, so it can never
#: collide with a real one.
_OTHER = "__other__"
#: Chooser value for the "AWS SSO" entry. Not a wire method: it collects a
#: ``profile`` connection, filtered to the sign-in-capable kinds below.
_SSO_METHOD = "sso"
#: Profile kinds listed under "AWS SSO" (IAM Identity Center and ``aws login``);
#: every other kind is listed under "AWS profile".
_SSO_KINDS: Final[frozenset[str]] = frozenset({"sso", "login"})


def _k():  # noqa: ANN202 - generated module
    from local_operator.i18n.keys import wire_login_bedrock

    return wire_login_bedrock


def _t(message: Any) -> str:
    from local_operator.i18n.messages import render_resolved

    return render_resolved(message)


class InputInvalid(ValueError):
    """A connect request outside the contract's bounds; ``code``/``field`` say which."""

    def __init__(self, code: str, field: str, value: str | None = None) -> None:
        super().__init__(f"{code}:{field}")
        self.code = code
        self.field = field
        #: Only for NON-secret fields (the Region), so a message can name it.
        self.value = value


@dataclass(frozen=True)
class ConnectionInput:
    """One connection as the user chose it.

    Secret fields are plain ``str`` here because this object lives only for
    the duration of one connect call; ``repr`` is overridden so a stray log
    line or traceback frame can never print them.
    """

    method: str
    region: str | None = None
    profile: str | None = None
    api_key: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None

    def __repr__(self) -> str:
        return (
            f"ConnectionInput(method={self.method!r}, region={self.region!r}, "
            f"profile={self.profile!r})"
        )


def validate_input(inp: ConnectionInput) -> None:
    """The contract's bounds (contract delta 2). Raises :class:`InputInvalid`.

    Region ``^[a-z]{2}(-[a-z]+)+-\\d{1,2}$``, profile ``^[\\w .@+/:=,-]+$`` 1-128,
    access key id ``^[A-Z0-9]{16,128}$``. The renderer checks the same shapes;
    this is the server half of that fence.
    """
    if inp.method not in INPUT_METHODS:
        raise InputInvalid("bedrock_input_invalid", "method")
    if inp.region is not None and not bprofiles.valid_region_input(inp.region):
        raise InputInvalid("bedrock_region_invalid", "region", inp.region[:40])
    if inp.method != DEFAULT_CHAIN and inp.region is None:
        raise InputInvalid("bedrock_region_invalid", "region", "")
    if inp.method == "profile" and not bprofiles.valid_profile_input(inp.profile):
        raise InputInvalid("bedrock_input_invalid", "profile")
    if inp.method == "api_key" and not (inp.api_key or "").strip():
        raise InputInvalid("bedrock_input_invalid", "api_key")
    if inp.method == "access_key":
        if not bprofiles.ACCESS_KEY_ID_RE.fullmatch(inp.access_key_id or ""):
            raise InputInvalid("bedrock_input_invalid", "access_key_id")
        if not (inp.secret_access_key or "").strip():
            raise InputInvalid("bedrock_input_invalid", "secret_access_key")


def effective_region(inp: ConnectionInput) -> str:
    """The Region this connection will use; the chosen one, else §5.2 precedence."""
    if inp.region:
        return inp.region
    profile_region = bprofiles.profile_region(inp.profile) if inp.profile else None
    region, _source = bprofiles.default_region(profile_region_value=profile_region)
    return region


def _key_fingerprint(access_key_id: str) -> str:
    # An access key ID is an identifier, not a secret (AWS prints it in
    # CloudTrail), but the row label still shows only its ends.
    return f"{access_key_id[:4]}…{access_key_id[-4:]}"


def _descriptor_connection(inp: ConnectionInput, region: str | None) -> bconn.Connection:
    method = DESCRIPTOR_METHOD[inp.method]
    return bconn.Connection(
        method=method,  # type: ignore[arg-type]  # validated against INPUT_METHODS
        region=region,
        profile=inp.profile if inp.method == "profile" else None,
    )


def label_for(inp: ConnectionInput, region: str) -> str:
    if inp.method == "access_key" and inp.access_key_id:
        return f"{_key_fingerprint(inp.access_key_id)} · {region}"
    return bconn.account_label(_descriptor_connection(inp, region), region)


def build_row(inp: ConnectionInput, region: str) -> dict[str, Any]:
    """The auth.db row (design §5.2). Secret material rides in ``secret`` only.

    The AuthStore seal hook replaces the placeholder ``ref`` with a fresh one
    after sealing ``secret``, and drops ``secret`` before serialisation.
    ``account_id`` is the row's non-secret identity (re-saving the same
    profile + Region replaces the row instead of adding a duplicate).
    """
    descriptor_method = DESCRIPTOR_METHOD[inp.method]
    # The default chain keeps the Region the user explicitly picked; with none
    # it follows the environment at use time (§5.2), so none is stored.
    stored_region = region if (inp.method != DEFAULT_CHAIN or inp.region) else None
    label = label_for(inp, region)
    meta: dict[str, Any] = {"method": descriptor_method, "label": label}
    if stored_region:
        meta["region"] = stored_region
    row: dict[str, Any] = {"type": "api_key", "source": "login", "bedrock": meta}
    if inp.method in ("api_key", "access_key"):
        placeholder = bconn.REF_PREFIX + "0" * 32
        row["key"] = bconn.format_descriptor(
            descriptor_method, region=stored_region, ref=placeholder
        )
        if inp.method == "api_key":
            row["secret"] = (inp.api_key or "").strip()
            row["account_id"] = f"api-key@{region}"
        else:
            row["secret"] = {
                "access_key_id": inp.access_key_id,
                "secret_access_key": (inp.secret_access_key or "").strip(),
                "session_token": (inp.session_token or "").strip() or None,
            }
            row["account_id"] = f"access-key:{_key_fingerprint(inp.access_key_id or '')}@{region}"
    else:
        row["key"] = bconn.format_descriptor(
            descriptor_method, region=stored_region, profile=inp.profile
        )
        who = inp.profile if inp.method == "profile" else DEFAULT_CHAIN
        row["account_id"] = f"{descriptor_method}:{who}@{region}"
        if inp.profile:
            meta["profile"] = inp.profile
    return row


async def check_input(
    inp: ConnectionInput, region: str, *, timeout: float = CHECK_TIMEOUT_S
) -> "KeyCheck":
    """Validate BEFORE storing (design §5.5). Never raises.

    Typed secret material is checked from memory (it has no ``ref`` yet);
    profile / default-chain connections are checked through their descriptor,
    i.e. the exact resolution path the runtime will use.
    """
    from local_operator.providers.bedrock import auth as bauth
    from local_operator.providers.bedrock import check as bcheck

    if inp.method == "api_key":
        resolved = bauth.ResolvedAuth(region=region, bearer=(inp.api_key or "").strip())
        return await bcheck.probe(resolved, timeout)
    if inp.method == "access_key":
        from botocore.credentials import Credentials

        creds = Credentials(
            inp.access_key_id or "",
            (inp.secret_access_key or "").strip(),
            (inp.session_token or "").strip() or None,
        ).get_frozen_credentials()
        return await bcheck.probe(bauth.ResolvedAuth(region=region, credentials=creds), timeout)
    descriptor = _descriptor_connection(inp, region).descriptor()
    return await bcheck.check_connection(descriptor, timeout)


# -- terminal / TUI flow ------------------------------------------------------

Choice = tuple[str, str, str]  # (value, label, description)


async def _await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _report(hook: Any, *args: Any) -> None:
    from local_operator.providers.oauth.callback_server import report_safely

    await report_safely(hook, *args)


def _cancelled() -> Exception:
    from local_operator.providers.oauth.callback_server import LoginCancelledError

    return LoginCancelledError(_t(_k().cancelled()))


def _no_tty_error() -> Exception:
    # A LoginError, not ValueError: every host reports a LoginError as "Login
    # failed: <reason>", while a bare ValueError reached `lop login`'s generic
    # crash handler and printed a stack trace for a plain "no terminal" case.
    from local_operator.providers.oauth.callback_server import LoginError

    return LoginError(_no_tty_text())


def _no_tty_text() -> str:
    return _t(
        _k().no_tty(bearer_env=bconn.BEARER_ENV, auth_env=f"{bconn.AUTH_ENV}={DEFAULT_CHAIN}")
    )


async def choose(
    callbacks: "LoginCallbacks", title: str, options: Sequence[Choice], default: str | None
) -> str | None:
    """One single-select question; ``None`` is a cancel.

    Uses ``on_choice`` when the host has a picker. Hosts without one (older
    embedders) get a numbered list through ``on_setup_input`` so they keep
    working — the fallback the design promised (§8), not a second UI.
    """
    hook = getattr(callbacks, "on_choice", None)
    if hook is not None:
        value = await _await(hook(title, list(options), default))
        return value if value in {option[0] for option in options} else None
    setup = callbacks.on_setup_input
    if setup is None:
        raise _no_tty_error()
    k = _k()
    lines = [title] + [
        f"  {index}. {label}" + (f" — {description}" if description else "")
        for index, (_value, label, description) in enumerate(options, start=1)
    ]
    default_index = next(
        (str(index) for index, option in enumerate(options, 1) if option[0] == default), ""
    )
    for _attempt in range(_MAX_ATTEMPTS):
        answer = await _await(setup("\n".join(lines), default_index, False))
        if answer is None:
            return None
        answer = str(answer).strip() or default_index
        if not answer:
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1][0]
        await _report(callbacks.on_warning or callbacks.on_progress, _t(k.choice_invalid()))
    return None


async def _ask(callbacks: "LoginCallbacks", label: str, *, secret: bool) -> str | None:
    setup = callbacks.on_setup_input
    if setup is None:
        raise _no_tty_error()
    value = await _await(setup(label, "", secret))
    return None if value is None else str(value)


def _session_text(session: bprofiles.ProfileSession) -> str | None:
    k = _k()
    if session.state == "valid":
        return _t(k.session_valid(expires=session.expires_at or ""))
    if session.state == "expired_refreshable":
        return _t(k.session_refreshable())
    if session.state == "expired":
        return _t(k.session_expired())
    if session.state == "missing":
        return _t(k.session_missing())
    return None


def kind_text(kind: str) -> str:
    k = _k()
    return _t(
        {
            "static": k.kind_static,
            "sso": k.kind_sso,
            "login": k.kind_login,
            "assume_role": k.kind_assume_role,
            "process": k.kind_process,
            "web_identity": k.kind_web_identity,
        }.get(kind, k.kind_other)()
    )


def profile_choice(profile: bprofiles.AwsProfile) -> Choice:
    k = _k()
    region = profile.region or _t(k.region_unset())
    session = _session_text(profile.session)
    kind = kind_text(profile.kind)
    description = (
        _t(k.profile_desc(kind=kind, region=region, session=session))
        if session
        else _t(k.profile_desc_short(kind=kind, region=region))
    )
    return (profile.name, profile.name, description)


def needs_sign_in(profile: bprofiles.AwsProfile) -> bool:
    """SSO / ``aws login`` with no usable token. Refreshable sessions need none."""
    return profile.kind in ("sso", "login") and profile.session.state in ("expired", "missing")


async def choose_region(callbacks: "LoginCallbacks", profile_region: str | None) -> str | None:
    """Region picker: the default first (profile's Region beats the rest), then Other….

    ``None`` only when the user cancels the PICKER itself. "Other…" types a
    Region: an invalid id re-asks, and an empty or cancelled entry (or too many
    invalid ones) goes back to the picker rather than ending the login — by
    this step the user may already have typed a key or secret, and losing it
    to a typo is what #4270 design D2 flagged.
    """
    k = _k()
    region, source = bprofiles.default_region(profile_region_value=profile_region)
    source_text = {
        "config": _t(k.region_source_config()),
        "profile": _t(k.region_source_profile()),
        "fallback": _t(k.region_source_fallback()),
    }.get(
        source, source
    )  # AWS_REGION / AWS_DEFAULT_REGION stay env var names
    labels = bprofiles.region_labels()
    options: list[Choice] = [(region, region, _t(k.region_default_desc(source=source_text)))]
    options += [
        (candidate, candidate, labels.get(candidate, ""))
        for candidate in bprofiles.BEDROCK_REGIONS
        if candidate != region
    ]
    options.append((_OTHER, _t(k.region_other()), _t(k.region_other_desc())))
    while True:
        picked = await choose(callbacks, _t(k.region_title()), options, region)
        if picked != _OTHER:
            return picked
        for _attempt in range(_MAX_ATTEMPTS):
            typed = await _ask(callbacks, _t(k.region_prompt()), secret=False)
            if typed is None or not typed.strip():
                break
            typed = typed.strip().lower()
            if bprofiles.valid_region_input(typed):
                return typed
            await _report(
                callbacks.on_warning or callbacks.on_progress,
                _t(k.region_invalid(region=typed[:40])),
            )


async def _sign_in_if_needed(
    callbacks: "LoginCallbacks", profile: bprofiles.AwsProfile, signal: "AbortSignal | None"
) -> None:
    """Offer ``aws sso login`` for an expired SSO / ``aws login`` profile.

    Runs it only when the user picks "Sign in now" (design hard rule). "I'll
    run it myself", a refusal (no ``aws``, headless) or a failure ends the
    login with the command to run — never a silent retry.
    """
    from local_operator.providers.bedrock import sso_login
    from local_operator.providers.oauth.callback_server import (
        LoginDeferredError,
        LoginError,
    )

    k = _k()
    command = bprofiles.login_command(profile.name, profile.kind) or ""
    title = (
        k.sso_title(profile=profile.name)
        if profile.session.state == "expired"
        else k.sso_title_missing(profile=profile.name)
    )
    options: list[Choice] = [
        ("run", _t(k.sso_run()), _t(k.sso_run_desc(command=command))),
        ("self", _t(k.sso_self()), _t(k.sso_self_desc(command=command))),
    ]
    picked = await choose(callbacks, _t(title), options, "run")
    if picked is None:
        raise _cancelled()
    if picked != "run":
        # A choice, not a failure (#4270 D4/Q7): hosts render this neutrally,
        # the CLI exits 0, and the command is carried separately so it can be
        # set apart for copying. Raised before any row exists: nothing stored.
        raise LoginDeferredError(_t(k.sso_deferred()), command=command)
    await _report(callbacks.on_progress, _t(k.sso_started(profile=profile.name)))
    try:
        result = await sso_login.run_sso_login(
            profile.name, kind=profile.kind, callbacks=callbacks, abort=signal
        )
    except sso_login.SsoLoginRefused as refused:
        if refused.code == "aws_cli_missing":
            raise LoginError(_t(k.sso_cli_missing())) from None
        if refused.code == "headless":
            raise LoginError(_t(k.sso_headless(command=command))) from None
        raise LoginError(_t(k.sso_not_applicable(profile=profile.name))) from None
    except sso_login.SsoLoginFailed as failed:
        if failed.timed_out:
            raise LoginError(_t(k.sso_timeout(profile=profile.name))) from None
        raise LoginError(_t(k.sso_failed(profile=profile.name, command=command))) from None
    await _report(callbacks.on_progress, _t(k.sso_done(profile=profile.name)))
    if result.session.state in ("expired", "missing"):
        # The CLI exited 0 but no token landed (another cache dir, a cancelled
        # browser step): say what to run rather than store a dead connection.
        raise LoginError(_t(k.sso_run_yourself(command=command)))


def is_sso_profile(profile: bprofiles.AwsProfile) -> bool:
    """Listed under "AWS SSO" rather than "AWS profile": the sign-in-capable kinds."""
    return profile.kind in _SSO_KINDS


async def _choose_method(
    callbacks: "LoginCallbacks", found: Sequence[bprofiles.AwsProfile]
) -> str | None:
    """The method chooser, in the canonical order the desktop shares (#4270 D1).

    AWS profile · AWS SSO · Access key · Bedrock API key · Default credentials.
    The order and labels are one list across the TUI, the CLI and the desktop
    connect dialog; change them in all three or in none.
    """
    k = _k()
    ambient_on, sources = bprofiles.ambient_sources()
    plain = [p for p in found if not is_sso_profile(p)]
    sso = [p for p in found if is_sso_profile(p)]
    profile_desc = (
        _t(k.method_profile_desc(count=len(plain))) if plain else _t(k.method_profile_none())
    )
    sso_desc = _t(k.method_sso_desc(count=len(sso))) if sso else _t(k.method_sso_none())
    ambient_desc = (
        _t(k.method_ambient_desc(sources=", ".join(sources)))
        if sources
        else _t(k.method_ambient_desc_none())
    )
    options: list[Choice] = [
        ("profile", _t(k.method_profile()), profile_desc),
        (_SSO_METHOD, _t(k.method_sso()), sso_desc),
        ("access_key", _t(k.method_access_key()), _t(k.method_access_key_desc())),
        ("api_key", _t(k.method_api_key()), _t(k.method_api_key_desc())),
        (DEFAULT_CHAIN, _t(k.method_ambient()), ambient_desc),
    ]
    if plain:
        default = "profile"
    elif sso:
        default = _SSO_METHOD
    else:
        default = "api_key" if not ambient_on else DEFAULT_CHAIN
    return await choose(callbacks, _t(k.method_title()), options, default)


async def _collect(
    callbacks: "LoginCallbacks",
    method: str,
    found: Sequence[bprofiles.AwsProfile],
    signal: "AbortSignal | None",
) -> ConnectionInput | None:
    """The inputs for one method; ``None`` sends the user back to the chooser."""
    k = _k()
    if method in ("profile", _SSO_METHOD):
        # Both entries store a ``profile`` connection; they differ only in
        # which profiles they list, so SSO users are not hunting through
        # static-key profiles for theirs (#4270 D1).
        want_sso = method == _SSO_METHOD
        listed = [p for p in found if is_sso_profile(p) == want_sso]
        if not listed:
            empty = k.no_sso_profiles() if want_sso else k.no_profiles()
            await _report(callbacks.on_warning or callbacks.on_progress, _t(empty))
            return None
        default = next((p.name for p in listed if p.name == "default"), listed[0].name)
        name = await choose(
            callbacks,
            _t(k.sso_profile_title() if want_sso else k.profile_title()),
            [profile_choice(p) for p in listed],
            default,
        )
        if name is None:
            raise _cancelled()
        profile = next(p for p in listed if p.name == name)
        if needs_sign_in(profile):
            await _sign_in_if_needed(callbacks, profile, signal)
        region = await choose_region(callbacks, profile.region)
        if region is None:
            raise _cancelled()
        return ConnectionInput("profile", region=region, profile=name)
    if method == "api_key":
        key = await _ask(callbacks, _t(k.api_key_prompt()), secret=True)
        if key is None or not key.strip():
            raise _cancelled()
        region = await choose_region(callbacks, None)
        if region is None:
            raise _cancelled()
        return ConnectionInput("api_key", region=region, api_key=key.strip())
    if method == "access_key":
        key_id: str | None = None
        for _attempt in range(_MAX_ATTEMPTS):
            typed = await _ask(callbacks, _t(k.access_key_id_prompt()), secret=False)
            if typed is None or not typed.strip():
                raise _cancelled()
            if bprofiles.ACCESS_KEY_ID_RE.fullmatch(typed.strip()):
                key_id = typed.strip()
                break
            await _report(
                callbacks.on_warning or callbacks.on_progress, _t(k.access_key_id_invalid())
            )
        if key_id is None:
            raise _cancelled()
        secret = await _ask(callbacks, _t(k.secret_access_key_prompt()), secret=True)
        if secret is None or not secret.strip():
            raise _cancelled()
        token = await _ask(callbacks, _t(k.session_token_prompt()), secret=True)
        if token is None:
            raise _cancelled()
        region = await choose_region(callbacks, None)
        if region is None:
            raise _cancelled()
        return ConnectionInput(
            "access_key",
            region=region,
            access_key_id=key_id,
            secret_access_key=secret.strip(),
            session_token=token.strip() or None,
        )
    region = await choose_region(callbacks, None)
    if region is None:
        raise _cancelled()
    return ConnectionInput(DEFAULT_CHAIN, region=region)


async def login_bedrock(
    callbacks: "LoginCallbacks", *, signal: "AbortSignal | None" = None, **_kwargs: Any
) -> dict[str, Any]:
    """The ``lop login bedrock`` / TUI ``/login bedrock`` flow (design §8).

    Returns the row for ``ProviderController.login`` to upsert. A definite
    rejection raises ``LoginError`` with the keyed reason (nothing stored); an
    unverified check (timeout, list-denied) still returns the row, as
    ``save_key`` stores an unverifiable key, and says so as a warning.
    """
    from local_operator.providers.oauth.callback_server import LoginError

    if callbacks.on_setup_input is None and getattr(callbacks, "on_choice", None) is None:
        raise _no_tty_error()
    k = _k()
    try:
        found = bprofiles.list_profiles()
    except Exception as exc:  # noqa: BLE001 - a broken ~/.aws/config must not end the login
        logger.debug("AWS profiles unreadable: %s", type(exc).__name__)
        await _report(
            callbacks.on_warning or callbacks.on_progress,
            _t(k.context_unreadable(detail=str(exc).strip()[:300] or type(exc).__name__)),
        )
        found = []
    inp: ConnectionInput | None = None
    while inp is None:
        method = await _choose_method(callbacks, found)
        if method is None:
            raise _cancelled()
        inp = await _collect(callbacks, method, found, signal)
    validate_input(inp)
    region = effective_region(inp)
    await _report(callbacks.on_progress, _t(k.checking(region=region)))
    verdict = await check_input(inp, region)
    if verdict.valid is False:
        raise LoginError(verdict.reason or _t(k.cancelled()))
    if verdict.valid is None and verdict.reason:
        await _report(callbacks.on_warning or callbacks.on_progress, verdict.reason)
    return build_row(inp, region)


setattr(login_bedrock, LOGIN_KIND_ATTR, LOGIN_KIND)
