"""Compatibility readers for legacy Radient clients, backed by AuthStore.

Legacy clients accept SecretStr and perform synchronous HTTP; credential selection
must nevertheless share provider precedence and the sole OAuth refresh store.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from pydantic import SecretStr

from local_operator.providers.auth_store import AuthStore, OAuthAccess
from local_operator.providers.registry import get_provider_definition

if TYPE_CHECKING:
    from local_operator.config import ConfigManager
    from local_operator.network.credentials.store import MeshAwareAuthStore


def configured_radient_base_url(config_manager: "ConfigManager") -> str:
    """The hub API root a ``ConfigManager``'s configuration resolves to.

    THE one read of ``config.yml``'s ``radient_base_url`` for hub consumers:
    the NESTED ``values.radient_base_url`` the config store actually holds wins
    when set (a flat document-root key is dropped by the migration and would be
    silently ignored), and :func:`resolve_radient_api_base_url` supplies the
    ``RADIENT_API_BASE_URL``/canonical default otherwise, version segment
    included. The CLI's ``_radient_hub_base_url`` and the sync coordinator's
    ``agent_sync.hub_base_url`` both delegate here rather than each spelling
    the rule — one configuration resolving two destinations is the bug the
    single-place rule exists to prevent (agent review round 1, n2; the shape is
    asserted in ``tests/unit/test_radient_hub_base_resolution.py``).
    """

    from local_operator.env import resolve_radient_api_base_url

    return resolve_radient_api_base_url(config_manager.get_config_value("radient_base_url", None))


def _radient_api_key(config_dir: Path | None) -> SecretStr:
    """The `RADIENT_API_KEY` value: the provider store row, then the environment.

    Store-first, matching every other provider-key reader. The legacy
    ``credentials.env`` rung is GONE (PR2a): a name the store does not hold and
    the environment does not export resolves to an EMPTY ``SecretStr`` rather
    than to a file the consolidation no longer reads.
    """
    from local_operator.providers.registry import provider_secret_value

    stored = provider_secret_value("RADIENT_API_KEY", base=config_dir)
    if stored:
        return SecretStr(stored)
    return SecretStr(os.environ.get("RADIENT_API_KEY", ""))


def canonical_radient_destination(base_url: str) -> bool:
    definition = get_provider_definition("radient")
    if definition is None or not definition.base_url:
        return False
    try:
        requested = urlsplit(base_url)
        canonical = urlsplit(definition.base_url)
        return (
            not requested.username
            and not requested.password
            and not requested.query
            and not requested.fragment
            and (requested.scheme, requested.netloc) == (canonical.scheme, canonical.netloc)
            # Legacy marketplace CLI methods historically join their own /v1.
            and requested.path.rstrip("/") in {"", canonical.path.rstrip("/")}
        )
    except ValueError:
        return False


#: Explicit opt-in for a NON-canonical organization hub (the harness's own
#: local/QA runs, a deliberately hosted staging hub): see
#: :func:`org_oauth_destination_allowed`. OFF by default; nothing infers it.
ORG_ALLOW_NONCANONICAL_ENV = "RADIENT_ORG_ALLOW_NONCANONICAL_BASE"

#: The remedy for an org call with no signed-in account. ONE copy: the CLI
#: prints it behind its own ``Error:`` prefix and the local server answers a
#: 401 with it verbatim, so the two surfaces cannot drift apart (agent review
#: round 1, MINOR-2 -- each side used to pin only its own copy).
ORG_LOGIN_REMEDY = (
    "organization operations need a signed-in Radient account. Run `lop login radient`."
)

#: The configuration remedy for an org call at a hub the account's bearer must
#: not be sent to, with two placeholders: the canonical host is quoted at
#: runtime from ``env.DEFAULT_RADIENT_API_BASE_URL`` (a URL-shaped literal in
#: cli.py or the server routes would break the single-reader invariant
#: ``test_radient_hub_base_resolution`` pins) and the opt-in is quoted from
#: :data:`ORG_ALLOW_NONCANONICAL_ENV`. See :func:`org_destination_refused_sentence`.
ORG_DESTINATION_REFUSED_TEMPLATE = (
    "organization operations are refused because the configured Radient hub is not the "
    "Radient cloud API ({base_url}), and the signed-in account's bearer is not sent "
    "to other hosts. Point `radient_base_url` at {default_url}, or set "
    "{opt_in}=1 if this hub is local or staging."
)


def org_destination_refused_sentence(base_url: str, default_url: str) -> str:
    """The rendered configuration remedy for a hub at ``base_url``.

    ONE renderer for both consumers (the CLI's ``_resolve_org_client`` and the
    local server's ``_org_radient_credentials``): the sentence the user reads is
    the same one whichever surface refused.

    The base is quoted as its sanitized ORIGIN, never raw (security round 1 on
    #3588, S-1): both callers hand over the configured value, and a base
    spelled with URL credentials would otherwise print the password into a CLI
    error or a server response body. See :func:`_sanitized_origin`.
    """
    return ORG_DESTINATION_REFUSED_TEMPLATE.format(
        base_url=_sanitized_origin(base_url),
        default_url=default_url,
        opt_in=ORG_ALLOW_NONCANONICAL_ENV,
    )


#: What a credentialed base is shown as when no trustworthy scheme precedes its
#: credentials (security round 2 on #3588, S-4). Fixed text, so nothing from
#: the configured value can ride along, and no ``@`` so a log scan for one stays
#: a clean leak signal.
INVALID_HUB_BASE_PLACEHOLDER = "<invalid hub base>"

#: A leading ``scheme://``. The character class has no ``@``, so a match is by
#: construction a prefix with no ``@`` before it — the one condition under
#: which the scheme is provably not part of a credential.
_SCHEME_PREFIX = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")


def strip_base_credentials(base_url: str) -> tuple[str, bool]:
    """``base_url`` with every byte up to its LAST ``@`` gone, and whether it had one.

    ANY ``@`` IS CREDENTIALS: a hub API base never legitimately carries one,
    so this deliberately ignores RFC 3986's authority boundary (security round 1
    on #3588, S-1: a password with a raw ``/``, ``?`` or ``#`` ends that
    authority early).

    THE LAST ``@`` ACROSS THE WHOLE VALUE, not only after the first ``://``
    (security round 2, S-4): a missing-slashes typo with a URL later in the
    query (``https:ops:<pw>@hub/?r=https://x``) or a bare ``<pw>@https://hub``
    puts the credential BEFORE the first ``://``, and splitting on that first
    separator kept it as the "scheme". The scheme is re-attached only when the
    value itself starts with a valid ``scheme://`` (which cannot contain an
    ``@``); otherwise the whole value becomes
    :data:`INVALID_HUB_BASE_PLACEHOLDER`. Such a base is refused anyway, so
    the result only has to be credential-free, not faithful.

    Pure string work, never ``urlsplit`` (which raises on an unclosed ``[``),
    so it cannot fail on a malformed base and leave it unstripped.
    """

    if "@" not in base_url:
        return base_url, False
    remainder = base_url.rpartition("@")[2]
    scheme = _SCHEME_PREFIX.match(base_url)
    if scheme is None:
        return INVALID_HUB_BASE_PLACEHOLDER, True
    return scheme.group(0) + remainder, True


def _sanitized_origin(base_url: str) -> str:
    """``scheme://host[:port]`` of ``base_url``, lower-cased, with no credential byte.

    The ``hub_origin`` shape, built by hand so it can never raise on a malformed
    base (``urlsplit`` does on an unclosed ``[``). Credentials go first, through
    :func:`strip_base_credentials`, so no byte from before ANY ``@`` survives —
    including an ``@`` placed before the first ``://`` (security round 2 on
    #3588, S-4) — and a credentialed base without a trustworthy scheme is shown
    only as :data:`INVALID_HUB_BASE_PLACEHOLDER`.
    """
    stripped, _had_credentials = strip_base_credentials(str(base_url or "").strip())
    if stripped == INVALID_HUB_BASE_PLACEHOLDER:
        return stripped
    scheme, separator, rest = stripped.partition("://")
    if not separator:
        scheme, rest = "", scheme
    authority = rest
    for mark in "/?#\\":
        authority = authority.split(mark, 1)[0]
    origin = f"{scheme}://{authority}" if separator else authority
    return origin.lower()


#: The accepted spellings of a truthy environment switch.
_TRUTHY_ENV_VALUES = frozenset({"1", "true", "yes", "on"})


def org_oauth_destination_allowed(base_url: str) -> bool:
    """Whether the signed-in PERSON's org bearer may travel to ``base_url``.

    Organization calls attach the stored OAuth access token (design §8.3), and
    that token is a central credential: the boundary the public resolver keeps
    ("an explicit legacy gateway must not receive a centrally signed-in
    account's bearer") therefore applies here too, enforced by
    :func:`resolve_radient_oauth_access` BEFORE any credential is attached.
    Org routes exist only on the hub, so a non-canonical destination has no
    fallback credential to take -- it is refused, and the caller's remedy names
    the cause.

    ``RADIENT_ORG_ALLOW_NONCANONICAL_BASE`` (``1``/``true``/``yes``/``on``) is
    the explicit opt-in for local runs and a deliberately hosted non-canonical
    hub: the QA rig points the CLI at 127.0.0.1, and setting the variable is the
    operator accepting that the account's bearer travels there. Unset means
    off, and nothing sets it implicitly.
    """
    if canonical_radient_destination(base_url):
        return True
    value = os.environ.get(ORG_ALLOW_NONCANONICAL_ENV)
    return bool(value) and value.strip().lower() in _TRUTHY_ENV_VALUES


def _radient_auth_store(config_dir: Path | None) -> AuthStore | MeshAwareAuthStore:
    """The store both resolvers read through: one spelling of the plain-vs-mesh predicate.

    THE MESH RUNG (Radient org projection, 2026-09-29): ``build_auth_store`` is the
    session path's own predicate for "plain, or mesh-aware when this device borrows",
    and routing both resolvers through it is what lets a PAIRED device without the
    signed-in account borrow the owner's bearer instead of returning ``None`` —
    ``lop teams pull`` on that device then runs as the account the owner lent.

    THE DB PATH IS SPELLED EXPLICITLY, preserving what both resolvers always passed:
    ``config_dir`` alone does not set ``AuthStore``'s DATABASE (it feeds the
    env-override tier only), so a caller whose ``config_dir`` is not the ambient root
    — tests, embedded roots — would silently read the AMBIENT ``auth.db`` after a
    naive swap. ``build_auth_store``'s ``db_path`` default keeps every other caller
    byte-identical.

    Side effect, accepted and bounded: an ATTEMPTED borrow that is refused parks this
    device's observation (``PlacementState`` — key name, refusal code, retry-after;
    no credential material, 0600, bounded TTLs) beside the placement document. A
    resolve that never attempts a borrow (no entry, or not a holder) writes nothing.

    The return type names the two implementations ``build_auth_store`` can return
    (the plain store, or the mesh-aware wrapper); ``build_auth_store`` itself is
    typed ``Any``, and naming them here keeps the CALLERS checked instead of blind.
    """
    from local_operator.network.credentials import build_auth_store

    return build_auth_store(
        config_dir, db_path=(config_dir / "auth.db") if config_dir is not None else None
    )


async def resolve_radient_credential(
    config_dir: Path | None,
    base_url: str,
    *,
    store: AuthStore | MeshAwareAuthStore | None = None,
) -> SecretStr:
    if not canonical_radient_destination(base_url):
        # An explicit legacy gateway must not receive a centrally signed-in
        # account's bearer. Preserve its previous dedicated key lookup instead —
        # store-first, so a RADIENT_API_KEY saved via the store is the only value
        # this route resolves.
        return _radient_api_key(config_dir)
    owns_store = store is None
    # THE MESH RUNG: a device without the login borrows the owner's bearer (see
    # ``_radient_auth_store``); local rows still win, and a borrow that is refused
    # or absent falls through to the extension seam exactly as before.
    store = store or _radient_auth_store(config_dir)
    try:
        # Read-only avoids moving inference account stickiness for catalogue,
        # upload and speech helpers; a required refresh still persists centrally.
        value = await store.get_api_key("radient", read_only=True)
        if value:
            return SecretStr(value)
        # Preserve the per-key extension seam of older/custom credential
        # managers after all canonical tiers, never ahead of a central login.
        return _radient_api_key(config_dir) or SecretStr("")
    finally:
        if owns_store:
            store.close()


async def has_persisted_radient_credential(
    config_dir: Path | None,
    base_url: str,
    *,
    store: AuthStore | MeshAwareAuthStore,
) -> bool:
    """Whether Radient is LOGGED IN, from persisted rows alone (never env).

    The availability twin of :func:`resolve_radient_credential`, for the speech
    cascades' "advertise this rung" question. It mirrors that resolver's two
    branches and drops only what is not a stored login:

    - CANONICAL destination: the store's persisted rows
      (:meth:`AuthStore.has_persisted_credential`). The call-time resolver's
      trailing ``_radient_api_key`` seam -- which falls through to the process
      environment -- is deliberately NOT consulted: an exported
      ``RADIENT_API_KEY`` must not make a phone's voice picker offer a rung the
      user never signed in to.
    - LEGACY gateway: the provider-class store row for ``RADIENT_API_KEY`` and
      nothing else. The call-time legacy branch reads that row and THEN the
      environment; this keeps the first half. It is "a STORE row", never env.

    Never raises: any failure reads as "not logged in".
    """
    try:
        if not canonical_radient_destination(base_url):
            from local_operator.providers.registry import provider_secret_value

            return bool(provider_secret_value("RADIENT_API_KEY", base=config_dir))
        return await store.has_persisted_credential("radient")
    except Exception:  # noqa: BLE001 - a probe must never take its caller down
        return False


def resolve_radient_credential_sync(config_dir: Path | None, base_url: str) -> SecretStr:
    """CLI-only bridge; async hosts must await the shared resolver directly."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(resolve_radient_credential(config_dir, base_url))
    raise RuntimeError("Await resolve_radient_credential inside an async host")


async def resolve_radient_oauth_access(
    config_dir: Path | None,
    base_url: str,
    *,
    store: AuthStore | MeshAwareAuthStore | None = None,
) -> OAuthAccess | None:
    """The signed-in Radient account behind an ORGANIZATION (person-scoped) call.

    Design §8.3: organization calls authenticate with the stored Radient OAuth
    access token -- never the tenant API key, because an API key proves an
    application tenant, not a person's membership (§2.2). This resolver therefore
    answers ONLY when the store's cascade picks an OAuth row: a pasted
    ``radient-key`` login (``kind == "api_key"``) or no login at all resolves to
    ``None``, and the caller answers with the "run ``lop login radient``" remedy
    rather than the public hub paths' "RADIENT_API_KEY is required" one.

    The destination guard applies here exactly as it does to
    :func:`resolve_radient_credential`: an org call carries a CENTRAL credential,
    so only a destination allowed to receive it
    (:func:`org_oauth_destination_allowed` -- canonical by default, or the
    explicit ``RADIENT_ORG_ALLOW_NONCANONICAL_BASE`` opt-in) resolves a token at
    all. A refused destination comes back as ``None`` exactly like a missing
    login, which is what keeps any credential off a request to a host that must
    not receive one; the caller tells the two causes apart through the predicate
    above and names the right remedy.

    ``read_only`` resolves without blocking a credential or moving session
    stickiness, matching the other hub helpers; a required refresh still
    persists centrally. A grant the IdP has declared dead comes back as
    ``None`` (the cascade rotates away from unusable rows), which is exactly the
    "expired login" case the caller renders as the re-login remedy.

    ONE RUNG BELOW THE LOCAL READ, the mesh may serve this device a bearer the
    operator lent from a paired device (``build_auth_store``): the borrow is
    attempted only when this device is a named holder, the destination guard
    above still runs FIRST (a refused destination never asks an owner for a
    bearer), and the kind rule below still decides — a borrowed API key refuses
    exactly as a local one would, so person-scope is preserved over the mesh.
    """
    if not org_oauth_destination_allowed(base_url):
        return None
    owns_store = store is None
    # THE MESH RUNG: with a borrowable share, the read below borrows the owner's
    # bearer instead of returning ``None`` (see ``_radient_auth_store``).
    store = store or _radient_auth_store(config_dir)
    try:
        access = await store.get_oauth_access("radient", read_only=True)
    finally:
        if owns_store:
            store.close()
    if access is None or access.kind != "oauth" or access.credential_invalid:
        return None
    if not access.access_token:
        return None
    return access


def resolve_radient_oauth_access_sync(config_dir: Path | None, base_url: str) -> OAuthAccess | None:
    """CLI-only bridge; async hosts must await the shared resolver directly."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(resolve_radient_oauth_access(config_dir, base_url))
    raise RuntimeError("Await resolve_radient_oauth_access inside an async host")


def radient_oauth_login_usable_since(config_dir: Path, base_url: str) -> int | None:
    """When the newest Radient login :func:`resolve_radient_oauth_access` could use
    was written (epoch ms, the row's ``updated_at``), or ``None`` for no such login.

    LOCAL AND NO-NETWORK BY CONTRACT, for a status read that must not refresh a
    token (the hub links/status projections ask "is a stored ``no-credential``
    stale because the operator has since signed in?"). It answers with the
    resolver's own LOCAL refusals, so the two cannot drift apart silently:

    * the destination guard (:func:`org_oauth_destination_allowed`) -- a
      refused destination resolves nothing, whatever the store holds;
    * the kind rule -- only an ``oauth`` row is a hub sign-in; a pasted
      ``radient-key`` API key resolves to ``None`` there, so it does not count;
    * the terminal verdicts ``AuthStore._ensure_oauth_fresh`` raises
      ``CredentialInvalidError`` for before any network: the IdP's dead-grant
      tombstone, the family-moved marker, and an ENDED delegation on a token
      that needs a refresh -- plus a LIVE delegation on such a token, which that
      method refuses with ``RefreshDelegatedError``. These rows stay listed
      (they are deliberately not ``disabled_cause``), which is why a bare
      "any enabled OAuth row" probe reported a dead login as signed in (PR #3536
      review M1 / QA Q1);
    * an account-wide credential block (``AuthStore.is_blocked``) still in
      force -- the resolver's selection skips that row (review r2 m1).

    What it CANNOT see is an outcome only the network decides (a refresh that
    fails transiently, a block a LATER failed call writes). The timestamp is
    the guard for those: a caller compares it with the time the refusal was
    recorded and treats the refusal as stale only when this login is NEWER -- a
    refusal recorded after the login is a current fact.

    Runtime/config API-key overrides (``set_runtime_api_key`` /
    ``set_config_api_key``) also make the resolver answer ``None``, but they are
    per-instance, in-memory state: the resolver reads a FRESH store
    (:func:`_radient_auth_store`) that carries none, so they cannot apply to it
    and are not mirrored here.

    Never raises: any doubt (unreadable store, malformed row) reads as no usable
    login, which only leaves a stored refusal reported as it was.
    """
    try:
        if not org_oauth_destination_allowed(base_url):
            return None
        store = AuthStore.open_read_only(Path(config_dir) / "auth.db")
        try:
            rows = store.list_credentials("radient")
            blocked = {row.id for row in rows if store.is_blocked(row.id, "radient")}
        finally:
            store.close()
        now_ms = AuthStore._now_ms()
        newest: int | None = None
        for row in rows:
            if row.credential_type != "oauth" or not isinstance(row.data, dict):
                continue
            creds = row.data
            # An account-wide backoff a failed call wrote: the resolver's
            # selection skips a blocked row, so it is no sign-in here either
            # (PR #3536 review r2 m1). It expires on its TTL, after which the
            # row counts again -- the same moment the resolver would use it.
            if row.id in blocked:
                continue
            if AuthStore._grant_dead_at(creds) is not None:
                continue
            if AuthStore._family_moved_at(creds) is not None:
                continue
            if AuthStore._needs_refresh(creds) and (
                AuthStore._ended_delegation(creds, now_ms=now_ms) is not None
                or AuthStore._delegated(creds, now_ms=now_ms) is not None
            ):
                continue
            newest = max(newest or 0, int(row.updated_at or 0))
        return newest
    except Exception:  # noqa: BLE001 - a status probe must never fail its read
        return None
