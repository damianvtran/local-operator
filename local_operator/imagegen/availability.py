"""Sync availability probes and async key fetchers for the image rungs.

**D9 (frozen design):** the ``createIf`` gate answers "can this machine reach
ANY image provider", and every probe here is SYNC and LOCAL — sqlite rows, the
encrypted store, the process environment; NO sockets. A session-build path
must not open a connection (the browser gate's precedent), and a probe that
cannot answer reads as "not available" rather than raising — a broken probe
must never take a session's tool surface down with it.

**The one deliberate divergence between rungs:**

- **Radient** is advertised from PERSISTED LOGIN ROWS ONLY
  (``list_credentials`` non-empty; never ``RADIENT_API_KEY`` from the
  environment). This is the STT/mobile rule: a device must not offer a rung
  the user never signed in to.
- **FAL / OpenAI** accept a stored row **or an exported key** (``FAL_API_KEY``
  / ``OPENAI_API_KEY``). An exported key genuinely runs the call, and the
  strictness argument that governs a phone's voice picker has no local
  equivalent here.

The call-time resolvers (``resolve_radient_credential``, the async twin below)
deliberately keep their FULL precedence — availability decides whether the
tool EXISTS; the executor decides what a request authenticates with, and the
two are allowed to differ (an export lights a rung the gate would not... it
does here too for FAL/OpenAI by design; what must never differ is a rung
being advertised with one credential class and spending with a REFUSED one,
which is why OpenAI's probe and its async twin both read ``api_key`` rows
only — never a ChatGPT OAuth grant, which is not valid at ``/v1/images``).

**The subscription rung (media wave-2) is the first rung whose probe reads
the OAuth-GRANT class**: the grant that is invalid at ``/v1/images`` is
exactly what the Codex backend spends, so
:func:`openai_subscription_grant` reads ``oauth`` rows under ``openai`` — the
deliberate inverse of the ``openai-key`` rule, and the shape every breadth
rung follows: **probe the credential class the rung SPENDS with.**
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

from local_operator.providers.auth_store import AuthStore, OAuthAccess

logger = logging.getLogger(__name__)

#: The credential namespace the OpenAI image rung reads: the ``openai-key``
#: login's own, holding a platform API key. NOT ``openai`` — that provider's
#: only logins are ChatGPT OAuth grants, whose tokens are not valid at
#: ``/v1/images/generations``, so probing it would advertise a rung that can
#: only 401. Module-local on purpose (no ``stt`` import; the STT lane pinned
#: the same reading for ``/v1/audio``).
OPENAI_IMAGES_NAMESPACE = "openai-key"

#: Rung accepts API-key rows only, at probe AND call time. One constant so the
#: two cannot drift: an availability answer about one credential class and a
#: request sent with another is the defect this closes.
OPENAI_IMAGES_KINDS = frozenset({"api_key"})

FAL_ENV_KEY = "FAL_API_KEY"
OPENAI_ENV_KEY = "OPENAI_API_KEY"
GOOGLE_ENV_KEY = "GOOGLE_AI_STUDIO_API_KEY"
XAI_ENV_KEY = "XAI_API_KEY"


def _open_store(config_dir: Path | None) -> AuthStore:
    """A store rooted at ``config_dir`` (mirrors ``stt.cascade._ensure_store``).

    The db path is spelled explicitly: ``config_dir`` alone does not set
    ``AuthStore``'s DATABASE (it feeds the env-override tier only), so a
    caller whose ``config_dir`` is not the ambient root — tests, embedded
    roots — would silently read the AMBIENT ``auth.db`` after a naive swap.
    """
    db_path = (config_dir / "auth.db") if config_dir is not None else None
    return AuthStore(db_path, config_dir=config_dir)


def _api_key_from_rows(rows: Sequence[object]) -> str | None:
    """The first enabled ``api_key`` row's secret, or ``None``.

    The same field the AuthStore's own key tiers read (``data["key"]``), and
    only for rows the store already filtered as usable (``list_credentials``
    excludes disabled ones). Tolerant of a non-row seam: a probe must never
    raise.
    """
    for row in rows:
        credential_type = getattr(row, "credential_type", None)
        data = getattr(row, "data", None)
        if credential_type != "api_key" or not isinstance(data, dict):
            continue
        key = data.get("key")
        if isinstance(key, str) and key:
            return key
    return None


def radient_available(config_dir: Path | None = None) -> bool:
    """Whether Radient is LOGGED IN, from persisted rows alone (never env).

    The sync sibling of ``has_persisted_radient_credential``: ``list_credentials``
    returns the stored rows and does not touch the network. Any failure reads
    as "not available".
    """
    try:
        store = _open_store(config_dir)
        try:
            return bool(store.list_credentials("radient"))
        finally:
            store.close()
    except Exception:  # noqa: BLE001 - a probe must never take its caller down
        logger.debug("radient availability probe failed; reporting unavailable", exc_info=True)
        return False


def fal_key(config_dir: Path | None = None) -> str | None:
    """The FAL key the FAL rung would use: login row → store row → env.

    Store-first, matching every other provider-key reader in the tree. The
    login row wins because it is the credential the user actively signed in
    with; the store row (``lop credential update FAL_API_KEY``) and an
    exported environment value are kept so an operator who set the rung up
    that way still runs. Never raises.
    """
    try:
        store = _open_store(config_dir)
        try:
            key = _api_key_from_rows(store.list_credentials("fal"))
        finally:
            store.close()
        if key:
            return key
    except Exception:  # noqa: BLE001 - a probe must never take its caller down
        logger.debug("fal login-row probe failed; falling back to store/env", exc_info=True)

    from local_operator.providers.registry import provider_secret_value

    stored = provider_secret_value(FAL_ENV_KEY, base=config_dir)
    if stored:
        return stored
    exported = os.environ.get(FAL_ENV_KEY)
    return exported or None


def openai_images_key(config_dir: Path | None = None) -> str | None:
    """The OpenAI image key (sync form): ``api_key`` rows → store row → env.

    Same precedence as :func:`fal_key`; the row filter is the point (a ChatGPT
    OAuth row never answers — see the module docstring). Never raises.
    """
    try:
        store = _open_store(config_dir)
        try:
            key = _api_key_from_rows(store.list_credentials(OPENAI_IMAGES_NAMESPACE))
        finally:
            store.close()
        if key:
            return key
    except Exception:  # noqa: BLE001 - a probe must never take its caller down
        logger.debug("openai-key login-row probe failed; falling back to store/env", exc_info=True)

    from local_operator.providers.registry import provider_secret_value

    stored = provider_secret_value(OPENAI_ENV_KEY, base=config_dir)
    if stored:
        return stored
    exported = os.environ.get(OPENAI_ENV_KEY)
    return exported or None


def image_provider_reachable(config_dir: Path | None = None) -> bool:
    """The ``generate_image`` createIf gate: any rung's credential exists.

    All probes are sync and socket-free (see the module docstring), so this
    is safe on every session-build path. Sessions built before a login land
    the tool at the NEXT session — the commit-time gate is a snapshot by
    design, not a per-turn scan.
    """
    return bool(
        radient_available(config_dir)
        or fal_key(config_dir)
        or openai_images_key(config_dir)
        or openai_subscription_grant(config_dir)
        or google_key(config_dir)
        or xai_available(config_dir)
    )


def openai_subscription_grant(config_dir: Path | None = None) -> bool:
    """Whether a ChatGPT subscription GRANT is stored (the subscription rung).

    The deliberate INVERSE of :func:`openai_images_key`: the images API wants
    an ``api_key`` row and rejects an OAuth grant, while the Codex-backend
    rung spends exactly the OAuth grant — so this probe reads ``oauth`` rows
    under the ``openai`` namespace only, and the two rules must never be
    blurred into one. Never raises; deliberately NO env leg (a grant is
    stored by a sign-in, never exported).
    """
    try:
        store = _open_store(config_dir)
        try:
            rows = store.list_credentials("openai")
            return any(getattr(row, "credential_type", None) == "oauth" for row in rows)
        finally:
            store.close()
    except Exception:  # noqa: BLE001 - a probe must never take its caller down
        logger.debug("openai subscription probe failed; reporting unavailable", exc_info=True)
        return False


async def openai_call_key(store: AuthStore, session_id: str | None = None) -> str | None:
    """The bearer the OpenAI rung would send: persisted api_key rows, then env.

    The ASYNC twin of :func:`openai_images_key` — same credential class, same
    failure contract (never raises, ``None`` means "no key"). The persisted
    read goes through ``get_persisted_api_key``, which itself covers the
    provider-class store row (``legacy_store_keys``), and the environment is
    appended because an exported key genuinely runs the call.
    """
    try:
        key = await store.get_persisted_api_key(
            OPENAI_IMAGES_NAMESPACE, session_id, kinds=OPENAI_IMAGES_KINDS
        )
    except Exception:  # noqa: BLE001 - a probe must never take its caller down
        logger.warning("openai persisted-key read failed; reporting none")
        key = None
    if key:
        return key
    exported = os.environ.get(OPENAI_ENV_KEY)
    return exported or None


def google_key(config_dir: Path | None = None) -> str | None:
    """The Google AI Studio key (sync form): ``api_key`` rows → store → env.

    Same precedence as :func:`fal_key`. The namespace is the registry row's
    ``google`` (``lop login google`` stores there); the env leg is the
    ``GOOGLE_AI_STUDIO_API_KEY`` name the row declares, and an exported value
    genuinely runs the call. Never raises.
    """
    try:
        store = _open_store(config_dir)
        try:
            key = _api_key_from_rows(store.list_credentials("google"))
        finally:
            store.close()
        if key:
            return key
    except Exception:  # noqa: BLE001 - a probe must never take its caller down
        logger.debug("google login-row probe failed; falling back to store/env", exc_info=True)

    from local_operator.providers.registry import provider_secret_value

    stored = provider_secret_value(GOOGLE_ENV_KEY, base=config_dir)
    if stored:
        return stored
    exported = os.environ.get(GOOGLE_ENV_KEY)
    return exported or None


def xai_available(config_dir: Path | None = None) -> bool:
    """Whether an xAI credential exists: a stored row of EITHER class, or env.

    xAI serves both an API key and the Grok OAuth token at the same images
    route (the ``xai-oauth`` login stores under the ``xai`` namespace), so
    this probe accepts both row classes — the one-class rule holds because
    BOTH genuinely run the call, and there is no second credential class this
    rung could silently spend instead. The registry's store row and an
    exported ``XAI_API_KEY`` are honoured like FAL's. Never raises.
    """
    try:
        store = _open_store(config_dir)
        try:
            rows = store.list_credentials("xai")
            if any(getattr(row, "credential_type", None) in ("api_key", "oauth") for row in rows):
                return True
        finally:
            store.close()
    except Exception:  # noqa: BLE001 - a probe must never take its caller down
        logger.debug("xai row probe failed; falling back to store/env", exc_info=True)

    from local_operator.providers.registry import provider_secret_value

    if provider_secret_value(XAI_ENV_KEY, base=config_dir):
        return True
    return bool(os.environ.get(XAI_ENV_KEY))


async def openai_sub_access(store: AuthStore, session_id: str | None = None) -> OAuthAccess | None:
    """The identity-carrying grant the subscription rung would send.

    The ASYNC twin of :func:`openai_subscription_grant`, and deliberately the
    chat path's own resolver (``get_oauth_access``) so refresh, rotation and
    backoff behave exactly as a chat turn's credential would — one place for
    those rules. ``None`` means "no grant"; a stored grant that cannot mint a
    bearer surfaces as a rung failure and fails forward. Never raises.
    """
    try:
        return await store.get_oauth_access("openai", session_id)
    except Exception:  # noqa: BLE001 - a probe must never take its caller down
        logger.warning("openai subscription access read failed; reporting none")
        return None


async def google_call_key(store: AuthStore, session_id: str | None = None) -> str | None:
    """The bearer the Google rung would send: persisted rows, then env.

    The ASYNC twin of :func:`google_key` — same credential class, same
    failure contract (never raises, ``None`` means "no key").
    """
    try:
        key = await store.get_persisted_api_key("google", session_id, kinds={"api_key"})
    except Exception:  # noqa: BLE001 - a probe must never take its caller down
        logger.warning("google persisted-key read failed; reporting none")
        key = None
    if key:
        return key
    exported = os.environ.get(GOOGLE_ENV_KEY)
    return exported or None


async def xai_call_bearer(store: AuthStore, session_id: str | None = None) -> str | None:
    """The bearer the xAI rung would send: the store's own pick, then env.

    Deliberately ``get_oauth_access``: it resolves EITHER row class (the
    ``xai`` and ``xai-oauth`` logins share one namespace) with refresh,
    rotation and backoff exactly as a chat turn's credential would — one
    place for those rules. ``None`` means "no credential"; a stored row that
    cannot mint a bearer surfaces as a rung failure and fails forward. Never
    raises.
    """
    try:
        access = await store.get_oauth_access("xai", session_id)
    except Exception:  # noqa: BLE001 - a probe must never take its caller down
        logger.warning("xai credential read failed; reporting none")
        access = None
    if access is not None and access.access_token:
        return access.access_token
    exported = os.environ.get(XAI_ENV_KEY)
    return exported or None
