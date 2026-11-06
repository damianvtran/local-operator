"""Which ACCOUNT served a call, and the file that tells a host which one is current.

Two consumers, one derivation:

* the failover iterator stamps :func:`account_ref` onto every ``Usage`` so a
  ledger can attribute a call to an account (``Usage.account_ref``);
* :func:`refresh_account_file` publishes ``<config dir>/account.json`` so a
  terminal host (Tower) can attribute a whole terminal's usage to the login in
  force at launch, the way it reads ``~/.claude.json`` for Claude Code.

Both go through :func:`account_ref` with the same inputs read from the same
stored row (and the same store's ``store_uid`` namespace), so the per-call stamp
and the file cannot disagree about who the account is. A mesh borrower does not
derive a ref at all: it carries the owner's (``OAuthAccess.account_ref``).
The contract is documented in ``docs/account-file.md``.

The file names the serving account only where the credential cascade makes it
certain (see :func:`build_account_document`) and publishes ``null`` otherwise;
it never guesses from row order.

NON-SECRET BY CONSTRUCTION. The ref is a truncated SHA-256 over the provider id
and an identity, so it is stable for one account and opaque to a reader. It is
not a stored identifier in cleartext and cannot be enumerated without already
knowing the candidate identity; it is not a secret against a guess, though: a
64-bit digest of a low-entropy id (an email-shaped ``sub``) can be confirmed by
someone who guesses that id. The file carries the ref, the credential kind and
a coarse plan slug per account, nothing else.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

#: File name inside the config directory. Part of the host contract.
ACCOUNT_FILE_NAME = "account.json"

#: Schema version of the document. Bump only for a breaking shape change.
ACCOUNT_FILE_VERSION = 1

#: Hex characters of the digest kept. 64 bits: far beyond the number of accounts
#: one machine holds, short enough to read in a log line.
_REF_HEX_CHARS = 16

#: A plan/org type is published only as a short lowercase slug. Anything else
#: (a stored organization NAME in a field an older login wrote loosely) is
#: dropped rather than published.
_ORG_TYPE_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


@functools.lru_cache(maxsize=512)
def _digest(provider: str, tail: str) -> str:
    return hashlib.sha256(f"{provider}\0{tail}".encode()).hexdigest()[:_REF_HEX_CHARS]


def account_ref(
    provider: str,
    *,
    kind: str | None,
    credential_id: int = 0,
    org_id: str | None = None,
    account_id: str | None = None,
    store_uid: str | None = None,
    created_at: int = 0,
) -> str | None:
    """The opaque id of the account behind one credential, or ``None``.

    * OAuth row with an IdP identity: ``sha256("<provider>\\0<org_id or account_id>")``.
      The identity, not the row id or the store, so a re-login (new row, same
      account) keeps the ref and every host that knows the account agrees on it.
    * Row-derived (a stored API key, or an OAuth row whose IdP returned no
      identity: Kimi, xAI/Anthropic without an id_token):
      ``sha256("<kind>\\0<store_uid>\\0<credential_id>\\0<created_at>")`` with
      ``kind`` ``key`` or ``oauth``. A SQLite row id means nothing outside its
      ``auth.db``, so the store's random ``store_uid`` namespaces it (two
      machines' row 1 differ) and ``created_at`` keeps a reused id after a
      delete from aliasing the old account. It needs a positive ``credential_id``
      and a ``store_uid``; without both there is no durable identity and the
      answer is ``None``.
    * No stored row (``credential_id`` 0: environment variable, CLI/config
      override, fallback resolver): ``None``. A synthetic broker id (negative, a
      mesh borrower's stand-in for an owner row) never yields a row-derived ref;
      a borrower carries the OWNER's ref instead (``OAuthAccess.account_ref``).

    ``provider`` is canonicalised to its storage id so a login flavour
    (``openai-device``) and its provider (``openai``) are one account.
    """
    from local_operator.providers.registry import credential_provider_id

    if not credential_id:
        return None
    provider = credential_provider_id(provider)
    if kind == "oauth":
        identity = org_id or account_id
        if identity:
            return _digest(provider, str(identity))
    if credential_id <= 0 or not store_uid:
        return None
    kind_tag = "oauth" if kind == "oauth" else "key"
    return _digest(provider, f"{kind_tag}\0{store_uid}\0{credential_id}\0{created_at}")


def org_type_for(provider: str, kind: str, data: Mapping[str, Any]) -> str | None:
    """A plan/organization TYPE this machine already holds for the row, or ``None``.

    No profile or network call exists for this; it is read from the stored OAuth
    payload only. Today the single source is OpenAI's login, which stores the
    ``chatgpt_plan_type`` claim (``plus``, ``pro``, ...) in ``org_name``
    (``providers/oauth/openai.py``). Anthropic's login stores the organization's
    NAME, not its type, so Anthropic reports ``None``.
    """
    from local_operator.providers.registry import credential_provider_id

    if kind != "oauth" or credential_provider_id(provider) != "openai":
        return None
    candidate = str(data.get("org_name") or "").strip().lower()
    return candidate if _ORG_TYPE_SLUG.match(candidate) else None


def _entry_for(row: Any, store_uid: str | None) -> dict[str, Any] | None:
    """One provider's ``{account_ref, kind, org_type}`` entry from a stored row."""
    kind = "oauth" if row.credential_type == "oauth" else "api_key"
    data = row.data if isinstance(row.data, Mapping) else {}
    ref = account_ref(
        row.provider,
        kind=kind,
        credential_id=row.id,
        org_id=data.get("org_id"),
        account_id=data.get("account_id"),
        store_uid=store_uid,
        created_at=int(getattr(row, "created_at", 0) or 0),
    )
    if ref is None:
        return None
    return {"account_ref": ref, "kind": kind, "org_type": org_type_for(row.provider, kind, data)}


def _tier(row: Any) -> int:
    """The cascade tier of a stored row: 3 OAuth, 4 login-stored key, 6 other key.

    Mirrors ``AuthStore._resolve``: OAuth rows first, then API keys persisted by
    ``login`` (``data.source == "login"``), then (after the environment variable,
    tier 5, which no stored row can describe) any other stored key.
    """
    if row.credential_type == "oauth":
        return 3
    data = row.data if isinstance(row.data, Mapping) else {}
    return 4 if data.get("source") == "login" else 6


def build_account_document(
    rows: list[Any],
    hosting: str | None,
    store_uid: str | None = None,
    blocked_ids: Collection[int] = (),
) -> dict[str, Any]:
    """The ``account.json`` document for ``rows`` and the configured ``hosting``.

    PER-PROVIDER ENTRY. ``accounts`` lists every enabled row's
    ``{account_ref, kind, org_type}`` in cascade order (tier, then row id): an
    inventory, not a claim about which one serves. The entry's own ``account_ref``,
    ``kind`` and ``org_type`` name the serving account only when lop's credential
    cascade (``AuthStore._resolve``) is certain of it: the winning tier is the
    first non-empty of OAuth rows (3), login-stored keys (4), other stored keys
    (6), and it must hold exactly ONE row. With several rows the pick is made per
    session (sticky, usage-aware, round-robin), so the file cannot say; a tier-6
    key is outranked by an environment variable, which no file here can see. In
    both cases the three fields are ``null``. Disabled rows never count.

    BLOCKS. The sole winner's id must also not be in ``blocked_ids`` (credentials
    with an unexpired ``auth_credential_blocks`` row, any scope). A blocked winner
    means the cascade is serving a lower tier only until the block expires, and
    nothing is written when it does, so the serving account is time-dependent:
    the fields are ``null``, never a ref that turns wrong on expiry. Demotions
    (``AuthStore._deprioritized``, a per-process in-memory reordering of at most
    120 s) are not representable in a cross-process file and are not consulted. A
    credential from an environment variable, a config override or the runtime is
    not represented at all.

    DEFAULT RULE. ``default`` is ``{"provider", "account_ref", "kind", "org_type"}``
    (the entry without the ``accounts`` inventory) for the provider of
    the configured default model (``hosting`` in ``config.yml`` mapped through
    ``credential_provider_id``) when that provider has an entry, else ``null``;
    its ``account_ref`` is ``null`` when the serving account is not determinable.
    Per-agent and per-request model overrides are not consulted.
    """
    from local_operator.providers.local import LOCAL_PROVIDER_IDS
    from local_operator.providers.registry import (
        credential_provider_id,
        get_provider_definition,
    )

    by_provider: dict[str, list[Any]] = {}
    for row in rows:
        if row.disabled_cause is not None or row.credential_type not in ("oauth", "api_key"):
            continue
        # Only chat providers the registry owns: the same table also holds MCP
        # and console credentials, which are not an account lop spends.
        if get_provider_definition(row.provider) is None or row.provider in LOCAL_PROVIDER_IDS:
            continue
        by_provider.setdefault(row.provider, []).append(row)

    blocked = set(blocked_ids)
    providers: dict[str, Any] = {}
    for provider, group in by_provider.items():
        ordered = sorted(group, key=lambda r: (_tier(r), r.id))
        winning = [r for r in ordered if _tier(r) == _tier(ordered[0])]
        entry: dict[str, Any] = {"account_ref": None, "kind": None, "org_type": None}
        if _tier(ordered[0]) in (3, 4) and len(winning) == 1 and winning[0].id not in blocked:
            sole = _entry_for(winning[0], store_uid)
            if sole is not None:
                entry = sole
        entry["accounts"] = [e for e in (_entry_for(r, store_uid) for r in ordered) if e]
        providers[provider] = entry

    default: dict[str, Any] | None = None
    if hosting:
        provider = credential_provider_id(str(hosting))
        chosen = providers.get(provider)
        if chosen is not None:
            default = {
                "provider": provider,
                **{k: chosen[k] for k in ("account_ref", "kind", "org_type")},
            }
    return {
        "version": ACCOUNT_FILE_VERSION,
        "default": default,
        "providers": dict(sorted(providers.items())),
    }


def _render(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


#: Lock file beside ``account.json``. Not part of the host contract.
_LOCK_FILE_NAME = ".account-file.lock"

#: How long a publisher waits for another to finish. A publish is a SQLite read,
#: a config read and one small write, so a holder is normally done in
#: milliseconds; the bound only keeps a wedged holder from wedging a login.
_LOCK_WAIT_S = 5.0
_LOCK_POLL_S = 0.01


def refresh_account_file(
    config_dir: Path,
    *,
    only_if_changed: bool = True,
    read_state: Callable[[], tuple[list[Any], Collection[int]]] | None = None,
    store_uid: str | None = None,
) -> bool:
    """Rewrite ``<config_dir>/account.json`` from the credential store and config.

    Returns True when the file was written. Never raises: the file is a
    convenience for hosts, and no login, logout or config write may fail over it.

    SERIALIZED, REBUILT FROM CURRENT STATE. Reading the rows, reading the config,
    building, comparing and replacing all happen under one exclusive lock
    (``<config_dir>/.account-file.lock``, across processes and threads). Without
    it two publishers were unordered: one that read the rows before a logout could
    replace the file after the logout's own publish and bring the signed-out
    account back. Whoever holds the lock reads the store as it is NOW, so the last
    publisher to run always reflects the latest committed rows and config. A
    caller therefore cannot hand in rows; ``read_state`` is a callback invoked
    inside the lock returning ``(rows, blocked_ids)`` (the store's write hooks
    pass their own handle's reads plus its ``store_uid``). By default the store is
    opened read-only inside the lock,
    which creates nothing. When there is no ``auth.db`` and no earlier file there
    is nothing to say and nothing is created, not even the lock file, so a bare
    config directory stays bare.
    """
    try:
        config_dir = Path(config_dir)
        if read_state is None:
            if (
                not (config_dir / "auth.db").exists()
                and not (config_dir / ACCOUNT_FILE_NAME).exists()
            ):
                return False
        from local_operator.config_writer import nonblocking_file_claim

        # Resolved only for the lock path: the claim refuses a symlinked parent,
        # and a config directory reached through a link is a legitimate layout.
        lock_path = config_dir.resolve() / _LOCK_FILE_NAME
        deadline = time.monotonic() + _LOCK_WAIT_S
        while True:
            # A fresh descriptor per attempt, so threads of this process exclude
            # one another as well as other processes.
            with nonblocking_file_claim(lock_path) as acquired:
                if acquired:
                    return _publish_locked(config_dir, only_if_changed, read_state, store_uid)
            if time.monotonic() >= deadline:
                logger.warning("timed out waiting to refresh %s", ACCOUNT_FILE_NAME)
                return False
            time.sleep(_LOCK_POLL_S)
    except Exception:  # noqa: BLE001 — never fail a login/logout/config write over this
        # No exception text: a SQL or payload error could echo identity fields.
        logger.warning("could not refresh %s", ACCOUNT_FILE_NAME)
        return False


def _publish_locked(
    config_dir: Path,
    only_if_changed: bool,
    read_state: Callable[[], tuple[list[Any], Collection[int]]] | None,
    store_uid: str | None,
) -> bool:
    """The read-build-write body of :func:`refresh_account_file`; lock held."""
    target = config_dir / ACCOUNT_FILE_NAME
    if read_state is not None:
        rows, blocked_ids = read_state()
    else:
        from local_operator.providers.auth_store import AuthStore

        store = AuthStore.open_read_only(config_dir / "auth.db")
        try:
            rows = store.list_credentials(include_disabled=True)
            blocked_ids = store.active_block_ids()
            store_uid = store.store_uid
        finally:
            store.close()
    from local_operator.config import read_config_values

    values = read_config_values(config_dir)
    hosting = values.get("hosting") if isinstance(values, dict) else None
    payload = _render(
        build_account_document(
            rows, hosting if isinstance(hosting, str) else None, store_uid, blocked_ids
        )
    )
    if only_if_changed:
        try:
            if target.read_bytes() == payload and (target.stat().st_mode & 0o777) == 0o600:
                return False
        except OSError:
            pass
    # Same-directory temp + ``os.replace`` at 0600: a host polling the file
    # must never read a torn document.
    from local_operator.owned_process import _atomic_write_bytes

    _atomic_write_bytes(target, payload, mode=0o600)
    return True
