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

NON-SECRET BY CONSTRUCTION. The ref is a truncated SHA-256 over the provider id
and an identity, so it is stable for one account, opaque to a reader, and not
reversible to an email, an organization name or a raw IdP id. The file carries
the ref, the credential kind and a coarse plan slug, nothing else.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import re
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


def build_account_document(
    rows: list[Any], hosting: str | None, store_uid: str | None = None
) -> dict[str, Any]:
    """The ``account.json`` document for ``rows`` and the configured ``hosting``.

    PER-PROVIDER RULE. The row lop would reach first in its credential cascade:
    the lowest-id enabled OAuth row (the cascade walks OAuth before stored API
    keys), else the lowest-id enabled API-key row. Disabled rows never count. A
    provider with several accounts is reported as its first; the per-call stamp
    on ``Usage`` is the exact record of which one served.

    DEFAULT RULE. The provider of the configured default model, i.e.
    ``hosting`` in ``config.yml`` mapped through ``credential_provider_id``.
    ``null`` when no default is configured or that provider has no usable
    credential. Per-agent and per-request model overrides are not consulted.
    """
    from local_operator.providers.local import LOCAL_PROVIDER_IDS
    from local_operator.providers.registry import (
        credential_provider_id,
        get_provider_definition,
    )

    chosen: dict[str, Any] = {}
    for row in sorted(rows, key=lambda r: (r.credential_type != "oauth", r.id)):
        if row.disabled_cause is not None or row.credential_type not in ("oauth", "api_key"):
            continue
        # Only chat providers the registry owns: the same table also holds MCP
        # and console credentials, which are not an account lop spends.
        if get_provider_definition(row.provider) is None or row.provider in LOCAL_PROVIDER_IDS:
            continue
        if row.provider in chosen:
            continue
        entry = _entry_for(row, store_uid)
        if entry is not None:
            chosen[row.provider] = entry

    default: dict[str, Any] | None = None
    if hosting:
        provider = credential_provider_id(str(hosting))
        entry = chosen.get(provider)
        if entry is not None:
            default = {"provider": provider, **entry}
    return {
        "version": ACCOUNT_FILE_VERSION,
        "default": default,
        "providers": dict(sorted(chosen.items())),
    }


def _render(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def refresh_account_file(
    config_dir: Path,
    rows: list[Any] | None = None,
    *,
    only_if_changed: bool = True,
    store_uid: str | None = None,
) -> bool:
    """Rewrite ``<config_dir>/account.json`` from the credential store and config.

    Returns True when the file was written. Never raises: the file is a
    convenience for hosts, and no login, logout or config write may fail over it.

    ``rows`` are the store's rows when the caller already holds them (the store's
    own write hooks); otherwise the store is opened read-only, which creates
    nothing. When there is no ``auth.db`` and no earlier file there is nothing to
    say and nothing is created, so a bare config directory stays bare.
    """
    try:
        config_dir = Path(config_dir)
        target = config_dir / ACCOUNT_FILE_NAME
        if rows is None:
            db_path = config_dir / "auth.db"
            if not db_path.exists() and not target.exists():
                return False
            from local_operator.providers.auth_store import AuthStore

            store = AuthStore.open_read_only(db_path)
            try:
                rows = store.list_credentials(include_disabled=True)
                store_uid = store.store_uid
            finally:
                store.close()
        from local_operator.config import read_config_values

        values = read_config_values(config_dir)
        hosting = values.get("hosting") if isinstance(values, dict) else None
        payload = _render(
            build_account_document(rows, hosting if isinstance(hosting, str) else None, store_uid)
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
    except Exception:  # noqa: BLE001 — never fail a login/logout/config write over this
        # No exception text: a SQL or payload error could echo identity fields.
        logger.warning("could not refresh %s", ACCOUNT_FILE_NAME)
        return False
