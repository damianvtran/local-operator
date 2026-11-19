"""AuthStore seal hook for Bedrock connections (the Muse precedent, design §5.2).

A Bedrock row's ``key`` is a NON-secret connection descriptor. A login that
collected secret material (a Bedrock API key, or an access-key pair) hands the
store a row carrying ``secret``; this hook seals that value in the encrypted
store under a fresh ``BEDROCK_SECRET_<uuid4 hex>`` ref, writes the ref into the
descriptor, and DROPS ``secret`` before anything is serialised to auth.db — so
auth.db never holds Bedrock secret plaintext. The ref is regex-pinned
(``connection.REF_PATTERN``): a descriptor can never point the resolver, or
this module's deletes, at an arbitrary operator secret.

Row shape after sealing::

    {"type": "api_key", "source": "login", "key": "bedrock://access-key?region=...&ref=...",
     "account_id": "access-key:AKIA…WXYZ@us-east-1",
     "bedrock": {"method": "access-key", "region": "us-east-1", "label": "..."}}

A raw API key row with no ``secret`` and no descriptor (the generic
``PUT /key`` path) is also sealed: its plaintext ``key`` is moved into the
store and replaced by an ``api-key`` descriptor, so even the fallback path
keeps the secret out of auth.db.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from local_operator.providers.bedrock import connection as bconn

SECRET_FIELD = "secret"


def _ref_of(credential: dict[str, Any]) -> str | None:
    key = credential.get("key")
    if not bconn.is_descriptor(key):
        return None
    try:
        conn = bconn.parse(key)
    except bconn.DescriptorError:
        return None
    return conn.ref


def _secret_text(secret: Any) -> str:
    if isinstance(secret, str):
        return secret.strip()
    if isinstance(secret, dict):
        # Access-key pair: only the three AWS fields, nothing else rides along.
        pair = {
            name: secret.get(name)
            for name in ("access_key_id", "secret_access_key", "session_token")
            if secret.get(name)
        }
        if not pair.get("access_key_id") or not pair.get("secret_access_key"):
            raise ValueError("an access-key connection needs an id and a secret")
        return json.dumps(pair, separators=(",", ":"))
    raise ValueError("unsupported Bedrock secret shape")


def needs_seal(credential: dict[str, Any]) -> bool:
    if SECRET_FIELD in credential:
        return True
    key = credential.get("key")
    return isinstance(key, str) and bool(key.strip()) and not bconn.is_descriptor(key)


def seal_credential(credential: dict[str, Any], base: Path | None) -> dict[str, Any]:
    """Return the row to persist; the secret (if any) is now in the store.

    Raises ``ValueError`` for a malformed row, and the store's own error if the
    write fails — a write the operator asked for must not look like success.
    """
    from local_operator.providers.registry import store_provider_key

    row = {name: value for name, value in credential.items() if name != SECRET_FIELD}
    if SECRET_FIELD in credential:
        secret = _secret_text(credential[SECRET_FIELD])
        key = credential.get("key")
        conn = bconn.parse(key) if bconn.is_descriptor(key) else None
        method = conn.method if conn is not None else "api-key"
        if method not in ("api-key", "access-key"):
            raise ValueError(f"a {method} Bedrock connection carries no secret")
        region = conn.region if conn is not None else None
    else:
        # Raw pasted Bedrock API key (generic PUT /key / paste login).
        secret = str(credential.get("key") or "").strip()
        method, region = "api-key", None
    if not secret:
        raise ValueError("empty Bedrock secret")
    ref = bconn.REF_PREFIX + uuid.uuid4().hex
    store_provider_key(ref, secret, description="Amazon Bedrock connection secret", base=base)
    row["key"] = bconn.format_descriptor(method, region=region, ref=ref)
    row.setdefault("type", "api_key")
    meta = dict(row.get("bedrock") or {})
    meta.setdefault("method", method)
    if region:
        meta.setdefault("region", region)
    row["bedrock"] = meta
    return row


def delete_key(credential: dict[str, Any], base: Path | None) -> None:
    """Remove the sealed secret a row references; missing is already removed."""
    from local_operator.providers.registry import remove_provider_key

    ref = _ref_of(credential)
    if ref and bconn.REF_PATTERN.fullmatch(ref):
        remove_provider_key(ref, base=base)


def identity_key(credential: dict[str, Any]) -> str | None:
    """``bedrock:<method>:<who>@<region>`` so re-saving a connection replaces it.

    ``who`` is the profile name, an access-key fingerprint carried in
    ``account_id``, or ``ambient``/``env-bearer``. A raw/sealed API key has no
    stable non-secret identity beyond its row, so it returns ``None`` (API keys
    never dedupe, matching every other provider).
    """
    key = credential.get("key")
    if not bconn.is_descriptor(key):
        return None
    try:
        conn = bconn.parse(key)
    except bconn.DescriptorError:
        return None
    region = conn.region or ""
    if conn.method == "profile":
        return f"bedrock:profile:{conn.profile}@{region}"
    if conn.method in ("ambient", "env-bearer"):
        return f"bedrock:{conn.method}@{region}"
    if conn.method == "access-key":
        account = credential.get("account_id")
        if isinstance(account, str) and account:
            return f"bedrock:access-key:{account}"
    return None
