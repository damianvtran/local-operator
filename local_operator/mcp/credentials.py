"""MCP-only encrypted credentials: declared references, never provider settings.

The owner executes this off-record. Values never enter slash arguments, receipts,
or the session environment; the only durable write is the existing encrypted store.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from local_operator.mcp.config import load_all_mcp_configs
from local_operator.paths import config_dir
from local_operator.secrets import access
from local_operator.secrets.errors import SecretNotFound
from local_operator.secrets.keys import store_path


class MCPCredentials(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1, max_length=256)
    values: dict[str, SecretStr] = Field(min_length=1, max_length=32)
    confirmed_replace: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def bounded(self):
        if (
            any(
                not key or len(key) > 128 or not 0 < len(value.get_secret_value()) <= 32768
                for key, value in self.values.items()
            )
            or sum(len(value.get_secret_value()) for value in self.values.values()) > 65536
        ):
            raise ValueError("Invalid MCP credential fields")
        if not set(self.confirmed_replace) <= self.values.keys():
            raise ValueError("Invalid replacement confirmation")
        return self


def credential_source(key: str, base: Path) -> str:
    """Which store holds ``key``: ``encrypted``, ``missing``, or ``unavailable``.

    **Nothing here creates a credential-shaped file.** ``CredentialManager``'s
    plain constructor calls ``_ensure_config_exists``, which CREATES the
    plaintext ``credentials.env`` — and a metadata probe that creates the store
    it is describing is a side effect the caller never asked for, on a file the
    consolidation retires.

    The plaintext fallback was REMOVED here: the legacy file has no writers and
    no readers left, so a ``legacy`` answer would describe a store nothing
    maintains. A key present only in an old ``credentials.env`` simply reads as
    the store's own ``missing`` until ``lop secret migrate-env`` moves it.
    """
    try:
        if store_path(base).exists():
            try:
                access.open_store(base).describe(key)
                return "encrypted"
            except SecretNotFound:
                pass
        return "missing"
    except Exception:
        # Store diagnostics may quote its input. Public state is intentionally
        # bounded and must not turn refusal/corruption into permission to fall back.
        return "unavailable"


async def store_credentials(session: Any, body: MCPCredentials) -> dict[str, Any]:
    from local_operator.mcp.secret_refs import public_secret_refs

    manager = getattr(session, "mcp_manager", None)
    base = Path(getattr(session, "config_dir", None) or config_dir())
    cwd = getattr(manager, "cwd", None)
    result: dict[str, Any] = {
        "name": body.name,
        "saved_ids": [],
        "failed_ids": [],
        "code": "invalid_target",
    }
    if manager is None or cwd is None:
        return result
    configs, _ = await asyncio.to_thread(load_all_mcp_configs, cwd)
    cfg = configs.get(body.name)
    if cfg is None:
        return result
    declared = {ref["id"] for ref in public_secret_refs(cfg)}
    if not body.values.keys() <= declared:
        return result
    # Registration precedes even a refused write. Do not use store_credential:
    # it also injects the value into every unrelated bash child environment.
    for value in body.values.values():
        secret = value.get_secret_value()
        variables = getattr(session, "variables", None)
        if variables is not None:
            variables.register_redaction(secret)
        # The manager's own sink too: a manager driven directly (no session
        # alongside it, as the review's canary probe does) still has to scrub
        # what it resolved. Both are scrubbing-only registrations — never an
        # environment injection, so the value stays unreadable to every child.
        sink = getattr(manager, "register_secret_redaction", None)
        if callable(sink):
            sink(secret)

    def persist() -> dict[str, Any]:
        saved: list[str] = []
        failed = list(body.values)
        try:
            store = access.open_store(base, create=True)
            store.initialize()
            existing: set[str] = set()
            for key in body.values:
                try:
                    store.describe(key)
                    existing.add(key)
                except SecretNotFound:
                    pass
            if not existing <= set(body.confirmed_replace):
                return {**result, "failed_ids": failed, "code": "replace_confirmation_required"}
            for key, value in body.values.items():
                write = store.update if key in existing else store.set
                write(key, value.get_secret_value().encode(), session_id=str(session.session_id))
                saved.append(key)
                failed.remove(key)
        except Exception:
            return {**result, "saved_ids": saved, "failed_ids": failed, "code": "store_unavailable"}
        return {**result, "saved_ids": saved, "failed_ids": [], "code": "saved"}

    return await asyncio.to_thread(persist)
