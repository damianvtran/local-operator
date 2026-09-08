"""Encrypted long-term secret store (``lop secret``).

Design: ``docs/design/secret-store.md``. This package ships PR 1 of that
design — the crypto, the SQLite schema and the CLI, reading the master key
file directly. The broker daemon that holds the key in RAM and authenticates
callers by process ancestry is PR 2; :mod:`local_operator.secrets.access` is
the seam it slides into.

**What this is, stated without overclaiming, because the design is explicit
that it must not be dressed up:** it makes secrets invisible to the "scan the
disk for credential files" malware a bad link actually drops, and it fails
closed against an attacker who edits the database. In the default ``keyfile``
mode it does NOT stop an attacker who reads the master key file next to the
store, and nothing confined to one macOS user account stops a process that
simply runs ``lop secret get`` itself. See §9 of the design for the full list.
It is not a vault and should not be described as one.

Nothing heavy is imported here: the CLI's argument registration must stay off
the crypto stack's import path (``tests/unit/test_import_graph.py``), so
callers import the submodule they need.

``secrets`` — the lazy mapping an agent uses from ``eval`` (design §5.3, and
``from local_operator.secrets import secrets`` is the documented spelling) — is
exported through ``__getattr__`` for exactly that reason. Naming it in the
import list above would put :mod:`local_operator.secrets.runtime` on the path of
every importer of this package; behind ``__getattr__`` it is loaded on first
attribute access and the errors stay stdlib-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    # Import-time cost is the whole reason for the ``__getattr__`` below, and a
    # TYPE_CHECKING import costs nothing at runtime while letting a type
    # checker (and an editor) resolve ``from local_operator.secrets import
    # secrets`` to the real object.
    from local_operator.secrets.runtime import secrets as secrets

from local_operator.secrets.errors import (
    BrokerUnavailable,
    IncompatibleStore,
    InsecurePermissions,
    InvalidSecretName,
    SecretCorrupt,
    SecretExists,
    SecretNotFound,
    SecretStoreError,
)


def __getattr__(name: str) -> object:
    """Resolve ``secrets`` on first access; see the module docstring."""
    if name == "secrets":
        from local_operator.secrets.runtime import secrets

        return secrets
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BrokerUnavailable",
    "IncompatibleStore",
    "InsecurePermissions",
    "InvalidSecretName",
    "SecretCorrupt",
    "SecretExists",
    "SecretNotFound",
    "SecretStoreError",
    "secrets",
]
