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
"""

from __future__ import annotations

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

__all__ = [
    "BrokerUnavailable",
    "IncompatibleStore",
    "InsecurePermissions",
    "InvalidSecretName",
    "SecretCorrupt",
    "SecretExists",
    "SecretNotFound",
    "SecretStoreError",
]
