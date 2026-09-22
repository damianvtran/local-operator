"""The credential-store-backed ``SecretResolver``.

This is the SECOND runner module allowed to reach into the application (the
first is ``provider_client.py``, for the same reason: a real episode needs a
real credential, and the store is where operators keep them). Everything
else under ``runner/`` must stay free of the store so an episode's evidence
cannot silently depend on the operator's own configuration --
``tests/unit/evaluation/runner/test_isolation.py`` asserts both the
exception and the rule.

The store is the ENCRYPTED secret store (``local_operator.secrets``), whose
agent-class rows hold the names this resolver serves — an AWS key, an eval
harness token, anything an operator stored with ``lop secret set`` or moved out
of the plaintext file with ``lop secret migrate-env``. Reads go through the
store's agent namespace, so a ``LOP_PROVIDER_*`` row is deliberately NOT visible
here: the evaluation runner resolves the names its spec lists, and a provider
key is not one of them.

The legacy ``CredentialManager`` mapping (``~/.local-operator/credentials.env``)
is consulted as a TRANSITION fallback for a name the store does not hold, so an
install that has not migrated yet keeps resolving. That fallback disappears with
the rest of the plaintext store.

``CredentialManager.get_credential`` returns an empty ``SecretStr`` for an
unknown key rather than raising, and it also falls back to ``os.environ`` for
a name it does not hold. The fallback is deliberately NOT used here: the
runner's contract is that the environment is reachable only through an
explicit ``EnvSecretResolver`` over names the caller listed, and a store
resolver that quietly served ambient variables would make "resolved from the
store" a false claim in the operator's own proof. The legacy leg therefore
reads the manager's loaded mapping directly (``get_credentials``), never the
env-falling-back getter.
"""

from __future__ import annotations

from typing import Any, Sequence

from local_operator.evaluation.adapters.api import ResolvedSecret
from local_operator.evaluation.runner.secrets import (
    MissingSecret,
    build_resolved_secret,
)


class CredentialStoreResolver:
    """Resolve ``SecretRef`` names from the harness secret store.

    ``credential_manager`` is the object whose ``config_dir`` locates the
    encrypted store and whose ``get_credentials()`` provides the legacy
    transition fallback. It is typed ``Any`` so this module never imports
    ``local_operator.credentials`` at module scope; the store is imported
    lazily in :meth:`resolve` for the same reason.

    The encrypted store is read first, in the AGENT namespace, so a
    ``LOP_PROVIDER_*`` row can never satisfy an evaluation ref. Only a name the
    store does not hold falls through to the legacy mapping — which is why
    ``get_credentials`` and not the env-falling-back getter is used (see the
    module docstring).
    """

    def __init__(self, credential_manager: Any) -> None:
        self._manager = credential_manager

    def _stored_secrets(self) -> dict[str, str]:
        """Agent-class store values by name, or ``{}`` when no store is readable.

        Best-effort, matching the readers elsewhere: an absent, locked or
        damaged store is "this leg has nothing" rather than a failure, so the
        legacy fallback (or ``MissingSecret``) still decides the outcome. A
        single unreadable row is skipped, not fatal — one damaged record must
        not hide every other credential, the store's own ``list`` rule.
        """
        from local_operator.secrets.access import open_store, retrieve_secret
        from local_operator.secrets.errors import SecretStoreError
        from local_operator.secrets.keys import store_path
        from local_operator.secrets.store import is_provider_secret_name

        base = getattr(self._manager, "config_dir", None)
        values: dict[str, str] = {}
        try:
            if not store_path(base).exists():
                return {}
            names = [
                record.name
                for record in open_store(base).list()
                if not is_provider_secret_name(record.name)
            ]
        except (SecretStoreError, OSError, ValueError):
            return {}
        for name in names:
            try:
                values[name] = retrieve_secret(name, base, role="agent").decode(
                    "utf-8", "replace"
                )
            except (SecretStoreError, OSError, ValueError):
                continue
        return values

    def resolve(self, names: Sequence[str]) -> tuple[ResolvedSecret, ...]:
        try:
            legacy = self._manager.get_credentials()
        except Exception as error:
            # The store's own error could quote the file path; it never quotes
            # a value, but the chained cause is still kept off the message so
            # the diagnostic stays "name only".
            raise MissingSecret(names[0] if names else "") from error
        stored = self._stored_secrets()
        resolved: list[ResolvedSecret] = []
        for name in names:
            # Encrypted store first (agent namespace); the legacy mapping is the
            # transition fallback for a name it does not hold.
            if name in stored:
                value = stored[name]
            else:
                secret = legacy.get(name)
                value = secret.get_secret_value() if secret is not None else ""
            resolved.append(build_resolved_secret(name, value))
        return tuple(resolved)

