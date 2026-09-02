"""The credential-store-backed ``SecretResolver``.

This is the SECOND runner module allowed to reach into the application (the
first is ``provider_client.py``, for the same reason: a real episode needs a
real credential, and the store is where operators keep them). Everything
else under ``runner/`` must stay free of the store so an episode's evidence
cannot silently depend on the operator's own configuration --
``tests/unit/evaluation/runner/test_isolation.py`` asserts both the
exception and the rule.

The store is ``CredentialManager`` (``~/.local-operator/credentials.env`` or
the ``--config-dir`` an operator names), the same object the CLI builds for
``AuthStore``. It is taken as a constructor argument rather than built here so
the caller that opens the store for the model client hands the SAME instance
to the secret resolver: one place decides which config dir is authoritative.

``CredentialManager.get_credential`` returns an empty ``SecretStr`` for an
unknown key rather than raising, and it also falls back to ``os.environ`` for
a name it does not hold. Both are folded into the one contract every resolver
shares: an empty value is ``MissingSecret(name)``.
"""

from __future__ import annotations

from typing import Any, Sequence

from local_operator.evaluation.adapters.api import ResolvedSecret
from local_operator.evaluation.runner.secrets import MissingSecret


class CredentialStoreResolver:
    """Resolve ``SecretRef`` names from the harness credential store.

    ``credential_manager`` is any object with ``get_credential(name)``
    returning a ``SecretStr``-like value (``get_secret_value()``); it is typed
    ``Any`` so this module never imports ``local_operator.credentials`` at
    module scope. Even the lazy import is avoided: the caller already holds a
    live manager, and constructing a second one here would re-read the file
    and could disagree with the one the model client was built from.
    """

    def __init__(self, credential_manager: Any) -> None:
        self._manager = credential_manager

    def resolve(self, names: Sequence[str]) -> tuple[ResolvedSecret, ...]:
        resolved: list[ResolvedSecret] = []
        for name in names:
            try:
                value = self._manager.get_credential(name).get_secret_value()
            except Exception as error:
                # The store's own error could quote the key path or the file;
                # it never quotes a value, but the chained cause is still kept
                # off the message so the diagnostic stays "name only".
                raise MissingSecret(name) from error
            if not value:
                raise MissingSecret(name)
            resolved.append(ResolvedSecret(name=name, value=value))
        return tuple(resolved)
