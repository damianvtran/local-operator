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
a name it does not hold. The fallback is deliberately NOT used here: the
runner's contract is that the environment is reachable only through an
explicit ``EnvSecretResolver`` over names the caller listed, and a store
resolver that quietly served ambient variables would make "resolved from the
store" a false claim in the operator's own proof. This reads the store's
loaded mapping directly, so a name the file lacks is ``MissingSecret(name)``
even when the process environment happens to carry it.
"""

from __future__ import annotations

from typing import Any, Sequence

from local_operator.evaluation.adapters.api import ResolvedSecret
from local_operator.evaluation.runner.secrets import (
    MissingSecret,
    build_resolved_secret,
)


class CredentialStoreResolver:
    """Resolve ``SecretRef`` names from the harness credential store.

    ``credential_manager`` is any object with ``get_credentials()`` returning
    a ``{name: SecretStr-like}`` mapping (``get_secret_value()``); it is typed
    ``Any`` so this module never imports ``local_operator.credentials`` at
    module scope. Even the lazy import is avoided: the caller already holds a
    live manager, and constructing a second one here would re-read the file
    and could disagree with the one the model client was built from.
    ``get_credentials`` rather than ``get_credential`` because only the
    former is free of the environment fallback (see the module docstring).
    """

    def __init__(self, credential_manager: Any) -> None:
        self._manager = credential_manager

    def resolve(self, names: Sequence[str]) -> tuple[ResolvedSecret, ...]:
        try:
            stored = self._manager.get_credentials()
        except Exception as error:
            # The store's own error could quote the file path; it never quotes
            # a value, but the chained cause is still kept off the message so
            # the diagnostic stays "name only".
            raise MissingSecret(names[0] if names else "") from error
        resolved: list[ResolvedSecret] = []
        for name in names:
            secret = stored.get(name)
            value = secret.get_secret_value() if secret is not None else ""
            resolved.append(build_resolved_secret(name, value))
        return tuple(resolved)
