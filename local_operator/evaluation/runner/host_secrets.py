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

The legacy plaintext mapping (``~/.local-operator/credentials.env``) is NO LONGER
consulted (PR2a). It used to be a TRANSITION fallback for a name the store did
not hold; the plaintext file is no longer a credential source, so a ref the store
does not hold is reported missing rather than served from the file. This was the
last reader leg to leave, and its holder is now deleted outright (PR2b).

The store resolver never touches the environment: the runner's contract is that
the environment is reachable only through an explicit ``EnvSecretResolver`` over
names the caller listed, and a store resolver that quietly served ambient
variables would make "resolved from the store" a false claim in the operator's
own proof.
"""

from __future__ import annotations

from typing import Any, Sequence

from local_operator.evaluation.adapters.api import ResolvedSecret
from local_operator.evaluation.runner.secrets import build_resolved_secret


class CredentialStoreResolver:
    """Resolve ``SecretRef`` names from the harness secret store.

    ``config_dir`` locates the encrypted store, or ``None`` for the
    HOME-derived default. It used to be a ``CredentialManager`` whose
    ``config_dir`` this read, and is now the bare PATH that class used to carry
    (PR2b) — the readers only ever needed the root, and ``ConfigManager`` owns it.

    The encrypted store is read in the AGENT namespace, so a
    ``LOP_PROVIDER_*`` row can never satisfy an evaluation ref.
    """

    def __init__(self, config_dir: Any) -> None:
        # Accepts the config ROOT (a ``Path``, the post-PR2b shape) or any object
        # that carries a ``config_dir`` — ``ConfigManager``, or an evaluation
        # host's stand-in. The attribute read is kept because a host builds this
        # before it needs the config stack, so this module imports no
        # configuration machinery at module scope.
        if hasattr(config_dir, "config_dir"):
            config_dir = config_dir.config_dir
        self._config_dir = config_dir

    def _stored_secrets(self) -> dict[str, str]:
        """Agent-class store values by name, or ``{}`` when no store is readable.

        Best-effort, matching the readers elsewhere: an absent, locked or
        damaged store is "this leg has nothing" rather than a failure, so a name
        the store cannot produce is reported missing rather than turning the
        resolve into an error. A single unreadable row is skipped, not fatal —
        one damaged record must not hide every other credential, the store's own
        ``list`` rule.
        """
        from local_operator.secrets.access import open_store, retrieve_secret
        from local_operator.secrets.errors import SecretStoreError
        from local_operator.secrets.keys import store_path
        from local_operator.secrets.store import is_provider_secret_name

        base = self._config_dir
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
                values[name] = retrieve_secret(name, base, role="agent").decode("utf-8", "replace")
            except (SecretStoreError, OSError, ValueError):
                continue
        return values

    def resolve(self, names: Sequence[str]) -> tuple[ResolvedSecret, ...]:
        stored = self._stored_secrets()
        resolved: list[ResolvedSecret] = []
        for name in names:
            # The encrypted store, agent namespace. A name the store does not
            # hold resolves to an EMPTY value — the legacy plaintext fallback
            # this used to consult is GONE (PR2a), so a ref the operator never
            # stored is honestly missing rather than quietly served from a file
            # the store's own resolution no longer reads.
            resolved.append(build_resolved_secret(name, stored.get(name, "")))
        return tuple(resolved)
