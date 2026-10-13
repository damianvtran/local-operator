"""Owner-specific provider admission, not authentication or a version threshold.

A current daemon can offer a provider the still-running conversation owner never
learned. Accepting that choice and later calling its missing sealed-key reader
misreports a healthy sign-in as absent. Positive owner evidence is authoritative;
legacy missing/truncated inventory is only unknown, never proof of no support.
"""

from __future__ import annotations

from collections.abc import Collection


class RuntimeUpgradeNeededError(ValueError):
    """Typed, non-mutating refusal; a sign-in must not be suggested as repair."""

    code = "runtime_upgrade_needed"

    def __init__(self, provider: str, *, support: str = "unverified") -> None:
        self.provider = provider
        self.support = support
        action = "verify support for" if support == "unverified" else "use"
        super().__init__(
            f"This conversation's runtime cannot {action} "
            f"{provider}. Your saved sign-in is unchanged. Start a new conversation, "
            "or resume this one after its runtime updates at an idle boundary."
        )


def native_chat_provider_ids() -> list[str]:
    """A bounded registry claim made only by the process loading this source."""
    from local_operator.providers.registry import (
        PROVIDER_REGISTRY,
        credential_provider_id,
    )

    return sorted(
        {
            credential_provider_id(row.id)
            for row in PROVIDER_REGISTRY
            if not (row.decision_only or row.speech_only or row.media_only)
        }
    )


def require_owner_provider(
    provider: str,
    *,
    established: Collection[str],
    native: Collection[str] | None,
    positive_catalogue: Collection[str] = (),
    stale: bool = False,
) -> None:
    """No model-id allowlist: a new id under an affirmed provider stays valid.

    Unknown/custom providers retain the existing compatible-host contract. Only
    daemon-known builtin choices require owner evidence, and only stale legacy
    owners need the unreported-capability refusal. No credentials are read here.
    """
    from local_operator.providers.registry import (
        credential_provider_id,
        get_provider_definition,
    )

    if get_provider_definition(provider) is None:
        return
    canonical = credential_provider_id(provider)
    known = {credential_provider_id(item) for item in established}
    if canonical in known:
        return
    if native is not None:
        if canonical not in {credential_provider_id(item) for item in native}:
            raise RuntimeUpgradeNeededError(provider, support="unsupported")
        return
    if stale and canonical not in {credential_provider_id(item) for item in positive_catalogue}:
        raise RuntimeUpgradeNeededError(provider)
