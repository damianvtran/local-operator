"""Provider codes cross snapshots intact; only presentation resolves their labels.

This boundary is deliberately TUI-only: backend DTOs and HTTP state codes remain
machine-readable. Retaining codes avoids credential-store reads on locale repaint
and prevents translated text from becoming an authentication predicate.
"""

from dataclasses import dataclass
from types import MappingProxyType

from local_operator.i18n.keys import tui_provider_state as _state
from local_operator.i18n.text import tr


@dataclass(frozen=True)
class ProviderStateRow:
    """An opaque provider identity paired with an untranslated backend state."""

    provider_id: str
    state_code: str


_resolve = tr("tui.provider_state")

_STATE_MESSAGES = MappingProxyType(
    {
        "logged_in": _state.logged_in(),
        "env_key": _state.env_key(),
        "needs_login": _state.needs_login(),
        "local_ready": _state.local_ready(),
        "local_unconfigured": _state.local_unconfigured(),
    }
)


def provider_state_label(state_code: str) -> str:
    """Resolve at paint time; unknown codes fail loudly rather than hiding drift."""
    return _resolve(_STATE_MESSAGES[state_code])
