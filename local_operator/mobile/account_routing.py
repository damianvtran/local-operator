"""One authenticated relay setting projection, not arbitrary config access.

The daemon owns authentication/origin checks before calling this module. Both
phone clients consume backend metadata and effective values; neither carries a
second account router or a device-local account-selection preference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from local_operator import settings_io
from local_operator.config import CONFIG_FILE_NAME, ConfigManager
from local_operator.i18n.keys import wire_settings_account_policy as _ap
from local_operator.i18n.messages import Msg
from local_operator.i18n.resolve import resolve_language

POLICY_KEY = "retry.accountSelectionPolicy"
CAPABILITY = "account-routing-v1"

#: ADR 0008 desk presence: the second single-setting projection. The phone's
#: per-computer "send to phone when I'm away for N" row reads and writes it;
#: the relay advertises ``capabilities.phone_away_after`` for it.
PHONE_AWAY_AFTER_KEY = "notifications.phone_away_after_s"
PHONE_AWAY_AFTER_CAPABILITY = "phone-away-after-v1"


class AccountRoutingError(Exception):
    def __init__(self, code: str, message: str, status: int):
        super().__init__(message)
        self.code = code
        self.status = status


def policy_payload(root: Path, patch: Any = None, *, write: bool = False) -> dict[str, Any]:
    """Read or patch the one non-security setting through the existing facade."""
    return setting_payload(
        root,
        POLICY_KEY,
        CAPABILITY,
        patch,
        write=write,
        unavailable_code="account_routing_unavailable",
        unavailable=_ap.error_unavailable(),
    )


def setting_payload(
    root: Path,
    key: str,
    capability: str,
    patch: Any = None,
    *,
    write: bool = False,
    unavailable_code: str,
    unavailable: Msg,
) -> dict[str, Any]:
    """Read or patch ONE named registry setting, for an authenticated relay route.

    Shared by every relay route that projects a single setting (account routing,
    and ADR 0008's ``notifications.phone_away_after_s``), so each such route is
    one ``key`` the server names, never a caller-supplied path. Validation and
    persistence are ``settings_io``'s, the same writer the TUI page and
    ``PATCH /v1/settings`` use. The refusal copy is generic ("Expected only a
    value field", "could not be saved"), which is why it is shared too.
    """
    # Resolve from the backend config owner on each request, like relay metadata;
    # envelope() with no locale= renders the process's ACTIVE locale, which the
    # TUI's ambient state can move; it is not a request-scoped authority, so
    # the language is pinned explicitly from config below. Do not introduce a
    # client language override.
    setting = settings_io.resolve_key(key)
    if setting is None:
        raise AccountRoutingError(
            unavailable_code,
            unavailable.envelope(locale=resolve_language())["text"],
            404,
        )
    try:
        settings_io.require_readable_config_path(root / CONFIG_FILE_NAME)
        manager = ConfigManager(root)
        settings_io._require_readable_config(manager)
        if write:
            if not isinstance(patch, dict) or set(patch) != {"value"}:
                raise AccountRoutingError(
                    "invalid_setting",
                    _ap.error_invalid_shape().envelope(locale=resolve_language())["text"],
                    422,
                )
            # Interactive authenticated user action: retain the same attribution
            # as TUI/desktop HTTP, not the agent-only settings_control seam.
            settings_io.write_setting(manager, setting, patch["value"])
        value = settings_io.read_setting(manager, setting)
    except settings_io.ConfigUnreadableError as exc:
        raise AccountRoutingError("config_unreadable", str(exc), 409) from exc
    except ValueError as exc:
        raise AccountRoutingError("invalid_setting", str(exc), 422) from exc
    except OSError as exc:
        raise AccountRoutingError(
            "setting_write_failed",
            _ap.error_write_failed().envelope(locale=resolve_language())["text"],
            503,
        ) from exc
    return {
        "capability": capability,
        "setting": {
            "key": setting.key,
            "label": setting.label,
            "help": setting.help,
            "kind": setting.kind.value,
            "value": value,
            "default": setting.default,
            "is_default": settings_io.is_default(manager, setting),
            "choices": [
                {"value": choice.value, "label": choice.label, "description": choice.description}
                for choice in setting.resolved_choices
            ],
        },
    }
