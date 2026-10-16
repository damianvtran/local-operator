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

POLICY_KEY = "retry.accountSelectionPolicy"
CAPABILITY = "account-routing-v1"


class AccountRoutingError(Exception):
    def __init__(self, code: str, message: str, status: int):
        super().__init__(message)
        self.code = code
        self.status = status


def policy_payload(root: Path, patch: Any = None, *, write: bool = False) -> dict[str, Any]:
    """Read or patch the one non-security setting through the existing facade."""
    setting = settings_io.resolve_key(POLICY_KEY)
    if setting is None:
        raise AccountRoutingError(
            "account_routing_unavailable", _ap.error_unavailable().envelope()["text"], 404
        )
    try:
        settings_io.require_readable_config_path(root / CONFIG_FILE_NAME)
        manager = ConfigManager(root)
        settings_io._require_readable_config(manager)
        if write:
            if not isinstance(patch, dict) or set(patch) != {"value"}:
                raise AccountRoutingError(
                    "invalid_setting", _ap.error_invalid_shape().envelope()["text"], 422
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
            "setting_write_failed", _ap.error_write_failed().envelope()["text"], 503
        ) from exc
    return {
        "capability": CAPABILITY,
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
