"""Desktop settings are a projection of settings_io, never a second schema.

Only registered keys cross this boundary. Arbitrary config mappings (variables,
credentials, provider headers) are deliberately not serialized. Writes use the
same merge/reset facade as the CLI and TUI so literal dotted keys and concurrent
edits retain their established semantics.
"""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.server.dependencies import get_config_manager
from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse

router = APIRouter(tags=["Settings"], dependencies=[Depends(require_desktop)])
# The underlying manager writes a whole YAML snapshot. Serialize this server's
# read/merge/write operations, not just the final rename, to preserve siblings.
_write_lock = asyncio.Lock()


class SettingEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: Any
    base: dict[str, list[str]] | None = None


class SettingView(BaseModel):
    key: str
    section: str
    label: str
    kind: str
    help: str
    value: Any
    default: Any
    is_default: bool
    minimum: float | None
    maximum: float | None
    members: list[str]
    choices: list[dict[str, Any]]
    empty_unsets: bool
    redacted: bool = False


class SettingsView(BaseModel):
    sections: list[dict[str, str]]
    settings: list[SettingView]


def _private_endpoint(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        return False
    try:
        url = urlsplit(value)
        # Settings is not a secret editor. Userinfo, query parameters and
        # fragments may carry inline credentials under arbitrary vendor names.
        return bool(url.username or url.password or url.query or url.fragment)
    except ValueError:
        return True


def _view(manager: ConfigManager, setting: settings_io.Setting) -> SettingView:
    value = (
        settings_io.read_chains(manager)
        if setting.kind is settings_io.Kind.CASCADE
        else settings_io.read_setting(manager, setting)
    )
    return SettingView(
        key=setting.key,
        section=setting.section,
        label=setting.label,
        kind=setting.kind.value,
        help=setting.help,
        value=None if _private_endpoint(value) else value,
        default=setting.default,
        redacted=_private_endpoint(value),
        is_default=settings_io.is_default(manager, setting),
        minimum=setting.minimum,
        maximum=setting.maximum,
        members=list(setting.members),
        choices=[
            {"value": c.value, "label": c.label, "description": c.description}
            for c in setting.resolved_choices
        ],
        empty_unsets=setting.empty_unsets,
    )


def _setting(key: str) -> settings_io.Setting:
    setting = settings_io.resolve_key(key)
    if setting is None:
        raise HTTPException(404, "This setting is not registered.")
    return setting


@router.get("/v1/settings", response_model=CRUDResponse[SettingsView])
async def list_settings(manager: ConfigManager = Depends(get_config_manager)):
    async with _write_lock:
        try:
            settings_io._require_readable_config(manager)
            manager.reload()
        except settings_io.ConfigUnreadableError:
            raise HTTPException(
                409, "The configuration file cannot be read. Repair it before saving."
            )
        return CRUDResponse(
            status=200,
            message="Settings retrieved.",
            result=SettingsView(
                sections=[
                    {
                        "name": section.name,
                        "title": section.title,
                        "scope": section.scope.value,
                        "description": section.description,
                    }
                    for section in settings_io.SECTIONS
                ],
                settings=[_view(manager, setting) for setting in settings_io.SETTINGS],
            ),
        )


def _write_cascade(manager: ConfigManager, edit: SettingEdit) -> None:
    # Base is mandatory for a GUI edit: an unchanged chain from a stale screen
    # must never overwrite a newer chain written in a terminal session.
    if edit.base is None or not isinstance(edit.value, dict):
        raise ValueError("Provide the edited chains and their original snapshot.")
    for key, hops in edit.value.items():
        if not isinstance(key, str) or not key.strip() or len(key) > 512:
            raise ValueError("Each chain needs a non-empty name of at most 512 characters.")
        if not isinstance(hops, list) or len(hops) > 100:
            raise ValueError("Each chain must be a list of at most 100 models.")
        for hop in hops:
            if not isinstance(hop, str) or len(hop) > 1024:
                raise ValueError("Each model must be a selector of at most 1024 characters.")
            # Existing display labels can carry effort. Only new text must
            # clear the same validator the TUI uses; otherwise a harmless edit
            # to a different chain would make preserved effort uneditable.
            if hop not in edit.base.get(key, []):
                problem = settings_io.validate_hop(hop)
                if problem:
                    raise ValueError(problem)
    settings_io.write_chains(manager, edit.value, base=edit.base)


@router.patch("/v1/settings/{key}", response_model=CRUDResponse[SettingView])
async def edit_setting(
    key: str, edit: SettingEdit, manager: ConfigManager = Depends(get_config_manager)
):
    setting = _setting(key)
    if _private_endpoint(edit.value):
        raise HTTPException(422, "Use an endpoint without inline credentials or query parameters.")
    async with _write_lock:
        try:
            if setting.kind is settings_io.Kind.CASCADE:
                _write_cascade(manager, edit)
            elif setting.empty_unsets and edit.value == "":
                settings_io.reset_setting(manager, setting)
            else:
                settings_io.write_setting(manager, setting, edit.value)
        except settings_io.ConfigUnreadableError:
            raise HTTPException(
                409, "The configuration file cannot be read. Repair it before saving."
            )
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        return CRUDResponse(status=200, message="Setting saved.", result=_view(manager, setting))


@router.post("/v1/settings/{key}/reset", response_model=CRUDResponse[SettingView])
async def reset_setting(key: str, manager: ConfigManager = Depends(get_config_manager)):
    setting = _setting(key)
    async with _write_lock:
        try:
            settings_io.reset_setting(manager, setting)
        except settings_io.ConfigUnreadableError:
            raise HTTPException(
                409, "The configuration file cannot be read. Repair it before saving."
            )
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        return CRUDResponse(status=200, message="Setting reset.", result=_view(manager, setting))
