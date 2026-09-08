"""Conversation model identity, shared by owners and runtime-less viewers.

The journal is authoritative, not config.yml and not the effective fallback
route. Version 2 rows own the initial selection as well as explicit switches.
Legacy writers only journalled switches; their later frontend checkpoints can
therefore be newer evidence of the primary they actually selected on resume.
Reading never creates a directory or migrates a transcript: only its leased
owner may append the upgraded selection when it admits real work.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

SELECTED_MODEL_CUSTOM_TYPE = "selected_model"
SELECTION_VERSION = 2


@dataclass(frozen=True)
class StoredModelSelection:
    provider: str
    model_id: str
    effort: str | None = None
    authoritative: bool = False
    recovered: bool = False
    # This is the conversation's birth selector, not the pair configured while
    # constructing a resumed owner. Existing /effort journalling depends on it.
    boot_selector: str | None = None

    @property
    def selector(self) -> str:
        return f"{self.provider}/{self.model_id}"


def _selection(selector: Any, effort: Any, *, authoritative: bool = False, boot: Any = None):
    if not isinstance(selector, str) or "/" not in selector:
        return None
    provider, model_id = selector.split("/", 1)
    if not provider or not model_id:
        return None
    from local_operator.providers.registry import get_provider_definition

    if get_provider_definition(provider) is None:
        return None
    return StoredModelSelection(
        provider,
        model_id,
        effort if isinstance(effort, str) and effort else None,
        authoritative,
        boot_selector=(
            boot if isinstance(boot, str) and "/" in boot and all(boot.split("/", 1)) else None
        ),
    )


def selection_from_payloads(payloads: Iterable[dict[str, Any]]) -> StoredModelSelection | None:
    """Resolve CUSTOM-entry payloads; both callers enforce the envelope first."""
    authoritative = None
    legacy = None
    unusable = False
    for payload in payloads:
        kind = payload.get("custom_type")
        details = payload.get("details")
        if not isinstance(details, dict):
            continue
        if kind == SELECTED_MODEL_CUSTOM_TYPE:
            version = details.get("version")
            selected = _selection(
                details.get("selector"),
                details.get("effort"),
                authoritative=version == SELECTION_VERSION,
                boot=details.get("boot"),
            )
            if selected is not None:
                if selected.authoritative:
                    authoritative = selected
                    unusable = False
                elif version is None:
                    legacy = selected
            elif version == SELECTION_VERSION or (version is None and authoritative is None):
                unusable = True
        elif kind == "frontend_state_checkpoint_v1":
            state = details.get("state")
            model = state.get("selected_model") if isinstance(state, dict) else None
            if isinstance(model, dict):
                selected = _selection(
                    f"{model.get('provider', '')}/{model.get('model_id', '')}",
                    model.get("reasoning_effort"),
                )
                if selected is not None:
                    # A checkpoint refreshes the observed primary, not its
                    # birth. Retain known provenance only for the SAME primary;
                    # an abandoned old switch cannot lend its birth to a new one.
                    if legacy is not None and legacy.selector == selected.selector:
                        selected = replace(selected, boot_selector=legacy.boot_selector)
                    legacy = selected
    selected = authoritative or legacy
    return replace(selected, recovered=unusable) if selected is not None else None


def read_model_selection(directory: Path) -> StoredModelSelection | None:
    """Read only identity rows without loading message attachments or history."""

    def payloads():
        try:
            with (directory / "transcript.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    # Large transcripts mostly contain messages. Do not parse
                    # their payloads just to recover a two-field selector.
                    if '"custom_type"' not in line:
                        continue
                    try:
                        row = json.loads(line)
                    except (ValueError, TypeError):
                        continue
                    if (
                        isinstance(row, dict)
                        and row.get("type") == "custom"
                        and isinstance(row.get("payload"), dict)
                    ):
                        yield row["payload"]
        except (OSError, UnicodeError):
            return

    return selection_from_payloads(payloads())
