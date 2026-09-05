"""Optional native metadata over the existing HTTP discovery client.

Chat remains OpenAI-compatible. Native reads enrich that same catalogue under
one deadline; absence of an extension never makes a compatible server unusable.
No request here loads, downloads, unloads, or changes a model.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import httpx


def discover_local(
    provider: str, base_url: str, api_key: str | None, client: httpx.Client, timeout: float
):
    from local_operator.model.discovery import (
        DiscoveredModel,
        _entry_list,
        _row_from_openai_entry,
    )
    from local_operator.providers.local import DEFAULT_LOCAL_CONTEXT

    deadline = time.monotonic() + timeout
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    root = base_url.removesuffix("/v1")

    def get(path: str, payload: dict[str, Any] | None = None, *, optional: bool = False):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            response = (
                client.post(
                    path, headers=headers, json=payload, timeout=remaining, follow_redirects=False
                )
                if payload is not None
                else client.get(path, headers=headers, timeout=remaining, follow_redirects=False)
            )
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError):
            if optional:
                return None
            raise

    body = get(base_url + "/models")
    entries = _entry_list(body, "data", "models")
    if entries is None:
        raise ValueError("The server did not return an OpenAI-compatible model list.")
    rows: dict[str, DiscoveredModel] = {}
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("type") in {"embedding", "embeddings"}:
            continue
        row = _row_from_openai_entry(entry, provider)
        if row is not None:
            active = _positive(entry.get("max_model_len"))
            maximum = _positive(entry.get("max_context_length")) or row.max_context_window
            # A model's advertised/training maximum is not its loaded runtime
            # allocation. Only a serving limit (vLLM) or native active metadata
            # can seed that budget; retain maxima separately for inspection.
            context = active or min(DEFAULT_LOCAL_CONTEXT, maximum or DEFAULT_LOCAL_CONTEXT)
            rows[row.id] = dataclasses.replace(
                row,
                context_window=context,
                max_context_window=maximum,
                free=provider != "openai-compatible" or row.free,
                active_context_window=active,
                supports_tools=_boolean(entry.get("supports_tools")),
                supports_images=(
                    _boolean(entry.get("supports_images"))
                    if "supports_images" in entry
                    else row.supports_images
                ),
                reasoning=_boolean(entry.get("reasoning")),
            )

    if provider == "lmstudio":
        native = get(root + "/api/v1/models", optional=True)
        if isinstance(native, dict) and isinstance(native.get("models"), list):
            for entry in native["models"]:
                if not isinstance(entry, dict):
                    continue
                model_id = entry.get("key") or entry.get("id")
                if not isinstance(model_id, str):
                    continue
                if entry.get("type") == "embedding":
                    rows.pop(model_id, None)
                    continue
                instances = entry.get("loaded_instances", [])
                active = (
                    [
                        _positive(i.get("config", {}).get("context_length"))
                        for i in instances
                        if isinstance(i, dict) and isinstance(i.get("config", {}), dict)
                    ]
                    if isinstance(instances, list)
                    else []
                )
                active_context = min((n for n in active if n), default=None)
                capabilities = entry.get("capabilities", {})
                capabilities = capabilities if isinstance(capabilities, dict) else {}
                reasoning = capabilities.get("reasoning")
                if isinstance(reasoning, dict) and isinstance(
                    reasoning.get("allowed_options"), list
                ):
                    reasoning = "on" in reasoning["allowed_options"]
                served = {model_id: active_context}
                for instance in instances if isinstance(instances, list) else []:
                    if isinstance(instance, dict) and isinstance(instance.get("id"), str):
                        config = instance.get("config", {})
                        served[instance["id"]] = (
                            _positive(config.get("context_length"))
                            if isinstance(config, dict)
                            else None
                        )
                # LM Studio exposes both a library key and user-named loaded
                # IDs. Only enrich IDs the compatible list actually serves.
                for served_id, active_context in served.items():
                    if served_id not in rows:
                        continue
                    row = rows[served_id]
                    rows[served_id] = dataclasses.replace(
                        row,
                        name=entry.get("display_name") or row.name,
                        context_window=active_context
                        or min(
                            _positive(entry.get("max_context_length")) or DEFAULT_LOCAL_CONTEXT,
                            DEFAULT_LOCAL_CONTEXT,
                        ),
                        max_context_window=_positive(entry.get("max_context_length")),
                        active_context_window=active_context,
                        supports_images=_boolean(capabilities.get("vision")),
                        supports_tools=_boolean(capabilities.get("trained_for_tool_use")),
                        reasoning=_boolean(reasoning),
                        free=True,
                    )
    elif provider == "ollama":
        running = get(root + "/api/ps", optional=True)
        active_by_id = (
            {
                m.get("name") or m.get("model"): _positive(m.get("context_length"))
                for m in running.get("models", [])
                if isinstance(m, dict)
            }
            if isinstance(running, dict)
            else {}
        )
        # Only a bounded prefix gets optional detail. Cached listings and /ps
        # still describe the rest; a large local library cannot block the TUI.
        for index, (model_id, row) in enumerate(list(rows.items())):
            detail = (
                get(root + "/api/show", {"model": model_id}, optional=True) if index < 8 else None
            )
            detail = detail if isinstance(detail, dict) else {}
            info = detail.get("model_info", {})
            info = info if isinstance(info, dict) else {}
            trained = max(
                (_positive(v) or 0 for k, v in info.items() if k.endswith(".context_length")),
                default=0,
            )
            caps = detail.get("capabilities")
            if isinstance(caps, list) and "embedding" in caps and "completion" not in caps:
                rows.pop(model_id)
                continue
            active_context = active_by_id.get(model_id)
            rows[model_id] = dataclasses.replace(
                row,
                context_window=active_context
                or min(trained or DEFAULT_LOCAL_CONTEXT, DEFAULT_LOCAL_CONTEXT),
                max_context_window=trained or row.max_context_window,
                active_context_window=active_context,
                supports_tools=("tools" in caps) if isinstance(caps, list) else row.supports_tools,
                supports_images=(
                    ("vision" in caps) if isinstance(caps, list) else row.supports_images
                ),
                reasoning=("thinking" in caps) if isinstance(caps, list) else row.reasoning,
            )
    elif provider == "llamacpp":
        props = get(root + "/props", optional=True)
        settings = props.get("default_generation_settings", {}) if isinstance(props, dict) else {}
        active_context = _positive(settings.get("n_ctx")) if isinstance(settings, dict) else None
        if active_context:
            rows = {
                key: dataclasses.replace(
                    row, context_window=active_context, active_context_window=active_context
                )
                for key, row in rows.items()
            }
    return list(rows.values())


def _positive(value: object) -> int | None:
    return value if type(value) is int and value > 0 else None


def _boolean(value: object) -> bool | None:
    return value if type(value) is bool else None
