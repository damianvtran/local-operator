"""Shared policy for user-operated OpenAI-compatible servers.

The endpoint belongs to the provider, not to a UI or transport. Keeping its
normalization and model overrides here prevents discovery from inspecting one
server while inference sends a conversation (and its bearer) to another.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

LOCAL_PRESETS = {
    "lmstudio": ("LM Studio", "http://localhost:1234/v1", "https://lmstudio.ai"),
    "ollama": ("Ollama", "http://localhost:11434/v1", "https://ollama.com"),
    "vllm": ("vLLM", "http://localhost:8000/v1", "https://vllm.ai"),
    "llamacpp": ("llama.cpp", "http://localhost:8080/v1", "https://github.com/ggml-org/llama.cpp"),
    "openai-compatible": (
        "OpenAI-compatible",
        "",
        "https://platform.openai.com/docs/api-reference",
    ),
}
LOCAL_PROVIDER_IDS = frozenset(LOCAL_PRESETS)
DEFAULT_MODEL_OVERRIDES = "{}"
DEFAULT_LOCAL_CONTEXT = 4096
DEFAULT_LOCAL_MAX_OUTPUT = 1024


def normalize_base_url(value: str) -> str:
    """Accept a server or API root, never credentials disguised as a URL.

    Proxy prefixes survive normalization. A URL copied from a request rather
    than a server root is rejected rather than silently retargeted.
    """
    value = value.strip()
    try:
        parsed = urlsplit(value)
        valid = parsed.scheme in {"http", "https"} and parsed.hostname and parsed.port != 0
    except ValueError:
        raise ValueError("Enter an HTTP or HTTPS server URL with a valid port.") from None
    if not valid or any(c.isspace() for c in value):
        raise ValueError("Enter an HTTP or HTTPS server URL, including its port if needed.")
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Use a server URL without credentials, query parameters, or a fragment.")
    path = parsed.path.rstrip("/")
    if path.endswith(("/models", "/chat/completions", "/responses")):
        raise ValueError("Use the server API root, not a models or completions URL.")
    while path.endswith("/v1"):
        path = path[:-3].rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path + "/v1", "", ""))


def provider_settings(provider: str, values: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    if values is None:
        from local_operator.config import ConfigManager
        from local_operator.paths import config_dir

        values = ConfigManager(config_dir()).get_config().values
    providers = values.get("providers", {})
    entry = providers.get(provider, {}) if isinstance(providers, Mapping) else {}
    return entry if isinstance(entry, Mapping) else {}


def resolve_base_url(
    provider: str, *, override: str | None = None, values: Mapping[str, Any] | None = None
) -> str:
    from local_operator.providers.registry import get_provider_definition

    definition = get_provider_definition(provider)
    if definition is None:
        raise ValueError(f"Unknown provider: {provider}")
    if provider not in LOCAL_PROVIDER_IDS:
        return override or definition.base_url or ""
    configured = provider_settings(provider, values).get("base_url")
    value = (
        override
        or (configured if isinstance(configured, str) else None)
        or LOCAL_PRESETS[provider][1]
    )
    return normalize_base_url(value) if value else ""


def validate_endpoint_setting(value: str) -> str:
    # Empty means reset to a preset (or leave a generic gateway unconfigured).
    return normalize_base_url(value) if value.strip() else ""


def endpoint_cache_key(key: str, endpoint: str) -> str:
    # Include the proxy path as well as the origin: two models services may
    # share one gateway, and their arbitrary IDs are not interchangeable.
    digest = hashlib.sha256(endpoint.encode()).hexdigest()[:20]
    return f"{key.removesuffix('.listing')}.endpoint.{digest}.listing"


def model_overrides(value: object) -> dict[str, dict[str, Any]]:
    """Validate the settings editor's JSON text and hand-written YAML equally."""
    if isinstance(value, str):
        if len(value.encode("utf-8")) > 65_536:
            raise ValueError("Model overrides must fit within 64 KiB.")
        try:
            value = json.loads(value)
        except ValueError:
            raise ValueError(
                "Enter a JSON object keyed by exact model ID, for example "
                '{"my-model":{"context_window":8192}}.'
            ) from None
    if not isinstance(value, dict):
        raise ValueError("Model overrides must be an object keyed by exact model ID.")
    if len(value) > 256:
        raise ValueError("Configure at most 256 model overrides per server.")
    allowed = {
        "context_window",
        "max_output_tokens",
        "supports_tools",
        "supports_images",
        "reasoning",
        "supports_sampling_params",
    }
    for model, fields in value.items():
        if (
            not isinstance(model, str)
            or not model.strip()
            or len(model) > 512
            or not isinstance(fields, dict)
        ):
            raise ValueError("Each model ID must name an object of overrides.")
        for key, item in fields.items():
            if key not in allowed:
                raise ValueError(f"Unsupported model override: {key}.")
            if key in {"context_window", "max_output_tokens"}:
                if type(item) is not int or item <= 0:
                    raise ValueError(f"{key} must be a positive whole number.")
            elif type(item) is not bool:
                raise ValueError(f"{key} must be true or false.")
    return value


def local_api_key(provider: str) -> str | None:
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore()
    try:
        endpoint = resolve_base_url(provider)
        for row in reversed(store.list_credentials(provider)):
            if row.data.get("endpoint") == endpoint and row.credential_type == "api_key":
                return str(row.data.get("key") or "") or None
        return None
    finally:
        store.close()


def local_model_spec(
    provider: str, model_id: str, *, api_key: str | None = None, cached_only: bool = False
):
    """Resolve only server evidence and explicit overrides, never cloud-name heuristics.

    The fallback is a conservative working budget, not a claim about the model.
    An active server limit always caps overrides: changing a client budget does
    not resize the model loaded in the inference process.
    """
    from local_operator.harness.types import ModelSpec
    from local_operator.model.discovery import available_models, cached_available_models

    endpoint = resolve_base_url(provider)
    if not endpoint:
        raise ValueError(
            "Configure the server URL with /login openai-compatible before choosing a model."
        )
    if cached_only:
        rows, _ = cached_available_models(provider)
    else:
        rows, _ = available_models(
            provider,
            api_key=api_key or local_api_key(provider),
            base_url=endpoint,
            want_id=model_id,
        )
    row = next((r for r in rows if r.id == model_id), None)
    overrides = model_overrides(
        provider_settings(provider).get("models", DEFAULT_MODEL_OVERRIDES)
    ).get(model_id, {})
    context = overrides.get(
        "context_window",
        row.context_window if row and row.context_window else DEFAULT_LOCAL_CONTEXT,
    )
    active = row.active_context_window if row else None
    if active:
        context = min(context, active)
    output = min(
        overrides.get(
            "max_output_tokens",
            row.max_tokens if row and row.max_tokens else DEFAULT_LOCAL_MAX_OUTPUT,
        ),
        max(1, context // 2),
    )
    return ModelSpec(
        provider=provider,
        model_id=model_id,
        base_url=endpoint,
        context_window=context,
        max_output_tokens=output,
        context_metadata_resolved=True,
        max_context_window=active,
        default_context_window=row.context_window if row and row.context_window else None,
        supports_tools=overrides.get(
            "supports_tools", row.supports_tools is not False if row else True
        ),
        supports_images=overrides.get(
            "supports_images", row.supports_images is True if row else False
        ),
        reasoning=overrides.get("reasoning", row.reasoning is True if row else False),
        supports_sampling_params=overrides.get("supports_sampling_params", True),
        temperature=None,
        top_p=None,
    )


def local_model_info(provider: str, model_id: str):
    from local_operator.model.discovery import cached_available_models
    from local_operator.model.registry import ModelInfo

    spec = local_model_spec(provider, model_id, cached_only=True)
    rows, _ = cached_available_models(provider)
    row = next((row for row in rows if row.id == model_id), None)
    free = provider != "openai-compatible" or bool(row and row.free)
    return ModelInfo(
        id=model_id,
        name=model_id,
        context_window=spec.context_window,
        max_tokens=spec.max_output_tokens,
        supports_images=spec.supports_images,
        supports_prompt_cache=False,
        input_price=0.0 if free else row.input_price if row and row.input_price > 0 else -1.0,
        output_price=0.0 if free else row.output_price if row and row.output_price > 0 else -1.0,
        description="User-operated model; limits use server metadata or explicit overrides.",
    )
