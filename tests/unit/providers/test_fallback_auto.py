"""Explicit chain auto resets target effort without changing legacy carry rules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator import settings_io
from local_operator.config import ConfigManager
from local_operator.harness.types import (
    ChatRequest,
    Message,
    StreamTextDelta,
    TextContent,
)
from local_operator.model.configure import SessionStreamFn, build_model_spec
from local_operator.model.effort import FALLBACK_EFFORTS, SUPPORTED_EFFORTS
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.clients import client_for_spec
from local_operator.providers.failover import (
    CHAIN_EFFORT_LADDER,
    FailoverRouteState,
    FallbackTarget,
    RetrySettings,
    expand_fallback_targets,
    spec_for_target,
    stream_with_failover,
)
from local_operator.server.routes.settings import SettingEdit, _write_cascade


def test_auto_normalizes_without_becoming_an_ordinal_rung(caplog: pytest.LogCaptureFixture) -> None:
    settings = RetrySettings.from_settings(
        {
            "retry": {
                "fallbackChains": {
                    "default": [{"provider": "radient", "model": "auto", "effort": " AUTO "}]
                }
            }
        }
    )
    assert expand_fallback_targets("openai/gpt-6.1-sol", settings.fallback_chains["default"]) == [
        FallbackTarget("radient/auto", "auto")
    ]
    assert FALLBACK_EFFORTS == SUPPORTED_EFFORTS | {"auto"}
    assert set(CHAIN_EFFORT_LADDER) == SUPPORTED_EFFORTS
    assert "auto" not in CHAIN_EFFORT_LADDER
    RetrySettings.from_settings(
        {
            "retry": {
                "fallbackChains": {
                    "default": [{"provider": "radient", "model": "auto", "effort": "invalid"}]
                }
            }
        }
    )
    advertised = caplog.records[-1].getMessage().split("expected one of ")[1].split(", ")
    assert set(advertised) == FALLBACK_EFFORTS


@pytest.mark.parametrize(
    "primary_effort, expected", [("high", [FallbackTarget("radient/auto", "auto")]), ("auto", [])]
)
def test_auto_identity_obeys_existing_current_effort_skip(primary_effort, expected) -> None:
    # Absence, a legacy string and identical explicit effort still skip the
    # current route. The reset remains separate when the current dial is high.
    assert (
        expand_fallback_targets(
            "radient/auto",
            [
                "radient/auto",
                {"provider": "radient", "model": "auto"},
                {"provider": "radient", "model": "*", "effort": "auto"},
            ],
            primary_effort=primary_effort,
        )
        == expected
    )


@pytest.mark.parametrize(
    "selector, expected",
    [
        ("radient/auto", "auto"),
        ("anthropic/claude-sonnet-5-5", "high"),
        ("openai/gpt-6.1-sol", None),
        ("deepseek/deepseek-flash", "high"),
        ("alibaba/qwen3-coder-plus", None),
    ],
)
def test_reset_resolves_target_default_not_literal_auto(
    selector: str, expected: str | None
) -> None:
    base = build_model_spec("openai", "gpt-6.1-sol").model_copy(
        update={"reasoning_effort": "max", "fast_mode": True}
    )
    target = spec_for_target(base, FallbackTarget(selector, "auto"))
    assert target.reasoning_effort == expected
    assert target.reasoning_effort == target.reasoning_default_effort
    # A reset must not early-return before unrelated capability carry policy.
    assert target.fast_mode == bool(target.supports_fast_mode)


def test_legacy_and_fixed_effort_still_carry_or_win() -> None:
    base = build_model_spec("openai", "gpt-6.1-sol").model_copy(update={"reasoning_effort": "high"})
    assert spec_for_target(base, FallbackTarget("radient/auto")).reasoning_effort == "high"
    assert spec_for_target(base, FallbackTarget("radient/auto", "low")).reasoning_effort == "low"


def test_auto_survives_settings_write_display_and_original_snapshot(tmp_path: Path) -> None:
    manager = ConfigManager(tmp_path)
    setting = settings_io.BY_KEY["retry.fallbackChains"]
    hop = {"provider": "radient", "model": "auto", "effort": "auto"}
    chains = {"default": [hop], "openai/*": ["deepseek/deepseek-flash"]}
    assert settings_io.validate(setting, chains) is None
    settings_io.write_setting(manager, setting, chains)
    original = settings_io.read_chains(manager)
    assert original["default"] == ["radient/auto (auto)"]
    edited = {**original, "openai/*": ["anthropic/claude-sonnet-5-5"]}
    _write_cascade(manager, SettingEdit(value=edited, base=original))
    reloaded = ConfigManager(tmp_path).get_config_value("retry")["fallbackChains"]
    assert reloaded["default"] == [hop]
    assert reloaded["openai/*"] == ["anthropic/claude-sonnet-5-5"]
    # Fixed choices keep the old validation; typos and 'none' stay rejected.
    for invalid in ("automatic", "none"):
        assert settings_io.validate(setting, {"default": [{**hop, "effort": invalid}]})


class _Auth:
    """Synthetic account pools; never load the operator's auth store."""

    def __init__(self, primary: str) -> None:
        self.keys = {
            primary: ["synthetic-first", "synthetic-sibling"],
            "radient": ["synthetic-target"],
        }
        self.rotations: list[str] = []

    async def get_api_key(
        self, provider: str, session_id: str | None = None, **kwargs: Any
    ) -> str | None:
        return next(iter(self.keys.get(provider, [])), None)

    def rotate_sibling(
        self,
        provider: str,
        session_id: str | None,
        error: Any,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> bool:
        self.rotations.append(provider)
        pool = self.keys.get(provider, [])
        if api_key in pool:
            pool.remove(api_key)
        return bool(pool)


@pytest.mark.asyncio
@pytest.mark.parametrize("primary, absent_effort", [("openai", "high"), ("anthropic", "auto")])
@pytest.mark.parametrize("explicit_auto", [False, True])
async def test_real_client_wire_siblings_pin_and_subsequent_turn(
    primary: str, absent_effort: str, explicit_auto: bool
) -> None:
    bodies: list[tuple[str, dict[str, Any]]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append((request.url.host, body))
        if request.url.host != "api.radienthq.com":
            return httpx.Response(
                429,
                json={
                    "error": {"message": "synthetic quota exhausted", "type": "insufficient_quota"}
                },
            )
        chunk = {
            "id": "offline-completion",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "auto",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "served offline"},
                    "finish_reason": "stop",
                }
            ],
        }
        return httpx.Response(
            200,
            content="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
            headers={"content-type": "text/event-stream"},
        )

    base = build_model_spec(
        primary, "gpt-6.1-sol" if primary == "openai" else "claude-sonnet-5-5"
    ).model_copy(update={"reasoning_effort": "high"})
    hop = {"provider": "radient", "model": "auto"}
    if explicit_auto:
        hop["effort"] = "auto"
    settings = {"retry": {"maxRetries": 0, "fallbackChains": {"default": [hop]}}}
    auth = _Auth(primary)
    route = FailoverRouteState()
    request = ChatRequest(
        model=base,
        max_tokens=64,
        messages=[Message(role="user", content=[TextContent(text="offline effort routing probe")])],
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:

        def client_for(spec):
            return client_for_spec(spec, http_client=http)

        for _ in range(2):
            events = [
                event
                async for event in stream_with_failover(
                    request,
                    auth,
                    settings,
                    client_for,
                    session_id="synthetic-auto-hop",
                    route_state=route,
                )
            ]
            assert (
                "".join(e.delta for e in events if isinstance(e, StreamTextDelta))
                == "served offline"
            )
    assert [host for host, _ in bodies] == [
        "api.openai.com" if primary == "openai" else "api.anthropic.com"
    ] * 2 + ["api.radienthq.com"] * 2
    assert auth.rotations == [primary, primary]
    assert route.active == FallbackTarget("radient/auto", "auto" if explicit_auto else None)
    for _, body in bodies[-2:]:
        assert body["model"] == "auto"
        assert body["messages"][0]["role"] == "user"
        assert body["reasoning_effort"] == ("auto" if explicit_auto else absent_effort)


def test_preflight_expansion_and_restored_route_preserve_auto(tmp_path: Path) -> None:
    stream = SessionStreamFn(
        auth_store=AuthStore(tmp_path / "auth.db"),
        session_id="synthetic-auto-hop",
        settings={
            "retry": {
                "fallbackChains": {
                    "default": [{"provider": "radient", "model": "auto", "effort": "auto"}]
                }
            }
        },
    )
    base = build_model_spec("openai", "gpt-6.1-sol").model_copy(update={"reasoning_effort": "high"})
    assert stream._fallback_targets(base) == [FallbackTarget("radient/auto", "auto")]
    stream.restore_fallback("radient/auto", "auto", "openai/gpt-6.1-sol")
    active = stream._route_state.active
    assert active is not None
    assert active == FallbackTarget("radient/auto", "auto")
    assert spec_for_target(base, active).reasoning_effort == "auto"
