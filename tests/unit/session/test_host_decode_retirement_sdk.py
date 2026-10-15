"""Real typed-SDK session: a slow host's retirement reaches the NEXT request and the user.

``tests/unit/model/test_host_decode_retirement.py`` proves the policy against the
stream funnel. This proves the chain a user actually rides: ``sdk.open_session``
(the real ``session_factory.create_session``) -> real ``Session`` and agent loop
-> the real ``SessionStreamFn`` -> the real ``_record_stream`` funnel -> the real
``OpenAICompatClient._build_body`` rendering ``provider.ignore`` -> the session's
``NoticeEvent`` bridge. ONLY the transport call is scripted
(``OpenAICompatClient.stream``): the scripted stream renders the request through
the client's own body builder, so what is asserted is the byte-for-byte
``provider`` object OpenRouter would have been sent, and the scripted router
honours it the way OpenRouter does (an ``ignore``d host is never served).

Zero spend, zero network. A tripwire replaces the HTTP transports for the whole
test and the case fails if ANYTHING tried to leave the process, so "no network"
is observed, not assumed. The measured timing is controlled by a fake monotonic
clock the scripted stream advances: the decode window is the real
first-to-last-delta measurement, with an exact, reproducible duration.
"""

from __future__ import annotations

import os
import threading
import time as _real_time
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
import pytest
import requests
import yaml

from local_operator.harness.types import (
    ChatRequest,
    StreamEndEvent,
    StreamStartEvent,
    StreamTextDelta,
    Usage,
)
from local_operator.model import configure as configure_mod
from local_operator.model.registry import ModelInfo

MODEL = "deepseek/deepseek-v4.1-flash"
#: Decode rates (tok/s) of the two scripted hosts: the measured slow host and a
#: healthy one, from the audit this change answers (20.3 vs ~130-172).
_TPS = {"SlowHost": 20.3, "FastHost": 150.0}


class _Clock:
    now = 50_000.0


class _TimeProxy:
    def monotonic(self) -> float:
        return _Clock.now

    def __getattr__(self, name: str) -> Any:
        return getattr(_real_time, name)


@pytest.fixture
def offline_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """An isolated root, a pinned model row, a fake clock and a network tripwire."""
    from local_operator import agent_shell
    from local_operator.sdk import SessionRoots
    from local_operator.skills import api as skills_api
    from local_operator.skills.embeddings import LocalEmbedder

    home = tmp_path / "home"
    root = home / ".local-operator"
    agent_home = home / "local-operator-home"
    work = home / "work"
    for directory in (root, agent_home, work):
        directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "decoy-config"))
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(tmp_path / "decoy-home"))
    # A synthetic credential: it exists only so the request is allowed to reach the
    # (scripted) client; nothing ever transmits it.
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-synthetic-not-a-real-key")
    for name in (
        agent_shell.AGENT_SHELL_ENV,
        agent_shell.MAY_DELEGATE_ENV,
        agent_shell.ALLOW_NESTED_SESSION_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    for key in [k for k in os.environ if k.startswith(("CMUX_", "LOP_"))]:
        monkeypatch.delenv(key, raising=False)

    # The session reads its model capability row through this seam. Pinning it
    # keeps the catalogue/price fetches (real network) out of the run, and states
    # the one fact the affinity gate needs: this model caches prompts.
    monkeypatch.setattr(
        configure_mod,
        "resolve_model_info",
        lambda provider, model_id, *a, **k: ModelInfo(
            input_price=0.1,
            output_price=0.3,
            max_tokens=32768,
            context_window=1_000_000,
            supports_images=False,
            supports_prompt_cache=True,
            cache_writes_price=0.0,
            cache_reads_price=0.01,
            description="synthetic row",
            id=MODEL,
            name="synthetic",
            recommended=False,
        ),
    )
    # Knowledge indexing would embed over HTTP with the synthetic OpenRouter key.
    monkeypatch.setattr(skills_api, "default_backend_from_env", lambda *a, **k: LocalEmbedder())

    # The analytics recorder prices every call on its own background thread, and
    # pricing resolves the model catalogue over HTTP. That thread outlives the
    # request, so it raced this test's tripwire (measured: 3 of 6 combined runs
    # tripped it on a /v1/models listing) -- and it would also write a row to an
    # analytics store this test has no business touching. Analytics is not under
    # test here, so the funnel's hand-off is replaced with a no-op.
    import local_operator.analytics as analytics

    monkeypatch.setattr(analytics, "record_call", lambda snapshot: None)

    attempts: list[str] = []

    owner = threading.current_thread()

    def tripwire(label: str) -> Any:
        def trip(self: Any, request: Any, *a: Any, **k: Any) -> Any:
            # Only the TEST'S OWN thread counts. The process also hosts background
            # threads this test does not own -- the analytics writer, catalogue
            # revalidation -- which an EARLIER test in the same worker may have
            # started and which fetch a pricing listing over HTTP. Counting them
            # made this case fail 10 of 15 combined runs for a reason unrelated to
            # the routing under test. The conversation itself (the funnel, the
            # client, classification, knowledge indexing) all runs on this thread.
            if threading.current_thread() is owner:
                attempts.append(f"{label}: {getattr(request, 'url', request)}")
            raise AssertionError(f"network attempt blocked: {label}")

        return trip

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", tripwire("httpx-async"))
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", tripwire("httpx-sync"))
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", tripwire("requests"))

    _Clock.now = 50_000.0
    monkeypatch.setattr(configure_mod, "time", _TimeProxy())

    def write_settings(values: dict[str, Any]) -> None:
        # Classification would ask a vendor model to grade each prompt: off, so the
        # only provider traffic is the conversation under test.
        merged = {"classification": {"auto": False}, **values}
        (root / "config.yml").write_text(yaml.safe_dump({"values": merged}))

    write_settings({})
    return {
        "roots": SessionRoots(
            config_dir=root, agent_home=agent_home, cwd=work, allow_volatile=True
        ),
        "attempts": attempts,
        "write_settings": write_settings,
    }


class _Router:
    """Scripted OpenRouter: renders the REAL body, honours ``ignore``, decodes at a rate."""

    def __init__(self, plan: list[str]) -> None:
        self.plan = list(plan)
        self.sent: list[dict[str, Any] | None] = []
        self.served: list[str] = []


def _install_router(monkeypatch: pytest.MonkeyPatch, router: _Router) -> None:
    from local_operator.providers.clients import OpenAICompatClient

    async def scripted(
        self: Any, request: ChatRequest, api_key: str | None, oauth_access: Any = None
    ) -> AsyncIterator[Any]:
        if request.purpose != "turn":
            # A helper call (naming, titles): a different, cheap shape that must
            # neither score nor move anything. Answered by an unnamed host.
            yield StreamStartEvent(response_id="helper")
            yield StreamTextDelta(delta="ok")
            yield StreamEndEvent(stop_reason="stop", usage=Usage(input_tokens=10, output_tokens=2))
            return
        body = self._build_body(request)
        router.sent.append(body.get("provider"))
        wanted = router.plan.pop(0)
        ignored = set((body.get("provider") or {}).get("ignore", []))
        host = "FastHost" if wanted in ignored else wanted
        router.served.append(host)
        tokens = 1700
        yield StreamStartEvent(response_id="r")
        _Clock.now += 8.0  # queue + prefill: must not count toward decode
        for index in range(40):
            yield StreamTextDelta(delta="x")
            if index < 39:
                _Clock.now += (tokens / _TPS[host]) / 39
        yield StreamEndEvent(
            stop_reason="stop",
            served_provider=host,
            usage=Usage(input_tokens=100, output_tokens=tokens),
        )

    monkeypatch.setattr(OpenAICompatClient, "stream", scripted)


async def _converse(world: Any, plan: list[str], monkeypatch: pytest.MonkeyPatch) -> Any:
    from local_operator import sdk
    from local_operator.session.spec import ApprovalPolicy, SessionSpec

    router = _Router(plan)
    _install_router(monkeypatch, router)
    notices: list[str] = []
    spec = SessionSpec(hosting="openrouter", model=MODEL, approvals=ApprovalPolicy.auto())
    async with sdk.open_session(spec, roots=world["roots"]) as session:
        session.subscribe(
            lambda event: (
                notices.append(event.text) if getattr(event, "type", "") == "notice" else None
            )
        )
        for index in range(len(plan)):
            await session.prompt(f"turn {index + 1}")
    return router, notices


@pytest.mark.asyncio
async def test_a_slow_host_is_retired_and_the_next_request_names_it_in_ignore(
    offline_world: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fast host first (the proof a faster route exists), then three slow turns.

    Turn 1 is served by FastHost and pins it; the router then moves the
    conversation to SlowHost for turns 2-4 (as OpenRouter's own balancing did in
    the measured run). On the third slow turn the host is retired: request 5 must
    carry ``provider.ignore == ["SlowHost"]`` and NO ``order`` for it, the router
    must serve FastHost, and the user must have been told why.
    """
    router, notices = await _converse(
        offline_world, ["FastHost", "SlowHost", "SlowHost", "SlowHost", "SlowHost"], monkeypatch
    )

    # Request 1: nothing is known yet, so the wire carries no `provider` object.
    assert router.sent[0] is None
    # Requests 2-4 are pinned to the host that just served (the cache affinity).
    assert router.sent[1] == {"order": ["FastHost"]}
    assert router.sent[2] == {"order": ["SlowHost"]}
    assert router.sent[3] == {"order": ["SlowHost"]}
    # Request 5, after the third slow turn: the retired host is IGNORED, and the
    # pin it held is gone (order is absent because the host just left).
    assert router.sent[4] == {"ignore": ["SlowHost"]}, router.sent
    assert router.served == ["FastHost", "SlowHost", "SlowHost", "SlowHost", "FastHost"]

    speed = [text for text in notices if "SlowHost" in text]
    assert len(speed) == 1, notices
    assert "20 tok/s" in speed[0] and "floor 25" in speed[0] and "cold prompt cache" in speed[0]
    assert offline_world["attempts"] == [], "something tried to leave the process"


@pytest.mark.asyncio
async def test_with_the_floor_at_zero_the_same_conversation_never_leaves_the_slow_host(
    offline_world: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The base behaviour, selected by the documented off switch (the control)."""
    offline_world["write_settings"]({"providers": {"openrouter": {"retire_below_tps": 0}}})
    router, notices = await _converse(
        offline_world, ["FastHost", "SlowHost", "SlowHost", "SlowHost", "SlowHost"], monkeypatch
    )

    assert all(sent is None or "ignore" not in sent for sent in router.sent), router.sent
    assert router.served == ["FastHost", "SlowHost", "SlowHost", "SlowHost", "SlowHost"]
    assert [text for text in notices if "SlowHost" in text] == []
    assert offline_world["attempts"] == []


@pytest.mark.asyncio
async def test_a_conversation_that_only_ever_saw_the_slow_host_is_never_ignored(
    offline_world: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last viable host: no measured-fast alternative, so the pin is released
    (OpenRouter's own routing decides) and `ignore` is NEVER sent."""
    router, notices = await _converse(offline_world, ["SlowHost"] * 5, monkeypatch)

    assert all(sent is None or "ignore" not in sent for sent in router.sent), router.sent
    # SlowHost serves turn 1 as well (decode speed does not depend on being
    # pinned), so its three strikes land on turns 1-3: requests 2 and 3 are still
    # pinned to it and the release applies from request 4 on.
    assert router.sent[:3] == [None, {"order": ["SlowHost"]}, {"order": ["SlowHost"]}]
    assert router.sent[3] is None and router.sent[4] is None
    release = [text for text in notices if "SlowHost" in text]
    assert len(release) == 1 and "no longer pinning" in release[0], notices
    assert offline_world["attempts"] == []
