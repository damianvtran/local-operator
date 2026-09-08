"""Assembled factory/owner/provider routing with no paid or external calls.

Only the network transport is a local recorder: the real composition root,
config watcher, SessionStreamFn, agent loop and journal stay in the path.
This catches label-only fixes and clients configured before saved identity was
resolved, neither of which a Session facade assertion can observe.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pytest

from local_operator.agents import AgentRegistry
from local_operator.config import ConfigManager
from local_operator.config_watch import process_watcher
from local_operator.credentials import CredentialManager
from local_operator.providers.clients import MockClient
from local_operator.session_factory import create_session

pytestmark = pytest.mark.e2e


@pytest.mark.asyncio
async def test_persisted_server_adapters_do_not_treat_defaults_as_explicit_flags(
    headless_tui_env: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from local_operator.env import get_env_config
    from local_operator.server.utils.operator import create_operator
    from tests.unit.test_session_factory import _agent_fields

    config = ConfigManager(headless_tui_env)
    config.update_config({"hosting": "test", "model_name": "server-a"})
    registry = AgentRegistry(headless_tui_env)
    agent = registry.create_agent(
        _agent_fields("stable-server").model_copy(update={"hosting": None, "model": None})
    )
    credentials = CredentialManager(headless_tui_env)
    calls = []
    original_stream = MockClient.stream

    async def recording_stream(self, request, api_key, oauth_access=None):
        calls.append((request.model.provider, request.model.model_id))
        async for event in original_stream(self, request, api_key, oauth_access):
            yield event

    monkeypatch.setattr(MockClient, "stream", recording_stream)
    for text in ("first persisted request", "second persisted request"):
        operator = create_operator(
            "",
            "",
            credentials,
            config,
            registry,
            get_env_config(),
            current_agent=agent,
            persist_conversation=True,
        )
        await operator.handle_user_input(text)
        config.update_config({"hosting": "invalid-default", "model_name": "server-b"})
    assert calls == [("test", "server-a"), ("test", "server-a")]
    print("persisted server provider requests:", calls)


@pytest.mark.asyncio
async def test_factory_routes_existing_and_resumed_conversations_to_saved_model(
    headless_tui_env: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config_dir = headless_tui_env
    config = ConfigManager(config_dir)
    config.set_config_value("hosting", "test")
    config.set_config_value("model_name", "birth-a")
    calls = []
    original_stream = MockClient.stream

    async def recording_stream(self, request, api_key, oauth_access=None):
        calls.append((request.model.provider, request.model.model_id))
        async for event in original_stream(self, request, api_key, oauth_access):
            yield event

    monkeypatch.setattr(MockClient, "stream", recording_stream)

    async def build(resume=None, **extra):
        return await create_session(
            argparse.Namespace(
                hosting=extra.get("hosting"),
                model=extra.get("model"),
                model_selection_override=extra.get("override", True),
                agent_name=None,
                agent_id=None,
                yolo=True,
                train=False,
                resume=resume,
            ),
            ConfigManager(config_dir),
            CredentialManager(config_dir),
            AgentRegistry(config_dir),
            has_ui=False,
            cwd=str(config_dir),
        )

    first = await build()
    sibling = await build()
    session_id = first.session_id
    try:
        await first.prompt("before edit")
        edited = subprocess.run(
            [
                sys.executable,
                "-m",
                "local_operator.cli",
                "config",
                "edit",
                "model_name",
                "default-b",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "Successfully updated model_name" in edited.stdout
        process_watcher(config_dir).poll_now()
        await first.prompt("after edit")
        await sibling.prompt("sibling after edit")
        assert calls == [("test", "birth-a")] * 3
        newer = await build()
        try:
            await newer.prompt("new conversation")
            assert calls[-1] == ("test", "default-b")
        finally:
            await newer.dispose()
    finally:
        await first.dispose()
        await sibling.dispose()

    config = ConfigManager(config_dir)
    config.set_config_value("hosting", "invalid-default-provider")
    resumed = await build(session_id)
    try:
        await resumed.prompt("resume with invalid global default")
        assert calls[-1] == ("test", "birth-a")
    finally:
        await resumed.dispose()
    overridden = await build(session_id, model="explicit-c")
    try:
        await overridden.prompt("deliberate resume override")
        assert calls[-1] == ("test", "explicit-c")
    finally:
        await overridden.dispose()
    # Detached owners and server adapters carry already-resolved argument
    # pairs, not new user intent. The journal must win over those stale seeds.
    synthetic = await build(session_id, hosting="test", model="stale-seed", override=False)
    try:
        await synthetic.prompt("resume from synthesized bootstrap")
        assert calls[-1] == ("test", "explicit-c")
    finally:
        await synthetic.dispose()
    print("provider-boundary requests:", calls)


@pytest.mark.asyncio
async def test_cold_viewer_carries_birth_model_into_real_detached_owner(
    headless_tui_env: Path,
):
    import asyncio

    from local_operator.session.remote import RemoteSession
    from tests.e2e.watchdog import bounded

    config = ConfigManager(headless_tui_env)
    config.set_config_value("hosting", "test")
    config.set_config_value("model_name", "cold-birth-a")

    async def never_take_over():
        raise AssertionError("viewer must remain a viewer")

    viewer = await RemoteSession.cold(
        "modelbirth0001",
        config_dir=headless_tui_env,
        cwd=str(headless_tui_env),
        takeover_factory=never_take_over,
    )
    seen = []
    ended = asyncio.Event()

    def observe(event):
        if event.type == "provider_turn_start":
            seen.append((event.provider, event.model_id))
        if event.type == "agent_end":
            ended.set()

    unsubscribe = viewer.subscribe(observe)
    try:
        # No owner exists yet. A sibling changes defaults between painting the
        # initial band and engaging the detached runtime on the first request.
        config.set_config_value("model_name", "cold-default-b")
        with bounded(60, "cold model admission and provider request"):
            await viewer.prompt("first detached request")
            await ended.wait()
        assert seen == [("test", "cold-birth-a")]
        print("detached provider-turn-start:", seen)
    finally:
        unsubscribe()
        if viewer._client is not None:
            await viewer._client.request_stop()
        await viewer.dispose()
