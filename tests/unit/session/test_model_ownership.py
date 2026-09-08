"""Model ownership is proved at the request boundary, not just in the band.

All providers are local recording streams. Shared config is still live for
other settings, but its provider/model pair is a birth default, not a lease on
conversations that happened to start from it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.config import ConfigManager
from local_operator.config_watch import ConfigWatcher
from local_operator.harness.types import ModelSpec
from local_operator.session.model_selection import read_model_selection
from local_operator.session.remote import RemoteSession
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.session_factory import resolve_hosting_model_with_source
from tests.e2e.harness import ScriptedStream, text_turn


@pytest.mark.asyncio
async def test_child_inherits_launching_conversation_not_new_global_default(tmp_path, monkeypatch):
    from local_operator.harness.subagent import _build_child_session, _dispose_child

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    config = ConfigManager(tmp_path)
    config.update_config({"hosting": "test", "model_name": "global-b"})
    parent, stream = session(tmp_path / "sessions" / "parent")
    child = await _build_child_session(
        label="child", prompt="inherit", parent_session=parent, model_spec=None, job_id="child-1"
    )
    try:
        await child.prompt("which model")
        assert stream.requests[-1].model.model_id == A.model_id
    finally:
        await _dispose_child(child)
        await parent.dispose()


@pytest.mark.asyncio
async def test_unusable_legacy_selection_recovers_visibly_once(tmp_path):
    directory = tmp_path / "sessions" / "legacy-invalid"
    transcript = Transcript(directory)
    await transcript.append_custom("selected_model", {"selector": "missing-provider/model"})
    owner, stream = session(directory)
    notices = []
    owner.subscribe(lambda event: notices.append(event) if event.type == "notice" else None)
    await owner.prompt("recover")
    assert stream.requests[-1].model.model_id == A.model_id
    assert any("Saved model information is incomplete" in event.text for event in notices)
    await owner.dispose()
    resumed, _ = session(directory, model=B)
    try:
        assert not resumed._model_migration_notice
        assert resumed.model.model_id == A.model_id
    finally:
        await resumed.dispose()


A = ModelSpec(provider="test", model_id="conversation-a", context_window=100_000)
B = A.model_copy(update={"model_id": "default-b"})


def session(directory: Path, *, model=A, source="config", defer=False):
    stream = ScriptedStream([text_turn("ok") for _ in range(8)])
    owner = Session(
        model=model,
        model_source=source,
        stream_fn=stream,
        tools=[],
        transcript=Transcript(directory, defer_materialise=defer),
        system_blocks_provider=lambda: [],
        cwd=str(directory.parent),
    )
    return owner, stream


@pytest.mark.asyncio
async def test_sibling_default_changes_never_change_subsequent_requests(tmp_path):
    config_dir = tmp_path / "config"
    config = ConfigManager(config_dir)
    config.set_config_value("hosting", "test")
    config.set_config_value("model_name", A.model_id)
    watcher = ConfigWatcher(config_dir)
    owners = [session(config_dir / "sessions" / name) for name in ("a", "b")]
    for owner, _ in owners:
        owner.add_dispose_hook(watcher.subscribe(owner._apply_config_change))
    try:
        for owner, _ in owners:
            await owner.prompt("before")
        writer = ConfigManager(config_dir)
        writer.set_config_value("model_name", B.model_id)
        watcher.poll_now()
        for owner, stream in owners:
            await owner.prompt("after")
            assert [r.model.model_id for r in stream.requests] == [A.model_id, A.model_id]
        # A deliberate selection belongs to exactly its owner.
        owners[0][0].set_model(B, explicit=True)
        for owner, _ in owners:
            await owner.prompt("explicit")
        assert owners[0][1].requests[-1].model.model_id == B.model_id
        assert owners[1][1].requests[-1].model.model_id == A.model_id
    finally:
        for owner, _ in owners:
            await owner.dispose()


@pytest.mark.asyncio
async def test_birth_selection_is_durable_only_when_work_is_admitted(tmp_path):
    directory = tmp_path / "sessions" / "birth"
    owner, stream = session(directory, defer=True)
    assert not directory.exists()
    await owner.prompt("first")
    saved = read_model_selection(directory)
    assert saved is not None and saved.authoritative and saved.selector == "test/conversation-a"
    await owner.dispose()
    resumed, requests = session(directory, model=B)
    try:
        await resumed.prompt("resume after default changed")
        assert requests.requests[-1].model.model_id == A.model_id
        assert (
            len(
                [
                    r
                    for r in resumed.transcript.entries()
                    if r.payload.get("custom_type") == "selected_model"
                ]
            )
            == 1
        )
    finally:
        await resumed.dispose()
    unused, _ = session(tmp_path / "unused", defer=True)
    await unused.dispose()
    assert not (tmp_path / "unused").exists()


@pytest.mark.asyncio
async def test_explicit_switch_survives_changed_default_and_immediate_fork(tmp_path):
    directory = tmp_path / "sessions" / "parent"
    owner, _ = session(directory)
    await owner.prompt("seed")
    owner.set_model(B, explicit=True)
    # Exercise the session's snapshot method, which must await the switch row.
    result = await owner.fork_snapshot()
    await owner.dispose()
    resumed, stream = session(directory, model=A.model_copy(update={"model_id": "new-default"}))
    try:
        await resumed.prompt("resume")
        assert stream.requests[-1].model.model_id == B.model_id
    finally:
        await resumed.dispose()
    fork_id = result["fork_id"]
    forked, fork_stream = session(tmp_path / "sessions" / fork_id, model=A)
    try:
        await forked.prompt("fork")
        assert fork_stream.requests[-1].model.model_id == B.model_id
    finally:
        await forked.dispose()


@pytest.mark.asyncio
async def test_legacy_migration_prefers_latest_primary_observation(tmp_path):
    directory = tmp_path / "sessions" / "legacy"
    transcript = Transcript(directory)
    await transcript.append_custom(
        "selected_model", {"selector": "test/old-switch", "boot": "test/old-boot"}
    )
    await transcript.append_custom(
        "frontend_state_checkpoint_v1",
        {
            "state": {
                "selected_model": B.model_dump(),
                "effective_model": A.model_dump(),
            }
        },
    )
    saved = read_model_selection(directory)
    assert saved is not None and saved.selector == "test/default-b"
    owner, stream = session(directory)
    await owner.prompt("migrate")
    assert stream.requests[-1].model.model_id == B.model_id
    await owner.dispose()
    # New authority wins even over a stale later checkpoint or old writer row.
    transcript = Transcript(directory)
    await transcript.append_custom(
        "frontend_state_checkpoint_v1", {"state": {"selected_model": A.model_dump()}}
    )
    saved = read_model_selection(directory)
    assert saved is not None and saved.selector == "test/default-b"
    assert saved.authoritative


@pytest.mark.asyncio
async def test_model_resolution_before_invalid_defaults_and_profile_changes(tmp_path):
    config = ConfigManager(tmp_path)
    directory = tmp_path / "sessions" / "saved"
    owner, _ = session(directory)
    await owner.prompt("seed")
    await owner.dispose()
    config.set_config_value("hosting", "not-a-provider")
    config.set_config_value("model_name", "")
    agent: Any = SimpleNamespace(hosting="openai", model="gpt-6")
    args = argparse.Namespace(resume="saved", hosting=None, model=None)
    assert resolve_hosting_model_with_source(agent, args, config) == ("test", A.model_id, "resume")
    args.model = "explicit-model"
    assert resolve_hosting_model_with_source(agent, args, config) == (
        "test",
        "explicit-model",
        "flag",
    )
    args.hosting = "openai"
    args.model = None
    provider, model_id, source = resolve_hosting_model_with_source(agent, args, config)
    assert provider == "openai" and model_id and source == "flag"
    # Same model id on another provider must not compare equal.
    args.model = A.model_id
    assert resolve_hosting_model_with_source(agent, args, config) == ("openai", A.model_id, "flag")
    args.model_selection_override = False
    assert resolve_hosting_model_with_source(agent, args, config) == ("test", A.model_id, "resume")


@pytest.mark.asyncio
async def test_cold_viewer_birth_and_resume_use_same_selection(tmp_path):
    config = ConfigManager(tmp_path)
    config.set_config_value("hosting", "test")
    config.set_config_value("model_name", A.model_id)

    async def takeover():
        raise AssertionError("read-only viewer must not take ownership")

    cold = await RemoteSession.cold(
        "new", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=takeover
    )
    try:
        config.set_config_value("model_name", B.model_id)
        assert cold.model.model_id == A.model_id
        assert cold._birth_model is not None
        assert cold._birth_model.model_id == A.model_id
        assert not (tmp_path / "sessions" / "new").exists()
        # A viewer attaching to a warmed (not yet materialized) owner learns
        # its primary too. Recovery must retain that observation in memory.
        from local_operator.session.frontend_state import FrontendModelSpec

        cold._install_frontend(
            cold.frontend_state.model_copy(
                update={
                    "epoch": "owner-epoch",
                    "selected_model": FrontendModelSpec(provider="test", model_id="owner-picked"),
                }
            )
        )
        assert cold._birth_model.model_id == "owner-picked"
        newer = await RemoteSession.cold(
            "newer", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=takeover
        )
        assert newer.model.model_id == B.model_id
        await newer.dispose()
    finally:
        await cold.dispose()
    owner, _ = session(tmp_path / "sessions" / "saved")
    await owner.prompt("save")
    await owner.dispose()
    config.set_config_value("hosting", "invalid")
    resumed = await RemoteSession.cold(
        "saved", config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=takeover
    )
    try:
        assert resumed.model.provider == "test"
        assert resumed.model.model_id == A.model_id
    finally:
        await resumed.dispose()
