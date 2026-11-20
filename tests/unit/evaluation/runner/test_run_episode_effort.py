"""``--reasoning-effort`` on the episode script.

The published numbers this harness is compared against are produced at maximum
effort, so a run meant to be read beside them has to be able to say which level
it used — and to fail loudly when it cannot use the one it was given.

That second half is the point of the validation. The provider clients DROP an
effort a model does not list (``providers/clients._reasoning_effort``), which is
the right call mid-episode where the alternative is losing the turn, but here it
would mean a paid run quietly executing at the default while the operator
believed otherwise.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from local_operator.model.configure import build_model_spec
from scripts import run_episode

# A real route with a real ladder: validating against a fabricated spec would
# assert our own fixture rather than the model table the script actually reads.
_PROVIDER, _MODEL = "anthropic", "claude-opus-5"


def _client_kwargs(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "auth_store": object(),
        "settings": {},
        "provider": _PROVIDER,
        "model": _MODEL,
        "route": run_episode._route_identity(_PROVIDER, _MODEL),
        "artifact_root": Path("/nonexistent"),
        "episode_id": "ep-effort",
        "task_id": "task",
        "keep_recent_frames": 3,
    }
    base.update(overrides)
    return base


def test_an_invalid_effort_fails_before_any_provider_or_adapter_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The refusal must happen before the client is constructed, not inside it.

    A VM is allocated and billed downstream of this call, so an effort the
    model cannot accept has to be caught while the run is still free.
    """
    called = False

    def _never(**_kwargs: Any) -> object:
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(
        "local_operator.evaluation.runner.provider_client.create_provider_model_client",
        _never,
    )

    with pytest.raises(ValueError) as error:
        run_episode._model_client(**_client_kwargs(reasoning_effort="turbo"))

    assert called is False, "the model client was built despite an unusable effort"
    # The message has to name the ladder: "invalid" alone leaves the operator
    # guessing at a per-model vocabulary.
    message = str(error.value)
    assert "turbo" in message
    for level in build_model_spec(_PROVIDER, _MODEL).reasoning_efforts:
        assert level in message


def test_a_valid_effort_reaches_the_model_spec_the_request_is_built_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The level must land on the spec the client bills against.

    ``_reasoning_effort`` re-reads ``request.model.reasoning_effort`` at send
    time, so this field IS the wire behaviour; a flag that parsed and went
    nowhere would look identical from the outside.
    """
    captured: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "local_operator.evaluation.runner.provider_client.create_provider_model_client",
        _capture,
    )

    ladder = build_model_spec(_PROVIDER, _MODEL).reasoning_efforts
    chosen = ladder[-1]
    run_episode._model_client(**_client_kwargs(reasoning_effort=chosen))

    spec = captured["model_spec"]
    assert spec.reasoning_effort == chosen
    # And the client would actually send it: same membership check the wire
    # clients apply, rather than a second opinion about the ladder.
    assert spec.reasoning_effort in spec.reasoning_efforts


def test_omitting_the_flag_leaves_the_spec_default_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No flag means no opinion: the model table's default stands."""
    captured: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "local_operator.evaluation.runner.provider_client.create_provider_model_client",
        _capture,
    )

    run_episode._model_client(**_client_kwargs())

    expected = build_model_spec(_PROVIDER, _MODEL).reasoning_effort
    assert captured["model_spec"].reasoning_effort == expected


@pytest.mark.asyncio
async def test_session_arm_refuses_invalid_effort_before_source_or_resource_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The effective Session route can differ from the reply/manifest route.
    # This route has no such level, even though the manifest route accepts it.
    async def unexpected(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("invalid effort started the managed owner")

    monkeypatch.setattr(run_episode, "_run_managed_frontend", unexpected)
    args = run_episode.build_parser().parse_args(
        [
            "--selector",
            str(tmp_path / "does-not-exist.json"),
            "--task-id",
            "t",
            "--route",
            f"{_PROVIDER}/{_MODEL}",
            "--run-root",
            "r",
            "--engagement",
            "session",
            "--session-route",
            "test/mock",
            "--reasoning-effort",
            "max",
        ]
    )
    assert await run_episode.run(args) == run_episode.EXIT_PREFLIGHT
    assert "test/mock does not accept reasoning effort 'max'" in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested", "auto", "expected"),
    [("high", True, "high"), ("max", True, "max"), (None, True, "low"), (None, False, "medium")],
)
async def test_first_session_turn_serializes_effective_effort(
    requested: str | None,
    auto: bool,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator import sdk
    from local_operator.config import ConfigManager
    from local_operator.evaluation import session_arm, source_pin
    from local_operator.harness.types import ChatRequest
    from local_operator.providers import failover
    from local_operator.providers.clients import AnthropicClient
    from local_operator.session.spec import SessionRoots

    home = tmp_path / "home"
    config_dir = home / ".local-operator"
    config_dir.mkdir(parents=True)
    for key in [key for key in os.environ if key.startswith(("CMUX_", "LOP_"))]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(home / "local-operator-home"))
    monkeypatch.setenv("LOP_RUN_SCRATCH_ROOT", str(tmp_path))
    config = ConfigManager(config_dir=config_dir)
    config.set_config_value("model_effort", "medium")
    config.set_config_value("effort", {"auto": auto})
    monkeypatch.setattr(source_pin, "verify_core_source", lambda _: None)
    monkeypatch.setattr(SessionRoots, "assert_durable", lambda self: None)

    turn_efforts: list[str] = []
    events = (
        {"type": "message_start", "message": {"id": "fixture", "usage": {"input_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "OK"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    )
    response = "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()

    async def open_and_prompt(*, session_spec: Any, roots: SessionRoots, **_kwargs: Any) -> Any:
        # Replace the adapter/guest, but open and prompt the same typed Session.
        async with sdk.open_session(session_spec, roots=roots, mode="own") as session:
            # A known low auto-effort prompt; explicit levels must beat it.
            await session.prompt("show status")
        return session_arm.SessionArmOutcome("completed", "effort-wire", record_root=None)

    monkeypatch.setattr(session_arm, "run_session_episode", open_and_prompt)

    def bind_wire(turn: bool) -> httpx.MockTransport:
        def receive(request: httpx.Request) -> httpx.Response:
            if turn:
                # Never retain prompts, tool schemas, headers or cache keys.
                turn_efforts.append(json.loads(request.content)["output_config"]["effort"])
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=response
            )

        return httpx.MockTransport(receive)

    async def synthetic_failover(
        request: ChatRequest, *_args: Any, **_kwargs: Any
    ) -> AsyncIterator[Any]:
        # Replace credential selection with a dummy key and an in-process HTTP
        # transport. The Anthropic request builder and serializer are real.
        async with httpx.AsyncClient(transport=bind_wire(request.purpose == "turn")) as http:
            async for event in AnthropicClient(http_client=http).stream(request, "fixture-key"):
                yield event

    monkeypatch.setattr(failover, "stream_with_failover", synthetic_failover)
    argv = [
        "--selector",
        "s.json",
        "--task-id",
        "t",
        "--route",
        f"{_PROVIDER}/{_MODEL}",
        "--run-root",
        str(tmp_path),
        "--engagement",
        "session",
    ]
    if requested is not None:
        argv.extend(["--reasoning-effort", requested])
    result = await run_episode._run_session_arm(
        args=run_episode.build_parser().parse_args(argv),
        spec=SimpleNamespace(secret_refs=(), episode_id="effort-wire"),
        config=None,
        selector=SimpleNamespace(core_source=object()),
        resolver=SimpleNamespace(resolve=lambda _: ()),
        route=(_PROVIDER, _MODEL),
    )

    assert result == run_episode.EXIT_OK
    assert turn_efforts and turn_efforts[0] == expected


def test_the_parser_defaults_the_flag_to_none() -> None:
    """Default None is what "leave the spec alone" is spelled as."""
    args = run_episode.build_parser().parse_args(
        ["--selector", "s.json", "--task-id", "t", "--route", "test/model", "--run-root", "r"]
    )
    assert args.reasoning_effort is None

    args = run_episode.build_parser().parse_args(
        [
            "--selector",
            "s.json",
            "--task-id",
            "t",
            "--route",
            "test/model",
            "--run-root",
            "r",
            "--reasoning-effort",
            "max",
        ]
    )
    assert args.reasoning_effort == "max"


def test_the_chosen_effort_lands_in_the_episode_metadata(tmp_path) -> None:
    """A score is not comparable across effort levels, so the bundle records it.

    Asserted on ``build_spec``'s output rather than a completed run because
    this is the seam that carries it: the manifest metadata is what a reader
    has months later, and "which flags did I pass" is not.
    """
    # build_spec hashes the task's real bytes for the manifest, so the
    # workspace has to hold one.
    (tmp_path / "tasks").mkdir()
    (tmp_path / "tasks" / "task.py").write_text("TASK = {}\n")
    spec = run_episode.build_spec(
        episode_id="ep-effort",
        selector=run_episode.AdapterSelector.model_validate(
            {
                "schema_version": "1.7",
                "adapter_id": "tiny",
                "distribution": "tiny-adapter",
                "version": "1.0",
                "entry_point": "tiny:create",
                "package_digest": "a" * 64,
                "release_digest": "b" * 64,
                "python_executable": str(tmp_path / "python"),
                "workspace": str(tmp_path),
                "workspace_digest": "c" * 64,
                "route_capability": "computer",
            }
        ),
        task_id="task",
        route=run_episode._route_identity(_PROVIDER, _MODEL),
        benchmark_id="bench",
        benchmark_release="release",
        secret_refs=(),
        infra_values=(),
        max_usd_micros=1,
        max_wall_ms=1000,
        max_steps=1,
        metadata={"route": f"{_PROVIDER}/{_MODEL}", "reasoning_effort": "max"},
    )

    assert spec.metadata["reasoning_effort"] == "max"
    # Beside the route, not instead of it: both are needed to read the score.
    assert spec.metadata["route"] == f"{_PROVIDER}/{_MODEL}"
