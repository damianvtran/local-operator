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
import socket
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from local_operator.evaluation.runner.secrets import StaticSecretResolver
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


@pytest.fixture
def isolated_session_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    from local_operator.model import configure
    from local_operator.session.spec import SessionRoots

    # resolve_model_info memoizes per provider/model and TTL bucket, not per
    # config root, so a ladder one case serves from its fixture listing would
    # otherwise be what the next case in the same process reads (QA r1 Q1: the
    # [False] case's ("high",) ladder refused [True]'s valid 'max'). Clearing on
    # both sides keeps every case independent of order and of xdist placement.
    configure.invalidate_model_info_cache()

    home = tmp_path / "home"
    config_dir = home / ".local-operator"
    config_dir.mkdir(parents=True)
    for key in [key for key in os.environ if key.startswith(("CMUX_", "LOP_"))]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("LOCAL_OPERATOR_HOME", str(home / "local-operator-home"))
    monkeypatch.setenv("LOP_RUN_SCRATCH_ROOT", str(tmp_path))
    monkeypatch.setattr(SessionRoots, "assert_durable", lambda self: None)
    yield config_dir
    configure.invalidate_model_info_cache()


def _session_preflight_args(tmp_path: Path, session_route: str, *, managed: bool = False) -> Any:
    selector = tmp_path / "selector.json"
    selector.write_text(
        json.dumps(
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
                "core_source": {"commit": "f" * 40, "module_hashes": {}},
            }
        )
    )
    argv = [
        "--selector",
        str(selector),
        "--task-id",
        "t",
        "--route",
        f"{_PROVIDER}/{_MODEL}",
        "--run-root",
        str(tmp_path / "run"),
        "--engagement",
        "session",
        "--session-route",
        session_route,
        "--reasoning-effort",
        "max",
        "--owner-backend=detached" if managed else "--in-process-host",
    ]
    args = run_episode.build_parser(internal_test_support=True).parse_args(argv)
    args._raw_argv = argv
    return args


def _budget_disclosure_line(args: Any) -> str:
    """The Session budget disclosure ``run`` prints before any owner starts.

    Why: every admitted Session run discloses its budget on stderr before the
    owner launches (PR #4316), so the effort refusal -- which can only be decided
    after owner admission, from scoped route metadata -- always follows that
    line. The record comes from ``run_episode``'s own helper so a change to the
    disclosed fields cannot silently drift from these exact-stderr pins; only
    the one-line prefix is restated, as ``run`` writes it inline.
    """
    budget = run_episode._session_budget_disclosure(
        args,
        configured_usd_micros=run_episode._usd_micros(args.max_usd, flag="--max-usd"),
        max_cycle_usd_micros=(
            run_episode._usd_micros(args.max_cycle_usd, flag="--max-cycle-usd")
            if args.max_cycle_usd is not None
            else None
        ),
    )
    return (
        "session budget preflight (null limits are not enforced): "
        f"session_budget={json.dumps(budget, sort_keys=True, allow_nan=False)}\n"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["pin", "isolation", "owner"])
async def test_session_gates_refuse_before_metadata_lookup(
    refusal: str,
    tmp_path: Path,
    isolated_session_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from local_operator.evaluation import source_pin
    from local_operator.model import configure

    def unexpected(*_args: Any, **_kwargs: Any) -> Any:
        pytest.fail("an unadmitted run reached model metadata or a store")

    def verify_pin(_: Any) -> None:
        if refusal == "pin":
            raise source_pin.SourcePinError("fixture bad source pin")

    monkeypatch.setattr(source_pin, "verify_core_source", verify_pin)
    monkeypatch.setattr(configure, "build_model_spec", unexpected)
    monkeypatch.setattr(configure, "resolve_model_info", unexpected)
    monkeypatch.setattr(configure, "_catalogue_credential", unexpected)
    monkeypatch.setattr(run_episode, "_open_store", unexpected)
    args = _session_preflight_args(tmp_path, "deepseek/deepseek-flash")
    if refusal == "isolation":
        monkeypatch.delenv("LOCAL_OPERATOR_CONFIG_DIR")
    if refusal == "owner":
        args.owned_host = True
    assert await run_episode.run(args) == run_episode.EXIT_PREFLIGHT
    expected = {
        "pin": "fixture bad source pin",
        "isolation": "requires LOCAL_OPERATOR_CONFIG_DIR",
        "owner": "this environment carries no owner identity",
    }
    assert expected[refusal] in capsys.readouterr().err
    assert not (tmp_path / "world").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("valid", [False, True])
async def test_admitted_session_effort_uses_scoped_route_before_allocation(
    valid: bool,
    tmp_path: Path,
    isolated_session_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from local_operator import owned_process
    from local_operator.evaluation import source_pin
    from local_operator.model import configure, discovery

    args = _session_preflight_args(tmp_path, "deepseek/fixture-dynamic")
    args.owned_host = True
    calls: list[str] = []

    def admit(_: Any) -> None:
        assert calls == ["pin"]
        calls.append("owner admitted")

    def credentials(provider: str, *, base: Path | None = None) -> tuple[str, bool, None]:
        assert calls == ["pin", "owner admitted"]
        assert provider == "deepseek"
        assert base == isolated_session_env.resolve()
        calls.append("scoped metadata")
        return "fixture-only", False, None

    def listing(provider: str, **kwargs: Any) -> Any:
        assert provider == "deepseek"
        assert kwargs["config_dir"] == isolated_session_env.resolve()
        assert kwargs["api_key"] == "fixture-only"
        ladder = ("high", "max") if valid else ("high",)
        row = discovery.DiscoveredModel(
            id="fixture-dynamic", name="fixture-dynamic", reasoning_efforts=ladder
        )
        return [row], "live"

    class ReachedPrepare(Exception):
        """Stop a valid preflight before any preparation, store, or generation."""

    def stop_before_prepare(*_args: Any, **_kwargs: Any) -> Any:
        raise ReachedPrepare

    monkeypatch.setattr(source_pin, "verify_core_source", lambda _: calls.append("pin"))
    monkeypatch.setattr(configure, "_catalogue_credential", credentials)
    monkeypatch.setattr(discovery, "available_models", listing)
    # Ref parsing also runs before admission; this boundary is only after effort validation.
    monkeypatch.setattr(run_episode, "_ensure_lease_outlasts_wall", stop_before_prepare)
    channel = owned_process.OwnerStatusChannel(
        tmp_path / "status.json",
        owner_id="fixture-owner",
        backend="detached",
        job_label=None,
        pid=123,
        birth="fixture-birth",
    )
    monkeypatch.setattr(run_episode, "_owned_host_status_channel", lambda _: channel)

    if valid:
        with pytest.raises(ReachedPrepare):
            await run_episode.run(args, _test_owner_admission=admit)
        status = owned_process.read_owner_status(channel.path)
        assert status is not None and status.phase == "admitted"
    else:
        assert (
            await run_episode.run(args, _test_owner_admission=admit) == run_episode.EXIT_PREFLIGHT
        )
        disclosure, refusal = capsys.readouterr().err.splitlines(keepends=True)
        # Budget disclosure first (it precedes owner admission), then the refusal.
        assert disclosure == _budget_disclosure_line(args)
        assert (
            "session route deepseek/fixture-dynamic does not accept reasoning effort 'max'"
            in refusal
        )
        assert "its ladder is high" in refusal
        status = owned_process.read_owner_status(channel.path)
        assert status is not None and status.phase == "ended"
        assert run_episode._report_owner_outcome(None, status) == run_episode.EXIT_PREFLIGHT
        assert capsys.readouterr().err == refusal
    assert calls == ["pin", "owner admitted", "scoped metadata"]
    assert not (tmp_path / "world").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("admitted", [True, False], ids=["after-admission", "admission-timeout"])
@pytest.mark.parametrize("receipt", ["matching", "missing", "wrong-pid", "wrong-owner"])
async def test_frontend_checks_terminal_receipt_when_owner_disappears(
    admitted: bool,
    receipt: str,
    tmp_path: Path,
    isolated_session_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from local_operator import owned_process
    from local_operator.evaluation import source_pin

    args = _session_preflight_args(tmp_path, "deepseek/fixture-dynamic", managed=True)
    refusal = "session route deepseek/fixture-dynamic does not accept reasoning effort 'max'"
    monkeypatch.setattr(source_pin, "verify_core_source", lambda _: None)
    monkeypatch.setattr(run_episode, "resolve_core_pin", lambda: "fixture-source-pin")
    monkeypatch.setattr(run_episode, "refuse_volatile_root", lambda *a, **kw: None)
    if not admitted:
        # The owner has started, but the frontend is descheduled until its
        # admission deadline. No status has been seen and no real wait is needed.
        ticks = iter((0.0, args.owner_startup_timeout_s))
        monkeypatch.setattr(run_episode, "time", SimpleNamespace(monotonic=lambda: next(ticks)))

    def publish(lease: owned_process.OwnedProcessLease, phase: str) -> None:
        terminal = phase == "ended"
        owned_process.write_owner_status(
            lease.paths.status_path,
            owner_id="foreign-owner" if terminal and receipt == "wrong-owner" else lease.owner_id,
            phase=phase,
            pid=lease.pid + 1 if terminal and receipt == "wrong-pid" else lease.pid,
            birth=lease.birth,
            backend=lease.backend,
            job_label=lease.job_label,
            detail=refusal if terminal else None,
            receipt={"session_effort_preflight_refused": True} if terminal else None,
        )

    def start(
        spec: owned_process.OwnedProcessSpec, *, startup_timeout_s: float
    ) -> owned_process.OwnedProcessLease:
        lease = owned_process.OwnedProcessLease(
            owner_id=spec.owner_id,
            backend=spec.backend,
            job_label=None,
            pid=123,
            birth_scheme=None,
            birth="fixture-birth",
            source_pin=spec.source_pin,
            run_root=spec.run_root,
        )
        if admitted:
            publish(lease, "admitted")
        return lease

    def observe(lease: owned_process.OwnedProcessLease) -> run_episode.JobObservation:
        status = owned_process.read_owner_status(lease.paths.status_path)
        assert (status.phase if status else None) == ("admitted" if admitted else None)
        # Seal while liveness is being read, not before the frontend's first read.
        if receipt != "missing":
            publish(lease, "ended")
        return run_episode.JobObservation(run_episode.JobState.ABSENT, None, "fixture owner exited")

    monkeypatch.setattr(owned_process, "start_owned_process", start)
    monkeypatch.setattr(run_episode, "_owner_observation", observe)
    result = await run_episode.run(args)
    captured = capsys.readouterr()
    # The frontend discloses the budget before launching the owner, so it is
    # printed whatever the owner later reports; only the relayed refusal varies.
    disclosure = _budget_disclosure_line(args)
    if receipt == "matching":
        assert result == run_episode.EXIT_PREFLIGHT
        assert captured.err == disclosure + refusal + "\n"
        assert captured.out == ""
    else:
        assert result == run_episode.EXIT_EPISODE
        assert captured.err == disclosure
        assert json.loads(captured.out)["status"] == "owner_lost_unverified"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "requested", "auto", "expected"),
    [
        ("anthropic", "high", True, "high"),
        ("anthropic", "max", True, "max"),
        ("anthropic", None, True, "low"),
        ("anthropic", None, False, "medium"),
        ("deepseek", "max", True, "max"),
        ("deepseek", "none", True, "none"),
        # The unchanged auto classifier rates "show status" tier 'lo', which maps
        # to the lowest rung: 'low' on Anthropic's ladder but 'none' on DeepSeek's
        # (none, low, high, max), serialized as thinking disabled (QA r1 Q2).
        ("deepseek", None, True, "none"),
        ("deepseek", None, False, "max"),
    ],
)
async def test_first_session_turn_serializes_effective_effort(
    provider: str,
    requested: str | None,
    auto: bool,
    expected: str,
    tmp_path: Path,
    isolated_session_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_operator import sdk
    from local_operator.config import ConfigManager
    from local_operator.evaluation import session_arm, source_pin
    from local_operator.harness.types import ChatRequest
    from local_operator.model import configure, discovery
    from local_operator.providers import failover, registry
    from local_operator.providers.clients import AnthropicClient, OpenAICompatClient
    from local_operator.session.spec import SessionRoots

    config = ConfigManager(config_dir=isolated_session_env)
    configured = "medium"
    if provider == "deepseek":
        # Auto-off must differ from both the built-in high and the auto low.
        configured = "high" if auto else "max"
    config.set_config_value("model_effort", configured)
    config.set_config_value("effort", {"auto": auto})
    monkeypatch.setattr(source_pin, "verify_core_source", lambda _: None)
    # Metadata and credential *sources* are offline; the real Session and
    # model spec still derive the effort and the native client serializes it.
    monkeypatch.setattr(configure, "_catalogue_credential", lambda *a, **kw: ("", False, None))
    monkeypatch.setattr(discovery, "available_models", lambda *a, **kw: ([], "static"))
    monkeypatch.setattr(configure, "_from_price_catalogue", lambda p, m, info, **kw: info)
    monkeypatch.setattr(registry, "provider_env_key", lambda *a, **kw: None)

    attempted_egress: list[str] = []
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex

    def refuse_egress(*_args: Any, **_kwargs: Any) -> Any:
        attempted_egress.append("unexpected network access")
        pytest.fail(attempted_egress[-1])

    def guard_connect(sock: socket.socket, address: Any) -> None:
        if sock.family != socket.AF_UNIX:
            refuse_egress()
        connect(sock, address)

    def guard_connect_ex(sock: socket.socket, address: Any) -> int:
        if sock.family != socket.AF_UNIX:
            refuse_egress()
        return connect_ex(sock, address)

    monkeypatch.setattr(socket, "getaddrinfo", refuse_egress)
    monkeypatch.setattr(socket.socket, "connect", guard_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guard_connect_ex)

    turn_fields: list[dict[str, Any]] = []
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
    response = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    if provider == "deepseek":
        response = (
            "".join(
                "data: "
                + json.dumps({"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
                + "\n\n"
                for delta, finish in (({"content": "OK"}, None), ({}, "stop"))
            )
            + "data: [DONE]\n\n"
        )

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
                body = json.loads(request.content)
                if provider == "deepseek":
                    assert str(request.url) == "https://api.deepseek.com/v1/chat/completions"
                    assert body["model"] == "deepseek-flash"
                    turn_fields.append(
                        {key: body[key] for key in ("thinking", "reasoning_effort") if key in body}
                    )
                else:
                    turn_fields.append({"effort": body["output_config"]["effort"]})
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=response.encode()
            )

        return httpx.MockTransport(receive)

    async def synthetic_failover(
        request: ChatRequest, *_args: Any, **_kwargs: Any
    ) -> AsyncIterator[Any]:
        # Substitute only credential/failover and HTTP I/O at the wire edge.
        async with httpx.AsyncClient(transport=bind_wire(request.purpose == "turn")) as http:
            if request.purpose == "turn":
                assert request.model.provider == provider
            client = (
                OpenAICompatClient("https://api.deepseek.com/v1", http_client=http)
                if provider == "deepseek"
                else AnthropicClient(http_client=http)
            )
            async for event in client.stream(request, "fixture-key"):
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
    if provider == "deepseek":
        argv.extend(["--session-route", "deepseek/deepseek-flash"])
    if requested is not None:
        argv.extend(["--reasoning-effort", requested])
    result = await run_episode._run_session_arm(
        args=run_episode.build_parser().parse_args(argv),
        spec=SimpleNamespace(secret_refs=(), episode_id="effort-wire"),
        config=None,
        selector=SimpleNamespace(core_source=object()),
        resolver=StaticSecretResolver({}),
        route=(_PROVIDER, _MODEL),
    )

    assert result == run_episode.EXIT_OK
    assert not attempted_egress
    if provider == "anthropic":
        expected_fields: dict[str, Any] = {"effort": expected}
    elif expected == "none":
        expected_fields = {"thinking": {"type": "disabled"}}
    else:
        expected_fields = {"thinking": {"type": "enabled"}, "reasoning_effort": expected}
    assert turn_fields and turn_fields[0] == expected_fields


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
