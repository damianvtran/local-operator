"""Efficiency reports must count provider usage once and preserve wire content."""

import asyncio
import json
import os
import subprocess

from local_operator.harness.types import (
    AgentTool,
    ChatRequest,
    Message,
    ModelSpec,
    Usage,
)
from local_operator.providers.clients import AnthropicClient
from scripts import bench_cache_rate, bench_complex_tasks
from scripts.bench_live_workspace import initialize_workspace


def test_live_workspace_git_discovery_stays_under_synthetic_root(tmp_path):
    """A scratch cwd alone must not expose an enclosing repository's files.

    Exercise real Git rather than mocking init: removing the initializer makes
    both rev-parse and status escape to the ancestor, reproducing the live-trial
    leak. No provider or live worker is imported or invoked by this fixture test.
    """
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    synthetic_home = tmp_path / "home"
    synthetic_home.mkdir()
    environment["HOME"] = str(synthetic_home)
    environment["XDG_CONFIG_HOME"] = str(synthetic_home / ".config")
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    ancestor = tmp_path / "ancestor"
    ancestor.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", "--template=", str(ancestor)],
        env=environment,
        check=True,
    )
    (ancestor / "unrelated-sentinel.txt").write_text("unrelated fixture")
    workspace = ancestor / "trial" / "workspace"
    # Explicit Git roots are inherited by subprocesses as well as discovered.
    # The initializer must clear them for all subsequent tools in this worker.
    environment["GIT_DIR"] = str(ancestor / ".git")
    environment["GIT_WORK_TREE"] = str(ancestor)
    initialize_workspace(workspace, environment=environment)
    (workspace / "ledger.py").write_text("synthetic fixture")
    top_level = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], cwd=workspace, env=environment, text=True
    ).strip()
    assert top_level == str(workspace)
    status = subprocess.check_output(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=workspace,
        env=environment,
        text=True,
    )
    assert status == "?? ledger.py\n"
    assert not any(key.startswith("GIT_") for key in environment)
    for key, expected in {
        "core.fsmonitor": "false",
        "core.excludesFile": os.devnull,
        "user.email": "benchmark@example.invalid",
    }.items():
        assert (
            subprocess.check_output(
                ["git", "config", "--local", "--get", key],
                cwd=workspace,
                env=environment,
                text=True,
            ).strip()
            == expected
        )


def test_cache_denominators_follow_the_provider_counting_contract():
    assert (
        bench_cache_rate._prompt_tokens(
            "anthropic", Usage(input_tokens=100, cache_read_tokens=800, cache_write_tokens=100)
        )
        == 1000
    )
    assert (
        bench_cache_rate._prompt_tokens(
            "openrouter", Usage(input_tokens=1000, cache_read_tokens=800, cache_write_tokens=100)
        )
        == 1000
    )


def test_complex_tally_deduplicates_message_and_turn_end_and_prices_cached_usage(
    tmp_path, monkeypatch
):
    from local_operator.model import configure, registry

    calls = []

    def price(provider, info, usage):
        calls.append(usage)
        return usage.cache_read_tokens / 1000

    monkeypatch.setattr(configure, "cost_for_usage", price)
    monkeypatch.setattr(registry, "get_model_info", lambda *args: object())
    message = {
        "id": "message-1",
        "role": "assistant",
        "usage": {
            "input_tokens": 1000,
            "cache_read_tokens": 900,
            "output_tokens": 50,
            "context_tokens": 1000,
        },
    }
    path = tmp_path / "capture.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"type": kind, "message": message})
            for kind in ["message_end", "turn_end", "message_end"]
        )
    )
    tokens, cost = bench_complex_tasks.tally_cost(path)
    assert tokens == {"input": 1000, "output": 50, "max_context": 1000}
    assert cost == 0.9
    assert len(calls) == 1


def test_structural_probe_uses_real_schemas_and_hierarchy_without_claiming_hits():
    async def noop(*args):
        raise AssertionError("wire construction must never execute tools")

    async def run():
        client = AnthropicClient()
        try:
            tool = AgentTool(
                name="read", description="first description", parameters={}, execute=noop
            )
            request = ChatRequest(
                model=ModelSpec(provider="anthropic", model_id="test"),
                tools=[tool],
                system_blocks=["system"],
                messages=[Message.user("hello")],
            )
            before = bench_cache_rate._serialize_request(request, client)
            assert (
                before.index(b'"tools"') < before.index(b'"system"') < before.index(b'"messages"')
            )
            tool.description = "changed description"
            after = bench_cache_rate._serialize_request(request, client)
            assert before != after
            assert b"first description" in before
            assert b"cache_control" not in before
            assert request.messages[0].id.encode() not in before
        finally:
            await client.aclose()

    asyncio.run(run())


def test_unknown_benchmark_pricing_is_not_reported_as_free(tmp_path, monkeypatch):
    from local_operator.model import registry

    def unknown(*args):
        raise ValueError("unknown model price")

    monkeypatch.setattr(registry, "get_model_info", unknown)
    path = tmp_path / "capture.jsonl"
    path.write_text(
        json.dumps({"type": "message_end", "message": {"id": "1", "usage": {"input_tokens": 100}}})
    )
    _, cost = bench_complex_tasks.tally_cost(path)
    assert cost is None
