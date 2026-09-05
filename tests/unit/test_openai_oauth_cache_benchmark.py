"""Cache diagnostics must measure the real wire without touching live state."""

import argparse
import copy
import io
import json
import os
import sqlite3
import time

import httpx
import pytest

from local_operator.harness.types import ChatRequest, Message, ModelSpec
from local_operator.providers.auth_store import OAuthAccess
from local_operator.providers.clients import OpenAICompatClient
from scripts import bench_openai_oauth_cache as bench


def test_openai_denominator_counts_cached_input_once():
    result = bench.ScenarioResult("reproduction", [bench.TurnUsage(1000, 1000)])
    assert result.denom == 1000
    assert result.cache_rate == 1.0
    result.turns.append(bench.TurnUsage(3000, 1000))
    assert result.cache_rate == 0.5  # weighted by input, not mean of per-call rates


def test_write_inclusive_input_estimate_is_not_a_subscription_charge():
    turn = bench.TurnUsage(1000, 300, 200)
    assert turn.input_price_equivalent_tokens == 780
    assert bench.ScenarioResult("empty").cache_rate == 0


def test_credential_read_is_unmodified_and_skips_expired_rows(tmp_path):
    path = tmp_path / "auth.db"
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE auth_credentials (id INTEGER, provider TEXT, "
            "credential_type TEXT, disabled_cause TEXT, data TEXT)"
        )
        for index, expires in enumerate([0, (time.time() + 3600) * 1000]):
            db.execute(
                "INSERT INTO auth_credentials VALUES (?, 'openai', 'oauth', NULL, ?)",
                (
                    index,
                    json.dumps(
                        {
                            "access": "synthetic-access",
                            "org_id": "synthetic-account",
                            "expires": expires,
                        }
                    ),
                ),
            )
    before = path.read_bytes()
    access = bench._oauth_access(path)
    assert access.credential_id == 1
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("auth.db-*"))


def test_missing_auth_is_failure_and_does_not_create_database(tmp_path, capsys):
    path = tmp_path / "missing.db"
    assert bench.main(["--live", "--auth-db", str(path)]) == 1
    assert not path.exists()
    assert "Benchmark failed" in capsys.readouterr().err


def test_live_requires_explicit_opt_in_and_budget_is_bounded():
    for args in [
        [],
        ["--live", "--turns", "9"],
        ["--live", "--gap", "0"],
        ["--live", "--gap", "nan"],
    ]:
        with pytest.raises(SystemExit) as error:
            bench.main(args)
        assert error.value.code == 2


def test_main_isolates_home_and_config_restores_environment(tmp_path, monkeypatch):
    home = tmp_path / "operator-home"
    config = tmp_path / "operator-config"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config))
    observed = []

    async def run(args, access, output):
        from pathlib import Path

        isolated = Path.home()
        observed.append(isolated)
        assert isolated != home
        assert os.environ["LOCAL_OPERATOR_CONFIG_DIR"] == str(isolated / ".local-operator")
        (isolated / "synthetic-artifact").write_text("safe")
        assert access is None
        return 0

    monkeypatch.setattr(bench, "_run", run)
    assert bench.main(["--dry-run"]) == 0
    assert not observed[0].exists()
    assert os.environ["HOME"] == str(home)
    assert os.environ["LOCAL_OPERATOR_CONFIG_DIR"] == str(config)
    assert not home.exists() and not config.exists()


def _sse(usage=None, output=None):
    usage = (
        usage
        if usage is not None
        else {
            "input_tokens": 1000,
            "input_tokens_details": {"cached_tokens": 700, "cache_write_tokens": 100},
            "output_tokens": 5,
        }
    )
    output = (
        output
        if output is not None
        else [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "actual answer"}],
            }
        ]
    )
    events = [
        {"type": "response.created", "response": {"id": "resp_synthetic"}},
        {"type": "response.output_text.delta", "delta": "actual answer"},
        {
            "type": "response.completed",
            "response": {"model": "gpt-6-astra", "output": output, "usage": usage},
        },
    ]
    return "".join("data: " + json.dumps(event) + "\n\n" for event in events).encode()


@pytest.mark.asyncio
async def test_observer_preserves_real_body_usage_and_actual_native_reply():
    wire = []

    def respond(request):
        wire.append(request)
        return httpx.Response(200, stream=httpx.ByteStream(_sse()))

    capture = bench._CaptureTransport(httpx.MockTransport(respond))
    record = capture.record
    async with httpx.AsyncClient(transport=capture) as http:
        client = OpenAICompatClient("https://api.openai.com/v1", http_client=http)
        request = ChatRequest(
            model=ModelSpec(provider="openai", model_id="gpt-6-astra", supports_prompt_cache=True),
            system_blocks=["stable"],
            messages=[Message.user("synthetic")],
            prompt_cache_key="synthetic-lineage",
        )
        access = OAuthAccess("synthetic-access", 1, org_id="synthetic-account")
        usage, assistant = await bench._stream_turn(client, request, access, record)
    assert usage == bench.TurnUsage(1000, 700, 100, 5)
    assert assistant.text == "actual answer"
    assert assistant.provider_payload is not None
    assert record["raw_usage"]["input_tokens_details"] == {
        "cached_tokens": 700,
        "cache_write_tokens": 100,
    }
    assert record["requested_model"] == record["returned_model"] == "gpt-6-astra"
    assert record["total_s"] >= record["ttft_s"] >= 0
    assert json.loads(wire[0].content)["instructions"] == "stable"
    assert "prompt_cache_options" not in json.loads(wire[0].content)
    assert "synthetic-access" not in json.dumps(record)
    assert "synthetic-account" not in json.dumps(record)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    [
        {},
        {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 101}},
        {"input_tokens": 100, "input_tokens_details": {"cached_tokens": -1}},
    ],
)
async def test_missing_or_invalid_usage_is_not_a_successful_zero(usage):
    capture = bench._CaptureTransport(
        httpx.MockTransport(lambda _: httpx.Response(200, stream=httpx.ByteStream(_sse(usage))))
    )
    async with httpx.AsyncClient(transport=capture) as http:
        client = OpenAICompatClient("https://api.openai.com/v1", http_client=http)
        request = ChatRequest(
            model=ModelSpec(provider="openai", model_id="gpt-6-astra"),
            messages=[Message.user("synthetic")],
        )
        with pytest.raises(ValueError):
            await bench._stream_turn(
                client, request, OAuthAccess("synthetic", 1, org_id="account"), capture.record
            )


@pytest.mark.parametrize(
    "field", ["input_tokens", "output_tokens", "cached_tokens", "cache_write_tokens"]
)
@pytest.mark.parametrize("value", [None, True, False, -1, 0.0, "0", [], {}])
def test_cost_critical_raw_fields_require_exact_nonnegative_integers(field, value):
    raw = {
        "input_tokens": 1000,
        "output_tokens": 5,
        "input_tokens_details": {"cached_tokens": 700, "cache_write_tokens": 100},
    }
    target = (
        raw["input_tokens_details"] if field in ("cached_tokens", "cache_write_tokens") else raw
    )
    target[field] = value
    with pytest.raises(ValueError, match=field):
        bench._validated_usage(raw)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field",
    [
        "input_tokens_details",
        "input_tokens",
        "output_tokens",
        "cached_tokens",
        "cache_write_tokens",
    ],
)
async def test_missing_raw_fields_preserve_error_not_fabricated_measurements(field):
    raw = {
        "input_tokens": 1000,
        "output_tokens": 5,
        "input_tokens_details": {"cached_tokens": 700, "cache_write_tokens": 100},
    }
    target = (
        raw["input_tokens_details"] if field in ("cached_tokens", "cache_write_tokens") else raw
    )
    target.pop(field)
    expected = copy.deepcopy(raw)
    capture = bench._CaptureTransport(
        httpx.MockTransport(lambda _: httpx.Response(200, stream=httpx.ByteStream(_sse(raw))))
    )
    async with httpx.AsyncClient(transport=capture) as http:
        client = OpenAICompatClient("https://api.openai.com/v1", http_client=http)
        request = ChatRequest(
            model=ModelSpec(provider="openai", model_id="gpt-6-astra"),
            messages=[Message.user("synthetic")],
        )
        with pytest.raises(ValueError):
            await bench._stream_turn(
                client, request, OAuthAccess("synthetic", 1, org_id="account"), capture.record
            )
    assert capture.record["raw_usage"] == expected
    assert "normalized_usage" not in capture.record
    assert "raw_usage_validated" not in capture.record
    assert "public_list_input_equivalent_tokens_estimate" not in capture.record


@pytest.mark.parametrize("details", [None, True, [], "malformed"])
def test_malformed_raw_details_are_not_an_empty_known_bucket(details):
    with pytest.raises(ValueError):
        bench._validated_usage(
            {"input_tokens": 1000, "output_tokens": 5, "input_tokens_details": details}
        )


def test_dry_run_uses_real_builder_without_auth_or_network(tmp_path):
    target = tmp_path / "dry.jsonl"
    assert (
        bench.main(
            [
                "--dry-run",
                "--scenario",
                "long_session",
                "--model",
                "gpt-6-astra",
                "--output",
                str(target),
            ]
        )
        == 0
    )
    row = json.loads(target.read_text())
    assert row["requested_model"] == "gpt-6-astra"
    assert row["endpoint"].endswith("/codex/responses")
    assert row["body_shape"]["tool_choice"] == "auto"
    assert row["body_shape"]["store"] is False
    assert "raw_usage" not in row  # a dry run never claims a cache hit
    before = target.read_bytes()
    assert bench.main(["--dry-run", "--output", str(target)]) == 1
    assert target.read_bytes() == before


@pytest.mark.asyncio
async def test_empty_exception_message_still_stops_the_run(monkeypatch):
    async def timeout(*args):
        raise TimeoutError()

    monkeypatch.setattr(bench, "_stream_turn", timeout)
    monkeypatch.setattr(bench, "_build_prefix", lambda *_: ([], ["synthetic prefix"]))
    args = argparse.Namespace(
        seed="fixture",
        model="gpt-6-astra",
        prefix_rows=0,
        turns=3,
        dry_run=False,
        gap=4,
        scenario="long_session",
    )
    output = io.StringIO()
    assert await bench._run(args, OAuthAccess("synthetic", 1, org_id="account"), output) == 1
    assert len(output.getvalue().splitlines()) == 1
    assert json.loads(output.getvalue())["error"] == "TimeoutError"


@pytest.mark.asyncio
async def test_http_failure_is_recorded_redacted_and_stops_budget(monkeypatch):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(400, json={"error": {"message": "refused synthetic-secret"}})

    cls = bench._CaptureTransport
    monkeypatch.setattr(bench, "_CaptureTransport", lambda: cls(httpx.MockTransport(respond)))
    monkeypatch.setattr(bench, "_build_prefix", lambda *_: ([], ["synthetic prefix"]))
    args = argparse.Namespace(
        seed="fixture",
        model="gpt-6-astra",
        prefix_rows=0,
        turns=3,
        dry_run=False,
        gap=4,
        scenario="long_session",
    )
    output = io.StringIO()
    result = await bench._run(
        args, OAuthAccess("synthetic-secret", 1, org_id="fixture-account"), output
    )
    assert result == 1
    assert len(calls) == 1
    assert "synthetic-secret" not in output.getvalue()
    assert json.loads(output.getvalue())["http_status"] == 400
