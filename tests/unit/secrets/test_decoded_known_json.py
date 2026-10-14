"""Known-only scalar coverage without replacing the original shape judgement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    ToolContext,
)
from local_operator.redaction_shapes import (
    REDACTION_MARKER,
    scrub_known_json_string_values,
    scrub_secrets_with_hits,
)
from local_operator.secrets.store import SecretStore
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools.builtin import (
    build_read_tool,
    build_write_tool,
    execute_read_variable,
)
from local_operator.tools.secret_tool import execute_secret
from local_operator.variables import VariableStore


class ScalarProvider:
    """Script real tools and retain the requests that actually reach the provider."""

    def __init__(self, calls: list[tuple[str, dict[str, Any]]]):
        self.calls = calls
        self.requests: list[Any] = []

    def __call__(self, request: Any, signal: Any = None):
        index = len(self.requests)
        self.requests.append(request)

        async def generate():
            if index < len(self.calls):
                name, arguments = self.calls[index]
                yield StreamToolCallDelta(
                    index=0,
                    id=f"synthetic-{index}",
                    name=name,
                    argument_delta=json.dumps(arguments),
                )
                yield StreamEndEvent(stop_reason="toolUse")
            else:
                yield StreamTextDelta(delta="done")
                yield StreamEndEvent(stop_reason="stop")

        return generate()


def session_at(path: Path, provider: ScalarProvider) -> Session:
    return Session(
        model=ModelSpec(provider="test", model_id="synthetic", context_window=100_000),
        stream_fn=provider,
        tools=[build_read_tool(), build_write_tool()],
        transcript=Transcript(path / "session"),
        system_blocks_provider=lambda *_: [],
        yolo=True,
        cwd=str(path),
        variables=VariableStore(cwd=str(path)),
    )


@pytest.mark.parametrize("value", ["a\nb", "a\tb", "a\rb", 'a"b', "a\\b", "秘", "Ж", "अ"])
@pytest.mark.parametrize("case", ["lower", "upper"])
def test_known_short_json_scalars_are_decoded_before_reserializing(value, case):
    spelling = json.dumps(value)[1:-1]
    assert len(spelling) < 12
    if case == "upper":
        # Uppercase hexadecimal escape digits, not ordinary scalar characters.
        import re

        spelling = re.sub(r"\\u([0-9a-f]{4})", lambda m: "\\u" + m[1].upper(), spelling)
    text = '{"payload": "' + spelling + '", "near": "ordinary", "n": 3}'
    store = VariableStore(cwd=".")
    store.store_credential("SYNTHETIC_VALUE", value)
    before, _ = scrub_secrets_with_hits(text, store.redaction_values())
    assert before == text, "this must discriminate the preexisting short-spelling gap"
    masked = store.redact(text)
    assert json.loads(masked) == {"payload": REDACTION_MARKER, "near": "ordinary", "n": 3}
    assert store.redaction_values() == [value]


@pytest.mark.parametrize(
    ("token", "value"),
    [(r'"a\/b"', "a/b"), (r'"\u0061\nb"', "a\nb"), (r'"\u79d8"', "秘")],
)
def test_valid_noncanonical_json_escapes_denote_the_same_scalar(token, value):
    assert json.loads(token) == value
    assert json.loads(scrub_known_json_string_values(token, [value])) == REDACTION_MARKER


def test_word_boundaries_and_surrogate_serialization_are_preserved():
    assert scrub_known_json_string_values(r'"\u0063athedral"', ["cat"]) == r'"\u0063athedral"'
    assert scrub_known_json_string_values(r'"\u0063at"', ["cat"]) == '"[redacted]"'
    masked = scrub_known_json_string_values(r'"\ud800\u79d8"', ["秘"])
    assert json.loads(masked) == "\ud800" + REDACTION_MARKER
    masked.encode("utf-8")  # changed scalars cannot introduce unencodable surrogates


@pytest.mark.parametrize(
    "text",
    [
        '{"\\u79d8": "ordinary"}',  # keys remain identifiers in the additive pass
        '{"payload": "\\u79d8\\q"}',  # malformed escape: no tolerant unescaping
        '{"payload": "\\u79d8',  # incomplete token: no guessed closing delimiter
        '{"payload": "\\u79"}',
        '{"payload": "literal \\\\u79d8"}',  # literal backslash-u is not the value
        '{"payload": "\\u79d9", "zh-CN": "目录说明：普通介绍，连接设置。"}',
        'JSON documentation: "\\n" is an escaped newline; use "\\t" for tabs.',
        '{"payload": "Журнал अज्ञात 目录 mixed ordinary text", "items": []}',
    ],
)
def test_unrelated_keys_malformed_and_literal_documentation_keep_bytes(text):
    assert scrub_known_json_string_values(text, ["秘"]) == text
    store = VariableStore(cwd=".")
    store.store_credential("SYNTHETIC_VALUE", "秘")
    before, _ = scrub_secrets_with_hits(text, store.redaction_values())
    assert store.redact(text) == before == text


def test_decoded_word_boundaries_and_strict_json_surrogates():
    assert scrub_known_json_string_values('"\\u0061lphabet"', ["alpha"]) == '"\\u0061lphabet"'
    assert json.loads(scrub_known_json_string_values('"\\u0061lpha!"', ["alpha"])) == "[redacted]!"
    masked = scrub_known_json_string_values('"\\ud800\\u79d8"', ["秘"])
    assert json.loads(masked) == "\ud800[redacted]"
    masked.encode("utf-8")  # reserialization must not publish an unencodable surrogate


def test_original_shape_coverage_and_learning_precede_additive_known_pass():
    # An unknown password containing the escaped known value is still judged on
    # the ORIGINAL carrier. Pre-decoding/masking would break that shape's input.
    short = "a\nb"
    password = "synthetic-prefix-a\\nb-suffix-713"
    text = json.dumps({"password": password, "payload": short})
    store = VariableStore(cwd=".")
    store.store_credential("SYNTHETIC_SHORT", short)
    before, hits = scrub_secrets_with_hits(text, store.redaction_values())
    assert hits, "the negative control must actually exercise an unknown shape"
    masked, report = store.redact_with_report(text)
    assert REDACTION_MARKER in masked
    assert password not in masked
    assert report == __import__(
        "local_operator.redaction_shapes", fromlist=["shape_report"]
    ).shape_report(hits)
    assert set(store.redaction_values()) >= {short, *(hit.value for hit in hits if hit.value)}
    # Every already-hidden scalar stays hidden; only the formerly-readable known
    # scalar can change. No shape is reclassified using a pre-masked input.
    assert json.loads(masked)["password"] == json.loads(before)["password"]
    assert json.loads(masked)["payload"] == REDACTION_MARKER


@pytest.mark.parametrize(
    "unknown",
    [
        "秘密資料認証接続番号八四七九",
        "СекретПарольЖурналЗапись",
        "गुप्तप्रवेशप्रमाणकूटशब्द",
        "Ж秘अMixedCredential991",
    ],
)
def test_unknown_nonlatin_full_shapes_keep_original_coverage_and_learning(unknown):
    text = f"postgresql://synthetic_user:{unknown}@synthetic.invalid/db"
    baseline, hits = scrub_secrets_with_hits(text, ["a\nb"])
    assert REDACTION_MARKER in baseline and unknown not in baseline
    store = VariableStore(cwd=".")
    store.store_credential("SYNTHETIC_KNOWN", "a\nb")
    assert store.redact(text) == baseline
    assert hits and unknown in store.redaction_values()


@pytest.mark.asyncio
async def test_real_session_read_write_provider_and_transcript_keep_executor_bytes(tmp_path):
    short = "a\nb"
    body = json.dumps({"payload": short, "near": "Журнал अज्ञात 普通目录"}) + "\n"
    source = tmp_path / "source.json"
    target = tmp_path / "target.json"
    source.write_text(body)
    provider = ScalarProvider(
        [
            ("read", {"path": str(source)}),
            ("write", {"path": str(target), "content": body}),
            ("read", {"path": str(target)}),
        ]
    )
    session = session_at(tmp_path, provider)
    session.variables.store_credential("SYNTHETIC_SHORT", short)
    await session.prompt("Execute the synthetic file checks")
    assert len(provider.requests) == 4
    assert source.read_bytes() == target.read_bytes() == body.encode()
    history = list(session.transcript.build_llm_history())
    results = [message for message in history if getattr(message, "role", None) == "tool"]
    assert len(results) == 3, [getattr(message, "role", None) for message in history]
    for result in results:
        assert json.dumps(short)[1:-1] not in result.text
    assert REDACTION_MARKER in results[0].text
    # Read the durable rows, not only the in-memory hook, and inspect the provider
    # requests too. Neither view may recover the executor's escaped scalar.
    durable = (tmp_path / "session" / "transcript.jsonl").read_text()
    assert json.dumps(json.dumps(short)[1:-1])[1:-1] not in durable
    assert REDACTION_MARKER in durable
    assert REDACTION_MARKER in repr(provider.requests[-1])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    ["SYNTHETIC_SERVICE_ACCESS_TOKEN", "OPENAI_API_KEY", "GH_TOKEN", "AWS_SECRET_ACCESS_KEY"],
)
async def test_noncolliding_secret_names_and_read_variable_refusal_survive(
    tmp_path, store: SecretStore, monkeypatch, name
):
    monkeypatch.setattr("local_operator.secrets.access.open_store", lambda **_: store)
    session = session_at(tmp_path, ScalarProvider([]))
    session.variables.store_credential(name, "synthetic-value-83612")
    context = ToolContext(cwd=str(tmp_path), variables=session.variables)
    stored = await execute_secret(
        "synthetic-store",
        {"op": "store", "name": name, "value": "synthetic-value-83612"},
        None,
        None,
        context,
    )
    listed = await execute_secret("synthetic-list", {"op": "list"}, None, None, context)
    refused = await execute_read_variable("synthetic-readvar", {"name": name}, None, None, context)
    assert not stored.is_error and not listed.is_error
    assert refused.is_error
    for result in [stored, listed, refused]:
        assert name in result.text
        assert session._redact_tool_result_text(result.text) == result.text
    shell_identifier = f'printf %s "${name}"'
    assert session._redact_tool_result_text(shell_identifier) == shell_identifier
    assert stored.details["name"] == name


@pytest.mark.asyncio
async def test_identical_name_registered_value_collision_keeps_masking_precedence(tmp_path):
    session = session_at(tmp_path, ScalarProvider([]))
    name = "SYNTHETIC_IDENTIFIER"
    session.variables.store_credential(name, "synthetic-other-value-291")
    session.variables.store_credential("SYNTHETIC_COLLISION", name)
    result = await execute_read_variable(
        "synthetic-collision",
        {"name": name},
        None,
        None,
        ToolContext(cwd=str(tmp_path), variables=session.variables),
    )
    assert name in result.text
    assert name not in session._redact_tool_result_text(result.text)
    # No name-based value exemption: a value with exactly that spelling is masked.
    assert (
        session._redact_tool_result_text(json.dumps({"payload": name}))
        == '{"payload": "[redacted]"}'
    )
