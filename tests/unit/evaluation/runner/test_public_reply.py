"""Offline public-memory parity through the real provider and shared compactor.

The erasure test deliberately varies ONLY image bytes first. Different IDs or
observation text would leak the answer into the control and hide the defect.
Summary output below is deterministic plumbing evidence, not an LLM quality claim.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

from local_operator.compaction.api import serialize_conversation
from local_operator.compaction.pass_ import run_compaction_pass
from local_operator.compaction.snapcompact import serialize_for_snapcompact
from local_operator.compaction.thresholds import CompactionSettings
from local_operator.evaluation.evidence.models import (
    ActionBatchPayload,
    ModelRequestPayload,
    ModelResponsePayload,
    canonical_digest,
)
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.receipts import RedactionSet
from local_operator.evaluation.runner.episode import EpisodeRunner
from local_operator.evaluation.runner.model import DecisionRejected, EpisodeTurn
from local_operator.evaluation.runner.provider_client import (
    DecisionParseError,
    _ContextBuilder,
    build_system_prompt,
    parse_decision,
)
from local_operator.evaluation.runner.public_reply import (
    MAX_PUBLIC_OBSERVATIONS_CHARS,
    REJECTED_PUBLIC_REPLY,
    decode_public_reply,
    public_reply_contract,
    redact_public_reply,
)
from local_operator.harness.types import ImageContent, Message, ModelSpec, TextContent
from tests.unit.evaluation.runner.conftest import (
    FakeAdapter,
    ScriptedModel,
    build_config,
    build_spec,
    payloads,
    selector,
)
from tests.unit.evaluation.runner.test_episode import _rescue_ok
from tests.unit.evaluation.runner.test_provider_client import (
    ROUTE,
    RecordingStream,
    _client,
    _wait_reply,
    finish_payload,
    observation,
    type_payload,
)


def envelope(batch: str, notes: Any = "") -> str:
    return json.dumps(
        {
            "reply_version": "1.0",
            "action_batch": json.loads(batch),
            "public_observations": notes,
        }
    )


@pytest.mark.parametrize("notes", ["", "Visible status: ready", "x" * 2000, "東京"])
def test_public_reply_binds_without_changing_legacy_batch_bytes(notes: str) -> None:
    current = observation()
    legacy = parse_decision(type_payload(current), current, route=ROUTE)
    visible = envelope(type_payload(current), notes)
    decision = parse_decision(visible, current, route=ROUTE)
    assert decision.public_reply == visible
    assert legacy.public_reply is None
    assert decision.action_batch.to_canonical_json() == legacy.action_batch.to_canonical_json()
    assert canonical_digest("adapter-action-batch-v1", decision.action_batch) == canonical_digest(
        "adapter-action-batch-v1", legacy.action_batch
    )
    assert current == observation()


@pytest.mark.parametrize("notes", [None, [], {}, 1, True, "x" * 2001, "\ud800"])
def test_invalid_or_oversized_notes_reject_entire_decision(notes: Any) -> None:
    with pytest.raises(DecisionParseError):
        parse_decision(envelope(type_payload(observation()), notes), observation(), route=ROUTE)


@pytest.mark.parametrize(
    "change",
    [
        "missing-version",
        "wrong-version",
        "extra-envelope",
        "extra-batch",
        "nested-envelope",
        "nested-batch",
        "duplicate-note",
        "duplicate-action",
        "duplicate-version",
        "trailing-prose",
        "trailing-batch",
        "leading-prose",
        "wrong-observation",
    ],
)
def test_envelope_is_single_unambiguous_current_decision(change: str) -> None:
    current = observation()
    value = json.loads(envelope(type_payload(current), "visible fact"))
    if change == "missing-version":
        del value["reply_version"]
    elif change == "wrong-version":
        value["reply_version"] = "2.0"
    elif change == "extra-envelope":
        value["actions"] = json.loads(type_payload(current))["actions"]
    elif change == "extra-batch":
        value["action_batch"]["episode_id"] = "another-episode"
    elif change == "nested-envelope":
        value = {"wrapper": value}
    elif change == "nested-batch":
        value["action_batch"] = {"wrapper": value["action_batch"]}
    elif change == "wrong-observation":
        value["action_batch"] = json.loads(type_payload(observation(1)))
    raw = json.dumps(value)
    if change == "duplicate-note":
        raw = raw.replace(
            '"public_observations":', '"public_observations":"other", "public_observations":'
        )
    elif change == "duplicate-action":
        raw = raw.replace('"kind":', '"kind":"finish", "kind":')
    elif change == "duplicate-version":
        raw = raw.replace('"reply_version":', '"reply_version":"2.0", "reply_version":')
    elif change == "trailing-prose":
        raw += " commentary"
    elif change == "trailing-batch":
        raw += finish_payload(current)
    elif change == "leading-prose":
        raw = "commentary " + raw
    with pytest.raises(DecisionParseError):
        parse_decision(raw, current, route=ROUTE)


def test_legacy_trailing_envelope_cannot_supersede_current_decision() -> None:
    current = observation()
    with pytest.raises(DecisionParseError, match="second action batch"):
        parse_decision(
            type_payload(current) + envelope(finish_payload(current)), current, route=ROUTE
        )
    # Preserve the original tolerance for old quoted decisions.
    decision = parse_decision(
        type_payload(current) + envelope(finish_payload(observation(1))), current, route=ROUTE
    )
    assert decision.public_reply is None


@pytest.mark.asyncio
@pytest.mark.parametrize("escaped_keys", [False, True])
async def test_rejected_envelope_notes_are_not_replayed_as_facts(
    tmp_path: Path,
    escaped_keys: bool,
) -> None:
    secret = "invalid-note-must-not-enter-memory"
    raw = envelope(type_payload(observation(1)), secret)
    if escaped_keys:
        raw = raw.replace("public_observations", "public_\\u006fbservations")
        raw = raw.replace("reply_version", "reply_\\u0076ersion")
        raw = raw.replace("action_batch", "action_\\u0062atch")
    stream = RecordingStream(raw)
    client = _client(stream, tmp_path)
    turns = [EpisodeTurn(observation=observation())]
    with pytest.raises(DecisionRejected) as error:
        await client.decide(observation(), turns)
    assert error.value.reply == REJECTED_PUBLIC_REPLY
    stream.reply = finish_payload(observation())
    await client.decide(observation(), turns)
    replay = "\n".join(message.text for message in stream.requests[-1].messages)
    assert secret not in replay
    assert REJECTED_PUBLIC_REPLY in replay
    assert not stream.summary_requests


@pytest.mark.parametrize(
    "transform",
    [
        str,
        lambda s: quote(s, safe=""),
        lambda s: base64.b64encode(s.encode()).decode(),
    ],
)
def test_resolved_secret_notes_are_redacted_before_replay(transform: Any) -> None:
    secret = "known-secret/credential-value"
    raw = envelope(type_payload(observation()), transform(secret))
    # JSON escapes cannot hide decoded credentials from the existing boundary.
    raw = raw.replace("known", "\\u006bnown")
    clean = redact_public_reply(raw, RedactionSet.from_resolved_values([secret]))
    assert json.loads(clean)["public_observations"] == "[redacted public observations]"
    assert json.loads(clean)["action_batch"] == json.loads(raw)["action_batch"]
    assert secret not in clean


def test_contract_identity_is_separate_from_action_surface() -> None:
    metadata = public_reply_contract()
    contract = json.loads(metadata["model_reply_contract"])
    assert metadata["model_reply_contract_digest"] == canonical_digest(
        "runner-model-reply-v1", contract
    )
    schema = contract["schema"]
    assert schema["properties"]["public_observations"]["maxLength"] == MAX_PUBLIC_OBSERVATIONS_CHARS
    assert schema["properties"]["action_batch"]["additionalProperties"] is False
    prompt = build_system_prompt()
    assert '"reply_version": "1.0"' in prompt
    assert "concise NEW factual data" in prompt
    assert "private reasoning" in prompt and "credentials/secrets" in prompt


def test_context_builder_retains_notes_as_append_only_shared_messages(tmp_path: Path) -> None:
    first, second = observation(), observation(1)
    decision = parse_decision(
        envelope(type_payload(first), "Visible status: ready"), first, route=ROUTE
    )
    context = _ContextBuilder(artifact_root=tmp_path, keep_recent_frames=3, rebuild_every_frames=12)
    context.append_new_turns([EpisodeTurn(observation=first)])
    prefix = list(context.messages)
    turns = [
        EpisodeTurn(
            observation=first, batch=decision.action_batch, public_reply=decision.public_reply
        ),
        EpisodeTurn(observation=second),
    ]
    context.append_new_turns(turns)
    assert context.messages[0] is prefix[0]
    assert context.messages[1].role == "assistant"
    assert all(isinstance(block, TextContent) for block in context.messages[1].content)
    assert context.messages[1].text == decision.public_reply
    closed = list(context.messages)
    context.append_new_turns(turns)
    assert all(a is b for a, b in zip(closed, context.messages, strict=True))


def image_history(variant: str, *, notes: bool) -> list[Message]:
    history = []
    for index in range(12):
        # IDs/text/actions are fixed: only the oldest screenshot's bytes vary.
        pixels = variant if index == 0 else f"unchanged-frame-{index}"
        history.append(
            Message.user(
                f"Observation {index}",
                [
                    ImageContent(
                        data=base64.b64encode(pixels.encode()).decode(),
                        mime_type="image/png",
                    )
                ],
            )
        )
        batch = type_payload(observation(index))
        fact = f"Visible label: {variant}" if index == 0 else ""
        reply = envelope(batch, fact) if notes else batch
        history.append(Message.assistant(reply))
    # Message IDs default to UUIDs; fix them so the paired controls genuinely
    # differ only in the old image (and, for the treatment, its public note).
    return [
        message.model_copy(update={"id": f"message-{index}"})
        for index, message in enumerate(history)
    ]


@pytest.mark.asyncio
async def test_twelve_frame_erasure_and_public_facts_after_prune_then_text_compaction() -> None:
    model = ModelSpec(provider="provider", model_id="model", context_window=128_000)

    async def prune(history: list[Message]) -> Any:
        return await run_compaction_pass(
            history,
            model=model,
            settings=CompactionSettings(keep_recent_frames=3),
            summarize=None,
            now_ms=10_000,
            last_activity_ms=10_000,
        )

    left, right = image_history("violet", notes=False), image_history("amber", notes=False)
    assert left != right
    # Neither serializer could rescue pixels simply by running before prune.
    assert serialize_conversation(left) == serialize_conversation(right)
    assert serialize_for_snapcompact(left) == serialize_for_snapcompact(right)
    erased_left, erased_right = await prune(left), await prune(right)
    assert erased_left.frames_dropped == erased_right.frames_dropped == 9
    assert not erased_left.ran and not erased_right.ran
    assert erased_left.messages == erased_right.messages

    summaries = []
    for variant in ("violet", "amber"):
        result = await prune(image_history(variant, notes=True))
        assert result.frames_dropped == 9 and not result.ran
        fact = f"Visible label: {variant}"
        assert fact in serialize_conversation(result.messages)
        assert fact in serialize_for_snapcompact(result.messages)
        prompts = []

        async def summarize(prompt: str) -> str:
            prompts.append(prompt)
            # Only extract from the actual serialized request, never inject
            # an external expected fact into the compactor's output.
            return next(
                f"Visible label: {v}"
                for v in ("violet", "amber")
                if f"Visible label: {v}" in prompt
            )

        compacted = await run_compaction_pass(
            result.messages,
            model=model,
            settings=CompactionSettings(keep_recent_tokens=100, strategy="context-full"),
            summarize=summarize,
            now_ms=10_000,
            last_activity_ms=10_000,
            respect_threshold=False,
        )
        assert compacted.ran and len(prompts) == 1
        assert fact in compacted.messages[0].text
        summaries.append(compacted.messages)
    assert summaries[0] != summaries[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("secret", [False, True])
async def test_real_provider_runner_records_and_replays_public_evidence(
    tmp_path: Path,
    episode_id: str,
    secret: bool,
) -> None:
    note = "known-secret/credential-value" if secret else "Visible status: ready"
    calls = 0

    def reply(message: Message) -> str:
        nonlocal calls
        calls += 1
        raw = _wait_reply(message)
        if calls == 2:
            oid = json.loads(raw)["actions"][0]["observation_id"]
            raw = json.dumps(
                {
                    "actions": [
                        {
                            "kind": "finish",
                            "observation_id": oid,
                            "status": "done",
                            "reason": "complete",
                        }
                    ]
                }
            )
        return envelope(raw, note if calls == 1 else "")

    config = build_config(tmp_path)
    stream = RecordingStream(reply)
    client = _client(stream, config.artifact_root)
    runner = EpisodeRunner(
        build_spec(episode_id),
        config,
        selector=selector(tmp_path),
        model=client,
        launch=lambda _: FakeAdapter(tmp_path, episode_id),
        rescue=_rescue_ok,
        redactions=RedactionSet.from_resolved_values([note] if secret else []),
    )
    outcome = await runner.run()
    assert outcome.status == "completed"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    responses = payloads(root, ModelResponsePayload)
    assert len(responses) == calls == 2 and not stream.summary_requests
    ref = responses[0].redacted_response
    assert ref is not None
    recorded = (root / "artifacts" / ref.sha256).read_text()
    expected = "[redacted public observations]" if secret else note
    assert json.loads(recorded)["public_observations"] == expected
    replay = next(m.text for m in stream.requests[1].messages if m.role == "assistant")
    assert replay == recorded
    batches = payloads(root, ActionBatchPayload)
    batch = json.loads((root / "artifacts" / batches[0].action_artifact.sha256).read_text())
    assert batch["actions"] == json.loads(recorded)["action_batch"]["actions"]
    manifest = json.loads((root / "manifest.json").read_text())
    assert all(manifest["metadata"][key] == value for key, value in public_reply_contract().items())
    request = payloads(root, ModelRequestPayload)[0]
    assert request.tool_schema_digest == canonical_digest(
        "runner-tool-schema-v1", json.loads(manifest["metadata"]["action_surface"])
    )
    if secret:
        assert note not in "\n".join(m.text for m in stream.requests[1].messages)
        assert not any(
            note.encode() in path.read_bytes() for path in root.rglob("*") if path.is_file()
        )


@pytest.mark.asyncio
async def test_old_fake_client_does_not_claim_a_new_reply_contract(
    tmp_path: Path,
    episode_id: str,
) -> None:
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path),
        selector=selector(tmp_path),
        model=ScriptedModel(["finish"]),
        launch=lambda _: FakeAdapter(tmp_path, episode_id),
        rescue=_rescue_ok,
    )
    outcome = await runner.run()
    root = outcome.bundle_root
    assert root is not None and verify_bundle(root).valid
    manifest = json.loads((root / "manifest.json").read_text())
    assert "model_reply_contract" not in manifest["metadata"]
    assert payloads(root, ModelResponsePayload)[0].redacted_response is None
    assert decode_public_reply(envelope(type_payload(observation())))
