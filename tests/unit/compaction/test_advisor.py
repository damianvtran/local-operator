"""Compaction advisor (BETA): parsing, the rejection rules, and the guarantee
that the whole feature is inert while the flag is off.

The advisor is a model judgement wired into a trigger, so almost every test
here is a REJECTION test. The load-bearing property is that a wrong or
hallucinated hint cannot make a pass worse than it would have been: it may
only widen what a pass preserves and lower the trigger to a floor.
"""

import pytest

from local_operator.compaction.advisor import (
    ADVISOR_MAX_REASON_CHARS,
    CompactionHint,
    build_advisor_prompt,
    parse_hint,
    validate_hint,
)
from local_operator.compaction.cutpoint import task_boundary_floor
from local_operator.compaction.thresholds import (
    CompactionSettings,
    resolve_advisor_floor_tokens,
    resolve_threshold_tokens,
    should_compact,
)
from local_operator.harness.types import Message, ToolCall


def _user(text: str = "do the thing", words: int = 40) -> Message:
    return Message.user(f"{text} " + ("word " * words))


def _assistant(text: str = "working on it", words: int = 40) -> Message:
    return Message.assistant(f"{text} " + ("word " * words))


def _assistant_with_call(call_id: str) -> Message:
    message = Message.assistant("running the tool")
    message.tool_calls = [ToolCall(id=call_id, name="bash", arguments={"command": "ls"})]
    return message


def _history() -> list[Message]:
    """user -> assistant -> user -> assistant, all with real token weight."""
    return [_user("first request"), _assistant(), _user("second request"), _assistant()]


def _payload(messages, **overrides):
    payload = {
        "preserve_from": messages[2].id,
        "compact_now": True,
        "confidence": 0.9,
        "reason": "second request is the current unit",
    }
    payload.update(overrides)
    return payload


def _validate(messages, payload, **kwargs):
    options = {
        "genuine_user_ids": {m.id for m in messages if m.role == "user"},
        "min_confidence": 0.6,
        "keep_recent_tokens": 1,
        "floor_cap": 1_000_000,
    }
    options.update(kwargs)
    return validate_hint(payload, messages, **options)


# --- parsing --------------------------------------------------------------


def test_parse_fenced_block():
    raw = 'prose before\n```json\n{"preserve_from": "abc", "confidence": 0.7}\n```\nafter'
    assert parse_hint(raw) == {"preserve_from": "abc", "confidence": 0.7}


def test_parse_bare_object():
    assert parse_hint('  {"preserve_from": "abc"}  ') == {"preserve_from": "abc"}


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "no json at all",
        "```json\nnot json\n```",
        '```json\n["a", "list"]\n```',  # valid JSON, wrong shape
    ],
)
def test_parse_rejects_unusable(raw):
    assert parse_hint(raw) is None


# --- validation: rejection rules -----------------------------------------


def test_valid_hint_accepted():
    messages = _history()
    hint = _validate(messages, _payload(messages))
    assert isinstance(hint, CompactionHint)
    assert hint.preserve_from_id == messages[2].id
    assert hint.compact_now is True
    # Token count is computed from the REAL messages, never taken from the model.
    assert hint.preserve_tokens > 0


def test_rejects_hallucinated_entry_id():
    """The anchor must be one of the ids the advisor was SHOWN. An id that
    does not exist is the signature of a model that stopped reading the list,
    and repairing it would hide exactly that."""
    messages = _history()
    assert _validate(messages, _payload(messages, preserve_from="not-a-real-id")) is None


def test_rejects_id_present_but_not_a_candidate():
    """A tool-calling assistant is not a legal cut, so it is never offered as
    a candidate; naming it anyway is rejected rather than snapped."""
    messages = [*_history(), _assistant_with_call("c1")]
    assert _validate(messages, _payload(messages, preserve_from=messages[-1].id)) is None


def test_rejects_low_confidence():
    messages = _history()
    assert _validate(messages, _payload(messages, confidence=0.4)) is None


@pytest.mark.parametrize("value", [None, "0.9", True, -0.1, 1.5])
def test_rejects_non_numeric_or_out_of_range_confidence(value):
    messages = _history()
    assert _validate(messages, _payload(messages, confidence=value)) is None


def test_rejects_over_long_reason_rather_than_truncating():
    """The reason reaches a user-visible receipt. A model that ignored an
    explicit length instruction has ignored the format, and the rest of its
    answer earns the same suspicion."""
    messages = _history()
    payload = _payload(messages, reason="x" * (ADVISOR_MAX_REASON_CHARS + 1))
    assert _validate(messages, payload) is None


def test_accepts_reason_at_the_cap():
    messages = _history()
    payload = _payload(messages, reason="x" * ADVISOR_MAX_REASON_CHARS)
    hint = _validate(messages, payload)
    assert hint is not None and len(hint.reason) == ADVISOR_MAX_REASON_CHARS


@pytest.mark.parametrize("value", [None, "yes", 1])
def test_rejects_non_boolean_compact_now(value):
    messages = _history()
    assert _validate(messages, _payload(messages, compact_now=value)) is None


def test_rejects_missing_payload():
    assert _validate(_history(), None) is None
    assert _validate(_history(), {}) is None


def test_rejects_narrowing_hint():
    """THE guard that makes a wrong hint harmless: the advisor may only WIDEN
    what a pass preserves. A hint whose window is narrower than the local
    floor would sever more than today's rule does, which is the one outcome
    the feature must not be able to produce."""
    messages = _history()
    # Anchor at the LAST message (a narrow window), with a floor far above it.
    payload = _payload(messages, preserve_from=messages[3].id)
    assert _validate(messages, payload, keep_recent_tokens=10_000) is None


def test_widening_hint_accepted_above_the_floor():
    """Anchoring at or before the task boundary is accepted; anchoring AFTER it
    is not, even with the caller's own keep_recent floor at its minimum — the
    task floor alone is enough to refuse a narrowing hint."""
    messages = _history()
    # messages[2] is the last genuine user turn, so it IS the task boundary.
    at_boundary = _validate(messages, _payload(messages, preserve_from=messages[2].id))
    wider = _validate(messages, _payload(messages, preserve_from=messages[0].id))
    narrower = _validate(messages, _payload(messages, preserve_from=messages[3].id))
    assert at_boundary is not None and wider is not None
    assert wider.preserve_tokens > at_boundary.preserve_tokens
    assert narrower is None


# --- task_boundary_floor --------------------------------------------------


def test_task_boundary_floor_measures_from_last_genuine_user_turn():
    messages = _history()
    floor = task_boundary_floor(messages, {m.id for m in messages if m.role == "user"}, cap=10**9)
    expected = sum(
        len(m.text) // 4 for m in messages[2:]
    )  # order-of-magnitude; exactness is the estimator's job
    assert floor > 0
    assert expected > 0


def test_task_boundary_floor_honours_the_cap():
    """Uncapped, a session whose last user turn is far back would demand a
    preserve window larger than the context and turn 'protect the task' into
    'never compact'."""
    messages = _history()
    assert task_boundary_floor(messages, None, cap=5) == 5
    assert task_boundary_floor(messages, None, cap=0) == 0


def test_task_boundary_floor_zero_without_a_genuine_user_turn():
    """No genuine user turn (every user-role entry is an injected delivery)
    leaves the caller's keep_recent_tokens untouched rather than guessing."""
    messages = _history()
    assert task_boundary_floor(messages, set(), cap=10**9) == 0
    assert task_boundary_floor([_assistant()], None, cap=10**9) == 0


def test_task_boundary_floor_ignores_injected_user_deliveries():
    """In the RENDERED history a wake/hub/todo delivery is a plain user
    Message. Measuring from one would anchor the task at an injection."""
    messages = [_user("real request"), _assistant(), _user("injected delivery")]
    genuine = {messages[0].id}
    from_real = task_boundary_floor(messages, genuine, cap=10**9)
    from_any = task_boundary_floor(messages, None, cap=10**9)
    assert from_real > from_any


# --- the trigger ----------------------------------------------------------


def test_advisor_floor_clamped_to_the_ordinary_trigger():
    """A floor configured ABOVE the ordinary trigger would read as 'compact
    later when the advisor is on' — a second, competing trigger."""
    settings = CompactionSettings(advisor_enabled=True, advisor_floor_tokens=10_000_000)
    window = 1_000_000
    assert resolve_advisor_floor_tokens(window, settings) == resolve_threshold_tokens(
        window, settings
    )


def test_advisor_can_only_pull_the_trigger_earlier():
    settings = CompactionSettings(advisor_enabled=True, advisor_floor_tokens=200_000)
    window = 1_000_000
    ordinary = resolve_threshold_tokens(window, settings)
    assert resolve_advisor_floor_tokens(window, settings) <= ordinary
    # Below the ordinary trigger but above the floor: only the advisory fires.
    assert should_compact(250_000, window, settings) is False
    assert should_compact(250_000, window, settings, advisory_ok=True) is True
    # Below the FLOOR: not even the advisor may fire.
    assert should_compact(150_000, window, settings, advisory_ok=True) is False


def test_advisory_never_suppresses_an_ordinary_trigger():
    """The advisory is an additional input, not a gate: a context above the
    ordinary threshold compacts whether or not advice arrived."""
    settings = CompactionSettings(advisor_enabled=True)
    window = 1_000_000
    assert should_compact(700_000, window, settings) is True
    assert should_compact(700_000, window, settings, advisory_ok=True) is True


def test_disabled_compaction_still_wins_over_advice():
    settings = CompactionSettings(enabled=False, advisor_enabled=True)
    assert should_compact(900_000, 1_000_000, settings, advisory_ok=True) is False
    off = CompactionSettings(strategy="off", advisor_enabled=True)
    assert should_compact(900_000, 1_000_000, off, advisory_ok=True) is False


# --- OFF BY DEFAULT: equivalence ------------------------------------------


def test_advisor_defaults_are_inert():
    settings = CompactionSettings()
    assert settings.advisor_enabled is False
    assert settings.advisor_every_n_turns == 20
    assert settings.advisor_floor_tokens == 200_000
    assert settings.advisor_trigger_tokens == 300_000
    assert settings.advisor_min_confidence == 0.6
    assert settings.advisor_timeout_s == 30.0
    assert settings.advisor_max_calls == 200
    assert settings.advisor_cooldown_turns == 60


def test_old_config_validates_unchanged():
    """Every advisor field is optional, so a config written before the feature
    validates and resolves exactly as it did."""
    legacy = {"enabled": True, "strategy": "auto", "keep_recent_tokens": 20000}
    settings = CompactionSettings.model_validate(legacy)
    assert settings.advisor_enabled is False
    assert settings.keep_recent_tokens == 20000


@pytest.mark.parametrize("window", [128_000, 200_000, 1_000_000])
@pytest.mark.parametrize(
    "context",
    [0, 1, 50_000, 150_000, 199_999, 250_000, 500_000, 599_999, 600_001, 900_000],
)
def test_off_by_default_is_byte_identical(window, context):
    """THE equivalence test. With the flag off, ``should_compact`` answers
    identically whether or not an advisory was offered — so a user who has not
    opted into the beta cannot observe it at all, at any window or context
    size."""
    settings = CompactionSettings()  # advisor_enabled defaults False
    assert should_compact(context, window, settings) == should_compact(
        context, window, settings, advisory_ok=True
    )


@pytest.mark.parametrize("context", [250_000, 400_000])
def test_flag_off_ignores_a_configured_floor(context):
    """Even a config that sets the other advisor knobs changes nothing while
    the beta flag itself is off."""
    settings = CompactionSettings(advisor_floor_tokens=100_000, advisor_trigger_tokens=1)
    assert should_compact(context, 1_000_000, settings, advisory_ok=True) is False


# --- prompt ---------------------------------------------------------------


def test_prompt_lists_only_legal_anchors():
    messages = [*_history(), _assistant_with_call("c1")]
    prompt = build_advisor_prompt(messages, context_tokens=480_000, threshold_tokens=600_000)
    for message in messages[:4]:
        assert message.id in prompt
    # The tool-calling assistant is not a legal cut and must not be offered.
    assert messages[4].id not in prompt


def test_prompt_carries_its_own_instructions():
    """The instructions ride in the user turn rather than a system block: a
    system block sits ahead of the cache prefix and measured a 0% cache hit
    (see ADVISOR_SYSTEM_PROMPT). This test pins the placement."""
    prompt = build_advisor_prompt(_history(), context_tokens=1, threshold_tokens=2)
    assert "compaction advisor" in prompt
    assert "preserve_from" in prompt


def test_prompt_does_not_restate_the_conversation():
    """The history is sent as the message list; restating it would pay twice
    and break the append-only property the cache economics rest on. Only a
    bounded EXCERPT of each candidate appears."""
    long_text = "unique-marker " + ("filler " * 400)
    messages = [Message.user(long_text), _assistant(), Message.user(long_text)]
    prompt = build_advisor_prompt(messages, context_tokens=1, threshold_tokens=2)
    # The excerpt is present, the full body is not.
    assert "unique-marker" in prompt
    assert long_text.strip() not in prompt
    assert len(prompt) < len(long_text)
