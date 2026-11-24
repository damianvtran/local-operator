"""Where a cache-breaking request first departed from the one before it.

Anthropic names the SECTION that broke a cached prefix and charges for the
rewrite, but never says which block or message moved. That gap is why two
incidents on this fleet had to be diagnosed by hand from transcripts. These
cells pin the local answer: a content-blind label, an index, and no divergence
reported for the ordinary cases (a first request, an appended turn).

Nothing here may carry prompt text: every kind is the message's own role and
harness name, so a log line built from it cannot leak an excerpt or an argument.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.analytics.model import (
    first_prefix_divergence,
    message_kind,
    prefix_sections,
)


def _message(
    role: str = "user", text: str = "", *, custom_type: str | None = None, details: Any = None
) -> Any:
    return SimpleNamespace(
        role=role,
        content=[SimpleNamespace(type="text", text=text)] if text else [],
        tool_calls=[],
        is_error=False,
        custom_type=custom_type,
        provider_payload={"details": details} if details else None,
    )


def _request(
    *messages: Any,
    system: tuple[str, ...] = ("system",),
    tools: tuple[Any, ...] = (),
    diagnostics: bool = True,
) -> Any:
    return SimpleNamespace(
        system_blocks=list(system),
        messages=list(messages),
        tools=list(tools),
        cache_diagnostics=diagnostics,
    )


def test_no_previous_comparison_says_nothing() -> None:
    """A first request on a chain has no baseline: `""`, never a fake index."""
    assert first_prefix_divergence(None, prefix_sections(_request(_message(text="a")))) == ""


def test_an_appended_turn_is_not_a_divergence() -> None:
    """The ordinary turn: the cached prefix is intact and only the tail grew."""
    previous = prefix_sections(_request(_message("user", "one")))
    current = prefix_sections(_request(_message("user", "one"), _message("assistant", "two")))
    assert first_prefix_divergence(previous, current) == ""


def test_unchanged_context_says_nothing() -> None:
    request = _request(_message("user", "one"), _message("assistant", "two"))
    assert first_prefix_divergence(prefix_sections(request), prefix_sections(request)) == ""


def test_a_changed_message_is_named_by_index_and_kind() -> None:
    previous = prefix_sections(
        _request(_message("user", "one"), _message("user", "two", custom_type="wake_prompt"))
    )
    current = prefix_sections(
        _request(_message("user", "one"), _message("user", "two EDITED", custom_type="wake_prompt"))
    )
    assert first_prefix_divergence(previous, current) == "messages[1]:user_wake_prompt"


def test_an_inserted_message_shifts_the_index_to_the_first_difference() -> None:
    previous = prefix_sections(_request(_message("user", "one"), _message("assistant", "two")))
    current = prefix_sections(
        _request(
            _message("user", "one"),
            _message("user", "injected", custom_type="session_state"),
            _message("assistant", "two"),
        )
    )
    assert first_prefix_divergence(previous, current) == "messages[1]:user_session_state"


def test_a_changed_system_block_is_named_before_tools_and_messages() -> None:
    previous = prefix_sections(_request(_message("user", "one")))
    current = prefix_sections(_request(_message("user", "one"), system=("system", "repo guidance")))
    assert first_prefix_divergence(previous, current) == "system[1]"


def test_a_changed_block_in_place_keeps_its_index() -> None:
    previous = prefix_sections(_request(system=("a", "b")))
    current = prefix_sections(_request(system=("a", "b rewritten")))
    assert first_prefix_divergence(previous, current) == "system[1]"


def test_changed_tools_are_named() -> None:
    previous = prefix_sections(_request(_message("user", "one")))
    tool = SimpleNamespace(name="bash", description="run", parameters={"type": "object"})
    current = prefix_sections(_request(_message("user", "one"), tools=(tool,)))
    assert first_prefix_divergence(previous, current) == "tools"


def test_a_shrunk_context_diverges_where_it_ends() -> None:
    """Compaction re-renders a shorter history: the cut point is the answer."""
    previous = prefix_sections(
        _request(_message("user", "one"), _message("assistant", "two"), _message("user", "three"))
    )
    current = prefix_sections(_request(_message("user", "one")))
    assert first_prefix_divergence(previous, current) == "messages[1]"


def test_a_tool_message_is_named_as_a_tool() -> None:
    previous = prefix_sections(_request(_message("tool", "one"), _message("tool", "two")))
    current = prefix_sections(_request(_message("tool", "one"), _message("tool", "three")))
    assert first_prefix_divergence(previous, current) == "messages[1]:tool"


# --------------------------------------------------------------------------
# The label is content-blind by construction
# --------------------------------------------------------------------------


def test_a_kind_is_the_role_and_harness_name_only() -> None:
    secret = "sk-live-DO-NOT-LOG"
    assert message_kind(_message("user", secret, custom_type="team_bridge_message")) == (
        "user_team_bridge_message"
    )
    assert secret not in message_kind(_message("user", secret))


def test_a_custom_type_with_punctuation_is_normalised_not_echoed() -> None:
    label = message_kind(_message("user", "x", custom_type="Weird Type!\nwith:specials"))
    assert label == "user_weird_type_with_specials"


def test_a_very_long_custom_type_is_capped() -> None:
    assert len(message_kind(_message("user", "x", custom_type="t" * 500))) <= 48


def test_a_custom_type_in_a_payload_never_reaches_the_label() -> None:
    """PRIVACY BLOCKER (review round 1): a tool's own words are not a name.

    ``provider_payload['details']`` is harness bookkeeping on some rows and tool
    metadata on others, so a label that read a type out of it could echo a word
    the tool produced. A synthetic secret under that key must leave the label as
    the bare role.
    """
    message = _message("user", "x", details={"custom_type": "SecretPassword123"})
    assert message_kind(message) == "user"


def test_the_renderer_injection_marker_names_an_injected_turn() -> None:
    """A BOOLEAN the renderer owns — never a string found in the payload."""
    message = _message("user", "x")
    message.provider_payload = {"harness_injected": True}
    assert message_kind(message) == "user_injected"


def test_a_payload_string_under_the_marker_is_not_a_claim() -> None:
    message = _message("user", "x")
    message.provider_payload = {"harness_injected": "SecretPassword123"}
    assert message_kind(message) == "user"


def test_an_unknown_message_still_gets_a_label() -> None:
    assert message_kind(object()) == "message"


# --------------------------------------------------------------------------
# The log line
# --------------------------------------------------------------------------


def _break_log(caplog: pytest.LogCaptureFixture, divergence: str) -> str:
    from local_operator.harness.types import Usage
    from local_operator.model.configure import SessionStreamFn

    # A PARTIAL stream fn is the point: the line only reads ``_session_id``,
    # and the recording cells build the real class the same way.
    stub = cast(Any, SimpleNamespace(_session_id="abc"))
    usage = Usage(
        input_tokens=1_000,
        cache_read_tokens=5_000,
        cache_write_tokens=200_000,
        context_tokens=206_000,
    )
    with caplog.at_level(logging.INFO, logger="local_operator.model.configure"):
        SessionStreamFn._log_cache_break(
            stub,
            usage,
            "messages_changed",
            200_000,
            provider="anthropic",
            first_divergence=divergence,
        )
    return caplog.text


def test_a_large_break_names_the_first_divergence(caplog: pytest.LogCaptureFixture) -> None:
    text = _break_log(caplog, "messages[42]:user_team_bridge_message")
    assert "first_divergence=messages[42]:user_team_bridge_message" in text


def test_a_break_without_a_comparison_says_so(caplog: pytest.LogCaptureFixture) -> None:
    """`-` rather than an empty field: "not compared" is a fact worth reading."""
    assert "first_divergence=-" in _break_log(caplog, "")


# --------------------------------------------------------------------------
# The baseline lifecycle, and what the comparison cannot see
# --------------------------------------------------------------------------


def _stream_stub() -> Any:
    """A stream fn with only the prefix attributes set (see the recording cells)."""
    from local_operator.model.configure import SessionStreamFn

    stub = object.__new__(SessionStreamFn)
    stub._cache_first_divergence = ""
    stub._cache_prefix_last = None
    stub._cache_prefix_pending = None
    return stub


def _commit(stub: Any) -> None:
    """The accepted-response step: pending becomes the baseline."""
    stub._cache_prefix_last = stub._cache_prefix_pending
    stub._cache_prefix_pending = None


def test_a_failed_snapshot_drops_the_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    """MAJOR (review round 1): a missing snapshot must not leave an older one.

    The provider chain advances on an accepted response whatever happens here,
    so keeping the old snapshot would compare the NEXT turn against a request
    that is no longer the previous one — and name a divergence that never
    happened. The true answer after a failed snapshot is "no baseline".
    """
    import local_operator.analytics as analytics

    stub = _stream_stub()
    stub._note_cache_prefix(_request(_message("user", "one")))
    _commit(stub)
    assert stub._cache_prefix_last is not None

    def _boom(_request: Any) -> Any:
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(analytics, "prefix_sections", _boom)
    stub._note_cache_prefix(_request(_message("user", "two")))
    assert stub._cache_prefix_last is None
    assert stub._cache_first_divergence == ""

    monkeypatch.undo()
    stub._note_cache_prefix(_request(_message("user", "two")))
    assert stub._cache_first_divergence == ""
    _commit(stub)
    stub._note_cache_prefix(_request(_message("user", "two")))
    assert stub._cache_first_divergence == ""


def test_a_turn_that_did_not_opt_in_costs_nothing() -> None:
    stub = _stream_stub()
    stub._note_cache_prefix(_request(_message("user", "one"), diagnostics=False))
    assert stub._cache_prefix_pending is None
    assert stub._cache_first_divergence == ""


def test_media_bytes_are_outside_the_comparison_by_contract() -> None:
    """Documented limitation, pinned: an image's pixels are not in the digest.

    The canonical form is the prompt hash's, which deliberately excludes media
    bytes (they would be copied on the event loop). A changed image therefore
    compares equal; the provider's own section label is the only diagnosis for
    that break, and this field says ``-`` rather than guessing. An image ADDED
    or REMOVED still moves the digest, because the part markers are covered.
    """
    one = _message("user", "look")
    one.content.append(SimpleNamespace(type="image", data=b"pixels-A"))
    two = _message("user", "look")
    two.content.append(SimpleNamespace(type="image", data=b"pixels-B"))
    assert (
        first_prefix_divergence(prefix_sections(_request(one)), prefix_sections(_request(two)))
        == ""
    )

    three = _message("user", "look")
    three.content.append(SimpleNamespace(type="image", data=b"pixels-A"))
    three.content.append(SimpleNamespace(type="image", data=b"second"))
    assert (
        first_prefix_divergence(prefix_sections(_request(one)), prefix_sections(_request(three)))
        == "messages[0]:user"
    )
