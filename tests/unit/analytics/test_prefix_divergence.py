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
    assert first_prefix_divergence(previous, current) == "messages[1]:user"


def test_an_inserted_message_shifts_the_index_to_the_first_difference() -> None:
    previous = prefix_sections(_request(_message("user", "one"), _message("assistant", "two")))
    current = prefix_sections(
        _request(
            _message("user", "one"),
            _message("user", "injected", custom_type="session_state"),
            _message("assistant", "two"),
        )
    )
    injected = _message("user", "injected")
    injected.provider_payload = {"harness_injected": True}
    current = prefix_sections(
        _request(_message("user", "one"), injected, _message("assistant", "two"))
    )
    assert first_prefix_divergence(previous, current) == "messages[1]:user_injected"


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


def test_a_duck_typed_role_is_not_a_role() -> None:
    """R3-4: an arbitrary object's ``role`` attribute is data, not a claim."""
    assert message_kind(SimpleNamespace(role="SecretPassword123")) == "message"
    assert message_kind(SimpleNamespace(role="user", provider_payload=None, content=[])) == "user"


def test_the_injection_marker_is_only_read_on_a_user_turn() -> None:
    """The renderer marks USER turns; a tool object wearing it is not an injection."""
    tool = SimpleNamespace(role="tool", content=[], provider_payload={"harness_injected": True})
    assert message_kind(tool) == "tool"


def test_a_kind_is_the_role_only() -> None:
    """No value from any data source names a message: role, or role_injected."""
    message = _message("user", "an ordinary sent message")
    assert message_kind(message) == "user"
    assert "ordinary" not in message_kind(message)


def test_a_custom_type_with_punctuation_is_normalised_not_echoed() -> None:
    label = message_kind(_CustomTypeHolder("Weird Type!\nwith:specials", role="user"))
    assert label == "user"


def test_a_very_long_type_value_is_not_echoed_at_all() -> None:
    assert message_kind(_CustomTypeHolder("t" * 500, role="user")) == "user"


class _CustomTypeHolder:
    """A message-shaped object whose ``custom_type`` carries arbitrary text."""

    def __init__(self, custom_type: str, *, role: str = "message") -> None:
        self.custom_type = custom_type
        self.role = role
        self.content = []
        self.tool_calls = []
        self.is_error = False
        self.provider_payload = None


def test_a_custom_type_the_type_owns_never_reaches_the_label() -> None:
    """R2-1: an unrestricted ``str`` on a harness type is still DATA.

    ``CustomMessage.custom_type`` is not an enum, so "the type owns it" was
    never the same claim as "the value is trusted". The branch is gone; the
    reviewer's exact repro now yields the neutral label.
    """
    from local_operator.harness.types import CustomMessage

    message = CustomMessage(custom_type="SecretPassword123", details={"text": "x"})
    assert message_kind(message) == "message"


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
    stub._cache_prefix_failed = False
    return stub


def _commit(stub: Any) -> None:
    """The accepted-response step: a diagnosed start commits the baseline."""
    stub._note_cache_start(SimpleNamespace(cache_miss_type="unchanged", response_id="r"))


def test_a_failed_snapshot_that_is_never_accepted_keeps_the_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R2-3: a request the provider never took did not become the previous one."""
    import local_operator.analytics as analytics

    stub = _stream_stub()
    stub._note_cache_prefix(_request(_message("user", "one")))
    stub._note_cache_start(SimpleNamespace(cache_miss_type="unchanged", response_id="r1"))
    assert stub._cache_prefix_last is not None

    def _boom(_request: Any) -> Any:
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(analytics, "prefix_sections", _boom)
    stub._note_cache_prefix(_request(_message("user", "one"), _message("user", "two")))
    assert stub._cache_prefix_last is not None, "not accepted yet: the baseline still holds"

    monkeypatch.undo()
    stub._note_cache_prefix(_request(_message("user", "one")))
    assert stub._cache_first_divergence == "", "an unchanged prefix compared against r1"


def test_a_failed_snapshot_that_is_accepted_drops_the_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    stub._note_cache_prefix(_request(_message("user", "one"), _message("user", "two")))
    # Accepted, and its own snapshot is missing: the old baseline is no longer
    # adjacent to anything the provider holds.
    stub._note_cache_start(SimpleNamespace(cache_miss_type="messages_changed", response_id="r2"))
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
    assert stub._cache_prefix_failed is False
    assert stub._cache_first_divergence == ""


def test_a_changed_image_mime_is_named_before_a_later_text_change() -> None:
    """R3-1: the provider serialises the MIME, so a PNG->JPEG swap is a change."""
    one = _message("user", "look")
    one.content.append(SimpleNamespace(type="image", data=b"same-bytes", mime_type="image/png"))
    two = _message("assistant", "answer two")
    jpeg = _message("user", "look")
    jpeg.content.append(SimpleNamespace(type="image", data=b"same-bytes", mime_type="image/jpeg"))
    assert (
        first_prefix_divergence(
            prefix_sections(_request(one)), prefix_sections(_request(jpeg, two))
        )
        == "messages[0]:user"
    )


def test_hashing_a_large_attachment_does_not_copy_it_whole() -> None:
    """R3-2: a 10 MB attachment cost a 20 MB peak when it was JSON-encoded.

    The payload is already in memory; the comparison may hash it but must not
    make an encoded copy of it on the event loop.
    """
    import tracemalloc

    message = _message("user", "look")
    message.content.append(SimpleNamespace(type="image", data="A" * (10 * 1024 * 1024)))
    tracemalloc.start()
    try:
        prefix_sections(_request(message))
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024 * 1024, f"peak {peak} bytes is an encoded copy, not a chunk"


def test_a_document_filename_is_part_of_the_comparison() -> None:
    """A renamed document is a different request to a provider that sends it."""
    one = _message("user", "see file")
    one.content.append(
        SimpleNamespace(
            type="document", data=b"same", mime_type="application/pdf", filename="a.pdf"
        )
    )
    two = _message("user", "see file")
    two.content.append(
        SimpleNamespace(
            type="document", data=b"same", mime_type="application/pdf", filename="b.pdf"
        )
    )
    assert (
        first_prefix_divergence(prefix_sections(_request(one)), prefix_sections(_request(two)))
        == "messages[0]:user"
    )


def test_a_changed_image_is_named_before_a_later_text_change() -> None:
    """R2-2: inherited kind-only media coverage would have blamed message 1.

    The prompt hash leaves media bytes out on purpose (it is a stored column);
    this comparison must not, or an image that changed in message 0 lets a text
    edit in message 1 be reported as the FIRST divergence — a false accusation
    the review reproduced.
    """
    one = _message("user", "look")
    one.content.append(SimpleNamespace(type="image", data=b"pixels-A"))
    two = _message("assistant", "answer one")
    three = _message("assistant", "answer TWO")

    before = prefix_sections(_request(one, two))
    after = prefix_sections(_request(_with_pixels(b"pixels-B"), three))
    assert first_prefix_divergence(before, after) == "messages[0]:user"


def _with_pixels(pixels: bytes) -> Any:
    message = _message("user", "look")
    message.content.append(SimpleNamespace(type="image", data=pixels))
    return message


def test_an_added_image_moves_the_digest() -> None:
    one = _with_pixels(b"pixels-A")
    both = _message("user", "look")
    both.content.append(SimpleNamespace(type="image", data=b"pixels-A"))
    both.content.append(SimpleNamespace(type="image", data=b"second"))
    assert (
        first_prefix_divergence(prefix_sections(_request(one)), prefix_sections(_request(both)))
        == "messages[0]:user"
    )


def test_a_media_payload_is_never_written_anywhere_readable() -> None:
    """The digest is one-way; the label carries no pixel and no byte count."""
    message = _with_pixels(b"SECRET-PIXELS")
    label = message_kind(message)
    assert "secret" not in label.lower()
    snapshot = prefix_sections(_request(message))
    assert b"SECRET-PIXELS" not in snapshot.messages[0]


def test_a_google_style_start_without_a_response_id_commits_the_baseline() -> None:
    """R3-3 / QA round 2: Google's start carries `cache_miss_type` and NO id.

    This is the real ``StreamStartEvent`` shape, not a duck type: gating the
    commit on ``response_id`` left every Google chain uncommitted and every
    Google break reading ``-``.
    """
    from local_operator.harness.types import StreamStartEvent

    stub = _stream_stub()
    stub._note_cache_prefix(_request(_message("user", "one")))
    google_start = StreamStartEvent(
        response_id=None, cache_miss_type="baseline", cache_missed_tokens=0
    )
    stub._note_cache_start(google_start)
    assert stub._cache_prefix_last is not None, "a diagnosed Google start IS an accept"

    # A changed first message is the comparison the baseline exists to make
    # (an APPENDED turn is not a divergence, by design — see the cell above).
    stub._note_cache_prefix(_request(_message("user", "one EDITED")))
    assert stub._cache_first_divergence == "messages[0]:user"

    # NEGATIVE CONTROL: an UNDIAGNOSED start on the same chain commits nothing.
    before = stub._cache_prefix_last
    stub._note_cache_prefix(_request(_message("user", "three")))
    stub._note_cache_start(StreamStartEvent(response_id=None, cache_miss_type=None))
    assert stub._cache_prefix_last is before


def test_the_stream_loop_gates_the_commit_on_the_diagnosis_not_a_response_id() -> None:
    """R3-3: the wiring itself, pinned — the old gate cannot come back silently.

    A cell that only drives ``_note_cache_start`` would keep passing if the loop
    went back to ``and event.response_id``, which is exactly the regression QA
    found. So the call site is read for what it does NOT contain.
    """
    import ast
    import inspect
    import textwrap

    from local_operator.model import configure as configure_module

    source = textwrap.dedent(inspect.getsource(configure_module.SessionStreamFn))
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_note_cache_start"
    ]
    assert len(calls) == 1, "the stream loop calls the commit step exactly once"
    enclosing = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and any(inner is calls[0] for inner in ast.walk(node))
    ]
    assert enclosing, "the commit step is guarded, not run on every event"
    gate = ast.unparse(enclosing[0].test)
    assert "cache_diagnostics" in gate
    assert "response_id" not in gate, f"the Google-shaped regression is back: {gate}"
