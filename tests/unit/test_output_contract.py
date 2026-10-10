"""The final-response output contract: extraction, strictness, schemas, refusals.

What this file is the referee for: the candidate ladder (§5 of the design) and
the per-format strictness table (§4). The loop-side behaviour (retry, exhaustion,
events) lives in ``tests/unit/harness/test_loop.py``; the CLI/SDK plumbing lives
in the exec-startup and sdk/spec tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import pytest
from pydantic import BaseModel

from local_operator.harness.types import Message
from local_operator.output_contract import (
    OUTPUT_FORMATS,
    MarkdownSchema,
    OutputContract,
    OutputContractError,
    OutputDecodeError,
    decode_output,
)

# ---------------------------------------------------------------------------
# Candidate order — labelled fences, unknown-label fences, whole text, prose scan
# ---------------------------------------------------------------------------


def test_labelled_fence_beats_an_earlier_unlabelled_block() -> None:
    """The label is how the model says WHICH block is the payload."""
    contract = OutputContract(format="json")
    text = "```\nnot json at all\n```\nThe answer:\n```json\n[1, 2]\n```"
    assert contract.check(text).payload_text == "[1, 2]"


def test_unlabelled_fence_beats_the_whole_text() -> None:
    contract = OutputContract(format="json")
    text = "Here you go.\n```\n[1, 2]\n```\nI hope that helps."
    check = contract.check(text)
    assert check.ok
    assert check.payload_text == "[1, 2]"


def test_whole_text_beats_prose_scan_for_exact_payloads() -> None:
    contract = OutputContract(format="json")
    check = contract.check('  {"a": 1}  ')
    assert check.payload_text == '{"a": 1}'


def test_cross_labelled_fence_is_skipped_never_crossed() -> None:
    """A ```yaml block is an explicit claim of a different format, even when
    its bytes would also parse as JSON: skipped, never crossed. The bare json
    scan honours the same skip, so a lone cross-labelled block is a rejection
    rather than a silently accepted payload, and the payload behind it is
    found only by a LATER candidate (the retry message is the remedy)."""
    contract = OutputContract(format="json")
    swallowed = contract.check('```yaml\n{"a": 1}\n```')
    assert not swallowed.ok
    check = contract.check('```yaml\n{"a": 1}\n```\n{"b": 2}')
    assert check.ok
    assert check.payload_text == '{"b": 2}'


def test_cross_labelled_tilde_fence_is_skipped_for_yaml() -> None:
    contract = OutputContract(format="yaml")
    # YAML has no prose-scan tier, so a cross-labelled block ALONE is a
    # rejection rather than a silently accepted payload.
    assert not contract.check("~~~toml\na = 1\n~~~").ok
    # The payload behind it is found only by a later candidate.
    check = contract.check("~~~toml\na = 1\n~~~\n~~~yaml\nb: 2\n~~~")
    assert check.ok
    assert check.payload_text == "b: 2"


def test_prose_scan_finds_a_bare_json_span_last() -> None:
    contract = OutputContract(format="json")
    check = contract.check('The result is {"total": 3} as requested.')
    assert check.ok
    assert check.payload_text == '{"total": 3}'


def test_prose_scan_is_json_only() -> None:
    """YAML/TOML deliberately have no prose tier: every text is a YAML scalar,
    and a TOML document cannot be meaningfully located inside prose."""
    yaml_contract = OutputContract(format="yaml")
    check = yaml_contract.check("The result is\na: 1\nas requested.")
    # No fence and no bare-scan tier, so this is a plain decode failure of the
    # whole text -- not a silently extracted fragment.
    assert not check.ok


def test_scalar_payload_accepted_for_json_when_exact() -> None:
    contract = OutputContract(format="json")
    check = contract.check("42")
    assert check.ok
    assert check.payload_text == "42"


def test_fenced_scalar_rejected_for_yaml() -> None:
    """The structured-value rule: a bare scalar is prose, not a payload."""
    contract = OutputContract(format="yaml")
    check = contract.check("```yaml\n42\n```")
    assert not check.ok
    assert "bare scalar" in check.error


def test_same_length_inner_fence_closes_the_outer() -> None:
    """Documented limit: content inside an outer fence is never rescanned, so
    a same-length inner fence closes the outer one — and the trailing stray
    block opens a fresh, unterminated fence."""
    from local_operator.output_contract import _scan_fences

    blocks = _scan_fences("```\nouter\n```\nstray\n```\n")
    assert [(b.info, b.content, b.closed) for b in blocks] == [
        ("", "outer", True),
        ("", "", False),
    ]


def test_a_shorter_marker_run_does_not_close_a_longer_fence() -> None:
    from local_operator.output_contract import _scan_fences

    blocks = _scan_fences("````\ncode\n```\nmore\n````\n")
    assert [(b.content, b.closed) for b in blocks] == [("code\n```\nmore", True)]


def test_a_fence_line_carrying_an_info_string_does_not_close() -> None:
    """A closer has an empty info string; ````` text`` is content."""
    from local_operator.output_contract import _scan_fences

    blocks = _scan_fences("```\na\n``` text\nb\n```\n")
    assert [(b.content, b.closed) for b in blocks] == [("a\n``` text\nb", True)]


def test_empty_and_whitespace_text_report_the_pinned_reason() -> None:
    for text in ("", "   \n\t "):
        check = OutputContract(format="json").check(text)
        assert not check.ok
        assert check.error == "the final message was empty"


def test_markdown_unterminated_fence_is_rejected() -> None:
    check = OutputContract(format="markdown").check("text\n```py\ncode without a close")
    assert not check.ok
    assert "unterminated" in check.error


def test_first_schema_failure_is_preferred_over_later_decode_failures() -> None:
    contract = OutputContract(format="json", schema={"type": "object", "required": ["name"]})
    # First candidate decodes but fails the schema; the trailing prose does not
    # decode at all. The reported reason must be the schema failure.
    check = contract.check('```json\n{"age": 3}\n```\nno payload here at all')
    assert not check.ok
    assert check.error.startswith("payload does not match the schema:")
    assert "name" in check.error


# ---------------------------------------------------------------------------
# Per-format strictness (the design's table, pinned)
# ---------------------------------------------------------------------------


def test_json_trailing_comma_is_rejected() -> None:
    check = OutputContract(format="json").check('{"a": 1,}')
    assert not check.ok
    assert check.error.startswith("not valid JSON:")


def test_json_single_quotes_are_rejected() -> None:
    assert not OutputContract(format="json").check("{'a': 1}").ok


def test_json_non_finite_literals_are_rejected() -> None:
    """``NaN``/``Infinity``/``-Infinity`` are Python literals, not JSON."""
    for literal in ("NaN", "Infinity", "-Infinity"):
        check = OutputContract(format="json").check('{"a": %s}' % literal)
        assert not check.ok, literal
        assert "not valid JSON" in check.error


def test_yaml_safe_loader_semantics_are_deliberate_tolerances() -> None:
    contract = OutputContract(format="yaml")
    # YAML 1.1 booleans and duplicate last-wins are PyYAML's documented
    # semantics; the module docstring/spec states this is deliberately not
    # hand-strictified.
    assert contract.check("flag: yes").ok
    assert contract.check("a: 1\na: 2").ok


def test_yaml_multi_document_is_rejected() -> None:
    check = OutputContract(format="yaml").check("a: 1\n---\nb: 2")
    assert not check.ok
    assert "not valid YAML" in check.error


def test_yaml_empty_document_is_a_decode_failure() -> None:
    assert not OutputContract(format="yaml").check("null").ok


def test_toml_duplicate_keys_are_rejected() -> None:
    check = OutputContract(format="toml").check("a = 1\na = 2")
    assert not check.ok
    assert "not valid TOML" in check.error


def test_toml_root_is_a_table() -> None:
    check = OutputContract(format="toml").check("[server]\nport = 80")
    assert check.ok
    assert check.payload_text == "[server]\nport = 80"


# ---------------------------------------------------------------------------
# Schema paths
# ---------------------------------------------------------------------------


class Invoice(BaseModel):
    total: int
    currency: str


@dataclass
class Item:
    name: str
    quantity: int


class Header(TypedDict):
    title: str


RAW_SCHEMA = {
    "type": "object",
    "required": ["name"],
    "properties": {"name": {"type": "string"}},
}


def test_pydantic_model_schema() -> None:
    contract = OutputContract(format="json", schema=Invoice)
    ok = contract.check('{"total": 3, "currency": "EUR"}')
    assert ok.ok
    bad = contract.check('{"total": "not a number", "currency": "EUR"}')
    assert not bad.ok
    assert bad.error.startswith("payload does not match the schema:")
    assert "total" in bad.error


def test_dataclass_schema() -> None:
    contract = OutputContract(format="json", schema=Item)
    assert contract.check('{"name": "widget", "quantity": 2}').ok
    assert not contract.check('{"name": "widget"}').ok


def test_typed_dict_schema() -> None:
    contract = OutputContract(format="json", schema=Header)
    assert contract.check('{"title": "hello"}').ok


def test_raw_json_schema_mapping() -> None:
    contract = OutputContract(format="yaml", schema=RAW_SCHEMA)
    assert contract.check("name: x").ok
    bad = contract.check("age: 3")
    assert not bad.ok
    assert "$" in bad.error  # jsonschema's instance path for the root
    assert "name" in bad.error


def test_markdown_schema_required_sections_in_order() -> None:
    contract = OutputContract(
        format="markdown", schema=MarkdownSchema(required_sections=("Summary", "Risks"))
    )
    assert contract.check("# Summary\nx\n## Risks\ny").ok
    missing = contract.check("# Summary\nx")
    assert missing.error == "missing required section 'Risks'"
    disordered = contract.check("## Risks\nx\n# Summary\ny")
    assert disordered.error == (
        "section 'Risks' appears before 'Summary'; required order: Summary, Risks"
    )


def test_markdown_schema_mapping_spelling_matches_the_class() -> None:
    from_mapping = OutputContract(
        format="markdown", schema={"required_sections": ["Summary", "Risks"]}
    )
    from_class = OutputContract(
        format="markdown", schema=MarkdownSchema(required_sections=("Summary", "Risks"))
    )
    text = "# SUMMARY\ny\n## risks\nx"
    assert from_mapping.check(text).ok
    assert from_class.check(text).ok


def test_markdown_section_comparison_is_case_insensitive_and_collapsed() -> None:
    contract = OutputContract(
        format="markdown", schema=MarkdownSchema(required_sections=("Release Notes",))
    )
    assert contract.check("##   release   notes ##\ntext").ok


def test_markdown_sections_match_as_subsequence() -> None:
    """An extra heading between required ones does not break the order rule."""
    contract = OutputContract(
        format="markdown", schema=MarkdownSchema(required_sections=("Summary", "Risks"))
    )
    assert contract.check("# Summary\n## Notes\n## Risks\nx").ok


# ---------------------------------------------------------------------------
# Construction refusals
# ---------------------------------------------------------------------------


def test_unknown_format_refused() -> None:
    with pytest.raises(OutputContractError, match="format must be one of"):
        OutputContract(format="xml")  # type: ignore[arg-type]


@pytest.mark.parametrize("retries", [-1, 6, 100])
def test_retries_range_refused(retries: int) -> None:
    with pytest.raises(OutputContractError, match="between 0 and 5"):
        OutputContract(format="json", retries=retries)


def test_markdown_schema_on_a_structured_format_refused() -> None:
    with pytest.raises(OutputContractError, match="markdown"):
        OutputContract(format="json", schema=MarkdownSchema())


def test_type_schema_on_markdown_refused() -> None:
    with pytest.raises(OutputContractError, match="markdown schema"):
        OutputContract(format="markdown", schema=Invoice)


def test_non_object_json_schema_for_toml_refused() -> None:
    with pytest.raises(OutputContractError, match="table"):
        OutputContract(format="toml", schema={"type": "array"})


def test_toml_schema_without_type_is_allowed() -> None:
    """No declared type is not a declared non-object type."""
    contract = OutputContract(format="toml", schema={"required": ["server"]})
    assert contract.check("[server]\nport = 80").ok
    assert not contract.check("port = 80").ok


def test_invalid_json_schema_refused_at_construction() -> None:
    """The meta-schema check happens before the first prompt, not at validate."""
    with pytest.raises(OutputContractError, match="not a valid JSON Schema"):
        OutputContract(format="json", schema={"type": "nonsense"})


def test_markdown_schema_mapping_requires_exactly_required_sections() -> None:
    with pytest.raises(OutputContractError):
        OutputContract(format="markdown", schema={"required_sections": [1, 2]})


# ---------------------------------------------------------------------------
# Budgets, retry text, event-facing fields
# ---------------------------------------------------------------------------


def test_max_attempts_is_retries_plus_one() -> None:
    assert OutputContract(format="json").max_attempts == 3
    assert OutputContract(format="json", retries=0).max_attempts == 1
    assert OutputContract(format="json", retries=5).max_attempts == 6


def test_label_is_the_format() -> None:
    assert OutputContract(format="yaml").label == "yaml"


def _retry_message(contract: OutputContract, *, attempt: int, error: str) -> Message:
    """``retry_message`` narrowed to the plain user ``Message`` it constructs.

    The declared return type is the ``AgentMessage`` union (a retry is an
    ordinary row beside the host-authored custom entries), and every
    assertion below wants the concrete class: one narrow, one spelling.
    """
    message = contract.retry_message(attempt=attempt, error=error)
    assert isinstance(message, Message)
    return message


def test_retry_message_shape() -> None:
    message = _retry_message(
        OutputContract(format="json"), attempt=2, error="not valid JSON: Expecting value"
    )
    assert message.role == "user"
    text = message.text
    assert text.startswith("Harness output check: not valid JSON: Expecting value\n")
    assert "The required output format is json." in text
    assert "exactly one JSON value" in text
    assert "attempt 2 of 3" in text


def test_retry_message_bounds_and_collapses_the_reason() -> None:
    collapsed = _retry_message(
        OutputContract(format="json"), attempt=1, error="first\nsecond   line"
    )
    assert "Harness output check: first second line\n" in collapsed.text
    long_reason = "x" * 5000 + "\n\nsecond"
    message = _retry_message(OutputContract(format="json"), attempt=1, error=long_reason)
    reason = message.text.split("\n", 1)[0].removeprefix("Harness output check: ")
    assert len(reason) <= 600
    assert reason.endswith("… (truncated)")
    assert "second" not in reason


def test_retry_message_for_markdown_names_required_sections() -> None:
    contract = OutputContract(
        format="markdown", schema=MarkdownSchema(required_sections=("Summary",))
    )
    text = _retry_message(contract, attempt=1, error="missing required section 'Summary'").text
    assert "Reply with the corrected markdown document only." in text
    assert "Required sections, in order: Summary." in text


def test_exhausted_error_text() -> None:
    contract = OutputContract(format="json", retries=2)
    assert contract.exhausted_error(attempts=3, error="boom") == (
        "final response did not satisfy the output contract (json) after 3 attempts: boom"
    )


def test_check_errors_are_bounded() -> None:
    contract = OutputContract(
        format="json",
        schema={"type": "object", "properties": {"x": {"type": "string"}}},
    )
    payload = '{"x": ' + "9" * 4000 + "}"
    check = contract.check(payload)
    assert not check.ok
    assert len(check.error) <= 600


# ---------------------------------------------------------------------------
# The system block (the model-facing announcement, built once)
# ---------------------------------------------------------------------------


def test_system_block_for_json_without_schema() -> None:
    block = OutputContract(format="json").system_block()
    assert block.startswith("Output contract: the final response for this session is enforced.")
    assert "- Format: json." in block
    assert "a fenced ```json block is also accepted" in block


def test_system_block_embeds_the_whole_schema() -> None:
    block = OutputContract(format="json", schema=RAW_SCHEMA).system_block()
    assert "- It must validate against this schema (JSON Schema): {" in block
    assert '"required"' in block


def test_system_block_for_markdown_lists_sections() -> None:
    block = OutputContract(
        format="markdown", schema=MarkdownSchema(required_sections=("Summary", "Risks"))
    ).system_block()
    assert "- Format: markdown." in block
    assert "- Required sections, in order: Summary, Risks." in block


def test_no_contract_is_no_block() -> None:
    """Contract construction is the only source of a block; nothing exists to
    append when there is none (the byte-identical default)."""
    assert OutputContract(format="json").system_block()  # never empty when set


# ---------------------------------------------------------------------------
# decode_output — the public counterpart
# ---------------------------------------------------------------------------


def test_decode_output_uses_the_same_ladder() -> None:
    assert decode_output('prose ```json\n{"a": 1}\n``` tail', "json") == {"a": 1}
    assert decode_output("a: 1\nb: 2", "yaml") == {"a": 1, "b": 2}
    assert decode_output("[server]\nport = 80", "toml") == {"server": {"port": 80}}
    assert decode_output("# Title\ntext", "markdown") == "# Title\ntext"


def test_decode_output_raises_with_the_contract_reason() -> None:
    with pytest.raises(OutputDecodeError, match="not valid JSON"):
        decode_output("no payload here", "json")
    with pytest.raises(OutputDecodeError, match="the final message was empty"):
        decode_output("   ", "json")
    with pytest.raises(OutputDecodeError, match="unterminated"):
        decode_output("x\n```\ncode", "markdown")


def test_decode_output_rejects_an_unknown_format() -> None:
    with pytest.raises(OutputDecodeError, match="format must be one of"):
        decode_output("{}", "xml")


def test_decode_output_and_check_agree_on_the_winning_payload() -> None:
    text = 'Answer:\n```json\n{"a": [1, 2]}\n```'
    contract = OutputContract(format="json")
    check = contract.check(text)
    assert check.ok
    assert decode_output(text, "json") == {"a": [1, 2]}


def test_formats_tuple_is_the_cli_vocabulary() -> None:
    assert OUTPUT_FORMATS == ("markdown", "json", "yaml", "toml")


# ---------------------------------------------------------------------------
# The quiet-end tool under a contract (docs/design/quiet-turns.md §4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_quiet_end_tool_is_absent_under_an_output_contract(tmp_path) -> None:
    """The contract gate reads a textless end as a MISSING final response
    (``loop.py``'s ``final_response_gate``), so ``no_reply`` must not exist in
    a session that sets one: the session binds ``quiet_end`` to None and the
    tool's createIf gate follows (rung 3). Not "present but refused" — a
    refusal would still spend the call, and the model would be asked for a
    payload its own quiet signal had already declined to produce."""
    from local_operator.harness.types import StreamEndEvent
    from local_operator.tools.registry import create_tools
    from tests.unit.session.test_session import ScriptedStream, make_session

    session = make_session(tmp_path, ScriptedStream([[StreamEndEvent(stop_reason="stop")]]))
    try:
        assert callable(session._quiet_end_callable()), "binds before the contract"
        assert "no_reply" in {tool.name for tool in session._tools}, "mounted before"
        session.set_output_contract(OutputContract(format="json"))
        assert session._quiet_end_callable() is None
        assert session._build_tool_context().quiet_end is None
        assert create_tools(session._build_tool_context(), enabled=["no_reply"]) == []
        # The setter runs AFTER construction (exec_startup/sdk), when the
        # constructor's merge has already mounted the tool — so it must drop it
        # from the LIVE inventory, not only null the per-turn door: an
        # advertised tool whose every call could only refuse is exactly the
        # "inert" shape the design forbids.
        assert "no_reply" not in {
            tool.name for tool in session._tools
        }, "a contract must drop the mounted tool"
        # And clearing restores it, because the door reopens: absent and
        # present must follow the capability in both directions.
        session.set_output_contract(None)
        assert callable(session._quiet_end_callable())
        assert "no_reply" in {tool.name for tool in session._tools}, "clearing restores it"
    finally:
        await session.dispose()
