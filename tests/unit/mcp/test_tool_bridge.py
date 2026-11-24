"""Tool bridge: name mangling, arg hygiene, result formatting, retriable errors."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
    Tool,
)

from local_operator.harness.intent import INTENT_PROPERTY
from local_operator.harness.types import CapturedFrame
from local_operator.harness.types import ImageContent as HarnessImageContent
from local_operator.harness.types import ToolContext
from local_operator.mcp.tool_bridge import (
    INTENT_FIELD,
    MAX_NOTE_DECLARED_NAMES,
    MAX_NOTE_NAME_CHARS,
    MAX_NOTE_REJECTED_NAMES,
    build_agent_tool,
    capture_producer_scope,
    clean_outbound_args,
    create_mcp_tool_name,
    format_mcp_result,
    is_retriable_connection_error,
    normalize_input_schema,
    outbound_args_note,
    prepare_outbound_args,
)
from local_operator.tools import spill


class TestCreateMcpToolName:
    def test_basic_shape(self) -> None:
        assert create_mcp_tool_name("linear", "create_issue") == "mcp__linear_create_issue"

    def test_redundant_server_prefix_stripped(self) -> None:
        assert (
            create_mcp_tool_name("puppeteer", "puppeteer_screenshot") == "mcp__puppeteer_screenshot"
        )

    def test_sanitization_lowercase_runs_edges(self) -> None:
        # Non-[a-z_] runs (incl. digits) collapse to one underscore; edges
        # trimmed; lowercase.
        assert create_mcp_tool_name("My-Server 2", "Do The Thing!") == "mcp__my_server_do_the_thing"

    def test_prefix_strip_after_sanitization(self) -> None:
        # Sanitization must run before the prefix check: 'Git Hub' + 'git_hub_status'.
        assert create_mcp_tool_name("Git Hub", "git_hub_status") == "mcp__git_hub_status"

    def test_empty_parts_fall_back(self) -> None:
        assert create_mcp_tool_name("!!!", "???") == "mcp__server_tool"

    def test_partial_prefix_not_stripped(self) -> None:
        # 'puppeteer' is not a prefix+underscore of 'puppet_screenshot'.
        assert (
            create_mcp_tool_name("puppeteer", "puppet_screenshot")
            == "mcp__puppeteer_puppet_screenshot"
        )


class TestNormalizeInputSchema:
    def test_ensures_object_and_defaults(self) -> None:
        assert normalize_input_schema(None) == {"type": "object", "properties": {}, "required": []}
        assert normalize_input_schema({"type": "string"})["type"] == "object"

    def test_preserves_existing(self) -> None:
        schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
        assert normalize_input_schema(schema) == schema


class TestPrepareOutboundArgs:
    def test_strips_undeclared_intent_when_strict(self) -> None:
        """The harness-injected 'i' field must not reach a strict server."""
        out = prepare_outbound_args(
            {INTENT_FIELD: "why", "query": "x"},
            {"query": {"type": "string"}},
            required=["query"],
            additional_properties=False,
        )
        assert out == {"query": "x"}

    def test_strips_any_undeclared_when_additional_false(self) -> None:
        """additionalProperties explicitly false: drop every undeclared key."""
        out = prepare_outbound_args(
            {"a": 1, "stray": 2},
            {"a": {"type": "number"}},
            required=[],
            additional_properties=False,
        )
        assert out == {"a": 1}

    def test_absent_additional_properties_keeps_undeclared(self) -> None:
        """MCP-04 blocker: absent additionalProperties is OPEN per JSON
        Schema — arbitrary undeclared keys must survive, or composed/open
        servers would silently run calls with {}."""
        out = prepare_outbound_args({"path": "/x", "limit": 5}, {}, [], None)
        assert out == {"path": "/x", "limit": 5}

    def test_empty_additional_properties_keeps_undeclared(self) -> None:
        """MCP-14: additionalProperties {} is an open sub-schema."""
        assert prepare_outbound_args({"a": 1}, {}, [], {}) == {"a": 1}

    def test_open_schema_keeps_undeclared_drops_intent(self) -> None:
        """Open schema: arbitrary undeclared keys survive; the harness 'i'
        field still drops unless the server declares it."""
        out = prepare_outbound_args(
            {"a": 1, INTENT_FIELD: "why"},
            {"a": {"type": "number"}},
            required=[],
            additional_properties=True,
        )
        assert out == {"a": 1}

    def test_intent_stripped_even_when_open(self) -> None:
        """'i' is the harness's field: dropped unless the server declares it,
        regardless of how permissive the schema is."""
        out = prepare_outbound_args(
            {INTENT_FIELD: "why", "q": "x"}, {"q": {"type": "string"}}, [], None
        )
        assert out == {"q": "x"}
        out = prepare_outbound_args({INTENT_FIELD: "why"}, {}, [], True)
        assert out == {}

    def test_keeps_declared_intent(self) -> None:
        """A server that declares 'i' as its own parameter is unaffected."""
        out = prepare_outbound_args(
            {INTENT_FIELD: "why"},
            {INTENT_FIELD: {"type": "string"}},
            required=[INTENT_FIELD],
            additional_properties=False,
        )
        assert out == {INTENT_FIELD: "why"}

    def test_drops_empty_optionals(self) -> None:
        out = prepare_outbound_args(
            {"keep": "v", "empty_str": "", "empty_dict": {}, "none_val": None, "req_empty": ""},
            {
                "keep": {"type": "string"},
                "empty_str": {"type": "string"},
                "empty_dict": {"type": "object"},
                "none_val": {"type": "string"},
                "req_empty": {"type": "string"},
            },
            required=["req_empty"],
            additional_properties=False,
        )
        assert out == {"keep": "v", "req_empty": ""}

    def test_required_placeholder_kept(self) -> None:
        assert prepare_outbound_args({"r": None}, {"r": {}}, ["r"], True) == {"r": None}

    def test_falsy_is_not_placeholder(self) -> None:
        assert prepare_outbound_args(
            {"n": 0, "f": False, "l": []},
            {"n": {}, "f": {}, "l": {}},
            [],
            True,
        ) == {"n": 0, "f": False, "l": []}

    def test_non_dict_input(self) -> None:
        assert prepare_outbound_args(None, {"a": {}}) == {}

    def test_input_not_mutated(self) -> None:
        args = {"a": 1, INTENT_FIELD: "why"}
        prepare_outbound_args(args, {"a": {}}, [], False)
        assert args == {"a": 1, INTENT_FIELD: "why"}


class TestRejectedOutboundArgs:
    """The channel beside the cleaned dict: which model-supplied keys a strict
    schema made the bridge withhold, and the note a result carries for them.

    The recorded case: a worker passed ``timeout`` to a tool declaring
    ``timeout_s``; the key was dropped silently, the 60 s default applied, six
    calls timed out and nothing told the model its argument never arrived.
    """

    STRICT = {"command": {"type": "string"}, "timeout_s": {"type": "integer"}}

    def test_undeclared_model_key_under_strict_schema_is_reported(self) -> None:
        out = clean_outbound_args(
            {"command": "sleep 90", "timeout": 120}, self.STRICT, ["command"], False
        )
        assert out.cleaned == {"command": "sleep 90"}  # still dropped, as before
        assert out.rejected == ("timeout",)
        note = outbound_args_note(out.rejected, self.STRICT)
        assert note is not None
        assert "timeout" in note and "NOT sent" in note
        # The declared names are what lets the model correct itself.
        assert "Declared arguments: command, timeout_s." in note

    def test_intent_field_drop_is_not_reported(self) -> None:
        """The harness added ``i``; the model made no mistake."""
        out = clean_outbound_args(
            {"command": "ls", INTENT_FIELD: "Listing files"}, self.STRICT, ["command"], False
        )
        assert out.cleaned == {"command": "ls"}
        assert out.rejected == ()
        assert outbound_args_note(out.rejected, self.STRICT) is None

    @pytest.mark.parametrize("placeholder", [None, "", {}])
    def test_placeholder_drops_are_not_reported(self, placeholder: object) -> None:
        """Neither a declared optional placeholder nor an undeclared one: an
        empty value is a slot the model did not mean to fill."""
        out = clean_outbound_args(
            {"command": "ls", "timeout_s": placeholder, "timeout": placeholder},
            self.STRICT,
            ["command"],
            False,
        )
        assert out.cleaned == {"command": "ls"}
        assert out.rejected == ()

    @pytest.mark.parametrize("additional", [None, True, {}])
    def test_non_strict_schema_forwards_and_reports_nothing(self, additional: Any) -> None:
        out = clean_outbound_args({"command": "ls", "timeout": 120}, self.STRICT, [], additional)
        assert out.cleaned == {"command": "ls", "timeout": 120}
        assert out.rejected == ()

    def test_cleaned_matches_the_dict_entry_point(self) -> None:
        """One pass feeds both: the dict callers get is the one reported on."""
        args = {"command": "ls", "timeout": 5, "timeout_s": "", INTENT_FIELD: "why"}
        assert clean_outbound_args(
            args, self.STRICT, ["command"], False
        ).cleaned == prepare_outbound_args(args, self.STRICT, ["command"], False)

    def test_non_dict_input_reports_nothing(self) -> None:
        out = clean_outbound_args(None, self.STRICT, [], False)
        assert out.cleaned == {} and out.rejected == ()

    def test_note_names_every_rejected_key_in_plural(self) -> None:
        note = outbound_args_note(("timeout", "cwd"), self.STRICT)
        assert note is not None
        assert "arguments not declared by this tool were NOT sent: timeout, cwd." in note
        assert "was sent without them" in note

    def test_note_is_bounded(self) -> None:
        """A server may declare dozens of properties and a model may invent
        any number of keys: both lists are cut and the remainder counted."""
        declared = {f"p{n}": {} for n in range(MAX_NOTE_DECLARED_NAMES + 5)}
        rejected = tuple(f"x{n}" for n in range(MAX_NOTE_REJECTED_NAMES + 3)) + ("y" * 500,)
        note = outbound_args_note(rejected, declared)
        assert note is not None
        assert "(+4 more)" in note  # rejected: 3 extra + the long one
        assert "(+5 more)" in note  # declared
        assert f"p{MAX_NOTE_DECLARED_NAMES}" not in note
        assert "y" * (MAX_NOTE_NAME_CHARS + 1) not in note
        assert len(note) < 1200

    def test_long_name_is_cut(self) -> None:
        note = outbound_args_note(("z" * 500,), {})
        assert note is not None
        assert "z" * MAX_NOTE_NAME_CHARS + "..." in note
        assert "z" * (MAX_NOTE_NAME_CHARS + 1) not in note
        assert "Declared arguments: (none)." in note


class TestFormatMcpResult:
    def test_text_joined(self) -> None:
        result = CallToolResult(
            content=[
                TextContent(type="text", text="one"),
                TextContent(type="text", text="two"),
            ],
            is_error=False,
        )
        formatted = format_mcp_result(result, "id1", "mcp__srv_tool")
        assert formatted.text == "one\n\ntwo"
        assert formatted.is_error is False
        assert formatted.tool_call_id == "id1"
        assert formatted.tool_name == "mcp__srv_tool"

    def test_image_block_is_inlined(self) -> None:
        """A decodable image under the cap reaches the model AS an image block.

        This is the property the benchmark arm depends on: a computer-use
        observation comes back through an MCP tool result, and the model must
        SEE the frame, not a caption of it.
        """
        payload = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        result = CallToolResult(
            content=[ImageContent(type="image", data=payload, mime_type="image/png")],
            is_error=False,
        )
        formatted = format_mcp_result(result)
        assert len(formatted.content) == 1
        block = formatted.content[0]
        assert block.type == "image"
        assert block.data == payload
        assert block.mime_type == "image/png"
        # No text block was invented for an image-only result.
        assert formatted.text == ""

    def test_image_after_text_caption(self) -> None:
        """Text leads, images follow -- the order a local image read uses."""
        payload = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        result = CallToolResult(
            content=[
                TextContent(type="text", text="step 1"),
                ImageContent(type="image", data=payload, mime_type="image/png"),
            ],
            is_error=False,
        )
        formatted = format_mcp_result(result)
        assert [block.type for block in formatted.content] == ["text", "image"]
        assert formatted.text == "step 1"

    def test_oversized_image_falls_back_to_placeholder(self) -> None:
        """Past the cap the payload is never inlined; the model gets a caption."""
        from local_operator.imaging import IMAGE_MAX_BYTES

        payload = base64.b64encode(b"x" * (IMAGE_MAX_BYTES + 1)).decode()
        result = CallToolResult(
            content=[ImageContent(type="image", data=payload, mime_type="image/png")],
            is_error=False,
        )
        formatted = format_mcp_result(result)
        assert [block.type for block in formatted.content] == ["text"]
        assert formatted.text == "[Image: image/png]"

    def test_undecodable_image_falls_back_to_placeholder(self) -> None:
        """Garbage must not enter history as an image the provider will 400 on."""
        result = CallToolResult(
            content=[ImageContent(type="image", data="not base64!!", mime_type="image/png")],
            is_error=False,
        )
        formatted = format_mcp_result(result)
        assert [block.type for block in formatted.content] == ["text"]
        assert formatted.text == "[Image: image/png]"

    def test_dict_image_inlines_under_snake_case_mime(self) -> None:
        """Cached/raw-dict results carry the snake_case mime key too."""
        payload = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        result = {
            "content": [{"type": "image", "data": payload, "mime_type": "image/png"}],
            "isError": False,
        }
        formatted = format_mcp_result(result)
        assert [block.type for block in formatted.content] == ["image"]
        image = formatted.content[0]
        # The HARNESS type, not mcp.types': the bridge rewrites the SDK shape
        # into the one the loop carries.
        assert isinstance(image, HarnessImageContent)
        assert image.data == payload

    def test_capture_claim_rides_into_harness_provenance_with_harness_scope(self) -> None:
        """A claim in ``_meta`` stamps CapturedFrame; identity stays the harness's.

        The producer's own ``producer_scope`` (forged here) must be ignored:
        compaction may only ever supersede captures within the ACTUAL producer
        scope the harness composed at the call site.
        """
        payload = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        result = CallToolResult(
            content=[
                ImageContent(
                    type="image",
                    data=payload,
                    mime_type="image/png",
                    _meta={
                        "local_operator/capture": {
                            "stream_id": "screen",
                            "revision_id": "7",
                            "producer_scope": "forge-me",
                        }
                    },
                )
            ],
            is_error=False,
        )
        formatted = format_mcp_result(result, producer_scope="mcp:sess:osworld:copy")
        image = formatted.content[0]
        assert isinstance(image, HarnessImageContent)
        assert image.captured_frame == CapturedFrame(
            stream_id="screen", revision_id="7", producer_scope="mcp:sess:osworld:copy"
        )

    def test_capture_claims_absent_or_malformed_keep_images_protected(self) -> None:
        """Every bad claim degrades to PROTECTED, never to a stamped identity."""
        payload = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        cases = [
            {},  # no meta at all
            {"_meta": {"local_operator/capture": "not-a-dict"}},
            {"_meta": {"local_operator/capture": {"stream_id": "", "revision_id": "r"}}},
            {"_meta": {"local_operator/capture": {"stream_id": "s", "revision_id": "r" * 257}}},
            {"_meta": {"local_operator/capture": {"stream_id": "s", "revision_id": 7}}},
        ]
        for extra in cases:
            block = {"type": "image", "data": payload, "mime_type": "image/png", **extra}
            formatted = format_mcp_result(
                {"content": [block]}, producer_scope="mcp:sess:osworld:copy"
            )
            image = formatted.content[0]
            assert isinstance(image, HarnessImageContent)
            assert image.captured_frame is None
        # And with NO scope, even a valid claim must not stamp: without a real
        # producer identity the image stays protected.
        claimed = {
            "type": "image",
            "data": payload,
            "mime_type": "image/png",
            "_meta": {"local_operator/capture": {"stream_id": "s", "revision_id": "r"}},
        }
        formatted = format_mcp_result({"content": [claimed]})
        stamped = formatted.content[0]
        assert isinstance(stamped, HarnessImageContent)
        assert stamped.captured_frame is None

    def test_capture_producer_scope_composes_from_actual_origin_only(self) -> None:
        assert (
            capture_producer_scope("osworld", "screenshot", "abc123")
            == "mcp:abc123:osworld:screenshot"
        )
        assert capture_producer_scope("osworld", "screenshot", "") is None
        assert capture_producer_scope("", "screenshot", "abc123") is None
        assert capture_producer_scope("s" * 200, "t" * 200, "abc") is None

    def test_resource_uri_plus_text(self) -> None:
        """A text resource contributes its body; a blob resource is URI-only."""
        result = CallToolResult(
            content=[
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri="file:///a", text="body", mime_type="text/plain"
                    ),
                ),
                EmbeddedResource(
                    type="resource",
                    resource=BlobResourceContents(
                        uri="file:///b",
                        blob=base64.b64encode(b"\x00\x01\x02").decode(),
                        mime_type="application/octet-stream",
                    ),
                ),
            ],
            is_error=False,
        )
        assert (
            format_mcp_result(result).text == "[Resource: file:///a]\nbody\n\n[Resource: file:///b]"
        )

    def test_is_error_prefix_and_flag(self) -> None:
        result = CallToolResult(content=[TextContent(type="text", text="boom")], is_error=True)
        formatted = format_mcp_result(result)
        assert formatted.is_error is True
        assert formatted.text == "Error: boom"

    def test_dict_result_shape(self) -> None:
        """Cached/serialized results (plain dicts) flatten identically."""
        result = {
            "content": [{"type": "text", "text": "hi"}],
            "isError": True,
        }
        formatted = format_mcp_result(result)
        assert formatted.is_error is True
        assert formatted.text == "Error: hi"

    def test_empty_content(self) -> None:
        assert format_mcp_result(CallToolResult(content=[], is_error=False)).text == ""

    def test_details_carry_the_server_payload(self) -> None:
        """``details['server_result']`` round-trips the server's own result."""
        result = CallToolResult(content=[TextContent(type="text", text="ok")], is_error=False)
        details = format_mcp_result(result).details
        assert details is not None
        assert details["server_result"] == result.model_dump()

    def test_small_contextual_result_is_unchanged(self, tmp_path, monkeypatch) -> None:
        """Adding spill context is a no-op below the established 8 KiB budget."""
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
        result = {
            "content": [{"type": "text", "text": "ok"}],
            "structuredContent": {"rows": [{"id": 1}]},
            "isError": False,
            "meta": {"cursor": "next"},
        }
        formatted = format_mcp_result(result, context=ToolContext(session_id="small"))
        assert formatted.text == "ok"
        assert formatted.details == {"server_result": result}

    def test_spill_removes_only_duplicate_text_and_recovers_it(self, tmp_path, monkeypatch) -> None:
        """Spilled text has one durable copy while structured/media metadata survives."""
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
        body = "MCP evidence line\n" * 1_000
        result = {
            "content": [
                {"type": "text", "text": body},
                {"type": "image", "data": "pixels", "mimeType": "image/png"},
                {
                    "type": "resource",
                    "resource": {"uri": "file:///report", "text": body, "mimeType": "text/plain"},
                },
            ],
            "structuredContent": {"rows": [{"id": 1, "value": "kept"}]},
            "isError": True,
            "meta": {"cursor": "next"},
        }
        formatted = format_mcp_result(
            result, "call", "mcp__demo_large", ToolContext(session_id="mcp-spill")
        )
        details = formatted.details
        assert details is not None
        handle = details["spill"]["handle"]
        assert formatted.is_error is True
        assert formatted.text.startswith("Error: MCP evidence line")
        assert handle in formatted.text
        assert details["server_result"] == {
            "content": [
                {"type": "image", "data": "pixels", "mimeType": "image/png"},
                {
                    "type": "resource",
                    "resource": {"uri": "file:///report", "mimeType": "text/plain"},
                },
            ],
            "structuredContent": {"rows": [{"id": 1, "value": "kept"}]},
            "isError": True,
            "meta": {"cursor": "next"},
        }
        stored = spill.get_store().read_lines(handle, 1, 2)
        assert stored is not None
        lines, total = stored
        assert lines == ["Error: MCP evidence line", "MCP evidence line"]
        assert total > 1_000

    @staticmethod
    def _bd_fetch_result() -> tuple[dict[str, Any], str]:
        """A result at bd_tools' own 12,000-char cap: the benchmark's commonest fetch.

        One JSON line, the shape every MCP JSON result has (no line structure).
        """
        # Compact separators: the shape an MCP server emits and a fixed point
        # of ``mcp.result_normalization`` (no long whitespace-bearing leaf), so
        # these spill-bound assertions keep measuring the spill path alone.
        frame = json.dumps(
            {"url": "https://example.test/judges", "text": ""}, separators=(",", ":")
        )
        page = json.dumps(
            {"url": "https://example.test/judges", "text": "J" * (12_000 - len(frame))},
            separators=(",", ":"),
        )
        assert len(page) == 12_000
        return {"content": [{"type": "text", "text": page}], "isError": False}, page

    def test_without_read_a_result_at_the_servers_cap_arrives_whole(
        self, tmp_path, monkeypatch
    ) -> None:
        """REPRODUCTION (benchmark, 2026-11-20): under ``--tools mcp__bd_search,
        mcp__bd_fetch`` lop cut 62 of 159 MCP results at 8 KiB and pointed at a
        ``spill://`` handle the run had no ``read`` to open, so the model fetched
        the handle with its MCP ``fetch`` or re-searched. A session that cannot
        read spills now gets the result inline, with no route it cannot take."""
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
        result, page = self._bd_fetch_result()
        assert len(page) > 8 * 1024, "precondition: over the readable-session cap"
        context = ToolContext(session_id="no-read", spill_readable=False)

        formatted = format_mcp_result(result, "call", "mcp__bd_fetch", context)

        assert formatted.text == page
        assert "spill://" not in formatted.text and "read(" not in formatted.text
        assert formatted.details == {"server_result": result}

    def test_without_read_an_oversized_result_names_no_unreachable_route(
        self, tmp_path, monkeypatch
    ) -> None:
        from local_operator.tools.builtin import (
            BASH_TRUNCATION_MARKER,
            UNREADABLE_SPILL_INLINE_LIMIT_CHARS,
        )

        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
        body = "row\n" * 20_000  # 80,000 chars: past even the wider bound
        result = {"content": [{"type": "text", "text": body}], "isError": False}
        context = ToolContext(session_id="no-read-big", spill_readable=False)

        formatted = format_mcp_result(result, "call", "mcp__bd_fetch", context)

        assert len(formatted.text) <= UNREADABLE_SPILL_INLINE_LIMIT_CHARS
        assert BASH_TRUNCATION_MARKER in formatted.text
        assert "spill://" not in formatted.text and "read(" not in formatted.text
        # The full result is still STORED, for the harness rather than the
        # model (passage ranking, monitors, renderers: review round 1, R2), and
        # the de-duplicated ``server_result`` rides along as it does with read.
        details = formatted.details or {}
        assert details["spill"]["complete"] is True
        assert "server_result" in details

    def test_with_read_the_same_result_keeps_todays_cap_and_footer(
        self, tmp_path, monkeypatch
    ) -> None:
        """Byte-for-byte the shape that shipped: ``spill_readable`` defaults True,
        and an explicit True is the same as a context that predates the field."""
        from local_operator.tools.builtin import spill_truncate

        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
        result, page = self._bd_fetch_result()
        readable = ToolContext(session_id="with-read", spill_readable=True)

        formatted = format_mcp_result(result, "call", "mcp__bd_fetch", readable)
        expected, _details = spill_truncate(page, "mcp__bd_fetch", readable)

        assert ToolContext(session_id="default").spill_readable is True
        assert formatted.text == expected
        assert "spill://" in formatted.text and 'read(path="spill://' in formatted.text
        assert len(formatted.text) < len(page), "cut at the 8 KiB cap, as before"


class TestIsRetriableConnectionError:
    @pytest.mark.parametrize(
        "message",
        [
            "connect ECONNREFUSED 127.0.0.1:3000",
            "read ECONNRESET",
            "Transport closed",
            "transport not connected",
            "HTTP 404: session not found",
            "HTTP 502: bad gateway",
            "HTTP 503: service unavailable",
            "fetch failed",
            "network error during request",
        ],
    )
    def test_retriable(self, message: str) -> None:
        assert is_retriable_connection_error(RuntimeError(message)) is True

    @pytest.mark.parametrize(
        "message",
        ["tool not found", "invalid arguments", "timeout waiting"],
    )
    def test_not_retriable(self, message: str) -> None:
        assert is_retriable_connection_error(RuntimeError(message)) is False


class TestBuildAgentTool:
    def test_wraps_sdk_tool_model(self) -> None:
        mcp_tool = Tool(
            name="search",
            description="Search things",
            input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        )

        async def call_fn(*args, **kwargs):  # pragma: no cover - not invoked here
            raise AssertionError

        tool = build_agent_tool("linear", mcp_tool, call_fn)
        assert tool.name == "mcp__linear_search"
        assert tool.label == "linear/search"
        assert tool.description == "Search things"
        assert tool.parameters["type"] == "object"
        # The harness injects `i` FIRST and leaves the server's own properties
        # after it, untouched.
        assert tool.parameters["properties"] == {
            INTENT_FIELD: dict(INTENT_PROPERTY),
            "q": {"type": "string"},
        }
        assert list(tool.parameters["properties"]) == [INTENT_FIELD, "q"]
        assert INTENT_FIELD not in tool.parameters.get("required", [])

    def test_wraps_cached_dict(self) -> None:
        entry = {"name": "ping", "description": "", "inputSchema": None}

        async def call_fn(*args, **kwargs):  # pragma: no cover
            raise AssertionError

        tool = build_agent_tool("srv", entry, call_fn)
        assert tool.name == "mcp__srv_ping"
        assert tool.parameters == {
            "type": "object",
            "properties": {INTENT_FIELD: dict(INTENT_PROPERTY)},
            "required": [],
        }


def test_auth_challenges_are_retriable_once() -> None:
    """Mid-call hosted MCP auth challenges must enter the manager's existing
    one-reconnect/one-retry path; otherwise a refreshed OAuth token is never
    used until the next user call."""
    from local_operator.mcp.tool_bridge import is_retriable_connection_error

    assert is_retriable_connection_error(RuntimeError("HTTP 401 Unauthorized"))
    assert is_retriable_connection_error(RuntimeError("WWW-Authenticate: Bearer scope=x"))
    assert is_retriable_connection_error(RuntimeError("mcp/www_authenticate challenge"))


_FORGE_WIKI_FETCH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "mcp_results"
    / "forge_fetch_wikipedia_mark_sutcliffe.json"
)

# A Postmedia-shaped page head: tracking pixel, share icons and nav bullets,
# the markup that dominated the measured run's fetched news pages.
_POSTMEDIA_HEAD = "\n".join(
    ["![](https://sb.scorecardresearch.com/p?c1=2&c2=10276888)"]
    + [
        f"![{alt}](https://dcs-static.gprod.postmedia.digital/22.2.2/websites/images/"
        f"share-icons/icon-{i}.svg)"
        for i, alt in enumerate(["", "Close", "", "Reddit", "X", "", "Share via email"] * 3)
    ]
    + [
        "* [News](https://ottawacitizen.com/category/news/)",
        "* [Local News](/category/news/local-news/)",
    ]
    + [
        "",
        "Mark Sutcliffe was sworn in as the 59th mayor of Ottawa on Nov. 15, 2022, after a "
        "campaign built on [affordability](https://ottawacitizen.com/tag/affordability) "
        "and transit.",
    ]
)


def _text_result(text: str, *, is_error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


class TestFormatMcpResultNormalization:
    """``format_mcp_result`` applies ``mcp.result_normalization`` once, at creation."""

    def test_recorded_forge_fetch_shrinks_and_keeps_cited_facts(self) -> None:
        raw = _FORGE_WIKI_FETCH.read_text(encoding="utf-8")
        result = _text_result(raw)
        formatted = format_mcp_result(result, "c1", "mcp__forge_tools_fetch")
        # Measured on this payload: 12,389 -> 6,748 chars under prose-links.
        assert len(formatted.text) <= 0.6 * len(raw)
        parsed = json.loads(formatted.text)
        assert parsed["url"] == json.loads(raw)["url"]  # the citation key, byte-equal
        for fact in ("59th", "mayor of Ottawa", "July 14, 1968"):
            assert fact in parsed["content"]
        assert "#cite_note" not in parsed["content"]
        assert formatted.details is not None
        assert formatted.details["server_result"]["content"][0]["text"] == raw

    def test_icon_images_and_nav_bullets_are_removed(self) -> None:
        formatted = format_mcp_result(_text_result(_POSTMEDIA_HEAD))
        assert "postmedia.digital" not in formatted.text
        assert "scorecardresearch" not in formatted.text
        assert "Local News" not in formatted.text
        assert "59th mayor of Ottawa" in formatted.text
        assert "Reddit" in formatted.text  # alt text is content, kept

    def test_strip_links_mode_drops_prose_link_targets(self) -> None:
        prose = format_mcp_result(_text_result(_POSTMEDIA_HEAD), normalization="prose-links").text
        stripped = format_mcp_result(
            _text_result(_POSTMEDIA_HEAD), normalization="strip-links"
        ).text
        assert "(https://ottawacitizen.com/tag/affordability)" in prose
        assert "ottawacitizen.com" not in stripped
        assert "affordability and transit" in stripped

    def test_off_is_byte_identical_to_unnormalized_flattening(self) -> None:
        raw = _FORGE_WIKI_FETCH.read_text(encoding="utf-8")
        result = {
            "content": [{"type": "text", "text": raw}, {"type": "text", "text": _POSTMEDIA_HEAD}]
        }
        assert format_mcp_result(result, normalization="off").text == f"{raw}\n\n{_POSTMEDIA_HEAD}"

    def test_error_results_are_never_normalized(self) -> None:
        formatted = format_mcp_result(_text_result(_POSTMEDIA_HEAD, is_error=True))
        assert formatted.text == f"Error: {_POSTMEDIA_HEAD}"

    def test_spill_stores_the_normalized_text(self, tmp_path, monkeypatch) -> None:
        """Normalized BEFORE the cut, so the inline head and the spill carry no
        markup; the duplicate raw text is still dropped from ``server_result``
        (one durable copy per large result, as before this change)."""
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
        raw = (_POSTMEDIA_HEAD + "\n\n") * 60
        formatted = format_mcp_result(
            _text_result(raw), "call", "mcp__demo_page", ToolContext(session_id="norm-spill")
        )
        details = formatted.details
        assert details is not None
        assert "postmedia.digital" not in formatted.text  # inline head is normalized
        read = spill.get_store().read_lines(details["spill"]["handle"])
        assert read is not None
        stored = "\n".join(read[0])
        assert "59th mayor of Ottawa" in stored and "postmedia.digital" not in stored
        assert details["server_result"]["content"] == []

    def test_spill_of_an_unchanged_text_still_drops_the_duplicate(
        self, tmp_path, monkeypatch
    ) -> None:
        """No normalization delta: the existing de-duplication is unchanged."""
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
        body = "MCP evidence line\n" * 1_000
        formatted = format_mcp_result(
            _text_result(body.rstrip("\n")),
            "call",
            "mcp__demo_plain",
            ToolContext(session_id="plain"),
        )
        assert formatted.details is not None
        assert "spill" in formatted.details
        assert formatted.details["server_result"]["content"] == []  # text-only: dropped


class TestResultNormalizationSetting:
    """``read_result_normalization_mode`` reads the key per call and never raises."""

    @staticmethod
    def _write_config(tmp_path, monkeypatch, body: str | None) -> None:
        cfg = tmp_path / "cfg"
        cfg.mkdir()
        if body is not None:
            (cfg / "config.yml").write_text(body, encoding="utf-8")
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(cfg))

    def test_absent_key_is_the_default(self, tmp_path, monkeypatch) -> None:
        from local_operator.mcp.manager import read_result_normalization_mode

        self._write_config(tmp_path, monkeypatch, None)
        assert read_result_normalization_mode() == "prose-links"

    @pytest.mark.parametrize("mode", ["off", "strip-links", "prose-links"])
    def test_configured_mode_is_read(self, tmp_path, monkeypatch, mode: str) -> None:
        from local_operator.mcp.manager import read_result_normalization_mode

        self._write_config(
            tmp_path, monkeypatch, f'values:\n  mcp:\n    result_normalization: "{mode}"\n'
        )
        assert read_result_normalization_mode() == mode

    def test_unknown_value_falls_back_to_the_default(self, tmp_path, monkeypatch) -> None:
        from local_operator.mcp.manager import read_result_normalization_mode

        self._write_config(
            tmp_path, monkeypatch, "values:\n  mcp:\n    result_normalization: everything\n"
        )
        assert read_result_normalization_mode() == "prose-links"

    def test_bare_yaml_off_means_off(self, tmp_path, monkeypatch) -> None:
        """YAML 1.1 parses an unquoted ``off`` as False; it still means off."""
        from local_operator.mcp.manager import read_result_normalization_mode

        self._write_config(
            tmp_path, monkeypatch, "values:\n  mcp:\n    result_normalization: off\n"
        )
        assert read_result_normalization_mode() == "off"

    def test_settings_registry_write_is_what_the_reader_sees(self, tmp_path, monkeypatch) -> None:
        """The /settings writer and this reader agree on the path."""
        from local_operator import settings_io
        from local_operator.config import ConfigManager
        from local_operator.mcp.manager import read_result_normalization_mode

        self._write_config(tmp_path, monkeypatch, None)
        settings_io.write_setting(
            ConfigManager(tmp_path / "cfg"), settings_io.BY_KEY["mcp.result_normalization"], "off"
        )
        assert read_result_normalization_mode() == "off"
