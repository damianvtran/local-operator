"""Tool bridge: name mangling, arg hygiene, result formatting, retriable errors."""

from __future__ import annotations

import base64

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
from local_operator.mcp.tool_bridge import (
    INTENT_FIELD,
    build_agent_tool,
    create_mcp_tool_name,
    format_mcp_result,
    is_retriable_connection_error,
    normalize_input_schema,
    prepare_outbound_args,
)


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

    def test_image_placeholder(self) -> None:
        result = CallToolResult(
            content=[
                ImageContent(
                    type="image",
                    data=base64.b64encode(b"\x89PNG\r\n\x1a\n").decode(),
                    mime_type="image/png",
                )
            ],
            is_error=False,
        )
        assert format_mcp_result(result).text == "[Image: image/png]"

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
        ["tool not found", "invalid arguments", "HTTP 401: unauthorized", "timeout waiting"],
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
