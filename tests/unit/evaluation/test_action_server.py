"""The episode action server: the wire, the tool shape, and the failure contract.

What these tests defend, in the order the module promises it:

* **One statement of the wire.** The frame helpers are what the bridge and the
  server both speak; a protocol skew must fail its FIRST frame loudly, so the
  skew cases are asserted rather than assumed.
* **The offered tool IS the projection.** ``tools/list`` serves
  ``action_tool_parameters`` verbatim -- the same function the loop-driven
  action tool projects -- so the MCP channel cannot offer a contract the other
  channel has moved past.
* **A dead bridge is a tool RESULT.** The model must receive a sentence it can
  act on, never an exception into the session or a hung call.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.evaluation.action_server import (
    SERVER_NAME,
    WIRE_PROTOCOL,
    WIRE_READ_LIMIT_BYTES,
    ActionBridgeUnreachable,
    build_mcp_server,
    decode_call,
    decode_response,
    encode_call,
    encode_response,
    format_forward_error,
    forward_call,
    main,
    surface_from_json,
    surface_to_json,
)
from local_operator.evaluation.action_surface import ActionSurface
from local_operator.evaluation.protocol import MAX_ENVELOPE_BYTES
from local_operator.evaluation.runner.action_tool import (
    ACTION_TOOL_NAME,
    action_tool_parameters,
)
from local_operator.harness.types import ImageContent, TextContent

PNG = b"\x89PNG\r\n\x1a\n"


async def _start_stub(tmp_path: Path) -> tuple[str, asyncio.AbstractServer]:
    """A bridge stub that answers every call with text + an image."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        request = decode_call(await reader.readline())
        assert request["tool"] == ACTION_TOOL_NAME
        writer.write(
            encode_response(
                [
                    TextContent(text=f"ok:{request['arguments']['marker']}"),
                    ImageContent(data=base64.b64encode(PNG).decode("ascii"), mime_type="image/png"),
                ],
                details={"echo": request["arguments"]},
            )
        )
        await writer.drain()
        writer.close()

    endpoint = str(tmp_path / "bridge.sock")
    server = await asyncio.start_unix_server(handle, path=endpoint)
    return endpoint, server


class TestSurfaceSerialization:
    def test_round_trip(self) -> None:
        surface = ActionSurface(paste_text=True, type_text_mode="ascii", max_type_chars=50)
        assert surface_from_json(surface_to_json(surface)) == surface

    def test_missing_field_is_refused(self) -> None:
        payload = json.loads(surface_to_json(ActionSurface()))
        payload.pop("paste_text")
        with pytest.raises(ValueError, match="must name exactly"):
            surface_from_json(json.dumps(payload))

    def test_unknown_field_is_refused(self) -> None:
        payload = json.loads(surface_to_json(ActionSurface()))
        payload["future_field"] = True
        with pytest.raises(ValueError, match="must name exactly"):
            surface_from_json(json.dumps(payload))

    def test_non_object_is_refused(self) -> None:
        with pytest.raises(ValueError):
            surface_from_json("[1, 2]")


class TestFrames:
    def test_call_round_trip(self) -> None:
        frame = encode_call({"actions": []}, call_id="c1")
        assert frame.endswith(b"\n")
        assert decode_call(frame) == {
            "protocol": WIRE_PROTOCOL,
            "tool": ACTION_TOOL_NAME,
            "call_id": "c1",
            "arguments": {"actions": []},
        }

    def test_wrong_protocol_is_refused(self) -> None:
        frame = json.dumps({"protocol": 999, "tool": ACTION_TOOL_NAME, "arguments": {}}).encode()
        with pytest.raises(ValueError, match="protocol"):
            decode_call(frame)

    def test_wrong_tool_is_refused(self) -> None:
        frame = json.dumps({"protocol": WIRE_PROTOCOL, "tool": "other", "arguments": {}}).encode()
        with pytest.raises(ValueError, match="tool"):
            decode_call(frame)

    def test_response_round_trip(self) -> None:
        frame = encode_response([TextContent(text="hi")], is_error=False, details={"a": 1})
        reply = decode_response(frame)
        assert reply["content"] == [{"type": "text", "text": "hi"}]
        assert reply["is_error"] is False
        assert reply["details"] == {"a": 1}

    def test_response_requires_known_block_kinds(self) -> None:
        with pytest.raises(ValueError):
            decode_response(json.dumps({"protocol": WIRE_PROTOCOL, "content": [{}]}).encode())

    def test_response_rejects_wrong_protocol(self) -> None:
        with pytest.raises(ValueError, match="protocol"):
            decode_response(json.dumps({"protocol": 0, "content": []}).encode())


class TestToolShape:
    @pytest.mark.asyncio
    async def test_tool_schema_is_the_projection(self) -> None:
        import mcp.types as types

        surface = ActionSurface(paste_text=True, max_type_chars=200)
        from local_operator.evaluation.runner.action_tool import ACTION_TOOL_DESCRIPTION

        server = build_mcp_server("/nonexistent.sock", surface, timeout=0.1)
        entry = server.get_request_handler("tools/list")
        assert entry is not None
        result = cast(Any, await entry.handler(cast(Any, None), types.PaginatedRequestParams()))
        assert len(result.tools) == 1
        tool = result.tools[0]
        assert tool.name == ACTION_TOOL_NAME
        assert tool.description == ACTION_TOOL_DESCRIPTION
        assert tool.input_schema == action_tool_parameters(surface)

    @pytest.mark.asyncio
    async def test_call_tool_forwards_and_returns_media(self, tmp_path: Path) -> None:
        import mcp.types as types

        endpoint, stub = await _start_stub(tmp_path)
        server = build_mcp_server(endpoint, ActionSurface(), timeout=5.0)
        entry = server.get_request_handler("tools/call")
        assert entry is not None
        try:
            result = cast(
                Any,
                await entry.handler(
                    cast(Any, None),
                    types.CallToolRequestParams(name=ACTION_TOOL_NAME, arguments={"marker": "1"}),
                ),
            )
        finally:
            stub.close()
            await stub.wait_closed()
        assert result.is_error is False
        assert [block.type for block in result.content] == ["text", "image"]
        assert result.content[0].text == "ok:1"
        assert result.content[1].data == base64.b64encode(PNG).decode("ascii")

    @pytest.mark.asyncio
    async def test_unknown_tool_is_refused(self) -> None:
        import mcp.types as types

        server = build_mcp_server("/nonexistent.sock", ActionSurface(), timeout=0.1)
        entry = server.get_request_handler("tools/call")
        assert entry is not None
        result = cast(
            Any,
            await entry.handler(
                cast(Any, None), types.CallToolRequestParams(name="nope", arguments={})
            ),
        )
        assert result.is_error is True

    @pytest.mark.asyncio
    async def test_dead_bridge_becomes_a_tool_error(self, tmp_path: Path) -> None:
        import mcp.types as types

        server = build_mcp_server(str(tmp_path / "missing.sock"), ActionSurface(), timeout=0.1)
        entry = server.get_request_handler("tools/call")
        assert entry is not None
        result = cast(
            Any,
            await entry.handler(
                cast(Any, None),
                types.CallToolRequestParams(name=ACTION_TOOL_NAME, arguments={"actions": []}),
            ),
        )
        assert result.is_error is True
        assert "action bridge" in result.content[0].text


class TestForwardCall:
    @pytest.mark.asyncio
    async def test_forward_call_passes_blocks_through(self, tmp_path: Path) -> None:
        endpoint, stub = await _start_stub(tmp_path)
        try:
            reply = await forward_call(endpoint, {"marker": "9"}, timeout=5.0)
        finally:
            stub.close()
            await stub.wait_closed()
        assert reply["content"][0]["text"] == "ok:9"
        assert reply["details"] == {"echo": {"marker": "9"}}

    @pytest.mark.asyncio
    async def test_a_reply_the_size_of_one_observation_round_trips(self, tmp_path: Path) -> None:
        """Replies carry the observation's images, so they exceed 64 KiB by design.

        Regression, measured on the paid session arm 2026-09-28 (task_017): a
        476 KiB frame produced a reply whose read raised "Separator is not
        found, and chunk exceed the limit" -- asyncio's default line limit --
        so every executed batch was reported to the model as unreachable
        while the desktop had already acted. The connection is opened with
        the wire's own limit; this pins that a reply that size reads back.
        """
        big = base64.b64encode(b"\x00" * (700 * 1024)).decode("ascii")

        async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                await reader.readline()
                writer.write(
                    encode_response(
                        [
                            TextContent(text="observed"),
                            ImageContent(data=big, mime_type="image/png"),
                        ]
                    )
                )
                await writer.drain()
            finally:
                writer.close()

        endpoint = str(tmp_path / "big.sock")
        server = await asyncio.start_unix_server(handle, path=endpoint)
        try:
            reply = await forward_call(endpoint, {}, timeout=5.0)
        finally:
            server.close()
            await server.wait_closed()
        assert reply["content"][0]["text"] == "observed"
        assert reply["content"][1]["data"] == big

    def test_the_read_limit_covers_a_full_protocol_envelope(self) -> None:
        # The sizing rule, pinned: the reader must not be a smaller contract
        # than the protocol's own cap.
        assert WIRE_READ_LIMIT_BYTES >= MAX_ENVELOPE_BYTES

    @pytest.mark.asyncio
    async def test_missing_endpoint_raises_reachability(self, tmp_path: Path) -> None:
        with pytest.raises(ActionBridgeUnreachable):
            await forward_call(str(tmp_path / "missing.sock"), {}, timeout=0.1)

    @pytest.mark.asyncio
    async def test_silent_bridge_times_out(self, tmp_path: Path) -> None:
        async def silent(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            # The handler must CLOSE its own transport: a server whose handler
            # leaves the connection open wedges ``wait_closed()`` on 3.12+, and
            # the fd outlives the test.
            try:
                await reader.readline()
                await asyncio.sleep(0.5)
            finally:
                writer.close()

        endpoint = str(tmp_path / "silent.sock")
        server = await asyncio.start_unix_server(silent, path=endpoint)
        try:
            with pytest.raises(ActionBridgeUnreachable, match="did not answer"):
                await forward_call(endpoint, {}, timeout=0.2)
        finally:
            server.close()
            await server.wait_closed()

    def test_format_forward_error_names_the_cause(self) -> None:
        blocks, is_error = format_forward_error(ActionBridgeUnreachable("boom"))
        assert is_error is True
        assert "boom" in blocks[0]["text"]


class TestMain:
    def test_bad_surface_exits_before_serving(self, tmp_path: Path) -> None:
        assert main(["--endpoint", str(tmp_path / "x.sock"), "--surface", "{}"]) == 2

    def test_server_name_is_the_declared_one(self) -> None:
        assert SERVER_NAME == "episode-actions"
