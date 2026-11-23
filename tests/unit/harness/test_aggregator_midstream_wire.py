"""End-to-end over a REAL loopback gateway: the aggregator upstream cut.

The loop and driver tests next door hand ``stream_with_failover`` a client that
raises the recorded failure in process. This one replaces that stand-in with an
ACTUAL OpenAI-compatible HTTP server that sends partial output and then an
in-band error. It exercises the original relay-502 frame and a synthetic
code-less frame reproducing the diagnostic reported on Radient auto. That
covers the shapes that only meet on the wire: the SSE parser,
the client's ``_compat_stream_error`` composition, the driver's ``forwarded_any``
marking, the loop's composed continuation instruction, and the re-issued call's
real side effect.

Only two things stand in, and neither is under test: a one-bearer credential
store, and the gateway itself. There is NO external network — the server binds
loopback on an ephemeral port and is torn down with the test — and the ``write``
tool really writes, so the re-issued call has something to prove. Committed as
the permanent guard for the shape the PR body's ad-hoc rig first showed.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from local_operator.harness.loop import AgentLoop, LoopContext
from local_operator.harness.rows import is_harness_chrome
from local_operator.harness.types import (
    AbortSignal,
    AgentEndEvent,
    AgentTool,
    ChatRequest,
    LoopConfig,
    Message,
    NoticeEvent,
    TextContent,
    ToolResult,
)
from local_operator.model.configure import build_model_spec
from local_operator.providers.clients import client_for_spec
from local_operator.providers.failover import ProviderError, stream_with_failover

MODEL_ID = "deepseek/deepseek-v4.1-flash"

#: The upstream body the incident's gateway relayed: this one is marker-bearing,
#: so the test would pass on the marker path alone; the marker-less variant is
#: covered in ``tests/unit/providers/test_failover.py``. Here the point is the
#: wire, not the classifier.
_UPSTREAM_RAW = json.dumps(
    {
        "error": {
            "message": (
                "Upstream error from Together: Stream error: h2 protocol error: "
                "error reading a body from connection"
            )
        }
    }
)

_H2_BODY_READ_ERROR = "Stream error: h2 protocol error: error reading a body from connection"

LLM = Message


def _chunk(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def _delta(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "gen-e2e-1",
        "object": "chat.completion.chunk",
        "created": 1776000000,
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


class _OneBearerAuth:
    """Minimal ``FailoverAuthStore``: one bearer, no siblings to rotate to."""

    async def get_api_key(
        self, provider: str, session_id: str | None = None, **kwargs: Any
    ) -> str | None:
        return "stub-key"

    def rotate_sibling(self, *args: Any, **kwargs: Any) -> bool:
        return False


@pytest.fixture
def notes_path(tmp_path):
    return tmp_path / "notes.txt"


@pytest.fixture
def gateway(notes_path):
    """A loopback OpenAI-compatible gateway that dies in band on request 1.

    Yields the base URL plus the state the assertions read: how many
    chat/completions calls were served, and the body of each, so the re-issued
    request can be checked on the WIRE rather than in the driver's memory.
    """
    state: dict[str, Any] = {
        "calls": 0,
        "bodies": [],
        "terminal": {
            "error": {
                "code": 502,
                "message": "Provider returned error",
                "metadata": {
                    "error_type": "provider_unavailable",
                    "provider_name": "Together",
                    "raw": _UPSTREAM_RAW,
                },
            },
        },
    }

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — stdlib naming
            pass

        def do_POST(self) -> None:  # noqa: N802 — stdlib
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            if not self.path.rstrip("/").endswith("/chat/completions"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")
                return
            state["bodies"].append(json.loads(raw or b"{}"))
            call = state["calls"]
            state["calls"] += 1

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            if call == 0:
                # THE INCIDENT: prose, then a call still being dictated, then
                # the gateway's in-band report that its upstream host died.
                self.wfile.write(
                    _chunk(_delta({"role": "assistant", "content": "Let me write that down. "}))
                )
                self.wfile.write(
                    _chunk(
                        _delta(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "write",
                                            "arguments": (
                                                f'{{"path": "{notes_path}", "content": "hel'
                                            ),
                                        },
                                    }
                                ]
                            }
                        )
                    )
                )
                if prior_delta := state.get("prior_delta"):
                    self.wfile.write(_chunk(prior_delta))
                self.wfile.write(
                    _chunk(
                        {
                            **_delta({}, "error"),
                            **state["terminal"],
                        }
                    )
                )
            elif call == 1:
                # The continued turn: the model re-issues the aborted call, this
                # time complete, and the harness must actually run it.
                self.wfile.write(
                    _chunk(_delta({"role": "assistant", "content": "Reissued on a fresh host. "}))
                )
                self.wfile.write(
                    _chunk(
                        _delta(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_2",
                                        "type": "function",
                                        "function": {
                                            "name": "write",
                                            "arguments": json.dumps(
                                                {"path": str(notes_path), "content": "hello"}
                                            ),
                                        },
                                    }
                                ]
                            }
                        )
                    )
                )
                self.wfile.write(_chunk(_delta({}, finish="tool_calls")))
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.wfile.write(_chunk(_delta({"role": "assistant", "content": "done"})))
                self.wfile.write(_chunk(_delta({}, finish="stop")))
                self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def write_tool(notes_path, executed):
    """The tool the incident's aborted call named. It really writes the file."""

    async def execute(tool_call_id, args, signal, on_update, context):
        executed.append("write")
        payload = args if isinstance(args, dict) else json.loads(args or "{}")
        notes_path.write_text(str(payload.get("content", "")))
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name="write",
            content=[TextContent(text=f"wrote {notes_path}")],
        )

    return AgentTool(
        name="write",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        },
        execute=execute,
    )


@pytest.fixture
def executed():
    return []


@pytest.mark.parametrize(
    ("provider", "model_id", "wire_error", "recovers"),
    [
        pytest.param("radient", MODEL_ID, None, True, id="relay-502"),
        pytest.param("radient", "auto", {"message": _H2_BODY_READ_ERROR}, True, id="radient-auto"),
        pytest.param(
            "radient-key", "auto", {"message": _H2_BODY_READ_ERROR}, True, id="radient-key-auto"
        ),
        pytest.param(
            "radient", "auto", {"code": 400, "message": _H2_BODY_READ_ERROR}, False, id="http-400"
        ),
        pytest.param(
            "radient",
            "auto",
            {"code": "policy_violation", "message": _H2_BODY_READ_ERROR},
            False,
            id="explicit-policy-code",
        ),
        pytest.param(
            "radient",
            "auto",
            {"status": 403, "message": _H2_BODY_READ_ERROR},
            False,
            id="explicit-status",
        ),
        pytest.param(
            "radient",
            "auto",
            {"type": "content_policy_violation", "message": _H2_BODY_READ_ERROR},
            False,
            id="explicit-refusal-type",
        ),
        pytest.param("radient", "auto", {"message": "Stream error: refused"}, False, id="refusal"),
        pytest.param("openai", "gpt-4o", {"message": _H2_BODY_READ_ERROR}, False, id="first-party"),
        pytest.param(
            "openrouter", MODEL_ID, {"message": _H2_BODY_READ_ERROR}, False, id="openrouter"
        ),
    ],
)
@pytest.mark.asyncio
async def test_the_incident_is_recovered_over_the_wire(
    gateway, write_tool, notes_path, executed, provider, model_id, wire_error, recovers
):
    """A qualified error continues the turn and the re-issued call runs for real.

    Otherwise, the original stream error ends the pass without another request
    or any writes. The incomplete call must never run in either case.
    """
    _, state = gateway
    if wire_error is not None:
        state["terminal"] = {"error": wire_error}
    await _assert_wire_outcome(
        gateway, write_tool, notes_path, executed, provider, model_id, recovers
    )


@pytest.mark.parametrize(
    ("terminal_fields", "prior_delta"),
    [
        pytest.param(_delta({}, "content_filter"), None, id="choice-content-filter"),
        pytest.param(
            {
                "choices": [
                    *_delta({}, "error")["choices"],
                    {"index": 1, "finish_reason": "content_filter"},
                ]
            },
            None,
            id="later-choice-content-filter",
        ),
        pytest.param(_delta({"refusal": "Cannot assist."}, "error"), None, id="choice-refusal"),
        pytest.param(
            {"choices": [{"finish_reason": "error", "native_finish_reason": "SAFETY"}]},
            None,
            id="native-safety-finish",
        ),
        pytest.param(
            {"choices": [{"finish_reason": "error", "message": {"refusal": "Cannot assist."}}]},
            None,
            id="message-refusal",
        ),
        pytest.param({}, _delta({"refusal": "Cannot assist."}), id="prior-refusal"),
        pytest.param({"status": 403}, None, id="envelope-status-403"),
        pytest.param({"code": "content_filter"}, None, id="envelope-policy-code"),
        pytest.param({"type": "refusal"}, None, id="envelope-refusal-type"),
        pytest.param(
            {"error": {"message": _H2_BODY_READ_ERROR, "detail": "Request refused"}},
            None,
            id="competing-error-detail",
        ),
        pytest.param(
            {"metadata": {"error_type": "policy_violation"}}, None, id="envelope-metadata"
        ),
        pytest.param(
            {
                "error": {
                    "message": _H2_BODY_READ_ERROR,
                    "metadata": {"error_type": "policy_violation"},
                }
            },
            None,
            id="error-metadata",
        ),
        pytest.param(
            {"error": {"message": _H2_BODY_READ_ERROR, "metadata": {"provider_code": "SAFETY"}}},
            None,
            id="error-provider-code",
        ),
        pytest.param(
            {"choices": [{"finish_reason": "error", "error": {"code": 403}}]},
            None,
            id="choice-error",
        ),
        pytest.param(
            {
                "error": {
                    "message": _H2_BODY_READ_ERROR,
                    "metadata": {
                        "raw": json.dumps({"error": {"code": 403, "message": _H2_BODY_READ_ERROR}})
                    },
                }
            },
            None,
            id="relayed-upstream-status",
        ),
    ],
)
@pytest.mark.asyncio
async def test_mixed_transport_and_disposition_stays_terminal(
    gateway, write_tool, notes_path, executed, terminal_fields, prior_delta
):
    _, state = gateway
    state["terminal"] = {"error": {"message": _H2_BODY_READ_ERROR}, **terminal_fields}
    state["prior_delta"] = prior_delta
    await _assert_wire_outcome(gateway, write_tool, notes_path, executed, "radient", "auto", False)


async def _assert_wire_outcome(
    gateway, write_tool, notes_path, executed, provider, model_id, recovers
):
    base_url, state = gateway
    spec = build_model_spec(provider, model_id).model_copy(update={"base_url": base_url})
    auth = _OneBearerAuth()
    failures: list[ProviderError] = []

    async def client_for(model: Any) -> Any:
        return client_for_spec(model)

    async def stream_fn(request: ChatRequest, signal: AbortSignal | None):
        try:
            async for event in stream_with_failover(
                request,
                auth,
                {"retry": {"baseDelayMs": 1, "maxRetries": 1}},
                client_for,
                signal=signal,
            ):
                yield event
        except ProviderError as failure:
            failures.append(failure)
            raise

    config = LoopConfig(
        model=spec,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=stream_fn,
    )

    printed: list[str] = []
    events: list[Any] = []
    async for event in AgentLoop().run(
        [Message.user("Triage the reported customer issue.")],
        LoopContext(tools=[write_tool]),
        config,
        None,
    ):
        events.append(event)
        if event.type == "message_update":
            printed.append(getattr(event, "delta", ""))

    ends = [e for e in events if isinstance(e, AgentEndEvent)]
    assert len(ends) == 1
    assert len(failures) == 1
    failure = failures[0]
    wire_error = state["terminal"]["error"]
    code = wire_error.get("code")
    assert failure.status == (code if isinstance(code, int) else None)
    if model_id == "auto":
        assert state["bodies"][0]["model"] == "auto"
    assert not failure.transport  # This was an HTTP 200 error frame, not a local read failure.
    if not recovers:
        assert (state["calls"], executed, notes_path.exists()) == (1, [], False)
        assert ends[0].error and wire_error["message"] in ends[0].error
        assert not failure.connectivity_loss
        return
    assert failure.connectivity_loss
    assert ends[0].error is None, "a gateway's upstream death must not end the pass"
    assert ends[0].aborted is False

    # The partial answer is shown ONCE, followed by its continuation and the
    # turn's own ending — never the partial text again.
    assert "".join(printed) == "Let me write that down. Reissued on a fresh host. done"

    # The side effect: the call dropped as truncated JSON was re-issued and RAN.
    assert executed == ["write"]
    assert notes_path.read_text() == "hello"

    # Three wire calls: the incident, the continuation that re-issued the call,
    # and the wrap-up after the tool result.
    assert state["calls"] == 3, "the gateway was asked again, which is the re-route"

    # The operator is told what happened, in the wording that is true of both
    # families this branch serves.
    notices = [e.text for e in events if isinstance(e, NoticeEvent)]
    assert notices == ["response stream cut mid-answer — resuming the turn (1/3)"]

    # And the continuation reached the MODEL as harness chrome: the re-issued
    # request carries the instruction, naming the aborted call — and every
    # surface will suppress that row on replay rather than paint it as the
    # operator's own words (round-1 M1).
    follow_up = state["bodies"][1]["messages"]
    instruction = follow_up[-1]["content"]
    assert follow_up[-1]["role"] == "user"
    # The signed-in auto route can serialize the same text as cacheable blocks.
    if isinstance(instruction, list):
        instruction = "".join(part["text"] for part in instruction)
    assert "write" in instruction and "aborted" in instruction
    assert is_harness_chrome(instruction)
    # The truncated call never reached the wire: no unpaired tool_call in the
    # re-issued history, so the request stays legal on every vendor's wire.
    assert all(not entry.get("tool_calls") for entry in follow_up)
