from __future__ import annotations

from local_operator.browser_bridge.gen_ts import main as gen_ts_main
from local_operator.browser_bridge.gen_ts import render as gen_ts_render
from local_operator.browser_bridge.protocol import (
    METHODS,
    PROTO_VERSION,
    ErrorCode,
    ErrorDetail,
    Hello,
    Request,
    Response,
)


def test_protocol_round_trips() -> None:
    hello = Hello(proto=PROTO_VERSION, token="secret", extension_version="0.1.0", browser="Chrome")
    assert Hello.model_validate_json(hello.model_dump_json()) == hello
    request = Request(id="r-1", method="goto", params={"url": "https://example.com"})
    assert Request.model_validate_json(request.model_dump_json()) == request
    response = Response(
        id=request.id,
        ok=False,
        error=ErrorDetail(code=ErrorCode.NAV_FAILED, message="net::ERR_NAME_NOT_RESOLVED"),
    )
    assert Response.model_validate_json(response.model_dump_json()) == response


def test_generated_types_are_current() -> None:
    assert gen_ts_main(["--check"]) == 0


def test_scroll_and_logs_are_protocol_methods() -> None:
    # The two new capabilities must be real wire methods, not just tool-side
    # aliases, or the daemon's method allowlist would reject them.
    assert "scroll" in METHODS
    assert "logs" in METHODS


def test_scroll_and_logs_requests_round_trip() -> None:
    scroll = Request(id="r-s", method="scroll", params={"direction": "bottom"})
    assert Request.model_validate_json(scroll.model_dump_json()) == scroll
    logs = Request(id="r-l", method="logs", params={"level": "error", "limit": 50})
    assert Request.model_validate_json(logs.model_dump_json()) == logs


def test_tabs_is_a_protocol_method_with_a_daemon_timeout() -> None:
    # `tabs` is the multi-surface discovery verb: parallel sessions each own a
    # tab now, so agents need to list what is being driven. A METHODS entry
    # without a daemon timeout would pass the allowlist and then die as
    # "unknown method", so the two tables are pinned together here.
    from local_operator.browser_bridge.daemon import COMMAND_TIMEOUTS

    assert "tabs" in METHODS
    assert "tabs" in COMMAND_TIMEOUTS
    # Every advertised method must have a timeout, and vice versa — the ONE
    # source of truth contract METHODS documents.
    assert set(METHODS) == set(COMMAND_TIMEOUTS)


def test_tab_limit_is_a_typed_error() -> None:
    # The surface cap refusal must arrive as a typed code the tool can map to
    # "close one first", not a generic internal error.
    assert ErrorCode.TAB_LIMIT.value == "tab_limit"


def test_generated_method_union_and_types_cover_new_methods() -> None:
    # The generated TS is the extension's only view of the method set; a method
    # added to METHODS without regenerating would let the extension ship a
    # handler the type union rejects. Also assert the return-shape interfaces the
    # new commands rely on are emitted.
    rendered = gen_ts_render()
    assert "'scroll'" in rendered and "'logs'" in rendered
    assert "interface LogEntry" in rendered
    assert "interface ScrollResult" in rendered


def test_generated_ts_emits_origin_prompt_timeout() -> None:
    # origins.ts imports this from protocol.gen.ts so the extension's 60 s
    # deny, the daemon's 65 s prompt window, and the session client's HTTP
    # timeout all derive from one Python constant and cannot drift (finding
    # A3: the chain extension deny < daemon window < client timeout is what
    # turns a mid-prompt wait into a typed answer instead of "unreachable").
    assert "export const ORIGIN_PROMPT_TIMEOUT_MS = 60000 as const;" in gen_ts_render()


def test_access_flow_methods_are_wired_end_to_end() -> None:
    # The approval flow only helps if every layer knows the methods: wire
    # protocol, daemon timeout table, and generated TS union. A miss in any of
    # them reproduces the original incident class (a call that times out
    # opaquely instead of failing typed).
    from local_operator.browser_bridge.daemon import COMMAND_TIMEOUTS

    for method in ("request_access", "await_access", "cancel_access"):
        assert method in METHODS
        assert method in COMMAND_TIMEOUTS
    rendered = gen_ts_render()
    assert all(
        f"'{method}'" in rendered for method in ("request_access", "await_access", "cancel_access")
    )
    assert "ORIGIN_NOT_ALLOWED" in rendered


def test_await_access_daemon_timeout_covers_the_extension_slice() -> None:
    # The extension caps one await slice at 20s (access.ts AWAIT_SLICE_MS);
    # the daemon's bound must sit above it or a full slice would race the
    # daemon timeout and lose — recreating the misleading-timeout incident.
    from local_operator.browser_bridge.daemon import COMMAND_TIMEOUTS

    assert COMMAND_TIMEOUTS["await_access"] > 20.0
