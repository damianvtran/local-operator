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


def test_generated_method_union_and_types_cover_new_methods() -> None:
    # The generated TS is the extension's only view of the method set; a method
    # added to METHODS without regenerating would let the extension ship a
    # handler the type union rejects. Also assert the return-shape interfaces the
    # new commands rely on are emitted.
    rendered = gen_ts_render()
    assert "'scroll'" in rendered and "'logs'" in rendered
    assert "interface LogEntry" in rendered
    assert "interface ScrollResult" in rendered
