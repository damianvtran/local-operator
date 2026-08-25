from __future__ import annotations

from local_operator.browser_bridge.gen_ts import main as gen_ts_main
from local_operator.browser_bridge.protocol import (
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
