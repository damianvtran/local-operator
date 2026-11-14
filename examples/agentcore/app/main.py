"""Local Operator on Amazon Bedrock AgentCore Runtime — the container entrypoint.

AgentCore Runtime runs this container in a dedicated microVM per
``runtimeSessionId`` and speaks a two-route HTTP contract on port 8080
(``GET /ping``, ``POST /invocations``). ``BedrockAgentCoreApp`` implements that
contract; this module only:

1. starts the loopback Bedrock bridge (``bedrock_bridge``) so lop's
   ``openai-compatible`` hosting reaches Bedrock with SigV4 role credentials,
2. writes lop's config pointing at that bridge (no ``/login``, no API key),
3. maps each invocation onto a lop session for its ``runtimeSessionId``
   (``lop_runtime.SessionPool``) and streams the turn back as SSE.

Modelled on the AWS sample that hosts Hermes Agent on AgentCore
(aws-samples/sample-host-hermesagent-on-amazon-bedrock-agentcore), minus its
messaging channels.

Streamed events (each one SSE ``data:`` line of JSON)::

    {"type": "session", "session_id": "...", "model": "...",  # model in effect
     "notice": "..."}                                       # only if a model was ignored
    {"type": "text", "text": "..."}                       # assistant text deltas
    {"type": "tool_start", "tool": "bash", "args": {...}}
    {"type": "tool_end", "tool": "bash", "is_error": false, "result": "..."}
    {"type": "end", "error": null}                         # turn finished
    {"type": "error", "error": "..."}                      # invocation refused
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator

from bedrock_agentcore.runtime import (  # pyright: ignore[reportMissingImports]
    BedrockAgentCoreApp,
)
from bedrock_bridge import start_bridge
from lop_runtime import PayloadError, RuntimeSettings, SessionPool, parse_payload

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("lop_agentcore")

# Fail at boot, not on the first invocation: a runtime with no model id would
# pass /ping and then error on every request.
SETTINGS = RuntimeSettings.from_env()
_bridge = start_bridge(
    SETTINGS.model_id,
    port=int(os.environ.get("BRIDGE_PORT", "0")),
    # Set by the runtime template; falls back to the platform's AWS_REGION.
    region=os.environ.get("LOP_BEDROCK_REGION") or None,
)
_bridge_url = f"http://127.0.0.1:{_bridge.server_address[1]}/v1"
POOL = SessionPool(SETTINGS)
POOL.configure(_bridge_url)
log.info(
    "lop runtime ready model=%s approvals=%s bridge=%s home=%s",
    SETTINGS.model_id,
    SETTINGS.approvals,
    _bridge_url,
    SETTINGS.home,
)

app = BedrockAgentCoreApp()

# Outside AgentCore (``python main.py`` locally) there is no session header;
# one fixed id keeps a local multi-turn conversation together.
_LOCAL_SESSION = "local"


@app.entrypoint
async def invoke(payload: Any, context: Any) -> AsyncIterator[dict[str, Any]]:
    """Run one turn of the lop session bound to this runtime session."""
    session_id = getattr(context, "session_id", None) or _LOCAL_SESSION
    try:
        turn = parse_payload(payload)
    except PayloadError as exc:
        yield {"type": "error", "error": str(exc)}
        return
    try:
        # run_turn's first frame is the ``session`` event, naming the model the
        # session actually runs on (see lop_runtime.session_event).
        async for event in POOL.run_turn(session_id, turn):
            yield event
    except Exception as exc:  # surface, never swallow: the caller sees why
        log.exception("turn failed for session %s", session_id)
        yield {"type": "error", "error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", "8080")))
