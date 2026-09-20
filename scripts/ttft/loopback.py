"""A loopback provider: an OpenAI-compatible SSE endpoint served inside the bench.

WHY THE HARNESS NEEDS ITS OWN PROVIDER
======================================
The harness has to measure two things the built-in ``test`` mock cannot give it,
and both of them are the point of this ticket:

1. **A reasoning phase.** ``MockClient`` (``providers/clients.py``) emits text and
   nothing else, so a mock-provider bench has no reasoning delta to be dropped
   and cannot show the wait the operator is complaining about at all.
2. **A deterministically controllable provider floor.** With a real provider the
   provider's own time swamps every local measurement; with a cached-prefix
   prompt it costs 0.4-1.6 s on this machine (see ``probe_provider_ttfb.py``). A
   gate on < 300 ms has to be measured with the provider's contribution at a
   known, chosen value.

So the harness serves its own endpoint and points the real
``openai-compatible`` LOCAL provider at it through the config the app already
supports (``providers.<id>.base_url``). Nothing in ``local_operator/`` is
changed, and every layer the operator's traffic crosses is the real one: the real
``OpenAICompatClient`` parses the SSE (including the ``reasoning_content`` branch
whose ``StreamReasoningDelta`` the runtime currently drops), the real failover
wrapper forwards it, the real session and loop consume it, and the real daemon /
TUI / phone relay carry it to a front end. Only the model is a stub.

THE FLOORS ARE KNOBS, AND THAT IS THE DESIGN
============================================
``prefill_ms`` is the provider's time to the first reasoning token and
``prefill_ms + reasoning_ms`` its time to the first text token. Zero means
"measure local-operator's own overhead", which is the GATE configuration, since
only local-operator's share of the wait is a budget this repository can hold. A
non-zero ``reasoning_ms`` is how the harness DEMONSTRATES the invisible phase:
the provider puts reasoning on the wire at T and text at T+reasoning_ms, and the
front end sees nothing until the text — a gap that is measured, not argued.

A REQUEST IS CORRELATED BY A TOKEN IN ITS LAST USER MESSAGE, not by request id:
the provider is a different HTTP conversation from the harness's submit, and the
measured turn can be one of eight concurrent ones. The harness puts
``[bench:<token>]`` in the prompt it submits, this module reads it back out of the
last user message, and the stamps land under that token. The token is generated
per turn and never reused, so a helper call (auto-naming, effort classification,
a compaction summary) cannot be mistaken for the measured turn — it carries no
token at all.

ONE EVENT LOOP, DELIBERATELY
===========================
The endpoint is served on THIS process's event loop, so its stamps are on the
same ``time.monotonic()`` clock as the harness's submit timestamps and need no
clock-skew correction, and the stub's own scheduling cost is the loop's own. That
is the right trade for a stub whose job is to have a *known* cost: a real socket
and a real SSE parser are still exercised, and the fake part is fake by
construction rather than by accident.
"""

from __future__ import annotations

import asyncio
import json
import re
import socket
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

# FastAPI's route machinery resolves a handler's annotations from the module
# globals. These are module-level rather than imported inside ``_build_app`` for
# exactly that reason: an annotation naming a CLOSURE-LOCAL ``Request`` cannot be
# resolved, FastAPI falls back to treating that parameter as a required query
# parameter, and every request then fails with a 422 asking for ``?request=`` —
# measured, not theorised, which is why the import sits here.
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

#: Marker the harness puts in a submitted prompt and this provider reads back out.
TOKEN_PATTERN = re.compile(r"\[bench:([0-9a-f]{8,32})\]")


def request_token(body: Any) -> str:
    """The measurement token of the last user message, or ``""`` for a helper call."""
    messages = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            text = ""
        match = TOKEN_PATTERN.search(text)
        return match.group(1) if match else ""
    return ""


def last_user_text(body: Any) -> str:
    """The last user message's text, truncated — enough to tell a turn from a helper call."""
    messages = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content[:100]
        if isinstance(content, list):
            joined = " ".join(
                str(part.get("text", ""))
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
            return joined[:100]
    return ""


@dataclass
class ProviderStamps:
    """When the provider put each kind of token on the wire, per measured turn.

    Stored as WALL-CLOCK epochs and not on this process's monotonic clock, because
    for most channels the submit happens in another process (the TUI arm's child,
    the exec arm's interpreter) and the wall clock is the only clock they share.
    A missing token means the provider never sent that kind for that turn — a fact
    about the run, not a missing measurement — and is reported as
    :data:`scripts.ttft.metrics.UNAVAILABLE`.
    """

    reasoning: dict[str, float] = field(default_factory=dict)
    text: dict[str, float] = field(default_factory=dict)

    def stamp_reasoning(self, token: str) -> None:
        if token:
            self.reasoning.setdefault(token, time.time())

    def stamp_text(self, token: str) -> None:
        if token:
            self.text.setdefault(token, time.time())

    def delta_ms(self, kind: str, token: str, submit_epoch: float) -> float:
        """Submit -> the provider had that token on the wire, in ms.

        ``-1`` (unavailable) for a token the provider never stamped, or for one
        whose delta is nonsensical (negative, or beyond any real turn), which is
        what a wall-clock step would produce. The provider numbers are reported
        and never asserted for exactly this reason.
        """
        from scripts.ttft.metrics import UNAVAILABLE

        table = self.reasoning if kind == "reasoning" else self.text
        stamp = table.get(token)
        if stamp is None:
            return UNAVAILABLE
        delta = (stamp - submit_epoch) * 1000
        if delta < 0 or delta > 120_000:
            return UNAVAILABLE
        return delta


class LoopbackProvider:
    """The harness's own OpenAI-compatible SSE endpoint, on this process's loop.

    ``prefill_ms``/``reasoning_ms`` are the emulated provider floors described in
    the module docstring. ``stamps`` records what the provider did and when, for
    the reported-never-asserted provider columns.
    """

    def __init__(
        self,
        *,
        model: str = "bench-loopback",
        prefill_ms: float = 0.0,
        reasoning_ms: float = 0.0,
    ) -> None:
        self.model = model
        self.prefill_ms = prefill_ms
        self.reasoning_ms = reasoning_ms
        self.stamps = ProviderStamps()
        self.requests = 0
        #: Every request this endpoint served, so a mis-paired provider column is
        #: AUDITABLE rather than argued about: each entry says which token it
        #: carried, when it arrived, and whether it streamed. A turn that produces
        #: two requests carrying the same token (a replay, a retry, a helper call
        #: that quotes the prompt) is visible here as two rows.
        self.log: list[dict[str, Any]] = []
        self._listener: socket.socket | None = None
        self._server: Any = None
        self._task: asyncio.Task[Any] | None = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def base_url(self) -> str:
        if self._listener is None:
            raise RuntimeError("provider not started")
        return f"http://127.0.0.1:{self._listener.getsockname()[1]}/v1"

    async def start(self) -> None:
        import uvicorn

        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        self._listener = listener
        server = uvicorn.Server(uvicorn.Config(self._build_app(), log_level="error"))
        self._server = server
        self._task = asyncio.create_task(server.serve(sockets=[listener]))
        for _ in range(100_000):
            if server.started:
                return
            await asyncio.sleep(0)
        raise AssertionError("loopback provider did not start")

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._task), 20)
            except Exception:  # noqa: BLE001 — an unresponsive server is cancelled, not awaited
                self._task.cancel()
            self._task = None
        if self._listener is not None:
            self._listener.close()
            self._listener = None

    # -- the wire ----------------------------------------------------------

    def _build_app(self) -> Any:
        app = FastAPI()

        @app.get("/v1/models")
        async def models() -> JSONResponse:
            if len(self.log) < 400:
                self.log.append({"path": "/v1/models", "at": time.time()})
            # The local provider resolves its listing from here, and its absence
            # degrades the model to a 4096-token window and a refusal rather than
            # a slow turn — see providers/local.py:local_model_spec.
            return JSONResponse({"object": "list", "data": [{"id": self.model, "object": "model"}]})

        @app.post("/v1/chat/completions")
        async def completions(request: Request) -> StreamingResponse:
            body = await request.json()
            self.requests += 1
            token = request_token(body)
            if len(self.log) < 400:
                self.log.append(
                    {
                        "path": "/v1/chat/completions",
                        "token": token or "(helper)",
                        "at": time.time(),
                        "stream": bool(body.get("stream")),
                        # The label is what distinguishes the MEASURED turn from a
                        # helper call (effort classification, auto-naming) that
                        # happens to quote the same prompt. Without it a duplicated
                        # token is indistinguishable from a retry.
                        "last_user": last_user_text(body),
                    }
                )
            return StreamingResponse(self._sse(token), media_type="text/event-stream")

        return app

    def _chunk(self, delta: dict[str, Any], finish: str | None = None) -> str:
        payload = {
            "id": "chatcmpl-loopback",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    async def _sse(self, token: str) -> AsyncIterator[str]:
        """One canned turn: reasoning, then text, then a close and a usage trailer."""
        started = time.monotonic()

        async def sleep_until(offset_ms: float) -> None:
            wait = offset_ms / 1000.0 - (time.monotonic() - started)
            if wait > 0:
                await asyncio.sleep(wait)

        yield self._chunk({"role": "assistant", "reasoning_content": ""})
        await sleep_until(self.prefill_ms)
        # The first REASONING token is the provider's floor for this turn, and
        # it is what a front end cannot see today: this is the branch
        # `clients.py` turns into `StreamReasoningDelta`.
        self.stamps.stamp_reasoning(token)
        yield self._chunk({"reasoning_content": "Working through the question."})
        yield self._chunk({"reasoning_content": " The answer is short."})
        await sleep_until(self.prefill_ms + self.reasoning_ms)
        self.stamps.stamp_text(token)
        yield self._chunk({"content": "Hello"})
        yield self._chunk({"content": " from the loopback provider!"})
        yield self._chunk({}, "stop")
        yield (
            'data: {"id":"chatcmpl-loopback","object":"chat.completion.chunk",'
            f'"model":"{self.model}","choices":[],"usage":{{"prompt_tokens":10,'
            '"completion_tokens":8,"total_tokens":18,'
            '"completion_tokens_details":{"reasoning_tokens":4}}}}\n\n'
        )
        yield "data: [DONE]\n\n"
