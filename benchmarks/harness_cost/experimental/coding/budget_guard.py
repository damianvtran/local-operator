"""Conservative per-trial AND cumulative USD guard for an isolated coding lane.

WHY: a response can disconnect before a meter logs usage. Reservations therefore
remain charged durably, even after failures/restarts: unused capacity is never
silently refunded. This intentionally overestimates spend. No billing tier or
price is inferred; the operator supplies upper-bound rates for the frozen tier.
CONSTRAINTS: one guard owns a ledger (exclusive lock); reserve before forwarding,
fail closed on corrupt accounting, missing caps, unpriced models and bad limits.
The downstream meter must cap actual output to the requested limit and report
all usage, including helpers/cache charges. Bounds must cover that billing tier.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

RUN = re.compile(r"coding__[a-z0-9-]+__[a-zA-Z0-9_-]+__r[0-9]+\Z")


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("positive finite USD cap/rate required")
    return value


class Budget:
    """Durably retain upper bounds; no response/log timing gap can free spend."""

    def __init__(self, ledger, total, trial, input_rate, output_rate, models):
        self.path = Path(ledger)
        self.total, self.trial = positive(total), positive(trial)
        self.input_rate, self.output_rate = positive(input_rate), positive(output_rate)
        self.models = frozenset(models)
        if not self.models or not all(isinstance(m, str) and m for m in self.models):
            raise ValueError("explicit priced models required")
        # An interrupted atomic write is ambiguous: refusing it is safer than
        # assuming the preceding reservation never reached the upstream.
        if self.path.with_suffix(self.path.suffix + ".pending").exists():
            raise ValueError("interrupted reservation needs explicit reconciliation")
        self.used = json.loads(self.path.read_text()) if self.path.exists() else {}
        if not isinstance(self.used, dict) or any(
            not RUN.fullmatch(k) or not math.isfinite(float(v)) or float(v) < 0
            for k, v in self.used.items()
        ):
            raise ValueError("corrupt reservation ledger")

    def reserve(self, run_id, raw):
        if not RUN.fullmatch(run_id):
            raise ValueError("coding run ID required")
        body = json.loads(raw)
        if body.get("model") not in self.models:
            raise ValueError("unpriced model")
        if body.get("service_tier") not in (None, os.environ.get("CODING_BILLING_TIER")):
            raise ValueError("request billing tier differs from explicit frozen tier")
        maximum = body.get("max_tokens", body.get("max_output_tokens"))
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            raise ValueError("explicit positive integer output limit required")

        # UTF-8 bytes upper-bound text tokens; supplied input rate must also
        # bound cache writes/reads. Images/audio are outside this priced lane.
        def text_only(value):
            if isinstance(value, dict):
                if value.get("type") in {
                    "image",
                    "image_url",
                    "input_image",
                    "audio",
                    "input_audio",
                    "document",
                    "input_file",
                    "file",
                }:
                    raise ValueError("multimodal billing is not priced")
                for child in value.values():
                    text_only(child)
            elif isinstance(value, list):
                for child in value:
                    text_only(child)

        text_only(body)
        amount = len(raw) * self.input_rate / 1_000_000 + maximum * self.output_rate / 1_000_000
        current = self.used.get(run_id, 0)
        if current + amount > self.trial or sum(self.used.values()) + amount > self.total:
            raise OverflowError("conservative trial or total USD cap exceeded")
        updated = {**self.used, run_id: current + amount}
        # Same-directory atomic replace keeps crashes fail-closed. Reserve must
        # complete on disk before a single upstream byte is sent.
        temporary = self.path.with_suffix(self.path.suffix + ".pending")
        with temporary.open("w") as stream:
            json.dump(updated, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)
        self.used = updated
        return amount


def create_app():
    """Load optional ASGI dependencies only for an explicitly configured service."""
    import fcntl

    import httpx
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    ledger = Path(os.environ["CODING_RESERVATION_LEDGER"])
    ledger.parent.mkdir(parents=True, exist_ok=True)
    owner = ledger.with_suffix(ledger.suffix + ".lock").open("a")
    fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    budget = Budget(
        ledger,
        os.environ["CODING_TOTAL_BUDGET_USD"],
        os.environ["CODING_TRIAL_BUDGET_USD"],
        os.environ["CODING_INPUT_USD_PER_M"],
        os.environ["CODING_OUTPUT_USD_PER_M"],
        os.environ["CODING_PRICED_MODELS"].split(","),
    )
    meter = os.environ["CODING_METER_URL"].rstrip("/")
    bridge = os.environ["CODING_BRIDGE_URL"].rstrip("/")
    current = Path(os.environ["CODING_CURRENT_RUN"])
    client = httpx.AsyncClient(timeout=httpx.Timeout(1800, connect=30))
    lock = asyncio.Lock()

    async def forward(request):
        path = request.path_params["path"]
        codex = path == "codex/v1/responses"
        if codex:
            run_id = current.read_text().strip() if current.exists() else ""
            if not run_id.startswith("coding__codex-best__"):
                return JSONResponse({"error": "Codex trial not reserved"}, status_code=409)
            target = bridge + "/v1/responses"
        else:
            parts = path.split("/")
            if len(parts) != 4 or parts[0] != "r" or parts[2:] != ["v1", "messages"]:
                return JSONResponse({"error": "unpriced route refused"}, status_code=403)
            run_id = parts[1]
            target = meter + "/" + path
        raw = await request.body()
        try:
            async with lock:
                budget.reserve(run_id, raw)
        except OverflowError:
            return JSONResponse({"error": "budget_exceeded"}, status_code=402)
        except (ValueError, TypeError, AttributeError):
            return JSONResponse({"error": "invalid or unpriced request"}, status_code=400)
        # Real auth is never accepted from task containers or persisted; the
        # externally configured meter reads OPENROUTER_API_KEY in its own env.
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower()
            not in {
                "authorization",
                "x-api-key",
                "host",
                "content-length",
                "connection",
                "accept-encoding",
            }
        }
        response = await client.send(
            client.build_request("POST", target, content=raw, headers=headers), stream=True
        )

        async def stream():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()

        headers = {
            k: v
            for k, v in response.headers.items()
            if k.lower()
            not in {"content-length", "content-encoding", "transfer-encoding", "connection"}
        }
        return StreamingResponse(stream(), status_code=response.status_code, headers=headers)

    async def health(request):
        return JSONResponse({"ok": True, "reserved_usd": sum(budget.used.values())})

    @asynccontextmanager
    async def lifespan(app):
        """Release only this guard's client and ledger lock on every ASGI exit.

        Starlette 1.x removed the legacy on_shutdown constructor argument.
        Lifespan works on the supported modern ASGI surface and makes real
        startup/shutdown testable without dispatching a paid upstream call.
        The finally nesting also releases the exclusive ledger lock if HTTP
        client shutdown itself fails; a dead guard must not retain ownership.
        """
        try:
            yield
        finally:
            try:
                await client.aclose()
            finally:
                owner.close()

    app = Starlette(
        routes=[Route("/health", health), Route("/{path:path}", forward, methods=["POST"])],
        lifespan=lifespan,
    )
    app.state.ledger_owner = owner
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        create_app(),
        host="127.0.0.1",
        port=int(os.environ.get("CODING_GUARD_PORT", "8788")),
        log_level="warning",
    )
