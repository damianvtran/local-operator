"""Experimental text-only OpenRouter meter with explicit, fail-closed budgets.

All credentials come from OPENROUTER_API_KEY and stay in memory. Total and
per-trial caps reserve a conservative byte/token/rate bound before dispatch;
unknown billing, disconnects, and malformed ledgers stop later traffic. Only a
pinned, explicitly priced model and default service tier are accepted. This is
not provider-side billing enforcement: configured rates must bound all charges.
Prompt capture is opt-in; logs contain usage/count metadata, never auth headers.
"""

from __future__ import annotations

import fcntl
import math
import uuid
from contextlib import contextmanager
import re
import json
import os
import time
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

UPSTREAM = os.environ.get("BENCH_UPSTREAM", "https://openrouter.ai/api").rstrip("/")
MODEL = os.environ.get("BENCH_MODEL", "claude-sonnet-5-5")
PIN = os.environ.get("BENCH_PIN_MODEL") == "1"
AUTO_CACHE = os.environ.get("BENCH_AUTO_CACHE", "0") == "1"
LOG = Path(os.environ.get("BENCH_LOG", "requests.jsonl"))
BODIES = Path(os.environ.get("BENCH_BODIES", "bodies"))
CAPTURE_BODIES = os.environ.get("BENCH_CAPTURE_BODIES") == "1"
IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
INJECT_IDENTITY = False
_seen_runs: set[str] = set()
client = httpx.AsyncClient(timeout=httpx.Timeout(600, connect=30))


async def upstream_key() -> str:
    """Never consult a private auth database or write the environment key to disk."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise ValueError("OPENROUTER_API_KEY is required")
    return key


def positive(name: str) -> float:
    value = float(os.environ[name])
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


class BudgetGuard:
    """Durable, serialized reservations prevent retries/concurrency overspending.

    Unknown billing keeps the reservation and blocks ALL later requests. Recovery
    requires operator reconciliation, never a reset on malformed/missing usage.
    Rate ceilings must include cache-write premiums and every enabled surcharge;
    only text/custom tools and the explicitly selected default tier are supported.
    """

    def __init__(self):
        self.total = positive("BENCH_BUDGET_USD")
        self.trial = positive("BENCH_TRIAL_BUDGET_USD")
        self.reserve = positive("BENCH_REQUEST_RESERVE_USD")
        self.input_rate = positive("BENCH_INPUT_USD_PER_MILLION")
        self.output_rate = positive("BENCH_OUTPUT_USD_PER_MILLION")
        self.max_bytes = int(os.environ["BENCH_MAX_INPUT_BYTES"])
        self.max_tokens = int(os.environ["BENCH_MAX_OUTPUT_TOKENS"])
        if self.max_bytes <= 0 or self.max_tokens <= 0:
            raise ValueError("positive byte/token bounds required")
        if os.environ.get("BENCH_SERVICE_TIER") != "default":
            raise ValueError("explicit BENCH_SERVICE_TIER=default required")
        self.path = Path(os.environ["BENCH_SPEND_FILE"])
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def ledger(self):
        # A separate lock inode survives the atomic ledger replacement.
        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                data = json.loads(self.path.read_text())
                if (set(data) != {"usd", "trials", "pending", "blocked"}
                        or not isinstance(data["blocked"], bool)
                        or not isinstance(data["pending"], dict)
                        or not isinstance(data["trials"], dict)):
                    raise ValueError("invalid budget ledger")
                for value in [data["usd"], *data["trials"].values()]:
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError("invalid ledger spend")
            else:
                data = {"usd": 0.0, "trials": {}, "pending": {}, "blocked": False}
            yield data
            tmp = self.path.with_suffix(self.path.suffix + ".new")
            with tmp.open("w") as fh:
                json.dump(data, fh, allow_nan=False)
                fh.flush()
                os.fsync(fh.fileno())
            tmp.replace(self.path)

    def validate(self, body: dict, raw: bytes) -> None:
        if or_model(body.get("model")) != or_model(MODEL):
            raise ValueError("model is not explicitly priced")
        tokens = body.get("max_tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or not 0 < tokens <= self.max_tokens:
            raise ValueError("max_tokens is missing or exceeds priced limit")
        if len(raw) > self.max_bytes:
            raise ValueError("input exceeds priced byte bound")
        if body.get("service_tier", "default") != "default":
            raise ValueError("unpriced service tier")
        allowed = {"model", "messages", "system", "max_tokens", "stream", "tools",
                   "tool_choice", "temperature", "top_p", "top_k", "stop_sequences",
                   "metadata", "thinking", "output_config", "cache_control", "service_tier", "provider"}
        if set(body) - allowed:
            raise ValueError("unpriced request extension")

        def check_content(value):
            if isinstance(value, list):
                for child in value:
                    check_content(child)
            elif isinstance(value, dict):
                kind = value.get("type")
                if kind is not None and kind not in {
                    "text", "tool_use", "tool_result", "thinking", "redacted_thinking"}:
                    raise ValueError("unpriced content block")
                # Arbitrary tool arguments are data, not provider modalities.
                if "content" in value:
                    check_content(value["content"])
        check_content(body.get("messages"))
        check_content(body.get("system"))
        for tool in body.get("tools", []):
            if tool.get("type", "custom") != "custom":
                raise ValueError("unpriced server tool")
        # UTF-8 bytes overbound text tokens; 4x plus framing slack deliberately
        # over-reserves rather than estimating from a favorable tokenizer.
        bound = ((4 * self.max_bytes + 65536) * self.input_rate
                 + self.max_tokens * self.output_rate) / 1_000_000
        if bound > self.reserve:
            raise ValueError("request reservation is below worst-case token/rate bound")

    def acquire(self, run_id: str) -> str:
        token = uuid.uuid4().hex
        with self.ledger() as data:
            if data["blocked"] or data["pending"]:
                raise ValueError("unreconciled or in-flight request; budget fails closed")
            if (data["usd"] + self.reserve > self.total
                    or data["trials"].get(run_id, 0.0) + self.reserve > self.trial):
                raise ValueError("total or per-trial budget exhausted")
            data["pending"][token] = {"run_id": run_id, "usd": self.reserve}
        return token

    def settle(self, token: str, cost) -> None:
        with self.ledger() as data:
            pending = data["pending"][token]
            if (isinstance(cost, bool) or not isinstance(cost, (int, float))
                    or not math.isfinite(cost) or cost < 0):
                data["blocked"] = True
                return
            data["usd"] += cost
            run_id = pending["run_id"]
            data["trials"][run_id] = data["trials"].get(run_id, 0.0) + cost
            if cost > pending["usd"]:
                data["blocked"] = True
            del data["pending"][token]


def or_model(model: str | None) -> str | None:
    """Anthropic id -> OpenRouter slug: claude-sonnet-5-5 -> anthropic/claude-sonnet-5.5."""
    if not model or model.startswith("anthropic/"):
        return model
    m = re.match(r"^(claude-[a-z]+)-(\d+)-(\d+)(?:-\d{8})?$", model)
    if m:
        return f"anthropic/{m.group(1)}-{m.group(2)}.{m.group(3)}"
    m = re.match(r"^(claude-[a-z]+)-(\d+)(?:-\d{8})?$", model)
    return f"anthropic/{m.group(1)}-{m.group(2)}" if m else f"anthropic/{model}"


def _patch_body(body: dict) -> dict:
    meta = {"requested_model": body.get("model")}
    # Model pin is OFF by default: side-calls a harness deliberately routes to
    # a cheaper model (e.g. Claude Code's Haiku title/summary calls) are part
    # of its real cost profile. The report prices each row by the model the
    # response says served it. BENCH_PIN_MODEL=1 forces everything to MODEL.
    if PIN:
        body["model"] = MODEL
    system = body.get("system")
    has_identity = False
    if isinstance(system, str):
        has_identity = system.startswith(IDENTITY)
        if not has_identity and INJECT_IDENTITY:
            body["system"] = [{"type": "text", "text": IDENTITY}, {"type": "text", "text": system}] if system else [{"type": "text", "text": IDENTITY}]
    elif isinstance(system, list):
        first = system[0] if system else {}
        has_identity = isinstance(first, dict) and str(first.get("text", "")).startswith(IDENTITY)
        if not has_identity and INJECT_IDENTITY:
            body["system"] = [{"type": "text", "text": IDENTITY}] + system
    elif INJECT_IDENTITY:
        body["system"] = [{"type": "text", "text": IDENTITY}]
    meta["harness_sent_identity"] = has_identity
    meta["identity_injected"] = (not has_identity) and INJECT_IDENTITY
    meta["thinking"] = body.get("thinking")
    meta["output_config"] = body.get("output_config")
    meta["max_tokens"] = body.get("max_tokens")
    meta["n_tools"] = len(body.get("tools") or [])
    meta["n_messages"] = len(body.get("messages") or [])
    meta["tools_bytes"] = len(json.dumps(body.get("tools") or []))
    meta["system_bytes"] = len(json.dumps(body.get("system") or ""))
    return meta


def _merge_usage(acc: dict, usage: dict | None) -> None:
    if not usage:
        return
    for k, v in usage.items():
        if isinstance(v, (int, float)) and v is not None:
            if k in {"output_tokens", "cost"}:
                acc[k] = v  # cumulative in message_delta; settle the final bill
            else:
                acc[k] = v if k not in acc or acc[k] == 0 else acc[k]
        elif k == "cache_creation" and isinstance(v, dict):
            acc["cache_creation"] = v
        elif k == "cost_details" and isinstance(v, dict):
            acc["cost_details"] = v
        elif k == "server_tool_use" and isinstance(v, dict):
            acc["server_tool_use"] = v


async def proxy(request: Request) -> Response:
    run_id = request.path_params["run_id"]
    if run_id == "@current":
        # Codex reaches us through a LiteLLM translator with one fixed
        # api_base, so its runs are tagged via a file the runner rewrites
        # before each (strictly sequential) Codex run.
        run_id = Path(os.environ.get("BENCH_CURRENT_RUN_FILE", "current_run")).read_text().strip()
    path = request.path_params["path"]
    raw = await request.body()
    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in {"host", "content-length", "x-api-key", "authorization", "accept-encoding", "connection"}
    }
    if path.rstrip("/") == "api/hello":
        return JSONResponse({"ok": True})  # Claude Code connectivity ping
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", run_id):
        return JSONResponse({"error": "invalid run identifier"}, status_code=400)
    if path != "v1/messages" or request.method != "POST":
        return JSONResponse({"error": "only priced message requests are allowed"}, status_code=400)
    try:
        guard = BudgetGuard()
        body = json.loads(raw)
        # Validate before mutation so model pinning cannot conceal an unpriced call.
        guard.validate(body, raw)
        key = await upstream_key()
    except (KeyError, ValueError, TypeError, AttributeError, OSError):
        return JSONResponse({"error": "invalid request or missing explicit budget/auth configuration"}, status_code=402)
    headers["authorization"] = f"Bearer {key}"
    beta = [b.strip() for b in headers.get("anthropic-beta", "").split(",") if b.strip() and not b.strip().startswith("oauth-")]
    if beta:
        headers["anthropic-beta"] = ",".join(beta)
    else:
        headers.pop("anthropic-beta", None)
    headers.setdefault("anthropic-version", "2023-06-01")
    # httpx would otherwise advertise gzip; we pass raw bytes through and strip
    # content-encoding, so a compressed SSE body reached clients undecodable
    # (Claude Code then silently retried non-streaming -> double-counted calls).
    headers["accept-encoding"] = "identity"

    meta: dict = {}
    is_messages = path.rstrip("/").endswith("v1/messages") and request.method == "POST"
    is_count = path.rstrip("/").endswith("count_tokens")
    if (is_messages or is_count) and raw:
        body = json.loads(raw)
        if CAPTURE_BODIES and is_messages and run_id not in _seen_runs:
            _seen_runs.add(run_id)
            BODIES.mkdir(parents=True, exist_ok=True)
            (BODIES / f"{run_id}.first.json").write_text(json.dumps(body))
        if CAPTURE_BODIES and is_messages:
            # Last body per run holds the whole transcript: used offline to see
            # which skills a harness actually read. Overwritten each call.
            BODIES.mkdir(parents=True, exist_ok=True)
            (BODIES / f"{run_id}.last.json").write_text(json.dumps(body))
        meta = _patch_body(body) if is_messages else {"requested_model": body.get("model")}
        if is_messages and AUTO_CACHE and "cache_control" not in raw.decode("utf-8", "ignore"):
            # A harness that sends NO cache markers (Codex via the LiteLLM
            # Responses->Messages translator) would otherwise pay full input
            # price on every turn. Anthropic's top-level automatic caching is
            # the minimal, documented fix; flagged per row so the report can
            # also price the no-cache counterfactual.
            body["cache_control"] = {"type": "ephemeral"}
            meta["auto_cache"] = True
        if is_count and PIN:
            body["model"] = MODEL
        meta["upstream_model"] = body["model"] = or_model(body.get("model"))
        body["service_tier"] = "default"
        body["provider"] = {
            "order": ["anthropic"], "allow_fallbacks": False,
            "max_price": {"prompt": guard.input_rate, "completion": guard.output_rate},
        }
        raw = json.dumps(body).encode()
        try:
            guard.validate(body, raw)
        except (ValueError, TypeError, AttributeError):
            return JSONResponse({"error": "final wire body exceeds explicit bounds"}, status_code=402)

    url = f"{UPSTREAM}/{path}"
    if request.url.query:
        url += "?" + request.url.query
    try:
        reservation = guard.acquire(run_id)
    except (ValueError, KeyError, OSError, TypeError):
        return JSONResponse({"error": "budget ledger unavailable, exhausted, or unreconciled"}, status_code=402)
    started = time.time()
    upstream_req = client.build_request(request.method, url, headers=headers, content=raw)
    try:
        resp = await client.send(upstream_req, stream=True)
    except Exception:
        # The upstream may have billed before a disconnect: leave it unresolved.
        guard.settle(reservation, None)
        return JSONResponse({"error": "upstream transport failed; reconcile reservation"}, status_code=502)
    ctype = resp.headers.get("content-type", "")
    row = {"ts": started, "run_id": run_id, "path": path, "status": resp.status_code, **meta}
    passthrough_headers = {
        k: v for k, v in resp.headers.items() if k.lower() not in {"content-length", "content-encoding", "transfer-encoding", "connection"}
    }

    def write_row(usage: dict, extra: dict | None = None) -> None:
        # OpenRouter returns the billed amount in usage.cost: it feeds the
        # hard budget stop and is reported beside our list-price computation.
        guard.settle(reservation, usage.get("cost"))
        row["latency_s"] = round(time.time() - started, 3)
        row["usage"] = usage
        if extra:
            row.update(extra)
        with LOG.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    if "text/event-stream" in ctype:
        async def gen():
            usage: dict = {}
            buf = b""
            stop = None
            err = None
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
                    buf = (buf + chunk).replace(b"\r\n", b"\n")
                    while b"\n\n" in buf:
                        event, buf = buf.split(b"\n\n", 1)
                        for line in event.split(b"\n"):
                            if line.startswith(b"data:"):
                                try:
                                    d = json.loads(line[5:].strip())
                                except Exception:
                                    continue
                                t = d.get("type")
                                if t == "message_start":
                                    row["served_model"] = d.get("message", {}).get("model")
                                    _merge_usage(usage, d.get("message", {}).get("usage"))
                                elif t == "message_delta":
                                    _merge_usage(usage, d.get("usage"))
                                    stop = d.get("delta", {}).get("stop_reason") or stop
                                elif t == "error":
                                    err = "upstream_error"
            finally:
                await resp.aclose()
                write_row(usage, {"stream": True, "stop_reason": stop, "error": err})

        return StreamingResponse(gen(), status_code=resp.status_code, headers=passthrough_headers)

    content = await resp.aread()
    await resp.aclose()
    usage, extra = {}, {"stream": False}
    try:
        d = json.loads(content)
        if is_messages:
            usage = d.get("usage") or {}
            row["served_model"] = d.get("model")
            extra["stop_reason"] = d.get("stop_reason")
        if is_count:
            extra["count_input_tokens"] = d.get("input_tokens")
        if resp.status_code >= 400:
            extra["error"] = "upstream_error"
    except Exception:
        pass
    write_row(usage, extra)
    return Response(content, status_code=resp.status_code, headers=passthrough_headers)


async def health(_: Request) -> Response:
    return JSONResponse({"ok": True, "model": MODEL})


app = Starlette(
    routes=[
        Route("/health", health),
        Route("/r/{run_id}/{path:path}", proxy, methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"]),
    ]
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("BENCH_PORT", "8787")), log_level="warning")
