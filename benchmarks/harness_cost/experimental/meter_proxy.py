"""Experimental text-only OpenRouter meter with explicit, fail-closed budgets.

All credentials come from OPENROUTER_API_KEY and stay in memory. Total and
per-trial caps reserve each request's own byte/token upper bound at its model's
live rate before dispatch; unknown billing, disconnects, cancellations and
malformed ledgers stop later traffic. Only the natively requested, live-priced
models (``NATIVE_MODELS``) and the default service tier are accepted; anything
else is refused and voids the run. This is not provider-side billing
enforcement: the rate card must bound all charges.
Prompt capture is opt-in; logs contain usage/count metadata, never auth headers.

One ledger serves every paid channel. ``channel`` is ``model`` for Messages
traffic priced here and ``classifier`` for Radient decisions priced by
``lop_edge.py``; both reserve and settle through the same ``BudgetGuard`` file,
so total and per-trial caps bound the sum of both, not each separately.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

# Only the provider JSON boundary is dynamic: nested schemas, tool arguments,
# and provider usage extensions must survive without a lossy local schema.
WireObject = dict[str, Any]

# Local Operator sends ``max_tokens=128000`` on main turns (its native output
# budget for the benchmark model). A lower cap refuses every LoP turn with a
# local 402 the product cannot recover from, so a configuration below the
# native value is rejected instead of silently measuring a broken harness.
NATIVE_MAIN_OUTPUT_TOKENS = 128000
CHANNELS = ("model", "classifier")

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


# Every model a benchmarked harness natively requests on the wire (captured in
# logs/requests.jsonl: all seven send Sonnet 5.5 main turns; omp and OpenCode
# also send Haiku 5.5 helpers -- 425 omp Haiku calls over 25 runs, 57 OpenCode;
# Claude Code's 203 captured rows are all Sonnet). Pricing only the main model
# would refuse those helpers on paid runs and silently change those harnesses,
# so the card is exactly this allowlist:
# nothing less (a missing native model refuses startup) and nothing more (any
# other model -- e.g. a failover target -- is refused AND voids the run).
NATIVE_MODELS = ("anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5")
# A card is fetched from the live price endpoint at run start; an old one may
# not describe today's price, so it is refused rather than trusted.
RATE_CARD_MAX_AGE_S = 6 * 3600
# Tokens per request beyond the body bytes: role/tool framing, the system
# prompt template and image-free provider overhead. Generous on purpose; any
# bill above the reservation still latches at settle.
FRAMING_TOKENS = 8192
# The durable ledger file's exact schema (a pre-voids ledger reads as invalid).
LEDGER_KEYS = ("usd", "trials", "pending", "blocked", "channels", "models", "voids", "cost_caps")


class CostCapReached(ValueError):
    """The trial's SETTLED spend reached its cap: a primary cost-cap failure.

    PREREGISTRATION §9.2 counts a cost-cap stop as Y=0 with its cost retained,
    never as an invalid attempt to rerun, so callers must not void on it.
    """


def _provider_fault(status: int) -> bool:
    """Statuses that are the provider's fault, not the harness's (§9.2).

    Only overload/rate-limit/server errors (429, 5xx incl. Anthropic's 529)
    are external infrastructure. Any other 4xx (e.g. "prompt is too long") is
    a response to what the harness sent: its own behaviour, counted as is.
    """
    return status == 429 or status >= 500


def _per_million(pricing: dict[str, Any], key: str, *, required: bool) -> float:
    """OpenRouter states USD per token as a decimal string; return USD/M tokens."""
    raw = pricing.get(key)
    if raw is None and not required:
        return 0.0
    if not isinstance(raw, str):
        raise ValueError(f"price {key} missing or not a decimal string")
    value = float(raw) * 1_000_000
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"price {key} is not a finite nonnegative rate")
    return value


def rate_card_from_models_listing(
    raw: bytes, *, source_url: str, fetched_at: float
) -> dict[str, Any]:
    """Build the run's rate card from the raw live ``/api/v1/models`` response.

    The input rate used for reservations is the MAXIMUM of prompt and
    cache-write rates, so a request whose whole prefix is a cache write is still
    over-reserved. The response digest and fetch time are kept as provenance so
    a published result names the exact price statement it was metered under.
    """
    listing = json.loads(raw)
    rows = {row.get("id"): row.get("pricing") for row in listing.get("data", [])}
    models: dict[str, dict[str, float]] = {}
    for model in NATIVE_MODELS:
        pricing = rows.get(model)
        if not isinstance(pricing, dict):
            raise ValueError(f"live price listing has no {model}")
        prompt = _per_million(pricing, "prompt", required=True)
        completion = _per_million(pricing, "completion", required=True)
        write = _per_million(pricing, "input_cache_write", required=False)
        models[model] = {
            "prompt": prompt,
            "completion": completion,
            "input_cache_write": write,
            "reserve_input": max(prompt, write),
        }
    return {
        "source": source_url,
        "fetched_at": fetched_at,
        "response_sha256": hashlib.sha256(raw).hexdigest(),
        "models": models,
    }


def _load_rate_card(path: Path) -> dict[str, dict[str, float]]:
    card = json.loads(path.read_text())
    models = card.get("models")
    if not isinstance(models, dict) or set(models) != set(NATIVE_MODELS):
        raise ValueError("rate card must price exactly the native benchmark models")
    age = time.time() - float(card["fetched_at"])
    if not 0 <= age <= RATE_CARD_MAX_AGE_S:
        raise ValueError("rate card is stale; refetch the live price listing")
    if not isinstance(card.get("response_sha256"), str) or not card.get("source"):
        raise ValueError("rate card lacks price provenance")
    for rates in models.values():
        for key in ("prompt", "completion", "reserve_input"):
            value = rates.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("rate card holds a non-finite or negative rate")
    return models


def _pid_alive(pid: object) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class BudgetGuard:
    """Durable, serialized reservations prevent retries/concurrency overspending.

    Built ONCE when a meter or edge starts: a missing or invalid setting fails
    startup instead of failing open on some later request.

    Unknown billing keeps the reservation and blocks ALL later requests. Recovery
    requires operator reconciliation, never a reset on malformed/missing usage.
    Only text/custom tools and the explicitly selected default tier are supported.

    Each reservation is that request's OWN upper bound: (body bytes + framing)
    input tokens at the model's highest input rate plus its own ``max_tokens``
    at the output rate. UTF-8 bytes overbound text tokens. A bill above it
    latches at settle. Reservations gate ONLY the total campaign budget (with
    every pending reservation summed), so the total allowance is never
    overspent.

    The per-trial cap is enforced on SETTLED spend: a request is admitted while
    the trial's settled spend is below the cap, whatever its native
    ``max_tokens``. Gating the trial cap on the worst-case reservation would
    make the effective cap ``cap - reserve(next request)`` -- about $3.7 of $5
    for the six 128k-output harnesses but $4.6 for Claude Code (32k), i.e. not
    the same limit for all harnesses (PREREGISTRATION §9.1). Overshoot past
    the cap is recorded in ``cost_caps`` and the trial ends as a primary
    cost-cap failure (cost counted, Y=0, §9.2), never a void.

    A reservation that never settles must not fail open (the base rule was:
    any pending entry refuses everything). Three backstops restore that:
    the owning process settles-or-latches on every exit path, including
    cancellation; ``close()`` latches its own unsettled tokens at teardown; and
    admission latches when a pending entry's owner process is dead or the entry
    is older than ``BENCH_MAX_REQUEST_S``.

    ``voids`` records, per run, an infrastructure or metering reason it can no
    longer be published as measured (invalid + at most one rerun, §9.2): a
    provider 5xx/429, a transport failure, an unpriced request the meter
    refused, a campaign-budget or admission-deadline refusal. A void is per
    run and does not block other runs; ``blocked`` is global and means spend
    may be unaccounted for.
    """

    def __init__(self):
        self.total = positive("BENCH_BUDGET_USD")
        self.trial = positive("BENCH_TRIAL_BUDGET_USD")
        self.rates = _load_rate_card(Path(os.environ["BENCH_RATE_CARD"]))
        self.max_bytes = int(os.environ["BENCH_MAX_INPUT_BYTES"])
        self.max_tokens = int(os.environ["BENCH_MAX_OUTPUT_TOKENS"])
        if self.max_bytes <= 0 or self.max_tokens <= 0:
            raise ValueError("positive byte/token bounds required")
        if self.max_tokens < NATIVE_MAIN_OUTPUT_TOKENS:
            raise ValueError("BENCH_MAX_OUTPUT_TOKENS must admit the native 128000 main cap")
        # Bounded queueing, not refusal: harnesses natively overlap side calls
        # (LoP's title errand and classifier run beside the turn; Claude Code
        # issues concurrent helpers). Refusing them turns a priced call into a
        # native fallback (e.g. a placeholder title) and changes behaviour.
        self.admission_wait = positive("BENCH_ADMISSION_WAIT_S")
        self.max_request_s = positive("BENCH_MAX_REQUEST_S")
        if os.environ.get("BENCH_SERVICE_TIER") != "default":
            raise ValueError("explicit BENCH_SERVICE_TIER=default required")
        self.path = Path(os.environ["BENCH_SPEND_FILE"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mine: set[str] = set()

    def rates_for(self, model: str | None) -> dict[str, float]:
        rates = self.rates.get(model or "")
        if rates is None:
            raise ValueError(f"unpriced model {model}")
        return rates

    def reservation_usd(self, model: str | None, raw: bytes, max_tokens: int) -> float:
        rates = self.rates_for(model)
        input_tokens = len(raw) + FRAMING_TOKENS
        return (input_tokens * rates["reserve_input"] + max_tokens * rates["completion"]) / 1e6

    @contextmanager
    def ledger(self):
        # A separate lock inode survives the atomic ledger replacement.
        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                data = json.loads(self.path.read_text())
                if (
                    set(data) != set(LEDGER_KEYS)
                    or not isinstance(data["blocked"], bool)
                    or not all(isinstance(data[k], dict) for k in LEDGER_KEYS if k != "blocked"
                               and k != "usd")
                    or set(data["channels"]) != set(CHANNELS)
                ):
                    raise ValueError("invalid budget ledger")
                pending_usd = [
                    entry.get("usd") if isinstance(entry, dict) else None
                    for entry in data["pending"].values()
                ]
                for value in [
                    data["usd"],
                    *data["trials"].values(),
                    *data["channels"].values(),
                    *data["models"].values(),
                    *pending_usd,
                ]:
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value < 0
                    ):
                        raise ValueError("invalid ledger spend")
            else:
                data = {
                    "usd": 0.0,
                    "trials": {},
                    "pending": {},
                    "blocked": False,
                    "channels": {name: 0.0 for name in CHANNELS},
                    "models": {},
                    "voids": {},
                    "cost_caps": {},
                }
            yield data
            tmp = self.path.with_suffix(self.path.suffix + ".new")
            with tmp.open("w") as fh:
                json.dump(data, fh, allow_nan=False)
                fh.flush()
                os.fsync(fh.fileno())
            tmp.replace(self.path)

    def _record_cost_cap(self, data: dict[str, Any], run_id: str) -> None:
        """Record (or refresh, as late in-flight bills settle) a cost-cap stop."""
        data["cost_caps"][run_id] = {
            "settled_usd": data["trials"].get(run_id, 0.0),
            "cap_usd": self.trial,
        }

    @staticmethod
    def _void(data: dict[str, Any], run_id: str, reason: str) -> None:
        reasons = data["voids"].setdefault(run_id, [])
        if reason not in reasons:
            reasons.append(reason)

    def void(self, run_id: str, reason: str) -> None:
        """Mark one run invalid (rerun policy) without blocking other runs."""
        with self.ledger() as data:
            self._void(data, run_id, reason)

    def latch(self, run_id: str, reason: str) -> None:
        """Void the run AND block every later admission until reconciliation.

        For traffic whose bill or intent is unknown (a denied POST, an
        unclassifiable CONNECT, an abandoned reservation): spend can no longer
        be shown to be fully measured.
        """
        with self.ledger() as data:
            data["blocked"] = True
            self._void(data, run_id, reason)

    def validate(self, body: WireObject, raw: bytes) -> None:
        self.rates_for(or_model(body.get("model")))
        tokens = body.get("max_tokens")
        if (
            isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or not 0 < tokens <= self.max_tokens
        ):
            raise ValueError("max_tokens is missing or exceeds priced limit")
        if len(raw) > self.max_bytes:
            raise ValueError("input exceeds priced byte bound")
        if body.get("service_tier", "default") != "default":
            raise ValueError("unpriced service tier")
        allowed = {
            "model",
            "messages",
            "system",
            "max_tokens",
            "stream",
            "tools",
            "tool_choice",
            "temperature",
            "top_p",
            "top_k",
            "stop_sequences",
            "metadata",
            "thinking",
            "output_config",
            "cache_control",
            "service_tier",
            "provider",
            # LoP's first-party cache diagnostics opt-in (configure.py turn
            # requests; clients.py adds it when the base is api.anthropic.com).
            # Price-neutral request metadata: forwarded unchanged so the
            # product's native diagnostics/refusal-retry path stays intact.
            "diagnostics",
        }
        if set(body) - allowed:
            raise ValueError("unpriced request extension")

        def check_content(value):
            if isinstance(value, list):
                for child in value:
                    check_content(child)
            elif isinstance(value, dict):
                kind = value.get("type")
                if kind is not None and kind not in {
                    "text",
                    "tool_use",
                    "tool_result",
                    "thinking",
                    "redacted_thinking",
                }:
                    raise ValueError("unpriced content block")
                # Arbitrary tool arguments are data, not provider modalities.
                if "content" in value:
                    check_content(value["content"])

        check_content(body.get("messages"))
        check_content(body.get("system"))
        for tool in body.get("tools", []):
            if tool.get("type", "custom") != "custom":
                raise ValueError("unpriced server tool")

    def _reap(self, data: dict[str, Any]) -> None:
        """Latch on any reservation nobody can settle any more (F1 backstop)."""
        now = time.time()
        for entry in data["pending"].values():
            if entry.get("abandoned"):
                continue
            if not _pid_alive(entry.get("pid")):
                reason = "orphaned reservation: owner process exited without settling"
            elif now - float(entry.get("ts", 0)) > self.max_request_s:
                reason = "stale reservation older than BENCH_MAX_REQUEST_S"
            else:
                continue
            entry["abandoned"] = True
            data["blocked"] = True
            self._void(data, entry["run_id"], reason)

    def try_acquire(self, run_id: str, channel: str, usd: float, model: str) -> str | None:
        """Reserve ``usd`` against settled spend PLUS every pending reservation.

        Returns a token when the reservation fits now, ``None`` when it would fit
        once in-flight reservations settle (the caller may wait), and raises when
        it can never fit, the run is void, or the ledger is latched. Summing
        pending exposure is what makes concurrent dispatch safe.
        """
        if channel not in CHANNELS:
            raise ValueError("unknown ledger channel")
        token = uuid.uuid4().hex
        refusal = None
        token_or_wait: str | None = None
        with self.ledger() as data:
            # A reaped orphan must be PERSISTED even though this admission is
            # refused, so the refusal is returned out of the ledger block
            # (raising inside it would skip the write-back).
            self._reap(data)
            trial = data["trials"].get(run_id, 0.0)
            capped = False
            if data["blocked"]:
                refusal = "unreconciled request; budget fails closed"
            elif run_id in data["voids"]:
                refusal = "run is void; no further admissions"
            elif trial >= self.trial:
                capped = True
                self._record_cost_cap(data, run_id)
                refusal = "trial cost cap reached on settled spend"
            elif data["usd"] + usd > self.total:
                refusal = "total campaign budget exhausted"
            if refusal is None:
                token_or_wait = self._reserve(data, token, run_id, channel, usd, model)
        if refusal is not None:
            raise (CostCapReached if capped else ValueError)(refusal)
        if token_or_wait is not None:
            self._mine.add(token_or_wait)
        return token_or_wait

    def _reserve(
        self,
        data: dict[str, Any],
        token: str,
        run_id: str,
        channel: str,
        usd: float,
        model: str,
    ) -> str | None:
        """Add the reservation if it fits the TOTAL beside every pending one, else ``None``.

        No per-trial exposure term on purpose (F7): the trial cap is decided on
        settled spend in ``try_acquire``, so it binds identically for every
        harness regardless of native ``max_tokens``.
        """
        exposure = sum(p["usd"] for p in data["pending"].values())
        if data["usd"] + exposure + usd > self.total:
            return None
        data["pending"][token] = {
            "run_id": run_id,
            "usd": usd,
            "channel": channel,
            "model": model,
            "pid": os.getpid(),
            "ts": time.time(),
        }
        return token

    async def acquire(self, run_id: str, channel: str, usd: float, model: str) -> str:
        """Wait (bounded by BENCH_ADMISSION_WAIT_S) for room, then reserve.

        A refusal for an infrastructure/metering reason -- campaign budget,
        deadline, latched ledger -- changes what the harness natively does next,
        so it voids the run with its reason (F5). A trial cost-cap stop is the
        harness's own primary outcome and is NOT voided (F7).
        """
        deadline = time.monotonic() + self.admission_wait
        try:
            while True:
                token = self.try_acquire(run_id, channel, usd, model)
                if token is not None:
                    return token
                if time.monotonic() >= deadline:
                    raise ValueError("admission wait expired with exposure still pending")
                await asyncio.sleep(0.02)
        except CostCapReached:
            raise
        except ValueError as exc:
            with contextlib.suppress(ValueError, OSError):
                self.void(run_id, f"admission refused: {exc}")
            raise

    def abandon(self, token: str, reason: str) -> None:
        """Settle-or-latch fallback: dispatched, never settled -> latch, keep exposure."""
        with self.ledger() as data:
            entry = data["pending"].get(token)
            if entry is not None and not entry.get("abandoned"):
                entry["abandoned"] = True
                data["blocked"] = True
                self._void(data, entry["run_id"], f"abandoned reservation: {reason}")
        self._mine.discard(token)

    def close(self) -> None:
        """Teardown backstop: any reservation this guard still owns latches."""
        for token in list(self._mine):
            with contextlib.suppress(ValueError, OSError):
                self.abandon(token, "teardown with request in flight")

    def settle(self, token: str, cost, qualified: bool = True) -> None:
        """Settle a reservation from a billed amount; anything unknown latches.

        An invalid/missing cost keeps the reservation as exposure and blocks
        all later admissions. ``qualified=False`` records a finite stated bill
        (it was charged) but still latches: the bill's contract was not the
        reviewed one, so later spend cannot be bounded.
        """
        self._mine.discard(token)
        with self.ledger() as data:
            pending = data["pending"][token]
            if (
                isinstance(cost, bool)
                or not isinstance(cost, (int, float))
                or not math.isfinite(cost)
                or cost < 0
            ):
                pending["abandoned"] = True
                data["blocked"] = True
                self._void(data, pending["run_id"], "unbilled or unknown charge")
                return
            data["usd"] += cost
            run_id = pending["run_id"]
            data["trials"][run_id] = data["trials"].get(run_id, 0.0) + cost
            channel = pending.get("channel", "model")
            data["channels"][channel] = data["channels"][channel] + cost
            model = pending.get("model") or "unknown"
            data["models"][model] = data["models"].get(model, 0.0) + cost
            if cost > pending["usd"] or not qualified:
                data["blocked"] = True
                self._void(data, run_id, "charge above reservation or unqualified bill")
            if data["trials"][run_id] >= self.trial:
                self._record_cost_cap(data, run_id)
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


def _patch_body(body: WireObject) -> dict[str, object]:
    meta: dict[str, object] = {"requested_model": body.get("model")}
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
            body["system"] = (
                [{"type": "text", "text": IDENTITY}, {"type": "text", "text": system}]
                if system
                else [{"type": "text", "text": IDENTITY}]
            )
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


def _merge_usage(acc: WireObject, usage: WireObject | None) -> None:
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


async def proxy(guard: BudgetGuard, request: Request) -> Response:
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
        if k.lower()
        not in {
            "host",
            "content-length",
            "x-api-key",
            "authorization",
            "accept-encoding",
            "connection",
        }
    }
    if path.rstrip("/") == "api/hello":
        return JSONResponse({"ok": True})  # Claude Code connectivity ping
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", run_id):
        return JSONResponse({"error": "invalid run identifier"}, status_code=400)
    if path != "v1/messages" or request.method != "POST":
        return JSONResponse({"error": "only priced message requests are allowed"}, status_code=400)
    try:
        body = json.loads(raw)
        # Validate before mutation so model pinning cannot conceal an unpriced call.
        guard.validate(body, raw)
        key = await upstream_key()
    except (KeyError, ValueError, TypeError, AttributeError, OSError) as exc:
        # The harness asked for something the benchmark cannot price; refusing
        # it changes the run, so the run is void with the reason (D3).
        reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
        with contextlib.suppress(ValueError, OSError):
            guard.void(run_id, f"refused request: {reason}")
        return JSONResponse({"error": "request refused by the meter", "reason": reason},
                            status_code=402)
    headers["authorization"] = f"Bearer {key}"
    beta = [
        b.strip()
        for b in headers.get("anthropic-beta", "").split(",")
        if b.strip() and not b.strip().startswith("oauth-")
    ]
    if beta:
        headers["anthropic-beta"] = ",".join(beta)
    else:
        headers.pop("anthropic-beta", None)
    headers.setdefault("anthropic-version", "2023-06-01")
    # httpx would otherwise advertise gzip; we pass raw bytes through and strip
    # content-encoding, so a compressed SSE body reached clients undecodable
    # (Claude Code then silently retried non-streaming -> double-counted calls).
    headers["accept-encoding"] = "identity"

    meta: dict[str, object] = {}
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
            "order": ["anthropic"],
            "allow_fallbacks": False,
            "max_price": {"prompt": guard.rates_for(body["model"])["prompt"],
                          "completion": guard.rates_for(body["model"])["completion"]},
        }
        raw = json.dumps(body).encode()
        try:
            guard.validate(body, raw)
        except (ValueError, TypeError, AttributeError) as exc:
            with contextlib.suppress(ValueError, OSError):
                guard.void(run_id, f"refused final wire body: {exc}")
            return JSONResponse(
                {"error": "final wire body exceeds explicit bounds"}, status_code=402
            )

    url = f"{UPSTREAM}/{path}"
    if request.url.query:
        url += "?" + request.url.query
    priced_model = body["model"]
    reserve = guard.reservation_usd(priced_model, raw, body["max_tokens"])
    try:
        reservation = await guard.acquire(run_id, "model", reserve, priced_model)
    except CostCapReached:
        return JSONResponse(
            {"error": "trial cost cap reached; the trial ends as a cost-cap failure"},
            status_code=402,
        )
    except (ValueError, KeyError, OSError, TypeError):
        return JSONResponse(
            {"error": "budget ledger unavailable, exhausted, or unreconciled"}, status_code=402
        )
    started = time.time()
    upstream_req = client.build_request(request.method, url, headers=headers, content=raw)
    # Settle-or-latch (F1): from here until a settle, ANY exit -- an exception,
    # a cancellation at shutdown, a mid-body reset -- abandons the reservation,
    # which latches the ledger and voids the run. ``owned`` is handed to the
    # streaming generator once it takes over settlement.
    owned = True
    try:
        try:
            resp = await client.send(upstream_req, stream=True)
        except Exception:
            # The upstream may have billed before a disconnect: leave it unresolved.
            owned = False
            guard.settle(reservation, None)
            return JSONResponse(
                {"error": "upstream transport failed; reconcile reservation"}, status_code=502
            )
        ctype = resp.headers.get("content-type", "")
        row = {
            "ts": started,
            "run_id": run_id,
            "channel": "model",
            "priced_model": priced_model,
            "reserve_usd": reserve,
            "path": path,
            "status": resp.status_code,
            **meta,
        }
        passthrough_headers = {
            k: v
            for k, v in resp.headers.items()
            if k.lower()
            not in {"content-length", "content-encoding", "transfer-encoding", "connection"}
        }
        status = resp.status_code

        def write_row(usage: WireObject, extra: dict[str, object] | None = None) -> None:
            # OpenRouter returns the billed amount in usage.cost: it feeds the
            # hard budget stop and is reported beside our list-price computation.
            guard.settle(reservation, usage.get("cost"))
            if _provider_fault(status):
                # Provider overload/rate-limit/server error: external
                # infrastructure, so the attempt is invalid and may be rerun
                # once (§9.2), never counted as a measured result. A harness-
                # caused 4xx is NOT here: its bill settled above like any other
                # (a missing bill still latches) and the trial continues.
                guard.void(run_id, f"provider status {status}")
            row["latency_s"] = round(time.time() - started, 3)
            row["usage"] = usage
            if extra:
                row.update(extra)
            with LOG.open("a") as fh:
                fh.write(json.dumps(row) + "\n")

        if "text/event-stream" in ctype:

            async def gen():
                usage: WireObject = {}
                buf = b""
                stop = None
                err = None
                settled = False
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
                    write_row(usage, {"stream": True, "stop_reason": stop, "error": err})
                    settled = True
                finally:
                    await resp.aclose()
                    if not settled:
                        # Client disconnect or cancellation mid-stream: the
                        # partial usage is not a complete bill (unknown -> latch).
                        write_row(usage, {"stream": True, "stop_reason": stop, "error": err,
                                          "incomplete": True})
                        guard.abandon(reservation, "stream ended before completion")

            owned = False  # the generator's finally now settles or latches
            return StreamingResponse(gen(), status_code=status, headers=passthrough_headers)

        try:
            content = await resp.aread()
        finally:
            await resp.aclose()
        usage: WireObject = {}
        extra: dict[str, object] = {"stream": False}
        try:
            d = json.loads(content)
            if is_messages:
                usage = d.get("usage") or {}
                row["served_model"] = d.get("model")
                extra["stop_reason"] = d.get("stop_reason")
            if status >= 400:
                extra["error"] = "upstream_error"
        except Exception:
            pass
        owned = False
        write_row(usage, extra)
        return Response(content, status_code=status, headers=passthrough_headers)
    finally:
        if owned:
            guard.abandon(reservation, "request ended before settlement")


def make_app(guard: BudgetGuard) -> Starlette:
    """The meter app around ONE guard built at startup (F6)."""

    async def health(_: Request) -> Response:
        return JSONResponse({"ok": True, "models": sorted(guard.rates)})

    async def route(request: Request) -> Response:
        return await proxy(guard, request)

    return Starlette(
        routes=[
            Route("/health", health),
            Route(
                "/r/{run_id}/{path:path}",
                route,
                methods=["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS"],
            ),
        ]
    )


if __name__ == "__main__":
    import uvicorn

    # Built before the socket opens: a misconfigured meter never serves.
    _guard = BudgetGuard()
    try:
        uvicorn.run(
            make_app(_guard),
            host="127.0.0.1",
            port=int(os.environ.get("BENCH_PORT", "8787")),
            log_level="warning",
        )
    finally:
        _guard.close()
