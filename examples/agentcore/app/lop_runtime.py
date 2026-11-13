"""Local Operator sessions keyed by AgentCore ``runtimeSessionId``.

AgentCore Runtime gives every ``runtimeSessionId`` its own microVM and keeps it
alive between invocations until the session idles out. This module keeps ONE
Local Operator session per runtime session id inside that process, so turn two
of a conversation lands in the same lop session as turn one (history, files in
the working directory, tool state) without the caller re-sending history.

It drives lop through its public embedding surface, ``local_operator.sdk``:
the same session object ``lop exec`` drives, with lop's own tools, compaction
and approvals. There is deliberately no agent loop here.

Kept free of the AgentCore SDK so it can be unit-tested offline; ``main.py``
wires it to ``BedrockAgentCoreApp``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

log = logging.getLogger("lop_agentcore.runtime")

# The lop hosting id for "any OpenAI-compatible server at an explicit URL"
# (docs/LOCAL_PROVIDERS.md). The Bedrock bridge is such a server.
HOSTING = "openai-compatible"

# Events forwarded to the caller. Everything else (usage ticks, compaction
# bookkeeping, full message snapshots) stays in the microVM's logs.
_FORWARDED = {
    "message_update",
    "tool_execution_start",
    "tool_execution_end",
    "agent_end",
}

# Tool results can be large (a file dump); the stream carries a preview only.
_RESULT_PREVIEW_CHARS = 2000


class PayloadError(ValueError):
    """The invocation payload is not a prompt this runtime can run."""


@dataclass(frozen=True)
class RuntimeSettings:
    """Everything the runtime reads from its environment, resolved once.

    Every value is an AgentCore runtime environment variable set by the
    CloudFormation template, so a deployment changes behaviour without a
    rebuild.
    """

    model_id: str
    home: Path
    approvals: str = "auto"
    tools: tuple[str, ...] | None = None
    context_window: int = 200_000
    max_output_tokens: int = 16_000

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "RuntimeSettings":
        env = dict(os.environ if env is None else env)
        model_id = env.get("LOP_MODEL_ID", "").strip()
        if not model_id:
            raise RuntimeError("LOP_MODEL_ID is required (the Bedrock model or inference profile)")
        tools_raw = env.get("LOP_TOOLS", "").strip()
        return cls(
            model_id=model_id,
            # NOT under /tmp: the SDK refuses volatile roots by default because
            # a purged store loses transcripts mid-run.
            home=Path(env.get("LOP_HOME", "/var/lib/lop")),
            approvals=env.get("LOP_APPROVALS", "auto").strip() or "auto",
            tools=tuple(t.strip() for t in tools_raw.split(",") if t.strip()) or None,
            context_window=int(env.get("LOP_CONTEXT_WINDOW", "200000")),
            max_output_tokens=int(env.get("LOP_MAX_OUTPUT_TOKENS", "16000")),
        )


@dataclass(frozen=True)
class Turn:
    """One invocation's request, normalised from the payload."""

    prompt: str
    model: str | None = None


def parse_payload(payload: Any) -> Turn:
    """Accept ``{"prompt": ...}`` (the AgentCore convention) or ``{"message": ...}``.

    ``model`` optionally selects another Bedrock model for a NEW session; it
    must also be granted to the execution role, or Bedrock answers
    AccessDenied (which is surfaced to the caller verbatim).
    """
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError:
            payload = {"prompt": payload}
    if not isinstance(payload, dict):
        raise PayloadError('payload must be a JSON object such as {"prompt": "..."}')
    prompt = payload.get("prompt", payload.get("message"))
    if not isinstance(prompt, str) or not prompt.strip():
        raise PayloadError('payload needs a non-empty "prompt" string')
    model = payload.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise PayloadError('"model" must be a non-empty string when given')
    return Turn(prompt=prompt, model=model.strip() if model else None)


def write_lop_config(
    config_dir: Path,
    *,
    base_url: str,
    settings: RuntimeSettings,
    extra_models: tuple[str, ...] = (),
) -> Path:
    """Point lop's ``openai-compatible`` hosting at the bridge, non-interactively.

    This is the same ``providers.<id>`` block ``/login openai-compatible``
    writes after its prompts (``base_url`` plus per-model overrides). The model
    budget is declared because the bridge cannot report a context window and
    lop must not guess one (an undeclared model gets a conservative 4K window
    and a normal prompt is refused as too large). ``extra_models`` declares
    per-session model overrides the same way. JSON is valid YAML, so no YAML
    dependency here.
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    model_overrides = {
        model: {
            "context_window": settings.context_window,
            "max_output_tokens": settings.max_output_tokens,
            "supports_tools": True,
            "supports_images": True,
        }
        for model in (settings.model_id, *extra_models)
    }
    # lop's store reads settings only under ``values:``; a top-level key is
    # ignored with a warning. ``hosting``/``model_name`` make the bridge the
    # default too, so ``lop`` run by hand inside the microVM uses it as well.
    config = {
        "values": {
            "hosting": HOSTING,
            "model_name": settings.model_id,
            "providers": {HOSTING: {"base_url": base_url, "models": model_overrides}},
        }
    }
    path = config_dir / "config.yml"
    path.write_text(json.dumps(config, indent=2) + "\n")
    return path


def approval_policy(name: str) -> Any:
    """Map ``LOP_APPROVALS`` onto an SDK ``ApprovalPolicy``.

    * ``auto`` (default): every gated tool call is approved. Nobody is
      available to answer a prompt inside a headless microVM, and the microVM
      is the sandbox: it is per session, discarded on idle, and its only
      credentials are the execution role's (scoped to Bedrock invoke and
      logging by the template).
    * ``refuse``: gated calls (writes, shell) fail with a typed refusal — a
      read-only research agent.
    * ``tool_a,tool_b``: only the named tools are reachable and stand as their
      own approval (``ApprovalPolicy.declared``).
    """
    from local_operator.sdk import ApprovalPolicy

    if name == "auto":
        return ApprovalPolicy.auto()
    if name == "refuse":
        return ApprovalPolicy.refuse()
    tools = [t.strip() for t in name.split(",") if t.strip()]
    if not tools:
        raise RuntimeError(f"LOP_APPROVALS={name!r} is not auto, refuse or a tool list")
    return ApprovalPolicy.declared(tools)


def _result_text(result: Any) -> str:
    """The text a tool returned: lop wraps it as ``{"content": [{"type": "text"}]}``."""
    if isinstance(result, str):
        return result
    if isinstance(result, dict) and isinstance(result.get("content"), list):
        texts = [
            part.get("text", "")
            for part in result["content"]
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        if texts:
            return "\n".join(texts)
    return json.dumps(result, default=str)


def project_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """Reduce lop's ``exec --json`` event shape to what a remote caller needs.

    Input is ``headless_print.printable_event`` output — lop's one JSON
    projection of its events — so this only filters and trims, it does not
    re-serialize lop's event model.
    """
    kind = event.get("type")
    if kind not in _FORWARDED:
        return None
    if kind == "message_update":
        delta = event.get("delta")
        text = delta.get("text") if isinstance(delta, dict) else delta
        if not isinstance(text, str) or not text:
            return None
        return {"type": "text", "text": text}
    if kind == "tool_execution_start":
        return {
            "type": "tool_start",
            "tool": event.get("tool_name"),
            "args": event.get("args"),
        }
    if kind == "tool_execution_end":
        preview = _result_text(event.get("result"))
        return {
            "type": "tool_end",
            "tool": event.get("tool_name"),
            "is_error": bool(event.get("is_error")),
            "result": preview[:_RESULT_PREVIEW_CHARS],
        }
    # agent_end: the turn's terminal frame.
    return {"type": "end", "error": event.get("error")}


def session_event(key: str, model_in_effect: str, requested: str | None) -> dict[str, Any]:
    """The first frame of every turn: which session and which model ran it.

    A lop session keeps the model it was opened with, so a ``model`` sent on a
    later turn of the same runtime session cannot take effect. Rather than
    ignore it silently (and mislabel the turn), the frame reports the model in
    effect and carries a ``notice`` saying the request was not applied.
    """
    event: dict[str, Any] = {"type": "session", "session_id": key, "model": model_in_effect}
    if requested and requested != model_in_effect:
        event["notice"] = (
            f"requested model {requested!r} ignored: this session already runs "
            f"{model_in_effect!r}; a model applies only when a session starts "
            "(use a new session id to change it)"
        )
    return event


@dataclass
class _Entry:
    session: Any
    # The model the lop session was opened with. A session's model is fixed for
    # its lifetime here, so this (not the latest request's ``model``) is what
    # every turn of the session actually runs on and what the caller is told.
    model: str = ""
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SessionPool:
    """Lazily opened lop sessions, one per runtime session id.

    All sessions share ONE set of SDK roots: the SDK allows a single root per
    process, and AgentCore already isolates runtime sessions by microVM, so in
    production a process only ever serves one id. Locally (``python main.py``)
    several ids share the working directory — that is a dev convenience, not
    the isolation boundary.
    """

    def __init__(self, settings: RuntimeSettings) -> None:
        self.settings = settings
        self._stack = contextlib.AsyncExitStack()
        self._entries: dict[str, _Entry] = {}
        self._open_lock = asyncio.Lock()
        home = settings.home
        self.config_dir = home / ".local-operator"
        self.agent_home = home
        self.workdir = home / "work"
        self._base_url: str | None = None
        self._extra_models: tuple[str, ...] = ()

    def configure(self, base_url: str) -> Path:
        """Write lop's config for the bridge at ``base_url`` (call once at boot)."""
        self._base_url = base_url
        return self._write_config()

    def declare_model(self, model: str) -> bool:
        """Declare a per-session model override's budget before lop resolves it.

        Whether the execution role may actually invoke that model is Bedrock's
        (IAM's) decision, surfaced to the caller as the turn's error.
        Returns True when the config was rewritten.
        """
        if self._base_url is None:
            raise RuntimeError("SessionPool.configure() must run before sessions open")
        if model == self.settings.model_id or model in self._extra_models:
            return False
        self._extra_models += (model,)
        self._write_config()
        return True

    def _write_config(self) -> Path:
        if self._base_url is None:
            raise RuntimeError("SessionPool.configure() must run before sessions open")
        return write_lop_config(
            self.config_dir,
            base_url=self._base_url,
            settings=self.settings,
            extra_models=self._extra_models,
        )

    def roots(self) -> Any:
        from local_operator.sdk import SessionRoots

        return SessionRoots(
            config_dir=self.config_dir, agent_home=self.agent_home, cwd=self.workdir
        )

    async def _session(self, key: str, model: str | None) -> _Entry:
        async with self._open_lock:
            entry = self._entries.get(key)
            if entry is not None:
                return entry
            from local_operator.sdk import SessionSpec, open_session

            if model:
                self.declare_model(model)
            self.workdir.mkdir(parents=True, exist_ok=True)
            opened_model = model or self.settings.model_id
            spec = SessionSpec(
                hosting=HOSTING,
                model=opened_model,
                approvals=approval_policy(self.settings.approvals),
                tools=self.settings.tools,
                name=f"agentcore {key[:12]}",
            )
            log.info("opening lop session for runtime session %s model=%s", key, spec.model)
            session = await self._stack.enter_async_context(open_session(spec, roots=self.roots()))
            entry = _Entry(session=session, model=opened_model)
            self._entries[key] = entry
            return entry

    async def run_turn(self, key: str, turn: Turn) -> AsyncIterator[dict[str, Any]]:
        """Run one prompt in the session for ``key``, yielding projected events.

        Turns within one session are serialised: AgentCore may deliver a second
        invocation while the first still streams, and a lop session runs one
        turn at a time.
        """
        from local_operator.headless_print import printable_event
        from local_operator.sdk import events

        entry = await self._session(key, turn.model)
        yield session_event(key, entry.model, turn.model)
        async with entry.lock:
            stream = events(entry.session)  # subscribe BEFORE prompting: no replay
            prompt_task = asyncio.create_task(entry.session.prompt(turn.prompt))
            # ``prompt`` can raise before it emits anything (TurnInFlight while
            # lop's own compaction holds the turn, a disposed session, a
            # capability gate). Then no ``agent_end`` ever arrives, so the loop
            # below would wait forever and the invocation would hang until the
            # caller's read timeout. Closing the stream when the task finishes
            # bounds the loop by the prompt itself; ``close()`` queues its
            # sentinel AFTER any already-queued ``agent_end``, so a normal turn
            # still ends on ``end``. ``await prompt_task`` then re-raises the
            # failure for ``main.py`` to report as an ``error`` event.
            prompt_task.add_done_callback(lambda _task: stream.close())
            try:
                async for event in stream:
                    projected = project_event(printable_event(event))
                    if projected is None:
                        continue
                    yield projected
                    if projected["type"] == "end":
                        break
                await prompt_task
            finally:
                if not prompt_task.done():
                    prompt_task.cancel()
                await stream.aclose()

    async def aclose(self) -> None:
        """Close every open lop session.

        ``main.py`` deliberately never calls this: AgentCore discards the whole
        microVM when a runtime session idles out or the runtime is updated, and
        ``BedrockAgentCoreApp`` exposes no shutdown hook to run it from. Tests
        and local embedders that outlive a pool should call it.
        """
        await self._stack.aclose()
