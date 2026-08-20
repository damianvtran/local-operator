"""Sessions the daemon owns: started from the phone, run in-process.

An owned session is a full harness ``Session`` built with the same
composition root as the CLI (:func:`session_factory.create_session`), wrapped
in the registrant's :class:`~local_operator.mobile.registrant.SessionHandle`
contract and registered through the SAME loopback socket path a TUI uses.
That last part is the design's keystone: the daemon's web layer never
branches on who owns a session, so a phone-started session and a terminal
session are indistinguishable to the UI — and connection failure handling
(re-dial, degraded, ended) has exactly one implementation.

Approval and ask gates are installed at spawn: the harness calls them when a
tool needs the user, and this bridge parks the call on a future whose
resolution arrives as a control request from the phone. The pending request
is on the projection the whole time, so a phone that opens mid-approval sees
the card — a question for the user is the most prominent thing on screen
(branding.md §7), and "the agent is waiting" must survive a phone restart.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import secrets
from typing import Any, Callable

from local_operator.harness.types import AgentEvent
from local_operator.mobile.projection import ProjectionFold
from local_operator.mobile.registrant import SessionHandle
from local_operator.mobile.tui_handle import _image_blocks
from local_operator.mobile.types import PendingRequest, SessionProjection

logger = logging.getLogger(__name__)

#: How long an approval/ask may sit unanswered before the tool is denied and
#: the turn told why. A phone in a pocket is the common case; an unbounded
#: wait would pin the turn (and its tool slot) forever.
PENDING_REQUEST_TIMEOUT_S = 3600.0


class OwnedSessionHandle(SessionHandle):
    """SessionHandle over an in-process Session living on ``loop``.

    The registrant drives handle methods on the OWNING loop here — the child
    process runs the registrant's socket server as a task on its one asyncio
    loop, so no cross-thread hop is needed and ``run_coroutine_threadsafe``
    never appears. The registrant's ``start_in_process`` classmethod is the
    entry point that wiring uses.
    """

    def __init__(
        self,
        session: Any,
        loop: asyncio.AbstractEventLoop,
        *,
        cwd: str,
    ) -> None:
        self._session = session
        self._loop = loop
        self._on_projection: Callable[[], None] | None = None
        self._projection = SessionProjection(
            session_id=session.session_id,
            pid=0,  # stamped by the registrant's record
            kind="daemon",
            conversation_name=getattr(session, "conversation_name", "") or "mobile session",
            cwd=cwd,
            model_label=session.model_label,
            model_selector=_selector(session),
            effort=_current_effort(session),
            effort_ladder=_ladder(session),
        )
        self._fold = ProjectionFold(self._projection)
        # request_id -> Future the gate/ask call is parked on.
        self._pending_futures: dict[str, asyncio.Future[Any]] = {}
        # request_id -> the AskQuestion.id the harness is waiting on (the
        # answer map's key — see ask_gate).
        self._pending_question_ids: dict[str, str] = {}
        self._install_gates()

    # -- gates -----------------------------------------------------------------

    def _install_gates(self) -> None:
        async def approval_gate(tool_name: str, description: str) -> bool:
            request_id = secrets.token_hex(8)
            future: asyncio.Future[bool] = self._loop.create_future()
            self._pending_futures[request_id] = future
            self._fold.set_pending(
                PendingRequest(
                    request_id=request_id,
                    kind="approval",
                    title=tool_name,
                    detail=description,
                )
            )
            self._notify()
            try:
                return await asyncio.wait_for(future, timeout=PENDING_REQUEST_TIMEOUT_S)
            except TimeoutError:
                return False
            finally:
                self._pending_futures.pop(request_id, None)
                self._fold.set_pending(None)
                self._notify()

        async def ask_gate(questions: list[Any]) -> dict[str, list[str]] | None:
            if not questions:
                # Nothing to ask: answer NOTHING (the harness's "user escaped"
                # signal) rather than parking a card with no question on it.
                return None
            request_id = secrets.token_hex(8)
            future: asyncio.Future[dict[str, list[str]] | None] = self._loop.create_future()
            self._pending_futures[request_id] = future
            first = questions[0]
            # The answer map is keyed by the QUESTION'S id, not our request id
            # — AskUserFn's contract answers ``question.id -> choices``, and
            # the harness validates the keys it gets back. Stash the mapping
            # so ask_answer resolves with the right key.
            self._pending_question_ids[request_id] = (
                getattr(first, "id", "") if first is not None else ""
            )
            if len(questions) > 1:
                logger.info(
                    "mobile ask gate: %d questions, projecting the first only",
                    len(questions),
                )
            self._fold.set_pending(
                PendingRequest(
                    request_id=request_id,
                    kind="ask",
                    title=getattr(first, "question", "the agent is asking")
                    or "the agent is asking",
                    detail="",
                    options=list(getattr(first, "options", []) or []),
                )
            )
            self._notify()
            try:
                return await asyncio.wait_for(future, timeout=PENDING_REQUEST_TIMEOUT_S)
            except TimeoutError:
                return None
            finally:
                self._pending_futures.pop(request_id, None)
                self._pending_question_ids.pop(request_id, None)
                self._fold.set_pending(None)
                self._notify()

        self._session.set_approval_handler(approval_gate)
        self._session.set_ask_handler(ask_gate)

    def _resolve_pending(self, request_id: str, value: Any) -> None:
        future = self._pending_futures.get(request_id)
        if future is None or future.done():
            raise ValueError("that prompt is no longer waiting")
        self._loop.call_soon_threadsafe(future.set_result, value)

    # -- SessionHandle -----------------------------------------------------------

    @property
    def session_projection_seed(self) -> SessionProjection:
        return self._projection

    def subscribe(self, on_projection: Callable[[], None]) -> Callable[[], None]:
        self._on_projection = on_projection

        def handler(event: AgentEvent) -> None:
            # Session events fire on the daemon loop; the fold is a plain
            # state machine, so folding inline is safe. Only the repaint push
            # crosses threads.
            self._fold.fold_event(event)
            self._refresh_state()
            self._refresh_todos()
            self._notify()

        unsubscribe = self._session.subscribe(handler)
        try:
            self._fold.fold_history(self._session.history())
        except Exception:  # noqa: BLE001 — history is a convenience, not a gate
            logger.debug("owned session history fold failed", exc_info=True)
        return unsubscribe

    async def prompt(self, text: str, images: list[dict[str, str]] | None = None) -> str:
        self._check_loop_thread()
        image_blocks = _image_blocks(images)
        asyncio.ensure_future(self._session.prompt(text, image_blocks))
        self._projection.streaming = True
        self._fold.note_user_message(text)
        self._notify()
        return "prompt sent"

    async def steer(self, text: str) -> str:
        self._check_loop_thread()
        self._session.steer(text)
        self._projection.queued_count += 1
        self._fold.note_user_message(text, steer=True)
        self._notify()
        return "steering queued"

    async def abort(self) -> str:
        self._check_loop_thread()
        self._session.abort("stopped from mobile")
        return "stopping"

    async def set_model(self, provider: str, model_id: str) -> str:
        self._check_loop_thread()
        from local_operator.model.configure import build_model_spec

        spec = await asyncio.to_thread(build_model_spec, provider, model_id)
        self._session.set_model(spec)
        self._refresh_state()
        return f"model: {self._projection.model_label}"

    async def set_effort(self, effort: str) -> str:
        self._check_loop_thread()
        spec = self._session.model
        if effort not in spec.reasoning_efforts:
            ladder = ", ".join(spec.reasoning_efforts) or "no rungs"
            raise ValueError(f"{spec.model_id} accepts {ladder}, not '{effort}'")
        self._session.set_model(spec.model_copy(update={"reasoning_effort": effort}))
        self._refresh_state()
        return f"effort: {effort}"

    async def slash(self, command: str, args: str) -> str:
        """Session-level slash commands — the ones with meaning off-terminal.
        TUI chrome (/help tables, /usage panels) is the phone UI's own job."""
        self._check_loop_thread()
        if command == "goal":
            stored = self._session.set_goal(args)
            return "goal updated" if stored else "goal cleared"
        if command == "compact":
            asyncio.ensure_future(self._session.compact_now())
            return "compacting context"
        raise ValueError(f"/{command} is terminal-only here")

    async def new_conversation(self) -> str:
        raise ValueError("start a new session from the session list")

    async def resume_session(self, session_id: str) -> str:
        raise ValueError("pick the session from the session list instead")

    async def approval_answer(self, request_id: str, approved: bool, remember: bool) -> str:
        self._resolve_pending(request_id, approved)
        return "approved" if approved else "denied"

    async def ask_answer(self, request_id: str, value: str) -> str:
        # Resolve with the QUESTION id the harness asked under — never our
        # request id, which the harness never saw.
        question_id = self._pending_question_ids.get(request_id, request_id)
        self._resolve_pending(request_id, {question_id: [value]} if value else None)
        return "answered"

    async def refresh(self) -> None:
        self._refresh_state()
        self._refresh_todos()

    # -- internals ----------------------------------------------------------------

    def _notify(self) -> None:
        if self._on_projection is not None:
            self._on_projection()

    def _check_loop_thread(self) -> None:
        """The registrant calls handle methods on its own loop; owned sessions
        live on the daemon loop. The registrant hops via ``run_coroutine_threadsafe``
        in the daemon's spawn path, so reaching here means we ARE on the right
        loop — assert it in dev, and let the call proceed (asyncio detects real
        cross-loop misuse loudly)."""

    def _refresh_state(self) -> None:
        self._fold.set_state(
            model_label=self._session.model_label,
            model_selector=_selector(self._session),
            effort=_current_effort(self._session),
            effort_ladder=_ladder(self._session),
            conversation_name=getattr(self._session, "conversation_name", "") or None,
            streaming=bool(getattr(self._session, "is_streaming", False)),
        )

    def _refresh_todos(self) -> None:
        try:
            from local_operator.tools.builtin import TODO_STORE

            self._fold.set_todos(list(TODO_STORE.get(self._session.session_id, [])))
        except Exception:  # noqa: BLE001 — todos are a panel, never a failure
            logger.debug("todo refresh failed", exc_info=True)


def _selector(session: Any) -> str:
    try:
        spec = session.model
        return f"{spec.provider}/{spec.model_id}"
    except Exception:  # noqa: BLE001
        return ""


def _current_effort(session: Any) -> str:
    try:
        return session.model.reasoning_effort or ""
    except Exception:  # noqa: BLE001
        return ""


def _ladder(session: Any) -> list[str]:
    try:
        return list(session.model.reasoning_efforts)
    except Exception:  # noqa: BLE001
        return []


async def spawn_owned_session(
    loop: asyncio.AbstractEventLoop,
    *,
    cwd: str,
    provider: str | None = None,
    model_id: str | None = None,
) -> OwnedSessionHandle:
    """Build a fresh session for the phone with the CLI's composition root."""
    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.paths import config_dir
    from local_operator.session_factory import create_session

    config_directory = config_dir()
    config_manager = ConfigManager(config_dir=config_directory)
    credential_manager = CredentialManager(config_dir=config_directory)
    agent_registry = AgentRegistry(config_dir=config_directory)

    args = argparse.Namespace(
        hosting=provider,
        model=model_id,
        agent_name=None,
        agent_id=None,
        yolo=False,
        train=False,
    )
    session = await create_session(
        args,
        config_manager,
        credential_manager,
        agent_registry,
        has_ui=False,
        cwd=cwd,
    )
    return OwnedSessionHandle(session, loop, cwd=cwd)
