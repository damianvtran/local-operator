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
from typing import TYPE_CHECKING, Any, Callable

from local_operator.harness.types import AgentEvent, ModelChangeEvent

if TYPE_CHECKING:
    from local_operator.harness.types import ImageContent
from local_operator.mobile.projection import ProjectionFold
from local_operator.mobile.registrant import SessionHandle
from local_operator.mobile.registrant import image_blocks as _image_blocks
from local_operator.mobile.types import (
    PendingRequest,
    SessionProjection,
    ask_pending_request,
)

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
        auto_approve: bool = False,
    ) -> None:
        self._session = session
        self._loop = loop
        # When the owner's saved default is full-auto (``tool_approval_mode:
        # auto``), the phone must not park a card the TUI would never show —
        # the gate is answered ``True`` inline instead. Stored so a future
        # per-session toggle can flip it without reconstructing the handle.
        self._auto_approve = auto_approve
        self._on_projection: Callable[[], None] | None = None
        self._projection = SessionProjection(
            session_id=session.session_id,
            pid=0,  # stamped by the registrant's record
            kind="daemon",
            # A restored session carries its stored name; a brand-new one has
            # none yet and the naming errand (see prompt()) fills it after the
            # first substantive turn. Left empty rather than a stand-in like
            # "mobile session" so the phone's own fallback (the header shows
            # "untitled", the list shows the cwd) is what the user sees until
            # the real title lands, instead of a placeholder that never moves.
            conversation_name=getattr(session, "conversation_name", "") or "",
            cwd=cwd,
            model_label=_effective_label(session),
            model_selector=_selector(session),
            effort=_current_effort(session),
            effort_ladder=_ladder(session),
        )
        self._fold = ProjectionFold(self._projection)
        # Conversation naming is a TUI-only errand today (OperatorApp owns the
        # naming worker), so a phone-started session used to stay "mobile
        # session" forever — the session list and the header both read the
        # conversation name and had nothing better to show. This latch mirrors
        # OperatorApp._name_requested: the first substantive prompt fires ONE
        # background naming call, alongside the turn it decorates.
        self._name_requested = False
        # Opener of a naming attempt that returned nothing. Isolated naming
        # often 429s on a dead primary BEFORE the turn pins a fallback; the
        # route edge re-fires this opener once a serving model exists.
        self._pending_name_text = ""
        # Strong references to detached background tasks (the naming errand),
        # so the event loop cannot garbage-collect one mid-flight and drop the
        # title silently. Each task removes itself on completion.
        self._background_tasks: set[asyncio.Future[Any]] = set()
        # request_id -> Future the gate/ask call is parked on.
        self._pending_futures: dict[str, asyncio.Future[Any]] = {}
        # request_id -> the AskQuestion.id the harness is waiting on (the
        # answer map's key — see ask_gate).
        self._pending_question_ids: dict[str, str] = {}
        self._install_gates()

    # -- gates -----------------------------------------------------------------

    def _install_gates(self) -> None:
        async def approval_gate(tool_name: str, description: str) -> bool:
            # Full-auto: the owner's saved default is to approve every tier,
            # exactly as the TUI adopts ``tool_approval_mode: auto`` at boot
            # (see OperatorApp._load_approvals_default). Answer inline so the
            # turn never stalls on a card no front end would present.
            if self._auto_approve:
                return True
            request_id = secrets.token_hex(8)
            future: asyncio.Future[bool] = self._loop.create_future()
            self._pending_futures[request_id] = future
            # push_pending, not set_pending: a parallel tool batch can open two
            # approvals concurrently, and each must get its own card. Clearing
            # by request_id (pop_pending) is what keeps them independent — one
            # answered card must not dismiss the sibling that is still waiting.
            self._fold.push_pending(
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
                self._fold.pop_pending(request_id)
                self._notify()

        async def ask_gate(questions: list[Any]) -> dict[str, list[str]] | None:
            if not questions:
                # Nothing to ask: answer NOTHING (the harness's "user escaped"
                # signal) rather than parking a card with no question on it.
                return None
            total = len(questions)
            if total > 1:
                logger.info(
                    "mobile ask gate: %d questions, projecting question-by-question",
                    total,
                )
            # Answer the questions one at a time on the SAME card so a
            # multi-question ask is answerable end to end from the phone rather
            # than the first answer resolving the whole set and dropping the
            # rest (U1). Each question parks its own future; the phone's
            # ask_answer resolves the FRONT one, and we advance to the next.
            answers: dict[str, list[str]] = {}
            for index, question in enumerate(questions):
                request_id = secrets.token_hex(8)
                future: asyncio.Future[dict[str, list[str]] | None] = self._loop.create_future()
                self._pending_futures[request_id] = future
                # The answer map is keyed by the QUESTION'S id, not our request
                # id — AskUserFn's contract answers ``question.id -> choices``.
                self._pending_question_ids[request_id] = getattr(question, "id", "") or ""
                # Built through the shared seam so this projection carries option
                # descriptions (U3), the secret flag (D1/U2, never the value),
                # and the "N of M" position (U1) identically to the TUI path.
                self._fold.push_pending(
                    ask_pending_request(
                        request_id,
                        question,
                        question_index=index,
                        question_total=total,
                    )
                )
                self._notify()
                try:
                    answer = await asyncio.wait_for(future, timeout=PENDING_REQUEST_TIMEOUT_S)
                except TimeoutError:
                    # A timed-out question ends the whole ask: report whatever
                    # earlier questions collected (partial, like the terminal's
                    # Escape) rather than blocking forever on the next one.
                    return answers or None
                finally:
                    self._pending_futures.pop(request_id, None)
                    self._pending_question_ids.pop(request_id, None)
                    self._fold.pop_pending(request_id)
                    self._notify()
                if not answer:
                    # The user answered nothing on this question. On the FIRST
                    # question that is "escaped" — fall back to the model's
                    # recommendation (None). Past it, keep the partial map, the
                    # same rule the terminal picker follows on Escape.
                    return answers or None
                answers.update(answer)
            return answers or None

        # Kept as attributes as well as registered on the session: the handle
        # is the single owner of the gate behaviour, and holding the reference
        # lets tests (and any future direct caller) exercise the exact closure
        # the harness will await, rather than a re-implementation of it.
        self._approval_gate = approval_gate
        self._ask_gate = ask_gate
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
            if isinstance(event, ModelChangeEvent):
                self._retry_naming_after_route_change()
            self._notify()

        unsubscribe = self._session.subscribe(handler)
        try:
            self._fold.fold_history(self._session.history())
        except Exception:  # noqa: BLE001 — history is a convenience, not a gate
            logger.debug("owned session history fold failed", exc_info=True)
        # Seed the live flag ONCE at attach (a phone subscribing mid-turn never
        # saw the AgentStartEvent). After this the fold's own lifecycle events
        # own ``streaming`` — see ``_reconcile_streaming``.
        self._reconcile_streaming()
        return unsubscribe

    async def prompt(self, text: str, images: list[dict[str, str]] | None = None) -> str:
        self._check_loop_thread()
        image_blocks = _image_blocks(images)
        # A rejected prompt must not leave a ghost user row on the phone, so
        # the echo waits until Session.prompt has ACCEPTED the turn (the lock
        # is the honest signal) — see _run_turn_task. Check the cheap guard up
        # front so the common busy case answers immediately without a task.
        if self._session.is_streaming or getattr(self._session, "_compacting", False):
            return "not sent: session is busy — steer instead, or retry in a moment"
        self._maybe_name_conversation(text)
        self._run_turn_task(text, image_blocks)
        return "prompt sent"

    def _maybe_name_conversation(self, text: str) -> None:
        """Name a still-unnamed conversation from its first real prompt.

        The TUI's OperatorApp runs the full naming/re-titling machinery; the
        phone only needs the FIRST-name half, because a mobile session opens
        unnamed and the list/header have nothing to show until it is named.
        Mirrors OperatorApp._maybe_name_conversation: skip low-signal openers
        (a bare "hi" is usually followed by the real ask, and latching on it
        would leave the session named after the greeting), fire at most once,
        and run the call as a background task so the title arrives ALONGSIDE
        the turn rather than after it.
        """
        from local_operator.session import naming

        if self._name_requested or naming.is_low_signal(text):
            return
        if getattr(self._session, "conversation_name", ""):
            # Already named (a restored session, or a prior prompt named it).
            self._name_requested = True
            return
        # Wear the opener on the phone immediately — same stand-in the TUI
        # band shows — so the list/header are not "untitled" for the whole
        # first turn (or forever, if the isolated naming call 429s).
        label = naming.provisional_title(text)
        if label:
            self._fold.set_state(conversation_name=label)
            self._notify()
        self._name_requested = True
        # Hold a strong reference until the task settles: a bare ensure_future
        # is only weakly held by the loop and can be collected before it runs.
        task = asyncio.ensure_future(self._name_conversation_worker(text))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _name_conversation_worker(self, text: str) -> None:
        """Ask the model for a title once, cheaply, off the turn's lock.

        ``session.complete_once`` is the same isolated, single-attempt, cheap
        completion the TUI's naming worker uses — a 429 here is swallowed by
        ``generate_title`` and cannot touch the turn. On success the title is
        stored on the session (which persists it), then the projection is
        refreshed and pushed so the phone's header and list update live.
        """
        from local_operator.session import naming

        try:
            title = await naming.generate_title(text, self._session.complete_once)
        except Exception:  # noqa: BLE001 — naming is decoration; never fail a turn
            logger.debug("mobile conversation naming failed", exc_info=True)
            return
        if not title or getattr(self._session, "conversation_name", ""):
            # No title, or a user/restore named it while we were in flight:
            # allow a later substantive prompt to retry only when still unnamed.
            if not getattr(self._session, "conversation_name", ""):
                self._name_requested = False
                self._pending_name_text = text
            return
        self._pending_name_text = ""
        self._session.set_conversation_name(title, user_set=False)
        self._refresh_state()
        self._notify()

    def _run_turn_task(self, text: str, image_blocks: list["ImageContent"]) -> None:
        """Run the turn as a background task; a rejection surfaces as a notice,
        never as a ghost user row.

        The session emits MessageStartEvent for a user turn only AFTER the
        turn lock is acquired (see Session._run_turn), so that event IS the
        acceptance signal — the fold paints the row from it, and there is no
        optimistic echo to undo. Session.prompt raises RuntimeError when a
        turn started in the gap between the guard above and the lock; catching
        it here keeps the un-awaited-future warning out of the log and gives
        the phone a quiet "not sent" notice instead of a fake sent row.
        """

        async def run() -> None:
            try:
                await self._session.prompt(text, image_blocks)
            except RuntimeError as exc:
                self._projection.streaming = self._session.is_streaming
                self._fold.note_prompt_rejected(str(exc))
                self._notify()
                logger.warning("mobile prompt rejected: %s", exc)

        asyncio.ensure_future(run())

    async def steer(self, text: str, images: list[dict[str, str]] | None = None) -> str:
        self._check_loop_thread()
        # Images ride the steer too — a screenshot sent mid-turn IS the
        # correction, and session.steer already carries them.
        self._session.steer(text, _image_blocks(images))
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
        # ``explicit``: the phone's model switch is a deliberate choice, so a
        # pinned fallback route is withdrawn even when it re-selects the model
        # the fallback displaced — see ``Session.set_model``.
        self._session.set_model(spec, explicit=True)
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

    async def ask_answer(
        self, request_id: str, value: str, question_index: int | None = None
    ) -> str:
        # ``question_index`` is accepted for protocol parity with the TUI handle
        # (U8 guard). An owned session assigns a DISTINCT request_id per question
        # (the gate loops one future per question), so the request_id is already
        # the per-question identity: a stale tap targets an id whose future is
        # gone and is rejected below. No separate index check is needed here.
        del question_index
        # Resolve with the QUESTION id the harness asked under — never our
        # request id, which the harness never saw.
        question_id = self._pending_question_ids.get(request_id, request_id)
        future = self._pending_futures.get(request_id)
        if future is None or future.done():
            # Human, reconciling copy (U4): a stale tap means this question
            # already settled elsewhere — say so rather than the developer-
            # worded "no longer waiting", so the phone user learns their tap
            # lost a race instead of the card silently vanishing.
            raise ValueError("that question was already answered")
        self._loop.call_soon_threadsafe(
            future.set_result, {question_id: [value]} if value else None
        )
        return "answered"

    async def refresh(self) -> None:
        self._refresh_state()
        self._refresh_todos()
        # Command boundary: safe to reconcile from the session flag (no
        # terminal event is mid-flight here, unlike the per-event path).
        self._reconcile_streaming()

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
            model_label=_effective_label(self._session),
            model_selector=_selector(self._session),
            effort=_current_effort(self._session),
            effort_ladder=_ladder(self._session),
            conversation_name=getattr(self._session, "conversation_name", "") or None,
            # NOTE: ``streaming`` is deliberately NOT set here. This runs after
            # every folded event, and the session clears ``is_streaming`` only
            # in the turn's ``finally`` -- AFTER the AgentEndEvent has been
            # emitted and folded. Reading the still-True flag on that terminal
            # event re-stuck the projection to True with no later event to fix
            # it, pinning the phone to "in progress" forever. The fold's own
            # lifecycle events are authoritative; ``_reconcile_streaming``
            # covers attach and command boundaries.
        )

    def _reconcile_streaming(self) -> None:
        """Seed/align ``streaming`` from the session flag at attach and command
        boundaries, delegating the safety rule to ``ProjectionFold`` (which
        ignores the flag once it has folded a turn-terminal event -- see
        ``ProjectionFold.reconcile_streaming``). NEVER called from the
        per-event handler: the fold's lifecycle events own ``streaming``
        there."""
        self._fold.reconcile_streaming(bool(getattr(self._session, "is_streaming", False)))

    def _retry_naming_after_route_change(self) -> None:
        """Re-fire a failed naming attempt once a fallback is actually serving.

        Isolated naming has no fallback chain, so a quota 429 on the primary
        returns None while the turn is still pinning the rescue route. The
        opener is stashed; this spends it the moment the serving model exists.
        """
        pending = self._pending_name_text
        if not pending or getattr(self._session, "conversation_name", ""):
            return
        self._pending_name_text = ""
        self._maybe_name_conversation(pending)

    def _refresh_todos(self) -> None:
        try:
            from local_operator.tools.builtin import TODO_STORE

            self._fold.set_todos(list(TODO_STORE.get(self._session.session_id, [])))
        except Exception:  # noqa: BLE001 — todos are a panel, never a failure
            logger.debug("todo refresh failed", exc_info=True)


def _effective_label(session: Any) -> str:
    """``provider/model`` of the model actually serving requests.

    A display that reads ``session.model_label`` during a provider fallback
    names a model that is not answering — the stale composer chip.
    """
    label = str(getattr(session, "effective_model_label", "") or "")
    return label or str(getattr(session, "model_label", "") or "")


def _selector(session: Any) -> str:
    try:
        spec = session.model
        return f"{spec.provider}/{spec.model_id}"
    except Exception:  # noqa: BLE001
        return ""


def _current_effort(session: Any) -> str:
    try:
        spec = getattr(session, "effective_model", None) or session.model
        return spec.reasoning_effort or ""
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
    resume: str | None = None,
) -> OwnedSessionHandle:
    """Build a session for the phone with the CLI's composition root.

    ``resume`` names an existing session id to reopen: it flows into
    ``args.resume`` exactly as the CLI's ``--resume`` does, so the factory
    reuses that transcript directory and the session replays its history —
    the phone's "open this past conversation" button.
    """
    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.paths import config_dir
    from local_operator.session_factory import create_session

    config_directory = config_dir()
    config_manager = ConfigManager(config_dir=config_directory)
    credential_manager = CredentialManager(config_dir=config_directory)
    agent_registry = AgentRegistry(config_dir=config_directory)

    # The owner's saved tool-approval default. The TUI reads the SAME key at
    # boot (OperatorApp._load_approvals_default) and adopts ``auto`` as
    # "approve every tier"; a phone-started session must honour it too, or a
    # device set to full-auto still pops an approval card the desktop would
    # not. ``yolo`` stays False so the gate is INSTALLED (a per-session toggle
    # can still switch to asking); the handle short-circuits it when auto.
    try:
        approval_mode = (
            str(config_manager.get_config_value("tool_approval_mode", "ask")).strip().lower()
        )
    except Exception:  # noqa: BLE001 — a missing/odd config means "ask", never a crash
        logger.debug("could not read tool_approval_mode; defaulting to ask", exc_info=True)
        approval_mode = "ask"
    auto_approve = approval_mode == "auto"

    args = argparse.Namespace(
        hosting=provider,
        model=model_id,
        agent_name=None,
        agent_id=None,
        yolo=False,
        train=False,
        resume=resume,
    )
    session = await create_session(
        args,
        config_manager,
        credential_manager,
        agent_registry,
        has_ui=False,
        cwd=cwd,
    )
    return OwnedSessionHandle(session, loop, cwd=cwd, auto_approve=auto_approve)
