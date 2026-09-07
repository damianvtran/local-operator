"""Source-owned work that must outlive a terminal's current presentation.

A session ID alone is insufficient: reconnect/takeover can replace its facade.
The binding token identifies this exact owner-facing incarnation. Widgets do
not belong here; hidden work updates this state and only the current binding
may turn those updates into terminal UI mutations.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from local_operator.session.protocol import SessionProtocol

if TYPE_CHECKING:
    from local_operator.tui.widgets.transcript import NoticeKind


@dataclass
class TurnInteraction:
    provider_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    epoch: int = 0
    operation: int = 0
    open: bool = False
    notified: bool = False
    accrued_cost: float = 0.0
    pending_echoes: list[Any] = field(default_factory=list)
    submitted_draft: SessionDraft | None = None
    completion_deferred: bool = False
    settled_child_ids: set[str] = field(default_factory=set)
    waiting_kind: str | None = None
    approvals_denied_epoch: int | None = None


@dataclass
class LoopInteraction:
    running: bool = False
    cancelled: bool = False
    goal: str = ""
    suppress_completion: bool = False
    worker: Any = None
    operation: int = 0


@dataclass
class ShellInteraction:
    signal: Any = None
    worker: Any = None
    call_id: str = ""
    result: Any = None
    progress: str = ""


@dataclass
class CompactionInteraction:
    active: bool = False
    held_prompt: str = ""
    held_typed: str = ""
    held_images: dict[int, Any] = field(default_factory=dict)
    accepted_message_id: str = ""
    accepted_draft: SessionDraft | None = None


@dataclass
class SessionAccounting:
    total: float = 0.0
    child_costs: dict[str, float] = field(default_factory=dict)
    is_floor: bool = False


@dataclass
class SessionNaming:
    requested: bool = False
    provisional: str = ""
    generation: int = 0
    checked_at: float = 0.0
    last_titled_turn_count: int = 0
    refresh_count: int = 0
    pending_text: str = ""


@dataclass
class SessionDraft:
    text: str = ""
    attachments: dict[int, Any] = field(default_factory=dict)
    selection: Any = None
    shell_mode: bool = False
    focus_id: str = ""
    scroll_anchor_id: str = ""
    scroll_anchor_part: int = 0
    scroll_offset: int = 0
    following_tail: bool = True
    aside: Any = None
    aside_main_text: str | None = None
    aside_main_images: dict[int, Any] = field(default_factory=dict)
    aside_main_shell_mode: bool = False
    approve_all: bool | None = None
    recoveries: list[SessionDraft] = field(default_factory=list)
    notices: list[tuple[str, NoticeKind]] = field(default_factory=list)
    #: Prompt-history recall position and the draft it displaced, so Up in
    #: one session and Down after switching back lands on THAT session's
    #: draft rather than whatever the shared editor stashed last. In-memory
    #: only: a switch is what needs it, not a restart.
    history_index: int | None = None
    history_stash: str = ""
    history_stash_attachments: dict[int, Any] = field(default_factory=dict)


@dataclass
class SessionInteraction:
    session: SessionProtocol | None
    token: str = field(default_factory=lambda: uuid.uuid4().hex)
    turn: TurnInteraction = field(default_factory=TurnInteraction, repr=False)
    loop: LoopInteraction = field(default_factory=LoopInteraction, repr=False)
    shell: ShellInteraction = field(default_factory=ShellInteraction, repr=False)
    compaction: CompactionInteraction = field(default_factory=CompactionInteraction, repr=False)
    accounting: SessionAccounting = field(default_factory=SessionAccounting, repr=False)
    draft: SessionDraft = field(default_factory=SessionDraft, repr=False)
    naming: SessionNaming = field(default_factory=SessionNaming, repr=False)
    active_workers: int = 0
    controller: Any = None
    retired: bool = False
    presentation_revision: int = 0
    restored_goal_notice: str = ""
    aside_open: bool = False
    aside_generation: int = 0
    preparations: int = 0
    # A gate draft can contain a secret. It stays only in this live context,
    # never in the general draft spill or diagnostic repr.
    gate_draft: tuple[tuple[Any, ...], Any] | None = field(default=None, repr=False)
    gate_view_generation: int = 0
    unsubscribe_frontend: Any = field(default=None, repr=False)

    @property
    def unsent(self) -> list[SessionDraft]:
        return self.draft.recoveries

    @property
    def notices(self) -> list[tuple[str, NoticeKind]]:
        return self.draft.notices

    @property
    def retained_for_local_work(self) -> bool:
        """Retention this process OWNS: our coroutines, the user's unsent state.

        Every clause is something only WE hold — an in-flight worker awaiting
        this socket, a shell or loop we started, a held prompt, a gate answer
        the user typed and has not sent. Disposing the source breaks a live
        worker or loses that answer, so nothing may drain these. They are also
        bounded by user action: a turn has to be started before it can be
        abandoned by switching away.
        """
        return bool(
            self.active_workers
            or self.loop.running
            or self.shell.worker is not None
            or self.compaction.held_prompt
            or self.gate_draft is not None
        )

    @property
    def retained_for_auto_work(self) -> bool:
        """Retention driven by the OWNER's turn — a remote fact, and unbounded.

        Kept so an auto-approving viewer keeps noticing a background turn finishing;
        but the viewer is a read-only projection (``no_takeover``), so dropping
        it cannot stop or corrupt the owner's turn. Unbounded is the problem:
        prewarming any busy background agent mints one, and each retained
        viewer costs a deep state copy on EVERY owner delta, so closing the
        sidebar must be allowed to drop these (see ``_release_sidebar_source``).
        """
        state = getattr(self.session, "frontend_state", None)
        return bool(
            self.draft.approve_all
            and (
                getattr(self.session, "is_streaming", False)
                or any(
                    getattr(job, "status", "") == "running" for job in getattr(state, "jobs", ())
                )
            )
        )

    @property
    def must_retain(self) -> bool:
        return self.retained_for_local_work or self.retained_for_auto_work

    def worker_group(self, name: str) -> str:
        return f"{name}:{self.token}"
