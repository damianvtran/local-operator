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
from typing import Any

from local_operator.harness.types import ImageContent
from local_operator.session.protocol import SessionProtocol


@dataclass
class TurnInteraction:
    provider_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    epoch: int = 0
    open: bool = False
    notified: bool = False
    accrued_cost: float = 0.0
    pending_echoes: list[Any] = field(default_factory=list)
    submitted_draft: SessionDraft | None = None


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


@dataclass
class SessionInteraction:
    session: SessionProtocol | None
    token: str = field(default_factory=lambda: uuid.uuid4().hex)
    turn: TurnInteraction = field(default_factory=TurnInteraction)
    loop: LoopInteraction = field(default_factory=LoopInteraction)
    shell: ShellInteraction = field(default_factory=ShellInteraction)
    compaction: CompactionInteraction = field(default_factory=CompactionInteraction)
    accounting: SessionAccounting = field(default_factory=SessionAccounting)
    draft: SessionDraft = field(default_factory=SessionDraft)
    # Accepted-but-unsent input is data, not a widget reference. It remains
    # recoverable even if this conversation's presentation has been evicted.
    unsent: list[tuple[str, list[ImageContent]]] = field(default_factory=list)
    notices: list[tuple[str, str]] = field(default_factory=list)
    active_workers: int = 0
    controller: Any = None
    retired: bool = False
    presentation_revision: int = 0
    preparations: int = 0

    @property
    def must_retain(self) -> bool:
        return bool(
            self.active_workers
            or self.loop.running
            or self.shell.worker is not None
            or self.compaction.held_prompt
            or self.unsent
        )

    def worker_group(self, name: str) -> str:
        return f"{name}:{self.token}"
