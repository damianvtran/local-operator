"""A fake ``/community`` page backend built from the client's REAL response models.

The page's tests and its screenshot rig drive the real widget inside the real
``OperatorApp`` (so ``local_operator.tcss`` applies) with this in place of
``LiveCommunityBackend``. Rows are ``community.models`` instances parsed from
contract-shaped dicts, so a model change breaks these fixtures too.
"""

from __future__ import annotations

import asyncio
from typing import Any

from local_operator.community import errors as e
from local_operator.community.models import Channel, Community, Message
from local_operator.community.service import PostOutcome
from local_operator.tui.community_backend import Home

COMMUNITY = Community.model_validate(
    {
        "id": "local-operator",
        "slug": "local-operator",
        "name": "Local Operator Community",
        "state": "active",
    }
)
GUIDELINES = (
    "1  Be kind and stay on topic. Disagree with ideas, not people.\n"
    "2  No spam or undisclosed promotion. Say when it's your own work.\n"
    "3  Keep private information private: no personal details, keys, tokens.\n"
    "Not allowed anywhere: sexual content; child sexual abuse or exploitation; illegal "
    "sales; hacking help or malware; scams, fraud and impersonation; harassment, hate "
    "and threats."
)


def channel(
    cid: str, name: str, *, visibility: str = "public", unread: int = 0, locked: bool = False
) -> Channel:
    return Channel.model_validate(
        {
            "id": cid,
            "name": name,
            "visibility": visibility,
            "member": True,
            "unread": unread,
            "locked": locked,
        }
    )


CHANNELS = (
    channel("c_welcome", "welcome-and-rules"),
    channel("c_general", "general", unread=3),
    channel("c_help", "help", unread=12),
    channel("c_feedback", "feedback"),
    channel("c_show", "show-and-tell"),
    channel("c_beta", "beta-testers", visibility="private", unread=2),
)


def message(
    seq: int,
    name: str,
    text: str,
    *,
    kind: str = "human",
    handle: str | None = None,
    at: str = "2026-11-18T10:02:00Z",
    replies: int = 0,
) -> Message:
    return Message.model_validate(
        {
            "id": f"m{seq}",
            "conversation_id": "c_help",
            "seq": seq,
            "author": {"kind": kind, "display_name": name, "agent_handle": handle},
            "created_at": at,
            "text": text,
            "reply_count": replies or None,
        }
    )


HELP = [
    message(1, "Priya Raman", "Does the relay drop for anyone after sleep?"),
    message(2, "Sam Lee", "Yes, every time.", at="2026-11-18T10:05:00Z", replies=2),
    message(
        3,
        "Sam Lee",
        "Logs show the tunnel restarts on wake. Try `lop tunnel doctor`.",
        kind="agent",
        handle="Juno",
        at="2026-11-18T10:06:00Z",
    ),
    message(
        4, "Alex Kim", "Fixed in 0.42.3 — update with `lop-update`.", at="2026-11-18T10:08:00Z"
    ),
]


class FakeBackend:
    """Scriptable: set ``next_outcome`` / ``history_error`` / ``home_value``."""

    def __init__(self, *, joined: bool = True, channels: tuple[Any, ...] = CHANNELS) -> None:
        self.joined = joined
        self.channels = channels
        self.histories: dict[str, list[Any]] = {"c_help": list(HELP), "c_general": list(HELP[:1])}
        self.history_error: dict[str, Exception] = {}
        self.home_error: Exception | None = None
        self.next_outcome: PostOutcome = PostOutcome("posted", submission_id="k", message_id="m9")
        self.cancel_outcome: PostOutcome = PostOutcome("cancelled", submission_id="k")
        self.submit_gate: asyncio.Event | None = None
        self.submitted: list[tuple[str, str]] = []
        self.joins: list[str] = []
        self.left = 0
        self.memberships: list[dict[str, Any]] = [
            {
                "tenant_id": "t_tran",
                "tenant_name": "Tran Robotics",
                "status": "active",
                "is_home": True,
            },
            {"tenant_id": "t_fern", "tenant_name": "Fernhill Data", "status": "active"},
        ]
        self.saved_selection: str | None = None

    async def home(self, community_id: str | None = None) -> Home:
        if self.home_error is not None:
            raise self.home_error
        return Home(
            community=COMMUNITY,
            joined=self.joined,
            guidelines_version="2026-11",
            guidelines_text=GUIDELINES,
            accepted_version="2026-11" if self.joined else None,
            channels=self.channels if self.joined else (),
        )

    async def history(self, community_id: str, channel_id: str) -> list[Any]:
        if channel_id in self.history_error:
            raise self.history_error[channel_id]
        return list(self.histories.get(channel_id, []))

    async def submit(
        self, community_id: str, channel_id: str, text: str, key: str, version: str | None
    ) -> PostOutcome:
        self.submitted.append((channel_id, text))
        if self.submit_gate is not None:
            await self.submit_gate.wait()
        return self.next_outcome

    async def cancel(self, community_id: str, key: str) -> PostOutcome:
        return self.cancel_outcome

    async def join(self, community_id: str, version: str) -> None:
        self.joins.append(version)
        self.joined = True

    async def leave(self, community_id: str) -> None:
        self.left += 1

    async def browse(self, community_id: str, query: str, filter: str) -> list[Any]:
        rows = [c for c in self.channels if query.lower() in c.name]
        if filter == "private":
            rows = [c for c in rows if c.visibility == "private"]
        return rows

    async def orgs(self) -> tuple[list[dict[str, Any]], list[Any]]:
        return list(self.memberships), [COMMUNITY] if self.joined else []

    def last_selection(self) -> str | None:
        return self.saved_selection

    def remember_selection(self, key: str) -> None:
        self.saved_selection = key


UNAVAILABLE = e.CommunityUnavailable()
