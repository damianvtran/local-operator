"""The ``/community`` page's I/O: the community client plus the org list, behind one seam.

WHY A SEAM. The page (``widgets/community_view.py``) paints and routes keys;
every network call lives here so the Textual tests drive the real widget with a
fake backend (``tests/unit/tui/test_community_view.py``) and the production
path is one small adapter over the shared pieces — ``community.client``,
``community.service`` (the sender-only outcome mapping the CLI shares) and the
person's ``GET /me/memberships`` read the hub-org gate already uses
(``hub_org.eligibility.person_client``).

ISOLATION (RFC §7.2). Nothing here touches the private Team Bridge client,
receiver, store or cursors: the private orgs come from the account's membership
list (the same read the desktop switcher uses), and selecting one only records
the selection. The bridge's existing private-org behaviour is unchanged.

THE LAST SELECTION is a small JSON file under ``<config>/community/`` holding
one org key — no content, no tokens. It feeds Amendment A's third rule
("else the last selection"); a missing or corrupt file means "none".
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from local_operator.community import service

logger = logging.getLogger(__name__)

SELECTION_FILE = "org_selection.json"


@dataclass(frozen=True)
class Home:
    """What the page needs to paint its first frame."""

    community: Any
    joined: bool
    guidelines_version: str
    guidelines_text: str
    accepted_version: str | None
    channels: tuple[Any, ...]


class LiveCommunityBackend:
    """The production adapter. One client per call (the client is cheap)."""

    def __init__(self, config_dir: Path | str, *, client_factory: Any = None) -> None:
        self._config_dir = Path(config_dir)
        self._factory = client_factory

    def _client(self) -> Any:
        return service.make_client(self._config_dir, self._factory)

    async def home(self, community_id: str | None = None) -> Home | None:
        client = self._client()
        community, joined = await service.any_community(client, community_id)
        if community is None:
            return None
        boot = await client.bootstrap(community.id)
        member = boot.membership is not None
        channels = (
            await service.list_channels(client, community.id, filter="joined") if member else []
        )
        return Home(
            community=community,
            joined=member,
            guidelines_version=boot.guidelines.version,
            guidelines_text=boot.guidelines.text,
            accepted_version=boot.membership.guidelines_version if boot.membership else None,
            channels=tuple(channels),
        )

    async def history(self, community_id: str, channel_id: str) -> list[Any]:
        page = await self._client().history(community_id, channel_id)
        return sorted(page.items, key=lambda m: m.seq)

    async def submit(
        self, community_id: str, channel_id: str, text: str, key: str, version: str | None
    ) -> service.PostOutcome:
        try:
            result = await self._client().submit(
                community_id,
                channel_id,
                text=text,
                idempotency_key=key,
                guidelines_version=version,
            )
        except Exception as exc:  # noqa: BLE001 - every failure is one outcome
            return service.outcome_from_error(exc, submission_id=key)
        return service.outcome_from_submission(result)

    async def cancel(self, community_id: str, key: str) -> service.PostOutcome:
        try:
            result = await self._client().cancel_submission(community_id, key)
        except Exception as exc:  # noqa: BLE001
            return service.outcome_from_error(exc, submission_id=key)
        outcome = service.outcome_from_submission(result)
        if outcome.kind == "posted":
            # Cancel is best effort: the post went through first (T5).
            return service.PostOutcome("too_late", submission_id=key, message_id=outcome.message_id)
        return outcome

    async def join(self, community_id: str, version: str) -> None:
        await self._client().join(community_id, guidelines_version=version, accept_guidelines=True)

    async def leave(self, community_id: str) -> None:
        await self._client().leave(community_id)

    async def browse(self, community_id: str, query: str, filter: str) -> list[Any]:
        directory_filter = (
            None if filter == "all" else ("public" if filter == "not_joined" else filter)
        )
        rows = await service.list_channels(
            self._client(), community_id, filter=directory_filter, q=query or None
        )
        if filter == "not_joined":
            rows = [row for row in rows if not row.member]
        return rows

    async def orgs(self) -> tuple[list[dict[str, Any]], list[Any]]:
        """``(memberships, joined communities)``; either may fail independently."""

        memberships: list[dict[str, Any]] = []
        communities: list[Any] = []
        try:
            memberships = await asyncio.to_thread(self._memberships)
        except Exception:  # noqa: BLE001 - the communities group still paints
            logger.debug("community page: membership read failed", exc_info=True)
        try:
            listing = await self._client().list_communities()
            communities = list(listing.joined)
        except Exception:  # noqa: BLE001
            logger.debug("community page: community list failed", exc_info=True)
        return memberships, communities

    def _memberships(self) -> list[dict[str, Any]]:
        from local_operator.hub_org.eligibility import person_client

        person = person_client(self._config_dir)
        if person is None:
            return []
        rows = person[0].list_memberships()
        return [row for row in rows if isinstance(row, dict)]

    # -- last selection --------------------------------------------------------

    def _selection_path(self) -> Path:
        return self._config_dir / "community" / SELECTION_FILE

    def last_selection(self) -> str | None:
        try:
            data = json.loads(self._selection_path().read_text())
        except (OSError, ValueError):
            return None
        key = data.get("key") if isinstance(data, dict) else None
        return key if isinstance(key, str) and 0 < len(key) <= 128 else None

    def remember_selection(self, key: str) -> None:
        path = self._selection_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"key": key}))
            os.replace(tmp, path)
        except OSError:
            logger.debug("community page: selection not saved", exc_info=True)


__all__ = ["Home", "LiveCommunityBackend", "SELECTION_FILE"]
