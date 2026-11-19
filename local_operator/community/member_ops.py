"""The member-operation allowlist, shared by the desktop routes and relay web.

WHY ONE TABLE. RFC §7.2 gives the community two local HTTP doors — the desktop
plane (``/v1/desktop/community/*``, ``server/routes/desktop_community.py``) and
the mobile daemon's relay surface (``/api/community/*``) — and requires the
relay to expose "an allowlist of member operations only (no staff operations
over relay)". Spelling that allowlist once, here, is what keeps the two doors
from drifting: both register exactly :data:`MEMBER_OPERATIONS`, the tests
assert that, and an operation absent from this table has no local route at all.

Each entry names its local path suffix (no ``:action`` segments, so neither
router needs a colon-aware template), its method, the input models it accepts,
and :func:`run`'s branch that calls :class:`~local_operator.community.client.
CommunityClient`. The handlers return plain JSON-able dicts; each door wraps
them in its own envelope and maps :class:`~local_operator.community.errors.
CommunityError` with :func:`refusal`.

INPUT BODIES ARE STRICT (``extra="forbid"``): the renderer and relay web are
ours on both ends, so an unknown field is a client bug that must fail here, not
silently drop a user's choice — the desktop Team Bridge routes' ``_Strict``
rule. Validation failures never echo input: the desktop plane's 422 handler
scrubs them, and the relay answers a fixed ``invalid_request`` body.

IMAGES travel as base64 inside JSON on both doors. The alternative — a
multipart body parsed by Starlette — spools any part over 1 MiB to a temporary
file, and RFC §6.5/§7.2 forbid a submission body touching disk. Base64 of the
largest allowed image (5 MiB) is 6.7 MiB, under the relay's 10 MiB body
ceiling; the client re-streams the decoded bytes upstream from memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from local_operator.community.errors import CommunityError, CommunityInvalidRequest
from local_operator.community.limits import (
    MAX_CURSOR_CHARS,
    MAX_EMOJI_CHARS,
    MAX_IMAGE_BYTES,
    MAX_IMAGES,
    MAX_TEXT_CHARS,
)
from local_operator.community.models import IDEMPOTENCY_KEY_PATTERN, DirectoryFilter

if TYPE_CHECKING:  # the client (httpx) stays off the router import path
    from local_operator.community.client import CommunityClient, CommunityImage

#: Base64 length of the largest allowed image: a field-level bound so an
#: oversized upload fails validation before it is decoded.
MAX_IMAGE_BASE64_CHARS = 4 * math.ceil(MAX_IMAGE_BYTES / 3)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ImageIn(_Strict):
    content_type: Literal["image/png", "image/jpeg", "image/webp"]
    data_base64: str = Field(min_length=1, max_length=MAX_IMAGE_BASE64_CHARS)


class JoinIn(_Strict):
    """Explicit acceptance of ONE guidelines version (contract JoinRequest)."""

    guidelines_version: str = Field(min_length=1, max_length=128)
    accept_guidelines: Literal[True]


class SubmitIn(_Strict):
    text: str = Field(max_length=MAX_TEXT_CHARS)
    images: list[ImageIn] = Field(default_factory=list, max_length=MAX_IMAGES)
    thread_root_id: str | None = Field(default=None, max_length=128)
    guidelines_version: str | None = Field(default=None, min_length=1, max_length=128)
    #: Reuse an earlier attempt's key to replay it; omitted, the door mints one
    #: and returns it (in the outcome or in the refusal body) so a retry or a
    #: status read can name the same attempt. Only for resending the SAME
    #: draft: the hub answers a reused key with the first attempt's stored
    #: outcome whatever the body (#414 R1-M2), so a changed draft omits it.
    idempotency_key: str | None = Field(default=None, pattern=IDEMPOTENCY_KEY_PATTERN)


class EditIn(_Strict):
    text: str = Field(max_length=MAX_TEXT_CHARS)
    images: list[ImageIn] = Field(default_factory=list, max_length=MAX_IMAGES)
    guidelines_version: str | None = Field(default=None, min_length=1, max_length=128)
    idempotency_key: str | None = Field(default=None, pattern=IDEMPOTENCY_KEY_PATTERN)


class ReactIn(_Strict):
    emoji: str = Field(min_length=1, max_length=MAX_EMOJI_CHARS)
    remove: bool = False


class _Query(BaseModel):
    """Query models are NOT strict: the desktop door reads the owner guard
    (``expected_account_id``) from the same query string, and a FastAPI query
    model with ``extra="forbid"`` would refuse it."""


class ChannelsQuery(_Query):
    q: str | None = Field(default=None, max_length=100)
    filter: DirectoryFilter | None = None
    cursor: str | None = Field(default=None, max_length=MAX_CURSOR_CHARS)


class CursorQuery(_Query):
    cursor: str | None = Field(default=None, max_length=MAX_CURSOR_CHARS)


class EventsQuery(_Query):
    after: str | None = Field(default=None, max_length=MAX_CURSOR_CHARS)


@dataclass(frozen=True)
class MemberOperation:
    name: str
    method: Literal["GET", "POST", "DELETE"]
    #: Path suffix under each door's prefix, Starlette/FastAPI template syntax.
    path: str
    body: type[BaseModel] | None = None
    query: type[BaseModel] | None = None


_C = "/communities/{community_id}"

#: THE allowlist. Adding a row here exposes it on BOTH doors; a staff or
#: moderation operation must never be added (RFC §7.2).
MEMBER_OPERATIONS: tuple[MemberOperation, ...] = (
    MemberOperation("list_communities", "GET", "/communities"),
    MemberOperation("bootstrap", "GET", _C + "/bootstrap"),
    MemberOperation("join", "POST", _C + "/join", body=JoinIn),
    MemberOperation("leave", "POST", _C + "/leave"),
    MemberOperation("channels", "GET", _C + "/channels", query=ChannelsQuery),
    MemberOperation(
        "history", "GET", _C + "/conversations/{conversation_id}/messages", query=CursorQuery
    ),
    MemberOperation(
        "submit", "POST", _C + "/conversations/{conversation_id}/messages", body=SubmitIn
    ),
    MemberOperation("submission_status", "GET", _C + "/submissions/{idempotency_key}"),
    MemberOperation("submission_cancel", "POST", _C + "/submissions/{idempotency_key}/cancel"),
    MemberOperation("edit", "POST", _C + "/messages/{message_id}/edit", body=EditIn),
    MemberOperation("delete", "DELETE", _C + "/messages/{message_id}"),
    MemberOperation("react", "POST", _C + "/messages/{message_id}/reactions", body=ReactIn),
    MemberOperation("events", "GET", _C + "/events", query=EventsQuery),
)

OPERATIONS_BY_NAME: dict[str, MemberOperation] = {op.name: op for op in MEMBER_OPERATIONS}


def _images(items: list[ImageIn]) -> list["CommunityImage"]:
    from local_operator.community.client import CommunityImage

    return [CommunityImage.from_base64(i.content_type, i.data_base64) for i in items]


def _dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


async def run(
    name: str,
    client: "CommunityClient",
    path: dict[str, str],
    *,
    query: BaseModel | None = None,
    body: BaseModel | None = None,
) -> tuple[int, dict[str, Any] | None]:
    """Run one allowlisted operation: ``(status, json)`` or a ``CommunityError``.

    A submit/edit refusal carries the attempt's ``idempotency_key`` in its
    ``details`` (see :class:`SubmitIn`), so a surface that timed out can read
    the outcome rather than re-sending blind.
    """

    cid = path.get("community_id", "")
    if name == "list_communities":
        return 200, _dump(await client.list_communities())
    if name == "bootstrap":
        return 200, _dump(await client.bootstrap(cid))
    if name == "join":
        if not isinstance(body, JoinIn):
            raise CommunityInvalidRequest()
        return 200, _dump(
            await client.join(
                cid,
                guidelines_version=body.guidelines_version,
                accept_guidelines=body.accept_guidelines,
            )
        )
    if name == "leave":
        return 200, {"left": True, "already_left": await client.leave(cid)}
    if name == "channels":
        q = query if isinstance(query, ChannelsQuery) else ChannelsQuery()
        return 200, _dump(await client.channels(cid, q=q.q, filter=q.filter, cursor=q.cursor))
    if name == "history":
        c = query if isinstance(query, CursorQuery) else CursorQuery()
        return 200, _dump(await client.history(cid, path["conversation_id"], cursor=c.cursor))
    if name in ("submit", "edit"):
        if not isinstance(body, (SubmitIn, EditIn)):
            raise CommunityInvalidRequest()
        from local_operator.community.client import new_idempotency_key

        key = body.idempotency_key or new_idempotency_key()
        try:
            images = _images(body.images)
            if isinstance(body, SubmitIn):
                outcome = await client.submit(
                    cid,
                    path["conversation_id"],
                    text=body.text,
                    images=images,
                    thread_root_id=body.thread_root_id,
                    guidelines_version=body.guidelines_version,
                    idempotency_key=key,
                )
            else:
                outcome = await client.edit(
                    cid,
                    path["message_id"],
                    text=body.text,
                    images=images,
                    guidelines_version=body.guidelines_version,
                    idempotency_key=key,
                )
        except CommunityError as refusal:
            refusal.attempt_key = key
            raise
        return 200, _dump(outcome)
    if name == "submission_status":
        return 200, _dump(await client.submission(cid, path["idempotency_key"]))
    if name == "submission_cancel":
        return 200, _dump(await client.cancel_submission(cid, path["idempotency_key"]))
    if name == "delete":
        await client.delete_message(cid, path["message_id"])
        return 200, {"deleted": True}
    if name == "react":
        if not isinstance(body, ReactIn):
            raise CommunityInvalidRequest()
        return 200, _dump(
            await client.react(cid, path["message_id"], emoji=body.emoji, remove=body.remove)
        )
    if name == "events":
        e = query if isinstance(query, EventsQuery) else EventsQuery()
        return 200, _dump(await client.events(cid, after=e.after))
    raise KeyError(name)  # unreachable: both doors register only MEMBER_OPERATIONS


def refusal(error: CommunityError) -> tuple[int, dict[str, Any]]:
    """The local status and body for one refusal (codes and facts only)."""

    body = error.wire()
    if error.attempt_key is not None:
        body["idempotency_key"] = error.attempt_key
    return error.http_status, body


__all__ = [
    "MAX_IMAGE_BASE64_CHARS",
    "MEMBER_OPERATIONS",
    "OPERATIONS_BY_NAME",
    "ChannelsQuery",
    "CursorQuery",
    "EditIn",
    "EventsQuery",
    "ImageIn",
    "JoinIn",
    "MemberOperation",
    "ReactIn",
    "SubmitIn",
    "refusal",
    "run",
]
