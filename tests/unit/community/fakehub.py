"""A fake ``/community/v1`` hub and credential store, shared by the community tests.

The hub is an ``httpx.MockTransport`` handler that answers like the contract in
agent-server ``docs/community-api.yaml`` (feat/community-registry): every request
that "left" is recorded with its fully read body, so tests assert on wire bytes,
not on mocks of the client's own logic. The store mirrors the exact
``AuthStore`` surface the client touches (``list_credentials``,
``get_credential``, ``ensure_oauth_fresh_or_raise``), so the owner-pin and
forced-refresh rules run against rows, the Team Bridge client tests' pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from local_operator.community.client import CommunityClient
from local_operator.env import DEFAULT_RADIENT_API_BASE_URL
from local_operator.providers.auth_store import AuthStoreError

COMMUNITY = {
    "id": "local-operator",
    "slug": "local-operator",
    "name": "Local Operator",
    "state": "active",
}
GUIDELINES = {"version": "2026-11-01", "text": "Be kind."}
BOOTSTRAP = {
    "community_id": "local-operator",
    "community": COMMUNITY,
    "guidelines": GUIDELINES,
    "membership": None,
}
JOINED = {
    **BOOTSTRAP,
    "membership": {
        "status": "active",
        "guidelines_version": "2026-11-01",
        "accepted_at": "2026-11-18T10:00:00Z",
    },
}
MESSAGE = {
    "id": "msg_1",
    "conversation_id": "conv_general",
    "seq": 1,
    "author": {"kind": "human", "display_name": "Ada", "public_id": "pub_ada"},
    "text": "hello",
    "created_at": "2026-11-18T10:00:00Z",
}
KEY = "k" * 32


def outcome(state: str = "published", **extra: Any) -> dict[str, Any]:
    return {"idempotency_key": KEY, "state": state, "message_id": "msg_2", **extra}


#: The fake hub's default answers, keyed by (method, path under /community/v1).
ROUTES: dict[tuple[str, str], tuple[int, Any]] = {
    ("GET", "/communities"): (200, {"joined": [], "featured": [COMMUNITY]}),
    ("GET", "/communities/local-operator/bootstrap"): (200, BOOTSTRAP),
    ("POST", "/communities/local-operator/join"): (200, JOINED),
    ("POST", "/communities/local-operator/leave"): (204, None),
    ("GET", "/communities/local-operator/directory/channels"): (
        200,
        {
            "items": [{"id": "conv_general", "name": "general", "visibility": "public"}],
            "next_cursor": None,
        },
    ),
    ("GET", "/communities/local-operator/conversations/conv_general/messages"): (
        200,
        {"items": [MESSAGE], "next_cursor": "c2"},
    ),
    ("POST", "/communities/local-operator/conversations/conv_general/submissions"): (
        200,
        outcome(),
    ),
    ("GET", f"/communities/local-operator/submissions/{KEY}"): (200, outcome("checking")),
    ("POST", f"/communities/local-operator/submissions/{KEY}:cancel"): (200, outcome("cancelled")),
    ("POST", "/communities/local-operator/messages/msg_1:edit"): (200, outcome()),
    ("DELETE", "/communities/local-operator/messages/msg_1"): (204, None),
    ("POST", "/communities/local-operator/messages/msg_1/reactions"): (
        200,
        {"emoji": "👍", "count": 2, "mine": True},
    ),
    ("GET", "/communities/local-operator/events"): (
        200,
        {
            "events": [{"type": "conversation_changed", "conversation_id": "conv_general"}],
            "next": "e2",
        },
    ),
}


@dataclass
class Sent:
    method: str
    path: str
    params: dict[str, str]
    headers: httpx.Headers
    body: bytes
    timeout: dict[str, Any]


class FakeHub:
    """Records every request and answers from ``ROUTES`` unless overridden."""

    PREFIX = "/community/v1"

    def __init__(self) -> None:
        self.sent: list[Sent] = []
        self.override: Callable[[Sent], httpx.Response | None] | None = None

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        body = await request.aread()
        path = request.url.raw_path.decode("ascii").split("?", 1)[0]
        sent = Sent(
            request.method,
            path,
            dict(request.url.params),
            request.headers,
            body,
            dict(request.extensions.get("timeout", {})),
        )
        self.sent.append(sent)
        if self.override is not None:
            answer = self.override(sent)
            if answer is not None:
                return answer
        if not path.startswith(self.PREFIX):
            return httpx.Response(404, json={"message": "not found"})
        status, payload = ROUTES.get(
            (request.method, path[len(self.PREFIX) :]), (404, {"message": "community not found"})
        )
        if payload is None:
            return httpx.Response(status)
        return httpx.Response(status, json=payload)

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


@dataclass
class FakeRow:
    id: int
    data: dict[str, Any] = field(default_factory=dict)
    provider: str = "radient"
    credential_type: str = "oauth"


class FakeStore:
    """``AuthStore``'s three-method surface. ``refresh`` replaces the snapshot a
    FORCED refresh returns (e.g. a re-login onto another account)."""

    def __init__(
        self,
        rows: list[FakeRow] | None = None,
        *,
        refresh: dict[str, Any] | None = None,
        refresh_error: Exception | None = None,
    ) -> None:
        self.rows = (
            list(rows)
            if rows is not None
            else [FakeRow(1, data={"access": "token-1", "account_id": "acc_1"})]
        )
        self.refresh = refresh
        self.refresh_error = refresh_error
        self.forced: list[int] = []

    def list_credentials(self, provider: str) -> list[FakeRow]:
        return [r for r in self.rows if r.provider == provider]

    def get_credential(self, credential_id: int) -> FakeRow | None:
        return next((r for r in self.rows if r.id == credential_id), None)

    async def ensure_oauth_fresh_or_raise(self, credential_id: int, *, force: bool = False):
        row = self.get_credential(credential_id)
        if row is None:
            raise AuthStoreError("gone")
        if force:
            self.forced.append(credential_id)
            if self.refresh_error is not None:
                raise self.refresh_error
            if self.refresh is not None:
                row.data = dict(self.refresh)
        return dict(row.data)


def make_client(
    hub: FakeHub,
    store: FakeStore | None = None,
    *,
    base_url: str = DEFAULT_RADIENT_API_BASE_URL,
    expected_account_id: str | None = None,
) -> CommunityClient:
    return CommunityClient(
        base_url=base_url,
        store=store if store is not None else FakeStore(),
        transport=hub.transport(),
        expected_account_id=expected_account_id,
    )


# Smallest valid images: the client sniffs the magic bytes against the type.
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x01" * 64
