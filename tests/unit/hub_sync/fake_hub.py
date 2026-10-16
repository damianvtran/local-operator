"""A hub fake built from the SERVER'S OWN DECODERS (round-2 remediation).

Why a second hub fake exists: the permissive ones accepted request shapes the
real server refuses (the doubled envelope on a commit, above all), so a suite
against them could pass while the wire was broken (round-2 R11/QN1 origin). This
fake encodes the server's decode rules -- read from agent-server@bdd03ab,
internal/controllers/generalization_preview.go, internal/requests/skills.go,
internal/responses/skill_publish.go and docs/HUB-REVISIONS.md -- so a client
that sends something else fails HERE:

* preview takes ``{preview_version, document[, base_document]}`` and nothing else;
* a commit reads the BARE document; anyone re-sending the envelope gets the
  server's own 422 ``preview_version is not a recognised field``;
* the pin travels in ``X-Radient-Preview-Token``, expires (15 minutes), and a
  second commit with the SAME token is an idempotent replay, not a new write;
* ``If-Match`` must equal the row's CURRENT opaque ETag, quotes verbatim; a
  mismatch is 412 ``remote-revision-changed`` WITHOUT disclosing the current
  document;
* an org/preview/publish request carries ``Authorization``; a PUBLIC skill GET
  is served unauthenticated, and an org row addressed publicly is 404 with no
  tenant hint.

It is deliberately small: one agent row. Extend it beside the behaviour it
needs to prove, never with a permissive branch.
"""

from __future__ import annotations

import json
import time
from typing import Any, Mapping

import httpx

REVISION_COUNTER = [0]

#: The server's CLOSED top-level field sets per family, read from
#: agent-server@bdd03ab (internal/requests/instruction_set.go
#: ``instructionSetAllowedFields``, internal/requests/hub_team.go
#: ``teamDocumentAllowedFields``, internal/requests/skills.go
#: ``skillDocumentAllowedFields``). The real decoders stream the top-level keys
#: and refuse the first one outside the set BY NAME (``firstKeyOutside`` +
#: ``DisallowUnknownFields``); this fake now says the same thing, so a client
#: that sends a field the server drops-or-refuses fails HERE (round-3 R19).
CLOSED_DOCUMENT_FIELDS: dict[str, frozenset[str]] = {
    "agents": frozenset(
        {
            "document_type",
            "document_version",
            "name",
            "description",
            "instructions",
            "kind",
            "when_to_use",
            "tools",
            "effort",
            "delegate",
            "version",
            "categories",
            "tags",
            "model_suggestion",
        }
    ),
    "teams": frozenset(
        {
            "name",
            "description",
            "manager",
            "members",
            "instructions",
            "project",
            "version",
            "model_suggestion",
        }
    ),
    "skills": frozenset({"document_version", "name", "description", "files", "version"}),
}


def _closed_fields_for(path: str) -> frozenset[str]:
    """The family whose decoder answers this publish path."""

    if "/agents" in path:
        return CLOSED_DOCUMENT_FIELDS["agents"]
    if "/teams" in path:
        return CLOSED_DOCUMENT_FIELDS["teams"]
    return CLOSED_DOCUMENT_FIELDS["skills"]


def _unknown_key(body: Mapping[str, Any], allowed: frozenset[str]) -> str | None:
    for key in body:
        if key not in allowed:
            return str(key)
    return None


def _etag(n: int) -> str:
    return f'"revision-v1-{n:064d}"'


class FakeHub:
    """One agent row, served over an ``httpx.MockTransport`` handler."""

    def __init__(self, *, document: Mapping[str, Any] | None = None) -> None:
        self.document: dict[str, Any] = dict(
            document or {"instructions": "# v1\n", "description": "d"}
        )
        self.revision: str = _etag(1)
        self.preview_calls: list[httpx.Request] = []
        self.commit_calls: list[httpx.Request] = []
        self.pins: dict[str, dict[str, Any]] = {}
        self.receipts: dict[str, dict[str, Any]] = {}
        self.accepts_anonymous_skill_get = True
        #: The real pins expire (15 minutes); a test can shrink this to drive
        #: the expiry branch (a commit with a dead pin is refused, never served).
        self.pin_ttl_seconds = 900.0
        self.skill_public: dict[str, Any] = {
            "document_version": 1,
            "name": "s",
            "description": "d",
            "files": [
                {
                    "path": "SKILL.md",
                    "content": "---\nname: s\ndescription: d\n---\n\nbody\n",
                }
            ],
            "revision": '"skill-rev-1"',
            "sync_contract": {"revision_cas": 1, "skills": 1},
            "id": "sk-1",
            "visibility": "public",
        }
        self.org_via_public_404 = True

    # -- transport plumbing --------------------------------------------------

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and "/skills/" in path:
            return self._skill_get(request)
        if path.endswith("/publish/preview"):
            return self._preview(request)
        if path.endswith("/publish"):
            return self._commit(request)
        return httpx.Response(404, json={"error": {"code": "route-not-found"}})

    def dispatcher(self) -> Any:
        return _MockDispatcher(self)

    def install(self, monkeypatch: Any) -> None:
        """Route the runner's real transport through this fake."""

        import local_operator.hub_sync.outbox as outbox_mod

        hub = self
        monkeypatch.setattr(outbox_mod, "AsyncDispatcher", lambda *a, **kw: hub.dispatcher())

    # -- routes --------------------------------------------------------------

    def _skill_get(self, request: httpx.Request) -> httpx.Response:
        if self.accepts_anonymous_skill_get and not request.headers.get("Authorization"):
            return httpx.Response(200, json={"result": dict(self.skill_public)})
        if request.headers.get("Authorization"):
            # Signed-in reads answer the same public row; an org row addressed
            # through the PUBLIC path answers 404 with NO tenant hint.
            if self.org_via_public_404 and self.skill_public.get("visibility") == "org":
                return httpx.Response(404, json={"error": {"code": "skill_not_found"}})
            return httpx.Response(200, json={"result": dict(self.skill_public)})
        return httpx.Response(401, json={"error": {"code": "unauthorized"}})

    def _preview(self, request: httpx.Request) -> httpx.Response:
        if not request.headers.get("Authorization"):
            return httpx.Response(401, json={"error": {"code": "unauthorized"}})
        body = json.loads(request.content)
        if set(body) - {"preview_version", "document", "base_document"}:
            return httpx.Response(
                422, json={"error": {"code": "invalid_preview_request", "field": "envelope"}}
            )
        if body.get("preview_version") != 1 or not isinstance(body.get("document"), Mapping):
            return httpx.Response(
                422, json={"error": {"code": "invalid_preview_request", "field": "preview_version"}}
            )
        unknown = _unknown_key(body["document"], _closed_fields_for(request.url.path))
        if unknown is not None:
            return httpx.Response(
                422,
                json={
                    "error": {
                        "code": "invalid_skill_document",
                        "field": unknown,
                        "rule": "is not a recognised field",
                    }
                },
            )
        expected = request.headers.get("If-Match", "")
        if expected != self.revision:
            # 412 without disclosing the current document (revision only).
            return httpx.Response(412, json={"error": {"code": "remote-revision-changed"}})
        REVISION_COUNTER[0] += 1
        token = f"pin-{REVISION_COUNTER[0]}"
        self.pins[token] = {
            "revision": expected,
            "document": dict(body["document"]),
            "expires_at_monotonic": time.monotonic() + self.pin_ttl_seconds,
        }
        self.preview_calls.append(request)
        return httpx.Response(
            200,
            json={
                "msg": "ok",
                "result": {
                    "revision": expected,
                    "sync_contract": {"revision_cas": 1},
                    "status": "ready",
                    "document": dict(body["document"]),
                    "unresolved": [],
                    "pin": {
                        "token": token,
                        "expires_at": "2999-01-01T00:00:00Z",
                        "commit_header": "X-Radient-Preview-Token",
                        "accept_header": "X-Radient-Accept-Unresolved",
                        "accept_ids": [],
                    },
                },
            },
        )

    def _commit(self, request: httpx.Request) -> httpx.Response:
        if not request.headers.get("Authorization"):
            return httpx.Response(401, json={"error": {"code": "unauthorized"}})
        body = json.loads(request.content)
        # THE CLOSED DECODERS (server source, internal/requests/skills.go +
        # internal/controllers/hub_skill.go): the commit reads the bare document;
        # the envelope is refused by name.
        if "preview_version" in body or "document" in body:
            return httpx.Response(
                422,
                json={"error": {"code": "invalid_skill_document", "field": "preview_version"}},
            )
        unknown = _unknown_key(body, _closed_fields_for(request.url.path))
        if unknown is not None:
            return httpx.Response(
                422,
                json={
                    "error": {
                        "code": "invalid_skill_document",
                        "field": unknown,
                        "rule": "is not a recognised field",
                    }
                },
            )
        token = request.headers.get("X-Radient-Preview-Token", "")
        pin = self.pins.get(token)
        if pin is None:
            return httpx.Response(422, json={"error": {"code": "invalid_preview_request"}})
        if pin.get("expires_at_monotonic") is not None and (
            time.monotonic() > float(pin["expires_at_monotonic"])
        ):
            return httpx.Response(
                422, json={"error": {"code": "invalid_preview_request", "rule": "preview expired"}}
            )
        if token in self.receipts:
            # Idempotent replay: the SAME receipt, even if the row moved since.
            return httpx.Response(200, json={"msg": "ok", "result": dict(self.receipts[token])})
        expected = request.headers.get("If-Match", "")
        if expected != pin["revision"] or expected != self.revision:
            return httpx.Response(412, json={"error": {"code": "remote-revision-changed"}})
        if body != pin["document"]:
            return httpx.Response(422, json={"error": {"code": "preview_mismatch"}})
        REVISION_COUNTER[0] += 1
        self.revision = _etag(REVISION_COUNTER[0])
        self.document = dict(body)
        self.commit_calls.append(request)
        result = {
            "revision": self.revision,
            "content_revision": REVISION_COUNTER[0],
            "accepted_document": dict(body),
            "operation_id": f"op-{REVISION_COUNTER[0]}",
        }
        self.receipts[token] = dict(result)
        return httpx.Response(200, json={"msg": "ok", "result": dict(result)})

    # -- assertions-ish helpers ---------------------------------------------

    def bare_commit_bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.commit_calls]


class _MockDispatcher:
    """An ``AsyncDispatcher``-shaped object over ``httpx.MockTransport``."""

    def __init__(self, hub: FakeHub) -> None:
        self._client = httpx.AsyncClient(transport=httpx.MockTransport(hub.handler))

    async def send(
        self,
        method: str,
        url: str,
        *,
        budget: Any,
        json_body: Any,
        headers: Any,
        params: Any,
        stage: str,
    ) -> httpx.Response:
        return await self._client.request(
            method, url, json=json_body, headers=dict(headers), params=dict(params or {})
        )

    async def aclose(self) -> None:
        await self._client.aclose()
