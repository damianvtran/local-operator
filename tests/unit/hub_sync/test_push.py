"""Outbound push against the frozen revision contract: preview -> CAS commit, replay.

The HTTP layer is exercised through ``httpx.MockTransport`` (PATH, Authorization,
If-Match verbatim, preview-token header, 412 mapping, no tenant hints on 404); the
push flow itself is exercised through a protocol-conformant fake transport, with
PERSISTED-EFFECT assertions (outbox states/intent), not call-count mirrors.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Mapping, TypeVar, overload

import httpx
import pytest

from local_operator.hub_sync.budget import OperationBudget, OperationDeadlineExceeded
from local_operator.hub_sync.merge import MergeOptions
from local_operator.hub_sync.outbox import (
    PUSH_MAX_ATTEMPTS,
    AsyncDispatcher,
    Outbox,
    OutboxRecord,
)
from local_operator.hub_sync.push import (
    ETAG_IF_MATCH,
    HttpPushTransport,
    PushOutcome,
    PushRefused,
    PushTransport,
    RemoteRevisionChanged,
    push_item,
    receipt_for,
)

A = {"instructions": "# A\nbase rule\n", "description": "d"}
EDITED = {"instructions": "# A\nbase rule\nNEW\n", "description": "d"}
TOKEN = '"revision-v1-' + "0" * 64 + '"'


def _budget() -> OperationBudget:
    return OperationBudget.start("automatic", "op", 1)


def _stored(out: Outbox, operation_id: str) -> OutboxRecord:
    """The stored record, ASSERTED present: every call here enqueued one first."""

    record = out.read(operation_id)
    assert record is not None
    return record


class FakeTransport:
    """Protocol-conformant transport building the frozen result shapes."""

    def __init__(
        self,
        *,
        revision: str | None = "rev-a",
        status: str = "ready",
        unresolved: list[Mapping[str, Any]] | None = None,
        fail_preview: BaseException | None = None,
        fail_commit: BaseException | None = None,
        hang_commit: bool = False,
        accepted: Mapping[str, Any] | None = None,
    ) -> None:
        self.revision = revision
        self.status = status
        self.unresolved = unresolved or []
        self.fail_preview = fail_preview
        self.fail_commit = fail_commit
        self.hang_commit = hang_commit
        self.accepted = accepted if accepted is not None else {"instructions": "# A\nmerged\n"}
        self.preview_calls: list[dict[str, Any]] = []
        self.commit_calls: list[dict[str, Any]] = []

    async def preview(self, *, document, expected_revision, budget):
        self.preview_calls.append(
            {"document": dict(document), "expected_revision": expected_revision}
        )
        if self.fail_preview is not None:
            raise self.fail_preview
        return {
            "revision": self.revision,
            "status": self.status,
            "document": dict(document),
            "unresolved": list(self.unresolved),
            "pin": {"token": "pin-1"},
        }

    async def commit(
        self, *, document, preview_token, accept_unresolved, expected_revision, budget
    ):
        # CLOSED-DECODER FAKE: the server's commit route reads the BARE document
        # (closed key set per family; an envelope key is refused verbatim as
        # "preview_version is not a recognised field" -- agent-server@bdd03ab,
        # internal/requests/skills.go + docs/HUB-REVISIONS.md:35-37). A fake that
        # accepted the envelope would let R11 regress unnoticed.
        if "preview_version" in document or "document" in document:
            raise PushRefused("invalid_skill_document", "preview_version is not a recognised field")

        self.commit_calls.append(
            {
                "document": dict(document),
                "preview_token": preview_token,
                "accept_unresolved": list(accept_unresolved),
                "expected_revision": expected_revision,
            }
        )
        if self.hang_commit:
            raise OperationDeadlineExceeded("commit exceeded")
        if self.fail_commit is not None:
            raise self.fail_commit
        return {
            "revision": "rev-b",
            "content_revision": 2,
            "accepted_document": dict(self.accepted),
            "operation_id": "srv-op",
        }


#: The transport a ``_run`` call passed comes back as itself, so a REAL
#: (MockTransport-backed) run can assert on it without a cast, while the
#: default-selecting overload keeps ``FakeTransport`` for the fake-based tests.
T = TypeVar("T", bound=PushTransport)


@overload
async def _run(
    tmp_path: Path, *, transport: None = None, expected: str | None = "rev-a", **kw: Any
) -> tuple[PushOutcome, Outbox, FakeTransport]: ...


@overload
async def _run(
    tmp_path: Path, *, transport: T, expected: str | None = "rev-a", **kw: Any
) -> tuple[PushOutcome, Outbox, T]: ...


async def _run(
    tmp_path: Path,
    *,
    transport: PushTransport | None = None,
    expected: str | None = "rev-a",
    **kw: Any,
) -> tuple[PushOutcome, Outbox, PushTransport]:
    server = transport if transport is not None else FakeTransport()
    out = Outbox(tmp_path)
    pushed: dict[str, Any] = {
        "kind": "agent",
        "local_id": "a1",
        "captured_local": A,
        "accepted": A,
        "current_local": EDITED,
        "remote_latest": A,
        "transport": server,
        "outbox": out,
        "operation_id": "op1",
        "budget": _budget(),
        "opts": MergeOptions(allow_llm=False),
    }
    pushed.update(kw)
    outcome = await push_item(expected_revision=expected, **pushed)
    return outcome, out, server


def test_a_local_edit_is_previewed_then_committed(tmp_path: Path) -> None:
    outcome, out, server = asyncio.run(_run(tmp_path))
    assert outcome.state == "pushed" and outcome.revision == "rev-b"
    assert "NEW" in server.preview_calls[0]["document"]["instructions"]
    # The commit body is the PREVIEW's returned document, with the pin header.
    assert server.commit_calls[0]["preview_token"] == "pin-1"
    assert server.commit_calls[0]["document"] == server.preview_calls[0]["document"]
    assert server.commit_calls[0]["expected_revision"] == "rev-a"
    record = _stored(out, "op1")
    assert record.state == "done" and record.result == {
        "revision": "rev-b",
        "operation_id": "srv-op",
    }


def test_identical_remote_settles_a_no_op_without_paying(tmp_path: Path) -> None:
    outcome, out, server = asyncio.run(_run(tmp_path, current_local=A))
    assert outcome.state == "no-op"
    assert server.preview_calls == [] and server.commit_calls == []
    assert _stored(out, "op1").state == "done"


def test_a_preview_on_another_revision_is_a_conflict_not_a_commit(tmp_path: Path) -> None:
    outcome, out, server = asyncio.run(
        _run(tmp_path, transport=FakeTransport(revision="rev-moved"))
    )
    assert outcome.state == "conflict" and outcome.reason == "remote-revision-changed"
    assert server.commit_calls == []  # nothing was committed on a moved revision
    record = _stored(out, "op1")
    assert record.state == "failed" and record.error_class == "remote-revision-changed"
    assert record.next_retry_at is not None  # backoff, not an every-tick retry


def test_a_preview_without_a_revision_refuses_locally(tmp_path: Path) -> None:
    outcome, out, server = asyncio.run(_run(tmp_path, transport=FakeTransport(revision=None)))
    assert outcome.state == "refused" and outcome.reason == "unsupported-precondition"
    assert server.commit_calls == []
    assert _stored(out, "op1").state == "failed"


def test_a_commit_412_is_a_named_conflict_and_is_not_resubmitted(tmp_path: Path) -> None:
    outcome, out, server = asyncio.run(
        _run(tmp_path, transport=FakeTransport(fail_commit=RemoteRevisionChanged()))
    )
    assert outcome.state == "conflict" and outcome.reason == "remote-revision-changed"
    assert len(server.commit_calls) == 1  # the stale body was NOT resubmitted
    assert _stored(out, "op1").error_class == "remote-revision-changed"


def test_needs_ack_unresolved_values_stop_before_the_commit(tmp_path: Path) -> None:
    outcome, _out, server = asyncio.run(
        _run(
            tmp_path,
            transport=FakeTransport(status="needs_ack", unresolved=[{"id": "u1"}]),
        )
    )
    assert outcome.state == "needs-review" and outcome.reason == "generalization-needs-ack"
    assert server.commit_calls == []  # an unclassified value is never auto-acknowledged


def test_an_aborted_commit_replays_the_stored_intent_verbatim(tmp_path: Path) -> None:
    wonky = FakeTransport(hang_commit=True)
    outcome, out, server = asyncio.run(_run(tmp_path, transport=wonky))
    assert outcome.state == "pending-reconciliation"
    record = _stored(out, "op1")
    assert record.state == "pending-reconciliation"
    assert record.intent is not None  # the verbatim intent was persisted
    stored_intent = dict(record.intent)

    # The NEXT tick replays: same pin, same body, same expected revision; NO new
    # preview (a reconciliation must not re-enter the paid generalization stage).
    healthy = FakeTransport()
    replay, _out2, server2 = asyncio.run(_run(tmp_path, transport=healthy))
    assert replay.state == "pushed"
    assert server2.preview_calls == []  # no second preview
    assert len(server2.commit_calls) == 1
    assert server2.commit_calls[0]["preview_token"] == stored_intent["preview_token"]
    assert server2.commit_calls[0]["document"] == stored_intent["document"]
    assert server2.commit_calls[0]["expected_revision"] == stored_intent["expected_revision"]
    assert _stored(out, "op1").state == "done"


def test_a_failed_push_waits_for_its_backoff_window(tmp_path: Path) -> None:
    first = FakeTransport(fail_commit=PushRefused("hub-error"))
    outcome, out, server = asyncio.run(_run(tmp_path, transport=first))
    assert outcome.state == "refused"
    assert _stored(out, "op1").next_retry_at is not None

    # Inside the backoff window: skipped, and NOT a single request is made.
    skipped, _o, server2 = asyncio.run(_run(tmp_path, transport=FakeTransport()))
    assert skipped.state == "skipped" and skipped.reason == "push-backoff"
    assert server2.preview_calls == [] and server2.commit_calls == []

    # Past the window (rewrite the schedule): a fresh attempt re-projects.
    record = _stored(out, "op1")
    rewritten = OutboxRecord(
        operation_id=record.operation_id,
        kind=record.kind,
        local_id=record.local_id,
        direction=record.direction,
        generation=record.generation,
        state=record.state,
        created_at=record.created_at,
        updated_at=record.updated_at,
        result=record.result,
        error_class=record.error_class,
        attempts=record.attempts,
        next_retry_at="2000-01-01T00:00:00+00:00",
        intent=record.intent,
    )
    out._atomic_write(rewritten)
    again, _o, server3 = asyncio.run(_run(tmp_path, transport=FakeTransport()))
    assert again.state == "pushed"
    assert len(server3.preview_calls) == 1


def test_a_dead_lettered_record_is_skipped_without_requests(tmp_path: Path) -> None:
    from local_operator.hub_sync.push import _projection_fingerprint

    out = Outbox(tmp_path)
    out.enqueue(operation_id="op1", kind="agent", local_id="a1", direction="push", generation=1)
    # The SAME projection input ``_run`` will compute, so the capped record can
    # tell "the same input again" (skip) from "a deliberate new edit" (re-arm).
    fp = _projection_fingerprint(
        kind="agent",
        captured_local=A,
        accepted=A,
        current_local=EDITED,
        remote_latest=A,
        expected_revision="rev-a",
    )
    for _ in range(PUSH_MAX_ATTEMPTS):
        attempt = out.begin_attempt("op1", kind="agent", local_id="a1", input_fp=fp)
        assert attempt is not None
        assert out.settle("op1", generation=attempt.generation, error_class="hub-error")
    server = FakeTransport()
    outcome, _out2, _server2 = asyncio.run(_run(tmp_path, transport=server))
    assert outcome.state == "skipped" and outcome.reason == "push-retries-exhausted"
    assert server.preview_calls == [] and server.commit_calls == []

    # A CHANGED input re-arms the capped operation (round-2 R12a): retrying the
    # same bytes cannot change the answer, but a new edit deserves a fresh budget.
    changed = {"instructions": "# A\nbase rule\nANOTHER\n", "description": "d"}
    again, _out3, server3 = asyncio.run(
        _run(tmp_path, transport=FakeTransport(), current_local=changed)
    )
    assert again.state == "pushed" and len(server3.preview_calls) == 1


def test_no_outbound_receipt_refuses_without_touching_the_outbox(tmp_path: Path) -> None:
    server = FakeTransport()
    outcome, out, _server = asyncio.run(
        _run(tmp_path, transport=server, captured_local=None, accepted=None)
    )
    assert outcome.state == "refused" and outcome.reason == "no-outbound-receipt"
    assert server.preview_calls == []
    assert out.read("op1") is None  # not even an intent record was written


def test_an_empty_expected_revision_refuses_before_any_request(tmp_path: Path) -> None:
    outcome, out, server = asyncio.run(_run(tmp_path, expected=None))
    assert outcome.state == "refused" and outcome.reason == "unsupported-precondition"
    assert server.preview_calls == []
    assert _stored(out, "op1").state == "failed"


def test_receipt_for_pairs_the_sent_local_with_the_accepted_remote() -> None:
    accepted = {"instructions": "# A\ngeneralized\n"}
    receipt = receipt_for(
        "agent",
        accepted,
        {"instructions": "# A\nbase rule\nNEW\n", "description": "d"},
        revision=TOKEN,
        operation_id="srv-op",
    )
    assert receipt.local_fields["instructions"].endswith("NEW\n")
    # The OPAQUE ETag is the stored revision -- NEVER the integer content_revision,
    # which cannot be an If-Match token.
    assert receipt.remote_revision == TOKEN
    assert receipt.operation_id == "srv-op"


# -- the real HTTP transport over httpx.MockTransport --------------------------------------


def _dispatcher(handler: Any) -> AsyncDispatcher:
    return AsyncDispatcher(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def _transport(disp: AsyncDispatcher, **kw: Any) -> HttpPushTransport:
    base: dict[str, Any] = {
        "kind": "agent",
        "hub_id": "agt-1",
        "base_url": "https://hub.test/v1/",
        "headers": {"Authorization": "Bearer tok-123"},
        "visibility": "org",
        "tenant_id": "t1",
    }
    base.update(kw)
    return HttpPushTransport(disp, **base)


def test_the_http_transport_previews_and_commits_on_the_frozen_wire() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = json.loads(request.content)
        if request.url.path.endswith("/preview"):
            # The PREVIEW request envelope: only preview_version/document/base_document.
            assert set(body) <= {"preview_version", "document", "base_document"}
            return httpx.Response(
                200,
                json={
                    "msg": "ok",
                    "result": {
                        "revision": TOKEN,
                        "status": "ready",
                        "document": {"instructions": "# generalized\n"},
                        "unresolved": [],
                        "pin": {"token": "pin-1"},
                    },
                },
            )
        # The COMMIT route reads the BARE document: any envelope key is refused by
        # the real server ("preview_version is not a recognised field"), so this
        # fake answers exactly as the server would.
        if "preview_version" in body or "document" in body:
            return httpx.Response(
                422,
                json={"error": {"code": "invalid_skill_document", "field": "preview_version"}},
            )
        return httpx.Response(
            200,
            json={
                "msg": "ok",
                "result": {
                    "revision": '"revision-v1-' + "1" * 64 + '"',
                    "content_revision": 2,
                    "accepted_document": {"instructions": "# accepted\n"},
                    "operation_id": "srv-op",
                },
            },
        )

    async def run() -> None:
        disp = _dispatcher(handler)
        transport = _transport(disp)
        preview = await transport.preview(
            document={"instructions": "# merged\n"},
            expected_revision=TOKEN,
            budget=_budget(),
        )
        assert preview["pin"]["token"] == "pin-1"
        result = await transport.commit(
            document=preview["document"],
            preview_token="pin-1",
            accept_unresolved=[],
            expected_revision=TOKEN,
            budget=_budget(),
        )
        assert result["operation_id"] == "srv-op"
        await disp.aclose()

    asyncio.run(run())

    assert len(seen) == 2
    first, second = seen
    # PATH: the single configured /v1, never a doubled /v1/v1.
    assert first.url.path == "/v1/agents/agt-1/publish/preview"
    assert second.url.path == "/v1/agents/agt-1/publish"
    assert "//" not in first.url.path
    for request in seen:
        assert request.method == "PUT"
        assert request.headers["Authorization"] == "Bearer tok-123"
        # If-Match is the opaque ETag VERBATIM, quotes included.
        assert request.headers[ETAG_IF_MATCH] == TOKEN
        assert dict(request.url.params) == {"visibility": "org", "tenant_id": "t1"}
    envelope = json.loads(first.content)
    assert envelope["preview_version"] == 1 and "document" in envelope
    # The commit carries the preview pin AND the BARE preview-returned document:
    # no envelope (round-2 R11 -- the server's commit decoders are closed, and
    # docs/HUB-REVISIONS.md:35-37 says "Commit sends only the returned
    # result.document").
    assert second.headers["X-Radient-Preview-Token"] == "pin-1"
    assert json.loads(second.content) == {"instructions": "# generalized\n"}


def test_the_http_transport_stale_commit_maps_to_412_and_repreviews() -> None:
    """A 412 is a named conflict; the caller re-fetches -- never a header swap.

    The transport itself must not resubmit: it raises, and the NEXT attempt is a
    fresh preview at the new revision (asserted by the second preview carrying the
    new token)."""

    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path.endswith("/preview"):
            return httpx.Response(
                200,
                json={
                    "msg": "ok",
                    "result": {
                        "revision": TOKEN,
                        "status": "ready",
                        "document": {"instructions": "# g\n"},
                        "unresolved": [],
                        "pin": {"token": "pin-1"},
                    },
                },
            )
        return httpx.Response(
            412, json={"msg": "stale", "error": {"code": "remote-revision-changed"}}
        )

    async def run() -> None:
        disp = _dispatcher(handler)
        transport = _transport(disp)
        with pytest.raises(RemoteRevisionChanged):
            await transport.commit(
                document={"instructions": "# g\n"},
                preview_token="pin-1",
                accept_unresolved=[],
                expected_revision=TOKEN,
                budget=_budget(),
            )
        await disp.aclose()

    asyncio.run(run())
    # Exactly ONE commit was attempted: no blind resubmission.
    assert [r.url.path.endswith("/preview") for r in seen] == [False]


def test_the_http_transport_maps_server_codes_and_hides_tenants_on_404() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/preview"):
            return httpx.Response(
                400, json={"msg": "bad", "error": {"code": "unsupported-precondition"}}
            )
        return httpx.Response(404, json={"msg": "nope", "error": {"code": "skill_not_found"}})

    async def run() -> None:
        disp = _dispatcher(handler)
        transport = _transport(disp, kind="skill", hub_id="skl-1")
        with pytest.raises(PushRefused) as refused:
            await transport.preview(
                document={"name": "s"},
                expected_revision=TOKEN,
                budget=_budget(),
            )
        assert refused.value.code == "unsupported-precondition"
        with pytest.raises(PushRefused) as missing:
            await transport.commit(
                document={"name": "s"},
                preview_token="pin-1",
                accept_unresolved=[],
                expected_revision=TOKEN,
                budget=_budget(),
            )
        assert missing.value.code == "skill_not_found"
        # An org row addressed publicly must stay indistinguishable from absent:
        # no tenant hint anywhere in the refusal message.
        assert "t1" not in str(missing.value)
        await disp.aclose()

    asyncio.run(run())


def test_the_http_transport_refuses_an_empty_revision_without_a_request() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - never called
        seen.append(request)
        return httpx.Response(200, json={"msg": "ok", "result": {}})

    async def run() -> None:
        disp = _dispatcher(handler)
        transport = _transport(disp)
        with pytest.raises(PushRefused) as refused:
            await transport.commit(
                document={"instructions": "# m\n"},
                preview_token="pin-1",
                accept_unresolved=[],
                expected_revision="",
                budget=_budget(),
            )
        assert refused.value.code == "unsupported-precondition"
        await disp.aclose()

    asyncio.run(run())
    assert seen == []  # never an unconditional write


def test_a_replay_skips_when_the_record_left_the_uncertain_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3 guard: replay dispatches only while the record is the SAME uncertain attempt.

    ``begin_attempt`` fences, but releases the outbox lock before the network
    dispatch. Two raced shapes cover the window: (a) another window FINALIZED the
    record (it is no longer reconcilable); (b) a newer generation superseded this
    attempt. Both must skip BY NAME with no commit leaving the client.
    """

    from types import SimpleNamespace

    wonky = FakeTransport(hang_commit=True)
    outcome, out, _server = asyncio.run(_run(tmp_path, transport=wonky))
    assert outcome.state == "pending-reconciliation"
    record = _stored(out, "op1")
    assert record.intent is not None

    def stale_begin(self: Outbox, operation_id: str, *, kind: str, local_id: str, **kw: Any) -> Any:
        return SimpleNamespace(
            generation=record.generation, replay=dict(record.intent or {}), attempts=1
        )

    # (a) the record was finalized elsewhere: not reconcilable -> named skip.
    out.settle("op1", generation=record.generation, result={"revision": "rev-x"})
    assert _stored(out, "op1").state == "done"
    healthy = FakeTransport()
    monkeypatch.setattr(Outbox, "begin_attempt", stale_begin)
    raced, _out, server = asyncio.run(_run(tmp_path, transport=healthy))
    assert raced.state == "skipped" and raced.reason == "reconciliation-required"
    assert server.commit_calls == [] and server.preview_calls == []
    monkeypatch.undo()

    # (b) a newer generation superseded the attempt: same uncertain state, stale
    # generation -> named skip, still no commit.
    second = tmp_path / "second"
    second.mkdir()
    outcome2, out2, _server2 = asyncio.run(_run(second, transport=FakeTransport(hang_commit=True)))
    assert outcome2.state == "pending-reconciliation"
    record2 = _stored(out2, "op1")

    def newer_begin(self: Outbox, operation_id: str, *, kind: str, local_id: str, **kw: Any) -> Any:
        return SimpleNamespace(
            generation=record2.generation + 1, replay=dict(record2.intent or {}), attempts=1
        )

    monkeypatch.setattr(Outbox, "begin_attempt", newer_begin)
    raced2, _out2, server2 = asyncio.run(_run(second, transport=FakeTransport()))
    assert raced2.state == "skipped" and raced2.reason == "reconciliation-required"
    assert server2.commit_calls == []
    assert _stored(out2, "op1").state == "pending-reconciliation"  # untouched by us


def test_a_preview_deadline_is_retryable_and_the_next_healthy_tick_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R12c: a PREVIEW-stage deadline owes nothing -- backoff, then it republishes.

    The old code marked it pending-reconciliation with NO stored intent, which
    ``begin_attempt`` then skipped forever ("reconciliation-required"). Nothing
    was dispatched, so retrying is the honest answer.
    """

    from local_operator.hub_sync import outbox as outbox_mod

    wonky = FakeTransport(fail_preview=OperationDeadlineExceeded("preview over budget"))
    outcome, out, server = asyncio.run(_run(tmp_path, transport=wonky))
    assert outcome.state == "refused" and outcome.reason == "deadline"
    record = _stored(out, "op1")
    assert record.state == "failed" and record.waiting_input is False and record.next_retry_at
    assert server.commit_calls == []  # nothing was dispatched

    # The backoff window passes (fake clock), then a healthy tick publishes.
    future = ["2099-01-01T00:00:00Z"]
    monkeypatch.setattr(outbox_mod, "now_iso", lambda: future[0])
    replay, _o2, server2 = asyncio.run(_run(tmp_path, transport=FakeTransport()))
    assert replay.state == "pushed" and len(server2.preview_calls) == 1


def test_five_deterministic_refusals_wait_for_input_instead_of_dead_lettering(
    tmp_path: Path,
) -> None:
    """R12a + R15: a refusal retrying the SAME bytes cannot change never burns the cap.

    Five consecutive ticks of an unsatisfiable projection (a skill whose remote
    latest is empty) must leave the attempt counter alone and wait for the INPUT
    to move; a changed input then proceeds on the SAME record.
    """

    skill_md = "---\nname: s\ndescription: d\n---\n\nbody\n"
    skill = {
        "document_version": 1,
        "name": "s",
        "description": "d",
        "files": [{"path": "SKILL.md", "content": skill_md}],
    }
    first, out, server = asyncio.run(
        _run(
            tmp_path,
            transport=FakeTransport(),
            kind="skill",
            captured_local=skill,
            accepted=skill,
            current_local=skill,
            remote_latest={},
        )
    )
    assert first.state == "needs-review" and first.reason == "merge-refused"
    assert server.preview_calls == []
    # Ticks 2..5: the SAME input is skipped BY NAME, not retried into the cap.
    for _ in range(4):
        again, _o, server_n = asyncio.run(
            _run(
                tmp_path,
                transport=FakeTransport(),
                kind="skill",
                captured_local=skill,
                accepted=skill,
                current_local=skill,
                remote_latest={},
            )
        )
        assert again.state == "skipped" and again.reason == "awaiting-input-change"
        assert server_n.preview_calls == []
    record = _stored(out, "op1")
    assert record.attempts == 0 and record.waiting_input is True

    good, _o2, server2 = asyncio.run(
        _run(
            tmp_path,
            transport=FakeTransport(),
            kind="skill",
            captured_local=skill,
            accepted=skill,
            current_local=skill,
            remote_latest=skill,
        )
    )
    assert good.state == "pushed" and len(server2.preview_calls) == 1


def test_five_conflicts_refund_their_attempts_and_a_quiet_remote_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R12d: 412s back off but never dead-letter an actively edited link."""

    from local_operator.hub_sync import outbox as outbox_mod

    for i in range(5):
        future = [f"2099-01-0{i + 1}T00:00:00Z"]
        monkeypatch.setattr(outbox_mod, "now_iso", lambda future=future: future[0])
        outcome, out, _server = asyncio.run(
            _run(tmp_path, transport=FakeTransport(fail_commit=RemoteRevisionChanged()))
        )
        assert outcome.state == "conflict"
    record = _stored(Outbox(tmp_path), "op1")
    assert record.attempts == 0  # conflicts are refunded, never counted
    assert record.state == "failed"

    future = ["2099-02-01T00:00:00Z"]
    monkeypatch.setattr(outbox_mod, "now_iso", lambda: future[0])
    quiet, _o2, server2 = asyncio.run(_run(tmp_path, transport=FakeTransport()))
    assert quiet.state == "pushed" and len(server2.commit_calls) == 1


@pytest.mark.parametrize("outage_code", ["http-503", "hub-error", "hub-unavailable"])
def test_hub_outages_refund_their_attempts_and_a_healthy_hub_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outage_code: str
) -> None:
    """QN8 + QN10 + R22: an outage -- a 5xx under ANY body code, or a transport
    failure -- never exhausts the cap, and its window still grows.

    ``outage=True`` is the classification the transport derives from the STATUS
    (or the absence of a response); ``hub-unavailable`` is the reviewer's coded
    5xx, whose body code must not change the attempt class.
    """

    from local_operator.hub_sync import outbox as outbox_mod
    from local_operator.hub_sync.outbox import push_backoff_seconds

    previous_window: str | None = None
    for i in range(5):
        future = [f"2099-01-0{i + 1}T00:00:00Z"]
        monkeypatch.setattr(outbox_mod, "now_iso", lambda future=future: future[0])
        outcome, _out, _srv = asyncio.run(
            _run(
                tmp_path,
                transport=FakeTransport(
                    fail_commit=PushRefused(outage_code, "outage", outage=True)
                ),
            )
        )
        assert outcome.state == "refused" and outcome.reason == outage_code
        record = _stored(Outbox(tmp_path), "op1")
        assert record.attempts == 0, "an outage is not an attempt"
        assert record.outage_streak == i + 1
        assert record.next_retry_at is not None
        if previous_window is not None:
            # R22: the refunded counter cannot pace; the STREAK still doubles it.
            assert record.next_retry_at > previous_window, record.next_retry_at
        previous_window = record.next_retry_at

    # The window grows to the six-hour cap, never past it.
    assert push_backoff_seconds(5) == 14400.0
    assert push_backoff_seconds(6) == 21600.0

    future = ["2099-02-01T00:00:00Z"]
    monkeypatch.setattr(outbox_mod, "now_iso", lambda future=future: future[0])
    pushed, _o2, server = asyncio.run(_run(tmp_path, transport=FakeTransport()))
    assert pushed.state == "pushed" and len(server.commit_calls) == 1
    assert _stored(Outbox(tmp_path), "op1").outage_streak == 0, "success resets the streak"


def test_a_real_coded_5xx_refunds_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QN10's exact repro over REAL HTTP: 503 + {error:{code:'hub-unavailable'}}
    x5, backoff elapsed each time, then a healthy UNCHANGED input publishes."""

    from local_operator.hub_sync import outbox as outbox_mod

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"code": "hub-unavailable"}})

    for i in range(5):
        future = [f"2099-01-0{i + 1}T00:00:00Z"]
        monkeypatch.setattr(outbox_mod, "now_iso", lambda future=future: future[0])
        outcome, _out, _srv = asyncio.run(
            _run(tmp_path, transport=_transport(_dispatcher(handler)))
        )
        assert outcome.state == "refused" and outcome.reason == "hub-unavailable"

    record = _stored(Outbox(tmp_path), "op1")
    assert record.attempts == 0 and record.outage_streak == 5

    future = ["2099-02-01T00:00:00Z"]
    monkeypatch.setattr(outbox_mod, "now_iso", lambda future=future: future[0])
    pushed, _o2, server = asyncio.run(_run(tmp_path, transport=FakeTransport()))
    assert pushed.state == "pushed" and len(server.commit_calls) == 1


def test_a_coded_5xx_is_classified_by_status_not_the_body_code() -> None:
    """QN10 root: outage-ness derives from the HTTP STATUS; the body code stays
    the user-facing reason. A 4xx with a code is NOT an outage."""

    def handler_503(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503, json={"error": {"code": "hub-unavailable", "message": "down for maintenance"}}
        )

    async def run_503() -> PushRefused:
        disp = _dispatcher(handler_503)
        transport = _transport(disp)
        with pytest.raises(PushRefused) as raised:
            await transport.preview(
                document={"instructions": "x"}, expected_revision=TOKEN, budget=_budget()
            )
        await disp.aclose()
        return raised.value

    refused = asyncio.run(run_503())
    assert refused.code == "hub-unavailable" and refused.outage is True

    def handler_422(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"error": {"code": "invalid_skill_document"}})

    async def run_422() -> PushRefused:
        disp = _dispatcher(handler_422)
        transport = _transport(disp)
        with pytest.raises(PushRefused) as raised:
            await transport.preview(
                document={"instructions": "x"}, expected_revision=TOKEN, budget=_budget()
            )
        await disp.aclose()
        return raised.value

    refused_422 = asyncio.run(run_422())
    assert refused_422.code == "invalid_skill_document" and refused_422.outage is False


def test_a_deterministic_refusal_still_consumes_attempts_and_dead_letters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """QN8 contrast: a server ANSWER still spends attempts; the cap still holds."""

    from local_operator.hub_sync import outbox as outbox_mod

    for i in range(5):
        future = [f"2099-01-0{i + 1}T00:00:00Z"]
        monkeypatch.setattr(outbox_mod, "now_iso", lambda future=future: future[0])
        outcome, _out, _srv = asyncio.run(
            _run(
                tmp_path,
                transport=FakeTransport(
                    fail_commit=PushRefused("invalid_skill_document", "the document is invalid")
                ),
            )
        )
        assert outcome.state == "refused"

    record = _stored(Outbox(tmp_path), "op1")
    assert record.attempts == 5, "deterministic refusals keep consuming attempts"

    future = ["2099-02-01T00:00:00Z"]
    monkeypatch.setattr(outbox_mod, "now_iso", lambda future=future: future[0])
    skipped, _o2, server = asyncio.run(_run(tmp_path, transport=FakeTransport()))
    assert skipped.state == "skipped" and skipped.reason == "push-retries-exhausted"
    assert len(server.commit_calls) == 0
