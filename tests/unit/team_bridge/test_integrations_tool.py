"""The ``integrations`` tool over a faked agent-server (RFC Team Integrations §8, Phase 3a).

The HTTP layer is faked at the Team Bridge client's ``_authorized`` seam, so the
real ``IntegrationsClient`` runs: its refusal mapping, the house envelope parse
and the 204 path. The fake server models the parts of agent-server PR #401 the
harness relies on — a draft answers a publish commitment, ``prepare: true``
answers one for a post-publish action, every commitment is single use
(``409 commitment_already_used`` on replay), and a mint answers the raw secret
exactly once. The ask queue and the secret store are the REAL ones under a
sandboxed config dir, so "sealed", "refused" and "nothing leaked" are read back
from the same objects production uses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets as pysecrets
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator.asks.queue import AskQueue
from local_operator.harness.types import ToolContext
from local_operator.paths import CONFIG_DIR_ENV
from local_operator.secrets.crypto import generate_master_key
from local_operator.secrets.store import SecretStore
from local_operator.team_bridge import integrations as integ
from local_operator.team_bridge.integration_consent import IntegrationConsent
from local_operator.tools import integrations_tool
from local_operator.tools.integrations_tool import (
    build_integrations_tool,
    execute_integrations,
)

TENANT = "ten_acme"
APP = "iap_" + "a" * 32
CHANNEL = "bcv_" + "c" * 32
BASE = 1_700_000_000_000


# ---------------------------------------------------------------------------
# the fake agent-server
# ---------------------------------------------------------------------------


class FakeServer:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.commitments: dict[str, dict[str, Any]] = {}
        self.minted: list[str] = []
        self.app_status = "draft"
        self.refuse: tuple[int, dict[str, Any]] | None = None
        self.dark = False

    def _commit(self, action: str) -> dict[str, Any]:
        digest = pysecrets.token_hex(32)
        self.commitments[digest] = {"action": action, "used": False}
        return {
            "action": action,
            "confirmation_commitment": digest,
            "expires_at": "2026-11-18T12:15:00Z",
            "revision": 1,
        }

    def _consume(self, body: dict[str, Any], action: str) -> tuple[int, dict[str, Any]] | None:
        c = self.commitments.get(str(body.get("confirmation_commitment") or ""))
        if not body.get("ask_ref"):
            return 403, {
                "error": "A confirmation commitment is required.",
                "code": "commitment_required",
            }
        if c is None or c["used"]:
            return 409, {
                "error": "This confirmation commitment has already been used or expired.",
                "code": "commitment_already_used",
            }
        if c["action"] != action:
            return 409, {"error": "mismatch", "code": "commitment_mismatch"}
        c["used"] = True
        return None

    def _secret(self, prefix: str) -> str:
        value = prefix + pysecrets.token_urlsafe(32).replace("-", "x").replace("_", "y") + "_Ab12Cd"
        self.minted.append(value)
        return value

    def handle(self, method: str, path: str, body: dict[str, Any] | None) -> tuple[int, Any]:
        self.calls.append((method, path, body))
        if path == "/messaging/status":
            return 200, {
                "tenants": [{"tenant_id": TENANT, "tenant_name": "Acme", "eligible": True}]
            }
        if self.dark:
            # Echo's router 404 for an unregistered route: no integration code.
            return 404, {"error": "Not Found", "code": "not_found_route_x"}
        if self.refuse is not None:
            return self.refuse
        root = f"/tenants/{TENANT}/integrations"
        body = body or {}
        if path == f"{root}/channels":
            return 200, {
                "channels": [{"id": CHANNEL, "name": "alerts-prod", "installed_app_ids": []}],
                "can_manage_integrations": True,
            }
        if path == f"{root}/apps" and method == "GET":
            return 200, {"apps": [], "app_summaries": [], "visibility": "full"}
        if path == f"{root}/apps" and method == "POST":
            return 201, {
                "app": {
                    "id": APP,
                    "name": body["name"],
                    "status": "draft",
                    "scopes": body["scopes"],
                },
                "commitment": self._commit("publish"),
            }
        if path == f"{root}/apps/{APP}" and method == "GET":
            return 200, {
                "visibility": "full",
                "app": {
                    "id": APP,
                    "name": "Alertmanager",
                    "status": self.app_status,
                    "scopes": ["chat:write"],
                    "proposed_channel_ids": [CHANNEL],
                    "initial_credential": {"kind": "token"},
                },
                "installations": [],
                "tokens": [],
                "hooks": [],
            }
        if path == f"{root}/apps/{APP}/publish":
            if body.get("prepare"):
                return 200, {"prepared": True, "commitment": self._commit("publish")}
            refused = self._consume(body, "publish")
            if refused:
                return refused
            self.app_status = "active"
            return 200, {
                "app": {"id": APP, "name": "Alertmanager", "status": "active"},
                "installations": [{"conversation_id": CHANNEL}],
                "token": {
                    "id": "itk_" + "1" * 32,
                    "prefix_hint": "AbCdEf",
                    "scopes": ["chat:write"],
                },
                "secret": self._secret("rdi_at_"),
            }
        if path == f"{root}/apps/{APP}/tokens":
            if body.get("prepare"):
                return 200, {"prepared": True, "commitment": self._commit("mint_token")}
            refused = self._consume(body, "mint_token")
            if refused:
                return refused
            return 201, {
                "token": {
                    "id": "itk_" + "2" * 32,
                    "prefix_hint": "ZyXwVu",
                    "scopes": body.get("scopes"),
                },
                "secret": self._secret("rdi_at_"),
            }
        if path == f"{root}/apps/{APP}/hooks":
            if body.get("prepare"):
                return 200, {"prepared": True, "commitment": self._commit("create_hook")}
            refused = self._consume(body, "create_hook")
            if refused:
                return refused
            return 201, {
                "hook": {"id": "ihk_" + "3" * 32, "vendor": body["vendor"], "status": "off"},
                "ingest_path": "/v1/integrations/hooks/ihk_" + "3" * 32,
                "secret": self._secret("rdi_hk_"),
            }
        if path == f"{root}/apps/{APP}/installations":
            if body.get("prepare"):
                return 200, {"prepared": True, "commitment": self._commit("install")}
            refused = self._consume(body, "install")
            if refused:
                return refused
            return 201, {"installation": {"conversation_id": body["conversation_id"]}}
        if method == "DELETE" and path.startswith(f"{root}/apps/{APP}/tokens/"):
            return 204, None
        return 404, {"error": "The integration resource was not found.", "code": "not_found"}


class FakeBridge:
    """Stands in for ``TeamBridgeClient``: only ``_authorized`` is reached."""

    def __init__(self, server: FakeServer) -> None:
        self.server = server

    async def _authorized(
        self, method: str, path: str, *, body=None, params=None, **_: Any
    ) -> httpx.Response:
        status, payload = self.server.handle(method, path, dict(body) if body is not None else None)
        request = httpx.Request(method, "https://api.example.test/v1" + path)
        if payload is None:
            return httpx.Response(status, request=request)
        content = payload if status >= 400 else {"msg": "ok", "result": payload}
        return httpx.Response(status, json=content, request=request)


# ---------------------------------------------------------------------------
# the harness side: a real queue, a real store, a session-shaped seam
# ---------------------------------------------------------------------------


class _Session:
    def credential_names(self) -> list[str]:
        return []

    def ask_reach(self) -> list[str]:
        return ["tui"]

    def _spawn_background(self, coro):
        coro.close()

    async def deliver_ask_messages(self, messages) -> None:
        return None


class Rig:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        root = tmp_path / "config"
        root.mkdir()
        (tmp_path / "home").mkdir()
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        monkeypatch.setenv(CONFIG_DIR_ENV, str(root))
        self.store = SecretStore(generate_master_key(), base=root)
        self.store.initialize()
        assert self.store.path.is_relative_to(root), "store escaped the sandbox"
        import local_operator.secrets.access as access

        monkeypatch.setattr(access, "open_store", lambda base=None, create=False: self.store)
        self.now = [BASE]
        self.queue = AskQueue(
            _Session(), config_dir=root, session_id="s1", clock=lambda: self.now[0]
        )
        self.answered: set[str] = set()
        self.seam = IntegrationConsent(
            lambda: self.queue, lambda: set(self.answered), session_id="s1"
        )
        self.server = FakeServer()
        self.client = integ.IntegrationsClient(FakeBridge(self.server))  # type: ignore[arg-type]
        self.results: list[Any] = []

    def context(self, **overrides: Any) -> ToolContext:
        fields: dict[str, Any] = {
            "ask_user": lambda *a, **k: None,
            "integration_consent": self.seam,
            "integrations_client_factory": lambda: self.client,
            "session_id": "s1",
        }
        fields.update(overrides)
        return ToolContext(**fields)

    def call(self, **args: Any) -> dict[str, Any]:
        result = asyncio.run(execute_integrations("call-1", args, context=self.context()))
        self.results.append(result)
        return dict(result.details or {})

    def answer(self, ask_id: str, pick: str, *, person: bool = True) -> None:
        receipt = self.queue.respond(ask_id, {integ.QUESTION_ID: [pick]}, by="user")
        assert receipt.get("ok"), receipt
        if person:
            # What Session._run_answered_ask_ids holds once the person's
            # answer row is classified (ask_answer_is_person).
            self.answered.add(ask_id)

    def consent(self, ask_id: str) -> dict[str, Any]:
        record = self.queue.find(ask_id)
        assert record is not None
        return dict(record["consent"])

    def mutating_calls(self) -> list[tuple[str, str, dict[str, Any] | None]]:
        return [c for c in self.server.calls if c[0] != "GET" and not (c[2] or {}).get("prepare")]

    def transcript_text(self) -> str:
        return "\n".join(
            json.dumps([getattr(c, "text", "") for c in r.content])
            + json.dumps(r.details or {}, default=str)
            for r in self.results
        )


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Rig:
    return Rig(tmp_path, monkeypatch)


def _draft(rig: Rig, **extra: Any) -> dict[str, Any]:
    args = {
        "op": "draft_app",
        "name": "Alertmanager",
        "scopes": ["chat:write"],
        "channel_ids": [CHANNEL],
        "credential": "token",
        "secret_name": "RADIENT_APP_AM_TOKEN",
    }
    args.update(extra)
    return rig.call(**args)


# ---------------------------------------------------------------------------
# the happy path, and the zero-plaintext guarantee
# ---------------------------------------------------------------------------


def test_draft_files_one_human_ask_and_creates_nothing_live(rig: Rig) -> None:
    out = _draft(rig)
    assert out["state"] == "queued" and out["app_id"] == APP
    consent = rig.consent(out["ask_id"])
    draft_commitment = [b for (_, p, b) in rig.server.calls if p.endswith("/apps") and b]
    assert consent["op"] == integ.CONSENT_OP
    assert (
        consent["integration_op"] == "publish_app"
        and consent["secret_name"] == "RADIENT_APP_AM_TOKEN"
    )
    # The frozen commitment is the one the server answered the draft with.
    assert consent["commitment"] in rig.server.commitments
    assert rig.server.commitments[consent["commitment"]]["action"] == "publish"
    assert draft_commitment and rig.mutating_calls() == [
        ("POST", f"/tenants/{TENANT}/integrations/apps", draft_commitment[0])
    ]
    record = rig.queue.find(out["ask_id"])
    assert record is not None and record["status"] == "open"
    question = record["questions"][0]
    assert "Alertmanager" in question["question"] and "#alerts-prod" in question["question"]
    assert "RADIENT_APP_AM_TOKEN" in question["question"]
    assert [o["label"] for o in question["options"]] == ["Approve and publish", "Cancel"]


def test_publish_before_an_answer_reuses_the_queued_ask_and_calls_nothing(rig: Rig) -> None:
    out = _draft(rig)
    before = list(rig.server.calls)
    again = rig.call(op="publish_app", app_id=APP)
    assert again["state"] == "queued" and again["ask_id"] == out["ask_id"]
    assert [c for c in rig.server.calls[len(before) :] if c[1] != "/messaging/status"] == []


def test_approved_publish_seals_the_token_and_never_leaks_it(
    rig: Rig, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    out = _draft(rig)
    frozen = rig.consent(out["ask_id"])["commitment"]
    rig.answer(out["ask_id"], "Approve and publish")
    done = rig.call(op="publish_app", app_id=APP)
    assert done["state"] == "done", done
    # The server received exactly the frozen commitment and an ask_ref naming the ask.
    publish = [
        b
        for (m, p, b) in rig.server.calls
        if p.endswith("/publish") and not (b or {}).get("prepare")
    ]
    assert publish == [{"confirmation_commitment": frozen, "ask_ref": f"lop:s1:{out['ask_id']}"}]
    (raw,) = rig.server.minted
    assert raw.startswith("rdi_at_")
    # Sealed: the store holds the exact bytes under the frozen name.
    assert rig.store.get("RADIENT_APP_AM_TOKEN") == raw.encode()
    from local_operator.secrets import runtime

    assert raw in runtime.registered_values()
    meta = done["credential"]
    assert (
        meta["secret_name"] == "RADIENT_APP_AM_TOKEN"
        and meta["length"] == len(raw)
        and meta["sealed"]
    )
    # ZERO secret bytes in the tool result (text and details) or in any log record.
    assert raw not in rig.transcript_text()
    assert raw[7:] not in rig.transcript_text()
    assert raw not in caplog.text and raw[7:] not in caplog.text
    # The grant is spent: a repeat cannot publish twice.
    again = rig.call(op="publish_app", app_id=APP)
    assert again["state"] != "done"


def test_mint_token_two_step_seals_and_preflights_the_name(rig: Rig) -> None:
    rig.server.app_status = "active"
    args = {
        "op": "mint_token",
        "app_id": APP,
        "scopes": ["chat:write", "chat:write.thread"],
        "secret_name": "SCRAPER_TOKEN",
    }
    first = rig.call(**args)
    assert first["state"] == "queued"
    assert rig.server.minted == []
    rig.answer(first["ask_id"], "Approve")
    done = rig.call(**args)
    assert done["state"] == "done" and done["token"]["prefix_hint"] == "ZyXwVu"
    assert rig.store.get("SCRAPER_TOKEN") == rig.server.minted[0].encode()
    assert rig.server.minted[0] not in rig.transcript_text()


def test_a_taken_secret_name_is_refused_before_anything_is_minted(rig: Rig) -> None:
    rig.store.set("TAKEN", b"existing-value")
    out = _draft(rig, secret_name="TAKEN")
    assert out["state"] == "secret_exists"
    assert rig.mutating_calls() == []
    rig.server.app_status = "active"
    out = rig.call(op="mint_token", app_id=APP, secret_name="TAKEN")
    assert out["state"] == "secret_exists"
    assert [c for c in rig.server.calls if c[1].endswith("/tokens")] == []
    assert rig.store.get("TAKEN") == b"existing-value"


# ---------------------------------------------------------------------------
# every non-human answer is refused, with zero server mutation
# ---------------------------------------------------------------------------


def test_cancel_is_a_decline_with_zero_retry(rig: Rig) -> None:
    out = _draft(rig)
    rig.answer(out["ask_id"], "Cancel")
    res = rig.call(op="publish_app", app_id=APP)
    assert res["state"] == "declined"
    assert not [c for c in rig.server.calls if c[1].endswith("/publish")]


def test_declined_to_answer_is_not_consent(rig: Rig) -> None:
    out = _draft(rig)
    rig.queue.decline(out["ask_id"], by="user")
    assert rig.call(op="publish_app", app_id=APP)["state"] == "declined"
    assert not [c for c in rig.server.calls if c[1].endswith("/publish")]


def test_answered_in_chat_is_never_consent(rig: Rig) -> None:
    out = _draft(rig)
    receipt = rig.queue.withdraw(
        out["ask_id"],
        reason="answered_in_chat",
        answers={integ.QUESTION_ID: ["Approve and publish"]},
    )
    assert receipt.get("ok"), receipt
    res = rig.call(op="publish_app", app_id=APP)
    assert res["state"] in ("not_human", "timed_out")
    assert not [c for c in rig.server.calls if c[1].endswith("/publish")]


def test_an_affirmative_answer_not_given_by_a_person_this_run_is_refused(rig: Rig) -> None:
    out = _draft(rig)
    rig.answer(out["ask_id"], "Approve and publish", person=False)
    res = rig.call(op="publish_app", app_id=APP)
    assert res["state"] == "timed_out"
    assert not [c for c in rig.server.calls if c[1].endswith("/publish")]


def test_a_typed_answer_is_neither_yes_nor_no(rig: Rig) -> None:
    out = _draft(rig)
    rig.answer(out["ask_id"], "yes please do it")
    assert rig.call(op="publish_app", app_id=APP)["state"] == "not_human"
    assert not [c for c in rig.server.calls if c[1].endswith("/publish")]


def test_timeout_keeps_the_draft_and_reask_files_a_fresh_commitment(rig: Rig) -> None:
    out = _draft(rig)
    old = rig.consent(out["ask_id"])["commitment"]
    rig.now[0] = BASE + (integ.ASK_TIMEOUT_S + 5) * 1000
    lapsed = rig.call(op="publish_app", app_id=APP)
    assert lapsed["state"] == "timed_out"
    assert not [c for c in rig.server.calls if c[1].endswith("/publish")]
    fresh = rig.call(op="publish_app", app_id=APP, reask=True)
    assert fresh["state"] == "queued" and fresh["ask_id"] != out["ask_id"]
    new = rig.consent(fresh["ask_id"])
    assert new["commitment"] != old and new["secret_name"] == "RADIENT_APP_AM_TOKEN"
    rig.answer(fresh["ask_id"], "Approve and publish")
    assert rig.call(op="publish_app", app_id=APP)["state"] == "done"


def test_a_replayed_commitment_maps_to_commitment_already_used(rig: Rig) -> None:
    out = _draft(rig)
    rig.server.commitments[rig.consent(out["ask_id"])["commitment"]]["used"] = True
    rig.answer(out["ask_id"], "Approve and publish")
    res = rig.call(op="publish_app", app_id=APP)
    assert res["state"] == "commitment_already_used" and res["code"] == "commitment_already_used"
    assert rig.server.minted == []


def test_an_ask_cited_for_a_different_action_grants_nothing(rig: Rig) -> None:
    rig.server.app_status = "active"
    first = rig.call(op="mint_token", app_id=APP, scopes=["chat:write"], secret_name="A_TOKEN")
    rig.answer(first["ask_id"], "Approve")
    # Same ask id, wider scope: not what the person approved.
    res = rig.call(
        op="mint_token",
        app_id=APP,
        scopes=["alerts:ingest"],
        secret_name="A_TOKEN",
        ask_id=first["ask_id"],
    )
    # The cited ask is filtered out by its action key, so a NEW card is filed
    # for the wider request instead of the old yes being spent on it.
    assert res["state"] == "queued" and res["ask_id"] != first["ask_id"]
    assert rig.server.minted == []


# ---------------------------------------------------------------------------
# the five-state server refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "body", "state", "needle"),
    [
        (
            403,
            {
                "error": "Team Integrations requires an active Radient Team plan or free trial.",
                "code": "team_plan_required",
                "details": {"trial_available": True},
            },
            "team_plan_required",
            integ.BILLING_URL,
        ),
        (
            403,
            {
                "error": "Only organization owners and admins can manage integrations.",
                "code": "insufficient_role",
            },
            "insufficient_role",
            "“Acme”",
        ),
    ],
)
def test_plan_and_role_refusals_map_to_their_user_states(
    rig: Rig, status: int, body: dict[str, Any], state: str, needle: str
) -> None:
    rig.server.refuse = (status, body)
    res = _draft(rig)
    assert res["state"] == state and needle in res["message"]
    assert rig.queue.records() == []


def test_flag_off_reads_as_unavailable(rig: Rig) -> None:
    rig.server.dark = True
    res = _draft(rig)
    assert res["state"] == "unavailable"
    status = rig.call(op="status")
    assert status["organizations"][0]["state"] == "unavailable"


def test_a_coded_not_found_is_not_mistaken_for_flag_off() -> None:
    refusal = integ.BridgeRefusal(
        "not_found", "The integration resource was not found.", http_status=404
    )
    assert integ.map_refusal(refusal).state == "refused"


def test_status_reports_ready_with_channels(rig: Rig) -> None:
    out = rig.call(op="status")
    (org,) = out["organizations"]
    assert org["state"] == "ready" and org["channels"][0]["name"] == "alerts-prod"


def test_revoke_runs_without_an_ask(rig: Rig) -> None:
    res = rig.call(op="revoke", app_id=APP, token_id="itk_" + "9" * 32)
    assert res["state"] == "done"
    assert rig.queue.records() == []


def test_ids_are_validated_before_any_path_is_built(rig: Rig) -> None:
    res = rig.call(op="publish_app", app_id="../../admin")
    assert res["state"] == "invalid"
    assert all("admin" not in p for (_, p, _) in rig.server.calls)


# ---------------------------------------------------------------------------
# the createIf gate
# ---------------------------------------------------------------------------


def test_the_tool_exists_only_with_a_person_a_seam_and_a_login(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(integrations_tool, "_gate_bridge_login", lambda: True)
    assert build_integrations_tool(rig.context(integrations_client_factory=None)) is not None
    assert build_integrations_tool(rig.context(ask_user=None)) is None
    assert build_integrations_tool(rig.context(integration_consent=None)) is None
    monkeypatch.setattr(integrations_tool, "_gate_bridge_login", lambda: False)
    assert build_integrations_tool(rig.context(integrations_client_factory=None)) is None


def test_the_tool_is_deferred_and_never_held_by_a_child() -> None:
    from local_operator.tools.deferral import DEFERRED_TOOL_PURPOSES, DEFERRED_TOOLS

    assert "integrations" in DEFERRED_TOOLS and "integrations" in DEFERRED_TOOL_PURPOSES


def test_rdi_credentials_are_masked_by_shape() -> None:
    from local_operator.redaction_shapes import _VENDOR_PATTERN

    for prefix in ("rdi_at_", "rdi_hk_", "rdi_whs_"):
        value = prefix + "Q3x9ZkP2mWv7LtR8nYb4"
        assert _VENDOR_PATTERN.search(f"token {value} here"), prefix
