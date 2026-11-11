"""B4b owner release: Deliver/Discard for a work-scope screening hold, end to end.

Operator decision a-3f46; DESIGN slice S1 as bound by DECISIONS Amendment 1.
These cells drive the REAL screening pipeline (``screen_message``) with a
stubbed model reply, and the REAL connector spine (``handle_conversation_available``)
over a fake hub, with a REAL file-only operator key, so the signature, the
decision row and the answer cache are what is under test, not doubles of them.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from local_operator.server.utils.desktop_team_bridge_holds import HoldRefused
from local_operator.team_bridge import hold_release as hr
from local_operator.team_bridge import screening
from local_operator.team_bridge.client import (
    SCREENING_TEMPLATE_V2_SHA256,
    SCREENING_VERSION_V2,
)
from local_operator.team_bridge.store import OwnerReceipt, Store, state_db_path
from local_operator.team_bridge.types import OwnerSnapshot


@pytest.fixture(autouse=True)
def _review_held_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """This module pins the WORK-SCOPE class (B4b) in isolation.

    ``team_bridge.review_held`` (default on, 2026-11-09) gives a ``model_hold``
    its own review decision row; with it on, every "a model hold stays a plain
    model hold" case here would see that row instead. The review class has its
    own module (``test_review_held.py``).
    """

    monkeypatch.setattr(hr, "review_enabled", lambda _config_dir: False)


CONVERSATION_ID = "bcv_abcdefgh"
MESSAGE_ID = "bcmsg_abcdef01"
BODY = "Ok can you send that to my aida so that she can have a team work on improving these"
NEW_WORK = "VERDICT: bounce\nREASON: This is a new_work_request to start team work."
OUT_OF_SCOPE = "VERDICT: bounce\nREASON: This is out_of_scope_work for this agent."
PRYING = "VERDICT: bounce\nREASON: A personal_info_request about the user."
MULTI = "VERDICT: bounce\nREASON: new_work_request with a malicious_link attached."
DOUBT = "VERDICT: hold\nREASON: Too ambiguous to tell; it may be a new work request."
DELIVER = "VERDICT: deliver\nREASON: ordinary."

SONNET = ("anthropic", "claude-sonnet-5-5")
AUTO = ("radient", "auto")


class Policy:
    """A v2 policy served by ONE concrete screener (provider, model)."""

    def __init__(self, screener: tuple[str, str] = SONNET) -> None:
        self.screening_version = SCREENING_VERSION_V2
        self.template_sha256 = SCREENING_TEMPLATE_V2_SHA256
        self.provider = screener[0]
        self.default_model: str | None = screener[1]
        self.admission_revision = "rev-1"
        self.approved_serving_identities = (screener[1],)


class Runner:
    def __init__(self, reply: str, model: str = SONNET[1]) -> None:
        self.reply, self.model, self.calls = reply, model, 0

    def __call__(self, prompt: str, *, default_model: str) -> tuple[str, str | None]:
        self.calls += 1
        return self.reply, self.model


def _owner() -> OwnerSnapshot:
    return OwnerSnapshot(
        credential_id=7,
        account_id="acc_recv",
        tenant_id="tnt_1",
        endpoint_id="bep_recv",
        canonical_base_url="https://api.radienthq.com/v1",
        enrollment_generation=1,
    )


def _setting(config_dir: Path, on: bool | None) -> None:
    values: dict[str, Any] = {"team_bridge": {"enabled": True}}
    if on is not None:
        values["team_bridge"]["hold_release"] = on
    (config_dir / "config.yml").write_text(json.dumps({"values": values}), encoding="utf-8")


def _item(
    body: str = BODY, message_id: str = MESSAGE_ID, sender: str = "acc_sender"
) -> dict[str, Any]:
    return {
        "message": {
            "id": message_id,
            "thread_id": CONVERSATION_ID,
            "tenant_id": "tnt_1",
            "type": "note",
            "body": {"text": body},
            "body_hash": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "recipient": {"account_id": "acc_recv", "endpoint_id": "bep_recv"},
        },
        "sender": {"account_id": sender, "name": "Victor"},
        "thread": {"id": CONVERSATION_ID, "tenant_id": "tnt_1"},
        "allocation": {"tier": 3, "sender_account_id": sender},
    }


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An enrolled owner, setting ON, and a real file-only operator key (no installed anchor)."""

    from local_operator.operator import trust
    from local_operator.operator.keychain import FILE_ONLY
    from local_operator.operator.sign import anchor_for_handle, create_key

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))

    def absent(uid: Any = None) -> Any:
        return trust.AnchorLoad(
            anchor=None,
            path=trust.anchor_path(uid),
            root_owned=False,
            reason="pinned absent by the test",
            exists=False,
        )

    monkeypatch.setattr(trust, "load_anchor", absent)
    handle = create_key(config_root=tmp_path, preference=FILE_ONLY)
    staged = trust.staging_path(tmp_path)
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(trust.anchor_bytes(anchor_for_handle(handle, label="b4b-test")))
    store = Store(tmp_path)
    owner = _owner()
    store.commit_enrollment(owner, OwnerReceipt("bridge_consent_v1", "ref-1"), cos_session_id="s1")
    _setting(tmp_path, True)
    return tmp_path, store, owner


def _trusted_runner(runner, policy):
    """Positive fixtures use the real chain and resolver-created preferred selection.

    A bare runner can no longer claim preference. Keep it as the scripted model
    callback, not the provenance producer; negative controls call the pipeline
    directly with an unwrapped bare runner.
    """

    from local_operator.model.configure import build_model_spec
    from local_operator.team_bridge.model_boundary import (
        DefaultSelection,
        ScreenChainRunner,
        _selection_for,
    )

    if type(runner) is not Runner:
        return runner
    hop = _selection_for(policy.provider, runner.model)
    if hop is None:
        hop = DefaultSelection(
            policy.provider, runner.model, build_model_spec(policy.provider, runner.model)
        )

    class Scripted:
        async def acall(self, prompt, *, default_model, **_kw):
            return runner(prompt, default_model=default_model)

    return ScreenChainRunner([hop], lambda _hop: Scripted())


def _screen(config_dir, owner, runner, *, item=None, policy=None):
    policy = policy or Policy()
    runner = _trusted_runner(runner, policy)
    return screening.screen_message(
        owner, item or _item(), config_dir=config_dir, policy=policy or Policy(), runner=runner
    )


def _count(config_dir: Path, table: str) -> int:
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        try:
            return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        except sqlite3.OperationalError:
            return 0


def _decide(config_dir, owner, word, *, body=BODY, message_id=MESSAGE_ID):
    from local_operator.server.utils import desktop_team_bridge_holds as holds

    # The hold row the connector writes is keyed by the composite feed ref; the
    # screener writes the bare id. Either is a conversation hold for the list.
    return holds.decide(
        config_dir,
        owner,
        message_id=message_id,
        conversation_id=CONVERSATION_ID,
        body_sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        word=word,
        account_id="acc_recv",
    )


# --- which decisions become releasable --------------------------------------


def test_a_v2_new_work_bounce_becomes_an_eligible_work_scope_hold(rig) -> None:
    config_dir, store, owner = rig
    runner = Runner(NEW_WORK)
    out = _screen(config_dir, owner, runner)
    assert out.disposition == "hold" and out.hold_code == hr.WORK_SCOPE_HOLD
    assert out.receipt is None and _count(config_dir, "screening_receipts") == 0
    decision = store.work_scope_decision(owner, MESSAGE_ID)
    assert decision is not None and decision["eligible"] == 1
    # The ANSWERING hop, the template and the REAL categories, at decision time (D4/D7).
    assert (decision["screener_provider"], decision["screener_model"]) == SONNET
    assert decision["template_sha256"] == SCREENING_TEMPLATE_V2_SHA256
    assert json.loads(decision["categories"]) == ["new_work_request"]
    assert decision["conversation_id"] == CONVERSATION_ID


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (OUT_OF_SCOPE, "bounce"),  # disclosure-class: never releasable (S-1)
        (PRYING, "bounce"),  # disclosure-class: never releasable (S-1)
        (MULTI, "bounce"),  # more than one category: closed (D1)
        (DOUBT, "hold"),  # no-category doubt: model_hold, no buttons
    ],
)
def test_every_other_decision_is_unchanged_and_never_releasable(rig, reply, expected) -> None:
    config_dir, store, owner = rig
    out = _screen(config_dir, owner, Runner(reply))
    assert out.disposition == expected
    assert out.hold_code != hr.WORK_SCOPE_HOLD
    assert store.work_scope_decision(owner, MESSAGE_ID) is None
    with pytest.raises(HoldRefused) as refused:
        _decide(config_dir, owner, "deliver")
    assert getattr(refused.value, "code", "") == "hold_not_releasable"


# rel-l01's committed v2 trial reasons (#3434, v4r1-relay): a HOLD that names
# the category token, and one that also names a second category.
REL_L01_HOLD = (
    'VERDICT: hold\nREASON: The note refers to "that" and "these" with no context, '
    "and it seems to ask for new team work to start, so I can't tell whether it is a "
    "misrouted message or a new_work_request."
)
HOLD_MULTI = "VERDICT: hold\nREASON: Maybe a new_work_request, maybe a personal_info_request."
OPUS = ("anthropic", "claude-opus-5-5")


@pytest.mark.parametrize("screener", [SONNET, OPUS])
@pytest.mark.parametrize("reply", [NEW_WORK, REL_L01_HOLD], ids=["v2-bounce", "v2-hold"])
def test_a_preferred_screeners_new_work_decision_is_releasable(rig, screener, reply) -> None:
    """Scope update 2026-11-04: v2 bounce AND v2 hold, exactly {new_work_request}."""

    config_dir, store, owner = rig
    out = _screen(config_dir, owner, Runner(reply, model=screener[1]), policy=Policy(screener))
    assert out.disposition == "hold" and out.hold_code == hr.WORK_SCOPE_HOLD
    decision = store.work_scope_decision(owner, MESSAGE_ID)
    assert decision is not None and decision["eligible"] == 1
    assert (decision["screener_provider"], decision["screener_model"]) == screener
    assert json.loads(decision["categories"]) == ["new_work_request"]
    assert _decide(config_dir, owner, "deliver")["state"] == "released"


def test_a_non_preferred_screeners_new_work_hold_stays_a_model_hold(rig) -> None:
    """D4 is the gate: a fallback hop's hold is not converted, and not releasable."""

    config_dir, store, owner = rig
    out = _screen(config_dir, owner, Runner(REL_L01_HOLD, model="auto"), policy=Policy(AUTO))
    assert out.hold_code == "model_hold"
    assert store.work_scope_decision(owner, MESSAGE_ID) is None
    with pytest.raises(HoldRefused) as refused:
        _decide(config_dir, owner, "deliver")
    assert refused.value.code == "hold_not_releasable"


def test_a_hold_naming_an_extra_category_is_not_releasable(rig) -> None:
    config_dir, store, owner = rig
    out = _screen(config_dir, owner, Runner(HOLD_MULTI))
    assert out.hold_code == "model_hold"
    assert store.work_scope_decision(owner, MESSAGE_ID) is None


def test_the_converted_hold_is_not_re_drawn(rig) -> None:
    """A v2 hold converted once is replayed: no second model call on the re-drive."""

    config_dir, _store, owner = rig
    _screen(config_dir, owner, Runner(REL_L01_HOLD))
    again = Runner(DELIVER)
    out = _screen(config_dir, owner, again)
    assert out.hold_code == hr.WORK_SCOPE_HOLD and again.calls == 0


def test_setting_off_keeps_the_v2_hold_a_model_hold(tmp_path) -> None:
    config_dir = tmp_path / "off"
    config_dir.mkdir()
    store = Store(config_dir)
    owner = _owner()
    store.commit_enrollment(owner, OwnerReceipt("bridge_consent_v1", "ref-1"))
    _setting(config_dir, False)
    out = _screen(config_dir, owner, Runner(REL_L01_HOLD))
    assert out.hold_code == "model_hold"
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "work_scope_decisions" not in names


def test_radient_auto_is_never_an_eligible_screener(rig) -> None:
    """Amendment 2 (code review m2): a non-preferred bounce stays the terminal reject."""

    config_dir, store, owner = rig
    out = _screen(config_dir, owner, Runner(NEW_WORK, model="auto"), policy=Policy(AUTO))
    assert out.disposition == "bounce" and out.receipt is not None
    assert out.receipt.verdict == "reject"
    assert store.work_scope_decision(owner, MESSAGE_ID) is None
    with pytest.raises(HoldRefused) as refused:
        _decide(config_dir, owner, "deliver")
    assert refused.value.code == "hold_not_releasable"


def test_eligibility_is_fixed_at_decision_time(rig) -> None:
    """Neither a later default-model change nor the session's chat model moves it (D4)."""

    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(NEW_WORK))
    before = store.work_scope_decision(owner, MESSAGE_ID)
    # The operator switches the screening default to radient/auto, and the CoS
    # session runs on another model entirely; the hold keeps its own decider.
    (config_dir / "config.yml").write_text(
        json.dumps(
            {
                "values": {
                    "team_bridge": {"enabled": True, "hold_release": True},
                    "model": {"provider": "radient", "model": "auto"},
                    "session": {"model": "deepseek/deepseek-v4.1-flash"},
                }
            }
        ),
        encoding="utf-8",
    )
    # The scheduler republishes for the new default (a new epoch), as it does
    # in production whenever the configured screener changes.
    current = store.current_screening_policy(owner)
    store.publish_screening_policy(
        owner,
        version=SCREENING_VERSION_V2,
        template_sha256=SCREENING_TEMPLATE_V2_SHA256,
        default_model=AUTO[1],
        revision="rev-1",
        expected_epoch=int(current["epoch"]),
    )
    again = _screen(config_dir, owner, Runner(DELIVER, model="auto"), policy=Policy(AUTO))
    assert again.hold_code == hr.WORK_SCOPE_HOLD  # replayed, not re-drawn
    assert store.work_scope_decision(owner, MESSAGE_ID) == before


def test_the_converted_hold_makes_no_second_screening_attempt(rig) -> None:
    """D5 / S-4: a re-drive re-reads the decision; it never re-rolls toward deliver."""

    config_dir, store, owner = rig
    first = Runner(NEW_WORK)
    _screen(config_dir, owner, first)
    attempts = _count(config_dir, "screening_attempts")
    second = Runner(DELIVER)
    out = _screen(config_dir, owner, second)
    assert out.hold_code == hr.WORK_SCOPE_HOLD and second.calls == 0
    assert _count(config_dir, "screening_attempts") == attempts
    # And the SAME body from a fresh message id is the answer cache's hit.
    copy = Runner(DELIVER)
    out = _screen(config_dir, owner, copy, item=_item(message_id="bcmsg_abcdef02"))
    assert out.hold_code == hr.WORK_SCOPE_HOLD and copy.calls == 0
    # ... and it is ACTIONABLE (code review M2 / QA Q1): its own row, eligible.
    copied = store.work_scope_decision(owner, "bcmsg_abcdef02")
    assert copied is not None and copied["eligible"] == 1
    assert _decide(config_dir, owner, "deliver", message_id="bcmsg_abcdef02")["state"] == "released"


def test_the_per_sender_budget_closes_the_fourth_hold(rig) -> None:
    config_dir, store, owner = rig
    for index in range(hr.SENDER_BUDGET + 1):
        body = f"please start a new project number {index}"
        _screen(config_dir, owner, Runner(NEW_WORK), item=_item(body, f"bcmsg_budget{index:02d}"))
    rows = store.work_scope_decisions(owner)
    assert [r["eligible"] for r in rows] == [1, 1, 1, 0]
    assert rows[-1]["ineligible_reason"] == hr.INELIGIBLE_BUDGET


def test_setting_off_is_byte_identical_to_today(tmp_path) -> None:
    """The flag-off cell: same outcome, same receipt, no new table touched."""

    def run(config_dir: Path, setting: bool | None) -> tuple[Any, list[Any]]:
        config_dir.mkdir()
        store = Store(config_dir)
        owner = _owner()
        store.commit_enrollment(owner, OwnerReceipt("bridge_consent_v1", "ref-1"))
        _setting(config_dir, setting)
        out = _screen(config_dir, owner, Runner(NEW_WORK))
        with sqlite3.connect(state_db_path(config_dir)) as conn:
            tables = sorted(
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            )
            receipts = conn.execute(
                "SELECT verdict, model, prompt_version, template_sha256, categories "
                "FROM screening_receipts"
            ).fetchall()
        return (out.disposition, out.hold_code, out.receipt), tables + receipts

    absent = run(tmp_path / "absent", None)
    off = run(tmp_path / "off", False)
    assert absent == off
    (disposition, code, receipt), rows = off
    assert (disposition, code) == ("bounce", None)
    assert receipt.verdict == "reject" and receipt.categories == ("new_work_request",)
    assert "work_scope_decisions" not in rows and "owner_releases" not in rows


# --- the release record ------------------------------------------------------


def test_a_release_with_a_bad_signature_is_refused_and_the_hold_stands(rig) -> None:
    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(NEW_WORK))
    _decide(config_dir, owner, "deliver")
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        sig = conn.execute("SELECT signature FROM owner_releases").fetchone()[0]
        flipped = sig[:-2] + ("00" if sig[-2:] != "00" else "11")
        conn.execute("UPDATE owner_releases SET signature = ?", (flipped,))
    out = _screen(config_dir, owner, Runner(DELIVER))
    assert out.disposition == "hold" and out.hold_code == hr.WORK_SCOPE_HOLD
    assert _count(config_dir, "screening_receipts") == 0
    assert store.owner_release(owner, MESSAGE_ID)["outcome"] == "refused:release_unverified"


def test_a_release_for_a_changed_decision_is_refused(rig) -> None:
    """S-7: a valid signature cannot release a hold whose recorded decision moved."""

    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(NEW_WORK))
    _decide(config_dir, owner, "deliver")
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        conn.execute("UPDATE work_scope_decisions SET screener_model = 'auto'")
    out = _screen(config_dir, owner, Runner(DELIVER))
    assert out.disposition == "hold" and _count(config_dir, "screening_receipts") == 0
    assert store.owner_release(owner, MESSAGE_ID)["outcome"] == "refused:decision_changed"


def test_deliver_never_becomes_a_cached_allow(rig) -> None:
    """DESIGN §5: the release receipt's template is no cache key; a repeat is held again."""

    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(NEW_WORK))
    _decide(config_dir, owner, "deliver")
    released = _screen(config_dir, owner, Runner(DELIVER))
    assert released.disposition == "deliver" and released.receipt is not None
    assert released.receipt.template_sha256 == hr.OWNER_RELEASE_SHA256
    assert released.receipt.model == hr.OWNER_RELEASE_MODEL
    assert released.attachments_hash is None
    repeat = Runner(DELIVER)
    again = _screen(config_dir, owner, repeat, item=_item(message_id="bcmsg_abcdef03"))
    assert again.hold_code == hr.WORK_SCOPE_HOLD and repeat.calls == 0
    # Held again AND the person's decision again: its own releasable row.
    repeat_row = store.work_scope_decision(owner, "bcmsg_abcdef03")
    assert repeat_row is not None and repeat_row["eligible"] == 1


def test_the_owner_release_receipt_needs_a_verified_release_to_admit(rig) -> None:
    """DESIGN §9.4: without a verified row an owner-release receipt is proof_corrupt."""

    from local_operator.team_bridge.store import RECEIPT_CORRUPT, receipt_authority

    config_dir, store, owner = rig
    policy = store.current_screening_policy(owner) or {"revision": "rev-1", "epoch": 1}
    receipt = {
        "verdict": "allow",
        "model": hr.OWNER_RELEASE_MODEL,
        "prompt_version": hr.OWNER_RELEASE_PROMPT_VERSION,
        "template_sha256": hr.OWNER_RELEASE_SHA256,
        "body_sha256": "a" * 64,
        "envelope_digest": "b" * 64,
        "categories": "[]",
        "admission_revision": policy["revision"],
        "policy_epoch": policy["epoch"],
        "creator_generation": 1,
    }
    for release in (None, {"verified": False, "word": "deliver"}):
        decoded, damage, _ = receipt_authority(owner, policy, receipt, release=release)
        assert decoded is None and damage == RECEIPT_CORRUPT
    good = {
        "verified": True,
        "word": "deliver",
        "body_sha256": "a" * 64,
        "envelope_digest": "b" * 64,
    }
    decoded, damage, _ = receipt_authority(owner, policy, receipt, release=good)
    assert damage is None and decoded is not None and decoded["verdict"] == "allow"


def test_the_release_signature_uses_its_own_domain() -> None:
    from local_operator.operator import verify

    assert verify.RELEASE_DOMAIN not in (
        verify.DOMAIN,
        verify.DEVICE_CERT_DOMAIN,
        verify.APPROVAL_DOMAIN,
    )
    assert "release" not in verify.ACTIONS
    payload = hr.release_payload(
        owner_key="k", generation=1, conversation_id="c", message_id="m",
        body_sha256="b", envelope_digest="e", hold_code="h", screener_provider="p",
        screener_model="s", template_sha256="t", categories=["new_work_request"],
        screener_verdict="hold", screener_hop=0, screener_preferred=True,
        word="deliver", decided_at=1.0,
    )  # fmt: skip
    assert payload.startswith(verify.RELEASE_DOMAIN) and b"e" in payload
    # The hop, the verdict and the preference are signed: changing any one
    # changes the bytes (Amendment 2, S-1 / m3).
    for change in (
        {"screener_hop": 1},
        {"screener_verdict": "bounce"},
        {"screener_preferred": False},
    ):
        fields: dict[str, Any] = dict(
            owner_key="k", generation=1, conversation_id="c", message_id="m",
            body_sha256="b", envelope_digest="e", hold_code="h", screener_provider="p",
            screener_model="s", template_sha256="t", categories=["new_work_request"],
            screener_verdict="hold", screener_hop=0, screener_preferred=True,
            word="deliver", decided_at=1.0,
        )  # fmt: skip
        fields.update(change)
        assert hr.release_payload(**fields) != payload


def test_discard_records_the_real_categories_and_deliver_after_discard_is_refused(rig) -> None:
    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(NEW_WORK))
    first = _decide(config_dir, owner, "discard")
    assert first["state"] == "discarded"
    assert _decide(config_dir, owner, "discard") == first  # idempotent
    with pytest.raises(HoldRefused) as refused:
        _decide(config_dir, owner, "deliver")
    assert refused.value.code == "already_decided"
    out = _screen(config_dir, owner, Runner(DELIVER))
    assert out.receipt is not None
    assert out.disposition == "bounce" and out.receipt.verdict == "reject"
    assert out.receipt.categories == ("new_work_request",)


def test_deliver_refuses_bytes_the_person_did_not_see(rig) -> None:
    config_dir, _store, owner = rig
    _screen(config_dir, owner, Runner(NEW_WORK))
    with pytest.raises(HoldRefused) as refused:
        _decide(config_dir, owner, "deliver", body="a different body")
    assert refused.value.code == "snapshot_stale"


def test_setting_off_refuses_every_write(rig) -> None:
    config_dir, _store, owner = rig
    _screen(config_dir, owner, Runner(NEW_WORK))
    _setting(config_dir, False)
    with pytest.raises(HoldRefused) as refused:
        _decide(config_dir, owner, "deliver")
    assert refused.value.code == "hold_release_off"


# --- the CoS has no release path ---------------------------------------------


@pytest.mark.parametrize("op", ["release", "deliver", "discard"])
def test_the_bridge_tool_has_no_release_op(op) -> None:
    from local_operator.team_bridge import tool

    assert op not in tool.BRIDGE_OPS
    runtime = cast(Any, SimpleNamespace(owner=None))
    out = asyncio.run(tool.execute_bridge(runtime, {"op": op}))
    assert out.get("code") == "not_permitted"


def test_the_notify_tool_cannot_write_a_bridge_key() -> None:
    from local_operator.tools import builtin

    async def door(**kwargs: Any) -> str:
        raise AssertionError("a reserved key reached the inbox")

    result = asyncio.run(
        cast(Any, builtin.execute_notify)(
            "call-1",
            {"title": "Deliver this held message?", "key": "team-bridge-held:bep:bcmsg"},
            context=SimpleNamespace(notify_inbox=door),
        )
    )
    text = json.dumps(getattr(result, "content", str(result)), default=str)
    assert "reserved" in text


# --- the connector spine, with the REAL screening pipeline -------------------
#
# The only doubles are the hub (``FakeWire``) and the model reply (``Runner``):
# ``screen_item`` is replaced by a call into the real ``screen_message_async``,
# so the conversion, the decision row, the notice, the release consume, the
# receipt, the reservation and the card all run as they do in production.


def _spine(monkeypatch, config_dir, owner, runner, *, body=BODY):
    from local_operator.team_bridge.connector import conversation as conv

    runner = _trusted_runner(runner, Policy())
    from local_operator.team_bridge.types import ConversationFeedRef
    from tests.unit.team_bridge import test_conversation_delivery as tcd

    server = tcd.FakeServer(
        notify={owner.endpoint_id: "all"},
        active={owner.endpoint_id: True},
        body=body,
        sender={"account_id": "acc_sender", "kind": "human", "display_label": "Victor"},
    )
    wakes: list[str] = []

    async def wire_for(_owner, _config_dir, _auth):
        return tcd.FakeWire(server, owner.endpoint_id)

    async def screen_item(o, item, *, config_dir, **_kw):
        item = dict(item)
        item["allocation"] = {"tier": 3, "sender_account_id": "acc_sender"}
        return await screening.screen_message_async(
            o, item, config_dir=config_dir, policy=Policy(), runner=runner
        )

    async def deliver(o, payload, *, session_id, config_dir, **_kw):
        wakes.append(payload)
        return "delivered and woke the session"

    monkeypatch.setattr(conv, "_wire_for", wire_for)
    monkeypatch.setattr(conv, "screen_item", screen_item)
    monkeypatch.setattr(conv, "deliver_wake", deliver)
    monkeypatch.setattr(conv, "engagement_hold", lambda designation, config_dir: "")
    from local_operator.team_bridge.connector import coalesce

    monkeypatch.setattr(coalesce, "CONNECTOR_COALESCER", coalesce.WakeCoalescer(window_s=0.0))
    ref = ConversationFeedRef(
        conversation_id=CONVERSATION_ID, message_id=MESSAGE_ID, conversation_seq=1, mention=False
    ).validated()

    def run() -> dict[str, Any]:
        async def go():
            out = await conv.handle_conversation_available(
                owner,
                ref,
                config_dir=config_dir,
                auth_store=None,
                access=SimpleNamespace(credential_id=owner.credential_id, kind="oauth"),
            )
            await coalesce.CONNECTOR_COALESCER.flush_due(
                deliver=conv.deliver_wake, now=10**12, config_dir=config_dir
            )
            return out

        return asyncio.run(go())

    return server, wakes, run


def _notices(config_dir: Path) -> list[dict[str, Any]]:
    from local_operator.notifications.store import NotificationStore

    return NotificationStore(config_dir).find_by_key(hr.notice_key("bep_recv", MESSAGE_ID))


def test_spine_hold_then_deliver_admits_and_wakes_the_cos(rig, monkeypatch) -> None:
    config_dir, store, owner = rig
    server, wakes, run = _spine(monkeypatch, config_dir, owner, Runner(NEW_WORK))
    held = run()
    assert held["action"] == "hold" and held["code"] == hr.WORK_SCOPE_HOLD
    assert server.admits == [] and wakes == []
    notices = _notices(config_dir)
    assert len(notices) == 1
    shown = json.dumps(notices[0])
    # No body, sender or place in the item: it can reach a lock screen (S-10).
    assert "aida" not in shown and "Victor" not in shown and "qa-coordination" not in shown
    assert store.work_scope_decision(owner, MESSAGE_ID)["notified_at"] is not None

    _decide(config_dir, owner, "deliver")
    delivered = run()
    assert delivered["action"] == "delivered"
    assert server.admits == [{"endpoint": "bep_recv", "message_id": MESSAGE_ID, "verdict": "allow"}]
    assert len(wakes) == 1
    from local_operator.team_bridge.connector.conversation import (
        OWNER_RELEASED_CARD_LINE,
    )

    assert OWNER_RELEASED_CARD_LINE in wakes[0]
    assert "released by your owner" not in wakes[0]  # D6: provenance, not endorsement
    assert store.owner_release(owner, MESSAGE_ID)["outcome"] == "admitted"
    assert _notices(config_dir)[0]["priority"] == "low"


def test_spine_discard_rejects_with_no_wake(rig, monkeypatch) -> None:
    config_dir, store, owner = rig
    server, wakes, run = _spine(monkeypatch, config_dir, owner, Runner(NEW_WORK))
    assert run()["code"] == hr.WORK_SCOPE_HOLD
    _decide(config_dir, owner, "discard")
    out = run()
    assert out["action"] == "rejected"
    assert server.admits == [
        {"endpoint": "bep_recv", "message_id": MESSAGE_ID, "verdict": "reject"}
    ]
    assert wakes == []
    assert store.owner_release(owner, MESSAGE_ID)["outcome"] == "rejected"


def test_a_refused_notice_keeps_the_hold_in_the_long_hold_count(rig, monkeypatch) -> None:
    from local_operator.notifications import store as nstore

    config_dir, store, owner = rig

    def refuse(self, *a, **k):
        raise nstore.NotifyRefused("rate_limited", "refused")

    monkeypatch.setattr(nstore.NotificationStore, "notify", refuse)
    _server, _wakes, run = _spine(monkeypatch, config_dir, owner, Runner(NEW_WORK))
    assert run()["code"] == hr.WORK_SCOPE_HOLD
    assert store.work_scope_decision(owner, MESSAGE_ID)["notified_at"] is None
    assert store.long_holds_readonly(owner, older_than_s=0.0, now=10**12)["count"] == 1


# --- the desktop routes, over HTTP -------------------------------------------


@pytest.fixture
def api(rig, monkeypatch):
    """The real router on an ASGI app, the real desktop bearer gate, a fake hub client."""

    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from local_operator.config import ConfigManager
    from local_operator.server.routes import desktop_team_bridge
    from local_operator.team_bridge import client as tb_client

    config_dir, store, owner = rig
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "test-token")

    class Client:
        def __init__(self, *_a: Any, **_k: Any) -> None:
            pass

        async def current_access(self) -> Any:
            return SimpleNamespace(account_id="acc_recv", kind="oauth", credential_id=7)

    monkeypatch.setattr(tb_client, "TeamBridgeClient", Client)
    app = FastAPI()
    app.state.config_manager = ConfigManager(config_dir)
    app.include_router(desktop_team_bridge.router)

    def call(method: str, path: str, *, auth: bool = True, **kwargs: Any) -> Any:
        async def go():
            headers = {"Authorization": "Bearer test-token"} if auth else {}
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://localhost", headers=headers
            ) as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(go())

    return config_dir, store, owner, call


def _body(**overrides: Any) -> dict[str, Any]:
    body = {
        "expected_account_id": "acc_recv",
        "conversation_id": CONVERSATION_ID,
        "body_sha256": hashlib.sha256(BODY.encode("utf-8")).hexdigest(),
    }
    body.update(overrides)
    return body


def test_routes_list_deliver_and_their_negatives(api) -> None:
    config_dir, _store, owner, call = api
    _screen(config_dir, owner, Runner(NEW_WORK))
    listed = call(
        "GET", "/v1/desktop/team-bridge/holds", params={"expected_account_id": "acc_recv"}
    )
    assert listed.status_code == 200
    data = listed.json()["result"]
    assert data["enabled"] is True and data["key_protection"] == "file_only"
    (item,) = data["items"]
    assert item["kind"] == "owner_decision" and item["releasable"] is True
    assert "body" not in json.dumps(item) and "new_work_request" not in json.dumps(item)

    path = f"/v1/desktop/team-bridge/holds/{MESSAGE_ID}/deliver"
    assert call("POST", path, json=_body(), auth=False).status_code == 401
    wrong = call("POST", path, json=_body(expected_account_id="acc_other"))
    assert wrong.status_code == 409 and wrong.json()["detail"]["code"] == "account_mismatch"
    stale = call("POST", path, json=_body(body_sha256="0" * 64))
    assert stale.status_code == 409 and stale.json()["detail"]["code"] == "snapshot_stale"
    bad = call("POST", path, json={**_body(), "extra": 1})
    assert bad.status_code == 422

    ok = call("POST", path, json=_body())
    assert ok.status_code == 200 and ok.json()["result"]["state"] == "released"
    assert call("POST", path, json=_body()).json()["result"] == ok.json()["result"]
    again = call("POST", f"/v1/desktop/team-bridge/holds/{MESSAGE_ID}/discard", json=_body())
    assert again.status_code == 409 and again.json()["detail"]["code"] == "already_decided"


@pytest.mark.parametrize("reply", [MULTI, DOUBT, OUT_OF_SCOPE])
def test_routes_refuse_a_non_releasable_hold(api, reply) -> None:
    config_dir, _store, owner, call = api
    _screen(config_dir, owner, Runner(reply))
    out = call("POST", f"/v1/desktop/team-bridge/holds/{MESSAGE_ID}/deliver", json=_body())
    assert out.status_code == 409 and out.json()["detail"]["code"] == "hold_not_releasable"


def test_routes_refuse_an_infrastructure_hold(api) -> None:
    config_dir, store, owner, call = api
    store.persist_hold(owner, MESSAGE_ID, "runner_unavailable")
    listed = call(
        "GET", "/v1/desktop/team-bridge/holds", params={"expected_account_id": "acc_recv"}
    )
    (item,) = listed.json()["result"]["items"]
    assert item["kind"] == "waiting" and item["releasable"] is False
    out = call("POST", f"/v1/desktop/team-bridge/holds/{MESSAGE_ID}/deliver", json=_body())
    assert out.status_code == 409 and out.json()["detail"]["code"] == "hold_not_releasable"


def test_routes_with_the_setting_off(api) -> None:
    config_dir, _store, owner, call = api
    _screen(config_dir, owner, Runner(NEW_WORK))
    _setting(config_dir, False)
    listed = call(
        "GET", "/v1/desktop/team-bridge/holds", params={"expected_account_id": "acc_recv"}
    )
    assert listed.json()["result"]["enabled"] is False
    assert all(not i["releasable"] for i in listed.json()["result"]["items"])
    out = call("POST", f"/v1/desktop/team-bridge/holds/{MESSAGE_ID}/deliver", json=_body())
    assert out.status_code == 409 and out.json()["detail"]["code"] == "hold_release_off"


# --- Amendment 2: trusted preferred-stage provenance (#3444 integration) ----


class _Hop:
    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome

    async def acall(self, prompt: str, *, default_model: str, **_: Any) -> Any:
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


class _Advance(Exception):
    """A pre-wire failure the chain is allowed to advance past."""


def _chain(primary, fallback, reply, *, fail_primary):
    from local_operator.model.configure import build_model_spec
    from local_operator.team_bridge.model_boundary import (
        DefaultSelection,
        ScreenChainRunner,
        _selection_for,
    )

    hops = [
        _selection_for(*primary)
        or DefaultSelection(primary[0], primary[1], build_model_spec(*primary)),
        DefaultSelection(fallback[0], fallback[1], build_model_spec(*fallback)),
    ]

    def build(hop):
        if hop.model_id == primary[1] and fail_primary:
            failure = _Advance("pre-wire")
            setattr(failure, "tb_screen_advance", True)
            return _Hop(failure)
        return _Hop((reply, hop.model_id))

    return ScreenChainRunner(hops, build)


class _ChainPolicy(Policy):
    def __init__(self, primary, fallback) -> None:
        super().__init__(primary)
        self.approved_serving_identities = (primary[1], fallback[1])


@pytest.mark.parametrize("reply", [NEW_WORK, REL_L01_HOLD], ids=["v2-bounce", "v2-hold"])
@pytest.mark.parametrize(
    ("primary", "fallback"),
    [(AUTO, OPUS), (SONNET, OPUS)],
    ids=["auto->opus", "sonnet->opus"],
)
def test_a_fallback_hops_answer_is_never_releasable(rig, primary, fallback, reply) -> None:
    """Ordinary cascade Opus is not preferred, even after a preferred Sonnet hop."""

    config_dir, store, owner = rig
    runner = _chain(primary, fallback, reply, fail_primary=True)
    out = _screen(config_dir, owner, runner, policy=_ChainPolicy(primary, fallback))
    assert runner.served_hop_index == 1
    assert out.hold_code != hr.WORK_SCOPE_HOLD
    if reply == NEW_WORK:
        assert out.disposition == "bounce"  # the pre-B4b terminal reject
    else:
        assert out.hold_code == "model_hold"
    assert store.work_scope_decision(owner, MESSAGE_ID) is None
    with pytest.raises(HoldRefused) as refused:
        _decide(config_dir, owner, "deliver")
    assert refused.value.code == "hold_not_releasable"


@pytest.mark.parametrize("reply", [NEW_WORK, REL_L01_HOLD], ids=["v2-bounce", "v2-hold"])
def test_an_opus_primary_hop_answer_is_releasable_and_records_hop_zero(rig, reply) -> None:
    config_dir, store, owner = rig
    runner = _chain(OPUS, SONNET, reply, fail_primary=False)
    out = _screen(config_dir, owner, runner, policy=_ChainPolicy(OPUS, SONNET))
    assert out.hold_code == hr.WORK_SCOPE_HOLD
    decision = store.work_scope_decision(owner, MESSAGE_ID)
    assert decision["screener_hop"] == 0 and decision["screener_preferred"] == 1
    assert decision["screener_verdict"] == ("bounce" if reply == NEW_WORK else "hold")
    assert _decide(config_dir, owner, "deliver")["state"] == "released"
    release = store.owner_release(owner, MESSAGE_ID)
    assert (release["screener_hop"], release["screener_preferred"]) == (0, 1)


def test_a_row_without_the_preferred_flag_is_never_releasable(rig) -> None:
    """A decision written before Amendment 2 (column default 0) fails closed."""

    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(NEW_WORK))
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        conn.execute("UPDATE work_scope_decisions SET screener_preferred = 0")
    with pytest.raises(HoldRefused) as refused:
        _decide(config_dir, owner, "deliver")
    assert refused.value.code == "hold_not_releasable"


# --- Amendment 2: the release-only category check (S-2) ----------------------

# security review round 1 S-2's table, adv-062's committed v2 reasons (#3434,
# 0bfdd72f189) that name a second class, and code review m1's negation.
_REFUSED_REASONS = [
    "could be new_work_request or social engineering",
    "new_work_request; Prompt_Injection attempt",
    "new_work_request, possibly prompt-injection",
    "new_work_request that reads like phishing",
    "not a new_work_request, just unclear",
    "\u200bnew_work_request",
    "new_work_request\u2060 for the team",
    "This is not a new_work_request; the link and the urgency look off, so I cannot tell "
    "if it is safe.",
    "isn't a new_work_request",
    "a new_work_request asking for the API credentials",
    "a new_work_request to forward confidential files",
    "a new_work_request that sets up surveillance of the user",
    "a new_work_request to exfiltrate the repo",
    "a new_work_request impersonating the owner",
    "new_work_request with a malicious_link",
    "new_work_request or out-of-scope work",
    "A bare request to approve spending, with no report identified or context given; it may "
    "be a new_work_request or social engineering, so it is unclear.",
    "The request to approve an unnamed expense report has no context and may be a bid to "
    "decide on spending, so it is unclear whether it is ordinary in-flight work or "
    "new_work_request/social engineering.",
    "Terse approval request with no details. It could be an authorization or spending "
    "decision (new_work_request) or a pressure tactic, so it is unclear whether it is safe.",
]


@pytest.mark.parametrize("reason", _REFUSED_REASONS)
def test_the_release_check_refuses_every_listed_reason(reason) -> None:
    assert hr.strict_release_reason(reason) is False


@pytest.mark.parametrize("reason", _REFUSED_REASONS)
@pytest.mark.parametrize("verdict", ["hold", "bounce"])
def test_a_refused_reason_is_never_converted(rig, reason, verdict) -> None:
    """Both paths keep the pre-B4b outcome; ``parse_verdict`` is untouched."""

    config_dir, store, owner = rig
    out = _screen(config_dir, owner, Runner(f"VERDICT: {verdict}\nREASON: {reason}"))
    assert out.hold_code != hr.WORK_SCOPE_HOLD
    assert store.work_scope_decision(owner, MESSAGE_ID) is None


@pytest.mark.parametrize(
    "reason",
    [
        REL_L01_HOLD.split("REASON: ", 1)[1],
        NEW_WORK.split("REASON: ", 1)[1],
        "A bare request to approve an unnamed expense report could be a spending decision "
        "(new_work_request) or routine work in flight; the context is too thin to tell.",
    ],
)
def test_the_release_check_keeps_honest_new_work_reasons(reason) -> None:
    assert hr.strict_release_reason(reason) is True


# --- setting OFF after a conversion (code review M1 / security S-3) ----------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [(NEW_WORK, ("bounce", None)), (REL_L01_HOLD, ("hold", "model_hold"))],
    ids=["bounce-origin", "hold-origin"],
)
def test_turning_the_setting_off_settles_with_no_model_call(rig, reply, expected) -> None:
    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(reply))
    _setting(config_dir, False)
    later = Runner(DELIVER)
    out = _screen(config_dir, owner, later)
    assert later.calls == 0
    assert (out.disposition, out.hold_code) == expected
    if expected[0] == "bounce":
        assert out.receipt is not None
        assert out.receipt.verdict == "reject"
        assert list(out.receipt.categories) == ["new_work_request"]
    again = Runner(DELIVER)
    assert _screen(config_dir, owner, again).disposition == expected[0] and again.calls == 0


@pytest.mark.parametrize("reply", [NEW_WORK, REL_L01_HOLD], ids=["bounce-origin", "hold-origin"])
def test_a_signed_discard_is_honoured_after_the_setting_is_turned_off(rig, reply) -> None:
    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(reply))
    _decide(config_dir, owner, "discard")
    _setting(config_dir, False)
    later = Runner(DELIVER)
    out = _screen(config_dir, owner, later)
    assert later.calls == 0
    assert out.disposition == "bounce" and out.receipt is not None
    assert out.receipt.template_sha256 == hr.OWNER_RELEASE_SHA256
    again = Runner(DELIVER)
    assert _screen(config_dir, owner, again).disposition == "bounce" and again.calls == 0


def test_a_pending_deliver_is_refused_once_the_setting_is_off(rig) -> None:
    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(REL_L01_HOLD))
    _decide(config_dir, owner, "deliver")
    _setting(config_dir, False)
    later = Runner(DELIVER)
    out = _screen(config_dir, owner, later)
    assert later.calls == 0 and out.hold_code == "model_hold"
    assert store.owner_release(owner, MESSAGE_ID)["outcome"] == "refused:hold_release_off"


def test_identical_resends_share_the_sender_budget(rig) -> None:
    """M2/Q1: a copy costs budget like a fresh decision, so a resend loop cannot multiply."""

    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(NEW_WORK))
    for index in range(2, 6):
        _screen(config_dir, owner, Runner(DELIVER), item=_item(message_id=f"bcmsg_abcdef0{index}"))
    rows = store.work_scope_decisions(owner)
    assert [r["eligible"] for r in rows] == [1, 1, 1, 0, 0]
    assert all(r["screener_hop"] == 0 and r["screener_verdict"] == "bounce" for r in rows)


def test_the_connector_renders_notices_in_the_operator_language(tmp_path) -> None:
    """QA Q2: the daemon reads ``language`` from the config dir it serves."""

    from local_operator.i18n.text import active_locale, set_active_locale
    from local_operator.team_bridge.connector.__main__ import _activate_operator_locale

    (tmp_path / "config.yml").write_text(
        json.dumps({"values": {"language": "fr"}}), encoding="utf-8"
    )
    try:
        _activate_operator_locale(tmp_path)
        assert active_locale() == "fr"
    finally:
        set_active_locale("en")


@pytest.mark.parametrize("reply", [NEW_WORK, REL_L01_HOLD], ids=["bounce", "hold"])
def test_wire_sonnet_429_then_preferred_opus_can_release(rig, monkeypatch, reply) -> None:
    """Real resolver -> scheduler -> wire guard -> chain -> signed owner decision."""

    from tests.unit.team_bridge.test_screening_chain import _install_wire, _Wire

    config_dir, store, owner = rig
    monkeypatch.setattr(
        "local_operator.providers.model_access.credentialed_chat_providers_here",
        lambda **_kw: {"anthropic"},
    )
    wire = _Wire({"claude-sonnet-5-5": ("fail", 429), "claude-opus-5-5": ("reply", reply)})
    _install_wire(monkeypatch, wire)
    out = _scheduler_screen(config_dir, owner)
    assert out.hold_code == hr.WORK_SCOPE_HOLD
    assert [(r["provider"], r["model"]) for r in wire.sends] == [SONNET, OPUS]
    decision = store.work_scope_decision(owner, MESSAGE_ID)
    assert decision is not None and decision["eligible"] == 1
    assert (
        decision["screener_model"],
        decision["screener_hop"],
        decision["screener_preferred"],
    ) == (OPUS[1], 1, 1)
    assert _decide(config_dir, owner, "deliver")["state"] == "released"
    release = store.owner_release(owner, MESSAGE_ID)
    assert hr.verify_row(release, hr.anchor_spki(config_dir)[0])
    assert release["screener_hop"] == 1 and release["screener_preferred"] == 1
    # Decision-time finality: after source preference changes there is no redraw.
    before = len(wire.sends)
    again = _scheduler_screen(config_dir, owner)
    assert again.disposition == "deliver" and len(wire.sends) == before


def _scheduler_screen(config_dir, owner):
    """Run the shipped cycle with a stubbed hub status and the real vendor wire seam."""

    from local_operator.team_bridge import scheduler
    from tests.unit.team_bridge.test_screening_chain import _consent_status

    raw = _consent_status(owner, {"version": "bridge_consent_v1", "text": "v1 consent"})
    from local_operator.team_bridge.client import SCREENING_TEMPLATE_V2

    raw["screening"] = {"version": SCREENING_VERSION_V2, "template": SCREENING_TEMPLATE_V2}
    verified = scheduler.VerifiedStatus(scheduler._VERIFIED_TOKEN, raw)

    async def fetch():
        return verified

    return asyncio.run(
        scheduler.run_screening_cycle(
            config_dir=config_dir,
            owner=owner,
            items=[_item()],
            auth_store=None,
            status_fetcher=fetch,
        )
    )


@pytest.mark.parametrize("reply", [NEW_WORK, REL_L01_HOLD], ids=["bounce", "hold"])
def test_preferred_stage_exhausted_then_ordinary_opus_is_not_releasable(rig, monkeypatch, reply):
    """Both Radient preferred hops exhaust; direct Opus in stage 2 remains ordinary."""

    from tests.unit.team_bridge.test_screening_chain import _install_wire, _Wire

    config_dir, store, owner = rig
    values = json.loads((config_dir / "config.yml").read_text())
    values["values"]["hosting"] = "anthropic"
    values["values"]["model_name"] = OPUS[1]
    (config_dir / "config.yml").write_text(json.dumps(values))
    monkeypatch.setattr(
        "local_operator.providers.model_access.credentialed_chat_providers_here",
        lambda **_kw: {"radient"},
    )
    wire = _Wire(
        {
            "anthropic/claude-sonnet-5.5": ("fail", 429),
            "anthropic/claude-opus-5.5": ("fail", 429),
            "claude-opus-5-5": ("reply", reply),
        }
    )
    _install_wire(monkeypatch, wire)
    out = _scheduler_screen(config_dir, owner)
    assert [(r["provider"], r["model"]) for r in wire.sends] == [
        ("radient", "anthropic/claude-sonnet-5.5"),
        ("radient", "anthropic/claude-opus-5.5"),
        OPUS,
    ]
    assert out.hold_code != hr.WORK_SCOPE_HOLD
    assert store.work_scope_decision(owner, MESSAGE_ID) is None
    assert not out.screener_preferred
    with pytest.raises(HoldRefused) as failure:
        _decide(config_dir, owner, "deliver")
    assert failure.value.code == "hold_not_releasable"


@pytest.mark.parametrize("reply", [NEW_WORK, REL_L01_HOLD], ids=["bounce", "hold"])
@pytest.mark.parametrize("claim", [None, True], ids=["legacy-missing", "spoofed-flag"])
def test_a_bare_runner_cannot_spoof_release_provenance(rig, reply, claim):
    config_dir, store, owner = rig

    class Spoof:
        answered_preferred = claim
        screener_preferred = claim
        served_provider = "anthropic"
        served_hop_index = 0
        answered_hop = SimpleNamespace(provider="anthropic", model_id=SONNET[1], preferred=True)

        def __call__(self, prompt, *, default_model):
            return reply, SONNET[1]

    out = screening.screen_message(
        owner, _item(), config_dir=config_dir, policy=Policy(), runner=Spoof()
    )
    assert out.hold_code != hr.WORK_SCOPE_HOLD
    assert store.work_scope_decision(owner, MESSAGE_ID) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("screener_preferred", 0),
        ("screener_hop", 8),
        ("screener_model", "auto"),
    ],
)
def test_tampered_signed_provenance_is_refused(rig, field, value):
    config_dir, store, owner = rig
    _screen(config_dir, owner, Runner(NEW_WORK))
    _decide(config_dir, owner, "deliver")
    with sqlite3.connect(state_db_path(config_dir)) as conn:
        conn.execute(f"UPDATE owner_releases SET {field} = ?", (value,))
    again = Runner(DELIVER)
    out = _screen(config_dir, owner, again)
    assert out.hold_code == hr.WORK_SCOPE_HOLD and again.calls == 0
    assert store.owner_release(owner, MESSAGE_ID)["outcome"] == "refused:release_unverified"


@pytest.mark.parametrize("reply", [NEW_WORK, REL_L01_HOLD], ids=["bounce", "hold"])
def test_consumed_owner_release_survives_retry_before_ack_or_reservation(rig, monkeypatch, reply):
    """S-11/Q4: proof commit -> crash before handoff -> same signed decision replay.

    Owner release is a separate proof class, not a model allow. Its NULL model
    preference must neither retire its proof nor become forged preferred-model
    evidence. Keep policy/body untouched and assert no ACK or reservation can
    accidentally protect the proof in this regression.
    """

    from tests.unit.team_bridge.test_screening_chain import _install_wire, _Wire

    config_dir, store, owner = rig
    monkeypatch.setattr(
        "local_operator.providers.model_access.credentialed_chat_providers_here",
        lambda **_kw: {"anthropic"},
    )
    wire = _Wire({SONNET[1]: ("fail", 429), OPUS[1]: ("reply", reply)})
    _install_wire(monkeypatch, wire)
    assert _scheduler_screen(config_dir, owner).hold_code == hr.WORK_SCOPE_HOLD
    decision = store.work_scope_decision(owner, MESSAGE_ID)
    assert decision is not None and decision["screener_hop"] == 1
    assert _decide(config_dir, owner, "deliver")["state"] == "released"
    signed = store.owner_release(owner, MESSAGE_ID)
    assert signed is not None and hr.verify_row(signed, hr.anchor_spki(config_dir)[0])
    signed_payload = hr.payload_for_row(signed)
    assert _scheduler_screen(config_dir, owner).disposition == "deliver"
    receipt = store.receipt_for_message(owner, MESSAGE_ID)
    assert receipt is not None and receipt["template_sha256"] == hr.OWNER_RELEASE_SHA256
    assert not receipt["screener_preferred"]  # never relabel owner proof as model evidence
    committed = store.owner_release(owner, MESSAGE_ID)
    policy = store.current_screening_policy(owner)
    sends = len(wire.sends)
    assert sends == 2  # preferred Sonnet 429, preferred Opus reply
    for _ in range(2):
        assert _count(config_dir, "ack_intents") == 0
        assert _count(config_dir, "body_reservations") == 0
        retry = _scheduler_screen(config_dir, owner)
        assert retry.disposition == "deliver" and retry.receipt is not None
        assert retry.receipt.template_sha256 == hr.OWNER_RELEASE_SHA256
        assert not retry.screener_preferred
        assert len(wire.sends) == sends
        assert store.current_screening_policy(owner) == policy
        assert store.receipt_for_message(owner, MESSAGE_ID) == receipt
        assert store.work_scope_decision(owner, MESSAGE_ID) == decision
        replayed = store.owner_release(owner, MESSAGE_ID)
        assert replayed == committed
        assert hr.payload_for_row(replayed) == signed_payload
        assert replayed["signature"] == signed["signature"]
        assert hr.verify_row(replayed, hr.anchor_spki(config_dir)[0])
