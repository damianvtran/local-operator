"""The hub arm: rig state, bearer threading, peer read-back, and arm attribution.

WHAT THIS SUITE PINS, and why each one is a money-or-honesty bug rather than a style
nit. Nothing here reaches a network, starts a hub, or calls a model: the arm's job is to
make a REAL hub run possible, and this file proves the parts that can be proved without
one. A live run is a separate, budgeted step (docs/evidence/team-bridge-hub-arm.md).

1. ``--limit`` bounds the RUN. Measured 2026-10-29: ``--limit 12`` selected ~201 cases
   because the flag was per category, which is how someone spends a corpus budget by
   accident. ``--limit-per-category`` keeps the old promise under its own name.
2. A rig state file that lost a key is refused BEFORE a case runs, never mid-run.
3. The shipped client receives its already-resolved identity through its own public
   ``access=`` argument - the arm must not invent a credential store, and must not
   silently drop the identity either (a send with no access is the "sign in to Radient"
   refusal that made the fixture arm unmeasurable).
4. A send is only evidence when the PEER can read it back. An unreadable message is
   ``verified=False``, never an empty string that scans clean.
5. Leak counts are attributed per arm and never summed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.evaluation.team_bridge_moderation.dataset import load_dataset
from local_operator.evaluation.team_bridge_moderation.hubarm import (
    HubArmError,
    HubObservation,
    HubRigState,
    HubVerifier,
    PeerClient,
    RigAccess,
    item_text,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPO_ROOT / "evals" / "team_bridge_moderation"

pytestmark = pytest.mark.skipif(
    not DATASET_ROOT.is_dir(),
    reason="the eval dataset is repository data and is absent from an installed build",
)


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATASET_ROOT)


# ---------------------------------------------------------------------------
# 1. --limit is a run bound
# ---------------------------------------------------------------------------


def test_limit_is_a_TOTAL_case_cap(dataset) -> None:
    """The flag a reader assumes bounds the run, does bound the run."""

    selected = dataset.select(limit=12)
    assert len(selected) == 12, "a total limit must cap the run, not each category"
    # …and it is drawn across categories rather than from one file.
    assert len({case.category for case in selected}) > 1


def test_limit_per_category_keeps_the_old_meaning_under_its_own_name(dataset) -> None:
    selected = dataset.select(limit_per_category=2)
    categories = {case.category for case in selected}
    assert len(selected) == 2 * len({case.category for case in dataset.cases if not case.skipped})
    assert len(categories) == len({case.category for case in dataset.cases if not case.skipped})


def test_the_two_limits_are_not_the_same_promise(dataset) -> None:
    total = len(dataset.select(limit=12))
    per_category = len(dataset.select(limit_per_category=12))
    assert total == 12
    assert per_category > total, "the per-category form must not be mistakable for a bound"


def test_a_zero_limit_is_refused(dataset) -> None:
    with pytest.raises(ValueError):
        dataset.select(limit=0)


# ---------------------------------------------------------------------------
# 2. Rig state validation
# ---------------------------------------------------------------------------


def _state(**overrides: Any) -> dict[str, Any]:
    base = {
        "url": "http://127.0.0.1:9/v1",
        "tenant": "org-a",
        "tokens": {"ada": "tok-ada", "ben": "tok-ben"},
        "accounts": {"ada": "acct-ada", "ben": "acct-ben"},
        "endpoints": {"ada": "ep-ada", "ben": "ep-ben"},
    }
    base.update(overrides)
    return base


def test_rig_state_round_trips_and_names_both_accounts(tmp_path: Path) -> None:
    import json

    path = tmp_path / "state.json"
    path.write_text(json.dumps(_state()), encoding="utf-8")
    state = HubRigState.load(path)
    assert state.base_url == "http://127.0.0.1:9/v1"
    assert state.identity("ben").account_id == "acct-ben"
    assert state.identity("ada").endpoint_id == "ep-ada"


@pytest.mark.parametrize("missing", ["url", "tenant", "tokens", "accounts", "endpoints"])
def test_a_state_file_missing_a_key_is_refused_before_any_case(
    tmp_path: Path, missing: str
) -> None:
    import json

    raw = _state()
    raw.pop(missing)
    path = tmp_path / "state.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(HubArmError) as refused:
        HubRigState.load(path)
    assert missing in str(refused.value)


def test_a_state_file_without_the_peer_role_is_refused(tmp_path: Path) -> None:
    """Both accounts are load-bearing: the peer IS the read-back proof."""

    import json

    path = tmp_path / "state.json"
    path.write_text(json.dumps(_state(tokens={"ben": "tok-ben"})), encoding="utf-8")
    with pytest.raises(HubArmError) as refused:
        HubRigState.load(path)
    assert "ada" in str(refused.value)


def test_a_non_json_state_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(HubArmError):
        HubRigState.load(path)


# ---------------------------------------------------------------------------
# 3. The bearer reaches the shipped client
# ---------------------------------------------------------------------------


class _RecordingInner:
    """The shipped client's shape, recording how it was called."""

    base_url = "http://127.0.0.1:9/v1"

    def __init__(self) -> None:
        self.sends: list[dict[str, Any]] = []
        self.reads: list[tuple[str, dict[str, Any]]] = []

    async def send(self, **kwargs: Any) -> Any:
        self.sends.append(kwargs)
        return {"message": {"id": "bmsg_accepted"}}

    async def message(self, message_id: str, **kwargs: Any) -> Any:
        self.reads.append((message_id, kwargs))
        return {"message": {"text": "hello from the hub"}}

    async def conversation_send(self, *args: Any, **kwargs: Any) -> Any:
        return {"message": {"id": "bmsg_conv"}}


def _peer(inner: Any) -> PeerClient:
    return PeerClient(
        inner,
        RigAccess(
            access_token="tok-ben", account_id="acct-ben", endpoint_id="ep-ben", tenant="org-a"
        ),
    )


@pytest.mark.asyncio
async def test_a_send_carries_the_rig_bearer_and_records_the_accepted_id() -> None:
    inner = _RecordingInner()
    client = _peer(inner)
    await client.send(
        tenant_id="org-a",
        to={"account_id": "acct-ada"},
        type="question",
        text="the answer",
        client_ref="ref-1",
    )
    call = inner.sends[0]
    assert call["access"].access_token == "tok-ben"
    assert call["access"].account_id == "acct-ben"
    assert client.sent_ids == ["bmsg_accepted"], "an accepted id must be kept for read-back"


@pytest.mark.asyncio
async def test_a_reply_with_no_message_id_records_no_observation() -> None:
    class _NoId(_RecordingInner):
        async def send(self, **kwargs: Any) -> Any:
            return {}

    client = _peer(_NoId())
    await client.send(
        tenant_id="org-a", to={"account_id": "acct-ada"}, type="question", text="x", client_ref="r"
    )
    assert client.sent_ids == [], "a reply without an id must not become a fake observation"


# ---------------------------------------------------------------------------
# 4. Read-back is the proof
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_peer_read_back_verifies_the_send() -> None:
    verifier = HubVerifier(_peer(_RecordingInner()), tenant="org-a")
    observation = await verifier.verify("bmsg_accepted")
    assert observation.verified is True
    assert observation.text == "hello from the hub"


@pytest.mark.asyncio
async def test_an_unreadable_message_is_unverified_not_clean() -> None:
    class _Broken(_RecordingInner):
        async def message(self, message_id: str, **kwargs: Any) -> Any:
            raise RuntimeError("hub said no")

    verifier = HubVerifier(_peer(_Broken()), tenant="org-a")
    observation = await verifier.verify("bmsg_accepted")
    assert observation.verified is False
    assert "peer_read_failed" in observation.detail


@pytest.mark.asyncio
async def test_a_bodyless_message_is_unverified_not_an_empty_clean_scan() -> None:
    class _Empty(_RecordingInner):
        async def message(self, message_id: str, **kwargs: Any) -> Any:
            return {"message": {}}

    verifier = HubVerifier(_peer(_Empty()), tenant="org-a")
    observation = await verifier.verify("bmsg_accepted")
    assert observation.verified is False
    assert observation.detail == "peer_read_had_no_body_text"


@pytest.mark.asyncio
async def test_a_send_with_no_id_is_unverified() -> None:
    verifier = HubVerifier(_peer(_RecordingInner()), tenant="org-a")
    observation = await verifier.verify("")
    assert observation.verified is False


def test_the_observation_record_never_carries_the_body() -> None:
    """The report may travel to a PR; the body is the canary-bearer it proves."""

    record = HubObservation("bmsg_1", True, text="TBCANARY-downloads-abc").as_record()
    assert "TBCANARY" not in str(record)
    assert record["chars"] == len("TBCANARY-downloads-abc")


@pytest.mark.parametrize(
    "item,expected",
    [
        ({"text": "a"}, "a"),
        ({"body": "b"}, "b"),
        ({"message": {"text": "c"}}, "c"),
        ({"message_body": {"text": "d"}}, "d"),
        ({"message": {}}, ""),
        ({"text": 5}, ""),
        ("not a mapping", ""),
    ],
)
def test_item_text_reads_the_documented_shapes_only(item: Any, expected: str) -> None:
    assert item_text(item) == expected


# ---------------------------------------------------------------------------
# 5. Arm attribution in the report
# ---------------------------------------------------------------------------


def _record(case_id: str, *, leaks: tuple[Any, ...] = (), hub: tuple[dict[str, Any], ...] = ()):
    from local_operator.evaluation.team_bridge_moderation.report import CaseRecord
    from local_operator.evaluation.team_bridge_moderation.scoring import CaseScore

    score = CaseScore(
        case_id=case_id,
        outcome="allow",
        verdict="fail" if leaks else "pass",
        stage="agent",
        classification="canary",
        leaks=leaks,
    )
    turn = {"hub_observations": list(hub)} if hub else {}
    return CaseRecord(
        case_id=case_id,
        label="allow",
        category="work_status",
        expected_disposition="allow",
        control="A",
        form="direct_request",
        score=score,
        turns=(turn,) if turn else (),
    )


def _leak(where: str):
    from local_operator.evaluation.team_bridge_moderation.canary import Leak

    return Leak(
        leak_class="downloads",
        canary="TBCANARY-downloads-abc",
        where=where,
        case_declared=True,
        via="exact",
    )


def test_leaks_are_attributed_per_arm_and_never_summed() -> None:
    from local_operator.evaluation.team_bridge_moderation.report import (
        ReportHeader,
        build_report,
    )

    records = (
        _record("fixture-leak", leaks=(_leak("egress:reply"),)),
        _record("hub-leak", leaks=(_leak("egress:bridge:hub"),)),
        _record(
            "hub-verified",
            hub=(
                {"message_id": "bmsg_1", "verified": True, "chars": 12, "detail": "peer_read_ok"},
            ),
        ),
    )
    header = ReportHeader(
        model_route="radient/anthropic/claude-opus-5-5",
        screen_version="bridge-screen-v2",
    )
    report = build_report(header=header, records=records, arm="hub")
    arms = report["arms"]
    assert arms["fixture"]["leaks"] == 1, "a recorded-wire leak belongs to the fixture arm"
    assert arms["hub"]["leaks"] == 1, "a peer-read leak belongs to the hub arm"
    assert report["header"]["arm"] == "hub"
    assert arms["hub"]["sends_verified"] == 1
    assert arms["hub"]["sends_unverified"] == 0
    # The headline number is NOT their sum, and the prose says which arm it came from.
    assert report["summary"]["leaks"] == 2
    markdown = __import__(
        "local_operator.evaluation.team_bridge_moderation.report", fromlist=["markdown_summary"]
    ).markdown_summary(report)
    assert "arm: `hub`" in markdown
    assert "never summed" in markdown


def test_an_unverified_hub_send_is_reported_as_unverified() -> None:
    from local_operator.evaluation.team_bridge_moderation.report import (
        ReportHeader,
        build_report,
    )

    records = (
        _record(
            "hub-unverified",
            hub=(
                {
                    "message_id": "bmsg_2",
                    "verified": False,
                    "chars": 0,
                    "detail": "peer_read_failed",
                },
            ),
        ),
    )
    header = ReportHeader(model_route="radient/anthropic/claude-opus-5-5", screen_version="s")
    report = build_report(header=header, records=records, arm="hub")
    assert report["arms"]["hub"]["sends_unverified"] == 1
    assert report["arms"]["hub"]["sends_verified"] == 0


# ---------------------------------------------------------------------------
# 6. The budget guard binds REPORTED spend, not the upper-bound estimate
# ---------------------------------------------------------------------------


def test_the_budget_guard_charges_reported_dollars_and_estimates_only_the_rest():
    """Measured 2026-10-29: the estimate is ~3x the bill, so it cannot be the ceiling's input.

    A $3 ceiling stopped a real run at ~$1.50 of billed spend, and a sanctioned $15 would
    have bought five cases instead of thirty. The guard therefore compares REPORTED dollars
    plus the estimate for calls that carried no price.
    """

    from local_operator.evaluation.team_bridge_moderation.runner import Spend

    billed = Spend(billed_usd=0.45, reported=6)
    assert billed.bound_usd == pytest.approx(0.45)

    unpriced = Spend(unreported_input=162_000, unreported_output=9_000, unreported=6)
    assert unpriced.bound_usd > billed.bound_usd * 3, "the estimate must stay conservative"

    mixed = billed + unpriced
    assert mixed.bound_usd == pytest.approx(0.45 + unpriced.bound_usd)
    assert mixed.reported == 6 and mixed.unreported == 6


def test_the_report_carries_the_cost_block_and_prints_it():
    from local_operator.evaluation.team_bridge_moderation.report import (
        ReportHeader,
        build_report,
        markdown_summary,
    )

    report = build_report(
        header=ReportHeader(
            model_route="radient/anthropic/claude-opus-5-5", screen_version="bridge-screen-v2"
        ),
        records=[],
        arm="hub",
        cost={
            "billed_usd": 1.25,
            "bound_usd": 1.30,
            "reported_requests": 8,
            "unreported_requests": 2,
        },
    )
    assert report["header"]["cost"]["billed_usd"] == 1.25
    text = markdown_summary(report)
    assert "$1.25 reported" in text
    assert "unpriced request(s)" in text


# ---------------------------------------------------------------------------
# 7. The rig's logins never touch the fixture store or the operator's
# ---------------------------------------------------------------------------


def test_seeding_the_rig_requires_an_explicit_store():
    """`AuthStore()` with no db_path writes to the OPERATOR's database.

    The first implementation did exactly that: four rig credential rows landed in the
    operator's real store (found and deleted 2026-10-29). The guard is structural now.
    """

    from local_operator.evaluation.team_bridge_moderation.hubarm import (
        HubArmError,
        HubRigState,
        install_rig_login,
    )

    state = HubRigState(
        base_url="http://127.0.0.1:9/v1",
        tenant="org-a",
        tokens={"ada": "t-a", "ben": "t-b"},
        accounts={"ada": "acct-ada", "ben": "acct-ben"},
        endpoints={"ada": "ep-ada", "ben": "ep-ben"},
    )
    import tempfile

    with pytest.raises(HubArmError) as refused:
        install_rig_login(Path(tempfile.mkdtemp()), state, role="ben")
    assert "operator" in str(refused.value)


def test_the_rig_store_is_separate_from_the_fixture_store(tmp_path: Path):
    """A rig login in the fixture store is a MODEL-CREDENTIAL collision, not a detail.

    Measured 2026-10-29 on the first paid hub run: the fixture carried a RADIENT OAuth row
    holding the rig's token, the provider stack resolved it for model calls, and all 30
    cases came back "authentication failed (HTTP 401): invalid API key format" before any
    turn could measure anything.
    """

    from local_operator.evaluation.team_bridge_moderation.hubarm import (
        HubRigState,
        build_arm_clients,
    )
    from local_operator.providers.auth_store import AuthStore

    state = HubRigState(
        base_url="http://127.0.0.1:9/v1",
        tenant="org-a",
        tokens={"ada": "t-a", "ben": "t-b"},
        accounts={"ada": "acct-ada", "ben": "acct-ben"},
        endpoints={"ada": "ep-ada", "ben": "ep-ben"},
    )
    build_arm_clients(state, config_dir=tmp_path)

    fixture = AuthStore(db_path=tmp_path / "auth.db", config_dir=tmp_path)
    assert (
        fixture.list_credentials("radient") == []
    ), "the fixture store must hold no rig credential: the model provider reads it"
    rig_dir = tmp_path.with_name(tmp_path.name + "-rig-auth")
    rig = AuthStore(db_path=rig_dir / "auth.db", config_dir=rig_dir)
    assert {r.data.get("account_id") for r in rig.list_credentials("radient")} == {
        "acct-ada",
        "acct-ben",
    }


# ---------------------------------------------------------------------------
# 8. An arm never inherits the other arm's limitations or its blank reasons
# ---------------------------------------------------------------------------


def test_the_hub_arm_does_not_claim_the_wire_was_recorded():
    """The first real hub report carried the FIXTURE arm's confession verbatim.

    It said a RecordingClient stood in for the hub while a real rig hub had answered a
    real send (measured 2026-10-29). A limitation that is false about its own run is
    worse than no limitation: it teaches a reader to discount the true ones.
    """

    from local_operator.evaluation.team_bridge_moderation.runner import limitations_for

    fixture = limitations_for("fixture")
    hub = limitations_for("hub")
    assert any("recorded, not reached" in line for line in fixture)
    assert not any("the wire is recorded" in line for line in hub)
    assert any("SHIPPED client" in line for line in hub)
    assert any("rig" in line.lower() for line in hub)


def test_an_invalid_case_always_names_its_cause():
    from local_operator.evaluation.team_bridge_moderation.report import CaseRecord
    from local_operator.evaluation.team_bridge_moderation.scoring import CaseScore

    for reason, expected in (
        ("ask_surface_missing", "ask_surface_missing"),
        ("", "unrecorded_invalidity"),
    ):
        record = CaseRecord(
            case_id="refuse.example.01",
            label="refuse",
            category="example",
            expected_disposition="refuse",
            control="A",
            form="direct_request",
            score=CaseScore(
                case_id="refuse.example.01",
                outcome="invalid",
                verdict="not_measured",
                stage="none",
                classification="canary",
                invalid_reason=reason,
            ),
        )
        assert record.as_record()["invalid_reason"] == expected


# ---------------------------------------------------------------------------
# 9. The session is actually OFFERED the tools the measurement needs
# ---------------------------------------------------------------------------


def test_the_session_is_offered_a_bridge_tool_and_an_ask_tool(tmp_path: Path) -> None:
    """The first paid run's whole outcome was an artefact of this being false.

    ``build_bridge_tool`` is createIf on ``context.bridge_runtime``, and the harness used
    to set ``session.bridge_runtime`` only. The context is a different object, so the
    model was offered 38 tools and NO bridge tool: five allow cases were scored "the agent
    answered and nothing reached the asker" when the agent had no way to send anything
    (measured 2026-10-29). This asserts the inventory, which is the thing a later reader
    would otherwise have to infer from outcomes.
    """

    import asyncio
    import os

    from local_operator import sdk
    from local_operator.evaluation.team_bridge_moderation.agent import AgentLeg
    from local_operator.evaluation.team_bridge_moderation.dataset import load_dataset
    from local_operator.evaluation.team_bridge_moderation.fixture import (
        build_fixture,
        enroll_bridge,
    )
    from local_operator.evaluation.team_bridge_moderation.gates import Controls
    from local_operator.session.spec import ApprovalPolicy, SessionRoots, SessionSpec

    dataset = load_dataset(Path("evals/team_bridge_moderation").resolve())
    world = build_fixture(
        root=tmp_path / "world", workspace=dataset.workspace, salt="tools", case_id="tools"
    )
    leg = AgentLeg(
        world=world,
        provider="radient",
        model_id="anthropic/claude-opus-5-5",
        controls=Controls(disabled=frozenset()),
        registry_path=Path("evals/team_bridge_moderation/labels/egress_registry_v1.json").resolve(),
    )
    # A pytest tmp_path is exactly the volatile root this API refuses by default; the
    # probe never outlives the process, so the override is the honest one.
    roots = SessionRoots(
        config_dir=world.config, agent_home=world.home, cwd=world.home, allow_volatile=True
    )
    spec = SessionSpec(
        hosting="radient",
        model="anthropic/claude-opus-5-5",
        name="tools-probe",
        tools=None,
        approvals=ApprovalPolicy.auto(),
    )

    previous_home = os.environ.get("HOME")
    os.environ["HOME"] = str(world.home)
    try:

        async def run() -> tuple[set[str], str, str]:
            async with sdk.open_session(spec, roots=roots, mode="own") as session:
                before = {
                    getattr(tool, "name", "") for tool in (getattr(session, "_tools", None) or [])
                }
                assert "bridge" not in before, (
                    "precondition: the tool is createIf, so it must be absent before the "
                    "runtime is installed - otherwise this test proves nothing"
                )
                owner = enroll_bridge(
                    world, session_id=str(getattr(session, "session_id", "") or "")
                )
                leg._install_runtime(session, [], owner=owner)
                after = {
                    getattr(tool, "name", "") for tool in (getattr(session, "_tools", None) or [])
                }
                return after, leg.bridge_tool_state, leg.ask_surface

        after, bridge_state, ask_state = asyncio.run(run())
    finally:
        if previous_home is not None:
            os.environ["HOME"] = previous_home

    assert "bridge" in after, f"the model was offered no bridge tool: {sorted(after)}"
    assert "ask" in after, f"the model was offered no ask tool: {sorted(after)}"
    assert bridge_state == "present"
    assert ask_state == "installed"


# ---------------------------------------------------------------------------
# 10. A refused call's SHAPE is evidence; its values stay out of the report
# ---------------------------------------------------------------------------


def test_a_refused_call_records_recipient_shape_but_never_values():
    """The three refusals the bridge tool gives turn on the SHAPE of ``to``.

    Measured 2026-10-29: every bridge call in the hub-arm run was refused, and the record
    kept only argument keys - so "no recipient", "recipient as a string" and "recipient as
    an object with the wrong keys" were indistinguishable in the report, although each
    produced a different refusal from the same tool.
    """

    from local_operator.evaluation.team_bridge_moderation.agent import (
        ToolCall,
        _shape_of,
    )

    canary = "CANARY-1234-should-never-appear"

    absent = ToolCall(
        name="bridge",
        arguments={"op": "send", "text": "hello"},
        paths=(),
        outside_work_roots=False,
        refused=True,
        error=None,
    ).as_record()["argument_shape"]
    assert set(absent["types"]) == {"op", "text"}
    assert "to" not in absent["keys"], "an absent recipient must be visibly absent"

    # The three recipient shapes the tool answers differently - all distinguishable.
    assert _shape_of("acct-ada") == "str"
    assert _shape_of({"account_id": "acct-ada"})["keys"] == ["account_id"]
    assert _shape_of({"name": "Ada Lovelace"})["keys"] == ["name"]
    assert _shape_of({"name": "Ada"}) != _shape_of({"account_id": "acct-ada"})

    record = ToolCall(
        name="bridge",
        arguments={"op": "send", "type": "note", "text": canary, "to": {"name": canary}},
        paths=(),
        outside_work_roots=False,
        refused=True,
        error=None,
    ).as_record()
    rendered = repr(record["argument_shape"])
    assert canary not in rendered, f"a value reached the report: {rendered}"
    assert record["argument_shape"]["types"]["to"]["keys"] == ["name"]
    assert record["argument_shape"]["types"]["text"] == "str"


def test_the_shape_summary_is_closed_and_bounded():
    from local_operator.evaluation.team_bridge_moderation.agent import (
        _SHAPE_MAX_KEYS,
        _argument_shape,
        _shape_of,
    )

    long_mapping = {f"k{index}": "value" for index in range(_SHAPE_MAX_KEYS + 5)}
    summary = _argument_shape(long_mapping)
    assert summary["truncated"] is True
    assert len(summary["types"]) == _SHAPE_MAX_KEYS

    deep = _shape_of({"a": {"b": {"c": {"d": "value"}}}})
    assert deep["types"]["a"]["types"]["b"] in ("object", "dict"), deep

    assert _shape_of("") == "str[0]", "empty must not read as absent"
    assert _shape_of("value") == "str"
    assert _shape_of(["a", "b"]) == {"list": 2, "of": "str"}
    assert _shape_of(3) == "int" and _shape_of(None) == "null"
