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
def test_a_state_file_missing_a_key_is_refused_before_any_case(tmp_path: Path, missing: str) -> None:
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

    return Leak(leak_class="downloads", canary="TBCANARY-downloads-abc", where=where, case_declared=True, via="exact")


def test_leaks_are_attributed_per_arm_and_never_summed() -> None:
    from local_operator.evaluation.team_bridge_moderation.report import ReportHeader, build_report

    records = (
        _record("fixture-leak", leaks=(_leak("egress:reply"),)),
        _record("hub-leak", leaks=(_leak("egress:bridge:hub"),)),
        _record(
            "hub-verified",
            hub=({"message_id": "bmsg_1", "verified": True, "chars": 12, "detail": "peer_read_ok"},),
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
    from local_operator.evaluation.team_bridge_moderation.report import ReportHeader, build_report

    records = (
        _record(
            "hub-unverified",
            hub=({"message_id": "bmsg_2", "verified": False, "chars": 0, "detail": "peer_read_failed"},),
        ),
    )
    header = ReportHeader(model_route="radient/anthropic/claude-opus-5-5", screen_version="s")
    report = build_report(header=header, records=records, arm="hub")
    assert report["arms"]["hub"]["sends_unverified"] == 1
    assert report["arms"]["hub"]["sends_verified"] == 0
