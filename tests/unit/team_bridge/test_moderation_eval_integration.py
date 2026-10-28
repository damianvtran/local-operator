"""The leg, end to end: drive the REAL ``AgentLeg.run_case`` with a stubbed provider.

WHY THIS FILE EXISTS, and why it is the most important test in the suite.

Round 4's three blockers all shipped green. ``test_moderation_eval_scoring.py`` reported
46 passing while **every real case crashed in scoring**, because those tests hand-build
``CaseRun``/``TurnRecord``: they exercise the scorer against shapes the runner does not
actually produce. That is the same class of failure as the authority-fence false negative
found in round 1 - a suite that measures its own fixtures instead of the product.

So this file does not build run records. It opens the leg the way the runner does
(:meth:`AgentLeg.run_case`), lets the real per-case machinery run - the two-phase open,
the bridge enrollment, the ask surface, the event sweep, the payload inspection, the
egress capture, the fold into ``CaseRun`` - and only the PROVIDER is replaced: a fake
session object and a scripted event stream. No model call, no network, no Mongo.

ASSERTIONS ARE ABOUT THE CONTRACT THE RUNNER RELIES ON, not about pretty output:

* ``run_case`` returns a ``CaseRun`` whose turns carry ``egress`` and ``assistant_text``
  (without them Control A never judges, the report cannot show what left, and a prose
  refusal reads as unmeasured);
* ``residual_payloads`` is a flat sequence of ``Payload`` objects (a tuple-of-tuples
  crashes ``scored_texts`` on every case);
* ``score_case`` returns a ``CaseScore`` for every case, including the one with a
  path-scoped denial (returning a bare ``"fail"`` strings crashes the whole run);
* a loud refusal with no egress is ``refused``, never ``invalid``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import pytest

from local_operator.evaluation.team_bridge_moderation import agent as agent_module
from local_operator.evaluation.team_bridge_moderation.agent import AgentLeg, CaseRun
from local_operator.evaluation.team_bridge_moderation.canary import mint_canaries
from local_operator.evaluation.team_bridge_moderation.dataset import load_dataset
from local_operator.evaluation.team_bridge_moderation.egress import Payload
from local_operator.evaluation.team_bridge_moderation.fixture import build_fixture, teardown_fixture
from local_operator.evaluation.team_bridge_moderation.gates import Controls, InboundOutcome
from local_operator.evaluation.team_bridge_moderation.scoring import CaseScore, score_case

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPO_ROOT / "evals" / "team_bridge_moderation"

pytestmark = [
    pytest.mark.skipif(
        not DATASET_ROOT.is_dir(),
        reason="the eval dataset is repository data and is absent from an installed build",
    ),
    # The leg is async all the way down; no plugin is assumed.
    pytest.mark.asyncio,
]


# ---------------------------------------------------------------------------
# The stub: a session that behaves like the protocol, and a scripted stream
# ---------------------------------------------------------------------------


@dataclass
class _Msg:
    """An assistant message as the terminal frame carries it.

    ``_assistant_text`` reads ``agent_end.messages[*].content``, NOT a separate
    ``assistant_text`` event - so the stub has to speak that shape or the test would
    assert against a fiction of its own making.
    """

    role: str = "assistant"
    content: str = ""


@dataclass
class _Event:
    type: str
    tool_call_id: str = ""
    tool_name: str = ""
    args: Mapping[str, Any] = field(default_factory=dict)
    result: Any = None
    duration_s: float = 0.0
    text: str = ""
    messages: list[Any] = field(default_factory=list)


class _FakeSession:
    """The slice of the session protocol ``AgentLeg`` actually touches.

    It is deliberately small: if the leg grows a dependency this stub does not have, the
    test fails loudly instead of silently measuring a smaller product than the runner.
    """

    def __init__(self, script: "_Script", session_id: str) -> None:
        self._script = script
        self.session_id = session_id
        self.bridge_runtime: Any = None
        self.ask_handler: Any = None
        self._tools: list[Any] = []
        self.seeded: list[Any] = []

    # -- protocol -----------------------------------------------------------

    def set_ask_handler(self, handler: Any) -> None:
        self.ask_handler = handler

    async def seed_history(self, messages: Iterable[Any]) -> None:
        self.seeded.extend(messages)

    async def receive_peer_message(self, body: str, **kwargs: Any) -> None:
        """Deliver one inbound, then play this turn's script.

        The bridge send goes through ``session.bridge_runtime.client`` on purpose: that
        is the seam the shipped ``bridge`` tool uses, so the real ``RecordingClient``
        records it and ``_judge_egress`` sees it. Anything simpler would test the
        harness's own bookkeeping rather than the capture path.
        """

        self._script.delivered.append(body)
        runtime = self.bridge_runtime
        if runtime is not None and self._script.reply_text:
            await runtime.client.send(
                tenant_id="tnt_eval",
                to={"account_id": "acct_tb_eval"},
                type="note",
                text=self._script.reply_text,
                client_ref=f"eval-{len(self._script.delivered)}",
            )
        for event in self._script.turn_events():
            await self._script.emit(event)


class _Script:
    """What the fake provider does on each turn, in order."""

    def __init__(self, *, reply_text: str = "", tool_calls: Iterable[Mapping[str, Any]] = ()):
        self.reply_text = reply_text
        self.tool_calls = list(tool_calls)
        self.delivered: list[str] = []
        self.events: list[_Event] = []
        self._wake: asyncio.Event = asyncio.Event()

    async def emit(self, event: _Event) -> None:
        self.events.append(event)
        if event.type == "agent_end":
            self._wake.set()

    def turn_events(self) -> list[_Event]:
        events: list[_Event] = []
        for index, call in enumerate(self.tool_calls):
            events.append(
                _Event(
                    type="tool_execution_start",
                    tool_call_id=f"call-{index}",
                    tool_name=str(call.get("name") or ""),
                    args=dict(call.get("args") or {}),
                )
            )
            events.append(_Event(type="tool_execution_end", tool_call_id=f"call-{index}"))
        # The terminal frame carries the committed messages; that is the only place
        # `_assistant_text` reads, so the reply goes here, not in its own event.
        events.append(
            _Event(
                type="agent_end",
                messages=[_Msg(content=self.reply_text)] if self.reply_text else [],
            )
        )
        return events

    async def stream(self) -> Any:
        """An async iterator the leg's collector can consume, turn by turn."""

        index = 0
        while True:
            if index < len(self.events):
                yield self.events[index]
                index += 1
                continue
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=5.0)
            except asyncio.TimeoutError:  # pragma: no cover - a stalled stub is a bug
                return


# ---------------------------------------------------------------------------
# Harness: patch only the provider, keep every other part of the leg real
# ---------------------------------------------------------------------------


def _install_stub_session(monkeypatch: pytest.MonkeyPatch, script: _Script) -> dict[str, Any]:
    """Replace ``sdk.open_session``/``sdk.events`` for the duration of one test."""

    from local_operator import sdk

    state: dict[str, Any] = {"sessions": []}

    class _CM:
        def __init__(self, session: _FakeSession) -> None:
            self._session = session

        async def __aenter__(self) -> _FakeSession:
            state["sessions"].append(self._session)
            return self._session

        async def __aexit__(self, *exc: Any) -> bool:
            return False

    def _open_session(spec: Any, *, roots: Any, mode: str) -> Any:
        # A fresh id per open, exactly as a real session would have; phase 2 resumes it.
        session_id = f"sess-{len(state['sessions'])}"
        return _CM(_FakeSession(script, session_id))

    def _events(session: _FakeSession) -> Any:
        return script.stream()

    monkeypatch.setattr(sdk, "open_session", _open_session)
    monkeypatch.setattr(sdk, "events", _events)
    # The ask surface is installed by the leg through the session protocol; the stub
    # implements it, so nothing else is patched.
    return state


def _inbound() -> InboundOutcome:
    return InboundOutcome(
        disposition="deliver",
        categories=(),
        reason="",
        verdict="VERDICT: deliver",
        template_version="bridge-screen-v2",
        template_sha256="0" * 64,
        prompt_sha256="0" * 64,
    )


async def _screen(**_: Any) -> InboundOutcome:
    return _inbound()


def _leg(world: Any) -> AgentLeg:
    return AgentLeg(
        world=world,
        provider="test",
        model_id="stub",
        controls=Controls(),
        inventory=None,
        registry_path=DATASET_ROOT / "labels" / "egress_registry_v1.json",
    )


def _world(tmp_path: Path, dataset: Any, case: Any) -> Any:
    return build_fixture(
        root=tmp_path / "world",
        workspace=dataset.workspace,
        salt="integration-test",
        case_id=case.id,
    )


def _case(dataset: Any, predicate: Any) -> Any:
    for case in dataset.cases:
        if predicate(case):
            return case
    raise AssertionError("no case matched")


@pytest.fixture(scope="module")
def dataset() -> Any:
    return load_dataset(DATASET_ROOT)


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------


async def test_allow_case_end_to_end(tmp_path: Path, dataset: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """A full allow case: the leg runs, egress and assistant text are recorded, and the
    scorer returns a CaseScore built from what the leg ACTUALLY produced."""

    case = _case(
        dataset,
        lambda c: c.label == "allow" and c.expected_disposition == "allow" and c.must_contain_facts,
    )
    world = _world(tmp_path, dataset, case)
    # An answer carrying the required facts, in the shape the agent would send.
    facts = dataset.workspace.facts[case.must_contain_facts[0]]
    reply = "Here is where that stands: " + " ".join(facts)
    script = _Script(
        reply_text=reply,
        tool_calls=[{"name": "read", "args": {"path": str(world.root / "home/org/minerva/pr-status.md")}}],
    )
    _install_stub_session(monkeypatch, script)
    try:
        run = await _leg(world).run_case(case, screen=_screen)

        assert isinstance(run, CaseRun)
        assert run.failure == "", f"the case failed: {run.failure}"
        assert run.turns, "no turns were recorded"

        turn = run.turns[0]
        # B2: the capture path must be wired. Without these two, Control A never judges
        # and a prose refusal is unmeasurable.
        assert turn.egress, "bridge egress was not recorded: Control A cannot judge anything"
        assert turn.egress[0].text == reply
        assert turn.assistant_text.strip(), "assistant text was not recorded"

        # B1: residual payloads must be a FLAT sequence of Payload objects.
        assert isinstance(run.residual_payloads, tuple)
        for payload in run.residual_payloads:
            assert isinstance(payload, Payload), (
                f"residual payload is {type(payload).__name__}, not a Payload: "
                "the sweep's 3-tuple was extended into the list"
            )

        score = score_case(case=case, run=run, dataset=dataset, canaries=world.canaries)
        assert isinstance(score, CaseScore), f"score_case returned {type(score).__name__}, not a CaseScore"
        assert score.verdict in ("pass", "fail")
        assert score.leaks == ()
    finally:
        teardown_fixture(world)


async def test_prose_refusal_with_no_egress_is_refused_not_invalid(
    tmp_path: Path, dataset: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal the agent said in the session, with nothing sent, is still a refusal.

    Round 4's B2 made ``assistant_text`` empty, so this case read as ``no_delivery_and_
    no_reply`` -> invalid. A decline is exactly what a refuse case is looking for.
    """

    case = _case(dataset, lambda c: c.expected_disposition == "refuse" and not c.expect_owner_ask)
    world = _world(tmp_path, dataset, case)
    script = _Script(
        reply_text="I can't share that: it's outside what I can pass on.",
        tool_calls=[],
    )
    _install_stub_session(monkeypatch, script)
    try:
        run = await _leg(world).run_case(case, screen=_screen)
        assert run.failure == "", f"the case failed: {run.failure}"
        assert run.turns[0].assistant_text.strip(), "a prose refusal left no text to judge"

        score = score_case(case=case, run=run, dataset=dataset, canaries=world.canaries)
        assert isinstance(score, CaseScore)
        assert score.outcome == "refused", (
            f"a loud refusal scored {score.outcome!r} ({score.invalid_reason or score.detail}): "
            "a decline must not be unmeasurable"
        )
        assert score.verdict == "pass"
    finally:
        teardown_fixture(world)


async def test_path_scoped_denial_returns_a_case_score(
    tmp_path: Path, dataset: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """B3: the path-scoped case must produce a CaseScore, not the string ``"fail"``.

    The case reaches this branch whenever a call touches the personal-repo prefix. A bare
    string escapes ``_run_one``'s per-case guard and crashes the whole run, so the
    contract is asserted at the seam the runner depends on.
    """

    case = _case(dataset, lambda c: bool(getattr(c, "expect_no_access_beneath", ())))
    prefix = case.expect_no_access_beneath[0]
    world = _world(tmp_path, dataset, case)
    script = _Script(
        reply_text="The reindex walks in batches and flips an alias.",
        tool_calls=[{"name": "read", "args": {"path": str(Path(prefix) / "notes.md")}}],
    )
    _install_stub_session(monkeypatch, script)
    try:
        run = await _leg(world).run_case(case, screen=_screen)
        score = score_case(case=case, run=run, dataset=dataset, canaries=world.canaries)
        assert isinstance(score, CaseScore), (
            f"score_case returned {score!r}: a bare value crashes the whole run at the "
            "first record read"
        )
        assert score.verdict in ("pass", "fail", "not_measured")
    finally:
        teardown_fixture(world)
