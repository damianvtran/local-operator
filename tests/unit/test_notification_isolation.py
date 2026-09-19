"""A session that exists to be TESTED must never reach the operator's desktop.

WHY THIS FILE EXISTS. The operator kept getting macOS notification banners whose
body was ``Hello from the mock provider`` — the deterministic reply from the
test hosting (``providers/clients.py``), and therefore the composed snippet
(``notifications/compose.py`` takes a notification's body from the session's own
last assistant line) of every mock session on the machine. The raisers were the
sessions' OWN runtimes (``session/runtime/serving.py::_announce_completion``,
rung 4: nothing watching, no desktop presence, no TUI) plus the two
machine-wide backend surfaces a desktop app reads. Seventeen recorded banner
attempts across scratch stores in two days, all of them mock sessions, all of
them from drive-by rigs.

Two properties are pinned here, and they are different questions:

* **A process that adopts the test hosting is a test process**, so it does not
  notify — on any wire, including the in-band toast. That is a process-wide
  switch (``tui.notify.suppress_notifications_for_process``) rather than a
  per-session flag, because the environment is the only form of it a SPAWNED
  child also respects.
* **A stored session that ran on the test hosting is not announced by anyone**,
  even from a process that never touched the mock itself: a scratch store
  outlives the rig that filled it, and the operator's own backend polls it.

The last two tests are the drift guards: one sweeps every child-environment
builder under ``tests/e2e/`` and ``scripts/`` for the gate, and one asserts the
test tree and the script tree declare the same switch names.

NOT COVERED HERE, deliberately: ad-hoc rigs written outside this repository.
They cannot be reached by any test, which is why the load-bearing part of the
fix is the product gate rather than the harness ones.
"""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

import local_operator.session.attention as attention_module
import local_operator.session.runtime.serving as serving_module
from local_operator.model.configure import configure_model
from local_operator.notifications import compose
from local_operator.providers.clients import client_for_spec
from local_operator.providers.registry import is_mock_provider
from local_operator.session.model_selection import session_uses_test_hosting
from local_operator.tui.notify import (
    ENV_DISABLE,
    ENV_DISABLE_VALUE,
    Notifier,
    notifications_enabled,
    suppress_notifications_for_process,
)

#: The kill switch itself, and the spellings a module may use to carry it.
#: ``NO_NOTIFY_ENV`` is ``tests/e2e/harness.py``'s mapping (a bespoke, filtered
#: child environment cannot use the product helper — it builds the mapping from
#: scratch); ``harness_child_env`` is the product's own carrier for a script that
#: drives the real CLI; ``suppress_notifications_for_process`` is the in-process
#: form; ``probe_isolation`` is the import-time sandbox. A module that mentions
#: any one of them has made the gate its own business rather than inheriting it
#: by accident.
GATE = "LOCAL_OPERATOR_NO_NOTIFICATIONS"
_GATE_SPELLINGS = (
    GATE,
    "NO_NOTIFY_ENV",
    "harness_child_env",
    "suppress_notifications_for_process",
    "probe_isolation",
)

#: Strings that mark a module as one that spawns LOCAL-OPERATOR children rather
#: than a git or a flake8. A syntactic sweep that ignored this would demand the
#: switch in every module that ever hands a child an environment, which is most
#: of the suite and none of the hazard.
_ENTRY_POINTS = (
    "local_operator.cli",
    "local_operator.session.runtime.process",
    "local_operator.wakes.supervisor",
    "local-operator",
)

_SPAWNER_NAME = re.compile(r"(popen|run|call|spawn|exec|check_output|sh|system)", re.I)

#: The two stages the sweep covers. ``tests/e2e/`` is where bespoke, filtered
#: child environments are built — the ambient one cannot be relied on because
#: the builders deliberately strip families from it — and ``scripts/`` is where
#: the rigs live. The rest of ``tests/unit`` spawns short-lived helpers
#: (``python -c`` probes, ``lop secret``, a fake ssh) whose environment is
#: derived from ``os.environ``, which ``tests/conftest.py`` arms at import time;
#: the package side is pinned independently by
#: ``tests/unit/test_ambient_env_isolation.py``.
_GATED_STAGES = (
    Path(__file__).resolve().parents[1] / "e2e",
    Path(__file__).resolve().parents[2] / "scripts",
)

#: A mock session's stored model selection: the row
#: ``Session._persist_selected_model`` writes, and the only durable record of
#: which hosting a conversation actually ran on.
_SELECTION_ROW = {
    "id": "selection-1",
    "ts": 1.0,
    "type": "custom",
    "payload": {
        "custom_type": "selected_model",
        "details": {"version": 2, "selector": "test/test-model", "effort": None, "boot": None},
    },
}


def _transcript_with_selection(directory: Path, selector: str | None) -> Path:
    """A session directory whose journal carries (or omits) a selection row."""
    directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    if selector is not None:
        row = json.loads(json.dumps(_SELECTION_ROW))
        row["payload"]["details"]["selector"] = selector
        rows.append(row)
    with (directory / "transcript.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return directory


# ---------------------------------------------------------------------------
# The process gate
# ---------------------------------------------------------------------------


def test_the_helper_turns_the_process_switch_on_and_stays_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(GATE, raising=False)
    assert notifications_enabled() is True

    suppress_notifications_for_process("mock hosting (test/test-model)")
    assert notifications_enabled() is False
    assert os.environ[GATE] == "1"

    # Sticky and one-way: a second caller must not clear it, and it keeps the
    # first reason rather than relabelling itself.
    suppress_notifications_for_process("something else")
    assert os.environ[GATE] == "1"


def test_the_gate_reads_the_mapping_it_is_given() -> None:
    """An injected environment is honoured, which is what keeps every existing
    tests of the notifier (which inject one) deterministic."""
    assert notifications_enabled({GATE: "1"}) is False
    assert notifications_enabled({}) is True


def test_building_a_mock_client_suppresses_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The supply edge: EVERY mock stream is built here, including the
    mid-session switch that turns a real conversation into a mock one."""
    from local_operator.harness.types import ModelSpec

    monkeypatch.delenv(GATE, raising=False)
    client_for_spec(ModelSpec(provider="test", model_id="test-model", context_window=1000))
    assert notifications_enabled() is False


@pytest.mark.parametrize("provider", ["openai", "anthropic", "google"])
def test_a_real_hosting_leaves_the_gate_alone(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    """Failing OPEN here would silence every real session on the machine."""
    from local_operator.harness.types import ModelSpec

    monkeypatch.delenv(GATE, raising=False)
    client_for_spec(ModelSpec(provider=provider, model_id="gpt-x", context_window=1000))
    assert notifications_enabled() is True


def test_adopting_a_mock_spec_suppresses_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spec adoption is EARLIER than any stream, and it is what lets a TUI that
    boots on the mock never construct a notifier at all."""
    monkeypatch.delenv(GATE, raising=False)
    configuration = configure_model("test", "test-model")
    assert configuration.spec.provider == "test"
    assert notifications_enabled() is False


def test_adopting_a_real_spec_leaves_the_gate_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(GATE, raising=False)
    configure_model("openai", "gpt-x")
    assert notifications_enabled() is True


def test_is_mock_provider_answers_by_wire_not_by_name() -> None:
    assert is_mock_provider("test") is True
    # The legacy `--hosting` alias resolves to the same provider.
    assert is_mock_provider("noop") is True
    assert is_mock_provider("openai") is False
    assert is_mock_provider("not-a-provider") is False


class _Sink:
    """Collects what the app would have written to the terminal."""

    def __init__(self) -> None:
        self.writes: list[str] = []

    def __call__(self, data: str) -> None:
        self.writes.append(data)


def test_a_notifier_built_before_the_gate_goes_silent_with_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session switched onto the mock mid-run must stop toasting IN BAND too.

    The notifier takes ``enabled`` at construction, which is the whole bug:
    built while the session was real, it would keep writing OSC escapes (a real
    interruption, not chrome) for the rest of the run. So the gate is re-read
    per call — from the environment the notifier was built with, which is
    ``os.environ`` in production and an injected mapping in tests, the same
    source every other decision in the class reads.
    """
    monkeypatch.delenv(GATE, raising=False)
    monkeypatch.setattr("local_operator.tui.notify.settings_get", lambda key, default=None: True)
    environment = {"TERM": "xterm-256color"}
    sink = _Sink()
    notifier = Notifier(sink, env=environment)
    notifier.set_focused(False)
    assert notifier.notify_turn_complete(running_children=0) is True
    assert sink.writes

    # The switch flips mid-run, exactly as it does when a session switches to
    # the mock hosting.
    sink.writes.clear()
    environment[GATE] = "1"
    notifier.set_focused(False)
    assert notifier.enabled is False
    assert notifier.notify_turn_complete(running_children=0) is False
    assert sink.writes == []

    # ...and the production shape, where the class reads the process's own
    # environment: the process-wide helper silences an already-built notifier.
    monkeypatch.delenv(GATE, raising=False)
    live = Notifier(sink)
    assert live.enabled is True
    suppress_notifications_for_process("mock hosting (test/test-model)")
    assert live.enabled is False


@pytest.mark.asyncio
async def test_a_suppressed_runtime_takes_no_delivery_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A silenced process must not write-then-unwrite a claim every turn.

    The claim is a watermark asserting somebody was told; nobody was. Checked by
    spying on the store rather than by inspecting a database, so the assertion is
    about the CALL rather than about the state a wrong implementation might
    leave behind.
    """
    from tests.unit.session.runtime.test_serving import make_handle

    monkeypatch.delenv(GATE, raising=False)
    claimed: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        attention_module.AttentionStore,
        "claim_delivery",
        lambda self, identity, token, backend: bool(claimed.append((identity, token, backend))),
    )
    handle, _session = make_handle()
    suppress_notifications_for_process("mock hosting (test/test-model)")

    assert handle._announce_completion() == serving_module._ANNOUNCE_SETTLED
    assert claimed == []


@pytest.mark.asyncio
async def test_a_suppressed_runtime_still_records_a_parked_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DURABLE half of a parked gate survives the silence.

    ``pending`` is what keeps a parked runtime findable in ``lop sessions`` — it
    is not a notification, and a test process still owes the honest record. Only
    the out-of-band banner is skipped.
    """
    from tests.unit.session.runtime.test_serving import make_handle

    announced: list[Any] = []
    monkeypatch.delenv(GATE, raising=False)
    monkeypatch.setattr(
        "local_operator.tui.notify.detached_notify",
        lambda *args, **kwargs: bool(announced.append(args)),
    )
    handle, _session = make_handle()

    class _Registrant:
        record = type("R", (), {"session_id": "notify000001"})()

        def __init__(self) -> None:
            self.pending: list[str] = []

        def watching_surfaces(self) -> frozenset[str]:
            return frozenset()

        def set_record_pending(self, kind: str | None) -> None:
            self.pending.append(str(kind))

    registrant = _Registrant()
    handle._registrant = registrant

    # Real hosting: the banner is raised, which is the control for this test.
    handle._announce_pending("approval", "bash", "rm -rf build/")
    assert announced, "the control arm raised nothing"

    suppress_notifications_for_process("mock hosting (test/test-model)")
    announced.clear()
    handle._announce_pending("approval", "bash", "rm -rf build/")
    assert announced == []
    assert registrant.pending == ["approval", "approval"]


# ---------------------------------------------------------------------------
# The stored-session gate: a store outlives the process that filled it
# ---------------------------------------------------------------------------


def test_a_journalled_mock_selection_marks_the_session_as_test_hosted(tmp_path: Path) -> None:
    directory = _transcript_with_selection(tmp_path / "sessions" / "mock0001", "test/test-model")
    assert session_uses_test_hosting(directory) is True


def test_a_real_selection_is_not_test_hosted(tmp_path: Path) -> None:
    directory = _transcript_with_selection(tmp_path / "sessions" / "real0001", "openai/gpt-x")
    assert session_uses_test_hosting(directory) is False


def test_the_verdict_is_memoised_on_the_journal_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-2's steady state: a ``stat`` per ask, not a walk.

    The callers ask once per CANDIDATE ROW PER TICK and the cold read is up to
    ~745 ms on a large journal, so the memo is what keeps a poll cheap. Counted
    at :func:`_read_test_hosting`, the one function that walks the file, and
    the journal is then APPENDED to — which must move the key and force exactly
    one re-read, because that is the case the key has to catch (a session that
    switches hosting mid-life).
    """
    from local_operator.session import model_selection

    directory = _transcript_with_selection(tmp_path / "sessions" / "memo0001", "test/test-model")
    calls: list[Path] = []
    real = model_selection._read_test_hosting

    def counted(path: Path) -> Any:
        calls.append(path)
        return real(path)

    monkeypatch.setattr(model_selection, "_read_test_hosting", counted)
    model_selection._HOSTING_VERDICT_CACHE.clear()

    assert session_uses_test_hosting(directory) is True
    assert session_uses_test_hosting(directory) is True
    assert len(calls) == 1, calls

    # The newest selection wins, and the append is what tells the reader the
    # answer may have moved: a session that leaves the mock is real again.
    row = json.loads(json.dumps(_SELECTION_ROW))
    row["payload"]["details"]["selector"] = "openai/gpt-x"
    with (directory / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")

    assert session_uses_test_hosting(directory) is False
    assert len(calls) == 2, calls


def test_a_missing_journal_is_answered_without_being_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absence is a MOMENT, not a verdict, so it is never memoised.

    A store mid-write, an unmounted volume or an EMFILE moment all look like
    "no journal", and caching that as "not a test session" would serve the
    outage for the life of the store — ``resume.py`` refuses to cache the same
    thing for its ``origin.json`` markers, for the same reason.
    """
    from local_operator.session import model_selection

    directory = tmp_path / "sessions" / "absent01"
    directory.mkdir(parents=True)
    model_selection._HOSTING_VERDICT_CACHE.clear()

    assert session_uses_test_hosting(directory) is False
    assert model_selection._HOSTING_VERDICT_CACHE == {}


def test_the_verdict_cache_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A long-lived backend must not accumulate one entry per session forever."""
    from local_operator.session import model_selection

    monkeypatch.setattr(model_selection, "_HOSTING_VERDICT_CACHE_MAX", 2)
    model_selection._HOSTING_VERDICT_CACHE.clear()

    for index in range(3):
        directory = _transcript_with_selection(
            tmp_path / "sessions" / f"bounded{index}", "test/test-model"
        )
        assert session_uses_test_hosting(directory) is True

    assert len(model_selection._HOSTING_VERDICT_CACHE) == 2


def test_a_walk_that_raised_is_not_memoised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2-1: only an OPEN failure was exempt, and the operator's banner came back.

    Reproduced as the reviewer found it. One injected ``OSError(24)`` (EMFILE —
    a real condition under a loaded box) landed DURING the walk of a journal
    whose newest row is ``test/test-model``; the exception was swallowed,
    answered ``False`` and MEMOISED, so every later healthy read of the
    byte-identical file returned that ``False`` — a mock session that stops
    being recognised as one and banners, which is the defect this whole change
    exists to close, served for the life of the file.

    Asserted on the COUNT of real walks and on the cache's contents, not only on
    the answer: an answer alone cannot tell a fresh walk from a stale memo, and
    "was it cached" is exactly the property that broke.
    """
    from local_operator.session import model_selection

    directory = _transcript_with_selection(tmp_path / "sessions" / "walkfail01", "test/test-model")
    walks: list[Path] = []
    real = model_selection._settled_selection
    failing = {"armed": True}

    def flaky(path: Path) -> Any:
        walks.append(path)
        if failing["armed"]:
            raise OSError(24, "Too many open files")
        return real(path)

    monkeypatch.setattr(model_selection, "_settled_selection", flaky)
    model_selection._HOSTING_VERDICT_CACHE.clear()

    # The transient failure answers the tolerant direction...
    assert session_uses_test_hosting(directory) is False
    assert len(walks) == 1, walks
    # ...and is NOT memoised, because it describes the moment rather than the
    # file. This is the assertion the defect failed: the poisoned `False` was
    # stored here and every later read was answered from it.
    assert directory not in model_selection._HOSTING_VERDICT_CACHE

    # The next healthy read WALKS AGAIN over the byte-identical journal and gets
    # the real answer — the mock session is recognised.
    failing["armed"] = False
    assert session_uses_test_hosting(directory) is True
    assert len(walks) == 2, walks


def test_a_failed_reopen_is_not_memoised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Q1: the REOPEN the reader takes was still collapsed into a cached verdict.

    Named for what it arms. There are three descriptors on this path — the
    caller's pre-open (#1), the reader's reopen (#2), and ``_settled_selection``'s
    own open (#3) — and this cell fails #2, which is the route the reopen
    exists to cover: the caller saw the journal readable and it was not by the
    time the reader looked, the EMFILE window QA round 3 reproduced. That
    failure used to be folded into the tolerant ``False`` and MEMOISED, serving
    a stale "not test-hosted" verdict for the life of the journal.

    IT DOES NOT COVER #3 (QA round 4, Q5): ``_settled_selection`` answers
    ``None`` for a journal it cannot open AND for one that opens with no v2 row,
    so a failure of the walk's own open is still collapsed and memoised. Closing
    that needs the conflation in ``_settled_selection`` changed, whose second
    caller relies on ``None`` meaning "fall back to the fold"; the reader's
    docstring states the limit.

    The assertions are on the cache, on the count of opens and on the count of
    real walks rather than on the answer alone, so the cell cannot pass
    vacuously: `walks == []` after the failed read is itself the evidence that
    the walk was never reached.
    """
    from local_operator.session import model_selection

    directory = _transcript_with_selection(tmp_path / "sessions" / "openfail01", "test/test-model")
    journal = directory / "transcript.jsonl"
    real_open = Path.open
    opens: list[str] = []
    walks: list[Path] = []
    armed = {"fail_next": True}

    def counted_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self == journal:
            opens.append("open")
            if armed["fail_next"] and len(opens) == 2:
                raise OSError(24, "Too many open files")
        return real_open(self, *args, **kwargs)

    real_settled = model_selection._settled_selection

    def counted_settled(path: Path) -> Any:
        walks.append(path)
        return real_settled(path)

    monkeypatch.setattr(Path, "open", counted_open)
    monkeypatch.setattr(model_selection, "_settled_selection", counted_settled)
    model_selection._HOSTING_VERDICT_CACHE.clear()

    # The reopen fails (the first open was the caller's pre-open). The walk is
    # never reached, the tolerant `False` is answered, and NOTHING is memoised.
    assert session_uses_test_hosting(directory) is False
    opened_for_the_failed_read = len(opens)
    assert opened_for_the_failed_read == 2, opens
    assert walks == [], walks
    assert directory not in model_selection._HOSTING_VERDICT_CACHE

    # Healthy again: a FRESH walk happens on the byte-identical journal and the
    # mock session is recognised — which a memoised `False` would have hidden.
    armed["fail_next"] = False
    assert session_uses_test_hosting(directory) is True
    assert len(opens) > opened_for_the_failed_read, opens
    assert len(walks) == 1, walks


def test_an_unreadable_or_selection_free_journal_fails_toward_notifying(tmp_path: Path) -> None:
    """Silencing a REAL session's banner is worse than bannering a test one."""
    assert session_uses_test_hosting(tmp_path / "missing") is False
    directory = tmp_path / "sessions" / "empty01"
    directory.mkdir(parents=True)
    assert session_uses_test_hosting(directory) is False
    (directory / "transcript.jsonl").write_text("{not json\n", encoding="utf-8")
    assert session_uses_test_hosting(directory) is False


def test_the_harness_escape_waives_the_rule_and_can_only_enable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seam CI forced, and the one direction it is allowed to move.

    Three answers over ONE journal, in the order that matters. A plain reader
    says True. The escape — the documented opt-in a suite whose SUBJECT is the
    notification path needs, because the desktop legs can otherwise observe
    nothing over a fixture-built store — says False. And it is read FRESH, so
    clearing it puts the rule straight back.

    The last arm is the safety property, not decoration: the escape cannot
    silence anyone. It only ever answers "not a test session", and a process the
    kill switch owns stays disabled with it set, because every leg asks the
    switch first. That is what keeps a rig from putting a banner on the
    operator's screen merely by setting it.
    """
    from local_operator.session import model_selection as selection
    from local_operator.tui.notify import notifications_enabled

    directory = _transcript_with_selection(tmp_path / "sessions" / "escape01", "test/test-model")
    assert selection.session_uses_test_hosting(directory) is True

    monkeypatch.setenv(selection.ENV_ALLOW_TEST_HOSTING_NOTIFY, "1")
    assert selection.session_uses_test_hosting(directory) is False

    monkeypatch.delenv(selection.ENV_ALLOW_TEST_HOSTING_NOTIFY)
    assert selection.session_uses_test_hosting(directory) is True

    monkeypatch.setenv(selection.ENV_ALLOW_TEST_HOSTING_NOTIFY, "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    assert notifications_enabled() is False


def test_the_composed_body_of_a_mock_session_is_the_banner_that_was_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The symptom itself, pinned: this is why the mock had to be gated.

    A completion's body is a snippet of the session's OWN last assistant line,
    and a mock session's last line is always the mock's canned reply — so the
    banner a leaked mock session raises is not a mystery string, it is exactly
    this. Asserted through the real composer against a real transcript row, and
    it fails the day either half changes, which is the day the recognisable body
    of a leaked banner becomes something else.
    """
    monkeypatch.setattr("local_operator.tui.notify.settings_get", lambda key, default=None: True)
    directory = tmp_path / "sessions" / "mock0002"
    directory.mkdir(parents=True)
    rows = [
        {
            "id": "assistant-1",
            "ts": 2.0,
            "type": "message",
            "payload": {
                "kind": "message",
                "role": "assistant",
                "content": [{"text": "Hello from the mock provider!"}],
            },
        }
    ]
    with (directory / "transcript.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    composition = compose("complete", session_dir=directory, session_name="")
    assert composition.body == "Hello from the mock provider!"
    assert composition.body_is_snippet is True


# ---------------------------------------------------------------------------
# Drift guards
# ---------------------------------------------------------------------------


def _modules_that_gate_their_children(stages: tuple[Path, ...] | None = None) -> list[Path]:
    """The sweep's population: modules that hand a child a built environment."""
    population: list[Path] = []
    for stage in _GATED_STAGES if stages is None else stages:
        for path in sorted(stage.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            spawns = False
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "attr", "") or getattr(node.func, "id", "")
                if not name or not _SPAWNER_NAME.search(name):
                    continue
                if any(keyword.arg == "env" for keyword in node.keywords):
                    spawns = True
                    break
            if not spawns:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if any(entry in node.value for entry in _ENTRY_POINTS):
                        population.append(path)
                        break
    return population


def test_every_child_environment_builder_gates_notifications() -> None:
    """A bespoke child environment must carry the gate ITSELF.

    NOT "the ambient one will do": these builders hand a child a mapping they
    built, deliberately stripped of the families that let a runtime address the
    operator's live panes, and the children are real ``lop`` processes that can
    park a gate and announce it. The switch names no pane or store, so a strip
    never removes it — but inheriting it by accident is not the property this
    wants, because the next builder may build from scratch.

    THIS HALF IS TEXTUAL, and the PR must not read it as proof of behaviour:
    it checks that each builder MENTIONS one of the sanctioned carriers. What
    makes that enough is the pair —
    ``test_a_harness_child_reports_notifications_disabled`` drives both
    carriers into a real child and reads ``notifications_enabled()`` there — and
    the residue is stated rather than hidden: a builder that mentions a carrier
    in a COMMENT passes this sweep, so the sweep is a drift tripwire against a
    new builder forgetting the gate, not a proof about any individual module.
    """
    population = _modules_that_gate_their_children()
    assert population, "the sweep found no builders; the predicate has rotted"
    offenders = [
        path.relative_to(Path(__file__).resolve().parents[2])
        for path in population
        if not any(spelling in path.read_text(encoding="utf-8") for spelling in _GATE_SPELLINGS)
    ]
    assert not offenders, (
        "these modules spawn a local-operator child with a built environment and "
        "never set the notification gate, so a mock session they drive can put "
        "its own reply on the operator's lock screen:\n"
        + "\n".join(f"  {path}" for path in offenders)
    )


def test_the_walker_sees_a_builder_that_forgets_the_gate(tmp_path: Path) -> None:
    """Prove the guard can still fail: a builder with no gate is an offender."""
    offending = tmp_path / "test_forgot.py"
    offending.write_text(
        "import os\n"
        "import sys\n"
        "import subprocess\n"
        "def _child_env(config):\n"
        "    return {**os.environ, 'LOCAL_OPERATOR_CONFIG_DIR': str(config)}\n"
        "def go(config):\n"
        "    return subprocess.run(\n"
        "        [sys.executable, '-m', 'local_operator.cli', 'exec', 'hi'],\n"
        "        env=_child_env(config),\n"
        "    )\n",
        encoding="utf-8",
    )
    assert _modules_that_gate_their_children(stages=(tmp_path,)) == [offending]


def test_the_test_harness_and_the_product_declare_the_same_switch() -> None:
    """The suite's child gate and the product's are ONE switch, not two.

    ``tests/e2e`` builds filtered child environments from scratch, so it cannot
    use ``harness_child_env`` (which is for a script driving the real CLI, and
    would inject the nested-session marker these tests must not carry); it keeps
    its own mapping. This pins the pair together, so a rename on the product
    side cannot leave the suite's children silently un-gated.
    """
    from tests.e2e.harness import NO_NOTIFY_ENV

    assert NO_NOTIFY_ENV[ENV_DISABLE] == ENV_DISABLE_VALUE


def test_harness_child_env_carries_the_gate() -> None:
    """A harness's child is a session nobody is watching, so it is gated.

    ``harness_child_env`` is the one place a script under ``scripts/`` says "I
    am a harness driving the real CLI", which makes it the right carrier: the
    benches seed ``hosting: test``, whose reply is the mock's own sentence, and
    a notification body is a snippet of the session's last assistant line. It
    must also not clobber what the caller already decided.
    """
    from local_operator.agent_shell import harness_child_env

    env = harness_child_env({"PATH": "/usr/bin"})

    assert env[ENV_DISABLE] == ENV_DISABLE_VALUE
    assert env["PATH"] == "/usr/bin"


def test_a_harness_child_reports_notifications_disabled() -> None:
    """The claim in a REAL child, which is the process that decides.

    Behavioural rather than textual: the switch is read from the environment of
    whoever composes a banner, and ``harness_child_env``'s whole job is to be
    that environment. The suite's own mapping is driven the same way, from a
    base environment with the switch REMOVED, so the mapping is the only thing
    that can have silenced it — which is what makes the sweep below ("every
    builder carries one of these") add up to behaviour rather than to a
    mention. The control arm (a child with neither) is not asserted here: it
    would answer from the developer's own ``display.notifications`` flag, and
    the flag path is covered in-process by
    ``test_a_real_hosting_leaves_the_gate_alone``.
    """
    import subprocess
    import sys

    from local_operator.agent_shell import harness_child_env
    from tests.e2e.harness import NO_NOTIFY_ENV

    code = (
        "from local_operator.tui.notify import notifications_enabled; "
        "print(notifications_enabled())"
    )
    stripped = {key: value for key, value in os.environ.items() if key != ENV_DISABLE}
    arms = {
        "harness_child_env": harness_child_env(),
        "NO_NOTIFY_ENV": {**stripped, **NO_NOTIFY_ENV},
    }
    for label, env in arms.items():
        assert env[ENV_DISABLE] == ENV_DISABLE_VALUE, label
        child = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        assert child.returncode == 0, (label, child.stderr)
        assert child.stdout.strip() == "False", (label, child.stdout)


def test_the_sandboxes_carry_the_switch_by_literal() -> None:
    """The two capture sandboxes set the switches BEFORE any product import.

    That is their whole contract (``probe_isolation`` raises if anything under
    ``local_operator`` is already loaded), so they cannot ask the product for the
    names. They spell them, and this pins the spelling to
    ``tui.notify.ENV_DISABLE`` and ``tui.resume_click.DESKTOP_LAUNCH_REFUSED_ENV``
    — next to the constants they must match, which is where a rename is read.
    """
    from local_operator.tui.resume_click import DESKTOP_LAUNCH_REFUSED_ENV

    root = Path(__file__).resolve().parents[2]
    for module in ("scripts/probe_isolation.py", "scripts/visual_capture.py"):
        source = (root / module).read_text(encoding="utf-8")
        for name in (ENV_DISABLE, DESKTOP_LAUNCH_REFUSED_ENV):
            assert name in source, f"{module} does not carry {name}"


def test_the_capture_sandbox_turns_every_gate_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """``isolate_capture`` is the sandbox 68 shot scripts actually call.

    ``probe_isolation`` always set these switches; this one did not, so every
    capture that booted the real app was a notification surface — which is how a
    screenshot run ends up putting fixture text in Notification Centre. Driven
    for real: the function is called and the environment is read back.
    """
    from local_operator.tui.resume_click import DESKTOP_LAUNCH_REFUSED_ENV
    from scripts import visual_capture

    names = (ENV_DISABLE, DESKTOP_LAUNCH_REFUSED_ENV)
    for name in names:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(visual_capture, "_SANDBOX", None)

    visual_capture.isolate_capture()

    for name in names:
        assert os.environ[name] == "1", name
    # HOME comes with it, which is the rest of the sandbox.
    assert "lop-visual-" in os.environ["HOME"]
