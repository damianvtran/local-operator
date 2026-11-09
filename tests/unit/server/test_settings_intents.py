"""Durable desktop settings intents: the routes, the store and the applier.

What is pinned, and why:

* The ROUTES never call the runtime (the fake driver's ``apply`` is not reached
  by a POST), answer 202 for a new pick and 200 for a replayed ``request_id``,
  and refuse an unknown session (404), a malformed shape (422) and a stale
  cancel (409 ``generation_moved``).
* COALESCING is last-wins with a session-wide generation: an attempt in flight
  for an older generation is discarded and the newer pick runs next.
* SUPERSEDE: a selection changed elsewhere (equal to neither base nor value)
  stops the pick rather than overwriting the newer choice.
* The RETRY SCHEDULE (0, 1, 2, 4, 8, 15, then 30 s) and the budget run on a fake
  clock, so the numbers are asserted exactly rather than slept through.
* RESUME: a slot on disk from a previous process is re-armed and applied.
* cwd: a busy owner parks the slot in ``waiting_idle`` (every 5 s, no budget);
  another viewer fails it with ``move_exclusive`` after 120 s.
* The machine-wide CONCURRENCY cap holds across a burst of sessions.

Every test runs on its own ``tmp_path`` config root with ``CMUX_*`` stripped; no
runtime is spawned here (``tests/e2e/test_settings_intents_e2e.py`` does that).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from local_operator.server.routes import desktop_settings_intents
from local_operator.server.utils.settings_intents import (
    INTENT_FILE,
    RETRY_SCHEDULE_S,
    Current,
    Outcome,
    SettingsApplier,
    classify_exception,
    retry_delay,
)

SID = "a1b2c3d4e5f6"
TOKEN = "test-desktop-token"


class FakeClock:
    """Wall time that only moves when the applier sleeps."""

    def __init__(self) -> None:
        self.now = 1_800_000_000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(round(delay, 3))
        self.now += delay
        await asyncio.sleep(0)


class FakeDriver:
    """Scripted runtime side: ``outcomes`` per kind, consumed in order.

    ``current`` is mutable so a test can model the journal moving (our own
    apply landing, or someone else's ``/model``).
    """

    def __init__(self, outcomes: dict[str, list[Outcome]] | None = None) -> None:
        self.outcomes = outcomes or {}
        self.reading = Current(model="openai/gpt-6", effort="medium", fast=False, cwd="/old")
        self.calls: list[tuple[str, str, Any, str | None]] = []
        self.gate: asyncio.Event | None = None
        self.in_flight = 0
        self.peak = 0

    async def current(self, session_id: str) -> Current:
        return self.reading

    async def apply(self, session_id: str, kind: str, value: Any, effort: str | None) -> Outcome:
        self.calls.append((session_id, kind, value, effort))
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            if self.gate is not None:
                await self.gate.wait()
            queue = self.outcomes.get(kind) or []
            outcome = queue.pop(0) if queue else Outcome("applied")
            if outcome.status == "applied":
                if kind == "model":
                    self.reading = Current(
                        model=str(value),
                        effort=effort or self.reading.effort,
                        fast=False,
                        cwd=self.reading.cwd,
                    )
                else:
                    fields = {**self.reading.__dict__, kind: value}
                    self.reading = Current(**fields)
            return outcome
        finally:
            self.in_flight -= 1


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("CMUX_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", TOKEN)
    monkeypatch.delenv("LOP_SETTINGS_INTENT_BUDGET_S", raising=False)


def _session(root: Path, sid: str = SID) -> Path:
    directory = root / "sessions" / sid
    directory.mkdir(parents=True, exist_ok=True)
    return directory


class RecordingApplier(SettingsApplier):
    """The real applier, with every published frame recorded for assertions."""

    frames: list[tuple[str, str, Any]]


def _applier(
    root: Path, driver: FakeDriver, clock: FakeClock | None = None, **kw
) -> RecordingApplier:
    frames: list[tuple[str, str, Any]] = []
    applier = RecordingApplier(
        root,
        driver,
        publish=lambda sid, kind, intent: frames.append((sid, kind, intent)),
        clock=clock or FakeClock(),
        sleep=(clock.sleep if clock else None),
        **kw,
    )
    applier.frames = frames
    return applier


class FrozenClock(FakeClock):
    """Time that never moves: sleeps yield for real but no deadline approaches."""

    async def sleep(self, delay: float) -> None:
        self.sleeps.append(round(delay, 3))
        await asyncio.sleep(0.005)


async def _settle(applier: SettingsApplier, sid: str = SID, limit: float = 5.0) -> None:
    """Wait for the session's worker to finish (all slots terminal and pruned/idle)."""
    state = applier._sessions.get(sid)
    if state is None or state.task is None:
        return
    await asyncio.wait_for(asyncio.shield(state.task), timeout=limit)


async def _until(predicate, limit: float = 3.0) -> None:
    """Poll in REAL time: the store hops to a worker thread, so ``sleep(0)`` loops
    cannot promise the worker reached its next await."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limit
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.01)


async def _submit(applier: SettingsApplier, kind: str, value: Any, rid: str, **kw):
    return await applier.submit(
        SID,
        kind=kind,
        value=value,
        effort=kw.get("effort"),
        request_id=rid,
        base=kw.get("base", (await applier.driver.current(SID)).of(kind)),
    )


# -- routes -------------------------------------------------------------------


@pytest_asyncio.fixture
async def api(tmp_path: Path, monkeypatch):
    from local_operator.config import ConfigManager

    _session(tmp_path)
    app = FastAPI()
    app.state.config_manager = ConfigManager(tmp_path)
    driver = FakeDriver()
    driver.gate = asyncio.Event()  # the POST must answer while the runtime is stuck
    app.state.settings_applier = _applier(tmp_path, driver)
    app.include_router(desktop_settings_intents.router)

    async def accept(provider_model: str, effort: str | None) -> None:
        if provider_model == "nope/missing":
            from fastapi import HTTPException

            raise HTTPException(422, {"code": "model_unknown", "text": "unknown"})

    monkeypatch.setattr(desktop_settings_intents, "_validate_model", accept)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://localhost",
        headers={"Authorization": f"Bearer {TOKEN}"},
    ) as client:
        yield client, app, driver
    driver.gate.set()
    await app.state.settings_applier.close()


@pytest.mark.asyncio
async def test_post_answers_202_at_once_and_a_replay_answers_200(api, tmp_path) -> None:
    client, app, driver = api
    body = {"request_id": "r1", "kind": "model", "value": "anthropic/claude-x", "effort": "high"}
    first = await asyncio.wait_for(
        client.post(f"/v1/desktop/sessions/{SID}/settings-intents", json=body), timeout=2
    )
    assert first.status_code == 202, first.text
    slot = first.json()["result"]["intent"]
    assert slot["kind"] == "model" and slot["value"] == "anthropic/claude-x"
    assert slot["effort"] == "high"
    assert slot["base"] == "openai/gpt-6"
    assert slot["state"] in {"pending", "applying"}
    assert slot["generation"] == 1
    replay = await client.post(f"/v1/desktop/sessions/{SID}/settings-intents", json=body)
    assert replay.status_code == 200
    assert replay.json()["result"]["intent"]["generation"] == 1
    # Durable: the file holds the slot before anything is applied.
    stored = json.loads((tmp_path / "sessions" / SID / INTENT_FILE).read_text())
    assert stored["slots"]["model"]["request_id"] == "r1"
    # GET is the baseline every client hydrates from.
    listed = await client.get("/v1/desktop/settings-intents")
    assert listed.status_code == 200
    assert listed.json()["result"]["intents"][SID]["model"]["request_id"] == "r1"
    scoped = await client.get("/v1/desktop/settings-intents", params={"session_id": "ffffffffffff"})
    assert scoped.json()["result"]["intents"] == {}


@pytest.mark.asyncio
async def test_post_refusals_404_and_422(api) -> None:
    client, _, _ = api
    url = f"/v1/desktop/sessions/{SID}/settings-intents"
    missing = await client.post(
        "/v1/desktop/sessions/ffffffffffff/settings-intents",
        json={"request_id": "r", "kind": "fast", "value": True},
    )
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "session_unknown"
    for bad in (
        {"request_id": "r", "kind": "colour", "value": "x"},
        {"request_id": "r", "kind": "model", "value": ""},
        {"request_id": "r", "kind": "fast", "value": "on"},
        {"request_id": "r", "kind": "effort", "value": "high", "effort": "low"},
    ):
        response = await client.post(url, json=bad)
        assert response.status_code == 422, bad
    unknown = await client.post(
        url, json={"request_id": "r", "kind": "model", "value": "nope/missing"}
    )
    assert unknown.status_code == 422
    assert unknown.json()["detail"]["code"] == "model_unknown"


@pytest.mark.asyncio
async def test_post_requires_the_desktop_credential(api) -> None:
    client, _, _ = api
    response = await client.post(
        f"/v1/desktop/sessions/{SID}/settings-intents",
        json={"request_id": "r", "kind": "fast", "value": True},
        headers={"Authorization": "Bearer wrong"},
    )
    assert response.status_code in {401, 403}


@pytest.mark.asyncio
async def test_delete_cancels_and_a_stale_generation_is_409(api) -> None:
    client, _, _ = api
    url = f"/v1/desktop/sessions/{SID}/settings-intents"
    await client.post(url, json={"request_id": "a", "kind": "cwd", "value": "/"})
    newer = await client.post(url, json={"request_id": "b", "kind": "cwd", "value": "/"})
    generation = newer.json()["result"]["intent"]["generation"]
    stale = await client.delete(f"{url}/cwd", params={"generation": generation - 1})
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "generation_moved"
    assert stale.json()["detail"]["text"]
    ok = await client.delete(f"{url}/cwd", params={"generation": generation})
    assert ok.status_code == 204
    gone = await client.delete(f"{url}/cwd", params={"generation": generation})
    assert gone.status_code == 404


def test_the_capability_is_advertised() -> None:
    from local_operator.server.features import static_feature_flags

    assert static_feature_flags()["settings_intents"] == 1


# -- applier ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_and_effort_apply_paired_in_one_attempt(tmp_path) -> None:
    _session(tmp_path)
    driver = FakeDriver()
    driver.gate = asyncio.Event()
    clock = FakeClock()
    applier = _applier(tmp_path, driver, clock)
    await _submit(applier, "model", "anthropic/claude-x", "m1")
    await _submit(applier, "effort", "high", "e1")
    driver.gate.set()
    await _settle(applier)
    model_calls = [call for call in driver.calls if call[1] == "model"]
    assert model_calls[0][3] == "high", driver.calls
    assert not [call for call in driver.calls if call[1] == "effort"]
    assert driver.reading.model == "anthropic/claude-x" and driver.reading.effort == "high"
    # Applied slots publish, then are pruned after 60 s with a null frame.
    states = [(kind, intent and intent["state"]) for _, kind, intent in applier.frames]
    assert ("model", "applied") in states and ("effort", "applied") in states
    assert ("model", None) in states and ("effort", None) in states
    await applier.close()


@pytest.mark.asyncio
async def test_coalescing_discards_an_older_generation_in_flight(tmp_path) -> None:
    _session(tmp_path)
    driver = FakeDriver()
    driver.gate = asyncio.Event()
    applier = _applier(tmp_path, driver, FakeClock())
    await _submit(applier, "model", "a/one", "r1")
    await _until(lambda: driver.calls)
    assert driver.calls and driver.calls[0][2] == "a/one"
    # A newer pick lands while the first attempt is in flight.
    _, slot, _ = await _submit(applier, "model", "a/two", "r2", base="openai/gpt-6")
    assert slot["generation"] == 2
    driver.gate.set()
    await _settle(applier)
    assert [call[2] for call in driver.calls if call[1] == "model"] == ["a/one", "a/two"]
    finals = [
        intent for _, kind, intent in applier.frames if intent and intent["state"] == "applied"
    ]
    assert [intent["generation"] for intent in finals] == [2]
    await applier.close()


@pytest.mark.asyncio
async def test_a_selection_changed_elsewhere_supersedes_the_pick(tmp_path) -> None:
    _session(tmp_path)
    driver = FakeDriver()
    applier = _applier(tmp_path, driver, FakeClock())
    driver.reading = Current(model="openai/gpt-6")
    # Base recorded, then the TUI moves the session somewhere else before we apply.
    await applier.submit(
        SID, kind="model", value="a/mine", effort=None, request_id="r", base="openai/gpt-6"
    )
    driver.reading = Current(model="google/elsewhere")
    await _settle(applier)
    assert not driver.calls
    states = [intent and intent["state"] for _, _, intent in applier.frames]
    assert "superseded" in states and states[-1] is None
    assert (await applier.snapshot(SID)) == {}
    await applier.close()


@pytest.mark.asyncio
async def test_retry_schedule_and_budget_on_a_fake_clock(tmp_path) -> None:
    _session(tmp_path)
    busy = [Outcome("retry", "runtime_busy", "busy") for _ in range(100)]
    driver = FakeDriver({"model": busy})
    clock = FakeClock()
    applier = _applier(tmp_path, driver, clock, budget=100.0)
    await _submit(applier, "model", "a/b", "r")
    await _settle(applier)
    # 0 (immediate), then 1, 2, 4, 8, 15, 30, 30 … until the 100 s budget.
    waits = [delay for delay in clock.sleeps if delay > 0]
    assert waits[:6] == [1.0, 2.0, 4.0, 8.0, 15.0, 30.0], clock.sleeps
    assert sum(waits[:-1]) <= 100.0
    final = [intent for _, _, intent in applier.frames if intent][-1]
    assert final["state"] == "failed"
    assert final["error"]["code"] == "retry_budget_exhausted"
    assert final["error"]["last_code"] == "runtime_busy"
    assert final["error"]["retryable"] is True
    assert final["attempts"] == len(driver.calls)
    assert RETRY_SCHEDULE_S == (0.0, 1.0, 2.0, 4.0, 8.0, 15.0)
    assert [retry_delay(n) for n in range(8)] == [0.0, 1.0, 2.0, 4.0, 8.0, 15.0, 30.0, 30.0]
    await applier.close()


@pytest.mark.asyncio
async def test_budget_env_override(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOP_SETTINGS_INTENT_BUDGET_S", "30")
    applier = _applier(tmp_path, FakeDriver())
    assert applier.budget == 30.0
    monkeypatch.setenv("LOP_SETTINGS_INTENT_BUDGET_S", "junk")
    assert applier.budget == 600.0


@pytest.mark.asyncio
async def test_a_terminal_refusal_fails_at_once_and_a_new_post_retries(tmp_path) -> None:
    _session(tmp_path)
    driver = FakeDriver({"model": [Outcome("terminal", "model_unknown", "cannot switch")]})
    clock = FakeClock()
    applier = _applier(tmp_path, driver, clock)
    await _submit(applier, "model", "a/b", "r1")
    await _settle(applier)
    assert len(driver.calls) == 1
    slot = (await applier.snapshot(SID))[SID]["model"]
    assert slot["state"] == "failed"
    assert slot["error"] == {
        "code": "model_unknown",
        "message": "cannot switch",
        "retryable": False,
    }
    # One-click retry = a new POST with a fresh request id.
    await _submit(applier, "model", "a/b", "r2")
    await _settle(applier)
    assert len(driver.calls) == 2
    assert driver.reading.model == "a/b"
    await applier.close()


def test_classification_table() -> None:
    from fastapi import HTTPException

    from local_operator.model.configure import ModelSelectionRefused
    from local_operator.session.attached import RuntimeUnresponsiveError
    from local_operator.session.errors import RuntimeRetiring

    alive = ConnectionError("ack expired")
    alive.owner_alive = True  # type: ignore[attr-defined]
    table = [
        (RuntimeUnresponsiveError("x"), "retry", "runtime_busy"),
        (alive, "retry", "runtime_busy"),
        (ConnectionError("session is reconnecting"), "retry", "runtime_unreachable"),
        (TimeoutError(), "retry", "runtime_unreachable"),
        (RuntimeRetiring("leaving"), "retry", "runtime_retiring"),
        (HTTPException(503, {"code": "runtime_busy"}), "retry", "runtime_busy"),
        (HTTPException(404, {"code": "x"}), "terminal", "session_unknown"),
        (ModelSelectionRefused("unknown_model", "no such model"), "terminal", "model_unknown"),
        (ValueError("bad rung"), "terminal", "refused"),
    ]
    for error, status, code in table:
        outcome = classify_exception(error)
        assert (outcome.status, outcome.code) == (status, code), error


def test_slash_results_are_classified() -> None:
    from local_operator.server.utils.settings_intents import _slash_outcome

    assert _slash_outcome("model", {"kind": "notice", "style": "info"}).status == "applied"
    assert _slash_outcome("model", {"kind": "notice", "style": "error", "text": "x"}).code == (
        "model_unknown"
    )
    assert _slash_outcome("effort", {"kind": "notice", "style": "warning"}).code == (
        "effort_unsupported"
    )
    assert _slash_outcome("model", {"detail": "queued", "fields": {"queued": True}}).status == (
        "queued"
    )


@pytest.mark.asyncio
async def test_a_queued_ack_is_confirmed_from_the_journal(tmp_path) -> None:
    _session(tmp_path)
    driver = FakeDriver({"model": [Outcome("queued")]})
    # Frozen: the 60 s resend window must not elapse while the test observes
    # the queued state, or the "no resend" assertion would race the poller.
    applier = _applier(tmp_path, driver, FrozenClock())
    await _submit(applier, "model", "a/b", "r")
    await _until(
        lambda: any(
            intent and intent["state"] == "queued_next_step" for _, _, intent in applier.frames
        )
    )
    driver.reading = Current(model="a/b")  # the next step took it
    # Not ``_settle``: with frozen time the applied slot's 60 s prune never comes.
    await _until(
        lambda: any(intent and intent["state"] == "applied" for _, _, intent in applier.frames)
    )
    assert len(driver.calls) == 1  # no resend after a queued ack
    await applier.close()


@pytest.mark.asyncio
async def test_resume_rearms_a_slot_written_by_a_previous_process(tmp_path) -> None:
    _session(tmp_path)
    first = _applier(tmp_path, FakeDriver(), FakeClock())
    first._closed = True  # the old process dies before it applies anything
    await _submit(first, "fast", True, "r")
    stored = json.loads((tmp_path / "sessions" / SID / INTENT_FILE).read_text())
    stored["slots"]["fast"]["state"] = "applying"
    stored["slots"]["fast"]["attempts"] = 9
    (tmp_path / "sessions" / SID / INTENT_FILE).write_text(json.dumps(stored))

    driver = FakeDriver()
    second = _applier(tmp_path, driver, FakeClock())
    assert await second.resume() == 1
    await _settle(second)
    assert driver.calls == [(SID, "fast", True, None)]
    applied = [intent for _, _, intent in second.frames if intent and intent["state"] == "applied"]
    assert applied and applied[0]["attempts"] == 1  # attempts reset on resume
    await second.close()


@pytest.mark.asyncio
async def test_cwd_waits_for_idle_without_a_budget_then_applies(tmp_path) -> None:
    _session(tmp_path)
    busy = [Outcome("waiting_idle", "runtime_busy", "working") for _ in range(40)]
    driver = FakeDriver({"cwd": busy})
    clock = FakeClock()
    applier = _applier(tmp_path, driver, clock, budget=60.0)
    await _submit(applier, "cwd", "/new", "r")
    await _settle(applier, limit=30)
    # 40 refusals × 5 s = 200 s, far past the 60 s budget: still not failed.
    assert len(driver.calls) == 41
    assert set(delay for delay in clock.sleeps if delay > 0 and delay < 60) == {5.0}
    states = [intent["state"] for _, _, intent in applier.frames if intent]
    assert "waiting_idle" in states and "failed" not in states and states[-1] == "applied"
    await applier.close()


@pytest.mark.asyncio
async def test_cwd_blocked_by_another_viewer_fails_after_120_s(tmp_path) -> None:
    _session(tmp_path)
    viewed = [Outcome("exclusive", "move_exclusive", "open elsewhere") for _ in range(100)]
    driver = FakeDriver({"cwd": viewed})
    clock = FakeClock()
    applier = _applier(tmp_path, driver, clock)
    await _submit(applier, "cwd", "/new", "r")
    await _settle(applier)
    slot = (await applier.snapshot(SID))[SID]["cwd"]
    assert slot["state"] == "failed"
    assert slot["error"]["code"] == "move_exclusive"
    assert sum(delay for delay in clock.sleeps if delay > 0) >= 120.0
    assert len(driver.calls) <= 26
    await applier.close()


@pytest.mark.asyncio
async def test_at_most_four_attempts_in_flight_machine_wide(tmp_path) -> None:
    sids = [f"{index:012x}" for index in range(10)]
    for sid in sids:
        _session(tmp_path, sid)
    driver = FakeDriver()
    driver.gate = asyncio.Event()
    applier = _applier(tmp_path, driver)
    for sid in sids:
        await applier.submit(sid, kind="fast", value=True, effort=None, request_id="r", base=False)
    await _until(lambda: driver.in_flight == 4)
    await asyncio.sleep(0.2)  # the other six are parked on the cap, not racing in
    assert driver.in_flight == 4
    driver.gate.set()
    await _until(lambda: len(driver.calls) == 10)
    assert len(driver.calls) == 10
    assert driver.peak == 4 and applier.peak_in_flight == 4
    await applier.close()
