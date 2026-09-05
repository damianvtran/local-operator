"""Seeing and steering FAST MODE from the app: the band, the command, the spec.

The sibling of ``test_effort.py`` and deliberately shaped like it, because the
two dials sit on the same request and the failure modes are the same: a band
that agrees with the app's own variable while disagreeing with the spec the
request is built from.

Every assertion about "what is in force" therefore reads the SPEC the session
would send, not the app's remembered choice. The distinction matters more here
than for effort: this dial is billed at a premium, so a band claiming ``fast``
over a request that is not fast is a claim about the user's money.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.model.configure import build_model_spec
from local_operator.tui.app import OperatorApp
from local_operator.tui.widgets.status_line import ICON_FAST
from tests.unit.tui.test_app_pilot import FakeSession, _band, _factory
from tests.unit.tui.test_effort import _boot, _notices, _submit


class FastSession(FakeSession):
    """A session carrying a REAL ``ModelSpec``, so the support flag is the shipped one.

    Same reason ``EffortSession`` exists: ``FakeSession.model`` is ``None`` and
    its ``set_model`` a no-op, which would make every assertion here vacuous.
    """

    def __init__(self, provider: str = "anthropic", model_id: str = "claude-opus-5") -> None:
        super().__init__()
        self._spec = build_model_spec(provider, model_id)

    @property
    def model_label(self) -> str:
        return f"{self._spec.provider}/{self._spec.model_id}"

    @property
    def model(self) -> Any:
        return self._spec

    def set_model(self, model: Any, *, explicit: bool = False) -> None:
        self._spec = model


def _fast(app: OperatorApp) -> bool:
    """Whether the session would SEND the speed key on its next request."""
    assert app._session is not None, "the session must be up before reading its spec"
    return bool(getattr(app._session.model, "fast_mode", False))


@pytest.mark.asyncio
async def test_fast_turns_the_dial_on_in_the_spec_the_request_is_built_from() -> None:
    """The toggle has to reach the REQUEST, not just the transcript."""
    app = OperatorApp(lambda: _factory(FastSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        assert _fast(app) is False, "a premium dial must never start on"

        await _submit(pilot, app, "/fast")
        assert _fast(app) is True


@pytest.mark.asyncio
async def test_bare_fast_toggles_back_off() -> None:
    """Bare `/fast` is a TOGGLE: the same keystrokes must reverse it."""
    app = OperatorApp(lambda: _factory(FastSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/fast")
        assert _fast(app) is True

        await _submit(pilot, app, "/fast")
        assert _fast(app) is False


@pytest.mark.asyncio
async def test_on_and_off_name_the_resulting_state_rather_than_flipping() -> None:
    """`/fast on` twice must stay on — an argument names a state, not a flip."""
    app = OperatorApp(lambda: _factory(FastSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/fast on")
        assert _fast(app) is True
        await _submit(pilot, app, "/fast on")
        assert _fast(app) is True

        await _submit(pilot, app, "/fast off")
        assert _fast(app) is False
        await _submit(pilot, app, "/fast off")
        assert _fast(app) is False


@pytest.mark.asyncio
async def test_turning_it_on_names_the_premium_in_the_receipt() -> None:
    """The trade is money for latency, so a receipt naming only speed sells half of it."""
    app = OperatorApp(lambda: _factory(FastSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/fast")

        receipt = _notices(app)[-1]
        assert "premium" in receipt.lower()
        assert "on" in receipt.lower()


@pytest.mark.asyncio
async def test_the_band_shows_fast_only_while_it_is_on() -> None:
    """The segment's PRESENCE is the message: off must render nothing.

    A band that also printed `standard` would spend permanent width on the
    state every session is in by default.
    """
    app = OperatorApp(lambda: _factory(FastSession()))
    async with app.run_test(size=(160, 40)) as pilot:
        await _boot(pilot, app)
        # Asserted on the SEGMENT (icon + word), not on the bare word: this
        # worktree's own cwd contains "fast", and the cwd is painted on the
        # same band — a substring check passed for the wrong reason.
        segment = f"{ICON_FAST} fast"
        assert segment not in _band(app)

        await _submit(pilot, app, "/fast")
        assert segment in _band(app)

        await _submit(pilot, app, "/fast off")
        assert segment not in _band(app)


@pytest.mark.asyncio
async def test_a_route_without_a_fast_tier_says_so_and_changes_nothing() -> None:
    """Says so rather than accepting a toggle the wire would silently drop.

    `google/gemini-3-pro` speaks `generateContent`, whose request carries no
    service-tier field at all, so any state taken here would be a claim the
    request does not back.
    """
    app = OperatorApp(lambda: _factory(FastSession("google", "gemini-3-pro")))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/fast")

        assert _fast(app) is False
        assert "not available" in _notices(app)[-1].lower()


@pytest.mark.asyncio
async def test_status_reports_without_changing_anything() -> None:
    """The read-only question belongs beside the toggle, not behind another command."""
    app = OperatorApp(lambda: _factory(FastSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/fast status")
        assert _fast(app) is False
        assert "off" in _notices(app)[-1].lower()

        await _submit(pilot, app, "/fast on")
        await _submit(pilot, app, "/fast status")
        assert _fast(app) is True, "status must REPORT, never toggle"
        assert "on" in _notices(app)[-1].lower()


@pytest.mark.asyncio
async def test_an_unparseable_argument_is_refused_rather_than_guessed() -> None:
    """`/fast maybe` must not silently toggle: a premium dial gets no guesses."""
    app = OperatorApp(lambda: _factory(FastSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/fast maybe")

        assert _fast(app) is False
        assert "not one of on, off, status" in _notices(app)[-1].lower()


@pytest.mark.asyncio
async def test_the_speed_dial_does_not_disturb_the_depth_dial() -> None:
    """The two dials are orthogonal and must not read or write each other."""
    app = OperatorApp(lambda: _factory(FastSession()))
    async with app.run_test(size=(120, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/effort low")
        assert app._session is not None
        assert app._session.model.reasoning_effort == "low"

        await _submit(pilot, app, "/fast")
        assert _fast(app) is True
        assert app._session.model.reasoning_effort == "low", "fast mode moved the effort rung"


@pytest.mark.asyncio
async def test_the_remote_control_path_applies_the_dial_too() -> None:
    """`/fast` from a phone must reach the OWNER's spec, not a local no-op.

    The dial lives on the spec the owner builds its requests from, so it is an
    AUTHORITATIVE command rather than a frontend-local one (`/theme` and
    `/settings` are local because they act on the machine the user sits at).
    Without a producer here the capability is advertised and then answered with
    an "unsupported" warning — a success-shaped surface over an operation that
    never ran.
    """
    app = OperatorApp(lambda: _factory(FastSession()))
    async with app.run_test(size=(120, 34)) as pilot:
        await _boot(pilot, app)
        assert app._session is not None

        result = await app._slash_result("fast", "on", None)
        assert _fast(app) is True
        assert "premium" in result.text.lower()

        # `on` again REPORTS rather than toggling back, the same rule the
        # terminal path follows.
        again = await app._slash_result("fast", "on", None)
        assert _fast(app) is True
        assert "on" in again.text.lower()

        off = await app._slash_result("fast", "off", None)
        assert _fast(app) is False
        assert "standard" in off.text.lower()


@pytest.mark.asyncio
async def test_the_command_is_advertised_to_remote_clients() -> None:
    """A phone can only offer what the capability list names."""
    from local_operator.session.frontend_state import _slash_capabilities

    fast = [cap for cap in _slash_capabilities() if cap.command == "fast"]
    assert len(fast) == 1, "/fast must be advertised exactly once"
    # Authoritative: it mutates the owner's spec, so it cannot be frontend-local.
    assert fast[0].operation == "slash"


@pytest.mark.asyncio
async def test_a_provider_refusal_takes_the_dial_off_the_band_and_the_memory() -> None:
    """The app's half of review F1.

    When the session switches its own dial off after a provider refusal, the
    band must stop painting `fast` — and the app's REMEMBERED choice must be
    dropped too, or the next `/new`/`/model` would re-arm a tier the user was
    just told they cannot have.
    """
    app = OperatorApp(lambda: _factory(FastSession()))
    async with app.run_test(size=(160, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/fast")
        assert _fast(app) is True and app._fast_choice is True
        segment = f"{ICON_FAST} fast"
        assert segment in _band(app)

        # What `Session._on_fast_refused` does, without a provider in the loop:
        # the spec's dial comes off and the frontend state is republished.
        session = app._session
        assert session is not None
        session.set_model(session.model.model_copy(update={"fast_mode": False}))
        assert app._status is not None
        app._status.update(fast=_fast_label_for_test(session))
        app._pending_frontend_state = _State(session.model)
        app._apply_pending_frontend_state(getattr(app, "_frontend_session_generation", 0))
        for _ in range(4):
            await pilot.pause()

        assert segment not in _band(app)
        assert app._fast_choice is False, "the remembered choice must follow the spec"


class _State:
    """The two fields `_apply_frontend_state` reads for the reconciliation."""

    def __init__(self, selected: Any) -> None:
        self.selected_model = selected
        self.effective_model = selected
        self.effective_model_label = f"{selected.provider}/{selected.model_id}"
        self.jobs: list[Any] = []
        self.mcp_servers: list[Any] = []


def _fast_label_for_test(session: Any) -> str:
    from local_operator.tui.app import _fast_label

    return _fast_label(session)


class PublishingFastSession(FastSession):
    """A ``FastSession`` that publishes frontend state like the real Session.

    The round-2 F6 defect only reproduces when the app receives a frontend
    snapshot on adoption: `FastSession` publishes nothing, so a test built on
    it cannot see the reconciliation reading a fresh session's spec (dial
    off by default) as a provider refusal.
    """

    def __init__(self, provider: str = "anthropic", model_id: str = "claude-opus-5") -> None:
        super().__init__(provider, model_id)
        from local_operator.session.frontend_state import FrontendSessionState, FrontendStateStore

        self._store = FrontendStateStore(
            FrontendSessionState(session_id="fast-pub", epoch="e1", selected_model=self._spec)
        )

    @property
    def frontend_state(self) -> Any:
        return self._store.state

    def subscribe_frontend(self, handler: Any) -> Any:
        return self._store.subscribe(handler)

    def set_model(self, model: Any, *, explicit: bool = False) -> None:
        self._spec = model
        self._store.mutate(selected_model=model, effective_model=model)


@pytest.mark.asyncio
async def test_the_choice_survives_a_session_rebuild_that_publishes_state() -> None:
    """Round-2 F6: `/reload` on a state-publishing session must keep the dial.

    The adoption snapshot is painted BEFORE the remembered choice is restored
    onto the fresh spec, and that spec's dial defaults off — reconciling
    against the snapshot dropped the choice on every `/new` and `/reload`.
    """
    sessions = [PublishingFastSession(), PublishingFastSession()]
    app = OperatorApp(lambda: _factory(sessions.pop(0)))
    async with app.run_test(size=(160, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/fast")
        assert _fast(app) is True
        app._session_factory = lambda: _factory(sessions.pop(0))  # type: ignore[assignment]
        await app._reload_session(keep_context=True)
        for _ in range(40):
            await pilot.pause()
            if app._session is not None and not sessions:
                break
        for _ in range(6):
            await pilot.pause()
        assert _fast(app) is True, "the rebuild dropped the dial"
        assert app._fast_choice is True
        assert f"{ICON_FAST} fast" in _band(app)


@pytest.mark.asyncio
async def test_a_published_refusal_drops_the_remembered_choice() -> None:
    """The reconciliation still does its job on an ORDERED update: when the
    session takes the dial off (as `Session._on_fast_refused` does) after
    adoption, the app forgets the choice so the next rebuild cannot re-arm it."""
    app = OperatorApp(lambda: _factory(PublishingFastSession()))
    async with app.run_test(size=(160, 40)) as pilot:
        await _boot(pilot, app)
        await _submit(pilot, app, "/fast")
        assert app._fast_choice is True
        session = app._session
        assert session is not None
        session.set_model(session.model.model_copy(update={"fast_mode": False}))
        for _ in range(6):
            await pilot.pause()
        assert app._fast_choice is False
        assert f"{ICON_FAST} fast" not in _band(app)
