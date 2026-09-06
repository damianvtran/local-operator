"""The type-length bound, the per-character budget and the guest deadline agree.

Two real episodes died here, and the mechanism is worth stating once. The guest
types character by character, so a ``type`` action's cost is LINEAR in text
length; the deadline enforcing it is a CONSTANT socket timeout. While those two
numbers knew nothing about each other, the protocol admitted a 100000-character
``type`` that needs minutes, the batch was dispatched, the read timed out, and
the failure was ``GuestExecutionError`` — outcome unknown, never retried, so
fatal to an episode that had already cost real money.

Two defects had to be fixed and each has its own tests below:

* We passed ``interval=0.01`` to ``pyautogui.typewrite``, which is not a pacing
  hint but an unconditional ``time.sleep`` per character — a hard floor of 10 ms
  x len(text). ``test_the_interval_we_removed_really_did_cost_the_deadline``
  demonstrates that floor rather than asserting it from the fitted numbers.
* Nothing related ``MAX_TEXT_LENGTH`` to the timeout. ``MAX_TYPE_CHARS`` is now
  derived from the deadline and a named per-character budget, and the arithmetic
  tests fail if any one of the three moves without the others being reconsidered.
"""

from __future__ import annotations

import pytest
from lop_osworld_v2_adapter import actions
from lop_osworld_v2_adapter.providers.base import (
    GUEST_COMMAND_TIMEOUT_S,
    GUEST_TYPE_DEADLINE_FRACTION,
    GUEST_TYPE_MS_PER_CHAR,
)

from local_operator.evaluation.action_surface import ActionAdmissionError
from local_operator.evaluation.protocol import (
    MAX_TEXT_LENGTH,
    ActionBatch,
    ClickAction,
    TypeAction,
)
from tests.unit.evaluation.adapters.osworld.test_actions import _geo

# The two payloads that actually killed episodes ep-886aa2229672 and
# ep-762003dcc9c2, and the largest one that survived (2498 chars, 39.4 s).
# Real observations, not invented sizes: a bound that admits either fatal
# payload has not fixed the incident.
FATAL_PAYLOAD_CHARS = (6912, 7332)
LARGEST_SURVIVING_PAYLOAD_CHARS = 2498


def _batch(*actions_: object) -> ActionBatch:
    return ActionBatch(
        protocol_version="1.0",
        task_id="task",
        episode_id="episode",
        observation_id="obs",
        actions=tuple(actions_),  # type: ignore[arg-type]
    )


# ----------------------------------------------------------------------------
# The arithmetic: the three numbers cannot drift apart again
# ----------------------------------------------------------------------------


def test_the_bound_is_derived_from_the_deadline_and_the_per_character_budget() -> None:
    """Recompute the bound independently of the module that publishes it.

    This is the test that prevents recurrence. It does not check that typing is
    fast; it checks that the length we ADMIT and the deadline we ENFORCE are
    still expressions of one another, which is the property whose absence let a
    100000-character ``type`` be dispatched at all.
    """

    budgeted_seconds = actions.MAX_TYPE_CHARS * GUEST_TYPE_MS_PER_CHAR / 1000.0
    assert budgeted_seconds <= GUEST_COMMAND_TIMEOUT_S * GUEST_TYPE_DEADLINE_FRACTION
    # And it is the LARGEST length with that property: a bound that merely fits
    # would still pass if someone replaced the derivation with a smaller magic
    # number, which is the shape of the defect being guarded against.
    over_by_one = (actions.MAX_TYPE_CHARS + 1) * GUEST_TYPE_MS_PER_CHAR / 1000.0
    assert over_by_one > GUEST_COMMAND_TIMEOUT_S * GUEST_TYPE_DEADLINE_FRACTION


def test_the_constants_are_pinned_so_any_one_moving_is_a_deliberate_decision() -> None:
    """Pin all four values, so changing ANY of them lands here first.

    The derivation above is invariant under a coordinated change, so it alone
    would let someone halve the timeout and never notice the admission bound
    halving with it. Changing a value here is legitimate — update it, and say in
    the commit which measurement or deadline changed and why the others still
    hold.
    """

    assert GUEST_COMMAND_TIMEOUT_S == 90.0
    assert GUEST_TYPE_MS_PER_CHAR == 8.0
    assert GUEST_TYPE_DEADLINE_FRACTION == 0.6
    assert actions.MAX_TYPE_CHARS == 6750


def test_the_admission_bound_binds_before_the_protocol_length_cap() -> None:
    """The protocol's cap is a data limit; the deadline is an execution limit.

    ``MAX_TEXT_LENGTH`` stays where it is — it is right for ``paste_text``,
    whose clipboard cost does not grow with length. What must never return is a
    state where the only limit on a TYPED payload is the data cap, because that
    cap is ~15x what the guest can type inside its deadline.
    """

    assert actions.MAX_TYPE_CHARS < MAX_TEXT_LENGTH
    protocol_cap_seconds = MAX_TEXT_LENGTH * GUEST_TYPE_MS_PER_CHAR / 1000.0
    assert protocol_cap_seconds > GUEST_COMMAND_TIMEOUT_S


@pytest.mark.parametrize("length", FATAL_PAYLOAD_CHARS)
def test_the_payloads_that_killed_episodes_are_now_refused(length: int) -> None:
    assert length > actions.MAX_TYPE_CHARS
    with pytest.raises(ActionAdmissionError, match="paste_text"):
        actions.ACTION_SURFACE.validate_batch(
            _batch(TypeAction(observation_id="obs", text="a" * length))
        )


def test_the_largest_payload_that_actually_worked_is_still_admitted() -> None:
    """The bound must not be so tight that it refuses work the guest can do.

    2498 characters completed in 39.4 s on a real guest. A fix that rejects it
    would trade an episode-killing timeout for an episode-killing refusal.
    """

    assert LARGEST_SURVIVING_PAYLOAD_CHARS < actions.MAX_TYPE_CHARS
    actions.ACTION_SURFACE.validate_batch(
        _batch(TypeAction(observation_id="obs", text="a" * LARGEST_SURVIVING_PAYLOAD_CHARS))
    )


# ----------------------------------------------------------------------------
# Admission: rejected pre-dispatch, alternative named, nothing mutated
# ----------------------------------------------------------------------------


def test_an_over_long_type_is_rejected_before_anything_is_compiled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejection must precede the whole batch, not just the offending action.

    Same guarantee the ASCII rejection carries: a batch whose first action is a
    click and whose second is an inadmissible type must not land the click. A
    half-applied batch is precisely the unknown-outcome state that makes this
    class of failure unrecoverable rather than correctable.
    """

    batch = _batch(
        ClickAction(observation_id="obs", frame_id="screen", x=1, y=1),
        TypeAction(observation_id="obs", text="a" * (actions.MAX_TYPE_CHARS + 1)),
    )
    compiled: list[object] = []
    monkeypatch.setattr(actions, "compile_action", lambda *args: compiled.append(args))
    with pytest.raises(ActionAdmissionError) as raised:
        actions.compile_batch(batch, _geo())
    assert not compiled
    # The message must name the alternative and the actual numbers. A model told
    # only "too long" shortens its text and retries, spending the retry budget
    # bisecting for a limit it was never shown.
    message = str(raised.value)
    assert "paste_text" in message
    assert str(actions.MAX_TYPE_CHARS) in message
    assert str(actions.MAX_TYPE_CHARS + 1) in message


def test_a_payload_at_the_bound_still_compiles_to_one_typewrite() -> None:
    text = "a" * actions.MAX_TYPE_CHARS
    statements = actions.compile_batch(_batch(TypeAction(observation_id="obs", text=text)), _geo())
    assert statements == [f"pyautogui.typewrite({text!r})"]


def test_paste_text_is_not_bounded_by_the_typing_deadline() -> None:
    """The named alternative has to actually accept what type refused.

    ``paste_text`` writes the clipboard and sends one chord, so its guest cost
    is O(1) in length. Bounding it by the typing budget would leave a model with
    a rejection and nowhere to go.
    """

    from local_operator.evaluation.protocol import PasteTextAction

    actions.ACTION_SURFACE.validate_batch(
        _batch(
            PasteTextAction(
                observation_id="obs",
                text="a" * MAX_TEXT_LENGTH,
                keys=("CTRL", "v"),
                clipboard_policy="overwrite",
            )
        )
    )


# ----------------------------------------------------------------------------
# The mechanism: what interval= actually cost
# ----------------------------------------------------------------------------


class _StubPyAutoGui:
    """pyautogui's documented ``typewrite`` contract, with a virtual clock.

    Only the guest is stubbed, never the thing under test: the statements
    executed below are the REAL output of ``compile_action``. The loop body here
    is pyautogui 0.9.54's own (``__init__.py``: ``for c in message: press(c,
    _pause=False); time.sleep(interval)``), and ``interval`` is the documented
    "number of seconds in between each press", so the sleep is unconditional and
    per character. Time is accumulated rather than slept, because the point is
    to show a 73-second cost in milliseconds.
    """

    def __init__(self) -> None:
        self.slept_seconds = 0.0
        self.presses = 0

    def typewrite(self, message: str, interval: float = 0.0) -> None:
        for _ in message:
            self.presses += 1
            self.slept_seconds += float(interval)


def test_the_interval_we_removed_really_did_cost_the_deadline() -> None:
    """Demonstrate the 10 ms/char floor instead of inferring it from the fit.

    The regression on real batches attributed 10.00 of 14.18 ms/char to our own
    ``interval``; this executes both statement forms against the documented
    contract and shows that 71% directly. The old statement is written out as a
    frozen literal because it is a historical artifact — the compiler no longer
    produces it, and that is the property being protected.
    """

    text = "a" * max(FATAL_PAYLOAD_CHARS)  # 7332, the larger fatal payload

    before = _StubPyAutoGui()
    exec(f"pyautogui.typewrite({text!r}, interval=0.01)", {"pyautogui": before})

    compiled = actions.compile_action(TypeAction(observation_id="obs", text=text), None)
    assert compiled is not None
    after = _StubPyAutoGui()
    exec(compiled, {"pyautogui": after})

    # Same keystrokes delivered either way: the interval bought nothing.
    assert before.presses == after.presses == len(text)
    # Pure mandated sleep, before a single keystroke's real cost is counted.
    assert before.slept_seconds >= 73.0
    assert before.slept_seconds > GUEST_COMMAND_TIMEOUT_S * GUEST_TYPE_DEADLINE_FRACTION
    # What we emit now adds none of it.
    assert after.slept_seconds == 0.0
