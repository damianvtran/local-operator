"""Action compilation and coordinate fidelity.

The compiler is a pure function — the most valuable unit-testable piece in the
adapter. Coordinates always go through ``frame.geometry.model_to_native``,
never hand arithmetic. Out-of-frame coordinates raise (a host bug), never
clamp silently. Adversarial text goes through ``repr()``, never f-string
interpolation.
"""

from __future__ import annotations

import pytest
from lop_osworld_v2_adapter import actions
from lop_osworld_v2_adapter.actions import CompilationError

from local_operator.evaluation.protocol import (
    ClickAction,
    DoubleClickAction,
    FrameGeometry,
    FrameSize,
    KeyAction,
    ScrollAction,
    TypeAction,
    WaitAction,
)


def _geo(native=(1920, 1080), visible=None) -> FrameGeometry:
    return FrameGeometry(
        native=FrameSize(width=native[0], height=native[1]),
        model_visible=FrameSize(width=(visible or native)[0], height=(visible or native)[1]),
    )


def test_click_compiles_at_identity_geometry() -> None:
    action = ClickAction(observation_id="o", frame_id="screen", x=100, y=200, button="right")
    assert actions.compile_action(action, _geo()) == "pyautogui.click(x=100, y=200, button='right')"


def test_click_maps_model_to_native_at_a_downscale() -> None:
    # A 2x downscale: model-visible 960x540 over native 1920x1080 doubles each
    # coordinate. The conversion is the geometry's floor-and-clamp, not ours.
    geo = _geo(native=(1920, 1080), visible=(960, 540))
    action = ClickAction(observation_id="o", frame_id="screen", x=50, y=25)
    assert actions.compile_action(action, geo) == "pyautogui.click(x=100, y=50, button='left')"


def test_double_click_is_two_left_clicks() -> None:
    action = DoubleClickAction(observation_id="o", frame_id="screen", x=10, y=10)
    assert actions.compile_action(action, _geo()) == (
        "pyautogui.click(x=10, y=10, clicks=2, interval=0.1, button='left')"
    )


def test_type_uses_repr_for_adversarial_text() -> None:
    text = 'he said "hi"\nC:\\path\\to\twith a quote\''
    compiled = actions.compile_action(TypeAction(observation_id="o", text=text), None)
    # The emitted statement must contain the repr, so re-parsing the argument
    # yields exactly the original string.
    assert compiled == f"pyautogui.typewrite({text!r})"


@pytest.mark.parametrize("length", [1, 100, 6092, 7332, 20_000])
def test_type_never_emits_an_interval_at_any_length(length: int) -> None:
    """Upstream parity: pyautogui's default interval of 0.0, at every size.

    ``typewrite`` sleeps ``interval`` seconds per character unconditionally, so
    any interval at all is a hard floor of ``interval x len(text)`` on guest
    time rather than a pacing hint. Passing 0.01 killed two episodes at 7332 and
    6912 characters. Parametrised over length because the defect was invisible
    at the short strings the other compiler tests use.
    """

    compiled = actions.compile_action(TypeAction(observation_id="o", text="a" * length), None)
    assert compiled is not None
    assert "interval" not in compiled
    assert compiled == f"pyautogui.typewrite({'a' * length!r})"


def test_single_key_compiles_to_press() -> None:
    assert actions.compile_action(KeyAction(observation_id="o", keys=("ENTER",)), None) == (
        "pyautogui.press('enter')"
    )


def test_chord_compiles_to_hotkey() -> None:
    assert actions.compile_action(
        KeyAction(observation_id="o", keys=("CTRL", "ALT", "DELETE")), None
    ) == ("pyautogui.hotkey('ctrl', 'alt', 'delete')")


def test_meta_maps_to_the_linux_super_key() -> None:
    # On a Linux guest the META key is Super, which pyautogui calls "win".
    assert actions.compile_action(KeyAction(observation_id="o", keys=("META",)), None) == (
        "pyautogui.press('win')"
    )


def test_function_keys() -> None:
    assert actions.compile_action(KeyAction(observation_id="o", keys=("F5",)), None) == (
        "pyautogui.press('f5')"
    )


def test_scroll_emits_only_nonzero_axes() -> None:
    geo = _geo()
    vertical = ScrollAction(observation_id="o", frame_id="screen", x=5, y=5, delta_y=-3)
    assert actions.compile_action(vertical, geo) == "pyautogui.moveTo(5, 5); pyautogui.scroll(-3)"
    horizontal = ScrollAction(observation_id="o", frame_id="screen", x=5, y=5, delta_x=4)
    assert actions.compile_action(horizontal, geo) == "pyautogui.moveTo(5, 5); pyautogui.hscroll(4)"
    both = ScrollAction(observation_id="o", frame_id="screen", x=5, y=5, delta_x=1, delta_y=2)
    compiled_both = actions.compile_action(both, geo)
    assert compiled_both is not None
    assert "hscroll(1)" in compiled_both
    assert "scroll(2)" in compiled_both


def test_wait_is_the_osworld_special_token() -> None:
    assert (
        actions.compile_action(WaitAction(observation_id="o", duration_ms=250), None) == "WAIT 250"
    )


def test_out_of_frame_coordinate_raises_not_clamps() -> None:
    # validate_for rejects this upstream; seeing it here is a host bug.
    geo = _geo()
    action = ClickAction(observation_id="o", frame_id="screen", x=1920, y=0)
    with pytest.raises(CompilationError):
        actions.compile_action(action, geo)


def test_every_named_key_has_a_mapping() -> None:
    from local_operator.evaluation.protocol import NAMED_KEYS

    for key in sorted(NAMED_KEYS):
        compiled = actions.compile_action(KeyAction(observation_id="o", keys=(key,)), None)
        assert compiled is not None
        assert compiled.startswith("pyautogui.press(")


def test_coordinate_corners_round_trip() -> None:
    geo = _geo()
    for x, y in ((0, 0), (1919, 1079)):
        action = ClickAction(observation_id="o", frame_id="screen", x=x, y=y)
        compiled = actions.compile_action(action, geo)
        assert compiled is not None
        assert f"x={x}" in compiled and f"y={y}" in compiled
