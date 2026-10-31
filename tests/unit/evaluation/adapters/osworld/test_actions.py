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
    ActionBatch,
    ClickAction,
    DoubleClickAction,
    DragAction,
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


@pytest.mark.parametrize(
    ("alias", "compiled"),
    [
        (("Return",), "pyautogui.press('enter')"),
        (("Escape",), "pyautogui.press('esc')"),
        (("Page_Down",), "pyautogui.press('pagedown')"),
        (("super",), "pyautogui.press('win')"),
        (("Control_L", "minus"), "pyautogui.hotkey('ctrl', '-')"),
        (("ctrl", "shift", "equal"), "pyautogui.hotkey('ctrl', 'shift', '=')"),
        (("ctrl", "plus"), "pyautogui.hotkey('ctrl', '+')"),
    ],
)
def test_an_alias_compiles_exactly_as_its_canonical_key(
    alias: tuple[str, ...], compiled: str
) -> None:
    """The guest receives the canonical key; the alias never reaches it.

    Normalisation happens in the protocol validator, so the compiler needs no
    alias table of its own -- this pins that the compiled statement for an
    alias is the one its canonical spelling has always produced.
    """

    assert actions.compile_action(KeyAction(observation_id="o", keys=alias), None) == compiled


def test_coordinate_corners_round_trip() -> None:
    geo = _geo()
    for x, y in ((0, 0), (1919, 1079)):
        action = ClickAction(observation_id="o", frame_id="screen", x=x, y=y)
        compiled = actions.compile_action(action, geo)
        assert compiled is not None
        assert f"x={x}" in compiled and f"y={y}" in compiled


def _decoded_guest(statement: str) -> str:
    import ast

    return ast.literal_eval(statement[len("exec(") : -1])


def test_drag_compiles_both_endpoints_through_one_geometry() -> None:
    identity = DragAction(
        observation_id="o",
        frame_id="screen",
        start_x=10,
        start_y=20,
        end_x=110,
        end_y=220,
    )
    compiled = actions.compile_action(identity, _geo())
    assert compiled is not None
    assert "_drag_x11(10, 20, 110, 220, 'left', 500)" in _decoded_guest(compiled)
    # Deterministic byte-for-byte: identical action + geometry produce the
    # same source, with no random easing or duration.
    assert compiled == actions.compile_action(identity, _geo())
    # A 2x downscale maps BOTH endpoints exactly like every other pointer
    # action: 50,25 -> 100,50 and 100,75 -> 200,150, through the geometry's
    # own floor-and-clamp policy.
    scaled = DragAction(
        observation_id="o",
        frame_id="screen",
        start_x=50,
        start_y=25,
        end_x=100,
        end_y=75,
        duration_ms=1200,
    )
    scaled_source = actions.compile_action(scaled, _geo(native=(1920, 1080), visible=(960, 540)))
    assert scaled_source is not None
    assert "_drag_x11(100, 50, 200, 150, 'left', 1200)" in _decoded_guest(scaled_source)


def test_drag_admission_is_gated_by_the_surface_capability() -> None:
    from local_operator.evaluation.action_surface import (
        ActionAdmissionError,
        ActionSurface,
    )

    batch = ActionBatch.model_validate(
        {
            "protocol_version": "1.0",
            "task_id": "task",
            "episode_id": "episode",
            "observation_id": "obs",
            "actions": [
                {
                    "kind": "drag",
                    "observation_id": "obs",
                    "frame_id": "screen",
                    "start_x": 1,
                    "start_y": 2,
                    "end_x": 3,
                    "end_y": 4,
                }
            ],
        }
    )
    # A backend that does not declare the translator refuses the batch before
    # producing any statement: no guest call can happen for an unsupported
    # kind.
    with pytest.raises(ActionAdmissionError, match="not supported"):
        ActionSurface().validate_batch(batch)
    # This adapter declares it, so the same batch compiles to one statement
    # (one observation, existing ordered-batch semantics).
    assert actions.ACTION_SURFACE.drag is True
    statements = actions.compile_batch(batch, _geo())
    assert len(statements) == 1
    assert "_drag_x11" in _decoded_guest(statements[0])


# ---------------------------------------------------------------------------
# Held modifiers. Evidence: osworld-21 t25k3c-t004 steps 5-6 executed
# ``[click 111,302][key SHIFT][click 111,671]`` for a shift-click range
# selection and selected one slide, because a standalone key is released
# before the next click. The compiled statement must hold the key across the
# pointer action, and release it even when that action raises.

_FAKE_GUI = """
import json, os, pathlib
_log = pathlib.Path(os.environ["MOD_TEST_ROOT"]) / "calls.jsonl"
FAILSAFE = True
held = []


def _record(call, *args):
    with _log.open("a") as handle:
        handle.write(json.dumps([call, *args, FAILSAFE]) + "\\n")


def keyDown(key):
    held.append(key)
    _record("keyDown", key)
    if os.environ.get("MOD_FAILURE") == "down-" + key:
        raise RuntimeError("keyDown injected then failed")


def keyUp(key):
    _record("keyUp", key)
    if os.environ.get("MOD_FAILURE") == "up":
        raise RuntimeError("release refused")
    held.remove(key)


def click(x, y, button="left", clicks=1, interval=0.0):
    _record("click", x, y, button, clicks)
    if os.environ.get("MOD_FAILURE") == "click":
        raise RuntimeError("click refused (FailSafeException)")


def moveTo(x, y, duration=0.0):
    _record("moveTo", x, y)


def mouseDown(button="left"):
    _record("mouseDown", button)


def mouseUp(button="left"):
    _record("mouseUp", button)


def scroll(clicks):
    _record("scroll", clicks)


def hscroll(clicks):
    _record("hscroll", clicks)
"""


def _run_guest(statement: str, tmp_path, failure: str | None = None):
    """Execute a compiled statement the way the guest does, against a fake GUI."""
    import json
    import os
    import subprocess
    import sys

    from local_operator.computer_input import python_source_argv

    (tmp_path / "pyautogui.py").write_text(_FAKE_GUI)
    # The guest prefixes every statement with its own import, exactly as the
    # controller's ``pkgs_prefix`` does.
    source = f"import pyautogui; {statement}\nprint(pyautogui.held)"
    env = {**os.environ, "PYTHONPATH": str(tmp_path), "MOD_TEST_ROOT": str(tmp_path)}
    if failure:
        env["MOD_FAILURE"] = failure
    result = subprocess.run(
        [sys.executable, *python_source_argv(source)[1:]],
        env=env,
        text=True,
        capture_output=True,
        timeout=12,
    )
    log = tmp_path / "calls.jsonl"
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, calls


def _shift_click(**changes) -> ClickAction:
    return ClickAction.model_validate(
        {"observation_id": "o", "frame_id": "screen", "x": 111, "y": 302, **changes}
    )


def test_shift_click_holds_shift_across_the_click(tmp_path) -> None:
    statement = actions.compile_action(_shift_click(modifiers=["shift"]), _geo())
    assert statement is not None
    result, calls = _run_guest(statement, tmp_path)
    assert result.returncode == 0, result.stderr
    assert [call[:-1] for call in calls] == [
        ["keyDown", "shift"],
        ["click", 111, 302, "left", 1],
        ["keyUp", "shift"],
    ]
    assert result.stdout.strip() == "[]"


def test_modifiers_press_in_order_and_release_in_reverse(tmp_path) -> None:
    statement = actions.compile_action(
        _shift_click(modifiers=["ctrl", "shift", "meta"], button="right"), _geo()
    )
    assert statement is not None
    result, calls = _run_guest(statement, tmp_path)
    assert result.returncode == 0, result.stderr
    # META reuses the key compiler's own mapping (pyautogui's "win").
    assert [call[:-1] for call in calls] == [
        ["keyDown", "ctrl"],
        ["keyDown", "shift"],
        ["keyDown", "win"],
        ["click", 111, 302, "right", 1],
        ["keyUp", "win"],
        ["keyUp", "shift"],
        ["keyUp", "ctrl"],
    ]


@pytest.mark.parametrize(
    ("action", "inner"),
    [
        (
            DragAction(
                observation_id="o",
                frame_id="screen",
                start_x=1,
                start_y=2,
                end_x=3,
                end_y=4,
                modifiers=("SHIFT",),
            ),
            [["moveTo", 1, 2], ["mouseDown", "left"], ["moveTo", 3, 4], ["mouseUp", "left"]],
        ),
        (
            ScrollAction(
                observation_id="o", frame_id="screen", x=5, y=6, delta_y=-3, modifiers=("CTRL",)
            ),
            [["moveTo", 5, 6], ["scroll", -3]],
        ),
        (
            DoubleClickAction(observation_id="o", frame_id="screen", x=7, y=8, modifiers=("CTRL",)),
            [["click", 7, 8, "left", 2]],
        ),
    ],
    ids=["drag", "scroll", "double_click"],
)
def test_every_pointer_action_runs_inside_the_held_modifier(action, inner, tmp_path) -> None:
    statement = actions.compile_action(action, _geo())
    assert statement is not None
    result, calls = _run_guest(statement, tmp_path)
    assert result.returncode == 0, result.stderr
    key = "shift" if isinstance(action, DragAction) else "ctrl"
    assert [call[:-1] for call in calls] == [["keyDown", key], *inner, ["keyUp", key]]


def test_a_failing_click_still_releases_every_modifier(tmp_path) -> None:
    statement = actions.compile_action(_shift_click(modifiers=["ctrl", "shift"]), _geo())
    assert statement is not None
    result, calls = _run_guest(statement, tmp_path, failure="click")
    # The action's own failure still reaches the caller (no success claim) ...
    assert result.returncode != 0
    assert "click refused" in result.stderr
    # ... and nothing is left held. FAILSAFE is disarmed ONLY for the releases:
    # a click that ends in a screen corner must not make keyUp raise too.
    assert [call[:-1] for call in calls][-2:] == [["keyUp", "shift"], ["keyUp", "ctrl"]]
    assert [call[-1] for call in calls if call[0] == "keyUp"] == [False, False]
    assert "could not be released" not in result.stderr


def test_a_key_that_fails_mid_press_is_still_released(tmp_path) -> None:
    # keyDown can inject and then raise: the key that failed is released too,
    # and the click never runs.
    statement = actions.compile_action(_shift_click(modifiers=["ctrl", "shift"]), _geo())
    assert statement is not None
    result, calls = _run_guest(statement, tmp_path, failure="down-shift")
    assert result.returncode != 0
    assert [call[:-1] for call in calls] == [
        ["keyDown", "ctrl"],
        ["keyDown", "shift"],
        ["keyUp", "shift"],
        ["keyUp", "ctrl"],
    ]


def test_an_unobservable_release_is_reported_not_laundered(tmp_path) -> None:
    statement = actions.compile_action(_shift_click(modifiers=["ctrl", "shift"]), _geo())
    assert statement is not None
    result, calls = _run_guest(statement, tmp_path, failure="up")
    assert result.returncode != 0
    assert "could not be released" in result.stderr
    # Every release is attempted even after the first one fails.
    assert [call[1] for call in calls if call[0] == "keyUp"] == ["shift", "ctrl"]


def test_adapter_advertises_the_modifier_translator_it_ships() -> None:
    assert actions.ACTION_SURFACE.modifiers is True
    action = ClickAction(observation_id="o", frame_id="screen", x=1, y=2, modifiers=("shift",))
    from local_operator.evaluation.protocol import ActionBatch

    batch = ActionBatch(
        protocol_version="1.0",
        task_id="task",
        episode_id="episode",
        observation_id="o",
        actions=(action,),
    )
    actions.ACTION_SURFACE.validate_batch(batch)
    assert "shift" in actions.compile_action(action, _geo())
