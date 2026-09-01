"""Compile protocol actions to pyautogui source for the OSWorld guest.

Why pyautogui and not ``computer_13``: OSWorld's own controller COMPILES
``computer_13`` dicts into pyautogui source anyway (python.py:769-844), and
does so with randomised ``duration``/easing we did not ask for — a
nondeterminism source in a benchmark whose evidence must be reproducible.
Emitting pyautogui directly removes that translation layer and is the action
space every published OSWorld result and official runner uses.

The compiler is a pure function of the action plus the frame geometry — the
single most valuable unit-testable piece in the adapter. Coordinates are
converted with ``frame.geometry.model_to_native(x, y)``, never with hand
arithmetic, because the geometry's floor-and-clamp policy is the protocol's
and duplicating it here would be a second, divergent implementation.

Out-of-frame coordinates are NOT clamped here: ``ActionBatch.validate_for``
already rejected them upstream, so seeing one is a host bug and must raise,
not be silently repaired.
"""

from __future__ import annotations

from local_operator.evaluation.protocol import (
    ActionBatch,
    AskUserAction,
    ClickAction,
    ComputerAction,
    DoubleClickAction,
    FinishAction,
    FrameGeometry,
    KeyAction,
    ScrollAction,
    TypeAction,
    WaitAction,
)

# Our named keys (protocol._NAMED_KEYS) -> pyautogui key names. pyautogui
# uses lowercase single words; the two that differ structurally are META
# (which on a Linux guest is the Super/Win key, pyautogui's "win") and the
# pagedown/pageup pair. Printable single characters pass through lowercased.
_NAMED_KEY_MAP = {
    "ALT": "alt",
    "BACKSPACE": "backspace",
    "CAPSLOCK": "capslock",
    "CTRL": "ctrl",
    "DELETE": "delete",
    "DOWN": "down",
    "END": "end",
    "ENTER": "enter",
    "ESC": "esc",
    "HOME": "home",
    "INSERT": "insert",
    "LEFT": "left",
    "META": "win",  # Linux guest: the Super key is what pyautogui calls "win"
    "PAGEDOWN": "pagedown",
    "PAGEUP": "pageup",
    "RIGHT": "right",
    "SHIFT": "shift",
    "SPACE": "space",
    "TAB": "tab",
    "UP": "up",
    **{f"F{i}": f"f{i}" for i in range(1, 25)},
}


class CompilationError(ValueError):
    """An action could not be compiled; always a host bug, never model output."""


def _key_name(key: str) -> str:
    if key in _NAMED_KEY_MAP:
        return _NAMED_KEY_MAP[key]
    if len(key) == 1:
        return key.lower()
    raise CompilationError(f"no pyautogui name for key {key!r}")


def _native_point(geometry: FrameGeometry, x: int, y: int) -> tuple[int, int]:
    visible = geometry.model_visible
    if x >= visible.width or y >= visible.height:
        # validate_for rejected this upstream; reaching here means the adapter
        # was handed a batch the host never validated, which is a host bug.
        raise CompilationError(
            f"coordinate {x},{y} is outside the model-visible frame "
            f"{visible.width}x{visible.height}"
        )
    point = geometry.model_to_native(x, y)
    return point.x, point.y


def compile_action(action: ComputerAction, geometry: FrameGeometry | None = None) -> str | None:
    """Compile one action to one pyautogui statement (or an OSWorld special).

    Returns ``None`` for actions the guest does not execute as pyautogui:
    ``FinishAction`` (mapped to OSWorld's ``DONE``/``FAIL`` step tokens by the
    caller, which needs the batch context) and ``AskUserAction`` (handled by
    ``ask_user_exchange``, not the guest).

    ``geometry`` is required for pointer actions (click/double_click/scroll)
    and ignored for the rest.
    """

    if isinstance(action, ClickAction):
        assert geometry is not None
        nx, ny = _native_point(geometry, action.x, action.y)
        return f"pyautogui.click(x={nx}, y={ny}, button={action.button!r})"

    if isinstance(action, DoubleClickAction):
        assert geometry is not None
        nx, ny = _native_point(geometry, action.x, action.y)
        # button is Literal["left"] by type (protocol.py:408): no translation
        # needed, and emitting it explicitly documents the closure.
        return f"pyautogui.click(x={nx}, y={ny}, clicks=2, interval=0.1, button='left')"

    if isinstance(action, TypeAction):
        # repr(), NEVER f-string interpolation: the text is model output and
        # may contain quotes, backslashes, or newlines. repr() produces a
        # Python literal that re-parses to exactly the same string.
        return f"pyautogui.typewrite({action.text!r}, interval=0.01)"

    if isinstance(action, KeyAction):
        names = [_key_name(key) for key in action.keys]
        if len(names) == 1:
            return f"pyautogui.press({names[0]!r})"
        joined = ", ".join(repr(name) for name in names)
        return f"pyautogui.hotkey({joined})"

    if isinstance(action, ScrollAction):
        assert geometry is not None
        nx, ny = _native_point(geometry, action.x, action.y)
        # Emit only the non-zero axes: the validator guarantees at least one
        # is non-zero (protocol.py:474-478), and emitting a zero scroll is a
        # guest no-op that still costs a round trip.
        parts = [f"pyautogui.moveTo({nx}, {ny})"]
        if action.delta_x:
            parts.append(f"pyautogui.hscroll({action.delta_x})")
        if action.delta_y:
            parts.append(f"pyautogui.scroll({action.delta_y})")
        return "; ".join(parts)

    if isinstance(action, WaitAction):
        # OSWorld's own special action, not a pyautogui line: env.step("WAIT",
        # pause=seconds). The caller wraps this; here we emit the canonical
        # token so the batch compiler can assemble a uniform statement list.
        return f"WAIT {action.duration_ms}"

    if isinstance(action, (FinishAction, AskUserAction)):
        return None

    raise CompilationError(f"unhandled action kind {type(action).__name__!r}")


def compile_batch(batch: ActionBatch, geometry: FrameGeometry) -> list[str]:
    """Compile a whole batch to an ordered list of guest statements.

    A batch with a terminal action (finish/ask_user) is always a singleton
    (protocol.py:551-555); those produce no guest statement and are the
    caller's responsibility. The returned list is one statement per
    non-terminal action, in order, executed with one settle and one
    observation at the end.
    """

    statements: list[str] = []
    for action in batch.actions:
        statement = compile_action(action, geometry)
        if statement is not None:
            statements.append(statement)
    return statements
