"""One contract test for the ``config`` frame's shape (wave-5 slice D).

Two surfaces publish this frame — the desktop feed and the relay daemon's
``/api/config/events`` — and clients parse both with one reader. These tests
keep that from drifting: the SHAPE is pinned here (a change to it must be a
deliberate edit to this file AND to ``i18n/frames.py`` together), and both
surfaces are pinned to build their payloads THROUGH the shared builder, so a
re-inlined dict in either one fails here rather than at a client's parser.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

from local_operator.i18n.frames import config_frame_payload

_TREE = Path(__file__).resolve().parents[2]
_SURFACES = (
    "local_operator/server/utils/desktop_feed.py",
    "local_operator/mobile/daemon.py",
)


def test_the_builder_emits_the_pinned_shape() -> None:
    payload = config_frame_payload(("display.dock", "language"), "en")
    assert payload == {"changed_keys": ["display.dock", "language"], "language": "en"}
    # SORTED here, not by callers: the two surfaces cannot disagree on order.
    assert config_frame_payload(("language", "display.dock"), "en") == payload
    # JSON-safe as-is: SSE legs serialize the dict verbatim.
    assert json.loads(json.dumps(payload)) == payload


def test_the_baseline_spelling_is_the_empty_change_list() -> None:
    """The connect-time baseline frame's marker (no history, current value):
    an EMPTY ``changed_keys`` and the resolved tag — pinned so the two
    surfaces' baselines cannot diverge either."""
    assert config_frame_payload((), "en") == {"changed_keys": [], "language": "en"}


def test_both_surfaces_build_the_frame_through_the_shared_module() -> None:
    """MECHANICAL pin, two clauses (CI round 1, NIT-1 strengthened it):

    1. each surface's source CALLS ``config_frame_payload`` at all; and
    2. neither surface contains an INLINE dict literal carrying the frame's
       two literal keys — because a surviving call next to a re-inlined frame
       (the drift this file exists to prevent) would pass clause 1 alone.
    """
    for relative in _SURFACES:
        source = (_TREE / relative).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "config_frame_payload"
        ]
        assert calls, f"{relative} does not build the config frame through i18n.frames"
        re_inlined = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Dict)
            and {
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            >= {"changed_keys", "language"}
        ]
        assert not re_inlined, (
            f"{relative} re-inlines a config frame payload (both frame keys in one dict "
            "literal) — build it through i18n.frames.config_frame_payload instead"
        )
