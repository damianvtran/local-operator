"""Cross-package contracts: the settings registry and the backend's tool.

The ``library`` tool is owned by the backend-core branch
(``local_operator/tools/library_tool.py``); this package only names it. The
last test SKIPS until that module lands, and is the one to re-run on the
merged branch — the brief's integration note.
"""

from __future__ import annotations

import pytest

from local_operator import settings_io
from local_operator.librarian import bootstrap, proactive


def test_the_settings_registry_matches_the_consumers_defaults() -> None:
    """The registry's literals must equal the constants the engine reads.

    ``settings_io`` keeps its defaults as literals so it never imports the
    feature package on every CLI start; this is the pin that keeps that split
    from drifting.
    """
    rows = {setting.key: setting for setting in settings_io.SETTINGS}
    assert rows["librarian.enabled"].default is proactive.DEFAULT_ENABLED
    assert rows["librarian.name"].default == bootstrap.SESSION_TITLE
    assert rows["librarian.cadence.at"].default == proactive.DEFAULT_CADENCE_AT
    assert rows["librarian.cadence.paused"].default is proactive.DEFAULT_PAUSED
    assert (
        rows["librarian.cadence.max_extra_per_day"].default == proactive.DEFAULT_MAX_EXTRA_PER_DAY
    )
    assert rows["librarian.cadence.min_gap_minutes"].default == proactive.DEFAULT_MIN_GAP_MINUTES
    # The name cap is shared between the facade and the display reader.
    assert settings_io._LIBRARIAN_NAME_MAX_CHARS == bootstrap.MAX_NAME_CHARS


def test_the_kill_switch_is_documented_in_the_package() -> None:
    assert bootstrap.state.ENV_DISABLE == "LOCAL_OPERATOR_NO_LIBRARIAN"


def test_the_library_tool_is_available_to_the_seed() -> None:
    """SKIPS until the backend core branch lands ``tools/library_tool.py``."""
    tool_module = pytest.importorskip(
        "local_operator.tools.library_tool",
        reason="backend core owns local_operator/tools/library_tool.py",
    )
    # The tool is what the seed drives; its operation vocabulary is the slice
    # of the design the seed's instructions name.
    source = tool_module.__doc__ or ""
    for op in ("stats", "list", "get", "update", "tags", "conventions"):
        assert op in source or hasattr(tool_module, op), op
