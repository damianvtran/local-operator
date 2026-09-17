"""Guards for the sigil grammar shared by the composer and the resolver.

The grammar was extracted out of ``tui/widgets/command_picker.py`` so that
``local_operator/references.py`` — a session-layer module — can read it without
importing a Textual widget. Two of the tests below exist only because of that
move: ``test_the_sigil_module_imports_no_host`` pins the property the extraction
was FOR, and ``test_command_picker_still_exposes_the_moved_names`` pins the
re-export that keeps every pre-existing caller working. The rest pin the grammar
itself, which the move was not allowed to change.

This file deliberately does NOT live under ``tests/unit/tui/``. Filing it there
would assert the opposite of what the move is for.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from local_operator.sigils import SlashContext, is_boundary

REPO = Path(__file__).resolve().parent.parent.parent

# Copied from tests/unit/test_import_graph.py:30-47, whose reasoning applies
# here verbatim: an in-process ``sys.modules`` assertion is worthless because
# pytest has already imported half the tree by the time a test body runs, so a
# host module this one wrongly pulls would look "already imported" and the
# assertion would pass on a real regression. Printed as JSON on stdout so a
# stray warning on stderr cannot corrupt the result.
_PROBE = """
import json, importlib, sys
importlib.import_module(sys.argv[1])
print(json.dumps(sorted(sys.modules)))
"""


def _imported_modules(target: str) -> set[str]:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, target],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert proc.returncode == 0, f"importing {target} failed:\n{proc.stderr[-3000:]}"
    return set(json.loads(proc.stdout.strip().splitlines()[-1]))


# --- The boundary rule --------------------------------------------------------


def test_a_sigil_at_the_line_start_is_a_boundary() -> None:
    assert is_boundary("@foo", 0) is True


def test_a_sigil_after_whitespace_is_a_boundary() -> None:
    # A tab counts for the same reason a space does: ``str.isspace`` is the
    # whole rule, so a pasted draft indented with tabs behaves like a typed one.
    assert is_boundary("ask @foo", 4) is True
    assert is_boundary("ask\t@foo", 4) is True


def test_a_sigil_glued_to_a_word_is_not_a_boundary() -> None:
    # The single most load-bearing line in the module: this is what keeps an
    # email address from opening a picker, ``costs$5`` from opening a skill
    # list, and ``src/foo`` from opening a command list, on every keystroke of
    # ordinary prose.
    assert is_boundary("user@host.com", 4) is False
    assert is_boundary("a$b", 1) is False
    assert is_boundary("src/foo", 3) is False


# --- The span type ------------------------------------------------------------


def test_slash_context_carries_start_query_and_end() -> None:
    # Trivial by design: it pins that the NamedTuple FIELD ORDER survived the
    # move, which is the one thing a copy-paste can silently reverse and which
    # every positional construction in the composer depends on.
    context = SlashContext(3, "res", 7)
    assert context.start == 3
    assert context.query == "res"
    assert context.end == 7


# --- The module boundary the extraction exists to create ----------------------


def test_the_sigil_module_imports_no_host() -> None:
    # The test that justifies the slice. ``references.py`` reaches this module
    # from the session layer, and session_factory must never load ``textual``
    # (tests/unit/test_import_graph.py:163 — "TUI front end; the server has no
    # terminal"). A host import added here would travel that path silently.
    modules = _imported_modules("local_operator.sigils")
    hosts = sorted(
        name
        for name in modules
        if name == "textual" or name.startswith(("textual.", "rich", "local_operator.tui"))
    )
    assert hosts == [], f"local_operator.sigils must stay host-free, but it imported: {hosts}"


def test_command_picker_still_exposes_the_moved_names() -> None:
    # The re-export contract: the move is only pure if every pre-existing caller
    # of the old private name keeps resolving to the very same object.
    # Imported inside the test body, not at module scope: importing the widget
    # module at import time would pull Textual into every run of this file and
    # defeat the point of the test above it.
    from local_operator.tui.widgets.command_picker import (
        SlashContext as PickerSlashContext,
    )
    from local_operator.tui.widgets.command_picker import _is_boundary, is_boundary

    assert PickerSlashContext is SlashContext
    assert _is_boundary is is_boundary
