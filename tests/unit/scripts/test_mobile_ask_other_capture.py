"""The `Other`-row capture rig's own contract: it must run as documented, and a
question's text must reach the page as data rather than as JavaScript syntax.

Both assertions exist for the reason the sibling rig's do
(`test_mobile_asks_capture.py`): frames a later round cannot re-derive are not
evidence. The Run line is the only way to re-derive them, so it is exercised
exactly as written from the repository root; and the row is found by the text the
question paints, so a quote or a newline in that text must not be able to end the
selector early or start a new statement.

The module is imported directly for the helper test. It imports
`scripts.mobile_overflow_capture` (a class library with no import-time work) and
`scripts.probe_isolation` is not in that chain, so importing it here neither
re-homes the test process nor trips the isolation guard.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _clean_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(extra)
    return env


def test_the_documented_run_line_reaches_main(monkeypatch):
    """`python scripts/mobile_ask_other_capture.py` must import, not die on `scripts`.

    Invoked exactly as its docstring documents it — from the repository root,
    with PYTHONPATH cleared — so the module's own two-line `sys.path` bootstrap is
    what has to make `scripts.mobile_overflow_capture` resolve. Run with no
    arguments it stops at `main()`'s usage branch; that is the point: reaching
    usage proves the imports above it worked, while a `ModuleNotFoundError` or a
    traceback would mean nobody following the Run line gets that far.
    """
    monkeypatch.delenv("PYTHONPATH", raising=False)
    proc = subprocess.run(
        [sys.executable, "scripts/mobile_ask_other_capture.py"],
        cwd=REPO_ROOT,
        env=_clean_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "ModuleNotFoundError" not in proc.stderr, proc.stderr
    assert "Traceback (most recent call last)" not in proc.stderr, proc.stderr
    # Either usage spelling is fine: the docstring exit the rig uses today, or the
    # argparse exit 2 a later refactor may choose. Both mean main() was reached.
    assert proc.returncode in (1, 2), (proc.returncode, proc.stderr)
    assert "Capture the phone ask card" in proc.stderr, proc.stderr


def test_a_quote_or_newline_in_the_question_cannot_break_the_selector():
    """A question's text is the row's identity, so it must be escaped, not pasted.

    The text goes into `question_block_js` as a JavaScript string literal. A raw
    `"` would end that literal and a raw newline would end the statement, which
    is why `_js_literal` is `json.dumps` rather than f-string interpolation; the
    assertion is on the built selector, because that is what runs.
    """
    from scripts.mobile_ask_other_capture import _js_literal, question_block_js

    text = 'Ship "it"\nnow'
    literal = _js_literal(text)
    assert literal == json.dumps(text)
    assert json.loads(literal) == text
    assert "\n" not in literal
    assert '\\"' in literal

    js = question_block_js(text)
    assert 'Ship \\"it\\"\\nnow' in js
    assert "\n" not in js
