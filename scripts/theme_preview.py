"""Render one SVG frame per theme, with representative transcript content.

The palette gate (``tests/unit/tui/test_palette_contrast.py``) proves the
numbers; this proves the LOOK. It drives the real ``OperatorApp`` (the one
that loads ``local_operator.tcss``) through ``run_test``, fills the screen
with the content mix a real session shows — user prompt, assistant markdown
with a code fence, tool rows in all three outcomes, a notice, the status
band — switches the theme live through the same ``_apply_theme`` path the
``/theme`` command uses, and saves a frame per theme.

Usage::

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/theme_preview.py OUTDIR [theme ...]

With no theme arguments it renders every registered theme. SVGs land in
OUTDIR as ``<theme>.svg``; convert or open them and LOOK — the gate cannot
see a hue clash, only a ratio.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# The pilot needs a colour terminal and stilled animation for stable frames.
os.environ.setdefault("LOCAL_OPERATOR_NO_SHIMMER", "1")
os.environ.pop("NO_COLOR", None)

from local_operator.tui import theme as theme_mod  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.assistant import AssistantBlock  # noqa: E402
from local_operator.tui.widgets.tool_card import ToolCard  # noqa: E402
from local_operator.tui.widgets.transcript import NoticeBlock, UserBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: Markdown exercising the inks a theme must keep distinct: prose, bold,
#: inline code (signal), a fenced block (syntax ramp), a list, a link.
_ASSISTANT_MD = """\
Here's the plan — I checked the **routing table** and `config.yml` first.

```python
def relight(theme: str) -> int:
    # comments sit at dim; strings at the literal hue
    return apply(theme, mode="live")
```

1. Switch the ramp atomically
2. Re-ink every settled block
3. [Persist the choice](https://example.com) to config
"""


def _populate(app: OperatorApp) -> None:
    """One screen of representative content, appended through the real seams."""
    app._append_block(
        UserBlock(
            "Switch my theme and show me every kind of row.\n\n"
            "Second paragraph, for the gutter rule."
        )
    )
    assistant = AssistantBlock()
    assistant.update_text(_ASSISTANT_MD)
    assistant.finalize_text()
    app._append_block(assistant)

    done = ToolCard("call-1", "bash", {"command": "pytest -q tests/unit"})
    app._append_block(done)
    done.mark_done("2701 passed in 204s")

    failed = ToolCard("call-2", "read", {"path": "/tmp/missing.txt"})
    app._append_block(failed)
    failed.mark_failed("no such file: /tmp/missing.txt")

    running = ToolCard("call-3", "edit", {"path": "app.py"})
    app._append_block(running)

    app._append_block(NoticeBlock("theme: dark → monokai (saved as your default)", "note"))
    app._append_block(NoticeBlock("credential store unreachable — retrying", "warning"))


async def _render(names: list[str], outdir: Path) -> None:
    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(110, 34)) as pilot:
        await pilot.pause()
        _populate(app)
        await pilot.pause()
        for name in names:
            app._apply_theme(name)
            await pilot.pause()
            await pilot.pause()
            target = outdir / f"{name}.svg"
            app.save_screenshot(str(target))
            print(f"wrote {target}")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    outdir = Path(sys.argv[1])
    outdir.mkdir(parents=True, exist_ok=True)
    names = sys.argv[2:] or theme_mod.available_themes()
    unknown = [name for name in names if name not in theme_mod.available_themes()]
    if unknown:
        raise SystemExit(f"unknown themes: {', '.join(unknown)}")
    asyncio.run(_render(names, outdir))


if __name__ == "__main__":
    main()
