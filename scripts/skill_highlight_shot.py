"""Capture the ``$skill`` composer ink: resolved, inert, in-progress, guards.

Run from the worktree root, once per tree — the before/after halves come from
running this SAME script on a base checkout and on the branch:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/skill_highlight_shot.py OUTDIR [COLSxROWS]

The ``$name`` ink is user-visible, so a passing test is not evidence that it
reads right (AGENTS.md, "Visual validation"). What has to be judged from a
frame and cannot be judged from an assertion: that a resolved token reads as
structure rather than as prose, that the inert ink is quiet enough to be a
de-emphasis and not an alarm, that the in-progress token (list open) is visibly
UNPAINTED, and that the money/shell guards paint nothing — in BOTH ramps, since
every ink here is a theme role and light is where a cool hue is most likely to
collapse into the paper.

Drives the real ``OperatorApp`` rather than a bare widget host on purpose: the
lightweight hosts in the test files declare no ``CSS_PATH``, so
``local_operator.tcss`` never applies to them and a still captured from one
cannot show what the user sees.

The vocabulary is BUILT HERE, in a temporary directory (the documented
``LOCAL_OPERATOR_SKILL_EXTRA_ROOTS`` makes it the ONLY root, so the developer's
own skills cannot leak in) rather than pointed at the repo: a capture of
whatever happened to be installed is not comparable between two runs, and the
before/after pair is only evidence if the only thing that changed is the code.

Every state is captured in both ramps in one run, and each frame's REPORT
prints the numbers behind it (the composer strip's segment inks, the picker's
state, the input's size against the screen's) so a reviewer can read the claim
as well as look at it.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402

#: One entry per state: (label, draft). In capture order, and the order is the
#: story: the resolved name, the inert word, the in-progress prefix, then the
#: three guards that must stay prose.
STATES = (
    ("01-resolved", "$research fix this "),
    ("02-unknown", "$zzz"),
    ("03-picker-open", "$res"),
    ("04-money", "$5 for the redesign"),
    ("05-shell-var", "echo $PATH"),
    ("06-slash-claim", "/model $5"),
)

#: The fixture vocabulary. ``secret-audit`` is hidden on purpose, but for the
#: INK that makes no difference — hidden skills fire when named — so what it
#: really pins is that its name resolves like any other.
SKILLS = (
    ("research", "Investigate a question from primary sources."),
    ("code-review", "Review a merge request against the guardrails."),
    ("secret-audit", "Audit the credential store for stale entries."),
)


def _seed_skills(root: Path) -> None:
    """Write the fixture skills and wire them in as the only root."""
    for name, description in SKILLS:
        skill_dir = root / "skills" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description}\n---\n\nFixture body.\n"
        )
    os.environ["LOCAL_OPERATOR_SKILL_EXTRA_ROOTS"] = str(root / "skills")


def _hex(style: object) -> str | None:
    color = getattr(style, "color", None)
    return color.get_truecolor().hex.lower() if color else None


def _report(app: OperatorApp, editor: Editor, theme: str, label: str) -> None:
    """The numbers behind the still: the strip's inks, and the layout facts."""
    picker = editor.picker
    painted = [
        f"{segment.text!r}={_hex(segment.style)}"
        for segment in editor.render_line(0)._segments
        if segment.text.strip()
    ]
    names = [name for name, _ in picker.suggestions()]
    print(f"[{theme}/{label}] draft={editor.text!r}")
    print(f"    strip: {' '.join(painted)}")
    print(
        f"    picker: mode={picker.mode.value} open={picker.is_open()} matches={names}",
        flush=True,
    )
    print(
        f"    layout: editor={editor.size.width}x{editor.size.height} "
        f"screen={app.screen.size} virtual={app.screen.virtual_size} "
        f"scrollbar={app.screen.show_vertical_scrollbar}"
    )


async def _capture(out: Path, size: tuple[int, int]) -> None:
    from local_operator.tui import theme as theme_mod
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=size) as pilot:
        for _ in range(40):
            await pilot.pause()
            if app._session is not None:
                break
        editor = app.query_one(Editor)
        editor.focus()
        await pilot.pause()

        for theme in ("dark", "light"):
            theme_mod.set_theme(theme)
            app.refresh_css()
            await pilot.pause()
            for label, draft in STATES:
                editor.load_text(draft)
                editor.move_cursor(editor._end_of_buffer())
                # Two pauses: the app answers `SkillQueryOpened` one message-loop
                # tick after the keystroke, so a single pause captures the list
                # mid-fill — and the answer is what settles the ink.
                await pilot.pause()
                await pilot.pause()
                _report(app, editor, theme, label)
                save_capture(app, out / f"{label}-{theme}.svg")
            # Blank the draft between ramps so nothing carries across the theme
            # switch (the ink is theme-resolved at paint time; the buffer is not).
            editor.load_text("")
            await pilot.pause()


def main() -> None:
    out = Path(sys.argv[1]).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].lower().split("x")
        size = (int(cols), int(rows))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _seed_skills(root)
        os.chdir(root)
        asyncio.run(_capture(out, size))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
