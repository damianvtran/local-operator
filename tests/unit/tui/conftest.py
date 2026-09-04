"""Hermetic environment and shared harnesses for the TUI suite.

Snapshot frames were captured with colour enabled and shimmer off; a caller
that exports NO_COLOR (a common developer default) would otherwise fail all
three snapshots for reasons that have nothing to do with the code under
test. The pins live in fixtures — scoped and reverted — instead of module
import time, so collection order never leaks environment into other suites.

This file also reclaims the temporary directories this suite allocates; see
``_reclaim_bare_mkdtemp`` for why the cleanup lives here rather than at the
~30 call sites that create them.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from textual.app import App, ComposeResult

from local_operator.tui import animation
from local_operator.tui import theme as theme_mod
from local_operator.tui.settings import settings_reload
from local_operator.tui.widgets.transcript import TranscriptView

#: Whatever occupied ``tempfile.mkdtemp`` when this conftest was imported — in
#: practice the CPython original, since nothing in this venv patches it before
#: import, but the guarantee is "the import-time binding", not "the stdlib
#: function". If a plugin ever did patch it earlier, delegating here would
#: preserve that plugin's semantics rather than bypass them, which is the
#: behaviour we want either way.
#:
#: ``_reclaim_bare_mkdtemp`` delegates to THIS rather than to whatever occupies
#: the slot at setup time, which is what keeps wrappers from stacking: a test
#: that patches ``mkdtemp`` itself has its ``monkeypatch`` undone AFTER this
#: fixture (pytest resolves ``monkeypatch`` first, so it tears down later),
#: leaving OUR wrapper in the slot for the next test to capture as its "real"
#: function. Reading the live slot per test therefore grows the chain by one
#: layer per patching test — measured 2, 3, 4, 5, 6, 7 across six such tests,
#: and depth 60 on a 60-test storm; anchoring here holds it flat. Behaviour
#: survived the stack (every layer delegates), but the growth was unbounded
#: within a session.
_STDLIB_MKDTEMP = tempfile.mkdtemp

#: The real stylesheet, so styled tests exercise the shipped rules rather
#: than a convenient approximation of them.
TCSS_PATH = str(Path(theme_mod.__file__).parent / "local_operator.tcss")


@pytest.fixture(autouse=True)
def _reclaim_bare_mkdtemp() -> Iterator[None]:
    """Remove the ``tempfile.mkdtemp()`` dirs this suite abandons, per test.

    WHY THIS EXISTS. Five files here (``test_slash_echo``, ``test_team_chart``,
    ``test_slash_command_images``, ``test_paste_collapse``,
    ``test_credential_command``) build ``TeamRegistry``/``AgentRegistry`` roots
    from a bare ``tempfile.mkdtemp()`` and never remove them: 65 directories
    per serial run of ``tests/unit/tui``, unbounded in COUNT across runs. It is
    the residue of #564, which fixed the unbounded-in-SIZE evidence bundles
    (43,627 abandoned dirs / ~30 GB filled the operator's disk) and explicitly
    deferred this one.

    WHY IT IS A WRAPPER HERE AND NOT A FIXTURE AT THE CALL SITES. This leak has
    been fixed and reverted twice, both times for the same reason. Migrating
    the five files to a ``tmp_path``-backed ``scratch_dir`` fixture added
    fixture dependencies to those modules and perturbed xdist's ``worksteal``
    scheduling, landing an unrelated timing-sensitive pilot test
    (``test_app_pilot.py`` ::
    ``test_a_swap_leaves_the_ledger_matching_the_new_sessions_history``) in a
    failing window 4/4 in CI. A second attempt using a plain-function helper in
    ``tests/unit/tui/_scratch.py`` destabilised a different shard
    (``test_word_caret``, 2/2). Both were reverted.

    So the constraint this fix is written against is: **do not change what
    pytest collects, schedules, or resolves.** This fixture is autouse with no
    parameters, so it introduces no new fixture-graph edges and no per-test
    dependency on ``tmp_path``; the suite's other autouse fixture already makes
    every test in this package carry one. Collection, ordering, test ids and
    the per-test fixture closure are unchanged from the pre-fix tree — the only
    difference is that ``mkdtemp`` is a different function object while a test
    body runs.

    WHAT IT RECLAIMS, AND WHAT IT DELIBERATELY DOES NOT. The rule is by PARENT
    PATH, not by keyword: only allocations whose parent is the system temp root
    are tracked, which is exactly the bare ``mkdtemp()`` shape that leaks. That
    is what leaves ``dir=``-staged directories alone —
    ``TeamRegistry._save_team_locked`` stages a row under ``teams_dir`` and its
    backup under ``target.parent``, both outside the temp root, and sweeping
    either would delete a live registry's staging directory mid-write. Note the
    rule is deliberately about WHERE the directory landed rather than HOW it was
    asked for: an explicit ``mkdtemp(dir=<the temp root>)`` IS swept, because it
    is indistinguishable on disk from the bare call this reclaims, and a caller
    staging into the shared temp root has not opted out of cleanup.
    ``tempfile.TemporaryDirectory`` routes through ``mkdtemp`` and so is seen,
    but it removes itself first and the sweep below is then a no-op.
    ``tmp_path`` does not route through ``mkdtemp`` at all (pytest uses
    ``make_numbered_dir``), so it is untouched and keeps its own rotation.

    Sweeping per test rather than at process exit keeps the disk flat during a
    long run instead of only at the end of one, and attributes nothing across
    tests: the list is per-test state, so a directory is only ever removed by
    the test that allocated it.

    WHY THIS RECLAIMS RATHER THAN DETECTS, unlike the evidence suite's guard in
    ``tests/unit/evaluation/evidence/conftest.py``. That one can only DETECT: its
    leaks are created inside embedded subprocess scripts, which no in-process
    finalizer can reach, so failing the test is the only lever it has. These
    leaks are ordinary in-process allocations, so they can simply be reclaimed —
    and prevention beats detection here, because a new call site added later is
    cleaned up automatically instead of failing a test that a future author then
    has to remediate. There is deliberately no new detector test file to go with
    this: CI shards by ``sorted(glob('tests/unit/**/test_*.py'))`` with
    ``i % 5``, so adding ONE file re-indexes the files sorting after it into
    different shards (30 of 349 for a plausibly-named file here, up to ~100 for
    one sorting to the front) — scheduling churn this fix exists to avoid. The
    decisive reason, though, is that reclaiming PREVENTS by construction where a
    detector only reports: a bare ``mkdtemp()`` added later is cleaned up with
    no new test to maintain, and a detector folded into this conftest would have
    to assert on a temp root that concurrent sibling worktrees also write to,
    making it flaky by construction.

    To re-measure the leak (serial and private, so sibling worktrees cannot
    pollute the count). The expected count is whatever the CONTROL arm reports,
    not a number pinned here — it moves as call sites come and go; it was 65
    directories / 724 KB when this landed, and the invariant that matters is
    that this arm reports zero::

        T=/tmp/leakprobe; rm -rf $T; mkdir -p $T
        env -u NO_COLOR TERM=xterm-256color TMPDIR=$T \\
            .venv/bin/python -m pytest tests/unit/tui -q -p no:randomly -n0
        ls -1 $T | grep -v '^pytest-\\|data-gym-cache' | wc -l   # 0 here

    The ``grep -v`` is load-bearing, not decoration: pytest's own ``pytest-of-*``
    basetemp and tiktoken's ``data-gym-cache`` both live in the same directory
    and are not leaks. Dropping the filter when retyping the command is the
    fastest way to get a wrong answer out of it.
    """
    # Anchored to the import-time original, never to the live slot — see
    # ``_STDLIB_MKDTEMP`` for the stacking this avoids.
    real_mkdtemp = _STDLIB_MKDTEMP
    # Resolved once per test: comparing REAL paths keeps macOS's
    # /var -> /private/var symlink from making every parent comparison false.
    temp_root = os.path.realpath(tempfile.gettempdir())
    created: list[str] = []

    def tracking_mkdtemp(*args: Any, **kwargs: Any) -> str:
        path = real_mkdtemp(*args, **kwargs)
        if os.path.realpath(os.path.dirname(path)) == temp_root:
            created.append(path)
        return path

    # Not `monkeypatch`: this fixture takes no parameters on purpose (see the
    # docstring), and `try`/`finally` restores on every exit path anyway.
    tempfile.mkdtemp = tracking_mkdtemp  # type: ignore[assignment]
    try:
        yield
    finally:
        # Restore only while we still own the slot: a test that patched
        # `mkdtemp` and has not yet had its own `monkeypatch` undone is holding
        # the slot legitimately, and clobbering it here would strip a patch the
        # test still expects to be in force.
        #
        # KNOWN LIMIT (review round 2, MINOR 2), latent and deliberately not
        # designed around: if a test that patches `mkdtemp` is the LAST one in
        # this package, this declines to restore, its `monkeypatch` then
        # reinstates our wrapper, and nothing afterwards removes it — an
        # out-of-package test would allocate into a `created` list nobody
        # sweeps. No test in `tests/unit/tui` patches `mkdtemp` today (the only
        # one in the repo is `tests/unit/test_agents.py`, another package), so
        # this is unreachable. The fix, should it ever bite, is to restore when
        # the slot is not ours but our wrapper is still reachable in the chain;
        # it is not applied now because it trades a real, measured guarantee for
        # a hypothetical one.
        if tempfile.mkdtemp is tracking_mkdtemp:
            tempfile.mkdtemp = real_mkdtemp  # type: ignore[assignment]
        for path in created:
            # A test that cleaned up after itself, or handed the directory to
            # something that did, is the normal case rather than an error.
            shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def hermetic_tui_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_SHIMMER", "1")
    # Terminal-focus animation gating is a module global (it has to be: the
    # surfaces reading it are built at different times and one is not a
    # Widget). The suite shares a process, so a test that blurs the app would
    # otherwise leave every later test's timers at the reduced cadence.
    # Reverted here rather than in each test for the same reason the env pins
    # above are: a leak like this fails somewhere else entirely.
    animation.reset_animation_focus()
    # The display-flag cache is a module global for the same reason, and leaks
    # the same way (QA round 5, Q5-M1). Any test that writes a `display.*` key
    # and then reads one — the config-watch propagation tests do exactly that —
    # leaves the flag it wrote in `_cache`, which outlives its `tmp_path`. A
    # later test reading `motion_enabled()` or a glyph then sees another test's
    # config: `test_render_throttling.py` and `test_welcome.py` both fail that
    # way, and only when they land on the same xdist worker AFTER the polluter,
    # which is why it presents as a load flake rather than an ordering bug.
    # Reset BEFORE each test rather than after, so a test that populates the
    # cache cannot poison its successors no matter how it exits.
    settings_reload()
    # Pin the tool-row icon mode host-independently. `nerd_icons_enabled()` now
    # autodetects from terminal-emulator env markers (glyphs.py), so the icon a
    # row leads with depends on the HOST: a dev box in ghostty/cmux renders the
    # Nerd glyphs, a bare CI runner with no markers renders the ASCII fallback.
    # The rendering/snapshot tests here assert row CONTENT written against the
    # historical Nerd-on default, so without a pin they pass locally and fail on
    # CI (the plain `write` icon is `+`, which `test_unknown_counts_render_nothing`
    # asserts is absent). Seed a positive ghostty marker so the whole suite
    # renders in Nerd mode everywhere; the kill switch is cleared so a host that
    # exports it cannot force the opposite. Tests that exercise the detection
    # itself set their own markers/settings and override this within the test.
    monkeypatch.delenv("LOCAL_OPERATOR_NO_NERD_ICONS", raising=False)
    monkeypatch.setenv("GHOSTTY_BIN", "/usr/bin/ghostty")
    # `WelcomeView` pins the opening tip from `TERM_PROGRAM` alone
    # (`terminals.is_apple_terminal` does not look at `GHOSTTY_BIN`). A host
    # running the suite inside Terminal.app would otherwise open every splash
    # on the paste tip and fail assertions that the first frame is `TIPS[0]`
    # (review round 1, F1). Tests that exercise the Apple_Terminal pin set
    # this themselves.
    monkeypatch.delenv("TERM_PROGRAM", raising=False)
    # The splash starts a one-shot PyPI probe on mount. Unit tests must not
    # pay a 5 s timeout (or a real GET) for news they are not asserting.
    # Patch the worker, not ``check_latest``: ``/update`` needs the real
    # function so its own mocks can drive newer/same/error.
    monkeypatch.setattr(
        "local_operator.tui.app.OperatorApp._check_for_update",
        lambda self: None,
    )
    # The caret used to be pinned here too: `TextArea.cursor_blink` was patched
    # off for the whole suite because a blinking caret made whether a captured
    # frame contained one a coin flip, and the boot snapshot failed against a
    # file it had just regenerated. The product now ships a solid caret (see
    # `Editor.__init__`), so the pin is gone: a fixture that forces the
    # behaviour under test would make the editor's own caret tests vacuous, and
    # the strobe it hid from this suite was exactly the one users were seeing.


class StyledTranscriptApp(App[None]):
    """A transcript under the REAL sheet and the real brand variables.

    Widget-level assertions answer "does this build the right content"; this
    app answers "does the shipped CSS then turn that content into the rows
    and colours we claim", which is a different question — and the one that
    catches height, spacing, and hover regressions that unit tests cannot
    see. Nothing else is mounted, so no other rule can interfere.
    """

    CSS_PATH = TCSS_PATH

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        variables.update(theme_mod.tcss_variable_map())
        return variables

    def compose(self) -> ComposeResult:
        yield TranscriptView()


#: Mirrored from ``local_operator.tui.app`` so this helper does not import
#: the app module (conftest loads for every TUI test). The match is pinned
#: by ``test_composer_markers_match_the_app``.
PROMPT_CHEVRON = "❯"
SHELL_CHEVRON = "$"


def composer_cells(app: App[None]) -> list[tuple[str, str | None, str | None]]:
    """(text, fg hex, bg hex) for every segment of the composer's row.

    Shared because the composer's focus state is not a widget attribute: a
    caret is a cell whose colours have been swapped, and the chevron's ink is
    a colour the stylesheet resolved. Both are only answerable from what the
    terminal was SENT, which is what ``render_strips`` returns.

    Located by ``#prompt-chevron``'s laid-out row rather than by scanning
    for a glyph. Bang-mode paints ``$`` instead of ``❯`` (#385), and ``$``
    is ordinary prose — a scan would steal the first dollar in the
    transcript. The widget's ``region.y`` is the compositor strip index
    (measured); the ``/resume`` picker is a pushed Screen, so while it is
    up the composer is genuinely off the frame and the raise is the honest
    answer rather than a missed row.
    """
    try:
        chevron = app.query_one("#prompt-chevron")
    except Exception as exc:
        raise AssertionError("the composer row is not on the frame at all") from exc
    y = chevron.region.y
    strips = list(app.screen._compositor.render_strips())
    if y < 0 or y >= len(strips):
        raise AssertionError("the composer row is not on the frame at all")
    cells = []
    for segment in strips[y]._segments:
        style = segment.style
        fg = style.color.get_truecolor().hex.lower() if style and style.color else None
        bg = style.bgcolor.get_truecolor().hex.lower() if style and style.bgcolor else None
        cells.append((segment.text, fg, bg))
    return cells


def caret_cells(cells: list[tuple[str, str | None, str | None]]) -> list[str]:
    """What the caret is sitting ON: cells drawn with its inverted ground.

    The TEXT is returned rather than a count because "is there a caret" and
    "is the caret eating a letter" are the two questions this app has got
    wrong, and only the second one needs the content.
    """
    caret_ground = theme_mod.semantic_color("fg").lower()
    return [text for text, _, bg in cells if bg == caret_ground]


def chevron_colour(cells: list[tuple[str, str | None, str | None]]) -> str | None:
    """The prompt marker's ink — `fg` while the composer has focus, else `dim`.

    NOT the accent (D5): green means a turn is live, and a marker that is lit
    in nearly every frame cannot also mean that. Focus is a brightness step in
    the same neutral ramp, which is a 3.86x luminance move against `dim` where
    the accent was 2.15x.
    """
    markers = {PROMPT_CHEVRON, SHELL_CHEVRON}
    # Exact cell, not a substring: `$` is ordinary prose, so a typed
    # `$ ls` on the same strip must not steal the marker's ink.
    return next(fg for text, fg, _ in cells if text.strip() in markers)
