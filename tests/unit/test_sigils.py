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

import pytest

from local_operator.sigils import (
    SKILL_BLOCK_CLOSE,
    SKILL_BLOCK_OPEN,
    SKILL_BLOCK_PREAMBLE,
    SlashContext,
    defuse_skill_block,
    is_boundary,
    skill_block_names,
    skill_block_spans,
    skill_mentions,
    skill_name_spans,
)

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


# --- the `$skill` trigger -----------------------------------------------------

NAMES = frozenset({"research", "code-review", "secret_audit", "a.b", "a-b", "path"})


def _mentions(text: str, names: frozenset[str] = NAMES):
    return skill_mentions(text, names)


def _names(text: str, names: frozenset[str] = NAMES) -> list[str]:
    return [mention.name for mention in _mentions(text, names)]


@pytest.mark.parametrize(
    "text",
    [
        # Money and glue: the vocabulary decides, and `$` is not a token here.
        "costs $5",
        "$100 for it",
        "a$b",
        "costs$5",
        # Environment variables carry no lowercase evidence inline.
        "echo $PATH",
        "echo $DEBUG",
        # A `$` with nothing usable after it, or an ASCII-impossible head.
        "$",
        "$ research",
        "$_x",
        "$ßeta",
        # Explicit escapes with no new grammar: the character before `$` must be
        # whitespace or the start of the text.
        "$$research",
        "`$research`",
        "($research)",
        '"$research"',
        # An unknown name and a name that only prefixes a real one.
        "$unknown-skill do the thing",
        "$researc please",
        # A `$` glued to a preceding word on a later line is still glued.
        "first line\nx$research",
    ],
)
def test_no_mention_fires(text: str) -> None:
    assert _mentions(text) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Mid-sentence, line start, and after a newline all fire.
        ("$research fix it", ["research"]),
        ("fix it with $research", ["research"]),
        ("line one\n$research fix it", ["research"]),
        ("line one\nuse $research here", ["research"]),
        # Trailing punctuation is trimmed off the token at both positions.
        ("$research. now", ["research"]),
        ("use $research. now", ["research"]),
        ("$research, now", ["research"]),
        ("$research?", ["research"]),
        # Dots and hyphens inside a name are part of it.
        ("$a.b x", ["a.b"]),
        ("$a-b x", ["a-b"]),
        ("$code-review this", ["code-review"]),
        ("$secret_audit", ["secret_audit"]),
        # Several skills, text order.
        ("$code-review then $research", ["code-review", "research"]),
        # Dedupe by RESOLVED name, first-seen order.
        ("$research and $Research", ["research"]),
    ],
)
def test_mentions_fire_in_text_order(text: str, expected: list[str]) -> None:
    assert _names(text) == expected


def test_an_exact_name_fires_inline_whatever_its_case() -> None:
    """A skill genuinely named `DEBUG` is reachable mid-sentence."""
    assert _names("run $DEBUG now", frozenset({"DEBUG"})) == ["DEBUG"]


def test_inline_case_insensitive_matching_needs_a_lowercase_letter() -> None:
    assert _names("run $Research now") == ["research"]
    assert _names("run $research now") == ["research"]
    assert _names("run $RESEARCH now") == []


def test_a_leading_token_keeps_the_legacy_case_insensitive_rule() -> None:
    """The override the operator settled: leading is unchanged, inline guarded."""
    assert _names("$RESEARCH fix it") == ["research"]
    assert _names("  $RESEARCH fix it") == ["research"]
    assert _names("echo $RESEARCH") == []


def test_a_lowercase_env_var_fires_only_when_a_skill_has_that_name() -> None:
    """`$path` is evidence of intent only if the vocabulary contains `path`."""
    assert _names("echo $path") == ["path"]
    assert _names("echo $path", frozenset({"research"})) == []
    assert _names("echo $PATH") == []


def test_the_span_excludes_trimmed_punctuation() -> None:
    mention = _mentions("use $research. now")[0]
    assert (mention.start, mention.end, mention.text) == (4, 13, "$research")
    assert mention.leading is False


def test_the_leading_flag_is_whitespace_only_prefix() -> None:
    assert _mentions("   $research")[0].leading is True
    # Only whitespace precedes a token on a later line, and the legacy rule
    # (`lstrip` then match) treated exactly that as leading.
    assert _mentions("\n$research")[0].leading is True
    assert _mentions("fix $research")[0].leading is False


def test_a_skipped_span_is_ignored_before_the_dedupe() -> None:
    """A quoted name inside a block must not hide a real one after it.

    The spans are the block grammar's, so this is the same rule
    ``plan_attachments`` uses on a forwarded payload — and the dedupe order is
    what makes it a fix rather than a suppression.
    """
    text = "see $research in the body\n\nand really $research now"
    start = text.index("$research")
    end = start + len("$research")
    assert [m.name for m in skill_mentions(text, NAMES, skip_spans=[(start, end)])] == ["research"]
    assert [m.start for m in skill_mentions(text, NAMES, skip_spans=[(start, end)])] == [
        text.rindex("$research")
    ]


# --- the block grammar --------------------------------------------------------


def test_skill_name_spans_reports_the_untrimmed_run() -> None:
    assert skill_name_spans("$research. next") == [(0, 10, "research.")]
    assert skill_name_spans("costs$5") == []


def test_skill_block_spans_needs_the_preamble() -> None:
    assert skill_block_spans("<operator-skills>\n\nno preamble") == []


def test_skill_block_names_reads_heads_not_bodies() -> None:
    block = (
        SKILL_BLOCK_OPEN
        + "\n\n"
        + SKILL_BLOCK_PREAMBLE
        + "\n\n"
        + '<skill name="research">\nquoting <skill name="fake">\n</skill>\n\n'
        + SKILL_BLOCK_CLOSE
    )
    assert skill_block_names(block) == ["research"]


def test_block_names_ignore_a_head_quoted_inside_a_body() -> None:
    """The docstring's promise, tested structurally: depth, not line scanning."""
    block = (
        SKILL_BLOCK_OPEN
        + "\n\n"
        + SKILL_BLOCK_PREAMBLE
        + "\n\n"
        + '<skill name="research">\nbody quoting a head:\n<skill name="zzz">\n'
        + "</skill>\n\n"
        + SKILL_BLOCK_CLOSE
    )
    assert skill_block_names(block) == ["research"]


def test_defusing_neutralises_a_body_s_head() -> None:
    body = defuse_skill_block('a <skill name="zzz"> b')
    assert '<skill name="' not in body
    assert "\u200b" in body


def test_defusing_keeps_the_body_marker_out_of_the_scanner() -> None:
    body = defuse_skill_block("a <operator-skills> b </operator-skills> c")
    block = (
        SKILL_BLOCK_OPEN
        + "\n\n"
        + SKILL_BLOCK_PREAMBLE
        + "\n\n"
        + f'<skill name="x">\n{body}\n</skill>\n\n'
        + SKILL_BLOCK_CLOSE
    )
    assert len(skill_block_spans(block)) == 1
