"""Every frontend must discover the same command and prompt-consumption rules."""

import subprocess
import sys

import pytest

from local_operator.session.frontend_state import _slash_capabilities
from local_operator.slash_commands import (
    SLASH_COMMANDS,
    slash_command_for,
    whole_draft_command,
)


def test_canonical_frontend_capabilities_cover_the_shared_registry() -> None:
    commands = {command.name: command for command in SLASH_COMMANDS}
    capabilities = {cap.command: cap for cap in _slash_capabilities()}
    assert capabilities.keys() == commands.keys()
    for command in commands.values():
        for alias in command.names:
            assert slash_command_for(f"/{alias} argument") is command


def test_command_metadata_import_never_loads_the_textual_app() -> None:
    # Keep this in a fresh interpreter: the suite has already imported the TUI
    # during collection, which would make an in-process module census useless.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import local_operator.slash_commands; "
            "assert 'local_operator.tui.app' not in sys.modules; print('headless registry: ok')",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "headless registry: ok"


#: What a WHOLE draft invokes: the command's own name when the draft IS a
#: command, ``None`` when the draft is PROSE. Re-probed against this tree.
#:
#: The operator's report is the last block: a three-line draft whose first line
#: began with ``/mcp logout …`` was refused by the messages endpoint forever,
#: because the endpoint's blanket ``lstrip().startswith("/")`` test called it a
#: command and the composer had correctly planned it as prose. Nothing could
#: resend it. Every row here is one of the shapes that test got wrong.
WHOLE_DRAFT_TABLE = [
    # A bare word IS the command — Enter on the pre-selected picker row is
    # "accepting the suggestion", so the endpoint must keep refusing it. A
    # renderer that could not run it (capability off, dispatcher unwired) would
    # otherwise spend a paid turn on it.
    ("/compact", "compact"),
    ("/usage", "usage"),
    # Word + the value the command takes, on the same line.
    ("/model gpt-5", "model"),
    ("/theme dark", "theme"),
    ("/effort high", "effort"),
    ("/approvals plan", "approvals"),
    ("/loop stop", "loop"),
    # A start command's whole-draft form is still the command (`/goal clear` and
    # `/goal <objective>` are both forms of it); only the ARMED, whole-draft
    # `/goal` route is #209's business, and that is the composer's call.
    ("/goal ship it", "goal"),
    ("/goal", "goal"),
    ("/team ops fix this", "team"),
    ("/fork --switch here", "fork"),
    ("/btw what is this", "btw"),
    # A name-list command with NO name typed is the command: the list is open
    # in the picker and the listing IS the answer.
    ("/team", "team"),
    ("/mcp", "mcp"),
    # Aliases resolve to the same entry, so an alias spelling must not get a
    # different admission answer from its primary.
    ("/teams ops go", "team"),
    ("/models gpt-5", "model"),
    ("/agents ops", "agent"),
    # Surrounding whitespace is not part of the claim: this is still `/compact`.
    ("  /compact", "compact"),
    ("/compact   ", "compact"),
    # --- prose: the shapes the blanket leading-slash test refused -----------
    # A no-arg command followed by prose. The operator named this one first:
    # "`/compact` used to run and silently eat `hello`".
    ("/compact hello", None),
    ("/usage more prose", None),
    # A non-prefixing command word opening a draft: prose, so an MCP invocation
    # is never invented out of a sentence.
    ("/mcp logout seems to cause a crash", None),
    # A start command whose argument spans lines. The endpoint has no caret, so
    # it cannot tell "the command owns this" from "the user is still typing" —
    # and refusing it here would rebuild the permanent refusal this test exists
    # to prevent.
    ("/team ops fix this\nand then ship it", None),
    # QA round 2's Q4 shape, which is why the caret rule is not load-bearing for
    # safety: `/usage` takes no argument, so the body on the next line is prose.
    ("/usage\nfix it", None),
    ("/compact\nhello", None),
    # THE OPERATOR'S DRAFT, verbatim.
    (
        "/mcp logout seems to cause a crash on the TUI,\n"
        "can you review and fix that issue,\n"
        "replicate it and then fix and test end to end",
        None,
    ),
    # A word that names nothing, a path, a mid-sentence token, a token on a
    # later line — all prose, all accepted.
    ("/tema", None),
    ("/etc/hosts is wrong", None),
    ("/tmp/test\n\nThe above is a test file path", None),
    ("fix this /usage", None),
    ("can you check /usage for me", None),
    ("hello\n/team ops", None),
    # No leading slash at all is obviously prose; pinned so the predicate is
    # never rewritten to refuse on a substring test.
    ("what does /usage mean?", None),
    ("", None),
    ("   ", None),
    # --- the THIRD source: text the desktop validates or forwards -------------
    # These rows were this PR's round-1 MAJOR. Each is a whole-draft control the
    # desktop RUNS today (the route forwards the text, or the runtime dispatch
    # reads it), so each must stay REFUSED: a client whose command surface is off
    # plans `send` for everything, and a control accepted here would spend a paid
    # model turn on it.
    ("/mcp logout", "mcp"),
    ("/login openai", "login"),
    ("/logout openai", "logout"),
    ("/provider openai", "provider"),
    ("/accounts x", "accounts"),
    ("/rename x", "rename"),
    ("/rename my thing", "rename"),
    ("/resume abc", "resume"),
    ("/new foo", "new"),
    ("/reload abc", "reload"),
    ("/settings foo", "settings"),
    ("/search foo", "search"),
    ("/usage on", "usage"),
    ("/skills x", "skills"),
    ("/analytics view", "analytics"),
    ("/move ~/x", "move"),
    ("/move ~/my folder", "move"),
    ("/stop now", "stop"),
    ("/fast on", "fast"),
    # And each shape's PROSE side, which is what keeps them from swallowing the
    # operator's report: a sentence is not a selector, an unknown provider is not
    # a provider, and an over-long MCP invocation is not one either.
    ("/usage more prose", None),
    ("/mcp logout seems to cause a crash", None),
    ("/login zzz", None),
    ("/mcp zzz", None),
    ("/stop now and then", None),
    # A command whose path DROPS the trailing text is prose, by the same criterion
    # that keeps `/compact hello` prose: `context`'s owner dispatch calls
    # `_context_slash_result` with no args and `native_action` has no branch.
    ("/context x", None),
    # The word/argument split is the TOKENIZER's boundary, not a literal space:
    # `/usage\rfix it` is `usage` plus the two-token argument `fix it` (prose),
    # which is how both composers read it. Pinned because splitting on `" "` here
    # read it as one WORD-shaped token and refused a draft planned as prose.
    ("/usage\rfix it", None),
    # ...while the tab forms a SPACE would also have refused still decide as they
    # always did: one selector token, or a command whose text is its own.
    ("/goal\tship it", "goal"),
    ("/usage\ton", "usage"),
    # And the class the same change moves, pinned rather than left to the count.
    # Round 3 measured it and the invariant is stronger than the tally: a
    # tab-separated draft now decides exactly as the SAME draft with a space does
    # (294 of 294 agree), and 48 altered decision — 36 refused→prose, like these
    # two, and 12 prose→refused, like `/move\tsome prose`. Both directions are the
    # one boundary matching the tokenizer: the old literal-space split either read
    # the draft as a single WORD and refused prose both composers send, or it took
    # the word after the space as the whole argument and admitted chat for a
    # command both composers run. `Move` is the second case; these rows are the
    # first.
    ("/usage\tsome prose", None),
    ("/mcp logout\tand then", None),
    ("/move\tsome prose", "move"),
]


@pytest.mark.parametrize("draft,expected", WHOLE_DRAFT_TABLE)
def test_whole_draft_command_matches_the_policy_table(draft: str, expected: str | None) -> None:
    resolved = whole_draft_command(draft)
    if expected is None:
        assert resolved is None, f"{draft!r} must be PROSE, got /{resolved[0].name}"
    else:
        assert resolved is not None, f"{draft!r} must be the /{expected} command, got prose"
        assert resolved[0].name == expected


def test_any_newline_makes_a_draft_prose() -> None:
    """Stated separately because the rule is load-bearing, not incidental.

    Every prefixing command is reachable single-line, so the newline rule is the
    one thing standing between a body that opens with a command word and the
    permanent refusal the operator reported.

    ANY newline, including one that only TRAILS. The earlier "interior newline"
    test (``"\n" in text.strip()``) read ``/team ops\n`` as the command while the
    composer plans ``send`` for it — the caret sits on the empty last line — so
    the route refused a draft the composer had already decided to send, which is
    the permanent-refusal class this whole rule exists to remove. The endpoint
    has no caret, so the faithful translation of the composer's per-LINE decision
    is "a newline anywhere means prose".

    Whitespace that is NOT a newline still does not make prose: leading and
    trailing spaces are stripped, so ``  /compact`` and ``/compact   `` stay the
    command — neither turns the draft into two lines.
    """
    drafts = [
        # interior newlines: the operator's shape, and the QA round 2 Q4 regression
        "/team ops fix this\nand ship it",
        "/team ops fix this\n\nand ship it",
        "/goal ship it\nand then verify",
        "/loop keep going\nuntil the suite is green",
        "/usage\nfix it",
        "/compact\nhello",
        "/model gpt-5\nplease check the logs",
        "/mcp logout\nseems to be broken",
        "hello\n/team ops",
        "/tmp/test\n\nThe above is a test file path",
        # trailing newlines: a paste or Shift+Enter leaves the caret on the empty
        # last line, which is the case the two hosts disagreed about
        "/team ops\n",
        "/compact\n",
        "/compact\n   ",
        "/usage\n\t",
        "/goal ship it\n ",
        "\n/compact",
        "\n",
    ]
    for draft in drafts:
        assert whole_draft_command(draft) is None, draft

    # The non-newline whitespace counterpart, pinned so "any newline" is not
    # widened into "any whitespace at all is prose".
    assert whole_draft_command("/team ops") is not None
    assert whole_draft_command("  /compact") is not None
    assert whole_draft_command("/compact   ") is not None
    assert whole_draft_command("/compact\t") is not None
