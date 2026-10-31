#!/usr/bin/env python3
"""Generate the `$skill` token fixture shared by the TUI and the mobile web client.

WHY THIS FILE EXISTS
--------------------
`web/src/lib/skill-token.ts` re-implements the TUI's ``$name`` parser
(``local_operator/tui/widgets/command_picker.py``'s `skill_token`,
`skill_token_is_leading`, and the SKILL branch of `completion_for`). A port
that agrees by inspection is a port that drifts in silence: nothing in the web
bundle can read Python, and nothing in the Python suite can read the bundle.
So the bridge is ONE generated artifact,
``local_operator/mobile/web/src/lib/skill-token.parity.json``, read from both
sides:

* ``tests/unit/mobile/test_tui_bridge.py`` asserts the committed file is this
  script's own bytes (provenance), so a hand edit that shrinks coverage fails;
* ``web/src/lib/skill-token.test.ts`` runs every case through the TS port and
  asserts the same answer.

STRINGS, NEVER OFFSETS. The fixture must be language-neutral: Python indexes
code points and JavaScript indexes UTF-16 code units, so a stored caret or
token offset would mean different positions on the two sides the moment an
emoji appears. Every case therefore carries the caret as the ``head``/``tail``
split (caret = ``len(head)`` in each language's own units) and every expected
span as the strings around it (`prefix` / `token` / `suffix`) — no numbers
except the record's own identity.

CASES pin the rule the two surfaces must share, not a happy path:
line-start and inline tokens; the token's first and last cell and one past;
the Python-vs-JS whitespace disagreements (`\ufeff` and `\u200b` are NOT
whitespace in Python; `\x85`, `\x1c`–`\x1f` are); NBSP and an em space; an
emoji before the sigil (the UTF-16 case); CRLF; and the punctuation that must
NOT open a token (`costs$5`, `$$x`, backticks, parentheses, quotes). Each case
also carries the insertion result for the fixed name ``planning``, so the
span-replacement arithmetic is pinned by the same artifact.

    .venv/bin/python scripts/generate_skill_token_parity.py
    .venv/bin/python scripts/generate_skill_token_parity.py --check
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_operator.tui.widgets.command_picker import (  # noqa: E402
    CompletionMode,
    completion_for,
    skill_token,
    skill_token_is_leading,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "local_operator/mobile/web/src/lib/skill-token.parity.json"
)

#: The name every insertion case fills in. Not a real skill; chosen so the
#: result differs from every typed token in the set.
FILL = "planning"

#: ``(note, head, tail)`` — the caret sits between ``head`` and ``tail``.
#: Deliberately NOT a table of caret offsets: see the module docstring.
CASES: list[tuple[str, str, str]] = [
    ("bare $ at line start lists everything", "$", ""),
    ("caret before the sigil is inside the token", "", "$research"),
    ("caret after the sigil, name one char", "$", "research"),
    ("caret mid-name", "$re", "search"),
    ("caret on the token's last cell is inside", "$research", ""),
    ("caret one past the token closes it", "$research ", ""),
    ("inline token after prose", "fix ", "$research the bug"),
    ("inline token on a later line", "first line\n", "$second"),
    ("a token on another line does not open", "$x\n", "plain"),
    ("uppercase token parses; matching is the caller's gate", "uses ", "$PATH here"),
    ("money parses; matching is the caller's gate", "costs ", "$5"),
    ("glued sigil is punctuation", "costs$", "5"),
    ("double sigil opens on the first", "a ", "$$x"),
    # THE "LAST BOUNDARY $ AT OR BEFORE THE CARET" RULE (agent review m2): the
    # only multi-sigil case used to be `$$x`, which cannot distinguish "the last
    # one" from "the first one". These two can: an earlier token's `$` would
    # close the parse (caret past its end), and a sigil exactly AT the caret is
    # inside its own token.
    ("the last boundary $ before the caret wins", "$a $r", "e"),
    ("a sigil exactly at the caret opens its own token", "$a ", "$b"),
    ("backtick glue is not a boundary", "`", "$x`"),
    ("paren glue is not a boundary", "(", "$x)"),
    ("quote glue is not a boundary", '"', '$x"'),
    ("NBSP separates", "a\u00a0", "$x"),
    ("em space separates", "a\u2003", "$x"),
    ("NEL separates in Python but not in JS \\s", "a\x85", "$x"),
    ("BOM is NOT whitespace in Python but is in JS \\s", "a\ufeff", "$x"),
    ("ZWSP is not whitespace", "a\u200b", "$x"),
    ("file separator (\\x1c) separates in Python", "a\x1c", "$x"),
    ("emoji before the sigil (UTF-16 vs code points)", "🙂 ", "$x"),
    ("leading whitespace is preserved, token still opens", "  ", "$research"),
    ("CRLF: the caret's line is split on \\n and \\r stripped", "$a\r", "\nb"),
    ("CRLF: a token after the break is on its own line", "line\r\n", "$x"),
    ("dots are part of the name", "$", "code-review.v2"),
    ("underscores are part of the name", "$", "under_score"),
    ("dashes are part of the name", "$", "dash-name"),
    ("token with a trailing full stop includes it", "run ", "$research."),
]


def _insertion(text: str, caret: int) -> tuple[str, str] | None:
    """The buffer and caret accepting ``FILL`` produces, as the strings around it."""
    filled = completion_for(text, caret, CompletionMode.SKILL, FILL, ())
    if filled is None:
        return None
    completed, at = filled
    return completed[:at], completed[at:]


def build() -> dict[str, object]:
    """The fixture's content: every case with the TUI's own answer for it."""
    cases: list[dict[str, object]] = []
    for note, head, tail in CASES:
        text = head + tail
        caret = len(head)
        token = skill_token(text, caret)
        case: dict[str, object] = {"note": note, "head": head, "tail": tail}
        if token is None:
            case["token"] = None
        else:
            case["prefix"] = text[: token.start]
            case["token"] = text[token.start : token.end]
            case["query"] = token.query
            case["suffix"] = text[token.end :]
            case["leading"] = skill_token_is_leading(text, token)
            inserted = _insertion(text, caret)
            assert inserted is not None, (note, head, tail)
            case["insertHead"], case["insertTail"] = inserted
        cases.append(case)
    return {
        "_": (
            "Generated by scripts/generate_skill_token_parity.py — do not edit by "
            "hand. Read by the vitest suite (skill-token.test.ts) and the Python "
            "suite (tests/unit/mobile/test_tui_bridge.py) so the phone's ported "
            "$skill grammar and the TUI's original cannot drift apart in silence. "
            "Caret = head|tail; spans are strings, never offsets."
        ),
        "cases": cases,
    }


def render() -> str:
    """The fixture's exact bytes — one case per line, so a diff names what moved."""
    data = build()
    cases = data["cases"]
    assert isinstance(cases, list)
    title = json.dumps(data["_"], ensure_ascii=False)
    # ASCII-escaped bodies on purpose: \x85, NBSP and emoji are invisible or
    # confusable in a diff, and this file's job is to be read when it changes.
    body = ",\n".join(f"  {json.dumps(case)}" for case in cases)
    return f'{{\n "_": {title},\n "cases": [\n{body}\n ]\n}}\n'


def main() -> None:
    """Write the fixture, or (``--check``) prove the committed one came from here."""
    cases = build()["cases"]
    assert isinstance(cases, list)
    if "--check" in sys.argv[1:]:
        if FIXTURE.read_text(encoding="utf-8") == render():
            print(f"{FIXTURE} is current ({len(cases)} cases)")
            return
        print(f"{FIXTURE} is STALE — regenerate it from this script", file=sys.stderr)
        raise SystemExit(1)
    FIXTURE.write_text(render(), encoding="utf-8")
    print(f"wrote {FIXTURE} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
