"""The element -> token binding table for the transcript and tool row.

Design proposal: `docs/proposals/theme-colour-budget.md` §2. Before this
module, ``element -> token`` lived as inline ``Style(color=semantic_color(...))``
calls scattered across ``markdown_theme.py`` and ``tool_card.py`` (§2.1) — no
object existed to enumerate elements from, so a rebinding was a multi-file
edit with no way to see the whole map, and the perceptual-distinctness gate
§4 wants could not be written at all.

:data:`BINDINGS` is the single source of truth. Widgets no longer call
``theme.semantic_color`` directly for the transcript/tool-row surfaces they
own; they resolve a stable element id through :func:`style`,
:func:`markdown_theme` or :func:`syntax_styles` instead. ``semantic_color``
itself stays public and unchanged (§2.2) — 365 call sites across 27 files
reach it, and this table covers only the two surfaces the colour-budget
proposal measured and changed.

This module imports :mod:`local_operator.tui.theme` and nothing else in the
TUI, so it stays importable from a test with no app running.

**Role, by convention** (not derived — written per binding so the table stays
self-documenting): a binding's :class:`Role` reflects what question THIS
USE of the ink answers, not merely which token it happens to spend.
``warning`` (amber) is one of the budget's three OUTCOME tokens (§1.2), but
the syntax ramp's ``code.string``/``code.number``/``code.keyword_constant``
spend it on code LITERALS, not on "did this work?" — so those three are
:attr:`Role.NEUTRAL` despite the token, matching §1.5's "syntax highlighting
... is inside budget" (an accepted existing use, not a new one). The inverse
case is ``markdown.h1``: it spends `accent`, whose canonical role is
LIVENESS, to buy document STRUCTURE (§1.3, Rule C) — there is no fourth
"structure" role in the enum, and §8 risk 2 frames the h1 accent explicitly
as growing the LIVENESS accent-site count from four to five ("the accent
list grows ... a sixth requires removing one"), so it is counted there
rather than invented a role that would let it dodge the budget it is
spending against.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from pygments.token import (
    Comment,
    Error,
    Generic,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Token,
    _TokenType,
)
from rich.style import Style
from rich.theme import Theme

from local_operator.tui import theme as theme_mod

_C = theme_mod.semantic_color


class Role(Enum):
    """Which of the budget's three spending questions (§1.2) a hue answers.

    ``NEUTRAL`` is the ramp everything else lives on. ``GROUND`` is not a
    spending category at all — it marks a binding that paints a SURFACE
    (a background token) rather than ink on one, kept separate so the
    budget conformance check (§2.5 check 3, §4.2) can filter grounds out
    before counting hues.
    """

    OUTCOME = "outcome"
    REFERENCE = "reference"
    LIVENESS = "liveness"
    NEUTRAL = "neutral"
    GROUND = "ground"


class Surface(Enum):
    """Where an element can appear on screen.

    The unit the distinctness gate (§4.2) scopes its pairwise checks to: a
    picker row and a code comment are never on screen together, so demanding
    they be perceptually distinct would be the over-strict gate that fails
    every theme for a collision nobody can see (§2.2).
    """

    TRANSCRIPT = "transcript"
    TOOL_ROW = "tool_row"
    PICKER = "picker"
    STATUS = "status"
    PANEL = "panel"


@dataclass(frozen=True)
class Binding:
    """One thing the user can see, and the ink it is painted in."""

    #: Stable id, e.g. ``"tool.status.success_glyph"`` or ``"markdown.h1"``.
    element: str
    #: A :data:`theme.SEMANTIC_TOKENS` member.
    token: str
    #: The token of the surface this element renders ON. Documentation for
    #: the future contrast/distinctness gates, not consumed by :func:`style`
    #: — the widget's own background (a TCSS rule or a sibling GROUND
    #: binding) is what actually paints the ground.
    ground: str
    role: Role
    surface: Surface
    bold: bool = False
    #: The design rationale, moved here VERBATIM from its old call site.
    note: str = ""


# ---------------------------------------------------------------------------
# Markdown prose (`markdown_theme.brand_markdown_theme`, §2.1 ~20 entries).
# Surface: TRANSCRIPT. Ground: `bg` — the island ground behind AssistantBlock
# and RichBlock text; there is no elevation step for prose (local_operator.tcss
# AssistantBlock/RichBlock carry no `background:` of their own).
# ---------------------------------------------------------------------------
_MARKDOWN_BINDINGS: tuple[Binding, ...] = (
    Binding("markdown.paragraph", "fg", "bg", Role.NEUTRAL, Surface.TRANSCRIPT),
    Binding("markdown.text", "fg", "bg", Role.NEUTRAL, Surface.TRANSCRIPT),
    Binding("markdown.em", "fg", "bg", Role.NEUTRAL, Surface.TRANSCRIPT),
    Binding("markdown.strong", "fg", "bg", Role.NEUTRAL, Surface.TRANSCRIPT, bold=True),
    Binding(
        "markdown.code",
        "signal",
        "bg",
        Role.REFERENCE,
        Surface.TRANSCRIPT,
        note=(
            "`signal`, not `string`: inline code in agent prose is overwhelmingly "
            "file paths and identifiers, which is the case theme.py reserves "
            "`signal` for. It also frees the greens to mean only "
            'added-or-succeeded rather than doubling as "code literal".'
        ),
    ),
    # `code_block` needs an explicit `bg` PAINT on top of its `fg` ink (the
    # original literal set both `color` and `bgcolor`) — one element, two
    # tokens. The table names one ink per element; `markdown_theme()` adds
    # the ground explicitly for this one binding via its own `ground` field.
    # See the module docstring in `markdown_theme.py` for why: a bare-ink
    # `Style` here would inherit whatever background rich composits under
    # it, which for a fenced block should always be the island `bg`.
    Binding("markdown.code_block", "fg", "bg", Role.NEUTRAL, Surface.TRANSCRIPT),
    Binding(
        "markdown.block_quote",
        "label",
        "bg",
        Role.REFERENCE,
        Surface.TRANSCRIPT,
        note=(
            "A quote is someone ELSE'S words inside the answer — a log line, "
            "a spec, a message being replied to. On `muted` it was the same "
            "ink as a settled tool name and h3, so the one thing a reader "
            "most needs to know about it (this is not the assistant talking) "
            "was carried only by the leading glyph. `label` marks it as meta "
            "and the italic does the rest; rich draws no quote gutter here, "
            "so ink and slant are the only channels available."
        ),
    ),
    Binding("markdown.list", "fg", "bg", Role.NEUTRAL, Surface.TRANSCRIPT),
    Binding(
        "markdown.item.bullet",
        "label",
        "bg",
        Role.REFERENCE,
        Surface.TRANSCRIPT,
        note=(
            "The MARKER, never the item's text. A list is the shape agent "
            "output takes most often — findings, steps, options — and on `dim` "
            "the markers were the same ink as the success glyph, the duration "
            "and the expand hint, so the one cue that says 'these are "
            "countable, parallel things' was the quietest mark on the row. "
            "`label` is the ramp's meta hue and already means 'structure "
            "about the content rather than the content', which is exactly "
            "what a bullet is. It is one glyph per line, so the hue costs "
            "almost no area — the item text stays `fg` and unchanged."
        ),
    ),
    Binding(
        "markdown.item.number",
        "label",
        "bg",
        Role.REFERENCE,
        Surface.TRANSCRIPT,
        note="The ordinal, same argument as `markdown.item.bullet`.",
    ),
    Binding(
        "markdown.hr",
        "dim",
        "bg",
        Role.NEUTRAL,
        Surface.TRANSCRIPT,
        note=(
            "`dim`, not `edge`. A horizontal rule is a drawn object the "
            "reader is meant to SEE; `edge` is the hairline-border token, and "
            "bound as ink it measures under 3:1 against `bg` — WCAG's floor "
            "for a non-text graphical object — in 54 of 54 themes, and under "
            "1.5:1 in 42 of them (min 1.10:1, rose-pine-moon). That is a "
            "binding defect, not a palette defect: no theme can fix it by "
            "choosing a better `edge`, because a visible `edge` would stop "
            "being a hairline. `dim` is the sheet's own separator ink and "
            "clears 3:1 in all 54 (min 3.61:1)."
        ),
    ),
    # h1 no longer spends `accent`. That token means "a turn is live" and is
    # enumerated as such in local_operator.tcss; h1 was the ONE binding
    # spending it for STRUCTURE rather than liveness, which made a document
    # title share ink with the running-tool icon and the shimmer crest.
    #
    # Measured, not preferred: an accent-BEARING ramp
    # (accent, label, signal, string, muted, dim) collides outright at min
    # dE76 0.00 on gruvbox-light, while the accent-free ramp below holds a
    # min dE76 of 5.25 across all 54 themes. Spending accent on prose made
    # the ramp measurably worse, so freeing it costs the ramp nothing and
    # returns the liveness token to meaning exactly one thing.
    Binding(
        "markdown.h1",
        "signal",
        "bg",
        Role.REFERENCE,
        Surface.TRANSCRIPT,
        bold=True,
        note=(
            "The document's title. `signal` is the ramp's reference hue and "
            "is already the loudest non-outcome ink in most palettes, which "
            "is what a title wants; h2 keeps `label` below it.\n\n"
            "Previously `accent`, which made a title the same ink as the "
            "live-tool indicator and the shimmer crest. See the comment "
            "above for why that was measurably worse, not merely "
            "off-message. `signal` also carries `markdown.code` and "
            "`markdown.link`, but those are inline runs inside a paragraph "
            "while this is a bolded line opening a document behind a `#` "
            "marker — the two are never confusable in position."
        ),
    ),
    Binding(
        "markdown.h2",
        "label",
        "bg",
        Role.REFERENCE,
        Surface.TRANSCRIPT,
        bold=True,
        note=(
            "`label`, not `fg`. h2 is the level assistants ACTUALLY emit — a "
            "reply is `## What I found` / `## What I changed` all the way "
            "down, and `#` is nearly never written — so this is the heading "
            "that carries structure in practice. On `fg` it was byte-identical "
            "to `markdown.strong` (same hex, same weight), which made a "
            "section header and a bolded phrase the same pixels: the reader "
            "had no way to tell where a section began. h1 already spends the "
            "one structural accent, and a second accent site would break the "
            "enumeration in local_operator.tcss, so h2 takes `label` — the "
            "ramp's meta hue, already distinct from prose, and the only token "
            "that separates the two without growing the accent budget."
        ),
    ),
    Binding(
        "markdown.h3",
        "warning",
        "bg",
        Role.NEUTRAL,
        Surface.TRANSCRIPT,
        bold=True,
        note=(
            "`warning` here is the third STRUCTURAL hue, not the outcome hue — "
            "the same argument `code.string`/`code.number` already make (see "
            "the module docstring): the token names an ink, and this use "
            "answers 'where am I in the document', not 'did this work'. It "
            "never appears on a row that also reports an outcome.\n\n"
            "Chosen by measurement, not taste. `string` looks like the "
            "natural third hue but is an ALIAS in most palettes — min dE76 "
            "0.0 against `success`, `warning` AND `signal` across 54 themes — "
            "so a ramp using it collided in 28 of them. `warning` is 44.5 "
            "from `signal` and 25.1 from `success`, the widest available "
            "separation, and it lifts the ramp's worst contrast from 3.61:1 "
            "to 4.26:1."
        ),
    ),
    # h4 carries the fourth hue UNBOLDED, which is how the ramp keeps
    # descending after hue runs out of ordering: h1-h4 are hue + weight,
    # h5/h6 fall back to the neutral ramp and separate by weight alone. The
    # `#` markers restored in `markdown_theme._flat_heading` state the level
    # outright, so hue is free to buy SCANNABILITY rather than rank — which
    # is the whole reason a hue-per-level ramp is safe here and was not
    # before.
    Binding("markdown.h4", "success", "bg", Role.NEUTRAL, Surface.TRANSCRIPT, bold=True),
    # h5 keeps a hue; h6 is the first level to fall off the colour ramp onto
    # the neutral one. NOT `dim` for either: `dim` measures below 4.5:1 on
    # `bg` in 30 of 54 themes, so the old tail shipped sub-AA heading text in
    # more than half the palettes. `muted` clears 4.84:1 in all 54, which is
    # what holds the ramp's floor at 4.26:1.
    #
    # WEIGHT is bold for h1-h5 and plain for h6 alone. That is deliberately
    # almost no signal: it makes a heading read as a heading (bold against
    # body prose) rather than encoding its RANK, and hands rank entirely to
    # hue plus the `#` markers when they are enabled. h6 drops the bold as
    # the one typographic full stop at the bottom of the ramp.
    #
    # Weight must still never REVERSE going down the ramp. An earlier draft
    # ran bold, plain, bold, plain through the tail — h4 unbolded while h5
    # was bold again — so weight claimed h5 outranked h4 while position said
    # the opposite. `test_heading_weight_descends_monotonically` pins that.
    #
    # With five levels sharing a weight, hue is doing the separating alone:
    # measured min pairwise dE76 among h1-h5 is 21.6 median across the 54
    # themes, with one theme (`light`, 5.5) under 10.
    Binding("markdown.h5", "accent", "bg", Role.NEUTRAL, Surface.TRANSCRIPT, bold=True),
    Binding("markdown.h6", "muted", "bg", Role.NEUTRAL, Surface.TRANSCRIPT),
    Binding(
        "markdown.heading_marker",
        "dim",
        "bg",
        Role.NEUTRAL,
        Surface.TRANSCRIPT,
        note=(
            "The heading's OWN level, stated rather than inferred. rich strips "
            "the `#` markers when it parses a heading and renders only the "
            "text, which forced all six levels onto the colour channel alone "
            "— the root cause of h3/h4 colliding, and of every proposal to "
            "fix it wanting a hue it should not have to spend. Restoring the "
            "marker returns the level to the channel markdown itself uses, "
            "where the count of `#` is unambiguous at every level, in all 54 "
            "themes, and cannot be inverted by a saturated palette.\n\n"
            "`dim` because the marker is structural metadata about the "
            "content rather than the content — the same argument as "
            "`markdown.item.bullet`, one rung quieter since the heading text "
            "beside it already carries hue and weight. It is the sheet's own "
            "separator ink and clears the 3:1 non-text floor in all 54 "
            "(min 3.61:1), the same floor `markdown.hr` is held to."
        ),
    ),
    Binding("markdown.link", "signal", "bg", Role.REFERENCE, Surface.TRANSCRIPT),
    Binding("markdown.link_url", "signal", "bg", Role.REFERENCE, Surface.TRANSCRIPT),
)

# ---------------------------------------------------------------------------
# Code fence syntax ramp (`IslandSyntaxTheme`, §2.1 14 entries). Surface:
# TRANSCRIPT. Ground: `bg`, same as the markdown prose it sits inside.
# ---------------------------------------------------------------------------
_CODE_BINDINGS: tuple[Binding, ...] = (
    Binding("code.token", "fg", "bg", Role.NEUTRAL, Surface.TRANSCRIPT),
    Binding("code.comment", "dim", "bg", Role.NEUTRAL, Surface.TRANSCRIPT),
    Binding(
        "code.keyword",
        "label",
        "bg",
        Role.NEUTRAL,
        Surface.TRANSCRIPT,
        bold=True,
        note=(
            "Still NOT `accent` — syntax highlighting must not spend the "
            "running-indicator budget on every `def` in the transcript. But "
            "`muted` was the opposite error: keyword, function, class and "
            "builtin ALL resolved to it, so a fence was one grey and the "
            "reader parsed structure from position alone. `label` is the "
            "meta hue, distinct from `signal` (functions) at min dE76 11.0 "
            "across 54 themes."
        ),
    ),
    # `warning` (amber) here is the code-literal hue, not the outcome hue —
    # see the module docstring for why this is NEUTRAL despite the token.
    Binding(
        "code.keyword_constant",
        "warning",
        "bg",
        Role.NEUTRAL,
        Surface.TRANSCRIPT,
        note=(
            "`warning` (amber), NOT `string`. The `string` token resolves to the "
            "same hex as `success`, which is the diff-added green — so a fence "
            'containing a literal put `"ok"` and a write row\'s `+12` in one '
            "viewport in one colour, and that pairing (a code block beside a "
            "tool row) is the single most common shape of agent output. Amber "
            "already carries Number and Keyword.Constant, so this groups every "
            "LITERAL under one hue and leaves the green meaning only "
            '"added or succeeded".'
        ),
    ),
    Binding("code.name", "fg", "bg", Role.NEUTRAL, Surface.TRANSCRIPT),
    Binding("code.name_function", "signal", "bg", Role.NEUTRAL, Surface.TRANSCRIPT),
    Binding("code.name_class", "warning", "bg", Role.NEUTRAL, Surface.TRANSCRIPT),
    Binding("code.name_builtin", "label", "bg", Role.NEUTRAL, Surface.TRANSCRIPT),
    Binding("code.string", "warning", "bg", Role.NEUTRAL, Surface.TRANSCRIPT),
    Binding("code.number", "warning", "bg", Role.NEUTRAL, Surface.TRANSCRIPT),
    Binding("code.operator", "label", "bg", Role.NEUTRAL, Surface.TRANSCRIPT),
    Binding("code.punctuation", "dim", "bg", Role.NEUTRAL, Surface.TRANSCRIPT),
    Binding("code.error", "danger", "bg", Role.NEUTRAL, Surface.TRANSCRIPT),
    Binding("code.generic", "fg", "bg", Role.NEUTRAL, Surface.TRANSCRIPT),
    # The fence's own background (`IslandSyntaxTheme.get_background_style`
    # and the `Syntax(..., background_color=...)` argument in
    # `IslandCodeBlock`). A GROUND binding renders ON itself: it establishes
    # the surface rather than sitting on one.
    Binding(
        "code.background",
        "raised",
        "raised",
        Role.GROUND,
        Surface.TRANSCRIPT,
        note=(
            "The fence gets a real SLAB. On `bg` a code block was the same "
            "paper as the prose around it, so a fence had no edges and a "
            "reader had to infer where code started from the syntax colours "
            "alone. `raised` is the elevation token the transcript never "
            "spent: measured across 54 themes it is visibly distinct from "
            "`bg` in 53 (the exception is `paper`, whose ramp is flat by "
            "design) and still holds `fg` at min 6.09:1, so the slab costs "
            "no legibility.\n\n"
            "This is the D2 slab decision REVISITED, not reversed: the "
            "original objection was to rich's default Monokai block, whose "
            "ground was unrelated to the theme and was the loudest chrome in "
            "the transcript. A ground drawn from the theme's own elevation "
            "ramp is one step of depth, not a competing surface."
        ),
    ),
)

#: `IslandSyntaxTheme`'s pygments token -> element name. Kept here, next to
#: the bindings it names, rather than in `markdown_theme.py`: this table IS
#: the "one place" the element vocabulary is declared.
_SYNTAX_TOKEN_ELEMENTS: tuple[tuple[_TokenType, str], ...] = (
    (Token, "code.token"),
    (Comment, "code.comment"),
    (Keyword, "code.keyword"),
    (Keyword.Constant, "code.keyword_constant"),
    (Name, "code.name"),
    (Name.Function, "code.name_function"),
    (Name.Class, "code.name_class"),
    (Name.Builtin, "code.name_builtin"),
    (String, "code.string"),
    (Number, "code.number"),
    (Operator, "code.operator"),
    (Punctuation, "code.punctuation"),
    (Error, "code.error"),
    (Generic, "code.generic"),
)

# ---------------------------------------------------------------------------
# Tool row (`widgets/tool_card.py`, §2.1 ~28 call sites). Surface: TOOL_ROW.
# Ground varies with the card's state: `surface` once settled, `raised`
# while running, `tint-danger` when failed (local_operator.tcss ToolCard
# rules). A handful of elements (args/output body ink) render across more
# than one of those grounds depending on state; each is given the ground it
# predominantly appears against and the ambiguity is called out in the
# handoff rather than modelled — the table has no per-state ground axis.
# ---------------------------------------------------------------------------
_TOOL_ROW_BINDINGS: tuple[Binding, ...] = (
    # -- _append_live_body: the running card's own block ------------------
    Binding("tool.live.dim", "dim", "raised", Role.NEUTRAL, Surface.TOOL_ROW),
    Binding(
        "tool.live.header",
        "accent",
        "raised",
        Role.LIVENESS,
        Surface.TOOL_ROW,
        note=(
            "Accent, the same ink the running icon spends: this row is the "
            'card\'s answer to "is it alive", and it has to survive a still frame.'
        ),
    ),
    # -- _append_input_body: arguments -------------------------------------
    Binding("tool.args.dim", "dim", "surface", Role.NEUTRAL, Surface.TOOL_ROW),
    Binding("tool.args.label", "label", "surface", Role.REFERENCE, Surface.TOOL_ROW),
    Binding("tool.args.value", "fg", "surface", Role.NEUTRAL, Surface.TOOL_ROW),
    # -- _append_output_body: plain result expansion -----------------------
    Binding("tool.output.dim", "dim", "surface", Role.NEUTRAL, Surface.TOOL_ROW),
    Binding("tool.output.error", "danger", "tint-danger", Role.OUTCOME, Surface.TOOL_ROW),
    # -- _append_search_body ------------------------------------------------
    Binding("tool.search.dim", "dim", "surface", Role.NEUTRAL, Surface.TOOL_ROW),
    Binding("tool.search.title", "fg", "surface", Role.NEUTRAL, Surface.TOOL_ROW, bold=True),
    Binding("tool.search.url", "signal", "surface", Role.REFERENCE, Surface.TOOL_ROW),
    Binding(
        "tool.search.snippet",
        "muted",
        "surface",
        Role.NEUTRAL,
        Surface.TOOL_ROW,
        note=(
            "Snippets and the actionable footer must meet normal-text "
            "contrast; `dim` is reserved for structural metadata."
        ),
    ),
    # -- _append_fetch_body ---------------------------------------------------
    Binding("tool.fetch.dim", "dim", "surface", Role.NEUTRAL, Surface.TOOL_ROW),
    Binding(
        "tool.fetch.signal",
        "signal",
        "surface",
        Role.REFERENCE,
        Surface.TOOL_ROW,
        note=(
            "D3: the final URL is the card's anchor, so it rides `signal` "
            "blue like web_search's URLs instead of receding into the dim "
            'metadata. The "Fetched:" label and the trailing `· status · '
            "ctype · cache` metadata stay dim; only the URL(s) lift.\n\n"
            "Reused for the `sparse/JS-gated` advisory row: the one advisory "
            "row earns attention: it is the signal to reach for browser, "
            "not structural chrome."
        ),
    ),
    Binding("tool.fetch.snippet", "muted", "surface", Role.NEUTRAL, Surface.TOOL_ROW),
    Binding(
        "tool.fetch.error",
        "danger",
        "surface",
        Role.OUTCOME,
        Surface.TOOL_ROW,
        bold=True,
        note=(
            "F1: the non-2xx error row rides `danger`, bold — the strongest "
            "treatment on the card, so a block/error page cannot be mistaken "
            "for successful content. Matches the tool result's is_error flag."
        ),
    ),
    # -- _append_diff_body ----------------------------------------------------
    Binding("tool.diff.added", "success", "surface", Role.OUTCOME, Surface.TOOL_ROW),
    Binding("tool.diff.removed", "danger", "surface", Role.OUTCOME, Surface.TOOL_ROW),
    Binding("tool.diff.hunk", "muted", "surface", Role.NEUTRAL, Surface.TOOL_ROW),
    Binding("tool.diff.context", "dim", "surface", Role.NEUTRAL, Surface.TOOL_ROW),
    # -- _build_row: the one-line summary ------------------------------------
    Binding("tool.row.dim", "dim", "surface", Role.NEUTRAL, Surface.TOOL_ROW),
    Binding(
        "tool.row.name_running",
        "string",
        "raised",
        Role.NEUTRAL,
        Surface.TOOL_ROW,
        note=(
            "Two-step fade on settle: the live row keeps the string green on "
            "the name and readable `muted` body text; a settled row drops both "
            "one step so the running row is the brightest thing on screen. "
            "Composing counts as live: the model is actively producing this "
            "call, and a row that dimmed while its arguments streamed would "
            "read as a finished action rather than as the one thing "
            "currently happening."
        ),
    ),
    Binding(
        "tool.row.name_settled",
        "muted",
        "surface",
        Role.NEUTRAL,
        Surface.TOOL_ROW,
        note=(
            "The fallback for a tool with no category — an MCP call, or a "
            "builtin this table has not classified. Categorised tools use "
            "`tool.row.name_<category>` below; this stays the neutral it has "
            "always been so an unknown tool is quiet rather than mis-filed."
        ),
    ),
    # A settled row's name, by what the tool DID. Before this the whole ledger
    # was one grey, so the only rows carrying any hue at all were the failed
    # one and the running one — you could see that something broke, but not
    # scan for what touched your files.
    #
    # Liveness still wins: `_build_row` only reaches these once a row has
    # settled, so the running row keeps `tool.row.name_running` and there is
    # exactly one mechanism per meaning. Identity is a property of a finished
    # row; "what is happening now" outranks "what kind of thing it was".
    Binding(
        "tool.row.name_read",
        "tool-read",
        "surface",
        Role.REFERENCE,
        Surface.TOOL_ROW,
        note=(
            "Looked at the world, changed nothing: read, glob, grep, "
            "web_fetch, web_search, browser, the variable readers. Derives "
            "from `signal`, whose contract is already 'links, file paths' — "
            "reading is reference-shaped, so the default costs no new hue."
        ),
    ),
    Binding(
        "tool.row.name_mutate",
        "tool-mutate",
        "surface",
        Role.NEUTRAL,
        Surface.TOOL_ROW,
        note=(
            "Changed a file: write, edit. Derives from `muted` — deliberately "
            "NOT `warning`. A row that edited a file is not a warning, and "
            "borrowing the alarm hue would both lie about the row and hand "
            "control of tool names to whoever tunes `warning`. A theme that "
            "wants edits to stand out authors `tool-mutate`."
        ),
    ),
    Binding(
        "tool.row.name_exec",
        "tool-exec",
        "surface",
        Role.NEUTRAL,
        Surface.TOOL_ROW,
        note=(
            "Ran something: bash, eval. Derives from `muted` for the same "
            "reason as mutate — the neutral ramp, not an outcome hue."
        ),
    ),
    Binding(
        "tool.row.name_meta",
        "tool-meta",
        "surface",
        Role.REFERENCE,
        Surface.TOOL_ROW,
        note=(
            "Coordination rather than work on the machine: task, agent, hub, "
            "todo, send, wake, ask. Derives from `label`, whose contract is "
            "already 'violet meta: tips, skill labels' — the closest existing "
            "match, so again no new hue by default."
        ),
    ),
    Binding("tool.row.summary_running", "muted", "raised", Role.NEUTRAL, Surface.TOOL_ROW),
    Binding("tool.row.summary_settled", "dim", "surface", Role.NEUTRAL, Surface.TOOL_ROW),
    Binding(
        "tool.row.chip_running",
        "dim",
        "raised",
        Role.NEUTRAL,
        Surface.TOOL_ROW,
        note=(
            "Chrome quieter than the fact while live (the summary is "
            "`muted` there); one step BRIGHTER than the fact once settled, "
            "when the summary drops to `dim` — the attribution is the thing "
            "a reader scrolling back is hunting for, and a chip that faded "
            "to the same ink as the command stopped separating the two "
            "(design round 1, D1). It survives every shed rung because the "
            "budget above already paid for it."
        ),
    ),
    Binding("tool.row.chip_settled", "muted", "surface", Role.NEUTRAL, Surface.TOOL_ROW),
    Binding(
        "tool.row.icon_running",
        "accent",
        "raised",
        Role.LIVENESS,
        Surface.TOOL_ROW,
        note=(
            "The icon carries the running state: accent while live (D26 — a "
            'still frame must read "live" without the shimmer), dim once '
            "settled. This is one of the five places in the app the accent "
            "green is spent."
        ),
    ),
    Binding("tool.row.icon_settled", "dim", "surface", Role.NEUTRAL, Surface.TOOL_ROW),
    Binding(
        "tool.row.slot_offer",
        "dim",
        "surface",
        Role.NEUTRAL,
        Surface.TOOL_ROW,
        note=(
            "Generic tool output stays quiet until the row is targeted. "
            "Search sources are the primary result, not diagnostics, so "
            "their disclosure remains visible at rest and in colorless "
            "terminals."
        ),
    ),
    Binding(
        "tool.row.slot_notice",
        "muted",
        "surface",
        Role.NEUTRAL,
        Surface.TOOL_ROW,
        note=(
            "`muted`, one step brighter than the offer's `dim`: this is not "
            "something the eye may skip. Walk the ladder — full phrase, then "
            "the three-cell glyph — and take the first rung the row can hold "
            "with nothing reserved for the summary. Only when even ⟨∅⟩ will "
            "not fit does the answer go unsaid, and by then the row is down "
            "to its icon and its outcome anyway."
        ),
    ),
    # -- _status_runs: the running-row duration ------------------------------
    Binding("tool.status.running_duration", "dim", "raised", Role.NEUTRAL, Surface.TOOL_ROW),
    # -- _diff_runs: the status column's +N/-N pill --------------------------
    Binding("tool.status.diff_added", "success", "surface", Role.OUTCOME, Surface.TOOL_ROW),
    Binding("tool.status.diff_removed", "danger", "surface", Role.OUTCOME, Surface.TOOL_ROW),
    # -- _outcome_runs: the settled glyph + duration -------------------------
    Binding("tool.status.duration", "dim", "surface", Role.NEUTRAL, Surface.TOOL_ROW),
    Binding(
        "tool.status.success_glyph",
        "success",
        "surface",
        Role.OUTCOME,
        Surface.TOOL_ROW,
        note=(
            "The GLYPH carries `success`; the duration stays `dim`. This "
            'narrows D12 ("success is quiet — check + duration both dim"), '
            "and the half it keeps is the important half: a ledger where "
            "every completed row shouts is a ledger nobody scans.\n\n"
            "What it overturns is quiet implemented as BRIGHTNESS, which is "
            "the axis the whole transcript already competes on. Measured on a "
            "rendered frame, the outcome (`✓`), the cost (`0.4s`) and the "
            "affordance (`⟨expand⟩`) came out in ONE hex on the same row — "
            "three different kinds of fact in one ink, so an operator hunting "
            "twenty rows for the one that failed was scanning for a shape and "
            "the colour channel carried nothing. Hue at low salience was "
            "available and unspent: `success` is defined by every theme and "
            "ships at >=4.0:1 on `surface` in all 54.\n\n"
            "The row stays quiet in the sense that matters — the glyph is ONE "
            "cell, the duration and command stay neutral, the row's ground is "
            "unchanged. Same trade theme.py:50-55 already made for "
            "`tint-select`, where pure luminance steps measured 1.096:1 and "
            "were found imperceptible: D12 reached for the axis that token "
            "had already been shown to have no room on.\n\n"
            "The duration is a COST, not an outcome, so it keeps `dim` — as "
            "does the expand hint, which is an affordance. `NoticeKind.success` "
            "stays `fg` (transcript.py): that rationale is about three greens "
            "on one surface, and this puts the green on the tool row, which is "
            "exactly where the notice was told not to put a second one."
        ),
    ),
    # `interrupted` is deliberately NOT re-tinted to an outcome hue: the
    # docstring's contract is that ✓/✗/⊘ separate by SHAPE alone in a
    # colourless frame, and interrupted stays on the neutral ramp so that
    # contract is not quietly walked back by a future palette author reaching
    # for a fourth outcome colour.
    Binding("tool.status.interrupted", "dim", "surface", Role.NEUTRAL, Surface.TOOL_ROW),
    Binding("tool.status.error_glyph", "danger", "tint-danger", Role.OUTCOME, Surface.TOOL_ROW),
)

BINDINGS: tuple[Binding, ...] = _MARKDOWN_BINDINGS + _CODE_BINDINGS + _TOOL_ROW_BINDINGS

#: Derived index. Built once at import time — `BINDINGS` is a module-level
#: constant tuple, never mutated after definition.
BY_ELEMENT: Mapping[str, Binding] = {binding.element: binding for binding in BINDINGS}

assert len(BY_ELEMENT) == len(BINDINGS), "duplicate element id in BINDINGS"


def style(element: str) -> Style:
    """Resolve one binding to a rich ``Style`` against the current theme.

    ``Role.GROUND`` bindings paint a background (they ARE a surface, not
    ink on one); everything else paint foreground colour, plus ``bold=True``
    when the binding asks for it. ``bold`` is passed only when ``True`` and
    left unset (``None``) otherwise, matching every pre-refactor call site —
    none of them passed ``bold=False`` explicitly. Rich's ``Style`` treats
    ``bold=None`` (inherit whatever composites underneath, e.g. `markdown.code`
    nested inside `markdown.strong`) and ``bold=False`` (force it off)
    differently, so passing ``False`` here would be a real behaviour change,
    not a no-op.
    """
    binding = BY_ELEMENT[element]
    if binding.role is Role.GROUND:
        return Style(bgcolor=_C(binding.token))
    if binding.bold:
        return Style(color=_C(binding.token), bold=True)
    return Style(color=_C(binding.token))


def markdown_theme() -> Theme:
    """Element-style theme for rich ``Markdown``, replacing the inline dict
    that used to live at ``markdown_theme.py:125-157``.

    Three elements need more than :func:`style` gives them: ``markdown.em``
    and ``markdown.block_quote`` are italic, and ``markdown.link_url`` is
    underlined. Neither is a colour
    decision the budget tracks, so they are composed on top here rather than
    adding non-colour fields to :class:`Binding`. ``markdown.code_block``
    similarly needs its `ground` painted explicitly (see the binding's note)
    on top of its ink.
    """
    styles = {element: style(element) for element in _element_names(_MARKDOWN_BINDINGS)}
    styles["markdown.em"] = styles["markdown.em"] + Style(italic=True)
    styles["markdown.block_quote"] = styles["markdown.block_quote"] + Style(italic=True)
    styles["markdown.link_url"] = styles["markdown.link_url"] + Style(underline=True)
    code_block = BY_ELEMENT["markdown.code_block"]
    styles["markdown.code_block"] = styles["markdown.code_block"] + Style(
        bgcolor=_C(code_block.ground)
    )
    return Theme(styles, inherit=True)


def syntax_styles() -> dict[_TokenType, Style]:
    """Pygments token -> ``Style``, replacing ``IslandSyntaxTheme.__init__``'s
    inline dict."""
    return {token_type: style(element) for token_type, element in _SYNTAX_TOKEN_ELEMENTS}


def ground_hex(element: str) -> str:
    """Resolve a ``Role.GROUND`` binding straight to a hex string.

    The fourth seam, alongside :func:`style`, :func:`markdown_theme` and
    :func:`syntax_styles` (§2.5 check 2's AST gate enumerates all four as
    sanctioned). It exists because two rich APIs
    (``SyntaxTheme.get_background_style`` and ``Syntax(background_color=...)``
    — see ``markdown_theme.IslandSyntaxTheme``/``IslandCodeBlock``) want a raw
    hex, not a ``Style``, and pulling the hex back out of
    ``style(element).bgcolor`` would couple that call site to rich's
    ``Color`` internals for no reason. Raises ``KeyError`` for an unknown
    element and ``ValueError`` for a non-``GROUND`` one — a ground reader
    reaching for ink would be its own kind of bug.
    """
    binding = BY_ELEMENT[element]
    if binding.role is not Role.GROUND:
        raise ValueError(f"{element!r} is not a Role.GROUND binding")
    return _C(binding.token)


def _element_names(bindings: tuple[Binding, ...]) -> tuple[str, ...]:
    return tuple(binding.element for binding in bindings)
