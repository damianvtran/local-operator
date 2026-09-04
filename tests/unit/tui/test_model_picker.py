"""Model picker: ranking, columns, and the states a catalogue actually reaches.

The interesting cases here are not "does it list things". They are the ones where a
naive list misleads the user: a model that looks free because its price is missing,
a family led by its oldest member, a locked provider that silently vanishes, and a
list that is longer than the window without saying so.
"""

from __future__ import annotations

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult

from local_operator.tui.widgets.command_picker import slash_argument
from local_operator.tui.widgets.model_picker import (
    MAX_VISIBLE_ROWS,
    ModelPicker,
    ModelRow,
    _is_parenthesised_tail,
    format_price_pair,
    format_window,
    rank_rows,
)

MODEL_COMMANDS = ("model", "models")


def _rows() -> list[ModelRow]:
    return [
        ModelRow("anthropic", "claude-opus-5", "Claude Opus 5", 500_000, 15.0, 75.0, True),
        ModelRow("anthropic", "claude-opus-4-1", "Claude Opus 4.1", 200_000, 15.0, 75.0, True),
        ModelRow("anthropic", "claude-sonnet-4-20250514", "Sonnet 4", 200_000, 3.0, 15.0, True),
        ModelRow("openai", "gpt-5.4", "GPT-5.4", 400_000, 1.25, 10.0, True),
        ModelRow("openai", "gpt-4o", "GPT-4o", 128_000, 2.5, 10.0, True),
        ModelRow("openrouter", "deepseek/deepseek-v4-flash", "", 1_000_000, 0.09, 0.4, True),
        ModelRow("kimi", "kimi-k2-thinking", "Kimi K2", 262_144, 0.6, 2.5, False),
        ModelRow("ollama", "qwen3:8b", "", 0, 0.0, 0.0, True),
    ]


# -- formatting --------------------------------------------------------------


def test_window_is_abbreviated_and_unknown_is_blank() -> None:
    """Blank, not a placeholder: the column is right-aligned, so an empty cell
    simply reads as nothing to say while a `—` would draw the eye to the one row
    that knows least."""
    assert format_window(500_000) == "500k"
    assert format_window(1_000_000) == "1m"
    assert format_window(1_500_000) == "1.5m"
    assert format_window(900) == "900"
    assert format_window(0) == ""
    assert format_window(-1) == ""


def test_a_name_that_only_recases_the_id_is_not_printed_beside_it() -> None:
    """The parenthetical exists to add what the selector did not say.

    ChatGPT's Codex catalogue titles its display names off the slug
    (``gpt-5.6-luna`` -> ``GPT-5.6-Luna``), so an exact comparison let every row
    in that family spend ~16 cells restating its own id. A name that genuinely
    differs keeps its parenthetical: that is the half a case-insensitive test
    could break, so both are pinned here.
    """
    picker = ModelPicker(lambda row: None)
    picker.set_rows(
        [
            ModelRow("openai", "gpt-5.6-luna", "GPT-5.6-Luna", 272_000, 1.25, 10.0, True),
            ModelRow("openai", "gpt-4.1-mini", "GPT-4.1 mini", 128_000, 0.4, 1.6, True),
        ],
        current=None,
    )
    picker.open("")
    painted = picker.render_text(100).plain

    assert "(GPT-5.6-Luna)" not in painted, painted
    assert "(GPT-4.1 mini)" in painted, painted


def test_a_large_price_keeps_its_decimal() -> None:
    """Rounding is not free in a price column: `$18.75` rendered as `$19` reads as
    a real quoted rate the provider does not charge."""
    assert format_price_pair(15.0, 18.75) == "$15/18.8"
    assert format_price_pair(15.0, 75.0) == "$15/75"


def test_only_a_genuine_pair_of_zeroes_reads_as_free() -> None:
    """The error this prevents is the one a user would act on: a provider that
    quotes no pricing is NOT free, and rendering an absent price as zero would
    advertise a paid model as costing nothing."""
    assert format_price_pair(3.0, 15.0) == "$3/15"
    assert format_price_pair(0.09, 0.4) == "$0.09/0.4"
    assert format_price_pair(0.0, 0.0) == "free"
    assert format_price_pair(-1.0, -1.0) == ""
    # One unknown leg is still unknown: a row that could not price half of
    # itself has not been quoted free, and must not print the sentinel either.
    assert format_price_pair(0.0, -1.0) == ""
    assert format_price_pair(-1.0, 0.0) == ""


def test_a_meta_route_reads_as_usage_based_rather_than_free_or_blank() -> None:
    """The fourth price state, and the two wrong labels it replaces.

    A router's cost depends on the model it dispatches to, so neither existing
    answer is true of it: ``free`` is a promise the user would act on, and a
    blank cell is the same thing this column says about a model nobody priced.
    Both were reachable for ``radient/auto`` depending only on which shape its
    listing happened to quote.
    """
    assert format_price_pair(-1.0, -1.0, routed=True) == "usage-based"
    # The flag is checked FIRST, and that is the point rather than an ordering
    # detail: a stale listing quoting a symmetric zero for a router (Radient's
    # does today) must not be able to out-vote the endpoint's own nature.
    assert format_price_pair(0.0, 0.0, routed=True) == "usage-based"
    # Unset, every pre-existing answer is unchanged.
    assert format_price_pair(0.0, 0.0) == "free"
    assert format_price_pair(-1.0, -1.0) == ""
    assert format_price_pair(3.0, 15.0) == "$3/15"


def test_the_routed_label_does_not_overflow_the_price_column() -> None:
    """`usage-based` is materially longer than `free`, and this column is
    width-constrained — so the width it costs is a property worth pinning.

    At the picker's narrowest painted width the row must still fit exactly,
    with the id truncated rather than the numbers run spilling past the edge.
    """
    row = ModelRow(
        provider="radient",
        model_id="auto",
        label="Automatic",
        context_window=1_048_576,
        input_price=-1.0,
        output_price=-1.0,
        aggregated=True,
        routed=True,
    )
    picker = ModelPicker(lambda chosen: None)
    picker.set_rows([row], current="radient/auto", status="")

    for width in (56, 60, 73, 100):
        painted = picker._row(0, width).plain
        assert cell_len(painted) == width, (width, painted)
        assert "usage-based" in painted, (width, painted)
        assert "1m" in painted, (width, painted)


def test_a_sub_cent_price_keeps_three_significant_figures() -> None:
    """The `$18.75 -> $19` argument does not stop at one cent.

    A flat two decimals under-quoted `zai/glm-5.3-flash` by 6.7% (`0.075` ->
    `$0.07`) on exactly the cheap models a user picks BECAUSE of the price, and
    resolved the same half-cent in opposite directions on two rows of one list:
    `gpt-5.2:batch` (0.875) rounded up while `gpt-5.1:batch` (0.625) rounded
    down, an artefact of binary floats rather than of anything either vendor
    quoted.
    """
    assert format_price_pair(0.075, 0.25) == "$0.075/0.25"
    assert format_price_pair(0.875, 1.75) == "$0.875/1.75"
    assert format_price_pair(0.625, 1.25) == "$0.625/1.25"
    assert format_price_pair(0.125, 0.5) == "$0.125/0.5"
    # Three SIGNIFICANT figures, not three decimals: the bound is relative, so
    # it says the same thing about a $5 model and a $0.05 one.
    assert format_price_pair(0.04815, 0.19305) == "$0.0481/0.193"
    # Unchanged where it was already right — no trailing-zero noise, and the
    # one-decimal rule above ten still governs.
    assert format_price_pair(3.0, 15.0) == "$3/15"
    assert format_price_pair(0.6, 2.5) == "$0.6/2.5"
    assert format_price_pair(15.0, 18.75) == "$15/18.8"


def test_a_name_that_is_already_a_parenthetical_is_not_wrapped_twice() -> None:
    """`GLM-5.2 (Token Plan)` rendered as `(GLM-5.2 (Token Plan))`; the stray `))`
    reads as a rendering fault rather than as a qualifier.

    Only a TRAILING parenthetical is unwrapped, so brackets can never be dropped
    from a name that needs them to read as an annotation at all.
    """
    assert _is_parenthesised_tail("GLM-5.2 (Token Plan)")
    assert _is_parenthesised_tail("Claude Opus 4.5 (2025-11-01)")
    assert not _is_parenthesised_tail("GPT-4.1 mini")
    # The group closes before the end, so the name continues past it.
    assert not _is_parenthesised_tail("(preview) thing")
    # Unbalanced: a naive `endswith(")")` would strip a bracket that is needed.
    assert not _is_parenthesised_tail("a) b")

    picker = ModelPicker(lambda row: None)
    picker.set_rows(
        [
            ModelRow(
                "alibaba-token-plan", "glm-5.2", "GLM-5.2 (Token Plan)", 1_000_000, -1.0, -1.0, True
            )
        ],
        current="",
    )
    picker.open("glm")
    painted = picker.render_text(100).plain
    assert "GLM-5.2 (Token Plan)" in painted, painted
    assert "))" not in painted, painted


# -- ranking -----------------------------------------------------------------


def test_a_substring_match_beats_a_merely_fuzzy_one() -> None:
    """`opus` is a SUBSEQUENCE of `anthropic/claude-sonnet-4` — o and p from
    "anthropic", u from "claude", s from "sonnet". Ranking subsequence hits
    alongside substring hits therefore answered a query naming one model with a
    list containing several unrelated ones."""
    names = [row.model_id for row in rank_rows(_rows(), "opus")]
    assert names == ["claude-opus-5", "claude-opus-4-1"]
    assert all("opus" in name for name in names)


def test_the_fuzzy_fallback_still_resolves_typos_and_elisions() -> None:
    """Which is the whole reason the fallback exists — the substring tier alone
    would answer these with nothing."""
    assert rank_rows(_rows(), "anthopus")[0].model_id == "claude-opus-5"
    assert rank_rows(_rows(), "sonnet4")[0].model_id == "claude-sonnet-4-20250514"


def test_a_provider_prefix_scopes_the_list() -> None:
    """Matching the DISPLAYED `provider/id` is what makes this work without a
    separate syntax for it."""
    assert {row.provider for row in rank_rows(_rows(), "anthropic/")} == {"anthropic"}


def test_the_newest_version_leads_its_family() -> None:
    """Alphabetical order leads every family with its OLDEST member — `gpt-4o`
    before `gpt-5.4`, `claude-opus-4-1` before `claude-opus-5` — which is the one
    the user is least likely to want."""
    assert [row.model_id for row in rank_rows(_rows(), "gpt")] == ["gpt-5.4", "gpt-4o"]
    assert rank_rows(_rows(), "opus")[0].model_id == "claude-opus-5"


def test_a_dash_minor_version_outranks_the_bare_major() -> None:
    """`claude-opus-4-1` is 4.1, not two numbers. Read as `4` it tied with
    `claude-opus-4` and lost the tiebreak to it."""
    rows = [
        ModelRow("anthropic", "claude-opus-4", connected=True),
        ModelRow("anthropic", "claude-opus-4-1", connected=True),
    ]
    assert [row.model_id for row in rank_rows(rows, "opus")] == [
        "claude-opus-4-1",
        "claude-opus-4",
    ]


def test_a_datestamp_is_not_read_as_a_version_number() -> None:
    """`claude-sonnet-4-20260101` carries the standalone number 20260101, which as
    a version would outrank every real one in the catalogue — including a
    genuinely newer `claude-sonnet-4-6`. The version is the FIRST number, so the
    date lands on the tiebreak rung instead."""
    rows = [
        ModelRow("anthropic", "claude-sonnet-4-20260101", connected=True),
        ModelRow("anthropic", "claude-sonnet-4-6", connected=True),
    ]
    assert rank_rows(rows, "sonnet")[0].model_id == "claude-sonnet-4-6"


def test_a_serial_suffix_is_not_read_as_a_version_number() -> None:
    """`kimi-k2-0905` carries 2 and 905. Under a largest-number rule it scored 905
    and led a list in which `kimi-k3` came sixth — the version is the first number,
    not the biggest."""
    ids = ("kimi-k2-0905", "kimi-k2", "kimi-k2.7-code", "kimi-k3")
    rows = [ModelRow("kimi", model_id, connected=True) for model_id in ids]
    assert [row.model_id for row in rank_rows(rows, "kimi")] == [
        "kimi-k3",
        "kimi-k2.7-code",
        "kimi-k2-0905",
        "kimi-k2",
    ]


def test_a_version_glued_to_a_letter_still_counts() -> None:
    """`k2`, `qwen3` and `v4` attach the version directly to a letter. A pattern
    that required a preceding non-word character matched NOTHING in those ids, so
    their version came from whatever serial followed."""
    rows = [
        ModelRow("kimi", "kimi-k3", connected=True),
        ModelRow("kimi", "kimi-k2-0905", connected=True),
    ]
    assert rank_rows(rows, "kimi")[0].model_id == "kimi-k3"
    rows = [
        ModelRow("alibaba", "qwen2.5:14b", connected=True),
        ModelRow("alibaba", "qwen3:8b", connected=True),
    ]
    assert rank_rows(rows, "qwen")[0].model_id == "qwen3:8b"


def test_a_direct_provider_outranks_a_reseller_of_the_same_model() -> None:
    """`openrouter/anthropic/claude-opus-5` and `anthropic/claude-opus-5` are the
    same model. After logging in to Anthropic the direct route is what the user
    meant: one hop, the credential they just created, and provider-native
    behaviour like cache-control breakpoints."""
    rows = [
        ModelRow("openrouter", "anthropic/claude-opus-5", connected=True, aggregated=True),
        ModelRow("anthropic", "claude-opus-5", connected=True),
    ]
    assert rank_rows(rows, "opus")[0].selector == "anthropic/claude-opus-5"
    assert rank_rows(rows, "")[0].selector == "anthropic/claude-opus-5"


def test_two_snapshots_of_one_model_sort_newest_first() -> None:
    """The datestamp is the only thing that distinguishes them, so it gets its own
    descending rung rather than falling through to an alphabetical tiebreak."""
    rows = [
        ModelRow("anthropic", "claude-sonnet-4-20250514", connected=True),
        ModelRow("anthropic", "claude-sonnet-4-20260101", connected=True),
    ]
    assert [row.model_id for row in rank_rows(rows, "sonnet")] == [
        "claude-sonnet-4-20260101",
        "claude-sonnet-4-20250514",
    ]


def test_usable_models_outrank_ones_needing_a_login() -> None:
    """Interleaving them scatters the rows a user can act on through a list of
    rows they cannot."""
    rows = rank_rows(_rows(), "k")
    assert rows, "premise: something matches"
    connected = [row.connected for row in rows]
    assert connected == sorted(connected, reverse=True), [row.selector for row in rows]


def test_the_widget_hides_nothing_it_is_given() -> None:
    """Which rows are OFFERED is the app's call — it holds the credential state and
    filters unreachable models out before they get here (see `_catalogue_rows`).
    The widget renders what it is handed, so an unconnected row that survived that
    filter (the current model, or every row when the store could not be read) still
    appears, dimmed and marked, rather than vanishing."""
    assert any(row.provider == "kimi" for row in rank_rows(_rows(), ""))


# -- widget state ------------------------------------------------------------


def test_opening_with_no_query_highlights_the_current_model() -> None:
    """So the first frame answers "what am I on" too, and the first Enter is a
    no-op rather than an unrequested switch to whatever sorted first."""
    picker = ModelPicker(lambda row: None)
    picker.set_rows(_rows(), current="openai/gpt-4o")
    picker.open("")
    assert picker.highlighted_selector() == "openai/gpt-4o"


def test_opening_with_a_query_highlights_the_best_match() -> None:
    """The user has already narrowed; preselecting the current model there would
    fight the query."""
    picker = ModelPicker(lambda row: None)
    picker.set_rows(_rows(), current="openai/gpt-4o")
    picker.open("opus")
    assert picker.highlighted_selector() == "anthropic/claude-opus-5"


def test_a_late_arriving_catalogue_does_not_move_the_highlight() -> None:
    """Discovery is asynchronous, so rows land while the user is reading. Keying
    the held selection by INDEX would slide the highlight onto a different model
    whenever a row arrived above it."""
    picker = ModelPicker(lambda row: None)
    picker.set_rows(_rows())
    picker.open("opus")
    picker.move(+1)
    held = picker.highlighted_selector()
    assert held == "anthropic/claude-opus-4-1"

    extra = ModelRow("anthropic", "claude-opus-6", "Claude Opus 6", 900_000, 20.0, 90.0, True)
    picker.set_rows([extra, *_rows()])
    assert picker.highlighted_selector() == held


def test_changing_the_query_resets_the_highlight() -> None:
    """The candidate set is different, so the row under the cursor means a
    different model — keeping the index would act on something unrelated."""
    picker = ModelPicker(lambda row: None)
    picker.set_rows(_rows())
    picker.open("")
    picker.move(+2)
    picker.set_query("gpt")
    assert picker.selected_index == 0
    assert picker.highlighted_selector() == "openai/gpt-5.4"


def test_the_highlight_wraps_but_paging_clamps() -> None:
    """Wrapping arrows are what a short list wants; a PgDn that silently returned
    to the top of a 300-model catalogue would look like the list reset itself."""
    picker = ModelPicker(lambda row: None)
    picker.set_rows(_rows())
    picker.open("")
    total = len(picker.suggestions())
    picker.move(-1)
    assert picker.selected_index == total - 1
    picker.move(+1)
    assert picker.selected_index == 0
    picker.page(-1)
    assert picker.selected_index == 0
    picker.page(+1)
    assert picker.selected_index == total - 1


def test_choosing_hands_the_row_to_the_callback() -> None:
    """The widget never switches the model itself — only the app knows whether a
    row means a switch or a login."""
    chosen: list[ModelRow] = []
    picker = ModelPicker(chosen.append)
    picker.set_rows(_rows())
    picker.open("opus")
    picker.choose(0)
    assert [row.selector for row in chosen] == ["anthropic/claude-opus-5"]


def test_closing_empties_the_matches_so_the_editor_stops_routing_keys() -> None:
    """`is_open` is what the editor's key handler branches on, so a closed picker
    holding matches would swallow Up/Down from the text."""
    picker = ModelPicker(lambda row: None)
    picker.set_rows(_rows())
    picker.open("")
    assert picker.is_open()
    picker.close()
    assert not picker.is_open()
    assert picker.highlighted() is None


# -- rendering ---------------------------------------------------------------


@pytest.mark.parametrize("width", [40, 56, 76, 120, 200])
def test_no_rendered_row_ever_exceeds_the_width(width: int) -> None:
    """A row wider than its box wraps, and a wrapped row in a height-pinned widget
    is a clipped row — the density contract the whole TUI is built on."""
    picker = ModelPicker(lambda row: None)
    picker.set_rows(_rows(), current="openai/gpt-4o", status="cached: openai")
    picker.open("")
    for line in picker.render_text(width).plain.split("\n"):
        assert cell_len(line) <= width, repr(line)


def test_the_numbers_are_dropped_before_the_model_id_is() -> None:
    """The id is the part being chosen. Two columns of metadata in a narrow box
    would leave nothing for it."""
    picker = ModelPicker(lambda row: None)
    picker.set_rows(_rows())
    picker.open("gpt-5")
    narrow = picker.render_text(40).plain
    wide = picker.render_text(90).plain
    assert "gpt-5.4" in narrow
    assert "400k" not in narrow
    assert "400k" in wide


def test_the_current_model_is_marked() -> None:
    picker = ModelPicker(lambda row: None)
    picker.set_rows(_rows(), current="openai/gpt-4o")
    picker.open("gpt-4o")
    assert "●" in picker.render_text(90).plain


def test_a_locked_row_says_what_is_missing_instead_of_a_price() -> None:
    """A price column on a row that cannot run is answering the wrong question."""
    picker = ModelPicker(lambda row: None)
    picker.set_rows(_rows())
    picker.open("kimi")
    rendered = picker.render_text(90).plain
    assert "login required" in rendered


def test_an_overflowing_list_says_how_much_it_is_hiding() -> None:
    """Otherwise a windowed list is indistinguishable from a complete one, and a
    user concludes their model does not exist."""
    picker = ModelPicker(lambda row: None)
    many = [ModelRow("openai", f"gpt-test-{index}", connected=True) for index in range(60)]
    picker.set_rows(many)
    picker.open("")
    footer = picker.render_text(90).plain.split("\n")[-1]
    assert "of 60" in footer


def test_the_status_line_surfaces_what_the_catalogue_does_not_know() -> None:
    """A failed provider is exactly when a user hunting for a model released
    this morning needs telling, rather than concluding it is not real."""
    picker = ModelPicker(lambda row: None)
    picker.set_rows(_rows(), status="stale list: anthropic")
    picker.open("")
    assert "stale list: anthropic" in picker.render_text(90).plain


def test_a_footer_clause_that_cannot_keep_its_first_value_is_dropped_at_the_seam() -> None:
    """The three-clause composite (access note, stale, empty) at 100 columns
    used to render ``… · no live list:…`` — a label whose only payload died,
    leaving a dangling colon that read as a glitch. The cut moves to the seam
    before that clause. Lines that fitted before, and cuts that land inside a
    list of names, render exactly as they did."""
    note = "2 hidden — /login <provider>"
    picker = ModelPicker(lambda row: None)
    picker.set_rows(_rows(), status=f"{note} · stale list: anthropic · no live list: radient")
    picker.open("")
    # 73-cell picker at a 100-column terminal, as the captured frames were.
    footer = picker.render_text(73).plain.split("\n")[-1].strip()
    assert footer == f"{note} · stale list: anthropic", footer
    assert "no live list" not in footer
    # Wide enough → every clause, untouched.
    assert picker.render_text(90).plain.split("\n")[-1].strip() == (
        f"{note} · stale list: anthropic · no live list: radient"
    )
    # A cut inside the provider list keeps the label and first name (the
    # 60-column frame from design round 1), not the seam rule.
    picker.set_rows(_rows(), status=f"{note} · stale list: anthropic, xai")
    assert picker.render_text(56).plain.split("\n")[-1].strip() == (
        f"{note} · stale list: anthropic…"
    )
    # The leading clause is never dropped, however narrow.
    picker.set_rows(_rows(), status=note)
    assert picker.render_text(24).plain.split("\n")[-1].strip() == "2 hidden — /login <p…"


def test_the_footer_names_stale_providers_and_stays_silent_for_cached_ones() -> None:
    """`stale` is a failed refresh; `cached` is "listed within the picker's
    fifteen-minute TTL", which is not worth a line nobody would read."""
    from local_operator.tui.app import _catalogue_status

    assert _catalogue_status({"anthropic": "cached", "openrouter": "ok"}) == ""
    assert (
        _catalogue_status({"xai": "stale", "anthropic": "stale", "openrouter": "cached"})
        == "stale list: anthropic, xai"
    )
    assert (
        _catalogue_status({"anthropic": "stale", "radient": "empty"})
        == "stale list: anthropic · no live list: radient"
    )
    # Unauthenticated is counted by the access note, not here.
    assert _catalogue_status({"google": "unauthenticated"}) == ""


def test_the_stale_clause_fits_beside_the_access_note_at_the_reference_width() -> None:
    """The access note leads the footer and leaves 39 cells for this clause at
    100 columns (73-cell picker, 2 gutter, 1 margin, 28-cell note, 3-cell seam).
    The first spelling was 54 cells and truncation ate the half that said the
    rows were still usable."""
    from local_operator.tui.app import _catalogue_status

    clause = _catalogue_status({"anthropic": "stale", "xai": "stale", "openrouter": "ok"})
    assert cell_len(clause) <= 39, clause


def test_every_provider_stale_collapses_to_a_nameless_clause() -> None:
    """Offline fails every listed provider at once; five names fit no picker
    width the app renders, and naming only helps when the failure is partial."""
    from local_operator.tui.app import _catalogue_status

    everyone = {p: "stale" for p in ("anthropic", "google", "mistral", "openai", "xai")}
    assert _catalogue_status(everyone) == "stale list: all providers"
    # A provider that never produced a listing is not one of "all providers".
    assert _catalogue_status({**everyone, "ollama": "static", "zai": "unauthenticated"}) == (
        "stale list: all providers"
    )
    # One stale provider is always named: "all providers" would say less.
    assert _catalogue_status({"anthropic": "stale"}) == "stale list: anthropic"
    # Empty is a listing too, so stale + empty is partial and keeps the names.
    assert _catalogue_status({"anthropic": "stale", "xai": "stale", "radient": "empty"}) == (
        "stale list: anthropic, xai · no live list: radient"
    )


def test_an_empty_result_says_so_rather_than_rendering_nothing() -> None:
    picker = ModelPicker(lambda row: None)
    picker.set_rows(_rows())
    picker.open("zzzznope")
    assert "no matching models" in picker.render_text(90).plain


# -- the buffer parse that drives it -----------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("/model ", ""),
        ("/model opus", "opus"),
        ("/models gpt", "gpt"),
        ("/MODEL Opus", "Opus"),
        ("/model", None),  # the command WORD is still open — that is the other picker
        ("/mo", None),
        ("/help ", None),
        # INLINE: a `/model` typed after a message is a real argument now — the
        # caret defaults to the buffer end, which is inside the selector.
        ("hello /model x", "x"),
        # A newline after the argument leaves the caret on the MESSAGE line, so
        # the argument list is not live there.
        ("/model x\nmore", None),
    ],
)
def test_slash_argument_hands_over_on_the_terminating_space(text: str, expected) -> None:
    """The handover is what lets one buffer drive two lists without either widget
    knowing about the other: `slash_context` is live while the word is open, this
    takes over the instant a space terminates it.

    Inline and caret-aware now: an argument typed mid-draft counts, and the caret
    (defaulting to the buffer end) decides which line's command is live."""
    assert slash_argument(text, MODEL_COMMANDS) == expected


# -- mouse wheel -------------------------------------------------------------


class _Wheel:
    """The only thing the scroll handlers use from a Textual event."""

    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def test_the_wheel_moves_the_highlight_one_row_at_a_time() -> None:
    picker = ModelPicker(lambda row: None)
    picker.set_rows(_rows())
    picker.open("")
    first = picker.selected_index
    picker.on_mouse_scroll_down(_Wheel())
    assert picker.selected_index == first + 1
    picker.on_mouse_scroll_up(_Wheel())
    assert picker.selected_index == first


def test_the_wheel_clamps_where_the_arrows_wrap() -> None:
    """``move`` wraps, which suits a discrete arrow press. A wheel that
    teleported from the last model to the first reads as the catalogue
    resetting itself, so the scroll path clamps like paging does."""
    picker = ModelPicker(lambda row: None)
    picker.set_rows(_rows())
    picker.open("")
    for _ in range(len(_rows()) + 10):
        picker.on_mouse_scroll_down(_Wheel())
    assert picker.selected_index == len(picker.suggestions()) - 1
    for _ in range(len(_rows()) + 10):
        picker.on_mouse_scroll_up(_Wheel())
    assert picker.selected_index == 0
    # The arrow key still wraps — the wheel change must not have altered it.
    picker.move(-1)
    assert picker.selected_index == len(picker.suggestions()) - 1


def test_the_wheel_is_stopped_so_the_transcript_behind_stays_put() -> None:
    picker = ModelPicker(lambda row: None)
    picker.set_rows(_rows())
    picker.open("")
    down, up = _Wheel(), _Wheel()
    picker.on_mouse_scroll_down(down)
    picker.on_mouse_scroll_up(up)
    assert down.stopped and up.stopped


def test_the_wheel_on_an_empty_list_is_a_no_op() -> None:
    picker = ModelPicker(lambda row: None)
    picker.set_rows([])
    picker.open("")
    picker.on_mouse_scroll_down(_Wheel())  # must not raise
    assert picker.suggestions() == []


@pytest.mark.asyncio
async def test_large_degraded_catalogue_leads_with_current_family_and_status() -> None:
    """At 80x24 a thousand-row list must answer what is running and what the
    catalogue failed to load before the user scrolls."""
    current = ModelRow("anthropic", "claude-opus-5", connected=True)
    siblings = [
        ModelRow("anthropic", "claude-sonnet-4", connected=True),
        ModelRow("anthropic", "claude-haiku-4", connected=True),
    ]
    bulk = [
        ModelRow("openrouter", f"vendor/model-{index}", connected=True, aggregated=True)
        for index in range(1_000)
    ]
    picker = ModelPicker(lambda row: None)
    picker.set_rows(
        [*bulk, *siblings, current],
        current=current.selector,
        status="stale list: all providers — provider timeout",
    )

    class _Host(App[None]):
        CSS = "ModelPicker { width: 100%; }"

        def compose(self) -> ComposeResult:
            yield picker

    app = _Host()
    async with app.run_test(size=(80, 24)) as pilot:
        picker.styles.width = 80
        await pilot.pause()
        picker.open("")
        await pilot.pause()
        picker._repaint()
        await pilot.pause()
        await pilot.pause()
        first = picker.suggestions()[:3]
        painted = "\n".join(strip.text for strip in app.screen._compositor.render_strips())

    assert first[0].selector == current.selector
    assert all(row.provider == "anthropic" for row in first)
    assert "stale list: all providers" in painted, painted


def test_a_row_shows_the_display_name_beside_the_selector() -> None:
    """The band prints the display name and nothing else once a model is running.
    A picker that offered only the selector gave the user two names for one model
    with no way to connect them, so the row carries both: the selector first,
    because that is what ``/model`` takes and what ``rank_rows`` matches, and the
    name after it as a parenthetical.
    """
    picker = ModelPicker(lambda row: None)
    picker.set_rows(_rows(), current="openai/gpt-4o")
    picker.open("opus-5")
    row = picker.render_text(120).plain
    assert "anthropic/claude-opus-5" in row
    assert "(Claude Opus 5)" in row


def test_a_row_whose_name_says_nothing_new_does_not_repeat_itself() -> None:
    """A model with no display name — a local Ollama tag, a resold aggregator id —
    carries its selector as its label, and printing that twice on one row would
    read as two different columns saying the same thing."""
    picker = ModelPicker(lambda row: None)
    picker.set_rows(
        [ModelRow("ollama", "qwen3:8b", "ollama/qwen3:8b", 0, 0.0, 0.0, True)],
        current="",
    )
    picker.open("qwen")
    row = picker.render_text(120).plain
    assert row.count("qwen3:8b") == 1, row


def test_a_name_that_cannot_be_read_whole_is_not_painted_at_all() -> None:
    """Truncation keeps the HEAD, and the head of a model name is the vendor word
    every sibling row already shares: at 60 columns two anthropic rows both read
    ``Claude…`` while the part that tells them apart is exactly what was cut. A
    secondary aid that cannot be read should not spend cells."""
    long_name = "Claude Opus 4.5 (2025-11-01)"
    picker = ModelPicker(lambda row: None)
    picker.set_rows(
        [ModelRow("anthropic", "claude-opus-4-5-20251101", long_name, 200_000, 5.0, 25.0, True)],
        current="",
    )
    picker.open("opus")
    wide = picker.render_text(120).plain
    # Painted whole and WITHOUT a second pair of brackets: this name already
    # ends in its own parenthetical, so the wrapper would read `))`.
    assert long_name in wide, wide
    assert f"({long_name})" not in wide, wide
    narrow = picker.render_text(60).plain
    assert "Claude" not in narrow, narrow
    assert "claude-opus-4-5-20251101" in narrow, narrow


def test_the_name_never_grows_as_the_window_narrows() -> None:
    """Sized against the PAINTED layout, the annotation grew as the window shrank:
    crossing below ``_NUMBERS_MIN_WIDTH`` freed ~13 cells and handed all of them
    here, so at 56 columns the row read ``Cla…`` and at 55 the fuller
    ``Claude Opus 4.5…``. Content appearing as space disappears is the kind of
    thing a reader stops trusting a layout over, so the room is measured against a
    layout that always reserves the numbers run."""
    picker = ModelPicker(lambda row: None)
    picker.set_rows(_rows(), current="")
    picker.open("")
    widths = list(range(110, 44, -1))
    painted = []
    for width in widths:
        row = picker.render_text(width).plain.splitlines()[0]
        painted.append(len(row.partition("(")[2].partition(")")[0]))
    assert painted == sorted(painted, reverse=True), list(zip(widths, painted))


def test_the_status_row_is_kept_once_painted_so_the_card_does_not_reflow() -> None:
    """The app paints `checking providers…` on the keystroke and clears it when
    the live list lands. Letting the row collapse shrank the card by a line and
    dropped everything above it while the user was reading; a blank row is less
    motion than a reflow. A fresh open starts without the row again."""
    picker = ModelPicker(lambda row: None)
    picker.set_rows(_rows(), status="checking providers… · d in /model saves this")
    picker.open("")
    with_status = picker.render_text(90).plain.split("\n")
    picker.set_rows(_rows(), status="d in /model saves this")
    settled = picker.render_text(90).plain.split("\n")
    assert len(settled) == len(with_status), (with_status, settled)
    assert settled[-2].strip() == "", settled
    assert settled[-1].strip() == "d in /model saves this", settled
    # Never painted a status this open → no held row.
    picker.close()
    picker.set_rows(_rows(), status="d in /model saves this")
    picker.open("")
    fresh = picker.render_text(90).plain.split("\n")
    assert len(fresh) == len(settled) - 1, fresh


@pytest.mark.asyncio
@pytest.mark.parametrize("height", [12, 30, 45])
async def test_the_held_row_is_paid_for_out_of_the_row_budget_when_mounted(height: int) -> None:
    """The unmounted test above cannot reach `_row_budget`'s held-row term: with
    no screen it short-circuits to `MAX_VISIBLE_ROWS`. Mounted, the budget is
    a third of the screen, and the held blank row must come out of that third —
    otherwise the card grows a line on settle and pushes the composer, which is
    the reflow the hold exists to prevent. Pinned at three heights because the
    cap switches between the screen fraction and `MAX_VISIBLE_ROWS` at 45."""
    hint = "d in /model saves this"
    bulk = [
        ModelRow("openrouter", f"vendor/model-{index}", connected=True, aggregated=True)
        for index in range(60)
    ]
    picker = ModelPicker(lambda row: None)
    picker.set_rows(bulk, status=f"checking providers… · {hint}")

    class _Host(App[None]):
        CSS = "ModelPicker { width: 100%; }"

        def compose(self) -> ComposeResult:
            yield picker

    cap = min(MAX_VISIBLE_ROWS, height // 3)
    app = _Host()
    async with app.run_test(size=(100, height)) as pilot:
        await pilot.pause()
        picker.open("")
        await pilot.pause()
        checking = picker.render_text(100).plain.split("\n")
        checking_budget = picker._row_budget()
        heights: list[int] = []
        budgets: list[int] = []
        for status in (hint, "", "stale list: anthropic"):
            picker.set_rows(bulk, status=status)
            await pilot.pause()
            heights.append(len(picker.render_text(100).plain.split("\n")))
            budgets.append(picker._row_budget())

    # The card never changes height across the app's real settle sequence and
    # never exceeds the screen-fraction cap.
    assert heights == [len(checking)] * 3, (len(checking), heights)
    assert len(checking) <= cap, (len(checking), cap)
    # Two footer rows while the hint is present (`checking…` and hint-only
    # alike), and once the hint goes the held blank row — or the clause that
    # replaces it — still costs one: without the hold the empty-status state
    # would get its row back and the card would grow by a line.
    assert checking_budget == max(1, cap - 2), (checking_budget, cap)
    assert budgets == [cap - 2, cap - 1, cap - 1], (cap, budgets)
