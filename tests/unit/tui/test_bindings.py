"""The binding table's own contract (§2.5 of the colour-budget proposal).

Four things asserted here:

- **coverage** (check 1) — every element the transcript and tool row actually
  paint resolves through :data:`bindings.BY_ELEMENT`. This is what stops the
  table drifting from reality: a rebinding that only touches the call site
  (or a call site that only touches a literal) would otherwise go unnoticed.
- **totality** — every binding resolves to a real :data:`theme.SEMANTIC_TOKENS`
  member and a real ground, for every registered theme, not just the two
  brand ramps.
- **no bypass** (check 2) — no migrated module calls ``theme.semantic_color``
  directly outside :mod:`bindings`'s own sanctioned seams (:func:`style`,
  :func:`markdown_theme`, :func:`syntax_styles`, :func:`ground_hex`). Coverage
  alone cannot catch this: a widget that reaches around the table and calls
  ``semantic_color`` straight from a fresh call site never references an
  element id at all, so nothing in check 1 sees it. This is the AST
  allow-list proposal §2.5 check 2 specifies.
- **equivalence with the pre-refactor call sites** — :func:`bindings.style`
  and the two adapters reproduce the exact rich ``Style`` objects the old
  inline ``Style(color=semantic_color(...))`` calls produced. This is the
  slice's hard requirement (§6 Slice 2: "behaviour must be byte-identical to
  post-slice-1") turned into a running assertion rather than a one-time
  manual diff.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator

import pytest
from rich.style import Style

from local_operator.tui import bindings, theme
from local_operator.tui.bindings import BINDINGS, BY_ELEMENT, Binding, Role, Surface


@pytest.fixture(autouse=True)
def _restore_theme() -> Iterator[None]:
    """`theme.current_theme()` is a module singleton (see `test_theme.py`);
    this file switches it repeatedly and must not leak a choice into whatever
    test runs next in the same process."""
    original = theme.current_theme()
    yield
    theme.set_theme(original)


# ---------------------------------------------------------------------------
# Totality: every binding is well-formed against the live theme registry.
# ---------------------------------------------------------------------------


def test_bindings_table_has_no_duplicate_elements() -> None:
    """One row per element — a duplicate would make `BY_ELEMENT` non-total
    in the direction that matters (silently keeping the second definition)."""
    elements = [binding.element for binding in BINDINGS]
    assert len(elements) == len(set(elements))


@pytest.mark.parametrize("theme_name", theme.available_themes())
def test_markdown_heading_levels_are_mutually_distinct(theme_name: str) -> None:
    """h1-h6 must resolve to six different (ink, weight) pairs in EVERY theme.

    Two collisions have shipped here, and neither was visible to the existing
    gates. h3 and h4 were both `muted` AND both bold — byte-identical in all
    54 themes by construction, a defect in the BINDING table that no palette
    could fix. `radient` then bound `label` to the same hex as `muted`, so its
    h2 and h3 collided too — a defect in the PALETTE that no binding could
    fix. The per-token contrast and distinctness checks miss both, because
    each tests tokens in isolation and a heading level is the PAIRING of a
    token with a weight.

    Parametrized over every registered theme on purpose: run on a three-theme
    sample this assertion passes while `radient` ships two identical heading
    levels.
    """
    theme.set_theme(theme_name)

    def _level(element: str) -> tuple[str | None, bool]:
        style = bindings.style(element)
        triplet = style.color.get_truecolor() if style.color else None
        return (triplet.hex if triplet else None, bool(style.bold))

    levels = [_level(f"markdown.h{n}") for n in range(1, 7)]
    duplicates = {level for level in levels if levels.count(level) > 1}
    assert not duplicates, (
        f"{theme_name}: heading levels are not mutually distinct — "
        f"{sorted(map(str, duplicates))} is used more than once in {levels}"
    )


@pytest.mark.parametrize("theme_name", theme.available_themes())
def test_markdown_heading_marker_is_dim(theme_name: str) -> None:
    """The `#` markers rich strips at parse time, restored by
    ``markdown_theme._flat_heading`` as structural metadata about the content
    rather than content — the same argument as `markdown.item.bullet`, one
    rung quieter because the heading text beside it carries hue and weight."""
    theme.set_theme(theme_name)
    assert bindings.style("markdown.heading_marker") == Style(color=theme.semantic_color("dim"))


@pytest.mark.parametrize("theme_name", theme.available_themes())
def test_every_binding_resolves_to_a_real_token(theme_name: str) -> None:
    """Every ``token`` and ``ground`` is a live semantic token, for EVERY
    registered theme (not just the two brand ramps `theme.py`'s own totality
    check already covers) — a curated palette missing a token this table
    reaches for would otherwise only fail once a user actually selected it."""
    for binding in BINDINGS:
        # Raises KeyError (an assertion failure, via pytest.fail below) for
        # an unknown token — exercised directly rather than through
        # `pytest.raises` so every bad binding is named, not just the first.
        try:
            theme.semantic_color(binding.token, theme_name)
            theme.semantic_color(binding.ground, theme_name)
        except KeyError as exc:
            pytest.fail(f"{binding.element}: {exc}")


def test_every_binding_declares_a_role_and_surface() -> None:
    """`role`/`surface` are what makes the budget (§1.2) and the future
    surface-scoped distinctness gate (§4.2) machine-checkable; a binding
    with neither would silently opt out of both."""
    for binding in BINDINGS:
        assert isinstance(binding.role, Role), binding.element
        assert isinstance(binding.surface, Surface), binding.element


def test_ground_bindings_carry_no_note_free_pass() -> None:
    """Sanity check on the table's own shape: a `Binding` is frozen, so this
    just confirms construction produced the dataclass this module expects
    (guards against a future edit accidentally swapping in a plain tuple)."""
    for binding in BINDINGS:
        assert isinstance(binding, Binding)


def test_ground_hex_matches_style_for_every_ground_binding() -> None:
    """`ground_hex()` (the fourth seam markdown_theme.py now calls instead of
    reaching for `theme.semantic_color` directly) must agree with `style()`
    for every `Role.GROUND` row, in every registered theme — the two are
    reading the same binding through two different rich-facing shapes
    (a raw hex vs. a `Style(bgcolor=...)`), and a divergence between them
    would mean one of the two seams drifted from `BY_ELEMENT`."""
    ground_elements = [b.element for b in BINDINGS if b.role is Role.GROUND]
    assert ground_elements, "expected at least one Role.GROUND binding"
    for theme_name in ("dark", "light", "rose-pine-dawn"):
        theme.set_theme(theme_name)
        for element in ground_elements:
            bgcolor = bindings.style(element).bgcolor
            # rich's Style.bgcolor is Color | None; a GROUND binding always sets
            # it, so this narrows the type for pyright rather than guarding a
            # real runtime possibility.
            assert bgcolor is not None
            assert bindings.ground_hex(element) == bgcolor.name


def test_ground_hex_rejects_a_non_ground_element() -> None:
    """`ground_hex` is for GROUND rows only — an ink binding reaching for it
    would be a call-site bug (the wrong seam for what it wants), not a
    legitimate use, so it raises rather than silently returning ink as if it
    were a surface."""
    with pytest.raises(ValueError):
        bindings.ground_hex("markdown.h1")


# ---------------------------------------------------------------------------
# Coverage: every element the transcript / tool row paint is IN the table.
# `bindings.markdown_theme()` / `syntax_styles()` are exercised directly —
# they are exhaustive over `_MARKDOWN_BINDINGS` / `_CODE_BINDINGS` by
# construction (§2.5 check 1's real content, since those two adapters ARE
# the exhaustive readers). The tool-row half is checked by asserting the
# ids actually referenced from `widgets/tool_card.py`'s source are present —
# see `test_tool_card_source_only_uses_declared_elements` below, which is
# what makes the gate demonstrably able to FAIL (§6 Slice 2 verification).
# ---------------------------------------------------------------------------


def test_markdown_theme_covers_every_rich_markdown_element() -> None:
    """`bindings.markdown_theme()` names every element the old inline dict
    at `markdown_theme.py:125-157` did — the coverage claim for prose."""
    expected = {
        "markdown.paragraph",
        "markdown.text",
        "markdown.em",
        "markdown.strong",
        "markdown.code",
        "markdown.code_block",
        "markdown.block_quote",
        "markdown.list",
        "markdown.item.bullet",
        "markdown.item.number",
        "markdown.hr",
        "markdown.h1",
        "markdown.h2",
        "markdown.h3",
        "markdown.h4",
        "markdown.h5",
        "markdown.h6",
        "markdown.link",
        "markdown.link_url",
    }
    got = set(bindings.markdown_theme().styles)
    assert expected <= got, expected - got


def test_syntax_styles_covers_every_pygments_token_the_ramp_named() -> None:
    """`bindings.syntax_styles()` names every pygments token
    `IslandSyntaxTheme.__init__` used to."""
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
    )

    expected = {
        Token,
        Comment,
        Keyword,
        Keyword.Constant,
        Name,
        Name.Function,
        Name.Class,
        Name.Builtin,
        String,
        Number,
        Operator,
        Punctuation,
        Error,
        Generic,
    }
    got = set(bindings.syntax_styles())
    assert expected <= got, expected - got


#: Every element id actually referenced in `widgets/tool_card.py`, gathered
#: by source inspection rather than hand transcription — the whole point of
#: a coverage test is that it reads the real call sites, not a list an
#: editor could let drift from them.
#:
#: Collects every string constant shaped like a binding id (``tool.*``),
#: not only literals passed straight to `bindings.style(...)`: several call
#: sites route through a variable chosen by an if/else first (``slot_element
#: = "tool.row.slot_offer"`` ... later ``bindings.style(slot_element)``), and
#: narrowing to direct-call literals would silently miss exactly those. The
#: shape filter (rather than intersecting with `BY_ELEMENT` up front) is
#: what keeps this able to catch a REMOVED binding — intersecting first
#: would make a missing element invisible by construction.
def _tool_card_referenced_elements() -> set[str]:
    import ast
    import inspect

    from local_operator.tui.widgets import tool_card

    source = inspect.getsource(tool_card)
    tree = ast.parse(source)
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value.startswith("tool.")
    }


def test_tool_card_source_only_uses_declared_elements() -> None:
    """Every literal element id `tool_card.py` passes to `bindings.style`
    names a real row in the table.

    This is the check that could not be written before this module existed
    (§2.1: "there is no object to enumerate elements from"), and it is the
    coverage gate proper: a call site referencing a removed or misspelled
    element fails HERE instead of at render time inside whichever widget
    happens to ask first.
    """
    referenced = _tool_card_referenced_elements()
    assert referenced, "expected to find bindings.style(...) calls in tool_card.py"
    missing = referenced - set(BY_ELEMENT)
    assert not missing, f"tool_card.py references undeclared elements: {missing}"


def test_removing_a_referenced_binding_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate actually FAILS on a deliberately introduced violation (§6
    Slice 2: "the coverage test must fail if a binding is removed —
    demonstrate that, don't assert it").

    Simulates the removal by shrinking `BY_ELEMENT` to omit one element the
    source references, then re-running the same check the test above makes
    inline (duplicated rather than refactored into a shared helper, so this
    test still fails even if a future edit changes how the real test reads
    `BY_ELEMENT`).
    """
    referenced = _tool_card_referenced_elements()
    victim = "tool.status.success_glyph"
    assert victim in referenced  # the removal has to be observable
    thinned = {element: binding for element, binding in BY_ELEMENT.items() if element != victim}
    monkeypatch.setattr(bindings, "BY_ELEMENT", thinned)
    missing = referenced - set(bindings.BY_ELEMENT)
    assert missing == {victim}


# ---------------------------------------------------------------------------
# No bypass (§2.5 check 2): a migrated module may not call
# `theme.semantic_color` directly. Coverage (above) catches a call site that
# references a REMOVED element; it cannot catch one that never references an
# element at all — a fresh `Style(color=theme_mod.semantic_color("dim"))`
# dropped into a migrated module reads a real token and renders a real
# style, so nothing about it looks broken, and no `BY_ELEMENT` lookup is
# anywhere near it. That is the gap §2.5 check 2's AST allow-list exists to
# close: it does not ask "does this resolve", it asks "does this call site
# exist at all outside the table's own seams".
# ---------------------------------------------------------------------------

#: Modules migrated onto the table (§2.5 check 2's ratchet — "a file joins it
#: when it is migrated, and can never leave"). `tool_card.py` and
#: `markdown_theme.py` are the two this slice covers.
#:
#: `transcript.py` is a KNOWN, DELIBERATE exception, not a failure: it is not
#: yet migrated (its ``UserBlock``/``NoticeBlock``/notice-kind call sites are
#: exactly the ones `local_operator.tcss`'s own §2.3-option-(b) split leaves
#: outside this table's scope for now), so it is not on this list. Adding it
#: without migrating its call sites first would fail this gate for a module
#: nobody has touched — the ratchet only tightens once, on purpose.
_MIGRATED_MODULES: tuple[str, ...] = (
    "local_operator.tui.markdown_theme",
    "local_operator.tui.widgets.tool_card",
)


def _semantic_color_call_sites(source: str) -> set[tuple[str, int]]:
    """Every direct ``semantic_color(...)`` call in ``source``, as
    ``(qualname, lineno)`` pairs — empty for a module that only reaches the
    ramp through :mod:`bindings`.

    Two aliasing shapes appear in this codebase and both are followed:
    ``theme_mod.semantic_color(...)`` (an attribute call through
    ``from local_operator.tui import theme as theme_mod``) and a bare-name
    alias — either ``_C = theme_mod.semantic_color`` (bindings.py's own
    pattern, before this slice) or ``from local_operator.tui.theme import
    semantic_color``. :func:`bindings.style`/:func:`bindings.markdown_theme`/
    :func:`bindings.syntax_styles`/:func:`bindings.ground_hex` are the
    sanctioned seams and live in ``bindings.py`` itself, which this check
    never inspects — a migrated module is expected to call THOSE, not
    ``semantic_color``.
    """
    tree = ast.parse(source)

    module_aliases: set[str] = set()
    name_aliases: set[str] = {"semantic_color"}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "local_operator.tui":
            for alias in node.names:
                if alias.name == "theme":
                    module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "local_operator.tui.theme":
                    module_aliases.add(alias.asname or "theme")
        elif isinstance(node, ast.ImportFrom) and node.module == "local_operator.tui.theme":
            for alias in node.names:
                if alias.name == "semantic_color":
                    name_aliases.add(alias.asname or alias.name)

    # Second pass: `_C = theme_mod.semantic_color` binds a bare name to the
    # attribute alias found above — needs the module aliases resolved first,
    # and the assignment may appear before or after the import in source.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "semantic_color"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id in module_aliases
        ):
            name_aliases.add(node.targets[0].id)

    stack: list[str] = []
    sites: set[tuple[str, int]] = set()

    class _Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        visit_FunctionDef = _visit_function
        visit_AsyncFunctionDef = _visit_function

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            is_hit = (isinstance(func, ast.Name) and func.id in name_aliases) or (
                isinstance(func, ast.Attribute)
                and func.attr == "semantic_color"
                and isinstance(func.value, ast.Name)
                and func.value.id in module_aliases
            )
            if is_hit:
                qualname = ".".join(stack) if stack else "<module>"
                sites.add((qualname, node.lineno))
            self.generic_visit(node)

    _Visitor().visit(tree)
    return sites


def _module_source_path(dotted_name: str):
    import importlib

    module = importlib.import_module(dotted_name)
    assert module.__file__ is not None, dotted_name
    from pathlib import Path

    return Path(module.__file__)


@pytest.mark.parametrize("module_name", _MIGRATED_MODULES)
def test_migrated_module_never_calls_semantic_color_directly(module_name: str) -> None:
    """§2.5 check 2, the real check: NOT "was a binding removed" (coverage,
    above) but "did a call site go around the table entirely". A migrated
    module must reach the ramp only through :mod:`bindings`'s own seams.

    See :func:`test_the_bypass_gate_actually_fails` for proof this is not a
    check that can never fail — it deliberately introduces the exact
    violation this test polices, watches it fail, then restores the file
    byte-identically.
    """
    source = _module_source_path(module_name).read_text()
    sites = _semantic_color_call_sites(source)
    assert not sites, f"{module_name} calls semantic_color directly outside bindings.py: {sites}"


def test_the_bypass_gate_actually_fails() -> None:
    """The gate is not decoration: prove it catches a REAL bypass by
    introducing one and watching the same check the parametrized test above
    runs report it.

    §6: "a gate not proven to fail is not a gate." This is that proof for
    check 2, the same way `test_removing_a_referenced_binding_is_caught` is
    that proof for check 1 — and it follows the same shape: read the real
    source, mutate the STRING, assert on the mutated string, never write
    back to disk. `tests/unit` runs under pytest-xdist (`-n auto` in
    `pyproject.toml`); several workers import `markdown_theme.py`
    concurrently, so writing a bypass to the real file (even briefly, even
    restored in a `finally`) would race a sibling worker's import against
    the mutated bytes — this was tried and reproduced exactly that failure.
    The disk-level version of this demonstration — edit the file, run the
    suite, `git diff` to confirm, restore, run green — is the one-time
    manual proof recorded in the slice's own handoff, not a standing test.
    """
    original = _module_source_path("local_operator.tui.markdown_theme").read_text()
    assert not _semantic_color_call_sites(original), "fixture module must start clean"

    bypass = original.replace(
        "from local_operator.tui import bindings\n",
        "from local_operator.tui import bindings\n"
        "from local_operator.tui import theme as theme_mod\n",
        1,
    ).replace(
        "    def get_background_style(self) -> Style:  # type: ignore[override]\n"
        '        return Style(bgcolor=bindings.ground_hex("code.background"))\n',
        "    def get_background_style(self) -> Style:  # type: ignore[override]\n"
        '        return Style(bgcolor=theme_mod.semantic_color("bg"))  # BYPASS\n',
        1,
    )
    assert bypass != original, "fixture edit did not match the file's current text"

    caught = _semantic_color_call_sites(bypass)
    assert caught, "the introduced bypass was not detected — gate is not a gate"
    assert any(line == "IslandSyntaxTheme.get_background_style" for line, _ in caught)


# ---------------------------------------------------------------------------
# Equivalence: bindings.style() reproduces the exact old inline Style().
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("theme_name", ["dark", "light", "rose-pine-dawn"])
class TestStyleMatchesPreRefactorLiterals:
    """One assertion per pre-refactor call site (§2.1's ~28 tool_card.py
    entries plus the two markdown/syntax dicts), spelled as the literal the
    old code used to construct — so a change to `bindings.style()` that
    silently altered a resolution would fail a *specific*, named case
    instead of a generic round-trip check."""

    def test_tool_status_success_glyph(self, theme_name: str) -> None:
        theme.set_theme(theme_name)
        assert bindings.style("tool.status.success_glyph") == Style(
            color=theme.semantic_color("success")
        )

    def test_tool_status_error_glyph(self, theme_name: str) -> None:
        theme.set_theme(theme_name)
        assert bindings.style("tool.status.error_glyph") == Style(
            color=theme.semantic_color("danger")
        )

    def test_tool_diff_added_and_removed(self, theme_name: str) -> None:
        theme.set_theme(theme_name)
        assert bindings.style("tool.diff.added") == Style(color=theme.semantic_color("success"))
        assert bindings.style("tool.diff.removed") == Style(color=theme.semantic_color("danger"))

    def test_tool_row_icon_running_is_accent(self, theme_name: str) -> None:
        theme.set_theme(theme_name)
        assert bindings.style("tool.row.icon_running") == Style(
            color=theme.semantic_color("accent")
        )

    def test_tool_row_name_running_is_string(self, theme_name: str) -> None:
        theme.set_theme(theme_name)
        assert bindings.style("tool.row.name_running") == Style(
            color=theme.semantic_color("string")
        )

    def test_markdown_h1_is_bold_signal_not_accent(self, theme_name: str) -> None:
        """h1 moved OFF `accent`.

        `accent` means "a turn is live" and is enumerated as such in
        local_operator.tcss; h1 was the one binding spending it for structure,
        which made a document title share ink with the running-tool icon.
        Measured, an accent-bearing heading ramp collides outright (min dE76
        0.00, gruvbox-light) while the accent-free ramp holds 5.25 across 54
        themes — so freeing it cost the ramp nothing.
        """
        theme.set_theme(theme_name)
        assert bindings.style("markdown.h1") == Style(
            color=theme.semantic_color("signal"), bold=True
        )
        assert (
            bindings.style("markdown.h1").color != Style(color=theme.semantic_color("accent")).color
        )

    def test_markdown_hr_is_dim_not_edge(self, theme_name: str) -> None:
        theme.set_theme(theme_name)
        assert bindings.style("markdown.hr") == Style(color=theme.semantic_color("dim"))

    def test_markdown_code_is_signal(self, theme_name: str) -> None:
        theme.set_theme(theme_name)
        assert bindings.style("markdown.code") == Style(color=theme.semantic_color("signal"))


@pytest.mark.parametrize("theme_name", theme.available_themes())
def test_code_fence_sits_on_its_own_slab(theme_name: str) -> None:
    """The fence ground is `raised`, and actually differs from the prose ground.

    On `bg` a code block was the same paper as the text around it, so a fence
    had no edges at all. `raised` is one of the elevation tokens the binding
    table never spent; it is a real step in every registered palette, and the
    fence is the surface that most needs one.
    """
    theme.set_theme(theme_name)
    ground = bindings.ground_hex("code.background")
    assert ground == theme.semantic_color("raised")
    assert ground != theme.semantic_color("bg"), (
        f"{theme_name}: the fence slab is the same ink as the prose ground, "
        "so a code block has no edges"
    )


@pytest.mark.parametrize("theme_name", theme.available_themes())
def test_the_syntax_ramp_is_not_one_grey(theme_name: str) -> None:
    """Keyword, function, class and builtin must not all be the same ink.

    They were all `muted`, so a fence carried no structure at all and the
    reader parsed it from position alone — the single largest block of grey
    in the transcript.
    """
    theme.set_theme(theme_name)
    inks = {
        element: bindings.style(element).color
        for element in (
            "code.keyword",
            "code.name_function",
            "code.name_class",
            "code.string",
        )
    }
    assert len(set(inks.values())) >= 3, (
        f"{theme_name}: the syntax ramp collapsed to {len(set(inks.values()))} "
        f"distinct inks — {inks}"
    )
