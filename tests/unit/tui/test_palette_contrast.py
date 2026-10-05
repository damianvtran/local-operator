"""Contrast floors every registered theme must clear — the palette gate.

"Readable" is a checked property here, not a review comment. The floors are
calibrated from the BRAND ramps' own measured values (the dark ramp's ``dim``
is 4.55:1 on the ground, the light ramp's is 3.77:1; every state hue clears
4.3:1 on both), with enough headroom under each brand value that the brand
ramps themselves pass their own gate. A curated palette that fails here is
returned to its author with the exact token and ratio — the failure message
is the review.

The floors, and why each exists:

- ``fg`` >= 7:1 on ``bg`` and ``surface``: body prose is read for hours;
  both brand ramps sit near 14:1 and WCAG AAA is 7:1.
- ``muted`` >= 4.5:1: secondary text is still text (WCAG AA).
- ``dim`` >= 3.4:1: micro-labels and separators — the light brand ramp's
  3.77:1 is the reference, 3.4 leaves solve room without dipping to
  imperceptible.
- state hues (``accent success warning danger signal label string``)
  >= 4.0:1 on ``bg`` AND ``surface``: they carry meaning in one word
  (a red ✗, a green ✓) and both brand ramps clear 4.3:1 everywhere.
- ``faint`` BELOW ``dim``: it is the "inert hint" rung; a faint brighter
  than dim inverts the ramp's whole hierarchy.
- elevation is monotonic in the theme's polarity: each step of
  bg → surface → raised → overlay moves AWAY from the polarity's floor
  (lighter on dark themes, darker on light ones) and ``sunken`` moves the
  other way. A theme whose "raised" is darker than its ground paints
  depth upside down.
- tints stay near the ground (< 2.2:1 vs ``bg``): a tint is a cast, not a
  slab — the brand ramps' loudest tint is 1.6:1.
"""

from __future__ import annotations

import pytest

from local_operator.tui import theme


def _linear(channel: int) -> float:
    scaled = channel / 255
    return scaled / 12.92 if scaled <= 0.04045 else ((scaled + 0.055) / 1.055) ** 2.4


def _luminance(hex_color: str) -> float:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return 0.2126 * _linear(red) + 0.7152 * _linear(green) + 0.0722 * _linear(blue)


def contrast(color_a: str, color_b: str) -> float:
    lum_a, lum_b = _luminance(color_a), _luminance(color_b)
    high, low = max(lum_a, lum_b), min(lum_a, lum_b)
    return (high + 0.05) / (low + 0.05)


#: Foreground floors: token -> minimum ratio, checked against BOTH ``bg``
#: and ``surface`` (text renders on both grounds).
_FG_FLOORS: dict[str, float] = {
    "fg": 7.0,
    "muted": 4.5,
    "dim": 3.4,
    "accent": 4.0,
    "success": 4.0,
    "warning": 4.0,
    "danger": 4.0,
    "signal": 4.0,
    "label": 4.0,
    "string": 4.0,
}

#: A tint is a cast at roughly the ground's luminance, never a slab.
_TINT_CEILING = 2.2

_ALL_THEMES = theme.available_themes()


@pytest.mark.parametrize("name", _ALL_THEMES)
def test_theme_is_total(name: str) -> None:
    """Every required token is present; nothing outside the known vocabulary is.

    ``theme.OPTIONAL_TOKENS`` (the tool-category family) sits outside the
    required set but is always resolved by ``register_theme`` — a theme that
    does not author one still ends up with a filled value, so ``spec.tokens``
    is exactly ``SEMANTIC_TOKENS | OPTIONAL_TOKENS`` for every registered
    theme, never a partial set.
    """
    spec = theme.theme_spec(name)
    assert set(theme.SEMANTIC_TOKENS) <= set(spec.tokens)
    assert set(spec.tokens) == set(theme.SEMANTIC_TOKENS) | set(theme.OPTIONAL_TOKENS)
    assert spec.label, f"{name} has no picker label"
    assert spec.description, f"{name} has no picker description"


@pytest.mark.parametrize("name", _ALL_THEMES)
def test_foreground_contrast_floors(name: str) -> None:
    spec = theme.theme_spec(name)
    tokens = spec.tokens
    failures: list[str] = []
    for token, floor in _FG_FLOORS.items():
        for ground in ("bg", "surface"):
            ratio = contrast(tokens[token], tokens[ground])
            if ratio < floor:
                failures.append(
                    f"{token} {tokens[token]} on {ground} {tokens[ground]}: "
                    f"{ratio:.2f} < {floor}"
                )
    assert not failures, f"{name}: " + "; ".join(failures)


@pytest.mark.parametrize("name", _ALL_THEMES)
def test_danger_reads_on_its_own_tint(name: str) -> None:
    """The failed tool row pairs ``danger`` ink WITH the ``tint-danger`` band.

    Review round 1 (D1) caught the gap: the state floors above check ``bg``
    and ``surface``, but the one place danger ink always renders is the failed
    row's own tinted ground (``tool_card.py`` paints the error reason in
    ``danger`` on ``tint-danger``), and two palettes cleared every other floor
    while dipping to 3.6–3.9:1 on exactly that pairing. Same 4.0 floor as the
    other state checks.
    """
    tokens = theme.theme_spec(name).tokens
    ratio = contrast(tokens["danger"], tokens["tint-danger"])
    assert ratio >= 4.0, (
        f"{name}: danger {tokens['danger']} on tint-danger {tokens['tint-danger']}: "
        f"{ratio:.2f} < 4.0 — the failed row's error text is illegible on its own band"
    )


def test_the_ask_panel_reads_on_its_own_ground(name: str | None = None) -> None:
    """The ask drawer's state inks pair with ITS ground, not with `bg`/`surface`.

    Review round 2 (D1) measured the gap this closes: the state floors above
    check `bg` and `surface`, but the ask panel paints on `overlay`, and the
    brand LIGHT ramp's accent lands at 3.49:1 and its `success` at 3.45:1 there
    — under the 4.0 state floor — while every check here passed. The remedy is
    the repo's own: `theme._fill_chip_live` derives `chip-live`,
    `chip-success` and `chip-warning` (the hue when it clears this ground, the
    ramp's neutral ink when it does not), which is exactly the family the panel
    now paints with.

    Gated at AA (4.5) rather than 4.0 because that is the floor the derivation
    itself uses: a ramp whose hue clears 4.0 but not 4.5 keeps the neutral ink,
    so every ramp clears this by construction — this test is what keeps it true
    if the derivation or a curated palette changes.
    """
    # ``name`` is OPTIONAL for the reason this suite's other per-palette checks
    # take one: `test_host_theme.py`'s agreement gate calls every check in here
    # with a DERIVED ramp's name, and a check that cannot be called that way has
    # to be exempted by hand (and then it is not checked for the ramps the host
    # probe admits at all). Called bare it walks the curated ramps.
    for name in ([name] if name else _ALL_THEMES):
        tokens = theme.theme_spec(name).tokens
        for token in ("chip-live", "chip-success", "chip-warning"):
            for ground in ("overlay", "tint-select"):
                if ground not in tokens:
                    continue
                ratio = contrast(tokens[token], tokens[ground])
                assert ratio >= 4.5, (
                    f"{name}: {token} {tokens[token]} on {ground} {tokens[ground]}: "
                    f"{ratio:.2f} < 4.5 — the ask panel's state ink is illegible on its own ground"
                )


def test_the_ask_marker_reads_on_every_sidebar_ground(name: str | None = None) -> None:
    """The sidebar's ask marker paints on FOUR grounds, one of which nobody solved for.

    Review round 2 (D12): the round-1 remedy moved the panel's marker to the
    derived `chip-live` and left the sidebar's on raw `accent`. A cursor row in
    the focused sidebar paints `tint-select-hi` — a ground in neither `accent`'s
    derivation (`bg`/`surface`) nor the derived family's (`overlay`/
    `tint-select`) — and raw `accent` there is 3.96:1 on the brand light ramp,
    under the repo's own 4.0 state-hue floor; 14 of 54 ramps are under it and 2
    under 3:1 (duskfox 2.78, kanagawa-lotus 2.91), which is where a
    meaning-carrying glyph stops being reliably there.

    The floor is 4.0, not AA: this glyph is a state hue, like `danger` on
    `tint-danger` above. MEASURED worst case after the fix: 4.66:1
    (`everforest` on `tint-select-hi`).
    """
    for name in ([name] if name else _ALL_THEMES):
        tokens = theme.theme_spec(name).tokens
        ink = tokens["chip-live"]
        for ground in ("bg", "surface", "tint-select", "tint-select-hi"):
            if ground not in tokens:
                continue
            ratio = contrast(ink, tokens[ground])
            assert ratio >= 4.0, (
                f"{name}: the ask marker {ink} on {ground} {tokens[ground]}: "
                f"{ratio:.2f} < 4.0 — a meaning-carrying glyph on a row ground"
            )


def test_the_legend_card_reads_on_its_ground() -> None:
    """The `? Keys` card's copy rides the `overlay` ground — AA there (D3), measured.

    Design review round 1 of the legend: the descriptions were `dim` — 3.43:1
    dark / 2.72:1 light on `overlay`, the exact rung the `warning` pin above
    moved off — and the fix paints keys `fg` and descriptions `muted`. This
    pins both pairs with their numbers, and the hierarchy claim that `dim`
    stays below the copy rung, so a re-demotion fails here rather than in a
    screenshot. The BRAND ramps only, like the `warning` and `[missing]` pins
    above: `muted` is the ramp's own secondary ink, and re-flooring every
    registered palette for one card's ground is the design stream's call, not
    a coder's — the host-derived ramp sweep legitimately trades
    `muted`-on-`overlay` below 4.5 while its own calibrated floors hold.
    """
    for name in ("light", "dark"):
        tokens = theme.theme_spec(name).tokens
        for token in ("muted", "fg"):
            ratio = contrast(tokens[token], tokens["overlay"])
            assert ratio >= 4.5, (
                f"{name}: {token} {tokens[token]} on overlay {tokens['overlay']}: "
                f"{ratio:.2f} < 4.5 — the legend card's copy must clear AA"
            )
        # The hierarchy claim, so `dim` cannot come back as a "subtle" choice:
        # it must stay BELOW the copy rung on this ground.
        assert contrast(tokens["dim"], tokens["overlay"]) < contrast(
            tokens["muted"], tokens["overlay"]
        ), f"{name}: dim reads at or above muted on overlay"


@pytest.mark.parametrize("name", _ALL_THEMES)
def test_faint_sits_below_dim(name: str) -> None:
    tokens = theme.theme_spec(name).tokens
    assert contrast(tokens["faint"], tokens["bg"]) < contrast(tokens["dim"], tokens["bg"]), (
        f"{name}: faint ({tokens['faint']}) reads louder than dim ({tokens['dim']}) — "
        "the hint rung outranks the label rung"
    )


@pytest.mark.parametrize("name", _ALL_THEMES)
def test_elevation_is_monotonic(name: str) -> None:
    spec = theme.theme_spec(name)
    tokens = spec.tokens
    ladder = [_luminance(tokens[step]) for step in ("bg", "surface", "raised", "overlay")]
    if spec.dark:
        assert ladder == sorted(ladder), f"{name}: dark elevation must lighten upward: {ladder}"
        assert _luminance(tokens["sunken"]) <= ladder[0], f"{name}: sunken must sit below bg"
    else:
        assert ladder == sorted(
            ladder, reverse=True
        ), f"{name}: light elevation must darken upward: {ladder}"
        assert (
            _luminance(tokens["sunken"]) <= _luminance(tokens["bg"])
            or contrast(tokens["sunken"], tokens["bg"]) < 1.35
        ), f"{name}: light sunken should stay near or below the paper"


@pytest.mark.parametrize("name", _ALL_THEMES)
def test_tints_are_casts_not_slabs(name: str) -> None:
    """Every tint stays a cast, not a slab: near the ground in luminance."""
    tokens = theme.theme_spec(name).tokens
    failures: list[str] = []
    for token in ("tint-danger", "tint-select", "tint-select-hi"):
        ratio = contrast(tokens[token], tokens["bg"])
        if ratio > _TINT_CEILING:
            failures.append(f"{token} {tokens[token]} vs bg: {ratio:.2f} > {_TINT_CEILING}")
    assert not failures, f"{name}: " + "; ".join(failures)


@pytest.mark.parametrize("name", _ALL_THEMES)
def test_select_tint_survives_hover(name: str) -> None:
    """Hover on the selected row must read as MORE, not the same.

    The brand ramp's D8 lesson: tint-select-hi exists because hover has to be
    additive. Equal hexes silently erase hover feedback for mouse users.
    """
    tokens = theme.theme_spec(name).tokens
    assert (
        tokens["tint-select"] != tokens["tint-select-hi"]
    ), f"{name}: tint-select-hi equals tint-select — hover on the selected row is invisible"


@pytest.mark.parametrize("name", _ALL_THEMES)
def test_selected_composer_text_stays_legible_on_its_band(name: str) -> None:
    """Selecting text must highlight it, never erase it.

    Design round 1, D1. `Editor .text-area--selection` set only a background,
    so the selected run kept Textual's built-in `#e0e0e0` ink — a dark-theme
    default masquerading as a neutral. On the light ramp that landed at
    1.003:1 against the band (`#e0e0e0` on `#e5e0d5`): a triple-click made the
    whole draft look deleted, which is precisely the "my draft disappeared"
    reading the multi-click gesture exists to prevent.

    The stylesheet now names `$lo-fg`, so this asserts the pair the user
    actually sees for EVERY registered theme — the rule is shared by all of
    them, and a new palette whose `edge` drifts toward its `fg` would reopen
    the defect silently. AA (4.5:1) is the floor: this is body text the user is
    reading and editing, not a label.
    """
    # Resolved through the SEMANTIC names the stylesheet uses (`$lo-fg`,
    # `$lo-edge`), not the raw ramp keys: the light ramp aliases those to `ink`
    # and `hairline`, so reading the tokens directly would check a pair the
    # rule never renders.
    ink = theme.semantic_color("fg", name)
    band = theme.semantic_color("edge", name)
    ratio = contrast(ink, band)
    assert ratio >= 4.5, (
        f"{name}: selected text reads {ratio:.2f}:1 ({ink} on {band}) — "
        "a selection that erases its own text"
    )


def test_default_theme_is_operator_dark() -> None:
    """The product default stays the island night, whatever gets registered."""
    assert theme.DEFAULT_THEME == "dark"
    assert theme.available_themes()[0] == "dark"


def test_warning_ink_clears_aa_on_the_card_ground() -> None:
    """The modal card's warning rows ride ``overlay`` — AA there, measured, not assumed.

    Design review round 1 of the mesh repair notice (D2): the light ramp's
    ``warning`` measured 3.97:1 on ``overlay``, under the 4.5:1 floor for normal
    text, while the state hues' own floor above (4.0) is checked against
    ``bg``/``surface`` only — and the card's warning rows (the repair notice, the
    first-run teaching rows) sit on neither. The light ink is solved for this
    ground too; this pins the pair with its number. The BRAND ramps only: the
    registered-palette gate is the calibrated one above, and re-flooring every
    palette for one pair is not this pin's job.

    IT ALSO PINS THE PROJECTS PICKER'S REFUSAL (P5b, UX review round 1, U3):
    ``projects_view._style_resolver``'s ``refusal`` key is this same ``warning``
    token, and the start-session card paints its refusal sentence on
    ``overlay`` with it — the reason that card did not follow the web's
    ``text-danger``, which measures 3.9:1 in the light ramp. One pair, two
    cards, one number.
    """
    for name in ("light", "dark"):
        ink = theme.semantic_color("warning", name)
        ground = theme.semantic_color("overlay", name)
        ratio = contrast(ink, ground)
        assert ratio >= 4.5, (
            f"{name}: warning ink reads {ratio:.2f}:1 ({ink} on {ground}) — "
            "the card's warning rows fall under AA on their own ground"
        )


@pytest.mark.parametrize("name", _ALL_THEMES)
def test_the_live_chip_ink_clears_aa_on_the_card_ground(name: str) -> None:
    """The quick-send card's ``[live]`` chip rides the card's ground — AA there (D8).

    The pairing the palette could not inherit: a ramp's accent is solved against
    ``bg``/``surface``, and the card's chips sit on ``overlay``, a step further
    from the polarity's floor — where the brand light accent measures **3.49:1**
    (4.29:1 on the selected row's ``tint-select``) and no green or blue token in
    that ramp clears 4.5 (success 3.45, signal 3.54). The ``chip-live`` token is
    DERIVED per ramp by measurement (``theme._fill_chip_live``): the accent when
    it clears every ground the chip can sit on, the ramp's own neutral ink when
    it does not — so this check holds for EVERY theme, the derived terminal ramp
    included, rather than for the two brand ramps the other ground pins name.
    """
    from local_operator.tui.widgets.projects_send import state_ink_key

    assert (
        state_ink_key("live") == "chip_live"
    ), "the card's [live] chip re-mapped — re-measure its ink on the card ground"
    ink = theme.semantic_color("chip-live", name)
    for ground_token in ("overlay", "tint-select"):
        ground = theme.semantic_color(ground_token, name)
        ratio = contrast(ink, ground)
        assert ratio >= 4.5, (
            f"{name}: the [live] chip reads {ratio:.2f}:1 ({ink} on {ground}) — "
            "under AA on the card's own ground"
        )


def test_the_missing_chip_ink_clears_aa_on_the_card_ground() -> None:
    """The quick-send card's ``[missing]`` chip rides the card's ground — AA there (R2-1).

    The card's state chips paint on ``overlay`` (and on ``tint-select`` when
    their row is selected), never on ``bg``/``surface`` — and the chip for a
    link whose session is gone used to take ``dim``: 3.43:1 (dark) and 2.72:1
    (light) on ``overlay``, under the 4.5:1 floor for normal text and the exact
    number D5 moved the modal note ink off. It takes ``muted`` now; this pins
    the mapping AND the pair, with the ratio in the failure message. The BRAND
    ramps only, like the ``warning`` pin above: ``muted`` is the ramp's own
    secondary ink, and re-sourcing every curated palette's quiet chip for this
    one ground is the design stream's call, not a coder's.
    """
    from local_operator.tui.widgets.projects_send import state_ink_key

    assert (
        state_ink_key("missing") == "status_done"
    ), "the card's [missing] chip re-mapped — re-measure its ink on the card ground"
    for name in ("light", "dark"):
        ink = theme.semantic_color("muted", name)
        for ground_token in ("overlay", "tint-select"):
            ground = theme.semantic_color(ground_token, name)
            ratio = contrast(ink, ground)
            assert ratio >= 4.5, (
                f"{name}: the [missing] chip reads {ratio:.2f}:1 ({ink} on {ground}) — "
                "under AA on the card's own ground"
            )
