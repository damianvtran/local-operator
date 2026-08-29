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
    spec = theme.theme_spec(name)
    assert set(spec.tokens) == set(theme.SEMANTIC_TOKENS)
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
