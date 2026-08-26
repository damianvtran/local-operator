#!/usr/bin/env python3
"""Regenerate every Chrome Web Store visual asset from the source frames.

WHY THIS EXISTS
---------------
The Web Store listing needs a fixed set of pixel-exact PNGs (a 128x128 store
icon, five 1280x800 marketing screenshots, a 440x280 small promo tile, and a
1400x560 marquee). Those assets are composites of *real* rendered extension
frames — never hand-drawn mock UI — so the only defensible way to own them is a
script that rebuilds each one deterministically from the captured evidence. If
a frame is re-captured or a caption is reworded, re-running this file
regenerates the whole package identically; nothing is edited by hand in an
image editor and then lost.

DESIGN LANGUAGE — the local-operator.com system, not an ad-hoc one
------------------------------------------------------------------
Every colour, face, size and radius below is lifted from the website's design
system (the site repo's `src/index.css` `@theme` block, which is itself the
projection of `docs/design-language.md`). The store creative is the first
thing a visitor sees between the site and the product, so it must sit in the
same visual world as both:

* Faces: Fraunces (opsz build) for editorial display headlines at weight 400 —
  the site deliberately rejects lighter display weights because Fraunces at
  300 on warm paper loses its hairlines; Figtree for body copy; JetBrains Mono
  for uppercase eyebrow labels and URLs. The exact WOFF2 variable builds the
  site ships are vendored in ./fonts/ (see fonts/README.md) so this script
  never depends on OS-installed fonts.
* Ground: warm paper `#f7f4ee`, never a cool grey and never the old
  near-black marketing canvas. The extension popup's own background is
  `(247,244,239)` — one RGB point off site paper — so the real frames and the
  marketing ground share one colour world by construction.
* Ink ramp: `ink #211e18` for headlines, `ink-muted #565147` for body,
  `ink-dim #6c675c` for tertiary. All clear AA on paper (ratios tabulated in
  the site's design-language doc).
* Accent: the re-solved brand green `#177b45` (AA on light grounds), used
  sparingly — the eyebrow dot and eyebrow text only. One accent, two places.
* Motif M5, the dot field: 1px dots on a 24px pitch in hairline-strong
  `#d5cfc2`, masked to fade before any edge. On the site it appears behind the
  hero figure only; here it sits behind the framed artefact for the same
  reason — texture concentrates behind the object, never wallpaper.
* Radii: the site's `--radius-md`/`--radius-lg` (10/14px). Cards carry a
  1px hairline-strong border; elevation is primarily the border plus a very
  soft shadow (the site has no shadow ladder, so the shadow here stays at the
  threshold of perception — separation, not drama).

SOURCES (all committed in the repo)
-----------------------------------
* Brand glyph:        static/local-operator-icon-2-{light,dark}-clear.png
                      2048x750 RGBA; the only non-transparent artwork is the
                      icon glyph, alpha bbox (888,202)-(1206,550).
* Rendered frames:    docs/evidence/browser-extension/*.png — the real popup,
                      options page, and agent-driven page captured from the
                      release build.
* Fonts:              docs/store/assets/fonts/*.woff2 — the site's own
                      variable builds (Fraunces opsz, Figtree, JetBrains Mono).

CONSTRAINTS (Chrome Web Store)
------------------------------
* Exact sizes: icon 128x128, screenshots 1280x800, small tile 440x280,
  marquee 1400x560. The driver prints each output's size/mode as a check.
* No alpha: every file is saved RGB. CWS rejects or flattens alpha
  unpredictably, so we flatten deliberately onto our own ground.
* The real frame is always the hero of each screenshot; the canvas only
  frames and captions it. Captions are the plain declarative lines from
  listing.md — no hype, benefit first.

Run:  python3 docs/store/assets/build_assets.py   (needs Pillow >= 10 with
FreeType >= 2.13 for WOFF2 variable-font loading; the repo venv qualifies).
Outputs land next to this script in docs/store/assets/.
"""

from __future__ import annotations

import os

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
EVID = os.path.join(REPO, "docs", "evidence", "browser-extension")
STATIC = os.path.join(REPO, "static")
FONTS = os.path.join(HERE, "fonts")
OUT = HERE


def evid(name: str) -> str:
    return os.path.join(EVID, name)


# ---------------------------------------------------------------------------
# Palette — verbatim from the site's `@theme` block (src/index.css).
# Names kept identical to the site tokens so a future re-theme is a diff of
# hex values, not a re-derivation.
# ---------------------------------------------------------------------------
PAPER = (247, 244, 238)  # --color-paper      #f7f4ee — the ground
SURFACE = (252, 251, 247)  # --color-surface    #fcfbf7
ELEVATED = (255, 254, 251)  # --color-elevated   #fffefb — card fill
SUNKEN = (239, 236, 227)  # --color-sunken     #efece3 — browser bar
INK = (33, 30, 24)  # --color-ink        #211e18 — headlines
INK_MUTED = (86, 81, 71)  # --color-ink-muted  #565147 — body copy
INK_DIM = (108, 103, 92)  # --color-ink-dim    #6c675c — tertiary
HAIRLINE = (229, 224, 213)  # --color-hairline   #e5e0d5 — decorative rule
HAIRLINE_STRONG = (213, 207, 194)  # --color-hairline-strong #d5cfc2 — card edge
ACCENT = (23, 123, 69)  # --color-accent     #177b45 — AA on paper
POPUP_BG = (247, 244, 239)  # the popup capture's own background; one RGB
#                             point off PAPER, so mats blend seamlessly.
WHITE = (255, 255, 255)

# ---------------------------------------------------------------------------
# Fonts — the site's own variable WOFF2 builds, vendored in ./fonts.
#
# Pillow (via FreeType >= 2.13) loads WOFF2 directly and exposes the variation
# axes. Axis order matters for set_variation_by_axes: Fraunces is [opsz, wght]
# (the opsz build is the whole point — it carries the display drawing the
# site's headlines use), Figtree and JetBrains Mono are [wght].
# ---------------------------------------------------------------------------
_FONT_FILES = {
    "display": os.path.join(FONTS, "fraunces-5.3.0-latin.woff2"),
    "sans": os.path.join(FONTS, "figtree-5.3.0-latin.woff2"),
    "mono": os.path.join(FONTS, "jetbrains-mono-5.3.0-latin.woff2"),
}
_font_cache: dict[tuple, ImageFont.FreeTypeFont] = {}


def font(family: str, size: int, weight: int = 400) -> ImageFont.FreeTypeFont:
    """A cached variable-font instance.

    For Fraunces the optical-size axis is set the way `font-optical-sizing:
    auto` would: opsz = the CSS-point equivalent of the pixel size (px * 0.75),
    clamped to the axis range. That is what makes 50px headlines here render
    with the same high-contrast display drawing as the site's h1, instead of
    the body-optical drawing scaled up.
    """
    key = (family, size, weight)
    if key not in _font_cache:
        f = ImageFont.truetype(_FONT_FILES[family], size)
        if family == "display":
            opsz = max(9.0, min(144.0, size * 0.75))
            f.set_variation_by_axes([opsz, weight])
        else:
            f.set_variation_by_axes([weight])
        _font_cache[key] = f
    return _font_cache[key]


# ---------------------------------------------------------------------------
# Text helpers with tracking.
#
# The site's display steps carry negative letter-spacing (-0.028em at
# display-lg) and the mono eyebrow carries +0.08em; Pillow has no
# letter-spacing, so we draw glyph by glyph using kerned pair advances
# (textlength(prev+ch) - textlength(prev) preserves the font's kerning) plus a
# constant tracking offset. Without this, Fraunces sets visibly looser than
# the site and the mono eyebrows lose their label quality.
# ---------------------------------------------------------------------------
def text_width(draw, text: str, fnt, tracking: float = 0.0) -> float:
    return draw.textlength(text, font=fnt) + tracking * max(0, len(text) - 1)


def draw_tracked(draw, xy, text: str, fnt, fill, tracking: float = 0.0):
    if not tracking:
        draw.text(xy, text, font=fnt, fill=fill)
        return
    x, y = xy
    prev = ""
    for ch in text:
        draw.text((x, y), ch, font=fnt, fill=fill)
        if prev:
            adv = draw.textlength(prev + ch, font=fnt) - draw.textlength(prev, font=fnt)
        else:
            adv = draw.textlength(ch, font=fnt)
        x += adv + tracking
        prev = ch


def wrap(draw, text, fnt, max_w, tracking: float = 0.0):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        trial = (cur + " " + wd).strip()
        if text_width(draw, trial, fnt, tracking) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


# ---------------------------------------------------------------------------
# Type scale for the 1280x800 canvases.
#
# Derived from the site's scale rather than invented: display ~ display-md/lg
# territory, weight 400 (the system's display weight — see the design doc's
# note about lighter weights losing their hairlines on paper), body ~ body-lg,
# eyebrow = the mono-label recipe (uppercase, +0.08em, weight 500).
# ---------------------------------------------------------------------------
TITLE_SIZE = 50
TITLE_LINE = round(TITLE_SIZE * 1.10)
TITLE_TRACK = -0.024 * TITLE_SIZE
SUB_SIZE = 22
SUB_LINE = round(SUB_SIZE * 1.55)
EYEBROW_SIZE = 15
EYEBROW_TRACK = 0.08 * EYEBROW_SIZE


def title_font():
    return font("display", TITLE_SIZE, 400)


def sub_font():
    return font("sans", SUB_SIZE, 400)


def eyebrow_font():
    return font("mono", EYEBROW_SIZE, 500)


# ---------------------------------------------------------------------------
# Motif M5 — the dot field.
#
# The site's DotField component: 1px dots on a 24px pitch in hairline-strong,
# masked with a radial gradient so the field fades out well before its box
# edges (a field that terminates on a hard edge reads as a broken texture).
# The site composites a radial and two linear masks; a single radial whose
# radius stays inside the box achieves the same "never touches an edge" result
# in a raster.
# ---------------------------------------------------------------------------
def dot_field(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    focus: tuple[float, float] = (0.5, 0.45),
    pitch: int = 24,
    max_alpha: int = 255,
):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    col = HAIRLINE_STRONG + (max_alpha,)
    for yy in range(pitch // 2, h, pitch):
        for xx in range(pitch // 2, w, pitch):
            d.ellipse([xx - 1, yy - 1, xx + 1, yy + 1], fill=col)

    # Radial falloff mask, computed at 1/8 scale then resized — cheap and
    # smooth. Full opacity to 30% of the radius, cosine-ish falloff to 85%.
    mw, mh = max(1, w // 8), max(1, h // 8)
    mask = Image.new("L", (mw, mh), 0)
    fx, fy = focus[0] * mw, focus[1] * mh
    rad = 0.62 * max(mw, mh)
    px = mask.load()
    for yy in range(mh):
        for xx in range(mw):
            dist = ((xx - fx) ** 2 + ((yy - fy) * 1.25) ** 2) ** 0.5 / rad
            if dist <= 0.3:
                px[xx, yy] = 255
            elif dist < 1.0:
                t = (dist - 0.3) / 0.7
                px[xx, yy] = int(255 * (1 - t) ** 2)
    mask = mask.resize((w, h), Image.BILINEAR)
    layer.putalpha(ImageChops.multiply(layer.split()[3], mask))
    canvas.paste(layer, (x0, y0), layer)


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------
def content_bbox(im: Image.Image, bg: tuple[int, int, int], thresh: int = 12):
    """Bounding box of everything that differs from the flat background.

    Popup captures are 640x960 with a large empty tail below the card; cropping
    to real content stops that void from dominating the composite.
    """
    rgb = im.convert("RGB")
    flat = Image.new("RGB", rgb.size, bg)
    diff = ImageChops.difference(rgb, flat).convert("L").point(lambda v: 255 if v > thresh else 0)
    return diff.getbbox()


def rounded_shadow_card(
    canvas: Image.Image,
    frame: Image.Image,
    xy: tuple[int, int],
    radius: int = 14,
    border: tuple[int, int, int] | None = HAIRLINE_STRONG,
    shadow: bool = True,
) -> tuple[int, int, int, int]:
    """Paste `frame` onto `canvas` at top-left `xy` inside a rounded card.

    Separation on warm paper comes from the 1px hairline-strong border; the
    shadow is kept at the threshold of perception (the site has no shadow
    ladder, so anything heavier would read foreign next to it).
    """
    x, y = xy
    w, h = frame.size
    if shadow:
        pad = 36
        sh = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
        sd = ImageDraw.Draw(sh)
        sd.rounded_rectangle(
            [pad, pad + 5, pad + w, pad + h + 5], radius=radius, fill=(33, 30, 24, 34)
        )
        sh = sh.filter(ImageFilter.GaussianBlur(14))
        canvas.paste(sh, (x - pad, y - pad), sh)

    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    canvas.paste(frame.convert("RGB"), (x, y), mask)
    if border:
        ImageDraw.Draw(canvas).rounded_rectangle(
            [x, y, x + w - 1, y + h - 1], radius=radius, outline=border, width=1
        )
    return (x, y, x + w, y + h)


def draw_eyebrow(draw, x, y, label: str) -> int:
    """The site's mono eyebrow: a 6px accent dot, then uppercase JetBrains
    Mono at +0.08em in the brand green. Returns the y below the block."""
    ef = eyebrow_font()
    asc, _ = ef.getmetrics()
    cy = y + asc // 2 + 2
    draw.ellipse([x, cy - 3, x + 6, cy + 3], fill=ACCENT)
    draw_tracked(draw, (x + 16, y), label.upper(), ef, ACCENT, EYEBROW_TRACK)
    return y + asc + 6


def draw_caption_block(
    draw,
    x,
    y,
    eyebrow: str,
    title: str,
    sub: str,
    max_w: int,
) -> int:
    """Eyebrow + Fraunces headline + Figtree subline, top at `y`.

    Returns the y below the block. Spacing steps (14 under the eyebrow, 18
    above the sub) echo the site's section-header rhythm.
    """
    tf, sf = title_font(), sub_font()
    if eyebrow:
        y = draw_eyebrow(draw, x, y, eyebrow) + 14
    for line in wrap(draw, title, tf, max_w, TITLE_TRACK):
        draw_tracked(draw, (x, y), line, tf, INK, TITLE_TRACK)
        y += TITLE_LINE
    if sub:
        y += 18
        for line in wrap(draw, sub, sf, max_w):
            draw.text((x, y), line, font=sf, fill=INK_MUTED)
            y += SUB_LINE
    return y


def caption_block_height(draw, eyebrow: str, title: str, sub: str, max_w: int) -> int:
    tf, sf = title_font(), sub_font()
    h = 0
    if eyebrow:
        h += eyebrow_font().getmetrics()[0] + 6 + 14
    h += len(wrap(draw, title, tf, max_w, TITLE_TRACK)) * TITLE_LINE
    if sub:
        h += 18 + len(wrap(draw, sub, sf, max_w)) * SUB_LINE
    return h


# ---------------------------------------------------------------------------
# 1. Store icon — 128x128, black glyph on warm-paper rounded plate
# ---------------------------------------------------------------------------
def build_store_icon(path: str):
    """Crop the black 'light' glyph, seat it on a warm-paper plate.

    The plate fills the full 128x128 with NO drawn corner rounding: the file
    must be flat RGB (no alpha), so any rounding we draw would expose square
    corner fills in whatever colour we chose — white corners on a dark store
    surface, dark corners on a light one. CWS applies its own rounded mask on
    most surfaces, so we bleed the plate to the edges and keep the important
    artwork inside the central safe area, exactly as the CWS guidance asks.

    Site paper `#f7f4ee` rather than the old cool `#F7F7F9`, so the icon
    matches the listing screenshots and the website; a 1px hairline-strong
    inset border gives the plate a boundary on a white store card, and the
    light plate itself supplies contrast on a dark card — which is why we ship
    a solid backplate rather than a transparent glyph that would vanish on one
    surface or the other.
    """
    S = 128
    glyph = Image.open(os.path.join(STATIC, "local-operator-icon-2-light-clear.png")).convert(
        "RGBA"
    )
    gb = glyph.split()[3].getbbox()  # (888,202,1206,550)
    g = glyph.crop(gb)  # 318x348 black glyph, tight

    # Target glyph height ~54% of the icon so it breathes inside the plate.
    target_h = int(S * 0.54)
    scale = target_h / g.height
    g = g.resize((max(1, round(g.width * scale)), target_h), Image.LANCZOS)

    icon = Image.new("RGB", (S, S), PAPER)
    icon.paste(g, ((S - g.width) // 2, (S - g.height) // 2), g)

    # Hairline-strong border at the full perimeter so the plate has an edge on
    # white store cards; CWS's own mask clips it into a rounded edge where the
    # surface rounds.
    ImageDraw.Draw(icon).rectangle([0, 0, S - 1, S - 1], outline=HAIRLINE_STRONG, width=1)
    icon.save(path)
    return path


def _icon_plate(size: int, radius: int) -> Image.Image:
    """Small rounded warm-paper brand chip carrying the black glyph, reused in
    the promo tiles as the product mark. On the paper canvas the chip is
    ELEVATED (a step up in lightness, exactly how the site does elevation)
    with a hairline-strong edge."""
    glyph = Image.open(os.path.join(STATIC, "local-operator-icon-2-light-clear.png")).convert(
        "RGBA"
    )
    g = glyph.crop(glyph.split()[3].getbbox())
    th = int(size * 0.54)
    g = g.resize((round(g.width * th / g.height), th), Image.LANCZOS)
    chip = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    chip.paste(Image.new("RGBA", (size, size), ELEVATED + (255,)), (0, 0), m)
    chip.paste(g, ((size - g.width) // 2, (size - g.height) // 2), g)
    ImageDraw.Draw(chip).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=radius, outline=HAIRLINE_STRONG + (255,), width=1
    )
    return chip


# ---------------------------------------------------------------------------
# Popup / options frame preparation
# ---------------------------------------------------------------------------
def prep_popup(name: str, pad: int = 28) -> Image.Image:
    """Load a 640x960 popup capture, crop to its real content, and re-mat it
    on the popup's OWN background colour with even padding.

    Matting on the capture's own `(247,244,239)` — not white — is what makes
    the mat invisible: the old white mat left a visible seam around the
    paper-toned popup card.
    """
    im = Image.open(evid(name)).convert("RGB")
    box = content_bbox(im, POPUP_BG)
    x0, y0, x1, y1 = box
    crop = im.crop((0, y0, im.width, y1))  # keep full width, trim vertical void
    card = Image.new("RGB", (crop.width, crop.height + pad * 2), POPUP_BG)
    card.paste(crop, (0, pad))
    return card


def mat(crop: Image.Image, pad: int, bg=POPUP_BG) -> Image.Image:
    """Re-mat a tight crop on a flat background with even padding on all sides.

    Needed for card composites: a crop taken flush to its content, once given
    rounded corners, would clip the artwork that touches the edge (the options
    'Local Operator' heading sat exactly on the card's left rounded corner).
    Padding restores the breathing room the real page has.
    """
    out = Image.new("RGB", (crop.width + pad * 2, crop.height + pad * 2), bg)
    out.paste(crop, (pad, pad))
    return out


# ---------------------------------------------------------------------------
# Shared screenshot furniture
# ---------------------------------------------------------------------------
MARGIN = 96  # canvas margin, echoing the site's container gutter rhythm


def _fit(im: Image.Image, max_h: int, max_w: int) -> Image.Image:
    scale = min(max_h / im.height, max_w / im.width, 1.0)
    if scale < 1.0:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    return im


def _split_canvas(
    eyebrow: str,
    title: str,
    sub: str,
    frame: Image.Image,
    frame_right_margin: int = 90,
) -> Image.Image:
    """The standard split layout: warm paper ground, dot field behind the
    right half (motif M5, focused on the frame), left caption column
    vertically centred, real frame in an elevated card on the right.

    The caption column is capped at 500px so it never reaches under the frame
    even when the frame is wide (the options card).
    """
    W, H = 1280, 800
    cv = Image.new("RGB", (W, H), PAPER)
    fx = W - frame.width - frame_right_margin
    fy = (H - frame.height) // 2
    # Field behind the right half only, focal point on the frame's centre —
    # the same "texture the object, not the page" rule as the site hero.
    dot_field(
        cv,
        (W // 2 - 60, 0, W, H),
        focus=(
            (fx + frame.width / 2 - (W // 2 - 60)) / (W / 2 + 60),
            0.5,
        ),
    )
    d = ImageDraw.Draw(cv)
    max_w = 500
    bh = caption_block_height(d, eyebrow, title, sub, max_w)
    draw_caption_block(d, MARGIN, (H - bh) // 2, eyebrow, title, sub, max_w)
    rounded_shadow_card(cv, frame, (fx, fy))
    return cv


def _browser_chrome(page: Image.Image, url: str) -> Image.Image:
    """Wrap a page screenshot in a minimal browser top-bar (traffic lights +
    URL pill) so the reached page reads as 'in a real browser tab' without
    exposing any personal chrome. The bar sits on the site's `sunken` tone
    with hairline rules, and the URL is set in JetBrains Mono — the same
    voice the extension's own URL fields use."""
    bar = 44
    w = page.width
    out = Image.new("RGB", (w, page.height + bar), WHITE)
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, w, bar], fill=SUNKEN)
    d.line([0, bar, w, bar], fill=HAIRLINE_STRONG, width=1)
    for i, col in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
        cx = 20 + i * 22
        d.ellipse([cx, bar // 2 - 6, cx + 12, bar // 2 + 6], fill=col)
    pill_x0 = 96
    d.rounded_rectangle(
        [pill_x0, 9, w - 20, bar - 9], radius=10, fill=ELEVATED, outline=HAIRLINE_STRONG, width=1
    )
    uf = font("mono", 15, 400)
    asc, desc = uf.getmetrics()
    d.text((pill_x0 + 14, (bar - asc - desc) // 2 + 1), url, font=uf, fill=INK_DIM)
    out.paste(page, (0, bar))
    return out


# ---------------------------------------------------------------------------
# 2. Screenshots — five 1280x800 canvases
# ---------------------------------------------------------------------------
def build_screenshot_connected(path: str):
    """(a) Connected / driving — the hero 'agent in your browser' state."""
    frame = _fit(prep_popup("popup-connected.png"), max_h=560, max_w=440)
    cv = _split_canvas(
        "Connected",
        "Your browser, with an agent in it.",
        "Paired and connected. Local Operator drives one tab and shows you what it "
        "reached — here, Page Two on your own machine.",
        frame,
    )
    cv.save(path)
    return path


def build_screenshot_allow(path: str):
    """(b) Per-site allow prompt — default-deny consent."""
    frame = _fit(prep_popup("popup-origin-prompt.png"), max_h=640, max_w=470)
    cv = _split_canvas(
        "Per-site consent",
        "You decide which sites.",
        "The first time the agent wants a new site it asks. Allow once, always "
        "allow, or deny — reversible any time in Settings.",
        frame,
    )
    cv.save(path)
    return path


def build_screenshot_pairing(path: str):
    """(c) Pairing — one code, no account."""
    frame = _fit(prep_popup("popup-pairing.png"), max_h=560, max_w=440)
    cv = _split_canvas(
        "Pairing",
        "Paired to your app and nothing else.",
        "A one-time code shown in your terminal ties this browser to your copy of "
        "Local Operator. No account, no sign-up.",
        frame,
    )
    cv.save(path)
    return path


def build_screenshot_action(path: str):
    """(d) The agent driving a page — real reached page beside the popup.

    Different layout from the split canvases: a top caption band, then the
    reached page as the wide hero with the connected popup overlapping it —
    the overlap is the visual claim that the popup owns the page.
    """
    W, H = 1280, 800
    cv = Image.new("RGB", (W, H), PAPER)
    dot_field(cv, (W // 3, H // 4, W, H), focus=(0.55, 0.55))
    d = ImageDraw.Draw(cv)
    y = draw_caption_block(
        d,
        MARGIN,
        56,
        "Agent at work",
        "It drives one tab — never yours.",
        "Ask in Local Operator; the agent opens its own tab, reaches the page, and "
        "reports the real result.",
        max_w=W - 2 * MARGIN,
    )
    top = y + 30

    # The reached page (browser content) as the wide hero. The real test page
    # is deliberately sparse ("Page Two / Second page loaded."), so crop to a
    # page-top band that keeps the heading legible with balanced whitespace
    # rather than a near-empty white expanse.
    page = Image.open(evid("action-screenshot-page2.png")).convert("RGB")
    cb = content_bbox(page, (255, 255, 255))
    page = page.crop((0, 0, min(page.width, cb[2] + 340), min(page.height, cb[3] + 150)))
    page = _fit(page, max_h=H - top - 64, max_w=740)
    page_card = _browser_chrome(page, "127.0.0.1:8791/lop-page2.html")
    px = MARGIN
    py = top + ((H - top - 64) - page_card.height) // 2
    rounded_shadow_card(cv, page_card, (px, py))

    # The connected popup overlapping to the right, proving the agent owns it.
    pop = _fit(prep_popup("popup-connected.png"), max_h=430, max_w=360)
    rounded_shadow_card(cv, pop, (W - pop.width - 84, py + page_card.height - pop.height + 12))
    cv.save(path)
    return path


def build_screenshot_options(path: str):
    """(e) Options / allowed sites — everything stays on your machine."""
    # Unlike the popup screenshots, this frame never needs the canvas size:
    # _split_canvas owns the 1280x800 geometry and the options capture is
    # fitted purely by max_h/max_w below.
    im = Image.open(evid("options-populated.png")).convert("RGB")
    box = content_bbox(im, POPUP_BG)
    # Trim the bottom so the card ends after the first full allowed-site row
    # instead of slicing the second row mid-height, then mat on the page's own
    # background so the heading clears the rounded corner.
    x0, y0, x1, y1 = box
    im = im.crop((x0, y0, x1, min(y1, 1075)))
    im = mat(im, 26)
    frame = _fit(im, max_h=650, max_w=580)
    cv = _split_canvas(
        "Local only",
        "Everything stays on your machine.",
        "The extension talks only to the local bridge on your own computer. Manage "
        "the daemon port and your allowed sites here.",
        frame,
        frame_right_margin=80,
    )
    cv.save(path)
    return path


# ---------------------------------------------------------------------------
# 3. Promo tiles
# ---------------------------------------------------------------------------
def build_small_tile(path: str):
    """440x280 small promo — mark, product name, one line.

    Tiny canvas, so the composition is a centred stack on paper with a faint
    dot field: chip, Fraunces name in ink, Figtree line in ink-muted, and the
    mono 'free & open source' eyebrow in accent green as the only colour.
    """
    W, H = 440, 280
    cv = Image.new("RGB", (W, H), PAPER)
    dot_field(cv, (0, 0, W, H), focus=(0.5, 0.28), max_alpha=200)
    chip = _icon_plate(72, radius=18)
    cv.paste(chip, ((W - chip.width) // 2, 36), chip)
    d = ImageDraw.Draw(cv)

    name_f = font("display", 33, 480)
    name = "Local Operator"
    track = -0.02 * 33
    nw = text_width(d, name, name_f, track)
    draw_tracked(d, ((W - nw) // 2, 128), name, name_f, INK, track)

    line_f = font("sans", 17, 400)
    line = "Your browser. Your logins. Your machine."
    d.text(((W - d.textlength(line, font=line_f)) // 2, 180), line, font=line_f, fill=INK_MUTED)

    tag_f = font("mono", 12, 500)
    tag = "FREE AND OPEN SOURCE"
    tw = text_width(d, tag, tag_f, 0.08 * 12)
    draw_tracked(d, ((W - tw) // 2, 222), tag, tag_f, ACCENT, 0.08 * 12)
    cv.save(path)
    return path


def build_marquee(path: str):
    """1400x560 marquee — left brand column, right connected-popup composite.

    The wide-canvas version of the split layout: brand chip + Fraunces name +
    Figtree line on the left, the reached page in browser chrome with the
    connected popup overlapping on the right, dot field behind the composite.
    Focal content stays centred-ish because merchandising crops vary.
    """
    W, H = 1400, 560
    cv = Image.new("RGB", (W, H), PAPER)
    dot_field(cv, (W // 2 - 80, 0, W, H), focus=(0.5, 0.5))
    d = ImageDraw.Draw(cv)

    # Left ~40%: vertically centred brand stack.
    x = MARGIN
    chip = _icon_plate(80, radius=20)
    name_f = font("display", 46, 400)
    line_f = font("sans", 23, 400)
    lines = wrap(d, "The agent works where you're already signed in.", line_f, 440)
    stack_h = 80 + 28 + round(46 * 1.1) + 14 + len(lines) * round(23 * 1.55) + 26 + 15
    y = (H - stack_h) // 2
    cv.paste(chip, (x, y), chip)
    y += 80 + 28
    draw_tracked(d, (x, y), "Local Operator", name_f, INK, -0.024 * 46)
    y += round(46 * 1.1) + 14
    for ln in lines:
        d.text((x, y), ln, font=line_f, fill=INK_MUTED)
        y += round(23 * 1.55)
    y += 26
    draw_tracked(d, (x, y), "FREE AND OPEN SOURCE", font("mono", 13, 500), ACCENT, 0.08 * 13)

    # Right ~60%: the reached page in browser chrome with the connected popup.
    page = Image.open(evid("action-screenshot-page2.png")).convert("RGB")
    cb = content_bbox(page, (255, 255, 255))
    page = page.crop((0, 0, min(page.width, cb[2] + 340), min(page.height, cb[3] + 150)))
    page = _fit(page, max_h=360, max_w=600)
    page_card = _browser_chrome(page, "127.0.0.1:8791/lop-page2.html")
    px = 640
    py = (H - page_card.height) // 2
    rounded_shadow_card(cv, page_card, (px, py))

    pop = _fit(prep_popup("popup-connected.png"), max_h=300, max_w=250)
    rounded_shadow_card(cv, pop, (W - pop.width - 76, py + page_card.height - pop.height + 10))
    cv.save(path)
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
BUILDERS = [
    ("store-icon-128.png", build_store_icon),
    ("screenshot-1-connected.png", build_screenshot_connected),
    ("screenshot-2-allow-site.png", build_screenshot_allow),
    ("screenshot-3-pairing.png", build_screenshot_pairing),
    ("screenshot-4-agent-driving.png", build_screenshot_action),
    ("screenshot-5-options.png", build_screenshot_options),
    ("promo-small-440x280.png", build_small_tile),
    ("promo-marquee-1400x560.png", build_marquee),
]


def main():
    for name, fn in BUILDERS:
        out = os.path.join(OUT, name)
        fn(out)
        with Image.open(out) as im:
            print(f"{name:32s} {im.size[0]}x{im.size[1]} {im.mode}")


if __name__ == "__main__":
    main()
