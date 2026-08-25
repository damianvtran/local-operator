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

SOURCES (all committed in the repo)
-----------------------------------
* Brand glyph:        static/local-operator-icon-2-{light,dark}-clear.png
                      2048x750 RGBA; the only non-transparent artwork is the
                      icon glyph, alpha bbox (888,202)-(1206,550).
* Rendered frames:    docs/evidence/browser-extension/*.png — the real popup,
                      options page, and agent-driven page captured from the
                      release build.

DESIGN DECISIONS (see docs/store/assets.md and listing.md)
----------------------------------------------------------
* Store-icon backplate: the validated treatment in icons-on-dark.png is a light
  off-white rounded square (#F7F7F9) carrying the BLACK ("light" variant) glyph.
  A light plate reads on a white store card (rounded edge + hairline border give
  it a boundary) AND on a dark card (the plate itself supplies contrast), which
  is why we ship a solid backplate rather than a transparent glyph that would
  vanish on one surface or the other.
* Voice: plain, short, declarative captions taken from listing.md's own
  headings ("Your browser, with an agent in it." etc.). No hype, benefit first.
* The real frame is always the hero of each screenshot; the canvas only frames
  and captions it.

Run:  .venv/bin/python docs/store/assets/build_assets.py
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
OUT = HERE


def evid(name: str) -> str:
    return os.path.join(EVID, name)


# ---------------------------------------------------------------------------
# Brand palette
# ---------------------------------------------------------------------------
# Sampled directly from the validated icons-on-dark.png and the rendered frames
# so the composites sit in the same colour world as the real UI.
INK = (17, 17, 19)            # near-black headline/glyph ink
INK_SOFT = (74, 74, 82)       # secondary caption text
PLATE = (247, 247, 249)       # validated off-white backplate (#F7F7F9)
CANVAS_DARK = (24, 24, 27)    # deep neutral marketing canvas
CANVAS_LIGHT = (247, 247, 249)
CARD_BORDER = (228, 228, 232)
WHITE = (255, 255, 255)
ACCENT = (176, 58, 46)        # the extension's own deny/unpair red, used sparingly

# ---------------------------------------------------------------------------
# Fonts — Helvetica Neue matches the extension UI's system sans closely.
# ---------------------------------------------------------------------------
_HN = "/System/Library/Fonts/HelveticaNeue.ttc"
_FACE = {"regular": 0, "bold": 1, "medium": 10, "light": 7}


def font(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_HN, size, index=_FACE[weight])


# ---------------------------------------------------------------------------
# Small drawing helpers
# ---------------------------------------------------------------------------
def content_bbox(im: Image.Image, bg: tuple[int, int, int], thresh: int = 12):
    """Bounding box of everything that differs from the flat background.

    Popup captures are 600x840 with a large empty tail below the card; cropping
    to real content stops that void from dominating the composite.
    """
    rgb = im.convert("RGB")
    flat = Image.new("RGB", rgb.size, bg)
    diff = ImageChops.difference(rgb, flat).convert("L").point(
        lambda v: 255 if v > thresh else 0
    )
    return diff.getbbox()


def rounded_shadow_card(
    canvas: Image.Image,
    frame: Image.Image,
    xy: tuple[int, int],
    radius: int = 20,
    border: tuple[int, int, int] | None = CARD_BORDER,
    shadow: bool = True,
) -> tuple[int, int, int, int]:
    """Paste `frame` onto `canvas` at top-left `xy` inside a rounded, softly
    shadowed white card. Returns the placed box (x0,y0,x1,y1).
    """
    x, y = xy
    w, h = frame.size
    if shadow:
        # Build the drop shadow on a padded transparent layer, blur, paste.
        pad = 40
        sh = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
        sd = ImageDraw.Draw(sh)
        sd.rounded_rectangle(
            [pad, pad + 6, pad + w, pad + h + 6], radius=radius, fill=(0, 0, 0, 70)
        )
        sh = sh.filter(ImageFilter.GaussianBlur(18))
        canvas.paste(sh, (x - pad, y - pad), sh)

    # Rounded-corner mask so the frame corners match the card.
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    canvas.paste(frame.convert("RGB"), (x, y), mask)
    if border:
        ImageDraw.Draw(canvas).rounded_rectangle(
            [x, y, x + w - 1, y + h - 1], radius=radius, outline=border, width=1
        )
    return (x, y, x + w, y + h)


def wrap(draw, text, fnt, max_w):
    words, lines, cur = text.split(), [], ""
    for wd in words:
        trial = (cur + " " + wd).strip()
        if draw.textlength(trial, font=fnt) <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = wd
    if cur:
        lines.append(cur)
    return lines


def draw_caption(draw, x, y, title, sub, title_fnt, sub_fnt, max_w, fill=INK,
                 sub_fill=INK_SOFT, line_gap=10, block_gap=16):
    """Render a headline + optional subline, return the y below the block."""
    for line in wrap(draw, title, title_fnt, max_w):
        draw.text((x, y), line, font=title_fnt, fill=fill)
        asc, desc = title_fnt.getmetrics()
        y += asc + desc + line_gap
    if sub:
        y += block_gap - line_gap
        for line in wrap(draw, sub, sub_fnt, max_w):
            draw.text((x, y), line, font=sub_fnt, fill=sub_fill)
            asc, desc = sub_fnt.getmetrics()
            y += asc + desc + 6
    return y


# ---------------------------------------------------------------------------
# 1. Store icon — 128x128, black glyph on off-white rounded plate
# ---------------------------------------------------------------------------
def build_store_icon(path: str):
    """Crop the black 'light' glyph, seat it on the validated off-white plate.

    The plate fills the full 128x128 (CWS may apply its own rounded mask, so we
    keep the important artwork inside the central safe area and let the plate
    bleed to the edges). A hairline border gives the plate a boundary on a white
    store card without darkening it.
    """
    S = 128
    glyph = Image.open(
        os.path.join(STATIC, "local-operator-icon-2-light-clear.png")
    ).convert("RGBA")
    gb = glyph.split()[3].getbbox()  # (888,202,1206,550)
    g = glyph.crop(gb)  # 318x348 black glyph, tight

    # Target glyph height ~54% of the icon so it breathes inside the plate.
    target_h = int(S * 0.54)
    scale = target_h / g.height
    g = g.resize((max(1, round(g.width * scale)), target_h), Image.LANCZOS)

    icon = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    plate_mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(plate_mask).rounded_rectangle([0, 0, S - 1, S - 1], radius=28, fill=255)
    plate = Image.new("RGBA", (S, S), PLATE + (255,))
    icon.paste(plate, (0, 0), plate_mask)

    gx = (S - g.width) // 2
    gy = (S - g.height) // 2
    icon.paste(g, (gx, gy), g)

    # Hairline inner border so the plate has an edge on white store cards.
    ImageDraw.Draw(icon).rounded_rectangle(
        [0, 0, S - 1, S - 1], radius=28, outline=(226, 226, 230, 255), width=1
    )
    icon.convert("RGB").save(path)
    return path


def _icon_plate(size: int, radius: int) -> Image.Image:
    """Small rounded off-white brand chip carrying the black glyph, reused in
    the promo tiles as the product mark."""
    glyph = Image.open(
        os.path.join(STATIC, "local-operator-icon-2-light-clear.png")
    ).convert("RGBA")
    g = glyph.crop(glyph.split()[3].getbbox())
    th = int(size * 0.54)
    g = g.resize((round(g.width * th / g.height), th), Image.LANCZOS)
    chip = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    m = Image.new("L", (size, size), 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size - 1, size - 1], radius=radius, fill=255)
    chip.paste(Image.new("RGBA", (size, size), PLATE + (255,)), (0, 0), m)
    chip.paste(g, ((size - g.width) // 2, (size - g.height) // 2), g)
    return chip


# ---------------------------------------------------------------------------
# Popup / options frame preparation
# ---------------------------------------------------------------------------
def prep_popup(name: str, bg=(255, 255, 255), pad=28) -> Image.Image:
    """Load a 600x840 popup capture, crop to its real content, and re-mat it on
    clean white with even padding so it reads as a self-contained popup card."""
    im = Image.open(evid(name)).convert("RGB")
    box = content_bbox(im, bg)
    x0, y0, x1, y1 = box
    crop = im.crop((0, y0, im.width, y1))  # keep full width, trim vertical void
    card = Image.new("RGB", (crop.width, crop.height + pad * 2), WHITE)
    card.paste(crop, (0, pad))
    return card


def mat(crop: Image.Image, pad: int, bg=(255, 255, 255)) -> Image.Image:
    """Re-mat a tight crop on a flat background with even padding on all sides.

    Needed for card composites: a crop taken flush to its content, once given
    rounded corners, would clip the artwork that touches the edge (the options
    'Local Operator' heading sat exactly on the card's left rounded corner).
    Padding restores the breathing room the real page has.
    """
    out = Image.new("RGB", (crop.width + pad * 2, crop.height + pad * 2), bg)
    out.paste(crop, (pad, pad))
    return out


def build_screenshot_connected(path: str):
    """(a) Connected / driving — the hero 'agent in your browser' state."""
    W, H = 1280, 800
    cv = Image.new("RGB", (W, H), CANVAS_DARK)
    _left_caption(
        cv,
        "Your browser, with an agent in it.",
        "Paired and connected. Local Operator drives one tab and shows you what it "
        "reached — here, Page Two on your own machine.",
    )
    frame = prep_popup("popup-connected.png")
    frame = _fit(frame, max_h=560, max_w=440)
    rounded_shadow_card(cv, frame, (W - frame.width - 90, (H - frame.height) // 2))
    cv.save(path)
    return path


def build_screenshot_allow(path: str):
    """(b) Per-site allow prompt — default-deny consent."""
    W, H = 1280, 800
    cv = Image.new("RGB", (W, H), CANVAS_DARK)
    _left_caption(
        cv,
        "You decide which sites.",
        "The first time the agent wants a new site it asks. Allow once, always "
        "allow, or deny — reversible any time in Settings.",
    )
    frame = prep_popup("popup-origin-prompt.png")
    frame = _fit(frame, max_h=640, max_w=470)
    rounded_shadow_card(cv, frame, (W - frame.width - 90, (H - frame.height) // 2))
    cv.save(path)
    return path


def build_screenshot_pairing(path: str):
    """(c) Pairing — one code, no account."""
    W, H = 1280, 800
    cv = Image.new("RGB", (W, H), CANVAS_DARK)
    _left_caption(
        cv,
        "Paired to your app and nothing else.",
        "A one-time code shown in your terminal ties this browser to your copy of "
        "Local Operator. No account, no sign-up.",
    )
    frame = prep_popup("popup-pairing.png")
    frame = _fit(frame, max_h=560, max_w=440)
    rounded_shadow_card(cv, frame, (W - frame.width - 90, (H - frame.height) // 2))
    cv.save(path)
    return path


def build_screenshot_action(path: str):
    """(d) The agent driving a page — real reached page beside the popup."""
    W, H = 1280, 800
    cv = Image.new("RGB", (W, H), CANVAS_DARK)
    # Top caption band so both the page and the popup can share the lower canvas.
    d = ImageDraw.Draw(cv)
    title_f = font(46, "bold")
    sub_f = font(23, "regular")
    y = draw_caption(
        d, 90, 66, "It drives one tab — never yours.",
        "Ask in Local Operator; the agent opens its own tab, reaches the page, and "
        "reports the real result.",
        title_f, sub_f, max_w=W - 180, fill=WHITE, sub_fill=(196, 196, 204),
    )
    top = y + 24

    # The reached page (browser content) as the wide hero. The real test page
    # is deliberately sparse ("Page Two / Second page loaded."), so crop to a
    # page-top band that keeps the heading legible with balanced whitespace
    # rather than a near-empty white expanse.
    page = Image.open(evid("action-screenshot-page2.png")).convert("RGB")
    cb = content_bbox(page, (255, 255, 255))  # (16,57,280,196)-ish
    # Keep the real left margin; extend right/below for a natural page top.
    page = page.crop((0, 0, min(page.width, cb[2] + 340), min(page.height, cb[3] + 150)))
    page = _fit(page, max_h=H - top - 70, max_w=760)
    page_card = _browser_chrome(page, "127.0.0.1:8791/lop-page2.html")
    px = 90
    py = top + ((H - top - 70) - page_card.height) // 2
    rounded_shadow_card(cv, page_card, (px, py), radius=14, border=CARD_BORDER)

    # The connected popup overlapping to the right, proving the agent owns it.
    pop = prep_popup("popup-connected.png")
    pop = _fit(pop, max_h=430, max_w=360)
    rounded_shadow_card(
        cv, pop, (W - pop.width - 80, py + page_card.height - pop.height + 10)
    )
    cv.save(path)
    return path


def build_screenshot_options(path: str):
    """(e) Options / allowed sites — everything stays on your machine."""
    W, H = 1280, 800
    cv = Image.new("RGB", (W, H), CANVAS_DARK)
    _left_caption(
        cv,
        "Everything stays on your machine.",
        "The extension talks only to the local bridge on your own computer. Manage "
        "the daemon port and your allowed sites here.",
    )
    im = Image.open(evid("options-populated.png")).convert("RGB")
    box = content_bbox(im, (250, 250, 252))
    # Trim the bottom so the card ends after the first full allowed-site row
    # instead of slicing the second row mid-height, then mat on the page's own
    # off-white so the heading clears the rounded corner.
    x0, y0, x1, y1 = box
    im = im.crop((x0, y0, x1, min(y1, 1075)))
    im = mat(im, 26, bg=(250, 250, 252))
    frame = _fit(im, max_h=650, max_w=580)
    rounded_shadow_card(cv, frame, (W - frame.width - 80, (H - frame.height) // 2))
    cv.save(path)
    return path


# ---------------------------------------------------------------------------
# Shared screenshot furniture
# ---------------------------------------------------------------------------
def _fit(im: Image.Image, max_h: int, max_w: int) -> Image.Image:
    scale = min(max_h / im.height, max_w / im.width, 1.0)
    if scale < 1.0:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    return im


def _left_caption(cv: Image.Image, title: str, sub: str):
    """Left-column caption block for the standard split layout, vertically
    centred against the right-hand frame."""
    d = ImageDraw.Draw(cv)
    title_f = font(52, "bold")
    sub_f = font(25, "regular")
    # Keep the caption column clear of the right-hand frame even when a wide
    # frame (the options card) reaches toward the middle of the canvas.
    max_w = 500
    # Measure total height to centre the block.
    tlines = wrap(d, title, title_f, max_w)
    slines = wrap(d, sub, sub_f, max_w)
    ta, td = title_f.getmetrics()
    sa, sd = sub_f.getmetrics()
    th = len(tlines) * (ta + td + 12)
    sh = len(slines) * (sa + sd + 6) + 20
    total = th + sh
    y = (cv.height - total) // 2
    draw_caption(
        d, 90, y, title, sub, title_f, sub_f, max_w,
        fill=WHITE, sub_fill=(196, 196, 204),
    )


def _browser_chrome(page: Image.Image, url: str) -> Image.Image:
    """Wrap a page screenshot in a minimal browser top-bar (traffic lights + URL
    pill) so the reached page reads as 'in a real browser tab' without exposing
    any personal chrome."""
    bar = 44
    w = page.width
    out = Image.new("RGB", (w, page.height + bar), WHITE)
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, w, bar], fill=(243, 243, 245))
    d.line([0, bar, w, bar], fill=(228, 228, 232), width=1)
    for i, col in enumerate([(237, 106, 94), (245, 191, 79), (98, 197, 84)]):
        cx = 20 + i * 22
        d.ellipse([cx, bar // 2 - 6, cx + 12, bar // 2 + 6], fill=col)
    # URL pill
    pill_x0 = 96
    d.rounded_rectangle([pill_x0, 10, w - 20, bar - 10], radius=10, fill=WHITE,
                        outline=(224, 224, 228), width=1)
    uf = font(18, "regular")
    d.text((pill_x0 + 16, bar // 2 - uf.getmetrics()[0] // 2 - 2), url,
           font=uf, fill=INK_SOFT)
    out.paste(page, (0, bar))
    return out


# ---------------------------------------------------------------------------
# 3. Promo tiles
# ---------------------------------------------------------------------------
def build_small_tile(path: str):
    """440x280 small promo — mark, product name, one line."""
    W, H = 440, 280
    cv = Image.new("RGB", (W, H), CANVAS_DARK)
    chip = _icon_plate(76, radius=18)
    cx = (W - chip.width) // 2
    cv.paste(chip, (cx, 44), chip)
    d = ImageDraw.Draw(cv)
    name_f = font(30, "bold")
    line_f = font(19, "regular")
    small_f = font(15, "regular")
    name = "Local Operator"
    d.text(((W - d.textlength(name, font=name_f)) // 2, 138), name, font=name_f, fill=WHITE)
    line = "Your browser. Your logins. Your machine."
    d.text(((W - d.textlength(line, font=line_f)) // 2, 182), line, font=line_f,
           fill=(198, 198, 206))
    tag = "Free and open source"
    d.text(((W - d.textlength(tag, font=small_f)) // 2, 220), tag, font=small_f,
           fill=(140, 140, 150))
    cv.save(path)
    return path


def build_marquee(path: str):
    """1400x560 marquee — left brand column, right connected-popup composite."""
    W, H = 1400, 560
    cv = Image.new("RGB", (W, H), CANVAS_DARK)
    d = ImageDraw.Draw(cv)
    # Left ~40%
    chip = _icon_plate(84, radius=20)
    cv.paste(chip, (96, 150), chip)
    name_f = font(52, "bold")
    line_f = font(28, "regular")
    small_f = font(20, "regular")
    d.text((96, 258), "Local Operator", font=name_f, fill=WHITE)
    for i, ln in enumerate(wrap(d, "The agent works where you're already signed in.",
                                line_f, 470)):
        d.text((96, 322 + i * 40), ln, font=line_f, fill=(198, 198, 206))
    d.text((96, 322 + 2 * 40 + 16), "Free and open source", font=small_f,
           fill=(140, 140, 150))

    # Right ~60%: the reached page in browser chrome with the connected popup.
    page = Image.open(evid("action-screenshot-page2.png")).convert("RGB")
    cb = content_bbox(page, (255, 255, 255))
    page = page.crop((0, 0, min(page.width, cb[2] + 340), min(page.height, cb[3] + 150)))
    page = _fit(page, max_h=360, max_w=620)
    page_card = _browser_chrome(page, "127.0.0.1:8791/lop-page2.html")
    px = 620
    py = (H - page_card.height) // 2
    rounded_shadow_card(cv, page_card, (px, py), radius=14)

    pop = prep_popup("popup-connected.png")
    pop = _fit(pop, max_h=300, max_w=250)
    rounded_shadow_card(cv, pop, (W - pop.width - 70, py + page_card.height - pop.height + 8))
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
