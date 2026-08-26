# Chrome Web Store asset inventory

This is the production brief and acceptance checklist. It identifies every
required or recommended image and what the screenshots must prove. Final
artifacts should be generated from the release build, not mocked source UI.

**Assumption:** docs/design/browser-extension.md lists the required UI states
but no final extension build or screenshots existed when this package was
written. Dimensions below follow the current CWS dashboard guidance reflected
in the task brief; recheck the upload form before export because Google can
change limits.

## Required assets

| Asset | Exact size / format | Qty | Content and production notes |
|---|---:|---:|---|
| Store icon | **128 × 128 px**, PNG | 1 | Square Local Operator icon, transparent or on a solid brand-safe background. Keep important artwork within the central ~96 × 96 px safe area; the Web Store may add its own rounded mask. Do not upload the full horizontal wordmark. |
| Screenshots | **1280 × 800 px** preferred, or **640 × 400 px**, PNG/JPEG | 1–5 (produce 5) | Use one size consistently. Capture real Chrome at 1280 × 800 where possible. No browser chrome that exposes personal profiles, bookmarks, or unrelated tabs. Each image should make one user benefit legible at listing scale. |
| Small promo tile | **440 × 280 px**, PNG/JPEG | 1 | Local Operator icon/name plus one short line: “Your browser. Your logins. Your machine.” Keep text and mark away from edges; no screenshots reduced to illegibility. |

CWS commonly treats the small promo tile as optional for basic publication but
required for some merchandising placements. Produce it now so the listing is
complete and promotion-ready.

## Optional asset

| Asset | Exact size / format | Qty | Content and production notes |
|---|---:|---:|---|
| Marquee promo tile | **1400 × 560 px**, PNG/JPEG | 1 | Wide brand composition. Show the dedicated browser tab and Local Operator terminal side-by-side, with the extension popup in the connected state. Keep focal content centered because merchandising crops vary. No claim such as “featured” or CWS badges. |

## Five-screenshot storyboard

All screenshots must be captured from the actual extension and app in the
state shown. Pairing codes, local paths, email addresses, account names, and
site content must be demo-safe. Use a clean browser profile with only the
Local Operator toolbar icon visible.

### 1. Connected and ready

**Frame:** Chrome with the Local Operator popup open in its paired,
connected, idle state. Keep enough of the normal browser visible to establish
that this is the user's real browser.

**Caption suggestion:**
> Your browser is connected. The agent is ready when you are.

**Must show:** Local Operator name, connected status, clear indication that it
uses this browser. Do not reuse the dead “Patched in” naming from the branding
exploration.

### 2. Pair this browser

**Frame:** Side-by-side composition: `lop browser pair` in a clean terminal on
the left and the extension's code-entry state on the right. Use a staged code
that is expired or generated for the shoot and cannot pair after publication.

**Caption suggestion:**
> One code pairs this browser to the app on your computer.

**Must show:** the six-digit direction is terminal to extension, a short
expiry if the final UI displays it, and no account/sign-up step.

### 3. Choose which sites

**Frame:** The popup's pending-origin prompt for a safe, recognizable demo
origin such as `example.com` or the Local Operator documentation site. Show
all three choices: Allow once, Always allow, Deny.

**Caption suggestion:**
> New site? You decide before the agent opens it.

**Must show:** the exact origin, default-deny nature, and reversible choice.
Avoid bank, health, social, or employer sites in store creative.

### 4. The agent drives one tab

**Frame:** Local Operator terminal/session on the left and the dedicated
browser tab on the right. Show a concrete request and verified result, for
example: “Open the documentation and find the install command.” The browser
should show the page reached; the terminal should show the real response.
Chrome's debugger banner should remain visible if attached.

**Caption suggestion:**
> Ask in Local Operator. Watch the result in your real browser.

**Must show:** side-by-side terminal and browser, existing-login-capable real
browser, one agent-owned tab, and a completed result rather than a plan. This
is the hero proof image.

### 5. Browser closed or bridge reconnecting

**Frame:** Popup in the disconnected/reconnecting state with its direct
recovery instruction. This completes the four states explicitly required by
the design: unpaired/code entry, paired/connected, origin prompt, and
disconnected/reconnecting.

**Caption suggestion:**
> If the app stops, the extension tells you what to do next.

**Must show:** honest disconnected status and a concrete action (`Open Local
Operator` or `Retry`, depending on final UI). Avoid red error styling that
suggests data loss or compromise.

## Icon source and crop

Source files supplied in the worktree:

- `static/local-operator-icon-2-dark-clear.png`
- `static/local-operator-icon-2-light-clear.png`

Both are **2048 × 750 RGBA PNGs**. Despite the “wordmark” description in the
task brief, direct alpha-channel inspection shows that the only nontransparent
artwork is the Local Operator icon glyph:

- nontransparent x bounds: approximately **888–1205**;
- nontransparent y bounds: approximately **204–548**;
- glyph bounds: approximately **318 × 345 px**;
- `dark` variant: white glyph, intended for a dark background;
- `light` variant: black glyph, intended for a light background.

Crop a square around that region with equal padding on each side, then resize
once from the source to 128 × 128 with a high-quality downsampler. A practical
source square is approximately `(x=875…1219, y=204…548)` (344 × 344), adjusted
by eye so the line weight and terminals feel centered. Do **not** stretch the
318 × 345 visible glyph itself to a square.

Recommended store treatment: use the black `light` glyph on the finalized
light brand background, or the white `dark` glyph on a finalized dark brand
background. **Open decision:** the design docs do not specify the extension's
store-icon background color. Decide this with the product designer before
export, and test the 128 px icon against both light and dark Web Store cards.
A transparent white icon will disappear on light surfaces; a transparent
black icon will disappear on dark surfaces, so a controlled square background
is safer.

## Promo-tile copy and layout

### 440 × 280 small tile

- Top/left or centered: square Local Operator mark.
- Product name: `Local Operator`.
- One line only: `Your browser. Your logins. Your machine.`
- Optional small qualifier: `Free and open source`.
- No button treatment, stars, rankings, user counts, Chrome logo, or claims
  that imply Google endorsement.

### 1400 × 560 marquee

- Left ~40%: product name, line `The agent works where you're already signed
  in.`, and small `Free and open source`.
- Right ~60%: clean terminal/browser side-by-side frame derived from
  screenshot 4, large enough to read.
- Preserve generous whitespace. Do not turn it into a feature list.

## Capture and export checks

- [ ] Capture the real release build in Chrome, not a design mock or test host.
- [ ] Capture all settling/connection states after motion has stopped; check a
      second consecutive frame for layout reflow.
- [ ] Use 1280 × 800 for all screenshots, or 640 × 400 for all, never a mix.
- [ ] No pairing secret that remains valid, API key, personal path, account
      avatar, email address, bookmarks bar, or unrelated browser history.
- [ ] Popup text remains readable in the uploaded 1280 × 800 composite.
- [ ] The Chrome debugger notice is visible where the extension is attached;
      do not edit out a browser safety disclosure.
- [ ] Captions do not promise unsupported multi-tab, active-tab takeover,
      downloads, uploads, Firefox, Safari, or remote access.
- [ ] PNG/JPEG is crisp at 100%; no accidental transparency in promo tiles.
- [ ] View the 128 px icon at actual size on both light and dark backgrounds.
- [ ] Keep editable source files and record fonts/colors used so future store
      updates are reproducible.

## Produced assets (inventory complete)

All shippable PNGs live in `docs/store/assets/` and are regenerated
deterministically by `docs/store/assets/build_assets.py` (Pillow, no manual
image-editor steps). Re-run with `python3 docs/store/assets/build_assets.py`
(any Python with Pillow ≥ 10 on FreeType ≥ 2.13, e.g. the repo venv — WOFF2
variable-font loading is what sets the floor).

| File | Size | Role | Source frame |
|---|---:|---|---|
| `store-icon-128.png` | 128 × 128 | Store icon | `static/local-operator-icon-2-light-clear.png` glyph on site-paper plate |
| `screenshot-1-connected.png` | 1280 × 800 | Connected / driving (hero) | `popup-connected.png` |
| `screenshot-2-allow-site.png` | 1280 × 800 | Per-site allow prompt | `popup-origin-prompt.png` |
| `screenshot-3-pairing.png` | 1280 × 800 | Pairing (no account) | `popup-pairing.png` |
| `screenshot-4-agent-driving.png` | 1280 × 800 | Agent drives one tab | `action-screenshot-page2.png` + `popup-connected.png` |
| `screenshot-5-options.png` | 1280 × 800 | Options / allowed sites | `options-populated.png` |
| `promo-small-440x280.png` | 440 × 280 | Small promo tile | brand chip + wordline |
| `promo-marquee-1400x560.png` | 1400 × 560 | Marquee promo tile | `action-screenshot-page2.png` + `popup-connected.png` |

**Production facts (for reproducible store updates):**

- Compositor: `build_assets.py`, Pillow 12.3, LANCZOS downsampling.
- Type: the local-operator.com design system's own faces, vendored as the
  site's variable WOFF2 builds in `docs/store/assets/fonts/` (see the README
  there) — Fraunces (opsz build) at weight 400 for display headlines,
  Figtree for captions, JetBrains Mono for uppercase eyebrow labels and URLs,
  with the site's letter-spacing recipe (-0.024em display, +0.08em mono
  eyebrows) applied per-glyph.
- Palette: the site `@theme` tokens verbatim — warm paper `#f7f4ee` canvas,
  ink `#211e18`, ink-muted `#565147`, ink-dim `#6c675c`, hairline-strong
  `#d5cfc2` card edges, accent green `#177b45` (eyebrow dot + label only).
  The popup captures' own background `(247,244,239)` is one RGB point off
  site paper, so the real frames sit in the same colour world by construction.
- Motif: the site's M5 dot field (1px dots, 24px pitch, hairline-strong,
  radially masked so it fades before any edge) behind the framed artefact
  only — texture concentrates behind the object, never wallpaper.
- Store-icon treatment: solid site-paper plate carrying the black glyph,
  bled square to the full 128 px (the file must be flat RGB with no alpha, so
  corner rounding is left to CWS's own surface mask), hairline-strong border
  for an edge on white cards; the light plate supplies its own contrast on
  dark cards. Verified at 128/64/32 px on white, paper, and two dark grounds.
- Captions are drawn from `listing.md`'s own section headings; none promise
  multi-tab, active-tab takeover, downloads/uploads, Firefox/Safari, or remote
  access.

**Known caveat — Chrome debugger banner (checklist §"debugger notice"):** the
supplied `action-screenshot-page2.png` capture is the page content only and
does not include Chrome's "…is debugging this browser" info bar. Screenshot 4
therefore conveys the agent-owns-this-tab fact through the popup's `Driving:
Page Two` line and the connected state rather than the browser banner. Before
final upload, re-capture the agent-driving frame with the debugger banner
visible (design §9.5 attaches CDP to the delegated tab) and drop it in as the
new source — `build_assets.py` will recomposite it unchanged.
