/**
 * The phone's reading width: a persisted, default-OFF "wide view" (issue #1870).
 *
 * WHY A TOGGLE AND NOT A META EDIT. The request was that pinching out should be
 * able to give a wider reading column. No STATIC viewport meta can do that while
 * leaving the default experience alone, because both engines clamp the zoom-out
 * floor UP to "the layout fits the screen":
 *
 *  - WebKit (`ViewportConfiguration::minimumScale`): when the content is narrower
 *    than the view at the configured minimum scale, the minimum becomes
 *    `viewWidth / contentWidth`. With `width=device-width` the content IS the view
 *    width, so the floor is 1 and `minimum-scale<1` is inert — the engine snaps it
 *    back.
 *  - Blink (`PageScaleConstraints::FitToContentsWidth`): the same clamp,
 *    `minimum_scale = max(minimum_scale, viewWidth / contentsWidth)`.
 *
 * A wider static `width=` has the other two failure modes: with `initial-scale=1`
 * the page opens panned (a 390px window onto a wider layout), and without it the
 * default text shrinks to the fit scale. Either changes everyone's default. So the
 * layout viewport widens only when the reader asks, by rewriting the meta at
 * runtime — which both engines honour (they re-run viewport configuration on a
 * meta `content` change) — and the default meta, in `index.html` and in the
 * server-rendered login page, stays exactly {@link DEFAULT_VIEWPORT_CONTENT}.
 *
 * WHAT STAYS TRUE IN WIDE MODE, because it is why the default meta is what it is:
 *  - the keyboard pin (`--lo-vvh`, `session-view.tsx`) reads `visualViewport.height`,
 *    which is in CSS px at any page scale, so it needs no scale term;
 *  - THE 16px FLOOR NEEDED A SCALE TERM (issue #2017): WebKit's focus-zoom
 *    (`_zoomToFocusRect:`) targets `16 / font-size` as an ABSOLUTE page scale and
 *    moves the page there when it differs from the current one, so in wide view
 *    — where the page sits at the fit scale, ~0.76 on a 390pt phone — a 16px CSS
 *    font still zooms ~1.31x on focus. `applyWideView` therefore publishes
 *    {@link FIT_SCALE_PROP}, and `styles/index.css` divides the field floor by
 *    it, so every field's target scale IS the scale the page already sits at.
 * The compensation derives from WebKit's rule (sources in issue #2017) and the
 * Chromium measurements are in the PR; it is NOT verified on a real iPhone —
 * none was available — so the device leg is handed to QA.
 */

/** The meta every page ships with. `index.html` and the login page in
 *  `local_operator/mobile/daemon.py` must carry exactly this; a test per side
 *  pins the pair so the copies cannot drift apart. */
export const DEFAULT_VIEWPORT_CONTENT = "width=device-width, initial-scale=1, viewport-fit=cover";

/** THE FORCED LEGIBILITY TRADE-OFF, stated where a reader of this file will find it.
 *
 * Widening the layout shrinks the default text, and no choice of width avoids it:
 * the engines open at the fit scale, so a 390pt phone lays out 512 CSS px at
 * `scale = 390/512 = 0.762` and the 13px description text lands at ~9.9 PHYSICAL
 * px (360pt: 0.703, ~9.1px; the 44px targets at 33.5 / 30.9px). Keeping 13px at
 * >= 12 physical px at 360pt needs `width <= 390` — i.e. no widening at all, which
 * is why this is a trade rather than a bug. It is opt-in, the default path is
 * untouched, and pinching IN restores the scale. Widening the transcript COLUMN
 * without shrinking the text is not something a viewport meta can do.
 */
/** The layout width, in CSS px, wide view asks for.
 *
 *  512 is the smallest round width that gives a visibly wider column (about 1.3x
 *  a 390pt phone's 390) while keeping the shrink-to-fit scale (`screen / 512`)
 *  high enough that 14px body text still lands near 10px on a 360pt phone
 *  (360/512 x 14 = 9.8px) — pinching IN from there is always available. A wider
 *  value buys columns the transcript's 85% bubbles cannot use and costs legibility
 *  linearly. No `initial-scale` is sent: the engines then open at the fit scale,
 *  which is the zoom-out floor this mode exists to reach. */
export const WIDE_VIEWPORT_WIDTH = 512;

export const WIDE_VIEWPORT_CONTENT = `width=${WIDE_VIEWPORT_WIDTH}, viewport-fit=cover`;

/** The root CSS property `styles/index.css` divides the 16px field floor by in
 *  wide view (issue #2017). A NUMBER so CSS `calc()` can divide by it. */
export const FIT_SCALE_PROP = "--lo-fit-scale";

/** The wide layout's fit scale as a 0..1 number: `screen.width / 512`, floored
 *  to three decimals and clamped at 1.
 *
 * THE RATIO ONLY EXISTS IN JS, which is why this is not a CSS `calc()`. The
 * field rule needs the INVERSE (`16px / scale`), and inside a `width=512`
 * layout every viewport unit reads 512 — no CSS function can observe the
 * physical screen width the engines use to pick the fit scale. A pure function
 * of `window.screen.width` so the clamps below are testable.
 *
 * CLAMPED AT 1 because a view as wide as (or wider than) the 512 layout is not
 * shrunk — the engines open it at scale 1 — and the fields must not shrink
 * below the 16px floor there either.
 *
 * FLOORED TO THREE DECIMALS so the inverse font size ROUNDS UP: WebKit rounds
 * the visible scale in its focus-zoom comparison, and a font at or under the
 * exact inverse can put its target scale on the wrong side of the current one
 * (a zoom-in on focus). The margin costs at most ~0.02 CSS px of field font.
 *
 * Degenerate widths (absent, zero, non-finite) fall back to 1 — the same 16px
 * the fields get in every other mode. */
export function wideFitScale(): number {
	const width = Number(window.screen?.width);
	if (!Number.isFinite(width) || width <= 0) return 1;
	const raw = width / WIDE_VIEWPORT_WIDTH;
	return Math.min(1, Math.max(0.1, Math.floor(raw * 1000) / 1000));
}

const KEY = "lo-mobile-wide-view";

/** Not content-bearing, so it deliberately survives sign-out like the theme. */
export function getWideView(): boolean {
	return localStorage.getItem(KEY) === "1";
}

function viewportMeta(): HTMLMetaElement | null {
	return document.querySelector<HTMLMetaElement>('meta[name="viewport"]');
}

/** Apply (and persist) the mode: the meta, plus the `data-view` attribute the
 *  column cap keys off (`--lo-column-max` in `styles/index.css`). */
export function applyWideView(wide: boolean): void {
	const meta = viewportMeta();
	if (meta) meta.setAttribute("content", wide ? WIDE_VIEWPORT_CONTENT : DEFAULT_VIEWPORT_CONTENT);
	if (wide) {
		document.documentElement.dataset.view = "wide";
		document.documentElement.style.setProperty(FIT_SCALE_PROP, String(wideFitScale()));
		localStorage.setItem(KEY, "1");
	} else {
		delete document.documentElement.dataset.view;
		document.documentElement.style.removeProperty(FIT_SCALE_PROP);
		localStorage.removeItem(KEY);
	}
}

/** Boot: restore the persisted mode before first paint, like `initTheme`. A
 *  default-off reader never touches the meta at all. */
export function initWideView(): void {
	if (getWideView()) applyWideView(true);
}
