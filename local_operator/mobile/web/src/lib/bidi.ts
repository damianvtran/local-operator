/**
 * Bidi isolation for copy painted inside a right-to-left document.
 *
 * WHY THIS EXISTS. The relay document's base direction follows the copy it
 * actually paints (Urdu copy → `dir="rtl"`). Legacy English copy still renders
 * inside that RTL paragraph — wire fallback the locale does not carry, chrome
 * that is not extracted yet, and assistant prose. Unicode bidi then re-orders
 * the NEUTRAL characters at the edges of each English run: a trailing ellipsis
 * moves to the visual start (`Message…` → `…Message`), a sentence-final period
 * starts the next line, and the leading slash of a path lands at the far end
 * (`/Users/…` → `Users/…/`). Measured in the i18n review round on the relay
 * (QA-1 Q-6 / design D4).
 *
 * The fix belongs HERE, at the render boundary, and never in the catalogues:
 * the source strings are the translators' contract and must stay byte-exact,
 * and a bidi control baked into a message would ship into every locale. These
 * helpers wrap a run in first-strong isolate (FSI/PDI) — zero-width, ignored by
 * screen readers, stripped by nothing that matters — or pick an element's
 * direction from its own first strong character.
 */

/** First-strong ranges of the scripts this product can receive RTL copy in. */
const RTL_STRONG =
	/[\u0590-\u05FF\u0600-\u06FF\u0700-\u074F\u0750-\u077F\u0780-\u07BF\u07C0-\u07FF\u0800-\u083F\u0840-\u085F\u0860-\u086F\u0870-\u089F\u08A0-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF]/;
const LTR_STRONG = /[A-Za-z\u00C0-\u02AF\u0370-\u058F\u0900-\u1FFF\u2C00-\uD7FF]/;

/**
 * The direction a rendered string should carry, from its FIRST strong
 * character. Numbers and punctuation are weak by design: a count or a separator
 * must not decide the paragraph, so they fall back to `ltr` (the English source
 * this surface paints until its own translation lands).
 */
export function textDirection(text: string): "ltr" | "rtl" {
	for (const character of text) {
		if (RTL_STRONG.test(character)) return "rtl";
		if (LTR_STRONG.test(character)) return "ltr";
	}
	return "ltr";
}

/** `text` as an isolated run: the bidi algorithm cannot re-order its edges. */
export function isolate(text: string): string {
	return `\u2068${text}\u2069`;
}

/* Paths and URLs only: the tokens whose LEADING character is neutral (`/`) and
   therefore the ones the algorithm can detach. Word-internal slashes (`and/or`,
   `3/4`) cannot match because the token must start at a boundary. */
const TECHNICAL_RUN =
	/(^|[\s([{<"'“‘])((?:https?:\/\/|~\/|\.{1,2}\/|\/)[^\s<>"'`]+)/g;

/**
 * Isolate path/URL tokens inside free prose (assistant text), leaving the rest
 * of the paragraph — and its punctuation — untouched.
 */
export function isolateTechnicalRuns(text: string): string {
	if (!text.includes("/")) return text;
	return text.replace(TECHNICAL_RUN, (_match, lead: string, token: string) => `${lead}${isolate(token)}`);
}
