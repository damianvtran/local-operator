// @vitest-environment happy-dom
//
// THE WINDOW IS NOT A SCROLLER — the "two scroll bars" contract.
//
// The operator's report: in a conversation the transcript scrolls AND the whole
// window scrolls, with a scroll indicator that has nowhere to go; a drag that
// starts on the composer makes the screen slide and bounce. The fix is a
// document-level contract, so this file asserts the document-level declaration
// rather than a component's class string.
//
// WHY THE STYLESHEET IS READ AND INJECTED. The vitest layer renders without
// `styles/index.css`, so `getComputedStyle` on a Tailwind class is `""` —
// `session-view.overflow.test.tsx` records that lesson. Importing the file as
// text and mounting its rules in happy-dom makes the assertion read the REAL
// declaration: delete `overflow: hidden` from the base layer and this fails.
//
// WHAT IT CANNOT PROVE, and which layer does. happy-dom has no layout and no
// touch, so this says the document is DECLARED unscrollable, not that an iPhone
// stops rubber-banding. That geometric half was measured on the real bundle in
// headless Chrome with real touch drags, before and after the change (frames
// and numbers on the PR, recorded by a session-local probe that reuses
// `scripts/mobile_overflow_capture.py`), and the iOS-specific elastic bounce is
// unverified on a device — Chrome has one viewport, iOS has a layout viewport
// and a pannable visual one.
import { afterAll, describe, expect, it } from "vitest";
// Read from disk rather than imported: vitest stubs CSS imports (even `?raw`),
// and the stylesheet this file is about must be the real one.
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

// vitest runs with the package root as cwd, so this is the same source the
// bundle is built from.
const cssSource = readFileSync("src/styles/index.css", "utf8");

/** The body of `@layer base { … }`, with the layer wrapper removed.
 *
 *  happy-dom's CSS parser does not descend into `@layer`, so mounting the file
 *  as-is resolves nothing and the test would pass on an empty page. The
 *  wrapper is stripped by brace matching rather than a regex because the layer
 *  contains a nested `@media` block, which a non-greedy `}` match truncates
 *  before the rules this file is about. */
function baseLayer(css: string): string {
	const start = css.indexOf("@layer base");
	if (start < 0) throw new Error("styles/index.css has no @layer base block");
	const open = css.indexOf("{", start);
	let depth = 0;
	for (let i = open; i < css.length; i += 1) {
		if (css[i] === "{") depth += 1;
		else if (css[i] === "}") {
			depth -= 1;
			if (depth === 0) return css.slice(open + 1, i);
		}
	}
	throw new Error("styles/index.css has an unbalanced @layer base block");
}

const style = document.createElement("style");
style.textContent = baseLayer(cssSource);
document.head.appendChild(style);

afterAll(() => style.remove());

describe("document scroll contract", () => {
	it("pins the document to the viewport and forbids it from scrolling", () => {
		const html = getComputedStyle(document.documentElement);
		expect(html.height).toBe("100%");
		expect(html.overflow).toBe("hidden");
		expect(html.overscrollBehavior).toBe("none");
	});

	it("bounds the body instead of letting it grow past the viewport", () => {
		// `min-height: 100%` was the shipped rule: a floor, so the body grew with
		// the pinned column's `top` offset and the extra height became document
		// scroll. `height` is the bounded box, and that is the difference this
		// assertion exists for.
		const body = getComputedStyle(document.body);
		expect(body.height).toBe("100%");
		expect(body.minHeight).toBe("");
		expect(body.overflow).toBe("hidden");
		expect(body.overscrollBehavior).toBe("none");
	});
});

// ---------------------------------------------------------------------------
// THE OTHER HALF OF THE SAME CONTRACT: A PINNED DOCUMENT CANNOT CARRY A SCREEN.
//
// `html`/`body { overflow: hidden }` removes the window scroller, so a screen
// whose root is a height FLOOR (a `min-height` of the viewport — a minimum, not
// a bound) grows past the viewport with nothing that scrolls. Every row past the
// fold becomes unreachable, and no rendered test notices: the DOM is complete,
// only the pixels are missing.
//
// That shipped on `#/past`: 22 rows put the last one 458px below a 390x844
// screen. The screen was fixed, but the NEXT one to be written the same way
// would fail the same way, and the reviewer's audit of the other screens is
// exactly the kind of check that rots. So the shape is refused here, at the
// source, the way `no-sideways-scroll.test.ts` refuses a horizontal rail.
//
// The rule is per FILE, not per element: a screen that declares a height floor
// must declare a scroller somewhere, because the only thing that can scroll a
// column the window no longer scrolls is one the screen owns. The `dvh` and
// `screen` spellings of that floor are included — the same floor under another
// name — and `overflow-x-*` deliberately does NOT satisfy it: a sideways
// scroller answers a different question.
const SCREENS = "src/screens";
const HEIGHT_FLOOR = /^min-h-(?:full|dvh|screen)$/;
const OWN_SCROLLER = /^overflow-(?:y-)?(?:auto|scroll|hidden)$/;

export interface DocumentScrollOffender {
	file: string;
	token: string;
}

/** Every class-list token a source file declares through the CLASS ATTRIBUTE.
 *
 *  Built from the `class` sites rather than the whole file text so prose in a
 *  comment may NAME a class (this file's own note does) without tripping the
 *  guard, and so the braced form — a template expression around a helper call —
 *  is read through its string literals instead of being missed.
 *
 *  NOTE FOR WHOEVER EDITS THIS COMMENT: `scripts/check-bundle.mjs` scans every
 *  `.ts`/`.tsx` under `src` for the class attribute, tests included, and reads a
 *  quote straight after the equals sign as the start of a class literal. A
 *  backtick there opened a template literal that swallowed this paragraph as
 *  fifty class names and failed `pnpm build` — so the attribute is spelled out
 *  in prose as the CLASS ATTRIBUTE, never as attribute-equals-quote. */
function classTokens(source: string): string[] {
	const tokens: string[] = [];
	for (const site of source.matchAll(/className\s*=\s*(?:"([^"]*)"|\{([^}]*)\})/g)) {
		if (site[1] !== undefined) {
			tokens.push(...site[1].split(/\s+/));
			continue;
		}
		for (const quoted of (site[2] ?? "").matchAll(/"([^"]*)"/g)) {
			tokens.push(...quoted[1].split(/\s+/));
		}
	}
	return tokens.filter(Boolean);
}

/** A class name assembled from its segments.
 *
 *  WHY IT IS NOT SPELLED OUT. `scripts/check-bundle.mjs` treats every
 *  class-shaped literal anywhere under `src/` — tests and comments included — as
 *  a class the app renders with, and Tailwind then emits a rule for it. A guard
 *  has to NAME the class it refuses, so writing it verbatim added four selectors
 *  to the shipped stylesheet that nothing renders with (two of them here, two in
 *  the render test beside this contract). Assembled at runtime it is the same
 *  string and invisible to both scanners. */
function classOf(...segments: string[]): string {
	return segments.join("-");
}

const FLOOR = classOf("min", "h", "full");
const FLOOR_DVH = classOf("min", "h", "dvh");
const SIDEWAYS = classOf("overflow", "x", "auto");

/** The screens that declare a height floor and no scroller of their own. */
export function documentScrollers(sources: Array<{ file: string; source: string }>): DocumentScrollOffender[] {
	return sources.flatMap(({ file, source }) => {
		const tokens = classTokens(source);
		const floor = tokens.find((token) => HEIGHT_FLOOR.test(token));
		if (floor === undefined) return [];
		if (tokens.some((token) => OWN_SCROLLER.test(token))) return [];
		return [{ file, token: floor }];
	});
}

describe("a pinned window leaves every screen its own scroller", () => {
	it("no routed screen declares a height floor without one", () => {
		const files = readdirSync(SCREENS)
			.filter((name) => name.endsWith(".tsx") && !name.endsWith(".test.tsx"))
			.map((name) => ({ file: `${SCREENS}/${name}`, source: readFileSync(join(SCREENS, name), "utf8") }));
		expect(files.length).toBeGreaterThan(5);
		expect(
			documentScrollers(files),
			"The window cannot scroll, so a screen whose root is a height FLOOR " +
				"(a min-height, not a bound) needs a bounded column (`h-dvh` + `overflow-hidden`) and " +
				"its own `overflow-y-auto` scroller — otherwise everything past the fold is " +
				"unreachable. See `#/past` (the 22-row report) and `screens/library.tsx` " +
				"for the shape.",
		).toEqual([]);
	});

	it("catches the shape that shipped on #/past (negative control)", () => {
		/* The exact pre-fix root of `screens/past-sessions.tsx`. Without this the
		   test above could pass forever on a scanner that matches nothing. */
		const shipped =
			`<div className="mx-auto flex ${FLOOR} w-full max-w-[var(--lo-column-max,28rem)] flex-col">` +
			`<main className="flex flex-1 flex-col gap-0.5 px-2 pb-3">`;
		expect(documentScrollers([{ file: "src/screens/past-sessions.tsx", source: shipped }])).toEqual([
			{ file: "src/screens/past-sessions.tsx", token: FLOOR },
		]);

		// ...and the shape it was fixed to passes, so the guard is not simply
		// "refuse everything".
		const fixed =
			`<div className="mx-auto flex h-dvh w-full max-w-[var(--lo-column-max,28rem)] flex-col overflow-hidden">` +
			`<main className="lo-scroll flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto px-2 pb-3">`;
		expect(documentScrollers([{ file: "src/screens/past-sessions.tsx", source: fixed }])).toEqual([]);

		// A sideways scroller is not an answer: it leaves the vertical overflow
		// clipped where it was.
		expect(
			documentScrollers([
				{
					file: "src/screens/sideways.tsx",
					source: `<div className="flex ${FLOOR_DVH} ${SIDEWAYS}">`,
				},
			]),
		).toEqual([{ file: "src/screens/sideways.tsx", token: FLOOR_DVH }]);
	});
});
