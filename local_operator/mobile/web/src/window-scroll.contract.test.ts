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
import { readFileSync } from "node:fs";

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
