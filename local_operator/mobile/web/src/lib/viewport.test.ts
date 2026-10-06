// @vitest-environment happy-dom
//
// The wide-view toggle's contract (issue #1870): default OFF and byte-identical to
// the shipped meta, persisted, and fully reversible. The META PAIR (index.html and
// the login page) is pinned on the Python side, next to the page that serves it
// (tests/unit/mobile/test_daemon.py); this file pins the runtime half.
//
// The fit-scale half (issue #2017) is pinned the same way the meta pair is: the
// VAR CONTRACT as behavior (`applyWideView` publishes/removes the property, with
// the rounding margin) and the STYLESHEET rule as source text — happy-dom does
// not resolve `max()`/`calc()` in computed styles, so a computed-style assertion
// here would be a fake. The real computed values are measured on the built
// client in the PR's capture rig.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	applyWideView,
	DEFAULT_VIEWPORT_CONTENT,
	FIT_SCALE_PROP,
	getWideView,
	initWideView,
	WIDE_VIEWPORT_CONTENT,
	wideFitScale,
} from "./viewport";

function meta(): HTMLMetaElement {
	return document.querySelector('meta[name="viewport"]') as HTMLMetaElement;
}

const ORIGINAL_SCREEN_WIDTH = window.screen.width;

function screenWidth(value: number): void {
	Object.defineProperty(window.screen, "width", { value, configurable: true });
}

beforeEach(() => {
	document.head.innerHTML = `<meta name="viewport" content="${DEFAULT_VIEWPORT_CONTENT}">`;
});

afterEach(() => {
	// Through the public path so the scale LISTENERS are detached too, not just
	// the property: a leaked listener would let one test's event reach the next.
	applyWideView(false);
	localStorage.clear();
	delete document.documentElement.dataset.view;
	document.documentElement.style.removeProperty(FIT_SCALE_PROP);
	Object.defineProperty(window.screen, "width", {
		value: ORIGINAL_SCREEN_WIDTH,
		configurable: true,
	});
});

describe("wide view", () => {
	it("ships the same meta the shell file does", () => {
		const html = readFileSync(join(process.cwd(), "index.html"), "utf8");
		expect(html.replace(/\s+/g, " ")).toContain(`content="${DEFAULT_VIEWPORT_CONTENT}"`);
	});

	it("is off by default and a default boot never touches the meta", () => {
		initWideView();
		expect(getWideView()).toBe(false);
		expect(meta().getAttribute("content")).toBe(DEFAULT_VIEWPORT_CONTENT);
		expect(document.documentElement.dataset.view).toBeUndefined();
	});

	it("widens the layout viewport but drops neither the safe-area fit nor device-width semantics by accident", () => {
		applyWideView(true);
		expect(meta().getAttribute("content")).toBe(WIDE_VIEWPORT_CONTENT);
		// No initial-scale: the engines open at the fit scale, the zoom-out floor.
		expect(WIDE_VIEWPORT_CONTENT).not.toContain("initial-scale");
		expect(WIDE_VIEWPORT_CONTENT).toContain("viewport-fit=cover");
		expect(document.documentElement.dataset.view).toBe("wide");
	});

	it("persists across a boot and restores the default exactly when turned off", () => {
		applyWideView(true);
		document.head.innerHTML = `<meta name="viewport" content="${DEFAULT_VIEWPORT_CONTENT}">`;
		delete document.documentElement.dataset.view;
		initWideView();
		expect(meta().getAttribute("content")).toBe(WIDE_VIEWPORT_CONTENT);

		applyWideView(false);
		expect(meta().getAttribute("content")).toBe(DEFAULT_VIEWPORT_CONTENT);
		expect(getWideView()).toBe(false);
		expect(document.documentElement.dataset.view).toBeUndefined();
	});
});

describe("wide view fit scale (issue #2017)", () => {
	it("publishes screen.width/512 while wide, floored so the inverse font rounds up", () => {
		screenWidth(390);
		applyWideView(true);
		// 390/512 = 0.76171875 → floored to 0.761, so `16px / var` = ~21.02px.
		expect(document.documentElement.style.getPropertyValue(FIT_SCALE_PROP)).toBe("0.761");

		screenWidth(360);
		applyWideView(true);
		// 360/512 = 0.703125 → floored to 0.703, so `16px / var` = ~22.76px.
		expect(document.documentElement.style.getPropertyValue(FIT_SCALE_PROP)).toBe("0.703");
	});

	it("clamps at 1 for views at least as wide as the layout, and falls back to 1 for degenerate widths", () => {
		for (const width of [512, 900, 0]) {
			screenWidth(width);
			expect(wideFitScale()).toBe(1);
		}
	});

	it("cleans the property when wide view is turned off", () => {
		screenWidth(390);
		applyWideView(true);
		expect(document.documentElement.style.getPropertyValue(FIT_SCALE_PROP)).toBe("0.761");

		applyWideView(false);
		expect(document.documentElement.style.getPropertyValue(FIT_SCALE_PROP)).toBe("");
	});

	it("recomputes the fit scale on orientationchange and resize while wide", () => {
		screenWidth(390);
		applyWideView(true);
		expect(document.documentElement.style.getPropertyValue(FIT_SCALE_PROP)).toBe("0.761");

		// An engine whose screen.width follows the orientation.
		screenWidth(844);
		window.dispatchEvent(new Event("orientationchange"));
		expect(document.documentElement.style.getPropertyValue(FIT_SCALE_PROP)).toBe("1");

		screenWidth(360);
		window.dispatchEvent(new Event("resize"));
		expect(document.documentElement.style.getPropertyValue(FIT_SCALE_PROP)).toBe("0.703");
	});

	it("stops listening once wide view is turned off", () => {
		screenWidth(390);
		applyWideView(true);
		applyWideView(false);
		expect(document.documentElement.style.getPropertyValue(FIT_SCALE_PROP)).toBe("");

		screenWidth(360);
		window.dispatchEvent(new Event("orientationchange"));
		window.dispatchEvent(new Event("resize"));
		expect(document.documentElement.style.getPropertyValue(FIT_SCALE_PROP)).toBe("");
	});

	it("does not stack listeners across re-applies (a leaked one would rewrite the var)", () => {
		screenWidth(390);
		applyWideView(true);
		applyWideView(true);
		applyWideView(false);

		screenWidth(360);
		window.dispatchEvent(new Event("resize"));
		expect(document.documentElement.style.getPropertyValue(FIT_SCALE_PROP)).toBe("");
	});

	it("pins the stylesheet's half of the contract as source text, like the meta pair", () => {
		const css = readFileSync(join(process.cwd(), "src/styles/index.css"), "utf8").replace(/\s+/g, " ");
		expect(css).toContain(":is(input, textarea, select) { font-size: max(16px, 1em); }");
		expect(css).toContain(
			'html[data-view="wide"] :is(input, textarea, select) { font-size: max(calc(16px / var(--lo-fit-scale)), 1em); }',
		);
	});
});
