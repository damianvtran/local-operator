// @vitest-environment happy-dom
//
// The wide-view toggle's contract (issue #1870): default OFF and byte-identical to
// the shipped meta, persisted, and fully reversible. The META PAIR (index.html and
// the login page) is pinned on the Python side, next to the page that serves it
// (tests/unit/mobile/test_daemon.py); this file pins the runtime half.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	applyWideView,
	DEFAULT_VIEWPORT_CONTENT,
	getWideView,
	initWideView,
	WIDE_VIEWPORT_CONTENT,
} from "./viewport";

function meta(): HTMLMetaElement {
	return document.querySelector('meta[name="viewport"]') as HTMLMetaElement;
}

beforeEach(() => {
	document.head.innerHTML = `<meta name="viewport" content="${DEFAULT_VIEWPORT_CONTENT}">`;
});

afterEach(() => {
	localStorage.clear();
	delete document.documentElement.dataset.view;
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
