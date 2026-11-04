// @vitest-environment happy-dom
/**
 * The footer fit's WIRING, not just its pure rule (review R2-4, R3-4).
 *
 * `use-foot-fit.test.ts` pins `footFitLevel`; these tests pin that the composer
 * actually consults it with the right inputs. Geometry comes from
 * `foot-fit-geometry-fixture.ts`, a flexbox model that derives each item's box
 * from the classes the composer renders. That is what lets these tests fail on
 * the round-3 defects: a shrinkable queued count is squeezed by the model
 * exactly as a browser squeezes it (design D8), and the reading's cached
 * widths differ between the tall and short layouts (review R3-5).
 *
 * The pixel truth lives in the round-3 rendered evidence.
 */
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Composer } from "./composer";
import { geometry, installFootGeometry, queuedWidth } from "../foot-fit-geometry-fixture";
import type { SessionProjection } from "../types";

vi.mock("../router", () => ({ navigate: vi.fn() }));
vi.mock("../api", () => ({
	sendCommand: vi.fn(), transcribeAudio: vi.fn(), getCommands: vi.fn(async () => ({ commands: [] })),
	getSkills: vi.fn(async () => ({ skills: [] })),
	getDirectories: vi.fn(async () => ({ home: "/", recent: [], tmp: "" })),
}));
let capabilities: { stt: { available: boolean; path: string | null } | false } = { stt: false };
vi.mock("../store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../store")>();
	return { ...actual, useCapabilities: () => capabilities, useDraft: () => ["", vi.fn()] };
});
vi.mock("./slash-sheet", () => ({ SlashSheet: () => null }));
vi.mock("./model-sheet", () => ({ ModelSheet: () => null }));
vi.mock("./directory-sheet", async (importOriginal) => ({ ...(await importOriginal<typeof import("./directory-sheet")>()), DirectorySheet: () => null }));
vi.mock("./usage-sheet", () => ({ UsageSheet: () => null }));

function projection(): SessionProjection {
	return { session_id: "s1", pid: 1, kind: "tui", conversation_name: "test", cwd: "~/work", model_label: "Claude Opus 5.5", model_selector: "model", effort: "high", effort_ladder: [], streaming: false, activity: "", activity_started_s: 0, stop_reason: "", queued_count: 0, ended: false, degraded: false, transcript: [], todos: [], subagents: [], pending: null, pending_count: 0, usage: {}, version: 1, context_tokens: 61_800, context_window: 200_000, cumulative_parent_cost: 2.47 } as SessionProjection;
}

let observerCallback: ResizeObserverCallback | undefined;
let disconnects = 0;

function stubObserver() {
	vi.stubGlobal("ResizeObserver", class {
		constructor(callback: ResizeObserverCallback) { observerCallback = callback; }
		observe() {}
		disconnect() { disconnects++; }
	});
}

function composer(p: SessionProjection, short: boolean) {
	return <div data-short={short ? "" : undefined}><Composer pid="s1" projection={p} onOpenModels={() => {}} onReturnToModels={() => {}} onOpenEffort={() => {}} onCloseEffort={() => {}} effortOpen={false} ended={false} short={short} /></div>;
}

const row = () => document.querySelector<HTMLElement>("[data-foot-level]")!;
const level = () => Number(row().dataset.footLevel);
/** A real resize: the observer fires and the hook re-measures in that flush. */
const resize = (width: number) => { geometry.available = width; act(() => observerCallback?.([], {} as ResizeObserver)); };

describe("the composer footer's fit wiring", () => {
	beforeEach(() => { stubObserver(); installFootGeometry(); });
	afterEach(() => {
		cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals();
		geometry.available = 110; geometry.tallAvailable = 304; observerCallback = undefined; disconnects = 0; capabilities = { stt: false };
	});

	it.each([110, 156, 226, 240])("keeps the model chip a named 44 px door at %i px", (width) => {
		geometry.available = width;
		const p = projection();
		p.queued_count = width <= 156 ? 2 : 0;
		render(composer(p, true));
		const model = row().querySelector<HTMLElement>('[data-foot="model"]')!;
		/* The chip is the model sheet's only door (Q2/D5): never hidden, and its
		   accessible name is the model at every level, the glyph-free L3 too. */
		expect(model.classList.contains("hidden")).toBe(false);
		expect(model.getAttribute("aria-label")).toBe("Model: Claude Opus 5.5 · high");
		/* Its floor is the 44 px target class, not a stubbed width (R3-4 iv). */
		expect(model.className).toMatch(/\bmin-w-11\b/);
		expect(model.className).toMatch(/\bmin-h-11\b/);
		const reading = row().querySelector<HTMLElement>('[data-foot="reading"]')!;
		expect(reading.classList.contains("hidden")).toBe(level() >= 2);
		expect(reading.dataset.footForm).toBe(level() === 0 ? "full" : "short");
	});

	it("pins the model chip at 44 px with the label's first characters at L3 (D10)", () => {
		geometry.available = 80;
		render(composer(projection(), true));
		expect(level()).toBe(3);
		const model = row().querySelector<HTMLElement>('[data-foot="model"]')!;
		/* `w-11 shrink-0`: exactly the 44 px floor; a `shrink-0` chip without the
		   width would take the label's whole width. No bare glyph. */
		expect(model.className).toMatch(/\bw-11\b/);
		expect(model.className).toMatch(/\bshrink-0\b/);
		expect(model.textContent).toBe("Claude Opus 5.5 · high");
		expect(model.querySelector(".truncate")).not.toBeNull();
		expect(model.textContent).not.toContain("◇");
		expect(model.getAttribute("aria-label")).toBe("Model: Claude Opus 5.5 · high");
	});

	it("never paints the queued count as a fragment, at any width (D8)", () => {
		for (const count of [2, 12]) {
			const whole = `${count} queued`;
			for (let width = 92; width <= 260; width += 2) {
				geometry.available = width;
				const p = projection();
				p.queued_count = count;
				const view = render(composer(p, true));
				const queued = row().querySelector<HTMLElement>('[data-foot="queued"]');
				if (level() === 4) {
					/* Moved out whole: still announced with the composer. */
					expect(queued).toBeNull();
					expect(document.querySelector("[data-queued-moved]")?.textContent).toBe(whole);
				} else {
					expect(queued?.textContent).toBe(whole);
					expect(queued?.className).toMatch(/\bshrink-0\b/);
					expect(queued?.getBoundingClientRect().width).toBe(queuedWidth(whole));
				}
				view.unmount();
			}
		}
	});

	it("moves the count out only when the last level cannot seat it whole", () => {
		const whole = queuedWidth("2 queued");
		const p = projection();
		p.queued_count = 2;
		/* Folder + count + model floor + two gaps: the smallest row that keeps it. */
		const seat = 44 + whole + 44 + 8;
		geometry.available = seat;
		const view = render(composer(p, true));
		expect(level()).toBe(2);
		view.unmount();
		geometry.available = seat - 1;
		render(composer(p, true));
		expect(level()).toBe(4);
	});

	it.each([
		["two widths", [200, 110, 150]],
		["a narrower trip", [150, 92, 150]],
		["a cold mount", [150]],
	] as const)("reaches the same level at one width whatever the history: %s (R3-1)", (_name, path) => {
		const p = projection();
		p.queued_count = 2;
		geometry.available = path[0];
		render(composer(p, true));
		for (const width of path.slice(1)) resize(width);
		const after = level();
		const shown = row().querySelector('[data-foot="queued"]')?.textContent ?? null;
		cleanup();
		geometry.available = 150;
		render(composer(p, true));
		expect(after).toBe(level());
		expect(shown).toBe(row().querySelector('[data-foot="queued"]')?.textContent ?? null);
	});

	it("gives a first open and a re-open the same level (R3-5)", () => {
		/* 360 + mic, no queue: the round-3 probe went L1, L0, L2 because the tall
		   layout's 96 px compact slot was cached as the short layout's width. */
		const p = projection();
		geometry.available = 150;
		const view = render(composer(p, true));
		const firstOpen = level();
		view.rerender(composer(p, false));
		act(() => observerCallback?.([], {} as ResizeObserver));
		view.rerender(composer(p, true));
		act(() => observerCallback?.([], {} as ResizeObserver));
		expect(level()).toBe(firstOpen);
		expect(firstOpen).toBe(1);
	});

	it("re-fits when the mic arrives, without waiting for the row's observer (Q2)", () => {
		vi.stubGlobal("isSecureContext", true);
		vi.stubGlobal("MediaRecorder", class {});
		Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: { getUserMedia: vi.fn() } });
		const p = projection();
		geometry.available = 150;
		const view = render(composer(p, true));
		expect(level()).toBe(1);
		/* The mic takes 46 px from the line. The observer is not fired: the
		   commit that adds the mic must itself re-fit. */
		capabilities = { stt: { available: true, path: "/stt" } };
		geometry.available = 104;
		try {
			view.rerender(composer({ ...p }, true));
			expect(screen.getByRole("button", { name: "start voice input" })).toBeTruthy();
			expect(level()).toBe(2);
		} finally {
			delete (navigator as { mediaDevices?: unknown }).mediaDevices;
		}
	});

	it("re-fits when the reading's text grows with usage unchanged (R3-6)", () => {
		const p = projection();
		geometry.available = 190;
		const view = render(composer(p, true));
		expect(level()).toBe(0);
		/* Spend grows the reading (88.5 -> 102 px) past the row while `usage`
		   does not move and the row itself keeps its width, so no observer fires:
		   the commit carrying the new text must re-fit on its own. */
		view.rerender(composer({ ...p, cumulative_parent_cost: 123.47 }, true));
		expect(row().querySelector('[data-foot="reading"]')?.getAttribute("aria-label")).toMatch(/\$123\.47/);
		expect(level()).toBe(1);
	});

	it("re-fits inside the observer's own callback, outside act (R3-4 ii)", () => {
		/* The row's ResizeObserver is not a React event: without `flushSync` its
		   update commits in a later task, after the frame that resized the row. */
		const p = projection();
		geometry.available = 226;
		render(composer(p, true));
		expect(level()).toBe(0);
		const flag = globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean };
		const previous = flag.IS_REACT_ACT_ENVIRONMENT;
		flag.IS_REACT_ACT_ENVIRONMENT = false;
		try {
			geometry.available = 110;
			observerCallback?.([], {} as ResizeObserver);
			expect(level()).toBe(2);
		} finally {
			flag.IS_REACT_ACT_ENVIRONMENT = previous;
		}
	});

	it("does not re-measure on a draft keystroke or a stream frame, and disconnects once", () => {
		const rects = vi.mocked(HTMLElement.prototype.getBoundingClientRect);
		const p = projection();
		geometry.available = 226;
		const result = render(composer(p, true));
		const before = rects.mock.calls.length;
		fireEvent.change(screen.getByRole("textbox"), { target: { value: "a draft" } });
		result.rerender(composer({ ...p, version: 2 }, true));
		expect(rects.mock.calls.length).toBe(before);
		/* A width change re-measures, through the observer's synchronous flush. */
		resize(156);
		expect(rects.mock.calls.length).toBeGreaterThan(before);
		result.unmount();
		expect(disconnects).toBe(1);
	});
});
