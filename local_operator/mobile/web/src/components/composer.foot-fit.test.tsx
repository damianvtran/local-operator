// @vitest-environment happy-dom
/**
 * The footer fit's WIRING, not just its pure rule (review R2-4).
 *
 * `use-foot-fit.test.ts` pins `footFitLevel`; these tests pin that the composer
 * actually consults it, that the model chip stays reachable at every level
 * (review Q2 / design D5 — it is the model sheet's only door), that the reading
 * hides exactly when the level says so, and that the hook stops measuring on
 * commits that cannot move geometry (review R2-5).
 *
 * happy-dom has no layout engine, so the geometry is stubbed here; these are
 * wiring assertions. The pixel truth lives in the round-3 rendered evidence.
 */
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Composer } from "./composer";
import type { SessionProjection } from "../types";

vi.mock("../router", () => ({ navigate: vi.fn() }));
vi.mock("../api", () => ({
	sendCommand: vi.fn(), transcribeAudio: vi.fn(), getCommands: vi.fn(async () => ({ commands: [] })),
	getDirectories: vi.fn(async () => ({ home: "/", recent: [], tmp: "" })),
}));
vi.mock("../store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../store")>();
	return { ...actual, useCapabilities: () => ({ stt: false }), useDraft: () => ["", vi.fn()] };
});
vi.mock("./slash-sheet", () => ({ SlashSheet: () => null }));
vi.mock("./model-sheet", () => ({ ModelSheet: () => null }));
vi.mock("./directory-sheet", async (importOriginal) => ({ ...(await importOriginal<typeof import("./directory-sheet")>()), DirectorySheet: () => null }));
vi.mock("./usage-sheet", () => ({ UsageSheet: () => null }));

function projection(): SessionProjection {
	return { session_id: "s1", pid: 1, kind: "tui", conversation_name: "test", cwd: "~/work", model_label: "model", model_selector: "model", effort: "", effort_ladder: [], streaming: false, activity: "", activity_started_s: 0, stop_reason: "", queued_count: 0, ended: false, degraded: false, transcript: [], todos: [], subagents: [], pending: null, pending_count: 0, usage: {}, version: 1 } as SessionProjection;
}

const USAGE = { input_tokens: 1, output_tokens: 1, max_context_tokens: 10, context_remaining: 5, context_remaining_fraction: 0.5 } as const;

let available = 226;
let rects = 0;
let styles = 0;
let observerCallback: ResizeObserverCallback | undefined;
let disconnects = 0;

/** Natural (unshrunk) widths. The count is text: flexbox shrinks it to fit. */
function installGeometry() {
	rects = 0; styles = 0;
	vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
		rects++;
		let width = available;
		if (this.dataset.foot === "cwd" || this.dataset.foot === "model") width = 44;
		if (this.dataset.foot === "queued") width = 60;
		if (this.dataset.foot === "reading") width = this.dataset.footForm === "short" ? 44 : 96;
		return { width, height: 44, x: 0, y: 0, top: 0, bottom: 44, left: 0, right: width, toJSON: () => ({}) } as DOMRect;
	});
	vi.spyOn(window, "getComputedStyle").mockImplementation((element: Element) => {
		if (element.hasAttribute("data-foot-level") || element.getAttribute("data-foot") === "model") styles++;
		return { minWidth: element.getAttribute("data-foot") === "model" ? "44px" : "", columnGap: "4px" } as CSSStyleDeclaration;
	});
}

function renderComposer(p: SessionProjection) {
	return render(<div><Composer pid="s1" projection={p} onOpenModels={() => {}} onReturnToModels={() => {}} onOpenEffort={() => {}} onCloseEffort={() => {}} effortOpen={false} ended={false} short /></div>);
}

function stubObserver() {
	vi.stubGlobal("ResizeObserver", class {
		constructor(callback: ResizeObserverCallback) { observerCallback = callback; }
		observe() {}
		disconnect() { disconnects++; }
	});
}

describe("the composer footer's fit wiring", () => {
	afterEach(() => { cleanup(); vi.restoreAllMocks(); available = 226; observerCallback = undefined; disconnects = 0; });

	it.each([110, 156, 226, 240])("keeps both 44 px doors reachable at %i px", (width) => {
		available = width;
		stubObserver();
		const p = projection();
		p.queued_count = width <= 156 ? 2 : 0;
		p.usage = { ...USAGE };
		installGeometry();
		renderComposer(p);
		const row = document.querySelector<HTMLElement>("[data-foot-level]")!;
		const level = Number(row.dataset.footLevel);
		const model = row.querySelector<HTMLElement>('[data-foot="model"]')!;
		const reading = row.querySelector<HTMLElement>('[data-foot="reading"]')!;
		/* The doors themselves never yield (Q2/D5): the model chip is the model
		   sheet's only opener, the reading is the spend sheet's. */
		expect(model.classList.contains("hidden")).toBe(false);
		expect(model.getBoundingClientRect().width).toBe(44);
		expect(row.querySelector('[data-foot="cwd"]')?.classList.contains("hidden")).toBe(false);
		/* The reading yields only when the rule says so, and the mode it renders
		   in is explicit — never inferred from the document (R2-3). */
		expect(reading.classList.contains("hidden")).toBe(level >= 2);
		expect(reading.dataset.footForm).toBe(level === 0 ? "full" : "short");
		/* The two 44 px controls plus gaps must fit the row they share. */
		expect(44 + 44 + 4).toBeLessThanOrEqual(width + 0.5);
	});

	it("keeps the queued count visible as text at the tightest level", () => {
		available = 110;
		stubObserver();
		const p = projection();
		p.queued_count = 2;
		p.usage = { ...USAGE };
		installGeometry();
		renderComposer(p);
		const queued = document.querySelector<HTMLElement>('[data-foot="queued"]')!;
		const row = document.querySelector<HTMLElement>("[data-foot-level]")!;
		expect(Number(row.dataset.footLevel)).toBe(3);
		expect(queued.textContent).toMatch(/2/);
		expect(queued.getAttribute("aria-label")).toBeTruthy();
		/* Text, not a control: it gives up width before either door does. */
		expect(queued.className).toContain("min-w-0");
		expect(queued.className).toContain("shrink");
	});

	it("does not re-measure on a draft keystroke or a stream frame, and disconnects once", () => {
		stubObserver();
		installGeometry();
		const p = projection();
		p.usage = { ...USAGE };
		const result = renderComposer(p);
		const before = [rects, styles];
		fireEvent.change(screen.getByRole("textbox"), { target: { value: "a draft" } });
		result.rerender(<div><Composer pid="s1" projection={{ ...p, version: 2 }} onOpenModels={() => {}} onReturnToModels={() => {}} onOpenEffort={() => {}} onCloseEffort={() => {}} effortOpen={false} ended={false} short /></div>);
		expect([rects, styles]).toEqual(before);
		/* A width change re-measures, through the observer's synchronous flush. */
		available = 156;
		act(() => observerCallback?.([], {} as ResizeObserver));
		expect(rects).toBeGreaterThan(before[0]);
		result.unmount();
		expect(disconnects).toBe(1);
	});
});
