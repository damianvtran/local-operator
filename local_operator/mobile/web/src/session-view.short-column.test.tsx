// @vitest-environment happy-dom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useLayoutEffect, useRef, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SessionScreen } from "./screens/session-view";
import { geometry, installFootGeometry } from "./foot-fit-geometry-fixture";
import type { SessionProjection } from "./types";

let slot: { projection: SessionProjection | null; connected: boolean } = { projection: null, connected: true };
vi.mock("./router", () => ({ navigate: vi.fn() }));
vi.mock("./api", async () => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentDetail: vi.fn(async () => null),
	imageUrl: vi.fn(() => ""),
	getCommands: vi.fn(async () => ({ commands: [] })),
	getSkills: vi.fn(async () => ({ skills: [] })),
	getModels: vi.fn(async () => ({ models: [] })),
	getDirectories: vi.fn(async () => ({ home: "/", recent: [], tmp: "" })),
	changeDirectory: vi.fn(async () => ({ ok: true })),
	sendCommand: vi.fn(async () => ({ ok: true })),
	markSessionSeen: vi.fn(async () => ({ ok: true })),
	setSessionPin: vi.fn(async () => ({ ok: true })),
	resumeSession: vi.fn(async () => ({ ok: true })),
}));
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return { ...actual, useProjection: vi.fn(() => slot), retainProjectionStream: vi.fn(() => () => {}), retainSessionListStream: vi.fn(() => () => {}), useDraft: vi.fn(() => ["", () => {}]) };
});
function projection(): SessionProjection {
	return { session_id: "s1", pid: 42, kind: "tui", conversation_name: "test", cwd: "", model_label: "", model_selector: "", effort: "", effort_ladder: [], streaming: false, activity: "", activity_started_s: 0, stop_reason: "", queued_count: 0, ended: false, degraded: false, transcript: [], todos: [], subagents: [], pending: null, pending_count: 0, usage: {}, version: 1 } as SessionProjection;
}

describe("the short layout follows the VISUAL viewport (review R1-1)", () => {
	afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
	it.each([440, 480])("sets data-short on the first commit at %i", (height) => {
		const vv = new EventTarget() as EventTarget & { height: number; offsetTop: number };
		vv.height = height; vv.offsetTop = 0;
		vi.stubGlobal("visualViewport", vv);
		slot = { projection: projection(), connected: true };
		const view = render(<SessionScreen sessionId="s1" />);
		expect(view.container.firstElementChild?.hasAttribute("data-short")).toBe(true);
	});
	it("stays in the full layout at 481", () => {
		const vv = new EventTarget() as EventTarget & { height: number; offsetTop: number };
		vv.height = 481; vv.offsetTop = 0;
		vi.stubGlobal("visualViewport", vv);
		slot = { projection: projection(), connected: true };
		const view = render(<SessionScreen sessionId="s1" />);
		expect(view.container.firstElementChild?.hasAttribute("data-short")).toBe(false);
	});
	/* THE FIRST COMMIT, READ BEFORE ANY EFFECT CAN RUN (review R3-4 i = QA Q4).
	   `flushSync(root.render)` and `render()` both flush a sync render's passive
	   effects before returning, so the round-2 tests also passed against a seed
	   of `false` corrected by an effect. A PARENT's layout effect runs after its
	   children's DOM is attached but before any effect state they set has
	   re-rendered — exactly the DOM the first paint would show — so this probe
	   records `data-short` there. Without a projection too (QA Q1): the
	   reserved frame's root must carry it, or a cold short load paints the
	   154 px two-line composer and then jumps 44 px. */
	function FirstCommit({ children, seen }: { children: ReactNode; seen: (column: Element | null) => void }) {
		const ref = useRef<HTMLDivElement>(null);
		useLayoutEffect(() => { seen(ref.current?.firstElementChild ?? null); }, [seen]);
		return <div ref={ref}>{children}</div>;
	}
	it.each([["with", true], ["without", false]] as const)("decides short inside the first commit %s a projection", (_name, withProjection) => {
		const vv = new EventTarget() as EventTarget & { height: number; offsetTop: number };
		vv.height = 440; vv.offsetTop = 0;
		vi.stubGlobal("visualViewport", vv);
		slot = { projection: withProjection ? projection() : null, connected: true };
		let first: boolean | null = null;
		let composer: boolean | null = null;
		render(<FirstCommit seen={(column) => {
			first = column?.hasAttribute("data-short") ?? null;
			composer = column?.querySelector(".lo-composer")?.closest("[data-short]") !== null;
		}}><SessionScreen sessionId="s1" /></FirstCommit>);
		expect(first).toBe(true);
		expect(composer).toBe(true);
	});
	it("commits a native keyboard resize synchronously, outside act (R3-4 ii)", () => {
		/* A real `visualViewport` resize is not a React event, so without
		   `flushSync` the flip would commit in a later task: CSS and the fit
		   would not land in the frame the keyboard moved. Dispatched OUTSIDE
		   act, which would otherwise flush for us, and read straight after. */
		const vv = new EventTarget() as EventTarget & { height: number; offsetTop: number };
		vv.height = 700; vv.offsetTop = 0;
		vi.stubGlobal("visualViewport", vv);
		slot = { projection: projection(), connected: true };
		const container = document.createElement("div");
		document.body.appendChild(container);
		const root = createRoot(container);
		act(() => root.render(<SessionScreen sessionId="s1" />));
		const column = container.firstElementChild as HTMLElement;
		const previous = (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT;
		(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = false;
		try {
			vv.height = 440; vv.dispatchEvent(new Event("resize"));
			expect(column.hasAttribute("data-short")).toBe(true);
			expect(container.querySelector("[data-foot-level]")?.hasAttribute("data-short")).toBe(true);
			vv.height = 700; vv.dispatchEvent(new Event("resize"));
			expect(column.hasAttribute("data-short")).toBe(false);
		} finally {
			(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = previous;
			act(() => root.unmount());
			container.remove();
		}
	});
	it("sets data-short from visualViewport.height even when innerHeight is large", () => {
		const vv = new EventTarget() as EventTarget & { height: number; offsetTop: number };
		vv.height = 844; vv.offsetTop = 0;
		vi.stubGlobal("visualViewport", vv);
		Object.defineProperty(window, "innerHeight", { configurable: true, value: 844 });
		slot = { projection: projection(), connected: true };
		const view = render(<SessionScreen sessionId="s1" />);
		const column = view.container.firstElementChild as HTMLElement;
		expect(column.hasAttribute("data-short")).toBe(false);
		act(() => { vv.height = 440; vv.dispatchEvent(new Event("resize")); });
		expect(column.hasAttribute("data-short")).toBe(true);
		expect(view.container.querySelector("[data-foot-level]")?.hasAttribute("data-short")).toBe(true);
		expect(window.innerHeight).toBe(844);
		act(() => { vv.height = 700; vv.dispatchEvent(new Event("resize")); });
		expect(column.hasAttribute("data-short")).toBe(false);
	});
});

/* The ⋮ menu's moved-out rows, through the REAL screen (review R3-4 iii): the
   gate lives in `SessionScreen`'s header, so testing `SessionMenu`'s props
   alone left the wiring unpinned. Geometry: the flexbox model, 320 + mic. */
describe("the session menu carries what the footer moved out (D8/D9)", () => {
	afterEach(() => { cleanup(); vi.restoreAllMocks(); vi.unstubAllGlobals(); geometry.available = 110; geometry.tallAvailable = 304; });
	function open(height: number, p: SessionProjection) {
		const vv = new EventTarget() as EventTarget & { height: number; offsetTop: number };
		vv.height = height; vv.offsetTop = 0;
		vi.stubGlobal("visualViewport", vv);
		vi.stubGlobal("ResizeObserver", class { observe() {} disconnect() {} });
		installFootGeometry();
		slot = { projection: p, connected: true };
		render(<SessionScreen sessionId="s1" />);
		return vv;
	}
	const readingOnly = (): SessionProjection => ({ ...projection(), model_label: "Claude Opus 5.5", usage: {}, context_tokens: 61_800, context_window: 200_000, cumulative_parent_cost: 2.47 } as SessionProjection);

	it("offers the spend row when the meter is hidden and has a reading, usage receipt or not (R3-2b)", () => {
		open(440, readingOnly());
		expect(document.querySelector("[data-foot-level]")?.getAttribute("data-foot-level")).toBe("2");
		fireEvent.click(screen.getByRole("button", { name: "session menu" }));
		expect(screen.getByRole("button", { name: /Spend & context/ })).toBeTruthy();
	});

	it("offers no spend row while the meter is on screen", () => {
		/* 360 without the mic: the compact meter fits (L1), so it is the door. */
		geometry.available = 196;
		open(440, readingOnly());
		expect(document.querySelector("[data-foot-level]")?.getAttribute("data-foot-level")).toBe("0");
		fireEvent.click(screen.getByRole("button", { name: "session menu" }));
		expect(screen.queryByRole("button", { name: /Spend & context/ })).toBeNull();
	});

	it("offers no spend row when there is no reading to open", () => {
		open(440, { ...projection(), model_label: "Claude Opus 5.5" });
		fireEvent.click(screen.getByRole("button", { name: "session menu" }));
		expect(screen.queryByRole("button", { name: /Spend & context/ })).toBeNull();
	});

	it("keeps both rows in the open menu when the keyboard closes under it (D9)", () => {
		const vv = open(440, { ...readingOnly(), queued_count: 2 });
		/* 320 + mic: folder + `2 queued` + the model floor need 149 > 110 px. */
		expect(document.querySelector("[data-foot-level]")?.getAttribute("data-foot-level")).toBe("4");
		expect(document.querySelector("[data-queued-moved]")?.textContent).toBe("2 queued");
		fireEvent.click(screen.getByRole("button", { name: "session menu" }));
		const dialog = screen.getByRole("dialog");
		expect(dialog.textContent).toContain("2 queued");
		expect(screen.getByRole("button", { name: /Spend & context/ })).toBeTruthy();
		act(() => { vv.height = 700; vv.dispatchEvent(new Event("resize")); });
		/* The footer has its own line again, and the menu has not changed. */
		expect(document.querySelector("[data-foot-level]")?.getAttribute("data-foot-level")).toBe("0");
		expect(screen.getByRole("dialog").textContent).toContain("2 queued");
		expect(screen.getByRole("button", { name: /Spend & context/ })).toBeTruthy();
	});
});
