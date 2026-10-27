// @vitest-environment happy-dom
//
// THE ABOVE-COMPOSER STACK'S ONE RHYTHM (operator report: the ended strip sat
// flush on the composer card, its text hugged its edge, the transcript's last
// line sat tight above it).
//
// The cause was structural: every item between the transcript and the composer
// chose its own outside padding (`mx-2` card, `px-3 py-1.5` tray, full-bleed
// `px-2`/`px-3` bands), so nothing shared a gap. What is pinnable here is that
// structure: every item renders INSIDE the one slot, the slot owns the gap and
// the inset, and no item re-adds an outer inset of its own. The pixels (8 px
// between items and to the composer, 8 px under the transcript's last line, the
// slot's edge on the composer's edge) are measured in the rendered frames —
// happy-dom does no layout, the same limit `composer.spacing.test.tsx` states.
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SessionScreen } from "./screens/session-view";
import type { PendingRequest, SessionProjection } from "./types";

vi.mock("./router", () => ({ navigate: vi.fn() }));
vi.mock("./api", async () => {
	const { HttpError } = await vi.importActual<typeof import("./api")>("./api");
	return {
		HttpError,
		getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
		getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
		getSubagentDetail: vi.fn(async () => null),
		imageUrl: vi.fn(() => ""),
		getCommands: vi.fn(async () => ({ commands: [] })),
		getModels: vi.fn(async () => ({ models: [] })),
		getDirectories: vi.fn(async () => ({ home: "/Users/tester", recent: [], tmp: "" })),
		changeDirectory: vi.fn(async () => ({ ok: true, pid: 1, session_id: "s1" })),
		sendCommand: vi.fn(async () => ({ ok: true, detail: "" })),
		markSessionSeen: vi.fn(async () => ({ ok: true })),
		setSessionPin: vi.fn(async () => ({ ok: true })),
		resumeSession: vi.fn(async () => ({ ok: true, pid: 42, session_id: "s1" })),
	};
});
let slot: { projection: SessionProjection | null; connected: boolean } = { projection: null, connected: true };
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useProjection: vi.fn(() => slot),
		retainProjectionStream: vi.fn(() => () => {}),
		retainSessionListStream: vi.fn(() => () => {}),
		useDraft: vi.fn(() => ["", () => {}]),
	};
});

afterEach(cleanup);

const APPROVAL: PendingRequest = {
	request_id: "req-1",
	kind: "approval",
	title: "Run pnpm deploy?",
	detail: "",
	options: [
		{ label: "Approve", description: "" },
		{ label: "Deny", description: "" },
	],
	secret: false,
	question_index: 0,
	question_total: 1,
};

function projection(over: Partial<SessionProjection> = {}): SessionProjection {
	return {
		session_id: "s1",
		pid: 4242,
		kind: "tui",
		conversation_name: "rhythm",
		cwd: "",
		model_label: "",
		model_selector: "",
		effort: "",
		effort_ladder: [],
		streaming: false,
		activity: "",
		activity_started_s: 0,
		stop_reason: "",
		queued_count: 0,
		ended: false,
		degraded: false,
		transcript: [],
		todos: [{ name: "Todos", items: [{ text: "audit", status: "pending", reason: "" }] }],
		subagents: [],
		pending: APPROVAL,
		pending_count: 1,
		usage: {},
		version: 1,
		...over,
	} satisfies SessionProjection;
}

/** The ONE slot between the transcript and the composer. */
function stack(): HTMLElement {
	const el = document.querySelector<HTMLElement>("[data-above-composer]");
	if (!el) throw new Error("no above-composer stack rendered");
	return el;
}

/** Tailwind spacing utilities that put space OUTSIDE an element's box, or on
    its inline sides — the stacked independent insets the slot replaces. */
const OUTER_INSET = /(?:^|\s)-?(?:m[xytblrse]?|px|ps|pe|pl|pr)-\S+/;

describe("everything stacked above the composer shares one slot", () => {
	it("owns ONE gap, ONE inset aligned with the composer card, and the gap to the composer", () => {
		slot = { projection: projection(), connected: true };
		render(<SessionScreen sessionId="s1" />);
		const cls = stack().className.split(/\s+/);
		// 8 px between items and 8 px down to the composer card's top edge.
		expect(cls).toContain("gap-2");
		/* `mb-2`, OUTSIDE the scrollport (design round 2, D6 = review R2-1): a
		   `pb-2` sits at the END of the scrolled content, so an overflowing slot
		   rested flush on the composer (slot bottom 144.6 = composer top 144.6
		   at 320x440) and its cut edge fused into the card below. A margin keeps
		   the 8 px whether or not the slot scrolls. */
		expect(cls).toContain("mb-2");
		expect(cls).not.toContain("pb-2");
		// The composer card's own outer inset (`COMPOSER_SHELL` is `mx-2`), so an
		// item's edge sits on the composer's edge.
		expect(cls).toContain("px-2");
		expect(document.querySelector(".lo-composer")?.className.split(/\s+/)).toContain("mx-2");
		// No top padding: the transcript's own `py-2` already leaves 8 px under
		// its last line, so a top pad would double the gap above the first item.
		expect(cls.some((c) => /^p[ty]-/.test(c))).toBe(false);
		// Nothing between it and the composer card in flow, so its `mb-2` IS the
		// gap to the card's top edge (sheets render nothing while closed).
		const composer = document.querySelector(".lo-composer")!;
		let next = stack().nextElementSibling;
		while (next && !next.contains(composer) && next !== composer) {
			expect(next.getBoundingClientRect().height).toBe(0);
			expect(next.textContent).toBe("");
			next = next.nextElementSibling;
		}
		expect(next).toBeTruthy();
	});

	it("places the pending card AND the tray inside it, with no outer inset of their own", () => {
		slot = { projection: projection(), connected: true };
		render(<SessionScreen sessionId="s1" />);
		const card = document.querySelector<HTMLElement>('[data-testid="pending-card"]');
		const tray = screen.getByLabelText("Session status");
		for (const item of [card, tray]) {
			expect(item).toBeTruthy();
			expect(item!.parentElement).toBe(stack());
			expect(item!.className).not.toMatch(OUTER_INSET);
		}
		// Order: the decision card first, the tray right above the composer.
		expect(card!.compareDocumentPosition(tray) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
	});

	it("keeps the tray's own 44 px row (its pills' hit bands live in it)", () => {
		slot = { projection: projection({ pending: null, pending_count: 0 }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		const tray = screen.getByLabelText("Session status");
		expect(tray.className.split(/\s+/)).toEqual(expect.arrayContaining(["min-h-11", "shrink-0", "py-1.5"]));
	});

	it("renders no slot at all when nothing is stacked, so it adds no gap of its own", () => {
		slot = { projection: projection({ pending: null, pending_count: 0, todos: [] }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		expect(document.querySelector("[data-above-composer]")).toBeNull();
	});

	it("holds no health banner on an ended or degraded session — the header dot carries those", () => {
		for (const flags of [{ ended: true }, { degraded: true }]) {
			slot = { projection: projection({ pending: null, pending_count: 0, ...flags }), connected: true };
			render(<SessionScreen sessionId="s1" />);
			const children = [...stack().children];
			expect(children).toHaveLength(1);
			expect(children[0]).toBe(screen.getByLabelText("Session status"));
			cleanup();
		}
	});
});

describe("the slot never paints over the composer (QA round 1, Q2)", () => {
	/* QA [redacted] at 320x700 with a 260 px keyboard (column 440) and the
	   composer's retry alert open: the composer grew to ~340 px, the column's
	   flex shrink took the deficit out of this slot (`min-h-0`) and out of the
	   card (`min-h-0`, down to its 20 px padding), and both kept PAINTING their
	   children past their shrunken boxes — the composer, later in flow, covered
	   the tray and the card's Approve by 43 px. The slot must own its overflow:
	   it is the scroller, so a short column scrolls it instead of letting its
	   items spill; and each item keeps its own height inside it (the card's
	   column cap still bounds it), so nothing is squeezed below its controls.

	   happy-dom does no layout, so this pins the CONTRACT that makes overlap
	   impossible; the measured gap (>= 0 at 320x440 with all three present) is
	   [redacted] in the rendered frame recorded on the PR. */
	it("is a clipped vertical scroller whose items keep their own height", () => {
		slot = { projection: projection(), connected: true };
		render(<SessionScreen sessionId="s1" />);
		const cls = stack().className.split(/\s+/);
		// The scroller: clips at its own edge instead of spilling onto the composer.
		expect(cls).toContain("overflow-y-auto");
		// Never sideways (the column rule, no-sideways-scroll.test.ts).
		expect(cls).toContain("overflow-x-hidden");
		// Still yields to the column so the composer keeps its place.
		expect(cls).toContain("min-h-0");
		// Each item keeps its height INSIDE the slot: the slot scrolls, the card
		// is not squeezed under its own pinned controls.
		expect(cls).toContain("*:shrink-0");
		// The house scrollbar, but NO mask: the card renders its pairing sheet IN
		// PLACE (sheets never portal), and a `mask-image` on an ancestor would
		// clip that fixed-position overlay to this slot's box. `overflow` does
		// not clip a fixed descendant; a mask does.
		expect(cls).toContain("lo-scroll");
		expect(cls.some((c) => c.startsWith("lo-fade"))).toBe(false);
	});

	it("keeps all three present at 320x440 — card, tray AND the composer's retry alert", () => {
		/* The state QA hit: a retained envelope mounts the composer with its
		   retry alert. The card and the tray stay inside the slot, the slot stays
		   ABOVE the composer in flow (never a sibling painted after it), and the
		   composer still carries the alert. */
		localStorage.setItem(
			"lo-mobile-command:s1",
			JSON.stringify({
				version: 1,
				saved_at: Date.now(),
				envelope: { command_id: "0f2f1d5e-6c3a-4b7e-9f1a-2b3c4d5e6f70", op: "prompt", text: "hello from QA" },
			}),
		);
		try {
			slot = { projection: projection(), connected: true };
			render(<SessionScreen sessionId="s1" />);
			const composer = document.querySelector(".lo-composer")!;
			expect(composer.textContent).toContain("Retry earlier instruction");
			const card = document.querySelector<HTMLElement>('[data-testid="pending-card"]')!;
			const tray = screen.getByLabelText("Session status");
			expect(card.parentElement).toBe(stack());
			expect(tray.parentElement).toBe(stack());
			expect(stack().compareDocumentPosition(composer) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
			expect(stack().contains(composer)).toBe(false);
		} finally {
			localStorage.removeItem("lo-mobile-command:s1");
		}
	});
});

describe("an overflowing slot rests on the card's decision (design D6 = review R2-1)", () => {
	/* At 320x440 with the card, the tray and the retry alert, the slot's 65 px
	   viewport showed only the card's "approval needed / bash" head and its
	   Approve sat below a fold with no cue (touch scrollbars are transient and
	   the slot carries no mask). Resting at the raw END was measured wrong too:
	   it shows the tray, whose "Approve" chip opens the approvals-mode sheet,
	   and leaves the card's own Approve/Deny above the fold. So the slot brings
	   the row that ANSWERS the request to its bottom edge.

	   happy-dom does no layout, so the boxes are stubbed on the slot and the
	   answering row alone, with the geometry measured on the rendered page:
	   slot 72-136.6, content 316 px, the Approve/Deny row 252-296 at rest. */
	function stubLayout(scrollHeight: number, clientHeight: number, row: { top: number; bottom: number }) {
		const isSlot = (el: Element) => el instanceof HTMLElement && el.hasAttribute("data-above-composer");
		const sh = Object.getOwnPropertyDescriptor(Element.prototype, "scrollHeight")!;
		const ch = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientHeight")!;
		const rect = Element.prototype.getBoundingClientRect;
		vi.spyOn(Element.prototype, "scrollHeight", "get").mockImplementation(function (this: Element) {
			return isSlot(this) ? scrollHeight : (sh.get!.call(this) as number);
		});
		vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockImplementation(function (this: HTMLElement) {
			return isSlot(this) ? clientHeight : (ch.get!.call(this) as number);
		});
		vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(function (this: Element) {
			if (isSlot(this)) return new DOMRect(0, 72, 320, 64.6);
			if (this instanceof HTMLElement && this.hasAttribute("data-slot-rest")) {
				return new DOMRect(8, row.top, 304, row.bottom - row.top);
			}
			return rect.call(this);
		});
	}
	afterEach(() => vi.restoreAllMocks());

	it("brings the card's Approve/Deny row to the slot's bottom edge, not the raw end", () => {
		stubLayout(316, 65, { top: 252, bottom: 296 });
		slot = { projection: projection(), connected: true };
		render(<SessionScreen sessionId="s1" />);
		const row = document.querySelector<HTMLElement>("[data-slot-rest]")!;
		// The marked row is the one that answers: it holds Approve and Deny.
		expect(row.textContent).toMatch(/approve/i);
		expect(row.textContent).toMatch(/deny/i);
		expect(screen.getByTestId("pending-card").contains(row)).toBe(true);
		// 296 - 136.6: the row's bottom lands on the slot's bottom edge.
		expect(stack().scrollTop).toBeCloseTo(159.4, 1);
		expect(stack().scrollTop).toBeLessThan(316 - 65);
	});

	it("leaves a slot alone when nothing in it asks for a decision", () => {
		stubLayout(316, 65, { top: 252, bottom: 296 });
		slot = { projection: projection({ pending: null, pending_count: 0 }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		expect(document.querySelector("[data-slot-rest]")).toBeNull();
		expect(stack().scrollTop).toBe(0);
	});

	it("does nothing when the slot fits, or the row is already in view", () => {
		stubLayout(120, 120, { top: 252, bottom: 296 });
		slot = { projection: projection(), connected: true };
		const fits = render(<SessionScreen sessionId="s1" />);
		expect(stack().scrollTop).toBe(0);
		fits.unmount();
		vi.restoreAllMocks();
		stubLayout(316, 65, { top: 80, bottom: 124 });
		render(<SessionScreen sessionId="s1" />);
		expect(stack().scrollTop).toBe(0);
	});
});

describe("an ask that offers options rests on them too (review R3-1)", () => {
	/* R3-1: `data-slot-rest` sat only on the approval row and the free-text
	   row, so an ask that offers OPTIONS had no rest target and an overflowing
	   slot stayed at its top, the question's head, with every option below the
	   fold. The first option is the target: resting its bottom on the slot's
	   bottom edge puts the start of the decision in view with the question
	   just above it, and the rest of the list is one scroll away. */
	const ASK_WITH_OPTIONS: PendingRequest = {
		request_id: "req-ask",
		kind: "ask",
		title: "Which region?",
		detail: "",
		options: [
			{ label: "eu-west", description: "" },
			{ label: "us-east", description: "" },
			{ label: "ap-south", description: "" },
		],
		secret: false,
		question_index: 0,
		question_total: 1,
	};
	afterEach(() => vi.restoreAllMocks());

	it("marks the FIRST option as the rest target, and rests on it", () => {
		const isSlot = (el: Element) => el instanceof HTMLElement && el.hasAttribute("data-above-composer");
		const sh = Object.getOwnPropertyDescriptor(Element.prototype, "scrollHeight")!;
		const ch = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientHeight")!;
		const rect = Element.prototype.getBoundingClientRect;
		vi.spyOn(Element.prototype, "scrollHeight", "get").mockImplementation(function (this: Element) {
			return isSlot(this) ? 316 : (sh.get!.call(this) as number);
		});
		vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockImplementation(function (this: HTMLElement) {
			return isSlot(this) ? 65 : (ch.get!.call(this) as number);
		});
		vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(function (this: Element) {
			if (isSlot(this)) return new DOMRect(0, 72, 320, 64.6);
			if (this instanceof HTMLElement && this.hasAttribute("data-slot-rest")) return new DOMRect(8, 200, 304, 44);
			return rect.call(this);
		});
		/* No `asks` field: an owner that publishes `asks` routes a pending ask
		   to the asks dock, not this card (`lib/asks.ts` `blockingPending`). */
		slot = { projection: projection({ pending: ASK_WITH_OPTIONS }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		const rests = document.querySelectorAll<HTMLElement>("[data-slot-rest]");
		expect(rests).toHaveLength(1);
		expect(rests[0]!.textContent).toContain("eu-west");
		expect(screen.getByTestId("pending-card").contains(rests[0]!)).toBe(true);
		// 244 - 136.6: the first option's bottom lands on the slot's bottom edge.
		expect(stack().scrollTop).toBeCloseTo(107.4, 1);
	});
});

describe("the rest re-runs when the slot's own box resizes, and lets go (review R3-3)", () => {
	/* R3-3: happy-dom's ResizeObserver never calls back, so the observer half
	   of the rest was untested — yet the keyboard opening (a resize of the
	   slot's box) is what makes the slot overflow in the first place. This
	   stub records every observer and lets the test fire its callback. */
	type Observed = { callback: ResizeObserverCallback; targets: Element[]; disconnected: boolean };
	let observers: Observed[] = [];
	function stubObserver() {
		observers = [];
		vi.stubGlobal(
			"ResizeObserver",
			class {
				private readonly record: Observed;
				constructor(callback: ResizeObserverCallback) {
					this.record = { callback, targets: [], disconnected: false };
					observers.push(this.record);
				}
				observe(target: Element) { this.record.targets.push(target); }
				unobserve() {}
				disconnect() { this.record.disconnected = true; }
			},
		);
	}
	/** The slot's box, switchable between "fits" and "overflows" mid-test. */
	let overflow = false;
	function stubSlot() {
		const isSlot = (el: Element) => el instanceof HTMLElement && el.hasAttribute("data-above-composer");
		const sh = Object.getOwnPropertyDescriptor(Element.prototype, "scrollHeight")!;
		const ch = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientHeight")!;
		const rect = Element.prototype.getBoundingClientRect;
		vi.spyOn(Element.prototype, "scrollHeight", "get").mockImplementation(function (this: Element) {
			return isSlot(this) ? 316 : (sh.get!.call(this) as number);
		});
		vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockImplementation(function (this: HTMLElement) {
			return isSlot(this) ? (overflow ? 65 : 316) : (ch.get!.call(this) as number);
		});
		vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(function (this: Element) {
			if (isSlot(this)) return new DOMRect(0, 72, 320, overflow ? 64.6 : 316);
			if (this instanceof HTMLElement && this.hasAttribute("data-slot-rest")) return new DOMRect(8, 252, 304, 44);
			return rect.call(this);
		});
	}
	const slotObserver = () => observers.filter((o) => o.targets.some((t) => t.hasAttribute("data-above-composer")));
	afterEach(() => {
		vi.restoreAllMocks();
		vi.unstubAllGlobals();
		overflow = false;
	});

	it("re-rests when the keyboard opening shrinks the slot after the card mounted", () => {
		stubObserver();
		stubSlot();
		slot = { projection: projection(), connected: true };
		render(<SessionScreen sessionId="s1" />);
		// Mounted while it fit: nothing to do.
		expect(stack().scrollTop).toBe(0);
		expect(slotObserver()).toHaveLength(1);
		// The keyboard opens: the slot's box shrinks and the observer fires.
		overflow = true;
		const watcher = slotObserver()[0]!;
		watcher.callback([], {} as ResizeObserver);
		// 296 - 136.6: the decision row's bottom on the slot's bottom edge.
		expect(stack().scrollTop).toBeCloseTo(159.4, 1);
	});

	it("disconnects its observer when the card changes and when the screen unmounts", () => {
		stubObserver();
		stubSlot();
		slot = { projection: projection(), connected: true };
		const view = render(<SessionScreen sessionId="s1" />);
		const first = slotObserver()[0]!;
		expect(first.disconnected).toBe(false);
		// A NEW decision (another request) re-keys the rest: the old observer
		// lets go and a fresh one watches the slot.
		slot = {
			projection: projection({ pending: { ...APPROVAL, request_id: "req-2" } }),
			connected: true,
		};
		view.rerender(<SessionScreen sessionId="s1" />);
		expect(first.disconnected).toBe(true);
		const second = slotObserver().at(-1)!;
		expect(second).not.toBe(first);
		expect(second.disconnected).toBe(false);
		view.unmount();
		expect(second.disconnected).toBe(true);
	});
});

describe("a long ask's first option is revealed inside the card (review R1-4)", () => {
	it("scrolls the card's inner scroller as well as the slot", () => {
		slot = { projection: projection({ pending: { ...APPROVAL, kind: "ask", title: "A very long question ".repeat(20), detail: "More detail ".repeat(50), options: [{ label: "first", description: "" }, { label: "second", description: "" }] } }), connected: true };
		const rowBox = new DOMRect(0, 300, 304, 44);
		const originalRect = Element.prototype.getBoundingClientRect;
		const originalScrollHeight = Object.getOwnPropertyDescriptor(Element.prototype, "scrollHeight")!;
		const originalClientHeight = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "clientHeight")!;
		vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(function (this: Element) {
			if (this.hasAttribute("data-above-composer")) return new DOMRect(0, 72, 320, 65);
			if (this.hasAttribute("data-slot-rest")) return new DOMRect(0, rowBox.y, 304, 44);
			if (this.classList.contains("lo-scroll") && this.closest("[data-testid=pending-card]")) return new DOMRect(0, 72, 304, 100);
			return originalRect.call(this);
		});
		vi.spyOn(Element.prototype, "scrollHeight", "get").mockImplementation(function (this: Element) {
			if (this.hasAttribute("data-above-composer")) return 400;
			if (this.classList.contains("lo-scroll") && this.closest("[data-testid=pending-card]")) return 800;
			return originalScrollHeight.get!.call(this) as number;
		});
		vi.spyOn(HTMLElement.prototype, "clientHeight", "get").mockImplementation(function (this: HTMLElement) {
			if (this.hasAttribute("data-above-composer")) return 65;
			if (this.classList.contains("lo-scroll") && this.closest("[data-testid=pending-card]")) return 100;
			return originalClientHeight.get!.call(this) as number;
		});
		render(<SessionScreen sessionId="s1" />);
		const scroller = document.querySelector<HTMLElement>("[data-testid=pending-card] .lo-scroll")!;
		expect(scroller.scrollTop).toBe(172);
		expect(document.querySelector<HTMLElement>("[data-above-composer]")!.scrollTop).toBeGreaterThan(0);
		vi.restoreAllMocks();
	});
});
