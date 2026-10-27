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
		expect(cls).toContain("pb-2");
		// The composer card's own outer inset (`COMPOSER_SHELL` is `mx-2`), so an
		// item's edge sits on the composer's edge.
		expect(cls).toContain("px-2");
		expect(document.querySelector(".lo-composer")?.className.split(/\s+/)).toContain("mx-2");
		// No top padding: the transcript's own `py-2` already leaves 8 px under
		// its last line, so a top pad would double the gap above the first item.
		expect(cls.some((c) => /^p[ty]-/.test(c))).toBe(false);
		// Nothing between it and the composer card in flow, so its `pb-2` IS the
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
