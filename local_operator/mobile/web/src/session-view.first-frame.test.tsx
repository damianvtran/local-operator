// @vitest-environment happy-dom
//
// THE RESERVED FIRST FRAME (first-paint lane T2, finding F8).
//
// Before the projection lands, `SessionScreen` painted a header and one centred
// sentence; the composer, the transcript box, the working line and the panels
// all mounted in the commit that accepted the projection — a full-layout swap.
// These tests pin the two halves of the fix: the boxes are THERE before the
// projection, and they are the SAME boxes the projection fills.
//
// The class-list comparison is the load-bearing one and it is deliberately not
// a geometry check: happy-dom computes no layout, so the guarantee has to be
// carried by the shared class strings the composer and its reserve both use. A
// future composer change that stops sharing them fails here rather than on a
// phone.
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SessionScreen } from "./screens/session-view";
import type { SessionProjection } from "./types";

vi.mock("./api", () => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentDetail: vi.fn(async () => null),
	imageUrl: vi.fn(() => ""),
	getCommands: vi.fn(async () => ({ commands: [] })),
	getModels: vi.fn(async () => ({ models: [] })),
	getDirectories: vi.fn(async () => ({ home: "/Users/tester", recent: [], tmp: "" })),
	changeDirectory: vi.fn(async () => ({ ok: true, pid: 1, session_id: "s1" })),
	sendCommand: vi.fn(async () => ({ ok: true, detail: "answer accepted" })),
	markSessionSeen: vi.fn(async () => ({ ok: true })),
}));

let slot: { projection: SessionProjection | null; connected: boolean } = {
	projection: null,
	connected: true,
};
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

function projection(): SessionProjection {
	return {
		session_id: "s1",
		pid: 1,
		kind: "daemon",
		conversation_name: "First paint",
		cwd: "/tmp/project",
		model_label: "mock-model",
		model_selector: "mock",
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
		todos: [],
		subagents: [],
		pending: null,
		pending_count: 0,
		usage: {},
		version: 1,
	} satisfies SessionProjection;
}

afterEach(() => {
	cleanup();
	localStorage.clear();
	slot = { projection: null, connected: true };
});

/** The composer's own field, both before and after the projection lands. */
function field(container: HTMLElement): HTMLTextAreaElement {
	const el = container.querySelector("textarea");
	expect(el).not.toBeNull();
	return el as HTMLTextAreaElement;
}

describe("the relay's first frame reserves the settled layout", () => {
	it("paints the composer's box and the header before any projection exists", () => {
		const { container } = render(<SessionScreen sessionId="s1" />);
		expect(container.querySelector("header")).not.toBeNull();
		expect(field(container)).toBeTruthy();
		// The transcript box, which is the space the rows will land in.
		expect(container.querySelector("div.flex-1.min-h-0")).not.toBeNull();
	});

	it("reserves the composer with the SAME field the settled frame uses", () => {
		const reserved = render(<SessionScreen sessionId="s1" />);
		const before = field(reserved.container).className;
		reserved.unmount();

		slot = { projection: projection(), connected: true };
		const settled = render(<SessionScreen sessionId="s1" />);
		const after = field(settled.container).className;
		expect(after).toBe(before);
	});

	it("carries the settled empty composer's glyphs and placeholder in the reserve", () => {
		/* Design review round 1 (D1). The reserve was a ring, a bordered empty box
		   and a sunken disc: no paperclip, no `Message…`, no arrow — so the reveal
		   popped three glyphs and a placeholder into place, all four of which are
		   static and knowable before any data arrives. In dark the disc sits at
		   1.05:1 against the canvas, so the row read as "ring + outline" with the
		   right-hand control missing. */
		const { container } = render(<SessionScreen sessionId="s1" />);
		const frame = container.querySelector("[aria-hidden='true'] textarea") as HTMLTextAreaElement;
		expect(frame).not.toBeNull();
		expect(frame.placeholder).toBe("Message…");
		// One paperclip definition, drawn in the attach disc's disabled styling.
		expect(container.querySelectorAll("[aria-hidden='true'] svg").length).toBe(1);
		const discs = Array.from(
			container.querySelectorAll("[aria-hidden='true'] span"),
		).filter((el) => el.className.includes("size-11"));
		const send = discs.find((el) => el.className.includes("bg-sunken"));
		expect(send).toBeTruthy();
		expect(send?.textContent).toBe("↑");
	});

	it("reports a dropped link in its own sentence, with no ladder rung stacked on it", () => {
		// The reserved frame's sentence IS the link report, and with no data on
		// screen a "last synced view" strip would be noise about noise — the rule
		// `session-view.health.test.tsx` (U11) already pins. This asserts the
		// reserved frame keeps it rather than growing a second telling.
		slot = { projection: null, connected: false };
		render(<SessionScreen sessionId="s1" />);
		expect(screen.getByText("connecting to session…")).toBeTruthy();
		expect(screen.queryByText(/reconnecting — showing the last synced view/)).toBeNull();
	});

	it("shows no ladder rung while the link is up", () => {
		render(<SessionScreen sessionId="s1" />);
		expect(screen.queryByText(/reconnecting — showing the last synced view/)).toBeNull();
	});
});
