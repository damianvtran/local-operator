// @vitest-environment happy-dom
//
// The delegated-work rung of the phone's list (PR A §3). A parent whose own turn
// is not running but which still owns children used to render as an idle row:
// the app has no "attached"/wake arms to carry it, so the SLOT is where the
// state has to appear, with the count as text beside it.
//
// Asserted against the REAL SessionListScreen, like the unread ladder it sits
// under, so the rung order, the reserved slot and the chip are read off the
// production card rather than a copy of it.
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SessionListScreen } from "./screens/session-list";
import type { SessionSummary } from "./types";

let sessionList: SessionSummary[] = [];
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useSessions: () => ({ sessions: sessionList, connected: true }),
		retainSessionListStream: () => () => {},
	};
});
vi.mock("./api", () => ({
	getDirectories: vi.fn(async () => ({ home: "", recent: [] })),
}));

function summary(over: Partial<SessionSummary>): SessionSummary {
	return {
		session_id: "s",
		section: "active",
		conversation_name: "Session",
		cwd: "",
		model_label: "",
		streaming: false,
		needs_attention: false,
		unseen: false,
		pending_kind: "",
		subagents_running: 0,
		subagents_queued: 0,
		todos_open: 0,
		mtime: 0,
		...over,
	};
}

function cardByName(name: string): HTMLButtonElement {
	return screen.getByRole("button", { name: new RegExp(name) });
}

/* The ONE reserved indicator slot: sized to the largest occupant (the spinner),
   so every title starts at the same x in every state. Selected by its own
   geometry class rather than by position, exactly as the unread ladder's test
   does — a decision+streaming row also carries an sr-only status node here. */
function slot(card: HTMLButtonElement): HTMLElement {
	return card.querySelector("div")!.querySelector(".size-3") as HTMLElement;
}

/* The dots the delegating rung paints inside the slot. */
function marks(card: HTMLButtonElement): HTMLElement[] {
	return Array.from(slot(card).querySelectorAll(".size-1"));
}

afterEach(() => {
	cleanup();
	sessionList = [];
});

describe("SessionCard delegated-work rung", () => {
	it("marks a parent that owns running children, and counts them in the shared noun", () => {
		sessionList = [
			summary({ session_id: "p1", conversation_name: "Parent", subagents_running: 2 }),
		];
		render(<SessionListScreen />);
		const card = cardByName("Parent");

		/* Two 4px dots, not one: this package carries no icon dependency, and a
		   single dot in the accent ink would be indistinguishable from the unread
		   mark one rung above it in the same slot. */
		const dots = marks(card);
		expect(dots).toHaveLength(2);
		for (const dot of dots) {
			expect(dot.className).toContain("bg-accent");
			/* Static: `busy` owns the animation in this slot. */
			expect(dot.className).not.toContain("lo-pulse");
		}

		/* "subagents" and not "agents": the record's own field name, `/info`'s
		   word, the composer chip's word. */
		expect(card.textContent).toContain("2 subagents");
		expect(card.textContent).not.toContain("2 agents");

		/* Geometry is untouched: the same 12px slot, so the title's start x is
		   unchanged from the idle row below it. */
		expect(slot(card).className).toContain("size-3");
	});

	it("singularises at one", () => {
		sessionList = [
			summary({ session_id: "p1", conversation_name: "Solo", subagents_running: 1 }),
		];
		render(<SessionListScreen />);
		expect(cardByName("Solo").textContent).toContain("1 subagent");
		expect(cardByName("Solo").textContent).not.toContain("1 subagents");
	});

	it("shows a queued-only parent, which must not read as idle", () => {
		sessionList = [
			summary({
				session_id: "p1",
				conversation_name: "Parked",
				subagents_running: 0,
				subagents_queued: 2,
			}),
		];
		render(<SessionListScreen />);
		const card = cardByName("Parked");
		expect(marks(card)).toHaveLength(2);
		expect(card.textContent).toContain("2 queued");
	});

	it("lets an unseen completion keep the slot, since a receipt outranks live work", () => {
		sessionList = [
			summary({
				session_id: "p1",
				conversation_name: "Unread parent",
				unseen: true,
				subagents_running: 2,
			}),
		];
		render(<SessionListScreen />);
		const card = cardByName("Unread parent");

		/* The single unread dot stays; the delegating pair does not stack on it. */
		expect(marks(card)).toHaveLength(0);
		expect(card.textContent).toContain("new");
		/* ...and the count is still told, because the chip is not the mark. */
		expect(card.textContent).toContain("2 subagents");
	});

	it("lets a streaming parent keep the spinner, and still counts its children", () => {
		sessionList = [
			summary({
				session_id: "p1",
				conversation_name: "Working parent",
				streaming: true,
				subagents_running: 3,
			}),
		];
		render(<SessionListScreen />);
		const card = cardByName("Working parent");
		expect(slot(card).querySelector(".lo-spinner")).toBeTruthy();
		expect(marks(card)).toHaveLength(0);
		expect(card.textContent).toContain("3 subagents");
	});

	it("renders NOTHING for an unreported count rather than asserting zero", () => {
		/* A daemon that predates the field, or a durable-only row: `null` means
		   "not reported", and the phone must not say "no subagents" about a
		   session it could not ask. Same rule the empty-vs-unreadable listing
		   follows. */
		sessionList = [
			summary({
				session_id: "p1",
				conversation_name: "Unknown",
				subagents_running: null,
				subagents_queued: null,
			}),
		];
		render(<SessionListScreen />);
		const card = cardByName("Unknown");
		expect(card.textContent).not.toContain("subagent");
		expect(card.textContent).not.toContain("queued");
		expect(card.textContent).not.toContain("0 ");
		/* The slot keeps its transparent spacer, so nothing reflows. */
		expect(marks(card)).toHaveLength(0);
		expect(slot(card).className).toContain("size-3");
	});

	it("renders nothing when the daemon reports an explicit zero", () => {
		sessionList = [
			summary({
				session_id: "p1",
				conversation_name: "Childless",
				subagents_running: 0,
				subagents_queued: 0,
			}),
		];
		render(<SessionListScreen />);
		const card = cardByName("Childless");
		expect(card.textContent).not.toContain("subagent");
		expect(marks(card)).toHaveLength(0);
	});
});
