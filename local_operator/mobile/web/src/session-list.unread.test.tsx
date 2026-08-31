// @vitest-environment happy-dom
//
// The four-state attention ladder (spec §1): NEEDS DECISION > WORKING >
// NEW/UNREAD > IDLE. Flags coexist in data; exactly one state renders. These
// tests render the REAL SessionListScreen so the render ladder, the reserved
// left dot slot, and the `new` word's classes/aria are asserted against the
// production card, not a copy of it.
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
		todos_open: 0,
		mtime: 0,
		...over,
	};
}

function cardByName(name: string): HTMLButtonElement {
	return screen.getByRole("button", { name: new RegExp(name) });
}

/* The reserved dot slot is the first child of the card's title row. */
function dot(card: HTMLButtonElement): HTMLElement {
	const row = card.querySelector("div")!;
	return row.firstElementChild as HTMLElement;
}

afterEach(() => {
	cleanup();
	sessionList = [];
});

describe("SessionCard attention ladder", () => {
	it("renders the unread card with a static accent dot and the word `new`", () => {
		sessionList = [
			summary({ session_id: "u1", conversation_name: "Alpha", unseen: true }),
		];
		render(<SessionListScreen />);
		const card = cardByName("Alpha");
		expect(card.textContent).toContain("new");

		const word = Array.from(card.querySelectorAll("span")).find(
			(span) => span.textContent === "new",
		)!;
		expect(word.className).toContain("text-accent");
		expect(word.getAttribute("aria-label")).toBe("new activity");

		const marker = dot(card);
		expect(marker.className).toContain("bg-accent");
		/* Motion is reserved for danger: the unread dot never pulses. */
		expect(marker.className).not.toContain("lo-pulse");
	});

	it("lets needs-decision outrank unread", () => {
		sessionList = [
			summary({
				session_id: "d1",
				conversation_name: "Beta",
				unseen: true,
				needs_attention: true,
				pending_kind: "approval",
			}),
		];
		render(<SessionListScreen />);
		const card = cardByName("Beta");
		expect(card.textContent).toContain("approval");
		/* The unread mark is suppressed, not stacked. */
		expect(card.textContent).not.toContain("new");

		const marker = dot(card);
		expect(marker.className).toContain("bg-danger");
		expect(marker.className).toContain("lo-pulse");
	});

	it("lets streaming outrank unread", () => {
		sessionList = [
			summary({
				session_id: "w1",
				conversation_name: "Gamma",
				unseen: true,
				streaming: true,
			}),
		];
		render(<SessionListScreen />);
		const card = cardByName("Gamma");
		expect(card.textContent).not.toContain("new");
		/* In-flight work keeps the spinner and shimmer, not the accent mark. */
		expect(card.querySelector(".lo-spinner")).toBeTruthy();
		expect(dot(card).className).toContain("bg-transparent");
	});

	it("keeps the reserved dot slot on an idle card", () => {
		sessionList = [
			summary({ session_id: "i1", conversation_name: "Delta" }),
		];
		render(<SessionListScreen />);
		const card = cardByName("Delta");
		expect(card.textContent).not.toContain("new");

		/* The slot is always rendered — indicators change colour, never
		   geometry — so an idle card still carries the transparent dot. */
		const marker = dot(card);
		expect(marker).toBeTruthy();
		expect(marker.className).toContain("size-1.5");
		expect(marker.className).toContain("bg-transparent");
		expect(marker.getAttribute("aria-hidden")).toBe("true");
	});
});
