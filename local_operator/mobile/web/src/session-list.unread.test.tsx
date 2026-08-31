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

/* The ONE reserved indicator slot: first child of the card's title row. It is
   sized to the largest occupant (the spinner) and holds either the spinner or
   the centred dot, so every title starts at the same x in all four states. */
function slot(card: HTMLButtonElement): HTMLElement {
	const row = card.querySelector("div")!;
	return row.firstElementChild as HTMLElement;
}

/* The dot inside the slot. Streaming rows have a spinner there instead. */
function dot(card: HTMLButtonElement): HTMLElement {
	return slot(card).firstElementChild as HTMLElement;
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
		/* D2/U1: the spinner renders INSIDE the reserved slot, not beside it.
		   Rendered alongside, a streaming row paid slot + gap + spinner and its
		   title sat ~19.5px right of every other row's. */
		expect(slot(card).querySelector(".lo-spinner")).toBeTruthy();
		expect(slot(card).className).toContain("size-3");
	});

	it("keeps the danger dot when a decision row is ALSO streaming", () => {
		/* D6: the approval gate runs INSIDE a turn — the harness blocks in
		   _execute_tool_calls between agent_start and agent_end — so an
		   ordinary "may I run this?" carries streaming AND pending at once.
		   An earlier slot tested `streaming` first and replaced the red pulse
		   with a neutral spinner on the loudest state in the ladder. */
		sessionList = [
			summary({
				session_id: "ds",
				conversation_name: "Blocked while streaming",
				needs_attention: true,
				pending_kind: "approval",
				streaming: true,
			}),
		];
		render(<SessionListScreen />);
		const card = cardByName("Blocked while streaming");

		const marker = dot(card);
		expect(marker.className).toContain("bg-danger");
		expect(marker.className).toContain("lo-pulse");
		/* The spinner must NOT have taken the slot. */
		expect(slot(card).querySelector(".lo-spinner")).toBeNull();
		/* The decision word still renders on the right. */
		expect(card.textContent).toContain("approval");
		/* "Working" is still conveyed — by the title shimmer, which costs no
		   geometry — so the row does not read as idle-but-blocked. */
		const title = slot(card).nextElementSibling as HTMLElement;
		expect(title.className).toContain("lo-shimmer");
		/* And the slot geometry is unchanged, so the title does not shift. */
		expect(slot(card).className).toContain("size-3");
	});

	it("gives every state the SAME indicator slot, so titles never shift", () => {
		/* D2/U1: the reserved-slot promise held for three states and broke on
		   streaming, which rendered the spinner IN ADDITION to the slot. One
		   slot, one size, all four states. */
		sessionList = [
			summary({ session_id: "a", conversation_name: "Decide", needs_attention: true, pending_kind: "approval" }),
			summary({ session_id: "b", conversation_name: "Working", streaming: true }),
			summary({ session_id: "c", conversation_name: "Unread", unseen: true }),
			summary({ session_id: "d", conversation_name: "Idle" }),
			summary({ session_id: "e", conversation_name: "Deciding", needs_attention: true, pending_kind: "ask", streaming: true }),
		];
		render(<SessionListScreen />);
		const names = ["Decide", "Working", "Unread", "Idle", "Deciding"];
		const slots = names.map((name) => slot(cardByName(name)));
		/* Identical slot geometry across all four; only the contents differ. */
		for (const s of slots) {
			expect(s.className).toContain("size-3");
			expect(s.className).toContain("shrink-0");
		}
		/* And the title is the slot's next sibling in every state — nothing is
		   inserted between the slot and the name. */
		for (const name of names) {
			const card = cardByName(name);
			const title = slot(card).nextElementSibling as HTMLElement;
			expect(title.textContent).toBe(name);
		}
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
		expect(slot(card).getAttribute("aria-hidden")).toBe("true");
	});
});
