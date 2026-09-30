// @vitest-environment happy-dom
//
// The agent-opened mark on the phone's list (design note §10.4; manager
// decision §14.4). A listed workstream an agent opened must be
// distinguishable from the operator's own conversation — the 2026-09-18
// confusion class, one surface out. Asserted against the REAL
// SessionListScreen, like the delegated-work rung beside it, so the mark is
// read off the production card rather than a copy of it.
import { cleanup, render, screen, within } from "@testing-library/react";
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

afterEach(() => {
	cleanup();
	sessionList = [];
});

describe("SessionCard agent-opened rung", () => {
	it("marks a row whose opener is on the wire", () => {
		sessionList = [
			summary({
				session_id: "ws1",
				conversation_name: "Release cutter",
				opened_by: { agent: "manager", label: "sessions-tool-0b39", session: "e842" },
			}),
		];
		render(<SessionListScreen />);
		const card = cardByName("Release cutter");
		expect(within(card).getByText("agent")).toBeTruthy();
		/* The prefix lives in the chip itself (asserted below by accessible
		   name); the visible word is found by its own text node. */
	});

	it("marks a row even when every member of the opener is null", () => {
		/* PRESENCE is the fact, not any member: a top-level requester's object
		   carries no role, label or id, and the row is still not the
		   operator's own conversation. */
		sessionList = [
			summary({
				session_id: "ws2",
				conversation_name: "Anonymous workstream",
				opened_by: { agent: null, label: null, session: null },
			}),
		];
		render(<SessionListScreen />);
		const card = cardByName("Anonymous workstream");
		expect(within(card).getByText("agent")).toBeTruthy();
	});

	it("announces the mark as an origin, not an identity (design round 1, D1)", () => {
		/* The VISIBLE word stays `agent` (the measured 40 px chip budget),
		   but the accessible name must read "opened by agent": a bare noun
		   trailing the title can be heard as the session's identity rather
		   than its origin, the same confusion the mark exists to end. */
		sessionList = [
			summary({
				session_id: "ws9",
				conversation_name: "Release cutter",
				opened_by: { agent: null, label: null, session: null },
			}),
		];
		render(<SessionListScreen />);
		expect(screen.getByRole("button", { name: /opened by agent/ })).toBeTruthy();
	});

	it("leaves the operator's own conversations unmarked", () => {
		sessionList = [
			summary({ session_id: "own", conversation_name: "Operator session" }),
			summary({
				session_id: "old",
				conversation_name: "Older daemon row",
				opened_by: null,
			}),
		];
		render(<SessionListScreen />);
		expect(within(cardByName("Operator session")).queryByText("agent")).toBeNull();
		expect(within(cardByName("Older daemon row")).queryByText("agent")).toBeNull();
	});

	it("carries the mark without moving the title off its left edge", () => {
		/* The chip is `shrink-0` and the title pays for it; the title must
		   still start at the card's left edge and only its clip box may give.
		   Asserted structurally (which span carries what), never by pixel. */
		sessionList = [
			summary({
				session_id: "ws3",
				conversation_name: "Release cutter",
				opened_by: { agent: null, label: null, session: null },
			}),
		];
		render(<SessionListScreen />);
		const card = cardByName("Release cutter");
		const chip = within(card).getByText("agent");
		expect(chip.className).toContain("shrink-0");
		const title = within(card).getByText("Release cutter");
		expect(title.className).toContain("truncate");
	});
});
