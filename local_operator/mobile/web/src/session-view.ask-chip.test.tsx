// @vitest-environment happy-dom
//
// The chip's whole journey through the real screen: an id in an agent row is
// resolved against THIS session's live asks, and pressing it opens the asks
// sheet on that ask (the same door the dock uses) rather than on the queue's
// default head. The pieces are unit-tested separately; this is the wiring —
// provider, facts, door and jump — that no unit test holds together.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { askOpenPolicy } from "./lib/ask-open-policy";
import { SessionScreen } from "./screens/session-view";
import type { PendingAsk, SessionProjection, TranscriptEntry } from "./types";

vi.mock("./api", () => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentDetail: vi.fn(async () => null),
	imageUrl: vi.fn(() => ""),
	getCommands: vi.fn(async () => ({ commands: [] })),
	getSkills: vi.fn(async () => ({ skills: [] })),
	getModels: vi.fn(async () => ({ models: [] })),
	getDirectories: vi.fn(async () => ({ home: "/Users/tester", recent: [], tmp: "" })),
	changeDirectory: vi.fn(async () => ({ ok: true, pid: 1, session_id: "s1" })),
	getAsks: vi.fn(async () => ({ asks: [] })),
	sendCommand: vi.fn(async () => ({ ok: true, detail: "accepted" })),
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
	};
});

const { getAsks } = await import("./api");

function ask(patch: Partial<PendingAsk> = {}): PendingAsk {
	return {
		ask_id: "a-745e",
		created_at: 1,
		expires_at: Date.now() + 900_000,
		timeout_s: 900,
		urgent: false,
		status: "open",
		delivered: false,
		questions: [
			{ id: "q1", question: "Which region?", options: [], multi: false, secret: false, persist: false },
		],
		...patch,
	};
}

function projection(patch: Partial<SessionProjection> = {}): SessionProjection {
	return {
		session_id: "s1",
		pid: 1,
		kind: "tui",
		conversation_name: "Asks",
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
		todos: [],
		subagents: [],
		pending: null,
		pending_count: 0,
		usage: {},
		version: 1,
		...patch,
	} satisfies SessionProjection;
}

function assistant(text: string): TranscriptEntry {
	return { id: "m1", kind: "assistant", text, final: true, details: {} } as unknown as TranscriptEntry;
}

beforeEach(() => {
	askOpenPolicy.reset();
	// The door is this test's subject; the open-by-default policy would open the
	// sheet before the chip is pressed.
	askOpenPolicy.setEnabled(false);
	(getAsks as ReturnType<typeof vi.fn>).mockResolvedValue({
		asks: [ask({ session_id: "s1" })],
	});
});

afterEach(() => {
	cleanup();
	localStorage.clear();
	vi.clearAllMocks();
	slot = { projection: null, connected: true };
	askOpenPolicy.setEnabled(true);
	askOpenPolicy.reset();
});

describe("an ask-id chip on the session screen", () => {
	it("opens the asks sheet on the ask its id names", async () => {
		slot = {
			projection: projection({
				asks: [ask()],
				asks_open: 1,
				transcript: [assistant("the new one is a-745e")],
			}),
			connected: true,
		};
		render(<SessionScreen sessionId="s1" />);
		const chip = await screen.findByRole("button", { name: /Ask a-745e, open/ });
		fireEvent.click(chip);
		const card = await waitFor(() => {
			const found = screen.queryByTestId("ask-card");
			if (!found) throw new Error("the sheet has not opened on the ask");
			return found;
		});
		expect(card.getAttribute("data-ask-id")).toBe("a-745e");
		await waitFor(() => expect(document.activeElement).toBe(screen.getByTestId("ask-question-text")));
	});

	it("leaves an id that is not this session's ask as plain text", async () => {
		slot = {
			projection: projection({
				asks: [ask()],
				asks_open: 1,
				transcript: [assistant("the old one was a-9999")],
			}),
			connected: true,
		};
		render(<SessionScreen sessionId="s1" />);
		await screen.findByTestId("ask-id-plain");
		expect(screen.queryByRole("button", { name: /Ask a-9999/ })).toBeNull();
	});
});
