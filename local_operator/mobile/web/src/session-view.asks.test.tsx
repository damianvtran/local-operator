// @vitest-environment happy-dom
//
// R7 ON THE PHONE (design §5.0), plus the §4 client rule N3.
//
// The claims this pins, and why each is a real failure rather than a style
// preference:
//
//  * With a queued ask published, the MIRRORED single-slot card must not also
//    render. The mirror exists for old clients; a new client that drew both
//    would show one ask twice, and the second card's answer is a refusal.
//  * The header must not read "needs you" for it — a queued ask is not a run
//    held hostage, and the gate control's word is what teaches a user which
//    states need them.
//  * With the ask surface MINIMIZED (the sheet closed), the composer is an
//    ordinary conversation composer: Enter sends `prompt`, never an answer.
//    This is the leak the design's routing rule exists to prevent — the blocked
//    path's composer swallow is exactly the behaviour being replaced.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SessionScreen } from "./screens/session-view";
import type { PendingAsk, PendingRequest, SessionProjection } from "./types";

vi.mock("./api", () => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentDetail: vi.fn(async () => null),
	imageUrl: vi.fn(() => ""),
	getCommands: vi.fn(async () => ({ commands: [] })),
	getModels: vi.fn(async () => ({ models: [] })),
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

const { sendCommand } = await import("./api");

function ask(patch: Partial<PendingAsk> = {}): PendingAsk {
	return {
		ask_id: "ask-7",
		created_at: 1,
		expires_at: Date.now() + 900_000,
		timeout_s: 900,
		urgent: false,
		status: "open",
		delivered: false,
		questions: [
			{
				id: "q1",
				question: "queue behind the flag?",
				options: [{ label: "yes", description: "" }],
				multi: false,
				secret: false,
				persist: false,
			},
		],
		...patch,
	};
}

function mirror(): PendingRequest {
	return {
		request_id: "ask-7.0",
		kind: "ask",
		title: "queue behind the flag?",
		detail: "",
		options: [{ label: "yes", description: "" }],
		secret: false,
		question_index: 0,
		question_total: 1,
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

afterEach(() => {
	cleanup();
	localStorage.clear();
	vi.clearAllMocks();
	slot = { projection: null, connected: true };
});

describe("SessionScreen and queued asks", () => {
	it("shows the minimized bar once the runtime publishes asks", () => {
		render(<SessionScreen sessionId="s1" />);
		slot = { projection: projection({ asks: [ask()], asks_open: 1 }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		expect(screen.getByTestId("ask-dock")).toBeTruthy();
		expect(screen.queryByTestId("pending-card")).toBeNull();
		expect(screen.getByRole("button", { name: /approvals in this session/ })).toBeTruthy();
		expect(screen.queryByText("needs you")).toBeNull();
	});

	it("still renders the mirrored card when the runtime predates asks (the skew window)", () => {
		render(<SessionScreen sessionId="s1" />);
		slot = { projection: projection({ pending: mirror(), pending_count: 1 }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		expect(screen.getByTestId("pending-card")).toBeTruthy();
		expect(screen.queryByTestId("ask-dock")).toBeNull();
	});

	it("renders the mirror ONLY through the queued surface once asks are published", () => {
		render(<SessionScreen sessionId="s1" />);
		slot = {
			projection: projection({ pending: mirror(), pending_count: 1, asks: [ask()], asks_open: 1 }),
			connected: true,
		};
		render(<SessionScreen sessionId="s1" />);
		expect(screen.queryByTestId("pending-card")).toBeNull();
		expect(screen.getByTestId("ask-dock")).toBeTruthy();
	});

	it("keeps the composer a CONVERSATION composer while the ask bar is showing (R7)", async () => {
		render(<SessionScreen sessionId="s1" />);
		slot = { projection: projection({ asks: [ask()], asks_open: 1 }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		const field = screen.getByPlaceholderText("Message…");
		fireEvent.change(field, { target: { value: "an ordinary message" } });
		fireEvent.keyDown(field, { key: "Enter" });
		await waitFor(() => expect(sendCommand).toHaveBeenCalled());
		for (const call of (sendCommand as ReturnType<typeof vi.fn>).mock.calls) {
			const op = (call[1] as { op?: string }).op;
			expect(op).not.toBe("ask_respond");
			expect(op).not.toBe("ask_answer");
		}
		expect(sendCommand).toHaveBeenCalledWith(
			"s1",
			expect.objectContaining({ op: "prompt", text: "an ordinary message" }),
		);
	});

	it("opens the answer surface from the header entry", async () => {
		render(<SessionScreen sessionId="s1" />);
		slot = { projection: projection({ asks: [ask()], asks_open: 1 }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		fireEvent.click(screen.getByRole("button", { name: /queued asks in this session \(1\)/ }));
		await waitFor(() => expect(screen.getByRole("dialog")).toBeTruthy());
	});
});
