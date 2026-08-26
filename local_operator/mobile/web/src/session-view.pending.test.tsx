// @vitest-environment happy-dom
//
// Regression: a multi-part ask (several questions, one request_id) must
// present tappable options on EVERY question. The card's transient state
// (busy/error/remember/draft) lives in component-local useState, and the
// daemon keeps `pending` non-null across questions (same request_id,
// question_index advanced), so unless the render site keys the card on
// request_id+question_index React reuses the instance and `busy` — set while
// answering question 1 — leaks into question 2, disabling every option.
// This test renders the REAL SessionScreen (the render site that owns the
// key), not PendingCard in isolation, so a reverted key fails here.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SessionScreen } from "./screens/session-view";
import type { PendingRequest, SessionProjection } from "./types";

vi.mock("./api", () => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentDetail: vi.fn(async () => null),
	imageUrl: vi.fn(() => ""),
	getCommands: vi.fn(async () => ({ commands: [] })),
	getModels: vi.fn(async () => ({ models: [] })),
	sendCommand: vi.fn(async () => ({ ok: true, detail: "answer accepted" })),
}));

/* SessionScreen subscribes to the projection store; the test is the store.
   Only the hooks the screen tree reads are faked — the rest of the module
   (drafts, stream plumbing) stays real so the render path is production. */
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
		useDraft: vi.fn(() => ["", () => {}]),
	};
});

function pendingQuestion(index: number, total: number): PendingRequest {
	return {
		request_id: "req-multi",
		kind: "ask",
		title: `Question ${index + 1}`,
		detail: "",
		options: [
			{ label: `opt-${index}-a`, description: "" },
			{ label: `opt-${index}-b`, description: "" },
		],
		secret: false,
		question_index: index,
		question_total: total,
	};
}

function projection(pending: PendingRequest): SessionProjection {
	return {
		session_id: "s1",
		pid: 1,
		kind: "tui",
		conversation_name: "Multi",
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
		pending,
		pending_count: 1,
		usage: {},
		version: 1,
	} satisfies SessionProjection;
}

const { sendCommand } = await import("./api");

afterEach(() => {
	cleanup();
	localStorage.clear();
	vi.clearAllMocks();
	slot = { projection: null, connected: true };
});

describe("SessionScreen multi-part ask", () => {
	it("keeps question 2 options tappable after answering question 1", async () => {
		const { rerender } = render(
			<SessionScreen sessionId="s1" />,
		);
		slot = { projection: projection(pendingQuestion(0, 2)), connected: true };
		rerender(<SessionScreen sessionId="s1" />);
		expect(screen.getByText("Question 1 of 2")).toBeTruthy();

		// Answer question 1. sendCommand resolves, the daemon advances the
		// picker, and the projection repaints to question 2 — same
		// request_id, question_index 1. `pending` never goes null.
		fireEvent.click(screen.getByRole("button", { name: /opt-0-a/ }));
		await waitFor(() =>
			expect(sendCommand).toHaveBeenCalledWith("s1", {
				op: "ask_answer",
				request_id: "req-multi",
				value: "opt-0-a",
				question_index: 0,
			}),
		);

		slot = { projection: projection(pendingQuestion(1, 2)), connected: true };
		rerender(<SessionScreen sessionId="s1" />);
		expect(screen.getByText("Question 2 of 2")).toBeTruthy();

		// The bug: the card instance was reused, busy stayed true, and every
		// option rendered disabled. A remount (render-site key) resets it.
		const q2 = screen.getByRole("button", { name: /opt-1-a/ });
		expect(q2.hasAttribute("disabled")).toBe(false);

		fireEvent.click(q2);
		await waitFor(() =>
			expect(sendCommand).toHaveBeenLastCalledWith("s1", {
				op: "ask_answer",
				request_id: "req-multi",
				value: "opt-1-a",
				question_index: 1,
			}),
		);
	});
});
