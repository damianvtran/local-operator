// @vitest-environment happy-dom
//
// Only an uncovered final result in the focused selected conversation is read.
import { act, cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SessionScreen } from "./screens/session-view";
import type { SessionProjection } from "./types";

vi.mock("./api", () => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	/* Never resolves: keeps AgentScreen on its loading branch so the test
	   exercises the mount path without needing a full detail fixture. */
	getSubagentDetail: vi.fn(() => new Promise(() => undefined)),
	imageUrl: vi.fn(() => ""),
	getCommands: vi.fn(async () => ({ commands: [] })),
	getModels: vi.fn(async () => ({ models: [] })),
	sendCommand: vi.fn(async () => ({ ok: true, detail: "" })),
	/* Settled by default: the answer the handshake is defined by. Tests that need
	   the UNREAD answer (a superseded token, or an older daemon) stage their own. */
	markSessionSeen: vi.fn(async () => ({
		ok: true,
		attention: {
			conversation_id: "session/s1",
			completion_token: "token-a",
			anchor_id: "result-a",
			kind: "complete",
			unseen: false,
			revision: [1, 1],
		},
	})),
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
		useDraft: vi.fn(() => ["", () => {}]),
		clearSessionUnseen: vi.fn(actual.clearSessionUnseen),
	};
});

function projection(): SessionProjection {
	return {
		session_id: "s1",
		pid: 1,
		kind: "tui",
		conversation_name: "Seen",
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
	} satisfies SessionProjection;
}

const { markSessionSeen } = await import("./api");
const { clearSessionUnseen } = await import("./store");

afterEach(() => {
	cleanup();
	vi.clearAllMocks();
	vi.restoreAllMocks();
	vi.useRealTimers();
	slot = { projection: null, connected: true };
});

function focusedResult() {
	vi.useFakeTimers();
	const p = projection();
	p.transcript = [{ id: "result-a", kind: "assistant", text: "Finished result", final: true, text_complete: true,
		tool_call_id: "", tool_name: "", tool_state: "done", summary: "", intent: "",
		diff_added: 0, diff_removed: 0, elapsed_s: 0, error: "", details: {} }];
	p.attention = { conversation_id: "session/s1", completion_token: "token-a", anchor_id: "result-a", kind: "complete", unseen: true, revision: [1, 0] };
	slot = { projection: p, connected: true };
	vi.spyOn(document, "hasFocus").mockReturnValue(true);
	vi.spyOn(document, "visibilityState", "get").mockReturnValue("visible");
	vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue(new DOMRect(0, 0, 200, 100));
	vi.spyOn(document, "elementFromPoint").mockImplementation(() => document.querySelector("[data-completion-anchor]"));
	return p;
}

async function sample() {
	await act(async () => { vi.advanceTimersByTime(600); });
}

describe("SessionScreen seen handshake", () => {
	it("acknowledges the rendered token once without optimistic clearing", async () => {
		focusedResult();
		render(<SessionScreen sessionId="s1" />);
		await sample();
		expect(markSessionSeen).toHaveBeenCalledWith("s1", "token-a");
		await sample();
		expect(markSessionSeen).toHaveBeenCalledTimes(1);
		expect(clearSessionUnseen).not.toHaveBeenCalled();
	});

	it("keeps polling while the answer does not say the conversation is read", async () => {
		// The shipped daemon answered a SUPERSEDED token with a 200 whose body
		// still said `unseen` (see the findings file). Latched on the resolved
		// call, the "new" mark never cleared, because nothing ever acknowledged
		// the completion the projection had moved on to. `unseen` is the verdict;
		// anything else keeps the poll running.
		const p = focusedResult();
		const seen = vi.mocked(markSessionSeen);
		seen.mockResolvedValue({
			ok: true,
			attention: { ...p.attention!, unseen: true },
		});
		render(<SessionScreen sessionId="s1" />);
		await sample();
		expect(seen).toHaveBeenCalledTimes(1);
		await sample();
		expect(seen).toHaveBeenCalledTimes(2);
		expect(clearSessionUnseen).not.toHaveBeenCalled();

		// The answer catches up: the receipt belongs to this conversation and the
		// daemon confirms it. Now -- and only now -- the attempt settles.
		seen.mockResolvedValue({
			ok: true,
			attention: { ...p.attention!, unseen: false, revision: [1, 1] },
		});
		await sample();
		const settledCalls = seen.mock.calls.length;
		await sample();
		expect(seen.mock.calls.length).toBe(settledCalls);
		expect(seen).toHaveBeenLastCalledWith("s1", "token-a");
	});

	it("keeps polling when the daemon refuses the receipt", async () => {
		// A refusal is not a read. An older daemon refuses nothing, but the
		// current one answers a superseded token with 409, and the honest
		// response to that is to wait for the token the projection names next --
		// never to stop, and never to clear the mark optimistically.
		focusedResult();
		const seen = vi.mocked(markSessionSeen);
		seen.mockRejectedValue(
			Object.assign(new Error("completion token superseded by a newer completion"), {
				status: 409,
				code: "superseded_completion_token",
			}),
		);
		render(<SessionScreen sessionId="s1" />);
		await sample();
		await sample();
		expect(seen.mock.calls.length).toBeGreaterThan(1);
		expect(clearSessionUnseen).not.toHaveBeenCalled();
	});

	it("bounds a refusal storm instead of retrying it forever", async () => {
		// The other half of "keep polling": a refusal that cannot resolve on its
		// own (a projection stuck on a superseded token) must not be re-attempted
		// at the flat cadence for as long as the tab is open, in silence. Three
		// consecutive refusals back the cadence off and one line explains why; a
		// fresh token re-runs the effect with the counter at zero.
		focusedResult();
		const seen = vi.mocked(markSessionSeen);
		seen.mockRejectedValue(
			Object.assign(new Error("completion token superseded by a newer completion"), {
				status: 409,
				code: "superseded_completion_token",
			}),
		);
		const warned = vi.spyOn(console, "warn").mockImplementation(() => undefined);
		render(<SessionScreen sessionId="s1" />);
		await sample();
		await sample();
		await sample();
		expect(warned).toHaveBeenCalledTimes(1);
		const attempts = seen.mock.calls.length;
		await sample();
		expect(seen.mock.calls.length).toBe(attempts);
		expect(clearSessionUnseen).not.toHaveBeenCalled();
	});

	it.each(["hidden", "blurred", "covered", "scrollback", "streaming", "disconnected", "truncated", "unknown completeness"])("does not acknowledge %s results", async (reason) => {
		const p = focusedResult();
		if (reason === "hidden") vi.spyOn(document, "visibilityState", "get").mockReturnValue("hidden");
		if (reason === "blurred") vi.spyOn(document, "hasFocus").mockReturnValue(false);
		if (reason === "covered") vi.spyOn(document, "elementFromPoint").mockReturnValue(document.body);
		if (reason === "scrollback") vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockReturnValue(new DOMRect(0, 0, 200, innerHeight + 100));
		if (reason === "streaming") p.streaming = true;
		if (reason === "truncated") p.transcript[0].text_complete = false;
		if (reason === "unknown completeness") delete p.transcript[0].text_complete;
		if (reason === "disconnected") slot.connected = false;
		render(<SessionScreen sessionId="s1" />);
		await sample();
		expect(markSessionSeen).not.toHaveBeenCalled();
	});

	it("does not acknowledge an empty mounted root route", async () => {
		slot = { projection: projection(), connected: true };
		render(<SessionScreen sessionId="s1" />);
		expect(markSessionSeen).not.toHaveBeenCalled();
		expect(clearSessionUnseen).not.toHaveBeenCalled();
	});

	it("never acknowledges the parent while a child route is loading", async () => {
		slot = { projection: projection(), connected: true };
		render(<SessionScreen sessionId="s1" jobId="job-1" />);
		expect(markSessionSeen).not.toHaveBeenCalled();
		expect(clearSessionUnseen).not.toHaveBeenCalled();
	});
});


describe("unresolved receipt budget", () => {
 for (const outcome of ["unread", "wrong-token", "alternating"] as const) {
  it(`bounds ${outcome} replies and resets only for a new token`, async () => {
   const p = focusedResult();
   const seen = vi.mocked(markSessionSeen);
   const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
   let calls = 0;
   seen.mockImplementation(async () => {
    calls += 1;
    if (outcome === "alternating" && calls % 2) throw new Error("refused");
    return { ok:true, attention: { ...p.attention!, unseen:outcome !== "wrong-token", completion_token:outcome === "wrong-token" ? "other" : "token-a" } };
   });
   const view = render(<SessionScreen sessionId="s1" />);
   for (let tick = 0; tick < 240; tick++) await act(async () => { vi.advanceTimersByTime(500); });
   expect(calls).toBeLessThanOrEqual(10);
   expect(warn).toHaveBeenCalledTimes(1);
   const previous = calls;
   slot = { ...slot, projection: { ...p, attention:{...p.attention!,completion_token:"token-b"} } };
   view.rerender(<SessionScreen sessionId="s1" />);
   await sample();
   expect(calls).toBe(previous + 1);
  });
 }
});
