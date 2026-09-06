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
