// @vitest-environment happy-dom
//
// Seen handshake (spec §3): opening a session — the root route OR a subagent
// route, both of which render through SessionScreen — must fire
// POST /api/sessions/{id}/seen once on mount and optimistically clear the
// client's `unseen` copy so back-navigation never flashes a stale `new` mark.
import { cleanup, render, waitFor } from "@testing-library/react";
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
	slot = { projection: null, connected: true };
});

describe("SessionScreen seen handshake", () => {
	it("fires POST /seen once and clears the store flag on the root route", async () => {
		slot = { projection: projection(), connected: true };
		render(<SessionScreen sessionId="s1" />);
		await waitFor(() => expect(markSessionSeen).toHaveBeenCalledWith("s1"));
		expect(markSessionSeen).toHaveBeenCalledTimes(1);
		expect(clearSessionUnseen).toHaveBeenCalledWith("s1");
	});

	it("fires POST /seen on the agent route too", async () => {
		slot = { projection: projection(), connected: true };
		render(<SessionScreen sessionId="s1" jobId="job-1" />);
		await waitFor(() => expect(markSessionSeen).toHaveBeenCalledWith("s1"));
		expect(markSessionSeen).toHaveBeenCalledTimes(1);
		expect(clearSessionUnseen).toHaveBeenCalledWith("s1");
	});
});
