// @vitest-environment happy-dom
//
// The outstanding-asks chip on a session row (design §4/§5.0). It is a SEPARATE
// statement from the approval state — the agent keeps working while an ask is
// queued — so it is asserted beside `needs_attention` rather than folded into
// it, and it is absent (not "0") at zero.
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

function summary(over: Partial<SessionSummary> = {}): SessionSummary {
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

afterEach(() => {
	cleanup();
	sessionList = [];
});

describe("session row outstanding asks", () => {
	it("states the count when asks are waiting", () => {
		sessionList = [summary({ asks_open: 2 })];
		render(<SessionListScreen />);
		/* The chip states ASKS, the field's own unit (agent review round 1, R3):
		   `asks_open` counts open asks and it was labelled with the bar's unit. */
		expect(screen.getByText("2 asks")).toBeTruthy();
	});

	it("says it in the singular for one, and is absent at zero", () => {
		sessionList = [summary({ session_id: "a", conversation_name: "One", asks_open: 1 })];
		render(<SessionListScreen />);
		expect(screen.getByText("1 ask")).toBeTruthy();
		cleanup();
		sessionList = [summary({ session_id: "b", conversation_name: "None", asks_open: 0 })];
		render(<SessionListScreen />);
		expect(screen.queryByText("0 questions")).toBeNull();
	});

	it("is absent while the runtime cannot report asks at all", () => {
		sessionList = [summary()];
		render(<SessionListScreen />);
		expect(screen.queryByText(/question/)).toBeNull();
	});

	it("does not borrow the approval state's reading", () => {
		sessionList = [summary({ asks_open: 1 })];
		render(<SessionListScreen />);
		/* No `needs_attention` was set: an outstanding ask is not a blocked run,
		   so the row must not carry the decision state's ink or words. */
		expect(screen.queryByText(/needs/i)).toBeNull();
	});
});
