// @vitest-environment happy-dom
//
// Regression: the phone's history list DROPPED the fork mark entirely. Both
// daemon payloads (`_past_sessions`, `_search_sessions`) built their JSON as
// `{id, name, mtime}` from rows that already carried `forked`, so a fork still
// wearing its parent's title rendered as a byte-identical row beside the
// parent — same name, same age — exactly the twin-row confusion the TUI's
// /resume picker had just stopped showing. The fix has to reach BOTH sides, so
// this renders the real screen rather than asserting on the payload.
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { PastSessionsScreen } from "./screens/past-sessions";

const NOW = Date.now() / 1000;

/** A fork and its parent wearing the identical inherited title. */
const rows = [
	{
		id: "a1b2c3d4e5f6",
		name: "Refactor the YAML loader to stream anchors",
		mtime: NOW,
		forked: true,
	},
	{
		id: "9f8e7d6c5b4a",
		name: "Refactor the YAML loader to stream anchors",
		mtime: NOW - 120,
	},
];

vi.mock("./api", () => ({
	searchSessions: vi.fn(async () => ({ sessions: rows, query: "" })),
	resumeSession: vi.fn(),
}));

describe("the phone's history list", () => {
	beforeEach(() => {
		vi.clearAllMocks();
	});

	it("tags a fork that is still wearing its parent's title", async () => {
		render(<PastSessionsScreen />);

		expect(await screen.findByText("[fork]")).toBeTruthy();
		// Both rows still READ the same, which is why the tag has to exist.
		expect(
			(await screen.findAllByText(/Refactor the YAML loader/)).length,
		).toBe(2);
	});

	it("leaves an ordinary row unmarked", async () => {
		render(<PastSessionsScreen />);
		await screen.findByText("[fork]");

		// One tag for two rows: the parent is not a fork and must not be tagged.
		expect(screen.getAllByText("[fork]").length).toBe(1);
	});
});
