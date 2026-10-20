// @vitest-environment happy-dom
//
// Conversation tags on the PAST SESSIONS list: the same static line as the live
// list, and deliberately nothing else.
//
// This surface has no long-press action sheet and no assignment control: a past
// conversation is reopened elsewhere, and the relay's write route takes a live
// session id. A tag control here would either write to the wrong thing or be a
// dead button, so "read only" is asserted, not assumed.
import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { PastSessionsScreen } from "./screens/past-sessions";
import type { PastSession } from "./types";

const searchSessions = vi.fn(async (_q: string, _limit?: number) => ({
	sessions: [] as PastSession[],
	query: "",
}));

vi.mock("./api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./api")>();
	return {
		...actual,
		searchSessions: (q: string, limit?: number) => searchSessions(q, limit),
		resumeSession: vi.fn(),
	};
});

function past(over: Partial<PastSession>): PastSession {
	return {
		id: "6714def86197",
		name: "Hello from the mock provider",
		mtime: 1_792_420_990,
		forked: false,
		...over,
	};
}

afterEach(() => {
	cleanup();
	searchSessions.mockClear();
});

describe("past rows", () => {
	it("renders the tag line with the same two-label budget as the live list", async () => {
		searchSessions.mockResolvedValue({
			sessions: [
				past({
					tags: [
						{ id: "work", name: "Work" },
						{ id: "research", name: "Research" },
						{ id: "home", name: "Home" },
					],
					primary: "work",
				}),
			],
			query: "",
		});
		render(<PastSessionsScreen />);
		const row = await screen.findByText("Hello from the mock provider");
		const line = within(row.closest("div") as HTMLElement).getByTestId(
			"session-tags",
		);
		expect(line.textContent).toContain("Work");
		expect(line.textContent).toContain("Research");
		expect(line.textContent).not.toContain("Home");
		expect(line.textContent).toContain("+1");
		expect(within(line).getByText("primary")).toBeTruthy();
	});

	it("renders no line for an untagged past conversation", async () => {
		searchSessions.mockResolvedValue({ sessions: [past({})], query: "" });
		render(<PastSessionsScreen />);
		await screen.findByText("Hello from the mock provider");
		expect(screen.queryByTestId("session-tags")).toBeNull();
	});

	it("offers no assignment control: the line is not interactive", async () => {
		searchSessions.mockResolvedValue({
			sessions: [
				past({
					tags: [{ id: "work", name: "Work" }],
					primary: "work",
				}),
			],
			query: "",
		});
		render(<PastSessionsScreen />);
		const line = await screen.findByTestId("session-tags");
		expect(within(line).queryByRole("button")).toBeNull();
		expect(within(line).queryByRole("checkbox")).toBeNull();
		expect(screen.queryByRole("button", { name: "Tags…" })).toBeNull();
	});
});
