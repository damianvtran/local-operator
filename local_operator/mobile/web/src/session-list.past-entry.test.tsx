// @vitest-environment happy-dom
//
// The PAST SESSIONS entry point (mobile UX batch 1, U10). `#/past` is a
// searchable, resumable history screen — and until this batch nothing in the
// app navigated to it: a source-wide search found only the router, the screen,
// daemon comments and a test that reached it by assigning `location.hash`
// directly. So the wiring is what this file pins: a `past` control exists in
// the footer (beside the other top-level entries), and tapping it lands on the
// route — which is what a thumb actually does.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SessionListScreen } from "./screens/session-list";
import type { SessionSummary } from "./types";

vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useSessions: () => ({ sessions: [] as SessionSummary[], connected: true }),
		retainSessionListStream: () => () => {},
	};
});

vi.mock("./api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./api")>();
	return {
		...actual,
		getDirectories: vi.fn(async () => ({ home: "", recent: [] })),
	};
});

afterEach(() => {
	cleanup();
	vi.clearAllMocks();
});

describe("past sessions entry point", () => {
	it("navigates to the past sessions route from the list footer", () => {
		render(<SessionListScreen />);

		const entry = screen.getByRole("button", { name: "past" });
		fireEvent.click(entry);

		expect(window.location.hash).toBe("#/past");
	});
});
