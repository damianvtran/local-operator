// @vitest-environment happy-dom
//
// The Projects ENTRY POINT on the sessions screen. The sheet's own behaviour
// lives in `projects-sheet.test.tsx`; what this file pins is the wiring the
// reader actually touches: a `projects` control exists in the footer, tapping
// it opens the sheet, and dismissing the sheet closes it — three things a
// refactor of the footer could break while every sheet test stayed green.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SessionListScreen } from "./screens/session-list";
import type { SessionSummary } from "./types";

const getProjects = vi.fn();

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
		getProjects: (...args: unknown[]) => getProjects(...args),
	};
});

afterEach(() => {
	cleanup();
	vi.clearAllMocks();
});

describe("projects entry point", () => {
	it("opens the Projects sheet from the footer and closes it again", async () => {
		getProjects.mockResolvedValue({ projects: [] });
		render(<SessionListScreen />);

		const entry = screen.getByRole("button", { name: "projects" });
		expect(entry).toBeTruthy();
		expect(screen.queryByText("no projects yet")).toBeNull();

		fireEvent.click(entry);
		expect(await screen.findByText("no projects yet")).toBeTruthy();
		expect(getProjects).toHaveBeenCalledTimes(1);

		fireEvent.click(screen.getByRole("button", { name: "close sheet" }));
		expect(screen.queryByText("no projects yet")).toBeNull();
	});

	it("hands focus back to the footer control when the sheet closes", async () => {
		/* `Sheet` restores focus to `returnFocusRef` (falling back to whatever was
		   focused when it opened). The projects sheet was the one caller that
		   passed no ref, so on a phone — where `document.activeElement` is
		   usually `<body>` — closing it dropped the reader at the top of the
		   document with no way back to the control they pressed. */
		getProjects.mockResolvedValue({ projects: [] });
		render(<SessionListScreen />);

		const entry = screen.getByRole("button", { name: "projects" });
		fireEvent.click(entry);
		await screen.findByText("no projects yet");
		fireEvent.click(screen.getByRole("button", { name: "close sheet" }));

		await waitFor(() => expect(document.activeElement).toBe(entry));
	});
});
