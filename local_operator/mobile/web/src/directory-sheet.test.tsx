// @vitest-environment happy-dom
//
// The composer's working-directory chip and the sheet it opens.
//
// This is the second half of the operator's ask: a new session is ONE TAP and
// the daemon resolves the directory, so the place a user can change it has to
// live in the conversation itself — the chip above the composer. What this file
// pins is the wiring the daemon cannot be asked about: that the chip shows the
// SESSION's directory (not a local echo), that a tap offers the rows the daemon
// publishes, that a commit posts the route with the picked path, and that a
// refusal is SHOWN with the sheet left open rather than swallowed.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { WorkingDirectoryChip } from "./components/directory-sheet";

const changeDirectory = vi.fn();
const getDirectories = vi.fn();

vi.mock("./api", () => ({
	getDirectories: (...args: unknown[]) => getDirectories(...args),
	changeDirectory: (...args: unknown[]) => changeDirectory(...args),
}));

const SESSION = "abcd1234ef56";

/** Open the sheet from the chip. The chip's own accessible name is
    "working directory: <path>" and a sheet ROW is named by its path, so the
    query is deliberately the exact label rather than a substring — `/working
    directory/` also matches rows that name a directory under one. */
async function openSheet() {
	fireEvent.click(
		await screen.findByRole("button", { name: "working directory: /Users/tester/src/app" }),
	);
}

afterEach(() => {
	// This suite is not run with vitest globals, so testing-library's own
	// auto-cleanup never registers: every render would otherwise stack its chip
	// in the document and the next test's queries would find two.
	cleanup();
	document.body.innerHTML = "";
});

beforeEach(() => {
	vi.clearAllMocks();
	getDirectories.mockResolvedValue({
		home: "/Users/tester",
		recent: ["/Users/tester/projects/other"],
		tmp: "/private/tmp",
		default: "/Users/tester/src/app",
	});
	changeDirectory.mockResolvedValue({ ok: true, pid: 99, session_id: SESSION });
});

/** Render the chip and settle its `home` fetch, so `~` shortening is real. */
async function renderChip(cwd = "/Users/tester/src/app") {
	const onMoved = vi.fn();
	render(<WorkingDirectoryChip sessionId={SESSION} cwd={cwd} onMoved={onMoved} />);
	await waitFor(() => expect(getDirectories).toHaveBeenCalled());
	return onMoved;
}

describe("the working-directory chip", () => {
	it("shows the session's own directory, shortened against home", async () => {
		await renderChip();
		expect(await screen.findByText("~/src/app")).toBeTruthy();
	});

	it("falls back to the raw path while home is still unknown", async () => {
		// Never settles: the chip must not wait for a cosmetic fetch to name the
		// directory it already knows from the session's own projection.
		getDirectories.mockReturnValue(new Promise(() => undefined));
		render(<WorkingDirectoryChip sessionId={SESSION} cwd="/Users/tester/src/app" onMoved={vi.fn()} />);
		expect(await screen.findByText("/Users/tester/src/app")).toBeTruthy();
	});

	it("offers the current directory, home, tmp and the recents when opened", async () => {
		await renderChip();
		await openSheet();

		expect(await screen.findByText("current")).toBeTruthy();
		expect(screen.getByText("home")).toBeTruthy();
		expect(screen.getByText("tmp")).toBeTruthy();
		expect(screen.getByText("recent")).toBeTruthy();
		expect(screen.getByText("~")).toBeTruthy();
		expect(screen.getByText("/private/tmp")).toBeTruthy();
		// The recents row is a directory the session is NOT already in, so this
		// also proves the list is deduped against the current row rather than
		// blindly concatenated.
		expect(screen.getByText("~/projects/other")).toBeTruthy();
	});

	it("commits a row against the session's directory route and closes", async () => {
		const onMoved = await renderChip();
		await openSheet();
		// The tmp row: the only row that is neither current nor home, so this
		// asserts the row's own path is what travels.
		fireEvent.click(await screen.findByText("/private/tmp"));

		await waitFor(() =>
			expect(changeDirectory).toHaveBeenCalledWith(SESSION, "/private/tmp"),
		);
		await waitFor(() => expect(onMoved).toHaveBeenCalledWith("/private/tmp"));
		// Success closes the sheet: the session repaints from its own stream, and
		// the composer's optimistic bridge carries the new path until it does.
		await waitFor(() => expect(screen.queryByText("/private/tmp")).toBeNull());
	});

	it("commits a typed path through its own button", async () => {
		const onMoved = await renderChip();
		await openSheet();
		const input = await screen.findByPlaceholderText("or type another path…");
		fireEvent.change(input, { target: { value: "/Users/tester/scratch" } });
		fireEvent.click(screen.getByRole("button", { name: "change" }));

		await waitFor(() =>
			expect(changeDirectory).toHaveBeenCalledWith(SESSION, "/Users/tester/scratch"),
		);
		expect(onMoved).toHaveBeenCalledWith("/Users/tester/scratch");
	});

	it("shows the daemon's refusal sentence and keeps the sheet open", async () => {
		const sentence = "This session already has messages, so its working directory can't change.";
		changeDirectory.mockRejectedValue(new Error(sentence));
		const onMoved = await renderChip();
		await openSheet();
		fireEvent.click(await screen.findByText("/private/tmp"));

		expect((await screen.findByRole("alert")).textContent).toBe(sentence);
		// STILL OPEN, with the choices on screen: the reader has to be able to try
		// another row, and the reason has to outlive the attempt.
		expect(screen.getByText("/private/tmp")).toBeTruthy();
		expect(onMoved).not.toHaveBeenCalled();
	});
});
