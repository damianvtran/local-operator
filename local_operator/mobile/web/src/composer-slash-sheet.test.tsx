// @vitest-environment happy-dom
//
// The slash sheet's OPEN LIFECYCLE (mobile UX batch 1, U5/U8), pinned on the
// REAL Composer over a mocked catalogue:
//
//   * a pick closes the sheet, and it stays closed while the arguments are
//     typed — it used to re-open on every keystroke (the trigger only asked
//     whether the draft started with `/word`) and steal focus to its ✕;
//   * Escape closes it FOR THE CURRENT DRAFT; typing more of that draft does
//     not bring it back;
//   * a FRESH `/…` draft still opens it, because clearing the draft re-arms
//     the trigger.
//
// U8 rides along: opening from typing focuses the sheet's filter (seeded with
// the token typed so far) instead of the ✕, so type-ahead keeps working.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Composer } from "./components/composer";
import type { SessionProjection, SlashCommand } from "./types";

//: `delete` is the argument-taking shape (a pick FILLS `/delete ` and waits);
//: `copy` runs outright. The split itself is pinned in composer-slash-tap.
const CATALOGUE: SlashCommand[] = [
	{
		name: "delete",
		description: "Delete this conversation for good; asks to confirm",
		aliases: [],
		arguments: "optional",
	},
	{ name: "copy", description: "Copy an agent message", aliases: [], arguments: "none" },
];

vi.mock("./api", () => ({
	getCommands: vi.fn(async () => ({ commands: CATALOGUE })),
	getModels: vi.fn(async () => ({ models: [] })),
	sendCommand: vi.fn(async () => ({ ok: true, detail: "" })),
}));

vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		// The draft is the composer's own state, so the mock is REAL state:
		// typing has to re-render for the sheet to open on the `/` keystroke.
		useDraft: () => useState(""),
	};
});

function projection(): SessionProjection {
	return {
		session_id: "s1",
		pid: 1,
		kind: "tui",
		conversation_name: "Goal",
		cwd: "",
		model_label: "",
		model_selector: "",
		effort: "",
		effort_ladder: [],
		streaming: false,
		activity: "",
		activity_started_s: 0,
		stop_reason: "completed",
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
	} as unknown as SessionProjection;
}

function renderComposer(): HTMLTextAreaElement {
	render(
		<Composer
			pid="p1"
			projection={projection()}
			onOpenModels={() => {}}
			onOpenEffort={() => {}}
			effortOpen={false}
			onCloseEffort={() => {}}
		/>,
	);
	return screen.getByPlaceholderText("Message…") as HTMLTextAreaElement;
}

afterEach(() => {
	cleanup();
	localStorage.clear();
	vi.clearAllMocks();
});

describe("the slash sheet's open lifecycle", () => {
	it("closes on a pick and stays closed while the arguments are typed (U5)", async () => {
		const field = renderComposer();
		fireEvent.change(field, { target: { value: "/delete" } });
		await waitFor(() => expect(screen.getByRole("button", { name: /\/delete/ })).toBeTruthy());

		fireEvent.click(screen.getByRole("button", { name: /\/delete/ }));

		// The fill lands mid-edit ("/delete ") and the sheet is out of the way...
		expect(field.value).toBe("/delete ");
		expect(screen.queryByRole("dialog")).toBeNull();

		// ...and stays out of the way for every keystroke of the argument —
		// the exact sequence the audit measured re-opening the sheet over.
		fireEvent.change(field, { target: { value: "/delete x" } });
		expect(field.value).toBe("/delete x");
		expect(screen.queryByRole("dialog")).toBeNull();
	});

	it("closes on Escape for the current draft, and typing on does not reopen it (U5)", async () => {
		const field = renderComposer();
		fireEvent.change(field, { target: { value: "/de" } });
		await waitFor(() => expect(screen.getByRole("button", { name: /\/delete/ })).toBeTruthy());

		fireEvent.keyDown(document.body, { key: "Escape" });
		await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());

		// Still the command TOKEN (no space yet) — the strict half of the rule:
		// once closed for this draft, typing does not bring it back.
		fireEvent.change(field, { target: { value: "/dele" } });
		expect(screen.queryByRole("dialog")).toBeNull();

		// A fresh draft re-arms the trigger: cleared, then `/` again.
		fireEvent.change(field, { target: { value: "" } });
		fireEvent.change(field, { target: { value: "/" } });
		await waitFor(() => expect(screen.getByRole("dialog")).toBeTruthy());
	});

	it("still opens on a fresh `/x` draft (the control for the two rules above) (U5)", async () => {
		const field = renderComposer();
		fireEvent.change(field, { target: { value: "/" } });
		await waitFor(() => expect(screen.getByRole("dialog")).toBeTruthy());

		// Refining the token keeps it open...
		fireEvent.change(field, { target: { value: "/del" } });
		expect(screen.getByRole("dialog")).toBeTruthy();
	});

	it("focuses the filter when it opens from typing, so type-ahead keeps filtering (U8)", async () => {
		const field = renderComposer();
		fireEvent.change(field, { target: { value: "/de" } });
		await waitFor(() => expect(screen.getByRole("dialog")).toBeTruthy());

		const filter = screen.getByPlaceholderText("filter commands") as HTMLInputElement;
		// Seeded with the token typed so far, and the caret is IN it — not on
		// the ✕, which moved focus out of every text field (and closes the
		// software keyboard on a device).
		expect(filter.value).toBe("de");
		expect(document.activeElement).toBe(filter);
	});
});
