// @vitest-environment happy-dom
//
// THE `$SKILL` PICKER, ON THE REAL COMPOSER (SKM-D5/D6).
//
// These pin the surface a user touches, not a copy of its maths: the real
// `Composer` over a mocked catalogue, a real draft (so typing re-renders), and
// real caret moves. The grammar itself is pinned by `lib/skill-token.test.ts`
// against the generated TUI fixture; this file pins the WIRING that grammar
// depends on:
//
//  - the tray opens only when the relay advertises `skill_invocation`;
//  - a slash draft wins, so `/team x $skill` never shows a skill list;
//  - a tap inserts the token's span and leaves the caret in the field;
//  - hardware Enter still SENDS while the tray is open;
//  - a receipt with skipped skills paints the one keyed notice;
//  - a fresh send asks for expansion (`expand_skills`) only when it carries `$`.
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Composer, isSlashDraft } from "./composer";
import type { CommandReceipt } from "../api";
import type { SessionProjection, SkillEntry } from "../types";

// Sorted by name, as `GET /api/skills` answers: the composer never re-sorts.
const SKILLS: SkillEntry[] = [
	{ name: "pathfinder", description: "Walk a directory tree", hidden: true },
	{ name: "planning", description: "Plan the work in phases", hidden: false },
	{ name: "research", description: "Read the sources before changing code", hidden: false },
];

const sendCommand = vi.hoisted(() =>
	vi.fn(
		async (..._args: unknown[]): Promise<CommandReceipt> => ({ ok: true, detail: "" }),
	),
);

vi.mock("../api", () => ({
	getCommands: vi.fn(async () => ({ commands: [] })),
	getSkills: vi.fn(async () => ({ skills: SKILLS })),
	getModels: vi.fn(async () => ({ models: [] })),
	getDirectories: vi.fn(async () => ({ home: "/Users/tester", recent: [], tmp: "" })),
	changeDirectory: vi.fn(async () => ({ ok: true, pid: 1, session_id: "s1" })),
	sendCommand: (...args: unknown[]) => sendCommand(...args),
	transcribeAudio: vi.fn(),
	HttpError: class HttpError extends Error {
		constructor(
			readonly status: number,
			message: string,
			readonly code = "",
		) {
			super(message);
		}
	},
}));

let capabilities: { skill_invocation?: { available: boolean } } | null = {
	skill_invocation: { available: true },
};

vi.mock("../store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../store")>();
	return {
		...actual,
		useCapabilities: () => capabilities,
		// The draft is the composer's own state: typing must re-render for the
		// tray to open on the `$` keystroke.
		useDraft: () => useState(""),
	};
});

function projection(): SessionProjection {
	return {
		session_id: "s1",
		pid: 1,
		kind: "tui",
		conversation_name: "Skills",
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

function mount() {
	render(
		<Composer
			ended={false}
			pid="p1"
			projection={projection()}
			onOpenModels={() => {}}
			onReturnToModels={() => {}}
			onOpenEffort={() => {}}
			effortOpen={false}
			onCloseEffort={() => {}}
		/>,
	);
	return screen.getByPlaceholderText("Ask anything…") as HTMLTextAreaElement;
}

/** Type `value` and put the caret at `caret` — what a phone reports as it types. */
function type(field: HTMLTextAreaElement, value: string, caret = value.length) {
	fireEvent.change(field, { target: { value } });
	field.setSelectionRange(caret, caret);
	fireEvent.select(field);
}

function tray(): HTMLElement | null {
	return document.querySelector("[data-skill-tray]");
}

afterEach(() => {
	cleanup();
	localStorage.clear();
	vi.clearAllMocks();
	capabilities = { skill_invocation: { available: true } };
});

describe("the slash predicate send() and the picker share", () => {
	it("is a leading /, on one line, with no attachment", () => {
		expect(isSlashDraft("/team ops", 0)).toBe(true);
		expect(isSlashDraft("  /compact", 0)).toBe(true);
		expect(isSlashDraft("/team\nops", 0)).toBe(false);
		expect(isSlashDraft("/team", 1)).toBe(false);
		expect(isSlashDraft("fix /team", 0)).toBe(false);
		expect(isSlashDraft("$research", 0)).toBe(false);
	});
});

describe("the skill tray", () => {
	it("opens on a leading $ and lists the catalogue, hidden skills included", async () => {
		const field = mount();
		type(field, "$", 1);
		await waitFor(() => expect(tray()).toBeTruthy());
		const rows = within(tray() as HTMLElement).getAllByRole("button");
		expect(rows.map((row) => row.textContent)).toEqual([
			expect.stringContaining("$pathfinder"),
			expect.stringContaining("$planning"),
			expect.stringContaining("$research"),
		]);
		// The hidden skill is offered and MARKED: typing is what `hide` leaves open.
		expect(screen.getByText("hidden")).toBeTruthy();
	});

	it("narrows to a case-insensitive prefix, including a phone-capitalised R", async () => {
		const field = mount();
		type(field, "fix $Re", 7);
		await waitFor(() => expect(tray()).toBeTruthy());
		expect(within(tray() as HTMLElement).getAllByRole("button")).toHaveLength(1);
		expect(screen.getByText("$research")).toBeTruthy();
	});

	it("stays closed for an inline uppercase $PATH", async () => {
		const field = mount();
		type(field, "translate to $PATH", 17);
		await Promise.resolve();
		expect(tray()).toBeNull();
	});

	it("stays closed for a bare inline $", async () => {
		const field = mount();
		type(field, "costs $", 7);
		await Promise.resolve();
		expect(tray()).toBeNull();
	});

	it("lets a slash draft win — /team x $res shows no skill list", async () => {
		const field = mount();
		type(field, "/team x $res", 12);
		await Promise.resolve();
		expect(tray()).toBeNull();
	});

	it("never opens when the relay does not advertise the capability", async () => {
		capabilities = null;
		const field = mount();
		type(field, "$", 1);
		await Promise.resolve();
		expect(tray()).toBeNull();
	});

	it("a tap replaces ONLY the token's span and leaves the caret in the field", async () => {
		const field = mount();
		type(field, "please $res now", 10);
		await waitFor(() => expect(tray()).toBeTruthy());
		const row = within(tray() as HTMLElement).getByText("$research");
		fireEvent.pointerDown(row, { pointerId: 1, clientX: 12, clientY: 400 });
		fireEvent.pointerUp(row, { pointerId: 1, clientX: 12, clientY: 400 });

		expect(field.value).toBe("please $research now");
		expect(document.activeElement).toBe(field);
		expect(field.selectionStart).toBe(17);
		expect(field.selectionEnd).toBe(17);
	});

	it("restores the caret even when the pick leaves the text byte-identical", async () => {
		const field = mount();
		type(field, "please $research now", 10);
		await waitFor(() => expect(tray()).toBeTruthy());
		const row = within(tray() as HTMLElement).getByText("$research");
		fireEvent.pointerDown(row, { pointerId: 3, clientX: 12, clientY: 400 });
		fireEvent.pointerUp(row, { pointerId: 3, clientX: 12, clientY: 400 });

		expect(field.value).toBe("please $research now");
		expect(field.selectionStart).toBe(17);
		await waitFor(() => expect(tray()).toBeNull());
	});

	it("does not pick when the finger is dragging the list to scroll", async () => {
		const field = mount();
		type(field, "please $res now", 10);
		await waitFor(() => expect(tray()).toBeTruthy());
		const row = within(tray() as HTMLElement).getByText("$research");
		fireEvent.pointerDown(row, { pointerId: 2, clientX: 12, clientY: 500 });
		fireEvent.pointerUp(row, { pointerId: 2, clientX: 12, clientY: 460 });
		expect(field.value).toBe("please $res now");
	});

	it("hardware Enter still SENDS while the tray is open", async () => {
		const field = mount();
		type(field, "$research", 9);
		await waitFor(() => expect(tray()).toBeTruthy());
		fireEvent.keyDown(field, { key: "Enter" });

		await waitFor(() => expect(sendCommand).toHaveBeenCalledOnce());
		const sent = sendCommand.mock.calls[0]?.[1] as Record<string, unknown>;
		expect(sent.op).toBe("prompt");
		expect(sent.text).toBe("$research");
		expect(sent.expand_skills).toBe(true);
	});

	it("does not ask for expansion on a message with no $", async () => {
		const field = mount();
		type(field, "plain prose", 11);
		fireEvent.keyDown(field, { key: "Enter" });

		await waitFor(() => expect(sendCommand).toHaveBeenCalledOnce());
		const sent = sendCommand.mock.calls[0]?.[1] as Record<string, unknown>;
		expect("expand_skills" in sent).toBe(false);
	});

	it("paints the one keyed notice when the receipt reports a skipped skill", async () => {
		sendCommand.mockResolvedValueOnce({
			ok: true,
			detail: "",
			skills: { attached: [], skipped: ["empty-one"] },
		});
		const field = mount();
		type(field, "$empty-one", 10);
		fireEvent.keyDown(field, { key: "Enter" });

		// The names are wrapped in bidi isolates at the call site (FSI/PDI), so the
		// assertion is on the sentence halves the translation owns.
		await waitFor(() => expect(screen.getByText(/^Sent without/)).toBeTruthy());
		expect(screen.getByText(/that skill has no content to attach/)).toBeTruthy();
		expect(screen.getByText(/^Sent without/).textContent).toContain("$empty-one");
	});
});
