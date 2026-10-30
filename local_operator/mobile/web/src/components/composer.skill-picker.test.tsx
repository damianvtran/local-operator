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
//  - a receipt with skipped skills paints the one keyed notice, which the next
//    draft clears;
//  - a fresh send asks for expansion (`expand_skills`) only when it carries `$`;
//  - the tray is CAPPED IN COLUMN UNITS, dismisses on an outside tap, a blur or
//    Escape, stays shut on a restored draft, and opens mid-composition;
//  - a catalogue name the `$` grammar cannot express is never offered.
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Composer, isSlashDraft } from "./composer";
import { HttpError } from "../api";
import type { CommandReceipt } from "../api";
import type { SessionProjection, SkillEntry } from "../types";

// Sorted by name, as `GET /api/skills` answers: the composer never re-sorts.
// `ux spaced` is the case the mention grammar cannot express (UX review U3):
// the token ends at whitespace, so offering it would promise a pick that can
// never fire.
const SKILLS: SkillEntry[] = [
	{ name: "pathfinder", description: "Walk a directory tree", hidden: true },
	{ name: "planning", description: "Plan the work in phases", hidden: false },
	{ name: "research", description: "Read the sources before changing code", hidden: false },
	{ name: "ux spaced", description: "A name no mention can reach", hidden: false },
];

const sendCommand = vi.hoisted(() =>
	vi.fn(
		async (..._args: unknown[]): Promise<CommandReceipt> => ({ ok: true, detail: "" }),
	),
);

const getSkills = vi.hoisted(() =>
	vi.fn(async (_pid: string): Promise<{ skills: SkillEntry[] }> => ({ skills: SKILLS })),
);

vi.mock("../api", () => ({
	getCommands: vi.fn(async () => ({ commands: [] })),
	getSkills: (...args: unknown[]) => getSkills(...(args as [string])),
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

/* The draft a freshly mounted composer starts with. Non-empty is the RESTORED
   case (a draft that survived navigation): the text is there and no caret has
   been reported yet, which is the state UX review U2 says must stay closed. */
let draftSeed = "";

vi.mock("../store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../store")>();
	return {
		...actual,
		useCapabilities: () => capabilities,
		// The draft is the composer's own state: typing must re-render for the
		// tray to open on the `$` keystroke.
		useDraft: () => useState(draftSeed),
	};
});

function projection(pending: SessionProjection["pending"] = null): SessionProjection {
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
		pending,
		pending_count: pending === null ? 0 : 1,
		usage: {},
		version: 1,
	} as unknown as SessionProjection;
}

function TestComposer({
	pid = "p1",
	pending = null,
}: {
	pid?: string;
	pending?: SessionProjection["pending"];
}) {
	return (
		<Composer
			ended={false}
			pid={pid}
			projection={projection(pending)}
			onOpenModels={() => {}}
			onReturnToModels={() => {}}
			onOpenEffort={() => {}}
			effortOpen={false}
			onCloseEffort={() => {}}
		/>
	);
}

function mount(pid = "p1", pending: SessionProjection["pending"] = null) {
	render(<TestComposer pid={pid} pending={pending} />);
	return screen.getByPlaceholderText("Ask anything…") as HTMLTextAreaElement;
}

/** `mount` that keeps the render handle, for a rerender at another `pid`. */
function mountWithHandle(pid = "p1") {
	const view = render(<TestComposer pid={pid} />);
	return { view, field: screen.getByPlaceholderText("Ask anything…") as HTMLTextAreaElement };
}

/** Type `value` and put the caret at `caret` — what a phone reports as it types.

    The trailing `select` is the caret report the `$` parse reads; `onChange`
    alone reports the position BEFORE the character landed. One test below
    deliberately types without it — see the Escape case. */
function type(field: HTMLTextAreaElement, value: string, caret = value.length) {
	fireEvent.change(field, { target: { value } });
	field.setSelectionRange(caret, caret);
	fireEvent.select(field);
}

function tray(): HTMLElement | null {
	return document.querySelector("[data-skill-tray]");
}

/** An ask waiting for an answer, as the projection publishes it (design D5). */
const ASK: SessionProjection["pending"] = {
	request_id: "r1",
	kind: "ask",
	title: "Which database should the migration target?",
	detail: "The two candidates differ in how they handle the rollback.",
	options: [],
	secret: false,
	question_index: 0,
	question_total: 1,
};

afterEach(() => {
	cleanup();
	localStorage.clear();
	vi.clearAllMocks();
	capabilities = { skill_invocation: { available: true } };
	draftSeed = "";
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
		/* `dispatchEvent` returns false exactly when `preventDefault` ran, which is
		   the whole focus-retention mechanism: happy-dom's pointer events never
		   move focus on their own, so asserting `activeElement` alone would pass
		   with the handler deleted (agent review m5). */
		expect(fireEvent.pointerDown(row, { pointerId: 1, clientX: 12, clientY: 400 })).toBe(false);
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

	it("clears the skipped-skill receipt when the next draft is typed (U5)", async () => {
		sendCommand.mockResolvedValueOnce({
			ok: true,
			detail: "",
			skills: { attached: [], skipped: ["empty-one"] },
		});
		const field = mount();
		type(field, "$empty-one", 10);
		fireEvent.keyDown(field, { key: "Enter" });
		await waitFor(() => expect(screen.getByText(/^Sent without/)).toBeTruthy());

		type(field, "a fresh message", 15);
		await waitFor(() => expect(screen.queryByText(/^Sent without/)).toBeNull());
	});

	it("names the remedy when the relay refuses over-large skill bodies (U6)", async () => {
		/* The relay's 422 is a refusal with a remedy, not a transient failure:
		   retrying the same draft is refused identically, so the retry sentence
		   sent the user in a circle (UX review U6). The typed code is what the
		   client keys on, so the relay's own prose can move. */
		sendCommand.mockRejectedValueOnce(
			new HttpError(
				422,
				"skill bodies total 409676 bytes, over the 393216 byte budget",
				"skills_too_large",
			),
		);
		const field = mount();
		type(field, "$ux-big $ux-big2 hi", 20);
		fireEvent.keyDown(field, { key: "Enter" });

		await waitFor(() => expect(screen.getByText(/too large to send together/)).toBeTruthy());
		expect(screen.queryByText("Couldn’t continue this conversation. Try again.")).toBeNull();
		/* The draft stays: dropping a skill is an edit, not a retype. */
		expect(field.value).toBe("$ux-big $ux-big2 hi");
	});

	it("caps the tray in COLUMN units, not in vh (D1/M2)", async () => {
		const field = mount();
		type(field, "$", 1);
		await waitFor(() => expect(tray()).toBeTruthy());
		/* The unit is the fix: `vh` is the DEVICE height, which a virtual keyboard
		   does not shrink, so a `vh` cap took the composer's foot off a 508 px
		   column. An inline value resolves under happy-dom while the Tailwind class
		   it replaced resolved to "" — the reason `lib/column.ts` caps inline. The
		   layout proof at keyboard heights is the rendered frames. */
		expect((tray() as HTMLElement).style.maxHeight).toBe(
			"calc(var(--lo-vvh, 100dvh) * 0.45)",
		);
		// The repo's existing "more below" cue (design D3), not a second one.
		expect((tray() as HTMLElement).className).toContain("lo-fade-y");
	});

	it("drops to TWO rows while a question is waiting (D5)", async () => {
		const field = mount("p1", ASK);
		type(field, "$", 1);
		await waitFor(() => expect(tray()).toBeTruthy());

		/* A question outranks a list. At every phone column with the keyboard up
		   the 45 % tray pushed the ask's question above the slot and left only
		   its decision row — 14 px of it clipped at 508 — so the user was asked
		   to approve what they could not read. Two 44 px rows plus the tray's
		   own `py-1` is 96 px, and the list scrolls inside it. */
		expect((tray() as HTMLElement).style.maxHeight).toBe("calc(2 * 2.75rem + 0.5rem)");
		/* The rows are still there to be scrolled through — the cap narrows the
		   window, it does not narrow the list. */
		expect(within(tray() as HTMLElement).getAllByRole("button")).toHaveLength(3);
	});

	it("opens mid-composition — Android composes every Latin word (M1)", async () => {
		const field = mount();
		/* Gboard reports a composition for `$res` as it is typed. The dropped guard
		   is what keeps the list open through it; `selectionStart` decides. */
		fireEvent.compositionStart(field);
		type(field, "$res", 4);
		await waitFor(() => expect(tray()).toBeTruthy());
		expect(within(tray() as HTMLElement).getByText("$research")).toBeTruthy();
		fireEvent.compositionEnd(field);
		expect(tray()).toBeTruthy();
	});

	it("closes on a tap outside the tray and the field, keeping the draft (U1)", async () => {
		const field = mount();
		type(field, "$res", 4);
		await waitFor(() => expect(tray()).toBeTruthy());

		fireEvent.pointerDown(document.body);
		await waitFor(() => expect(tray()).toBeNull());
		expect(field.value).toBe("$res");
	});

	it("closes when the field loses focus, and re-opens on the next edit (U1)", async () => {
		const field = mount();
		type(field, "$res", 4);
		await waitFor(() => expect(tray()).toBeTruthy());

		fireEvent.blur(field);
		await waitFor(() => expect(tray()).toBeNull());

		type(field, "$rese", 5);
		await waitFor(() => expect(tray()).toBeTruthy());
	});

	it("closes on Escape without rewriting the draft (U1)", async () => {
		const field = mount();
		type(field, "$research the bug", 9);
		await waitFor(() => expect(tray()).toBeTruthy());

		fireEvent.keyDown(field, { key: "Escape" });
		await waitFor(() => expect(tray()).toBeNull());
		expect(field.value).toBe("$research the bug");
	});

	it("closes on ONE Escape, with the selection report the same press makes (Q-R2-1)", async () => {
		const field = mount();
		/* Two things the old test was missing, both halves of the bug.

		   FOCUS: React reports the selection of the focused control, so without
		   focus there is no second event to reproduce.

		   NO `select` AFTER TYPING (hence no `type()` here): a real keypress
		   reports the selection as it was BEFORE the character landed, so the
		   event a real Escape press produces IS a change — the one path that
		   re-armed the dismissal. Handing the composer a selection report right
		   after the edit, as the helper does, hides exactly that. */
		field.focus();
		fireEvent.change(field, { target: { value: "$res" } });
		field.setSelectionRange(4, 4);
		await waitFor(() => expect(tray()).toBeTruthy());

		fireEvent.keyDown(field, { key: "Escape" });
		fireEvent.keyUp(field, { key: "Escape" });

		await waitFor(() => expect(tray()).toBeNull());
		expect(field.value).toBe("$res");
	});

	it("re-opens when the caret actually moves, or on a tap in the field (U1/m7)", async () => {
		const field = mount();
		type(field, "$res", 4);
		await waitFor(() => expect(tray()).toBeTruthy());

		fireEvent.keyDown(field, { key: "Escape" });
		fireEvent.select(field);
		fireEvent.keyUp(field, { key: "Escape" });
		await waitFor(() => expect(tray()).toBeNull());

		/* A caret move inside the token is a fresh decision. */
		field.setSelectionRange(3, 3);
		fireEvent.select(field);
		await waitFor(() => expect(tray()).toBeTruthy());

		/* And so is a tap that lands where the caret already was — the gesture,
		   not the position, is what re-arms (review m7: returning from another
		   app and tapping back into the field must not leave the list stuck
		   closed until the next keystroke). */
		fireEvent.keyDown(field, { key: "Escape" });
		fireEvent.select(field);
		await waitFor(() => expect(tray()).toBeNull());
		fireEvent.pointerDown(field, { pointerId: 4 });
		await waitFor(() => expect(tray()).toBeTruthy());
	});

	it("stays open when focus moves into a row, and the row still picks (U7)", async () => {
		const field = mount();
		type(field, "$res", 4);
		await waitFor(() => expect(tray()).toBeTruthy());
		const row = within(tray() as HTMLElement).getByText("$research").closest("button") as HTMLButtonElement;

		/* Shift+Tab from the field lands on the last row. The blur that precedes
		   it used to remove the target before the browser could focus it, so
		   focus fell through to the attach control and the row's keyboard path
		   was advertised but unreachable. */
		fireEvent.blur(field, { relatedTarget: row });
		await Promise.resolve();
		expect(tray()).toBeTruthy();

		/* A keyboard activation (no pointer detail) is the row's third path. */
		fireEvent.click(row, { detail: 0 });
		await waitFor(() => expect(field.value).toBe("$research "));
	});

	it("closes once focus leaves the tray for something outside it (U7)", async () => {
		const field = mount();
		type(field, "$res", 4);
		await waitFor(() => expect(tray()).toBeTruthy());

		fireEvent.focusIn(document.body);
		await waitFor(() => expect(tray()).toBeNull());
	});

	it("Escape from a focused row closes the list and returns to the field (U7)", async () => {
		const field = mount();
		type(field, "$res", 4);
		await waitFor(() => expect(tray()).toBeTruthy());
		const row = within(tray() as HTMLElement).getByText("$research").closest("button") as HTMLButtonElement;

		fireEvent.blur(field, { relatedTarget: row });
		await Promise.resolve();
		expect(tray()).toBeTruthy();

		/* The field's own handler cannot see a key pressed on a row, so the list
		   would otherwise have no keyboard exit once focus is inside it. */
		fireEvent.keyDown(row, { key: "Escape" });
		await waitFor(() => expect(tray()).toBeNull());
		expect(document.activeElement).toBe(field);
		expect(field.value).toBe("$res");
	});

	it("a restored draft stays closed until the user edits it (U2)", async () => {
		draftSeed = "$res";
		const field = mount();
		await Promise.resolve();
		expect(field.value).toBe("$res");
		/* No caret has been reported: the draft is not a gesture, and a list that
		   opened over it would be covering a message the user did not touch. */
		expect(tray()).toBeNull();

		type(field, "$rese", 5);
		await waitFor(() => expect(tray()).toBeTruthy());
	});

	it("never offers a name the mention grammar cannot express (U3)", async () => {
		const field = mount();
		type(field, "$", 1);
		await waitFor(() => expect(tray()).toBeTruthy());
		/* `ux spaced` is in the catalogue; the token ends at whitespace, so picking
		   it would insert `$ux spaced ` and send a mention that never fires. */
		expect(within(tray() as HTMLElement).queryByText(/spaced/)).toBeNull();
	});

	it("clears the catalogue when the composer is repointed at another session (m4)", async () => {
		const { view, field } = mountWithHandle();
		type(field, "$", 1);
		await waitFor(() => expect(tray()).toBeTruthy());

		/* The next session's fetch never answers, so any row on screen would be the
		   previous session's catalogue. */
		getSkills.mockImplementationOnce(() => new Promise(() => {}));
		view.rerender(<TestComposer pid="p2" />);
		await waitFor(() => expect(tray()).toBeNull());
	});
});
