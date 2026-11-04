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
//    Escape, stays shut on a restored draft, opens mid-composition, and yields
//    two rows whenever the above-composer slot draws a card or an ask's dock;
//    an asks-present EMPTY mirror (nothing drawn) keeps the 45 % cap;
//  - the rows' keyboard route is ArrowDown/ArrowUp from the field (sequential
//    focus cannot reach them), with Enter/Space native and Escape back out;
//  - a catalogue name the `$` grammar cannot express is never offered.
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Composer, isSlashDraft } from "./composer";
import { HttpError } from "../api";
import type { CommandReceipt } from "../api";
import type { PendingAsk, SessionProjection, SkillEntry } from "../types";

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

function projection(
	pending: SessionProjection["pending"] = null,
	asks?: SessionProjection["asks"],
): SessionProjection {
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
		asks,
		usage: {},
		version: 1,
	} as unknown as SessionProjection;
}

function TestComposer({
	pid = "p1",
	pending = null,
	asks,
}: {
	pid?: string;
	pending?: SessionProjection["pending"];
	asks?: SessionProjection["asks"];
}) {
	return (
		<Composer
			ended={false}
			pid={pid}
			projection={projection(pending, asks)}
			onOpenModels={() => {}}
			onReturnToModels={() => {}}
			onOpenEffort={() => {}}
			effortOpen={false}
			onCloseEffort={() => {}}
		/>
	);
}

function mount(
	pid = "p1",
	pending: SessionProjection["pending"] = null,
	asks?: SessionProjection["asks"],
) {
	render(<TestComposer pid={pid} pending={pending} asks={asks} />);
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

/** The queued ask whose `kind: "ask"` pending above is the dock's mirror —
    what a runtime that publishes `asks` sends (m8/U8). */
const QUEUED: PendingAsk = {
	ask_id: "ask-1",
	created_at: 1,
	expires_at: 1_800_000_000,
	timeout_s: 900,
	urgent: false,
	status: "open",
	delivered: false,
	questions: [
		{
			id: "q1",
			question: "Which database should the migration target?",
			options: [{ label: "sqlite", description: "" }],
			multi: false,
			secret: false,
			persist: false,
		},
	],
};

/** A blocking approval: one card, no asks capability, so the card is the only
    thing above the composer. */
const APPROVAL: SessionProjection["pending"] = {
	request_id: "approval-1",
	kind: "approval",
	title: "Run the migration against the staging store?",
	detail: "",
	options: [
		{ label: "Allow", description: "" },
		{ label: "Deny", description: "" },
	],
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

	it("drops to two rows for a REAL open ask whose dock draws (Q-R4-1)", async () => {
		/* The runtime publishes the queued ask (`asks`) AND mirrors it into
		   `pending`. `blockingPending` resolves the mirror to no card, but the
		   32 px dock still draws in the above-composer slot. The cap must yield
		   to it: at the 508 px keyboard column QA measured the 45 % tray leaving
		   the slot 4.5 px and the dock 0 px visible (round 4). */
		const field = mount("p1", ASK, [QUEUED]);
		type(field, "$", 1);
		await waitFor(() => expect(tray()).toBeTruthy());
		expect((tray() as HTMLElement).style.maxHeight).toBe("calc(2 * 2.75rem + 0.5rem)");
	});

	it("keeps the 45 % cap for an asks-present EMPTY mirror — nothing draws (m8/U8)", async () => {
		/* The same `kind: "ask"` pending with `asks: []` renders NO card and NO
		   dock. A slot with nothing in it must not cost the list its height. */
		const field = mount("p1", ASK, []);
		type(field, "$", 1);
		await waitFor(() => expect(tray()).toBeTruthy());
		expect((tray() as HTMLElement).style.maxHeight).toBe(
			"calc(var(--lo-vvh, 100dvh) * 0.45)",
		);
	});

	it("drops to two rows for a blocking approval (D5)", async () => {
		const field = mount("p1", APPROVAL);
		type(field, "$", 1);
		await waitFor(() => expect(tray()).toBeTruthy());
		expect((tray() as HTMLElement).style.maxHeight).toBe("calc(2 * 2.75rem + 0.5rem)");
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

	/* The keyboard's route into the rows (U7/M3). The old test handed
	   `fireEvent.blur` a `relatedTarget` the browser never supplies from that
	   key, so it pinned the handler and not the route: DOM order puts the tray
	   before the card's attach/mic controls, and neither Tab nor Shift+Tab can
	   reach a row. These drive the keys a real keyboard sends, in the order it
	   sends them, and fail if the arrow handler is removed. */
	it("ArrowDown enters the list at the first row, and ArrowDown/ArrowUp walk it (U7)", async () => {
		const field = mount();
		field.focus();
		type(field, "$", 1);
		await waitFor(() => expect(tray()).toBeTruthy());
		const rows = within(tray() as HTMLElement).getAllByRole("button") as HTMLButtonElement[];
		expect(rows).toHaveLength(3);

		/* `dispatchEvent` returns false exactly when `preventDefault` ran — the
		   caret must not move while the list takes the key (m12 pins the caret
		   itself, not only the cancellation). */
		expect(fireEvent.keyDown(field, { key: "ArrowDown" })).toBe(false);
		await waitFor(() => expect(document.activeElement).toBe(rows[0]));
		expect(field.selectionStart).toBe(1);
		expect(field.selectionEnd).toBe(1);
		/* Focus moved INTO the list, which is not a dismissal (the blur guard). */
		expect(tray()).toBeTruthy();

		fireEvent.keyDown(rows[0], { key: "ArrowDown" });
		await waitFor(() => expect(document.activeElement).toBe(rows[1]));
		fireEvent.keyDown(rows[1], { key: "ArrowUp" });
		await waitFor(() => expect(document.activeElement).toBe(rows[0]));

		/* ArrowUp at the first row hands the key back to the field, and the
		   caret is still where it was (m12). */
		fireEvent.keyDown(rows[0], { key: "ArrowUp" });
		await waitFor(() => expect(document.activeElement).toBe(field));
		expect(field.selectionStart).toBe(1);
		expect(field.selectionEnd).toBe(1);
		fireEvent.keyDown(field, { key: "ArrowDown" });
		await waitFor(() => expect(document.activeElement).toBe(rows[0]));

		/* A focused button's Enter/Space is its NATIVE activation; the click it
		   produces has `detail === 0`, which is the row's keyboard path. */
		fireEvent.click(rows[0], { detail: 0 });
		await waitFor(() => expect(field.value).toBe("$pathfinder "));
		expect(tray()).toBeNull();
	});

	it("ArrowUp enters at the last row, and Escape returns to the field (U7)", async () => {
		const field = mount();
		field.focus();
		type(field, "$", 1);
		await waitFor(() => expect(tray()).toBeTruthy());
		const rows = within(tray() as HTMLElement).getAllByRole("button") as HTMLButtonElement[];

		fireEvent.keyDown(field, { key: "ArrowUp" });
		await waitFor(() => expect(document.activeElement).toBe(rows[2]));
		expect(tray()).toBeTruthy();

		/* The field's own handler cannot see a key pressed on a row, so the list
		   would otherwise have no keyboard exit once focus is inside it. */
		fireEvent.keyDown(rows[2], { key: "Escape" });
		await waitFor(() => expect(tray()).toBeNull());
		expect(document.activeElement).toBe(field);
		expect(field.value).toBe("$");
		expect(field.selectionStart).toBe(1);
	});

	it("closes when focus leaves a row for a control outside the tray (U7)", async () => {
		const field = mount();
		field.focus();
		type(field, "$", 1);
		await waitFor(() => expect(tray()).toBeTruthy());
		const row = within(tray() as HTMLElement).getAllByRole("button")[0] as HTMLButtonElement;

		fireEvent.keyDown(field, { key: "ArrowDown" });
		await waitFor(() => expect(document.activeElement).toBe(row));

		/* Tab out of the list is a real focus move to the card's next control;
		   happy-dom does not synthesise the Tab, so the move itself is made. */
		(screen.getByLabelText("attach image") as HTMLButtonElement).focus();
		await waitFor(() => expect(tray()).toBeNull());
	});

	it("does not take the arrows when the tray is closed or an IME is composing (M3)", async () => {
		const field = mount();
		field.focus();
		type(field, "$res", 4);
		await waitFor(() => expect(tray()).toBeTruthy());
		fireEvent.keyDown(field, { key: "Escape" });
		await waitFor(() => expect(tray()).toBeNull());

		/* A closed tray leaves the arrows to the caret. */
		expect(fireEvent.keyDown(field, { key: "ArrowDown" })).toBe(true);
		expect(document.activeElement).toBe(field);

		/* An IME owns the arrows while it composes — they pick candidates. */
		type(field, "$", 1);
		await waitFor(() => expect(tray()).toBeTruthy());
		fireEvent.compositionStart(field);
		expect(fireEvent.keyDown(field, { key: "ArrowDown", isComposing: true })).toBe(true);
		expect(document.activeElement).toBe(field);
		expect(tray()).toBeTruthy();

		/* A modified arrow is a text gesture (selection, word/paragraph move),
		   not list navigation. */
		expect(fireEvent.keyDown(field, { key: "ArrowDown", shiftKey: true })).toBe(true);
		expect(document.activeElement).toBe(field);
	});

	it("leaves Alt/Ctrl arrows to the field while the list is open (m11)", async () => {
		const field = mount();
		field.focus();
		type(field, "$", 1);
		await waitFor(() => expect(tray()).toBeTruthy());

		/* Word/paragraph moves and selection gestures stay the field's: the
		   guard's modifier clause is what keeps a hardware keyboard's text
		   editing working under an open list. */
		for (const init of [{ altKey: true }, { ctrlKey: true }]) {
			expect(fireEvent.keyDown(field, { key: "ArrowDown", ...init })).toBe(true);
			expect(document.activeElement).toBe(field);
			expect(tray()).toBeTruthy();
		}
	});

	it("does not walk the rows on a modified or composing arrow (m11)", async () => {
		const field = mount();
		field.focus();
		type(field, "$", 1);
		await waitFor(() => expect(tray()).toBeTruthy());
		const rows = within(tray() as HTMLElement).getAllByRole("button") as HTMLButtonElement[];
		fireEvent.keyDown(field, { key: "ArrowDown" });
		await waitFor(() => expect(document.activeElement).toBe(rows[0]));

		/* The wrapper's handler repeats the field's rule, because a key pressed
		   on a row never reaches the field: a modified arrow is a text gesture
		   and an IME's arrow picks candidates, so neither moves focus. */
		for (const init of [
			{ shiftKey: true },
			{ altKey: true },
			{ ctrlKey: true },
			{ metaKey: true },
			{ isComposing: true },
		]) {
			expect(fireEvent.keyDown(rows[0], { key: "ArrowDown", ...init })).toBe(true);
			expect(document.activeElement).toBe(rows[0]);
		}
	});

	it("does not take an Escape aimed outside the list (m9/m11)", async () => {
		const field = mount();
		field.focus();
		type(field, "$", 1);
		await waitFor(() => expect(tray()).toBeTruthy());

		/* The document capture listener is scoped to `[data-skill-tray]`: an
		   Escape aimed at another control belongs to that control, so the list
		   stays open and the key is not cancelled. */
		const attach = screen.getByLabelText("attach image");
		expect(fireEvent.keyDown(attach, { key: "Escape" })).toBe(true);
		expect(tray()).toBeTruthy();
		expect(document.activeElement).toBe(field);
	});

	it("closes once focus leaves the tray for something outside it (U7)", async () => {
		const field = mount();
		type(field, "$res", 4);
		await waitFor(() => expect(tray()).toBeTruthy());

		fireEvent.focusIn(document.body);
		await waitFor(() => expect(tray()).toBeNull());
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
