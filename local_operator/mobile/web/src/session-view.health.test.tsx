// @vitest-environment happy-dom
//
// Session-state honesty on the phone (mobile UX batch 2): U7 (ended/degraded
// are consumed), U11 (the phone's own dropped link says so) and U2 (a refused
// pin says why). Asserted against the REAL SessionScreen, because that is
// where the ladder renders and where the pin refusal lands.
//
// WHY A LADDER AND ONE STRIP. `ended` (the process is gone; resume lives in
// the strip), `degraded` (the relay's dial is down) and `connected === false`
// (the phone's own link) are three different facts, and stacking three strips
// on a 320-wide phone would spend the vertical budget the batch-1 layout work
// bought back. The later rungs are only worth stating while the earlier are
// untrue, so at most one shows — pinned here, since an accidental second strip
// is exactly the kind of regression a single-state assertion misses.
//
// WHAT THIS LAYER CANNOT PROVE: happy-dom does no layout, so the geometry the
// overlay exists for — the transcript's top edge staying put while a rung
// appears and clears, and the rung's dead space handing touches back to the
// transcript — is a rendered-frame question, proved by the round's headless
// captures and their measured y-stability numbers. What IS pinned here is the
// structure that makes it true (`absolute` + `pointer-events-none` on the
// ladder container, `pointer-events-auto` on the resume control).
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SessionScreen } from "./screens/session-view";
import type { SessionProjection, TranscriptEntry } from "./types";

const mocks = vi.hoisted(() => ({
	setSessionPin: vi.fn(async () => ({ ok: true })),
	/* THE REAL SHAPE (agent-review NIT 1): the resume route echoes the id it was
	   given (`daemon.py:3094`/`4313`), so the navigation is SAME-ROUTE and the
	   test must pin that, not a fiction where a new id comes back. */
	resumeSession: vi.fn(async () => ({ ok: true, pid: 42, session_id: "s1" })),
	navigate: vi.fn(),
}));

vi.mock("./router", () => ({ navigate: mocks.navigate }));

vi.mock("./api", async () => {
	/* The REAL refusal class, so `instanceof HttpError` in the refusal helper
	   behaves exactly as in production; every function stays a stub (spreading
	   the real module would hand unlisted callers a live fetch). */
	const { HttpError } = await vi.importActual<typeof import("./api")>("./api");
	return {
		HttpError,
		getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
		getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
		getSubagentDetail: vi.fn(async () => null),
		imageUrl: vi.fn(() => ""),
		getCommands: vi.fn(async () => ({ commands: [] })),
		getModels: vi.fn(async () => ({ models: [] })),
	// THE CHIP'S TWO CALLS (the composer's working-directory cluster). Stubbed
	// like every other api function here rather than spread from the real module,
	// which would hand the unlisted callers a live fetch.
	getDirectories: vi.fn(async () => ({ home: "/Users/tester", recent: [], tmp: "" })),
	changeDirectory: vi.fn(async () => ({ ok: true, pid: 1, session_id: "s1" })),
		sendCommand: vi.fn(async () => ({ ok: true, detail: "" })),
		markSessionSeen: vi.fn(async () => ({ ok: true })),
		setSessionPin: mocks.setSessionPin,
		resumeSession: mocks.resumeSession,
	};
});

let slot: { projection: SessionProjection | null; connected: boolean } = {
	projection: null,
	connected: true,
};
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useProjection: vi.fn(() => slot),
		retainProjectionStream: vi.fn(() => () => {}),
		retainSessionListStream: vi.fn(() => () => {}),
		useDraft: vi.fn(() => ["", () => {}]),
	};
});

/** The projection shape the REAL SessionScreen tree reads.
    Copied from the sibling render tests rather than trimmed: the Composer
    reads `effort_ladder.length` and friends, and a partial fixture fails
    inside a component rather than in the assertion under test. */
function projection(over: Partial<SessionProjection> = {}): SessionProjection {
	return {
		session_id: "s1",
		pid: 1,
		kind: "tui",
		conversation_name: "health",
		cwd: "",
		model_label: "",
		model_selector: "",
		effort: "",
		effort_ladder: [],
		streaming: false,
		activity: "",
		activity_started_s: 0,
		stop_reason: "",
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
		...over,
	} satisfies SessionProjection;
}

/** One user row, so the transcript (not the empty state) is on screen — the
    U23 = D7 reserve only exists where a scroller does. */
function userRow(text: string): TranscriptEntry {
	return {
		id: text,
		kind: "user",
		text,
		tool_call_id: "",
		tool_name: "",
		tool_state: "done",
		summary: "",
		intent: "",
		diff_added: 0,
		diff_removed: 0,
		elapsed_s: 0,
		error: "",
		details: {},
		final: false,
	};
}

beforeEach(() => {
	slot = { projection: projection(), connected: true };
	mocks.setSessionPin.mockClear();
	mocks.setSessionPin.mockImplementation(async () => ({ ok: true }));
	mocks.resumeSession.mockClear();
	mocks.resumeSession.mockImplementation(async () => ({ ok: true, pid: 42, session_id: "s1" }));
	mocks.navigate.mockClear();
});
afterEach(() => {
	cleanup();
	/* The wide-view control writes a preference and a DOM attribute; leaving
	   either behind would leak into the next render in this file. */
	localStorage.clear();
	delete document.documentElement.dataset.view;
});

/** A minimal roster row, copied from the sibling render test rather than
    trimmed: SubagentsPanel reads most of these fields. */
function subagentRow(): SessionProjection["subagents"][number] {
	return {
		job_id: "j1",
		label: "audit",
		agent: "coder",
		status: "completed",
		progress: "",
		elapsed_s: 1,
		model_label: "",
		result_text: "",
		error_text: "",
		parent_job_id: null,
		session_id: "j1-session",
		prompt: "",
		launch_message_id: "",
		effort: "high",
		ancestors: [],
		ancestor_ids: [],
		child_ids: [],
		peer_ids: [],
		transcript: [],
		todos: [],
		activity: "",
	};
}

/** A retained projection carrying every disclosure the tray can hold. */
function retained(flags: Partial<SessionProjection>): Partial<SessionProjection> {
	return {
		todos: [{ name: "Todos", items: [{ text: "audit", status: "pending", reason: "" }] }],
		subagents: [subagentRow()],
		...flags,
	};
}

/* THE BANNER NEVER OWNS THE ONLY DOOR (agent review round 1, C1-1 = QA Q1-2).
   The ended/degraded banner used to REPLACE the tray, and the tray held the one
   setter that opens the Tasks and Agents sheets — so a finished conversation's
   own history became unreadable without restarting its runtime. These cases go
   through the real SessionScreen and assert the DOORS, then open them. */
describe.each<[string, Partial<SessionProjection>]>([
	["ended", { ended: true }],
	["degraded", { degraded: true }],
])("a %s conversation keeps its retained history reachable (C1-1)", (_state, flags) => {
	it("exposes the tasks and agents doors and opens both sheets from them", () => {
		slot = { projection: projection(retained(flags)), connected: true };
		render(<SessionScreen sessionId="s1" />);

		/* Named by their own words, not by a hidden gesture: the tray chips are
		   BUTTONs, `aria-expanded` describes the sheet they own, and the chip
		   primitive's `lo-hit` gives them the 44 px floor inside the tray's own
		   44 px strip. happy-dom does no layout, so the BAND is a rendered-audit
		   number; this pins the structure that makes it real. */
		const tasks = screen.getByRole("button", { name: /^Tasks/ });
		const agents = screen.getByRole("button", { name: /^Agents/ });
		for (const door of [tasks, agents]) {
			expect(door.tagName).toBe("BUTTON");
			expect(door.className).toContain("lo-hit");
			expect(door.closest("div")?.className).toContain("h-11");
			expect(door.getAttribute("aria-expanded")).toBe("false");
		}

		/* The sheet is MODAL, so it takes the rest of the screen out of the
		   accessibility tree while it is open — the door is read through the
		   handle captured above, and the sheet is asserted by its own name. */
		fireEvent.click(tasks);
		expect(tasks.getAttribute("aria-expanded")).toBe("true");
		expect(screen.getByRole("dialog", { name: "Tasks" })).toBeTruthy();
		fireEvent.click(screen.getByRole("button", { name: "close sheet" }));

		fireEvent.click(agents);
		expect(agents.getAttribute("aria-expanded")).toBe("true");
		expect(screen.getByRole("dialog", { name: "Agents" })).toBeTruthy();
	});
});

/* THE ENDED SESSION (U7), now the header's grey status dot (operator request:
   the permanent banner above the composer is gone). The resume path itself —
   acceptance line (U24), same-route survival (U16), the refusal voice (D5/U26)
   and where it reopens (U18, #1875) — moved with its code into the dot's sheet
   and is asserted there, in `session-view.status-dot.test.tsx`. What stays here
   is the ladder's honesty about the state. */
describe("an ended session (U7)", () => {
	/** The status dot's sheet — the one place an ended session offers Resume. */
	function openHealthSheet(): HTMLElement {
		fireEvent.click(document.querySelector<HTMLElement>("button[data-health]")!);
		return screen.getByRole("dialog");
	}

	it("says so in the header and offers the documented resume from the dot", async () => {
		slot = { projection: projection({ ended: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);

		// The reader is told what happened, in the header's own word.
		const control = document.querySelector<HTMLElement>("button[data-health]")!;
		expect(control.dataset.health).toBe("ended");
		expect(control.getAttribute("title")).toBe("Ended");
		expect(within(control).getByText("Ended")).toBeTruthy();

		fireEvent.click(within(openHealthSheet()).getByRole("button", { name: "Resume" }));
		await waitFor(() => expect(mocks.resumeSession).toHaveBeenCalledWith("s1"));
		await waitFor(() => expect(mocks.navigate).toHaveBeenCalledWith("/s/s1"));
	});

	it("an ended CUT-OFF session offers exactly ONE resume affordance (U15)", () => {
		/* The shape the daemon serves after a mid-turn death: `ended` and the
		   cut-off receipt coexist. The composer's `turn cut off — tap to resume`
		   sends `continue` to a runtime that is gone; the dot's Resume is the one
		   path that respawns, so it must be the only one offered — and it is not
		   painted at rest at all, so nothing in the thumb zone competes. */
		slot = {
			projection: projection({ ended: true, stop_reason: "aborted", cut_off: true }),
			connected: true,
		};
		render(<SessionScreen sessionId="s1" />);

		expect(screen.queryByRole("button", { name: /tap to resume/ })).toBeNull();
		expect(screen.queryByText("turn cut off — tap to resume")).toBeNull();
		expect(screen.queryByRole("button", { name: "Resume" })).toBeNull();
		expect(within(openHealthSheet()).getAllByRole("button", { name: "Resume" })).toHaveLength(1);
	});

	it("reaches wide view through the session menu, with its state visible (D2, #1870)", () => {
		slot = { projection: projection(), connected: true };
		render(<SessionScreen sessionId="s1" />);

		/* The header no longer carries a permanent `[wide]` toggle; the menu row
		   names the setting and states it. */
		expect(screen.queryByRole("button", { name: "wide view" })).toBeNull();
		fireEvent.click(screen.getByRole("button", { name: "session menu" }));

		const control = screen.getByRole("button", { name: "wide view" });
		expect(control.textContent).toContain("Wide view");
		expect(control.textContent).toContain("off");
		expect(control.getAttribute("aria-pressed")).toBe("false");

		fireEvent.click(control);

		const pressed = screen.getByRole("button", { name: "wide view" });
		expect(pressed.getAttribute("aria-pressed")).toBe("true");
		expect(pressed.textContent).toContain("on");
		expect(document.documentElement.dataset.view).toBe("wide");
		expect(localStorage.getItem("lo-mobile-wide-view")).toBe("1");

		fireEvent.click(pressed);
		expect(document.documentElement.dataset.view).toBeUndefined();
		expect(localStorage.getItem("lo-mobile-wide-view")).toBeNull();
	});

	it("keeps spend reachable from the composer on an ended session", () => {
		slot = { projection: projection({ ended: true, context_tokens: 12400, context_window: 200000 }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		const spend = screen.getByRole("button", { name: /Spend & context:/ });
		expect(spend.closest(".lo-composer")).toBeTruthy();
	});

	it("spends no transcript top inset and no strip above the composer", () => {
		slot = { projection: projection({ ended: true, transcript: [userRow("prompt")] }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		expect(document.querySelector("[data-scroll-top-inset]")).toBeNull();
		expect(screen.getByText("prompt")).toBeTruthy();
		expect(screen.queryByText("Session ended · history kept")).toBeNull();
		expect(document.querySelector("[data-above-composer]")).toBeNull();
	});
});

describe("the route's tab title (U21)", () => {
	it("uses the header's own fallback for an unnamed conversation", async () => {
		/* The header has always said `untitled`; the tab title said `session`, so
		   the task switcher and the screen disagreed about the same session. */
		slot = { projection: projection({ conversation_name: "" }), connected: true };
		render(<SessionScreen sessionId="s1" />);

		expect(screen.getByText("untitled")).toBeTruthy();
		await waitFor(() => expect(document.title).toBe("untitled — local operator"));
	});
});

describe("a degraded session (U7)", () => {
	it("states the dial is down and does not claim the conversation ended", () => {
		slot = { projection: projection({ degraded: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);

		const control = document.querySelector<HTMLElement>("button[data-health]")!;
		expect(control.dataset.health).toBe("not_answering");
		expect(within(control).getByText("Not answering")).toBeTruthy();
		// The full sentence (sends wait) is one tap away, in the dot's sheet.
		fireEvent.click(control);
		expect(within(screen.getByRole("dialog")).getByText("Not answering — sends will wait")).toBeTruthy();
		// `ended` and `degraded` are different facts; the degraded state must
		// not drag the resume affordance (and its restart offer) in with it.
		expect(screen.queryByRole("button", { name: "Resume" })).toBeNull();
		expect(screen.queryByText(/session ended/i)).toBeNull();
	});
});

describe("the health ladder renders exactly one rung", () => {
	it("prefers ended over degraded and over the phone's own link", () => {
		slot = {
			projection: projection({ ended: true, degraded: true }),
			connected: false,
		};
		render(<SessionScreen sessionId="s1" />);

		const control = document.querySelector<HTMLElement>("button[data-health]")!;
		expect(control.dataset.health).toBe("ended");
		expect(within(control).getByText("Ended")).toBeTruthy();
		expect(screen.queryByText("Not answering")).toBeNull();
		expect(screen.queryByText("Reconnecting…")).toBeNull();
		expect(document.querySelectorAll("[data-health-dot]")).toHaveLength(1);
	});

	it("prefers degraded over the phone's own link", () => {
		slot = { projection: projection({ degraded: true }), connected: false };
		render(<SessionScreen sessionId="s1" />);

		expect(document.querySelector<HTMLElement>("button[data-health]")!.dataset.health).toBe("not_answering");
		expect(screen.getByText("Not answering")).toBeTruthy();
		expect(screen.queryByText("Reconnecting…")).toBeNull();
		expect(document.querySelectorAll("[data-health-dot]")).toHaveLength(1);
	});
});

describe("the phone's own dropped link (U11)", () => {
	it("says reconnecting once a projection is on screen, and nothing while it has none", () => {
		slot = { projection: projection(), connected: false };
		const first = render(<SessionScreen sessionId="s1" />);
		expect(screen.getByText("Reconnecting…")).toBeTruthy();
		first.unmount();

		// With no data yet the view already says `connecting to session…`;
		// a second line would be noise about noise.
		slot = { projection: null, connected: false };
		render(<SessionScreen sessionId="s1" />);
		expect(screen.queryByText("Reconnecting…")).toBeNull();
	});
});

describe("a refused pin (U2)", () => {
	it("shows the daemon's reason in the strip instead of a silent flip-back", async () => {
		mocks.setSessionPin.mockImplementation(async () => {
			throw new Error("no saved messages yet — pin it after you send one");
		});
		const { getByRole } = render(<SessionScreen sessionId="s1" />);

		fireEvent.click(await screen.findByRole("button", { name: "session menu" }));
		const star = getByRole("button", { name: "Pin session" });
		fireEvent.click(star);

		await waitFor(() =>
			expect(screen.getByRole("alert").textContent).toContain(
				"Could not save the pin: no saved messages yet — pin it after you send one",
			),
		);
		// R1-3/U2: the refusal closed the sheet, so the in-flow receipt is the
		// visible thing. The optimistic mark went back with it — reopening the
		// menu offers the press again rather than showing a state the daemon
		// never accepted.
		expect(screen.queryByRole("dialog")).toBeNull();
		/* The sheet restores the header's inert/aria-hidden in its cleanup,
		   which lands a tick after the close; poll the accessible name. */
		fireEvent.click(await screen.findByRole("button", { name: "session menu" }));
		expect(await screen.findByRole("button", { name: "Pin session" })).toBeTruthy();
	});
});

describe("the refused pin's own exit (U17)", () => {
	it("the line clears once the message it asked for lands", async () => {
		mocks.setSessionPin.mockImplementation(async () => {
			throw new Error("no saved messages yet — pin it after you send one");
		});
		const { rerender } = render(<SessionScreen sessionId="s1" />);

		fireEvent.click(await screen.findByRole("button", { name: "session menu" }));
		fireEvent.click(await screen.findByRole("button", { name: "Pin session" }));
		const refusal = await waitFor(() => screen.getByRole("alert"));
		expect(refusal.textContent).toContain("pin it after you send one");

		/* The reader follows the instruction: one message is sent, the
		   transcript's projection now carries it — and the line that named that
		   exit retires instead of sitting above the new row as if it still
		   applied. */
		slot = {
			projection: projection({
				transcript: [
					{
						id: "u1",
						kind: "user",
						text: "first message",
					} as SessionProjection["transcript"][number],
				],
			}),
			connected: true,
		};
		rerender(<SessionScreen sessionId="s1" />);

		await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
	});
});
