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
import { cleanup, fireEvent, render, screen, waitFor, act } from "@testing-library/react";
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
afterEach(cleanup);

describe("an ended session (U7)", () => {
	it("says so and offers the documented resume, which reopens the conversation", async () => {
		slot = { projection: projection({ ended: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);

		// The reader is told what happened, in the strip's own words.
		expect(screen.getByText("session ended — history kept")).toBeTruthy();

		fireEvent.click(screen.getByRole("button", { name: "resume" }));
		await waitFor(() => expect(mocks.resumeSession).toHaveBeenCalledWith("s1"));
		// The route echoes the id, so the phone lands on the SAME session route —
		// the shape that makes U16 (below) a same-route survival problem.
		await waitFor(() => expect(mocks.navigate).toHaveBeenCalledWith("/s/s1"));
	});

	it("an ended CUT-OFF session offers exactly ONE resume affordance (U15)", () => {
		/* The shape the daemon serves after a mid-turn death: the terminal
		   repaint fills the end from the durable record, so `ended` and the
		   cut-off receipt coexist. The composer used to grow a second, prominent
		   `turn cut off — tap to resume` whose `continue` the dead runtime can
		   never take — the strip's resume is the one path that respawns, so it
		   must be the only one offered. */
		slot = {
			projection: projection({ ended: true, stop_reason: "aborted", cut_off: true }),
			connected: true,
		};
		render(<SessionScreen sessionId="s1" />);

		expect(screen.getByRole("button", { name: "resume" })).toBeTruthy();
		expect(screen.queryByRole("button", { name: /tap to resume/ })).toBeNull();
		expect(screen.queryByText("turn cut off — tap to resume")).toBeNull();
	});

	it("a SUCCEEDED resume that does not revive the session leaves the button usable (U16)", async () => {
		/* The same-route no-op: the POST answers, the navigation lands where the
		   component already is, and no live frame arrives. `busy` used to stay
		   set, stranding `resuming…` disabled with no way out. */
		slot = { projection: projection({ ended: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);

		fireEvent.click(screen.getByRole("button", { name: "resume" }));
		await waitFor(() => expect(mocks.resumeSession).toHaveBeenCalledTimes(1));
		// Retrying is safe (the daemon coalesces concurrent resumes), so the
		// control must come back — and a second press must actually fire again.
		const again = await screen.findByRole("button", { name: "resume" });
		expect((again as HTMLButtonElement).disabled).toBe(false);
		fireEvent.click(again);
		await waitFor(() => expect(mocks.resumeSession).toHaveBeenCalledTimes(2));
	});

	it("renders the daemon's sentence when the resume is refused, in one refusal voice (D5)", async () => {
		mocks.resumeSession.mockImplementation(async () => {
			throw new Error("no runtime to resume into");
		});
		slot = { projection: projection({ ended: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);

		fireEvent.click(screen.getByRole("button", { name: "resume" }));
		await waitFor(() =>
			expect(screen.getByRole("alert").textContent).toContain(
				"Could not resume: no runtime to resume into",
			),
		);
		// The affordance survives the refusal — retrying is the reader's call.
		expect(screen.getByRole("button", { name: "resume" })).toBeTruthy();
		expect(mocks.navigate).not.toHaveBeenCalled();
	});

	it("the ended strip rides the header as an overlay, not an in-flow row (U19/D4)", () => {
		slot = { projection: projection({ ended: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);

		/* The ladder container: absolutely positioned under the header, and
		   pointer-events-none so its dead space passes touches through to the
		   transcript beneath (scrolling from the strip's own row must work). */
		const row = screen.getByText("session ended — history kept");
		const ladder = row.closest("div.absolute");
		expect(ladder?.className).toContain("pointer-events-none");
		expect(ladder?.className).toContain("top-full");
		// ...while the one control the rung owns opts back in.
		expect(screen.getByRole("button", { name: "resume" }).className).toContain(
			"pointer-events-auto",
		);
	});

	it("says where an ended session reopens, and that a send does it too (U18, #1875)", () => {
		slot = { projection: projection({ ended: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		/* The words are spelled out, not a bare `~` (UX round 2, U25 = D8). */
		expect(screen.getByText("a send or resume reopens it in your home folder")).toBeTruthy();
	});

	it("a resume that succeeds without a live frame says so (U24)", async () => {
		slot = { projection: projection({ ended: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		fireEvent.click(screen.getByRole("button", { name: "resume" }));
		expect(
			await screen.findByText("reopening — this can take a few seconds"),
		).toBeTruthy();
	});

	it("and the line says what is true if the session still has not come up (U24)", async () => {
		vi.useFakeTimers();
		try {
			slot = { projection: projection({ ended: true }), connected: true };
			render(<SessionScreen sessionId="s1" />);
			fireEvent.click(screen.getByRole("button", { name: "resume" }));
			// Flush the POST's promise without advancing the clock.
			await act(async () => {
				await Promise.resolve();
			});
			expect(screen.getByText("reopening — this can take a few seconds")).toBeTruthy();
			await act(async () => {
				vi.advanceTimersByTime(21_000);
			});
			expect(screen.getByText("still reopening — it has not come up yet")).toBeTruthy();
			expect(screen.queryByText("reopening — this can take a few seconds")).toBeNull();
		} finally {
			vi.useRealTimers();
		}
	});

	it("maps the resume 404 onto the reader's sentence, not the raw id (U26)", async () => {
		const api = await import("./api");
		mocks.resumeSession.mockImplementation(async () => {
			throw new api.HttpError(404, "no such past session: s1");
		});
		slot = { projection: projection({ ended: true }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		fireEvent.click(screen.getByRole("button", { name: "resume" }));
		await waitFor(() =>
			expect(screen.getByRole("alert").textContent).toBe(
				"Could not resume: this session is no longer saved",
			),
		);
	});

	it("anchors the ladder under the spend/context glance, not over it (D9)", () => {
		slot = {
			projection: projection({
				ended: true,
				context_tokens: 12400,
				context_window: 200000,
			}),
			connected: true,
		};
		render(<SessionScreen sessionId="s1" />);
		const status = screen.getByTestId("session-status");
		const row = screen.getByText("session ended — history kept");
		const ladder = row.closest("div.absolute");
		/* One wrapper holds the header, the glance row and the ladder, so the
		   ladder's `top-full` anchor is BELOW the glance row by construction:
		   the strip can no longer paint over the numbers a struggling session
		   most needs. */
		expect(ladder).toBeTruthy();
		expect(status.parentElement).toBe(ladder?.parentElement);
	});

	it("reserves the rung's measured height inside the transcript (U23 = D7)", () => {
		/* happy-dom lays nothing out (every rect is 0), so the rung's height is
		   injected at the one place the screen measures it — the ladder
		   container's own rect — and the assertion reads the spacer the screen
		   then hands the transcript. */
		const original = Element.prototype.getBoundingClientRect;
		Element.prototype.getBoundingClientRect = function (this: Element) {
			if (
				typeof this.className === "string" &&
				this.className.includes("top-full")
			) {
				return { height: 52 } as unknown as DOMRect;
			}
			return original.call(this);
		};
		try {
			slot = {
				projection: projection({ ended: true, transcript: [userRow("prompt")] }),
				connected: true,
			};
			render(<SessionScreen sessionId="s1" />);
			const spacer = document.querySelector<HTMLElement>("[data-scroll-top-inset]");
			expect(spacer).toBeTruthy();
			expect(spacer?.style.height).toBe("52px");
			// Above every reachable row, not between them.
			expect(spacer?.parentElement?.firstElementChild).toBe(spacer);
		} finally {
			Element.prototype.getBoundingClientRect = original;
		}
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

		expect(screen.getByText("not answering — showing its last synced view")).toBeTruthy();
		// `ended` and `degraded` are different facts; the degraded strip must
		// not drag the resume affordance (and its restart offer) in with it.
		expect(screen.queryByRole("button", { name: "resume" })).toBeNull();
		expect(screen.queryByText(/session ended/)).toBeNull();
	});
});

describe("the health ladder renders exactly one rung", () => {
	it("prefers ended over degraded and over the phone's own link", () => {
		slot = {
			projection: projection({ ended: true, degraded: true }),
			connected: false,
		};
		render(<SessionScreen sessionId="s1" />);

		expect(screen.getByText("session ended — history kept")).toBeTruthy();
		expect(screen.queryByText("not answering — showing its last synced view")).toBeNull();
		expect(screen.queryByText("reconnecting — showing the last synced view")).toBeNull();
	});

	it("prefers degraded over the phone's own link", () => {
		slot = { projection: projection({ degraded: true }), connected: false };
		render(<SessionScreen sessionId="s1" />);

		expect(screen.getByText("not answering — showing its last synced view")).toBeTruthy();
		expect(screen.queryByText("reconnecting — showing the last synced view")).toBeNull();
	});
});

describe("the phone's own dropped link (U11)", () => {
	it("says reconnecting once a projection is on screen, and nothing while it has none", () => {
		slot = { projection: projection(), connected: false };
		const first = render(<SessionScreen sessionId="s1" />);
		expect(screen.getByText("reconnecting — showing the last synced view")).toBeTruthy();
		first.unmount();

		// With no data yet the view already says `connecting to session…`;
		// a second line would be noise about noise.
		slot = { projection: null, connected: false };
		render(<SessionScreen sessionId="s1" />);
		expect(screen.queryByText("reconnecting — showing the last synced view")).toBeNull();
	});
});

describe("a refused pin (U2)", () => {
	it("shows the daemon's reason in the strip instead of a silent flip-back", async () => {
		mocks.setSessionPin.mockImplementation(async () => {
			throw new Error("no saved messages yet — pin it after you send one");
		});
		render(<SessionScreen sessionId="s1" />);

		const star = await screen.findByRole("button", { name: "pin this session" });
		fireEvent.click(star);

		await waitFor(() =>
			expect(screen.getByRole("alert").textContent).toContain(
				"Could not save the pin: no saved messages yet — pin it after you send one",
			),
		);
		// The optimistic mark went back with the refusal — the button offers the
		// press again rather than showing a ★ the daemon never accepted.
		expect(screen.getByRole("button", { name: "pin this session" })).toBeTruthy();
	});
});

describe("the refused pin's own exit (U17)", () => {
	it("the line clears once the message it asked for lands", async () => {
		mocks.setSessionPin.mockImplementation(async () => {
			throw new Error("no saved messages yet — pin it after you send one");
		});
		const { rerender } = render(<SessionScreen sessionId="s1" />);

		fireEvent.click(await screen.findByRole("button", { name: "pin this session" }));
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
