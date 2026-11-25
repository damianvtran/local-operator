// @vitest-environment happy-dom
//
// REVIEW ROUND 5, M1 (PR #4522): a waking row that outlived its composer.
//
// `SessionScreen` is keyed per route (`app.tsx`), so a session switch unmounts
// the composer and its `wakingCommand` tracker ref, while the pending row
// (module-level in `pending-echo.ts`) survives with `waking: true`. The relay's
// `wake_failed` marker rides ONE frame and is never stored, and a re-subscribed
// cold session serves only its pid-0 seed, so nothing could fail that row any
// more: it read "waking session…" until a land, a retry or a reload. The
// remounted composer now re-attaches the tracker from the retained envelope +
// the held waking row, and past `WAKE_SETTLED_BOUND_MS` (the relay's 180 s
// boot deadline plus slack) a no-owner frame without the row is the failed
// send it must be.
//
// Measured RED on a9ecde89 (origin/main): the three failure cells (marker
// after a switch, missed marker past the bound, quiet stream past the bound)
// — the row stays and only the retained-retry alert shows. The running-owner
// and no-waking-row cells are guards that hold on both heads.
//
// Same render site as `composer.prompt-wake.test.tsx`: the REAL
// `SessionScreen` and `Composer`, with only the projection slot and the
// network faked.

import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SessionScreen } from "./screens/session-view";
import { WAKE_SETTLED_BOUND_MS } from "./components/composer";
import * as api from "./api";
import { clearPendingEchoes } from "./pending-echo";
import type { SessionProjection, TranscriptEntry } from "./types";

vi.mock("./api", async (importOriginal) => ({
	...(await importOriginal<typeof api>()),
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentDetail: vi.fn(async () => null),
	imageUrl: vi.fn(() => ""),
	getCommands: vi.fn(async () => ({ commands: [] })),
	getSkills: vi.fn(async () => ({ skills: [] })),
	getModels: vi.fn(async () => ({ models: [] })),
	sendCommand: vi.fn(),
	markSessionSeen: vi.fn(async () => ({ ok: true })),
}));

const COMMAND_ID = "12345678-1234-4678-9234-567812345678";
const SENT = "wake up and answer";
const STORAGE_KEY = ["lo-mobile-command", "s1"].join(":");
/* English `relay.health.ended_send_failed`, as `composer-resume-label.test.tsx` pins it. */
const FAILED_COPY =
	"This session has ended and couldn't be reopened just now. Try again, or tap “Ended” at the top and choose Resume.";
const WAKING = { ok: true, detail: "session is waking", waking: true };

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
	};
});

function userRow(id: string, text: string): TranscriptEntry {
	return {
		id, kind: "user", text, tool_call_id: "", tool_name: "", tool_state: "done",
		summary: "", intent: "", diff_added: 0, diff_removed: 0, elapsed_s: 0,
		error: "", details: {}, final: true,
	};
}

/** The COLD seed by default: a disk rebuild, `pid: 0`, not ended. */
function frame(overrides: Partial<SessionProjection> = {}): SessionProjection {
	return {
		session_id: "s1", pid: 0, kind: "daemon", conversation_name: "Cold", cwd: "",
		model_label: "", model_selector: "", effort: "", effort_ladder: [], streaming: false,
		activity: "", activity_started_s: 0, stop_reason: "", queued_count: 0, ended: false,
		degraded: false, transcript: [userRow("existing", "an older message")], todos: [],
		subagents: [], pending: null, pending_count: 0, usage: {}, version: 1,
		...overrides,
	};
}

afterEach(() => {
	cleanup();
	localStorage.clear();
	clearPendingEchoes();
	vi.unstubAllGlobals();
	vi.restoreAllMocks();
});

beforeEach(() => {
	vi.mocked(api.sendCommand).mockReset();
	vi.mocked(api.getHistory).mockReset();
	vi.mocked(api.getHistory).mockImplementation(async () => ({ entries: [], has_more: false }));
	vi.stubGlobal("crypto", { randomUUID: () => COMMAND_ID });
});

/** Mount the route; `publish` is what the SSE stream would deliver. */
function mount(first: SessionProjection = frame()) {
	slot = { projection: first, connected: true };
	const view = render(<SessionScreen sessionId="s1" />);
	return {
		...view,
		publish(next: SessionProjection, connected = true) {
			slot = { projection: next, connected };
			view.rerender(<SessionScreen sessionId="s1" />);
		},
	};
}

/** Leave the route and come back: what `app.tsx`'s per-route key does. */
function switchAway(view: ReturnType<typeof mount>, first?: SessionProjection) {
	view.unmount();
	return mount(first);
}

async function composer(): Promise<HTMLTextAreaElement> {
	return (await screen.findByRole("textbox", { name: "Message" })) as HTMLTextAreaElement;
}

function pendingRows(view: { container: HTMLElement }): Element[] {
	return Array.from(view.container.querySelectorAll(`[data-pending-echo="${COMMAND_ID}"]`));
}

/** Send SENT and let the relay answer `waking`. */
async function sendWaking(view: ReturnType<typeof mount>) {
	vi.mocked(api.sendCommand).mockResolvedValueOnce(WAKING);
	fireEvent.change(await composer(), { target: { value: SENT } });
	fireEvent.click(screen.getByRole("button", { name: "send" }));
	await waitFor(() => expect(vi.mocked(api.sendCommand)).toHaveBeenCalledOnce());
	await waitFor(() => expect(pendingRows(view)[0]?.textContent).toContain("waking session…"));
	expect(localStorage.getItem(STORAGE_KEY)).toContain(COMMAND_ID);
}

describe("a waking row that outlived its composer (M1)", () => {
	it("fails the send on the wake's marker after a session switch", async () => {
		const view = mount();
		await sendWaking(view);
		const back = switchAway(view);
		await waitFor(() => expect(pendingRows(back)[0]?.textContent).toContain("waking session…"));

		back.publish(frame({ ended: true, wake_failed: COMMAND_ID }));

		await waitFor(() => expect(screen.getByRole("alert").textContent).toContain(FAILED_COPY));
		expect(pendingRows(back)).toHaveLength(0);
		expect((await composer()).value).toBe(SENT);
		expect(localStorage.getItem(STORAGE_KEY)).toContain(COMMAND_ID);
	});

	it("fails the send once the wake must have settled, when the marker was missed", async () => {
		const start = Date.now();
		const view = mount();
		await sendWaking(view);
		/* Away while the wake failed: the one-shot marker went to nobody. */
		const clock = vi.spyOn(Date, "now").mockReturnValue(start + 180_000);
		const back = switchAway(view);
		back.publish(frame({ version: 2 }));
		/* Inside the bound, the pid-0 seed is what a RUNNING wake shows too.
		   (The remount's pre-existing retained-retry alert may be up; only the
		   failure copy would be wrong here.) */
		expect(screen.queryByText(FAILED_COPY)).toBeNull();
		expect(pendingRows(back)).toHaveLength(1);

		clock.mockReturnValue(start + WAKE_SETTLED_BOUND_MS + 1_000);
		back.publish(frame({ version: 3 }));

		await waitFor(() => expect(screen.getByRole("alert").textContent).toContain(FAILED_COPY));
		expect(pendingRows(back)).toHaveLength(0);
		expect((await composer()).value).toBe(SENT);
		expect(screen.getByRole("button", { name: "Retry earlier instruction" })).toBeTruthy();
		/* The envelope stays: it is the retry's body, under the SAME id. */
		expect(localStorage.getItem(STORAGE_KEY)).toContain(COMMAND_ID);
	});

	it("fails the send at the bound even when the stream stays quiet", async () => {
		/* A cold session publishes nothing after a failed wake, so the composer's
		   own deadline timer must re-enter the check. */
		vi.useFakeTimers({ shouldAdvanceTime: true, toFake: ["setTimeout", "clearTimeout", "Date"] });
		try {
			const view = mount();
			await sendWaking(view);
			const back = switchAway(view);
			await vi.advanceTimersByTimeAsync(WAKE_SETTLED_BOUND_MS + 1_000);
			await waitFor(() => expect(screen.getByRole("alert").textContent).toContain(FAILED_COPY));
			expect(pendingRows(back)).toHaveLength(0);
		} finally {
			vi.useRealTimers();
		}
	});

	it("does not fail past the bound while a running owner is serving the session", async () => {
		const start = Date.now();
		const view = mount();
		await sendWaking(view);
		vi.spyOn(Date, "now").mockReturnValue(start + WAKE_SETTLED_BOUND_MS + 1_000);
		/* The runtime is up (this wake, or another surface's): its command-id
		   dedupe owns the message, and the row waits for it to land. */
		const back = switchAway(view, frame({ pid: 4242, kind: "tui", version: 2 }));
		back.publish(frame({ pid: 4242, kind: "tui", version: 3 }));

		expect(screen.queryByText(FAILED_COPY)).toBeNull();
		expect(pendingRows(back)).toHaveLength(1);
		expect(localStorage.getItem(STORAGE_KEY)).toContain(COMMAND_ID);
	});

	/* REVIEW ROUND 1, M1 (PR #4584): the projection is the 80-row tail, so a
	   wake that delivered and then ran a long turn has pushed the user row out
	   of it. Past the bound that looked like "no row, no owner" and the
	   delivered message was reported failed.

	   REVIEW ROUND 2, R2-M1: round 1 read ANY moved tail as that delivery, and
	   a tail moves for other reasons too (the failed wake's runtime booting
	   late, the wake supervisor, another surface, a re-fold): the row, the
	   envelope and the Retry were cleared and the message was lost. The move
	   now only starts a search for the command's own id, in the tail and then
	   at most WAKE_CONFIRM_HISTORY_PAGES older pages. Measured RED on 7b45086f
	   (4 of the 5 cells below): unconfirmed, empty-baseline and live-ended (the
	   row is retired as delivered, the envelope cleared, no Retry) and found-on-
	   page-2 (no history is read at all). The found-in-tail cell is a guard
	   that holds on both heads. */
	const UNCONFIRMED_COPY =
		"Couldn’t confirm your earlier instruction was delivered. Retry it to be sure — it won’t be sent twice.";
	const filler = (from: number, n: number) =>
		Array.from({ length: n }, (_, i) => userRow(`later-${from + i}`, `row ${from + i}`));
	/** Serve `getHistory` from one chronological journal, paging like the relay. */
	function serveHistory(all: TranscriptEntry[]) {
		vi.mocked(api.getHistory).mockImplementation(async (_sid, before, limit = 80) => {
			const end = before ? all.findIndex((row) => row.id === before) : all.length;
			const start = Math.max(0, end - limit);
			return { entries: all.slice(start, end), has_more: start > 0 };
		});
	}
	/** The composer's confirmation pages (80 rows), not the transcript's own (120). */
	const confirmCalls = () => vi.mocked(api.getHistory).mock.calls.filter((call) => call[2] === 80);

	/** Wake, then come back past the bound to a pid-0 rebuild showing `tail`. */
	async function backPastBound(tail: TranscriptEntry[], overrides: Partial<SessionProjection> = {}) {
		const start = Date.now();
		const view = mount();
		await sendWaking(view);
		const clock = vi.spyOn(Date, "now").mockReturnValue(start + WAKE_SETTLED_BOUND_MS + 1_000);
		const back = switchAway(view, frame({ version: 2, transcript: tail, ...overrides }));
		back.publish(frame({ version: 3, transcript: tail, ...overrides }));
		return { back, clock };
	}

	it("treats a moved tail carrying the command's row as delivered", async () => {
		const tail = [...filler(0, 40), userRow(COMMAND_ID, SENT), ...filler(40, 39)];
		const { back, clock } = await backPastBound(tail);

		await waitFor(() => expect(localStorage.getItem(STORAGE_KEY)).toBeNull());
		expect(screen.queryByText(FAILED_COPY)).toBeNull();
		expect(screen.queryByText(UNCONFIRMED_COPY)).toBeNull();
		expect(pendingRows(back)).toHaveLength(0);
		expect((await composer()).value).toBe("");
		expect(screen.queryByRole("button", { name: "Retry earlier instruction" })).toBeNull();
		expect(confirmCalls()).toHaveLength(0);
		clock.mockRestore();
	});

	it("finds a delivery that scrolled out of the tail on history page 2", async () => {
		/* Journal: 240 older rows with ours at index 100, then the 80-row tail.
		   Page 1 is rows 160–239, page 2 rows 80–159. */
		const older = filler(0, 240);
		older[100] = userRow(COMMAND_ID, SENT);
		/* With no wire hints, an assistant-headed tail still takes the unpinned
		   branch of the legacy length-and-kind fallback. */
		const tail = filler(1000, 80);
		tail[0] = { ...tail[0], kind: "assistant" };
		serveHistory([...older, ...tail]);
		const { back, clock } = await backPastBound(tail);

		await waitFor(() => expect(localStorage.getItem(STORAGE_KEY)).toBeNull());
		expect(confirmCalls().map((call) => call[1])).toEqual(["later-1000", "later-160"]);
		expect(screen.queryByText(FAILED_COPY)).toBeNull();
		expect(screen.queryByText(UNCONFIRMED_COPY)).toBeNull();
		expect(pendingRows(back)).toHaveLength(0);
		expect((await composer()).value).toBe("");
		expect(screen.queryByRole("button", { name: "Retry earlier instruction" })).toBeNull();
		clock.mockRestore();
	});

	it("finds the oldest of 240 older rows behind an unpinned user-headed tail", async () => {
		/* The daemon omits a false pin. The three exclusive pages must cover
		   160–239, 80–159, then 0–79; overlap with the tail would miss row 0. */
		const older = filler(0, 240);
		older[0] = userRow(COMMAND_ID, SENT);
		const tail = filler(1000, 80);
		serveHistory([...older, ...tail]);
		const { back, clock } = await backPastBound(tail, { transcript_has_older: true });

		await waitFor(() => expect(localStorage.getItem(STORAGE_KEY)).toBeNull());
		expect(confirmCalls().map((call) => call[1])).toEqual(["later-1000", "later-160", "later-80"]);
		expect(screen.queryByText(UNCONFIRMED_COPY)).toBeNull();
		expect(pendingRows(back)).toHaveLength(0);
		expect(screen.queryByRole("button", { name: "Retry earlier instruction" })).toBeNull();
		clock.mockRestore();
	});

	it("keeps the envelope and Retry when a moved tail and the bounded history lack the id", async () => {
		/* Another writer moved the tail; our row is nowhere in the 3 pages the
		   search may read (it sits at row 10 of 400, past the cap). */
		const older = filler(0, 400);
		older[10] = userRow(COMMAND_ID, SENT);
		const tail = filler(1000, 80);
		serveHistory([...older, ...tail]);
		const { back, clock } = await backPastBound(tail);

		expect(await screen.findByText(UNCONFIRMED_COPY)).toBeTruthy();
		/* With neither wire hint, a full user-headed tail still uses the legacy
		   pinned-opener heuristic and the search still stops after three pages. */
		expect(confirmCalls().map((call) => call[1])).toEqual(["later-1001", "later-321", "later-241"]);
		expect(screen.queryByText(FAILED_COPY)).toBeNull();
		expect(pendingRows(back)).toHaveLength(0);
		expect((await composer()).value).toBe(SENT);
		expect(screen.getByRole("button", { name: "Retry earlier instruction" })).toBeTruthy();
		/* The retry's body, under the SAME id: a duplicate is deduped. */
		expect(localStorage.getItem(STORAGE_KEY)).toContain(COMMAND_ID);
		clock.mockRestore();
	});

	it("confirms a brand-new session's first rows by id too (empty baseline)", async () => {
		/* The waking answer saw no rows, so the baseline is "". */
		const start = Date.now();
		const view = mount(frame({ transcript: [] }));
		await sendWaking(view);
		const clock = vi.spyOn(Date, "now").mockReturnValue(start + WAKE_SETTLED_BOUND_MS + 1_000);
		/* A supervisor wake wrote two rows; ours is not among them. */
		const tail = filler(0, 2);
		serveHistory(tail);
		const back = switchAway(view, frame({ version: 2, transcript: tail }));
		back.publish(frame({ version: 3, transcript: tail }));

		expect(await screen.findByText(UNCONFIRMED_COPY)).toBeTruthy();
		expect(pendingRows(back)).toHaveLength(0);
		expect((await composer()).value).toBe(SENT);
		expect(localStorage.getItem(STORAGE_KEY)).toContain(COMMAND_ID);
		clock.mockRestore();
	});

	it("does not read an ended live frame's tail against the pid-0 baseline (R2-m1)", async () => {
		/* A dead live owner's last frame can end in a transient notice row no
		   rebuild reproduces: the ordinary failure path, no history search. */
		const tail = filler(0, 80);
		const { back, clock } = await backPastBound(tail, { pid: 4242, kind: "tui", ended: true });

		await waitFor(() => expect(screen.getByRole("alert").textContent).toContain(FAILED_COPY));
		expect(confirmCalls()).toHaveLength(0);
		expect(pendingRows(back)).toHaveLength(0);
		expect(localStorage.getItem(STORAGE_KEY)).toContain(COMMAND_ID);
		clock.mockRestore();
	});

	/* REVIEW ROUND 1, m1 (PR #4584): a Retry inside the bound, after a failure
	   nobody saw, starts a NEW relay task with a fresh deadline. Measuring from
	   the FIRST receipt failed a boot that was still running. Measured RED on
	   b4adabdd: the failure copy shows at first-receipt + bound. */
	it("measures the bound from the latest waking answer, a retry's too", async () => {
		const start = Date.now();
		const view = mount();
		await sendWaking(view);
		/* Away while the first wake failed; back 150 s later, the remount shows
		   the retained-retry affordance and the user taps it. */
		const clock = vi.spyOn(Date, "now").mockReturnValue(start + 150_000);
		const back = switchAway(view);
		vi.mocked(api.sendCommand).mockResolvedValueOnce(WAKING);
		fireEvent.click(await screen.findByRole("button", { name: "Retry earlier instruction" }));
		await waitFor(() => expect(vi.mocked(api.sendCommand)).toHaveBeenCalledTimes(2));
		await waitFor(() => expect(pendingRows(back)[0]?.textContent).toContain("waking session…"));

		/* Past the FIRST receipt's bound, inside the retry's: still booting. */
		clock.mockReturnValue(start + WAKE_SETTLED_BOUND_MS + 1_000);
		back.publish(frame({ version: 2 }));
		expect(screen.queryByText(FAILED_COPY)).toBeNull();
		expect(pendingRows(back)).toHaveLength(1);

		/* Past the retry's own bound: now it has settled, and failed. */
		clock.mockReturnValue(start + 150_000 + WAKE_SETTLED_BOUND_MS + 1_000);
		back.publish(frame({ version: 3 }));
		await waitFor(() => expect(screen.getByRole("alert").textContent).toContain(FAILED_COPY));
		expect(pendingRows(back)).toHaveLength(0);
		clock.mockRestore();
	});

	/* QA ROUND 2, Q2-1 (PR #4584): the search's first cursor was
	   `transcript[0]`, which on a capped projection is the PINNED OPENER —
	   history before it is empty, so a delivered wake read as unconfirmed.
	   Expected RED on c94b2d5e (not yet run locally — host load gate; see PR evidence): first cursor `opener`, unconfirmed copy shown. */
	it("pages from below a pinned opener, finding a delivery in the gap", async () => {
		/* 85 rows after ours: the tail is the opener plus the newest 79, so our
		   row and the 6 rows after it sit in the gap. */
		const opener = userRow("opener", "the first message");
		const after = filler(100, 85);
		serveHistory([opener, ...filler(0, 20), userRow(COMMAND_ID, SENT), ...after]);
		const tail = [opener, ...after.slice(-79)];
		const { back, clock } = await backPastBound(tail, {
			transcript_head_pinned: true, transcript_has_older: true,
		});

		await waitFor(() => expect(localStorage.getItem(STORAGE_KEY)).toBeNull());
		expect(confirmCalls()[0]?.[1]).toBe("later-106");
		expect(screen.queryByText(UNCONFIRMED_COPY)).toBeNull();
		expect(pendingRows(back)).toHaveLength(0);
		expect((await composer()).value).toBe("");
		expect(screen.queryByRole("button", { name: "Retry earlier instruction" })).toBeNull();
		clock.mockRestore();
	});

	it("reads the pinned opener itself as the command's row", async () => {
		/* Our message opened the session: it IS the pinned head, no walk needed. */
		const tail = [userRow(COMMAND_ID, SENT), ...filler(0, 79)];
		const { back, clock } = await backPastBound(tail, { transcript_head_pinned: true });

		await waitFor(() => expect(localStorage.getItem(STORAGE_KEY)).toBeNull());
		expect(confirmCalls()).toHaveLength(0);
		expect(pendingRows(back)).toHaveLength(0);
		clock.mockRestore();
	});

	/* REVIEW ROUND 3, R3-m3: an unreadable history is not a delivery. */
	it("reads a failed history walk as unconfirmed, keeping the envelope", async () => {
		vi.mocked(api.getHistory).mockRejectedValue(new Error("history unavailable"));
		const { back, clock } = await backPastBound(filler(1000, 80));

		expect(await screen.findByText(UNCONFIRMED_COPY)).toBeTruthy();
		expect(confirmCalls()).toHaveLength(1);
		expect(pendingRows(back)).toHaveLength(0);
		expect((await composer()).value).toBe(SENT);
		expect(localStorage.getItem(STORAGE_KEY)).toContain(COMMAND_ID);
		clock.mockRestore();
	});

	/* REVIEW ROUND 3, R3-m3: a retry that lands mid-walk owns the outcome; the
	   stale walk's verdict must not touch it. */
	it("ignores a walk's verdict once a retry has taken over", async () => {
		let release: (page: { entries: TranscriptEntry[]; has_more: boolean }) => void = () => {};
		vi.mocked(api.getHistory).mockImplementation(
			(_sid, _before, limit) =>
				limit === 80
					? new Promise((resolve) => { release = resolve; })
					: Promise.resolve({ entries: [], has_more: false }),
		);
		const { back, clock } = await backPastBound(filler(1000, 80));
		await waitFor(() => expect(confirmCalls()).toHaveLength(1));

		vi.mocked(api.sendCommand).mockResolvedValueOnce(WAKING);
		fireEvent.click(await screen.findByRole("button", { name: "Retry earlier instruction" }));
		await waitFor(() => expect(vi.mocked(api.sendCommand)).toHaveBeenCalledTimes(2));
		await waitFor(() => expect(pendingRows(back)[0]?.textContent).toContain("waking session…"));

		release({ entries: filler(0, 80), has_more: false });
		back.publish(frame({ version: 4, transcript: filler(1000, 80) }));
		await new Promise((resolve) => setTimeout(resolve, 0));
		expect(screen.queryByText(UNCONFIRMED_COPY)).toBeNull();
		expect(screen.queryByText(FAILED_COPY)).toBeNull();
		expect(pendingRows(back)[0]?.textContent).toContain("waking session…");
		expect(localStorage.getItem(STORAGE_KEY)).toContain(COMMAND_ID);
		clock.mockRestore();
	});

	/* REVIEW ROUND 3, R3-m1: the walk finds the row after another surface has
	   booted the session (a live frame, so the bound no longer holds). The row
	   lives in older history, not the tail, so only the composer can retire
	   it. Expected RED on c94b2d5e (not yet run locally — host load gate; see PR evidence): the row stays on screen as "sending…". */
	it("withdraws the row on a found verdict that resolves after the session went live", async () => {
		const older = [userRow(COMMAND_ID, SENT), ...filler(0, 10)];
		let release: () => void = () => {};
		vi.mocked(api.getHistory).mockImplementation(
			(_sid, _before, limit) =>
				limit === 80
					? new Promise((resolve) => { release = () => resolve({ entries: older, has_more: false }); })
					: Promise.resolve({ entries: [], has_more: false }),
		);
		const tail = filler(1000, 80);
		const { back, clock } = await backPastBound(tail);
		await waitFor(() => expect(confirmCalls()).toHaveLength(1));

		back.publish(frame({ version: 4, pid: 4242, kind: "tui", transcript: tail }));
		release();

		await waitFor(() => expect(localStorage.getItem(STORAGE_KEY)).toBeNull());
		expect(pendingRows(back)).toHaveLength(0);
		clock.mockRestore();
	});

	/* QA ROUND 2, row 4 and Q2-2 (PR #4584): a wake that failed while its
	   composer WATCHED withdrew the waking row, and the row was the only fact a
	   remount rebuilt the tracker from. Back on the route the composer saw a bare
	   envelope and showed the generic retained-retry copy, and never ran the
	   moved-tail check. Expected RED on c94b2d5e (not yet run locally — host load gate; see PR evidence): all three cells show
	   "An earlier instruction may have been delivered…" and make 0 walks. */
	const RETAINED_COPY =
		"An earlier instruction may have been delivered. Retry that earlier instruction before sending your current draft.";
	async function watchUntilFailure(transcript: TranscriptEntry[]) {
		const view = mount(frame({ transcript }));
		await sendWaking(view);
		view.publish(frame({ version: 2, transcript, ended: true, wake_failed: COMMAND_ID }));
		await waitFor(() => expect(screen.getByRole("alert").textContent).toContain(FAILED_COPY));
		expect(pendingRows(view)).toHaveLength(0);
		return view;
	}
	async function failWhileWatching(transcript: TranscriptEntry[]) {
		const start = Date.now();
		const view = await watchUntilFailure(transcript);
		const clock = vi.spyOn(Date, "now").mockReturnValue(start + WAKE_SETTLED_BOUND_MS + 60_000);
		return { view, clock };
	}

	it("keeps a watched failure's copy across a remount when the tail is unchanged (row 4)", async () => {
		const { view, clock } = await failWhileWatching([userRow("existing", "an older message")]);
		const back = switchAway(view, frame({ version: 3 }));
		back.publish(frame({ version: 4 }));

		await waitFor(() => expect(screen.getByRole("alert").textContent).toContain(FAILED_COPY));
		expect(screen.queryByText(RETAINED_COPY)).toBeNull();
		expect(screen.getByRole("button", { name: "Retry earlier instruction" })).toBeTruthy();
		expect((await composer()).value).toBe(SENT);
		expect(confirmCalls()).toHaveLength(0);
		expect(localStorage.getItem(STORAGE_KEY)).toContain(COMMAND_ID);
		clock.mockRestore();
	});

	it("walks history for an empty-baseline session written to after a watched failure (Q2-2)", async () => {
		const { view, clock } = await failWhileWatching([]);
		/* A notice row lands; our row is nowhere. */
		const tail = filler(0, 1);
		serveHistory(tail);
		const back = switchAway(view, frame({ version: 3, transcript: tail }));
		back.publish(frame({ version: 4, transcript: tail }));

		await waitFor(() => expect(confirmCalls()).toHaveLength(1));
		expect(confirmCalls()[0]?.[1]).toBe("later-0");
		await waitFor(() => expect(screen.getByRole("alert").textContent).toContain(FAILED_COPY));
		expect(screen.queryByText(RETAINED_COPY)).toBeNull();
		expect((await composer()).value).toBe(SENT);
		expect(localStorage.getItem(STORAGE_KEY)).toContain(COMMAND_ID);
		clock.mockRestore();
	});

	it("settles a watched failure as delivered when the walk finds the row after all", async () => {
		/* The failed wake's runtime booted late and wrote our row, then a long turn. */
		const { view, clock } = await failWhileWatching([]);
		const tail = filler(0, 80);
		serveHistory([userRow(COMMAND_ID, SENT), ...tail]);
		const back = switchAway(view, frame({ version: 3, transcript: tail }));
		back.publish(frame({ version: 4, transcript: tail }));

		await waitFor(() => expect(localStorage.getItem(STORAGE_KEY)).toBeNull());
		await waitFor(() => expect(screen.queryByText(FAILED_COPY)).toBeNull());
		expect(screen.queryByText(RETAINED_COPY)).toBeNull();
		expect((await composer()).value).toBe("");
		expect(screen.queryByRole("button", { name: "Retry earlier instruction" })).toBeNull();
		clock.mockRestore();
	});

	it("reconciles a watched failure at the bound with only a moved cold seed", async () => {
		vi.useFakeTimers({ shouldAdvanceTime: true, toFake: ["setTimeout", "clearTimeout", "Date"] });
		try {
			const view = await watchUntilFailure([]);
			const tail = filler(1000, 80);
			serveHistory([userRow(COMMAND_ID, SENT), ...tail]);
			switchAway(view, frame({ version: 3, transcript: tail, transcript_has_older: true }));

			expect(screen.getByRole("alert").textContent).toContain(FAILED_COPY);
			expect(screen.getByRole("button", { name: "Retry earlier instruction" })).toBeTruthy();
			expect((await composer()).value).toBe(SENT);
			expect(localStorage.getItem(STORAGE_KEY)).toContain(COMMAND_ID);
			expect(confirmCalls()).toHaveLength(0);

			/* No publish after remount: only the deadline can start this walk. */
			await act(async () => { await vi.advanceTimersByTimeAsync(WAKE_SETTLED_BOUND_MS); });
			await waitFor(() => expect(localStorage.getItem(STORAGE_KEY)).toBeNull());
			expect(confirmCalls().map((call) => call[1])).toEqual(["later-1000"]);
			expect(screen.queryByText(FAILED_COPY)).toBeNull();
			expect(screen.queryByRole("button", { name: "Retry earlier instruction" })).toBeNull();
			expect((await composer()).value).toBe("");

			await act(async () => { await vi.advanceTimersByTimeAsync(WAKE_SETTLED_BOUND_MS); });
			expect(confirmCalls()).toHaveLength(1);
		} finally {
			vi.useRealTimers();
		}
	});

	it("leaves a retained envelope with no waking row on its retained-retry path", async () => {
		/* A reload (or an ambiguous 504) retains an envelope too; with no waking
		   row there is no wake to settle, so the old affordance stands. */
		localStorage.setItem(
			STORAGE_KEY,
			JSON.stringify({
				version: 1,
				saved_at: Date.now(),
				envelope: { op: "prompt", command_id: COMMAND_ID, text: SENT },
			}),
		);
		vi.spyOn(Date, "now").mockReturnValue(Date.now() + WAKE_SETTLED_BOUND_MS + 1_000);
		const view = mount();
		view.publish(frame({ version: 2 }));

		expect(await screen.findByRole("button", { name: "Retry earlier instruction" })).toBeTruthy();
		expect(screen.queryByText(FAILED_COPY)).toBeNull();
	});
});
