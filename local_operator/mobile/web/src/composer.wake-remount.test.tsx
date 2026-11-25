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

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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
