// @vitest-environment happy-dom
//
// A COLD PROMPT ANSWERED `waking`: how the composer settles it from the stream
// (agent review round 2, R2-1 and R2-2 on PR #4522).
//
// THE TRAP THESE CELLS PIN. A cold conversation is served as a disk rebuild
// with `pid: 0`, and the header dot's health (`sessionHealth`) reads that as
// `ended` — before the wake, during it, and on every reconnect's seed. The
// first settle effect keyed the failure on that health, so:
//   - a boot that failed before its first live frame was never reported: the
//     row said "sending…" for good (cell 1);
//   - a reconnect blip mid-wake (health `reconnecting`, then the pid-0 seed
//     again) read as a failure while the boot carried on, and nothing cleared
//     the failure when the message then landed (cells 2 and 3).
// The failure signal is now a fact about the wake: the relay's
// `wake_failed: <command_id>` marker on the terminal frame it publishes for a
// failed wake, or (a relay without the marker) `ended` after a frame from a
// running owner (cell 4).
//
// Expected RED on e1a2d2a1 by construction (there the effect reads the
// screen's `ended`, true from the pid-0 seed onward, and the waking branch
// says nothing): R2-1a, R2-1b, both "lands after" cells and R2-2. The
// other-command marker and the no-marker fallback cells are guards that hold
// on both heads.
//
// Like `pending-echo.test.tsx`, the render site is the REAL `SessionScreen`
// and `Composer` — the health that caused this is computed between them —
// with only the projection slot and the network faked.

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SessionScreen } from "./screens/session-view";
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
/* The English copy of `relay.health.ended_send_failed`, as
   `composer-resume-label.test.tsx` pins it. */
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
		id,
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
		final: true,
	};
}

/** A frame. The default is the COLD seed: a disk rebuild, `pid: 0`, not ended —
 *  what the relay serves a conversation nobody is running. */
function frame(overrides: Partial<SessionProjection> = {}): SessionProjection {
	return {
		session_id: "s1",
		pid: 0,
		kind: "daemon",
		conversation_name: "Cold",
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
		transcript: [userRow("existing", "an older message")],
		todos: [],
		subagents: [],
		pending: null,
		pending_count: 0,
		usage: {},
		version: 1,
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

function open() {
	slot = { projection: frame(), connected: true };
	const view = render(<SessionScreen sessionId="s1" />);
	return {
		...view,
		/** Publish what the SSE stream would: a new frame and/or link state. */
		publish(next: SessionProjection, connected = true) {
			slot = { projection: next, connected };
			view.rerender(<SessionScreen sessionId="s1" />);
		},
	};
}

async function composer(): Promise<HTMLTextAreaElement> {
	return (await screen.findByRole("textbox", { name: "Message" })) as HTMLTextAreaElement;
}

async function sendText(text: string) {
	fireEvent.change(await composer(), { target: { value: text } });
	fireEvent.click(screen.getByRole("button", { name: "send" }));
}

function pendingRows(view: { container: HTMLElement }): Element[] {
	return Array.from(view.container.querySelectorAll(`[data-pending-echo="${COMMAND_ID}"]`));
}

/** Send SENT and let the relay answer `waking`: the row is up, the composer is
 *  empty, the envelope is retained, and nothing says it failed. */
async function sendWaking(view: ReturnType<typeof open>) {
	vi.mocked(api.sendCommand).mockResolvedValueOnce(WAKING);
	await sendText(SENT);
	await waitFor(() => expect(vi.mocked(api.sendCommand)).toHaveBeenCalledOnce());
	await waitFor(() => expect(pendingRows(view)).toHaveLength(1));
	expect((await composer()).value).toBe("");
	expect(screen.queryByRole("alert")).toBeNull();
	expect(localStorage.getItem(STORAGE_KEY)).toContain(COMMAND_ID);
	/* The header already reads the cold session as Ended: the precondition that
	   made the health-keyed effect wrong. */
	expect(document.querySelector<HTMLElement>("button[data-health]")?.dataset.health).toBe("ended");
}

describe("a cold prompt answered `waking`", () => {
	it("reports a boot that failed before any live frame: alert, row down, draft back (R2-1a)", async () => {
		const view = open();
		await sendWaking(view);

		/* `_publish_failed_wake`: terminal, ended, pid 0, naming the command. */
		view.publish(frame({ ended: true, wake_failed: COMMAND_ID }));

		await waitFor(() => expect(screen.getByRole("alert").textContent).toContain(FAILED_COPY));
		expect(pendingRows(view)).toHaveLength(0);
		expect((await composer()).value).toBe(SENT);
		expect(screen.getByRole("button", { name: "Retry earlier instruction" })).toBeTruthy();
		/* The envelope stays: it is the retry's body, under the SAME id. */
		expect(localStorage.getItem(STORAGE_KEY)).toContain(COMMAND_ID);
	});

	it("ignores another command's failure marker", async () => {
		const view = open();
		await sendWaking(view);

		view.publish(frame({ ended: true, wake_failed: "another-command" }));

		expect(screen.queryByRole("alert")).toBeNull();
		expect(pendingRows(view)).toHaveLength(1);
	});

	it("does not fail on a reconnect blip during the wake (R2-1b)", async () => {
		const view = open();
		await sendWaking(view);

		/* The link drops (health `reconnecting`), then the reconnect serves the
		   pid-0 seed again (health `ended`) while the boot is still running. */
		view.publish(frame(), false);
		view.publish(frame(), true);

		expect(screen.queryByRole("alert")).toBeNull();
		expect(pendingRows(view)).toHaveLength(1);
		expect((await composer()).value).toBe("");
		expect(localStorage.getItem(STORAGE_KEY)).toContain(COMMAND_ID);
	});

	it("settles cleanly when the message lands after a blip: no retry copy, no alert", async () => {
		const view = open();
		await sendWaking(view);
		view.publish(frame(), false);
		view.publish(frame(), true);

		/* The runtime is up and wrote the row under the envelope's id. */
		view.publish(
			frame({ pid: 4242, kind: "tui", transcript: [userRow("existing", "an older message"), userRow(COMMAND_ID, SENT)] }),
		);

		await waitFor(() => expect(localStorage.getItem(STORAGE_KEY)).toBeNull());
		expect(screen.queryByRole("alert")).toBeNull();
		expect(screen.queryByRole("button", { name: "Retry earlier instruction" })).toBeNull();
		expect((await composer()).value).toBe("");
	});

	it("takes the failure copy down when the message lands after a reported failure", async () => {
		const view = open();
		await sendWaking(view);
		view.publish(frame({ ended: true, wake_failed: COMMAND_ID }));
		await waitFor(() => expect(screen.getByRole("alert").textContent).toContain(FAILED_COPY));

		/* Another surface woke the session (or a retry joined) and the message
		   was written after all. Leaving "send failed" and the restored draft up
		   would invite a second send under a NEW id. */
		view.publish(
			frame({ pid: 4242, kind: "tui", transcript: [userRow("existing", "an older message"), userRow(COMMAND_ID, SENT)] }),
		);

		await waitFor(() => expect(screen.queryByRole("alert")).toBeNull());
		expect(localStorage.getItem(STORAGE_KEY)).toBeNull();
		expect(screen.queryByRole("button", { name: "Retry earlier instruction" })).toBeNull();
		expect((await composer()).value).toBe("");
	});

	it("falls back to `ended` after a live owner's frame when the relay sends no marker", async () => {
		const view = open();
		await sendWaking(view);

		view.publish(frame({ pid: 4242, kind: "tui" }));
		expect(screen.queryByRole("alert")).toBeNull();
		/* A terminal frame without `wake_failed`, as an older relay sends it. */
		view.publish(frame({ ended: true }));

		await waitFor(() => expect(screen.getByRole("alert").textContent).toContain(FAILED_COPY));
		expect(pendingRows(view)).toHaveLength(0);
		expect((await composer()).value).toBe(SENT);
	});

	it("says so when a new draft's send replays the earlier, still-waking instruction (R2-2)", async () => {
		const view = open();
		await sendWaking(view);

		/* The retained envelope goes first (same id, its own bytes); the relay
		   answers `waking` again; the new draft stays in the composer. */
		vi.mocked(api.sendCommand).mockResolvedValueOnce(WAKING);
		await sendText("a second thought");
		await waitFor(() => expect(vi.mocked(api.sendCommand)).toHaveBeenCalledTimes(2));
		expect(vi.mocked(api.sendCommand).mock.calls[1]?.[1]).toMatchObject({
			op: "prompt",
			command_id: COMMAND_ID,
			text: SENT,
		});

		await waitFor(() =>
			expect(screen.getByText("Earlier instruction is still sending. Your draft is ready to send.")).toBeTruthy(),
		);
		expect((await composer()).value).toBe("a second thought");
		expect(screen.queryByRole("alert")).toBeNull();
	});
});
