// @vitest-environment happy-dom
//
// The optimistic pending row: the message the user just sent, painted before
// the daemon has answered and reconciled against the receipt.
//
// What these cells pin, and why each one has to be able to fail:
//
//   - the row exists BEFORE the receipt resolves. Without the optimistic path
//     the user's text is only ever in the composer's textarea, which is the
//     reported "I tap send and nothing happens" — so this cell holds the
//     receipt open and reads the screen.
//   - exactly ONE row survives resolution. The pending row and the real row are
//     two renderings of one message, so a reconciliation that failed to drop
//     the pending one would show the user their own sentence twice. Counted at
//     the moment the projection carries the id — the frame a "which one wins"
//     bug would paint.
//   - a failure leaves NO row behind. A row that outlives a failed send is a
//     phantom message the user believes was delivered, and the honest state is
//     the composer's own alert with its same-envelope retry.
//   - the retry path paints no second row for a command the session already
//     wrote, and does not throw away a draft typed since.
//
// The render site is the REAL `SessionScreen` and the REAL `Composer`, with
// only the projection slot and the network faked, because the reconciliation
// lives in the wiring between them (who registers the echo, who reads the
// transcript) rather than inside any one component.

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SessionScreen } from "./screens/session-view";
import * as api from "./api";
import { clearPendingEchoes, registerPendingEcho } from "./pending-echo";
import type { SessionProjection, TranscriptEntry } from "./types";

vi.mock("./api", async (importOriginal) => ({
	...(await importOriginal<typeof api>()),
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentDetail: vi.fn(async () => null),
	imageUrl: vi.fn(() => ""),
	getCommands: vi.fn(async () => ({ commands: [] })),
	getModels: vi.fn(async () => ({ models: [] })),
	sendCommand: vi.fn(),
	markSessionSeen: vi.fn(async () => ({ ok: true })),
}));

/* The id the composer mints. It is also the id the session writes the user row
   under (`message_id=command.command_id` → `Message.id` → the fold's entry id),
   which is the whole basis of the reconciliation under test: the projection
   carries the SAME string. */
const COMMAND_ID = "12345678-1234-4678-9234-567812345678";
const SENT = "hello from the phone";

let slot: { projection: SessionProjection | null; connected: boolean } = {
	projection: null,
	connected: true,
};

/* Only the projection slot and the stream are faked; the rest of the store is
   real, so `useDraft` is the production draft store the composer types into. */
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useProjection: vi.fn(() => slot),
		retainProjectionStream: vi.fn(() => () => {}),
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

function projection(transcript: TranscriptEntry[], streaming = false): SessionProjection {
	return {
		session_id: "s1",
		pid: 1,
		kind: "tui",
		conversation_name: "Echo",
		cwd: "",
		model_label: "",
		model_selector: "",
		effort: "",
		effort_ladder: [],
		streaming,
		activity: "",
		activity_started_s: 0,
		stop_reason: "",
		queued_count: 0,
		ended: false,
		degraded: false,
		transcript,
		todos: [],
		subagents: [],
		pending: null,
		pending_count: 0,
		usage: {},
		version: 1,
	};
}

afterEach(() => {
	cleanup();
	localStorage.clear();
	/* The store is module state, which the suite shares across cells: a leaked
	   echo would make the next cell's "no row" assertion read someone else's. */
	clearPendingEchoes();
	vi.unstubAllGlobals();
	vi.restoreAllMocks();
});

beforeEach(() => {
	/* No queued reply survives into the next cell: a send left to fall through to
	   no implementation at all would fail for a reason this file is not about. */
	vi.mocked(api.sendCommand).mockReset();
});

function open(transcript: TranscriptEntry[] = [], streaming = false) {
	slot = { projection: projection(transcript, streaming), connected: true };
	const view = render(<SessionScreen sessionId="s1" />);
	return {
		...view,
		/** Publish a new projection snapshot, as the SSE repaint would. */
		publish(next: TranscriptEntry[], nextStreaming = false) {
			slot = { projection: projection(next, nextStreaming), connected: true };
			view.rerender(<SessionScreen sessionId="s1" />);
		},
	};
}

async function type(text: string) {
	const composer = (await screen.findByPlaceholderText(
		"Message Local Operator…",
	)) as HTMLTextAreaElement;
	fireEvent.change(composer, { target: { value: text } });
	return composer;
}

function send() {
	fireEvent.click(screen.getByRole("button", { name: "send" }));
}

/** The pending rows on screen, by the row's own identity rather than by its
 *  class names — and NOT by `getByText`, which also matches a textarea's value
 *  and so cannot tell a row from the draft handed back into the composer. */
function pendingRows(view: { container: HTMLElement }, commandId = COMMAND_ID): Element[] {
	return Array.from(view.container.querySelectorAll(`[data-pending-echo="${commandId}"]`));
}

/** A promise the test releases by hand, so "before the receipt" is a state the
 *  screen can be READ in rather than a race the assertion has to win. */
function heldReceipt() {
	let release!: (value: { ok: boolean; detail: string }) => void;
	const promise = new Promise<{ ok: boolean; detail: string }>((resolve) => {
		release = resolve;
	});
	return { promise, release };
}

describe("optimistic pending echo", () => {
	it("paints the message as a pending row before the receipt, and moves it out of the composer", async () => {
		const receipt = heldReceipt();
		vi.mocked(api.sendCommand).mockReturnValueOnce(receipt.promise);
		vi.stubGlobal("crypto", { randomUUID: () => COMMAND_ID });
		const view = open();

		const composer = await type(SENT);
		send();

		/* The daemon has NOT answered — `sendCommand` is still holding — and the
		   user's message is already in the conversation, as its own pending row. */
		await waitFor(() => expect(pendingRows(view)).toHaveLength(1));
		expect(pendingRows(view)[0]?.textContent).toContain(SENT);
		expect(screen.getByText("sending…")).toBeTruthy();
		expect(screen.getAllByText(SENT)).toHaveLength(1);
		/* The draft MOVED rather than being copied: the same words in the textarea
		   and in a row would be the double display the row exists to remove. */
		expect(composer.value).toBe("");
		/* And nothing about it claims an outcome: no failure alert, and the
		   request really is in flight. */
		expect(screen.queryByRole("alert")).toBeNull();
		expect(vi.mocked(api.sendCommand)).toHaveBeenCalledOnce();
		expect(vi.mocked(api.sendCommand).mock.calls[0]?.[1]).toMatchObject({
			op: "prompt",
			command_id: COMMAND_ID,
			text: SENT,
		});

		receipt.release({ ok: true, detail: "prompt admitted" });
		await waitFor(() => expect(pendingRows(view)).toHaveLength(1));
	});

	it("resolves to exactly one row once the projection writes it", async () => {
		vi.mocked(api.sendCommand).mockResolvedValueOnce({ ok: true, detail: "prompt admitted" });
		vi.stubGlobal("crypto", { randomUUID: () => COMMAND_ID });
		const view = open();

		await type(SENT);
		send();

		/* The receipt has landed, but the projection has not carried the row yet
		   — the window in which a "drop the pending row on the ACK" rule would
		   leave the user with nothing at all. The row stays, and there is one. */
		await waitFor(() => expect(pendingRows(view)).toHaveLength(1));
		expect(screen.getAllByText(SENT)).toHaveLength(1);

		view.publish([userRow(COMMAND_ID, SENT)]);

		await waitFor(() => expect(pendingRows(view)).toHaveLength(0));
		/* ONE row, not two: the projection's own row under the same id is the
		   only one left, and the pending row beside it is gone. */
		expect(screen.getAllByText(SENT)).toHaveLength(1);
	});

	it("takes the row down on failure and hands the text back for retry", async () => {
		vi.mocked(api.sendCommand).mockRejectedValueOnce(new Error("response lost"));
		vi.stubGlobal("crypto", { randomUUID: () => COMMAND_ID });
		const view = open();

		const composer = await type(SENT);
		send();

		await waitFor(() => expect(screen.getByRole("alert")).toBeTruthy());
		/* No phantom: the row is gone, because the message never went. */
		expect(pendingRows(view)).toHaveLength(0);
		expect(screen.queryByText("sending…")).toBeNull();
		/* The text is back where the user can act on it — it was moved out of the
		   composer at submit, so this is the only copy left. */
		expect(composer.value).toBe(SENT);
		/* And the retry is offered under the SAME envelope id, which is what keeps
		   a possibly-delivered instruction from being sent twice. */
		expect(screen.getByRole("button", { name: "Retry earlier instruction" })).toBeTruthy();
	});

	it("paints no second row for a retry of a command the session already wrote", async () => {
		vi.mocked(api.sendCommand).mockResolvedValueOnce({ ok: true, detail: "already admitted" });
		/* The session already has the row — the acknowledgement-loss case, where
		   the first attempt landed and only its reply was lost — and the immutable
		   envelope for that command is what the composer's mount restores. */
		localStorage.setItem(
			"lo-mobile-command:s1",
			JSON.stringify({
				version: 1,
				saved_at: Date.now(),
				envelope: { command_id: COMMAND_ID, op: "prompt", text: SENT },
			}),
		);
		open([userRow(COMMAND_ID, SENT)]);

		const retry = await screen.findByRole("button", { name: "Retry earlier instruction" });
		fireEvent.click(retry);

		await waitFor(() => expect(vi.mocked(api.sendCommand)).toHaveBeenCalledOnce());
		/* The SAME envelope id, never a fresh one: that is what makes a second
		   delivery of an already-admitted instruction impossible. */
		expect(vi.mocked(api.sendCommand).mock.calls[0]?.[1]).toMatchObject({
			op: "prompt",
			command_id: COMMAND_ID,
		});
		/* And ONE row throughout — the projection's, never a pending twin of it. */
		expect(screen.getAllByText(SENT)).toHaveLength(1);
		expect(screen.queryByText("sending…")).toBeNull();
	});

	it("states an attachment count on the pending row", async () => {
		/* The bytes are never duplicated into the row: the composer keeps its own
		   previews and revokes them on acknowledgement, so the row carries the
		   COUNT only — the shape the TUI's own prompt receipt uses. Registered
		   directly because the composer's attach path needs `createImageBitmap`,
		   which the test environment does not have; what is under test here is the
		   row's rendering of a count, not the downscale in front of it. */
		registerPendingEcho("s1", { commandId: COMMAND_ID, text: SENT, imageCount: 2 });
		const view = open([], true);

		await waitFor(() => expect(pendingRows(view)).toHaveLength(1));
		expect(screen.getByText("2 images attached")).toBeTruthy();
	});

	it("drops echoes when authentication changes ownership", async () => {
		/* Direct: the purge is called by `clearPrivateSessionStorage`, which both
		   the 401 handler and the login page's sweep reach. A row left standing
		   would show the next user at this device what the last one typed. */
		registerPendingEcho("s1", { commandId: COMMAND_ID, text: SENT, imageCount: 0 });
		const view = open([], true);
		await waitFor(() => expect(pendingRows(view)).toHaveLength(1));

		clearPendingEchoes();

		await waitFor(() => expect(pendingRows(view)).toHaveLength(0));
		view.publish([], true);
		expect(pendingRows(view)).toHaveLength(0);
	});
});
