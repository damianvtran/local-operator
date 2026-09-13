// @vitest-environment happy-dom
//
// Design round 2, D7: the resume button's WORD has to agree with the notice it
// sits under. `stop_reason: "aborted"` covers two acts — a deliberate stop and a
// harness cut-off — and once the cut-off taxonomy made an involuntary end paint
// `Stopped with an error — ...` in danger ink, a button reading `interrupted —
// tap to resume` named one act two ways. It also appeared in a case it never
// used to: before that change a cut-off produced `stop_reason: "completed"`, so
// the button did not render at all.
//
// Renders the REAL SessionScreen (the render site that owns the layout and the
// composer), the idiom session-view.pending.test.tsx established, so a reverted
// label fails here rather than in a copy of the string.
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SessionScreen } from "./screens/session-view";
import type { SessionProjection, TranscriptEntry } from "./types";

vi.mock("./api", () => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentDetail: vi.fn(async () => null),
	imageUrl: vi.fn(() => ""),
	getCommands: vi.fn(async () => ({ commands: [] })),
	getModels: vi.fn(async () => ({ models: [] })),
	sendCommand: vi.fn(async () => ({ ok: true, detail: "" })),
	markSessionSeen: vi.fn(async () => ({ ok: true })),
}));

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
		useDraft: vi.fn(() => ["", () => {}]),
	};
});

/** A cut-off screen: the danger notice #970 paints, and a resumable turn. */
function projection(opts: { cut_off: boolean | undefined }): SessionProjection {
	const p = {
		session_id: "s1",
		pid: 1,
		kind: "tui",
		conversation_name: "Cut off",
		cwd: "",
		model_label: "",
		model_selector: "",
		effort: "",
		effort_ladder: [],
		streaming: false,
		activity: "",
		activity_started_s: 0,
		stop_reason: "aborted",
		queued_count: 0,
		ended: false,
		degraded: false,
		transcript: [
			{
				id: "err-1",
				kind: "notice" as const,
				text: "Stopped with an error — the session's runtime stopped answering while this turn was running",
				details: { severity: "error" as const },
			} as unknown as TranscriptEntry,
		],
		todos: [],
		subagents: [],
		pending: null,
		pending_count: 0,
		usage: {},
		version: 1,
	} as SessionProjection;
	if (opts.cut_off !== undefined) p.cut_off = opts.cut_off;
	return p;
}

afterEach(() => {
	cleanup();
	localStorage.clear();
	slot = { projection: null, connected: true };
});

describe("the phone's resume button", () => {
	it("says the turn was cut off, under a cut-off notice", () => {
		const { rerender } = render(<SessionScreen sessionId="s1" />);
		slot = { projection: projection({ cut_off: true }), connected: true };
		rerender(<SessionScreen sessionId="s1" />);

		expect(screen.getByText(/Stopped with an error/)).toBeTruthy();
		expect(screen.getByRole("button", { name: "turn cut off — tap to resume" })).toBeTruthy();
		expect(screen.queryByText("interrupted — tap to resume")).toBeNull();
	});

	it("keeps the deliberate stop's own word", () => {
		const { rerender } = render(<SessionScreen sessionId="s1" />);
		slot = { projection: projection({ cut_off: false }), connected: true };
		rerender(<SessionScreen sessionId="s1" />);

		expect(screen.getByRole("button", { name: "interrupted — tap to resume" })).toBeTruthy();
	});

	it("keeps today's word against a daemon that does not send the field", () => {
		// An older daemon omits `cut_off` entirely. Absence must not invent a
		// verdict nobody sent, and it must not remove the affordance either —
		// that is why the flag rides beside `stop_reason` instead of replacing
		// it with a third value.
		const { rerender } = render(<SessionScreen sessionId="s1" />);
		const without = projection({ cut_off: undefined });
		expect("cut_off" in without).toBe(false);
		slot = { projection: without, connected: true };
		rerender(<SessionScreen sessionId="s1" />);

		expect(screen.getByRole("button", { name: "interrupted — tap to resume" })).toBeTruthy();
	});
});
