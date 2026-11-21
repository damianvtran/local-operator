// @vitest-environment happy-dom
//
// `?steer_journal_order=1` (store.ts) asks the relay to MOVE a delivered phone
// steer from the slot it was painted in at send time to its journal slot, and
// to repaint it as a `user` row — the order and kind `/history` already has.
// This client declares it unconditionally, which is only safe if the move can
// never duplicate, drop or scramble a row here. Two cases cover every client:
//
//  - HELD: the tab watched the steer being sent. The held list upserts by
//    `rowKey` (the id), so the moved row keeps the slot it already holds and
//    simply repaints; later rows still land at the tail.
//  - FRESH MOUNT: a tab opened after delivery takes the frame as-is, which is
//    now journal order — the same order a history page gives.
import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { Transcript } from "./transcript";
import type { TranscriptEntry } from "../types";

afterEach(cleanup);

const row = (id: string, kind: TranscriptEntry["kind"], text: string): TranscriptEntry => ({
	id, kind, text, tool_call_id: "", tool_name: "", tool_state: "done",
	summary: "", intent: "", diff_added: 0, diff_removed: 0, elapsed_s: 0, error: "",
	details: {}, final: true,
});

const U1 = row("u1", "user", "PROMPT-U1");
const A0 = row("a0", "assistant", "REPLY-A0");
const SENT = row("s1", "steer", "STEER-S1");
const DELIVERED = row("s1", "user", "STEER-S1");
const A1 = row("a1", "assistant", "REPLY-A1");
const A2 = row("a2", "assistant", "REPLY-A2");
const A3 = row("a3", "assistant", "REPLY-A3");

/** The order the markers appear in the rendered DOM, each counted. */
function order(container: HTMLElement): string[] {
	const text = container.textContent ?? "";
	const markers = ["PROMPT-U1", "REPLY-A0", "STEER-S1", "REPLY-A1", "REPLY-A2", "REPLY-A3"];
	for (const marker of markers) {
		// Never twice: a moved row must repaint in place, not mount a second copy.
		expect(text.split(marker).length - 1, marker).toBeLessThanOrEqual(1);
	}
	return markers.filter((m) => text.includes(m)).sort((a, b) => text.indexOf(a) - text.indexOf(b));
}

describe("steer_journal_order on the relay web client", () => {
	it("HELD: a steer the relay moves at delivery keeps its slot, once, and later rows land after it", () => {
		const { container, rerender } = render(<Transcript pid="s" entries={[U1, A0, SENT]} />);
		rerender(<Transcript pid="s" entries={[U1, A0, SENT, A1]} />);
		rerender(<Transcript pid="s" entries={[U1, A0, SENT, A1, A2]} />);
		// Delivered: the relay moves s1 past a1/a2 and flips it to `user`.
		rerender(<Transcript pid="s" entries={[U1, A0, A1, A2, DELIVERED]} />);
		rerender(<Transcript pid="s" entries={[U1, A0, A1, A2, DELIVERED, A3]} />);
		expect(order(container)).toEqual([
			"PROMPT-U1", "REPLY-A0", "STEER-S1", "REPLY-A1", "REPLY-A2", "REPLY-A3",
		]);
	});

	it("FRESH MOUNT: a tab opened after delivery shows the journal order", () => {
		const { container } = render(<Transcript pid="s" entries={[U1, A0, A1, A2, DELIVERED, A3]} />);
		expect(order(container)).toEqual([
			"PROMPT-U1", "REPLY-A0", "REPLY-A1", "REPLY-A2", "STEER-S1", "REPLY-A3",
		]);
	});
});
