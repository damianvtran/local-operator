// @vitest-environment happy-dom
//
// The phone's render of the third live tool state: a call that has been
// announced, whose dictation is over, and which nothing has started.
//
// The projection sets `queued`; this asserts the RENDERER does something
// honest with it, because a fold that emits a state no view reads is a fix
// that does not reach the user. Two things a person would notice:
//
//   * the glyph must not claim work that is not happening — a spinner for a
//     call waiting behind a sibling is the phone's half of the operator's
//     "stuck wake compose" report;
//   * the row stays LIVE-looking (raised background, dim state glyph) without
//     pulsing, which is the styling that says "queued" rather than "running"
//     on a screen with no room for a status column.
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Transcript } from "./components/transcript";
import type { TranscriptEntry } from "./types";

vi.mock("./api", () => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	imageUrl: vi.fn(() => ""),
}));

afterEach(cleanup);

function entry(over: Partial<TranscriptEntry>): TranscriptEntry {
	return {
		id: "e1",
		kind: "tool",
		text: "",
		tool_call_id: "call_wake",
		tool_name: "wake",
		tool_state: "queued",
		summary: "waiting to run wake",
		intent: "",
		diff_added: 0,
		diff_removed: 0,
		elapsed_s: 0,
		error: "",
		details: {},
		final: false,
		...over,
	};
}

function rowFor(state: TranscriptEntry["tool_state"]) {
	render(<Transcript pid="1" entries={[entry({ tool_state: state })]} />);
	return screen.getByText("wake").closest("button")?.parentElement ?? null;
}

describe("the phone's queued tool row", () => {
	it("draws the queued glyph rather than a spinner", () => {
		const row = rowFor("queued");
		expect(row?.textContent).toContain("⋯");
		expect(row?.textContent).not.toContain("⟳");
	});

	it("stays visibly pending but does not pulse", () => {
		const row = rowFor("queued");
		/* THE GROUND IS FLAT NOW (relay redesign §7.1/§7.5): every tool row sits
		   on the transcript's own ground, so the queued state cannot be carried by
		   a raised box. It is carried by the dim pending glyph with no pulse — the
		   pulse is the "work is happening" signal and nothing is happening while
		   the call waits its turn — which is exactly what distinguishes it from a
		   running row (asserted by the next test). */
		expect(row?.className ?? "").not.toContain("bg-elevated");
		expect(row?.textContent).toContain("⋯");
		const glyph = row?.querySelector("span");
		expect(glyph?.className ?? "").toContain("text-ink-dim");
		expect(glyph?.className ?? "").not.toContain("lo-pulse");
	});

	it("keeps the running row pulsing, so the two are distinguishable", () => {
		const row = rowFor("running");
		const glyph = row?.querySelector("span");
		expect(glyph?.className ?? "").toContain("lo-pulse");
	});

	it("shows the never-run reason on the row itself", () => {
		render(
			<Transcript
				pid="1"
				entries={[
					entry({
						tool_state: "failed",
						summary: "Tool not found: wake",
						error: "Tool not found: wake",
					}),
				]}
			/>,
		);
		expect(screen.getByText("Tool not found: wake")).toBeTruthy();
	});
});
