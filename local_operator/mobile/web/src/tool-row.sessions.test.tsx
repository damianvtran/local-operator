// @vitest-environment happy-dom
//
// The phone's sessions tool rows (PR C). The summary STRING is composed
// server-side — `harness/rows.py`'s `sessions_row_summary`, called from
// `mobile/projection._summarize_args` — and this pins the renderer's half of
// the contract: the op-leading summary reaches the row unchanged, two
// different ops on one target are two different rows, and the summary keeps
// the truncating lane that lets a narrow phone shed its tail instead of the
// name or the clock (the "the mode must survive" property the TUI's own
// `_send_summary` records, on the phone).
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
		tool_call_id: "call_sessions",
		tool_name: "sessions",
		tool_state: "done",
		summary: "",
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

describe("the phone's sessions tool row", () => {
	it("shows the op-leading summary the daemon composed", () => {
		render(
			<Transcript
				pid="1"
				entries={[entry({ summary: "spawn · workstream · night-audit" })]}
			/>,
		);
		expect(screen.getByText("spawn · workstream · night-audit")).toBeTruthy();
		expect(screen.getByText("sessions")).toBeTruthy();
	});

	it("keeps a stop and a peek on one target as two different rows", () => {
		/* The defect class the leading op closes: both used to render as the
		   same `op=…`-less target once the summary was the only text. */
		render(
			<Transcript
				pid="1"
				entries={[
					entry({ id: "e1", tool_call_id: "c1", summary: "stop · release-crew" }),
					entry({ id: "e2", tool_call_id: "c2", summary: "peek · release-crew · last 12" }),
				]}
			/>,
		);
		expect(screen.getByText("stop · release-crew")).toBeTruthy();
		expect(screen.getByText("peek · release-crew · last 12")).toBeTruthy();
	});

	it("renders the summary in the row's own truncating lane", () => {
		/* `min-w-0 flex-1 truncate`: on a 360px phone a long prompt must shed
		   from the summary's tail — the op and target lead — rather than push
		   the name or the clock out of the row. */
		render(
			<Transcript
				pid="1"
				entries={[entry({ summary: "spawn · ephemeral · " + "x".repeat(200) })]}
			/>,
		);
		const summary = screen.getByText(/spawn · ephemeral/);
		expect(summary.className).toContain("truncate");
		expect(summary.className).toContain("flex-1");
	});
});
