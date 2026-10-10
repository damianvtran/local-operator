// @vitest-environment happy-dom
//
// The relay web's quiet-group bar (local-operator quiet-turn design §5 + §8
// S4; `lib/quiet-groups` holds the definition and the shared parity fixture).
//
// WHAT THESE CELLS PIN, and why each has to be able to fail:
//   - consecutive receipts paint as ONE collapsed line and their cards are NOT
//     in the document while collapsed — the whole point of the fold is that a
//     dozen peer messages stop being a dozen cards on the phone;
//   - the press lists the receipts *with their actions* in place, exactly as
//     they paint at top level — an expansion that dropped the tool rows would
//     hide the work the receipts triggered;
//   - ONE receipt keeps its ordinary card — a single-receipt bar would be a
//     line of chrome over the same card;
//   - the open tail's count grows while the expansion survives the append —
//     the appended receipt lands inside the open group (its `qg:` key never
//     moves), not as a new card below a collapsed bar;
//   - the count is a MINIMUM (`N+`) while a page fetch has not proved there is
//     nothing older, and resolves to the exact count when the fetch answers
//     "no more" — the head-cut rule, on the same signal as the app's own
//     more-history hairline.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Transcript } from "./transcript";
import * as api from "../api";
import type { TranscriptEntry } from "../types";

vi.mock("../api", async (importOriginal) => ({
	...(await importOriginal<typeof api>()),
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
}));

afterEach(cleanup);

function entry(
	id: string,
	kind: TranscriptEntry["kind"],
	text: string,
	extra: Partial<TranscriptEntry> = {},
): TranscriptEntry {
	return {
		id,
		kind,
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
		...extra,
	};
}

const peer = (id: string, body: string) => entry(id, "peer_message", body);
const bash = (id: string) => entry(id, "tool", "", { tool_name: "bash", summary: "echo hi" });

describe("the quiet-group bar", () => {
	it("folds consecutive receipts into one bar, and the press lists them with their actions", () => {
		render(
			<Transcript pid="s1" entries={[peer("p1", "first"), bash("t1"), peer("p2", "second")]} />,
		);
		expect(screen.getByText("Peer messages")).toBeTruthy();
		/* The minimum: no page fetch has proved there is nothing older yet. */
		expect(screen.getByText("2+")).toBeTruthy();
		/* Collapsed: neither the receipt cards nor the work between them is
		   mounted — their absence is what the fold is for. */
		expect(screen.queryByText("first")).toBeNull();
		expect(screen.queryByText("second")).toBeNull();
		expect(screen.queryByText("echo hi")).toBeNull();

		fireEvent.click(screen.getByText("Peer messages").closest("button")!);
		expect(screen.getByText("first")).toBeTruthy();
		expect(screen.getByText("second")).toBeTruthy();
		expect(screen.getByText("echo hi")).toBeTruthy();
	});

	it("leaves a single receipt as its ordinary card", () => {
		render(<Transcript pid="s1" entries={[peer("p1", "only one")]} />);
		expect(screen.queryByText("Peer messages")).toBeNull();
		expect(screen.getByText("only one")).toBeTruthy();
	});

	it("grows the open tail's count in place and keeps the expansion across the append", () => {
		const first = [peer("p1", "first"), peer("p2", "second")];
		const view = render(<Transcript pid="s1" entries={first} />);
		fireEvent.click(screen.getByText("Peer messages").closest("button")!);
		expect(screen.getByText("first")).toBeTruthy();

		view.rerender(<Transcript pid="s1" entries={[...first, peer("p3", "third")]} />);
		/* The same group — the key never moved — one count higher, still open. */
		expect(screen.getByText("3+")).toBeTruthy();
		expect(screen.getByText("first")).toBeTruthy();
		expect(screen.getByText("third")).toBeTruthy();
	});

	it("states the count as a minimum until a page proves there is nothing older", async () => {
		const view = render(
			<Transcript pid="s1" entries={[peer("p1", "first"), peer("p2", "second")]} />,
		);
		expect(screen.getByText("2+")).toBeTruthy();

		/* Scrolling to the top asks for the older page; the stub answers "no
		   more", which is the signal the app's own hairline reads. The minimum
		   resolves to the exact count. */
		fireEvent.scroll(view.container.querySelector(".lo-scroll")!);
		expect(await screen.findByText("2")).toBeTruthy();
		expect(screen.queryByText("2+")).toBeNull();
	});
});
