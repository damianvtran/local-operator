// @vitest-environment happy-dom
//
// Team Bridge calls inside a condensed group. The count line splits the one
// `bridge` wire name by op, the way `jobs cancel` splits from `jobs` - a post
// (`channel_post`/`dm`/`send`/`reply`) is a `team message`, everything else a
// `team action` - so a directory lookup is never counted as something said.
// The op comes from the row's args (`bridgeOpOf`); args stripped under frame
// pressure fall to the generic action noun rather than inventing a post. And a
// bridge row is an ordinary foldable call: it must not break a group of its
// neighbours (`send` included) into pieces.
import { cleanup, fireEvent, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Transcript } from "./components/transcript";
import { foldCounts } from "./lib/condense-fold";
import type { TranscriptEntry } from "./types";

vi.mock("./api", () => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	imageUrl: vi.fn(() => ""),
}));

afterEach(cleanup);

const action = (name: string, op = "") => ({ name, failed: false, durationS: null, startedAtMs: null, endedAtMs: null, op });

function tool(id: string, name: string, extra: Partial<TranscriptEntry> = {}): TranscriptEntry {
	return {
		id, kind: "tool", text: "", tool_call_id: id, tool_name: name, tool_state: "done",
		summary: `${name} ${id}`, intent: "", diff_added: 0, diff_removed: 0, elapsed_s: 2,
		error: "", details: {}, final: true, text_complete: true, ...extra,
	};
}
const bridge = (id: string, op: string | null) =>
	tool(id, "bridge", { details: op === null ? {} : { args: { op, body: "hi" } } });

describe("bridge calls in the fold count line", () => {
	it("counts the posting ops as team messages", () => {
		expect(foldCounts(["channel_post", "dm", "send", "reply"].map((op) => action("bridge", op)))).toBe("4 team messages");
		expect(foldCounts([action("bridge", "dm")])).toBe("1 team message");
	});

	it("counts every other op as a team action", () => {
		expect(foldCounts([action("bridge", "directory"), action("bridge", "channel_history")])).toBe("2 team actions");
		expect(foldCounts([action("bridge", "ack")])).toBe("1 team action");
	});

	it("keeps an op-less bridge call off the message noun", () => {
		expect(foldCounts([action("bridge")])).toBe("1 team action");
	});

	it("splits a mixed run and leaves lop's own send as a message", () => {
		const line = foldCounts([
			action("bridge", "channel_post"),
			action("bridge", "directory"),
			action("send"),
		]);
		expect(line.split(" · ").sort()).toEqual(["1 message", "1 team action", "1 team message"]);
	});
});

describe("bridge rows inside a condensed tool group", () => {
	it("folds with its send neighbours as ONE group, read through the row's own op", () => {
		const view = render(
			<Transcript
				pid="s"
				entries={[
					bridge("b1", "channel_post"),
					tool("s1", "send"),
					bridge("b2", "dm"),
					bridge("b3", "directory"),
					bridge("b4", null),
				]}
				streaming
			/>,
		);
		const groups = view.container.querySelectorAll("[data-condense-fold]");
		expect(groups.length).toBe(1);
		const header = groups[0].textContent ?? "";
		expect(header).toContain("2 team messages");
		expect(header).toContain("2 team actions");
		expect(header).toContain("1 message");
		fireEvent.click(groups[0].querySelector("button")!);
		expect(view.container.querySelectorAll('svg[data-icon="messages-square"]').length).toBe(4);
	});
});
