// The ask-id chip's rules, exercised without a rendered tree: the token
// grammar (a citation, not a name that happens to end in hex) and the
// precedence between the live frame, the transcript receipts and the sheet's
// last aggregate read. The component's own behaviour is in ask-id.test.tsx.
import { describe, expect, it } from "vitest";
import {
	askChipFacts,
	askChipOpen,
	askChipWord,
	askIdSegments,
	isAskId,
} from "./lib/ask-chip";
import type { PendingAsk, TranscriptEntry } from "./types";

function ask(patch: Partial<PendingAsk> = {}): PendingAsk {
	return {
		ask_id: "a-745e",
		created_at: 1,
		expires_at: 2,
		timeout_s: 900,
		urgent: false,
		status: "open",
		delivered: false,
		questions: [{ id: "q1", question: "Which region?", options: [], multi: false, secret: false, persist: false }],
		...patch,
	};
}

function receipt(patch: { ask_id?: string; status?: string; kind?: string } = {}): TranscriptEntry {
	return {
		id: `row-${patch.ask_id ?? "a-745e"}`,
		kind: (patch.kind ?? "ask_response") as TranscriptEntry["kind"],
		text: "",
		details: {
			ask_id: patch.ask_id ?? "a-745e",
			status: (patch.status ?? "answered") as "answered",
			questions: [{ question: "Which region?" }],
		},
	} as unknown as TranscriptEntry;
}

describe("askIdSegments", () => {
	it("finds a bare id and keeps the sentence around it", () => {
		expect(askIdSegments("the old one, a-745e, is withdrawn")).toEqual([
			{ kind: "text", text: "the old one, " },
			{ kind: "ask", id: "a-745e" },
			{ kind: "text", text: ", is withdrawn" },
		]);
	});

	it("splits several ids in one run", () => {
		expect(askIdSegments("a-745e and a-e53b")).toEqual([
			{ kind: "ask", id: "a-745e" },
			{ kind: "text", text: " and " },
			{ kind: "ask", id: "a-e53b" },
		]);
	});

	it("does not match the tail of a longer name", () => {
		for (const text of ["a-745e6", "xa-745e", "foo-a-745e", "a-745ef", "logs/a-12ab.txt", "https://x/a-745e", "x.a-12ab", "a-12ab.json"]) {
			expect(askIdSegments(text)).toEqual([{ kind: "text", text }]);
		}
	});

	it("still ends a sentence on a trailing dot", () => {
		expect(askIdSegments("withdrawn in a-745e.")).toEqual([
			{ kind: "text", text: "withdrawn in " },
			{ kind: "ask", id: "a-745e" },
			{ kind: "text", text: "." },
		]);
	});

	it("requires lower-case hex (the runtime's id shape)", () => {
		expect(askIdSegments("a-745E")).toEqual([{ kind: "text", text: "a-745E" }]);
	});

	it("treats a code span holding only an id as an id", () => {
		expect(isAskId("a-745e")).toBe(true);
		expect(isAskId("a-745e ")).toBe(false);
		expect(isAskId("run a-745e")).toBe(false);
	});
});

describe("askChipWord", () => {
	it("names every state the queue publishes", () => {
		const states = ["open", "answered", "declined", "timed_out", "late", "dismissed", "withdrawn", "expired"];
		expect(states.map((state) => {
			const word = askChipWord(state);
			return word?.kind === "key" ? word.key : null;
		})).toEqual(states.map((state) => `relay.asks.chip.${state}`));
	});

	it("falls back to the runtime's own spelling for a state it invents", () => {
		expect(askChipWord("escalated")).toEqual({ kind: "literal", text: "escalated" });
		expect(askChipWord("")).toBeNull();
	});

	it("does not read a prototype member as a state", () => {
		expect(askChipWord("constructor")).toEqual({ kind: "literal", text: "constructor" });
	});

	it("draws only an open ask in the accent", () => {
		expect(askChipOpen("open")).toBe(true);
		expect(askChipOpen("timed_out")).toBe(false);
	});
});

describe("askChipFacts", () => {
	it("prefers the live frame over a transcript receipt and the aggregate", () => {
		const facts = askChipFacts(
			{
				frame: [ask({ status: "open" })],
				transcript: [receipt({ status: "answered" })],
				settled: [ask({ session_id: "mine", status: "declined" })],
			},
			"mine",
		);
		expect(facts.get("a-745e")).toEqual({ status: "open", question: "Which region?" });
	});

	it("knows a settled ask from the receipt after it left the frame", () => {
		const facts = askChipFacts({ frame: [], transcript: [receipt({ status: "timed_out", kind: "ask_timeout" })] }, "mine");
		expect(facts.get("a-745e")?.status).toBe("timed_out");
	});

	it("names a withdrawn ask from the sheet's aggregate — the one state no receipt carries", () => {
		const facts = askChipFacts(
			{ frame: [], transcript: [], settled: [ask({ session_id: "mine", status: "withdrawn" })] },
			"mine",
		);
		expect(facts.get("a-745e")?.status).toBe("withdrawn");
	});

	it("never reads another conversation's row: ids are unique per session only", () => {
		const facts = askChipFacts(
			{ frame: [], settled: [ask({ session_id: "other", status: "withdrawn" })] },
			"mine",
		);
		expect(facts.get("a-745e")).toBeUndefined();
	});

	it("does not let a stale open aggregate row outrank a live frame that dropped the ask", () => {
		const facts = askChipFacts(
			{ frame: [], settled: [ask({ session_id: "mine", status: "open" })] },
			"mine",
		);
		expect(facts.get("a-745e")).toBeUndefined();
	});

	it("treats an absent frame as no live reading, so the aggregate may still speak", () => {
		const facts = askChipFacts({ settled: [ask({ session_id: "mine", status: "open" })] }, "mine");
		expect(facts.get("a-745e")?.status).toBe("open");
	});
});
