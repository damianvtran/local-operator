// @vitest-environment happy-dom
//
// The phone's Team Bridge tool row. The FACTS (`details.bridge_row`, from
// `harness/rows.py::bridge_row`) and the WORDS (`details.bridge_copy`, rendered
// by the daemon from `wire.tool_rows.bridge`) are both server-side; this pins
// the renderer's half: the bridge glyph, an inbox-only send reading as a caveat
// with its MessageSquareOff mark (never success), a refusal in danger ink, the
// expansion's headline / markdown body / outcome sentence, and no opaque
// `bcv_`/`ep_` id ever reaching the headline.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Transcript } from "./components/transcript";
import { ToolRow } from "./components/tool-row";
import type { BridgeRowCopy, BridgeRowFacts, TranscriptEntry } from "./types";

vi.mock("./api", () => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	imageUrl: vi.fn(() => ""),
}));

afterEach(cleanup);

/* The daemon's rendered words for each outcome, as `bridge_copy.py` emits them. */
const WORD: Record<string, string> = {
	inbox_only: "not posted to a conversation",
	refused: "refused",
	failed: "failed",
	unconfirmed: "delivery unconfirmed",
	awaiting_acceptance: "awaiting acceptance",
};

function copyFor(facts?: BridgeRowFacts, over: BridgeRowCopy = {}): BridgeRowCopy | undefined {
	if (!facts) return undefined;
	const word = facts.outcome ? WORD[facts.outcome] : "";
	return {
		outcome: word && facts.code ? `${word} · ${facts.code}` : word,
		headline: "",
		outcome_sentence: "",
		...over,
	};
}

function entry(
	over: Partial<TranscriptEntry>,
	facts?: BridgeRowFacts,
	args?: Record<string, unknown>,
	copy?: BridgeRowCopy,
): TranscriptEntry {
	return {
		id: "e1",
		kind: "tool",
		text: "",
		tool_call_id: "call_bridge",
		tool_name: "bridge",
		tool_state: "done",
		summary: "post · #release",
		intent: "",
		diff_added: 0,
		diff_removed: 0,
		elapsed_s: 0,
		error: "",
		details: {
			...(facts ? { bridge_row: facts, bridge_copy: copyFor(facts, copy) } : {}),
			...(args ? { args } : {}),
		},
		final: false,
		...over,
	};
}

const POST: BridgeRowFacts = {
	op: "channel_post",
	action: "post",
	target: { kind: "channel", label: "#release" },
	preview: "Shipping 0.93",
	count: null,
	emoji: null,
	outcome: null,
	code: null,
};

describe("the phone's bridge tool row", () => {
	it("leads with the messages-square glyph and the server summary", () => {
		const { container } = render(<ToolRow entry={entry({}, POST)} />);
		expect(container.querySelector('svg[data-icon="messages-square"]')).toBeTruthy();
		expect(screen.getByText("post · #release")).toBeTruthy();
		/* An ordinary landed post carries no outcome chip: ✓ already says it. */
		expect(container.querySelector("[data-bridge-outcome]")).toBeNull();
	});

	it("never paints the glyph on another tool", () => {
		const { container } = render(<ToolRow entry={entry({ tool_name: "sessions" })} />);
		expect(container.querySelector('svg[data-icon="messages-square"]')).toBeNull();
	});

	it("states an inbox-only send as a caveat in warning ink, not success", () => {
		const { container } = render(
			<ToolRow entry={entry({ summary: "send · Priya" }, { ...POST, op: "send", target: { kind: "person", label: "Priya" }, outcome: "inbox_only" })} />,
		);
		const chip = container.querySelector('[data-bridge-outcome="inbox_only"]')!;
		expect(chip.textContent).toBe("not posted to a conversation");
		expect(chip.querySelector('svg[data-icon="message-square-off"]')).toBeTruthy();
		expect(chip.className).toContain("text-warning");
		expect(chip.className).not.toContain("text-success");
	});

	it("drops the summary's outcome tail when the chip states it", () => {
		/* The daemon's one-line summary ends with ` · <outcome>` for capless
		   cells; with a chip beside it the caveat would read twice. */
		render(
			<ToolRow
				entry={entry(
					{ summary: "Sent to a teammate · ping · not posted to a conversation" },
					{ ...POST, op: "send", target: { kind: "person", label: null as unknown as string }, outcome: "inbox_only" },
				)}
			/>,
		);
		expect(screen.getByText("Sent to a teammate · ping")).toBeTruthy();
		expect(screen.getAllByText(/not posted to a conversation/).length).toBe(1);
	});

	it("names a refusal with its typed code in danger ink", () => {
		const { container } = render(
			<ToolRow entry={entry({ tool_state: "failed" }, { ...POST, outcome: "refused", code: "not_member" })} />,
		);
		const chip = container.querySelector('[data-bridge-outcome="refused"]')!;
		expect(chip.textContent).toBe("refused · not_member");
		expect(chip.className).toContain("text-danger");
	});

	it("keeps unconfirmed and awaiting-acceptance as quiet words", () => {
		const { container } = render(
			<>
				<ToolRow entry={entry({ id: "a" }, { ...POST, outcome: "unconfirmed" })} />
				<ToolRow entry={entry({ id: "b" }, { ...POST, op: "dm", outcome: "awaiting_acceptance" })} />
			</>,
		);
		const quiet = [...container.querySelectorAll("[data-bridge-outcome]")];
		expect(quiet.map((n) => n.textContent)).toEqual(["delivery unconfirmed", "awaiting acceptance"]);
		for (const n of quiet) expect(n.className).toContain("text-ink-dim");
		expect(quiet[1].querySelector('svg[data-icon="hourglass"]')).toBeTruthy();
	});

	it("expands to the daemon's headline, the markdown body and the outcome sentence", () => {
		const { container } = render(
			<ToolRow
				entry={entry(
					{ summary: "Sent a DM to Priya Shah · Heads up" },
					{ ...POST, op: "dm", target: { kind: "dm", label: "Priya Shah" }, outcome: "awaiting_acceptance" },
					{ op: "dm", endpoint_id: "ep_01J9ZZ", conversation_id: "bcv_01J9ABCDEF", body: "**Heads up** — the `0.93` cut is *tonight*.", silent: true },
					{
						headline: "Direct message to Priya Shah",
						outcome_sentence: "Posted, but the recipient agent is invited — its owner must accept before it is woken.",
					},
				)}
			/>,
		);
		fireEvent.click(screen.getByRole("button"));
		expect(container.querySelector("[data-bridge-headline]")?.textContent).toBe("Direct message to Priya Shah");
		/* Markdown, not the raw asterisks. */
		expect(container.querySelector("strong")?.textContent).toBe("Heads up");
		expect(container.querySelector("code")?.textContent).toBe("0.93");
		const sentence = container.querySelector("[data-bridge-outcome-sentence]")!;
		expect(sentence.textContent).toMatch(/^Posted, but the recipient agent is invited/);
		expect(sentence.className).toContain("text-ink-dim");
		/* The generic args block keeps only what the expansion did not say. */
		const text = container.textContent ?? "";
		expect(text).toContain("silent");
		expect(text).not.toContain("bcv_01J9ABCDEF");
		expect(text).not.toContain("ep_01J9ZZ");
		expect(text).not.toContain("**Heads up**");
	});

	it("paints no chip when the daemon sent no words (older daemon, shed details)", () => {
		const facts = { ...POST, outcome: "inbox_only" as const };
		const e = entry({}, facts);
		delete e.details.bridge_copy;
		const { container } = render(<ToolRow entry={e} />);
		/* No English guessed on the phone: the server owns the sentence. */
		expect(container.querySelector("[data-bridge-outcome]")).toBeNull();
	});

	it("never surfaces a bridge id in the headline", () => {
		const { container } = render(
			<ToolRow
				entry={entry(
					{ summary: "post · bcv_01J9ABCDEF · bcmsg_77 acc_9 ep_1" },
					{ ...POST, target: { kind: "channel", label: "bcv_01J9ABCDEF" }, outcome: "refused", code: "acc_123" },
					undefined,
					{ outcome: "refused · acc_123" },
				)}
			/>,
		);
		const headline = container.querySelector("button")!.textContent ?? "";
		expect(headline).not.toMatch(/\b(?:bcv|bcmsg|acc|ep)_/);
	});

	it("keeps the summary in the truncating lane inside the transcript", () => {
		render(<Transcript pid="1" entries={[entry({ summary: "post · #release · " + "x".repeat(200) }, POST)]} />);
		const summary = screen.getByText(/post · #release ·/);
		expect(summary.className).toContain("truncate");
		expect(summary.className).toContain("min-w-0");
	});
});
