// @vitest-environment happy-dom
//
// The RENDERED half of the history-fold convergence. The python fold now
// emits rows the phone never used to receive (refusals, failed turns,
// unattended gate timeouts, refused compactions) plus two new `details`
// fields — `severity` on a notice and `user_run` on a tool row. A fold that
// emits a field no renderer reads is a fix that does not reach the user, so
// these tests render the REAL Transcript component over the REAL fold's
// output shape and assert what a person would actually see.
//
// The fixtures below are the verbatim JSON the python fold produced for one
// session containing every restored row (see the PR's before/after evidence);
// keeping them as literal rows rather than hand-built objects is what makes
// this a test of the contract between the two halves rather than of a mock.
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
		kind: "notice",
		text: "",
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
		...over,
	};
}

describe("rows the phone used to drop entirely", () => {
	it("shows a refusal, a failed turn and a gate timeout", () => {
		render(
			<Transcript
				pid="1"
				entries={[
					entry({ id: "a", kind: "assistant", text: "I started to answer but" }),
					entry({ id: "b", text: "content policy", details: { severity: "error" } }),
					entry({ id: "c", text: "turn failed", details: { severity: "error" } }),
					entry({
						id: "d",
						text: "waited 2h for approval with nobody attached, then denied it — bash · rm -rf /x",
						details: { severity: "warning" },
					}),
				]}
			/>,
		);

		// Each was invisible on the phone before this change.
		expect(screen.getByText("content policy")).toBeTruthy();
		expect(screen.getByText("turn failed")).toBeTruthy();
		expect(screen.getByText(/waited 2h for approval/)).toBeTruthy();
	});

	it("tints a notice by severity rather than flattening every row to grey", () => {
		// The severity distinction is the POINT of D7: a compaction that was
		// declined and one that FAILED are different things to the person
		// deciding what to do next, and one ink for both loses that.
		render(
			<Transcript
				pid="1"
				entries={[
					entry({ id: "a", text: "compaction failed", details: { severity: "error" } }),
					entry({ id: "b", text: "compaction skipped", details: { severity: "warning" } }),
					entry({ id: "c", text: "an ordinary receipt" }),
				]}
			/>,
		);

		/* The tint sits on the notice ROW, not on the text node: the row is
		   `<p class="…text-danger"><span aria-hidden>✗</span><span>text</span></p>`,
		   so the severity class lives on the text span's parent. */
		const row = (text: string) => screen.getByText(text).parentElement;

		expect(row("compaction failed")?.className).toContain("text-danger");
		expect(row("compaction skipped")?.className).toContain("text-warning");
		// Negative control: an unmarked notice keeps the quiet default rather
		// than acquiring the loudest ink in the palette.
		expect(row("an ordinary receipt")?.className).toContain("text-ink-dim");
	});

	it("fronts a notice with a glyph so severity survives losing colour", () => {
		/* Design D2: severity was HUE-ONLY here. danger-vs-ink-dim measures
		   1.30:1, and across all 31 themes those two sit within 1.6:1 on 30 of
		   them (1.00:1 on six) — so in grayscale a failure and a routine
		   receipt were one indistinguishable grey, while the TUI survived
		   desaturation because it fronts every notice with ✗/!/·. */
		render(
			<Transcript
				pid="1"
				entries={[
					entry({ id: "a", text: "compaction failed", details: { severity: "error" } }),
					entry({ id: "b", text: "compaction skipped", details: { severity: "warning" } }),
					entry({ id: "c", text: "an ordinary receipt" }),
				]}
			/>,
		);

		const glyphOf = (text: string) =>
			screen.getByText(text).parentElement?.textContent?.trimStart()[0];

		// The TUI's NOTICE_GLYPHS grammar exactly (transcript.py).
		expect(glyphOf("compaction failed")).toBe("✗");
		expect(glyphOf("compaction skipped")).toBe("!");
		expect(glyphOf("an ordinary receipt")).toBe("·");
	});

	it("gives a wake receipt its own affordance instead of an anonymous line", () => {
		/* Design D3 / review MINOR-1: `notice_kind: "wake"` was emitted and
		   typed but read by NO component, so the row rendered exactly as it
		   did before the field existed — a typed-but-unread field is worse
		   than none. A wake is a receipt the user scheduled, so it takes a
		   neutral clock rather than alarm ink. */
		render(
			<Transcript
				pid="1"
				entries={[
					entry({
						id: "a",
						text: "w-9 (1, every 6h) — Check the deploy pipeline",
						details: { notice_kind: "wake" },
					}),
				]}
			/>,
		);

		const row = screen.getByText("w-9 (1, every 6h) — Check the deploy pipeline");
		expect(row.parentElement?.textContent?.trimStart()[0]).toBe("○");
		// Not dressed as a failure: the user asked for this to fire.
		expect(row.parentElement?.className).not.toContain("text-danger");
	});
});

describe("a hub steer never leaks its envelope", () => {
	it("renders the parent's own words as a parent_message card", () => {
		render(
			<Transcript
				pid="1"
				entries={[entry({ id: "a", kind: "parent_message", text: "focus on the parser" })]}
			/>,
		);

		expect(screen.getByText("focus on the parser")).toBeTruthy();
		expect(document.body.textContent).not.toContain("<parent-message>");
	});
});

describe("tool rows", () => {
	it("shows an unanswered call as interrupted, not as succeeded", () => {
		// D10: `–` is the interrupted glyph, `✓` the success one. A call that
		// never returned wearing a ✓ is the phone asserting an outcome nobody
		// observed.
		render(
			<Transcript
				pid="1"
				entries={[
					entry({
						id: "a",
						kind: "tool",
						tool_name: "read",
						tool_call_id: "t9",
						tool_state: "interrupted",
						summary: "/var/log/deploy.log",
					}),
				]}
			/>,
		);

		expect(screen.getByText("–")).toBeTruthy();
		expect(screen.queryByText("✓")).toBeNull();
	});

	it("opens a bang-mode card expanded and leaves a model call collapsed", () => {
		// D11: the user typed `! ls -la` and is waiting to read its output.
		const rows = [
			entry({
				id: "a",
				kind: "tool",
				tool_name: "bash",
				tool_call_id: "sh1",
				summary: "ls -la",
				details: { user_run: true, output: "total 0\ndrwxr-xr-x" },
			}),
			entry({
				id: "b",
				kind: "tool",
				tool_name: "bash",
				tool_call_id: "m1",
				summary: "make deploy",
				details: { output: "MODEL RAN THIS" },
			}),
		];
		render(<Transcript pid="1" entries={rows} />);

		// The user's own command shows its output without a tap...
		expect(screen.getByText(/drwxr-xr-x/)).toBeTruthy();
		// ...while the model's call stays one line until asked.
		expect(screen.queryByText(/MODEL RAN THIS/)).toBeNull();
	});
});
