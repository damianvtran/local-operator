// @vitest-environment happy-dom
//
// The transcript's ask rows (`ask_response` / `ask_timeout`, design §4/§5).
// One line at rest, carrying the SHARED copy the runtime wrote (`harness/rows.py`)
// — the phone must not paraphrase it — and a disclosure that shows the
// structured Q&A, with a secret answer rendered as the key the runtime stored
// and never a value.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { AskRow } from "./components/ask-row";
import type { TranscriptEntry } from "./types";

function entry(patch: Partial<TranscriptEntry> = {}): TranscriptEntry {
	return {
		id: "e1",
		kind: "ask_response",
		text: "Answered — delivering (ask ask-1)",
		tool_call_id: "",
		tool_name: "",
		tool_state: "done",
		summary: "",
		intent: "",
		diff_added: 0,
		diff_removed: 0,
		elapsed_s: 0,
		error: "",
		details: {
			ask_id: "ask-1",
			status: "answered",
			questions: [
				{
					id: "q1",
					question: "ship the fix?",
					options: [{ label: "yes", description: "merge it today" }],
					multi: false,
					secret: false,
					persist: false,
				},
			],
			answers: { q1: ["yes"] },
		},
		final: true,
		...patch,
	} as TranscriptEntry;
}

afterEach(() => cleanup());

describe("AskRow", () => {
	it("rests as the runtime's own one-liner and expands to the Q&A", () => {
		render(<AskRow entry={entry()} />);
		expect(screen.getByText("Answered — delivering (ask ask-1)")).toBeTruthy();
		expect(screen.queryByText("ship the fix?")).toBeNull();
		fireEvent.click(screen.getByRole("button"));
		expect(screen.getByText("ship the fix?")).toBeTruthy();
		expect(screen.getByText("yes — merge it today")).toBeTruthy();
	});

	it("renders a late answer as late, in warning ink", () => {
		render(
			<AskRow
				entry={entry({ text: "Answered late — the agent was told (ask ask-1)", details: { ...entry().details, status: "late", severity: "warning" } })}
			/>,
		);
		expect(screen.getByTestId("ask-row").getAttribute("data-ask-status")).toBe("late");
	});

	it("shows a secret answer as the stored KEY, never a value", () => {
		render(
			<AskRow
				entry={entry({
					details: {
						ask_id: "ask-1",
						status: "answered",
						questions: [
							{
								id: "q1",
								question: "the token?",
								options: [],
								multi: false,
								secret: true,
								persist: false,
							},
						],
						answers: { q1: ["LINEAR_TOKEN"] },
					},
				})}
			/>,
		);
		fireEvent.click(screen.getByRole("button"));
		expect(screen.getByText("LINEAR_TOKEN")).toBeTruthy();
	});

	it("labels a timeout's expansion as what the AGENT was told", () => {
		render(
			<AskRow
				entry={entry({
					kind: "ask_timeout",
					text: "Timed out after 42m — the agent moved on; you can still answer (ask ask-1)",
					details: {
						ask_id: "ask-1",
						status: "timed_out",
						severity: "warning",
						waited_s: 2520,
						urgent: false,
						text: "Proceed without it: use your recommended option.",
					},
				})}
			/>,
		);
		fireEvent.click(screen.getByRole("button"));
		expect(screen.getByText("what the agent was told")).toBeTruthy();
		expect(screen.getByText(/Proceed without it/)).toBeTruthy();
	});
});
