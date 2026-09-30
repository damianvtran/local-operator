// @vitest-environment happy-dom
//
// The pin-integrity badge on the phone's roster row (PR review round 1,
// reviewer MAJOR). The wire has carried `SubagentRow.model_label` as the
// `A → B ⚠ fallback` string since the projection was written — but no phone
// component painted it, so a child a fallback took off its pinned model read
// exactly like a never-pinned one, while the README/test-docstring claimed the
// substitution showed here. This file pins the render that closes the gap.
//
// What happy-dom proves is the STRUCTURE: the badge line exists for a child
// off its pin (and only for one), it wears the warning ink, and it is a
// wrapped line rather than a truncating one — the tail of the badge string IS
// the alarm (`⚠ fallback`), so an ellipsis that ate it would leave two model
// names with nothing saying they disagree. The pixel evidence is the frames
// recorded on the PR.
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { AgentRow } from "./subagents-panel";
import type { SubagentRow } from "../types";

afterEach(() => {
	cleanup();
});

const BADGE = "anthropic/claude-sonnet-5-5 → DeepSeek Flash ⚠ fallback";

function agent(over: Partial<SubagentRow> = {}): SubagentRow {
	return {
		job_id: "round1-designer",
		label: "round1-designer",
		agent: "designer",
		status: "running",
		progress: "",
		elapsed_s: 120,
		model_label: "deepseek/deepseek-flash",
		result_text: "",
		error_text: "",
		parent_job_id: null,
		session_id: "s1",
		prompt: "",
		launch_message_id: "",
		effort: "",
		ancestors: [],
		ancestor_ids: [],
		child_ids: [],
		peer_ids: [],
		transcript: [],
		todos: [],
		activity: "",
		...over,
	};
}

describe("the roster row for a child off its pin", () => {
	it("renders the badge line the wire carried and nobody painted", () => {
		const { container } = render(
			<AgentRow sessionId="s1" agent={agent({ model_fallback: true, model_label: BADGE })} />
		);
		const badge = screen.getByText(BADGE);
		expect(container.textContent).toContain(BADGE);
		// Both models are on the row, which is the claim the docstring makes.
		expect(container.textContent).toContain("anthropic/claude-sonnet-5-5");
		expect(container.textContent).toContain("DeepSeek Flash");
		// Warning ink, like the notice treatment — and `break-words`, NOT
		// `truncate`: the alarm lives at the tail of the string, so cutting it
		// with an ellipsis would silently drop exactly the words that matter.
		expect(badge.className).toContain("text-warning");
		expect(badge.className).toContain("break-words");
		expect(badge.className).not.toContain("truncate");
	});

	it("keeps the badge in the sheet's richer row too (showMetadata)", () => {
		render(
			<AgentRow
				sessionId="s1"
				agent={agent({ model_fallback: true, model_label: BADGE, effort: "hi" })}
				showMetadata
			/>
		);
		// The metadata line still carries `role · effort`, and the badge line
		// rides beside it rather than being folded away.
		expect(screen.getByText(BADGE)).toBeTruthy();
		expect(screen.getByText(/designer · hi/)).toBeTruthy();
	});

	it("leaves a child on its pin — and an older payload without the field — exactly as it was", () => {
		const onPin = render(
			<AgentRow sessionId="s1" agent={agent({ model_fallback: false })} />
		);
		expect(onPin.container.textContent).not.toContain("⚠");
		// A payload from a runtime that predates the field: absent must read as
		// `no substitution`, not as a crash or an invented marker.
		cleanup();
		const old = render(<AgentRow sessionId="s1" agent={agent()} />);
		expect(old.container.textContent).not.toContain("⚠");
		// And the effective selector is NOT painted as a line: only children
		// off their pin grow the extra row.
		expect(old.container.textContent).not.toContain("deepseek/deepseek-flash");
	});
});
