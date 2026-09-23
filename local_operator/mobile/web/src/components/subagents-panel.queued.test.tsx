// @vitest-environment happy-dom
//
// A capacity-parked child is NOT running, and the roster header counts this
// field — so the two have to be pinned together (UX round 3).
//
// The fold used to map the runtime's `queued` onto the mobile `running` status
// (`mobile/projection.py`), which made this header print `2/2 running` for a
// parent with one spending child and one waiting for a slot — one tap after a
// list chip that had just been taught to keep the two apart. The fold now emits
// `queued`, and this file holds both halves of the header's arithmetic: the
// fraction counts only the running lane, and the waiting children are named in
// their own word rather than being folded into it.
//
// What happy-dom can prove is the STRUCTURE — which words and glyphs exist, and
// that the queued count is a `shrink-0` sibling so the truncating label yields
// instead of the number breaking (the mechanism `failed` already established).
// The pixel evidence is the phone tap frames recorded on the PR.
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { AGENT_GLYPH, AgentRoster, AgentRow } from "./subagents-panel";
import type { SubagentRow } from "../types";

afterEach(() => {
	cleanup();
});

function child(
	job_id: string,
	over: { status?: SubagentRow["status"]; parent_job_id?: string | null } = {}
): SubagentRow {
	return {
		job_id,
		label: job_id,
		agent: "task",
		status: over.status ?? "running",
		progress: "",
		elapsed_s: null,
		model_label: "fixture/model",
		result_text: "",
		error_text: "",
		parent_job_id: over.parent_job_id ?? null,
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
	};
}

describe("the roster header with a parked child", () => {
	it("counts only the running lane and names the waiting children", () => {
		render(
			<AgentRoster
				sessionId="s1"
				subagents={[
					child("running-child"),
					child("waiting-child", { status: "queued" }),
				]}
				parentJobId={null}
			/>
		);
		const header = document.body.textContent ?? "";
		expect(header).toContain("1/2 running");
		expect(header).toContain("1 queued");
		// The claim this change removes: a waiting child counted as spending.
		expect(header).not.toContain("2/2 running");
	});

	it("still says so when nothing is running", () => {
		render(
			<AgentRoster
				sessionId="s1"
				subagents={[
					child("w1", { status: "queued" }),
					child("w2", { status: "queued" }),
				]}
				parentJobId={null}
			/>
		);
		const header = document.body.textContent ?? "";
		expect(header).toContain("0/2 running");
		expect(header).toContain("2 queued");
	});

	it("leaves a healthy roster's header exactly as it was", () => {
		render(
			<AgentRoster
				sessionId="s1"
				subagents={[child("a"), child("b")]}
				parentJobId={null}
			/>
		);
		const header = document.body.textContent ?? "";
		expect(header).toContain("2/2 running");
		// No invented `0 queued`: the addend exists only when there is something
		// to say.
		expect(header).not.toContain("queued");
	});
});

describe("the roster row's treatment for a parked child", () => {
	it("renders the waiting glyph, never the spinner", () => {
		const { container } = render(
			<AgentRow sessionId="s1" agent={child("waiting", { status: "queued" })} />
		);
		expect(container.textContent).toContain(AGENT_GLYPH.queued);
		expect(container.textContent).not.toContain(AGENT_GLYPH.running);
		// And it is NOT on the animated accent lane the spinner owns.
		expect(container.innerHTML).not.toContain("lo-pulse");
	});

	it("keeps the running glyph and its pulse for a child that is spending", () => {
		const { container } = render(
			<AgentRow sessionId="s1" agent={child("busy")} />
		);
		expect(container.textContent).toContain(AGENT_GLYPH.running);
		expect(container.innerHTML).toContain("lo-pulse");
	});
});

describe("the roster's own vocabulary", () => {
	it("gives the parked child a glyph of its own", () => {
		// Three states in one gutter, three distinct glyphs: the ellipsis reads as
		// waiting in every font, where a pause bar or an ellipsis variant can
		// render as tofu on some phones.
		expect(AGENT_GLYPH.queued).toBe("…");
		expect(new Set(Object.values(AGENT_GLYPH)).size).toBe(
			Object.keys(AGENT_GLYPH).length
		);
	});

	it("does not render a roster with no direct children at all", () => {
		render(
			<AgentRoster
				sessionId="s1"
				subagents={[child("nested", { parent_job_id: "other" })]}
				parentJobId={null}
			/>
		);
		expect(screen.queryByText(/running/)).toBeNull();
	});
});
