// @vitest-environment happy-dom
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AgentConversation } from "./screens/agent-view";
import type { SessionProjection, SubagentDetail, SubagentRow, TranscriptEntry } from "./types";

vi.mock("./api", () => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	imageUrl: vi.fn(() => ""),
}));

function entry(id: string, kind: TranscriptEntry["kind"], text: string): TranscriptEntry {
	return {
		id, kind, text, tool_call_id: "", tool_name: "", tool_state: "done",
		summary: "", intent: "", diff_added: 0, diff_removed: 0, elapsed_s: 0,
		error: "", details: {}, final: true,
	};
}

function row(jobId: string, parentJobId: string | null): SubagentRow {
	return {
		job_id: jobId, label: jobId, agent: "coder", status: "running", progress: "working",
		elapsed_s: 1, model_label: "", result_text: "", error_text: "",
		parent_job_id: parentJobId, session_id: `${jobId}-session`, prompt: "", effort: "high",
		ancestors: [], ancestor_ids: [], child_ids: [], peer_ids: [], transcript: [], todos: [], activity: "working",
	};
}

function fixture(): { detail: SubagentDetail; projection: SessionProjection } {
	const detail: SubagentDetail = {
		...row("current", "parent"), version: 3, label: "current-agent",
		ancestor_ids: ["ancestor", "parent"], peer_ids: ["peer"], child_ids: ["child"],
		prompt: "One request", transcript: [
			entry("parent", "parent_message", "One request"),
			entry("assistant", "assistant", "One response"),
		],
	};
	const projection = {
		session_id: "root", pid: 1, kind: "tui", conversation_name: "Root", cwd: "",
		model_label: "", model_selector: "", effort: "", effort_ladder: [], streaming: true,
		activity: "", activity_started_s: 0, stop_reason: "", queued_count: 0, ended: false,
		degraded: false, transcript: [], todos: [],
		subagents: [row("ancestor", null), row("parent", "ancestor"), detail, row("peer", "parent"), row("child", "current")],
		pending: null, pending_count: 0, usage: {}, version: 3,
	} satisfies SessionProjection;
	return { detail, projection };
}

afterEach(() => {
	vi.restoreAllMocks();
	document.body.innerHTML = "";
});

describe("AgentConversation", () => {
	it("opens one Agents sheet with path, peers, and children, then navigates a row", () => {
		const pushState = vi.spyOn(history, "pushState");
		const { detail, projection } = fixture();
		render(<AgentConversation sessionId="root" jobId="current" projection={projection} connected detail={detail} />);
		fireEvent.click(screen.getByRole("button", { name: "open agent navigation" }));
		expect(screen.getByRole("dialog")).toBeTruthy();
		expect(screen.getByText("Path")).toBeTruthy();
		expect(screen.getByText("Peers")).toBeTruthy();
		expect(screen.getByText("Children")).toBeTruthy();
		fireEvent.click(screen.getByRole("button", { name: /peer/ }));
		expect(pushState).toHaveBeenCalledWith(expect.anything(), "", "#/s/root/a/peer");
		expect(screen.queryByRole("dialog")).toBeNull();
	});

	it("keeps the delegated request singular and removes dominant legacy chrome", () => {
		const { detail, projection } = fixture();
		render(<AgentConversation sessionId="root" jobId="current" projection={projection} connected detail={detail} />);
		expect(screen.getAllByText("One request")).toHaveLength(1);
		expect(screen.queryByRole("navigation", { name: "agent lineage" })).toBeNull();
		expect(screen.queryByText("Parent request")).toBeNull();
		expect(screen.queryByText(/send commands from the root conversation/i)).toBeNull();
		expect(screen.getByRole("button", { name: "Open parent to steer" })).toBeTruthy();
	});
});
