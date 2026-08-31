import { describe, expect, it } from "vitest";
import { groupAgents } from "./components/agents-sheet";
import { agentConversationEntries } from "./screens/agent-view";
import type { SubagentDetail, SubagentRow, TranscriptEntry } from "./types";

function entry(id: string, kind: TranscriptEntry["kind"], text: string): TranscriptEntry {
	return {
		id, kind, text, tool_call_id: "", tool_name: "", tool_state: "done",
		summary: "", intent: "", diff_added: 0, diff_removed: 0, elapsed_s: 0,
		error: "", details: {}, final: true,
	};
}

function row(jobId: string, parentJobId: string | null): SubagentRow {
	return {
		job_id: jobId, label: jobId, agent: "coder", status: "running", progress: "",
		elapsed_s: 1, model_label: "", result_text: "", error_text: "",
		parent_job_id: parentJobId, session_id: `${jobId}-session`, prompt: "", launch_message_id: "", effort: "high",
		ancestors: [], ancestor_ids: [], child_ids: [], peer_ids: [], transcript: [], todos: [], activity: "thinking",
	};
}

function detail(overrides: Partial<SubagentDetail> = {}): SubagentDetail {
	return {
		...row("current", "parent"), version: 1, ancestor_ids: ["ancestor", "parent"],
		peer_ids: ["peer"], child_ids: ["child"], ...overrides,
	};
}

describe("agent conversation composition", () => {
	it("uses durable launch identity while preserving later steering messages", () => {
		const launch = entry("subagent-launch:current", "user", "Role preamble\n\nImplement the route");
		const steer = entry("steer", "user", "Implement the route more narrowly");
		const messages = agentConversationEntries(detail({
			prompt: "Implement the route",
			launch_message_id: launch.id,
			transcript: [launch, entry("reply", "assistant", "Working on it"), steer],
		}));
		expect(messages).toEqual([
			expect.objectContaining({ id: launch.id, kind: "parent_message" }),
			expect.objectContaining({ id: "reply", kind: "assistant" }),
			steer,
		]);
		expect(messages.filter((message) => message.kind === "parent_message")).toHaveLength(1);
	});

	it("does not duplicate a delegated request already carried by parent_message", () => {
		const parent = entry("parent-message", "parent_message", "Implement the route");
		const messages = agentConversationEntries(detail({
			prompt: "Implement the route",
			transcript: [parent, entry("reply", "assistant", "Working on it")],
		}));
		expect(messages).toEqual([parent, expect.objectContaining({ id: "reply" })]);
		expect(messages.filter((message) => message.kind === "parent_message")).toHaveLength(1);
	});

	it("retains legacy prompt context as one transcript row", () => {
		const messages = agentConversationEntries(detail({ prompt: "Legacy task", transcript: [] }));
		expect(messages).toEqual([
			expect.objectContaining({ kind: "parent_message", text: "Legacy task" }),
		]);
	});

	// A persisted hub steer folds to `parent_message` server-side, so the head
	// guard can no longer key on that kind alone: for a legacy child (no
	// launch_message_id) a steer would otherwise stand in for the launch task
	// and the row naming the whole conversation would vanish from the phone.
	it("keeps the launch prompt head for a legacy child that was steered", () => {
		const steer = entry("s1", "parent_message", "Focus on retries");
		const messages = agentConversationEntries(detail({
			prompt: "Implement the route",
			launch_message_id: "",
			transcript: [steer],
		}));
		expect(messages).toEqual([
			expect.objectContaining({ kind: "parent_message", text: "Implement the route" }),
			steer,
		]);
	});

	// The complement: a real launch head must still suppress the synthetic one,
	// including when the row carries the role preamble the launch message has.
	it("does not re-add the head when a launch row already carries the prompt", () => {
		const launch = entry("launch", "parent_message", "Role preamble\n\nImplement the route");
		const messages = agentConversationEntries(detail({
			prompt: "Implement the route",
			launch_message_id: "",
			transcript: [launch, entry("s1", "parent_message", "Focus on retries")],
		}));
		expect(messages[0]).toBe(launch);
		expect(messages.filter((m) => m.kind === "parent_message")).toHaveLength(2);
	});

	// `prompt` arrives as a bounded preview, so a truncated head still matches.
	it("recognises a truncated prompt preview as the launch head", () => {
		const launch = entry("launch", "parent_message", `Preamble\n\n${"x".repeat(40)} tail`);
		const messages = agentConversationEntries(detail({
			prompt: `${"x".repeat(40)}…`,
			launch_message_id: "",
			transcript: [launch],
		}));
		expect(messages).toEqual([launch]);
	});
});

describe("Agents sheet hierarchy", () => {
	it("groups long recursive hierarchies into path, peers, and direct children", () => {
		const roster = [
			row("ancestor", null), row("parent", "ancestor"), row("current", "parent"),
			row("peer", "parent"), row("child", "current"), row("grandchild", "child"),
		];
		expect(groupAgents(detail(), roster)).toEqual({
			path: [roster[0], roster[1]],
			peers: [roster[3]],
			children: [roster[4]],
		});
	});

	it("skips stale relationship ids without losing retained navigation", () => {
		const retainedParent = row("parent", null);
		expect(groupAgents(detail({ ancestor_ids: ["expired", "parent"], peer_ids: ["gone"] }), [retainedParent])).toEqual({
			path: [retainedParent], peers: [], children: [],
		});
	});
});
