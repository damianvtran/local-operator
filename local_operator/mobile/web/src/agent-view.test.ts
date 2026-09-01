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

	// `prompt` arrives as a bounded preview (SUBAGENT_PROMPT_PREVIEW_CHARS = 1000,
	// `_compact` emitting `text[: limit - 1] + "…"`), so a head truncated by that
	// cap must still match — as a PREFIX, since its tail was cut off.
	it("recognises a truncated prompt preview as the launch head", () => {
		const body = "x".repeat(999);
		const launch = entry("launch", "parent_message", `Preamble\n\n${body} tail`);
		const messages = agentConversationEntries(detail({
			prompt: `${body}…`,
			launch_message_id: "",
			transcript: [launch],
		}));
		expect(messages).toEqual([launch]);
	});

	// Finding 8. `…` is one keystroke for a human, so "ends in an ellipsis" is
	// not evidence of truncation. A SHORT prompt ending that way must stay on
	// the anchored path — taking the loose prefix path let a steer that merely
	// quotes the task suppress the head, the exact defect the anchoring closed.
	it("does not treat an author's own trailing ellipsis as a truncated preview", () => {
		const prompt = "Investigate why the retry loop stalls…";
		const steer = entry("s1", "parent_message", `About 'Investigate why the retry loop stalls' - focus on retries first`);
		const messages = agentConversationEntries(detail({
			prompt,
			launch_message_id: "",
			transcript: [steer],
		}));
		expect(messages).toEqual([
			expect.objectContaining({ kind: "parent_message", text: prompt }),
			steer,
		]);
	});

	// Finding 8, the case that GUARDS the length gate. The test above asserts an
	// outcome both versions of the predicate produce: its steer QUOTES the task,
	// so it fails the loose path's `startsWith` whether or not the gate is there.
	// Removing only the gate leaves it green. A steer that BEGINS with the stem
	// does reach that branch, so it flips from "head kept" to "suppressed" the
	// moment the length gate stops holding a short author-typed `…` off the
	// loose path — which is the regression this pair exists to catch.
	it("keeps the head when a steer begins with an author's own ellipsis prompt", () => {
		const prompt = "Fix the retry loop…";
		const steer = entry("s1", "parent_message", "Fix the retry loop, but only for 5xx responses");
		const messages = agentConversationEntries(detail({
			prompt,
			launch_message_id: "",
			transcript: [steer],
		}));
		expect(messages).toEqual([
			expect.objectContaining({ kind: "parent_message", text: prompt }),
			steer,
		]);
	});

	// Finding 12. Head detection walks the row's paragraph-delimited suffixes.
	// Materialising and normalising each one re-scans the tail per break, which
	// is quadratic in row length (9.4s for a 580KB row with 2000 breaks), and
	// the wire `text` for these row kinds is not length-bounded. The linear form
	// normalises once and compares at offsets, so this must stay well under the
	// old cost while still anchoring at a paragraph boundary and nowhere else.
	it("matches a launch head deep in a long row without quadratic cost", () => {
		const prompt = "Implement the retry route";
		const filler = Array.from({ length: 2_000 }, (_, i) => `paragraph ${i} ${"y".repeat(300)}`);
		// The task sits at a real paragraph boundary far into the row.
		const head = entry("h1", "parent_message", `${filler.join("\n\n")}\n\n${prompt}`);
		// Same bulk, but the task is mid-sentence, so it must NOT suppress.
		const steer = entry("s1", "parent_message", `${filler.join("\n\n")} and ${prompt}`);

		const started = performance.now();
		expect(agentConversationEntries(detail({
			prompt, launch_message_id: "", transcript: [head],
		}))).toEqual([head]);
		expect(agentConversationEntries(detail({
			prompt, launch_message_id: "", transcript: [steer],
		}))).toEqual([
			expect.objectContaining({ kind: "parent_message", text: prompt }),
			steer,
		]);
		// Measured on this exact shape: the suffix-materialising walk takes ~4.3s
		// for the pair, the linear form ~20ms. The threshold sits two orders of
		// magnitude above the fixed cost and ~3x below the broken one, so it fails
		// on a reintroduced quadratic without flaking on a loaded CI runner.
		expect(performance.now() - started).toBeLessThan(1_500);
	});

	// Finding 9. End-anchoring alone accepted a steer that CLOSES by restating
	// the task. The launch row is structurally `{preamble}\n\n{prompt}`, so the
	// task always begins at a paragraph boundary; a restatement mid-sentence
	// does not, and must not suppress the head.
	it("does not treat a steer that ends by restating the task as the launch head", () => {
		const prompt = "Implement the route";
		const steer = entry("s1", "parent_message", "Actually ignore that and Implement the route");
		const messages = agentConversationEntries(detail({
			prompt,
			launch_message_id: "",
			transcript: [steer],
		}));
		expect(messages).toEqual([
			expect.objectContaining({ kind: "parent_message", text: prompt }),
			steer,
		]);
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
