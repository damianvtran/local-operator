// @vitest-environment happy-dom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AgentConversation, AgentUnavailable } from "./screens/agent-view";
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
		parent_job_id: parentJobId, session_id: `${jobId}-session`, prompt: "", launch_message_id: "", effort: "high",
		ancestors: [], ancestor_ids: [], child_ids: [], peer_ids: [], transcript: [], todos: [], activity: "working",
	};
}

function fixture(): { detail: SubagentDetail; projection: SessionProjection } {
	const detail: SubagentDetail = {
		...row("current", "parent"), version: 3, label: "current-agent",
		ancestor_ids: ["ancestor", "parent"], peer_ids: ["peer"], child_ids: ["child"],
		prompt: "One request", launch_message_id: "subagent-launch:current", transcript: [
			entry("subagent-launch:current", "user", "Role instructions\n\nOne request"),
			entry("assistant", "assistant", "One response"),
			entry("steer", "user", "Preserve this later steering message"),
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
	cleanup();
	vi.restoreAllMocks();
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

	it("contains dialog focus, hides the background, and restores the opener", async () => {
		const { detail, projection } = fixture();
		const { container } = render(
			<AgentConversation sessionId="root" jobId="current" projection={projection} connected detail={detail} />,
		);
		const opener = screen.getByRole("button", { name: "open agent navigation" });
		opener.focus();
		fireEvent.click(opener);

		const dialog = screen.getByRole("dialog");
		const close = screen.getByRole("button", { name: "close sheet" });
		await waitFor(() => expect(document.activeElement).toBe(close));
		expect(dialog.getAttribute("aria-modal")).toBe("true");
		for (const sibling of Array.from(dialog.parentElement!.children)) {
			if (sibling !== dialog) {
				expect(sibling.hasAttribute("inert")).toBe(true);
				expect(sibling.getAttribute("aria-hidden")).toBe("true");
			}
		}

		fireEvent.keyDown(close, { key: "Tab", shiftKey: true });
		await waitFor(() => expect(document.activeElement).toBe(screen.getByRole("button", { name: /child/ })));
		fireEvent.keyDown(document.activeElement!, { key: "Tab" });
		await waitFor(() => expect(document.activeElement).toBe(close));
		fireEvent.keyDown(dialog, { key: "Escape" });

		await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
		expect(document.activeElement).toBe(opener);
		expect(container.querySelector("[inert]")).toBeNull();
		expect(opener.closest("header")?.getAttribute("aria-hidden")).toBeNull();
	});

	it("uses result vocabulary and state-specific unavailable copy", () => {
		const { detail, projection } = fixture();
		const completed = { ...detail, status: "completed" as const, result_text: "Completed result" };
		const { rerender } = render(
			<AgentConversation sessionId="root" jobId="current" projection={projection} connected detail={completed} />,
		);
		expect(screen.getByText("✓ Result from current-agent")).toBeTruthy();
		expect(screen.queryByText(/handoff/i)).toBeNull();

		const retry = vi.fn();
		history.replaceState({}, "", "#/s/root/a/unavailable");
		const replaceState = vi.spyOn(history, "replaceState");
		const pushState = vi.spyOn(history, "pushState");
		rerender(<AgentUnavailable sessionId="root" parentPath="/s/root/a/parent" onRetry={retry} />);
		expect(screen.getByText("Unavailable")).toBeTruthy();
		expect(screen.queryByText("Loading activity")).toBeNull();
		fireEvent.click(screen.getByRole("button", { name: "Retry" }));
		expect(retry).toHaveBeenCalledOnce();
		fireEvent.click(screen.getByRole("button", { name: "Back to parent" }));
		expect(replaceState).toHaveBeenCalledWith(expect.anything(), "", "#/s/root/a/parent");
		fireEvent.click(screen.getByRole("button", { name: "View root" }));
		expect(pushState).toHaveBeenCalledWith(expect.anything(), "", "#/s/root");
	});

	it("gives tool, task, and child disclosures the shared mobile target floor", () => {
		const { detail, projection } = fixture();
		const tool = {
			...entry("tool", "tool", ""),
			tool_name: "edit",
			summary: "agent-view.tsx",
			intent: "Fixing mobile controls",
		};
		const withDisclosures = {
			...detail,
			transcript: [...detail.transcript, tool],
			todos: [{ name: "Remediation", items: [{ text: "Validate targets", status: "pending" as const, reason: "" }] }],
		};
		render(
			<AgentConversation sessionId="root" jobId="current" projection={projection} connected detail={withDisclosures} />,
		);
		for (const control of [
			screen.getByRole("button", { name: /edit agent-view/ }),
			screen.getByRole("button", { name: /tasks 0\/1/ }),
			screen.getByRole("button", { name: /child agents 1\/1 running/ }),
		]) {
			expect(control.className).toContain("min-h-11");
		}
	});

	it("keeps the delegated request singular and removes dominant legacy chrome", () => {
		const { detail, projection } = fixture();
		render(<AgentConversation sessionId="root" jobId="current" projection={projection} connected detail={detail} />);
		expect(screen.getAllByText(/One request/)).toHaveLength(1);
		expect(screen.getByText("Preserve this later steering message")).toBeTruthy();
		expect(screen.queryByRole("navigation", { name: "agent lineage" })).toBeNull();
		expect(screen.queryByText("Parent request")).toBeNull();
		expect(screen.queryByText(/send commands from the root conversation/i)).toBeNull();
		expect(screen.getByRole("button", { name: "Open parent to steer" })).toBeTruthy();
	});
});
