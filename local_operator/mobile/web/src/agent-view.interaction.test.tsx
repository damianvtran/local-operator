// @vitest-environment happy-dom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { App } from "./app";
import { AgentConversation, AgentUnavailable } from "./screens/agent-view";
import * as api from "./api";
import * as store from "./store";
import type { SessionProjection, SubagentDetail, SubagentRow, TranscriptEntry } from "./types";

/* Spreads the ACTUAL api module so class exports keep their identity: the
   screen distinguishes a terminal 404 by `instanceof HttpError`, which a
   partial factory would replace with `undefined`. Network functions are
   still stubbed per test. */
vi.mock("./api", async (importOriginal) => ({
	...(await importOriginal<typeof api>()),
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentDetail: vi.fn(),
	imageUrl: vi.fn(() => ""),
	sendCommand: vi.fn(async () => ({ ok: true, detail: "steering queued" })),
	markSessionSeen: vi.fn(async () => ({ ok: true })),
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
	localStorage.clear();
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

	it("opens the real parent composer and submits one UUID-addressed steer", async () => {
		const { detail, projection } = fixture();
		const topLevel = { ...detail, parent_job_id: null };
		const sendCommand = vi.mocked(api.sendCommand);
		vi.mocked(api.getSubagentDetail).mockResolvedValue(topLevel);
		vi.spyOn(store, "useProjection").mockReturnValue({ projection, connected: true });
		vi.spyOn(store, "retainProjectionStream").mockReturnValue(() => undefined);
		vi.stubGlobal("crypto", {
			randomUUID: vi.fn(() => "12345678-1234-4678-9234-567812345678"),
		});
		history.replaceState({}, "", "#/s/root/a/current");
		render(<App />);

		await waitFor(() =>
			expect(screen.getByRole("button", { name: "Open parent to steer" })).toBeTruthy(),
		);
		fireEvent.click(screen.getByRole("button", { name: "Open parent to steer" }));
		expect(location.hash).toBe("#/s/root");
		const composer = screen.getByPlaceholderText("Message Local Operator…");
		const steer = screen.getByRole("button", { name: "steer" });
		expect((steer as HTMLButtonElement).disabled).toBe(true);
		fireEvent.change(composer, { target: { value: "Please report back once" } });
		fireEvent.click(steer);

		await waitFor(() => expect(sendCommand).toHaveBeenCalledOnce());
		expect(sendCommand).toHaveBeenCalledWith("root", {
			op: "steer",
			command_id: "12345678-1234-4678-9234-567812345678",
			text: "Please report back once",
			images: undefined,
		});
		expect(location.hash).toBe("#/s/root");
		history.back();
		window.dispatchEvent(new PopStateEvent("popstate"));
		await waitFor(() => expect(location.hash).toBe("#/s/root/a/current"));
		expect(screen.getByText("Preserve this later steering message")).toBeTruthy();
	});

	it("acknowledges the lost-response envelope without discarding a later draft", async () => {
		const { detail, projection } = fixture();
		const topLevel = { ...detail, parent_job_id: null };
		let rejectFirst!: (reason: Error) => void;
		const firstResponse = new Promise<never>((_resolve, reject) => { rejectFirst = reject; });
		vi.mocked(api.getSubagentDetail).mockResolvedValue(topLevel);
		vi.mocked(api.sendCommand)
			.mockReturnValueOnce(firstResponse)
			.mockResolvedValueOnce({ ok: true, detail: "already admitted" })
			.mockResolvedValueOnce({ ok: true, detail: "steering queued" });
		const randomUUID = vi.fn()
			.mockReturnValueOnce("12345678-1234-4678-9234-567812345678")
			.mockReturnValueOnce("87654321-4321-4678-9234-567812345678");
		vi.stubGlobal("crypto", { randomUUID });
		vi.spyOn(store, "useProjection").mockReturnValue({ projection, connected: true });
		vi.spyOn(store, "retainProjectionStream").mockReturnValue(() => undefined);
		history.replaceState({}, "", "#/s/root/a/current");
		render(<App />);
		await waitFor(() =>
			expect(screen.getByRole("button", { name: "Open parent to steer" })).toBeTruthy(),
		);
		fireEvent.click(screen.getByRole("button", { name: "Open parent to steer" }));
		const composer = screen.getByPlaceholderText("Message Local Operator…") as HTMLTextAreaElement;
		fireEvent.change(composer, { target: { value: "Original instruction" } });
		fireEvent.click(screen.getByRole("button", { name: "steer" }));
		await waitFor(() => expect(api.sendCommand).toHaveBeenCalledTimes(1));
		fireEvent.change(composer, { target: { value: "Edited draft" } });
		rejectFirst(new Error("response lost"));
		await waitFor(() => expect(screen.getByRole("alert")).toBeTruthy());
		expect((screen.getByRole("button", { name: "steer" }) as HTMLButtonElement).disabled).toBe(true);

		fireEvent.click(screen.getByRole("button", { name: "Retry earlier instruction" }));
		await waitFor(() => expect(api.sendCommand).toHaveBeenCalledTimes(2));
		expect(vi.mocked(api.sendCommand).mock.calls[1]?.[1]).toEqual(
			vi.mocked(api.sendCommand).mock.calls[0]?.[1],
		);
		/* D11: the delivered acknowledgement is a success notice, not the danger
		   alert — there is no alert once recovery succeeds. */
		expect(screen.queryByRole("alert")).toBeNull();
		expect(
			screen.getByText("Earlier instruction delivered. Your edited draft is ready to send."),
		).toBeTruthy();
		expect(composer.value).toBe("Edited draft");

		fireEvent.click(screen.getByRole("button", { name: "steer" }));
		await waitFor(() => expect(api.sendCommand).toHaveBeenCalledTimes(3));
		expect(vi.mocked(api.sendCommand).mock.calls[2]?.[1]).toMatchObject({
			command_id: "87654321-4321-4678-9234-567812345678",
			text: "Edited draft",
		});
		await waitFor(() => expect(composer.value).toBe(""));
	});

	it("restores a lost prompt for touch retry after streaming reload", async () => {
		const { projection } = fixture();
		const idle = { ...projection, streaming: false };
		let rejectFirst!: (reason: Error) => void;
		const firstResponse = new Promise<never>((_resolve, reject) => { rejectFirst = reject; });
		vi.mocked(api.sendCommand)
			.mockReturnValueOnce(firstResponse)
			.mockResolvedValueOnce({ ok: true, detail: "already admitted" })
			.mockResolvedValueOnce({ ok: true, detail: "steering queued" });
		const randomUUID = vi.fn()
			.mockReturnValueOnce("12345678-1234-4678-9234-567812345678")
			.mockReturnValueOnce("87654321-4321-4678-9234-567812345678");
		vi.stubGlobal("crypto", { randomUUID });
		const projectionSpy = vi.spyOn(store, "useProjection").mockReturnValue({ projection: idle, connected: true });
		vi.spyOn(store, "retainProjectionStream").mockReturnValue(() => undefined);
		history.replaceState({}, "", "#/s/root");
		const mounted = render(<App />);
		const composer = screen.getByPlaceholderText("Message Local Operator…") as HTMLTextAreaElement;
		fireEvent.change(composer, { target: { value: "Original instruction" } });
		fireEvent.click(screen.getByRole("button", { name: "send" }));
		await waitFor(() => expect(api.sendCommand).toHaveBeenCalledTimes(1));
		fireEvent.change(composer, { target: { value: "Edited draft" } });
		rejectFirst(new TypeError("response lost"));
		await waitFor(() => expect(screen.getByRole("button", { name: "Retry earlier instruction" })).toBeTruthy());

		projectionSpy.mockReturnValue({ projection: { ...projection, streaming: true }, connected: true });
		mounted.unmount();
		render(<App />);
		const reloadedComposer = screen.getByPlaceholderText("Message Local Operator…") as HTMLTextAreaElement;
		expect(reloadedComposer.value).toBe("Edited draft");
		expect(screen.getByRole("alert").textContent).toContain("earlier instruction may have been delivered");
		const retry = screen.getByRole("button", { name: "Retry earlier instruction" });
		expect(retry.className).toContain("min-h-11");
		fireEvent.click(retry);

		await waitFor(() => expect(api.sendCommand).toHaveBeenCalledTimes(2));
		expect(vi.mocked(api.sendCommand).mock.calls[0]?.[1]).toMatchObject({ op: "prompt" });
		expect(vi.mocked(api.sendCommand).mock.calls[1]?.[1]).toEqual(
			vi.mocked(api.sendCommand).mock.calls[0]?.[1],
		);
		expect(screen.queryByRole("alert")).toBeNull();
		expect(
			screen.getByText("Earlier instruction delivered. Your edited draft is ready to send."),
		).toBeTruthy();
		fireEvent.click(screen.getByRole("button", { name: "steer" }));
		await waitFor(() => expect(api.sendCommand).toHaveBeenCalledTimes(3));
		expect(vi.mocked(api.sendCommand).mock.calls[2]?.[1]).toMatchObject({
			op: "steer",
			command_id: "87654321-4321-4678-9234-567812345678",
			text: "Edited draft",
		});
	});

	it("keeps a failed parent instruction with an actionable non-protocol error", async () => {
		const { detail, projection } = fixture();
		const topLevel = { ...detail, parent_job_id: null };
		vi.mocked(api.getSubagentDetail).mockResolvedValue(topLevel);
		vi.mocked(api.sendCommand).mockRejectedValueOnce(
			new Error("command_id must be a UUID string"),
		);
		vi.spyOn(store, "useProjection").mockReturnValue({ projection, connected: true });
		vi.spyOn(store, "retainProjectionStream").mockReturnValue(() => undefined);
		history.replaceState({}, "", "#/s/root/a/current");
		render(<App />);
		await waitFor(() =>
			expect(screen.getByRole("button", { name: "Open parent to steer" })).toBeTruthy(),
		);
		fireEvent.click(screen.getByRole("button", { name: "Open parent to steer" }));
		const composer = screen.getByPlaceholderText(
			"Message Local Operator…",
		) as HTMLTextAreaElement;
		fireEvent.change(composer, { target: { value: "Retry this instruction" } });
		fireEvent.click(screen.getByRole("button", { name: "steer" }));

		await waitFor(() =>
			expect(screen.getByRole("alert").textContent).toBe(
				"Couldn’t send this instruction. Try again.",
			),
		);
		expect(composer.value).toBe("Retry this instruction");
		expect(screen.queryByText(/command_id/)).toBeNull();
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

	it("lazily fetches the child transcript when the wire carries none", async () => {
		/* The projection no longer embeds subagent transcripts (they overran the
		   daemon's 1 MB control-frame cap and wedged real-time updates). A modern
		   detail arrives with an empty transcript, so the sheet must fetch its
		   newest page from the child-history endpoint and render THAT. */
		const { detail, projection } = fixture();
		const empty = { ...detail, transcript: [], prompt: "", launch_message_id: "" };
		const getSubagentHistory = vi.mocked(api.getSubagentHistory);
		getSubagentHistory.mockResolvedValueOnce({
			entries: [entry("fetched", "assistant", "Lazily loaded reply")],
			has_more: false,
		});
		render(
			<AgentConversation sessionId="root" jobId="current" projection={projection} connected detail={empty} />,
		);
		await waitFor(() =>
			expect(getSubagentHistory).toHaveBeenCalledWith("root", "current", null, 80, expect.anything()),
		);
		await waitFor(() => expect(screen.getByText("Lazily loaded reply")).toBeTruthy());
	});

	it("does not fetch when a legacy daemon still inlines the transcript", () => {
		/* Back-compat: a producer that still ships detail.transcript wins outright
		   and the sheet must render it without an extra network round trip. */
		const { detail, projection } = fixture();
		const getSubagentHistory = vi.mocked(api.getSubagentHistory);
		getSubagentHistory.mockClear();
		render(
			<AgentConversation sessionId="root" jobId="current" projection={projection} connected detail={detail} />,
		);
		expect(screen.getByText("One response")).toBeTruthy();
		expect(getSubagentHistory).not.toHaveBeenCalled();
	});

	it("surfaces a retry when a settled child's one transcript fetch fails (U1)", async () => {
		/* A settled child has no poll, so a single dropped fetch is terminal.
		   The body must not be a silent blank — it shows an error + retry, and
		   the retry re-pulls and renders the entries on success. */
		const { detail, projection } = fixture();
		const settled = {
			...detail,
			status: "completed" as const,
			result_text: "The result",
			transcript: [],
			prompt: "",
			launch_message_id: "",
		};
		const getSubagentHistory = vi.mocked(api.getSubagentHistory);
		getSubagentHistory.mockReset();
		getSubagentHistory
			.mockRejectedValueOnce(new Error("network down"))
			.mockResolvedValueOnce({
				entries: [entry("recovered", "assistant", "Recovered step")],
				has_more: false,
			});
		render(
			<AgentConversation sessionId="root" jobId="current" projection={projection} connected detail={settled} />,
		);
		// The failure surfaces a visible alert + retry, not a blank body.
		await waitFor(() => expect(screen.getByText("Couldn't load the transcript.")).toBeTruthy());
		const retry = screen.getByRole("button", { name: "Retry" });
		// The outcome tail still renders alongside the empty-body error.
		expect(screen.getByText("✓ Result from current-agent")).toBeTruthy();
		fireEvent.click(retry);
		await waitFor(() => expect(screen.getByText("Recovered step")).toBeTruthy());
		expect(screen.queryByText("Couldn't load the transcript.")).toBeNull();
	});

	it("treats a 404 child history as terminal, not a retry loop", async () => {
		/* A child the daemon cannot route has no transcript to serve: the route
		   answers 404 forever. The body must say so and offer a way back to the
		   parent instead of a Retry that can never succeed (the console error
		   this replaced). The way back REPLACES the hierarchy fallback rather
		   than pushing it, like the header's own "‹" (D2/MINOR-1): a pushed
		   entry leaves the dead child as the predecessor the phone's Back key
		   returns to. */
		const replaceState = vi.spyOn(history, "replaceState");
		const pushState = vi.spyOn(history, "pushState");
		history.replaceState({}, "", "#/s/root/a/current");
		const { detail, projection } = fixture();
		const settled = {
			...detail,
			status: "completed" as const,
			transcript: [],
			prompt: "",
			launch_message_id: "",
		};
		const getSubagentHistory = vi.mocked(api.getSubagentHistory);
		getSubagentHistory.mockReset();
		getSubagentHistory.mockRejectedValue(
			new api.HttpError(404, "subagent history unavailable"),
		);
		render(
			<AgentConversation sessionId="root" jobId="current" projection={projection} connected detail={settled} />,
		);
		await waitFor(() => expect(screen.getByText("No transcript for this agent.")).toBeTruthy());
		expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
		fireEvent.click(screen.getByRole("button", { name: "Back to parent" }));
		expect(replaceState).toHaveBeenCalledWith(expect.anything(), "", "#/s/root/a/parent");
		expect(pushState).not.toHaveBeenCalled();
	});

	it("renders the terminal 404 notice for a child that carries a launch prompt (D1/Q1)", async () => {
		/* The shape the phone actually receives for a launched child: the roster
		   push fills ``prompt``, ``agentConversationEntries`` synthesizes a head
		   row from it, and the history route 404s. The notice must therefore not
		   be gated on an empty body — it renders under that row and above the
		   result tail, with the launch row kept as context. */
		const { detail, projection } = fixture();
		const settled = {
			...detail,
			status: "completed" as const,
			result_text: "Found the seam.",
			transcript: [],
		};
		expect(settled.prompt).not.toBe("");
		const getSubagentHistory = vi.mocked(api.getSubagentHistory);
		getSubagentHistory.mockReset();
		getSubagentHistory.mockRejectedValue(
			new api.HttpError(404, "subagent history unavailable"),
		);
		render(
			<AgentConversation sessionId="root" jobId="current" projection={projection} connected detail={settled} />,
		);
		await waitFor(() => expect(screen.getByText("No transcript for this agent.")).toBeTruthy());
		expect(
			screen.getByText(
				"Its conversation steps couldn't be read, so only the result and activity below are shown.",
			),
		).toBeTruthy();
		// The launch row is kept, not suppressed to make room for the notice.
		const launchRow = screen.getAllByText(/One request/)[0];
		expect(launchRow).toBeTruthy();
		// Terminal: no Retry that cannot succeed.
		expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
		// Order matters: launch row, then the notice, then the result tail.
		const order = Array.from(document.querySelectorAll("*"));
		expect(order.indexOf(launchRow)).toBeLessThan(
			order.indexOf(screen.getByText("No transcript for this agent.")),
		);
		expect(order.indexOf(screen.getByText("No transcript for this agent."))).toBeLessThan(
			order.indexOf(screen.getByText(/Result from current-agent/)),
		);
	});

	it("keeps the retry card for a prompt-carrying child when the fetch drops (Q2)", async () => {
		/* The same suppression hit the transient branch: a dropped fetch on a
		   launched child (rows on screen from the synthesized head) must still
		   surface the error and recover in place, exactly as the prompt-less
		   control does. */
		const { detail, projection } = fixture();
		const settled = {
			...detail,
			status: "completed" as const,
			result_text: "Found the seam.",
			transcript: [],
		};
		const getSubagentHistory = vi.mocked(api.getSubagentHistory);
		getSubagentHistory.mockReset();
		getSubagentHistory
			.mockRejectedValueOnce(new Error("network down"))
			.mockResolvedValueOnce({
				entries: [entry("recovered", "assistant", "Recovered step")],
				has_more: false,
			});
		render(
			<AgentConversation sessionId="root" jobId="current" projection={projection} connected detail={settled} />,
		);
		await waitFor(() => expect(screen.getByText("Couldn't load the transcript.")).toBeTruthy());
		expect(screen.getAllByText(/One request/).length).toBeGreaterThan(0);
		// The live region carries the words only — never the control (D4).
		expect(screen.getByRole("alert").querySelector("button")).toBeNull();
		fireEvent.click(screen.getByRole("button", { name: "Retry" }));
		await waitFor(() => expect(screen.getByText("Recovered step")).toBeTruthy());
		expect(screen.queryByText("Couldn't load the transcript.")).toBeNull();
	});

	it("keeps the terminal notice's live region off its action (D4)", async () => {
		/* ``role="alert"`` wrapped the whole card, so the AX tree announced an
		   unnamed alert containing a button. The region is the text; the action
		   is its sibling. */
		const { detail, projection } = fixture();
		const settled = { ...detail, status: "completed" as const, transcript: [] };
		const getSubagentHistory = vi.mocked(api.getSubagentHistory);
		getSubagentHistory.mockReset();
		getSubagentHistory.mockRejectedValue(
			new api.HttpError(404, "subagent history unavailable"),
		);
		render(
			<AgentConversation sessionId="root" jobId="current" projection={projection} connected detail={settled} />,
		);
		await waitFor(() => expect(screen.getByText("No transcript for this agent.")).toBeTruthy());
		const alert = screen.getByRole("alert");
		expect(alert.textContent).toContain("No transcript for this agent.");
		expect(alert.textContent).not.toContain("Back to parent");
		expect(alert.querySelector("button")).toBeNull();
		expect(screen.getByRole("button", { name: "Back to parent" })).toBeTruthy();
	});

	it("shows the loading affordance under the launch row while the tail is in flight", async () => {
		/* Same guard, third state: a prompt-carrying child whose first fetch has
		   not landed renders its synthesized launch row AND the loading
		   affordance — not a launch row alone that reads like a complete
		   conversation (U2). */
		const { detail, projection } = fixture();
		const empty = { ...detail, transcript: [] };
		const getSubagentHistory = vi.mocked(api.getSubagentHistory);
		getSubagentHistory.mockReset();
		getSubagentHistory.mockReturnValueOnce(new Promise(() => undefined));
		render(
			<AgentConversation sessionId="root" jobId="current" projection={projection} connected detail={empty} />,
		);
		expect(screen.getAllByText(/One request/).length).toBeGreaterThan(0);
		expect(screen.getByText("Loading agent activity…")).toBeTruthy();
	});

	it("re-pulls a settled child's transcript when the link reconnects (U1)", async () => {
		/* The fetch effect lists ``connected`` as a dependency, mirroring the
		   detail loader, so a restored link re-pulls a settled child that missed
		   its one fetch while offline — without a manual retry. */
		const { detail, projection } = fixture();
		const settled = {
			...detail,
			status: "completed" as const,
			transcript: [],
			prompt: "",
			launch_message_id: "",
		};
		const getSubagentHistory = vi.mocked(api.getSubagentHistory);
		getSubagentHistory.mockReset();
		getSubagentHistory
			.mockRejectedValueOnce(new Error("offline"))
			.mockResolvedValueOnce({
				entries: [entry("afterreconnect", "assistant", "Back online reply")],
				has_more: false,
			});
		const { rerender } = render(
			<AgentConversation sessionId="root" jobId="current" projection={projection} connected={false} detail={settled} />,
		);
		await waitFor(() => expect(screen.getByText("Couldn't load the transcript.")).toBeTruthy());
		// Reconnect: the same detail, connected flips true → the effect re-pulls.
		rerender(
			<AgentConversation sessionId="root" jobId="current" projection={projection} connected detail={settled} />,
		);
		await waitFor(() => expect(screen.getByText("Back online reply")).toBeTruthy());
	});

	it("shows a loading affordance in the open→load gap, not a blank body (U2)", async () => {
		/* While the first transcript fetch is in flight and no entries are on
		   screen, the body must read as loading — a spinner + label — rather than
		   an empty window under a fully-painted header. */
		const { detail, projection } = fixture();
		const empty = { ...detail, transcript: [], prompt: "", launch_message_id: "" };
		const getSubagentHistory = vi.mocked(api.getSubagentHistory);
		getSubagentHistory.mockReset();
		let resolveFetch!: (value: { entries: TranscriptEntry[]; has_more: boolean }) => void;
		getSubagentHistory.mockReturnValueOnce(
			new Promise((resolve) => {
				resolveFetch = resolve;
			}),
		);
		render(
			<AgentConversation sessionId="root" jobId="current" projection={projection} connected detail={empty} />,
		);
		expect(screen.getByText("Loading agent activity…")).toBeTruthy();
		resolveFetch({ entries: [entry("landed", "assistant", "Landed reply")], has_more: false });
		await waitFor(() => expect(screen.getByText("Landed reply")).toBeTruthy());
		expect(screen.queryByText("Loading agent activity…")).toBeNull();
	});

	it("keeps the header identity stable while the detail loads (D1, D3)", async () => {
		/* Tapping a roster row that already carried the label/agent/effort/status
		   must not flicker the header to a generic "Agent" placeholder. The
		   loading header paints the known identity from the projection row, with
		   the effort tier spelled out ("hi" → "high", D3). */
		const { detail, projection } = fixture();
		// A fresh job id: the module-level detail cache must not already hold it,
		// or the screen renders the full conversation instead of the loading gap.
		const jobId = "d1-fresh-child";
		const tapped = {
			...row(jobId, "parent"),
			label: "code-reviewer",
			agent: "reviewer",
			effort: "hi",
		};
		const withRow: SessionProjection = {
			...projection,
			subagents: [...projection.subagents, tapped],
		};
		void detail;
		// Detail never lands: the header must rely on the projection identity.
		vi.mocked(api.getSubagentDetail).mockReturnValue(new Promise(() => undefined));
		vi.spyOn(store, "useProjection").mockReturnValue({ projection: withRow, connected: true });
		vi.spyOn(store, "retainProjectionStream").mockReturnValue(() => undefined);
		history.replaceState({}, "", `#/s/root/a/${jobId}`);
		render(<App />);
		// The tapped row's real label shows immediately, never the placeholder.
		await waitFor(() => expect(screen.getByText("code-reviewer")).toBeTruthy());
		expect(screen.queryByText("Agent")).toBeNull();
		expect(screen.queryByText("Loading activity")).toBeNull();
		// Effort tier spelled out to match the session footer (D3): "hi" → "high".
		expect(screen.getByText(/reviewer · high/)).toBeTruthy();
	});

	it("keeps a running child's dropped poll quiet while its rows are on screen", async () => {
		/* A running child polls every 1.5 s, so its own poll is the recovery path
		   for a dropped tick (see the catch block): the transient card must not
		   paint over rows that are still live — on the flaky link this surface
		   exists for it would flash an error once per drop over a transcript the
		   user can read — and the next successful tick restores the tail. The row
		   has to LAND first, so the first fetch succeeds and the failure is the
		   tick after it. */
		vi.useFakeTimers();
		try {
			const { detail, projection } = fixture();
			const running = {
				...detail,
				status: "running" as const,
				transcript: [],
				prompt: "",
				launch_message_id: "",
			};
			const live = entry("live", "assistant", "Live step one");
			const getSubagentHistory = vi.mocked(api.getSubagentHistory);
			getSubagentHistory.mockReset();
			getSubagentHistory
				.mockResolvedValueOnce({ entries: [live], has_more: false })
				.mockRejectedValueOnce(new Error("poll dropped"))
				.mockResolvedValueOnce({
					entries: [live, entry("tail", "assistant", "Live step two")],
					has_more: false,
				});
			render(
				<AgentConversation sessionId="root" jobId="current" projection={projection} connected detail={running} />,
			);
			await vi.waitFor(() => expect(screen.getByText("Live step one")).toBeTruthy());
			// One dropped poll tick: the fetch really ran and really failed...
			await vi.advanceTimersByTimeAsync(1600);
			expect(getSubagentHistory).toHaveBeenCalledTimes(2);
			await vi.advanceTimersByTimeAsync(10);
			// ...the rows are still there, and the drop is not announced over them.
			expect(screen.getByText("Live step one")).toBeTruthy();
			expect(screen.queryByText("Couldn't load the transcript.")).toBeNull();
			expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
			// The next tick lands the tail with no card in between, no retry needed.
			await vi.advanceTimersByTimeAsync(1600);
			await vi.waitFor(() => expect(screen.getByText("Live step two")).toBeTruthy());
			expect(screen.queryByText("Couldn't load the transcript.")).toBeNull();
		} finally {
			vi.useRealTimers();
		}
	});

	it("keeps a running child's dropped polls quiet once a prompt-carrying tail has landed", async () => {
		/* The same rule as the test above, in the shape the daemon actually
		   sends: ``prompt`` set (so ``agentConversationEntries`` synthesizes a
		   ``parent_message`` head) AND running AND polling. That head is a label
		   for the child, not a landed transcript row — the gate has to read what
		   the FETCH returned, or this shape would be silent about a drop it must
		   not report; a landed row still suppresses the card, and the poll still
		   recovers without the user asking. */
		vi.useFakeTimers();
		try {
			const { detail, projection } = fixture();
			// ``launch_message_id`` kept: the roster row for a launched child
			// carries both fields, and the entry it names is not in the wire tail.
			const running = { ...detail, status: "running" as const, transcript: [] };
			expect(running.prompt).not.toBe("");
			const live = entry("live", "assistant", "Live step one");
			const getSubagentHistory = vi.mocked(api.getSubagentHistory);
			getSubagentHistory.mockReset();
			// A mode rather than a call counter: the number of ticks a virtual
			// interval produces is an implementation detail of the timer, and the
			// property under test is "every tick since the landing one failed".
			let mode: "live" | "broken" | "recovered" = "live";
			getSubagentHistory.mockImplementation(async () => {
				if (mode === "broken") throw new Error("poll dropped");
				return {
					entries: mode === "live" ? [live] : [live, entry("tail", "assistant", "Live step two")],
					has_more: false,
				};
			});
			render(
				<AgentConversation sessionId="root" jobId="current" projection={projection} connected detail={running} />,
			);
			await vi.waitFor(() => expect(screen.getByText("Live step one")).toBeTruthy());
			// Every tick from here fails: the landed row stays and no card is painted.
			mode = "broken";
			await vi.advanceTimersByTimeAsync(3200);
			expect(getSubagentHistory.mock.calls.length).toBeGreaterThanOrEqual(3);
			expect(screen.getByText("Live step one")).toBeTruthy();
			expect(screen.getAllByText(/One request/).length).toBeGreaterThan(0);
			expect(screen.queryByText("Couldn't load the transcript.")).toBeNull();
			expect(screen.queryByRole("button", { name: "Retry" })).toBeNull();
			mode = "recovered";
			await vi.advanceTimersByTimeAsync(1600);
			await vi.waitFor(() => expect(screen.getByText("Live step two")).toBeTruthy());
			expect(screen.queryByText("Couldn't load the transcript.")).toBeNull();
		} finally {
			vi.useRealTimers();
		}
	});

	it("tells the user when a running child's launch-prompted tail never lands (MAJOR-1)", async () => {
		/* The production shape for a launched child — ``prompt`` set, running,
		   and the history route failing on EVERY poll — with nothing landed. The
		   synthesized head row is a label, not content: gating the exemption on
		   the painted list let it stand in for the transcript, so this child
		   painted its launch row and a live "running" header with no error, no
		   Retry and no loading affordance for as long as it ran. The card is the
		   only signal anything failed, and it is honest here because nothing is
		   in flight; a later tick that lands content clears it in place. */
		vi.useFakeTimers();
		try {
			const { detail, projection } = fixture();
			const running = { ...detail, status: "running" as const, transcript: [] };
			expect(running.prompt).not.toBe("");
			const getSubagentHistory = vi.mocked(api.getSubagentHistory);
			getSubagentHistory.mockReset();
			// Released on demand: the assertion is that the route fails on every
			// poll for a while and then recovers, not that it fails N times.
			let released = false;
			getSubagentHistory.mockImplementation(async () => {
				if (!released) throw new Error("history route dropped");
				return { entries: [entry("landed", "assistant", "Landed at last")], has_more: false };
			});
			render(
				<AgentConversation sessionId="root" jobId="current" projection={projection} connected detail={running} />,
			);
			await vi.waitFor(() => expect(getSubagentHistory).toHaveBeenCalledTimes(1));
			// ~6 s of polls, every one of them failing: the launch row is there,
			// the body is not silent about it.
			await vi.advanceTimersByTimeAsync(6800);
			expect(getSubagentHistory.mock.calls.length).toBeGreaterThanOrEqual(4);
			expect(screen.getAllByText(/One request/).length).toBeGreaterThan(0);
			expect(screen.getByText("Couldn't load the transcript.")).toBeTruthy();
			expect(screen.getByRole("button", { name: "Retry" })).toBeTruthy();
			// The next tick lands rows; the card clears without the user asking.
			released = true;
			await vi.advanceTimersByTimeAsync(1600);
			await vi.waitFor(() => expect(screen.getByText("Landed at last")).toBeTruthy());
			expect(screen.queryByText("Couldn't load the transcript.")).toBeNull();
		} finally {
			vi.useRealTimers();
		}
	});

	it("keeps the retry card when a settled child's later fetch fails (U1)", async () => {
		/* The mirror of the rule above, and the reason it keys on ``running``
		   rather than on rows alone: a settled child has NO poll, so nothing
		   recovers a dropped fetch — not even when rows from an earlier fetch are
		   still on screen, where a rows-only gate would lose the transcript
		   silently. Here the first fetch lands, the link drops, the effect re-runs
		   (``connected`` is a dependency) and fails: the card must be back under
		   the row that is still there. */
		const { detail, projection } = fixture();
		const settled = {
			...detail,
			status: "completed" as const,
			transcript: [],
			prompt: "",
			launch_message_id: "",
		};
		const getSubagentHistory = vi.mocked(api.getSubagentHistory);
		getSubagentHistory.mockReset();
		getSubagentHistory
			.mockResolvedValueOnce({
				entries: [entry("landed", "assistant", "Landed step")],
				has_more: false,
			})
			.mockRejectedValueOnce(new Error("link dropped"));
		const view = (connected: boolean) => (
			<AgentConversation sessionId="root" jobId="current" projection={projection} connected={connected} detail={settled} />
		);
		const { rerender } = render(view(true));
		await waitFor(() => expect(screen.getByText("Landed step")).toBeTruthy());
		rerender(view(false));
		await waitFor(() => expect(screen.getByText("Couldn't load the transcript.")).toBeTruthy());
		expect(screen.getByText("Landed step")).toBeTruthy();
		expect(screen.getByRole("button", { name: "Retry" })).toBeTruthy();
	});

	it("gives both notice cards the app's 8 px title/body pair gap (D6)", async () => {
		/* The D4 wrapper took the pair spacing over from the card's own
		   ``flex … gap-2``, so the title and body sat flush at 0 px where the
		   app's equivalent pair (``AgentUnavailable`` → ``InlineState``) is 8 px.
		   Layout is not measurable in happy-dom, so this pins the utility the
		   designer specified rather than the rendered geometry; the two branches
		   are one component and must not drift apart on it. */
		const { detail, projection } = fixture();
		const getSubagentHistory = vi.mocked(api.getSubagentHistory);
		getSubagentHistory.mockReset();
		getSubagentHistory.mockRejectedValue(
			new api.HttpError(404, "subagent history unavailable"),
		);
		const { unmount } = render(
			<AgentConversation
				sessionId="root"
				jobId="current"
				projection={projection}
				connected
				detail={{ ...detail, status: "completed" as const, transcript: [] }}
			/>,
		);
		await waitFor(() => expect(screen.getByText("No transcript for this agent.")).toBeTruthy());
		const terminal = screen.getByRole("alert");
		expect(terminal.classList.contains("gap-2")).toBe(true);
		expect(terminal.className).toContain("flex-col");
		unmount();
		// The transient branch takes the same class: same title/body pair rhythm.
		getSubagentHistory.mockReset();
		getSubagentHistory.mockRejectedValue(new Error("transient drop"));
		render(
			<AgentConversation
				sessionId="root"
				jobId="current"
				projection={projection}
				connected
				detail={{ ...detail, status: "completed" as const, transcript: [], prompt: "" }}
			/>,
		);
		await waitFor(() => expect(screen.getByText("Couldn't load the transcript.")).toBeTruthy());
		expect(screen.getByRole("alert").classList.contains("gap-2")).toBe(true);
	});

	it("keeps a running child's sheet live by polling its transcript", async () => {
		/* status===running means the sheet must re-pull the transcript on an
		   interval so an open sheet stays real-time; a settled child fetches once. */
		vi.useFakeTimers();
		try {
			const { detail, projection } = fixture();
			const running = { ...detail, status: "running" as const, transcript: [], prompt: "", launch_message_id: "" };
			const getSubagentHistory = vi.mocked(api.getSubagentHistory);
			getSubagentHistory.mockResolvedValue({ entries: [], has_more: false });
			getSubagentHistory.mockClear();
			const { unmount } = render(
				<AgentConversation sessionId="root" jobId="current" projection={projection} connected detail={running} />,
			);
			await vi.waitFor(() => expect(getSubagentHistory).toHaveBeenCalledTimes(1));
			await vi.advanceTimersByTimeAsync(1600);
			expect(getSubagentHistory.mock.calls.length).toBeGreaterThanOrEqual(2);
			/* Unmount must cancel the interval: no further fetches after teardown. */
			const afterUnmount = getSubagentHistory.mock.calls.length;
			unmount();
			await vi.advanceTimersByTimeAsync(3200);
			expect(getSubagentHistory.mock.calls.length).toBe(afterUnmount);
		} finally {
			vi.useRealTimers();
		}
	});
});
