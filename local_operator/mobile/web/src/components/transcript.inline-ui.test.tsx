// @vitest-environment happy-dom
import { act, cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { sendCommand } from "../api";
import { readInlineUiMessages } from "../inline-ui-parts";
import { SessionScreen } from "../screens/session-view";
import { FetchStreamFixture as Stream, settleStream } from "../sse-test-fixture";
import type { SessionProjection, TranscriptEntry } from "../types";
import { Transcript } from "./transcript";

const entry = (text: string, extra: Partial<TranscriptEntry> = {}): TranscriptEntry => ({
	id: "answer", kind: "assistant", text, tool_call_id: "", tool_name: "", tool_state: "done",
	summary: "", intent: "", diff_added: 0, diff_removed: 0, elapsed_s: 0, error: "",
	details: {}, final: true, text_complete: true, ...extra,
});
const transcript = (text: string, extra?: Partial<TranscriptEntry>, wire?: unknown) => {
	const rows = [entry(text, extra)];
	return <Transcript pid="s" entries={rows} inlineUi={readInlineUiMessages(wire, rows)} />;
};
const fence = (payload: unknown) => `\`\`\`lop-ui\n${JSON.stringify(payload)}\n\`\`\``;
const bar = { type: "bar", title: "Net change", unit: "USD", data: [["Q1", -3], ["Q2", 9]] };
const overlayText = "**📈 Before**\n\n*After*";
const overlay = (widgets: unknown = [{ offset: overlayText.indexOf("*After*"), payload: bar }]) => ({
	version: 1, messages: { answer: { version: 1, text: overlayText, widgets } },
});

afterEach(cleanup);

describe("mobile transcript inline UI", () => {
	it("mounts signed bars between surrounding Markdown in the completed assistant row", () => {
		render(transcript("Legacy prose", undefined, overlay()));
		const table = screen.getByRole("table", { name: "Net change (USD)" });
		const before = screen.getByText("📈 Before");
		const after = screen.getByText("After");
		expect(before.tagName).toBe("STRONG");
		expect(after.tagName).toBe("EM");
		expect(before.compareDocumentPosition(table) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
		expect(table.compareDocumentPosition(after) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
		expect(within(table).getByRole("rowheader", { name: "Q1" })).toBeTruthy();
		const negative = within(table).getByRole("cell", { name: "-3" });
		const positive = within(table).getByRole("cell", { name: "9" });
		expect(negative.querySelector("[aria-hidden] span")?.getAttribute("style")).toContain("width: 25%");
		expect(positive.querySelector("[aria-hidden] span")?.getAttribute("style")).toContain("left: 25%");
	});

	it.each([
		["line", { data: [["Q1", 19]] }, "19"],
		["pie", { data: [["Q1", 19]] }, "19"],
		["histogram", { data: [["0–5", 19]] }, "19"],
		["scatter", { xLabel: "Time", yLabel: "Count", data: [[1, 19]] }, "19"],
		["heatmap", { columns: ["Mon", "Tue"], rowLabels: ["Night"], rows: [[19, null]] }, "19"],
		["dual", { mode: "dual", unit: "USD", rightUnit: "kg", labels: ["Q1"], series: [
			{ axis: "left", label: "Cost", values: [1] }, { axis: "right", label: "Mass", values: [19] },
		] }, "19"],
		["grouped", { mode: "grouped", labels: ["Q1", "Q2"], series: [{ label: "A", values: [-19, 3] }] }, "-19"],
		["stacked", { mode: "stacked", labels: ["Q1"], series: [{ label: "A", values: [19] }, { label: "B", values: [-3] }] }, "19"],
		["percent", { mode: "percent", labels: ["Q1"], series: [{ label: "A", values: [19] }, { label: "B", values: [3] }] }, "19"],
		["table", { columns: ["A"], rows: [["<img src=x onerror=alert(1)>"]] }, "<img src=x onerror=alert(1)>"],
		["spreadsheet", { columns: ["A", "B"], rows: [["=2+2", null]] }, "=2+2"],
	] as const)("renders %s with a literal, named value grid", (type, fields, expected) => {
		const payload = { ...fields, type: "mode" in fields ? "series" : type, title: "Preview" };
		const { container } = render(transcript("Legacy prose", undefined, overlay([{ offset: 0, payload }])));
		const table = screen.getByRole("table", { name: "Preview" });
		expect(within(table).getByRole("cell", { name: expected })).toBeTruthy();
		expect(container.querySelector("img")).toBeNull();
		if (type === "dual") expect(within(table).getByRole("columnheader", { name: "Mass (kg)" })).toBeTruthy();
		if (["line", "scatter", "pie", "grouped", "stacked", "percent", "dual"].includes(type)) {
			const graphic = container.querySelector(`[data-inline-plot="${type}"]`);
			expect(graphic?.childElementCount).toBeGreaterThan(0);
			expect(graphic?.outerHTML).not.toMatch(/NaN|Infinity/);
		}
		if (type === "heatmap") {
			expect(within(table).getByRole("rowheader", { name: "Night" })).toBeTruthy();
			expect(table.querySelectorAll("td [aria-hidden]")).toHaveLength(1);
			expect(within(table).getByRole("row", { name: /Night/ }).lastElementChild?.textContent).toBe("");
		}
	});

	it("does not extract widgets from standalone or nested legacy literal fences", () => {
		const { container, rerender } = render(transcript(fence(bar)));
		expect(screen.queryByRole("table")).toBeNull();
		expect(container.textContent).toContain("Net change");
		rerender(transcript(`\`\`\`\`text\n${fence(bar)}\n\`\`\`\``));
		expect(screen.queryByRole("table")).toBeNull();
		expect(container.textContent).toContain("Net change");
	});

	it.each(["user", "steer", "parent_message"] as const)("does not interpret %s-authored fences", (kind) => {
		const { container } = render(transcript(fence(bar), { kind }));
		expect(screen.queryByRole("table")).toBeNull();
		expect(container.textContent).toContain("Net change");
	});

	it.each([
		[{ offset: -1, payload: bar }],
		[{ offset: overlayText.length + 1, payload: bar }],
		[{ offset: 2.5, payload: bar }],
		[{ offset: 3, payload: bar }], // Splits the emoji's UTF-16 surrogate pair.
		[{ offset: 8, payload: bar }, { offset: 4, payload: bar }],
		[{ offset: 0, payload: { ...bar, data: [["Q1", "not a number"]] } }],
	])("rejects invalid placed widgets: %j", (...placements) => {
		expect(readInlineUiMessages(overlay(placements), [entry("Fallback")])).toBeUndefined();
	});

	it("keeps zero, duplicate and end offsets without losing intervening prose", () => {
		const decoded = readInlineUiMessages(overlay([
			{ offset: 0, payload: bar },
			{ offset: 0, payload: bar },
			{ offset: overlayText.length, payload: bar },
		]), [entry("Legacy")]);
		expect(decoded?.get("answer")).toEqual([
			{ kind: "ui", value: bar }, { kind: "ui", value: bar },
			{ kind: "markdown", text: overlayText }, { kind: "ui", value: bar },
		]);
	});

	it("ignores unknown versions and rows outside the live assistant transcript", () => {
		const live = [entry("Legacy")];
		const wire = overlay();
		expect(readInlineUiMessages({ ...wire, version: 2 }, live)).toBeUndefined();
		expect(readInlineUiMessages({ ...wire, messages: { answer: { ...wire.messages.answer, version: 2 } } }, live)).toBeUndefined();
		expect(readInlineUiMessages(wire, [entry("A user", { kind: "user" })])).toBeUndefined();
		expect(readInlineUiMessages(wire, [entry("Other assistant", { id: "other" })])).toBeUndefined();
	});

});

describe("live session inline UI negotiation", () => {
	const sessionId = "inline-stream";
	const path = `/api/sessions/${sessionId}/events`;
	const legacyBar = { ...bar, title: "Legacy preview" };
	const fetchJson = vi.fn<(url: string, init?: RequestInit) => Promise<Response>>();
	const projection: SessionProjection = {
		session_id: sessionId, pid: 42, kind: "tui", conversation_name: "Inline UI session",
		cwd: "", model_label: "", model_selector: "", effort: "", effort_ladder: [],
		streaming: false, activity: "", activity_started_s: null, stop_reason: "", queued_count: 0,
		ended: false, degraded: false, transcript: [entry(`Legacy prose\n\n${fence(legacyBar)}`)],
		todos: [], subagents: [], pending: null, pending_count: 0, usage: {}, version: 1,
	};
	beforeEach(() => {
		vi.useFakeTimers();
		Stream.install();
		fetchJson.mockReset().mockImplementation(async () => Response.json({}));
		vi.stubGlobal("fetch", (url: string, init?: RequestInit) =>
			new Headers(init?.headers).get("Accept") === "text/event-stream"
				? Stream.fetch(url, init) : fetchJson(url, init),
		);
	});
	afterEach(async () => {
		cleanup();
		await vi.runOnlyPendingTimersAsync();
		vi.useRealTimers();
		vi.unstubAllGlobals();
	});

	async function openSession(expectedPath: string) {
		render(<SessionScreen sessionId={sessionId} />);
		await act(settleStream);
		const stream = Stream.instances.find(({ url }) => url.startsWith(path));
		expect(stream?.url).toBe(expectedPath);
		return (frame: unknown) => act(async () => {
			stream!.emit("projection", frame);
			await settleStream();
		});
	}

	it("renders exact overlay text through SSE and returns to source on a full repaint without it", async () => {
		const emit = await openSession(`${path}?steer_journal_order=1&inline_ui=1`);
		await emit({ ...projection, inline_ui: overlay() });
		const before = screen.getByText("📈 Before");
		const after = screen.getByText("After");
		const table = screen.getByRole("table", { name: "Net change (USD)" });
		expect(before.tagName).toBe("STRONG");
		expect(after.tagName).toBe("EM");
		expect(before.compareDocumentPosition(table) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
		expect(table.compareDocumentPosition(after) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
		expect(screen.queryByText("Legacy prose")).toBeNull();

		// The relay can change just the overlay without bumping projection.version.
		await emit({ ...projection, inline_ui: overlay([{ offset: -1, payload: bar }]) });
		expect(screen.queryByRole("table", { name: "Net change (USD)" })).toBeNull();
		expect(screen.queryByRole("table", { name: "Legacy preview (USD)" })).toBeNull();
		await emit({ ...projection, inline_ui: overlay() });
		expect(screen.getByRole("table", { name: "Net change (USD)" })).toBeTruthy();
		await emit(projection);
		expect(screen.queryByRole("table", { name: "Net change (USD)" })).toBeNull();
		expect(screen.getByText("Legacy prose")).toBeTruthy();
		expect(screen.queryByRole("table", { name: "Legacy preview (USD)" })).toBeNull();
	});

	it("negotiates older pages without replacing or reviving the live overlay", async () => {
		const historyPath = `/api/sessions/${sessionId}/history`;
		const historicalAnswer = entry("Legacy history", { id: "older-answer" });
		const historicalOverlay = { version: 1, text: "Historical prose", widgets: [
			{ offset: 0, payload: { ...bar, title: "Historical preview" } },
		] };
		fetchJson.mockImplementation(async (url) => Response.json(url.startsWith(historyPath) ? {
			entries: [historicalAnswer, entry("Duplicate live answer")], has_more: false,
			inline_ui: { version: 1, messages: { "older-answer": historicalOverlay, answer: historicalOverlay } },
		} : {}));
		const emit = await openSession(`${path}?steer_journal_order=1&inline_ui=1`);
		const pinned = { ...projection, transcript_head_pinned: true,
			transcript: [entry("Opener", { kind: "user", id: "opener" }), ...projection.transcript] };
		await emit({ ...pinned, inline_ui: overlay() });
		await act(settleStream);
		const request = fetchJson.mock.calls.find(([url]) => url.startsWith(historyPath))?.[0];
		expect(request).toContain("before=answer");
		expect(request).toContain("inline_ui=1");
		expect(screen.getByRole("table", { name: "Historical preview (USD)" })).toBeTruthy();
		expect(screen.getByText("Historical prose")).toBeTruthy();
		expect(screen.getByRole("table", { name: "Net change (USD)" })).toBeTruthy();
		await emit(pinned);
		expect(screen.getByRole("table", { name: "Historical preview (USD)" })).toBeTruthy();
		expect(screen.queryByRole("table", { name: "Legacy preview (USD)" })).toBeNull();
		expect(screen.queryByRole("table", { name: "Net change (USD)" })).toBeNull();
	});

	it("shows server-projected live prose with no widget before a typed overlay arrives", async () => {
		const emit = await openSession(`${path}?steer_journal_order=1&inline_ui=1`);
		await emit({ ...projection, streaming: true, transcript: [entry("Partial answer", { final: false })] });
		expect(screen.getByText("Partial answer")).toBeTruthy();
		expect(screen.queryByRole("table")).toBeNull();
		await emit({ ...projection, inline_ui: overlay() });
		expect(screen.getByRole("table", { name: "Net change (USD)" })).toBeTruthy();
	});

	it("sends the v1 declaration on prompt and steer but not unrelated commands", async () => {
		await sendCommand(sessionId, { op: "prompt", text: "Chart please", command_id: "test" });
		await sendCommand(sessionId, { op: "steer", text: "Try bars", command_id: "test2" });
		await sendCommand(sessionId, { op: "abort" });
		expect(fetchJson.mock.calls.map(([url, init]) => [url, JSON.parse(String(init?.body))])).toEqual([
			[`/api/sessions/${sessionId}/command`, { op: "prompt", text: "Chart please", command_id: "test", inline_ui: true }],
			[`/api/sessions/${sessionId}/command`, { op: "steer", text: "Try bars", command_id: "test2", inline_ui: true }],
			[`/api/sessions/${sessionId}/command`, { op: "abort" }],
		]);
	});
});
