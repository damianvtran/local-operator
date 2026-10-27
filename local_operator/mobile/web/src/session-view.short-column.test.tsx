// @vitest-environment happy-dom
import { act, cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SessionScreen } from "./screens/session-view";
import type { SessionProjection } from "./types";

let slot: { projection: SessionProjection | null; connected: boolean } = { projection: null, connected: true };
vi.mock("./router", () => ({ navigate: vi.fn() }));
vi.mock("./api", async () => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentDetail: vi.fn(async () => null),
	imageUrl: vi.fn(() => ""),
	getCommands: vi.fn(async () => ({ commands: [] })),
	getModels: vi.fn(async () => ({ models: [] })),
	getDirectories: vi.fn(async () => ({ home: "/", recent: [], tmp: "" })),
	changeDirectory: vi.fn(async () => ({ ok: true })),
	sendCommand: vi.fn(async () => ({ ok: true })),
	markSessionSeen: vi.fn(async () => ({ ok: true })),
	setSessionPin: vi.fn(async () => ({ ok: true })),
	resumeSession: vi.fn(async () => ({ ok: true })),
}));
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return { ...actual, useProjection: vi.fn(() => slot), retainProjectionStream: vi.fn(() => () => {}), retainSessionListStream: vi.fn(() => () => {}), useDraft: vi.fn(() => ["", () => {}]) };
});
function projection(): SessionProjection {
	return { session_id: "s1", pid: 42, kind: "tui", conversation_name: "test", cwd: "", model_label: "", model_selector: "", effort: "", effort_ladder: [], streaming: false, activity: "", activity_started_s: 0, stop_reason: "", queued_count: 0, ended: false, degraded: false, transcript: [], todos: [], subagents: [], pending: null, pending_count: 0, usage: {}, version: 1 } as SessionProjection;
}

describe("the short layout follows the VISUAL viewport (review R1-1)", () => {
	afterEach(() => { cleanup(); vi.unstubAllGlobals(); });
	it("sets data-short from visualViewport.height even when innerHeight is large", () => {
		const vv = new EventTarget() as EventTarget & { height: number; offsetTop: number };
		vv.height = 844; vv.offsetTop = 0;
		vi.stubGlobal("visualViewport", vv);
		Object.defineProperty(window, "innerHeight", { configurable: true, value: 844 });
		slot = { projection: projection(), connected: true };
		const view = render(<SessionScreen sessionId="s1" />);
		const column = view.container.firstElementChild as HTMLElement;
		expect(column.hasAttribute("data-short")).toBe(false);
		act(() => { vv.height = 440; vv.dispatchEvent(new Event("resize")); });
		expect(column.hasAttribute("data-short")).toBe(true);
		expect(window.innerHeight).toBe(844);
		act(() => { vv.height = 700; vv.dispatchEvent(new Event("resize")); });
		expect(column.hasAttribute("data-short")).toBe(false);
	});
});
