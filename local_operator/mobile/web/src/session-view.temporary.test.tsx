// @vitest-environment happy-dom
//
// The session screen's temporary-chat surfaces (contract §7.1 item 12), against
// the REAL SessionScreen with a faked relay:
//
//   * the dashed toggle sits LEFT OF ⋮ on an empty session only, and is absent
//     without `features.temporary_chats` (fail closed);
//   * a temporary chat's subtitle reads `Temporary · <status>` behind the mark;
//   * the ⋮ row keeps at once, and asks before making a chat with messages
//     temporary;
//   * a deleted temporary chat (404 `session_temporary_deleted`) shows the
//     tombstone, and "New temporary session" carries the unsent draft over.
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SessionScreen } from "./screens/session-view";
import type { Capabilities, SessionProjection, SessionSummary, TemporaryInfo, TranscriptEntry } from "./types";

const setSessionTemporary = vi.fn(async (id: string, on: boolean) => ({ session_id: id, temporary: on ? MARK : null }));
const startSession = vi.fn(async (_input?: unknown) => ({ ok: true, pid: 2, session_id: "fresh1" }));
vi.mock("./api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./api")>();
	return {
		HttpError: actual.HttpError,
		getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
		getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
		getSubagentDetail: vi.fn(async () => null),
		imageUrl: vi.fn(() => ""),
		getCommands: vi.fn(async () => ({ commands: [] })),
		getSkills: vi.fn(async () => ({ skills: [] })),
		getModels: vi.fn(async () => ({ models: [] })),
		getDirectories: vi.fn(async () => ({ home: "/Users/tester", recent: [], tmp: "" })),
		changeDirectory: vi.fn(async () => ({ ok: true, pid: 1, session_id: "s1" })),
		sendCommand: vi.fn(async () => ({ ok: true, detail: "" })),
		markSessionSeen: vi.fn(async () => ({ ok: true })),
		setSessionPin: vi.fn(async () => ({ ok: true, pinned: true })),
		setSessionTemporary: (id: string, on: boolean) => setSessionTemporary(id, on),
		startSession: (input?: unknown) => startSession(input),
	};
});

const MARK: TemporaryInfo = { marked_at: 1_800_000_000, idle_minutes: 30, deletes_at: 1_800_001_800, hold: null };
let slot: { projection: SessionProjection | null; connected: boolean } = { projection: null, connected: true };
let caps: Capabilities | null = null;
let rows: SessionSummary[] = [];
let tombstone: number | null | undefined = undefined;
const drafts = new Map<string, string>();
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useProjection: vi.fn(() => slot),
		retainProjectionStream: vi.fn(() => () => {}),
		retainSessionListStream: vi.fn(() => () => {}),
		useDraft: vi.fn(() => ["", () => {}]),
		useCapabilities: () => caps,
		useSessions: () => ({ sessions: rows, connected: true, revision: 0 }),
		useTemporaryTombstone: () => tombstone,
		getDraft: (id: string) => drafts.get(id) ?? "",
		setDraft: (id: string, text: string) => { drafts.set(id, text); },
	};
});

function projection(over: Partial<SessionProjection> = {}): SessionProjection {
	return {
		session_id: "s1", pid: 1, kind: "tui", conversation_name: "Quick regex question", cwd: "",
		model_label: "", model_selector: "", effort: "", effort_ladder: [], streaming: false, activity: "",
		activity_started_s: 0, stop_reason: "", queued_count: 0, ended: false, degraded: false,
		transcript: [], todos: [], subagents: [], pending: null, pending_count: 0, usage: {}, version: 1,
		...over,
	} satisfies SessionProjection;
}
function row(over: Partial<SessionSummary> = {}): SessionSummary {
	return {
		session_id: "s1", section: "active", conversation_name: "Quick regex question", cwd: "", model_label: "",
		streaming: false, needs_attention: false, unseen: false, pending_kind: "", subagents_running: 0,
		subagents_queued: 0, todos_open: 0, mtime: 0, pinned: false, temporary: null, ...over,
	};
}
const MESSAGE = { kind: "user", id: "u1", text: "regex for an ISO date?" } as unknown as TranscriptEntry;

function toggle(): HTMLElement | null {
	return document.querySelector("[data-temporary-toggle]");
}

beforeEach(() => {
	caps = { features: { temporary_chats: 1 } };
	rows = [row()];
	slot = { projection: projection(), connected: true };
	tombstone = undefined;
	drafts.clear();
	setSessionTemporary.mockClear();
	startSession.mockClear();
	location.hash = "#/s/s1";
});
afterEach(() => cleanup());

describe("header toggle", () => {
	it("sits left of ⋮ on an empty session and marks it temporary without asking", async () => {
		render(<SessionScreen sessionId="s1" />);
		const button = toggle();
		expect(button).not.toBeNull();
		expect(button?.getAttribute("aria-label")).toBe("Temporary session");
		expect(button?.getAttribute("aria-pressed")).toBe("false");
		expect(button?.className).not.toContain("bg-accent-wash");
		expect(button?.nextElementSibling?.getAttribute("aria-label")).toBe("session menu");
		await act(async () => { fireEvent.click(button as HTMLElement); });
		expect(setSessionTemporary).toHaveBeenCalledWith("s1", true);
		await waitFor(() => expect(toggle()?.getAttribute("aria-pressed")).toBe("true"));
		// ON is the relay's pressed pair (wash ground + accent ink), not hue
		// alone (design D4): the ground is what tells ON from OFF in the 18 of
		// 31 palettes where accent and muted ink are within 1.2:1.
		expect(toggle()?.className).toContain("bg-accent-wash");
		expect(toggle()?.className).toContain("text-accent");
	});

	it("is gone once the chat has a message", () => {
		slot = { projection: projection({ transcript: [MESSAGE] }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		expect(toggle()).toBeNull();
	});

	it("fails closed without the capability: no toggle, no subtitle mark, no menu row", () => {
		caps = { features: {} };
		rows = [row({ temporary: MARK })];
		render(<SessionScreen sessionId="s1" />);
		expect(toggle()).toBeNull();
		expect(document.querySelector("[data-temporary]")).toBeNull();
		fireEvent.click(screen.getByRole("button", { name: "session menu" }));
		expect(document.querySelector('[data-menu="temporary"]')).toBeNull();
	});
});

describe("subtitle", () => {
	it("reads Temporary · <status> behind the dashed mark", () => {
		rows = [row({ temporary: MARK })];
		render(<SessionScreen sessionId="s1" />);
		const line = document.querySelector("[data-temporary]") as HTMLElement;
		expect(line.textContent).toBe("Temporary · Idle");
		expect(line.querySelector("svg path")?.getAttribute("d")).toMatch(/^M10 17H7l-4 4v-7/);
	});

	it("follows the events frame's explicit null over a stale list row", () => {
		rows = [row({ temporary: MARK })];
		slot = { projection: projection({ temporary: null }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		expect(document.querySelector("[data-temporary]")).toBeNull();
	});
});

describe("⋮ menu row", () => {
	it("keeps a temporary chat at once", async () => {
		rows = [row({ temporary: MARK })];
		slot = { projection: projection({ transcript: [MESSAGE] }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		fireEvent.click(screen.getByRole("button", { name: "session menu" }));
		const item = document.querySelector('[data-menu="temporary"]') as HTMLElement;
		expect(item.textContent).toContain("Keep session");
		// An action row, not a toggle: "Keep session, pressed" contradicted itself (D7).
		expect(item.hasAttribute("aria-pressed")).toBe(false);
		await act(async () => { fireEvent.click(item); });
		expect(setSessionTemporary).toHaveBeenCalledWith("s1", false);
	});

	it("asks before making a chat with messages temporary, and Cancel sends nothing", async () => {
		slot = { projection: projection({ transcript: [MESSAGE] }), connected: true };
		render(<SessionScreen sessionId="s1" />);
		fireEvent.click(screen.getByRole("button", { name: "session menu" }));
		fireEvent.click(document.querySelector('[data-menu="temporary"]') as HTMLElement);
		expect(setSessionTemporary).not.toHaveBeenCalled();
		const confirm = document.querySelector('[data-menu="temporary-confirm"]') as HTMLElement;
		expect(screen.getByText("Make this session temporary?")).toBeTruthy();
		expect(confirm.textContent).toContain("Files it changed on your computer are kept.");
		fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
		expect(setSessionTemporary).not.toHaveBeenCalled();
		fireEvent.click(document.querySelector('[data-menu="temporary"]') as HTMLElement);
		// The commit button has its own label, without the row's ellipsis (D6).
		expect(screen.queryByRole("button", { name: "Make temporary…" })).toBeNull();
		await act(async () => { fireEvent.click(screen.getByRole("button", { name: "Make temporary" })); });
		expect(setSessionTemporary).toHaveBeenCalledWith("s1", true);
	});
});

describe("tombstone", () => {
	it("replaces the conversation and carries the unsent draft into a new temporary chat", async () => {
		tombstone = 1_800_002_000;
		drafts.set("s1", "half-written question");
		render(<SessionScreen sessionId="s1" />);
		expect(document.querySelector("[data-temporary-tombstone]")).not.toBeNull();
		expect(screen.getByText("This temporary session was deleted")).toBeTruthy();
		expect(screen.queryByText("Quick regex question")).toBeNull();
		expect(document.querySelector("[data-draft]")?.textContent).toBe("half-written question");
		await act(async () => { fireEvent.click(screen.getByRole("button", { name: "New temporary session" })); });
		expect(startSession).toHaveBeenCalledWith({ temporary: true });
		expect(drafts.get("fresh1")).toBe("half-written question");
		expect(drafts.get("s1")).toBe("");
		expect(location.hash).toBe("#/s/fresh1");
	});

	it("offers only Back to sessions without the capability", () => {
		tombstone = null;
		caps = null;
		render(<SessionScreen sessionId="s1" />);
		expect(screen.queryByRole("button", { name: "New temporary session" })).toBeNull();
		expect(screen.getAllByRole("button", { name: "Back to sessions" }).length).toBeGreaterThan(0);
	});
});
