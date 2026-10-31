// @vitest-environment happy-dom
//
// The agent's picture in the session header (core `agent_avatar` contract): LEFT
// of the name, drawn from the LIST row's agent_id/avatar_sha256 (the projection
// does not carry it), and — because the phone has no editor in this scope — a
// tap or long-press shows the translated "change it on the desktop" hint. A
// session without a picture keeps the header exactly as it was.
//
// Harness mirrors session-view.status-dot.test.tsx (real SessionScreen, stubbed
// api/store); the list row is supplied through a mocked useSessions.
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { useState } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SessionScreen } from "./screens/session-view";
import type { SessionProjection } from "./types";


const mocks = vi.hoisted(() => ({
	resumeSession: vi.fn(async () => ({ ok: true, pid: 42, session_id: "s1" })),
	sendCommand: vi.fn(async (_sessionId: string, _op: Record<string, unknown>) => ({ ok: true, detail: "" })),
	navigate: vi.fn(),
}));

vi.mock("./router", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./router")>();
	return { ...actual, navigate: mocks.navigate };
});

vi.mock("./api", async () => {
	/* The REAL refusal class so `resumeRefusalText`'s `instanceof` behaves as in
	   production; every function is a stub (a spread would hand unlisted callers
	   a live fetch). */
	const { HttpError } = await vi.importActual<typeof import("./api")>("./api");
	return {
		HttpError,
		getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
		getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
		getSubagentDetail: vi.fn(async () => null),
		imageUrl: vi.fn(() => ""),
		getCommands: vi.fn(async () => ({ commands: [] })),
		getSkills: vi.fn(async () => ({ skills: [] })),
		getModels: vi.fn(async () => ({ models: [] })),
		getDirectories: vi.fn(async () => ({ home: "/Users/tester", recent: [], tmp: "" })),
		changeDirectory: vi.fn(async () => ({ ok: true, pid: 1, session_id: "s1" })),
		sendCommand: mocks.sendCommand,
		markSessionSeen: vi.fn(async () => ({ ok: true })),
		setSessionPin: vi.fn(async () => ({ ok: true })),
		resumeSession: mocks.resumeSession,
	};
});

let rows: Array<Record<string, unknown>> = [];
let slot: { projection: SessionProjection | null; connected: boolean } = { projection: null, connected: true };
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useProjection: vi.fn(() => slot),
		retainProjectionStream: vi.fn(() => () => {}),
		retainSessionListStream: vi.fn(() => () => {}),
		/* Real draft state, so a send can actually be driven. */
		useDraft: () => useState(""),
		useSessions: () => ({ sessions: rows, connected: true, revision: 0 }),
	};
});

function projection(over: Partial<SessionProjection> = {}): SessionProjection {
	return {
		session_id: "s1",
		pid: 4242,
		kind: "tui",
		conversation_name: "Toronto mayoral candidates",
		cwd: "",
		model_label: "",
		model_selector: "",
		effort: "",
		effort_ladder: [],
		streaming: false,
		activity: "",
		activity_started_s: 0,
		stop_reason: "",
		queued_count: 0,
		ended: false,
		degraded: false,
		transcript: [{ id: "a1", kind: "assistant", text: "Polls are open until 8 p.m." } as SessionProjection["transcript"][number]],
		todos: [],
		subagents: [],
		pending: null,
		pending_count: 0,
		usage: {},
		version: 1,
		...over,
	} satisfies SessionProjection;
}


afterEach(() => {
	cleanup();
	rows = [];
	slot = { projection: null, connected: true };
});

describe("agent picture in the session header", () => {
	it("draws the picture left of the name and explains where to change it", async () => {
		slot = { projection: projection(), connected: true };
		rows = [{ session_id: "s1", agent_id: "agent-9", avatar_sha256: "cafe" }];
		render(<SessionScreen sessionId="s1" />);
		/* The control's NAME is the hint it shows (never "Change <whole title>'s
		   picture": it takes no such action), and the image is decorative (alt="")
		   so the title is not read twice. 28px, the native app's size. */
		const button = screen.getByRole("button", { name: "Change this picture in the desktop app" });
		const img = button.querySelector("img")!;
		expect(img.getAttribute("src")).toBe("/v1/agents/agent-9/avatar?v=cafe");
		expect(img.getAttribute("alt")).toBe("");
		expect(img.getAttribute("width")).toBe("28");
		expect(img.className).toContain("ring-control");
		// LEFT of the name: the picture's control precedes the title in the header.
		const header = button.closest("header")!;
		const title = within(header).getByText("Toronto mayoral candidates");
		expect(button.compareDocumentPosition(title) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
		expect(screen.queryByText("Change this picture in the desktop app")).toBeNull();
		fireEvent.click(button);
		expect(await screen.findByText("Change this picture in the desktop app")).toBeTruthy();
	});

	it("long-press (context menu) shows the same hint", async () => {
		slot = { projection: projection(), connected: true };
		rows = [{ session_id: "s1", agent_id: "agent-9", avatar_sha256: "cafe" }];
		render(<SessionScreen sessionId="s1" />);
		fireEvent.contextMenu(screen.getByRole("button", { name: /desktop app/ }));
		expect(await screen.findByText("Change this picture in the desktop app")).toBeTruthy();
	});

	it("draws nothing extra for a session without a picture", () => {
		slot = { projection: projection(), connected: true };
		rows = [{ session_id: "s1", agent_id: "agent-9", avatar_sha256: null }];
		render(<SessionScreen sessionId="s1" />);
		expect(screen.queryByRole("button", { name: /desktop app/ })).toBeNull();
		expect(document.querySelector("header img")).toBeNull();
	});

	it("falls back to EXACTLY the no-picture header when the picture 404s (D1)", () => {
		slot = { projection: projection(), connected: true };
		rows = [{ session_id: "s1", agent_id: "agent-9", avatar_sha256: "cafe" }];
		const { container } = render(<SessionScreen sessionId="s1" />);
		const headerHtml = () => container.querySelector("header")!.innerHTML;
		const img = document.querySelector("header img")!;
		expect(img).not.toBeNull();
		fireEvent.error(img);
		// No image, no (invisible) 44px button that still opens a hint...
		expect(document.querySelector("header img")).toBeNull();
		expect(screen.queryByRole("button", { name: /desktop app/ })).toBeNull();
		// ...and the header is byte-identical to one for a session with no picture.
		const failed = headerHtml();
		cleanup();
		rows = [{ session_id: "s1", agent_id: "agent-9", avatar_sha256: null }];
		const { container: plain } = render(<SessionScreen sessionId="s1" />);
		expect(failed).toBe(plain.querySelector("header")!.innerHTML);
	});
});
