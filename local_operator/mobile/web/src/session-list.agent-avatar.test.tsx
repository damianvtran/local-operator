// @vitest-environment happy-dom
//
// Agent profile pictures on the phone's list rows (core `agent_avatar` contract).
// A row whose session is bound to an agent with a picture draws it from the
// row's own `agent_id`/`avatar_sha256` — no fetch to learn it exists — and the
// status ladder's mark is KEPT as a corner badge: status must stay legible.
// A row without both fields renders exactly the slot it always did.
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { agentAvatarUrl } from "./components/agent-avatar";
import { SessionListScreen } from "./screens/session-list";
import type { SessionSummary } from "./types";

let sessionList: SessionSummary[] = [];
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useSessions: () => ({ sessions: sessionList, connected: true, revision: 0 }),
		retainSessionListStream: () => () => {},
	};
});
vi.mock("./api", () => ({
	getDirectories: vi.fn(async () => ({ home: "", recent: [] })),
}));

function summary(over: Partial<SessionSummary>): SessionSummary {
	return {
		session_id: "s",
		section: "active",
		conversation_name: "Session",
		cwd: "",
		model_label: "",
		streaming: false,
		needs_attention: false,
		unseen: false,
		pending_kind: "",
		subagents_running: 0,
		subagents_queued: 0,
		todos_open: 0,
		mtime: 0,
		...over,
	};
}

const card = (name: string) => screen.getByRole("button", { name: new RegExp(name) });

afterEach(() => {
	cleanup();
	sessionList = [];
});

describe("agent picture on list rows", () => {
	it("builds the content-addressed relay URL only when both fields are present", () => {
		expect(agentAvatarUrl("a1", "abc")).toBe("/v1/agents/a1/avatar?v=abc");
		expect(agentAvatarUrl("a1", null)).toBeNull();
		expect(agentAvatarUrl(null, "abc")).toBeNull();
		expect(agentAvatarUrl(undefined, undefined)).toBeNull();
	});

	it("draws the picture and keeps the streaming spinner as a corner badge", () => {
		sessionList = [
			summary({
				session_id: "s1",
				conversation_name: "Pictured",
				agent_id: "agent-1",
				avatar_sha256: "deadbeef",
				streaming: true,
			}),
		];
		render(<SessionListScreen />);
		const row = card("Pictured");
		const img = row.querySelector("img[data-agent-avatar]") as HTMLImageElement;
		expect(img.getAttribute("src")).toBe("/v1/agents/agent-1/avatar?v=deadbeef");
		expect(img.getAttribute("alt")).toBe("Pictured's picture");
		const badge = row.querySelector("[data-avatar-status=badge]")!;
		// The ladder's slot (and its spinner) is still in the row, inside the badge.
		expect(badge.querySelector(".size-3")).not.toBeNull();
		expect(badge.querySelector('[role="status"]')).not.toBeNull();
	});

	it("keeps the danger pulse visible for a decision on a pictured row", () => {
		sessionList = [
			summary({
				session_id: "s2",
				conversation_name: "Gatebot",
				agent_id: "agent-1",
				avatar_sha256: "deadbeef",
				needs_attention: true,
				pending_kind: "approval",
			}),
		];
		render(<SessionListScreen />);
		const badge = card("Gatebot").querySelector("[data-avatar-status=badge]")!;
		expect(badge.querySelector(".lo-pulse.bg-danger")).not.toBeNull();
	});

	it("draws no badge disc for an idle pictured row, and no picture without a sha", () => {
		sessionList = [
			summary({ session_id: "s3", conversation_name: "Idle", agent_id: "a", avatar_sha256: "f" }),
			summary({ session_id: "s4", conversation_name: "Glyph", agent_id: "a", avatar_sha256: null }),
		];
		render(<SessionListScreen />);
		const idle = card("Idle");
		expect(idle.querySelector("img[data-agent-avatar]")).not.toBeNull();
		expect(idle.querySelector("[data-avatar-status=badge] .bg-canvas")).toBeNull();
		const glyph = card("Glyph");
		expect(glyph.querySelector("img")).toBeNull();
		expect(glyph.querySelector("[data-avatar-status]")).toBeNull();
		expect(glyph.querySelector("div")!.querySelector(".size-3")).not.toBeNull();
	});
});
