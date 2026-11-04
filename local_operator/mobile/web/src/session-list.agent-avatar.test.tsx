// @vitest-environment happy-dom
//
// Agent profile pictures on the phone's list rows (core `agent_avatar` contract).
// A row whose session is bound to an agent with a picture draws it from the
// row's own `agent_id`/`avatar_sha256` — no fetch to learn it exists — and the
// status ladder's mark is KEPT as a corner badge: status must stay legible.
// A row without both fields renders exactly the slot it always did.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { agentAvatarUrl } from "./components/agent-avatar";
import { SessionListScreen } from "./screens/session-list";
import type { SessionSummary } from "./types";

let sessionList: SessionSummary[] = [];
/* Bumped when a test swaps the list in place: the screen memoises on it. */
let revision = 0;
/* null = a relay that predates `agent_avatar_sizes` (every pre-existing test). */
let caps: Record<string, unknown> | null = null;
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useSessions: () => ({ sessions: sessionList, connected: true, revision }),
		useCapabilities: () => caps,
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
	caps = null;
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
		// Decorative: the row's button already names the conversation (D2).
		expect(img.getAttribute("alt")).toBe("");
		expect(img.className).toContain("ring-control");
		expect(img.getAttribute("width")).toBe("24");
		const badge = row.querySelector("[data-avatar-status=badge]")!;
		// The ladder's slot (and its spinner) is still in the row, inside the badge.
		expect(badge.querySelector(".size-3")).not.toBeNull();
		expect(badge.querySelector('[role="status"]')).not.toBeNull();
		// 12px mark on a 16px disc: a real 2px ring in the ROW's surface colour.
		const disc = badge.querySelector(".size-4")!;
		expect(disc.className).toContain("bg-canvas");
		expect(disc.className).toContain("group-active:bg-surface");
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

	it.each([
		["idle", {}],
		["running", { streaming: true }],
	])("lays an %s row out as a pictureless one when its picture 404s (D1)", (_label, over) => {
		const base = { session_id: "s5", conversation_name: "Ghosted", cwd: "/w/proj", ...over };
		sessionList = [summary({ ...base, agent_id: "agent-1", avatar_sha256: "gone" })];
		const { container, unmount } = render(<SessionListScreen />);
		const row = card("Ghosted");
		fireEvent.error(row.querySelector("img[data-agent-avatar]")!);
		// No box, no stranded badge, no 48px title indent on the second line.
		expect(row.querySelector("img")).toBeNull();
		expect(row.querySelector("[data-avatar-status]")).toBeNull();
		expect(row.innerHTML).not.toContain("ps-8");
		const failed = container.innerHTML;
		unmount();
		sessionList = [summary({ ...base, agent_id: null, avatar_sha256: null })];
		const { container: plain } = render(<SessionListScreen />);
		expect(failed).toBe(plain.innerHTML);
	});

	it("retries a NEW sha after a 404 (the failure is keyed by URL)", async () => {
		sessionList = [summary({ session_id: "s6", conversation_name: "Retry", agent_id: "a", avatar_sha256: "old" })];
		const { rerender } = render(<SessionListScreen />);
		fireEvent.error(card("Retry").querySelector("img")!);
		expect(card("Retry").querySelector("img")).toBeNull();
		sessionList = [summary({ session_id: "s6", conversation_name: "Retry", agent_id: "a", avatar_sha256: "new" })];
		revision += 1;
		rerender(<SessionListScreen />);
		/* The painted rows are debounced on purpose (`usePaintedRows`), so the new
		   sha lands a beat after the store changes. */
		await waitFor(() =>
			expect(card("Retry").querySelector("img")?.getAttribute("src")).toBe("/v1/agents/a/avatar?v=new"),
		);
	});
});

// happy-dom checks markup only; the browser fixture proves actual ellipsis pixels.

it("keeps a mixed-direction conversation title's own truncation end in RTL chrome", () => {
 sessionList = [summary({ conversation_name: "Identifying prefix — لمبی گفتگو" })];
 const { container } = render(<div dir="rtl"><SessionListScreen /></div>);
 const title = screen.getByText("Identifying prefix — لمبی گفتگو");
 expect(title.getAttribute("dir")).toBe("ltr");
 expect(title.className).toContain("truncate");
 expect(title.closest('[dir="rtl"]')).toBe(container.firstElementChild);
});

// First-strong loses the Urdu beginning after API; neutral titles inherit chrome.
it.each([
 ["API منصوبے کی پیش رفت اور اگلے مرحلے کی تفصیلی گفتگو", "rtl"],
 ["Pure Latin identifying beginning and a long conversation title", "ltr"],
 ["تحقیقاتی منصوبے کی پیش رفت اور اگلے مرحلے کی تفصیلی گفتگو", "rtl"],
 ["2026-11-03 12345678901234567890", null],
])("resolves truncated title %s by dominant script", (name, direction) => {
 sessionList = [summary({ conversation_name: name })];
 render(<div dir="rtl"><SessionListScreen /></div>);
 const title = screen.getByText(name);
 expect(title.getAttribute("dir")).toBe(direction);
 expect(title.className).toContain("truncate");
 expect(title.closest("button")?.hasAttribute("dir")).toBe(false);
});

describe("right-sized avatar variants (core agent_avatar_sizes)", () => {
	it("requests the smallest variant whose edge is >= css px x devicePixelRatio", async () => {
		const { avatarVariant, AVATAR_VARIANTS } = await import("./components/agent-avatar");
		for (const css of [16, 20, 24, 28, 32, 36, 48, 64]) {
			for (const dpr of [1, 1.5, 2, 2.625, 3, 4]) {
				const edge = avatarVariant(css, dpr);
				const needed = css * dpr;
				if (edge === null) {
					// The 512 original: only when even the largest variant is too small.
					expect(needed).toBeGreaterThan(AVATAR_VARIANTS[AVATAR_VARIANTS.length - 1]);
				} else {
					expect(edge).toBeGreaterThanOrEqual(needed);
					// ...and the SMALLEST such variant, so bandwidth stays sensible.
					const smaller = AVATAR_VARIANTS.filter((v) => v < edge);
					for (const v of smaller) expect(v).toBeLessThan(needed);
				}
			}
		}
		expect(avatarVariant(24, 3)).toBe(128);
		expect(avatarVariant(28, 2)).toBe(64);
		expect(avatarVariant(24, Number.NaN)).toBe(64);
	});

	it("a rendered 24px row at 3x asks the relay for the 128 variant", () => {
		caps = { features: { agent_avatar_sizes: 1 } };
		const original = window.devicePixelRatio;
		Object.defineProperty(window, "devicePixelRatio", { value: 3, configurable: true });
		try {
			sessionList = [summary({ session_id: "s1", conversation_name: "Retina", agent_id: "a", avatar_sha256: "f00d" })];
			render(<SessionListScreen />);
			const img = card("Retina").querySelector("img[data-agent-avatar]") as HTMLImageElement;
			expect(img.getAttribute("src")).toBe("/v1/agents/a/avatar?v=f00d&size=128");
			expect(Number(img.getAttribute("width")) * 3).toBeLessThanOrEqual(128);
		} finally {
			Object.defineProperty(window, "devicePixelRatio", { value: original, configurable: true });
		}
	});

	it("appends size only when the relay advertises it", async () => {
		const { sizedAvatarUrl, avatarSizesServed } = await import("./components/agent-avatar");
		const base = "/v1/agents/a1/avatar?v=abc";
		expect(sizedAvatarUrl(base, 24, 3, true)).toBe(`${base}&size=128`);
		expect(sizedAvatarUrl(base, 24, 3, false)).toBe(base);
		expect(sizedAvatarUrl(base, 128, 3, true)).toBe(base);
		expect(avatarSizesServed({ features: { agent_avatar_sizes: 1 } } as never)).toBe(true);
		expect(avatarSizesServed({ features: {} } as never)).toBe(false);
		expect(avatarSizesServed(null)).toBe(false);
	});
});
