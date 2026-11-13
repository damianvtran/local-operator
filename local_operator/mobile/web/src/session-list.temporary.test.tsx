// @vitest-environment happy-dom
//
// The relay home's `Temporary · N` group (contract §2.2/§2.3), against the REAL
// SessionListScreen with a faked relay:
//
//   * with `features.temporary_chats`, temporary rows leave Pinned/Active/
//     Previous and sit in ONE collapsed group at the head of the list, which
//     opens on its own while it holds the chat the reader came back from;
//   * each row says its countdown or why it is being kept;
//   * WITHOUT the capability nothing changes: no group, rows stay where the
//     daemon ranked them (fail closed);
//   * a refused pin of a temporary chat offers Keep and pin, which keeps the
//     chat and then pins it.
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HttpError } from "./api";
import { SessionListScreen } from "./screens/session-list";
import { TEMPORARY_HOLD_CODES, type Capabilities, type SessionSummary, type TemporaryInfo } from "./types";

let sessionList: SessionSummary[] = [];
let caps: Capabilities | null = null;
let lastViewed: string | null = null;
const setSessionPin = vi.fn(async (_id: string, pinned: boolean) => ({ ok: true, pinned }));
const setSessionTemporary = vi.fn(async (id: string, _on: boolean) => ({ session_id: id, temporary: null }));
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useSessions: () => ({ sessions: sessionList, connected: true, revision: 0 }),
		useCapabilities: () => caps,
		retainSessionListStream: () => () => {},
		getLastViewedSession: () => lastViewed,
	};
});
vi.mock("./api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./api")>();
	return {
		...actual,
		getDirectories: vi.fn(async () => ({ home: "", recent: [] })),
		setSessionPin: (id: string, pinned: boolean) => setSessionPin(id, pinned),
		setSessionTemporary: (id: string, on: boolean) => setSessionTemporary(id, on),
	};
});

const NOW_MS = 1_800_000_000_000;
function mark(over: Partial<TemporaryInfo> = {}): TemporaryInfo {
	return { marked_at: NOW_MS / 1000 - 60, idle_minutes: 30, deletes_at: NOW_MS / 1000 + 12 * 60, hold: null, ...over };
}
function summary(over: Partial<SessionSummary>): SessionSummary {
	return {
		session_id: "s", section: "active", conversation_name: "Session", cwd: "", model_label: "",
		streaming: false, needs_attention: false, unseen: false, pending_kind: "",
		subagents_running: 0, subagents_queued: 0, todos_open: 0, mtime: 0, pinned: false,
		temporary: null, ...over,
	};
}
const ENABLED: Capabilities = { features: { temporary_chats: 1 } };

function sectionHeadings(): string[] {
	return screen.getAllByRole("heading", { level: 2 }).map((h) => h.textContent ?? "");
}

beforeEach(() => {
	vi.useFakeTimers({ shouldAdvanceTime: true });
	vi.setSystemTime(NOW_MS);
	sessionList = [
		summary({ session_id: "kept", conversation_name: "Kept chat" }),
		summary({ session_id: "t1", conversation_name: "Regex question", temporary: mark() }),
		summary({ session_id: "t2", conversation_name: "Held chat", section: "previous", temporary: mark({ hold: "open_ask", deletes_at: null }) }),
		summary({ session_id: "pin", conversation_name: "Pinned chat", pinned: true }),
	];
	caps = ENABLED;
	lastViewed = null;
	setSessionPin.mockClear();
	setSessionTemporary.mockClear();
});
afterEach(() => {
	cleanup();
	vi.useRealTimers();
});

describe("Temporary · N group", () => {
	it("leads the list, collapsed, and holds the temporary rows only", () => {
		render(<SessionListScreen />);
		const headings = sectionHeadings();
		expect(headings[0]).toContain("Temporary · 2");
		expect(headings.slice(1)).toEqual(["Pinned", "Active"]);
		const toggle = screen.getByRole("button", { name: /Temporary · 2/ });
		expect(toggle.getAttribute("aria-expanded")).toBe("false");
		expect(screen.queryByText("Regex question")).toBeNull();
		expect(screen.queryByText("Held chat")).toBeNull();
		expect(screen.getByText("Kept chat")).toBeTruthy();
		// Previous held only a temporary row, so it has no heading at all.
		expect(headings).not.toContain("Previous");
	});

	it("opens on tap and shows the countdown or the hold reason", () => {
		render(<SessionListScreen />);
		fireEvent.click(screen.getByRole("button", { name: /Temporary · 2/ }));
		const group = document.querySelector('[data-section="temporary"]') as HTMLElement;
		expect(within(group).getByText("Regex question")).toBeTruthy();
		const labels = [...group.querySelectorAll("[data-temporary-label]")].map((el) => el.textContent);
		expect(labels).toEqual(["Deletes in 12 min", "Kept while a question is open"]);
	});

	it("opens on its own while it holds the chat the reader came back from", () => {
		lastViewed = "t2";
		render(<SessionListScreen />);
		expect(screen.getByRole("button", { name: /Temporary · 2/ }).getAttribute("aria-expanded")).toBe("true");
		expect(screen.getByText("Held chat")).toBeTruthy();
	});

	it("paints the last five minutes in warning ink", () => {
		sessionList = [summary({ session_id: "t1", conversation_name: "Soon", temporary: mark({ deletes_at: NOW_MS / 1000 + 4 * 60 }) })];
		lastViewed = "t1";
		render(<SessionListScreen />);
		const label = document.querySelector("[data-temporary-label]") as HTMLElement;
		expect(label.textContent).toBe("Deletes in 4 min");
		expect(label.className).toContain("text-warning");
	});

	it("fails closed without the capability: no group, rows where the daemon ranked them", () => {
		caps = { features: {} };
		render(<SessionListScreen />);
		expect(sectionHeadings()).toEqual(["Pinned", "Active", "Previous"]);
		expect(document.querySelector('[data-section="temporary"]')).toBeNull();
		expect(document.querySelector("[data-temporary-label]")).toBeNull();
		expect(screen.getByText("Regex question")).toBeTruthy();
	});

	it("has English copy for every hold code (no raw key reaches the screen)", () => {
		lastViewed = "h0";
		sessionList = TEMPORARY_HOLD_CODES.map((hold, i) =>
			summary({ session_id: `h${i}`, conversation_name: `held ${hold}`, temporary: mark({ hold, deletes_at: null }) }));
		render(<SessionListScreen />);
		const labels = [...document.querySelectorAll("[data-temporary-label]")].map((el) => el.textContent ?? "");
		expect(labels).toHaveLength(11);
		for (const text of labels) {
			expect(text).toMatch(/^Kept/);
			expect(text).not.toContain("relay.chrome");
		}
	});
});

describe("Keep and pin", () => {
	it("offers the remedy on 409 session_temporary, then keeps and pins", async () => {
		setSessionPin.mockImplementationOnce(async () => {
			throw new HttpError(409, "This is a temporary chat. Keep it before pinning.", "session_temporary");
		});
		lastViewed = "t1";
		render(<SessionListScreen />);
		const group = document.querySelector('[data-section="temporary"]') as HTMLElement;
		fireEvent.click(within(group).getAllByRole("button", { name: "More" })[0]);
		fireEvent.click(await screen.findByRole("button", { name: /^Pin/ }));
		const remedy = await screen.findByRole("button", { name: "Keep and pin" });
		await act(async () => { fireEvent.click(remedy); });
		await waitFor(() => expect(setSessionTemporary).toHaveBeenCalledWith("t1", false));
		await waitFor(() => expect(setSessionPin).toHaveBeenLastCalledWith("t1", true));
		expect(setSessionPin).toHaveBeenCalledTimes(2);
	});

	it("offers no remedy for an ordinary refusal", async () => {
		setSessionPin.mockImplementationOnce(async () => {
			throw new HttpError(409, "no saved messages yet", "session_empty");
		});
		render(<SessionListScreen />);
		fireEvent.click(screen.getAllByRole("button", { name: "More" })[1]);
		fireEvent.click(await screen.findByRole("button", { name: /^Pin/ }));
		await waitFor(() => expect(setSessionPin).toHaveBeenCalled());
		expect(screen.queryByRole("button", { name: "Keep and pin" })).toBeNull();
	});
});
