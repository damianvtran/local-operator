// @vitest-environment happy-dom
//
// The one-gesture clear (issue #2016): the control is present only while the
// pile is non-empty (and hidden from the accessibility tree when collapsed),
// posts exactly the completions the daemon's unread read enumerated, and
// reports the store's per-item verdicts rather than claiming a clean sweep it
// did not get. Rendered against the REAL SessionListScreen, like the ladder
// test next door.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AttentionSeenManyReceipt } from "./api";
import { SessionListScreen } from "./screens/session-list";
import type { CompletionAttention } from "./types";
import type { SessionSummary } from "./types";

let sessionList: SessionSummary[] = [];
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useSessions: () => ({ sessions: sessionList, connected: true }),
		retainSessionListStream: () => () => {},
	};
});
vi.mock("./api", () => ({
	getDirectories: vi.fn(async () => ({ home: "", recent: [] })),
	getAttentionUnread: vi.fn(async () => ({ count: 0, conversations: [] })),
	markAllSeen: vi.fn(async () => ({ ok: true, read: [], superseded: [], unknown: [] })),
}));

const { getAttentionUnread, markAllSeen } = await import("./api");

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
		todos_open: 0,
		mtime: 0,
		...over,
	};
}

function attentionState(over: Partial<CompletionAttention>): CompletionAttention {
	return {
		conversation_id: "session/s",
		completion_token: "t",
		anchor_id: "a",
		kind: "complete",
		unseen: false,
		revision: [1, 1],
		...over,
	};
}

function receipt(over: Partial<AttentionSeenManyReceipt>): AttentionSeenManyReceipt {
	return { ok: true, read: [], superseded: [], unknown: [], ...over };
}

/** The control by its accessible name — null while the pile is empty, when the
    control is not rendered at all (with a brief exit window after it clears). */
function control(): HTMLElement | null {
	return screen.queryByRole("button", { name: "mark all as read" });
}

afterEach(() => {
	cleanup();
	sessionList = [];
	vi.mocked(getAttentionUnread).mockReset();
	vi.mocked(markAllSeen).mockReset();
});

describe("mark all as read", () => {
	it("is not rendered until a row is unseen", () => {
		sessionList = [summary({ session_id: "a", conversation_name: "Alpha" })];
		const { rerender } = render(<SessionListScreen />);
		expect(control()).toBeNull();

		sessionList = [summary({ session_id: "a", conversation_name: "Alpha", unseen: true })];
		rerender(<SessionListScreen />);
		expect(control()).not.toBeNull();
	});

	it("posts every enumerated completion and reports the receipt", async () => {
		sessionList = [
			summary({ session_id: "u1", conversation_name: "Alpha", unseen: true }),
			summary({ session_id: "u2", conversation_name: "Beta", unseen: true }),
		];
		vi.mocked(getAttentionUnread).mockResolvedValue({
			count: 2,
			revision: [2, 0, 0],
			degraded: [],
			conversations: [
				{ session_id: "u1", completion_token: "t1", kind: "complete", revision: [1, 0] },
				{ session_id: "u2", completion_token: "t2", kind: "complete", revision: [2, 0] },
			],
		});
		vi.mocked(markAllSeen).mockResolvedValue(
			receipt({
				read: [
					attentionState({ conversation_id: "session/u1", completion_token: "t1" }),
					attentionState({ conversation_id: "session/u2", completion_token: "t2" }),
				],
			}),
		);
		render(<SessionListScreen />);
		fireEvent.click(control()!);

		await screen.findByText("Marked 2 read.");
		expect(vi.mocked(markAllSeen)).toHaveBeenCalledWith([
			{ session_id: "u1", completion_token: "t1" },
			{ session_id: "u2", completion_token: "t2" },
		]);
	});

	it("names the buckets it could not clear instead of a clean sweep", async () => {
		sessionList = [summary({ session_id: "u1", conversation_name: "Alpha", unseen: true })];
		vi.mocked(getAttentionUnread).mockResolvedValue({
			count: 1,
			revision: [1, 0, 0],
			degraded: [],
			conversations: [
				{ session_id: "u1", completion_token: "t1", kind: "complete", revision: [1, 0] },
			],
		});
		vi.mocked(markAllSeen).mockResolvedValue(
			receipt({ superseded: ["u1"], unknown: ["deadbeef1234"] }),
		);
		render(<SessionListScreen />);
		fireEvent.click(control()!);

		const status = await screen.findByRole("status");
		expect(status.textContent).toBe(
			"1 has a newer result and stays unread. " +
				"1 could not be found on this machine and stays unread.",
		);
	});

	it("surfaces a failed write in the alert line", async () => {
		sessionList = [summary({ session_id: "u1", conversation_name: "Alpha", unseen: true })];
		vi.mocked(getAttentionUnread).mockResolvedValue({
			count: 1,
			revision: [1, 0, 0],
			degraded: [],
			conversations: [
				{ session_id: "u1", completion_token: "t1", kind: "complete", revision: [1, 0] },
			],
		});
		vi.mocked(markAllSeen).mockRejectedValue(new Error("store busy"));
		render(<SessionListScreen />);
		fireEvent.click(control()!);

		const alert = await screen.findByRole("alert");
		expect(alert.textContent).toBe("Could not mark read: store busy");
	});

	it("says so when the badge enumerates nothing, without posting", async () => {
		sessionList = [summary({ session_id: "u1", conversation_name: "Alpha", unseen: true })];
		vi.mocked(getAttentionUnread).mockResolvedValue({ count: 0, conversations: [] });
		render(<SessionListScreen />);
		fireEvent.click(control()!);

		await screen.findByText("Nothing unread.");
		expect(vi.mocked(markAllSeen)).not.toHaveBeenCalled();
	});
});
