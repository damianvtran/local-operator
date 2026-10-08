// @vitest-environment happy-dom
//
// The one-gesture clear (issue #2016), round-2 remediation: the control is
// present only while the pile is non-empty, states HOW MANY it will clear (and
// stops stating it once a read could not be verified), and posts the badge rows
// for conversations this list carries. The receipt names the store's per-item
// verdicts, never claims a sweep it did not get, is reported as unknown — not as
// an empty pile — when the unread read was degraded, lives in the store so a
// glance into a conversation cannot erase it, and expires from the moment the
// reader is actually looking at it. Rendered against the REAL SessionListScreen,
// like the ladder test next door.
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AttentionSeenManyReceipt } from "./api";
import { MARK_NOTICE_TTL_MS, publishMarkNotice } from "./store";
import { SessionListScreen } from "./screens/session-list";
import type { AttentionUnread, CompletionAttention, SessionSummary } from "./types";

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

function unread(over: Partial<AttentionUnread>): AttentionUnread {
	return { count: 0, revision: [0, 0, 0], degraded: [], conversations: [], ...over };
}

function conversation(session_id: string, completion_token: string) {
	return { session_id, completion_token, kind: "complete" as const, revision: [1, 0] as [number, number] };
}

/** The control by its accessible name — null while the pile is empty. The
    label carries the COUNT (design D2 / UX U4 / QA Q1). */
function control(): HTMLButtonElement | null {
	return screen.queryByRole("button", { name: /mark all \d+ read/ }) as HTMLButtonElement | null;
}

/** The control when it has dropped the count (UX round 2, U11: after a read that
    could not be verified it stops asserting a number as fact). */
function unnumberedControl(): HTMLButtonElement | null {
	return screen.queryByRole("button", { name: "mark all as read" }) as HTMLButtonElement | null;
}

/** The sticky band the control sits in — its own element, the class list is what
    the separator and focus-ring findings are about (design round 2, D6 / D7).
    It is the only `sticky` element on the screen. */
function band(): HTMLDivElement {
	return document.querySelector("div.sticky") as HTMLDivElement;
}

afterEach(() => {
	cleanup();
	sessionList = [];
	// The receipt is MODULE state now, so it outlives a test's unmount by design.
	publishMarkNotice(null);
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

	it("states how many it will clear", () => {
		sessionList = [
			summary({ session_id: "u1", conversation_name: "Alpha", unseen: true }),
			summary({ session_id: "u2", conversation_name: "Beta", unseen: true }),
		];
		const { rerender } = render(<SessionListScreen />);
		expect(control()?.textContent).toBe("mark all 2 read");

		sessionList = [summary({ session_id: "u1", conversation_name: "Alpha", unseen: true })];
		rerender(<SessionListScreen />);
		expect(control()?.textContent).toBe("mark all 1 read");
	});

	it("posts only the rows it is painting, and reports the receipt", async () => {
		sessionList = [
			summary({ session_id: "u1", conversation_name: "Alpha", unseen: true }),
			summary({ session_id: "u2", conversation_name: "Beta", unseen: true }),
		];
		// The badge enumerates a THIRD conversation the list is not painting: a
		// completion published since the last frame. The batch must be the
		// rendered set (agent MINOR-2), so u3 is never posted.
		vi.mocked(getAttentionUnread).mockResolvedValue(
			unread({ count: 3, conversations: [conversation("u1", "t1"), conversation("u2", "t2"), conversation("u3", "t3")] }),
		);
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
		vi.mocked(getAttentionUnread).mockResolvedValue(
			unread({ count: 1, conversations: [conversation("u1", "t1")] }),
		);
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

	it("treats a DEGRADED read as unknown, never as an empty pile", async () => {
		sessionList = [summary({ session_id: "u1", conversation_name: "Alpha", unseen: true })];
		// The shape the route serves when a read behind the aggregate failed:
		// no `count`, no `conversations` — only the sources that failed.
		vi.mocked(getAttentionUnread).mockResolvedValue({ degraded: ["attention"] });
		render(<SessionListScreen />);
		fireEvent.click(control()!);

		const alert = await screen.findByRole("alert");
		expect(alert.textContent).toBe("Could not read what is unread — nothing was cleared. Try again.");
		expect(vi.mocked(markAllSeen)).not.toHaveBeenCalled();
	});

	it("says nothing to clear when the pile really is empty, without posting", async () => {
		sessionList = [summary({ session_id: "u1", conversation_name: "Alpha", unseen: true })];
		vi.mocked(getAttentionUnread).mockResolvedValue(unread({ count: 0, conversations: [] }));
		render(<SessionListScreen />);
		fireEvent.click(control()!);

		await screen.findByText("Nothing to clear.");
		expect(vi.mocked(markAllSeen)).not.toHaveBeenCalled();
	});

	it("surfaces a failed write in the reader's words, naming the recovery", async () => {
		sessionList = [summary({ session_id: "u1", conversation_name: "Alpha", unseen: true })];
		vi.mocked(getAttentionUnread).mockResolvedValue(
			unread({ count: 1, conversations: [conversation("u1", "t1")] }),
		);
		vi.mocked(markAllSeen).mockRejectedValue(new TypeError("Failed to fetch"));
		render(<SessionListScreen />);
		fireEvent.click(control()!);

		const alert = await screen.findByRole("alert");
		expect(alert.textContent).toBe(
			"Nothing was cleared — the daemon could not be reached. Try again.",
		);
	});

	it("keeps the control focusable while the write is in flight", async () => {
		sessionList = [summary({ session_id: "u1", conversation_name: "Alpha", unseen: true })];
		vi.mocked(getAttentionUnread).mockResolvedValue(
			unread({ count: 1, conversations: [conversation("u1", "t1")] }),
		);
		let release: (value: AttentionSeenManyReceipt) => void = () => {};
		vi.mocked(markAllSeen).mockReturnValue(
			new Promise<AttentionSeenManyReceipt>((resolve) => {
				release = resolve;
			}),
		);
		render(<SessionListScreen />);
		const button = control()!;
		button.focus();
		fireEvent.click(button);

		// `aria-disabled`, never `disabled`: a disabled button drops focus to
		// <body> the instant it is pressed, which is UX round 1's U3. The label
		// changes to `marking…` here, so this queries by that name rather than
		// the count label `control()` matches.
		await screen.findByText("marking…");
		const marking = screen.getByRole("button", { name: "marking…" }) as HTMLButtonElement;
		expect(marking.getAttribute("aria-disabled")).toBe("true");
		expect(marking.disabled).toBe(false);
		expect(document.activeElement).toBe(marking);

		await act(async () => {
			release(receipt({ read: [attentionState({ conversation_id: "session/u1", completion_token: "t1" })] }));
		});
	});

	it("outlives a glance into a conversation and back", async () => {
		sessionList = [summary({ session_id: "u1", conversation_name: "Alpha", unseen: true })];
		vi.mocked(getAttentionUnread).mockResolvedValue(
			unread({ count: 1, conversations: [conversation("u1", "t1")] }),
		);
		vi.mocked(markAllSeen).mockResolvedValue(
			receipt({ read: [attentionState({ conversation_id: "session/u1", completion_token: "t1" })] }),
		);
		const view = render(<SessionListScreen />);
		fireEvent.click(control()!);
		await screen.findByText("Marked 1 read.");

		// Route away (the screen unmounts) and back: the receipt is the only
		// explanation of a partial clear, so it must survive (UX U2).
		view.unmount();
		render(<SessionListScreen />);
		expect(screen.getByText("Marked 1 read.")).toBeTruthy();
	});

	it("expires on the store's TTL and can be dismissed", async () => {
		vi.useFakeTimers();
		try {
			render(<SessionListScreen />);
			act(() => publishMarkNotice({ text: "Marked 2 read.", danger: false }));
			expect(screen.getByText("Marked 2 read.")).toBeTruthy();

			fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
			expect(screen.queryByText("Marked 2 read.")).toBeNull();

			act(() => publishMarkNotice({ text: "Marked 2 read.", danger: false }));
			act(() => {
				vi.advanceTimersByTime(MARK_NOTICE_TTL_MS + 1);
			});
			expect(screen.queryByText("Marked 2 read.")).toBeNull();
		} finally {
			vi.useRealTimers();
		}
	});

	it("moves focus off <body> when the control unmounts under the reader", () => {
		vi.useFakeTimers();
		try {
			sessionList = [
				summary({ session_id: "u1", conversation_name: "Alpha", unseen: true }),
				summary({ session_id: "u2", conversation_name: "Beta", unseen: true }),
			];
			const view = render(<SessionListScreen />);
			control()!.focus();
			expect(document.activeElement).toBe(control());

			// The pile clears elsewhere: the control goes inert, then unmounts.
			sessionList = [
				summary({ session_id: "u1", conversation_name: "Alpha" }),
				summary({ session_id: "u2", conversation_name: "Beta" }),
			];
			act(() => {
				view.rerender(<SessionListScreen />);
			});
			act(() => {
				vi.advanceTimersByTime(300);
			});
			expect(document.activeElement).not.toBe(document.body);
			expect(document.activeElement?.tagName).toBe("BUTTON");
		} finally {
			vi.useRealTimers();
		}
	});

	it("suppresses the count once a read could not be verified, and restores it when one can", async () => {
		// UX round 2, U11. The number is derived from the painted pile, so in the
		// one state where this screen has just said it could not read unread state,
		// the control must stop asserting it — and may only state it again once a
		// read has answered for real.
		sessionList = [
			summary({ session_id: "u1", conversation_name: "Alpha", unseen: true }),
			summary({ session_id: "u2", conversation_name: "Beta", unseen: true }),
		];
		vi.mocked(getAttentionUnread).mockResolvedValue({ degraded: ["attention"] });
		render(<SessionListScreen />);
		expect(control()!.textContent).toBe("mark all 2 read");

		fireEvent.click(control()!);
		await screen.findByRole("alert");
		expect(unnumberedControl()!.textContent).toBe("mark all as read");
		expect(control()).toBeNull();

		vi.mocked(getAttentionUnread).mockResolvedValue(
			unread({ count: 2, conversations: [conversation("u1", "t1"), conversation("u2", "t2")] }),
		);
		vi.mocked(markAllSeen).mockResolvedValue(receipt({ read: [] }));
		fireEvent.click(unnumberedControl()!);
		await screen.findByRole("button", { name: "mark all 2 read" });
	});

	it("pauses the receipt's window while the reader is away, and restarts it on return", async () => {
		// UX round 2, U9: a store-owned timer armed at the tap kept ticking through
		// a route change, so a glance longer than the TTL still erased the receipt.
		// The window is the reader's own viewing time now.
		vi.useFakeTimers();
		try {
			const view = render(<SessionListScreen />);
			act(() => publishMarkNotice({ text: "Marked 2 read.", danger: false }));
			expect(screen.queryByText("Marked 2 read.")).toBeTruthy();

			// Away for LONGER than the TTL: nothing is on screen to expire.
			view.unmount();
			act(() => {
				vi.advanceTimersByTime(MARK_NOTICE_TTL_MS * 2);
			});

			// Back: the receipt is still there, and gets a full fresh window.
			render(<SessionListScreen />);
			expect(screen.queryByText("Marked 2 read.")).toBeTruthy();
			act(() => {
				vi.advanceTimersByTime(MARK_NOTICE_TTL_MS - 1000);
			});
			expect(screen.queryByText("Marked 2 read.")).toBeTruthy();

			// Still bounded: the window closes TTL after the reader returned (U8).
			act(() => {
				vi.advanceTimersByTime(2000);
			});
			expect(screen.queryByText("Marked 2 read.")).toBeNull();
		} finally {
			vi.useRealTimers();
		}
	});

	it("leaves the focus alone when the pile was cleared by someone else", () => {
		// The guard U3 promises: a pile cleared from another surface must not steal
		// the reader's focus (agent round 2, NIT-3 — this path was unpinned).
		vi.useFakeTimers();
		try {
			sessionList = [
				summary({ session_id: "u1", conversation_name: "Alpha", unseen: true }),
				summary({ session_id: "u2", conversation_name: "Beta", unseen: true }),
			];
			const view = render(<SessionListScreen />);
			const search = screen.getByPlaceholderText("Search conversations…") as HTMLInputElement;
			search.focus();

			sessionList = [
				summary({ session_id: "u1", conversation_name: "Alpha" }),
				summary({ session_id: "u2", conversation_name: "Beta" }),
			];
			act(() => {
				view.rerender(<SessionListScreen />);
			});
			act(() => {
				vi.advanceTimersByTime(300);
			});
			expect(document.activeElement).toBe(search);
		} finally {
			vi.useRealTimers();
		}
	});

	it("lands the keyboard on the band — with a visible ring — when no card is registered", () => {
		// The fallback half of U3 (design round 2, D7 / QA Q-2): with a query that
		// registers no card, the focus goes to the band, and the band must SHOW it
		// (WCAG 2.4.7) — Tailwind's `outline-none` would have neutralised the ring.
		vi.useFakeTimers();
		try {
			sessionList = [summary({ session_id: "u1", conversation_name: "Alpha", unseen: true })];
			const view = render(<SessionListScreen />);
			fireEvent.change(screen.getByPlaceholderText("Search conversations…"), {
				target: { value: "zzz-no-such-conversation" },
			});
			control()!.focus();

			sessionList = [summary({ session_id: "u1", conversation_name: "Alpha" })];
			act(() => {
				view.rerender(<SessionListScreen />);
			});
			act(() => {
				vi.advanceTimersByTime(300);
			});

			expect(document.activeElement).toBe(band());
			expect(band().className).toContain("focus:outline-accent");
			expect(band().className).not.toContain("outline-none");
		} finally {
			vi.useRealTimers();
		}
	});

	it("separates the pinned band from the rows sliding under it", () => {
		// Design round 2, D6: with no edge on an opaque `bg-canvas`, a row slides
		// under the band and its text is cut mid-glyph against a flat fill.
		vi.useFakeTimers();
		try {
			sessionList = [summary({ session_id: "u1", conversation_name: "Alpha", unseen: true })];
			const view = render(<SessionListScreen />);
			expect(band().className).toContain("border-hairline");

			// The edge goes with the band's content: while the control is still
			// sliding out the band holds something, and an emptied band leaves no
			// stray line across the top of the list.
			sessionList = [summary({ session_id: "u1", conversation_name: "Alpha" })];
			act(() => {
				view.rerender(<SessionListScreen />);
			});
			expect(band().className).toContain("border-hairline");
			act(() => {
				vi.advanceTimersByTime(300);
			});
			expect(band().className).not.toContain("border-hairline");
		} finally {
			vi.useRealTimers();
		}
	});
});
