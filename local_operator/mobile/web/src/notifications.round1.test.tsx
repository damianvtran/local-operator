// @vitest-environment happy-dom
//
// Review round 1 on #2968 (RW1-RW10): each case reproduces a finding against
// the phone inbox and pins the fix. The store/api mocks mirror
// notifications.test.tsx; a list frame is simulated by changing the mocked
// `notifications` block, which is exactly what makes the reel refetch.
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Capabilities, NotificationDetail, NotificationItem, NotificationsBlock, NotificationsPage, SessionSummary } from "./types";

let capability: Capabilities | null = null;
let block: NotificationsBlock | undefined = undefined;
const listeners = new Set<() => void>();
/** Deliver a list frame: the reel's refetch signal is the block's numbers. */
function frame(next: NotificationsBlock) {
	block = next;
	for (const listener of listeners) listener();
}

vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	const react = await import("react");
	return {
		...actual,
		useSessions: () => ({ sessions: [] as SessionSummary[], connected: true, revision: 0 }),
		useCapabilities: () => capability,
		useNotificationsBlock: () => {
			const [, bump] = react.useReducer((n: number) => n + 1, 0);
			react.useEffect(() => { listeners.add(bump); return () => { listeners.delete(bump); }; }, []);
			return block;
		},
		retainSessionListStream: () => () => {},
	};
});

vi.mock("./api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./api")>();
	return {
		...actual,
		getDirectories: vi.fn(async () => ({ home: "", recent: [] })),
		getProjects: vi.fn(async () => ({ projects: [] })),
		getAsks: vi.fn(async () => ({ asks: [] })),
		getNotifications: vi.fn(),
		getNotification: vi.fn(),
		patchNotifications: vi.fn(),
	};
});

const api = await import("./api");
const { NotificationsScreen } = await import("./screens/notifications");
const { NotificationDetailScreen } = await import("./screens/notification-detail");
const { SessionListScreen } = await import("./screens/session-list");

const NOW = Date.now();
let n = 0;
function item(patch: Partial<NotificationItem> = {}): NotificationItem {
	n += 1;
	return {
		id: `n_${String(n).padStart(16, "a")}`,
		source: "notify",
		priority: "normal",
		title: `Update ${n}`,
		excerpt: "",
		created_at: NOW - 60_000,
		updated_at: NOW - 60_000,
		seq: n,
		revision: 1,
		read: false,
		read_via: null,
		dismissed: false,
		repeat_count: 1,
		session: { id: "abcdefabcdef", title: "fraud watch", agent_id: "", live: true, live_state: "idle" },
		projects: [],
		ask: null,
		actions: { dismiss: true, answer: false, read: true },
		...patch,
	};
}
function page(items: NotificationItem[], patch: Partial<NotificationsPage> = {}): NotificationsPage {
	return {
		items,
		waiting: [],
		observed: items.map(row => ({ id: row.id, revision: row.revision })),
		next_cursor: null,
		unread_count: items.filter(row => !row.read).length,
		waiting_count: 0,
		waiting_unread_count: 0,
		high_water: 7,
		...patch,
	};
}

beforeEach(() => {
	n = 0;
	capability = { features: { notifications: 1 } };
	block = { unread_count: 1, waiting_count: 0, waiting_unread_count: 0, high_water: 7 };
	localStorage.clear();
	// mockReset, not clearAllMocks: a queued *Once answer a test did not consume
	// must not leak into the next test's reads.
	vi.mocked(api.getNotifications).mockReset();
	vi.mocked(api.getNotification).mockReset();
	vi.mocked(api.patchNotifications).mockReset();
	vi.mocked(api.patchNotifications).mockResolvedValue({ changed: [], skipped: [] });
});
afterEach(() => {
	cleanup();
	vi.clearAllMocks();
	listeners.clear();
	history.replaceState(null, "", "#/");
});

describe("RW1: Mark all read sends the set the sheet was opened over", () => {
	it("does not sweep an item that arrived while the confirmation was open", async () => {
		const seen = item({ title: "Seen before the sheet" });
		vi.mocked(api.getNotifications).mockResolvedValueOnce(page([seen]));
		render(<NotificationsScreen />);
		fireEvent.click(await screen.findByRole("button", { name: "Mark all read" }));
		const late = item({ title: "Arrived under the sheet", seq: 9 });
		vi.mocked(api.getNotifications).mockResolvedValue(page([late, seen], { high_water: 9 }));
		await act(async () => frame({ unread_count: 2, waiting_count: 0, waiting_unread_count: 0, high_water: 9 }));
		await waitFor(() => expect(api.getNotifications).toHaveBeenCalledTimes(2));
		fireEvent.click(screen.getByRole("button", { name: "Mark read" }));
		await waitFor(() => expect(api.patchNotifications).toHaveBeenCalledTimes(1));
		expect(vi.mocked(api.patchNotifications).mock.calls[0][0]).toEqual({
			op: "read", all: true, through_seq: 7, observed: [{ id: seen.id, revision: 1 }],
		});
	});
});

describe("RW6: Mark all read names only unread rows", () => {
	it("leaves already-read rows out of the body on the All tab", async () => {
		localStorage.setItem("lo-mobile-notifications-tab", "all");
		const unread = item();
		const read = item({ read: true, read_via: "explicit" });
		vi.mocked(api.getNotifications).mockResolvedValue(page([unread, read]));
		render(<NotificationsScreen />);
		fireEvent.click(await screen.findByRole("button", { name: "Mark all read" }));
		expect(screen.getByText("Mark 1 notification read?")).toBeTruthy();
		fireEvent.click(screen.getByRole("button", { name: "Mark read" }));
		await waitFor(() => expect(api.patchNotifications).toHaveBeenCalledTimes(1));
		expect(vi.mocked(api.patchNotifications).mock.calls[0][0]).toMatchObject({ observed: [{ id: unread.id, revision: 1 }] });
	});
});

describe("RW3: a frame keeps the depth the reader loaded", () => {
	it("still shows, and bulk-covers, a page-two row after a refresh", async () => {
		const first = item({ title: "Page one row" });
		const second = item({ title: "Page two row" });
		vi.mocked(api.getNotifications)
			.mockResolvedValueOnce(page([first], { next_cursor: "c1" }))
			.mockResolvedValueOnce(page([second]));
		render(<NotificationsScreen />);
		fireEvent.click(await screen.findByRole("button", { name: "Load more" }));
		expect(await screen.findByText("Page two row")).toBeTruthy();
		const fresh = item({ title: "Fresh top row", seq: 8 });
		vi.mocked(api.getNotifications)
			.mockResolvedValueOnce(page([fresh, first], { next_cursor: "c2", high_water: 8 }))
			.mockResolvedValueOnce(page([second], { high_water: 8 }));
		await act(async () => frame({ unread_count: 3, waiting_count: 0, waiting_unread_count: 0, high_water: 8 }));
		expect(await screen.findByText("Fresh top row")).toBeTruthy();
		expect(screen.getByText("Page two row")).toBeTruthy();
		// The depth re-read is one consistent read: pinned to the new boundary.
		expect(vi.mocked(api.getNotifications).mock.calls[3][0]).toEqual({ state: "unread", cursor: "c2", through_seq: 8 });
		fireEvent.click(screen.getByRole("button", { name: "Mark all read" }));
		fireEvent.click(screen.getByRole("button", { name: "Mark read" }));
		await waitFor(() => expect(api.patchNotifications).toHaveBeenCalledTimes(1));
		expect(vi.mocked(api.patchNotifications).mock.calls[0][0]).toMatchObject({ through_seq: 8 });
		expect((vi.mocked(api.patchNotifications).mock.calls[0][0] as { observed: { id: string }[] }).observed.map(t => t.id)).toContain(second.id);
	});
});

describe("RW4: the drawer badge", () => {
	it("draws nothing at zero and caps at 99+", async () => {
		block = { unread_count: 0, waiting_count: 2, waiting_unread_count: 0, high_water: 3 };
		render(<SessionListScreen />);
		fireEvent.click(screen.getByRole("button", { name: "Open navigation" }));
		expect(screen.getByRole("link", { name: "Notifications" })).toBeTruthy();
		cleanup();
		block = { unread_count: 120, waiting_count: 1, waiting_unread_count: 1, high_water: 3 };
		render(<SessionListScreen />);
		fireEvent.click(screen.getByRole("button", { name: "Open navigation" }));
		expect(screen.getByRole("link", { name: "Notifications 99+" })).toBeTruthy();
	});
});

describe("RW5: a deep link on a relay without the inbox", () => {
	it("leaves for the list instead of reading a route that 404s", async () => {
		capability = { features: {} };
		block = undefined;
		history.replaceState(null, "", "#/notifications");
		render(<NotificationsScreen />);
		await waitFor(() => expect(location.hash).toBe("#/"));
		expect(api.getNotifications).not.toHaveBeenCalled();
	});
});

describe("RW10: blocked writes are really disabled", () => {
	it("disables the row dismiss and Mark all read while a refresh has failed", async () => {
		vi.mocked(api.getNotifications).mockResolvedValueOnce(page([item({ title: "Held row" })]));
		render(<NotificationsScreen />);
		await screen.findByText("Held row");
		vi.mocked(api.getNotifications).mockRejectedValue(new TypeError("Failed to fetch"));
		await act(async () => frame({ unread_count: 2, waiting_count: 0, waiting_unread_count: 0, high_water: 9 }));
		await screen.findByRole("alert");
		expect((screen.getByRole("button", { name: "Dismiss Held row" }) as HTMLButtonElement).disabled).toBe(true);
		expect((screen.getByRole("button", { name: "Mark all read" }) as HTMLButtonElement).disabled).toBe(true);
	});
});

describe("detail", () => {
	const detail = (patch: Partial<NotificationItem> = {}, links = [{ label: "Dashboard", url: "https://example.test/d" }]): NotificationDetail => ({
		item: { ...item({ read: true, read_via: "explicit", ...patch }), body_md: "Body.", links },
		history: [],
		session: { id: "abcdefabcdef", title: "fraud watch", agent_id: "", live: false, live_state: "", cwd: "/w" },
		anchor: null,
		projects: [],
		asks: [],
	});

	it("RW2: dismiss-all follows every page, pinned, and confirms the true total", async () => {
		vi.mocked(api.getNotification).mockResolvedValue(detail());
		const firstPage = Array.from({ length: 100 }, () => item());
		const rest = Array.from({ length: 30 }, () => item());
		vi.mocked(api.getNotifications)
			.mockResolvedValueOnce(page(firstPage, { next_cursor: "p2", high_water: 400 }))
			.mockResolvedValueOnce(page(rest, { high_water: 400 }));
		vi.mocked(api.patchNotifications).mockResolvedValue({ changed: [...firstPage, ...rest].map(r => r.id), skipped: [] });
		render(<NotificationDetailScreen itemId="n_x" />);
		fireEvent.click(await screen.findByRole("button", { name: "Dismiss all from this conversation" }));
		expect(await screen.findByText("Dismiss 130 updates from fraud watch?")).toBeTruthy();
		expect(vi.mocked(api.getNotifications).mock.calls[1][0]).toEqual({ state: "all", session_id: "abcdefabcdef", limit: 100, cursor: "p2", through_seq: 400 });
		fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
		await waitFor(() => expect(api.patchNotifications).toHaveBeenCalledTimes(1));
		const body = vi.mocked(api.patchNotifications).mock.calls[0][0] as { observed: unknown[]; through_seq: number };
		expect(body.observed).toHaveLength(130);
		expect(body.through_seq).toBe(400);
		expect(await screen.findByText("Dismissed 130")).toBeTruthy();
	});

	it("RW9: a failed dismiss-all read speaks in the reader's words", async () => {
		vi.mocked(api.getNotification).mockResolvedValue(detail());
		vi.mocked(api.getNotifications).mockRejectedValue(new TypeError("Failed to fetch"));
		render(<NotificationDetailScreen itemId="n_x" />);
		fireEvent.click(await screen.findByRole("button", { name: "Dismiss all from this conversation" }));
		expect((await screen.findByRole("alert")).textContent).toBe("Nothing changed: this computer could not be reached. Try again.");
	});

	it("RW7: Open conversation is disabled when the relay says it cannot be reached", async () => {
		vi.mocked(api.getNotification).mockResolvedValue(detail({ runtime_live: false, durable: false }));
		render(<NotificationDetailScreen itemId="n_x" />);
		expect(((await screen.findByRole("button", { name: "Open conversation" })) as HTMLButtonElement).disabled).toBe(true);
		cleanup();
		vi.mocked(api.getNotification).mockResolvedValue(detail({ runtime_live: false, durable: true }));
		render(<NotificationDetailScreen itemId="n_x" />);
		expect(((await screen.findByRole("button", { name: "Open conversation" })) as HTMLButtonElement).disabled).toBe(false);
	});

	it("RW8: drops a non-http link and keys duplicate URLs apart", async () => {
		const errors = vi.spyOn(console, "error").mockImplementation(() => {});
		vi.mocked(api.getNotification).mockResolvedValue(detail({}, [
			{ label: "Evil", url: "javascript:alert(1)" },
			{ label: "One", url: "https://example.test/same" },
			{ label: "Two", url: "https://example.test/same" },
		]));
		render(<NotificationDetailScreen itemId="n_x" />);
		expect(await screen.findByRole("link", { name: "One" })).toBeTruthy();
		expect(screen.getByRole("link", { name: "Two" })).toBeTruthy();
		expect(screen.queryByRole("link", { name: "Evil" })).toBeNull();
		expect(errors.mock.calls.some(call => String(call[0]).includes("same key"))).toBe(false);
		errors.mockRestore();
	});
});
