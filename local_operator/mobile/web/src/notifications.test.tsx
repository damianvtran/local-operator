// @vitest-environment happy-dom
//
// The phone inbox (#/notifications) against the #2939 relay wire. What these
// pin is the operator's read contract (decision D11) and the late-arrival rule:
// - the reel marks NOTHING read on mount, refocus or visibility change;
// - Mark all read is confirmed, and its PATCH echoes exactly the loaded
//   `observed` tokens plus the first page's `high_water`, never "everything";
// - receipts name every skipped bucket (live_ask, stale_revision), and Undo
//   reverts only what changed;
// - opening a LOADED detail sends one explicit read at the shown revision, and
//   a detail whose content never loaded sends none;
// - the drawer row exists only when the relay advertises the inbox.
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Capabilities, NotificationDetail, NotificationItem, NotificationsBlock, NotificationsPage, SessionSummary } from "./types";

let capability: Capabilities | null = null;
let block: NotificationsBlock | undefined = undefined;

vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useSessions: () => ({ sessions: [] as SessionSummary[], connected: true, revision: 0 }),
		useCapabilities: () => capability,
		useNotificationsBlock: () => block,
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
const { notificationsBadge } = await import("./store");

const NOW = Date.now();

function item(patch: Partial<NotificationItem> = {}): NotificationItem {
	return {
		id: "n_aaaaaaaaaaaaaaaa",
		source: "notify",
		priority: "normal",
		title: "Campaign run 41 finished",
		excerpt: "All 40 queries done",
		created_at: NOW - 60_000,
		updated_at: NOW - 60_000,
		seq: 3,
		revision: 1,
		read: false,
		read_via: null,
		dismissed: false,
		repeat_count: 1,
		session: { id: "abcdefabcdef", title: "article search", agent_id: "", live: true, live_state: "idle" },
		projects: [],
		ask: null,
		actions: { dismiss: true, answer: false, read: true },
		...patch,
	};
}

const askRow = item({
	id: "ask:abcdefabcdef:q1",
	source: "ask",
	priority: "high",
	title: "Merge PR #2931?",
	revision: 1,
	ask: { ask_id: "q1", status: "open", deadline_at: 0, question_count: 1, dismissible: true },
	actions: { dismiss: false, answer: true, read: true },
});

function page(patch: Partial<NotificationsPage> = {}): NotificationsPage {
	const items = patch.items ?? [item()];
	const waiting = patch.waiting ?? [askRow];
	return {
		items,
		waiting,
		observed: [...items, ...waiting].map(row => ({ id: row.id, revision: row.revision })),
		next_cursor: null,
		unread_count: items.filter(row => !row.read).length,
		waiting_count: waiting.length,
		waiting_unread_count: waiting.filter(row => !row.read).length,
		high_water: 7,
		...patch,
	};
}

beforeEach(() => {
	capability = null;
	block = undefined;
	localStorage.clear();
	vi.mocked(api.getNotifications).mockResolvedValue(page());
	vi.mocked(api.patchNotifications).mockResolvedValue({ changed: [], skipped: [] });
});

afterEach(() => {
	cleanup();
	vi.clearAllMocks();
	history.replaceState(null, "", "#/");
});

describe("badge formula (D16)", () => {
	it("sums unread updates and unread live asks, and is unknown when either half is", () => {
		expect(notificationsBadge({ unread_count: 3, waiting_count: 4, waiting_unread_count: 2, high_water: 9 })).toBe(5);
		expect(notificationsBadge({})).toBeUndefined();
		expect(notificationsBadge(undefined)).toBeUndefined();
		expect(notificationsBadge({ unread_count: 3 })).toBeUndefined();
	});
});

describe("drawer destination", () => {
	const openDrawer = () => {
		render(<SessionListScreen />);
		fireEvent.click(screen.getByRole("button", { name: "Open navigation" }));
		return screen.getByRole("navigation");
	};

	it("is absent on a relay without the inbox", () => {
		expect(within(openDrawer()).queryByRole("link", { name: /Notifications/ })).toBeNull();
	});

	it("is absent when core advertises the feature but this relay serves no block", () => {
		capability = { features: { notifications: 1 } };
		expect(within(openDrawer()).queryByRole("link", { name: /Notifications/ })).toBeNull();
	});

	it("shows the badge formula and links to the reel when both facts are present", () => {
		capability = { features: { notifications: 1 } };
		block = { unread_count: 2, waiting_count: 3, waiting_unread_count: 1, high_water: 7 };
		const link = within(openDrawer()).getByRole("link", { name: /Notifications/ });
		expect(link.textContent).toContain("3");
		expect(link.getAttribute("href")).toBe("#/notifications");
	});

	it("draws no number while the block is unreadable ({}), never a 0", () => {
		capability = { features: { notifications: 1 } };
		block = {};
		const link = within(openDrawer()).getByRole("link", { name: /Notifications/ });
		expect(link.textContent?.replace("◉", "")).toBe("Notifications");
	});
});

describe("reel", () => {
	it("renders the pinned asks and updates and marks nothing on mount, focus or visibility", async () => {
		render(<NotificationsScreen />);
		expect(await screen.findByText("Waiting on you · 1")).toBeTruthy();
		expect(screen.getByText("Campaign run 41 finished")).toBeTruthy();
		window.dispatchEvent(new Event("focus"));
		document.dispatchEvent(new Event("visibilitychange"));
		fireEvent.scroll(screen.getByRole("main"));
		expect(api.patchNotifications).not.toHaveBeenCalled();
	});

	it("confirms Mark all read and echoes exactly the loaded tokens and boundary", async () => {
		vi.mocked(api.patchNotifications).mockResolvedValue({ changed: ["n_aaaaaaaaaaaaaaaa", "ask:abcdefabcdef:q1"], skipped: [] });
		render(<NotificationsScreen />);
		fireEvent.click(await screen.findByRole("button", { name: "Mark all read" }));
		expect(api.patchNotifications).not.toHaveBeenCalled();
		fireEvent.click(screen.getByRole("button", { name: "Mark read" }));
		await waitFor(() => expect(api.patchNotifications).toHaveBeenCalledTimes(1));
		expect(vi.mocked(api.patchNotifications).mock.calls[0][0]).toEqual({
			op: "read",
			all: true,
			through_seq: 7,
			observed: [
				{ id: "n_aaaaaaaaaaaaaaaa", revision: 1 },
				{ id: "ask:abcdefabcdef:q1", revision: 1 },
			],
		});
		expect(await screen.findByText("Marked 2 read")).toBeTruthy();
	});

	it("dismisses in one tap, names the skipped buckets, and Undo reverts only what changed", async () => {
		const second = item({ id: "n_bbbbbbbbbbbbbbbb", title: "Fraud watch tick", revision: 4 });
		vi.mocked(api.getNotifications).mockResolvedValue(page({ items: [item(), second] }));
		vi.mocked(api.patchNotifications).mockResolvedValueOnce({ changed: ["n_aaaaaaaaaaaaaaaa"], skipped: [] });
		render(<NotificationsScreen />);
		fireEvent.click(await screen.findByRole("button", { name: "Dismiss Campaign run 41 finished" }));
		await screen.findByText("Dismissed 1");
		expect(vi.mocked(api.patchNotifications).mock.calls[0][0]).toEqual({ op: "dismiss", ids: [{ id: "n_aaaaaaaaaaaaaaaa", revision: 1 }] });
		vi.mocked(api.patchNotifications).mockResolvedValueOnce({ changed: ["n_aaaaaaaaaaaaaaaa"], skipped: [] });
		fireEvent.click(screen.getByRole("button", { name: "Undo" }));
		await waitFor(() => expect(api.patchNotifications).toHaveBeenCalledTimes(2));
		expect(vi.mocked(api.patchNotifications).mock.calls[1][0]).toEqual({ op: "undismiss", ids: [{ id: "n_aaaaaaaaaaaaaaaa", revision: 1 }] });
		expect(await screen.findByText("Restored 1")).toBeTruthy();
	});

	it("multi-select dismiss reports the live ask it skipped and a stale revision", async () => {
		vi.mocked(api.patchNotifications).mockResolvedValue({
			changed: [],
			skipped: [{ id: "ask:abcdefabcdef:q1", reason: "live_ask" }, { id: "n_aaaaaaaaaaaaaaaa", reason: "stale_revision" }],
		});
		render(<NotificationsScreen />);
		fireEvent.click(await screen.findByRole("button", { name: "Select" }));
		fireEvent.click(screen.getByRole("button", { name: "Select Merge PR #2931?" }));
		fireEvent.click(screen.getByRole("button", { name: "Select Campaign run 41 finished" }));
		// The button counts only what a dismiss can act on: the live ask is skipped.
		fireEvent.click(screen.getByRole("button", { name: "Dismiss 1" }));
		expect(await screen.findByText(/1 question stays until it's answered · 1 changed since you looked/)).toBeTruthy();
		expect(screen.queryByRole("button", { name: "Undo" })).toBeNull();
	});

	it("a failed write says nothing changed", async () => {
		vi.mocked(api.patchNotifications).mockRejectedValue(new TypeError("Failed to fetch"));
		render(<NotificationsScreen />);
		fireEvent.click(await screen.findByRole("button", { name: "Dismiss Campaign run 41 finished" }));
		expect((await screen.findByRole("alert")).textContent).toBe("Nothing changed: this computer could not be reached. Try again.");
	});

	it("shows the first-load error with a retry, and caught-up on an empty unread tab", async () => {
		vi.mocked(api.getNotifications).mockRejectedValueOnce(new Error("store_unavailable"));
		render(<NotificationsScreen />);
		expect(await screen.findByText("Couldn't load notifications.")).toBeTruthy();
		vi.mocked(api.getNotifications).mockResolvedValue(page({ items: [], waiting: [] }));
		fireEvent.click(screen.getByRole("button", { name: "Try again" }));
		expect(await screen.findByText("All caught up")).toBeTruthy();
	});
});

describe("detail", () => {
	const detail = (patch: Partial<NotificationItem> = {}): NotificationDetail => ({
		item: { ...item(patch), body_md: "**Forty** queries ran.", links: [{ label: "Dashboard", url: "https://example.test/d" }] },
		history: [],
		session: { id: "abcdefabcdef", title: "article search", agent_id: "", live: true, live_state: "idle", cwd: "/w" },
		anchor: { id: "toolu_1", excerpt: "notify: campaign run 41" },
		projects: [{ key: "campaigns", name: "Campaign watch", status: "active" }],
		asks: [],
	});

	it("sends ONE explicit read at the shown revision once the content has loaded", async () => {
		vi.mocked(api.getNotification).mockResolvedValue(detail({ revision: 3 }));
		vi.mocked(api.patchNotifications).mockResolvedValue({ changed: ["n_aaaaaaaaaaaaaaaa"], skipped: [] });
		render(<NotificationDetailScreen itemId="n_aaaaaaaaaaaaaaaa" />);
		expect(await screen.findByText("Forty")).toBeTruthy();
		expect(screen.getByRole("link", { name: "Dashboard" }).getAttribute("href")).toBe("https://example.test/d");
		expect(screen.getByText("Campaign watch")).toBeTruthy();
		await waitFor(() => expect(api.patchNotifications).toHaveBeenCalledTimes(1));
		expect(vi.mocked(api.patchNotifications).mock.calls[0][0]).toEqual({ op: "read", ids: [{ id: "n_aaaaaaaaaaaaaaaa", revision: 3 }] });
	});

	it("sends nothing when the content never loaded, or the item is already read", async () => {
		vi.mocked(api.getNotification).mockRejectedValueOnce(new api.HttpError(404, "unknown notification"));
		render(<NotificationDetailScreen itemId="n_aaaaaaaaaaaaaaaa" />);
		expect(await screen.findByText("This notification is gone")).toBeTruthy();
		cleanup();
		vi.mocked(api.getNotification).mockResolvedValueOnce(detail({ read: true, read_via: "explicit" }));
		render(<NotificationDetailScreen itemId="n_aaaaaaaaaaaaaaaa" />);
		expect(await screen.findByText("Forty")).toBeTruthy();
		expect(api.patchNotifications).not.toHaveBeenCalled();
	});

	it("does not read a detail that loaded while the page was hidden", async () => {
		const original = Object.getOwnPropertyDescriptor(Document.prototype, "visibilityState");
		Object.defineProperty(document, "visibilityState", { configurable: true, get: () => "hidden" });
		try {
			vi.mocked(api.getNotification).mockResolvedValue(detail());
			render(<NotificationDetailScreen itemId="n_aaaaaaaaaaaaaaaa" />);
			expect(await screen.findByText("Forty")).toBeTruthy();
			Object.defineProperty(document, "visibilityState", { configurable: true, get: () => "visible" });
			document.dispatchEvent(new Event("visibilitychange"));
			expect(api.patchNotifications).not.toHaveBeenCalled();
		} finally {
			delete (document as unknown as Record<string, unknown>).visibilityState;
			if (original) Object.defineProperty(Document.prototype, "visibilityState", original);
		}
	});
});
