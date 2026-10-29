// @vitest-environment happy-dom
//
// S2b on the session list: the capability-gated grouping toggle, the grouped
// render replacing Active/Previous, the search and flat fallbacks, and the
// catalogue load/error paths. The pure partition itself is covered by
// `lib/session-grouping.test.ts`; these tests prove the screen wires it to the
// relay capability, the local preference and the live rows without writing.
import { act, cleanup, fireEvent, render as renderView, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SessionListScreen } from "./screens/session-list";
import { SessionGroupToggle } from "./components/session-group-toggle";
import type { Capabilities, SessionSummary, TagCatalogueEntry } from "./types";

const KEY = "lo-mobile-group-tags";

let sessionList: SessionSummary[] = [];
let capabilities: Capabilities | null = null;

const CATALOGUE: TagCatalogueEntry[] = [
	{ id: "work", name: "Work", order: 2, conversations: 2 },
	{ id: "research", name: "Research", order: 0, conversations: 1 },
	{ id: "home", name: "Home", order: 1, conversations: 1 },
];

const getTagsMock = vi.fn(async (): Promise<{ tags: TagCatalogueEntry[] }> => ({
	tags: CATALOGUE,
}));
const setSessionTagsMock = vi.fn();
const setSessionPinMock = vi.fn(async () => ({ ok: true, pinned: true }));

vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useSessions: () => ({ sessions: sessionList, connected: true, revision: 0 }),
		useCapabilities: () => capabilities,
		retainSessionListStream: () => () => {},
		applySessionPin: () => {},
	};
});

vi.mock("./api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./api")>();
	return {
		...actual,
		getDirectories: vi.fn(async () => ({ home: "", recent: [] })),
		getTags: () => getTagsMock(),
		setSessionPin: () => setSessionPinMock(),
		setSessionTags: () => setSessionTagsMock(),
	};
});

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
		pinned: false,
		...over,
	};
}

function row(
	id: string,
	over: Partial<SessionSummary> = {},
): SessionSummary {
	return summary({ session_id: id, conversation_name: id, ...over });
}

function headings(): string[] {
	return screen
		.getAllByRole("heading", { level: 2, hidden: true })
		.map((heading) => heading.textContent ?? "");
}

function section(name: string): HTMLElement {
	const heading = screen.getByRole("heading", { name, level: 2, hidden: true });
	return heading.closest("section") as HTMLElement;
}

function sectionRows(name: string): string[] {
	/* The card's title is the `flex-1 truncate` span; the button's full text also
	   carries the ★ mark and the tag line, which are not what these assertions
	   are about. */
	return within(section(name))
		.getAllByRole("button", { hidden: true }).filter(button => button.hasAttribute("aria-haspopup"))
		.map((button) => button.querySelector("span.flex-1")?.textContent ?? "");
}

function toggle(): HTMLElement {
	openAppearance();
	return screen.getByRole("button", { name: "Group conversations by tag" });
}

async function flush(): Promise<void> {
	await act(async () => {});
}

beforeEach(() => {
	sessionList = [];
	capabilities = null;
	localStorage.clear();
	getTagsMock.mockReset();
	getTagsMock.mockResolvedValue({ tags: CATALOGUE });
	setSessionTagsMock.mockReset();
	setSessionPinMock.mockReset();
});

afterEach(() => {
	cleanup();
	vi.restoreAllMocks();
});

describe("the capability gate", () => {
	it("offers no control without the relay capability", () => {
		capabilities = null;
		sessionList = [row("a", { primary: "work" })];
		render(<SessionListScreen />);
		expect(screen.queryByRole("button", { name: "Group conversations by tag" })).toBeNull();
		expect(headings()).not.toContain("Work");
		expect(getTagsMock).not.toHaveBeenCalled();
	});

	it("offers no control when the capability is present but FALSE", () => {
		capabilities = { session_tags: { available: false } };
		sessionList = [row("a", { primary: "work" })];
		render(<SessionListScreen />);
		expect(screen.queryByRole("button", { name: "Group conversations by tag" })).toBeNull();
		expect(headings()).not.toContain("Work");
	});

	it("offers no dead control while the preference is saved but capability is absent", () => {
		localStorage.setItem(KEY, "1");
		capabilities = null;
		sessionList = [row("a", { primary: "work" })];
		render(<SessionListScreen />);
		expect(screen.queryByRole("button", { name: "Group conversations by tag" })).toBeNull();
		expect(headings()).not.toContain("Work");
		expect(getTagsMock).not.toHaveBeenCalled();
	});
});

describe("the preference", () => {
	it("persists on, and turning it off removes the key", async () => {
		capabilities = { session_tags: { available: true } };
		sessionList = [row("a", { primary: "work" })];
		render(<SessionListScreen />);
		const button = toggle();
		/* THE ONE STATIC LABEL, and the pressed treatment carries the state (round
		   1, D1/U1): the words no longer change, so the class pair is what a reader
		   actually sees. */
		expect(button.getAttribute("aria-pressed")).toBe("false");
		expect(button.textContent).toBe("Group conversations by tag");
		expect(button.className).toContain("bg-surface");
		expect(button.className).not.toContain("border-accent");

		fireEvent.click(button);
		expect(localStorage.getItem(KEY)).toBe("1");
		expect(toggle().getAttribute("aria-pressed")).toBe("true");
		expect(toggle().textContent).toBe("Group conversations by tag");
		expect(toggle().className).not.toContain("border-accent");
		expect(toggle().className).toContain("bg-accent-wash");
		expect(toggle().className).toContain("text-accent");

		fireEvent.click(toggle());
		expect(localStorage.getItem(KEY)).toBeNull();
		expect(toggle().getAttribute("aria-pressed")).toBe("false");
		expect(toggle().className).not.toContain("border-accent");
	});

	it("restores a saved preference and groups on mount", async () => {
		localStorage.setItem(KEY, "1");
		capabilities = { session_tags: { available: true } };
		sessionList = [row("a", { primary: "work" })];
		render(<SessionListScreen />);
		expect(await screen.findByRole("heading", { name: "Work", hidden: true })).toBeTruthy();
		expect(toggle().getAttribute("aria-pressed")).toBe("true");
	});
});

describe("grouped rendering", () => {
	beforeEach(() => {
		localStorage.setItem(KEY, "1");
		capabilities = { session_tags: { available: true } };
	});

	it("orders groups by catalogue order and keeps daemon row order", async () => {
		sessionList = [
			row("work-b", { primary: "work" }),
			row("research-a", { primary: "research" }),
			row("work-a", { primary: "work" }),
			row("home-a", { primary: "home" }),
		];
		render(<SessionListScreen />);
		await screen.findByRole("heading", { name: "Research", hidden: true });
		expect(headings()).toEqual(["Research", "Home", "Work"]);
		expect(sectionRows("Work")).toEqual(["work-b", "work-a"]);
	});

	it("puts unknown and deleted primaries in Untagged, last", async () => {
		sessionList = [
			row("gone", { primary: "deleted" }),
			row("tagged", { primary: "work" }),
			row("none", { primary: null }),
		];
		render(<SessionListScreen />);
		await screen.findByRole("heading", { name: "Work", hidden: true });
		expect(headings()).toEqual(["Work", "Untagged"]);
		expect(sectionRows("Untagged")).toEqual(["gone", "none"]);
	});

	it("keeps ★ Pinned its own top section and out of the groups", async () => {
		sessionList = [
			row("pin", { primary: "work", pinned: true }),
			row("a", { primary: "research" }),
		];
		render(<SessionListScreen />);
		await screen.findByRole("heading", { name: "Research", hidden: true });
		expect(headings()).toEqual(["Pinned", "Research"]);
		expect(sectionRows("Pinned")).toEqual(["pin"]);
		expect(sectionRows("Research")).toEqual(["a"]);
	});

	it("pauses grouping while the search query is non-empty, then restores it", async () => {
		sessionList = [
			row("alpha", { primary: "research", conversation_name: "Alpha" }),
			row("beta", { primary: "work", conversation_name: "Beta" }),
		];
		render(<SessionListScreen />);
		await screen.findByRole("heading", { name: "Research", hidden: true });
		expect(screen.queryByText("grouping paused while searching")).toBeNull();
		const input = openSearch();

		fireEvent.change(input, { target: { value: "alpha" } });
		expect(headings()).toEqual(["Active"]);
		expect(screen.queryByRole("heading", { name: "Research", hidden: true })).toBeNull();
		expect(screen.getByRole("button", { name: /Alpha/ })).toBeTruthy();
		/* THE PREFERENCE SURVIVES THE SEARCH (round 1, U2/Q-2): a hint explains the
		   flat list, and the toggle keeps saying it is on rather than flipping
		   itself off behind the reader's back. It stays a working control. */
		openAppearance();
		expect(screen.getByText("grouping paused while searching").getAttribute("role")).toBe(
			"status",
		);
		expect(toggle().getAttribute("aria-pressed")).toBe("true");
		expect((toggle() as HTMLButtonElement).disabled).toBe(false);

		fireEvent.click(screen.getByRole("button", { name: "close sheet" }));
		fireEvent.change(input, { target: { value: "" } });
		expect(await screen.findByRole("heading", { name: "Research", hidden: true })).toBeTruthy();
		expect(screen.queryByText("grouping paused while searching")).toBeNull();
	});

	it("stays flat when no unpinned row has a resolvable primary", async () => {
		sessionList = [
			row("a", { primary: null }),
			row("b", { primary: "deleted" }),
		];
		render(<SessionListScreen />);
		await waitFor(() => expect(getTagsMock).toHaveBeenCalled());
		await flush();
		expect(headings()).toEqual(["Active"]);
		expect(screen.queryByRole("heading", { name: "Untagged" })).toBeNull();
	});

	it("regroups live when a row's primary changes", async () => {
		sessionList = [row("a", { primary: "research" })];
		const view = render(<SessionListScreen />);
		await screen.findByRole("heading", { name: "Research", hidden: true });

		sessionList = [row("a", { primary: "work" })];
		view.rerender(<SessionListScreen />);
		await screen.findByRole("heading", { name: "Work", hidden: true });
		expect(screen.queryByRole("heading", { name: "Research", hidden: true })).toBeNull();
		/* ONE catalogue read for the session, not one per list frame. */
		expect(getTagsMock).toHaveBeenCalledTimes(1);
	});
});

describe("the catalogue", () => {
	beforeEach(() => {
		localStorage.setItem(KEY, "1");
		capabilities = { session_tags: { available: true } };
	});

	it("renders the flat view while the catalogue loads, then groups", async () => {
		let resolveTags: (value: { tags: TagCatalogueEntry[] }) => void = () => {};
		getTagsMock.mockReturnValue(
			new Promise((resolve) => {
				resolveTags = resolve;
			}),
		);
		sessionList = [row("a", { primary: "work" })];
		render(<SessionListScreen />);
		expect(headings()).toEqual(["Active"]);
		expect(screen.queryByRole("heading", { name: "Work", hidden: true })).toBeNull();
		/* THE FIRST READ IS VISIBLE WITHOUT BLOCKING (round 1, D3/Q-1): the list is
		   the flat one it would be without the preference, plus one status line. */
		expect(screen.getByText("loading tags…").getAttribute("role")).toBe("status");

		await act(async () => {
			resolveTags({ tags: CATALOGUE });
		});
		expect(await screen.findByRole("heading", { name: "Work", hidden: true })).toBeTruthy();
		expect(screen.queryByText("loading tags…")).toBeNull();
	});

	it("keeps the flat view after a read error and offers a working retry", async () => {
		getTagsMock.mockRejectedValueOnce(new Error("offline"));
		sessionList = [row("a", { primary: "work" })];
		render(<SessionListScreen />);
		const retry = await screen.findByRole("button", { name: "Retry tags" });
		expect(retry.textContent).toBe("retry tags");
		expect(headings()).toEqual(["Active"]);
		/* THE FAILURE IS STATED IN PLACE (round 1, D2/Q-2), beside the control that
		   acts on it, and never as the `loading tags…` line that preceded it. */
		expect(screen.getByText("could not load tags").getAttribute("role")).toBe("status");
		expect(screen.queryByText("loading tags…")).toBeNull();

		/* THE RETRY TRANSITION, deferred so the in-flight state is observable:
		   clicking retry drops the failure at once and names the new read, while
		   the flat list stays exactly as it was. */
		let resolveRetry: (value: { tags: TagCatalogueEntry[] }) => void = () => {};
		getTagsMock.mockReturnValue(
			new Promise((resolve) => {
				resolveRetry = resolve;
			}),
		);
		fireEvent.click(retry);
		expect(screen.queryByText("could not load tags")).toBeNull();
		expect(screen.getByText("loading tags…").getAttribute("role")).toBe("status");
		expect(headings()).toEqual(["Active"]);

		await act(async () => {
			resolveRetry({ tags: CATALOGUE });
		});
		expect(await screen.findByRole("heading", { name: "Work", hidden: true })).toBeTruthy();
		expect(screen.queryByRole("button", { name: "Retry tags" })).toBeNull();
		/* Both states are gone: no failure text, and no status line at all. */
		expect(screen.queryByText("could not load tags")).toBeNull();
		expect(screen.queryByRole("status")).toBeNull();
	});

	it("shows loading tags… while a retry runs against a cached catalogue", async () => {
		/* QA round 2, Q-r2-1: the cached catalogue keeps painting, so the read is
		   invisible unless `busy` tracks the ATTEMPT rather than `catalogue ===
		   null`. This is the case that regressed when busy was derived from the
		   cache's absence. */
		sessionList = [row("a", { primary: "work" })];
		render(<SessionListScreen />);
		await screen.findByRole("heading", { name: "Work", hidden: true });

		vi.spyOn(document, "visibilityState", "get").mockReturnValue("visible");
		getTagsMock.mockRejectedValueOnce(new Error("500"));
		await act(async () => {
			document.dispatchEvent(new Event("visibilitychange"));
		});
		const retry = await screen.findByRole("button", { name: "Retry tags" });
		expect(screen.getByText("could not load tags")).toBeTruthy();
		expect(screen.queryByRole("heading", { name: "Work", hidden: true })).toBeNull();

		let resolveRetry: (value: { tags: TagCatalogueEntry[] }) => void = () => {};
		getTagsMock.mockReturnValue(
			new Promise((resolve) => {
				resolveRetry = resolve;
			}),
		);
		fireEvent.click(retry);
		/* The stale groups are back, and the new read is named beside them. */
		expect(screen.queryByText("could not load tags")).toBeNull();
		expect(screen.getByText("loading tags…").getAttribute("role")).toBe("status");
		expect(screen.getByRole("heading", { name: "Work", hidden: true })).toBeTruthy();

		await act(async () => {
			resolveRetry({ tags: CATALOGUE });
		});
		expect(screen.queryByText("loading tags…")).toBeNull();
		expect(screen.getByRole("heading", { name: "Work", hidden: true })).toBeTruthy();
	});

	it("shows loading tags… while a refocus read runs, keeping the cached groups", async () => {
		/* Refocus truthfulness: the same attempt-based flag means a background
		   refresh is visible too, without replacing what is already painted. */
		sessionList = [row("a", { primary: "work" })];
		render(<SessionListScreen />);
		await screen.findByRole("heading", { name: "Work", hidden: true });

		let resolveRefresh: (value: { tags: TagCatalogueEntry[] }) => void = () => {};
		getTagsMock.mockReturnValueOnce(
			new Promise((resolve) => {
				resolveRefresh = resolve;
			}),
		);
		vi.spyOn(document, "visibilityState", "get").mockReturnValue("visible");
		await act(async () => {
			document.dispatchEvent(new Event("visibilitychange"));
		});
		expect(screen.getByRole("heading", { name: "Work", hidden: true })).toBeTruthy();
		expect(screen.getByText("loading tags…").getAttribute("role")).toBe("status");

		await act(async () => {
			resolveRefresh({ tags: CATALOGUE });
		});
		expect(screen.queryByText("loading tags…")).toBeNull();
		expect(screen.getByRole("heading", { name: "Work", hidden: true })).toBeTruthy();
	});

	it("refreshes the catalogue when the page becomes visible, not while hidden", async () => {
		/* SAME PATTERN AS `session-view.seen.test.tsx`: the effect reads
		   `document.visibilityState` when the event fires, so mocking the getter
		   decides which branch runs. */
		const visibility = vi
			.spyOn(document, "visibilityState", "get")
			.mockReturnValue("visible");
		sessionList = [row("a", { primary: "work" })];
		render(<SessionListScreen />);
		await screen.findByRole("heading", { name: "Work", hidden: true });
		expect(getTagsMock).toHaveBeenCalledTimes(1);

		visibility.mockReturnValue("hidden");
		await act(async () => {
			document.dispatchEvent(new Event("visibilitychange"));
		});
		expect(getTagsMock).toHaveBeenCalledTimes(1);

		visibility.mockReturnValue("visible");
		await act(async () => {
			document.dispatchEvent(new Event("visibilitychange"));
		});
		await waitFor(() => expect(getTagsMock).toHaveBeenCalledTimes(2));
	});
});

describe("the toggle's status precedence", () => {
	/* The screen rarely holds two of these at once — a retry clears the error
	   before the read starts — so the order itself is pinned here, where every
	   combination is constructible. Error > paused > loading: each later state
	   makes the one before it meaningless, and both adjacent pairs are asserted
	   so an inverted branch fails without needing a contrived screen state. */
	it.each([
		{ failed: true, paused: true, busy: true, text: "could not load tags" },
		{ failed: false, paused: true, busy: true, text: "grouping paused while searching" },
		{ failed: false, paused: false, busy: true, text: "loading tags…" },
	])("shows $text when failed=$failed paused=$paused busy=$busy", ({ failed, paused, busy, text }) => {
		render(
			<SessionGroupToggle
				grouped
				busy={busy}
				failed={failed}
				paused={paused}
				onToggle={() => {}}
				onRetry={() => {}}
			/>,
		);
		expect(screen.getByRole("status").textContent).toBe(text);
	});
});

describe("no writes", () => {
	it("toggling and searching never call a relay write", async () => {
		localStorage.setItem(KEY, "1");
		capabilities = { session_tags: { available: true } };
		sessionList = [row("alpha", { primary: "work", conversation_name: "Alpha" })];
		render(<SessionListScreen />);
		await screen.findByRole("heading", { name: "Work", hidden: true });
		fireEvent.click(toggle());
		fireEvent.click(toggle());
		fireEvent.change(openSearch(), {
			target: { value: "alp" },
		});
		expect(setSessionTagsMock).not.toHaveBeenCalled();
		expect(setSessionPinMock).not.toHaveBeenCalled();
	});
});

/** Search now expands in the app bar; exercise its real opening control. */
function openSearch() {
 const close = screen.queryByRole("button", { name: "close sheet" });
 if (close) fireEvent.click(close);
 const field = screen.queryByPlaceholderText("Search conversations…");
 if (field) return field;
 fireEvent.click(screen.getByRole("button", { name: "Search", hidden: true }));
 return screen.getByPlaceholderText("Search conversations…");
}

/** The preference moved to Appearance; these tests retain their membership,
 * persistence and read/refusal assertions while using that real two-tap path.
 * Section assertions read rendered background membership while the modal is
 * open; modal focus/inert behavior is covered by the primitive tests. */
function openAppearance() {
 if (screen.queryByRole("button", { name: "close sheet" })) return;
 fireEvent.click(screen.getByRole("button", { name: "Open navigation" }));
 fireEvent.click(screen.getByRole("button", { name: "Appearance" }));
}
function render(ui: React.ReactElement) {
 const view = renderView(ui);
 if (ui.type === SessionListScreen) openAppearance();
 return view;
}
