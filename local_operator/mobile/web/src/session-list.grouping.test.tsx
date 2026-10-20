// @vitest-environment happy-dom
//
// S2b on the session list: the capability-gated grouping toggle, the grouped
// render replacing Active/Previous, the search and flat fallbacks, and the
// catalogue load/error paths. The pure partition itself is covered by
// `lib/session-grouping.test.ts`; these tests prove the screen wires it to the
// relay capability, the local preference and the live rows without writing.
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SessionListScreen } from "./screens/session-list";
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
		.getAllByRole("heading", { level: 2 })
		.map((heading) => heading.textContent ?? "");
}

function section(name: string): HTMLElement {
	const heading = screen.getByRole("heading", { name, level: 2 });
	return heading.closest("section") as HTMLElement;
}

function sectionRows(name: string): string[] {
	/* The card's title is the `flex-1 truncate` span; the button's full text also
	   carries the ★ mark and the tag line, which are not what these assertions
	   are about. */
	return within(section(name))
		.getAllByRole("button")
		.map((button) => button.querySelector("span.flex-1")?.textContent ?? "");
}

function toggle(): HTMLElement {
	return screen.getByRole("button", { name: "Group by primary tag" });
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
});

describe("the capability gate", () => {
	it("offers no control without the relay capability", () => {
		capabilities = null;
		sessionList = [row("a", { primary: "work" })];
		render(<SessionListScreen />);
		expect(screen.queryByRole("button", { name: "Group by primary tag" })).toBeNull();
		expect(headings()).not.toContain("Work");
		expect(getTagsMock).not.toHaveBeenCalled();
	});

	it("offers no control when the capability is present but FALSE", () => {
		capabilities = { session_tags: { available: false } };
		sessionList = [row("a", { primary: "work" })];
		render(<SessionListScreen />);
		expect(screen.queryByRole("button", { name: "Group by primary tag" })).toBeNull();
		expect(headings()).not.toContain("Work");
	});

	it("offers no dead control while the preference is saved but capability is absent", () => {
		localStorage.setItem(KEY, "1");
		capabilities = null;
		sessionList = [row("a", { primary: "work" })];
		render(<SessionListScreen />);
		expect(screen.queryByRole("button", { name: "Group by primary tag" })).toBeNull();
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
		expect(button.getAttribute("aria-pressed")).toBe("false");
		expect(button.textContent).toBe("group");

		fireEvent.click(button);
		expect(localStorage.getItem(KEY)).toBe("1");
		expect(toggle().getAttribute("aria-pressed")).toBe("true");
		expect(toggle().textContent).toBe("grouped");

		fireEvent.click(toggle());
		expect(localStorage.getItem(KEY)).toBeNull();
		expect(toggle().getAttribute("aria-pressed")).toBe("false");
	});

	it("restores a saved preference and groups on mount", async () => {
		localStorage.setItem(KEY, "1");
		capabilities = { session_tags: { available: true } };
		sessionList = [row("a", { primary: "work" })];
		render(<SessionListScreen />);
		expect(await screen.findByRole("heading", { name: "Work" })).toBeTruthy();
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
		await screen.findByRole("heading", { name: "Research" });
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
		await screen.findByRole("heading", { name: "Work" });
		expect(headings()).toEqual(["Work", "Untagged"]);
		expect(sectionRows("Untagged")).toEqual(["gone", "none"]);
	});

	it("keeps ★ Pinned its own top section and out of the groups", async () => {
		sessionList = [
			row("pin", { primary: "work", pinned: true }),
			row("a", { primary: "research" }),
		];
		render(<SessionListScreen />);
		await screen.findByRole("heading", { name: "Research" });
		expect(headings()).toEqual(["★ Pinned", "Research"]);
		expect(sectionRows("★ Pinned")).toEqual(["pin"]);
		expect(sectionRows("Research")).toEqual(["a"]);
	});

	it("drops grouping while the search query is non-empty, then restores it", async () => {
		sessionList = [
			row("alpha", { primary: "research", conversation_name: "Alpha" }),
			row("beta", { primary: "work", conversation_name: "Beta" }),
		];
		render(<SessionListScreen />);
		await screen.findByRole("heading", { name: "Research" });
		const input = screen.getByPlaceholderText("Search conversations…");

		fireEvent.change(input, { target: { value: "alpha" } });
		expect(headings()).toEqual(["Active Sessions"]);
		expect(screen.queryByRole("heading", { name: "Research" })).toBeNull();
		expect(screen.getByRole("button", { name: /Alpha/ })).toBeTruthy();

		fireEvent.change(input, { target: { value: "" } });
		expect(await screen.findByRole("heading", { name: "Research" })).toBeTruthy();
	});

	it("stays flat when no unpinned row has a resolvable primary", async () => {
		sessionList = [
			row("a", { primary: null }),
			row("b", { primary: "deleted" }),
		];
		render(<SessionListScreen />);
		await waitFor(() => expect(getTagsMock).toHaveBeenCalled());
		await flush();
		expect(headings()).toEqual(["Active Sessions"]);
		expect(screen.queryByRole("heading", { name: "Untagged" })).toBeNull();
	});

	it("regroups live when a row's primary changes", async () => {
		sessionList = [row("a", { primary: "research" })];
		const view = render(<SessionListScreen />);
		await screen.findByRole("heading", { name: "Research" });

		sessionList = [row("a", { primary: "work" })];
		view.rerender(<SessionListScreen />);
		await screen.findByRole("heading", { name: "Work" });
		expect(screen.queryByRole("heading", { name: "Research" })).toBeNull();
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
		expect(headings()).toEqual(["Active Sessions"]);
		expect(screen.queryByRole("heading", { name: "Work" })).toBeNull();

		await act(async () => {
			resolveTags({ tags: CATALOGUE });
		});
		expect(await screen.findByRole("heading", { name: "Work" })).toBeTruthy();
	});

	it("keeps the flat view after a read error and offers a working retry", async () => {
		getTagsMock.mockRejectedValueOnce(new Error("offline"));
		sessionList = [row("a", { primary: "work" })];
		render(<SessionListScreen />);
		const retry = await screen.findByRole("button", {
			name: "Retry loading the tag catalogue",
		});
		expect(headings()).toEqual(["Active Sessions"]);

		getTagsMock.mockResolvedValueOnce({ tags: CATALOGUE });
		fireEvent.click(retry);
		expect(await screen.findByRole("heading", { name: "Work" })).toBeTruthy();
		expect(
			screen.queryByRole("button", { name: "Retry loading the tag catalogue" }),
		).toBeNull();
	});
});

describe("no writes", () => {
	it("toggling and searching never call a relay write", async () => {
		localStorage.setItem(KEY, "1");
		capabilities = { session_tags: { available: true } };
		sessionList = [row("alpha", { primary: "work", conversation_name: "Alpha" })];
		render(<SessionListScreen />);
		await screen.findByRole("heading", { name: "Work" });
		fireEvent.click(toggle());
		fireEvent.click(toggle());
		fireEvent.change(screen.getByPlaceholderText("Search conversations…"), {
			target: { value: "alp" },
		});
		expect(setSessionTagsMock).not.toHaveBeenCalled();
		expect(setSessionPinMock).not.toHaveBeenCalled();
	});
});
