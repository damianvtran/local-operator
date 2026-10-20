// @vitest-environment happy-dom
//
// Conversation tags on the SESSION LIST: the static row line, and the two ways
// it is reached — never as a control inside the row button, and only through
// the long-press sheet when the RELAY advertises the capability.
//
// The capability cases are the important ones: `features.session_tags` exists on
// released relays that have none of these routes, so a build gated on it would
// offer a screen whose requests 404. `capabilities.session_tags.available` is
// the only gate, and absence must leave NO control behind.
import { act, cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SessionListScreen } from "./screens/session-list";
import type { Capabilities, SessionSummary, TagCatalogueEntry } from "./types";

let sessionList: SessionSummary[] = [];
let capabilities: Capabilities | null = null;

const CATALOGUE: TagCatalogueEntry[] = [
	{ id: "work", name: "Work", order: 0, conversations: 1 },
	{ id: "research", name: "Research", order: 1, conversations: 1 },
];

const getTags = vi.fn(async () => ({ tags: CATALOGUE }));

vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useSessions: () => ({ sessions: sessionList, connected: true }),
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
		setSessionPin: vi.fn(async () => ({ ok: true, pinned: true })),
		getTags: () => getTags(),
		setSessionTags: vi.fn(),
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

function cardByName(name: string): HTMLElement {
	return screen.getByRole("button", { name: new RegExp(name) });
}

/* The pinned suite's long-press helper: hold past the threshold, lift, and let
   the platform's release click hit the scrim (which the sheet swallows). */
function longPress(card: HTMLElement) {
	vi.useFakeTimers();
	fireEvent.pointerDown(card, { clientX: 10, clientY: 10 });
	act(() => {
		vi.advanceTimersByTime(600);
	});
	vi.useRealTimers();
	fireEvent.pointerUp(card);
	fireEvent.click(screen.queryByTestId("sheet-scrim") ?? card);
}

beforeEach(() => {
	sessionList = [];
	capabilities = null;
	getTags.mockClear();
});

afterEach(() => {
	cleanup();
});

describe("the row's tag line", () => {
	it("renders no line for an untagged conversation, so the row keeps its height", () => {
		sessionList = [summary({ session_id: "a", conversation_name: "Alpha" })];
		render(<SessionListScreen />);
		expect(within(cardByName("Alpha")).queryByTestId("session-tags")).toBeNull();
	});

	it("renders one label with no count", () => {
		sessionList = [
			summary({
				session_id: "a",
				conversation_name: "Alpha",
				tags: [{ id: "work", name: "Work" }],
				primary: null,
			}),
		];
		render(<SessionListScreen />);
		const line = within(cardByName("Alpha")).getByTestId("session-tags");
		expect(line.textContent).toContain("Work");
		expect(line.textContent).not.toContain("+");
	});

	it("renders two labels and compresses past them to +N", () => {
		sessionList = [
			summary({
				session_id: "a",
				conversation_name: "Alpha",
				tags: [
					{ id: "work", name: "Work" },
					{ id: "research", name: "Research" },
					{ id: "home", name: "Home" },
					{ id: "later", name: "Later" },
				],
				primary: null,
			}),
		];
		render(<SessionListScreen />);
		const line = within(cardByName("Alpha")).getByTestId("session-tags");
		expect(line.textContent).toContain("Work");
		expect(line.textContent).toContain("Research");
		expect(line.textContent).not.toContain("Home");
		expect(line.textContent).toContain("+2");
		/* The count's accessible name says what the mark means. */
		expect(within(line).getByLabelText("2 more tags")).toBeTruthy();
	});

	it("truncates a long label instead of pushing the count off the row", () => {
		const long = "A category name long enough to run past any phone row";
		sessionList = [
			summary({
				session_id: "a",
				conversation_name: "Alpha",
				tags: [
					{ id: "work", name: long },
					{ id: "research", name: "Research" },
					{ id: "home", name: "Home" },
				],
				primary: null,
			}),
		];
		render(<SessionListScreen />);
		const line = within(cardByName("Alpha")).getByTestId("session-tags");
		expect(within(line).getByText(long).className).toContain("truncate");
		expect(within(line).getByText("+1")).toBeTruthy();
	});

	it("marks the primary with a dot and a screen-reader word, not colour alone", () => {
		sessionList = [
			summary({
				session_id: "a",
				conversation_name: "Alpha",
				tags: [
					{ id: "work", name: "Work" },
					{ id: "research", name: "Research" },
				],
				primary: "research",
			}),
		];
		render(<SessionListScreen />);
		const line = within(cardByName("Alpha")).getByTestId("session-tags");
		expect(within(line).getByText("primary")).toBeTruthy();
		expect(line.textContent).toContain("•");
	});

	it("carries no control inside the row button", () => {
		sessionList = [
			summary({
				session_id: "a",
				conversation_name: "Alpha",
				tags: [{ id: "work", name: "Work" }],
				primary: "work",
			}),
		];
		render(<SessionListScreen />);
		const line = within(cardByName("Alpha")).getByTestId("session-tags");
		expect(within(line).queryByRole("button")).toBeNull();
		expect(within(line).queryByRole("checkbox")).toBeNull();
	});
});

describe("reaching the assignment sheet", () => {
	it("offers no Tags action without the RELAY capability", async () => {
		capabilities = { features: { session_tags: 1 } };
		sessionList = [summary({ session_id: "a", conversation_name: "Alpha" })];
		render(<SessionListScreen />);
		longPress(cardByName("Alpha"));
		expect(await screen.findByRole("button", { name: "Pin to the top" })).toBeTruthy();
		expect(screen.queryByRole("button", { name: "Tags…" })).toBeNull();
	});

	it("closes the action sheet and opens the tag sheet when the capability is present", async () => {
		capabilities = { session_tags: { available: true } };
		sessionList = [
			summary({
				session_id: "a",
				conversation_name: "Alpha",
				tags: [{ id: "work", name: "Work" }],
				primary: "work",
			}),
		];
		render(<SessionListScreen />);
		longPress(cardByName("Alpha"));
		fireEvent.click(await screen.findByRole("button", { name: "Tags…" }));

		/* One panel at a time: the pin action sheet is gone. */
		expect(screen.queryByRole("button", { name: "Pin to the top" })).toBeNull();
		const work = (await screen.findByRole("checkbox", {
			name: "Work",
		})) as HTMLInputElement;
		expect(work.checked).toBe(true);
		expect(screen.getByText("tags")).toBeTruthy();
	});

	it("keeps the pin action working beside the new one", async () => {
		capabilities = { session_tags: { available: true } };
		sessionList = [summary({ session_id: "a", conversation_name: "Alpha" })];
		render(<SessionListScreen />);
		longPress(cardByName("Alpha"));
		expect(await screen.findByRole("button", { name: "Pin to the top" })).toBeTruthy();
		expect(await screen.findByRole("button", { name: "Tags…" })).toBeTruthy();
	});
});
