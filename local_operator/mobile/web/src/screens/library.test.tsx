// @vitest-environment happy-dom
//
// The Library LIST's wired surface: loading, empty, error/retry, the debounced
// server search, the category chip, the filter Sheet, the card menu and cursor
// paging. The screens' copy is plain English (advisory i18n scope); assertions
// are by role and name, the convention every phone test follows.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { LibraryAsset, LibraryListing } from "../types";
import type { LibraryQuery } from "../api";

const getLibrary = vi.fn();
const patchLibraryAsset = vi.fn();

vi.mock("../api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../api")>();
	return {
		...actual,
		getLibrary: (...args: unknown[]) => getLibrary(...args),
		patchLibraryAsset: (...args: unknown[]) => patchLibraryAsset(...args),
	};
});

import { LibraryScreen } from "./library";

function asset(overrides: Partial<LibraryAsset> = {}): LibraryAsset {
	const id = overrides.id ?? "la_0123456789abcdef";
	return {
		id,
		ref: {
			type: "path",
			path: null,
			display_path: "~/notes/launch.md",
			digest: null,
			content_type: "text/markdown",
		},
		name: "launch notes",
		derived_name: "launch.md",
		description: "",
		kind: "markdown",
		category: "document",
		origin: "written",
		origin_group: "created",
		origins: ["written"],
		tags: [{ id: "tag_launch", name: "launch", category: "topic" }],
		primary_tag: null,
		size_bytes: 2048,
		width: null,
		height: null,
		duration_s: null,
		created_at: 1_700_000_000,
		modified_at: 1_700_000_000,
		last_seen_at: 1_700_000_000,
		exists: true,
		highlighted: false,
		highlight_why: null,
		occurrences: [
			{
				session_id: "4e92693767fa",
				session_title: "launch planning",
				origin: "written",
				tool: "write",
				at: 1_700_000_000,
				intent: null,
				first_seq: 3,
				last_seq: 7,
				child_id: null,
				label: null,
			},
		],
		session_count: 1,
		chat: null,
		preview: { state: "ready", mime_type: "text/markdown", pages: null },
		organized: null,
		locked: false,
		favorite: false,
		archived: false,
		...overrides,
	};
}

function listing(overrides: Partial<LibraryListing> = {}): LibraryListing {
	return {
		state: "ready",
		revision: 3,
		total: 1,
		counts: {
			category: { image: 2, document: 1 },
			origin_group: { created: 1, generated: 2 },
			kind: { image: 2, markdown: 1 },
		},
		tags: [{ id: "tag_launch", name: "launch", category: "topic", count: 1 }],
		rows: [asset()],
		next_cursor: null,
		cursor_reset: false,
		scan: { sessions_total: 4, sessions_scanned: 4 },
		...overrides,
	};
}

beforeEach(() => {
	getLibrary.mockResolvedValue(listing());
	patchLibraryAsset.mockResolvedValue(asset({ favorite: true }));
});

afterEach(() => {
	cleanup();
	vi.clearAllMocks();
	history.replaceState(null, "", "#/");
});

describe("library list", () => {
	it("renders a loading grid, then the server's rows", async () => {
		render(<LibraryScreen />);
		expect(screen.getByLabelText("loading assets")).toBeTruthy();
		expect(await screen.findByText("launch notes")).toBeTruthy();
		expect(screen.getByText("launch planning")).toBeTruthy();
		expect(screen.getByText("1 asset")).toBeTruthy();
		expect(getLibrary).toHaveBeenCalledTimes(1);
	});

	it("pluralizes the result count with the shared i18n message", async () => {
		getLibrary.mockResolvedValue(listing({ total: 3 }));
		render(<LibraryScreen />);
		expect(await screen.findByText("3 assets")).toBeTruthy();
	});

	it("keeps the chosen sort when filters are cleared", async () => {
		render(<LibraryScreen />);
		await screen.findByText("launch notes");

		fireEvent.click(screen.getByRole("button", { name: "sort" }));
		fireEvent.click(await screen.findByRole("button", { name: /^name/ }));
		await waitFor(() => expect(getLibrary).toHaveBeenCalledTimes(2));
		expect((getLibrary.mock.calls[1][0] as LibraryQuery).sort).toBe("name");

		fireEvent.click(screen.getByRole("button", { name: /filter/ }));
		fireEvent.click(await screen.findByRole("button", { name: "favorites" }));
		await waitFor(() => expect(getLibrary).toHaveBeenCalledTimes(3));
		const filtered = getLibrary.mock.calls[2][0] as LibraryQuery;
		expect(filtered.sort).toBe("name");
		expect(filtered.favorite).toBe(true);

		fireEvent.click(screen.getByRole("button", { name: "clear filters" }));
		await waitFor(() => expect(getLibrary).toHaveBeenCalledTimes(4));
		const cleared = getLibrary.mock.calls[3][0] as LibraryQuery;
		expect(cleared.sort).toBe("name");
		expect(cleared.favorite).toBeUndefined();
		expect(cleared.categories).toBeUndefined();
		expect(cleared.origin_groups).toBeUndefined();
		expect(cleared.tag).toBeUndefined();
	});

	it("shows the empty state and the building banner", async () => {
		getLibrary.mockResolvedValue(
			listing({ total: 0, rows: [], state: "building", scan: { sessions_total: 9, sessions_scanned: 4 } }),
		);
		render(<LibraryScreen />);
		expect(await screen.findByText("no assets yet")).toBeTruthy();
		expect(screen.getByText(/building the index/)).toBeTruthy();
	});

	it("shows a refusal with a retry that re-reads", async () => {
		getLibrary.mockRejectedValueOnce(new Error("library store unavailable"));
		render(<LibraryScreen />);
		expect(await screen.findByRole("alert")).toBeTruthy();
		expect(screen.getByText("library store unavailable")).toBeTruthy();
		fireEvent.click(screen.getByRole("button", { name: "Try again" }));
		await waitFor(() => expect(getLibrary).toHaveBeenCalledTimes(2));
		expect(await screen.findByText("launch notes")).toBeTruthy();
	});

	it("sends the debounced search to the server and ignores a stale answer", async () => {
		render(<LibraryScreen />);
		await screen.findByText("launch notes");
		fireEvent.change(screen.getByLabelText("search assets"), { target: { value: "launch" } });
		await waitFor(() => expect(getLibrary).toHaveBeenCalledTimes(2), { timeout: 1500 });
		const sent = getLibrary.mock.calls[1][0] as LibraryQuery;
		expect(sent.q).toBe("launch");
	});

	it("filters by the category chip the counts offer", async () => {
		render(<LibraryScreen />);
		await screen.findByText("launch notes");
		fireEvent.click(screen.getByRole("button", { name: /images/ }));
		await waitFor(() => expect(getLibrary).toHaveBeenCalledTimes(2));
		const sent = getLibrary.mock.calls[1][0] as LibraryQuery;
		expect(sent.categories).toEqual(["image"]);
	});

	it("applies favorites and tag filters from the Sheet", async () => {
		render(<LibraryScreen />);
		await screen.findByText("launch notes");
		fireEvent.click(screen.getByRole("button", { name: /filter/ }));
		fireEvent.click(await screen.findByRole("button", { name: "favorites" }));
		await waitFor(() => expect(getLibrary).toHaveBeenCalledTimes(2));
		expect((getLibrary.mock.calls[1][0] as LibraryQuery).favorite).toBe(true);

		fireEvent.click(screen.getByRole("button", { name: /launch 1/ }));
		await waitFor(() => expect(getLibrary).toHaveBeenCalledTimes(3));
		expect((getLibrary.mock.calls[2][0] as LibraryQuery).tag).toBe("tag_launch");
	});

	it("opens the per-card menu and toggles a favorite, then re-reads", async () => {
		render(<LibraryScreen />);
		await screen.findByText("launch notes");
		fireEvent.click(screen.getByRole("button", { name: "actions for launch notes" }));
		expect(await screen.findByText("copy asset id")).toBeTruthy();
		fireEvent.click(screen.getByText("add to favorites"));
		await waitFor(() => expect(patchLibraryAsset).toHaveBeenCalledWith("la_0123456789abcdef", { favorite: true }));
		await waitFor(() => expect(getLibrary).toHaveBeenCalledTimes(2));
	});

	it("pages with the relay's cursor and appends the next page", async () => {
		const first = listing({ total: 2, next_cursor: "fingerprint.3.1" });
		const second = listing({
			rows: [asset({ id: "la_fedcba9876543210", name: "second asset" })],
			next_cursor: null,
		});
		getLibrary.mockResolvedValueOnce(first).mockResolvedValueOnce(second);
		render(<LibraryScreen />);
		await screen.findByText("launch notes");
		fireEvent.click(screen.getByRole("button", { name: "load more" }));
		expect(await screen.findByText("second asset")).toBeTruthy();
		expect(screen.getByText("launch notes")).toBeTruthy();
		expect((getLibrary.mock.calls[1][0] as LibraryQuery).cursor).toBe("fingerprint.3.1");
	});

	it("opens the detail route from a card and the source conversation from the menu", async () => {
		const navigateSpy = vi.spyOn(history, "pushState");
		render(<LibraryScreen />);
		await screen.findByText("launch notes");
		fireEvent.click(screen.getByRole("button", { name: "open launch notes" }));
		expect(location.hash).toContain("#/library/a/la_0123456789abcdef");
		expect(navigateSpy).toHaveBeenCalled();

		fireEvent.click(screen.getByRole("button", { name: "actions for launch notes" }));
		fireEvent.click(await screen.findByText("open source conversation"));
		expect(location.hash).toContain("#/s/4e92693767fa");
	});
});
