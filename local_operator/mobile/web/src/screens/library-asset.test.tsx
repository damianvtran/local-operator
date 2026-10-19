// @vitest-environment happy-dom
//
// The DETAIL screen's composer: it must create a session, send the asset's
// identity as the first message, record the link and navigate. A send that
// cannot be delivered must leave the text in that session's draft rather than
// dropping it. Also pins the two honest fallbacks (a missing file, an
// unsupported preview) and the metadata re-read.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { LibraryAsset } from "../types";

const getLibraryAsset = vi.fn();
const startSession = vi.fn();
const sendCommand = vi.fn();
const linkLibraryChat = vi.fn();
const setDraft = vi.fn();
const getLibraryTags = vi.fn();
const patchLibraryAsset = vi.fn();
const setLibraryTags = vi.fn();
const createLibraryTag = vi.fn();

vi.mock("../api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../api")>();
	return {
		...actual,
		getLibraryAsset: (...args: unknown[]) => getLibraryAsset(...args),
		startSession: (...args: unknown[]) => startSession(...args),
		sendCommand: (...args: unknown[]) => sendCommand(...args),
		linkLibraryChat: (...args: unknown[]) => linkLibraryChat(...args),
		getLibraryTags: (...args: unknown[]) => getLibraryTags(...args),
		patchLibraryAsset: (...args: unknown[]) => patchLibraryAsset(...args),
		setLibraryTags: (...args: unknown[]) => setLibraryTags(...args),
		createLibraryTag: (...args: unknown[]) => createLibraryTag(...args),
	};
});

vi.mock("../store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../store")>();
	return { ...actual, setDraft: (...args: unknown[]) => setDraft(...args) };
});

import { LibraryAssetScreen } from "./library-asset";

function asset(overrides: Partial<LibraryAsset> = {}): LibraryAsset {
	return {
		id: "la_0123456789abcdef",
		ref: {
			type: "path",
			path: null,
			display_path: "~/notes/launch.md",
			digest: null,
			content_type: "text/markdown",
		},
		name: "launch notes",
		derived_name: "launch.md",
		description: "why the launch moved",
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

beforeEach(() => {
	// The text preview fetches the bytes route; happy-dom has no relay, so the
	// suite stubs the one network call instead of letting it fail noisily.
	vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, text: async () => "# preview\n" })));
	getLibraryAsset.mockResolvedValue(asset());
	startSession.mockResolvedValue({ ok: true, pid: 42, session_id: "aaaaaaaaaaaa" });
	sendCommand.mockResolvedValue({ ok: true });
	linkLibraryChat.mockResolvedValue({ chat: { session_id: "aaaaaaaaaaaa", title: "t", updated_at: 1 } });
	getLibraryTags.mockResolvedValue({ tags: [{ id: "tag_launch", name: "launch", category: "topic" }] });
	patchLibraryAsset.mockResolvedValue(asset({ name: "renamed notes" }));
	setLibraryTags.mockResolvedValue({ tags: [{ id: "tag_launch", name: "launch", category: "topic" }] });
});

afterEach(() => {
	vi.unstubAllGlobals();
	cleanup();
	vi.clearAllMocks();
	history.replaceState(null, "", "#/");
});

describe("library asset detail", () => {
	it("creates a session, sends the asset context, links it and navigates", async () => {
		render(<LibraryAssetScreen assetId="la_0123456789abcdef" />);
		expect(await screen.findByText("launch notes")).toBeTruthy();
		expect(screen.getByText("why the launch moved")).toBeTruthy();
		expect(screen.getByText("launch planning")).toBeTruthy();

		const composer = screen.getByLabelText("ask about this asset");
		fireEvent.change(composer, { target: { value: "summarize the risky parts" } });
		fireEvent.click(screen.getByRole("button", { name: "ask" }));

		await waitFor(() => expect(sendCommand).toHaveBeenCalledTimes(1));
		const [sessionId, op] = sendCommand.mock.calls[0] as [string, { op: string; text: string }];
		expect(sessionId).toBe("aaaaaaaaaaaa");
		expect(op.op).toBe("prompt");
		expect(op.text).toContain("launch notes");
		expect(op.text).toContain("la_0123456789abcdef");
		expect(op.text).toContain("~/notes/launch.md");
		expect(op.text).toContain("summarize the risky parts");
		expect(linkLibraryChat).toHaveBeenCalledWith("la_0123456789abcdef", "aaaaaaaaaaaa");
		await waitFor(() => expect(location.hash).toBe("#/s/aaaaaaaaaaaa"));
	});

	it("sends into an existing linked chat without starting another session", async () => {
		getLibraryAsset.mockResolvedValue(
			asset({ chat: { session_id: "bbbbbbbbbbbb", title: "asset chat", updated_at: 1 } }),
		);
		render(<LibraryAssetScreen assetId="la_0123456789abcdef" />);
		await screen.findByText("chat: asset chat");
		fireEvent.change(screen.getByLabelText("ask about this asset"), { target: { value: "what changed?" } });
		fireEvent.click(screen.getByRole("button", { name: "ask" }));
		await waitFor(() => expect(sendCommand).toHaveBeenCalledTimes(1));
		expect(startSession).not.toHaveBeenCalled();
		expect(linkLibraryChat).not.toHaveBeenCalled();
		await waitFor(() => expect(location.hash).toBe("#/s/bbbbbbbbbbbb"));
	});

	it("keeps the words as that session's draft when the send fails", async () => {
		sendCommand.mockRejectedValue(new Error("session not ready"));
		render(<LibraryAssetScreen assetId="la_0123456789abcdef" />);
		await screen.findByText("launch notes");
		fireEvent.change(screen.getByLabelText("ask about this asset"), { target: { value: "look at this" } });
		fireEvent.click(screen.getByRole("button", { name: "ask" }));
		await waitFor(() => expect(setDraft).toHaveBeenCalledTimes(1));
		const [sessionId, text] = setDraft.mock.calls[0] as [string, string];
		expect(sessionId).toBe("aaaaaaaaaaaa");
		expect(text).toContain("look at this");
		await waitFor(() => expect(location.hash).toBe("#/s/aaaaaaaaaaaa"));
	});

	it("reports a missing file instead of pretending to preview it", async () => {
		getLibraryAsset.mockResolvedValue(asset({ exists: false }));
		render(<LibraryAssetScreen assetId="la_0123456789abcdef" />);
		expect(await screen.findByText(/no longer on this machine/)).toBeTruthy();
	});

	it("shows the honest PPTX fallback", async () => {
		getLibraryAsset.mockResolvedValue(
			asset({ kind: "pptx", category: "presentation", preview: { state: "ready", mime_type: "application/vnd.openxmlformats-officedocument.presentationml.presentation", pages: null } }),
		);
		render(<LibraryAssetScreen assetId="la_0123456789abcdef" />);
		expect(await screen.findByText(/Slide rendering isn't available/)).toBeTruthy();
		expect(screen.getByText("open")).toBeTruthy();
		expect(screen.getByText("download")).toBeTruthy();
	});

	it("saves metadata and re-reads the asset instead of patching locally", async () => {
		render(<LibraryAssetScreen assetId="la_0123456789abcdef" />);
		await screen.findByText("launch notes");
		fireEvent.click(screen.getByRole("button", { name: "edit" }));
		const name = await screen.findByDisplayValue("launch notes");
		fireEvent.change(name, { target: { value: "renamed notes" } });
		fireEvent.click(screen.getByRole("button", { name: "save" }));
		await waitFor(() => expect(patchLibraryAsset).toHaveBeenCalledTimes(1));
		expect(patchLibraryAsset).toHaveBeenCalledWith("la_0123456789abcdef", {
			name: "renamed notes",
			description: "why the launch moved",
		});
		expect(setLibraryTags).toHaveBeenCalledWith("la_0123456789abcdef", ["tag_launch"]);
		// Twice: the first read on mount, the second after the write.
		await waitFor(() => expect(getLibraryAsset).toHaveBeenCalledTimes(2));
	});

	it("offers a retry when the read fails", async () => {
		getLibraryAsset.mockRejectedValueOnce(new Error("library_asset_not_found"));
		render(<LibraryAssetScreen assetId="la_0123456789abcdef" />);
		expect(await screen.findByRole("alert")).toBeTruthy();
		expect(screen.getByText("library_asset_not_found")).toBeTruthy();
		fireEvent.click(screen.getByRole("button", { name: "Try again" }));
		await waitFor(() => expect(getLibraryAsset).toHaveBeenCalledTimes(2));
		expect(await screen.findByText("launch notes")).toBeTruthy();
	});
});
