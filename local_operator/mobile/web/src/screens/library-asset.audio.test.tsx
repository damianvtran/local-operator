// @vitest-environment happy-dom
//
// The asset screen's AUDIO seat: it must mount THE shared mobile player, and
// no native `<audio controls>` widget may survive beside it (one player per
// platform). QA round 1 (Q3) probed this path live; this pins it in the repo.
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { LibraryAsset } from "../types";

const getLibraryAsset = vi.fn();

vi.mock("../api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("../api")>();
	return {
		...actual,
		getLibraryAsset: (...args: unknown[]) => getLibraryAsset(...args),
	};
});

import { LibraryAssetScreen } from "./library-asset";

function audioAsset(): LibraryAsset {
	return {
		id: "la_0123456789abcdef",
		ref: {
			type: "artifact",
			path: null,
			display_path: "chime.mp3",
			digest: "a".repeat(32),
			content_type: "audio/mpeg",
		},
		name: "chime.mp3",
		derived_name: "chime.mp3",
		description: "",
		kind: "audio",
		category: "audio",
		origin: "generated",
		origin_group: "generated",
		origins: ["generated"],
		tags: [],
		primary_tag: null,
		size_bytes: 2048,
		width: null,
		height: null,
		duration_s: 3.5,
		created_at: 1_700_000_000,
		modified_at: 1_700_000_000,
		last_seen_at: 1_700_000_000,
		exists: true,
		highlighted: false,
		highlight_why: null,
		occurrences: [],
		session_count: 1,
		chat: null,
		preview: { state: "unsupported", mime_type: "audio/mpeg", pages: null },
		organized: null,
		locked: false,
		favorite: false,
		archived: false,
	};
}

beforeEach(() => {
	getLibraryAsset.mockResolvedValue(audioAsset());
	/* The peaks route answers 404 — the player must still mount, flat. */
	vi.stubGlobal(
		"fetch",
		vi.fn(async () => ({ ok: false, status: 404, json: async () => ({ error: "no waveform" }) })),
	);
	vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
});

afterEach(() => {
	vi.unstubAllGlobals();
	vi.restoreAllMocks();
	cleanup();
	vi.clearAllMocks();
	history.replaceState(null, "", "#/");
});

describe("library asset audio preview", () => {
	it("mounts the shared full player instead of a native audio control", async () => {
		const { container } = render(<LibraryAssetScreen assetId="la_0123456789abcdef" />);
		expect(await screen.findByRole("heading", { name: "chime.mp3" })).toBeTruthy();

		const player = container.querySelector("[data-sound-player]");
		expect(player?.getAttribute("data-sound-player")).toBe("full");
		expect(player?.getAttribute("data-waveform")).toBe("flat");
		/* The engine element is inside the shared player, hidden and control-less;
		   the browser widget that used to sit here must be gone. */
		expect(container.querySelector("audio[controls]")).toBeNull();
		expect(container.querySelectorAll("audio")).toHaveLength(1);
		expect(container.querySelector("audio")?.getAttribute("aria-hidden")).toBe("true");
	});
});
