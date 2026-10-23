// @vitest-environment happy-dom
//
// The Library tile's audio seat: play must play, and the rest of the well must
// keep opening the asset. The card's open target is an `absolute inset-0`
// button, so this is a genuine z-order/pointer test rather than a style one —
// the player is a sibling ABOVE that button, and a press on it must not reach
// the button underneath.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AssetCard } from "./asset-card";
import type { LibraryAsset } from "../../types";

beforeEach(() => {
	/* `useNearViewport` must report immediately or the sound never mounts. */
	class FiredIntersectionObserver implements IntersectionObserver {
		readonly root = null;
		readonly rootMargin = "0px";
		readonly thresholds = [0];
		constructor(private readonly callback: IntersectionObserverCallback) {}
		observe(target: Element): void {
			this.callback([{ isIntersecting: true, target } as IntersectionObserverEntry], this);
		}
		unobserve(): void {}
		disconnect(): void {}
		takeRecords(): IntersectionObserverEntry[] {
			return [];
		}
	}
	vi.stubGlobal("IntersectionObserver", FiredIntersectionObserver);
	/* The peaks route answers 404: the player draws its flat scrubber. */
	vi.stubGlobal(
		"fetch",
		vi.fn(async () => ({ ok: false, status: 404, json: async () => ({ error: "no waveform" }) })),
	);
	vi.spyOn(HTMLMediaElement.prototype, "play").mockResolvedValue(undefined);
	vi.spyOn(HTMLMediaElement.prototype, "pause").mockImplementation(() => {});
});

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
	vi.restoreAllMocks();
});

function asset(overrides: Partial<LibraryAsset> = {}): LibraryAsset {
	const id = overrides.id ?? "la_0123456789abcdef";
	return {
		id,
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
		...overrides,
	} as LibraryAsset;
}

describe("library asset card", () => {
	it("shows the player in an audio tile's well instead of the glyph", () => {
		const { container } = render(
			<AssetCard asset={asset()} onOpen={() => {}} onMenu={() => {}} />,
		);
		expect(container.querySelector("[data-sound-player]")).toBeTruthy();
		expect(container.querySelector("[data-sound-player]")?.getAttribute("data-waveform")).toBe(
			"flat",
		);
	});

	it("does not open the asset when play is pressed", () => {
		const onOpen = vi.fn();
		render(<AssetCard asset={asset()} onOpen={onOpen} onMenu={() => {}} />);
		fireEvent.click(screen.getByRole("button", { name: "play chime.mp3" }));
		expect(onOpen).not.toHaveBeenCalled();
		/* The rest of the well still opens it. */
		fireEvent.click(screen.getByRole("button", { name: "open chime.mp3" }));
		expect(onOpen).toHaveBeenCalledTimes(1);
	});

	it("stacks the player above the inset-0 open overlay, not merely beside it", () => {
		/* `fireEvent` dispatches at the target and never hit-tests, so the click
		   test above cannot see a stacking regression (round-1 review m2).
		   These two assertions CAN fail if `z-10` is dropped or the seating
		   moves after the overlay: the open button must follow the player in
		   document order AND paint under a real stacking context. */
		const { container } = render(
			<AssetCard asset={asset()} onOpen={() => {}} onMenu={() => {}} />,
		);
		const seat = container.querySelector("[data-sound-seat]");
		if (!seat) throw new Error("no sound seat rendered");
		const open = screen.getByRole("button", { name: "open chime.mp3" });
		expect(seat.className).toContain("z-10");
		expect(seat.compareDocumentPosition(open) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
	});

	it("keeps an image card on the thumbnail path with no player", () => {
		const { container } = render(
			<AssetCard
				asset={asset({ kind: "image", name: "photo.png", ref: { ...asset().ref } })}
				onOpen={() => {}}
				onMenu={() => {}}
			/>,
		);
		expect(container.querySelector("[data-sound-player]")).toBeNull();
		expect(container.querySelector("img")?.getAttribute("src")).toContain("/thumbnail");
	});
});
