// The image-generation adapter: the detection constant, the per-state
// mapping, and every absence path.
//
// WHY THIS FILE EXISTS
// --------------------
// The card renders ONLY what `imageGenView` returns, and the live-detail wire
// fields (queue position, progress fraction, log lines, error payload) are
// NOT frozen yet — the programme contract keeps their handling in this one
// module so the freeze day is a one-file change. That makes this suite the
// freeze point's own regression net, and it exists at the ADAPTER level
// (rather than only through the rendered card) because the property most
// worth pinning is subtractive: a field the feed does not carry — or carries
// malformed — must reduce (null / empty / the indeterminate branch), never
// become an invented number. A render test can miss that; a mapping table
// cannot.
import { describe, expect, it } from "vitest";
import { IMAGE_GEN_TOOLS, imageGenView } from "./image-gen";
import type { TranscriptEntry, TranscriptEntryDetails } from "../types";

/** Live-detail fields are NOT frozen yet, and their only home in src/ is the
    adapter — so the type does not declare them and the tests build the raw
    bag here. Deliberately untyped on the way in: this suite's whole job is
    malformed shapes, which a typed literal could not carry. */
function liveDetails(fields: Record<string, unknown>): TranscriptEntryDetails {
	return fields as TranscriptEntryDetails;
}

function entry(over: Partial<TranscriptEntry>): TranscriptEntry {
	return {
		id: "e1",
		kind: "tool",
		text: "",
		tool_call_id: "call_img",
		tool_name: "generate_image",
		tool_state: "running",
		summary: "",
		intent: "",
		diff_added: 0,
		diff_removed: 0,
		elapsed_s: 0,
		error: "",
		details: {},
		final: false,
		...over,
	};
}

describe("the detection constant", () => {
	it("names the frozen tool, and only it", () => {
		/* The harness lane froze ONE tool: image-to-image is a
		   `source_image_path` argument on the same call. A second name
		   appearing here without its tool shipping is a wire the adapter
		   would claim to know. */
		expect(IMAGE_GEN_TOOLS.has("generate_image")).toBe(true);
		expect(IMAGE_GEN_TOOLS.size).toBe(1);
		expect(IMAGE_GEN_TOOLS.has("generate_altered_image")).toBe(false);
		expect(IMAGE_GEN_TOOLS.has("read")).toBe(false);
		expect(IMAGE_GEN_TOOLS.has("")).toBe(false);
	});
});

describe("state mapping, wire vocabulary -> card vocabulary", () => {
	it.each([
		["composing", "queued"],
		["queued", "queued"],
		["running", "running"],
		["done", "done"],
		["failed", "failed"],
		["interrupted", "cancelled"],
	] as const)("%s maps to %s", (wire, view) => {
		expect(imageGenView(entry({ tool_state: wire })).state).toBe(view);
	});

	it("a pending cancel outranks every not-yet-settled state", () => {
		for (const live of ["composing", "queued", "running"] as const) {
			expect(
				imageGenView(entry({ tool_state: live }), true).state,
			).toBe("cancelling");
		}
	});

	it("but never a settle — the confirmation is the settle itself", () => {
		/* The provider note: never optimistically "cancelled". The mirror
		   half: never optimistically still-cancelling either — if the row
		   reports done/failed/interrupted while the press is outstanding,
		   that verdict is what renders. */
		expect(imageGenView(entry({ tool_state: "done" }), true).state).toBe("done");
		expect(imageGenView(entry({ tool_state: "failed" }), true).state).toBe(
			"failed",
		);
		expect(
			imageGenView(entry({ tool_state: "interrupted" }), true).state,
		).toBe("cancelled");
	});
});

describe("the cancel conflict (media_already_completed)", () => {
	it("settles as `finished` from either failure-shaped settle, never as an error", () => {
		const at = (wire: "failed" | "interrupted") =>
			imageGenView(
				entry({
					tool_state: wire,
					details: liveDetails({ error_type: "media_already_completed" }),
				}),
			).state;
		expect(at("failed")).toBe("finished");
		expect(at("interrupted")).toBe("finished");
	});

	it("a done settle is not re-worded, and other codes stay their own states", () => {
		expect(
			imageGenView(
				entry({
					tool_state: "done",
					details: liveDetails({ error_type: "media_already_completed" }),
				}),
			).state,
		).toBe("done");
		expect(
			imageGenView(
				entry({
					tool_state: "failed",
					details: liveDetails({ error_type: "media_rejected" }),
				}),
			).state,
		).toBe("failed");
	});
});

describe("live detail fields reduce honestly when absent or malformed", () => {
	it("empty details carry no numbers and no text", () => {
		const view = imageGenView(entry({}));
		expect(view.queuePosition).toBeNull();
		expect(view.progress).toBeNull();
		expect(view.logs).toEqual([]);
		expect(view.error).toBe("");
		expect(view.errorType).toBe("");
		expect(view.images).toEqual([]);
	});

	it("queue_position: integers pass, everything else reduces", () => {
		const at = (value: unknown) =>
			imageGenView(entry({ details: liveDetails({ queue_position: value }) }))
				.queuePosition;
		expect(at(2)).toBe(2);
		expect(at(0)).toBe(0);
		expect(at("2")).toBeNull();
		expect(at(2.5)).toBeNull();
		expect(at(-1)).toBeNull();
		expect(at(Number.NaN)).toBeNull();
	});

	it("progress: a fraction in 0..1 passes, everything else reduces", () => {
		const at = (value: unknown) =>
			imageGenView(entry({ details: liveDetails({ progress: value }) })).progress;
		expect(at(0)).toBe(0);
		expect(at(0.42)).toBe(0.42);
		expect(at(1)).toBe(1);
		/* NOT clamped. 42 is not a fraction, and clamping it into a full bar
		   would state 100% off a value that never meant one — the
		   indeterminate branch is the honest rendering. */
		expect(at(42)).toBeNull();
		expect(at(-0.1)).toBeNull();
		expect(at(Number.NaN)).toBeNull();
		expect(at(Number.POSITIVE_INFINITY)).toBeNull();
		expect(at("0.5")).toBeNull();
	});

	it("logs: strings only, capped to the tail", () => {
		const at = (value: unknown) =>
			imageGenView(entry({ details: liveDetails({ logs: value }) })).logs;
		expect(at(["a", "b", "c", "d"])).toEqual(["b", "c", "d"]);
		expect(at(["only"])).toEqual(["only"]);
		expect(at(["a", 7, null, "b"])).toEqual(["a", "b"]);
		expect(at("not-a-list")).toEqual([]);
		expect(at([])).toEqual([]);
	});

	it("error is the settle's text, verbatim — and only when it is text", () => {
		const kept = imageGenView(
			entry({ error: "This generation failed before producing output." }),
		);
		expect(kept.error).toBe("This generation failed before producing output.");
		expect(imageGenView(entry({ error: undefined as unknown as string })).error).toBe("");
	});

	it("error_type is the structured category, when carried", () => {
		const at = (value: unknown) =>
			imageGenView(entry({ details: liveDetails({ error_type: value }) })).errorType;
		expect(at("media_rejected")).toBe("media_rejected");
		expect(at(7)).toBe("");
		expect(at(undefined)).toBe("");
	});

	it("images: refs pass through, malformed entries are dropped", () => {
		const view = imageGenView(
			entry({
				images: [
					{ index: 0, mime_type: "image/png" },
					{ index: 1 } as unknown as { index: number; mime_type: string },
				],
			}),
		);
		expect(view.images).toEqual([{ index: 0, mime_type: "image/png" }]);
	});
});
