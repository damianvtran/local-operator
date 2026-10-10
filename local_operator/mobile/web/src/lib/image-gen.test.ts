// The image-generation adapter: the detection constant, the canonical stage
// mapping, and every absence path.
//
// WHY THIS FILE EXISTS
// --------------------
// The card renders ONLY what `imageGenView` returns, and the live-detail wire
// fields are the harness lane's FROZEN canonical bag (`stage`,
// `queue_position`, `progress_fraction`, `log_lines`, `error`, `error_type` —
// every key present on every update, `None` when unsupplied). The programme
// contract keeps their handling in this one module, and this suite is that
// module's regression net. It exists at the ADAPTER level
// (rather than only through the rendered card) because the property most
// worth pinning is subtractive: a field the feed does not carry — or carries
// malformed — must reduce (null / empty / the reduced state), never become
// an invented number. A render test can miss that; a mapping table cannot.
import { describe, expect, it } from "vitest";
import { IMAGE_GEN_TOOLS, imageGenView } from "./image-gen";
import type { TranscriptEntry, TranscriptEntryDetails } from "../types";

/** The canonical bag's only home in src/ is the adapter — the entry type
    does not declare it — so the tests build the raw bag here, deliberately
    untyped on the way in: this suite's whole job is malformed shapes, which
    a typed literal could not carry. */
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

	it("the canonical stage word names the live interims the fold cannot", () => {
		const at = (stage: unknown) =>
			imageGenView(
				entry({ tool_state: "running", details: liveDetails({ stage }) }),
			).state;
		expect(at("queued")).toBe("queued");
		expect(at("in_progress")).toBe("running");
		expect(at("cancelling")).toBe("cancelling");
	});

	it("settled-end stage words do not repaint a live row — the settle is the confirmation", () => {
		const at = (stage: unknown) =>
			imageGenView(
				entry({ tool_state: "running", details: liveDetails({ stage }) }),
			).state;
		expect(at("completed")).toBe("running");
		expect(at("cancelled")).toBe("running");
	});

	it("an unknown stage word reads as absent — a stranger may not repaint the card", () => {
		const at = (stage: unknown) =>
			imageGenView(
				entry({ tool_state: "running", details: liveDetails({ stage }) }),
			).state;
		expect(at("RUNNING")).toBe("running");
		expect(at("in-progress")).toBe("running");
		expect(at(7)).toBe("running");
		expect(at(null)).toBe("running");
	});

	it("a mid-walk failure (stage null, semantics in error/error_type) keeps the live view live", () => {
		const view = imageGenView(
			entry({
				details: liveDetails({
					stage: null,
					error: "Radient was rate limited; the walk continues.",
					error_type: "media_rate_limited",
				}),
			}),
		);
		expect(view.state).toBe("running");
	});

	it("a pending cancel still outranks the live words", () => {
		for (const stage of ["queued", "in_progress", "cancelling"] as const) {
			expect(
				imageGenView(
					entry({ tool_state: "running", details: liveDetails({ stage }) }),
					true,
				).state,
			).toBe("cancelling");
		}
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

	it("a failure-shaped settle whose stage says 'cancelled' is a cancel that landed", () => {
		expect(
			imageGenView(
				entry({
					tool_state: "failed",
					details: liveDetails({ stage: "cancelled" }),
				}),
			).state,
		).toBe("cancelled");
		/* The conflict still outranks the stage word... */
		expect(
			imageGenView(
				entry({
					tool_state: "failed",
					details: liveDetails({
						stage: "cancelled",
						error_type: "media_already_completed",
					}),
				}),
			).state,
		).toBe("finished");
		/* ...and any OTHER classified failure keeps the failure arm: the
		   refinement never outranks an error the feed did classify. */
		expect(
			imageGenView(
				entry({
					tool_state: "failed",
					details: liveDetails({
						stage: "cancelled",
						error_type: "media_rejected",
					}),
				}),
			).state,
		).toBe("failed");
	});
});

describe("the generating fact (the desktop's F3 rule, mirrored)", () => {
	it("running implies true; every reduced and settled state is false", () => {
		expect(imageGenView(entry({ tool_state: "running" })).generating).toBe(true);
		expect(imageGenView(entry({ tool_state: "queued" })).generating).toBe(false);
		expect(imageGenView(entry({ tool_state: "composing" })).generating).toBe(
			false,
		);
		for (const wire of ["done", "failed", "interrupted"] as const) {
			expect(imageGenView(entry({ tool_state: wire })).generating).toBe(false);
		}
	});

	it("a press keeps the state it replaced — a queued hold never conjures a body", () => {
		expect(imageGenView(entry({ tool_state: "queued" }), true).generating).toBe(
			false,
		);
		expect(imageGenView(entry({ tool_state: "running" }), true).generating).toBe(
			true,
		);
		/* The provider-side queue is a reduced `queued` view on this surface
		   (the stage refine's own arm): the press holds that card's claim —
		   nothing generating yet — rather than flipping it into a body. */
		expect(
			imageGenView(
				entry({
					tool_state: "running",
					details: liveDetails({ stage: "queued" }),
				}),
				true,
			).generating,
		).toBe(false);
	});

	it("the wire's own cancelling keeps what the row's word last said", () => {
		/* The frame no longer carries the stage the hold replaced, so the
		   row's own word decides: `running` had started executing (true);
		   `queued` had not (false — the reduced hold). */
		expect(
			imageGenView(
				entry({
					tool_state: "running",
					details: liveDetails({ stage: "cancelling" }),
				}),
			).generating,
		).toBe(true);
		expect(
			imageGenView(
				entry({
					tool_state: "queued",
					details: liveDetails({ stage: "cancelling" }),
				}),
			).generating,
		).toBe(false);
	});

	it("a live stage word keeps the fact current", () => {
		/* `in_progress` says the provider is working, even over a row whose
		   word still reads queued — the stage refines the fact the same way
		   it refines the state. */
		expect(
			imageGenView(
				entry({
					tool_state: "queued",
					details: liveDetails({ stage: "in_progress" }),
				}),
			).generating,
		).toBe(true);
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

	it("progress_fraction: a fraction in 0..1 passes, everything else reduces", () => {
		const at = (value: unknown) =>
			imageGenView(entry({ details: liveDetails({ progress_fraction: value }) }))
				.progress;
		expect(at(0)).toBe(0);
		expect(at(0.42)).toBe(0.42);
		expect(at(1)).toBe(1);
		/* NOT clamped. 42 is not a fraction, and clamping it into a full bar
		   would state 100% off a value that never meant one — the reduced
		   state (no bar at all; one draws only against a carried fraction)
		   is the honest rendering. */
		expect(at(42)).toBeNull();
		expect(at(-0.1)).toBeNull();
		expect(at(Number.NaN)).toBeNull();
		expect(at(Number.POSITIVE_INFINITY)).toBeNull();
		expect(at("0.5")).toBeNull();
	});

	it("log_lines: canonical {message, timestamp} rows paint their messages, tail only", () => {
		const at = (value: unknown) =>
			imageGenView(entry({ details: liveDetails({ log_lines: value }) })).logs;
		const row = (message: string) => ({ message, timestamp: "2026-10-09T00:00:00Z" });
		expect(at([row("a"), row("b"), row("c"), row("d")])).toEqual(["b", "c", "d"]);
		expect(at([row("only")])).toEqual(["only"]);
		/* Rows of any other shape drop: a bare string, a number, null, an
		   object without a string message — none is a rendering the feed
		   sent. */
		expect(at([row("a"), 7, null, "bare", { noMessage: 1 }, row("b")])).toEqual([
			"a",
			"b",
		]);
		expect(at(["bare"])).toEqual([]);
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
