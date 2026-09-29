// @vitest-environment happy-dom
//
// The pure dictation rules: the draft join (never clobber, both orders), the
// recorder mime preference (iOS Safari records mp4 and NOT webm — the order is
// load-bearing), the duration format, and the provenance state behind
// input_mode/input_path (sticky history; spans only bookkeep the path).
import { describe, expect, it } from "vitest";
import {
	annotationForSend,
	applyEdit,
	computeEdit,
	emptyProvenance,
	formatDuration,
	joinDraft,
	noteDictation,
	pickRecorderMime,
	RECORDER_MIME_CANDIDATES,
} from "./dictation";

describe("joinDraft", () => {
	it("appends to an empty draft with no leading space", () => {
		expect(joinDraft("", "hello world")).toBe("hello world");
	});

	it("replaces a whitespace-only draft with the cleaned join", () => {
		expect(joinDraft("  \n ", " hello world ")).toBe("hello world");
	});

	it("adds exactly one separator when the draft does not end in whitespace", () => {
		expect(joinDraft("note:", "hello")).toBe("note: hello");
	});

	it("reuses the draft's own whitespace as the separator", () => {
		expect(joinDraft("note: ", "hello")).toBe("note: hello");
		expect(joinDraft("note:\n", "hello")).toBe("note:\nhello");
	});

	it("keeps the transcript's casing and punctuation", () => {
		expect(joinDraft("", "Hello, World!")).toBe("Hello, World!");
	});

	it("leaves the draft untouched for an empty transcript", () => {
		expect(joinDraft("keep me", "   ")).toBe("keep me");
		expect(joinDraft("", "")).toBe("");
	});

	it("stacks consecutive dictations", () => {
		expect(joinDraft(joinDraft("", "one"), "two")).toBe("one two");
	});
});

describe("pickRecorderMime", () => {
	it("prefers audio/mp4 first (iOS Safari records AAC/mp4, not webm)", () => {
		const supported = (mime: string) =>
			RECORDER_MIME_CANDIDATES.includes(mime as (typeof RECORDER_MIME_CANDIDATES)[number]);
		expect(pickRecorderMime(supported)).toBe("audio/mp4");
	});

	it("falls through to the webm/opus codec on a Chrome-shaped probe", () => {
		expect(pickRecorderMime((mime) => mime === "audio/webm;codecs=opus")).toBe(
			"audio/webm;codecs=opus",
		);
	});

	it("answers the empty string when nothing is supported (browser default)", () => {
		expect(pickRecorderMime(() => false)).toBe("");
	});

	it("probes in the frozen order", () => {
		const probed: string[] = [];
		pickRecorderMime((mime) => {
			probed.push(mime);
			return false;
		});
		expect(probed).toEqual([...RECORDER_MIME_CANDIDATES]);
	});
});

describe("formatDuration", () => {
	it("renders m:ss", () => {
		expect(formatDuration(0)).toBe("0:00");
		expect(formatDuration(7)).toBe("0:07");
		expect(formatDuration(61)).toBe("1:01");
		expect(formatDuration(120)).toBe("2:00");
	});
});

describe("the provenance window", () => {
	it("annotates a purely typed draft as typed", () => {
		expect(annotationForSend(emptyProvenance())).toEqual({ input_mode: "typed" });
	});

	it("annotates a dictated draft as dictated with its path", () => {
		const provenance = noteDictation(emptyProvenance(), {
			start: 0,
			end: 5,
			path: "provider_stt_radient",
		});
		expect(annotationForSend(provenance)).toEqual({
			input_mode: "dictated",
			input_path: "provider_stt_radient",
		});
	});

	it("annotates dictation then typing as mixed", () => {
		let provenance = noteDictation(emptyProvenance(), {
			start: 0,
			end: 5,
			path: "provider_stt_radient",
		});
		provenance = applyEdit(provenance, { start: 6, endOld: 6, endNew: 12 });
		expect(annotationForSend(provenance).input_mode).toBe("mixed");
	});

	it("keeps the dictated flag across a deletion edit (history-based contract)", () => {
		let provenance = noteDictation(emptyProvenance(), {
			start: 0,
			end: 5,
			path: "provider_stt_radient",
		});
		// Delete PART of the dictated range and send: the history is sticky — a
		// dictation WAS seen, so the message never re-reads as purely typed — and
		// the deletion is itself a user edit, which is what makes it mixed (the
		// design's rule: any user edit sets typedSeen; neither flag ever resets).
		// The mode alone could not carry the dictation, which is exactly why the
		// flags are separate from the spans.
		provenance = applyEdit(provenance, { start: 0, endOld: 5, endNew: 0 });
		expect(provenance.spans).toEqual([]);
		expect(provenance.sawDictation).toBe(true);
		expect(annotationForSend(provenance)).toEqual({
			input_mode: "mixed",
			input_path: "provider_stt_radient",
		});
	});

	it("drops a path when a dictation had none and none was recorded before", () => {
		const provenance = noteDictation(emptyProvenance(), { start: 0, end: 3, path: "" });
		expect(annotationForSend(provenance)).toEqual({ input_mode: "dictated" });
	});

	it("shifts spans after an edit before them and keeps the latest path", () => {
		let provenance = noteDictation(emptyProvenance(), {
			start: 0,
			end: 3,
			path: "provider_stt_radient",
		});
		provenance = applyEdit(provenance, { start: 0, endOld: 0, endNew: 4 });
		expect(provenance.spans).toEqual([{ start: 4, end: 7, path: "provider_stt_radient" }]);
		expect(annotationForSend(provenance).input_path).toBe("provider_stt_radient");
	});

	it("computes one edit for a replacement, and none for equality", () => {
		expect(computeEdit("abc", "abc")).toBeNull();
		expect(computeEdit("abc", "abd")).toEqual({ start: 2, endOld: 3, endNew: 3 });
		expect(computeEdit("abc", "axc")).toEqual({ start: 1, endOld: 2, endNew: 2 });
		expect(computeEdit("", "abc")).toEqual({ start: 0, endOld: 0, endNew: 3 });
	});
});
