/**
 * Dictation: the pure logic behind the composer's voice mic.
 *
 * Everything here is deliberately testable without a browser: the recorder
 * mime choice, the draft join, the provenance bookkeeping behind
 * ``input_mode``/``input_path``, and the two copy-free helpers the composer's
 * state machine calls. The composer owns the MediaRecorder, the streams and
 * the request; this module owns the rules.
 *
 * Provenance contract (input-mode-v1, history-based and STICKY): once the
 * window since the draft's last accepted send/clear has SEEN typing or a
 * dictation, it stays seen — deletion and edits do not un-see it, and the
 * empty draft is what resets the window (a fresh message starts clean). The
 * dictated SPANS survive only to name the ``input_path`` of the most recent
 * dictation still present in the sent text.
 */
import type { InputMode } from "../types";

/** The recorder's hard stop; the server's byte cap is the other bound. */
export const MAX_RECORDING_MS = 120_000;

/**
 * First supported wins. ``audio/mp4`` leads because iOS Safari (>= 16.4)
 * records AAC/mp4 and does NOT support webm; the rest follow the codecs
 * Chrome/Firefox prefer. ``""`` means "browser default", which the server's
 * allowlist still has to recognise — hence the wide server allowlist.
 */
export const RECORDER_MIME_CANDIDATES = [
	"audio/mp4",
	"audio/webm;codecs=opus",
	"audio/webm",
	"audio/ogg;codecs=opus",
] as const;

export function pickRecorderMime(isSupported: (mime: string) => boolean): string {
	for (const mime of RECORDER_MIME_CANDIDATES) {
		if (isSupported(mime)) return mime;
	}
	return "";
}

/** ``m:ss`` for the recording status row. */
export function formatDuration(seconds: number): string {
	const safe = Math.max(0, Math.floor(seconds));
	const minutes = Math.floor(safe / 60);
	return `${minutes}:${String(safe % 60).padStart(2, "0")}`;
}

/**
 * Append a transcript to the draft — never replace, never reorder.
 *
 * With a whitespace-only base this returns the CLEANED join (no leading
 * space, trailing newlines collapsed to the separator); otherwise the join
 * keeps the existing text byte-for-byte and adds exactly one separator only
 * when it is missing. No punctuation surgery: the transcript keeps its own
 * casing and punctuation.
 */
export function joinDraft(base: string, transcript: string): string {
	const cleaned = transcript.trim();
	if (cleaned === "") return base;
	if (base.trim() === "") return cleaned;
	return /\s$/.test(base) ? `${base}${cleaned}` : `${base} ${cleaned}`;
}

/** One dictated range still tracked on the draft. */
export interface DictatedSpan {
	start: number;
	/** Exclusive. */
	end: number;
	path: string;
}

/** One edit between two draft strings, in OLD-text coordinates. */
export interface DraftEdit {
	start: number;
	/** Exclusive end in the old text. */
	endOld: number;
	/** Exclusive end in the new text. */
	endNew: number;
}

/**
 * The one edit between ``previous`` and ``next`` (common prefix + suffix).
 * A replacement is one edit; equal strings are no edit at all.
 */
export function computeEdit(previous: string, next: string): DraftEdit | null {
	if (previous === next) return null;
	let start = 0;
	const max = Math.min(previous.length, next.length);
	while (start < max && previous[start] === next[start]) start++;
	let endOld = previous.length;
	let endNew = next.length;
	while (endOld > start && endNew > start && previous[endOld - 1] === next[endNew - 1]) {
		endOld--;
		endNew--;
	}
	return { start, endOld, endNew };
}

export interface DictationProvenance {
	/** Any user edit since the window opened (sticky). */
	sawTyping: boolean;
	/** Any dictation since the window opened (sticky). */
	sawDictation: boolean;
	/** Dictated ranges — input_path bookkeeping only, never the mode. */
	spans: DictatedSpan[];
	/** The newest dictation's path, kept for the history edge. */
	lastPath: string;
}

export function emptyProvenance(): DictationProvenance {
	return { sawTyping: false, sawDictation: false, spans: [], lastPath: "" };
}

/** Record one dictation: sticky flag, the new span, and its path. */
export function noteDictation(
	provenance: DictationProvenance,
	span: DictatedSpan,
): DictationProvenance {
	return {
		...provenance,
		sawDictation: true,
		spans: [...provenance.spans, span],
		lastPath: span.path || provenance.lastPath,
	};
}

/**
 * Classify one user edit: sticky ``sawTyping``, and spans intersect-trimmed.
 *
 * A span the edit touches stops counting as a dictated range (the text under
 * it is no longer the transcript that came back), while spans strictly after
 * the edited range shift with the delta. The MODE is not affected — history is
 * sticky by the frozen contract.
 */
export function applyEdit(
	provenance: DictationProvenance,
	edit: DraftEdit,
): DictationProvenance {
	const delta = edit.endNew - edit.endOld;
	const spans = provenance.spans
		.filter((span) => span.end <= edit.start || span.start >= edit.endOld)
		.map((span) =>
			span.start >= edit.endOld
				? { ...span, start: span.start + delta, end: span.end + delta }
				: span,
		);
	return { ...provenance, sawTyping: true, spans };
}

export interface DraftAnnotation {
	input_mode: InputMode;
	input_path?: string;
}

/**
 * The annotation a send carries. Mode: both seen is a mixed message; only a
 * dictation is dictated; otherwise typed (sent explicitly by new clients —
 * absence is the legacy reading for everyone else). Path: the most recent
 * dictated span still present; on the deletion edge (nothing present but a
 * dictation was seen) the last recorded path, so the annotation never
 * contradicts the sticky mode.
 */
export function annotationForSend(provenance: DictationProvenance): DraftAnnotation {
	const mode: InputMode = !provenance.sawDictation
		? "typed"
		: provenance.sawTyping
			? "mixed"
			: "dictated";
	if (mode === "typed") return { input_mode: mode };
	const span = provenance.spans[provenance.spans.length - 1];
	const path = span?.path || provenance.lastPath;
	return path ? { input_mode: mode, input_path: path } : { input_mode: mode };
}
