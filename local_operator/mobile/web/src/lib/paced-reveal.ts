/**
 * Paced release of streamed answer text (relay web).
 *
 * WHY THIS EXISTS. The daemon sends the whole conversation in bursts (at most
 * every ~100 ms, and a throttled tab or stalled radio delivers several at once),
 * so a streamed answer reaches the store as lumps of 30-150 characters. Drawn as
 * received, each lump is one visible "chunk": the fade in `stream-reveal.ts`
 * can only soften a lump, not turn it into a flow. Measured on the relay's own
 * benchmark (`src/bench/streaming-bench.tsx`, before this module): median 28
 * characters per painted step, p95 74, max 139, and 31% of steps over 40
 * characters. This module owns the pacing the transport deliberately does not
 * do: it holds back what has been received and releases it a word or a few at a
 * time, so each tick adds a small suffix that `Markdown` fades in.
 *
 * It is the web twin of the mobile app's `paced-reveal.ts`
 * (damianvtran/local-operator-mobile#399) and keeps its contract; it is NOT a
 * copy (no per-character alpha runs here: the fade is a WAAPI opacity ramp on
 * the appended span, see `stream-reveal.ts`).
 *
 * CONSTRAINTS (each has a test in `paced-reveal.test.ts`):
 *  - NOTHING IS DROPPED, REORDERED OR DUPLICATED. `shown()` is always a prefix
 *    of the received text and `flush()` reveals the rest, so the text a row ends
 *    on is byte-for-byte the text it received.
 *  - BOUNDED LAG. Text is never held longer than `MAX_LAG_MS` after the snapshot
 *    that delivered it, however large a burst is.
 *  - WORD BOUNDARIES. A word is released once the whitespace after it has
 *    arrived, so a half-received "wor" never shows and then "ds" jumps in. The
 *    exceptions are the very first word (first-text latency must not regress),
 *    ideographic text (no spaces: each character is a unit), and the lag bound.
 *    A unit never ends before a combining mark, joiner or emoji modifier.
 *  - A REWRITE IS NOT PACED. If the new text is not an extension of the old (a
 *    correction), everything is shown at once: pacing text the reader has
 *    already seen would replay it.
 *
 * PURE: no timers, no React, no clock of its own; the caller passes `now`. The
 * scheduling (one ~30 Hz timer, only while something is held back) lives in
 * `use-paced-text.ts`.
 */

/** Tick period (ms): the cursor, hence the paragraph's re-render, moves no faster than ~30 Hz. */
export const TICK_MS = 34;
/** The backlog drains exponentially on this time constant: a burst is spread over a few ticks, not dropped in one. */
const DRAIN_MS = 200;
/** No text waits longer than this after the snapshot that delivered it. */
export const MAX_LAG_MS = 300;
/** Units scanned per tick when sizing the backlog; a first-open burst of thousands of words only needs to know it is "large". */
const SCAN_CAP = 400;

const WHITESPACE = /\s/u;
// Ideographs and kana carry no spaces: each character is its own unit, or the
// paragraph would be held until the lag bound and then dropped in one chunk.
const IDEOGRAPH = /[\u2e80-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef]/u;
// A unit must not end before one of these, or a cluster would be cut in two.
const JOINING_START = /^(?:\p{M}|\p{Emoji_Modifier}|\u200c|\u200d|\ufe0f)/u;

const codePointAt = (text: string, at: number) => String.fromCodePoint(text.codePointAt(at) ?? 0);

/**
 * The end of the unit starting at `from`, or -1 when no complete unit is there.
 * A unit is a word plus the whitespace after it (the space travels with the word
 * before it), or a single ideograph, or a bare whitespace run.
 */
export function unitEnd(text: string, from: number): number {
	const length = text.length;
	let at = from;
	if (at >= length) return -1;
	const first = codePointAt(text, at);
	if (!WHITESPACE.test(first) && IDEOGRAPH.test(first)) {
		const end = at + first.length;
		return JOINING_START.test(text.slice(end, end + 2)) ? -1 : end;
	}
	while (at < length) {
		const ch = codePointAt(text, at);
		if (WHITESPACE.test(ch) || IDEOGRAPH.test(ch)) break;
		at += ch.length;
	}
	const wordEnd = at;
	while (at < length) {
		const ch = codePointAt(text, at);
		if (!WHITESPACE.test(ch)) break;
		at += ch.length;
	}
	// No whitespace after the word yet = the word may still be growing.
	if (at === wordEnd && wordEnd === length) return -1;
	return JOINING_START.test(text.slice(at, at + 2)) ? -1 : at;
}

type Arrival = { len: number; at: number };

export class PacedReveal {
	private text: string;
	private cursor: number;
	private arrivals: Arrival[] = [];
	private lastTick: number;

	/**
	 * @param hold false = the row was already on screen when pacing started (a
	 *   cell mounting mid-stream, a reconnect seed): show what is there and pace
	 *   only what arrives afterwards.
	 */
	constructor(text: string, now: number, hold: boolean) {
		this.text = text;
		this.cursor = hold ? 0 : text.length;
		this.lastTick = now;
		if (hold) {
			this.arrivals.push({ len: text.length, at: now });
			this.revealFirst();
		}
	}

	/** The text the reader should see now: a prefix of everything received. */
	shown(): string {
		return this.text.slice(0, this.cursor);
	}

	/** Everything received has been revealed: no timer is needed. */
	idle(): boolean {
		return this.cursor >= this.text.length;
	}

	/** Reveal everything now (settle, selection, hidden tab, reduced motion). */
	flush(): void {
		this.cursor = this.text.length;
		this.arrivals = [];
	}

	/** Record what the stream has delivered so far. Idempotent for equal text. */
	push(text: string, now: number): void {
		if (text === this.text) return;
		if (!text.startsWith(this.text)) {
			// A correction rewrote the answer: show it, do not replay it.
			this.text = text;
			this.flush();
			return;
		}
		this.text = text;
		this.arrivals.push({ len: text.length, at: now });
		if (this.cursor === 0) this.revealFirst();
	}

	/** Release the next words. Returns true when the visible text changed. */
	tick(now: number): boolean {
		const dt = Math.max(0, Math.min(now - this.lastTick, 100));
		this.lastTick = now;
		if (this.cursor >= this.text.length) return false;
		const ends: number[] = [];
		for (let at = this.cursor; ends.length < SCAN_CAP; ) {
			const end = unitEnd(this.text, at);
			if (end < 0) break;
			ends.push(end);
			at = end;
		}
		let next = this.cursor;
		if (ends.length > 0) {
			const count = Math.min(ends.length, Math.max(1, Math.ceil((ends.length * Math.max(dt, 1)) / DRAIN_MS)));
			next = ends[count - 1] ?? next;
		}
		// The lag bound: whatever a snapshot delivered MAX_LAG_MS ago is shown now,
		// even mid-word, so a stalled half-word cannot hold the answer back.
		for (const arrival of this.arrivals) {
			if (now - arrival.at >= MAX_LAG_MS && arrival.len > next) next = arrival.len;
		}
		return this.reveal(next);
	}

	private revealFirst(): void {
		// First text must not wait for a boundary: reveal the first word, or all
		// of what exists if no word is complete yet.
		const end = unitEnd(this.text, 0);
		this.reveal(end > 0 ? end : this.text.length);
	}

	private reveal(next: number): boolean {
		if (next <= this.cursor) return false;
		this.cursor = Math.min(next, this.text.length);
		this.arrivals = this.arrivals.filter((arrival) => arrival.len > this.cursor);
		return true;
	}
}
