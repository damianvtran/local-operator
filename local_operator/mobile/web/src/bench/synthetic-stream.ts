/**
 * A deterministic, recorded-style synthetic assistant stream for the relay's
 * streaming benchmark (`streaming-bench.tsx`) and its unit test.
 *
 * WHY SYNTHETIC: a benchmark that is only reproducible against a live model is
 * not a benchmark. The same seed always yields the same text AND the same
 * arrival schedule, so a baseline build and a variant build replay byte-for-byte
 * identical traffic and any difference in the numbers is the code, not the input.
 *
 * SHAPE (matches what the daemon's `projection` events carry in practice):
 *   - ~3-4k tokens (~14k chars) of mixed prose, bold/code/link inlines,
 *     bullet and numbered lists, a fenced code block, a GFM table and headings;
 *   - arrival is BURSTY, not a metronome: mostly 15-40 ms gaps (a model
 *     streaming), runs of near-zero gaps (a transport catching up after a
 *     stalled radio / throttled tab) and occasional 150-400 ms stalls. That
 *     burst shape is what makes an un-paced reveal land as visible "chunks".
 *
 * `chunks[i].length` is the number of source characters visible after the i-th
 * event; the harness emits `text.slice(0, length)` as a whole-snapshot
 * projection, exactly as the daemon does.
 */

export interface StreamChunk {
	/** Wall-clock offset from replay start, ms (already divided by `speed`). */
	atMs: number;
	/** Characters of `text` that have arrived once this event is delivered. */
	length: number;
}

export interface SyntheticStream {
	text: string;
	chunks: StreamChunk[];
}

/** mulberry32: tiny, seedable, identical on every engine. */
function rng(seed: number) {
	let a = seed >>> 0;
	return () => {
		a = (a + 0x6d2b79f5) >>> 0;
		let t = a;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

const WORDS = (
	"relay session stream transcript render commit frame reveal pacing buffer cursor " +
	"markdown token message agent browser phone network socket projection snapshot " +
	"layout paint scroll anchor settle fade opacity compositor thread worker queue " +
	"update change result report review branch release version deploy service daemon " +
	"config setting policy gate check verify measure sample median budget latency " +
	"throughput backlog burst stall steady smooth chunk batch append rewrite final " +
	"signal handler listener observer selection range node text span block inline " +
	"table column header value record field index offset length boundary grapheme"
).split(" ");

export function buildSyntheticStream(seed = 7, targetChars = 14000, speed = 1): SyntheticStream {
	const rand = rng(seed);
	const pick = <T,>(items: readonly T[]) => items[Math.floor(rand() * items.length)];
	const word = () => pick(WORDS);
	const sentence = () => {
		const n = 8 + Math.floor(rand() * 12);
		const out: string[] = [];
		for (let i = 0; i < n; i++) {
			const r = rand();
			let w = word();
			if (r < 0.06) w = `**${w}**`;
			else if (r < 0.12) w = `\`${w}_${word()}\``;
			else if (r < 0.15) w = `[${w}](https://example.test/${word()})`;
			out.push(w);
		}
		const s = out.join(" ");
		return `${s[0].toUpperCase()}${s.slice(1)}.`;
	};
	const paragraph = () => Array.from({ length: 2 + Math.floor(rand() * 3) }, sentence).join(" ");
	const bullets = () => Array.from({ length: 4 + Math.floor(rand() * 3) }, () => `- ${sentence()}`).join("\n");
	const numbered = () => Array.from({ length: 3 + Math.floor(rand() * 3) }, (_, i) => `${i + 1}. ${sentence()}`).join("\n");
	const code = () => "```ts\n" + Array.from({ length: 8 + Math.floor(rand() * 5) }, (_, i) =>
		`const ${word()}${i} = ${word()}(${Math.floor(rand() * 100)}, "${word()}");`).join("\n") + "\n```";
	const table = () => {
		const rows = Array.from({ length: 4 + Math.floor(rand() * 3) }, () =>
			`| ${word()} | ${Math.floor(rand() * 900)} ms | \`${word()}\` |`);
		return ["| Metric | Value | Note |", "| --- | --- | --- |", ...rows].join("\n");
	};
	const heading = () => `## ${word()[0].toUpperCase()}${word().slice(1)} ${word()}`;

	const blocks: string[] = [];
	const kinds = [paragraph, paragraph, bullets, paragraph, heading, numbered, code, paragraph, table];
	let size = 0;
	for (let i = 0; size < targetChars; i++) {
		const block = (i === 0 ? heading : kinds[i % kinds.length])();
		blocks.push(block);
		size += block.length + 2;
	}
	const text = blocks.join("\n\n");

	// Token-ish pieces: a word plus its trailing whitespace, or one punctuation
	// run. Chunks group 1-6 pieces, the way a model's deltas arrive.
	const pieces = text.match(/\S+\s*|\s+/g) ?? [text];
	const chunks: StreamChunk[] = [];
	let at = 0;
	let length = 0;
	let i = 0;
	while (i < pieces.length) {
		const take = 1 + Math.floor(rand() * 6);
		for (let k = 0; k < take && i < pieces.length; k++, i++) length += pieces[i].length;
		const r = rand();
		// stall 6%, burst-run 22% (near-zero gap), normal otherwise.
		const gap = r < 0.06 ? 150 + rand() * 250 : r < 0.28 ? rand() * 3 : 15 + rand() * 25;
		at += gap;
		chunks.push({ atMs: at / speed, length });
	}
	return { text, chunks };
}
