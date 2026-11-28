/**
 * Streaming render benchmark for the relay's live transcript.
 *
 * Drives the REAL path end to end: a stubbed `fetch` returns a genuine SSE byte
 * stream, so the store's SSE parser, its rAF-coalesced emit, `useProjection`,
 * `Transcript`, `Markdown` and the WAAPI reveal all run exactly as on a phone.
 * Only the daemon is replaced, by a deterministic recorded-style stream
 * (`synthetic-stream.ts`). It is built by `vite.bench.config.ts` and is NOT part
 * of the shipped bundle.
 *
 * Run: open `streaming.html?mode=perf|fade&speed=2&burn=0&seed=7&label=x`. The
 * page starts itself, writes the result JSON into `#result` and sets
 * `document.title` to `done` (or `error`). Read it with the harness browser
 * tool's `read`.
 *
 *   mode=perf  frame cadence, long tasks, LoAF scripting, React commits,
 *              first-text latency, reveal steadiness and display lag.
 *   mode=fade  additionally samples the computed opacity of every live reveal
 *              animation on every frame (the visible-fade proof). Kept out of
 *              `perf` because `getComputedStyle` per animation per frame is
 *              itself main-thread work that would colour the perf numbers.
 *   burn=N     busy-waits N ms inside every animation frame: a crude stand-in
 *              for a slower CPU, because the harness browser exposes no CDP
 *              `Emulation.setCPUThrottlingRate`. It adds a constant per frame;
 *              it does not scale the page's own work 4x. Report it as that.
 */
import { Profiler, useEffect } from "react";
import type { ProfilerOnRenderCallback } from "react";
import { createRoot } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { Transcript } from "../components/transcript";
import { Markdown } from "../components/markdown";
import { projectionRevealEpoch, retainProjectionStream, useProjection } from "../store";
import { initTheme } from "../theme";
import type { TranscriptEntry } from "../types";
import { buildSyntheticStream } from "./synthetic-stream";
import "../styles/index.css";

const SID = "bench-session";
const params = new URLSearchParams(location.search);
const MODE = params.get("mode") === "fade" ? "fade" : "perf";
const SPEED = Number(params.get("speed") ?? 2);
/** Source length of the synthetic answer. speed=0.5 with chars=5000 is the
 *  realistic profile (~400 chars/s ~ 100 tokens/s, 100 ms-ish snapshot lumps);
 *  the default speed=2/14000 replays ~1500 chars/s, a deliberate stress profile
 *  roughly 4-5x faster than any model streams. */
const CHARS = Number(params.get("chars") ?? 14000);
const BURN = Number(params.get("burn") ?? 0);
const SEED = Number(params.get("seed") ?? 7);
const LABEL = params.get("label") ?? "";
/** Settled rows above the streaming one. 1 = the minimal page; 60 matches the
 *  mounted-cell count the native audit used to expose per-frame re-render of
 *  every settled row (hypothesis (c)). */
const ROWS = Math.max(1, Number(params.get("rows") ?? 1));
const REVEAL_ID = "lop-live-transcript";

const row = (id: string, text: string, final: boolean): TranscriptEntry => ({
	id, text, kind: "assistant", final, tool_call_id: "", tool_name: "", tool_state: "done",
	summary: "", intent: "", diff_added: 0, diff_removed: 0, elapsed_s: 0, error: "", details: {},
});

/* ---- the daemon stand-in: a real SSE byte stream behind fetch ----------- */
let sink: ReadableStreamDefaultController<Uint8Array> | null = null;
const encoder = new TextEncoder();
window.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
	const url = String(input instanceof Request ? input.url : input);
	if (!url.includes("/events")) return new Response("{}", { status: 404 });
	const body = new ReadableStream<Uint8Array>({ start: (c) => { sink = c; } });
	init?.signal?.addEventListener("abort", () => { try { sink?.close(); } catch { /* already closed */ } });
	return new Response(body, { headers: { "content-type": "text/event-stream" } });
}) as typeof fetch;

function emit(version: number, entries: TranscriptEntry[], streaming: boolean) {
	const data = JSON.stringify({ session_id: SID, pid: 42, version, streaming, transcript: entries });
	sink?.enqueue(encoder.encode(`event: projection\ndata: ${data}\n\n`));
}

/* ---- the page under test ------------------------------------------------ */
const commitMs: number[] = [];
const onRender: ProfilerOnRenderCallback = (_id, _phase, actual) => { commitMs.push(actual); };

function Probe() {
	const { projection, connected } = useProjection(SID);
	useEffect(() => retainProjectionStream(SID), []);
	return projection && (
		<Profiler id="transcript" onRender={onRender}>
			<div className="mx-auto w-full max-w-[390px] px-3">
				<Transcript pid={SID} entries={projection.transcript} streaming={projection.streaming}
					revealEpoch={projectionRevealEpoch(projection, connected)} />
			</div>
		</Profiler>
	);
}

/* ---- statistics --------------------------------------------------------- */
const sorted = (xs: number[]) => [...xs].sort((a, b) => a - b);
const pct = (xs: number[], p: number) => {
	if (!xs.length) return 0;
	const s = sorted(xs);
	return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))];
};
const sum = (xs: number[]) => xs.reduce((a, b) => a + b, 0);
const r1 = (n: number) => Math.round(n * 10) / 10;
const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));
const nextFrame = () => new Promise<number>((resolve) => requestAnimationFrame(resolve));
const answer = () => document.querySelector<HTMLElement>('[data-completion-anchor="answer"]');

/** Rendered text length of a source prefix, via the same component, un-animated. */
function renderedLength(text: string): number {
	const html = renderToStaticMarkup(<Markdown text={text} />);
	return new DOMParser().parseFromString(html, "text/html").body.textContent?.length ?? 0;
}

interface Ramp { samples: { t: number; o: number }[]; done: boolean }

async function run() {
	initTheme();
	createRoot(document.getElementById("root")!).render(<Probe />);
	while (!sink) await sleep(5);

	const { text, chunks } = buildSyntheticStream(SEED, CHARS, SPEED);
	// Calibration happens BEFORE the clock starts: it is not part of the measurement.
	const rendered = chunks.map((chunk) => renderedLength(text.slice(0, chunk.length)));
	const finalRendered = renderedLength(text);
	const history = Array.from({ length: ROWS }, (_, i) => row(`history-${i}`,
		`Previous answer ${i}: **settled** prose with \`code\`, a [link](https://example.test/${i}) and enough words to wrap on a phone column.\n\n- first point\n- second point`, true));
	emit(1, history, true);
	await nextFrame(); await nextFrame(); await sleep(50);

	/* observers */
	const longTasks: number[] = [];
	const loaf = { frames: 0, duration: 0, blocking: 0, script: 0 };
	const observe = (type: string, handle: (entry: PerformanceEntry & Record<string, any>) => void) => {
		try {
			new PerformanceObserver((list) => list.getEntries().forEach((e) => handle(e as never))).observe({ type, buffered: false });
			return true;
		} catch { return false; }
	};
	const hasLongTask = observe("longtask", (e) => longTasks.push(e.duration));
	const hasLoaf = observe("long-animation-frame", (e) => {
		loaf.frames++; loaf.duration += e.duration; loaf.blocking += e.blockingDuration ?? 0;
		for (const s of e.scripts ?? []) loaf.script += s.duration;
	});

	/* per-frame sampling */
	const frameT: number[] = [];
	const frameShown: number[] = [];
	const ramps = new Map<Animation, Ramp>();
	let sampling = true;
	const sample = (now: number) => {
		if (!sampling) return;
		if (BURN > 0) { const end = performance.now() + BURN; while (performance.now() < end) { /* busy */ } }
		frameT.push(now);
		frameShown.push(answer()?.textContent?.length ?? 0);
		if (MODE === "fade") {
			for (const animation of document.getAnimations()) {
				if (animation.id !== REVEAL_ID) continue;
				const target = (animation.effect as KeyframeEffect | null)?.target as HTMLElement | null;
				if (!target) continue;
				let ramp = ramps.get(animation);
				if (!ramp) {
					ramp = { samples: [], done: false };
					ramps.set(animation, ramp);
					const mark = (ok: boolean) => { if (ramp) ramp.done = ok; };
					animation.finished.then(() => mark(true), () => mark(false));
				}
				ramp.samples.push({ t: now, o: Number(getComputedStyle(target).opacity) });
			}
		}
		requestAnimationFrame(sample);
	};

	const commitsBefore = commitMs.length;
	const t0 = performance.now();
	requestAnimationFrame(sample);

	/* replay: every due chunk is delivered in the same task, so a late timer
	   under load produces exactly the burst a stalled radio would. */
	const emitAt: number[] = [];
	let version = 2;
	await new Promise<void>((resolve) => {
		let i = 0;
		const pump = () => {
			const now = performance.now() - t0;
			while (i < chunks.length && chunks[i].atMs <= now) {
				emit(version++, [...history, row("answer", text.slice(0, chunks[i].length), false)], true);
				emitAt[i++] = performance.now();
			}
			if (i < chunks.length) setTimeout(pump, Math.max(0, chunks[i].atMs - (performance.now() - t0)));
			else resolve();
		};
		pump();
	});
	const lastEmit = performance.now();
	await sleep(20);
	const finalEmit = performance.now();
	emit(version++, [...history, row("answer", text, true)], false);

	/* wait for the display to settle on the full text (bounded) */
	let drained = 0;
	const deadline = performance.now() + 8000;
	while (performance.now() < deadline) {
		await nextFrame();
		if (frameShown.length && frameShown[frameShown.length - 1] >= finalRendered - 1) {
			await sleep(300); // let the last fades finish before sampling stops
			drained = frameT[frameT.length - 1];
			break;
		}
	}
	// `drained` above includes the 300 ms settle; the display-complete frame is earlier.
	const completeIdx = frameShown.findIndex((n) => n >= finalRendered - 1);
	const completeAt = completeIdx >= 0 ? frameT[completeIdx] : 0;
	sampling = false;
	void drained;

	/* ---- derive ---- */
	const deltas = frameT.slice(1).map((t, i) => t - frameT[i]);
	const firstEmit = emitAt[0];
	// first visible text: first frame whose shown length covers the first chunk.
	const firstIdx = frameShown.findIndex((n) => n >= Math.max(1, rendered[0] - 2));
	const firstVisible = firstIdx >= 0 ? frameT[firstIdx] : NaN;

	// reveal steadiness: characters newly visible per frame while text is arriving
	const perFrame: number[] = [];
	for (let i = 1; i < frameShown.length; i++) {
		const d = frameShown[i] - frameShown[i - 1];
		if (d > 0) perFrame.push(d);
	}
	// display lag: when did each received chunk become visible? (running-max, two pointers)
	const lags: number[] = [];
	let f = 0;
	let need = 0;
	for (let i = 0; i < chunks.length; i++) {
		need = Math.max(need, rendered[i] - 6);
		while (f < frameShown.length && frameShown[f] < need) f++;
		const seen = f < frameT.length ? frameT[f] : frameT[frameT.length - 1];
		lags.push(Math.max(0, seen - emitAt[i]));
	}

	const rampStats = [...ramps.values()].filter((r) => r.done && r.samples.length >= 1).map((r) => {
		const os = r.samples.map((s) => s.o);
		return {
			frames: os.length,
			between: os.filter((o) => o > 0.03 && o < 0.97).length,
			start: os[0],
			monotonic: os.every((o, i) => i === 0 || o >= os[i - 1] - 0.005),
			spanMs: r.samples[r.samples.length - 1].t - r.samples[0].t,
		};
	});

	const streamMs = (completeAt || frameT[frameT.length - 1]) - t0;
	const slice = commitMs.slice(commitsBefore);
	const result = {
		label: LABEL, mode: MODE, speed: SPEED, burn: BURN, seed: SEED, rows: ROWS,
		charsPerSec: Math.round(text.length / ((chunks[chunks.length - 1]?.atMs ?? 1) / 1000)),
		chars: text.length, events: chunks.length,
		frames: {
			n: deltas.length, p50: r1(pct(deltas, 50)), p95: r1(pct(deltas, 95)), max: r1(Math.max(0, ...deltas)),
			over20: deltas.filter((d) => d > 20).length, over33: deltas.filter((d) => d > 33).length,
			over50: deltas.filter((d) => d > 50).length,
		},
		longTasks: { supported: hasLongTask, count: longTasks.length, totalMs: r1(sum(longTasks)), maxMs: r1(Math.max(0, ...longTasks)) },
		loaf: { supported: hasLoaf, frames: loaf.frames, durationMs: r1(loaf.duration), blockingMs: r1(loaf.blocking), scriptMs: r1(loaf.script) },
		react: {
			commits: slice.length, perSec: r1(slice.length / (streamMs / 1000)), totalMs: r1(sum(slice)),
			p50: r1(pct(slice, 50)), p95: r1(pct(slice, 95)), max: r1(Math.max(0, ...slice)),
		},
		firstTextMs: r1(firstVisible - firstEmit),
		reveal: {
			steps: perFrame.length, charsPerStepP50: pct(perFrame, 50), charsPerStepP95: pct(perFrame, 95),
			charsPerStepMax: Math.max(0, ...perFrame), stepsOver40Chars: perFrame.filter((d) => d > 40).length,
		},
		lagMs: { p50: r1(pct(lags, 50)), p95: r1(pct(lags, 95)), max: r1(Math.max(0, ...lags)) },
		completeAfterFinalMs: r1(completeAt ? completeAt - finalEmit : -1),
		completeAfterLastChunkMs: r1(completeAt ? completeAt - lastEmit : -1),
		fade: MODE === "fade" ? {
			animations: rampStats.length,
			cancelled: [...ramps.values()].filter((r) => !r.done).length,
			medianFrames: pct(rampStats.map((s) => s.frames), 50),
			medianFramesStrictlyBetween: pct(rampStats.map((s) => s.between), 50),
			p10FramesStrictlyBetween: pct(rampStats.map((s) => s.between), 10),
			minStartOpacity: Math.round(Math.min(...rampStats.map((s) => s.start), 1) * 100) / 100,
			monotonicPct: r1((rampStats.filter((s) => s.monotonic).length / Math.max(1, rampStats.length)) * 100),
			medianSpanMs: r1(pct(rampStats.map((s) => s.spanMs), 50)),
			ramps8PlusFrames: rampStats.filter((s) => s.between >= 8).length,
		} : undefined,
		// Nothing dropped: the settled DOM text length equals a fresh render of the full text.
		textIntact: Math.abs((answer()?.textContent?.length ?? 0) - finalRendered) <= 1,
	};
	document.getElementById("result")!.textContent = JSON.stringify(result);
	document.title = "done";
}

void run().catch((error) => {
	document.getElementById("result")!.textContent = `ERROR ${String(error?.stack ?? error)}`;
	document.title = "error";
});
