// @vitest-environment happy-dom
import { act, cleanup, render } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { Markdown } from "./components/markdown";
import { Transcript } from "./components/transcript";
import { TICK_MS } from "./lib/paced-reveal";
import { FetchStreamFixture as Stream, settleStream } from "./sse-test-fixture";
import { projectionRevealEpoch, retainProjectionStream, useProjection } from "./store";
import type { TranscriptEntry } from "./types";

/* The word-by-word release in front of the fade (`lib/paced-reveal.ts`,
   `lib/use-paced-text.ts`): the contract the operator asked for is that
   streamed text flows instead of landing in lumps, WITHOUT ever changing what
   the answer finally says. Each test pins one clause of that contract. */

const SID = "paced-reveal-fixture";
const row = (id: string, text: string, overrides: Partial<TranscriptEntry> = {}): TranscriptEntry => ({
	id, text, kind: "assistant", final: false, tool_call_id: "", tool_name: "", tool_state: "done",
	summary: "", intent: "", diff_added: 0, diff_removed: 0, elapsed_s: 0, error: "", details: {}, ...overrides,
});
const history = row("history", "Previous answer", { final: true });
const anchor = (id: string) => document.querySelector(`[data-completion-anchor="${id}"]`);
const noPreference = Object.assign(new EventTarget(), { matches: true });

function Probe() {
	const { projection, connected } = useProjection(SID);
	useEffect(() => retainProjectionStream(SID), []);
	return projection && <Transcript pid={SID} entries={projection.transcript} streaming={projection.streaming}
		inlineUi={projection.inline_ui} revealEpoch={projectionRevealEpoch(projection, connected)} />;
}

async function frame(source: Stream, version: number, entries: TranscriptEntry[], streaming = true) {
	await act(async () => {
		source.emit("projection", { session_id: SID, pid: 42, version, streaming, transcript: entries });
		await settleStream();
	});
}
const tick = (ms = TICK_MS) => act(async () => { await vi.advanceTimersByTimeAsync(ms); });

beforeEach(() => {
	vi.useFakeTimers();
	Stream.install();
	vi.spyOn(window, "matchMedia").mockImplementation(() => noPreference as unknown as MediaQueryList);
	noPreference.matches = true;
});

afterEach(async () => {
	cleanup();
	await vi.advanceTimersByTimeAsync(10001);
	vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks();
	Object.defineProperty(document, "hidden", { configurable: true, value: false });
});

it("shows a new answer's first word at once and releases the rest word by word, ending byte-identical", async () => {
	render(<Probe />);
	await settleStream();
	const source = Stream.instances[0];
	await frame(source, 1, [history]);
	const idleTimers = vi.getTimerCount();
	const burst = "Streaming answers arrive in lumps but should read as a steady flow of words.";
	await frame(source, 2, [history, row("answer", burst)]);
	// First word immediate: never an empty row, never the whole lump.
	expect(anchor("answer")?.textContent).toBe("Streaming ");
	const seen: string[] = [];
	for (let i = 0; i < 40 && anchor("answer")?.textContent !== burst; i++) {
		await tick();
		seen.push(anchor("answer")?.textContent ?? "");
	}
	expect(anchor("answer")?.textContent).toBe(burst);
	// More than one step, each a word-boundary prefix of the received text.
	expect(new Set(seen).size).toBeGreaterThan(2);
	for (const shown of seen) {
		expect(burst.startsWith(shown)).toBe(true);
		expect(shown === burst || /\s$/.test(shown)).toBe(true);
	}
	// A growing answer keeps pacing only what arrives after what is on screen.
	await frame(source, 3, [history, row("answer", `${burst} And a second burst of words.`)]);
	expect((anchor("answer")?.textContent ?? "").startsWith(burst)).toBe(true);
	expect(anchor("answer")?.textContent).not.toBe(`${burst} And a second burst of words.`);
	// Final: whatever is held back is released, text byte-identical, no timer left.
	await frame(source, 4, [history, row("answer", `${burst} And a second burst of words.`, { final: true })], false);
	expect(anchor("answer")?.textContent).toBe(`${burst} And a second burst of words.`);
	expect(vi.getTimerCount()).toBe(idleTimers);
});

it("never paces under reduced motion: the received text is the text on screen", async () => {
	noPreference.matches = false;
	render(<Probe />);
	await settleStream();
	const source = Stream.instances[0];
	await frame(source, 1, [history]);
	const idleTimers = vi.getTimerCount();
	const burst = "Reduced motion readers get every word at once without any held back text.";
	await frame(source, 2, [history, row("answer", burst)]);
	expect(anchor("answer")?.textContent).toBe(burst);
	expect(vi.getTimerCount()).toBe(idleTimers);
});

it("does not hold back a row that mounts mid-stream, only what arrives afterwards", () => {
	const seed = "Already on screen before the first live frame arrived.";
	const view = render(<Markdown text={seed} />);
	view.rerender(<Markdown text={seed} live />);
	expect(view.container.textContent).toBe(seed);
	view.rerender(<Markdown text={`${seed} Then a burst of newly arrived words follows.`} live />);
	expect(view.container.textContent?.startsWith(seed)).toBe(true);
	expect(view.container.textContent).not.toBe(`${seed} Then a burst of newly arrived words follows.`);
});

it("shows a correction at once instead of replaying the pace", async () => {
	const view = render(<Markdown text="First draft of an answer that is long enough " live />);
	view.rerender(<Markdown text="First draft of an answer that is long enough to pace the tail words " live />);
	view.rerender(<Markdown text="A different corrected answer replaces the draft entirely " live />);
	expect(view.container.textContent).toBe("A different corrected answer replaces the draft entirely ");
});

it("releases the backlog when the tab is hidden and when the reader selects the answer", async () => {
	const text = `${"lorem ipsum dolor ".repeat(30)}end`;
	const view = render(<Markdown text="lorem " live />);
	view.rerender(<Markdown text={text} live />);
	expect(view.container.textContent?.length).toBeLessThan(text.length);
	Object.defineProperty(document, "hidden", { configurable: true, value: true });
	await tick();
	expect(view.container.textContent).toBe(text);
	Object.defineProperty(document, "hidden", { configurable: true, value: false });

	const next = `${text} ${"more words arrive ".repeat(20)}tail`;
	view.rerender(<Markdown text={next} live />);
	expect(view.container.textContent?.length).toBeLessThan(next.length);
	const node = document.createTreeWalker(view.container, NodeFilter.SHOW_TEXT).nextNode()!;
	document.getSelection()!.setBaseAndExtent(node, 0, node, 5);
	await tick();
	expect(view.container.textContent).toBe(next);
});

it("clears its timer on unmount mid-reveal", async () => {
	const idleTimers = vi.getTimerCount();
	const view = render(<Markdown text="Hello " live />);
	view.rerender(<Markdown text={`Hello ${"words ".repeat(60)}`} live />);
	expect(vi.getTimerCount()).toBeGreaterThan(idleTimers);
	view.unmount();
	expect(vi.getTimerCount()).toBe(idleTimers);
});
