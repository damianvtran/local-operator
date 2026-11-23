// @vitest-environment happy-dom
import { act, cleanup, render } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { Transcript } from "./components/transcript";
import { FetchStreamFixture as Stream, settleStream } from "./sse-test-fixture";
import { projectionRevealEpoch, retainProjectionStream, useProjection } from "./store";
import type { TranscriptEntry } from "./types";

const SID = "stream-reveal-fixture";
const row = (id: string, text: string, overrides: Partial<TranscriptEntry> = {}): TranscriptEntry => ({
	id, text, kind: "assistant", final: false, tool_call_id: "", tool_name: "", tool_state: "done",
	summary: "", intent: "", diff_added: 0, diff_removed: 0, elapsed_s: 0, error: "", details: {}, ...overrides,
});
const history = row("history", "Previous answer", { final: true });

function Probe() {
	const { projection, connected } = useProjection(SID);
	useEffect(() => retainProjectionStream(SID), []);
	return projection && <Transcript pid={SID} entries={projection.transcript} streaming={projection.streaming}
		revealEpoch={projectionRevealEpoch(projection, connected)} />;
}

async function frame(source: Stream, version: number, entries: TranscriptEntry[], streaming = true) {
	await act(async () => {
		source.emit("projection", { session_id: SID, pid: 42, version, streaming, transcript: entries });
		await settleStream();
	});
}

const noPreference = Object.assign(new EventTarget(), { matches: true });
const effects: { target: HTMLElement; text: string; id: string; finished: Promise<void>; cancel: ReturnType<typeof vi.fn>; frames: Keyframe[] }[] = [];
const originalAnimate = Object.getOwnPropertyDescriptor(Element.prototype, "animate");
const originalGetAnimations = Object.getOwnPropertyDescriptor(Element.prototype, "getAnimations");
function restore(name: string, original: PropertyDescriptor | undefined) {
	if (original) Object.defineProperty(Element.prototype, name, original);
	else Reflect.deleteProperty(Element.prototype, name);
}
const anchor = (id: string) => document.querySelector(`[data-completion-anchor="${id}"]`);

beforeEach(() => {
	vi.useFakeTimers();
	Stream.install();
	vi.spyOn(window, "matchMedia").mockImplementation(() => noPreference as unknown as MediaQueryList);
	noPreference.matches = true;
	effects.length = 0;
	Object.defineProperty(Element.prototype, "animate", { configurable: true, value(this: HTMLElement, frames: Keyframe[]) {
		let timer: ReturnType<typeof setTimeout>;
		let reject!: (reason: unknown) => void;
		const finished = new Promise<void>((resolve, rejected) => { timer = setTimeout(resolve, 120); reject = rejected; });
		// Row effects do not subscribe, matching browser cancellation semantics.
		void finished.catch(() => undefined);
		const cancel = vi.fn(() => { clearTimeout(timer); reject(new DOMException("cancelled", "AbortError")); });
		const effect = { target: this, text: this.textContent ?? "", id: "", finished, cancel, frames };
		effects.push(effect);
		return effect;
	} });
	Object.defineProperty(Element.prototype, "getAnimations", { configurable: true, value(this: HTMLElement, { subtree } = { subtree: false }) {
		return effects.filter(effect => !effect.cancel.mock.calls.length &&
			(this === effect.target || (subtree && this.contains(effect.target))));
	} });
});

afterEach(async () => {
	cleanup();
	await vi.advanceTimersByTimeAsync(10001);
	vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks();
	restore("animate", originalAnimate); restore("getAnimations", originalGetAnimations);
});

it("reveals only live suffix and tool arrivals, preserving markup and settling correction/final immediately", async () => {
	render(<Probe />);
	await settleStream();
	const source = Stream.instances[0];
	await frame(source, 1, [history]);
	expect(effects).toHaveLength(0);
	await frame(source, 2, [history, row("answer", "**Ready** for this")]);
	expect(effects.map(effect => effect.text.trimStart())).toEqual(["Ready for this"]);
	const bold = anchor("answer")?.querySelector("strong");
	await frame(source, 3, [history, row("answer", "**Ready** for this new addition")]);
	expect(effects.map(effect => effect.text.trimStart())).toEqual(["Ready for this", "new addition"]);
	expect(effects[1].frames).toEqual([{ opacity: 0.65 }, { opacity: 1 }]);
	expect(effects[0].cancel).toHaveBeenCalledOnce();
	expect(anchor("answer")?.querySelector("strong")).toBe(bold);
	const tool = row("tool", "", { kind: "tool", tool_call_id: "read", tool_name: "read_file", summary: "README.md", final: true });
	await frame(source, 4, [history, row("answer", "**Ready** for this new addition"), tool]);
	expect(effects[2].target).toBe(anchor("tool"));
	await frame(source, 5, [history, row("answer", "Corrected right now"), tool]);
	expect(anchor("answer")?.textContent).toBe("Corrected right now");
	// A correction may remove the old suffix node entirely. Either way the
	// corrected row must have no running effect on its visible subtree.
	expect(anchor("answer")?.getAnimations({ subtree: true })).toHaveLength(0);
	await frame(source, 6, [history, row("answer", "Corrected right now. Final write.", { final: true }), tool], false);
	expect(anchor("answer")?.textContent).toBe("Corrected right now. Final write.");
	expect(effects).toHaveLength(3);
	expect(effects[2].cancel).toHaveBeenCalledOnce();
});

it("treats a clean SSE reconnect as a baseline and resumes only with the next live append", async () => {
	render(<Probe />);
	await settleStream();
	const first = Stream.instances[0];
	await frame(first, 100, [history, row("answer", "Original seed")]);
	await act(async () => { await vi.advanceTimersByTimeAsync(6000); first.end(); await vi.advanceTimersByTimeAsync(250); });
	const second = Stream.instances[1];
	await frame(second, 1, [history, row("answer", "Original seed plus reconnect catchup")]);
	expect(effects).toHaveLength(0);
	await frame(second, 2, [history, row("answer", "Original seed plus reconnect catchup and live")]);
	expect(effects.map(effect => effect.text.trimStart())).toEqual(["and live"]);
});

it("keeps joined RTL and emoji text intact and settles when reduced motion changes", async () => {
	render(<Probe />);
	await settleStream();
	const source = Stream.instances[0];
	await frame(source, 1, [history, row("answer", "مرحبا 👩")]);
	await frame(source, 2, [history, row("answer", "مرحبا 👩‍🔬 الباحثة")]);
	expect(anchor("answer")?.textContent).toBe("مرحبا 👩‍🔬 الباحثة");
	expect(effects.map(effect => effect.text.trimStart())).toEqual(["الباحثة"]);
	noPreference.matches = false;
	noPreference.dispatchEvent(new Event("change"));
	expect(effects[0].cancel).toHaveBeenCalledOnce();
	await frame(source, 3, [history, row("answer", "مرحبا 👩‍🔬 الباحثة هنا")]);
	expect(anchor("answer")?.textContent).toBe("مرحبا 👩‍🔬 الباحثة هنا");
	expect(effects).toHaveLength(1);
});

it("retires finished suffix wrappers through a sustained paragraph and same-text final", async () => {
	render(<Probe />);
	await settleStream();
	const source = Stream.instances[0];
	let text = "[Reference](https://example.test) followed by steady prose";
	await frame(source, 1, [history, row("answer", text)]);
	const paragraph = anchor("answer")!.querySelector("p")!;
	const link = paragraph.querySelector("a")!;
	link.focus();
	const selection = document.getSelection()!;
	let selected: Node | undefined;
	for (let i = 0; i < 200; i++) {
		text += ` word${i}`;
		await frame(source, i + 2, [history, row("answer", text)]);
		if (i === 0) {
			selected = effects[0].target.firstChild!;
			selection.setBaseAndExtent(selected, 1, selected, 5);
		}
		await act(async () => { await vi.advanceTimersByTimeAsync(120); });
	}
	await frame(source, 202, [history, row("answer", text, { final: true })], false);
	expect(paragraph.textContent).toBe(text.replace("[Reference](https://example.test)", "Reference"));
	// The sole extra wrapper contains the active selection. Later frames
	// cannot accumulate wrappers, and a collapsed caret must not retain it.
	expect(paragraph.querySelectorAll("span")).toHaveLength(2);
	expect(selection.anchorNode).toBe(selected);
	expect(selection.toString()).toBe("word");
	expect(document.activeElement).toBe(link);
	await act(async () => {
		selection.collapse(selected!, 1);
		document.dispatchEvent(new Event("selectionchange"));
	});
	expect(paragraph.querySelectorAll("span")).toHaveLength(1);
	expect(effects).toHaveLength(200);
});

it("seeds a reported outage's recovery frame without fading and reveals its first fresh append", async () => {
	render(<Probe />);
	await settleStream();
	await frame(Stream.instances[0], 100, [history, row("answer", "Original seed")]);
	const outage = new Response(null, { status: 503, headers: { "Retry-After": "3" } });
	Stream.fetch.mockResolvedValueOnce(outage);
	await act(async () => { Stream.instances[0].fail(); await vi.advanceTimersByTimeAsync(3750); });
	const recovered = Stream.instances[1];
	await frame(recovered, 1, [history, row("answer", "Original seed plus catchup")]);
	await act(async () => { await vi.advanceTimersByTimeAsync(2100); });
	expect(effects).toHaveLength(0);
	await frame(recovered, 2, [history, row("answer", "Original seed plus catchup and live")]);
	expect(effects.map(effect => effect.text.trimStart())).toEqual(["and live"]);
});
