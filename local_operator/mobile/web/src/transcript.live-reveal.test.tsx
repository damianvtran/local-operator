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
		inlineUi={projection.inline_ui} revealEpoch={projectionRevealEpoch(projection, connected)} />;
}

async function frame(source: Stream, version: number, entries: TranscriptEntry[], streaming = true, inline_ui?: unknown) {
	await act(async () => {
		source.emit("projection", { session_id: SID, pid: 42, version, streaming, transcript: entries, inline_ui });
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

it("keeps a negotiated widget mounted while only the following prose fades", async () => {
	render(<Probe />);
	await settleStream();
	const source = Stream.instances[0];
	const before = "**Ready** for the figures.\n\n";
	const payload = { type: "table", title: "Exact counts", columns: ["Total"], rows: [["9007199254740993"]] };
	const sidecar = (text: string) => ({ version: 1, messages: {
		answer: { version: 1, text, widgets: [{ offset: before.length, payload }] },
	} });
	await frame(source, 1, [history, row("answer", before)]);
	expect(anchor("answer")?.querySelector("table")).toBeNull();
	const closed = `${before}Counts verified`;
	await frame(source, 2, [history, row("answer", closed)], true, sidecar(closed));
	const table = anchor("answer")?.querySelector("table");
	expect(table?.textContent).toContain("9007199254740993");
	expect(effects).toHaveLength(0);
	const appended = `${closed} against the report.`;
	await frame(source, 3, [history, row("answer", appended)], true, sidecar(appended));
	expect(effects.map(effect => effect.text.trimStart())).toEqual(["against the report."]);
	expect(anchor("answer")?.querySelector("table")).toBe(table);
	expect(anchor("answer")?.querySelector("strong")?.textContent).toBe("Ready");
	await frame(source, 4, [history, row("answer", appended, { final: true })], false, sidecar(appended));
	expect(anchor("answer")?.querySelector("table")).toBe(table);
	expect(anchor("answer")?.getAnimations({ subtree: true })).toHaveLength(0);
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

it.each(["plain", "wrapped"] as const)("keeps both range endpoints when selection moves to the %s run after a retired fade", async (shape) => {
	render(<Probe />);
	await settleStream();
	const source = Stream.instances[0];
	const text = "Start here first second";
	await frame(source, 1, [history, row("answer", "Start here")]);
	const line = anchor("answer")!.querySelector("p > span")!;
	const selection = document.getSelection()!;
	await frame(source, 2, [history, row("answer", "Start here first")]);
	const first = effects[0].target.firstChild!;
	selection.setBaseAndExtent(first, 1, first, 6);
	const settle = () => act(async () => { await vi.advanceTimersByTimeAsync(120); });
	await settle();
	expect(selection.anchorNode).toBe(first);
	expect(first.isConnected).toBe(true);
	await frame(source, 3, [history, row("answer", text)]);
	const secondHost = effects[1].target;
	if (shape === "wrapped") {
		// Keep the second wrapper past animation settlement with a range
		// spanning both suffixes, then narrow to the right endpoint alone.
		selection.setBaseAndExtent(first, 1, secondHost.firstChild!, 7);
	}
	await settle();
	const second = shape === "plain" ? line.lastChild! : secondHost.firstChild!;
	expect(second.nodeType).toBe(Node.TEXT_NODE);
	expect(second.textContent).toBe(" second");
	expect(second.parentNode).toBe(shape === "plain" ? line : secondHost);
	await act(async () => {
		selection.setBaseAndExtent(second, 1, second, 7);
		document.dispatchEvent(new Event("selectionchange"));
	});
	await settle();
	expect(second.isConnected).toBe(true);
	expect([selection.anchorNode, selection.anchorOffset, selection.focusNode, selection.focusOffset])
		.toEqual([second, 1, second, 7]);
	expect(selection.toString()).toBe("second");
	expect(line.textContent).toBe(text);
	expect(effects).toHaveLength(2);
	await act(async () => {
		selection.removeAllRanges();
		document.dispatchEvent(new Event("selectionchange"));
	});
	expect(line.childNodes).toHaveLength(1);
	expect(line.firstChild?.nodeType).toBe(Node.TEXT_NODE);
	expect(line.textContent).toBe(text);
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
	for (let i = 0; i < 204; i++) {
		text += ` word${i}`;
		await frame(source, i + 2, [history, row("answer", text)]);
		if (i === 0) {
			selected = effects[0].target.firstChild!;
			selection.setBaseAndExtent(selected, 1, selected, 5);
		}
		await act(async () => { await vi.advanceTimersByTimeAsync(120); });
		if (i === 1) {
			selected = paragraph.firstElementChild!.lastChild!;
			expect(selected.nodeType).toBe(Node.TEXT_NODE);
			expect(selected.textContent).toBe(" word1");
			await act(async () => {
				selection.setBaseAndExtent(selected!, 1, selected!, 5);
				document.dispatchEvent(new Event("selectionchange"));
			});
		}
		expect(paragraph.querySelectorAll("span")).toHaveLength(2);
	}
	await frame(source, 206, [history, row("answer", text, { final: true })], false);
	expect(paragraph.textContent).toBe(text.replace("[Reference](https://example.test)", "Reference"));
	// The sole extra wrapper protects the following selected plain node.
	// A collapsed caret no longer blocks the merge at that seam.
	expect(paragraph.querySelectorAll("span")).toHaveLength(2);
	expect(selection.anchorNode).toBe(selected);
	expect(selection.toString()).toBe("word");
	expect(document.activeElement).toBe(link);
	await act(async () => {
		selection.collapse(selected!, 1);
		document.dispatchEvent(new Event("selectionchange"));
	});
	expect(paragraph.querySelectorAll("span")).toHaveLength(1);
	expect(effects).toHaveLength(204);
});

it("does not retain fade hosts for a range elsewhere or a caret in the fade", async () => {
	render(<Probe />);
	await settleStream();
	const source = Stream.instances[0];
	await frame(source, 1, [history, row("answer", "Start here")]);
	const line = anchor("answer")!.querySelector("p > span")!;
	const elsewhere = anchor("history")!.querySelector("p > span")!.firstChild!;
	const selection = document.getSelection()!;
	selection.setBaseAndExtent(elsewhere, 1, elsewhere, 4);
	await frame(source, 2, [history, row("answer", "Start here first")]);
	await act(async () => { await vi.advanceTimersByTimeAsync(120); });
	expect(line.childNodes).toHaveLength(1);
	expect(selection.anchorNode).toBe(elsewhere);
	expect(selection.toString()).toBe("rev");
	await frame(source, 3, [history, row("answer", "Start here first second")]);
	selection.collapse(effects[1].target.firstChild!, 1);
	await act(async () => { await vi.advanceTimersByTimeAsync(120); });
	expect(line.childNodes).toHaveLength(1);
	expect(line.textContent).toBe("Start here first second");
	expect(effects).toHaveLength(2);
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
