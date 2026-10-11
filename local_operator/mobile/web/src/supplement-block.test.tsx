// @vitest-environment happy-dom
//
// The block's RENDERED half: every disposition, the files arms, the action
// pair and the ops' refusals. What happy-dom cannot prove is the sandbox and
// the frame wire (no document loads in a happy-dom iframe); those are pinned
// in `supplement-frame.test.tsx` and, for the real thing, by the browser rig
// recorded on the PR.
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HttpError } from "./api";
import { SupplementBlock } from "./components/supplement-block";
import { Transcript } from "./components/transcript";
import type { SupplementBeat, SupplementComponent, SupplementFile } from "./lib/supplement";
import type { TranscriptEntry } from "./types";

const api = vi.hoisted(() => ({
	getHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	getSubagentHistory: vi.fn(async () => ({ entries: [], has_more: false })),
	imageUrl: vi.fn((_pid: string, entryId: string, index: number) => `img:${entryId}:${index}`),
	sendCommand: vi.fn(async () => ({ ok: true, detail: "" })),
	getSupplementDocument: vi.fn(async () => ({ html: "<html>component</html>" })),
	getSupplementFile: vi.fn(async () => ({
		blob: new Blob(["hello\nworld"], { type: "text/plain" }),
		mime: "text/plain",
	})),
	supplementCancel: vi.fn(async () => ({ ok: true, detail: "cancelled" })),
	supplementSteer: vi.fn(async () => ({ ok: true, detail: "steering" })),
	supplementRestart: vi.fn(async () => ({ ok: true, detail: "restarting" })),
}));

vi.mock("./api", async (importOriginal) => {
	const original = await importOriginal<typeof import("./api")>();
	return { ...original, ...api };
});

/*
 * THE MOUNT GATE'S OBSERVER (U3). `FrameSlot` reads `IntersectionObserver`
 * before it may fetch a document, and happy-dom ships a stub whose `observe`
 * is a TODO — its callback never fires, so the gate would hold every document
 * forever and no rendered test could see a fetch. The suite installs the real
 * behaviour instead: an observation reports the element as in view. The gate
 * test overrides it with `ManualIntersectionObserver` and fires it by hand.
 */
class FiredIntersectionObserver implements IntersectionObserver {
	readonly root = null;
	readonly rootMargin = "0px";
	readonly thresholds: readonly number[] = [0];
	private readonly callback: IntersectionObserverCallback;

	constructor(callback: IntersectionObserverCallback) {
		this.callback = callback;
	}

	observe(target: Element): void {
		this.callback([{ isIntersecting: true, target } as IntersectionObserverEntry], this);
	}

	unobserve(): void {}

	disconnect(): void {}

	takeRecords(): IntersectionObserverEntry[] {
		return [];
	}
}

/** The hand-fired observer: nothing reports until `fire()`. */
class ManualIntersectionObserver implements IntersectionObserver {
	readonly root = null;
	readonly rootMargin = "0px";
	readonly thresholds: readonly number[] = [0];
	private readonly callback: IntersectionObserverCallback;
	private readonly targets: Element[] = [];

	constructor(callback: IntersectionObserverCallback) {
		this.callback = callback;
		manualObserver = this;
	}

	observe(target: Element): void {
		this.targets.push(target);
	}

	unobserve(): void {}

	disconnect(): void {}

	takeRecords(): IntersectionObserverEntry[] {
		return [];
	}

	fire(): void {
		this.callback(
			this.targets.map((target) => ({ isIntersecting: true, target }) as IntersectionObserverEntry),
			this,
		);
	}
}

let manualObserver: ManualIntersectionObserver | null = null;

beforeEach(() => {
	manualObserver = null;
	vi.stubGlobal("IntersectionObserver", FiredIntersectionObserver);
});

afterEach(() => {
	cleanup();
	vi.clearAllMocks();
	vi.unstubAllGlobals();
});

const ANCHOR = "b2c3d4e5f60718293a4b5c6d7e8f90a1";
const JOB = "7d41c0aa9e13";
const DIGEST = "a".repeat(32);

function beat(over: Partial<SupplementBeat> = {}): SupplementBeat {
	return {
		anchor: ANCHOR,
		job: JOB,
		version: 1,
		state: "done",
		stage: "",
		elapsedS: 0,
		files: [],
		filesMore: 0,
		more: [],
		components: [],
		error: "",
		dismissed: false,
		...over,
	};
}

function file(over: Partial<SupplementFile> = {}): SupplementFile {
	return {
		path: "reports/latency.md",
		name: "latency.md",
		kind: "markdown",
		sizeBytes: 4120,
		mtime: 0,
		why: "",
		...over,
	};
}

function component(over: Partial<SupplementComponent> = {}): SupplementComponent {
	return {
		attachment: DIGEST,
		title: "latency by hour",
		source: "data/bench.csv",
		mime: "text/html",
		heightHint: 200,
		...over,
	};
}

function entry(over: Partial<TranscriptEntry>): TranscriptEntry {
	return {
		id: "e1",
		kind: "assistant",
		text: "the answer",
		tool_call_id: "",
		tool_name: "",
		tool_state: "done",
		summary: "",
		intent: "",
		diff_added: 0,
		diff_removed: 0,
		elapsed_s: 0,
		error: "",
		details: {},
		final: true,
		...over,
	};
}

describe("the states", () => {
	it("preparing: the quiet line with the pair, and Cancel rides the session command", async () => {
		render(<SupplementBlock beat={beat({ state: "running" })} sessionId="s1" jobLive={true} />);
		expect(screen.getByText("Preparing highlights…")).toBeTruthy();
		const cancel = screen.getByText("Cancel");
		fireEvent.click(cancel);
		await waitFor(() => expect(api.supplementCancel).toHaveBeenCalledWith("s1", ANCHOR, JOB));
	});

	it("preparing + Adjust…: the inline field sends a steer and closes", async () => {
		render(<SupplementBlock beat={beat({ state: "queued" })} sessionId="s1" jobLive={true} />);
		fireEvent.click(screen.getByText("Adjust…"));
		const field = screen.getByPlaceholderText("Adjust the graphic…") as HTMLInputElement;
		fireEvent.change(field, { target: { value: "  simpler labels  " } });
		fireEvent.click(screen.getByText("Send"));
		// The wire forwards the words verbatim with surrounding space trimmed —
		// the runtime is the one place they are flattened, but a blank-only
		// instruction never leaves this surface.
		await waitFor(() =>
			expect(api.supplementSteer).toHaveBeenCalledWith("s1", ANCHOR, JOB, "simpler labels"),
		);
		expect(screen.queryByPlaceholderText("Adjust the graphic…")).toBeNull();
	});

	it("a refused press paints the daemon's own sentence and re-arms", async () => {
		api.supplementCancel.mockRejectedValueOnce(
			new HttpError(409, "session not connected"),
		);
		render(<SupplementBlock beat={beat({ state: "running" })} sessionId="s1" jobLive={true} />);
		fireEvent.click(screen.getByText("Cancel"));
		expect(await screen.findByText("Could not cancel: session not connected")).toBeTruthy();
		// The control is usable again (the latch was released in `finally`).
		await waitFor(() =>
			expect((screen.getByText("Cancel") as HTMLButtonElement).disabled).toBe(false),
		);
	});

	it("cancelled and failed settle into the neutral lines with Retry", async () => {
		const { unmount } = render(
			<SupplementBlock beat={beat({ state: "cancelled" })} sessionId="s1" jobLive={false} />,
		);
		expect(screen.getByText("Highlights cancelled")).toBeTruthy();
		fireEvent.click(screen.getByText("Retry"));
		await waitFor(() => expect(api.supplementRestart).toHaveBeenCalledWith("s1", ANCHOR, JOB));
		unmount();

		render(<SupplementBlock beat={beat({ state: "failed" })} sessionId="s1" jobLive={false} />);
		expect(screen.getByText("Couldn't prepare highlights")).toBeTruthy();
	});

	it("the stale-row rule: a cold non-terminal row is cancelled · Retry, never a forever spinner", () => {
		render(<SupplementBlock beat={beat({ state: "queued" })} sessionId="s1" jobLive={false} />);
		expect(screen.getByText("Highlights cancelled")).toBeTruthy();
		expect(screen.queryByText("Preparing highlights…")).toBeNull();
	});

	it("nothing: superseded, dismissed and done-empty render no block at all", () => {
		for (const b of [
			beat({ error: "superseded" }),
			beat({ dismissed: true }),
			beat(),
		]) {
			const { container, unmount } = render(
				<SupplementBlock beat={b} sessionId="s1" jobLive={true} />,
			);
			expect(container.querySelector('[data-supplement-block]')).toBeNull();
			unmount();
		}
	});
});

describe("the files half", () => {
	it("summarizes ≤ 2 files with no components as the contents themselves", () => {
		render(
			<SupplementBlock
				beat={beat({
					files: [file(), file({ path: "data/bench.csv", name: "bench.csv" })],
				})}
				sessionId="s1"
				jobLive={false}
			/>,
		);
		expect(screen.getByText("2 files: latency.md, bench.csv")).toBeTruthy();
	});

	it("renders rows and the N more disclosure, expanding the stored tail in place", () => {
		render(
			<SupplementBlock
				beat={beat({
					files: [
						file(),
						file({ path: "data/bench.csv", name: "bench.csv" }),
						file({ path: "out/plot.html", name: "plot.html" }),
					],
					filesMore: 2,
					more: ["archive/one.txt", "archive/two.txt"],
				})}
				sessionId="s1"
				jobLive={false}
			/>,
		);
		expect(screen.queryByText("4 more files")).toBeNull();
		expect(screen.getByText("2 more files")).toBeTruthy();
		fireEvent.click(screen.getByText("2 more files"));
		expect(screen.getByText("archive/one.txt")).toBeTruthy();
		expect(screen.getByText("archive/two.txt")).toBeTruthy();
		// The label TOGGLES with the state, the desktop's grammar (U2):
		// `2 more files` ⇄ `Show fewer`.
		expect(screen.getByText("Show fewer")).toBeTruthy();
		expect(screen.queryByText("2 more files")).toBeNull();
	});

	it("previews a text callout in place, by the row's own index", async () => {
		// THREE files: the ≤ 2-file one-liner (frozen §2.8) would otherwise
		// summarize instead of rendering rows, and the affordance lives on rows.
		render(
			<SupplementBlock
				beat={beat({
					files: [file(), file({ path: "data/b.csv", name: "b.csv" }), file({ path: "c.txt", name: "c.txt" })],
				})}
				sessionId="s1"
				jobLive={false}
			/>,
		);
		fireEvent.click(screen.getByText("latency.md"));
		expect(api.getSupplementFile).toHaveBeenCalledWith("s1", JOB, 0);
		// testing-library normalises whitespace before matching, so the newline
		// in the served text reads as a space here; the `<pre>` keeps the bytes.
		expect(await screen.findByText(/hello\s+world/)).toBeTruthy();
	});

	it("renders a refused preview as the refusal, never as content", async () => {
		api.getSupplementFile.mockRejectedValueOnce(new HttpError(404, "no such file"));
		render(
			<SupplementBlock
				beat={beat({
					files: [file(), file({ path: "data/b.csv", name: "b.csv" }), file({ path: "c.txt", name: "c.txt" })],
				})}
				sessionId="s1"
				jobLive={false}
			/>,
		);
		fireEvent.click(screen.getByText("latency.md"));
		expect(await screen.findByText("Could not preview: no such file")).toBeTruthy();
	});
});

describe("the components half", () => {
	it("mounts the head frame's document and fetches the tail only through the Show all disclosure", async () => {
		// The live iframe itself cannot be asserted in happy-dom (see the frame
		// test's header: its two `load` events tear a frame down), so this pins
		// the MOUNTING LADDER: the head slot fetches once the viewport gate
		// reports (immediately, in these tests), the tail's fetch happens only
		// once the disclosure is opened — the memo's own condition
		// for mounting beyond the budget (an explicit action).
		const second = component({ attachment: "c".repeat(32), title: "second" });
		render(
			<SupplementBlock
				beat={beat({ components: [component(), second] })}
				sessionId="s1"
				jobLive={false}
			/>,
		);
		expect(screen.getByText("Show all 1 graphic")).toBeTruthy();
		await waitFor(() => expect(api.getSupplementDocument).toHaveBeenCalledTimes(1));
		expect(api.getSupplementDocument).toHaveBeenLastCalledWith("s1", "a".repeat(32));
		fireEvent.click(screen.getByText("Show all 1 graphic"));
		await waitFor(() => expect(api.getSupplementDocument).toHaveBeenCalledTimes(2));
		expect(api.getSupplementDocument).toHaveBeenLastCalledWith("s1", "c".repeat(32));
		// The graphics disclosure toggles its words the same way (U2).
		expect(screen.getByText("Show fewer")).toBeTruthy();
		expect(screen.queryByText("Show all 1 graphic")).toBeNull();
	});

	it("waits for the viewport gate before fetching a document (§2.11; U3)", async () => {
		// Manual gate: nothing may fetch until the observer reports the slot
		// approaching the viewport — the memo's "mounts only when its answer
		// row is in the viewport", the desktop slot's own rule.
		vi.stubGlobal("IntersectionObserver", ManualIntersectionObserver);
		render(
			<SupplementBlock
				beat={beat({ components: [component()] })}
				sessionId="s1"
				jobLive={false}
			/>,
		);
		// The slot (and its reserved box + caption) is there; the fetch is not.
		expect(screen.getByText("Source: data/bench.csv")).toBeTruthy();
		expect(manualObserver).not.toBeNull();
		expect(api.getSupplementDocument).not.toHaveBeenCalled();
		act(() => manualObserver?.fire());
		await waitFor(() => expect(api.getSupplementDocument).toHaveBeenCalledTimes(1));
		expect(api.getSupplementDocument).toHaveBeenLastCalledWith("s1", "a".repeat(32));
	});

	it("a failed document fetch paints the frame-level fallback, not a hole", async () => {
		api.getSupplementDocument.mockRejectedValueOnce(new HttpError(404, "no such supplement"));
		render(
			<SupplementBlock
				beat={beat({ components: [component()] })}
				sessionId="s1"
				jobLive={false}
			/>,
		);
		expect(await screen.findByText("Couldn't render this graphic")).toBeTruthy();
	});
});

describe("the transcript seat", () => {
	it("renders the block under the answer whose id is the beat's anchor, outside the completion anchor", () => {
		const { container } = render(
			<Transcript
				pid="s1"
				entries={[
					entry({ id: "user-1", kind: "user", text: "hi", final: false }),
					entry({ id: ANCHOR, text: "the answer" }),
				]}
				supplements={{ [ANCHOR]: beat({ files: [file()] }) }}
				projectionPid={4321}
			/>,
		);
		const block = container.querySelector("[data-supplement-block]");
		expect(block).toBeTruthy();
		const anchorDiv = block?.previousElementSibling;
		expect(anchorDiv?.getAttribute("data-completion-anchor")).toBe(ANCHOR);
	});

	it("does not attach to a non-author row or a missing anchor", () => {
		const { container } = render(
			<Transcript
				pid="s1"
				entries={[entry({ id: ANCHOR, kind: "user", text: "not the answer" })]}
				supplements={{ [ANCHOR]: beat({ files: [file()] }) }}
				projectionPid={4321}
			/>,
		);
		expect(container.querySelector("[data-supplement-block]")).toBeNull();
	});
});
