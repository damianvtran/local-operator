// @vitest-environment happy-dom
//
// The composer's voice mic against a stubbed recorder + transcription endpoint:
// visibility from capabilities, record → stop → APPEND (no clobber), cancel
// (no request), the permission-denied copy, the failure-copy split, send
// cancelling an in-flight dictation, and the send envelope carrying the
// provenance. happy-dom has no MediaRecorder or getUserMedia, so both are
// stubbed here — which is exactly the seam the design says cannot be
// unit-proven end to end (QA owns the real-engine cell).
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "./api";
import { Composer } from "./components/composer";
import type { Capabilities, SessionProjection } from "./types";

vi.mock("./api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./api")>();
	return {
		...actual,
		getCommands: vi.fn(async () => ({ commands: [] })),
		getModels: vi.fn(async () => ({ models: [] })),
		sendCommand: vi.fn(async (_sessionId: string, _op: Record<string, unknown>) => ({
			ok: true,
			detail: "",
		})),
		transcribeAudio: vi.fn(async (_blob: Blob) => ({
			text: "",
			provider: "",
			model: null as string | null,
			path: "",
		})),
	};
});

const mockedSendCommand = vi.mocked(api.sendCommand);
const mockedTranscribe = vi.mocked(api.transcribeAudio);

let capabilitySlot: Capabilities | null = null;
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useDraft: () => useState(""),
		useCapabilities: () => capabilitySlot,
	};
});

class FakeRecorder {
	static instances: FakeRecorder[] = [];
	static supported = ["audio/webm;codecs=opus"];

	state: "inactive" | "recording" = "inactive";
	mimeType: string;
	ondataavailable: ((event: { data: Blob }) => void) | null = null;
	onstop: (() => void) | null = null;

	constructor(
		readonly stream: unknown,
		options?: { mimeType?: string },
	) {
		this.mimeType = options?.mimeType ?? "";
		FakeRecorder.instances.push(this);
	}

	static isTypeSupported(mime: string): boolean {
		return FakeRecorder.supported.includes(mime);
	}

	start(): void {
		this.state = "recording";
	}

	stop(): void {
		this.state = "inactive";
		this.onstop?.();
	}

	/** Simulate the timeslice's buffered data arriving. */
	emitChunk(text: string): void {
		this.ondataavailable?.({ data: new Blob([text], { type: this.mimeType }) });
	}
}

let trackStop: ReturnType<typeof vi.fn>;
const getUserMedia = vi.fn();

function projection(): SessionProjection {
	return {
		session_id: "s1",
		pid: 1,
		kind: "tui",
		conversation_name: "Voice",
		cwd: "",
		model_label: "",
		model_selector: "",
		effort: "",
		effort_ladder: [],
		streaming: false,
		activity: "",
		activity_started_s: 0,
		stop_reason: "completed",
		queued_count: 0,
		ended: false,
		degraded: false,
		transcript: [],
		todos: [],
		subagents: [],
		pending: null,
		pending_count: 0,
		usage: {},
		version: 1,
	} as unknown as SessionProjection;
}

function renderComposer() {
	return render(
		<Composer
			pid="p1"
			projection={projection()}
			onOpenModels={() => {}}
			onOpenEffort={() => {}}
			effortOpen={false}
			onCloseEffort={() => {}}
		/>,
	);
}

function field(): HTMLTextAreaElement {
	return screen.getByPlaceholderText("Message…") as HTMLTextAreaElement;
}

function availableCapability(): Capabilities {
	return { stt: { available: true, path: "provider_stt_radient", reason: "" } };
}

/** Click the mic and wait for the recording state; returns the live recorder. */
async function startRecording(): Promise<FakeRecorder> {
	fireEvent.click(screen.getByRole("button", { name: "start voice input" }));
	await waitFor(() => expect(screen.getByRole("button", { name: "stop and transcribe" })).toBeTruthy());
	return FakeRecorder.instances.at(-1)!;
}

async function dictate(
	transcript: { text: string; path: string },
	chunk = "audio-bytes",
): Promise<void> {
	mockedTranscribe.mockResolvedValueOnce({
		text: transcript.text,
		provider: "radient",
		model: null,
		path: transcript.path,
	});
	const recorder = await startRecording();
	recorder.emitChunk(chunk);
	fireEvent.click(screen.getByRole("button", { name: "stop and transcribe" }));
}

beforeEach(() => {
	FakeRecorder.instances = [];
	capabilitySlot = availableCapability();
	trackStop = vi.fn();
	getUserMedia.mockResolvedValue({ getTracks: () => [{ stop: trackStop }] });
	vi.stubGlobal("MediaRecorder", FakeRecorder);
	vi.stubGlobal("isSecureContext", true);
	Object.defineProperty(navigator, "mediaDevices", {
		configurable: true,
		value: { getUserMedia },
	});
});

afterEach(() => {
	cleanup();
	localStorage.clear();
	vi.clearAllMocks();
	vi.unstubAllGlobals();
	Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: undefined });
});

describe("mic visibility", () => {
	it("is hidden when the daemon sends no capabilities (an older daemon)", () => {
		capabilitySlot = null;
		renderComposer();
		expect(screen.queryByRole("button", { name: "start voice input" })).toBeNull();
	});

	it("is hidden when the daemon says no path is available", () => {
		capabilitySlot = { stt: { available: false, path: null, reason: "no key" } };
		renderComposer();
		expect(screen.queryByRole("button", { name: "start voice input" })).toBeNull();
	});

	it("is hidden in an insecure context", () => {
		vi.stubGlobal("isSecureContext", false);
		renderComposer();
		expect(screen.queryByRole("button", { name: "start voice input" })).toBeNull();
	});

	it("is shown when a path is available and the browser can record", () => {
		renderComposer();
		expect(screen.getByRole("button", { name: "start voice input" })).toBeTruthy();
	});
});

describe("record → transcribe → append", () => {
	it("appends a dictation into an empty draft and sends it as dictated", async () => {
		renderComposer();
		await dictate({ text: "hello world", path: "provider_stt_radient" });

		await waitFor(() => expect(field().value).toBe("hello world"));
		const blob = mockedTranscribe.mock.calls[0]![0] as Blob;
		expect(blob.type).toBe("audio/webm;codecs=opus");

		fireEvent.click(screen.getByRole("button", { name: "send" }));
		await waitFor(() => expect(mockedSendCommand).toHaveBeenCalledOnce());
		const envelope = mockedSendCommand.mock.calls[0]![1] as Record<string, unknown>;
		expect(envelope.input_mode).toBe("dictated");
		expect(envelope.input_path).toBe("provider_stt_radient");
	});

	it("appends into a TYPED draft and sends it as mixed", async () => {
		renderComposer();
		fireEvent.change(field(), { target: { value: "note:" } });
		await dictate({ text: "spoken words", path: "provider_stt_radient" });

		await waitFor(() => expect(field().value).toBe("note: spoken words"));

		fireEvent.click(screen.getByRole("button", { name: "send" }));
		await waitFor(() => expect(mockedSendCommand).toHaveBeenCalledOnce());
		const envelope = mockedSendCommand.mock.calls[0]![1] as Record<string, unknown>;
		expect(envelope.input_mode).toBe("mixed");
		expect(envelope.input_path).toBe("provider_stt_radient");
	});

	it("a reopened window after clearing the draft annotates as typed", async () => {
		renderComposer();
		await dictate({ text: "hello", path: "provider_stt_radient" });
		await waitFor(() => expect(field().value).toBe("hello"));

		// The user clears the box: the provenance window ends with the draft.
		fireEvent.change(field(), { target: { value: "" } });
		fireEvent.change(field(), { target: { value: "fresh typing" } });

		fireEvent.click(screen.getByRole("button", { name: "send" }));
		await waitFor(() => expect(mockedSendCommand).toHaveBeenCalledOnce());
		const envelope = mockedSendCommand.mock.calls[0]![1] as Record<string, unknown>;
		expect(envelope.input_mode).toBe("typed");
		expect(envelope.input_path).toBeUndefined();
	});

	it("reveals the appended span without focusing the field (U1)", async () => {
		renderComposer();
		fireEvent.change(field(), { target: { value: "line one\nline two" } });
		const ta = field();
		// happy-dom computes no layout, so give the field the geometry a long draft
		// has in the browser: the append must scroll the appended span into view.
		Object.defineProperty(ta, "scrollHeight", { configurable: true, value: 480 });
		const focusSpy = vi.spyOn(ta, "focus");

		await dictate({ text: "tail words", path: "provider_stt_radient" });

		await waitFor(() => expect(ta.scrollTop).toBe(480));
		expect(focusSpy).not.toHaveBeenCalled();
		expect(document.activeElement).not.toBe(ta);
	});

	it("announces the append politely without stealing focus (U3)", async () => {
		renderComposer();
		await dictate({ text: "hello world", path: "provider_stt_radient" });

		await waitFor(() => expect(field().value).toBe("hello world"));
		expect(screen.getByText("Transcript added")).toBeTruthy();
		expect(document.activeElement).not.toBe(field());
	});
});

describe("cancel, refusal and failure", () => {
	it("cancel releases the tracks, discards the result and makes no request", async () => {
		renderComposer();
		const recorder = await startRecording();
		recorder.emitChunk("audio-bytes");

		fireEvent.click(screen.getByRole("button", { name: "cancel recording" }));

		await waitFor(() => expect(screen.getByRole("button", { name: "start voice input" })).toBeTruthy());
		expect(trackStop).toHaveBeenCalled();
		expect(mockedTranscribe).not.toHaveBeenCalled();
		expect(field().value).toBe("");
	});

	it("names the browser-settings fix when the microphone is blocked", async () => {
		getUserMedia.mockRejectedValueOnce(new DOMException("denied", "NotAllowedError"));
		renderComposer();

		fireEvent.click(screen.getByRole("button", { name: "start voice input" }));

		await waitFor(() =>
			expect(
				screen.getByText(
					"Microphone access is blocked for this site. Allow it in your browser settings and try again.",
				),
			).toBeTruthy(),
		);
		expect(mockedTranscribe).not.toHaveBeenCalled();
	});

	it("shows the daemon's own sentence for an actionable refusal", async () => {
		mockedTranscribe.mockRejectedValueOnce(
			new (await import("./api")).HttpError(422, "Unsupported audio format: audio/x-garbage."),
		);
		renderComposer();
		const recorder = await startRecording();
		recorder.emitChunk("audio-bytes");
		fireEvent.click(screen.getByRole("button", { name: "stop and transcribe" }));

		await waitFor(() =>
			expect(screen.getByText("Unsupported audio format: audio/x-garbage.")).toBeTruthy(),
		);
		// The mic is usable again — a failed attempt is not a disabled control.
		expect(screen.getByRole("button", { name: "start voice input" })).toBeTruthy();
	});

	it("shows the retry sentence for an upstream/transient failure", async () => {
		mockedTranscribe.mockRejectedValueOnce(
			new (await import("./api")).HttpError(502, "Transcription failed upstream."),
		);
		renderComposer();
		const recorder = await startRecording();
		recorder.emitChunk("audio-bytes");
		fireEvent.click(screen.getByRole("button", { name: "stop and transcribe" }));

		await waitFor(() => expect(screen.getByText("Couldn't transcribe that. Try again.")).toBeTruthy());
	});
});

describe("interaction with send", () => {
	it("send cancels an in-flight dictation: no request, typed envelope", async () => {
		renderComposer();
		fireEvent.change(field(), { target: { value: "typed text" } });
		const recorder = await startRecording();
		recorder.emitChunk("audio-bytes");

		fireEvent.click(screen.getByRole("button", { name: "send" }));

		await waitFor(() => expect(mockedSendCommand).toHaveBeenCalledOnce());
		expect(mockedTranscribe).not.toHaveBeenCalled();
		expect(trackStop).toHaveBeenCalled();
		const envelope = mockedSendCommand.mock.calls[0]![1] as Record<string, unknown>;
		expect(envelope.text).toBe("typed text");
		expect(envelope.input_mode).toBe("typed");
		expect(envelope.input_path).toBeUndefined();
	});

	it("the 120 s cap stops the recorder rather than discarding", async () => {
		vi.useFakeTimers();
		try {
			renderComposer();
			mockedTranscribe.mockResolvedValueOnce({
				text: "capped",
				provider: "radient",
				model: null,
				path: "provider_stt_radient",
			});
			fireEvent.click(screen.getByRole("button", { name: "start voice input" }));
			// Flush the getUserMedia microtask chain without running the clock.
			await vi.advanceTimersByTimeAsync(0);
			const recorder = FakeRecorder.instances.at(-1)!;
			expect(recorder.state).toBe("recording");
			recorder.emitChunk("audio-bytes");

			// The cap fires stopDictation — NOT a discard: the recorder stops and
			// the clip is transcribed.
			await vi.advanceTimersByTimeAsync(120_000);
			await vi.advanceTimersByTimeAsync(0);

			expect(recorder.state).toBe("inactive");
			expect(mockedTranscribe).toHaveBeenCalledOnce();
			expect(field().value).toBe("capped");
		} finally {
			vi.useRealTimers();
		}
	});
});

describe("empty transcript, discards and cancels", () => {
	it("answers an empty transcript with one non-alarming sentence (D2)", async () => {
		renderComposer();
		await dictate({ text: "", path: "provider_stt_radient" });

		await waitFor(() =>
			expect(screen.getByText("Didn't catch that — try again.")).toBeTruthy(),
		);
		expect(field().value).toBe("");
	});

	it("says so when a send discards an in-flight dictation (U2)", async () => {
		renderComposer();
		fireEvent.change(field(), { target: { value: "typed text" } });
		const recorder = await startRecording();
		recorder.emitChunk("audio-bytes");

		fireEvent.click(screen.getByRole("button", { name: "send" }));

		await waitFor(() => expect(mockedSendCommand).toHaveBeenCalledOnce());
		expect(screen.getByText("Voice input discarded.")).toBeTruthy();
		expect(mockedTranscribe).not.toHaveBeenCalled();
	});

	it("cancels an in-flight transcription from the status row (U5)", async () => {
		renderComposer();
		let held:
			| ((v: { text: string; provider: string; model: string | null; path: string }) => void)
			| undefined;
		mockedTranscribe.mockImplementationOnce(
			() =>
				new Promise((resolve) => {
					held = resolve;
				}),
		);
		const recorder = await startRecording();
		recorder.emitChunk("audio-bytes");
		fireEvent.click(screen.getByRole("button", { name: "stop and transcribe" }));

		const cancel = await screen.findByRole("button", { name: "cancel transcription" });
		expect(screen.getByText("Transcribing…")).toBeTruthy();
		const signal = mockedTranscribe.mock.calls[0]![1] as AbortSignal;
		expect(signal.aborted).toBe(false);

		fireEvent.click(cancel);

		expect(signal.aborted).toBe(true);
		await waitFor(() =>
			expect(screen.getByRole("button", { name: "start voice input" })).toBeTruthy(),
		);

		// The late answer is dropped: the abort was a discard, not a pause.
		held!({ text: "too late", provider: "radient", model: null, path: "provider_stt_radient" });
		await new Promise((r) => setTimeout(r, 0));
		expect(field().value).toBe("");
	});
});
