// @vitest-environment happy-dom
//
// The composer's voice mic against a stubbed recorder + transcription endpoint:
// visibility from capabilities, record → stop → APPEND (no clobber), cancel
// (no request), the permission-denied copy, the failure-copy split, Enter and
// Send inert while a take is live, and the send envelope carrying the
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
		getSkills: vi.fn(async () => ({ skills: [] })),
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
			ended={false}
			pid="p1"
			projection={projection()}
			onOpenModels={() => {}}
			onReturnToModels={() => {}}
			onOpenEffort={() => {}}
			effortOpen={false}
			onCloseEffort={() => {}}
		/>,
	);
}

function field(): HTMLTextAreaElement {
	return screen.getByPlaceholderText("Ask anything…") as HTMLTextAreaElement;
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

	it("offers Retry for an upstream/transient failure instead of a bare try-again line", async () => {
		mockedTranscribe.mockRejectedValueOnce(
			new (await import("./api")).HttpError(502, "Transcription failed upstream."),
		);
		renderComposer();
		const recorder = await startRecording();
		recorder.emitChunk("audio-bytes");
		fireEvent.click(screen.getByRole("button", { name: "stop and transcribe" }));

		await waitFor(() => expect(screen.getByText("Transcription failed")).toBeTruthy());
		expect(screen.getByRole("button", { name: "retry transcription" })).toBeTruthy();
		expect(screen.getByRole("button", { name: "discard recording" })).toBeTruthy();
		/* The server's upstream prose belongs in the daemon log, not on the phone. */
		expect(screen.queryByText("Transcription failed upstream.")).toBeNull();
	});
});

describe("interaction with send", () => {
	/* Before the Send recording slot, the disc at the Send spot (and Enter) sent the
	   typed draft and THREW THE TAKE AWAY. The disc now sends the take
	   (composer.send-recording.test.tsx); Enter during a take is the desktop's
	   "Insert transcript": stop, transcribe, put the words in the box, send nothing. */
	it("Enter while recording stops the take and inserts the transcript without sending", async () => {
		renderComposer();
		fireEvent.change(field(), { target: { value: "typed text" } });
		mockedTranscribe.mockResolvedValueOnce({ text: "spoken", provider: "radient", model: null, path: "provider_stt_radient" });
		const recorder = await startRecording();
		recorder.emitChunk("audio-bytes");

		fireEvent.keyDown(field(), { key: "Enter" });

		await waitFor(() => expect(field().value).toBe("typed text spoken"));
		expect(recorder.state).toBe("inactive");
		expect(mockedTranscribe).toHaveBeenCalledOnce();
		expect(mockedSendCommand).not.toHaveBeenCalled();
	});

	it("Enter with focus outside the box (window level) also inserts the transcript", async () => {
		renderComposer();
		mockedTranscribe.mockResolvedValueOnce({ text: "spoken", provider: "radient", model: null, path: "provider_stt_radient" });
		const recorder = await startRecording();
		recorder.emitChunk("audio-bytes");

		fireEvent.keyDown(document.body, { key: "Enter" });

		await waitFor(() => expect(field().value).toBe("spoken"));
		expect(recorder.state).toBe("inactive");
		expect(mockedSendCommand).not.toHaveBeenCalled();
	});

	it("Enter on a focused button keeps that button's own action (Enter on cancel is not Insert)", async () => {
		renderComposer();
		const recorder = await startRecording();
		recorder.emitChunk("audio-bytes");

		fireEvent.keyDown(screen.getByRole("button", { name: "cancel recording" }), { key: "Enter" });
		await new Promise((r) => setTimeout(r, 0));

		expect(mockedTranscribe).not.toHaveBeenCalled();
		expect(recorder.state).toBe("recording");
	});

	it("Shift+Enter during a take is not claimed (it stays the box's newline key)", async () => {
		renderComposer();
		const recorder = await startRecording();
		recorder.emitChunk("audio-bytes");

		fireEvent.keyDown(field(), { key: "Enter", shiftKey: true });
		await new Promise((r) => setTimeout(r, 0));

		expect(mockedTranscribe).not.toHaveBeenCalled();
		expect(recorder.state).toBe("recording");
	});

	it("Enter when not recording is not claimed: it still sends the typed draft", async () => {
		renderComposer();
		fireEvent.change(field(), { target: { value: "typed text" } });
		fireEvent.keyDown(field(), { key: "Enter" });
		await waitFor(() => expect(mockedSendCommand).toHaveBeenCalledOnce());
		expect(mockedTranscribe).not.toHaveBeenCalled();
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

	it("Enter while transcribing sends nothing: the old draft cannot overtake the words in flight", async () => {
		renderComposer();
		fireEvent.change(field(), { target: { value: "typed text" } });
		let held:
			| ((v: { text: string; provider: string; model: string | null; path: string }) => void)
			| undefined;
		mockedTranscribe.mockImplementationOnce(
			() => new Promise((resolve) => { held = resolve; }),
		);
		const recorder = await startRecording();
		recorder.emitChunk("audio-bytes");
		fireEvent.click(screen.getByRole("button", { name: "stop and transcribe" }));
		await screen.findByText("Transcribing…");

		fireEvent.keyDown(field(), { key: "Enter" });
		fireEvent.keyDown(document.body, { key: "Enter" });
		/* The disc is inert too: a press would abort the request and send the old draft. */
		expect((screen.getByRole("button", { name: "send" }) as HTMLButtonElement).disabled).toBe(true);
		expect(mockedSendCommand).not.toHaveBeenCalled();

		held!({ text: "more", provider: "radient", model: null, path: "provider_stt_radient" });
		await waitFor(() => expect(field().value).toBe("typed text more"));
		expect(mockedSendCommand).not.toHaveBeenCalled();
		/* Enter did not stop or discard the take: the one request was the tap's. */
		expect(mockedTranscribe).toHaveBeenCalledOnce();
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
		const signal = (mockedTranscribe.mock.calls[0]![1] as { signal: AbortSignal }).signal;
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

/** One recording whose transcription fails with `error`. */
async function failTake(error: unknown, chunk = "audio-bytes"): Promise<void> {
	mockedTranscribe.mockRejectedValueOnce(error);
	const recorder = await startRecording();
	recorder.emitChunk(chunk);
	fireEvent.click(screen.getByRole("button", { name: "stop and transcribe" }));
	await screen.findByRole("alert");
}

const retryControl = () => screen.getByRole("button", { name: "retry transcription" });
const discardControl = () => screen.getByRole("button", { name: "discard recording" });

/* The failed take is KEPT in place (design: a refused recording is recoverable)
   rather than dropped with the alert row's dismiss. These pin the four things
   that must stay true of that state: the same bytes go back, a retry never
   sends, the take outlives one failed retry, and every release path frees it. */
describe("a failed take: Retry, Discard, and what cannot be retried", () => {
	it("resends THE SAME blob through the success path on Retry, and never sends", async () => {
		renderComposer();
		await failTake(new api.HttpError(502, "Transcription failed upstream."));
		expect(screen.getByText("Transcription failed")).toBeTruthy();

		mockedTranscribe.mockResolvedValueOnce({
			text: "hello",
			provider: "radient",
			model: null,
			path: "provider_stt_radient",
		});
		fireEvent.click(retryControl());

		await waitFor(() => expect(field().value).toBe("hello"));
		expect(mockedTranscribe).toHaveBeenCalledTimes(2);
		const first = mockedTranscribe.mock.calls[0]!;
		const second = mockedTranscribe.mock.calls[1]!;
		/* The take itself, not a re-recording; a FRESH controller for the retry. */
		expect(second[0]).toBe(first[0]);
		expect((second[1] as { signal: AbortSignal }).signal).not.toBe(
			(first[1] as { signal: AbortSignal }).signal,
		);
		/* The success path ran — the completion line, no auto-send. */
		expect(screen.getByText("Transcript added")).toBeTruthy();
		expect(mockedSendCommand).not.toHaveBeenCalled();
		expect(screen.queryByText("Transcription failed")).toBeNull();
	});

	it("keeps the take and still offers Retry when the retry fails too", async () => {
		renderComposer();
		await failTake(new api.HttpError(502, "Transcription failed upstream."));
		mockedTranscribe.mockRejectedValueOnce(new api.HttpError(502, "Transcription failed upstream."));
		fireEvent.click(retryControl());

		await waitFor(() => expect(mockedTranscribe).toHaveBeenCalledTimes(2));
		await waitFor(() => expect(screen.getByText("Transcription failed")).toBeTruthy());
		expect(retryControl()).toBeTruthy();
	});

	it("offers Retry for a tunnel 5xx too: every 5xx is transient, as on the desktop", async () => {
		renderComposer();
		/* A tunnel's own 504 is the class the old [500, 502] pair dropped: it is
		   an upstream wait, not an answer to the request, so the same bytes are
		   worth sending again. */
		await failTake(new api.HttpError(504, "gateway timeout"));

		expect(screen.getByText("Transcription failed")).toBeTruthy();
		expect(retryControl()).toBeTruthy();
	});

	it("an unknown non-5xx status keeps Discard alone and does not promise 'Try again'", async () => {
		renderComposer();
		await failTake(new api.HttpError(418, "I'm a teapot"));

		expect(screen.getByText("Transcription failed")).toBeTruthy();
		/* The row draws no Retry here, so its copy must not offer one. */
		expect(screen.queryByText(/Try again/)).toBeNull();
		expect(screen.queryByRole("button", { name: "retry transcription" })).toBeNull();
		expect(discardControl()).toBeTruthy();
	});

	it("hands focus to the status row and the row STAYS focusable while it is rendered", async () => {
		renderComposer();
		await failTake(new api.HttpError(502, "Transcription failed upstream."));
		const row = screen.getByRole("alert").parentElement as HTMLElement;
		/* Script-focusable only, and already so while the failure is shown. */
		expect(row.getAttribute("tabindex")).toBe("-1");
		const retry = retryControl();
		retry.focus();
		/* Registered BEFORE the click: the Retry press calls the endpoint
		   synchronously, so a later `mockResolvedValueOnce` would arrive after the
		   retry already took the default (empty) answer. */
		mockedTranscribe.mockResolvedValueOnce({
			text: "hello",
			provider: "radient",
			model: null,
			path: "provider_stt_radient",
		});
		fireEvent.click(retry);

		await waitFor(() => expect(screen.getByText("Transcribing…")).toBeTruthy());
		/* A real browser drops focus to <body> when the focused button unmounts;
		   that strands a hardware keyboard's place in the form. */
		expect(document.activeElement).not.toBe(document.body);
		expect(document.activeElement).toBe(row);

		/* THE HANDOFF TARGET MUST OUTLIVE THE FAILURE STATE. Retry clears
		   `dictationFailure`, and Chrome resets activeElement to <body> the instant
		   the focused element loses its tabIndex (Gecko keeps it, so this is
		   invisible outside a real Chrome). happy-dom never moves focus for a
		   removed tabIndex, so the attribute IS the check that discriminates:
		   conditional tabIndex fails the two assertions below on the Retry render. */
		expect(row.getAttribute("tabindex")).toBe("-1");

		await waitFor(() => expect(field().value).toBe("hello"));
		/* Past the transcribing state, the row is still the same element and still
		   holds focus: the outcome notice keeps it mounted. */
		expect(screen.getByText("Transcript added")).toBeTruthy();
		expect(document.activeElement).toBe(row);
		expect(row.getAttribute("tabindex")).toBe("-1");
	});

	it("rings both actions for a keyboard and keeps them 44 px apart", async () => {
		renderComposer();
		await failTake(new api.HttpError(502, "Transcription failed upstream."));

		for (const control of [retryControl(), discardControl()]) {
			expect(control.className).toContain("focus-visible:outline-2");
			expect(control.className).toContain("min-h-11");
		}
		/* D1: Retry is a quiet bordered pill, Discard bare, and the pair reads as
		   two targets rather than one run of text. */
		expect(retryControl().className).toContain("rounded-full");
		expect(retryControl().parentElement?.className).toContain("gap-4");
	});

	it("Discard frees the take: no request, no row, nothing left to retry", async () => {
		renderComposer();
		await failTake(new api.HttpError(502, "Transcription failed upstream."));
		fireEvent.click(discardControl());

		expect(screen.queryByText("Transcription failed")).toBeNull();
		expect(screen.queryByRole("button", { name: "retry transcription" })).toBeNull();
		expect(mockedTranscribe).toHaveBeenCalledOnce();
	});

	it("a 402 keeps the daemon's own sentence and offers Discard alone", async () => {
		renderComposer();
		const sentence =
			"Transcription is unavailable: your Radient credit balance is too low. Add credits to continue.";
		await failTake(new api.HttpError(402, sentence));

		expect(screen.getByText(sentence)).toBeTruthy();
		expect(screen.queryByRole("button", { name: "retry transcription" })).toBeNull();
		expect(discardControl()).toBeTruthy();
	});

	it("a 503 (no executable path) offers Discard alone: a retry answers the same way", async () => {
		renderComposer();
		await failTake(new api.HttpError(503, "Voice input isn't available on this machine."));

		expect(screen.getByText("Voice input isn't available on this machine.")).toBeTruthy();
		expect(screen.queryByRole("button", { name: "retry transcription" })).toBeNull();
	});

	it("a 413 explains the tunnel's cap and refuses to retry", async () => {
		renderComposer();
		await failTake(new api.HttpError(413, "request exceeds 10 MiB"));

		expect(
			screen.getByText("That recording is too long to send. Discard it and record a shorter one."),
		).toBeTruthy();
		expect(screen.queryByRole("button", { name: "retry transcription" })).toBeNull();
		expect(discardControl()).toBeTruthy();
	});

	it("a retained take over the tunnel's 10 MiB cap is not retried, whatever the refusal says", async () => {
		renderComposer();
		/* The daemon's own cap is 20 MiB, so this take passed the client's checks and
		   the gateway's 413 is the only bound that can stop it — retrying would ask
		   the user to wait for the same answer. */
		await failTake(new api.HttpError(502, "Transcription failed upstream."), "x".repeat(10 * 1024 * 1024 + 1));

		expect(
			screen.getByText("That recording is too long to send. Discard it and record a shorter one."),
		).toBeTruthy();
		expect(screen.queryByRole("button", { name: "retry transcription" })).toBeNull();
	});

	it("a new take releases the failed one", async () => {
		renderComposer();
		await failTake(new api.HttpError(502, "Transcription failed upstream."));

		fireEvent.click(screen.getByRole("button", { name: "start voice input" }));
		await waitFor(() => expect(screen.getByRole("button", { name: "stop and transcribe" })).toBeTruthy());

		expect(screen.queryByText("Transcription failed")).toBeNull();
		expect(screen.queryByRole("button", { name: "retry transcription" })).toBeNull();
	});

	it("a route change (another conversation) releases the failed take", async () => {
		const { rerender } = renderComposer();
		await failTake(new api.HttpError(502, "Transcription failed upstream."));

		rerender(
			<Composer
				ended={false}
				pid="p2"
				projection={projection()}
				onOpenModels={() => {}}
				onReturnToModels={() => {}}
				onOpenEffort={() => {}}
				effortOpen={false}
				onCloseEffort={() => {}}
			/>,
		);

		expect(screen.queryByText("Transcription failed")).toBeNull();
		expect(screen.queryByRole("button", { name: "retry transcription" })).toBeNull();
	});

	it("announces the failure as an alert and keeps the controls path-sized", async () => {
		renderComposer();
		await failTake(new api.HttpError(502, "Transcription failed upstream."));

		const alert = screen.getByRole("alert");
		expect(alert.textContent).toContain("Transcription failed");
		expect(retryControl().className).toContain("min-h-11");
		expect(discardControl().className).toContain("min-h-11");
	});
});
