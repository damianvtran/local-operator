// @vitest-environment happy-dom
//
// "SEND RECORDING" on the phone, driven through the real Composer with a stubbed
// recorder and transcription endpoint: while the mic records, slot B is a Send
// recording disc (never Stop, never a Send that throws the take away); one press
// stops, transcribes and submits the words; and every way the words must NOT go
// out (empty, edited meanwhile, slash, cancel, failure, a pending retry) leaves
// them in the box and slot B back to its ordinary role.
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "./api";
import { Composer } from "./components/composer";
import { resetTurnStopForTests } from "./lib/turn-stop";
import type { Capabilities, SessionProjection } from "./types";

vi.mock("./api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./api")>();
	return {
		...actual,
		getCommands: vi.fn(async () => ({ commands: [] })),
		getSkills: vi.fn(async () => ({ skills: [] })),
		getModels: vi.fn(async () => ({ models: [] })),
		sendCommand: vi.fn(async (_sessionId: string, _op: Record<string, unknown>) => ({ ok: true, detail: "" })),
		transcribeAudio: vi.fn(async (_blob: Blob, _signal?: AbortSignal) => ({ text: "", provider: "", model: null as string | null, path: "" })),
	};
});

const mockedSendCommand = vi.mocked(api.sendCommand);
const mockedTranscribe = vi.mocked(api.transcribeAudio);

let capabilitySlot: Capabilities | null = null;
vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return { ...actual, useDraft: () => useState(""), useCapabilities: () => capabilitySlot };
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

function projection(streaming: boolean): SessionProjection {
	return {
		session_id: "s1", pid: 1, kind: "tui", conversation_name: "Voice", cwd: "", model_label: "",
		model_selector: "", effort: "", effort_ladder: [], streaming, activity: "", activity_started_s: 0,
		stop_reason: "completed", queued_count: 0, ended: false, degraded: false, transcript: [],
		todos: [], subagents: [], pending: null, pending_count: 0, usage: {}, version: 1,
	} as unknown as SessionProjection;
}

function renderComposer(streaming = false) {
	return render(
		<Composer
			ended={false} pid="p1" projection={projection(streaming)} onOpenModels={() => {}}
			onReturnToModels={() => {}} onOpenEffort={() => {}} effortOpen={false} onCloseEffort={() => {}}
		/>,
	);
}

const field = () => screen.getByPlaceholderText("Ask anything…") as HTMLTextAreaElement;
const roles = () => [...screen.getByTestId("composer-actions").querySelectorAll("[data-role]")].map((el) => el.getAttribute("data-role"));
const sendRecordingDisc = () => screen.getByRole("button", { name: "send recording" }) as HTMLButtonElement;

async function startRecording(): Promise<FakeRecorder> {
	fireEvent.click(screen.getByRole("button", { name: "start voice input" }));
	await waitFor(() => expect(screen.getByRole("button", { name: "stop and transcribe" })).toBeTruthy());
	const recorder = FakeRecorder.instances.at(-1)!;
	recorder.emitChunk("audio-bytes");
	return recorder;
}

type Transcript = { text: string; provider: string; model: string | null; path: string };
const transcript = (text: string): Transcript => ({ text, provider: "radient", model: null, path: "provider_stt_radient" });

/** A transcription that stays in flight until the test settles it. */
function holdTranscription() {
	let resolve!: (v: Transcript) => void;
	let reject!: (e: unknown) => void;
	mockedTranscribe.mockImplementationOnce(() => new Promise<Transcript>((res, rej) => { resolve = res; reject = rej; }));
	return { resolve: (v: Transcript) => resolve(v), reject: (e: unknown) => reject(e) };
}

beforeEach(() => {
	FakeRecorder.instances = [];
	/* Once-queues survive `clearAllMocks`: a test that leaves one unspent would feed the next. */
	mockedTranscribe.mockReset();
	mockedTranscribe.mockResolvedValue(transcript(""));
	mockedSendCommand.mockReset();
	mockedSendCommand.mockResolvedValue({ ok: true, detail: "" } as Awaited<ReturnType<typeof api.sendCommand>>);
	capabilitySlot = { stt: { available: true, path: "provider_stt_radient", reason: "" } };
	trackStop = vi.fn();
	getUserMedia.mockResolvedValue({ getTracks: () => [{ stop: trackStop }] });
	vi.stubGlobal("MediaRecorder", FakeRecorder);
	vi.stubGlobal("isSecureContext", true);
	Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: { getUserMedia } });
});
afterEach(() => {
	cleanup();
	resetTurnStopForTests();
	localStorage.clear();
	vi.useRealTimers();
	vi.clearAllMocks();
	vi.unstubAllGlobals();
	Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: undefined });
});

describe("slot B while recording", () => {
	it("is Send recording, enabled, with an empty box and no turn (it was a dead Send)", async () => {
		renderComposer();
		expect(roles()).toEqual(["send"]);
		await startRecording();
		expect(roles()).toEqual(["send-recording"]);
		expect(sendRecordingDisc().disabled).toBe(false);
	});

	it("is Send recording, never Stop, while a turn runs (a tap aimed at it must not interrupt the agent)", async () => {
		renderComposer(true);
		expect(roles()).toEqual(["stop"]);
		await startRecording();
		expect(roles()).toEqual(["send-recording"]);
		expect(screen.queryByRole("button", { name: "stop" })).toBeNull();
	});

	it("goes back to the ordinary role when the take is cancelled", async () => {
		renderComposer();
		await startRecording();
		fireEvent.click(screen.getByRole("button", { name: "cancel recording" }));
		await waitFor(() => expect(roles()).toEqual(["send"]));
		expect(mockedTranscribe).not.toHaveBeenCalled();
	});
});

describe("Send recording: stop, transcribe and submit in one press", () => {
	it("sends the transcript as a dictated message and never inserts a second copy", async () => {
		renderComposer();
		mockedTranscribe.mockResolvedValueOnce(transcript("hello world"));
		await startRecording();

		fireEvent.click(sendRecordingDisc());

		await waitFor(() => expect(mockedSendCommand).toHaveBeenCalledOnce());
		const envelope = mockedSendCommand.mock.calls[0]![1] as Record<string, unknown>;
		expect(envelope.op).toBe("prompt");
		expect(envelope.text).toBe("hello world");
		expect(envelope.input_mode).toBe("dictated");
		expect(envelope.input_path).toBe("provider_stt_radient");
		await waitFor(() => expect(field().value).toBe(""));
		expect(screen.queryByText("Voice input discarded.")).toBeNull();
	});

	it("joins the draft typed before the press with the transcript, in one message", async () => {
		renderComposer();
		fireEvent.change(field(), { target: { value: "please" } });
		mockedTranscribe.mockResolvedValueOnce(transcript("fix the build"));
		await startRecording();

		fireEvent.click(sendRecordingDisc());

		await waitFor(() => expect(mockedSendCommand).toHaveBeenCalledOnce());
		const envelope = mockedSendCommand.mock.calls[0]![1] as Record<string, unknown>;
		expect(envelope.text).toBe("please fix the build");
		expect(envelope.input_mode).toBe("mixed");
	});

	it("steers while a turn runs, through the same path as a plain Send", async () => {
		renderComposer(true);
		mockedTranscribe.mockResolvedValueOnce(transcript("actually use the other file"));
		await startRecording();

		fireEvent.click(sendRecordingDisc());

		await waitFor(() => expect(mockedSendCommand).toHaveBeenCalledOnce());
		expect((mockedSendCommand.mock.calls[0]![1] as Record<string, unknown>).op).toBe("steer");
	});

	it("holds slot B as a static, inert, busy disc while the transcript is in flight", async () => {
		renderComposer(true);
		const held = holdTranscription();
		await startRecording();

		fireEvent.click(sendRecordingDisc());

		await waitFor(() => expect(roles()).toEqual(["send-recording-busy"]));
		const disc = screen.getByRole("button", { name: "send recording" });
		expect(disc.getAttribute("aria-busy")).toBe("true");
		/* A second tap is a no-op: one transcription request, no abort, no send, and
		   the slot never becomes Stop (the box is empty and a turn is running). */
		fireEvent.click(disc);
		fireEvent.click(disc);
		expect(mockedTranscribe).toHaveBeenCalledOnce();
		expect((mockedTranscribe.mock.calls[0]![1] as AbortSignal).aborted).toBe(false);
		expect(roles()).toEqual(["send-recording-busy"]);
		expect(mockedSendCommand).not.toHaveBeenCalled();
		/* The mic is the one spinner and cannot start a second take. */
		expect((screen.getByRole("button", { name: "transcribing" }) as HTMLButtonElement).disabled).toBe(true);

		await act(async () => held.resolve(transcript("go")));
		await waitFor(() => expect(mockedSendCommand).toHaveBeenCalledOnce());
		/* No Stop between the send and its settle: the slot goes busy -> connecting -> steer/stop. */
		await waitFor(() => expect(roles()).not.toContain("send-recording-busy"));
	});

	it("Enter is refused while recording: the take is kept", async () => {
		renderComposer();
		fireEvent.change(field(), { target: { value: "typed" } });
		const recorder = await startRecording();
		fireEvent.keyDown(field(), { key: "Enter" });
		await new Promise((r) => setTimeout(r, 0));
		expect(mockedSendCommand).not.toHaveBeenCalled();
		expect(recorder.state).toBe("recording");
	});
});

describe("Send recording: when nothing is sent, the words stay in the box", () => {
	it("an empty transcript sends nothing and says so", async () => {
		renderComposer();
		mockedTranscribe.mockResolvedValueOnce(transcript("   "));
		await startRecording();
		fireEvent.click(sendRecordingDisc());
		await waitFor(() => expect(screen.getByText("Didn't catch that — try again.")).toBeTruthy());
		expect(mockedSendCommand).not.toHaveBeenCalled();
		expect(roles()).toEqual(["send"]);
		expect(field().value).toBe("");
	});

	it("a failed transcription sends nothing, says why, and frees the slot", async () => {
		renderComposer();
		mockedTranscribe.mockRejectedValueOnce(new api.HttpError(502, "Transcription failed upstream."));
		await startRecording();
		fireEvent.click(sendRecordingDisc());
		await waitFor(() => expect(screen.getByText("Couldn't transcribe that. Try again.")).toBeTruthy());
		expect(mockedSendCommand).not.toHaveBeenCalled();
		expect(roles()).toEqual(["send"]);
		expect(screen.getByRole("button", { name: "start voice input" })).toBeTruthy();
	});

	it("typing during the wait sends nothing; the transcript joins what is in the box for review", async () => {
		renderComposer();
		const held = holdTranscription();
		await startRecording();
		fireEvent.click(sendRecordingDisc());
		await waitFor(() => expect(roles()).toEqual(["send-recording-busy"]));

		fireEvent.change(field(), { target: { value: "wait, also" } });
		await act(async () => held.resolve(transcript("do this")));

		await waitFor(() => expect(field().value).toBe("wait, also do this"));
		expect(mockedSendCommand).not.toHaveBeenCalled();
		expect(roles()).toEqual(["send"]);
		expect(screen.getByText("Transcript added")).toBeTruthy();
	});

	it("a transcript that is a slash command is left in the box, never run", async () => {
		renderComposer();
		mockedTranscribe.mockResolvedValueOnce(transcript("/usage"));
		await startRecording();
		fireEvent.click(sendRecordingDisc());
		await waitFor(() => expect(field().value).toBe("/usage"));
		await waitFor(() => expect(roles()).toEqual(["send"]));
		expect(mockedSendCommand).not.toHaveBeenCalled();
	});

	it("cancelling the transcription from the status row aborts it, sends nothing and frees the slot", async () => {
		renderComposer();
		const held = holdTranscription();
		await startRecording();
		fireEvent.click(sendRecordingDisc());
		const cancel = await screen.findByRole("button", { name: "cancel transcription" });
		const signal = mockedTranscribe.mock.calls[0]![1] as AbortSignal;

		fireEvent.click(cancel);

		expect(signal.aborted).toBe(true);
		await waitFor(() => expect(roles()).toEqual(["send"]));
		await act(async () => held.resolve(transcript("too late")));
		await new Promise((r) => setTimeout(r, 0));
		expect(field().value).toBe("");
		expect(mockedSendCommand).not.toHaveBeenCalled();
	});

	it("a press on a recorder that already stopped (the 120 s cap) does not enter the phase: the slot is not stuck", async () => {
		renderComposer();
		const recorder = await startRecording();
		/* The cap's stop has happened but React has not repainted: the disc is still on screen. */
		recorder.state = "inactive";

		fireEvent.click(sendRecordingDisc());
		await new Promise((r) => setTimeout(r, 0));

		expect(roles()).toEqual(["send-recording"]);
		expect(mockedTranscribe).not.toHaveBeenCalled();
		expect(mockedSendCommand).not.toHaveBeenCalled();
	});

	it("leaving the page mid-transcription sends nothing", async () => {
		const view = renderComposer();
		const held = holdTranscription();
		await startRecording();
		fireEvent.click(sendRecordingDisc());
		await waitFor(() => expect(roles()).toEqual(["send-recording-busy"]));
		view.unmount();
		await act(async () => held.resolve(transcript("too late")));
		await new Promise((r) => setTimeout(r, 0));
		expect(mockedSendCommand).not.toHaveBeenCalled();
	});
});

describe("a send that is NOT Send recording still discards a live take, and says so", () => {
	it("tapping a no-argument slash command from the tray while recording runs it and drops the take", async () => {
		vi.mocked(api.getCommands).mockResolvedValue({
			commands: [{ name: "compact", description: "Compact the conversation context", aliases: [], arguments: "none" }],
		} as Awaited<ReturnType<typeof api.getCommands>>);
		renderComposer();
		const recorder = await startRecording();
		fireEvent.change(field(), { target: { value: "/" } });

		fireEvent.click(await screen.findByRole("button", { name: /\/compact/ }));

		await waitFor(() => expect(mockedSendCommand).toHaveBeenCalledOnce());
		expect(screen.getByText("Voice input discarded.")).toBeTruthy();
		expect(mockedTranscribe).not.toHaveBeenCalled();
		expect(recorder.state).toBe("inactive");
		await waitFor(() => expect(roles()).toEqual(["send"]));
	});
});

describe("a plain Send is inert while the take is being transcribed", () => {
	it("the disc is disabled (a press would abort the request and send the old draft)", async () => {
		renderComposer();
		fireEvent.change(field(), { target: { value: "typed" } });
		const held = holdTranscription();
		await startRecording();
		fireEvent.click(screen.getByRole("button", { name: "stop and transcribe" }));
		await screen.findByText("Transcribing…");
		expect((screen.getByRole("button", { name: "send" }) as HTMLButtonElement).disabled).toBe(true);
		await act(async () => held.resolve(transcript("more")));
		await waitFor(() => expect(field().value).toBe("typed more"));
		expect((screen.getByRole("button", { name: "send" }) as HTMLButtonElement).disabled).toBe(false);
	});
});
