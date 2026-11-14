/**
 * The composer's two action slots as PURE decisions: which control the primary
 * slot draws, whether the dictation slot is drawn, and when a press on a
 * destructive control is a reflex rather than a decision.
 *
 * WHY THIS IS A MODULE. The right-hand cluster is two FIXED 44 px slots, [A: mic]
 * then [B: primary], and the hazard they exist to remove is positional: a tap
 * aimed at one control landing on another because the row reflowed under the
 * finger. The old cluster drew Stop BESIDE Steer, so the cluster's width changed
 * with the turn, and moving the mic into it would have made the mic slide left
 * every time a turn began. Send, Steer and Stop are therefore ONE slot with ONE
 * occupant: they are never drawn together, and nothing else moves when it
 * changes. The decision is a function of five booleans so it can be tested
 * without a browser; the component only draws what it returns.
 */
import type { Capabilities } from "../types";

/** What slot B holds. First match wins in `primaryRole`. */
export type PrimaryRole =
	| "send-recording-busy"
	| "send-recording"
	| "connecting"
	| "stopping"
	| "stop"
	| "steer"
	| "send";

/** The mic's phase, as the composer's dictation state names it. */
export type DictationPhase = "idle" | "recording" | "transcribing";

/**
 * "Send recording": the press that stops the take, transcribes it and submits
 * the words as a message, in one tap. Mirrors the desktop composer's machine
 * (`SendRecordingPhase` in local-operator-ui `composer-slots.ts`); the phone's
 * differs in ONE place, `transcript`, because the phone appends the transcript
 * to the draft at the moment it LANDS (not at the press), so the "did the box
 * change under the wait" check is made there, against the draft and the
 * attachments the press saw.
 *
 *   idle -> awaiting-transcript            the press (recorder stopping, transcribing)
 *   awaiting-transcript -> ready-to-submit a usable transcript landed and the box
 *                                          is what the press saw
 *   awaiting-transcript -> idle            failure, empty transcript, discarded
 *                                          take, a cancel, or the box was edited
 *                                          meanwhile (the words stay in the box,
 *                                          nothing is sent)
 *   ready-to-submit -> submitting          the render holding the transcript is
 *                                          what was expected; the send has fired
 *   ready-to-submit -> idle                it is not: nothing is sent
 *   submitting -> idle                     the send settled (accepted or refused)
 */
export type SendRecordingPhase = "idle" | "awaiting-transcript" | "ready-to-submit" | "submitting";

export type SendRecordingEvent =
	| { type: "press" }
	/** The transcription settled. `usable`: a non-empty transcript landed AND the box is what the press saw. */
	| { type: "transcript"; usable: boolean }
	/** The take was thrown away (cancelled, too short, no audio, an error, an unmount). */
	| { type: "discarded" }
	/** The render after the transcript landed. `entered`: the send was started. */
	| { type: "checked"; entered: boolean }
	/** The send path has run. `sending` is its in-flight flag as of the next render. */
	| { type: "settled"; sending: boolean };

export function sendRecordingNext(
	phase: SendRecordingPhase,
	event: SendRecordingEvent,
): SendRecordingPhase {
	switch (event.type) {
		case "press":
			return phase === "idle" ? "awaiting-transcript" : phase;
		case "transcript":
			if (phase !== "awaiting-transcript") return phase;
			return event.usable ? "ready-to-submit" : "idle";
		case "discarded":
			return "idle";
		case "checked":
			if (phase !== "ready-to-submit") return phase;
			return event.entered ? "submitting" : "idle";
		case "settled":
			return phase === "submitting" && !event.sending ? "idle" : phase;
		default:
			return phase;
	}
}

/**
 * Whether the box is still what the "Send recording" press saw: the same draft
 * text and the SAME attachments (by identity, the list is replaced on every
 * change). Anything else means the user typed or attached during the wait; those
 * words were never part of the press, so nothing is sent and the transcript is
 * appended as an ordinary dictation for them to review.
 */
export function dictatedDraftIntact(
	atPress: { text: string; images: readonly unknown[] },
	now: { text: string; images: readonly unknown[] },
): boolean {
	return atPress.text === now.text && atPress.images === now.images;
}

export function primaryRole(input: {
	/** A turn is running (`projection.streaming`). */
	alive: boolean;
	/** A stop has been requested and the turn has not ended yet. */
	stopping: boolean;
	/** A send is in flight: the role is locked so a double tap cannot reach Stop. */
	sending: boolean;
	/** The box holds text or an attachment. */
	hasPayload: boolean;
	/** The mic's phase. While recording, slot B is "Send recording". */
	dictation: DictationPhase;
	/** "Send recording" has been pressed and has not settled. */
	sendRecording: SendRecordingPhase;
}): PrimaryRole {
	/* A "Send recording" in flight holds slot B as a static, inert disc that still
	   says what the press was. Without it the disc would flip to Stop (empty box,
	   working) or back to an ordinary Send under a finger that is still on it, and
	   a second tap would cancel the transcription the first one asked for. */
	if (input.sendRecording !== "idle") return "send-recording-busy";
	/* RECORDING: the disc at the Send spot sends the take. On the phone it used to
	   stay a plain Send/Steer/Stop, which threw the take away ("Voice input
	   discarded.") or stopped the agent: the desktop's recording row puts "Send
	   recording" here and never an interrupt. Stop stays reachable on the working
	   line. */
	if (input.dictation === "recording") return "send-recording";
	/* The send just cleared the box, so `alive && !hasPayload` is true for the
	   instant before the request settles. Without this lock the disc flips from
	   Steer to Stop under the finger that is still double-tapping it. */
	if (input.sending) return "connecting";
	/* Stopping outranks typed text: the pressed state is the answer to the tap,
	   and a Steer drawn into a dying turn would hide it. */
	if (input.alive && input.stopping) return "stopping";
	if (input.alive && !input.hasPayload) return "stop";
	return input.alive ? "steer" : "send";
}

/** How long a destructive role ignores a pointer press after it ARRIVES in its
    slot. The measured reflex re-press after a state change is 120-170 ms (the
    desktop interrupt record), so 500 ms clears it with margin while staying
    shorter than a deliberate second tap. */
export const GUARD_MS = 500;

/**
 * Whether a click on a guarded control is a reflex to swallow.
 *
 * `detail` is the click count: 0 for keyboard and assistive activation, which
 * is never a stray finger, so only a pointer/touch click (detail > 0) can be
 * ignored. The caller must NOT disable the button when this returns true: a
 * disabled button drops focus, and the press has to look like an ordinary tap
 * that did nothing yet.
 */
export function pressIsReflex(arrivedAt: number, now: number, detail: number): boolean {
	return detail > 0 && now - arrivedAt < GUARD_MS;
}

/** What slot A draws: the mic, an inert same-size box, or nothing. */
export type MicSlot = "mic" | "reserved" | "none";

/**
 * Slot A's occupancy.
 *
 * - The browser cannot record (plain HTTP, no getUserMedia/MediaRecorder): the
 *   mic can never exist, so the slot is collapsed from the first frame. A
 *   reserved void would be permanent for those users.
 * - The browser can, but the daemon has not answered yet (`capabilities` is
 *   null until the first list frame): reserve the 44 px so the answer does not
 *   shift the controls under a finger.
 * - The daemon answered: draw the mic iff a path can run, else collapse. That
 *   is one layout change at load and the choice is then stable for the session.
 */
export function micSlot(browserCanRecord: boolean, capabilities: Capabilities | null): MicSlot {
	if (!browserCanRecord) return "none";
	if (capabilities === null) return "reserved";
	return capabilities.stt?.available === true ? "mic" : "none";
}

/** A role that leaves its slot and returns within this long is a flicker of the
    input (a turn flag dipping across a reconnect or a tool boundary), not a
    new arrival. Re-arming the guard on it would swallow a deliberate Stop. */
export const FLICKER_MS = 1500;

export interface Arrival {
	role: PrimaryRole;
	/** `performance.now()` at which this role last ARRIVED in the slot. */
	arrivedAt: number;
	/** The role that held the slot before this one, with when it arrived and left. */
	previous: { role: PrimaryRole; arrivedAt: number; leftAt: number } | null;
}

/**
 * Fold a role observation into the arrival record. Unchanged role: same record.
 * A change records the arrival time, except a quick return to the role just
 * left, which keeps that role's ORIGINAL arrival (a flicker must not re-arm).
 */
export function trackArrival(prev: Arrival | null, role: PrimaryRole, now: number): Arrival {
	if (prev === null) return { role, arrivedAt: now, previous: null };
	if (prev.role === role) return prev;
	const back = prev.previous;
	const flicker = back !== null && back.role === role && now - back.leftAt < FLICKER_MS;
	return {
		role,
		arrivedAt: flicker ? back.arrivedAt : now,
		previous: { role: prev.role, arrivedAt: prev.arrivedAt, leftAt: now },
	};
}

/** Whether browser APIs for recording exist at all (secure context, getUserMedia,
    MediaRecorder). The old plain-HTTP case is deliberately excluded: a mic that
    appears and fails is worse than one that never does. */
export function browserCanRecord(): boolean {
	return (
		typeof window !== "undefined" &&
		window.isSecureContext === true &&
		typeof navigator !== "undefined" &&
		!!navigator.mediaDevices?.getUserMedia &&
		typeof MediaRecorder !== "undefined"
	);
}
