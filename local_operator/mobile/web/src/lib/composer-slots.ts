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
export type PrimaryRole = "connecting" | "stopping" | "stop" | "steer" | "send";

export function primaryRole(input: {
	/** A turn is running (`projection.streaming`). */
	alive: boolean;
	/** A stop has been requested and the turn has not ended yet. */
	stopping: boolean;
	/** A send is in flight: the role is locked so a double tap cannot reach Stop. */
	sending: boolean;
	/** The box holds text or an attachment. */
	hasPayload: boolean;
}): PrimaryRole {
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
