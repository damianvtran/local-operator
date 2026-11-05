/**
 * "A stop was requested for this session and its turn has not ended yet",
 * shared by every Stop control on the screen.
 *
 * WHY A MODULE-LEVEL STORE. The composer's Stop and the working line's Stop
 * mount in different subtrees but answer the same question, and a second tap on
 * either one used to send a second `{op:"abort"}`. One flag per session means
 * both controls show the pressed state together and the second tap is a no-op.
 * The turn ending (`alive` false) clears it, and so does a hard bound, because
 * a flag that outlives a lost event would leave Stop dead for good.
 */
import { useEffect, useSyncExternalStore } from "react";
import { sendCommand } from "../api";

/** The longest a stop stays "pending" without the turn ending (the desktop's
    bound for the same state). */
export const STOPPING_BOUND_MS = 15_000;

const pending = new Map<string, ReturnType<typeof setTimeout>>();
const listeners = new Set<() => void>();

function emit(): void {
	listeners.forEach((listener) => listener());
}

function release(pid: string): void {
	const timer = pending.get(pid);
	if (timer === undefined) return;
	clearTimeout(timer);
	pending.delete(pid);
	emit();
}

/** Ask the session to stop its turn. Resolves to the failure message, or null.
    A second call while one is pending is a no-op, not a second abort. */
export async function stopTurn(pid: string): Promise<string | null> {
	if (pending.has(pid)) return null;
	pending.set(pid, setTimeout(() => release(pid), STOPPING_BOUND_MS));
	emit();
	try {
		await sendCommand(pid, { op: "abort" });
		return null;
	} catch (e) {
		release(pid);
		return String((e as Error).message ?? e);
	}
}

/** Whether a stop is pending for `pid`; clears it when the turn is no longer `alive`. */
export function useTurnStopping(pid: string, alive: boolean): boolean {
	const stopping = useSyncExternalStore(
		(listener) => {
			listeners.add(listener);
			return () => listeners.delete(listener);
		},
		() => pending.has(pid),
		() => false,
	);
	useEffect(() => {
		if (!alive) release(pid);
	}, [alive, pid]);
	return stopping;
}

/** Test seam: forget every pending stop. */
export function resetTurnStopForTests(): void {
	for (const pid of [...pending.keys()]) release(pid);
}
