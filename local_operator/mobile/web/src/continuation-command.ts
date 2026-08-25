import { sendCommand } from "./api";
import type { PromptImage } from "./types";

export type ContinuationOp = "prompt" | "steer";

function commandStorageKey(sessionId: string): string {
	return `lo-mobile-command:${sessionId}`;
}

/** Keep one producer receipt across transport retries, then retire it only
 * after the daemon acknowledges the command. Reusing an acknowledged receipt
 * for a later steer would make the owner correctly discard new user input as a
 * duplicate. */
export async function submitContinuation(
	sessionId: string,
	op: ContinuationOp,
	text: string,
	images?: PromptImage[],
): Promise<{ ok: boolean; detail: string }> {
	const key = commandStorageKey(sessionId);
	const commandId = localStorage.getItem(key) ?? crypto.randomUUID();
	localStorage.setItem(key, commandId);
	const reply = await sendCommand(sessionId, {
		op,
		command_id: commandId,
		text,
		images,
	});
	localStorage.removeItem(key);
	return reply;
}
