import { sendCommand } from "./api";
import type { PromptImage } from "./types";

export type ContinuationOp = "prompt" | "steer";

export interface ContinuationEnvelope {
	command_id: string;
	op: ContinuationOp;
	text: string;
	images?: PromptImage[];
}

export interface ContinuationReceipt {
	ok: boolean;
	detail: string;
	envelope: ContinuationEnvelope;
}

function commandStorageKey(sessionId: string): string {
	return `lo-mobile-command:${sessionId}`;
}

export function hasPendingContinuation(sessionId: string): boolean {
	return storedEnvelope(commandStorageKey(sessionId)) !== null;
}

function storedEnvelope(key: string): ContinuationEnvelope | null {
	const raw = localStorage.getItem(key);
	if (!raw) return null;
	try {
		const value = JSON.parse(raw) as Partial<ContinuationEnvelope>;
		if (
			typeof value.command_id === "string" &&
			(value.op === "prompt" || value.op === "steer") &&
			typeof value.text === "string" &&
			(value.images === undefined || Array.isArray(value.images))
		) {
			return value as ContinuationEnvelope;
		}
	} catch {
		/* A pre-envelope receipt cannot safely recover its payload. Retiring it
		   avoids binding freshly typed content to an identity whose admitted body
		   is unknowable after an upgrade. */
	}
	localStorage.removeItem(key);
	return null;
}

/** Retain the exact producer envelope until acknowledgement. The UUID is the
 * identity of this body, not a mutable composer slot: after uncertain delivery,
 * retries replay these bytes while later edits remain a separate draft. */
export async function submitContinuation(
	sessionId: string,
	op: ContinuationOp,
	text: string,
	images?: PromptImage[],
): Promise<ContinuationReceipt> {
	const key = commandStorageKey(sessionId);
	const envelope = storedEnvelope(key) ?? {
		op,
		command_id: crypto.randomUUID(),
		text,
		images: images?.map((image) => ({ ...image })),
	};
	localStorage.setItem(key, JSON.stringify(envelope));
	const reply = await sendCommand(sessionId, envelope);
	localStorage.removeItem(key);
	return { ...reply, envelope };
}
