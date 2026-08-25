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

interface StoredContinuation {
	version: 1;
	saved_at: number;
	envelope: ContinuationEnvelope;
}

const COMMAND_PREFIX = "lo-mobile-command:";
const COMMAND_TTL_MS = 24 * 60 * 60 * 1000;
const MAX_STORED_COMMAND_CHARS = 4 * 1024 * 1024;
const MAX_COMMAND_TEXT_CHARS = 100_000;
const MAX_COMMAND_IMAGES = 8;
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function commandStorageKey(sessionId: string): string {
	return `${COMMAND_PREFIX}${sessionId}`;
}

function validEnvelope(value: Partial<ContinuationEnvelope>): value is ContinuationEnvelope {
	return (
		typeof value.command_id === "string" &&
		UUID_RE.test(value.command_id) &&
		(value.op === "prompt" || value.op === "steer") &&
		typeof value.text === "string" &&
		value.text.length <= MAX_COMMAND_TEXT_CHARS &&
		(value.images === undefined ||
			(Array.isArray(value.images) &&
				value.images.length <= MAX_COMMAND_IMAGES &&
				value.images.every(
					(image) =>
						typeof image === "object" &&
						image !== null &&
						typeof image.data_b64 === "string" &&
						typeof image.mime_type === "string" &&
						image.mime_type.startsWith("image/"),
				)))
	);
}

function saveEnvelope(key: string, envelope: ContinuationEnvelope, savedAt: number): void {
	const raw = JSON.stringify({ version: 1, saved_at: savedAt, envelope } satisfies StoredContinuation);
	if (raw.length > MAX_STORED_COMMAND_CHARS) {
		throw new Error("This instruction is too large to retain safely for retry.");
	}
	try {
		localStorage.setItem(key, raw);
	} catch {
		throw new Error("This instruction could not be retained safely for retry.");
	}
}

export function getPendingContinuation(sessionId: string): ContinuationEnvelope | null {
	return storedEnvelope(commandStorageKey(sessionId));
}

export function hasPendingContinuation(sessionId: string): boolean {
	return getPendingContinuation(sessionId) !== null;
}

export function clearPendingContinuation(sessionId: string): void {
	localStorage.removeItem(commandStorageKey(sessionId));
}

export function clearPendingContinuationsExcept(sessionId?: string): void {
	for (let index = localStorage.length - 1; index >= 0; index--) {
		const key = localStorage.key(index);
		if (key?.startsWith(COMMAND_PREFIX) && key !== commandStorageKey(sessionId ?? "")) {
			localStorage.removeItem(key);
		}
	}
}

function storedEnvelope(key: string): ContinuationEnvelope | null {
	const raw = localStorage.getItem(key);
	if (!raw || raw.length > MAX_STORED_COMMAND_CHARS) {
		localStorage.removeItem(key);
		return null;
	}
	try {
		const value = JSON.parse(raw) as Partial<StoredContinuation> & Partial<ContinuationEnvelope>;
		if (value.version === 1) {
			if (
				typeof value.saved_at === "number" &&
				value.saved_at <= Date.now() &&
				Date.now() - value.saved_at <= COMMAND_TTL_MS &&
				value.envelope &&
				validEnvelope(value.envelope)
			) {
				return value.envelope;
			}
			localStorage.removeItem(key);
			return null;
		}
		if (validEnvelope(value)) {
			/* Upgrade the short-lived raw envelope shipped by the previous build
			   without granting it unbounded storage lifetime. */
			saveEnvelope(key, value, Date.now());
			return value;
		}
	} catch {
		/* Corrupt or quota-truncated state has no trustworthy UUID/body pairing. */
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
	const stored = storedEnvelope(key);
	const envelope = stored ?? {
		op,
		command_id: crypto.randomUUID(),
		text,
		images: images?.map((image) => ({ ...image })),
	};
	if (!validEnvelope(envelope)) {
		clearPendingContinuation(sessionId);
		throw new Error("This instruction cannot be retained safely for retry.");
	}
	if (!stored) saveEnvelope(key, envelope, Date.now());
	try {
		const reply = await sendCommand(sessionId, envelope);
		clearPendingContinuation(sessionId);
		return { ...reply, envelope };
	} catch (error) {
		/* An HTTP response proves rejection; only transport uncertainty keeps the
		   UUID/body pair retryable. Replaying a rejected envelope is never useful. */
		if (
			typeof error === "object" &&
			error !== null &&
			"status" in error &&
			typeof error.status === "number"
		) {
			clearPendingContinuation(sessionId);
		}
		throw error;
	}
}
