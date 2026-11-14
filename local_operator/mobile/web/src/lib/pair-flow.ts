/**
 * Terminal-free pairing, the phone's half (design v2 §2.2 and §5.2, amendments
 * A1/A2): availability → claim → reveal/file → poll → certificate.
 *
 * WHY THIS EXISTS. Pairing used to need a code printed by a terminal command on
 * the computer, and a phone user who tapped Approve on an unpaired phone was told
 * to go and run it — a dead end for anyone who does not live in a terminal (Ben's
 * report, A1). Now the phone FILES a request and the person approves it on the
 * computer (a desktop popup, or the TUI), where the computer's own password is
 * the authority. Nothing here can approve anything: the phone only proposes its
 * public key and shows the six digits both ends derived.
 *
 * React-free on purpose: the screen and the in-place sheet share one engine, and
 * the tests can drive the protocol (and its refusals) without rendering.
 *
 * CADENCES ARE UI CADENCES, not protocol limits (contract §1, §5): the status
 * poll every 2 s and the availability re-check every 5 s only bound how quickly a
 * VISIBLE screen notices a change made on the computer. The computer enforces
 * the request window and the filing rate itself.
 */
import {
	HttpError,
	pairAvailability,
	pairClaim,
	pairFile,
	pairingStatus,
	pairRequestStatus,
	pairWithdraw,
	type PairRecord,
	type PairRequestStatus,
} from "../api";
import { base64UrlToBytes, bytesToBase64Url, freshNonce, sasCode, sasCommit } from "./pair-sas";
import {
	clearDeviceKeyLost,
	deviceIdFor,
	encodePoint,
	forgetDeviceKey,
	loadOrCreateDeviceKey,
	signWithDeviceKey,
	storeCertificate,
} from "./operator-device";

/** How often a visible waiting screen asks whether the computer has answered. */
export const STATUS_POLL_MS = 2000;
/** How often a visible "not available" screen re-checks availability (D15). */
export const AVAILABILITY_POLL_MS = 5000;
/** After this long on the waiting screen, add the "don't see it?" hint (U11). */
export const NOT_SEEN_HINT_MS = 60_000;

/** The pending request, persisted so a reload resumes it or withdraws it rather
    than leaving an orphan on the computer the person has to deny. Holds `np`
    because the SAS is recomputed from it on resume; by the time it is stored it
    has already been revealed to the computer, so it is no longer a secret. */
const PENDING_KEY = "lo-mobile-pair-request";

export type PairMode = "confirm" | "code";

export interface PendingPair {
	requestId: string;
	deviceId: string;
	approvalId: string;
	np: string;
	nd: string;
	computerPub: string;
	mode: PairMode;
	expiresAt: number;
}

export function savedPendingPair(): PendingPair | null {
	try {
		const raw = localStorage.getItem(PENDING_KEY);
		if (!raw) return null;
		const parsed = JSON.parse(raw) as PendingPair;
		return typeof parsed?.requestId === "string" ? parsed : null;
	} catch {
		return null;
	}
}

export function clearPendingPair(): void {
	localStorage.removeItem(PENDING_KEY);
}

function savePendingPair(pending: PendingPair): void {
	localStorage.setItem(PENDING_KEY, JSON.stringify(pending));
}

/** 32 lowercase hex chars, the contract's client-generated request id. */
export function newRequestId(): string {
	let out = "";
	for (const byte of crypto.getRandomValues(new Uint8Array(16))) {
		out += byte.toString(16).padStart(2, "0");
	}
	return out;
}

/** The record's `via` says whether the computer will show the code for the
    person to compare (Radient-attested route) or ask them to type it. */
export function modeFromRecord(record: PairRecord): PairMode {
	const via = record.what?.pair_phone?.via;
	return via?.route === "radient" && via?.attested === "edge-v1" ? "confirm" : "code";
}

export type Availability = "available" | "unavailable" | "unsupported";

/** A 404 is a daemon from before terminal-free pairing: do not offer the flow. */
export async function checkAvailability(): Promise<Availability> {
	try {
		const answer = await pairAvailability();
		if (answer.available) return "available";
		return answer.reason === "unsupported" ? "unsupported" : "unavailable";
	} catch (error) {
		if (error instanceof HttpError && error.status === 404) return "unsupported";
		throw error;
	}
}

/** Every outcome of filing the phone treats differently, keyed on `code`. */
export type FileOutcome =
	| { kind: "filed"; pending: PendingPair }
	| { kind: "busy_same"; requestId: string }
	| { kind: "busy_other" }
	/* `retryAfterS` is null when the refusal carried no usable wait: the screen
	   then says "try again later" rather than counting down an invented number. */
	| { kind: "rate_limited"; retryAfterS: number | null }
	| { kind: "revoked" }
	| { kind: "unavailable" };

/** Claim then reveal. The order is the security shape: the commit goes first,
    the computer's nonce comes back, and only then is `np` revealed. */
export async function fileRequest(name: string): Promise<FileOutcome> {
	const key = await loadOrCreateDeviceKey();
	const spki = encodePoint(key.point);
	const np = freshNonce();
	const commit = await sasCommit(np);
	try {
		const claim = await pairClaim({ spki, commit });
		const requestId = newRequestId();
		const record = await pairFile({
			request_id: requestId,
			spki,
			commit,
			np: bytesToBase64Url(np),
			name,
			platform: "web",
		});
		const pending: PendingPair = {
			requestId: record.request_id || requestId,
			deviceId: record.device?.device_id || (await deviceIdFor(key.point)),
			approvalId: record.approval_id,
			np: bytesToBase64Url(np),
			nd: claim.nd,
			computerPub: record.what?.pair_phone?.computer_id_pub || claim.computer_id_pub,
			mode: modeFromRecord(record),
			expiresAt: Number(record.expires_at ?? claim.expires_at ?? 0),
		};
		savePendingPair(pending);
		return { kind: "filed", pending };
	} catch (error) {
		const refused = classifyRefusal(error);
		if (refused === null) throw error;
		if (refused.kind === "revoked") await rekeyAfterRevocation();
		return refused;
	}
}

function classifyRefusal(error: unknown): Exclude<FileOutcome, { kind: "filed" }> | null {
	if (!(error instanceof HttpError)) return null;
	const body = error.body;
	switch (error.code) {
		case "pair_request_pending":
			if (body.same_key === true && typeof body.request_id === "string") {
				return { kind: "busy_same", requestId: body.request_id };
			}
			return { kind: "busy_other" };
		case "pair_filing_rate_limited": {
			const seconds = Number(body.retry_after_s);
			/* The daemon always sends `retry_after_s`; a refusal without it (a
			   proxy, a future daemon) gets no guessed countdown — there is no
			   measured default to show. */
			const usable = body.retry_after_s !== undefined && body.retry_after_s !== null && Number.isFinite(seconds);
			return { kind: "rate_limited", retryAfterS: usable ? Math.max(0, Math.ceil(seconds)) : null };
		}
		case "pair_device_revoked":
			return { kind: "revoked" };
		case "approval_anchor_unavailable":
			return { kind: "unavailable" };
		default:
			return null;
	}
}

/** The six digits for a pending request, recomputed from what was persisted. */
export async function sasFor(pending: PendingPair): Promise<string> {
	const key = await loadOrCreateDeviceKey();
	return await sasCode({
		np: base64UrlToBytes(pending.np),
		nd: base64UrlToBytes(pending.nd),
		devicePoint: key.point,
		computerPoint: base64UrlToBytes(pending.computerPub),
		approvalId: pending.approvalId,
	});
}

/** A 404 `unknown_request` is a request the computer no longer holds: expired. */
export async function pollStatus(requestId: string): Promise<PairRequestStatus> {
	try {
		return await pairRequestStatus(requestId);
	} catch (error) {
		if (error instanceof HttpError && error.status === 404) {
			return { state: "expired", reason: "expired" };
		}
		throw error;
	}
}

export const TERMINAL_REASONS = new Set([
	"strikes_exhausted",
	"denied_by_user",
	"withdrawn",
	"expired",
	"paired",
]);

const WITHDRAW_TAG = "lop-pair-withdraw-v1";

export type WithdrawOutcome = "withdrawn" | "in_progress";

/** Cancel this phone's own request, signed so only this key can. */
export async function withdrawRequest(requestId: string): Promise<WithdrawOutcome> {
	const message = new TextEncoder().encode(WITHDRAW_TAG + requestId);
	const sig = await signWithDeviceKey(message);
	try {
		await pairWithdraw(requestId, sig);
	} catch (error) {
		if (error instanceof HttpError && error.code === "approval_in_progress") return "in_progress";
		/* Unknown or already-gone requests have nothing left to cancel; treating
		   them as withdrawn lets the person start again instead of looping. */
		if (error instanceof HttpError && error.status === 404) {
			clearPendingPair();
			return "withdrawn";
		}
		throw error;
	}
	clearPendingPair();
	return "withdrawn";
}

/** `paired`: fetch and keep the certificate, then ask the browser to keep it. */
export async function completePairing(deviceId: string): Promise<{ authorityReady: boolean }> {
	const answer = await pairingStatus(deviceId);
	if (!answer.paired || !answer.certificate) throw new Error("certificate not ready");
	storeCertificate(answer.certificate, answer.operator_key_id ?? deviceId);
	clearPendingPair();
	clearDeviceKeyLost();
	await requestPersistentStorage();
	return { authorityReady: answer.authority_ready !== false };
}

/**
 * U15: the computer removed this phone. Discard the key and certificate and mint
 * a fresh key, so "pair again" is a NEW request the computer has never revoked —
 * and still needs a new approval there.
 */
export async function rekeyAfterRevocation(): Promise<void> {
	clearPendingPair();
	await forgetDeviceKey();
	await loadOrCreateDeviceKey();
}

/**
 * U16: ask the browser not to evict this origin's storage. Best effort — Safari
 * and Chrome decide by engagement and installed-ness, and some browsers lack the
 * API entirely — so a refusal is not an error; the Add-to-Home-Screen tip is the
 * other half of the same mitigation.
 */
export async function requestPersistentStorage(): Promise<boolean> {
	try {
		if (typeof navigator === "undefined" || !navigator.storage?.persist) return false;
		return await navigator.storage.persist();
	} catch {
		return false;
	}
}

/** Whether the page already runs as an installed app (no tip needed). */
export function isStandalone(): boolean {
	if (typeof window === "undefined") return false;
	const iosStandalone = (navigator as Navigator & { standalone?: boolean }).standalone === true;
	return iosStandalone || window.matchMedia?.("(display-mode: standalone)")?.matches === true;
}

/** Seconds → "m:ss" (N2), the format the computer's card counts down in. */
export function formatCountdown(seconds: number): string {
	const whole = Math.max(0, Math.ceil(seconds));
	return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}
