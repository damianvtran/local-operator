/**
 * The pairing short authentication string (SAS) and the commit that binds it.
 *
 * WHY A SIX-DIGIT CODE AT ALL. Terminal-free pairing (design v2 §2, A1) files a
 * request from the phone and lets the operator approve it on the computer. The
 * relay in between is not trusted to be honest about WHICH phone asked, so both
 * ends derive the same six digits from material each side contributed, and the
 * person compares them (confirm mode) or types them (code mode). A relay that
 * swapped keys would have to guess the other side's nonce before seeing it,
 * which is what the commit-then-reveal order prevents:
 *
 *   1. the phone sends `commit = sha256(np)` with its public key (claim);
 *   2. the computer answers with its nonce `nd` and its identity point;
 *   3. only then does the phone reveal `np` (file).
 *
 * THE DERIVATION IS THE RUNTIME'S, byte for byte (`pair_op.sas_code`):
 *   HKDF-SHA256(ikm = np ‖ nd ‖ device_point ‖ computer_point ‖ utf8(approval_id),
 *               salt = empty, info = "lop-pair-sas-v1", L = 8)
 * read as a big-endian uint64, mod 1 000 000, zero-padded to six digits. A
 * drift is not subtle — every comparison would mismatch — so `pair-sas.test.ts`
 * pins this against `src/fixtures/pair-sas-vectors.json`, the same file the
 * Python suite and the native apps pin.
 *
 * WebCrypto only (no hashing dependency): the PWA already requires
 * `crypto.subtle` for its non-extractable device key, so a browser that lacks
 * it cannot pair anyway.
 */

/** HKDF `info`, versioned so a future derivation cannot collide with this one. */
export const SAS_INFO = "lop-pair-sas-v1";

const encoder = new TextEncoder();

/** Unpadded url-safe base64 → bytes (the wire form of every pairing field). */
export function base64UrlToBytes(value: string): Uint8Array {
	const padded = value.replace(/-/g, "+").replace(/_/g, "/");
	const binary = atob(padded + "=".repeat((4 - (padded.length % 4)) % 4));
	const out = new Uint8Array(binary.length);
	for (let index = 0; index < binary.length; index += 1) out[index] = binary.charCodeAt(index);
	return out;
}

/** Bytes → unpadded url-safe base64. */
export function bytesToBase64Url(bytes: Uint8Array): string {
	let binary = "";
	for (const byte of bytes) binary += String.fromCharCode(byte);
	return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function concat(parts: Uint8Array[]): Uint8Array {
	const out = new Uint8Array(parts.reduce((sum, part) => sum + part.length, 0));
	let offset = 0;
	for (const part of parts) {
		out.set(part, offset);
		offset += part.length;
	}
	return out;
}

/** The phone's 32-byte nonce. Fresh per filed request, never reused: reusing it
    would let an observer of one SAS predict the commit of the next. */
export function freshNonce(): Uint8Array {
	return crypto.getRandomValues(new Uint8Array(32));
}

/** `commit = base64url(sha256(np))` — what the claim sends before `np` is revealed. */
export async function sasCommit(np: Uint8Array): Promise<string> {
	const digest = await crypto.subtle.digest("SHA-256", np as unknown as ArrayBuffer);
	return bytesToBase64Url(new Uint8Array(digest));
}

/** The six-digit SAS, as a zero-padded string ("004213"). */
export async function sasCode(input: {
	np: Uint8Array;
	nd: Uint8Array;
	devicePoint: Uint8Array;
	computerPoint: Uint8Array;
	approvalId: string;
}): Promise<string> {
	const ikm = concat([
		input.np,
		input.nd,
		input.devicePoint,
		input.computerPoint,
		encoder.encode(input.approvalId),
	]);
	const key = await crypto.subtle.importKey("raw", ikm as unknown as ArrayBuffer, "HKDF", false, [
		"deriveBits",
	]);
	const bits = await crypto.subtle.deriveBits(
		{
			name: "HKDF",
			hash: "SHA-256",
			salt: new Uint8Array(0),
			info: encoder.encode(SAS_INFO),
		},
		key,
		64,
	);
	/* BigInt, not Number: the 8-byte value exceeds 2^53, and a float modulo would
	   silently disagree with the runtime on most inputs. */
	const value = new DataView(bits).getBigUint64(0, false);
	return (value % 1_000_000n).toString().padStart(6, "0");
}

/** "482913" → "482 913": the grouping the desktop card draws, so the two read alike. */
export function groupSas(code: string): string {
	return `${code.slice(0, 3)} ${code.slice(3)}`;
}
