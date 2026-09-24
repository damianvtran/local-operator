/**
 * The phone's operator device key: WebCrypto ES256, non-extractable, and the
 * one place a challenge is signed (stage D of issue #1310, design §2.2/§2.3).
 *
 * WHY THIS LIVES ON THE PHONE AT ALL. Before this, the phone could not loosen a
 * running approval gate in ANY session: the authority was the spawn capability,
 * and the relay's spawn path never receives one, so `entry.operator_cap` was
 * always `None` on that surface. The redesign moves authority to a fact about the
 * OPERATOR — or the operator's device — so the phone becomes a signer, and it does
 * so without the relay gaining the ability to mint anything: the relay carries a
 * challenge out and a signature back, and can produce neither.
 *
 * THE PRIVATE HALF NEVER LEAVES THE DEVICE, and that is enforced by the platform
 * rather than by care: the key is generated with `extractable: false`, so
 * `crypto.subtle.exportKey` refuses on it and nothing here can be made to leak it
 * — not a bug in this file, not a compromise of the portal's own JavaScript. What
 * this machine holds is an operator-SIGNED certificate over the public point, and
 * a signature from anyone who stole that certificate verifies against nothing.
 *
 * THE MESSAGE FRAMING IS THE RUNTIME'S, byte for byte. `signedMessageBytes` here
 * reproduces `local_operator/operator/verify.py`'s `signed_message`: a domain tag
 * plus four length-prefixed fields. A drift between the two is not a subtle bug —
 * every phone signature would fail — which is why the test suite pins this
 * implementation against a vector produced BY the Python one
 * (`operator-device.test.ts`).
 *
 * AND THE SIGNATURE IS DER, not WebCrypto's native form. `crypto.subtle` returns
 * ECDSA signatures as raw `r||s` (64 bytes for P-256) while the runtime parses DER
 * (`MAX_SIGNATURE_BYTES` = 80 for exactly that reason), so `rawSignatureToDer`
 * does the conversion in one place. A phone that sent the raw form would be
 * refused as "not a signature" — a failure that looks like a key problem and is
 * not.
 */

/** The domain tag, matching `verify.py`'s `DOMAIN`. Versioned so a future message
    shape cannot be confused with this one during a migration. */
const DOMAIN = "lop-operator-v1\u0000";

/** `OPERATOR_CAP_BYTES * 2` in the runtime: a challenge is 32 random bytes, hex. */
export const CHALLENGE_HEX_CHARS = 64;

/** How many hex characters a key id has, matching `verify.key_id_for`. */
export const KEY_ID_HEX_CHARS = 32;

/** The scopes a paired device carries, in the runtime's spelling. */
export const DEVICE_SCOPES = ["loosen", "approve"] as const;

export type OperatorAction = "loosen" | "approve";

export interface OperatorProof {
	/** The DER signature, hex — the wire form the runtime parses. */
	sig: string;
	keyId: string;
	/** The operator-signed certificate this device presents with the signature. */
	certificate: string;
}

const textEncoder = new TextEncoder();

/** 4-byte big-endian length, then UTF-8 bytes — `verify.py`'s `_lp`. */
export function lengthPrefix(value: string): Uint8Array {
	const raw = textEncoder.encode(value);
	const out = new Uint8Array(4 + raw.length);
	new DataView(out.buffer).setUint32(0, raw.length, false);
	out.set(raw, 4);
	return out;
}

function concat(parts: Uint8Array[]): Uint8Array {
	const total = parts.reduce((sum, part) => sum + part.length, 0);
	const out = new Uint8Array(total);
	let offset = 0;
	for (const part of parts) {
		out.set(part, offset);
		offset += part.length;
	}
	return out;
}

/**
 * The ONE message a signature covers: `verify.py`'s `signed_message`.
 *
 * Every field is bound into it, which is what makes a signature per-action,
 * per-session and per-challenge rather than a bearer token. The length prefixes
 * are not decoration: without them an attacker controlling one field could shift
 * bytes between adjacent fields and produce a second, valid-looking message for a
 * different action.
 */
export function signedMessageBytes(input: {
	action: OperatorAction;
	sessionId: string;
	requestId: string;
	challenge: string;
}): Uint8Array {
	return concat([
		textEncoder.encode(DOMAIN),
		lengthPrefix(input.action),
		lengthPrefix(input.sessionId),
		lengthPrefix(input.requestId),
		lengthPrefix(input.challenge),
	]);
}

export function bytesToHex(bytes: Uint8Array): string {
	let out = "";
	for (const byte of bytes) out += byte.toString(16).padStart(2, "0");
	return out;
}

export function hexToBytes(hex: string): Uint8Array {
	if (hex.length % 2) throw new Error("hex string must have an even length");
	const out = new Uint8Array(hex.length / 2);
	for (let index = 0; index < out.length; index += 1) {
		const byte = Number.parseInt(hex.slice(index * 2, index * 2 + 2), 16);
		if (Number.isNaN(byte)) throw new Error("not a hex string");
		out[index] = byte;
	}
	return out;
}

/** The DER length octets for a P-256 integer. 33 bytes (a leading zero for a
    high bit) fits the short form either way, so no long-form handling is needed —
    and writing it anyway would be a branch nothing exercises. */
function derInteger(raw: Uint8Array): Uint8Array {
	let start = 0;
	while (start < raw.length - 1 && raw[start] === 0) start += 1;
	const trimmed = raw.subarray(start);
	const needsPad = (trimmed[0] & 0x80) !== 0;
	const out = new Uint8Array(2 + trimmed.length + (needsPad ? 1 : 0));
	out[0] = 0x02;
	out[1] = out.length - 2;
	if (needsPad) out[2] = 0x00;
	out.set(trimmed, needsPad ? 3 : 2);
	return out;
}

/**
 * `crypto.subtle`'s raw `r||s` → the DER the runtime parses.
 *
 * One conversion, in one place, because the failure it prevents is
 * misattributable: a raw 64-byte signature is a well-formed-looking hex string
 * that the verifier rejects as though the KEY were wrong.
 */
export function rawSignatureToDer(raw: Uint8Array): Uint8Array {
	if (raw.length !== 64) {
		throw new Error("a P-256 signature is 64 raw bytes");
	}
	const r = derInteger(raw.subarray(0, 32));
	const s = derInteger(raw.subarray(32, 64));
	const body = concat([r, s]);
	if (body.length > 0x7f) throw new Error("signature too long for short-form DER");
	return concat([new Uint8Array([0x30, body.length]), body]);
}

/** DER → raw `r||s`. The inverse, used by the tests to verify what we produce. */
export function derSignatureToRaw(der: Uint8Array): Uint8Array {
	if (der[0] !== 0x30) throw new Error("not a DER sequence");
	const out = new Uint8Array(64);
	let offset = 2;
	for (const half of [0, 1]) {
		if (der[offset] !== 0x02) throw new Error("not a DER integer");
		const length = der[offset + 1];
		let start = offset + 2;
		if (length === 33 && der[start] === 0x00) start += 1;
		out.set(der.subarray(start, offset + 2 + length), half * 32);
		offset += 2 + length;
	}
	return out;
}

// ---------------------------------------------------------------------------
// The key: generated here, non-extractable, stored as a CryptoKey
// ---------------------------------------------------------------------------

const IDB_NAME = "lo-mobile-operator";
const IDB_STORE = "device-key";
const CERT_KEY = "lo-mobile-operator-device-cert";
const ID_KEY = "lo-mobile-operator-device-key-id";

/** This device's signing key, and the public point the operator certifies. */
export interface DeviceKey {
	/** NON-EXTRACTABLE: `exportKey` on this handle throws, which is what makes "the
	    private half never leaves the device" a platform guarantee rather than a
	    promise this file keeps. */
	privateKey: CryptoKey;
	/** The uncompressed P-256 point, 65 bytes — what the operator's certificate
	    covers, and what the runtime verifies a signature against. */
	point: Uint8Array;
}

/** Used only when IndexedDB is unavailable, so the key then lives for the life of
    the page. A browser always has IndexedDB; the fallback exists so this module has
    ONE behaviour to reason about instead of a crash, and so the unit suite —
    happy-dom has no IDB — exercises the production path rather than a mock of it. */
let memoryKey: DeviceKey | null = null;

async function openStore(): Promise<IDBDatabase | null> {
	if (typeof indexedDB === "undefined") return null;
	return await new Promise<IDBDatabase | null>((resolve) => {
		const request = indexedDB.open(IDB_NAME, 1);
		request.onupgradeneeded = () => {
			request.result.createObjectStore(IDB_STORE);
		};
		request.onsuccess = () => resolve(request.result);
		request.onerror = () => resolve(null);
	});
}

async function readStoredKey(): Promise<DeviceKey | null> {
	const db = await openStore();
	if (db === null) return memoryKey;
	return await new Promise<DeviceKey | null>((resolve) => {
		const request = db.transaction(IDB_STORE, "readonly").objectStore(IDB_STORE).get("key");
		request.onsuccess = () => resolve((request.result as DeviceKey | undefined) ?? null);
		request.onerror = () => resolve(null);
	});
}

async function storeKey(key: DeviceKey): Promise<void> {
	memoryKey = key;
	const db = await openStore();
	if (db === null) return;
	await new Promise<void>((resolve) => {
		const tx = db.transaction(IDB_STORE, "readwrite");
		tx.objectStore(IDB_STORE).put(key, "key");
		tx.oncomplete = () => resolve();
		tx.onerror = () => resolve();
	});
}

/**
 * This device's key: the stored one, or a fresh pair.
 *
 * THE ONE-TICK TRANSIENT EXPORT, and why it is the only shape the platform allows.
 * `generateKey` sets `extractable` on BOTH halves at once, so a non-extractable
 * private key necessarily arrives with a non-extractable public one — and the
 * public point is exactly what the operator's certificate must cover, so it has to
 * be readable. The pair is therefore generated extractable, the point is read, and
 * the private half is immediately RE-IMPORTED non-extractable. The handle that is
 * stored is the non-extractable one; the exportable PKCS#8 copy exists only on this
 * function's stack across one `await` and is never written anywhere. What persists
 * on the device cannot be exfiltrated, which is the property a stolen phone's
 * storage would otherwise break.
 */
export async function loadOrCreateDeviceKey(): Promise<DeviceKey> {
	const existing = await readStoredKey();
	if (existing !== null) return existing;

	const generated = (await crypto.subtle.generateKey(
		{ name: "ECDSA", namedCurve: "P-256" },
		true,
		["sign", "verify"],
	)) as CryptoKeyPair;
	/* "raw" on an ECDSA public key IS the uncompressed point, so there is no
	   coordinate assembly here and therefore no way to get the leading 0x04 or the
	   x/y order wrong. */
	const point = new Uint8Array(await crypto.subtle.exportKey("raw", generated.publicKey));
	if (point.length !== 65 || point[0] !== 0x04) {
		throw new Error("this browser did not produce an uncompressed P-256 point");
	}
	const privateKey = await crypto.subtle.importKey(
		"pkcs8",
		await crypto.subtle.exportKey("pkcs8", generated.privateKey),
		{ name: "ECDSA", namedCurve: "P-256" },
		false,
		["sign"],
	);
	const key: DeviceKey = { privateKey, point };
	await storeKey(key);
	return key;
}

/** Unpadded url-safe base64 of a public point — the pairing request's shape. */
export function encodePoint(point: Uint8Array): string {
	let binary = "";
	for (const byte of point) binary += String.fromCharCode(byte);
	return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

/**
 * The device id in the runtime's spelling: the truncated SHA-256 of the point.
 *
 * Derived rather than chosen, in the browser as on the machine, so the two ends
 * agree without a registry — and so a revoked device cannot relabel itself by
 * claiming a different id.
 */
export async function deviceIdFor(point: Uint8Array): Promise<string> {
	const digest = new Uint8Array(
		await crypto.subtle.digest("SHA-256", point as unknown as ArrayBuffer),
	);
	return bytesToHex(digest).slice(0, KEY_ID_HEX_CHARS);
}

// ---------------------------------------------------------------------------
// The certificate: public data, kept in storage the 401 path does not clear
// ---------------------------------------------------------------------------

/** Store the operator-signed certificate this device presents. PUBLIC data — a
    statement plus a signature over a public key — so it lives in `localStorage`
    beside the other non-private preferences, and deliberately NOT in the private
    storage the sign-out path clears: losing it would silently unhook a paired
    phone and the reader would be told "authorise from your paired phone" by copy
    it could no longer satisfy. */
export function storeCertificate(certificate: string, keyId: string): void {
	localStorage.setItem(CERT_KEY, certificate);
	localStorage.setItem(ID_KEY, keyId);
}

export function storedCertificate(): { certificate: string; keyId: string } | null {
	const certificate = localStorage.getItem(CERT_KEY);
	const keyId = localStorage.getItem(ID_KEY);
	if (!certificate || !keyId) return null;
	return { certificate, keyId };
}

export function forgetCertificate(): void {
	localStorage.removeItem(CERT_KEY);
	localStorage.removeItem(ID_KEY);
}

// ---------------------------------------------------------------------------
// The two calls the phone makes
// ---------------------------------------------------------------------------

/** One typed failure the UI can tell apart from a transport problem. */
export class NotPairedError extends Error {
	constructor() {
		super("this phone is not paired with that machine yet");
		this.name = "NotPairedError";
	}
}

/**
 * Sign the runtime's challenge for one action.
 *
 * `null`/`NotPairedError` are supported answers rather than crashes: a phone that
 * has not been paired gets copy naming the pairing screen, which is strictly more
 * useful than a sentence about a signature. The caller renders the runtime's own
 * refusal when nothing here can produce a proof, so the two surfaces cannot
 * disagree about what happened.
 */
export async function signChallenge(input: {
	action: OperatorAction;
	sessionId: string;
	requestId: string;
	challenge: string;
}): Promise<OperatorProof> {
	const stored = storedCertificate();
	if (stored === null) throw new NotPairedError();
	if (input.challenge.length !== CHALLENGE_HEX_CHARS) {
		throw new Error("the runtime sent a challenge this build cannot sign");
	}
	const key = await loadOrCreateDeviceKey();
	const message = signedMessageBytes(input);
	const raw = new Uint8Array(
		await crypto.subtle.sign(
			{ name: "ECDSA", hash: "SHA-256" },
			key.privateKey,
			message as unknown as ArrayBuffer,
		),
	);
	return {
		sig: bytesToHex(rawSignatureToDer(raw)),
		keyId: stored.keyId,
		certificate: stored.certificate,
	};
}

/** The fields an authority-increasing frame carries when this phone signs it. */
export async function operatorFieldsFor(input: {
	action: OperatorAction;
	sessionId: string;
	requestId: string;
	challenge: string;
}): Promise<Record<string, string>> {
	const proof = await signChallenge(input);
	return {
		operator_sig: proof.sig,
		operator_key_id: proof.keyId,
		operator_cert: proof.certificate,
	};
}
