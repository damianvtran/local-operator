/**
 * The phone's signing path (stage D of issue #1310), pinned against the runtime.
 *
 * WHY THERE ARE CROSS-LANGUAGE VECTORS IN A JAVASCRIPT SUITE. Two facts have to
 * agree byte for byte with Python for a phone signature to be worth anything:
 *
 *   * the SIGNED MESSAGE (`verify.signed_message`), because a drift of one byte in
 *     the domain tag or a length prefix makes every phone signature fail — and it
 *     fails as a refusal, which reads as "you are not authorised" rather than as a
 *     bug;
 *   * the SIGNATURE ENCODING, because `crypto.subtle` returns raw `r||s` while the
 *     runtime parses DER.
 *
 * Both vectors below were produced by the Python side itself (`local_operator/
 * operator/verify.py` + `cryptography`) and are pasted verbatim, so this suite
 * fails the moment either end moves rather than the moment a phone is used.
 *
 * WHAT IS *NOT* TESTED HERE, deliberately: the runtime's VERDICT on a signature.
 * That is Python's, and it is exercised end to end on the real socket in
 * `tests/unit/session/runtime/test_approval_authority_seam.py` — the cell that
 * drives a real relay, a real runtime and a real device key. Duplicating the
 * policy here would give this suite a second, independent idea of what "valid"
 * means.
 */
import { beforeEach, describe, expect, it } from "vitest";
import {
	CHALLENGE_HEX_CHARS,
	NotPairedError,
	bytesToHex,
	derSignatureToRaw,
	deviceIdFor,
	encodePoint,
	forgetCertificate,
	hexToBytes,
	lengthPrefix,
	loadOrCreateDeviceKey,
	operatorFieldsFor,
	rawSignatureToDer,
	signChallenge,
	signedMessageBytes,
	storeCertificate,
} from "./operator-device";

/** From `verify.signed_message(action="loosen", session_id="session-abc",
    request_id="", challenge="ab"*32)`. */
const MESSAGE_HEX =
	"6c6f702d6f70657261746f722d763100000000066c6f6f73656e0000000b73657373696f6e2d616263" +
	"000000000000" +
	"004061626162616261626162616261626162616261626162616261626162616261626162616261626162616261626162616261626162616261626162616261626162";

/** The same message signed by the deterministic P-256 key below, in BOTH forms:
    `cryptography` gives DER, and `decode_dss_signature` splits it into the raw
    `r||s` WebCrypto produces. */
const DER_HEX =
	"3046022100f4e6a724265a0dc5c45066bd997067a41e8e4529445be6467c018bf81e75f20f" +
	"02210094938bfdb47170c94736b9347e5434a39a866d4b2014587ac0a9aabd97e033ef";
const RAW_HEX =
	"f4e6a724265a0dc5c45066bd997067a41e8e4529445be6467c018bf81e75f20f" +
	"94938bfdb47170c94736b9347e5434a39a866d4b2014587ac0a9aabd97e033ef";

const CHALLENGE = "ab".repeat(32);

describe("the signed message", () => {
	it("reproduces the runtime's framing byte for byte", () => {
		const message = signedMessageBytes({
			action: "loosen",
			sessionId: "session-abc",
			requestId: "",
			challenge: CHALLENGE,
		});
		expect(bytesToHex(message)).toBe(MESSAGE_HEX);
	});

	it("length-prefixes big-endian, so no field can be shifted into another", () => {
		expect(Array.from(lengthPrefix("loosen"))).toEqual([0, 0, 0, 6, 108, 111, 111, 115, 101, 110]);
		expect(Array.from(lengthPrefix(""))).toEqual([0, 0, 0, 0]);
		/* The binding is what makes a signature non-transferable: the same
		   challenge under a different ACTION or SESSION is a different message, so a
		   signature harvested for one cannot be presented for the other. */
		const base = { action: "loosen" as const, sessionId: "s", requestId: "", challenge: CHALLENGE };
		const other = signedMessageBytes({ ...base, action: "approve" });
		expect(bytesToHex(other)).not.toBe(bytesToHex(signedMessageBytes(base)));
		expect(bytesToHex(signedMessageBytes({ ...base, sessionId: "s2" }))).not.toBe(
			bytesToHex(signedMessageBytes(base)),
		);
		expect(bytesToHex(signedMessageBytes({ ...base, requestId: "card-1" }))).not.toBe(
			bytesToHex(signedMessageBytes(base)),
		);
	});
});

describe("the DER encoder", () => {
	it("turns WebCrypto's raw r||s into exactly the DER the runtime parses", () => {
		/* THE PIN THAT MAKES A PHONE SIGNATURE USABLE. The runtime's
		   `verify_signature` hands the bytes to `cryptography`'s DER parser, so a raw
		   64-byte signature is refused as though the key were wrong — a failure that
		   misattributes itself. */
		expect(bytesToHex(rawSignatureToDer(hexToBytes(RAW_HEX)))).toBe(DER_HEX);
	});

	it("round-trips, and pads a high-bit integer rather than emitting a negative one", () => {
		const der = rawSignatureToDer(hexToBytes(RAW_HEX));
		expect(bytesToHex(derSignatureToRaw(der))).toBe(RAW_HEX);
		/* `0xf4` has its top bit set, so DER requires a leading zero — without it the
		   integer would read as negative and `cryptography` would reject the
		   signature. Asserted explicitly because it is the one case a cheap
		   implementation gets wrong and a round-trip test cannot see. */
		expect(der[4]).toBe(0x00);
		expect(der[5]).toBe(0xf4);
	});

	it("refuses a signature that is not a P-256 one", () => {
		expect(() => rawSignatureToDer(new Uint8Array(63))).toThrow();
		expect(() => derSignatureToRaw(new Uint8Array([0x31, 0x00]))).toThrow();
	});
});

describe("the device key", () => {
	it("keeps the private half non-extractable, and reads the point instead", async () => {
		const key = await loadOrCreateDeviceKey();
		/* THE PLATFORM IS THE GUARANTEE, not this module's care: `exportKey` on a
		   non-extractable handle throws, so no bug here — and no compromise of the
		   portal's own JavaScript — can read the private half. */
		await expect(crypto.subtle.exportKey("pkcs8", key.privateKey)).rejects.toThrow();
		await expect(crypto.subtle.exportKey("jwk", key.privateKey)).rejects.toThrow();
		expect(key.privateKey.extractable).toBe(false);
		expect(key.point.length).toBe(65);
		expect(key.point[0]).toBe(0x04);
		/* One key per device: a second call returns the SAME handle, because a
		   regenerated key would silently invalidate the certificate the operator
		   already signed. */
		expect((await loadOrCreateDeviceKey()).point).toEqual(key.point);
	});

	it("derives the device id the way the machine does", async () => {
		const key = await loadOrCreateDeviceKey();
		const id = await deviceIdFor(key.point);
		expect(id).toHaveLength(32);
		expect(id).toMatch(/^[0-9a-f]{32}$/);
		/* Derived from the POINT, so a revoked device cannot relabel itself: the
		   machine asks its revocation list about this string and the string is the
		   key. */
		const digest = new Uint8Array(
			await crypto.subtle.digest("SHA-256", key.point as unknown as ArrayBuffer),
		);
		expect(id).toBe(bytesToHex(digest).slice(0, 32));
	});

	it("encodes the public point as unpadded url-safe base64", async () => {
		const key = await loadOrCreateDeviceKey();
		const encoded = encodePoint(key.point);
		expect(encoded).not.toContain("=");
		expect(encoded).not.toContain("+");
		expect(encoded).not.toContain("/");
	});
});

describe("signing a challenge", () => {
	beforeEach(() => {
		forgetCertificate();
	});

	it("is a NO-OP with a typed error when this phone is not paired", async () => {
		await expect(
			signChallenge({
				action: "loosen",
				sessionId: "s",
				requestId: "",
				challenge: CHALLENGE,
			}),
		).rejects.toBeInstanceOf(NotPairedError);
	});

	it("produces a DER signature that verifies against this phone's own point", async () => {
		storeCertificate("operator-signed-certificate", "0123456789abcdef0123456789abcdef");
		const proof = await signChallenge({
			action: "loosen",
			sessionId: "session-abc",
			requestId: "",
			challenge: CHALLENGE,
		});
		expect(proof.certificate).toBe("operator-signed-certificate");
		expect(proof.keyId).toBe("0123456789abcdef0123456789abcdef");
		expect(proof.sig).toMatch(/^[0-9a-f]+$/);
		expect(proof.sig.length).toBeLessThanOrEqual(160);

		/* THE VERIFICATION, with the platform's own verifier, over the SAME message
		   bytes the runtime checks: the DER form is converted back to raw only
		   because `subtle.verify` expects raw — the runtime gets the DER. */
		const key = await loadOrCreateDeviceKey();
		const publicKey = await crypto.subtle.importKey(
			"raw",
			key.point as unknown as ArrayBuffer,
			{ name: "ECDSA", namedCurve: "P-256" },
			true,
			["verify"],
		);
		const message = signedMessageBytes({
			action: "loosen",
			sessionId: "session-abc",
			requestId: "",
			challenge: CHALLENGE,
		});
		const raw = derSignatureToRaw(hexToBytes(proof.sig));
		expect(
			await crypto.subtle.verify(
				{ name: "ECDSA", hash: "SHA-256" },
				publicKey,
				raw as unknown as ArrayBuffer,
				message as unknown as ArrayBuffer,
			),
		).toBe(true);
	});

	it("binds the signature to the action, so a loosening cannot answer a card", async () => {
		storeCertificate("operator-signed-certificate", "0123456789abcdef0123456789abcdef");
		const loosen = await signChallenge({
			action: "loosen",
			sessionId: "session-abc",
			requestId: "",
			challenge: CHALLENGE,
		});
		const approve = await signChallenge({
			action: "approve",
			sessionId: "session-abc",
			requestId: "card-1",
			challenge: CHALLENGE,
		});
		expect(loosen.sig).not.toBe(approve.sig);

		/* And the runtime would refuse the loosening one for an approval: the signed
		   message covers the action it was minted for, so the two are not
		   interchangeable even though they came from one key. */
		const key = await loadOrCreateDeviceKey();
		const publicKey = await crypto.subtle.importKey(
			"raw",
			key.point as unknown as ArrayBuffer,
			{ name: "ECDSA", namedCurve: "P-256" },
			true,
			["verify"],
		);
		const approveMessage = signedMessageBytes({
			action: "approve",
			sessionId: "session-abc",
			requestId: "card-1",
			challenge: CHALLENGE,
		});
		expect(
			await crypto.subtle.verify(
				{ name: "ECDSA", hash: "SHA-256" },
				publicKey,
				derSignatureToRaw(hexToBytes(loosen.sig)) as unknown as ArrayBuffer,
				approveMessage as unknown as ArrayBuffer,
			),
		).toBe(false);
	});

	it("refuses a challenge this build cannot sign rather than signing garbage", async () => {
		storeCertificate("operator-signed-certificate", "0123456789abcdef0123456789abcdef");
		await expect(
			signChallenge({
				action: "loosen",
				sessionId: "s",
				requestId: "",
				challenge: "ab",
			}),
		).rejects.toThrow();
		expect(CHALLENGE_HEX_CHARS).toBe(64);
	});

	it("hands the frame the three fields the relay forwards", async () => {
		storeCertificate("operator-signed-certificate", "0123456789abcdef0123456789abcdef");
		const fields = await operatorFieldsFor({
			action: "approve",
			sessionId: "s",
			requestId: "card-1",
			challenge: CHALLENGE,
		});
		expect(Object.keys(fields).sort()).toEqual([
			"operator_cert",
			"operator_key_id",
			"operator_sig",
		]);
		/* NO `operator_cap`, EVER. That field is the MACHINE's proof material — the
		   relay mints its own when it is the spawner and drops any value arriving in
		   an HTTP body — so a phone that sent one would be presenting a forgery. */
		expect("operator_cap" in fields).toBe(false);
	});
});
