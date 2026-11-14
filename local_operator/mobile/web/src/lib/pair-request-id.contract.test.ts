// The pairing filing's request id crosses two codebases: the PWA mints it
// (`newRequestId`) and the computer refuses anything that is not
// `approvals.new_request_id`'s shape. They drifted once (PR #3949 QA Q48-1: 32
// bare hex chars, refused 422 on every terminal-free pairing while both suites
// stayed green, because each side tested its own idea of the shape). Both sides
// now pin the SAME fixture; the Python half is in tests/unit/mobile/
// test_pairing_e2e.py, which files these exact ids through the real relay.
import { describe, expect, it } from "vitest";
import contract from "../fixtures/pair-request-id.contract.json";
import { newRequestId, requestIdFromBytes } from "./pair-flow";

const pattern = new RegExp(contract.pattern);

function hexToBytes(hex: string): Uint8Array {
	return Uint8Array.from(hex.match(/../g) ?? [], (pair) => Number.parseInt(pair, 16));
}

describe("pairing request id contract (shared with the computer)", () => {
	for (const sample of contract.samples) {
		it(`the PWA's generator turns ${sample.bytes_hex} into ${sample.request_id}`, () => {
			expect(requestIdFromBytes(hexToBytes(sample.bytes_hex))).toBe(sample.request_id);
			expect(sample.request_id).toMatch(pattern);
		});
	}

	it("every freshly minted id has the shape the computer accepts", () => {
		const seen = new Set<string>();
		for (let index = 0; index < 200; index += 1) {
			const id = newRequestId();
			expect(id).toMatch(pattern);
			seen.add(id);
		}
		// Random, not a constant that happens to match.
		expect(seen.size).toBe(200);
	});

	it("the shape the computer refused (Q48-1) does not pass the pattern", () => {
		expect(contract.refused.length).toBeGreaterThan(0);
		for (const refused of contract.refused) expect(refused).not.toMatch(pattern);
	});
});
