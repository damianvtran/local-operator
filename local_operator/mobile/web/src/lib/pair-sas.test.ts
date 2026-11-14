// The SAS must agree with the runtime's `pair_op.sas_code` byte for byte: a
// drift makes every confirm-mode comparison mismatch and every typed code fail.
// Pinned to the SAME vector file the Python suite reads
// (tests/unit/operator/test_pair_sas_vectors.py), so neither side can move alone.
import { describe, expect, it } from "vitest";
import vectors from "../fixtures/pair-sas-vectors.json";
import { base64UrlToBytes, groupSas, SAS_INFO, sasCode, sasCommit } from "./pair-sas";

describe("pairing SAS", () => {
	it("pins the shared label", () => {
		expect(vectors.label).toBe(SAS_INFO);
		expect(vectors.vectors.length).toBeGreaterThan(0);
	});

	for (const [index, vector] of vectors.vectors.entries()) {
		it(`matches vector ${index} (${vector.approval_id})`, async () => {
			const np = base64UrlToBytes(vector.np);
			expect(await sasCommit(np)).toBe(vector.commit);
			const code = await sasCode({
				np,
				nd: base64UrlToBytes(vector.nd),
				devicePoint: base64UrlToBytes(vector.device_spki),
				computerPoint: base64UrlToBytes(vector.operator_spki),
				approvalId: vector.approval_id,
			});
			expect(code).toBe(vector.sas);
			expect(code).toMatch(/^\d{6}$/);
		});
	}

	it("groups as the desktop draws it", () => {
		expect(groupSas("004213")).toBe("004 213");
	});
});
