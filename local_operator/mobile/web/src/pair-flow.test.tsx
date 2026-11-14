// @vitest-environment happy-dom
//
// Terminal-free pairing on the phone (design v2 §2.2/§5.2, amendments A1/A2),
// driven through the REAL api.ts transport against a scripted relay: fetch is
// the only seam, so the request bodies, the refusal codes and the HttpError
// extras are exercised exactly as production parses them. The device key and
// the SAS are real WebCrypto, so the six digits on screen are checked against an
// independent recomputation from what the phone actually sent.
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import vectors from "./fixtures/pair-sas-vectors.json";
import { derSignatureToRaw, hexToBytes, storedCertificate } from "./lib/operator-device";
import { base64UrlToBytes, groupSas, sasCode } from "./lib/pair-sas";
import {
	markPairingActionGone,
	openPairingSheet,
	PairingSheetHost,
	PairPromptSheet,
} from "./screens/pair";

const COMPUTER_PUB = vectors.vectors[0].operator_spki;
const ND = vectors.vectors[0].nd;

type Call = { method: string; path: string; body: Record<string, unknown> };
type Reply = { status: number; body: unknown };
let calls: Call[] = [];
let handler: (call: Call) => Reply;

function json(status: number, body: unknown): Reply {
	return { status, body };
}

beforeEach(() => {
	calls = [];
	vi.useFakeTimers({ shouldAdvanceTime: true });
	vi.stubGlobal(
		"fetch",
		vi.fn(async (path: string, init?: RequestInit) => {
			const call: Call = {
				method: init?.method ?? "GET",
				path,
				body: init?.body ? (JSON.parse(String(init.body)) as Record<string, unknown>) : {},
			};
			calls.push(call);
			const reply = handler(call);
			return {
				ok: reply.status >= 200 && reply.status < 300,
				status: reply.status,
				json: async () => reply.body,
			};
		}),
	);
});

afterEach(() => {
	cleanup();
	vi.useRealTimers();
	vi.unstubAllGlobals();
	localStorage.clear();
});

/** A relay that files one request and answers status from a script. */
function relay(options: {
	available?: () => Reply;
	via?: Record<string, string>;
	file?: (call: Call) => Reply | null;
	reasons?: string[];
	withdraw?: () => Reply;
}) {
	const reasons = [...(options.reasons ?? ["waiting"])];
	handler = (call) => {
		if (call.path === "/api/pair/availability") {
			return options.available?.() ?? json(200, { available: true, reason: "ok" });
		}
		if (call.method === "POST" && call.path === "/api/pair/request") {
			if (!("request_id" in call.body)) {
				return json(200, { nd: ND, computer_id_pub: COMPUTER_PUB, anchor: "a", expires_at: 0 });
			}
			const refused = options.file?.(call);
			if (refused) return refused;
			return json(201, {
				approval_id: "appr_test",
				request_id: call.body.request_id,
				state: "requested",
				expires_at: Date.now() / 1000 + 300,
				device: { device_id: "dev-test" },
				what: { pair_phone: { via: options.via ?? { route: "radient", account: "a@b.c", attested: "edge-v1" }, computer_id_pub: COMPUTER_PUB } },
			});
		}
		if (call.path.endsWith("/withdraw")) return options.withdraw?.() ?? json(200, { state: "withdrawn" });
		if (call.path.startsWith("/api/pair/request/")) {
			const reason = reasons.length > 1 ? reasons.shift()! : reasons[0];
			return json(200, { state: "requested", reason, mode: "confirm", device_id: "dev-test" });
		}
		if (call.path === "/api/pair/dev-test") {
			return json(200, { paired: true, device_id: "dev-test", certificate: "CERT", operator_key_id: "KID", authority_ready: true });
		}
		if (call.path.includes("/command")) throw new Error(`unexpected command ${call.path}`);
		return json(404, { error: "no route", code: "not_found" });
	};
}

const filed = () => calls.filter((call) => call.method === "POST" && call.path === "/api/pair/request" && "request_id" in call.body);
const claims = () => calls.filter((call) => call.method === "POST" && call.path === "/api/pair/request" && !("request_id" in call.body));

async function tick(ms: number) {
	await act(async () => {
		await vi.advanceTimersByTimeAsync(ms);
	});
}

function openSheet() {
	render(<PairPromptSheet open onClose={() => undefined} />);
	fireEvent.click(screen.getByText("Pair this phone"));
}

/** The SAS recomputed from what the phone SENT, not from anything it showed. */
async function expectedSas(): Promise<string> {
	const reveal = filed().at(-1)!.body;
	return await sasCode({
		np: base64UrlToBytes(String(reveal.np)),
		nd: base64UrlToBytes(ND),
		devicePoint: base64UrlToBytes(String(reveal.spki)),
		computerPoint: base64UrlToBytes(COMPUTER_PUB),
		approvalId: "appr_test",
	});
}

describe("terminal-free pairing", () => {
	it("confirm mode: commit before reveal, the shared code, confirming, then the certificate", async () => {
		const persist = vi.fn(async () => true);
		vi.stubGlobal("navigator", { ...navigator, storage: { persist } });
		relay({ reasons: ["signing", "paired"] });
		openSheet();
		await screen.findByText("Check that your computer shows the same code, then approve it there.");
		// The claim carries only the commit; np is revealed in the second call.
		expect(claims()[0].body).not.toHaveProperty("np");
		expect(filed()[0].body).toMatchObject({ platform: "web", commit: claims()[0].body.commit });
		expect(String(filed()[0].body.request_id)).toMatch(/^[0-9a-f]{32}$/);
		const code = await expectedSas();
		expect(screen.getByText(groupSas(code)).getAttribute("aria-label")).toBe(
			`Pairing code ${code.split("").join(" ")}`,
		);
		expect(screen.getByText(/^Expires in \d:\d\d$/)).toBeTruthy();

		await tick(2000);
		await screen.findByText("Your computer is confirming — enter your password there.");
		// The countdown hides while the computer is confirming.
		expect(screen.queryByText(/^Expires in/)).toBeNull();

		await tick(2000);
		await screen.findByText("This phone is paired");
		expect(storedCertificate()).toEqual({ certificate: "CERT", keyId: "KID" });
		expect(persist).toHaveBeenCalledTimes(1);
		expect(localStorage.getItem("lo-mobile-pair-request")).toBeNull();
	});

	it("code mode: tells the person to type the code on the computer", async () => {
		relay({ via: { route: "local" } });
		openSheet();
		await screen.findByText("Type this code on your computer when it asks.");
		expect(screen.getByText(groupSas(await expectedSas()))).toBeTruthy();
	});

	it("resumes this phone's own request after a reload instead of filing again", async () => {
		relay({ reasons: ["waiting"] });
		openSheet();
		await screen.findByText("Check that your computer shows the same code, then approve it there.");
		const code = await expectedSas();
		cleanup();
		render(<PairPromptSheet open onClose={() => undefined} />);
		await screen.findByText(groupSas(code));
		expect(filed()).toHaveLength(1);
	});

	it("409 same key: cancels the earlier request with a device signature, then files anew", async () => {
		let busy = true;
		relay({
			file: () =>
				busy ? ((busy = false), json(409, { error: "x", code: "pair_request_pending", same_key: true, request_id: "e".repeat(32) })) : null,
		});
		openSheet();
		await screen.findByText("This phone already asked to pair.");
		// Nothing saved for that id on this phone, so it cannot show the code: no resume.
		expect(screen.queryByText("Continue that request")).toBeNull();
		fireEvent.click(screen.getByText("Cancel the earlier request"));
		await screen.findByText("Check that your computer shows the same code, then approve it there.");
		const withdraw = calls.find((call) => call.path.endsWith("/withdraw"))!;
		expect(withdraw.path).toBe(`/api/pair/request/${"e".repeat(32)}/withdraw`);
		// The signature verifies under the key the phone filed with.
		const point = base64UrlToBytes(String(filed().at(-1)!.body.spki));
		const publicKey = await crypto.subtle.importKey("raw", point as unknown as ArrayBuffer, { name: "ECDSA", namedCurve: "P-256" }, false, ["verify"]);
		const ok = await crypto.subtle.verify(
			{ name: "ECDSA", hash: "SHA-256" },
			publicKey,
			derSignatureToRaw(hexToBytes(String(withdraw.body.sig))) as unknown as ArrayBuffer,
			new TextEncoder().encode(`lop-pair-withdraw-v1${"e".repeat(32)}`) as unknown as ArrayBuffer,
		);
		expect(ok).toBe(true);
	});

	it("409 another key: says another phone is asking", async () => {
		relay({ file: () => json(409, { error: "x", code: "pair_request_pending", same_key: false }) });
		openSheet();
		await screen.findByText(/Another phone is asking to pair with your computer right now/);
	});

	it("429: counts down in m:ss and holds Try again until it reaches zero", async () => {
		relay({ file: () => json(429, { error: "x", code: "pair_filing_rate_limited", retry_after_s: 90 }) });
		openSheet();
		await screen.findByText("Too many pairing requests. Try again in 1:30.");
		expect((screen.getByText("Try again") as HTMLButtonElement).disabled).toBe(true);
		await tick(31_000);
		await screen.findByText(/Try again in 0:5\d\./);
		await tick(60_000);
		await waitFor(() => expect((screen.getByText("Try again") as HTMLButtonElement).disabled).toBe(false));
	});

	it("429 without retry_after_s: no invented countdown, Try again stays available", async () => {
		relay({ file: () => json(429, { error: "x", code: "pair_filing_rate_limited" }) });
		openSheet();
		await screen.findByText("Too many pairing requests. Wait a little, then try again.");
		expect(screen.queryByText(/\d:\d\d/)).toBeNull();
		expect((screen.getByText("Try again") as HTMLButtonElement).disabled).toBe(false);
	});

	it("Cancel withdraws; while the computer is confirming it stays put", async () => {
		let inProgress = true;
		relay({
			withdraw: () =>
				inProgress ? ((inProgress = false), json(409, { error: "x", code: "approval_in_progress" })) : json(200, { state: "withdrawn" }),
		});
		openSheet();
		await screen.findByText("Cancel");
		fireEvent.click(screen.getByText("Cancel"));
		await screen.findByText("Your computer is already confirming. Finish or cancel it there.");
		fireEvent.click(screen.getByText("Cancel"));
		await screen.findByText("Request cancelled.");
		expect(localStorage.getItem("lo-mobile-pair-request")).toBeNull();
	});

	it("U15: a revoked phone discards its key and pairs again under a fresh one", async () => {
		let revoked = true;
		relay({ file: () => (revoked ? ((revoked = false), json(422, { error: "x", code: "pair_device_revoked" })) : null) });
		localStorage.setItem("lo-mobile-operator-device-cert", "OLD");
		localStorage.setItem("lo-mobile-operator-device-key-id", "OLDKID");
		openSheet();
		await screen.findByText("This phone was removed from your computer. Ask again to pair it.");
		expect(storedCertificate()).toBeNull();
		const oldSpki = filed()[0].body.spki;
		fireEvent.click(screen.getByText("Pair again"));
		await screen.findByText("Check that your computer shows the same code, then approve it there.");
		expect(filed().at(-1)!.body.spki).not.toBe(oldSpki);
	});

	it("not available: names Settings → Phone access, re-checks every 5 s, then files by itself", async () => {
		let available = false;
		relay({ available: () => json(200, { available, reason: available ? "ok" : "no_operator_authority" }) });
		openSheet();
		const said = await screen.findByText(/Phone approvals aren’t turned on for your computer yet/);
		expect(said.textContent).toContain("Settings → Phone access");
		await tick(5000);
		expect(claims()).toHaveLength(0);
		available = true;
		await tick(5000);
		await screen.findByText("Check that your computer shows the same code, then approve it there.");
		expect(filed()).toHaveLength(1);
	});

	it("a pre-C2 computer (404) is not offered the flow", async () => {
		relay({ available: () => json(404, { error: "no route" }) });
		openSheet();
		await screen.findByText(/too old to pair phones from here/);
		expect(claims()).toHaveLength(0);
	});
});

describe("return to the action (D15/U14)", () => {
	it("opens titled for the approval, and after pairing asks for a fresh tap — nothing is sent", async () => {
		relay({ reasons: ["paired"] });
		render(<PairingSheetHost />);
		act(() => openPairingSheet({ purpose: "approve", actionId: "req-1" }));
		expect(screen.getByText("Pair this phone to approve from here")).toBeTruthy();
		fireEvent.click(screen.getByText("Pair this phone"));
		await screen.findByText("Check that your computer shows the same code, then approve it there.");
		await tick(2000);
		await screen.findByText("Paired. Tap Approve again to finish — nothing has been approved yet.");
		expect(calls.some((call) => call.path.includes("/command"))).toBe(false);
		fireEvent.click(screen.getByText("Back to the request"));
		await waitFor(() => expect(screen.queryByText("This phone is paired")).toBeNull());
	});

	it("says so when the action was handled or expired meanwhile", async () => {
		relay({ reasons: ["waiting", "paired"] });
		render(<PairingSheetHost />);
		act(() => openPairingSheet({ purpose: "approve", actionId: "req-2" }));
		fireEvent.click(screen.getByText("Pair this phone"));
		await screen.findByText("Check that your computer shows the same code, then approve it there.");
		act(() => markPairingActionGone("req-2"));
		await tick(4000);
		await screen.findByText(
			"That request was answered or expired while you were pairing. This phone can approve the next one.",
		);
		expect(screen.queryByText("Back to the request")).toBeNull();
		expect(screen.getByText("Done")).toBeTruthy();
	});

	it("U16: a certificate whose key was evicted reads as lost and offers pairing again", async () => {
		const { reconcileDeviceKey } = await import("./lib/operator-device");
		localStorage.setItem("lo-mobile-operator-device-cert", "CERT");
		localStorage.setItem("lo-mobile-operator-device-key-id", "KID");
		// happy-dom has no IndexedDB and this module's memory key is unset: no key.
		vi.resetModules();
		const fresh = await import("./lib/operator-device");
		expect(await fresh.reconcileDeviceKey()).toBe(true);
		expect(fresh.storedCertificate()).toBeNull();
		expect(fresh.deviceKeyWasLost()).toBe(true);
		expect(reconcileDeviceKey).toBeTypeOf("function");
		relay({});
		render(<PairPromptSheet open onClose={() => undefined} />);
		expect(screen.getByText(/pairing was lost when the browser cleared its storage/)).toBeTruthy();
	});
});
