import { afterEach, describe, expect, it, vi } from "vitest";

import { transcribeAudio } from "./api";

/**
 * The phone sends its locale as a hint, not as an assertion. The daemon
 * normalises it to an ISO-639-1 code and omits anything unmapped (Q20/Q21);
 * this test pins the client half so the hint never silently disappears.
 */
describe("transcribeAudio language hint", () => {
	afterEach(() => {
		vi.unstubAllGlobals();
	});

	it("sends navigator.language in the multipart body", async () => {
		Object.defineProperty(navigator, "language", {
			value: "en-US",
			configurable: true,
		});
		const fetchMock = vi.fn(
			async (_input: RequestInfo | URL, _init?: RequestInit) =>
				new Response(
					JSON.stringify({
						text: "hello",
						provider: "radient",
						model: null,
						path: "provider_stt_radient",
					}),
					{ status: 200, headers: { "content-type": "application/json" } },
				),
		);
		vi.stubGlobal("fetch", fetchMock);

		await transcribeAudio(new Blob(["RIFF"], { type: "audio/webm" }));

		const init = fetchMock.mock.calls[0]?.[1] as RequestInit | undefined;
		const body = init?.body as FormData;
		expect(body.get("language")).toBe("en-US");
	});

	it("names itself and its session so a daemon refusal can be found", async () => {
		const fetchMock = vi.fn(
			async (_input: RequestInfo | URL, _init?: RequestInit) =>
				new Response(
					JSON.stringify({ text: "hello", provider: "radient", model: null, path: "provider_stt_radient" }),
					{ status: 200, headers: { "content-type": "application/json" } },
				),
		);
		vi.stubGlobal("fetch", fetchMock);

		await transcribeAudio(new Blob(["RIFF"], { type: "audio/webm" }), { sessionId: "abcdef123456" });

		const body = (fetchMock.mock.calls[0]?.[1] as RequestInit).body as FormData;
		/* The daemon's log line and the request log's `reason` are the only places a
		   phone report can be joined to a failure; both fields are diagnostic and
		   the daemon re-validates the id. */
		expect(body.get("client")).toBe("web");
		expect(body.get("session_id")).toBe("abcdef123456");
	});

	it("omits an absent session (an older route has none to give)", async () => {
		const fetchMock = vi.fn(
			async (_input: RequestInfo | URL, _init?: RequestInit) =>
				new Response(
					JSON.stringify({ text: "hello", provider: "radient", model: null, path: "provider_stt_radient" }),
					{ status: 200, headers: { "content-type": "application/json" } },
				),
		);
		vi.stubGlobal("fetch", fetchMock);

		await transcribeAudio(new Blob(["RIFF"], { type: "audio/webm" }));

		const body = (fetchMock.mock.calls[0]?.[1] as RequestInit).body as FormData;
		expect(body.get("client")).toBe("web");
		expect(body.get("session_id")).toBeNull();
	});
});
