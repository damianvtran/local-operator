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
});
