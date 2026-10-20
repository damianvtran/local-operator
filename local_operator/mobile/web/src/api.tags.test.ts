/**
 * The conversation-tag routes, at the transport boundary.
 *
 * The relay's contract is exact and asymmetric in a way that is easy to get
 * wrong: `PUT /api/sessions/{id}/tags` takes `{tags, primary}` and NOTHING
 * else — `primary` is always present (an omitted key is a 422, never a silent
 * re-primary) — and the answer is the store's canonical read-back. These cases
 * pin the method, path, and the byte-exact body the daemon validates, plus the
 * refusal shape the sheet renders.
 */
import { afterEach, describe, expect, it, vi } from "vitest";

import { getTags, HttpError, setSessionTags } from "./api";

function answer(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), {
		status,
		headers: { "content-type": "application/json" },
	});
}

function stub(body: unknown, status = 200) {
	const fetchMock = vi.fn(async () => answer(body, status));
	vi.stubGlobal("fetch", fetchMock);
	return fetchMock;
}

function call(fetchMock: ReturnType<typeof stub>, index = 0): [string, RequestInit] {
	const [path, init] = fetchMock.mock.calls[index] as unknown as [string, RequestInit];
	return [path, init];
}

afterEach(() => {
	vi.unstubAllGlobals();
});

describe("getTags", () => {
	it("reads the registry from /api/tags", async () => {
		const fetchMock = stub({
			tags: [{ id: "work", name: "Work", order: 0, conversations: 2 }],
		});
		const result = await getTags();
		expect(call(fetchMock)[0]).toBe("/api/tags");
		expect(result.tags[0]).toEqual({
			id: "work",
			name: "Work",
			order: 0,
			conversations: 2,
		});
	});
});

describe("setSessionTags", () => {
	it("PUTs the exact desired-state body, with primary present even when null", async () => {
		const fetchMock = stub({ session_id: "s1", tags: [], primary: null });
		await setSessionTags("s1", ["work", "research"], null);
		const [path, init] = call(fetchMock);
		expect(path).toBe("/api/sessions/s1/tags");
		expect(init.method).toBe("PUT");
		/* BYTE-EXACT: an extra key or a missing `primary` is a 422 at the relay,
		   and `JSON.stringify` preserves this key order. */
		expect(init.body).toBe('{"tags":["work","research"],"primary":null}');
		expect((init.headers as Record<string, string>)["content-type"]).toBe(
			"application/json",
		);
	});

	it("sends the clear as the empty desired state", async () => {
		const fetchMock = stub({ session_id: "s1", tags: [], primary: null });
		await setSessionTags("s1", [], null);
		expect(call(fetchMock)[1].body).toBe('{"tags":[],"primary":null}');
	});

	it("carries the primary id when one is chosen", async () => {
		const fetchMock = stub({
			session_id: "s1",
			tags: [{ id: "work", name: "Work" }],
			primary: "work",
		});
		const result = await setSessionTags("s1", ["work"], "work");
		expect(call(fetchMock)[1].body).toBe('{"tags":["work"],"primary":"work"}');
		expect(result.primary).toBe("work");
	});

	it("encodes the session id in the path", async () => {
		const fetchMock = stub({ session_id: "a/b", tags: [], primary: null });
		await setSessionTags("a/b", [], null);
		expect(call(fetchMock)[0]).toBe("/api/sessions/a%2Fb/tags");
	});

	it("keeps the relay's own refusal sentence and status", async () => {
		stub({ error: "Unknown category id" }, 422);
		await expect(setSessionTags("s1", ["gone"], null)).rejects.toMatchObject({
			name: "HttpError",
			status: 422,
			message: "Unknown category id",
		});
	});

	it("surfaces a 404 for an unknown or remote session", async () => {
		stub({ error: "unknown session" }, 404);
		const refusal = await setSessionTags("ffffffffffff", [], null).catch(
			(error: unknown) => error,
		);
		expect(refusal).toBeInstanceOf(HttpError);
		expect((refusal as HttpError).status).toBe(404);
		expect((refusal as HttpError).message).toBe("unknown session");
	});
});
