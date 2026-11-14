// @vitest-environment happy-dom
//
// The store's two temporary-chat transport rules (contract §2.3, §4):
//
//   * the list stream sends `include_temporary=1` ONLY after a frame has
//     advertised `features.temporary_chats`, reopening once to ask, and drops
//     it again if a later frame withdraws the capability (fail closed);
//   * a 404 `session_temporary_deleted` on a session's events stream records a
//     tombstone, drops the cached transcript and STOPS retrying, while any
//     other 404 keeps the ordinary retry policy.
import { act, cleanup, render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { FetchStreamFixture as Stream, settleStream } from "./sse-test-fixture";
import { retainProjectionStream, retainSessionListStream, useTemporaryTombstone } from "./store";

afterEach(async () => {
	cleanup(); await vi.advanceTimersByTimeAsync(10001);
	vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks();
});

const listFrame = (features: Record<string, unknown>) => ({ sessions: [], capabilities: { features } });

it("asks for temporary rows only under the capability, and stops asking when it goes", async () => {
	vi.useFakeTimers(); Stream.install();
	const release = retainSessionListStream(); await settleStream();
	expect(Stream.instances.map((s) => s.url)).toEqual(["/api/sessions/events"]);
	// A frame WITHOUT the capability changes nothing.
	await act(async () => { Stream.instances[0].emit("sessions", listFrame({})); await settleStream(); });
	expect(Stream.instances).toHaveLength(1);
	// The capability arrives: one reopen, with the flag.
	await act(async () => { Stream.instances[0].emit("sessions", listFrame({ temporary_chats: 1 })); await settleStream(); });
	expect(Stream.instances.map((s) => s.url)).toEqual(["/api/sessions/events", "/api/sessions/events?include_temporary=1"]);
	expect(Stream.instances[0].closed).toBe(true);
	// The same capability again does not reopen.
	await act(async () => { Stream.instances[1].emit("sessions", listFrame({ temporary_chats: 1 })); await settleStream(); });
	expect(Stream.instances).toHaveLength(2);
	// Withdrawn (a downgraded relay): back to the plain stream.
	await act(async () => { Stream.instances[1].emit("sessions", listFrame({})); await settleStream(); });
	expect(Stream.instances.map((s) => s.url).at(-1)).toBe("/api/sessions/events");
	release();
});

function Probe({ id }: { id: string }) {
	const tombstone = useTemporaryTombstone(id);
	useEffect(() => retainProjectionStream(id), [id]);
	return <output data-testid="tomb">{tombstone === undefined ? "live" : `deleted:${tombstone}`}</output>;
}

function stubEventsFetch(status: number, body: unknown) {
	const fetch = vi.fn(async (url: string, init?: RequestInit) =>
		url.endsWith("/events") && url.startsWith("/api/sessions/") && url !== "/api/sessions/events"
			? new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } })
			: new Stream(url, init?.signal).response);
	vi.stubGlobal("fetch", fetch);
	return fetch;
}

it("turns a session_temporary_deleted 404 into a tombstone and stops retrying", async () => {
	vi.useFakeTimers(); Stream.install();
	const fetch = stubEventsFetch(404, { code: "session_temporary_deleted", deleted_at: 1_800_000_000 });
	render(<Probe id="gone1" />);
	await act(async () => { await settleStream(); });
	expect(screen.getByTestId("tomb").textContent).toBe("deleted:1800000000");
	const calls = fetch.mock.calls.length;
	await act(async () => { await vi.advanceTimersByTimeAsync(60000); });
	expect(fetch.mock.calls.length).toBe(calls);
});

it("keeps retrying an ordinary 404 and paints no tombstone", async () => {
	vi.useFakeTimers(); Stream.install();
	const fetch = stubEventsFetch(404, { code: "session_not_found" });
	render(<Probe id="missing1" />);
	await act(async () => { await settleStream(); });
	const calls = fetch.mock.calls.length;
	await act(async () => { await vi.advanceTimersByTimeAsync(60000); });
	expect(screen.getByTestId("tomb").textContent).toBe("live");
	expect(fetch.mock.calls.length).toBeGreaterThan(calls);
});
