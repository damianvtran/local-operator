// @vitest-environment happy-dom
import { act, cleanup, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { FetchStreamFixture as Stream, settleStream } from "./sse-test-fixture";
import { retainProjectionStream, retainSessionListStream, useProjection, useSessions } from "./store";

let releases: (() => void)[] = [];
beforeEach(() => {
	vi.useFakeTimers(); Stream.install();
	vi.spyOn(Math, "random").mockReturnValue(0.5);
});
afterEach(async () => {
	cleanup(); releases.forEach(release => release()); releases = [];
	await vi.advanceTimersByTimeAsync(10001);
	vi.useRealTimers(); vi.restoreAllMocks(); vi.unstubAllGlobals();
});

const projection = (version: number, text: string) => ({
	session_id: "ordering", pid: 42, version, streaming: false,
	transcript: [{ id: "answer", kind: "assistant", text, final: true, text_complete: true }],
	subagents: [], todos: [],
});

it("C1-1/Q1-1: seed plus short EOF backs off and stays disconnected until a stable recovery", async () => {
	const times: number[] = [];
	let failing = true;
	vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
		times.push(Date.now());
		const stream = new Stream(url, init?.signal);
		stream.emit("sessions", { sessions: [{ session_id: "retained" }] });
		if (failing) stream.end();
		return stream.response;
	}));
	const view = renderHook(() => useSessions());
	await act(async () => { releases.push(retainSessionListStream()); await vi.advanceTimersByTimeAsync(3000); });
	expect(times).toHaveLength(5); // old head: 17 requests, all 187.5ms apart
	expect(times.slice(1).map((at, index) => at - times[index])).toEqual([187, 375, 750, 1500]);
	expect(view.result.current.connected).toBe(false);
	expect(view.result.current.sessions[0].session_id).toBe("retained");
	// The next seed alone is not recovery. A continuously open leg is.
	failing = false;
	await act(async () => { window.dispatchEvent(new Event("online")); await settleStream(); });
	expect(view.result.current.connected).toBe(false);
	await act(async () => { await vi.advanceTimersByTimeAsync(2100); });
	expect(view.result.current.connected).toBe(true);
	// A second outage must report again after the first one genuinely recovered.
	await act(async () => { Stream.instances.at(-1)!.fail(); failing = true; await vi.advanceTimersByTimeAsync(3100); });
	expect(view.result.current.connected).toBe(false);
});

it("C1-2: same-paint descending snapshots retain the terminal winner and reset only on a new source", async () => {
	const view = renderHook(() => useProjection("ordering"));
	await act(async () => { releases.push(retainProjectionStream("ordering")); await settleStream(); });
	await act(async () => {
		Stream.instances[0].emit("projection", projection(9, "complete"));
		Stream.instances[0].emit("projection", projection(8, "partial"));
		await settleStream();
	});
	expect(view.result.current.projection?.version).toBe(9);
	expect(view.result.current.projection?.transcript[0].text).toBe("complete");
	await act(async () => {
		await vi.advanceTimersByTimeAsync(6000); Stream.instances[0].end();
		await vi.advanceTimersByTimeAsync(250);
		Stream.instances[1].emit("projection", projection(1, "new epoch"));
		await settleStream();
	});
	expect(view.result.current.projection?.version).toBe(1);
	expect(view.result.current.connected).toBe(true);
});

it("C1-3: empty/malformed reconnects cannot certify a retained list as connected", async () => {
	const view = renderHook(() => useSessions());
	await act(async () => {
		releases.push(retainSessionListStream()); await settleStream();
		Stream.instances[0].emit("sessions", { sessions: [{ session_id: "good" }] }); await settleStream();
	});
	expect(view.result.current.connected).toBe(true);
	vi.stubGlobal("fetch", vi.fn(async () => new Response("", { headers: { "content-type": "text/event-stream" } })));
	await act(async () => { Stream.instances[0].fail(); await vi.advanceTimersByTimeAsync(10000); });
	expect(view.result.current.connected).toBe(false);
	expect(view.result.current.sessions[0].session_id).toBe("good");
	Stream.install();
	await act(async () => {
		window.dispatchEvent(new Event("online")); await settleStream();
		Stream.instances[0].emit("sessions", { sessions: null });
		await vi.advanceTimersByTimeAsync(2100);
	});
	expect(view.result.current.connected).toBe(false);
	expect(view.result.current.sessions[0].session_id).toBe("good");
	await act(async () => { Stream.instances[0].emit("sessions", { sessions: [] }); await settleStream(); });
	expect(view.result.current.connected).toBe(true);
});

it.each(["online", "pageshow", "visibilitychange"])("C1-4: recovered retry does not churn on %s", async event => {
	releases.push(retainProjectionStream("ordering")); await settleStream();
	Stream.instances[0].fail(); await vi.advanceTimersByTimeAsync(250);
	Stream.instances[1].emit("projection", projection(1, "fresh")); await settleStream();
	(event === "visibilitychange" ? document : window).dispatchEvent(new Event(event));
	await settleStream(); expect(Stream.instances).toHaveLength(2);
});
