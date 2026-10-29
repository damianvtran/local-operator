// @vitest-environment happy-dom
import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FetchStreamFixture as Stream, settleStream } from "./sse-test-fixture";
import { openEventStream, retainSessionListStream, sseFramer, useSessions } from "./store";

let close: (() => void) | undefined;
beforeEach(() => { vi.useFakeTimers(); Stream.install(); vi.spyOn(Math, "random").mockReturnValue(0.5); });
afterEach(async () => {
	close?.(); close = undefined;
	await vi.advanceTimersByTimeAsync(10001);
	vi.useRealTimers(); vi.restoreAllMocks(); vi.unstubAllGlobals();
});

function connect() {
	const data = vi.fn(), opened = vi.fn(), disconnected = vi.fn();
	close = openEventStream("/events", "projection", data, opened, disconnected);
	return { data, opened, disconnected };
}

describe("fetch SSE transport", () => {
	it("parses split CRLF, multiline data, comments and unknown fields", () => {
		const frames: unknown[] = [];
		const push = sseFramer((event, data) => frames.push({ event, data }));
		for (const char of ": keepalive\r\nevent: projection\r\ndata: one\r\ndata: two\r\nid: ignored\r\n\r\ndata: default\n\n") push(char);
		expect(frames).toEqual([{ event: "projection", data: "one\ntwo" }, { event: "message", data: "default" }]);
	});
	it("rotates mature clean EOF without disconnect or second-long backoff", async () => {
		const hooks = connect(); await settleStream();
		Stream.instances[0].emit("projection", "first"); await settleStream();
		await vi.advanceTimersByTimeAsync(6000);
		Stream.instances[0].end(); await vi.advanceTimersByTimeAsync(250);
		expect(Stream.instances).toHaveLength(2);
		expect(hooks.disconnected).not.toHaveBeenCalled();
		expect(hooks.opened).toHaveBeenCalledTimes(2);
	});
	it("delivers frames in order before batching paints and ignores retired readers", async () => {
		const hooks = connect(); await settleStream();
		Stream.instances[0].emit("projection", "superseded");
		Stream.instances[0].emit("projection", "latest"); await settleStream();
		expect(hooks.data.mock.calls).toEqual([["superseded", true], ["latest", true]]);
		close?.(); Stream.instances[0].emit("projection", "closed"); await settleStream();
		expect(hooks.data).toHaveBeenCalledTimes(2);
	});
	it("detects byte silence and immediately resynchronizes", async () => {
		const hooks = connect(); await settleStream();
		Stream.instances[0].emit("projection", "seed"); await settleStream();
		await vi.advanceTimersByTimeAsync(35001);
		expect(Stream.instances).toHaveLength(2);
		Stream.instances[1].emit("projection", "resynced"); await settleStream();
		expect(hooks.data).toHaveBeenLastCalledWith("resynced", true);
		expect(hooks.disconnected).not.toHaveBeenCalled();
	});
	it("counts keepalive comment bytes as liveness", async () => {
		connect(); await settleStream();
		for (let index = 0; index < 4; index++) {
			await vi.advanceTimersByTimeAsync(15000);
			Stream.instances[0].chunk(": keepalive\n\n"); await settleStream();
		}
		expect(Stream.instances).toHaveLength(1);
	});
	it.each(["online", "pageshow", "visibilitychange"])("reopens a stale stream on %s", async (event) => {
		connect(); await settleStream(); await vi.advanceTimersByTimeAsync(21000);
		(event === "visibilitychange" ? document : window).dispatchEvent(new Event(event));
		await settleStream(); expect(Stream.instances).toHaveLength(2);
	});
	it("does not churn a healthy stream on wake", async () => {
		connect(); await settleStream(); window.dispatchEvent(new Event("online"));
		await settleStream(); expect(Stream.instances).toHaveLength(1);
	});
	it("backs off early EOF, reports sustained outage once, resets after a stable recovery", async () => {
		const hooks = connect(); await settleStream();
		for (let index = 0; index < 5; index++) {
			Stream.instances.at(-1)!.end();
			await vi.advanceTimersByTimeAsync(250 * 2 ** index);
		}
		expect(hooks.disconnected).toHaveBeenCalledTimes(1);
		Stream.instances.at(-1)!.emit("projection", "recovered"); await settleStream();
		await vi.advanceTimersByTimeAsync(2000);
		Stream.instances.at(-1)!.fail(); await vi.advanceTimersByTimeAsync(250);
		expect(hooks.disconnected).toHaveBeenCalledTimes(1);
	});
	it("honours a bounded Retry-After on 503 and wake skips backoff", async () => {
		vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 503, headers: { "Retry-After": "120" } })));
		const hooks = connect(); await vi.advanceTimersByTimeAsync(29999);
		expect(fetch).toHaveBeenCalledTimes(1);
		expect(hooks.disconnected).toHaveBeenCalledTimes(1);
		window.dispatchEvent(new Event("online")); await vi.advanceTimersByTimeAsync(0);
		expect(fetch).toHaveBeenCalledTimes(2);
	});
	it("aborts a fetch whose headers never arrive", async () => {
		const pending = vi.fn((_url, init: RequestInit) => new Promise((_resolve, reject) => init.signal!.addEventListener("abort", () => reject(new DOMException()))));
		vi.stubGlobal("fetch", pending); connect(); await vi.advanceTimersByTimeAsync(35001);
		expect(pending).toHaveBeenCalledTimes(2);
	});
	it("uses the shared 401 rule and never retries", async () => {
		const reload = vi.fn(); vi.stubGlobal("location", { reload });
		localStorage.setItem("lop:draft:test", "private");
		vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(null, { status: 401 })));
		connect(); await vi.advanceTimersByTimeAsync(60000);
		expect(reload).toHaveBeenCalledTimes(1); expect(fetch).toHaveBeenCalledTimes(1);
	});
});

describe("list stream", () => {
	it("counts applied frames, drops malformed frames and survives navigation linger", async () => {
		close = retainSessionListStream(); await settleStream();
		const { result, unmount } = renderHook(() => useSessions());
		const revision = result.current.revision;
		await act(async () => {
			Stream.instances[0].emit("sessions", { sessions: [{ session_id: "a" }] });
			Stream.instances[0].emit("sessions", { sessions: [{ session_id: "b" }] });
			await settleStream();
		});
		expect(result.current.revision).toBe(revision + 2);
		expect(result.current.sessions[0].session_id).toBe("b");
		close(); close = undefined; await vi.advanceTimersByTimeAsync(9000);
		close = retainSessionListStream(); expect(Stream.instances).toHaveLength(1);
		await act(async () => { Stream.instances[0].emit("sessions", "{"); await settleStream(); });
		expect(result.current.revision).toBe(revision + 2); unmount();
	});
});
