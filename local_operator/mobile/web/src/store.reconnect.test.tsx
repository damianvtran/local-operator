// @vitest-environment happy-dom
import { act, cleanup, render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { FetchStreamFixture as Stream, settleStream } from "./sse-test-fixture";
import { retainProjectionStream, useProjection } from "./store";

function emit(source: Stream, version: number, token: string, kind: string, sessionId = "mounted") {
	source.emit("projection", {
		session_id: sessionId, version, pid: 42,
		transcript: [{ id: token, kind: "assistant", text: token, final: true, text_complete: true }],
		subagents: [], todos: [],
		attention: { conversation_id: `session/${sessionId}`, completion_token: token,
			anchor_id: token, kind, unseen: true, revision: [version, 0] },
	});
}
function Probe() {
	const { projection, connected } = useProjection("mounted");
	useEffect(() => retainProjectionStream("mounted"), []);
	return <output data-testid="state">{connected ? "ready" : "loading"}:{projection?.attention?.kind}:{projection?.attention?.completion_token}:{projection?.transcript[0]?.text}</output>;
}

afterEach(async () => {
	cleanup(); await vi.advanceTimersByTimeAsync(10001);
	vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks();
});

it("retains the last good snapshot over rotation, resets the source fence and rejects stale/foreign frames", async () => {
	vi.useFakeTimers(); Stream.install(); render(<Probe />); await settleStream();
	const first = Stream.instances[0];
	await act(async () => { emit(first, 100, "A", "complete"); await settleStream(); });
	expect(screen.getByTestId("state").textContent).toBe("ready:complete:A:A");
	await act(async () => { await vi.advanceTimersByTimeAsync(6000); first.end(); await vi.advanceTimersByTimeAsync(250); });
	const second = Stream.instances[1];
	// Rotation alone never turns the retained view into a reconnecting strip.
	expect(screen.getByTestId("state").textContent).toBe("ready:complete:A:A");
	await act(async () => { emit(second, 500, "foreign", "complete", "another-session"); await settleStream(); });
	await act(async () => { emit(second, 1, "B", "error"); await settleStream(); });
	expect(screen.getByTestId("state").textContent).toBe("ready:error:B:B");
	await act(async () => { emit(second, 0, "stale", "complete"); await settleStream(); });
	expect(screen.getByTestId("state").textContent).toBe("ready:error:B:B");
});

it("lingers a projection stream for navigation but eventually releases its retained view", async () => {
	vi.useFakeTimers(); Stream.install();
	const release = retainProjectionStream("linger"); await settleStream();
	release(); await vi.advanceTimersByTimeAsync(4000);
	const again = retainProjectionStream("linger");
	expect(Stream.instances).toHaveLength(1); expect(Stream.instances[0].closed).toBe(false);
	again(); await vi.advanceTimersByTimeAsync(5001);
	expect(Stream.instances[0].closed).toBe(true);
});
