// @vitest-environment happy-dom
import { act, cleanup, render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { retainProjectionStream, useProjection } from "./store";

class Source {
	static instances: Source[] = [];
	onopen: (() => void) | null = null;
	onerror: (() => void) | null = null;
	listeners = new Map<string, (event: MessageEvent) => void>();
	closed = false;
	constructor(readonly url: string) { Source.instances.push(this); }
	addEventListener(name: string, listener: EventListenerOrEventListenerObject) {
		this.listeners.set(name, listener as (event: MessageEvent) => void);
	}
	close() { this.closed = true; }
	emit(version: number, token: string, kind: string, sessionId = "mounted") {
		this.listeners.get("projection")?.(new MessageEvent("projection", { data: JSON.stringify({
			session_id: sessionId, version, pid: 42,
			transcript: [{ id: token, kind: "assistant", text: token, final: true, text_complete: true }],
			subagents: [], todos: [],
			attention: { conversation_id: `session/${sessionId}`, completion_token: token,
				anchor_id: token, kind, unseen: true, revision: [version, 0] },
		}) }));
	}
}

function Probe() {
	const { projection, connected } = useProjection("mounted");
	useEffect(() => retainProjectionStream("mounted"), []);
	return <output data-testid="state">{connected ? "ready" : "loading"}:{projection?.attention?.kind}:{projection?.attention?.completion_token}:{projection?.transcript[0]?.text}</output>;
}

afterEach(() => {
	cleanup();
	vi.useRealTimers();
	vi.unstubAllGlobals();
	Source.instances = [];
});

it("accepts fresh completion/error/interruption snapshots without remounting or relaxing within-source ordering", () => {
	vi.useFakeTimers();
	vi.stubGlobal("EventSource", Source);
	render(<Probe />);
	const first = Source.instances[0];
	act(() => { first.onopen?.(); first.emit(100, "A", "complete"); });
	expect(screen.getByTestId("state").textContent).toBe("ready:complete:A:A");
	act(() => { first.onerror?.(); vi.advanceTimersByTime(1000); });
	const second = Source.instances[1];
	act(() => { second.onopen?.(); });
	// Reopening transport must not make its stale old transcript acknowledgeable.
	expect(screen.getByTestId("state").textContent).toBe("loading:complete:A:A");
	act(() => {
		first.emit(999, "retired", "complete");
		first.onopen?.(); first.onerror?.();
		second.emit(500, "foreign", "complete", "another-session");
		second.emit(1, "B", "error");
	});
	expect(second.closed).toBe(false);
	expect(screen.getByTestId("state").textContent).toBe("ready:error:B:B");
	act(() => { second.emit(0, "stale", "complete"); });
	expect(screen.getByTestId("state").textContent).toBe("ready:error:B:B");
	act(() => { second.onerror?.(); vi.advanceTimersByTime(1000); });
	const third = Source.instances[2];
	act(() => { third.onopen?.(); third.emit(0, "C", "interrupted"); second.emit(999, "late", "complete"); });
	expect(screen.getByTestId("state").textContent).toBe("ready:interrupted:C:C");
});
