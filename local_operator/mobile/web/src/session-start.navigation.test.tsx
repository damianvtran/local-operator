// @vitest-environment happy-dom
// Exercise the rendered form, real HTTP wrapper, router and SSE store together.
// A PID is a process generation, never the identity consumed by session routes.
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./app";

class EventSourceFixture {
	static instances: EventSourceFixture[] = [];
	onopen: (() => void) | null = null;
	onerror: (() => void) | null = null;
	listeners = new Map<string, (event: MessageEvent) => void>();
	constructor(readonly url: string) { EventSourceFixture.instances.push(this); }
	addEventListener(name: string, listener: EventListenerOrEventListenerObject) {
		this.listeners.set(name, listener as (event: MessageEvent) => void);
	}
	close() {}
}

const sessionId = "abcd1234ef56";
const pid = 4242;

beforeEach(() => {
	EventSourceFixture.instances = [];
	vi.stubGlobal("EventSource", EventSourceFixture);
	vi.stubGlobal("fetch", vi.fn(async (input: string) => {
		let body: unknown;
		if (input === "/api/directories") body = { home: "/tmp/fixture", recent: [] };
		else if (input === "/api/models") body = { models: [] };
		else if (input.startsWith("/api/sessions/search?")) body = {
			sessions: [{ id: sessionId, name: "Saved conversation", mtime: 1 }], query: "",
		};
		else if (input === "/api/sessions/start" || input === "/api/sessions/resume") {
			body = { ok: true, pid, session_id: sessionId };
		} else if (input.endsWith("/seen")) body = { ok: true };
		else if (input.includes("/history")) body = { entries: [], has_more: false };
		else throw new Error(`Unexpected request: ${input}`);
		return new Response(JSON.stringify(body), { status: 200 });
	}));
});

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

describe("session creation routes", () => {
	for (const route of ["new", "past"]) {
		it(`${route} opens the durable session stream and paints its welcome`, async () => {
			history.replaceState(null, "", `#/${route}`);
			render(<App />);
			const button = await screen.findByRole("button", {
				name: route === "new" ? "start" : /resume/i,
			});
			await waitFor(() => expect((button as HTMLButtonElement).disabled).toBe(false));
			fireEvent.click(button);
			await waitFor(() => expect(location.hash).toBe(`#/s/${sessionId}`));
			// History changes before React commits the new route's effect. Wait
			// for the actual stream publication, not the preceding router event.
			const stream = await waitFor(() => {
				const source = EventSourceFixture.instances.find((candidate) =>
					candidate.url === `/api/sessions/${sessionId}/events`);
				expect(source).toBeTruthy();
				return source!;
			});
			expect(EventSourceFixture.instances.some((source) =>
				source.url.includes(`/${pid}/`))).toBe(false);
			act(() => {
				stream!.onopen?.();
				stream!.listeners.get("projection")?.(new MessageEvent("projection", {
					data: JSON.stringify({
						session_id: sessionId, pid, version: 1, conversation_name: "Ready conversation",
						transcript: [], todos: [], subagents: [], usage: {}, effort_ladder: [],
					}),
				}));
			});
			expect(await screen.findByText("Ready conversation")).toBeTruthy();
			expect(screen.queryByText("waiting for projection…")).toBeNull();
		});
	}
});
