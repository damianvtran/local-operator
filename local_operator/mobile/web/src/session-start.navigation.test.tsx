// @vitest-environment happy-dom
// Exercise the rendered form, real HTTP wrapper, router and SSE store together.
// A PID is a process generation, never the identity consumed by session routes.
//
// THE ONE-TAP START IS THE POINT OF THE FIRST CASE. Starting a session used to
// be a route (`#/new`) that asked for a working directory first; the operator's
// ask is that it costs one tap on the list, lands in the conversation, and puts
// the cursor in the composer — with the directory resolved by the daemon rather
// than asked of the user. The three assertions that make that real are all here:
// the POST carries NO `cwd`, the hash lands on the session, and the composer's
// textarea is the active element (the last one matters because it is the whole
// reason `lib/pending-focus.ts` exists — the composer mounts a route later than
// the tap).
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

let startBodies: string[] = [];

beforeEach(() => {
	EventSourceFixture.instances = [];
	startBodies = [];
	vi.stubGlobal("EventSource", EventSourceFixture);
	vi.stubGlobal("fetch", vi.fn(async (input: string, init?: RequestInit) => {
		let body: unknown;
		if (input === "/api/directories") body = { home: "/tmp/fixture", recent: [], default: "/tmp/fixture" };
		else if (input === "/api/models") body = { models: [] };
		else if (input.startsWith("/api/sessions/search?")) body = {
			sessions: [{ id: sessionId, name: "Saved conversation", mtime: 1 }], query: "",
		};
		else if (input === "/api/sessions/start") {
			// Recorded, not asserted-on-later: the body is the contract under test
			// (no `cwd` may be sent), and it is gone once the request object is.
			startBodies.push(String(init?.body ?? ""));
			body = { ok: true, pid, session_id: sessionId };
		} else if (input === "/api/sessions/resume") {
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

/** Publish the projection a mounted session stream is waiting for. */
async function publishProjection(conversationName = "Ready conversation") {
	const stream = await waitFor(() => {
		const source = EventSourceFixture.instances.find((candidate) =>
			candidate.url === `/api/sessions/${sessionId}/events`);
		expect(source).toBeTruthy();
		return source!;
	});
	expect(EventSourceFixture.instances.some((source) =>
		source.url.includes(`/${pid}/`))).toBe(false);
	act(() => {
		stream.onopen?.();
		stream.listeners.get("projection")?.(new MessageEvent("projection", {
			data: JSON.stringify({
				session_id: sessionId, pid, version: 1, conversation_name: conversationName,
				transcript: [], todos: [], subagents: [], usage: {}, effort_ladder: [],
			}),
		}));
	});
	await screen.findByText(conversationName);
}

describe("session creation routes", () => {
	it("one tap on the list starts a session with no cwd and focuses the composer", async () => {
		history.replaceState(null, "", "#/");
		render(<App />);
		const button = await screen.findByRole("button", { name: "new session" });
		fireEvent.click(button);
		await waitFor(() => expect(location.hash).toBe(`#/s/${sessionId}`));
		// NO cwd: the daemon owns the default (GET /api/directories publishes the
		// same answer as `default`), which is what makes the tap a single step.
		expect(startBodies).toEqual(["{}"]);
		await publishProjection();
		// The focus intent crossed the route boundary: the composer mounted after
		// the projection, and the textarea is where the next keystroke lands.
		expect(document.activeElement).toBe(screen.getByPlaceholderText("Message…"));
	});

	it("resuming a past session opens the durable session stream and paints its welcome", async () => {
		history.replaceState(null, "", "#/past");
		render(<App />);
		const button = await screen.findByRole("button", { name: /resume/i });
		await waitFor(() => expect((button as HTMLButtonElement).disabled).toBe(false));
		fireEvent.click(button);
		await waitFor(() => expect(location.hash).toBe(`#/s/${sessionId}`));
		await publishProjection();
		// An ordinary navigation does NOT steal focus — the intent is a one-shot
		// and this route never carried one.
		expect(document.activeElement).not.toBe(screen.getByPlaceholderText("Message…"));
	});

	it("a deep link to an existing session does not autofocus the composer", async () => {
		history.replaceState(null, "", `#/s/${sessionId}`);
		render(<App />);
		await publishProjection("Existing conversation");
		expect(document.activeElement).not.toBe(screen.getByPlaceholderText("Message…"));
	});
});

describe("the retired new-session route", () => {
	it("resolves #/new to the session list", async () => {
		history.replaceState(null, "", "#/new");
		render(<App />);
		// The list's own control, not a directory picker: the hash is kept
		// working for bookmarks and stale links, and it lands somewhere sensible.
		expect(await screen.findByRole("button", { name: "new session" })).toBeTruthy();
		expect(screen.queryByPlaceholderText("or type another path…")).toBeNull();
	});
});
