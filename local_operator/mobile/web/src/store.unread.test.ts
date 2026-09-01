// @vitest-environment happy-dom
//
// Tab-title aggregate (spec §4) and the optimistic half of the seen
// handshake. The title is the one attention signal visible from the phone's
// app switcher, so it must track the list store on every emit: n = sessions
// with unseen || needs_attention.
import { afterEach, describe, expect, it, vi } from "vitest";
import { clearSessionUnseen, retainSessionListStream } from "./store";
import type { SessionSummary } from "./types";

class FakeEventSource {
	static instances: FakeEventSource[] = [];
	onopen: (() => void) | null = null;
	onerror: (() => void) | null = null;
	listeners = new Map<string, (event: MessageEvent) => void>();
	closed = false;
	constructor(readonly url: string) {
		FakeEventSource.instances.push(this);
	}
	addEventListener(name: string, listener: EventListenerOrEventListenerObject) {
		this.listeners.set(name, listener as (event: MessageEvent) => void);
	}
	close() {
		this.closed = true;
	}
}

function summary(over: Partial<SessionSummary>): SessionSummary {
	return {
		session_id: "s",
		section: "active",
		conversation_name: "Session",
		cwd: "",
		model_label: "",
		streaming: false,
		needs_attention: false,
		unseen: false,
		pending_kind: "",
		subagents_running: 0,
		todos_open: 0,
		mtime: 0,
		...over,
	};
}

let release: (() => void) | null = null;

function emitSessions(rows: SessionSummary[]) {
	const es = FakeEventSource.instances[0];
	const listener = es.listeners.get("sessions")!;
	listener({ data: JSON.stringify({ sessions: rows }) } as MessageEvent);
}

afterEach(() => {
	release?.();
	release = null;
	FakeEventSource.instances = [];
});

describe("tab title aggregate", () => {
	it("counts unseen and needs-decision sessions on every emit", () => {
		vi.stubGlobal("EventSource", FakeEventSource);
		release = retainSessionListStream();
		emitSessions([
			summary({ session_id: "a", unseen: true }),
			summary({ session_id: "b", needs_attention: true, pending_kind: "approval" }),
			summary({ session_id: "c" }),
		]);
		expect(document.title).toBe("(2) local operator");

		/* A session that is BOTH unseen and blocked counts once. */
		emitSessions([
			summary({ session_id: "a", unseen: true, needs_attention: true, pending_kind: "ask" }),
		]);
		expect(document.title).toBe("(1) local operator");

		emitSessions([]);
		expect(document.title).toBe("local operator");
	});

	it("clears the mark optimistically when the session is opened", () => {
		vi.stubGlobal("EventSource", FakeEventSource);
		release = retainSessionListStream();
		emitSessions([summary({ session_id: "a", unseen: true })]);
		expect(document.title).toBe("(1) local operator");

		clearSessionUnseen("a");
		expect(document.title).toBe("local operator");

		/* Clearing a session that is not unseen is a no-op: no redundant
		   store emission, title unchanged. */
		clearSessionUnseen("a");
		expect(document.title).toBe("local operator");
	});
});
