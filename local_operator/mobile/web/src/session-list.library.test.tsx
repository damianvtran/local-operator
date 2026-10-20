// @vitest-environment happy-dom
//
// The Library ENTRY POINT: the footer offers it only when the relay advertises
// `capabilities.library`, and tapping it lands on `#/library`. A relay that
// predates the module omits the key, so absence must render no control — not a
// dead one that opens a screen whose every request would 404.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Capabilities, SessionSummary } from "./types";

let capability: Capabilities | null = null;

vi.mock("./store", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./store")>();
	return {
		...actual,
		useSessions: () => ({ sessions: [] as SessionSummary[], connected: true, revision: 0 }),
		useCapabilities: () => capability,
		retainSessionListStream: () => () => {},
	};
});

vi.mock("./api", async (importOriginal) => {
	const actual = await importOriginal<typeof import("./api")>();
	return {
		...actual,
		getDirectories: vi.fn(async () => ({ home: "", recent: [] })),
		getProjects: vi.fn(async () => ({ projects: [] })),
	};
});

import { SessionListScreen } from "./screens/session-list";

beforeEach(() => {
	capability = null;
});

afterEach(() => {
	cleanup();
	vi.clearAllMocks();
	history.replaceState(null, "", "#/");
});

describe("library footer entry", () => {
	it("is absent when the relay does not advertise the capability", () => {
		render(<SessionListScreen />);
		expect(screen.queryByRole("button", { name: "library" })).toBeNull();
	});

	it("is present and navigates to the list when the capability is advertised", () => {
		capability = { library: 1 };
		render(<SessionListScreen />);
		fireEvent.click(screen.getByRole("button", { name: "library" }));
		expect(location.hash).toBe("#/library");
	});
});
