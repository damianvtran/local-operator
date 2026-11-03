// @vitest-environment happy-dom
//
// A steer (a message sent while a turn runs) can carry images, and the sender's
// bubble must show them like a prompt's. Regression: the `steer` row rendered
// only `entry.text`, so an image-only steer painted as an EMPTY bordered pill
// (the operator's iOS report) and a captioned one lost its thumbnails. The
// backend already upgrades the steer echo with image refs on the drained event
// (projection.absorb_user_event); only the renderer dropped them.
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Transcript } from "./transcript";
import * as api from "../api";
import type { TranscriptEntry } from "../types";

afterEach(cleanup);

const steer = (extra: Partial<TranscriptEntry>): TranscriptEntry => ({
	id: "m1", kind: "steer", text: "", tool_call_id: "", tool_name: "", tool_state: "done",
	summary: "", intent: "", diff_added: 0, diff_removed: 0, elapsed_s: 0, error: "",
	details: {}, final: true, ...extra,
});

describe("steer rows with attachments", () => {
	it("renders thumbnails (and no empty pill text) for an image-only steer", () => {
		const url = vi.spyOn(api, "imageUrl");
		render(<Transcript pid="s" entries={[steer({ images: [{ index: 0, mime_type: "image/jpeg" }, { index: 1, mime_type: "image/png" }] })]} />);
		expect(screen.getAllByAltText("attachment")).toHaveLength(2);
		expect(url).toHaveBeenCalledWith("s", "m1", 1);
	});
	it("keeps the caption beside the thumbnails", () => {
		render(<Transcript pid="s" entries={[steer({ text: "see this", images: [{ index: 0, mime_type: "image/png" }] })]} />);
		expect(screen.getByText("see this")).toBeTruthy();
		expect(screen.getAllByAltText("attachment")).toHaveLength(1);
	});
});

describe("tap to enlarge", () => {
	it("opens a full-screen viewer once loaded and closes on tap", () => {
		render(<Transcript pid="s" entries={[steer({ images: [{ index: 0, mime_type: "image/png" }] })]} />);
		fireEvent.load(screen.getByAltText("attachment"));
		fireEvent.click(screen.getByRole("button", { name: "open image attachment full screen" }));
		expect(screen.getByRole("dialog")).toBeTruthy();
		expect(screen.getByAltText("attachment full size")).toBeTruthy();
		// Portaled to <body> (not clipped by the transcript scroller) and the app is inert behind it.
		expect(document.body.querySelector("[data-attachment-viewer]")!.parentElement).toBe(document.body);
		expect(document.activeElement?.getAttribute("aria-label")).toBe("close");
		fireEvent.click(screen.getByRole("dialog"));
		expect(screen.queryByRole("dialog")).toBeNull();
	});
	it("is not offered while loading, and Escape closes the viewer", () => {
		render(<Transcript pid="s" entries={[steer({ images: [{ index: 0, mime_type: "image/png" }] })]} />);
		expect(screen.queryByRole("button", { name: "open image attachment full screen" })).toBeNull();
		fireEvent.load(screen.getByAltText("attachment"));
		fireEvent.keyDown(screen.getByRole("button", { name: "open image attachment full screen" }), { key: "Enter" });
		expect(screen.getByRole("dialog")).toBeTruthy();
		fireEvent.keyDown(window, { key: "Escape" });
		expect(screen.queryByRole("dialog")).toBeNull();
	});
});
