// @vitest-environment happy-dom
//
// A steer (a message sent while a turn runs) can carry images, and the sender's
// bubble must show them like a prompt's. Regression: the `steer` row rendered
// only `entry.text`, so an image-only steer painted as an EMPTY bordered pill
// (the operator's iOS report) and a captioned one lost its thumbnails. The
// backend already upgrades the steer echo with image refs on the drained event
// (projection.absorb_user_event); only the renderer dropped them.
import { cleanup, render, screen } from "@testing-library/react";
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
