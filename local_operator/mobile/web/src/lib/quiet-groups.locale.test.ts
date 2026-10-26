import { afterEach, describe, expect, it } from "vitest";
import { BUNDLED_OWNED } from "../i18n/bundled.gen";
import { messageRuntime } from "../store";
import { quietGroupLabel } from "./quiet-groups";

afterEach(() => {
	messageRuntime.registerBundle("relay.transcript", "en", { ...BUNDLED_OWNED["relay.transcript"] });
});

describe("localized quiet-group labels", () => {
	it("reads the mixed-family default from the catalogue, not an English literal", () => {
		const marker = "MIXED-LABEL-FROM-CATALOGUE";
		messageRuntime.registerBundle("relay.transcript", "en", {
			...BUNDLED_OWNED["relay.transcript"],
			"relay.transcript.label.messages": marker,
		});
		expect(quietGroupLabel("mixed")).toBe(marker);
	});
});
