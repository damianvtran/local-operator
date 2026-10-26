import { describe, expect, it } from "vitest";
import { actionCount, countLabel } from "./condense-copy";
import { BUNDLED_OWNED } from "../i18n/bundled.gen";
import { messageRuntime } from "../store";

describe("condensed action counts", () => {
	it("renders the partial marker instead of silently stating an exact count", () => {
		messageRuntime.registerBundle("relay.transcript", "en", { ...BUNDLED_OWNED["relay.transcript"] });
		expect(actionCount(3, true, countLabel(3, true, "en"))).toBe("3+ actions");
		expect(actionCount(3, false, countLabel(3, false, "en"))).toBe("3 actions");
	});

	it("formats the visible label with locale grouping, not raw digits", () => {
		expect(countLabel(1234, false, "en")).toBe("1,234");
		expect(countLabel(1234, true, "en")).toBe("1,234+");
		expect(countLabel(1234, false, "hi")).toBe(new Intl.NumberFormat("hi").format(1234));
		expect(actionCount(1234, true, countLabel(1234, true, "en"))).toBe("1,234+ actions");
	});
});
