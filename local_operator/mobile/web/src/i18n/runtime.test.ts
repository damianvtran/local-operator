import { describe, expect, it } from "vitest";
import { createMessageRuntime, isResolvedLocale, type LocaleSnapshot } from "./runtime";

type MutableContext = { -readonly [K in keyof LocaleSnapshot]: LocaleSnapshot[K] };

describe("explicit translation context", () => {
	it("has no device inference or implicit catalogue, and preserves unknown envelopes", () => {
		const context: MutableContext = { backendLocale: null, uiLocale: "en", direction: "ltr", revision: 0 };
		const runtime = createMessageRuntime(() => context, () => { context.revision++; });
		expect(runtime.uiLocaleFor("fr")).toBe("en");
		if (false) {
			// @ts-expect-error Unknown generated key.
			runtime.t("relay.unknown.message");
			// @ts-expect-error Required argument is absent.
			runtime.t("wire.exec.error_line");
			// @ts-expect-error String argument cannot be a number.
			runtime.t("wire.exec.error_line", { error: 42 });
			// @ts-expect-error Bare key cannot accept arguments.
			runtime.t("relay.login.page.title", { extra: "not allowed" });
		}
		expect(runtime.renderEnvelope("unknown.code", {}, "Original  raw text")).toEqual({ text: "Original  raw text", locale: null, direction: null });
		for (const invalid of ["auto", "fr-CA", "EN", "", null]) expect(isResolvedLocale(invalid)).toBe(false);
	});
	it("uses English formatting for fallback, served wire ownership, and same-locale revision", () => {
		const context: MutableContext = { backendLocale: "ru", uiLocale: "en", direction: "ltr", revision: 0 };
		const runtime = createMessageRuntime(() => context, () => { context.revision++; });
		runtime.registerWire("wire.exec", "en", { "wire.exec.error_line": "{count, plural, one {# item} other {# items}}" });
		expect(runtime.renderEnvelope("wire.exec.error_line", { count: 21 }, "fallback")).toEqual({ text: "21 items", locale: "en", direction: "ltr" });
		runtime.registerWire("wire.exec", "ru", { "wire.exec.error_line": "Русский {error}" });
		expect(runtime.t("wire.exec.error_line", { error: "fixture" })).toBe("Русский fixture");
		const revision = context.revision;
		runtime.registerWire("wire.exec", "ru", { "wire.exec.error_line": "Другой {error}" });
		expect(context.revision).toBe(revision + 1);
		expect(runtime.t("wire.exec.error_line", { error: "fixture" })).toBe("Другой fixture");
		expect(runtime.registerBundle("wire.exec", "ru", { "wire.exec.error_line": "Not allowed" })).toBe(false);
		expect(runtime.renderEnvelope("wire.exec.error_line", { error: {} }, "Keep exact")).toEqual({ text: "Keep exact", locale: null, direction: null });
	});
	it("keeps own bundles ahead of any wire map, and requires complete own copy for UI metadata", () => {
		const context: MutableContext = { backendLocale: "ur", uiLocale: "en", direction: "ltr", revision: 0 };
		const runtime = createMessageRuntime(() => context, () => { context.revision++; });
		runtime.registerBundle("relay.login", "en", { "relay.login.page.title": "English", "relay.login.submit.label": "Submit" });
		runtime.registerBundle("relay.login", "ur", { "relay.login.page.title": "اردو" });
		expect(runtime.uiLocaleFor("ur")).toBe("en");
		expect(runtime.registerWire("relay.login", "ur", { "relay.login.page.title": "Override" })).toBe(false);
		runtime.registerBundle("relay.login", "ur", { "relay.login.page.title": "اردو", "relay.login.submit.label": "داخل" });
		context.uiLocale = runtime.uiLocaleFor("ur");
		expect(context.uiLocale).toBe("ur");
		expect(runtime.renderEnvelope("relay.login.page.title", {}, "old")).toEqual({ text: "اردو", locale: "ur", direction: "rtl" });
	});
});
