/** Translation mechanics only: no catalogue, locale inference, storage or fetch.
 * The existing store owns the single snapshot; this runtime reads that explicit
 * context at render time so registration cannot create a second locale truth. */
import { IntlMessageFormat, type PrimitiveType } from "intl-messageformat";
import type { MessageKey, MessageParams, WithParams, WithoutParams } from "./keys.gen";

export interface LocaleSnapshot {
	readonly backendLocale: string | null;
	readonly uiLocale: string;
	readonly direction: "ltr" | "rtl";
	readonly revision: number;
}
export interface RenderedEnvelope {
	text: string;
	locale: string | null;
	direction: "ltr" | "rtl" | null;
}
const RESOLVED = new Set(["en", "fr", "es", "zh-CN", "ru", "vi", "ur", "hi"]);
export const isResolvedLocale = (value: unknown): value is string =>
	typeof value === "string" && RESOLVED.has(value);
export const directionFor = (locale: string): "ltr" | "rtl" => locale === "ur" ? "rtl" : "ltr";

type Catalogue = Readonly<Record<string, string>>;
function validMap(namespace: string, messages: Record<string, string>): boolean {
	return Object.entries(messages).every(([key, value]) => key.startsWith(`${namespace}.`) && typeof value === "string");
}
function equalMap(left: Catalogue | undefined, right: Catalogue): boolean {
	return !!left && Object.keys(left).length === Object.keys(right).length &&
		Object.entries(right).every(([key, value]) => left[key] === value);
}
function primitiveParams(value: unknown): Record<string, PrimitiveType> | null {
	if (value == null) return {};
	if (typeof value !== "object" || Array.isArray(value)) return null;
	const result: Record<string, PrimitiveType> = {};
	for (const [key, item] of Object.entries(value)) {
		if (item != null && !(item instanceof Date) && !["string", "number", "bigint", "boolean", "undefined"].includes(typeof item)) return null;
		Object.defineProperty(result, key, { value: item, enumerable: true });
	}
	return result;
}

export function createMessageRuntime(readContext: () => LocaleSnapshot, changed: () => void) {
	const bundles = new Map<string, Map<string, Catalogue>>();
	const wires = new Map<string, Map<string, Catalogue>>();
	const formats = new Map<string, IntlMessageFormat>();

	function register(target: Map<string, Map<string, Catalogue>>, namespace: string, locale: string, messages: Record<string, string>): boolean {
		if (!isResolvedLocale(locale) || !validMap(namespace, messages)) return false;
		let locales = target.get(namespace);
		if (!locales) { locales = new Map(); target.set(namespace, locales); }
		if (equalMap(locales.get(locale), messages)) return false;
		locales.set(locale, Object.freeze({ ...messages }));
		// Locale tags are a fixed eight-value contract; bound served namespace
		// registries as well so callers cannot grow memory by arbitrary names.
		if (target === wires && target.size > 16) target.delete(target.keys().next().value!);
		changed(); // same-locale content changes must repaint React too
		return true;
	}
	function registerBundle(namespace: string, locale: string, messages: Record<string, string>): boolean {
		return namespace.startsWith("relay.") && register(bundles, namespace, locale, messages);
	}
	function registerWire(namespace: string, locale: string, messages: Record<string, string>): boolean {
		return namespace.startsWith("wire.") && register(wires, namespace, locale, messages);
	}
	function unregisterWire(namespace: string, locale: string): void {
		const locales = wires.get(namespace);
		if (!locales?.delete(locale)) return;
		if (locales.size === 0) wires.delete(namespace);
		changed();
	}
	function uiLocaleFor(backendLocale: string | null): string {
		// No own catalogues in B: English/LTR even if backend preference is fr.
		// A later namespace extraction may register audited bundles. Require the
		// whole English key set so partial maps cannot mislabel fallback copy.
		if (!backendLocale || bundles.size === 0) return "en";
		for (const locales of bundles.values()) {
			const english = locales.get("en");
			const translated = locales.get(backendLocale);
			if (!english || !translated || Object.keys(english).length === 0 || !Object.keys(english).every(key => key in translated)) return "en";
		}
		return backendLocale;
	}
	function sourceFor(code: string): { source: string; locale: string } | null {
		const context = readContext();
		const target = code.startsWith("wire.") ? wires : bundles;
		const requested = code.startsWith("wire.") ? context.backendLocale : context.uiLocale;
		for (const locale of [...new Set([requested, "en"])]) {
			if (!locale) continue;
			for (const locales of target.values()) {
				const sources = locales.get(locale);
				if (sources && Object.hasOwn(sources, code)) return { source: sources[code], locale };
			}
		}
		return null;
	}
	function render(code: string, params: unknown): { text: string; locale: string } | null {
		const selected = sourceFor(code);
		const values = primitiveParams(params);
		if (!selected || !values) return null;
		try {
			const identity = JSON.stringify([selected.locale, selected.source]);
			let format = formats.get(identity);
			if (!format) {
				format = new IntlMessageFormat(selected.source, selected.locale, undefined, { ignoreTag: true });
				formats.set(identity, format);
				if (formats.size > 256) formats.delete(formats.keys().next().value!);
			}
			const text = format.format(values);
			return typeof text === "string" ? { text, locale: selected.locale } : null;
		} catch { return null; } // malformed copy must never break an error path
	}
	function t<K extends WithParams>(key: K, params: MessageParams[K]): string;
	function t(key: WithoutParams): string;
	function t(key: MessageKey, params?: Record<string, unknown>): string {
		return render(key, params)?.text ?? key;
	}
	function renderEnvelope(code: string, params: unknown, text: string): RenderedEnvelope {
		// Dynamic codes are not cast to generated keys. Unknown/malformed data
		// preserves the original wire sentence, with no guessed language tag.
		const rendered = render(code, params);
		return rendered ? { ...rendered, direction: directionFor(rendered.locale) } : { text, locale: null, direction: null };
	}
	return { t, renderEnvelope, registerBundle, registerWire, unregisterWire, uiLocaleFor };
}
