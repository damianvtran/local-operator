/**
 * Preserve the first substantive identifying run, not the majority script:
 * counts cannot infer whether a longer bilingual explanation is the identifier.
 * Arabic runs always qualify; Latin runs qualify unless every whitespace token
 * is an uppercase/digit acronym of at most five characters (API SDK). Neutral
 * separators belong to the span but never start a run or contribute letters.
 * With only acronyms, use the first strong script; without strong letters,
 * inherit chrome direction. Call on content leaves, excluding [fork]/chat:.
 * Accepted residual: DeepSeek followed by Urdu keeps the vendor identifier and
 * may clip the Urdu tail; the existing title/long-press exposes the full text.
 */
export function contentDir(text: string): "ltr" | "rtl" | undefined {
	type Direction = "ltr" | "rtl";
	let first: Direction | undefined;
	let run: Direction | undefined;
	let span = "";
	const substantive = () => run === "rtl" || (run === "ltr" &&
		!span.trim().split(/\s+/u).filter((token) => /\p{Script=Latin}/u.test(token))
			.every((token) => /^[A-Z0-9]{1,5}$/u.test(token)));

	for (const char of text) {
		let direction: Direction | undefined;
		if (/\p{Letter}/u.test(char)) {
			if (/[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]/u.test(char)) direction = "rtl";
			else if (/\p{Script=Latin}/u.test(char)) direction = "ltr";
		}
		if (direction) {
			first ??= direction;
			if (run && run !== direction) {
				if (substantive()) return run;
				span = "";
			}
			run = direction;
			span += char;
		} else if (run) {
			span += char;
		}
	}
	return substantive() ? run : first;
}
