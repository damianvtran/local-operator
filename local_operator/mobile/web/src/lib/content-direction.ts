/**
 * Truncated user content follows its dominant script, not its first letter.
 * First-strong (`dir="auto"`) makes acronym-leading Urdu titles lose their
 * identifying beginning; generated Latin prefixes such as [fork] and chat:
 * have the same failure. Call this on the content leaf, excluding such markers.
 * Neutral/tied content inherits chrome direction rather than defaulting to LTR.
 */
export function contentDir(text: string): "ltr" | "rtl" | undefined {
	let arabic = 0;
	let latin = 0;
	for (const char of text) {
		if (!/\p{Letter}/u.test(char)) continue;
		if (/[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]/u.test(char)) arabic++;
		else if (/\p{Script=Latin}/u.test(char)) latin++;
	}
	return arabic > latin ? "rtl" : latin > arabic ? "ltr" : undefined;
}
