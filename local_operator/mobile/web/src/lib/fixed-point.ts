/**
 * Python's fixed-precision formatting, reproduced exactly.
 *
 * `format(value, f".{digits}f")` — what every f-string in the Python sources
 * this bundle ports (`tui/costs.py`, `tui/widgets/status_line.py`,
 * `session/frontend_state.py`) compiles to — rounds HALF TO EVEN on the
 * value's exact binary expansion. `Number.prototype.toFixed` rounds half AWAY
 * FROM ZERO, so the two disagree at every representable tie, and the ties are
 * reachable: through the real TUI widget, one session state rendered
 * `1.2%/200k` in the band and `1.3%/200k` in the desktop strip (that repo's
 * round-1 finding, R1/Q1). The phone now prints the same class of readings —
 * cost digits and context percentages — so it adopts the desktop's rule
 * rather than a second one of its own.
 *
 * ## Why an exact-decimal implementation rather than `Intl.NumberFormat`
 *
 * `Intl.NumberFormat` gained `roundingMode: "halfEven"`, and it is the obvious
 * answer, and it is wrong here. It rounds the number's SHORTEST DECIMAL
 * REPRESENTATION, not its exact binary value, so it treats a literal that
 * merely looks like a tie as one:
 *
 * ```
 *                 exact binary value                 Python   Intl halfEven
 *   12.345   12.34500000000000063948846218409...     12.35    12.34   (wrong)
 *    2.675   2.67499999999999982236431605997...       2.67     2.68   (wrong)
 * ```
 *
 * Both of those are NOT ties — the stored double sits just above or just below
 * the midpoint — and Python's answer follows from the bits, not from how the
 * literal was typed. An implementation that reads the bits is the only one
 * that agrees on both the real ties and the near-ties, so this expands the
 * double itself.
 *
 * ## Why it is exact
 *
 * Every finite IEEE-754 double is `mantissa x 2^e` for integers `mantissa` and
 * `e`, and therefore has a FINITE decimal expansion: multiplying by `2^-e` for
 * negative `e` is multiplying by `5^-e` and shifting the decimal point. So the
 * value is carried here as a `BigInt` numerator over a power of ten with no
 * rounding anywhere, and the single rounding step at the end is the one the
 * caller asked for. No floating-point arithmetic happens after the bits are
 * read, which is what makes this exact rather than merely more accurate.
 *
 * Ported from the desktop strip's `fixed-point.ts` (same algorithm, same
 * comments where they still apply). What would make this mirror wrong is
 * CPython changing `float.__format__`'s rule, or a caller reaching for
 * `toFixed` directly instead of coming through here.
 */

/** A double split into `sign`, and an exact `num / 10^scale`. */
type ExactDecimal = { negative: boolean; num: bigint; scale: number };

const EXPONENT_MASK = 0x7ffn;
const MANTISSA_MASK = 0xfffffffffffffn;
/** Exponent bias plus mantissa width: `value = mantissa * 2^(exp - 1075)`. */
const EXPONENT_SHIFT = 1075;
/** The exponent a subnormal is stored with, once the implicit bit is absent. */
const SUBNORMAL_EXPONENT = -1074;

/**
 * The exact decimal value of a double, as an integer over a power of ten.
 *
 * Reads the IEEE-754 bits rather than any decimal string, because the decimal
 * string is precisely the lossy step that makes `Intl` disagree with Python.
 */
function exactDecimal(value: number): ExactDecimal {
	const view = new DataView(new ArrayBuffer(8));
	view.setFloat64(0, value);
	const bits = (BigInt(view.getUint32(0)) << 32n) | BigInt(view.getUint32(4));
	const negative = ((bits >> 63n) & 1n) === 1n;
	const rawExponent = Number((bits >> 52n) & EXPONENT_MASK);
	const rawMantissa = bits & MANTISSA_MASK;
	// A zero exponent is a subnormal (no implicit leading 1) or zero itself.
	const mantissa = rawExponent === 0 ? rawMantissa : rawMantissa | (1n << 52n);
	const exponent =
		rawExponent === 0 ? SUBNORMAL_EXPONENT : rawExponent - EXPONENT_SHIFT;
	// value = mantissa * 2^exponent. For exponent < 0 that is mantissa * 5^-e
	// over 10^-e, which is exact; for exponent >= 0 it is a plain shift.
	if (exponent >= 0)
		return { negative, num: mantissa << BigInt(exponent), scale: 0 };
	return {
		negative,
		num: mantissa * 5n ** BigInt(-exponent),
		scale: -exponent,
	};
}

/** Place the decimal point in a digit string holding `digits` fraction digits. */
function withPoint(value: bigint, digits: number): string {
	let text = value.toString();
	if (digits === 0) return text;
	// Left-pad so a value below 1 keeps its leading zero and full fraction.
	while (text.length <= digits) text = `0${text}`;
	return `${text.slice(0, text.length - digits)}.${text.slice(text.length - digits)}`;
}

/**
 * `format(value, f".{digits}f")` — Python's answer, digit for digit.
 *
 * Non-finite input returns `""`, because a reading that is NaN or infinite has
 * nothing honest to print and every caller in this feature already hides a
 * segment it cannot spell rather than inventing one.
 */
export function pyFixed(value: number, digits: number): string {
	if (!Number.isFinite(value)) return "";
	// `value`, not `Math.abs(value)`: the sign is part of the reading and must
	// survive into the string — a negative cost prints `$-0.0042` as Python
	// does, and stripping it here would also silently change which magnitude
	// branch a caller picks, since `cost < 0.01` is true for every negative.
	const { negative, num, scale } = exactDecimal(value);
	// Already coarser than the requested precision: scale up, nothing to round.
	if (digits >= scale)
		return (
			(negative ? "-" : "") +
			withPoint(num * 10n ** BigInt(digits - scale), digits)
		);
	const divisor = 10n ** BigInt(scale - digits);
	const quotient = num / divisor;
	const remainder = num % divisor;
	const half = divisor / 2n;
	// Half to even: round up only when past the midpoint, or exactly on it with
	// an odd quotient. This is the single rounding step in the whole function.
	const rounded =
		remainder > half || (remainder === half && (quotient & 1n) === 1n)
			? quotient + 1n
			: quotient;
	return (negative ? "-" : "") + withPoint(rounded, digits);
}
