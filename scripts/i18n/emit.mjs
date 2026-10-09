#!/usr/bin/env node
// Build-time emitter: Node `Intl` -> committed data tables for the Python runtime.
//
// WHY THIS EXISTS. The Python side ships NO i18n dependency (operator decision
// §11.5: no Babel, no ICU binding). Everything locale specific the Python
// runtime needs — plural categories, number separators and grouping, month and
// weekday names, date/time patterns, relative-time templates — is MEASURED here
// from the same `Intl` the TypeScript surfaces use, and committed as JSON under
// `local_operator/i18n/data/`. The wheel then carries data, not a dependency.
//
// DETERMINISM IS A CONTRACT, not a nicety: `scripts/i18n/generate.py --check`
// re-runs this emitter and requires BYTE-IDENTICAL output, so a change in this
// file's output that is not regenerated and committed fails CI. Therefore:
//   - no clock reads, no randomness, no environment reads;
//   - fixed probe dates/numbers and a fixed `timeZone: 'UTC'` on every
//     formatter (a host-local timezone would move every date pattern);
//   - sorted keys everywhere plus a stable, explicit locale order;
//   - NODE-MAJOR STABLE OUTPUT (see the NNBSP_RE note below): byte-identical on
//     Node 24 — CI's `setup-node` pin (.github/workflows/ci.yml) — and Node 26,
//     so a local regeneration reproduces the CI drift check whichever of the
//     two the host runs.
//
// WHAT IS DELIBERATELY *NOT* EMITTED. The tables hold raw data plus golden
// `probes` (input -> the exact string Node produced). The Python formatter
// implements the algorithm and its tests compare against the probes — that is
// the drift anchor. Storing only the probes and no algorithm would make Python
// a string-lookup table; storing only symbols would trust our algorithm against
// nothing.
//
// USAGE.
//   node scripts/i18n/emit.mjs --out <dir>     # write plural_rules.json + formats.json
// (generate.py passes a temp dir and diffs; CI never lets this write in place.)
//
// The locales are the wave list from the RFC (§6): en now, then fr/es, ru/vi,
// hi/zh-CN, ur last. All eight are emitted from day one so the tables cannot
// be the thing that blocks a wave.

import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const LOCALES = ["en", "fr", "es", "zh-CN", "ru", "vi", "ur", "hi"];

// ---------------------------------------------------------------------------
// Argument handling
// ---------------------------------------------------------------------------

function parseArgs(argv) {
  let out = null;
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--out") {
      out = argv[++i];
    } else if (arg === "--help" || arg === "-h") {
      process.stdout.write("usage: node scripts/i18n/emit.mjs --out <dir>\n");
      process.exit(0);
    } else {
      process.stderr.write(`emit.mjs: unknown argument ${arg}\n`);
      process.exit(2);
    }
  }
  if (!out) {
    // No default target on purpose: a bare run must not be able to touch the
    // committed tables by accident. generate.py owns every real write.
    process.stderr.write("emit.mjs: --out <dir> is required\n");
    process.exit(2);
  }
  return { out };
}

// ---------------------------------------------------------------------------
// Plural rules: categories per locale + integer samples for validation
// ---------------------------------------------------------------------------
//
// The SAMPLES are the point: `scripts/i18n/generate.py` validates the Python
// rule functions against every sample here, so the Python selector cannot
// drift from the `Intl.PluralRules` the TS surfaces use. The sweep covers all
// residues mod 100 (which is all the ru-style rules look at) plus the band
// where category rules change across the shipped locales; the specials cross
// into magnitudes a compact sample list would miss, and the decimals exercise
// `v` (visible fraction digits) — `1.5` carries v=1 in both JS and Python.

const PLURAL_SWEEP_MAX = 200;
const PLURAL_SPECIALS = [250, 300, 500, 1000, 10000, 100000, 1000000, 1234567];
const PLURAL_DECIMALS = [0.5, 1.5, 2.5, 3.5, 11.5, 21.5, 101.5];

function pluralSamples(locale) {
  const rules = new Intl.PluralRules(locale);
  const samples = {};
  const values = [];
  for (let n = 0; n <= PLURAL_SWEEP_MAX; n += 1) values.push(n);
  values.push(...PLURAL_SPECIALS, ...PLURAL_DECIMALS);
  for (const n of values) {
    samples[String(n)] = rules.select(n);
  }
  return { categories: [...new Set(Object.values(samples))].sort(), samples };
}

// ---------------------------------------------------------------------------
// Numbers: separators, grouping sizes, minimum-grouping-digits, probes
// ---------------------------------------------------------------------------

const NUMBER_PROBE_VALUES = [
  "0.25", "0.001", "0.12345", "1", "1.5", "999.5", "1000", "1234", "10000",
  "12345.67", "100000", "123456", "1234567", "12345678",
];

function partsSymbol(parts, type) {
  const part = parts.find((p) => p.type === type);
  return part ? part.value : null;
}

function groupingInfo(locale) {
  const fmt = new Intl.NumberFormat(locale, { useGrouping: true });
  // Group sizes: format 1234567890 and read the integer part's layout. The
  // CLDR shape is [primary, secondary]: en [3,3] ("1,234,567,890"), hi [3,2]
  // ("1,23,45,67,890"). Both locales' dominant pattern is visible in this one
  // sample because it is long enough to exercise the secondary size twice.
  const parts = fmt.formatToParts(1234567890);
  const groupSizes = [];
  let run = 0;
  for (let i = parts.length - 1; i >= 0; i -= 1) {
    if (parts[i].type === "integer") {
      run += parts[i].value.length;
    } else if (parts[i].type === "group") {
      groupSizes.push(run);
      run = 0;
    }
  }
  if (groupSizes.length === 0) {
    // No grouping at all in the rendered sample (should not happen for these
    // locales); state the CLDR default so the shape stays uniform.
    groupSizes.push(3, 3);
  } else if (groupSizes.length === 1) {
    groupSizes.push(groupSizes[0]);
  }
  // minimumGroupingDigits is MEASURED, not read: `resolvedOptions()` does not
  // expose it. It is the digit count at which grouping first appears, minus
  // the primary group size: en groups at 4 digits ("1,000", so 1), a locale
  // that withholds grouping until 5 digits measures as 2.
  let minimumGroupingDigits = 1;
  for (let exponent = 3; exponent <= 7; exponent += 1) {
    const probeParts = fmt.formatToParts(10 ** exponent);
    if (probeParts.some((p) => p.type === "group")) {
      minimumGroupingDigits = exponent + 1 - groupSizes[0];
      break;
    }
  }
  return {
    group: partsSymbol(parts, "group"),
    decimal: partsSymbol(fmt.formatToParts(0.5), "decimal") ?? ".",
    groupSizes: [groupSizes[0], groupSizes[1]],
    minimumGroupingDigits,
  };
}

function numberProbes(locale) {
  const fmt = new Intl.NumberFormat(locale, { useGrouping: true });
  const probes = {};
  for (const value of NUMBER_PROBE_VALUES) {
    probes[value] = fmt.format(Number(value));
  }
  return probes;
}

function percentInfo(locale) {
  const fmt = new Intl.NumberFormat(locale, { style: "percent", useGrouping: true });
  const parts = fmt.formatToParts(0.25);
  const pattern = parts
    .map((p) => (p.type === "integer" || p.type === "group" || p.type === "decimal" || p.type === "fraction" ? "{n}" : p.value))
    .join("");
  return {
    pattern,
    probes: { "0.25": fmt.format(0.25), "1": fmt.format(1), "0.075": fmt.format(0.075) },
  };
}

// ---------------------------------------------------------------------------
// Dates and times: tokenised patterns + name lists
// ---------------------------------------------------------------------------
//
// A pattern is built by walking `formatToParts` for ONE fixed fixture instant
// and mapping each part to a TOKEN the Python renderer understands; literal
// parts (separators, "年" etc.) are kept verbatim. Month NAMES arrive as
// `month` parts (e.g. en medium "Jul 9, 2025"): they are matched against the
// locale's own FORMAT-CASE name arrays — exact-match, and an unmatched name is
// a hard error rather than a baked literal. Tokens:
//   date: {yy} {yyyy} {m} {mm} {mon} {mon_full} {d} {dd}
//   time: {h} {hh} {H} {HH} {min} {ss} {ampm}
// Padding ({mm}/{dd}/{hh}) is detected from a LEADING ZERO in the measured
// part — which is why the fixture uses July 9 and the 04:xx hour: a
// two-digit month/day/hour could not distinguish padded from unpadded.
//
// The renderer substitutes from the caller's datetime; because the pattern is
// MEASURED, differences like "7/9/25" vs "9.7.25" vs "2025/7/9" cannot drift
// without a regeneration.

const DATE_FIXTURE = Date.UTC(2025, 6, 9, 4, 5, 9); // Wed 2025-07-09 04:05:09Z

function padded(value) {
  return value.length === 2 && value.startsWith("0");
}

// `formatToParts` and `format` DISAGREE on the narrow no-break space, and the
// disagreement is MAJOR-DEPENDENT: Node 24's parts carry U+202F in six slots
// (en time short/medium + datetime; ru date medium/long + datetime) where the
// string `format()` produces — and where Node 26's parts also carry — U+0020
// (measured 2026-10-09 across all eight locales and both majors; `format()`
// itself never emitted U+202F on either). We emit the format() spelling,
// because that is the string a surface actually shows, and normalising keeps
// the committed tables byte-identical across the two majors — without it the
// drift check passes on one Node and fails on the other (round-1 B1).
const NNBSP_RE = /\u202f/g;

function normalizeSpaces(value) {
  return value.replace(NNBSP_RE, " ");
}

function dateTokenFor(part, monthArray, monthToken, where) {
  // Shared by datePattern and datetimeInfo — the same fixture, the same
  // format-case month comparison, the same loud error on an unmatched name.
  // The value is normalised first (see NNBSP_RE) so the comparison sees the
  // format() spelling whatever the parts emitted.
  const value = normalizeSpaces(part.value);
  switch (part.type) {
    case "year":
      return value.length === 2 ? "{yy}" : "{yyyy}";
    case "month":
      if (/^\d+$/.test(value)) return padded(value) ? "{mm}" : "{m}";
      if (value === monthArray[6]) return monthToken;
      throw new Error(
        `emit.mjs: month part ${JSON.stringify(value)} for ${where} matches no ${monthToken} entry — refusing to bake a month name into a pattern`
      );
    case "day":
      return padded(value) ? "{dd}" : "{d}";
    default:
      return null;
  }
}

function timeTokenFor(part, hour12) {
  switch (part.type) {
    case "hour":
      if (hour12) return padded(part.value) ? "{hh}" : "{h}";
      return padded(part.value) ? "{HH}" : "{H}";
    case "minute":
      return "{min}";
    case "second":
      return "{ss}";
    case "dayPeriod":
      return "{ampm}";
    default:
      return null;
  }
}

function datePattern(locale, names) {
  return (style) => {
    const fmt = new Intl.DateTimeFormat(locale, { dateStyle: style, timeZone: "UTC" });
    const parts = fmt.formatToParts(new Date(DATE_FIXTURE));
    // Month NAMES are matched against the array built FROM THE SAME STYLE: the
    // format case and the stand-alone case differ (ru: dates say "июл."/"июля",
    // stand-alone says "июль"), so comparing against stand-alone names baked
    // "июл." into the pattern as a LITERAL — every month rendered July. The
    // style fixes their tokens: long patterns get {mon_full}, short/medium get
    // {mon}; dateTokenFor refuses to bake a name either way.
    const expected = style === "long" ? { months: names.monthsWide, token: "{mon_full}" } : { months: names.monthsShort, token: "{mon}" };
    return parts
      .map((p) => {
        const token = dateTokenFor(p, expected.months, expected.token, `${locale}/${style}`);
        return token === null ? normalizeSpaces(p.value) : token;
      })
      .join("");
  };
}

function datetimeInfo(locale, names) {
  // The date+time JOIN is per-locale — vi puts the time FIRST ("14:05 19 thg
  // 7, 2025"), zh-CN joins with a bare space, en with ", " — so it cannot be
  // a Python-side `date + " " + time`. One medium date + short time fixture,
  // tokens from the shared mappers.
  const fmt = new Intl.DateTimeFormat(locale, {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "UTC",
  });
  const parts = fmt.formatToParts(new Date(DATE_FIXTURE));
  const hour12 = parts.some((p) => p.type === "dayPeriod");
  return parts
    .map((p) => {
      const date = dateTokenFor(p, names.monthsShort, "{mon}", `${locale}/datetime`);
      if (date !== null) return date;
      const time = timeTokenFor(p, hour12);
      if (time !== null) return time;
      return normalizeSpaces(p.value);
    })
    .join("");
}

// Month arrays from a full DATE-STYLE rendering, in the FORMAT case the
// patterns use (see the note inside `datePattern`): monthsShort from medium,
// monthsWide from long. Weekdays stay stand-alone — no emitted pattern uses
// them yet; they exist for consumers like relative/absolute renderers later.
function monthStyleParts(locale, style) {
  const fmt = new Intl.DateTimeFormat(locale, { dateStyle: style, timeZone: "UTC" });
  const months = [];
  for (let m = 0; m < 12; m += 1) {
    const parts = fmt.formatToParts(new Date(Date.UTC(2025, m, 15)));
    const month = parts.find((p) => p.type === "month");
    months.push(month ? normalizeSpaces(month.value) : "");
  }
  return months;
}

function nameLists(locale) {
  const monthsShort = monthStyleParts(locale, "medium");
  const monthsWide = monthStyleParts(locale, "long");
  const weekdaysWide = [];
  const weekdaysShort = [];
  // 2025-06-01 is a Sunday (getUTCDay() === 0), so index directly by weekday.
  for (let wd = 0; wd < 7; wd += 1) {
    const d = new Date(Date.UTC(2025, 5, 1 + wd));
    weekdaysWide.push(new Intl.DateTimeFormat(locale, { weekday: "long", timeZone: "UTC" }).format(d));
    weekdaysShort.push(new Intl.DateTimeFormat(locale, { weekday: "short", timeZone: "UTC" }).format(d));
  }
  return { monthsWide, monthsShort, weekdaysWide, weekdaysShort };
}

function timeInfo(locale) {
  const build = (style) => {
    const fmt = new Intl.DateTimeFormat(locale, { timeStyle: style, timeZone: "UTC" });
    const parts = fmt.formatToParts(new Date(DATE_FIXTURE));
    const hour12 = parts.some((p) => p.type === "dayPeriod");
    const pattern = parts
      .map((p) => {
        const token = timeTokenFor(p, hour12);
        return token === null ? normalizeSpaces(p.value) : token;
      })
      .join("");
    return { pattern, fmt };
  };
  const short = build("short");
  const medium = build("medium");
  const cycle = short.fmt.resolvedOptions().hourCycle ?? null;
  let periods = null;
  if (cycle === "h11" || cycle === "h12") {
    const readPeriod = (hour) => {
      const fmt = new Intl.DateTimeFormat(locale, { hour: "numeric", timeZone: "UTC" });
      const parts = fmt.formatToParts(new Date(Date.UTC(2025, 6, 9, hour)));
      const part = parts.find((p) => p.type === "dayPeriod");
      return part ? part.value : null;
    };
    periods = { am: readPeriod(4), pm: readPeriod(16) };
  }
  return { short: short.pattern, medium: medium.pattern, cycle, periods };
}

// ---------------------------------------------------------------------------
// Relative time
// ---------------------------------------------------------------------------
//
// numeric: "always" on purpose: the templates stay a single `{n}` substitution
// per category ("1 day ago", "in 1 day" in en) rather than mixing special-case
// words ("yesterday"/"tomorrow") into a template table. The English specials
// are the ONLY reason this matters today; formatting them would add a second
// code path for one language's two words. A style-guide call can revisit it.

const RELATIVE_UNITS = ["second", "minute", "hour", "day", "week", "month", "year"];

// Representative integers per plural category, chosen so every category a
// locale produces is sampled: `1, 3, 5, 21, 1000000` covers one/few/many at
// small magnitudes and fr's `many` (which only fires at that magnitude), and
// `1.5` catches each locale's DECIMAL category — for ru every decimal is
// `other`, which none of the integer samples produce. The first sample per
// category wins, so decimals never replace an integer template.
const CATEGORY_SAMPLES = [1, 3, 5, 21, 1000000, 1.5];

function relativePart(fmt, n, unit) {
  return fmt
    .formatToParts(n, unit)
    .map((p) => (p.type === "integer" || p.type === "group" || p.type === "decimal" || p.type === "fraction" ? "{n}" : p.value))
    .join("")
    // A grouped number ("1,000,000") maps each digit run to a placeholder;
    // collapse the run into the single {n} the Python renderer substitutes.
    .replace(/(?:\{n\})+/, "{n}");
}

function relativeTemplates(locale) {
  const fmt = new Intl.RelativeTimeFormat(locale, { numeric: "always" });
  const rules = new Intl.PluralRules(locale);
  const out = {};
  for (const unit of RELATIVE_UNITS) {
    const past = {};
    const future = {};
    for (const n of CATEGORY_SAMPLES) {
      const category = rules.select(n);
      if (!(category in past)) past[category] = relativePart(fmt, -n, unit);
      if (!(category in future)) future[category] = relativePart(fmt, n, unit);
    }
    out[unit] = { past, future };
  }
  return out;
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

function buildPluralRules() {
  const locales = {};
  for (const locale of LOCALES) locales[locale] = pluralSamples(locale);
  return { schema: 1, locales };
}

function buildFormats() {
  const locales = {};
  for (const locale of LOCALES) {
    const names = nameLists(locale);
    const dateOf = datePattern(locale, names);
    locales[locale] = {
      number: { ...groupingInfo(locale), probes: numberProbes(locale) },
      percent: percentInfo(locale),
      date: {
        short: dateOf("short"),
        medium: dateOf("medium"),
        long: dateOf("long"),
        ...names,
      },
      time: timeInfo(locale),
      datetime: datetimeInfo(locale, names),
      relative: relativeTemplates(locale),
    };
  }
  return { schema: 1, locales };
}

// Stable serialisation: sorted keys, two-space indent, trailing newline. The
// --check path compares bytes, so this is part of the contract.
function stableStringify(value) {
  const sort = (v) => {
    if (Array.isArray(v)) return v.map(sort);
    if (v && typeof v === "object") {
      const out = {};
      for (const key of Object.keys(v).sort()) out[key] = sort(v[key]);
      return out;
    }
    return v;
  };
  return `${JSON.stringify(sort(value), null, 2)}\n`;
}

function main() {
  const { out } = parseArgs(process.argv.slice(2));
  mkdirSync(out, { recursive: true });
  const plural = stableStringify(buildPluralRules());
  const formats = stableStringify(buildFormats());
  writeFileSync(join(out, "plural_rules.json"), plural, "utf8");
  writeFileSync(join(out, "formats.json"), formats, "utf8");
  process.stdout.write(`emit.mjs: wrote ${join(out, "plural_rules.json")} and ${join(out, "formats.json")}\n`);
}

main();
