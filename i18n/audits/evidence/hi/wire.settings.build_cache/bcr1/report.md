# Independent audit — `wire.settings.build_cache` — hi — audit id `bcr1`

- **Verdict: passed** — 0 blocker, 0 major, 0 minor, 0 nit
- Locale `hi`, namespace `wire.settings.build_cache`, **10 keys**, audited 2026-11-08
- Audited revision `b24b7b78c63370cf17ca69be336fefb5535f455d` — the translated bytes. This audit's own commit adds the
  ledger rows and this bundle; it edits no catalogue value, so the audited bytes are
  still the current bytes (`current_bytes_match_audited_hashes: true`).
- **Auditor:** independent i18n-auditor (lopdev task subagent, build-cache settings
  lane), session `6368251d546b`, served model `deepseek/deepseek-flash`
- **Translator:** `i18n-translator`, model `anthropic/claude-opus-5-5`, commit
  `b24b7b78c63`, no fallback (identity supplied by the delegation brief — no
  translator receipt file exists in the tree; see `record.json`)

## Independence

I did not translate these values. I edited no value in this namespace, no sidecar, no
glossary, no style guide and no generated artifact; the only writes are the seven
ledger rows under `i18n/audit-history/` and these seven evidence bundles. The nits
below are reported, not fixed — an auditor who silently rewrites what they are
judging has stopped being an auditor.

## What was read

The English source and its `.context.json` sidecar (role, tone, and the two
load-bearing facts), `i18n/glossary.md`, this locale's `i18n/style-guides/hi.md`,
and all ten values here against the English. Material term questions were settled
against the **shipped, already-audited corpus for this locale**, which is what turns
"this looks wrong to me" into a finding or a rejected candidate.

## Role fidelity against the sidecar

The sidecar names the roles and the facts that must survive, so those were judged
before style:

| Requirement | Here |
|---|---|
| switch labels are action/noun phrases, help text is a sentence | labels and title carry no final punctuation; all six help/description strings are sentences with the locale's full stop |
| `enabled.help`: OFF means **nothing is ever removed** | present and absolute in this locale |
| `age_hours.help`: the live-build check protects a running compile; the age floor is the *slower second* guard | both guards present and correctly ranked |
| `age_hours`: the unit is HOURS and is never converted | hours preserved |
| `description` / `enabled.help`: removals are logged; the command still reports | both present |
| `dry_run.help`: a per-run override that does not write the setting | present |
| `include_extra_shapes`: off by default; the names mean output in some ecosystems and source in others — the user's call | present, and no claim that they are always output |

## Locale judgement

### Register and role

Guide pins **आप**, the polite imperative, and danda for prose. All hold: "बिल्ड कैश
हटाएँ", "केवल रिपोर्ट (ड्राई रन)", "build/ और dist/ शामिल करें"; instructions use
"चलाएँ" / "चालू करें"; all ten help/description strings end in । (mechanical check).
No "!" and no honorific padding.

### Term decisions checked against the guide and the shipped corpus

- Settled loans in Devanagari, as the guide lists: **कैश**, बिल्ड, फ़ाइल, फ़ोल्डर,
  कमांड, रिपॉज़िटरी, चेकआउट. Nukta kept consistently (फ़, ज़, ड़) — part of the
  spelling, per guide.
- `age_hours.label` **न्यूनतम उम्र (घंटे)** uses उम्र. I checked the shipped hi
  corpus before judging: उम्र appears 8× and आयु 2×, so उम्र is the settled
  rendering rather than a drift away from a formal form.
- `ड्राई रन` for "dry run" — the transliteration the corpus already uses, not a
  half-translated hybrid.

### Load-bearing facts preserved

Both facts the sidecar says make the feature safe survive: removal only when
nothing is using the directory ("जब कुछ भी उपयोग में न हो") and every removal logged
("हर बार हटाने को लॉग करता है"); plus the absolute "बंद: कुछ भी कभी नहीं हटता" for the
off state, and the two-guard distinction in `age_hours.help` ("चल रही बिल्ड की असली
सुरक्षा लाइव-बिल्ड जाँच करती है; यह दूसरी, धीमी सुरक्षा परत है").

## Findings

**None.** No blocker, no major, no minor, and no nit for this locale.

## Verification — real output in `audit.log`

Shared run over all seven locales (one namespace, one English source):

1. `check.py --catalogues-only` → **0 problem(s)**
2. Independent mechanical audit — 10/10 keys in every locale, key sets exactly
   equal, sidecar covers every key, **0 ICU/printf placeholders** (the
   "no placeholders" claim was verified, not assumed), and all ten literals
   (`target/`, `.next/`, `build/`, `dist/`, `lop build-cache reap`, `--dry-run`,
   `Rust`, `Next.js`, `CMake`, `Gradle`) byte-identical with the English counts.
3. `generate.py --check` **before** the ledger rows → stale (the translated
   namespaces had no passing row, so all seven locales un-shipped); **after** →
   **158 artifact(s) byte-identical**
4. Coverage gate, this file only, serial and bounded:
   `tests/unit/i18n/test_language_manifest.py` → **35 passed**
5. Shipping proof → all seven locales carry 10/10 keys with `status: passed` and a
   fresh `source_sha256`, and no `coverage-allowance` debt exists for this namespace

`source_sha256` is the repository's own rule — `catalogues.catalogue_sha256("en",
namespace)`, i.e. sha256 of the committed English file bytes. I computed it with that
API, then regenerated and re-checked: the value does not move, because the English
source is untouched by this commit.

## Limitations — what this audit does not claim

- **No rendered-frame or layout proof.** Widths were measured with
  `rich.cells.cell_len` on the strings; no settings page was painted and inspected.
  The longest label is 1.67× English, inside the ≤2× settings budget, but that is a
  measurement, not a rendered frame.
- **No native-speaker certification** and no audio/RTL hardware pass.
- **`ur` TUI layout** stays the documented degraded LTR stance; this audit makes no
  claim about terminal shaping or mirroring.
- **Served-model disclosure:** this lane opened on `anthropic/claude-opus-5-5`, hit
  provider quota twice, and ran under `deepseek/deepseek-flash`; the model is
  self-reported from the session's switch notices and not independently observable.
