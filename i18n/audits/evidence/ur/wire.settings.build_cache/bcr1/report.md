# Independent audit — `wire.settings.build_cache` — ur — audit id `bcr1`

- **Verdict: passed** — 0 blocker, 0 major, 0 minor, 0 nit
- Locale `ur`, namespace `wire.settings.build_cache`, **10 keys**, audited 2026-11-08
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
load-bearing facts), `i18n/glossary.md`, this locale's `i18n/style-guides/ur.md`,
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

### RTL and bidi

Storage is logical order and the render layer applies FSI/PDI per string, so the
guide forbids manual bidi controls. Verified: **zero** RLM/LRM/LRE/RLE/PDF/LRO/RLO/
LRI/RLI/FSI/PDI characters in all 10 values. Nothing to double-apply.

LTR islands stay byte-identical Latin, in source order: `Rust target/`,
`Next.js .next/`, `lop build-cache reap`, `--dry-run`, `build/`, `dist/`, `CMake`,
`Gradle`. `include_extra_shapes.label` begins with the LTR run **build/ اور dist/
شامل کریں** — that is source order preserved, which is what the guide requires for
a string whose first element is a literal; the bidi algorithm keeps the run intact
inside the isolate.

### Punctuation, per the ur guide

The guide names its own set and forbids Devanagari: ۔ full stop **10/10** help
finals, ، comma **7×**, ؛ semicolon **3×**, no danda (।) anywhere, and no Latin
comma-space in prose outside the literals. Earlier in this audit my own check
wrongly flagged ، and ؛ as foreign signs; the guide settles it (they *are* the
required Urdu punctuation) and the shipped ur corpus uses them heavily. The check
was corrected before any judgement was recorded.

### Register and idiom

آپ throughout; the one instruction uses the respectful imperative ("طلب پر
چلائیں"). "فولڈر کو ہاتھ نہیں لگاتا" is the idiomatic Urdu for leaving something
alone — a better rendering of "leaves a directory alone" than a literal verb would
be. The load-bearing clause survives intact: "آف: کبھی کچھ نہیں ہٹتا" (off: nothing
is ever removed).

### Observed, not a finding

The locale renders "cache" two ways across the corpus — کیش (19) and کیشے (16).
This namespace uses کیش twice, which is attested. The split is pre-existing, spans
namespaces this audit does not judge, and is out of scope here; flagging it as this
namespace's defect would misattribute it.

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
