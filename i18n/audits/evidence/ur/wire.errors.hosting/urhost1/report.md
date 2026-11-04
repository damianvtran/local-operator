# Audit — `ur` / `wire.errors.hosting` (one key, reader 1)

| Field | Value |
|---|---|
| Audit id | `ur26host1` |
| Locale | `ur` |
| Namespace | `wire.errors.hosting` (new file, one key) |
| Key | `wire.errors.hosting.not_configured` |
| Auditor | `i18n-auditor`, session `1b7a2a90c876`, reader 1 |
| Model requested | `radient/auto` |
| Model served | `radient/auto` (`provider: radient`, `model_id: auto`) |
| Underlying model | **UNKNOWN** (routing alias; not inferred) |
| Independence | Translator `1955dea69023`; code author `94dcef624dec`; second reader not read |
| Verdict | **passed** — 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT |
| Holds left open | native/rendered proof **NOT RUN**; repo validator/ICU/CLDR **NOT RUN** (reasons below) |

## Inputs and hashes (computed independently)

| Role | Path | SHA-256 | Check |
|---|---|---|---|
| EN | `…/sessions/3ef6295a5545/scratchpad/hosting-source-v1/wire.errors.hosting.json` | `458d8f8be304428ccb6a328ce22e5a86515e9f44f869f0138a5cc6dfde964b63` | matches supplied |
| EN context | same dir, `wire.errors.hosting.context.json` | `ed0ae50fa465a010a2907bc680771dae80bc716d3a186b0af37e31e0d7063cf9` | matches supplied |
| UR target | `…/sessions/1955dea69023/scratchpad/hosting-ur/targets/ur/wire.errors.hosting.json` | `8f7977e2ce06a53267f0672f50605e9e5a09b5c52e85b58619acbb90dfbdd732` | matches supplied (113 bytes) |
| EN worktree copy | `~/local-operator/.worktrees/startup-owned-copy/local_operator/i18n/catalogues/en/wire.errors.hosting.json` | `458d8f8b…b63` | byte-identical to the supplied EN |

The readable worktree copy corroborates the supplied EN bytes without relying on the session scratch copy alone.

## Copy under audit

- EN: `Hosting platform is not configured.`
- UR: `ہوسٹنگ پلیٹ فارم ترتیب نہیں دیا گیا۔`

Context (read in full): an **owned cause clause** composed into startup refusals, spliced after a colon; diagnostic tone that names the condition and never restates the remedy; must read as a complete statement and keep its terminal period; one copy for the condition; no placeholders or dynamic detail; the English is also the legacy exception's text but that string stays English by contract — this key is display copy only.

## Native assessment

**Meaning.** «ہوسٹنگ پلیٹ فارم ترتیب نہیں دیا گیا۔» = “the hosting platform has not been configured.” It names the owned condition, adds no remedy, no next step, no external-error wording and no detail absent from the English. Splice test: `…: ہوسٹنگ پلیٹ فارم ترتیب نہیں دیا گیا۔` reads as a complete statement and closes with the Urdu full stop `۔` (U+06D4), not an ASCII period; the string carries no colon of its own, so the frame still owns the colon.

**Terms, checked against the shipped ur corpus (not against other locales).**
- `ترتیب نہیں دیا گیا` is the corpus phrase for “not configured”: `wire.errors.desktop.aida_no_provider` (ur) reads `ابھی کوئی پرووائیڈر ترتیب نہیں دیا گیا، …`. Using it keeps the error-family register.
- `ہوسٹنگ` sits in the corpus's host family (`ہوسٹ`, `ہوسٹس`, `ہوسٹ کیے` in `wire.settings`, `wire.remote_compute`); no corpus entry uses `میزبانی` for web hosting, and the author recorded rejecting it for its hospitality reading. The choice is consistent with the corpus.
- `پلیٹ فارم` has no corpus or style-guide entry but is settled, unremarkable Urdu for “platform”; it mirrors the English noun. The corpus's hosting *setting* row is labelled `ڈیفالٹ پرووائیڈر` and its helper text says `پرووائیڈر`, so a provider reading exists nearby — but the English key says “Hosting platform”, and substituting `پرووائیڈر` here would narrow the source and add a mechanism the sentence does not state. Mirroring “platform” is the correct translation decision; if the product wants the provider named, that is a **source** change, not a translation finding.
- Agreement is right: `پلیٹ فارم` is masculine, so `دیا گیا` agrees; the corpus's `پرووائیڈر … دیا گیا` confirms the pattern.

**Register and script.** Impersonal passive, matching the rest of `wire.errors`. No second person, no exclamation, no honorific padding. All-Urdu script with zero ASCII letters, so there is no English-fallback text and no case to carry (Urdu has no sentence case). No digits, no hand-written date or number. No manual bidi controls.

## Mechanical (applicable surface only)

| Check | Result |
|---|---|
| JSON parses; no duplicate keys; single key in EN, context and target; key parity | PASS |
| UTF-8, no BOM, no CRLF, trailing newline, two-space indent | PASS |
| NFC normalised | PASS |
| Cf / bidi-control codepoints | none |
| ASCII letters in the value (English fallback test) | none |
| ICU placeholders / arguments / braces | none present (zero-parameter key) |
| Terminal punctuation | exactly one `۔` (U+06D4); no `.` and no `:` in the value |
| Leading/trailing whitespace | none |

**Deliberately not run, with reasons.** The repo ICU parser, `scripts/i18n/check.py`, generation, the scanner, typecheck, tests, build and any native/app run were not executed — this round is source-only and forbids process dispatch. Placeholder and CLDR plural checks are **N/A**: the key has zero arguments and no plural/select structure, so there is nothing for them to assert beyond the no-braces check above. Pseudo-localisation, RTL/bidi rendering, width and native TUI proof are **not claimed**: they belong to the separate consumer/renderer gate, and no measured or executed render is asserted here.

## Findings

None. No added remedy or detail, no omission, no polarity change, no punctuation or script error, no terminology break against the corpus, no overflow signal available or claimed.

Advisories (not findings, no action requested of the translator):
1. The English names the **platform** while the preflight condition is “no provider configured”. The Urdu mirrors the English as the brief requires; if the naming mismatch is unwanted, it is a source-copy decision, not a translation defect.
2. The second locale that used a “provider” term instead of “platform” is outside this audit; the judgement above rests on this key's English and the actual ur corpus, as instructed, not on uniform wording across locales.

## Scope guards and provenance

- The legacy `HostingNotConfiguredError` string remains English by contract; this key is display copy only. Nothing here modifies the API error, the caller, the SDK, a catalogue, the ledger, generated types or a baseline — no repository edit was made.
- The namespace is a new, uncommitted candidate and is absent at base `87ca846245b2`; that state is **manager/author-stated** — Git was not run in this round, so it is not independently verified here. No shipping, activation or frame clearance is claimed.
- Operator acceptance (“no English anywhere in fr and the other supported UI locales”) is **not** a whole-page or whole-locale claim from this one key.

## Limitations

1. Source-only verdict: no native render, no rendered width, no RTL/shaping evidence, no executed runtime fit; those are separate gates.
2. No repository validator, ICU/CLDR run, generation, scanner, tests, typecheck or build.
3. Reader-1 single pass; the manager's second independent reader is separate. No other reader's verdict or report was read before this one.
4. Git state (uncommitted source, absence at base) is not verified here.
5. The candidate is one key; any edit to its value invalidates this record.

## Artifacts

- `report.md` — this report.
- `record.json` — machine record: identity, model, hashes, mechanical results, native assessment summary, verdict, gates, limitations. Its `artifacts` block carries this report's SHA-256; the record's own SHA-256 is reported with the packet.
