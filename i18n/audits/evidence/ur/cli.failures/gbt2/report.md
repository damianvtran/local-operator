# ur · cli.failures · audit gbt2 (fresh independent round)

- Scope: all 5 keys of `cli.failures` (new: `cli.failures.github_borrow_timeout`).
- Worktree commit: 80a55b20802ed13a7379fef90644f2345b63066a (base f0689a27).
- Inputs read: `i18n/style-guides/ur.md`, `i18n/glossary.md`, `docs/I18N.md`,
  en catalogue + context sidecar, ur catalogue. No MT/translation service used;
  no earlier receipts/records for this key consulted.
- Auditor: audit2-ur subagent, session 4af8d78683c7. Primary model
  openai/gpt-6.1-sol; the harness switched to anthropic/claude-opus-5-5
  (provider quota HTTP 429) after orientation and before any catalogue string
  was read — the semantic audit was performed by anthropic/claude-opus-5-5.
- Translator: i18n-translator (timeout-ur), openai/gpt-6.1-sol with fallback
  anthropic/claude-opus-5-5 during its run (as reported by parent). Independent
  agent/session; NOT an independent model family.

## Verdict: passed — 0 BLOCKER, 0 MAJOR, 1 MINOR, 0 NIT

## Mechanical (audit.log is the execution log)
- Keys 5/5, none added/dropped; context sidecar covers all 5.
- Placeholders identical per key; no ICU plural/select in the group.
- No manual bidi controls (LRM/RLM/FSI/PDI/ALM); no Hindi danda.
- `GitHub` kept Latin; Urdu semicolon ؛ used.
- `scripts/i18n/check.py`: rc=0 (catalogues consistent).
- `scripts/i18n/generate.py --check`: rc=1 — stale/orphaned mobile
  `bundled.*.gen.ts` artifacts (ledger/regeneration state of the branch), not a
  catalogue-content defect; recorded, not judged here.
- New key 87 cells (`rich.cells.cell_len`); no maxCells hint; one stderr line.

## Key-by-key
| key | ur | judgement |
|---|---|---|
| error_line | `{prog}: خرابی: {message}` | Shape `<command>: <error word>: <message>` kept. OK |
| required | `درج ذیل آرگومنٹس درکار ہیں: {names}` | Accurate, natural. OK |
| invalid_choice | `آرگومنٹ {argument}: غلط انتخاب: {value} ({choices} میں سے چنیں)` | Accurate; respectful imperative. OK |
| invalid_command | `{group} کی کمانڈ درست نہیں: {command}` | "Invalid {group} command" rendered naturally. OK |
| github_borrow_timeout | `{owner} سے GitHub کریڈینشل {seconds} سیکنڈ کے اندر موصول نہیں ہوئی؛ کمانڈ دوبارہ چلائیں` | See below |

### github_borrow_timeout — accuracy against the context constraints
- "from {owner}" → `{owner} سے … موصول` = received from the device: correct;
  owner stays a device, not an account/token.
- "did not arrive within {seconds} s" → `{seconds} سیکنڈ کے اندر موصول نہیں
  ہوئی`: polarity and time bound preserved; `سیکنڈ` invariant after a number
  is standard; Latin digits per guide.
- "retry the command" → `کمانڈ دوبارہ چلائیں` ("run the command again"):
  respectful imperative, actionable; does not suggest an automatic or extra
  retry, denial, revocation, login, or token/lease extension. Nothing added.
- Terminology: glossary `credentials` cell for ur is still `—`; the catalogue's
  established rendering is کریڈینشل (dozens of strings), used here. Agreement is
  feminine (`موصول نہیں ہوئی`), matching existing ur copy («مالک کریڈینشل تبدیل
  نہیں ہوئی»). Some speakers would treat the loanword as masculine; consistency
  with the catalogue wins — not a finding.
- Register: concise diagnostic, آپ-form imperative, no padding, no «براہ کرم»
  (English has no "please"). Native-sounding, not a calque.

## Findings

### U1 — MINOR — RTL/bidi — `cli.failures.github_borrow_timeout`
- Current: `{owner} سے GitHub کریڈینشل {seconds} سیکنڈ کے اندر موصول نہیں ہوئی؛ کمانڈ دوبارہ چلائیں`
- Problem: the line opens with `{owner}`, a sanitized device display name that
  will usually be Latin (e.g. a laptop name), followed by Latin `GitHub`. This
  string is emitted as a plain stderr line inside a bash tool result
  (`tools/builtin.py::_github_borrow_timeout_note`), so a first-strong /
  `dir=auto` viewer (web/Electron transcript, bidi-capable terminal) resolves
  the paragraph LTR and lays out the Urdu clauses left-to-right, putting the
  ؛ and the remedy in the wrong visual position. Meaning is intact; legibility
  on RTL surfaces suffers. Pre-existing keys also open with Latin placeholders
  but their argparse shape (`{prog}: …`) is constrained; this one is free.
- Suggested: `{seconds} سیکنڈ کے اندر {owner} سے GitHub کریڈینشل موصول نہیں ہوئی؛ کمانڈ دوبارہ چلائیں`
  — digits are bidi-weak, so the first strong character is Urdu and the
  paragraph resolves RTL; leading time adverbial is natural Urdu word order.
  Do not add manual bidi controls (style guide).
- Non-blocking: passed status stands; fix opportunistically.
