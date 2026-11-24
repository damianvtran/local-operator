# Urdu independent audit A — tui.model_persistence

Verdict: **passed** for linguistic copy and scoped mechanical validation. All four keys were independently read in source order: persist, settings, saved, with_model. Findings: none (0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT). This is not B approval, consumer integration approval, native RTL certification, eligibility or release approval.

Auditor: i18n-auditor, session caf2c0ebbb90, actual model openai/gpt-6.1-sol. Host explicitly switched from anthropic/claude-opus-5-5 because of provider quota before source/target assessment. Translator session 3c1e111e2598 is different; its canonical author record explicitly identifies the same actual GPT fallback model. Same model does not mean same session; no translator conclusions were adopted as audit evidence. Method: direct LLM native-fluent assessment; no MT/translation service/API.

## Scope and authority

Audited revision b1c5ca8be245dcc3f9cae4463d9a5553b3e0c01f, frozen input hashes in record.json. Read Urdu guide and glossary first, English and context next, target next through approved read_public_copy. Read HOWTO, RFC sections 2.2/6, and canonical evidence README. Translation author record supplies translator provenance only. Actual validations below were executed independently here, not copied from author logs.

## Every-key assessment (audit_notes_not_execution_log)

- `tui.model_persistence.persist`: `/model default اس ماڈل کو نئے سیشنز کے لیے محفوظ کرتا ہے`. Correctly says the command saves this model for NEW sessions, without promising a current-session switch. محفوظ کرتا ہے is the natural factual counterpart of glossary محفوظ کریں; سیشنز is an ordinary software loanword plural, permitted by the guide's natural inflection rule. Literal two-word command retained in logical order. No change suggested.
- `tui.model_persistence.settings`: `/settings سے نئے سیشن کا ابتدائی ماڈل بھی بدلا جا سکتا ہے`. Natural, concise Urdu: /settings can also change the starting model for a new session. ابتدائی makes the boot-default meaning explicit without an opaque literal translation of “boot”. بھی correctly preserves “too”; as a complete clause it remains usable when this is the sole route. It does not introduce a forbidden /model command. No change suggested.
- `tui.model_persistence.saved`: `اس سیشن میں محفوظ ابتدائی ماڈل پر آنے کے لیے /model saved استعمال کریں`. Respectful imperative استعمال کریں; explicitly THIS session and the saved starting model. محفوظ modifies the destination model, not the action. پر آنے means moving/returning to that model, not saving it; the switch/save polarity is intact. Natural everyday Urdu rather than Hindi or over-Persianized wording. No change suggested.
- `tui.model_persistence.with_model`: `ماڈل: {label} — {routes}`. Human prefix uses glossary ماڈل. Arguments remain simple strings, in source order, each exactly once. Opaque ID remains unchanged, already-localized routes interpolate once. Colon and spaced em dash are appropriate to a compact mixed-direction UI envelope; this is not sentence prose requiring a terminal Urdu full stop. No change suggested.

All four fragments consistently omit terminal full stops like the source; no unsafe/destructive claim, added permission, negation or omission was found. No text width cap exists in these sidecars; the footer's 43-cell budget must not be imported.

## Actual execution

Run (absolute worktree path abbreviated as WT only in this explanatory command):

```
mkdir -p "$LOCAL_OPERATOR_SCRATCHPAD/ur-a-home"
env -i HOME="$LOCAL_OPERATOR_SCRATCHPAD/ur-a-home" LOCAL_OPERATOR_CONFIG_DIR="$LOCAL_OPERATOR_SCRATCHPAD/ur-a-home/config" PATH="$PATH" TERM=xterm-256color PYTHONPATH="$WT" /Users/damian/local-operator/.venv/bin/python "$WT/i18n/audits/evidence/ur/tui.model_persistence/a-caf2/script.py.txt" > "$WT/i18n/audits/evidence/ur/tui.model_persistence/a-caf2/audit.log" 2>&1
```

Exit 0. audit.log is actual output; script.py.txt is the executed reproduction. env -i strips CMUX and inherited configuration; the module paths printed in the log establish worktree catalogue/runtime imports with the primary .venv interpreter. No Git ran under the isolated HOME.

Official `check_catalogues()` executed with only its locale/namespace enumeration narrowed to en/ur + this namespace; real readers and validation logic were unchanged. Result `[]`. Missing-key and missing-label-argument negative fixtures were rejected by that same official checker, and both locales' runtime missing-label cases raised MessageFormatError. Exact ordered key parity and context shape passed. ICU argument names/types/count/order match. **Zero plural/select arms in all four source and target strings**: no CLDR arms were required or invented. Protected commands, markup/backticks (none), NFC and absence of manual stored bidi controls passed. Before/after input hashes matched.

72 plain-Rich wrapping cases: English first as same-code control, then Urdu; ordinary/missing-model/settings-only routes; 12- and 81-character opaque labels; widths 80/100/120; original and human-prose-doubled pseudo fixtures. The HOWTO advertises no official pseudo command, so pseudo is explicitly a protected-token diagnostic fixture, not an official generator. All line cell lengths fit the requested width. Urdu non-pseudo final lines were not lone commands. A long unbroken ID is folded by plain Rich (English control does the same); this is not a locale-specific overflow finding. No invented fixed cap, transcript-family geometry claim, or new all-width guarantee.

## RTL limits and next boundary

Logical Urdu sentence order is sound; /model default, /model saved and /settings remain exact Latin islands. Stored bidi controls are absent, as mandated. The UI must isolate complete two-word commands and opaque IDs at presentation time; storage must not be modified to compensate for a terminal. Plain Rich cell diagnostics do not establish visual order, Arabic shaping, native font behavior, balanced presentation isolates, or styled NoticeBlock budgets. Those integration proofs, including pristine/non-empty transcript families, remain with later styled/native QA. This linguistic pass says nothing about the consumer that still displays English.

## Discovery diagnostics (not validator failures)

An initial read of a presumed catalogues/__init__.py returned path-not-found; the module is catalogues.py, confirmed by real imported __file__. A discovery rg command included nonexistent pseudo.py and exited 2; HOWTO confirms no advertised one-command pseudo workflow. These were preflight lookup errors, not scoped validator output. They were not relabeled as successful execution.
