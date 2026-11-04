# Urdu core literals — translation handoff

- Translator: i18n-translator, session `6d443470fd42`; actual translation model `openai/gpt-6.1-sol` following quota fallbacks from `anthropic/claude-sonnet-5-5` and `anthropic/claude-opus-5-5`.
- Worktree: `/Users/damian/local-operator/.worktrees/lit-translate`.
- Written: `local_operator/i18n/catalogues/ur/cli.failures.json` (4 keys), `cli.group_help.json` (10 keys), `tui.setup_cues.json` (2 keys): **16 keys total**.
- Inputs read: all three English catalogues and context sidecars, complete Urdu style guide and glossary, AGENTS.md i18n/environment guidance, docs/I18N.md and RFC catalogue/audit sections. Assigned Urdu files did not previously exist.

## Terminology and interpretation

- Preserved `Hub` (consistent with existing `wire.slash.hub`), `Radient`, `AI`, and `pull` (the glossary explicitly keeps hub-operation `pull` English).
- Used glossary forms مہارت (skill), ایجنٹ (agent), ڈیوائس (device), سیشن (session), ٹول (tool); respectful imperative throughout.
- Errors: خرابی for error, آرگومنٹ for argument, کمانڈ for command. The outer `{prog}: خرابی: {message}` retains the required shape; already-quoted values and choices remain interpolation-only, with no added quotes.
- Safety meanings retained: any execute-permission bit causes bundle rejection; installation never executes code; disk reclamation requires demonstrated safety; the operator key may relax the approval condition rather than promising unconditional bypass; phone pairing authorisation is one-off.
- `without a prompt` means without asking the user each time, not absence of an AI prompt. `headless` means no visible window, expressed directly rather than literally. `load-balanced` is explained as distributing load across providers.
- Glossary proposals (not edits): harness → ہارنس; bundle → بنڈل; mesh network → میش نیٹ ورک; operator → آپریٹر; approval gate → منظوری کی شرط; encrypted → رمز بند; argument → آرگومنٹ. These terms are needed here but have no binding glossary rows.

## Width and mechanical checks actually run

Used `/Users/damian/local-operator/.venv/bin/python` with `rich.cells.cell_len` after interpolating `command='/login radient'`:

```text
پہلے AI اکاؤنٹ منسلک کریں: /login radient لکھیں
cell_len=47; budget=74; remaining=27
```

Read back all three files with Python. Actual output:

```text
cli.failures: 4 keys; canonical JSON, key order, placeholders and no bidi controls verified
cli.group_help: 10 keys; canonical JSON, key order, placeholders and no bidi controls verified
tui.setup_cues: 2 keys; canonical JSON, key order, placeholders and no bidi controls verified
Total keys: 16
```

Each file was checked against English key order, placeholder order/names, canonical `json.dumps(ensure_ascii=False, indent=2) + '\n'`, and absence of manual bidi controls. No strings exceeded the declared budget; no truncation or width exception.

No generate.py, catalogue validator, tests or git commands were run, per the explicit task restriction. Catalogue-wide validation and independent linguistic auditing remain for the coordinator/auditor; this is a translation handoff, not an audit or passed verdict. No ledger or generated file was changed.
