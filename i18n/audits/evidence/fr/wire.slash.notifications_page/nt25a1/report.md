# Independent French audit — `wire.slash.notifications_page`

Date: 2026-10-25. Auditor: translation-auditor / `766adb539e6a` / actual model `openai/gpt-6-astra` (runtime session header; no fallback). Independent of translator coder / Opus5.5 (translator provenance supplied in delegation).

## Verdict

**clean-with-findings / passed**, 1 English/French key pairs and 1 context entries read. Zero blockers/majors; 1 minor findings. This pass never changed catalogue, code, ledger or git.

Requested pin: `3c16a9e77c63dce7082af6137e151421b6cbf2be`. Observed checkout moved to `329b5c581784df1082dbd0630c0f814c7a3c0c8f`; the probe compared each English/French/context file to `git show <pin>:<path>` and proved byte identity. Runtime parser/formatter source also remains byte-identical to the requested pin.

## Actual checks

`audit.log` retains the real command, stdout, stderr and exit status. The initial probe failed before the assertions because `wcwidth` is not installed in this checkout. No dependency was installed; recovery used the already-installed `rich.cells.cell_len`. The subsequent core runtime probe exited **0**. Both attempts are retained, not rewritten as uninterrupted success. Reproduction source: `recheck.py.txt`.

- `parse_message` and `message_arguments`: 2 real EN/FR parses; exact placeholder names/types preserved.
- `render_message`: 1 real French outputs. Where applicable, `/synthetic/config/[alerts] & {literal}` remains verbatim, including braces.
- No plural/select syntax or markup appears in this namespace; plural selector checks confirm none.
- Negative controls reject malformed ICU and distinguish a dropped placeholder.
- No manual bidi controls found. French is LTR; RTL review is not applicable.
- French vous register, apostrophes and U+202F punctuation checked in the catalogue bytes.

- Current literal `read` occurs exactly once. Existing help measures **46 cells / maxCells 50** using Rich. The proposed consistent replacement measures **50/50**, with literal `read` still intact. Neither number claims rendered web/native fit.

## Finding

### FR-CORE-1 — MINOR — `wire.slash.notifications_page.help`

- Current: `Questions et infos des agents ; read : tout lu`
- Proposed: `Demandes et mises à jour d’agents ; read : tout lu`
- Reason: This command describes the same notification kinds as the UI but calls them Questions / infos instead of the UI’s Demandes / mises à jour; align discoverability vocabulary without changing literal read.
- Evidence: rich.cells.cell_len: current 46 cells; proposed 50 cells; sidecar maxCells=50. UI filter.kind_asks=Demandes and filter.kind_updates=Mises à jour were read in the audited target catalogue.
- Disposition: proposed; not edited; non-blocking cross-surface terminology finding

## Side-by-side inventory

### `wire.slash.notifications_page.help`

- EN: Asks and agent updates; read marks all read
- FR: Questions et infos des agents ; read : tout lu
- Context: `{"role": "command description line in /help, the completion picker and the desktop command palette", "tone": "terse", "maxCells": 50, "note": "The /notifications command: opens the Notifications page (open asks waiting on the operator plus updates agents posted with notify). `read` is a typed word the command reads, keep it verbatim. Never call the page an 'Inbox' in product copy. English-only child namespace, kept out of the audited wire.slash catalogue so its locale hashes do not move."}`

## Input hashes

- en: `local_operator/i18n/catalogues/en/wire.slash.notifications_page.json` — `9440f547bbf4e982c949202fd5ddb78a784a3d6da0f7e88694eb9ffdc3bc6594`
- en_context_sidecar: `local_operator/i18n/catalogues/en/wire.slash.notifications_page.context.json` — `2974cbbc68f15d61f45c8a79d99185ddb8b13afeb2f1920d592f752accdf45e0`
- target: `local_operator/i18n/catalogues/fr/wire.slash.notifications_page.json` — `b7e183e9efbd5a40c1819556433bca46ed5c98029fefd152e6a1352dca4b3952`
- glossary: `i18n/glossary.md` — `ef07b446668778822d0ff4dcd7b380543456ddf7b08e060dce5818cb8fd6fdcd`
- style_guide: `i18n/style-guides/fr.md` — `b916f3d1c30b0a2e18fc94ca51d130f60e86b3b17b1918aa51f5c5da6d52ec39`

## Limits

This is an independent LLM source/linguistic audit with actual parser/format execution, not a human/native-speaker audit, screenshot review, shipped-locale assertion or release approval. Rendered-fit evidence remains the later design/QA role’s scope.
