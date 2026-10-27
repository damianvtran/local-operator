# Independent French audit — `wire.errors.notifications`

Date: 2026-10-25. Auditor: translation-auditor / `766adb539e6a` / actual model `openai/gpt-6-astra` (runtime session header; no fallback). Independent of translator coder / Opus5.5 (translator provenance supplied in delegation).

## Verdict

**clean / passed**, 4 English/French key pairs and 4 context entries read. Zero blockers/majors; 0 minor findings. This pass never changed catalogue, code, ledger or git.

Requested pin: `3c16a9e77c63dce7082af6137e151421b6cbf2be`. Observed checkout moved to `329b5c581784df1082dbd0630c0f814c7a3c0c8f`; the probe compared each English/French/context file to `git show <pin>:<path>` and proved byte identity. Runtime parser/formatter source also remains byte-identical to the requested pin.

## Actual checks

`audit.log` retains the real command, stdout, stderr and exit status. The initial probe failed before the assertions because `wcwidth` is not installed in this checkout. No dependency was installed; recovery used the already-installed `rich.cells.cell_len`. The subsequent core runtime probe exited **0**. Both attempts are retained, not rewritten as uninterrupted success. Reproduction source: `recheck.py.txt`.

- `parse_message` and `message_arguments`: 8 real EN/FR parses; exact placeholder names/types preserved.
- `render_message`: 4 real French outputs. Where applicable, `/synthetic/config/[alerts] & {literal}` remains verbatim, including braces.
- No plural/select syntax or markup appears in this namespace; plural selector checks confirm none.
- Negative controls reject malformed ICU and distinguish a dropped placeholder.
- No manual bidi controls found. French is LTR; RTL review is not applicable.
- French vous register, apostrophes and U+202F punctuation checked in the catalogue bytes.

## Linguistic assessment

All four keys are clean. `store_busy` preserves transient retry and no-write consequences. `store_full` preserves the disk-space remedy and which volume to inspect. `store_unavailable` correctly makes retry insufficient and directs inspection of the supplied path/disk. `unknown_item` retains Notifications as the page name, never Inbox. No glossary, register, placeholder, plural or idiom finding. These are full-sentence errors, not fixed-cell controls; rendered wrapping belongs to later design/QA.

## Side-by-side inventory

### `wire.errors.notifications.store_busy`

- EN: The notifications store is busy right now, so nothing changed. Try again in a moment.
- FR: Le stockage des notifications est occupé pour le moment, donc rien n’a changé. Réessayez dans un instant.
- Context: `{"role": "error sentence painted by the desktop Notifications page", "tone": "calm, retryable", "note": "notifications.db met lock contention through its retry budget. Nothing was written. The client also reads the machine code store_busy to offer a retry. 'notifications store' is the storage behind the Notifications page; never call it an 'inbox'."}`

### `wire.errors.notifications.store_full`

- EN: This computer is out of disk space, so nothing changed. Free some space on the volume holding {where}, then try again.
- FR: Cet ordinateur n’a plus d’espace disque, donc rien n’a changé. Libérez de l’espace sur le volume qui contient {where}, puis réessayez.
- Context: `{"role": "error sentence painted by the desktop Notifications page", "tone": "plain, actionable", "note": "{where} is the config directory the store lives in (a path, keep it verbatim). The volume is full; retrying alone will not help."}`

### `wire.errors.notifications.store_unavailable`

- EN: The notifications store could not be read or written. Retrying will not help; check {where} and the disk it is on.
- FR: Le stockage des notifications n’a pas pu être lu ni écrit. Réessayer n’y changera rien ; vérifiez {where} et le disque qui le contient.
- Context: `{"role": "error sentence painted by the desktop Notifications page", "tone": "plain", "note": "{where} is the config directory (a path, keep it verbatim). The store could not be opened or is damaged; retrying will not help. 'notifications store' is the storage behind the Notifications page; never call it an 'inbox'."}`

### `wire.errors.notifications.unknown_item`

- EN: That notification is no longer in Notifications.
- FR: Cette notification ne figure plus dans Notifications.
- Context: `{"role": "error sentence for an item id the inbox does not hold", "tone": "neutral", "note": "Shown when a detail link names an item that was pruned, deleted with its conversation, or an ask that is no longer open. 'Notifications' is the page's name (the sidebar destination); never render it as 'Inbox' — that word belongs to Team Bridge (design §1)."}`

## Input hashes

- en: `local_operator/i18n/catalogues/en/wire.errors.notifications.json` — `f02bf1bc28f9c9e549a3ad8c34ead880315f4b39d25c36a869a1b3c5110f362c`
- en_context_sidecar: `local_operator/i18n/catalogues/en/wire.errors.notifications.context.json` — `07ad1e5da403a5fccde5648f04cd10bf1263cabc50f5e1804644630c7ce13106`
- target: `local_operator/i18n/catalogues/fr/wire.errors.notifications.json` — `a8b19b6cd569d4525bc84eb2e08e206b0d80523b9c760fa9bc8e5a719b672122`
- glossary: `i18n/glossary.md` — `ef07b446668778822d0ff4dcd7b380543456ddf7b08e060dce5818cb8fd6fdcd`
- style_guide: `i18n/style-guides/fr.md` — `b916f3d1c30b0a2e18fc94ca51d130f60e86b3b17b1918aa51f5c5da6d52ec39`

## Limits

This is an independent LLM source/linguistic audit with actual parser/format execution, not a human/native-speaker audit, screenshot review, shipped-locale assertion or release approval. Rendered-fit evidence remains the later design/QA role’s scope.
