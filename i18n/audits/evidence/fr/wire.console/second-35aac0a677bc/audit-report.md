# French second-reader audit: wire.console

**PASSED — independent staged linguistic/mechanical verdict for 8/8 keys.** No BLOCKER, MAJOR, MINOR or NIT findings. This is a separate namespace decision, not blanket approval of the three contributions or interface-wide completeness.

Pinned source: `0e7cde33ca8008d561e74f4aed774aa5f76ff7d5`. Auditor `i18n-auditor`, session `35aac0a677bc`; translator session `7b2ba4ed501d`; coder actual session `eae0961e4e78` (not job `196c3a94b346`). No translator/coder role performed by this reader and no other new reader report inspected.

**Actual served model:** `openai/gpt-6.1-sol`, from own completed persisted usage for probe authorship/execution and source-inventory derivation in `provenance.json`. Known own route notices selected Opus then OpenAI after primary Sonnet quota failures; route notices do not establish completed Opus usage. No requested-header or no-fallback inference.

## Input identity

- en: `1a591fbc1f6a939651cfe9bc5c4ff619a5a4ade2099cc9cfb348534d09a0011b`
- context: `453aadc8095e93122719cef9b8da2ed9f8b57170449804e6af716fe684ef2dd9`
- fr: `7f499e6ab9f8db5c9953e6782357d4da38554ac89250f6fc6616b77069d4c8b2`

Source and context packet bytes were compared directly with pinned git blobs. All original translator-packet manifest entries were rehashed; style/glossary snapshots are byte-identical to the French guides already read in this session. Original packet bytes remain untouched.

## Native assessment of every key

### `title`
Current: Capacité de la console
Assessment: Capacité de la console naturally names the capacity settings section.

### `description`
Current: Admission des terminaux actifs. L’historique des terminaux terminés n’occupe pas d’emplacements de processus. Les modifications s’appliquent aux prochains démarrages, sans jamais arrêter les tâches en cours.
Assessment: Admission of live terminals, completed history consuming no process slots, future starts only and never stopping current work all preserved.

### `max_live.label`
Current: Plafond de terminaux actifs
Assessment: Plafond de terminaux actifs is a concise ceiling label rather than a memory cap.

### `max_live.help`
Current: 32–256 ; 128 par défaut. La mémoire gérée, les cellules de terminal et les ressources natives disponibles peuvent réduire le nombre de terminaux admis. Ce réglage ne limite pas la mémoire des processus enfants.
Assessment: Inclusive 32–256, default 128 intact. Managed memory, terminal cells and native resources may lower admissions. Explicitly does not cap child-process memory.

### `max_owner.label`
Current: Terminaux d’agents par conversation
Assessment: Per-conversation agent-terminal scope retained; not per user or entire application.

### `max_owner.help`
Current: 1–32 ; 8 par défaut. Chaque conversation partage le quota global de terminaux actifs ; les terminaux terminés ne comptent pas.
Assessment: Inclusive 1–32, default 8, shared global live allowance, and exclusion of completed terminals all preserved.

### `user_reserved.label`
Current: Emplacements réservés aux terminaux utilisateur
Assessment: Reserved slots belong to user terminals, distinct from agent terminals.

### `user_reserved.help`
Current: 4–32 ; 16 par défaut, au maximum un quart du plafond de terminaux actifs. Les démarrages d’agents ne peuvent pas utiliser ces emplacements. Les tâches existantes ne sont jamais arrêtées.
Assessment: Inclusive 4–32/default 16 and maximum one quarter of live ceiling retained; agent starts cannot consume reserved slots and existing work is never stopped.

## Actual scoped mechanical execution

The existing repository `scripts.i18n.check.check_catalogues()` ran against this namespace alone, with its retained English/French catalogue and original English context. Independently authored checks validated duplicate-free JSON, exact key/order parity, names/types via the core ICU parser, numeric literals, technical tokens, no `Cf`, and all supplied Rich-cell limits. Core `render_message` exercised every string; opaque syntax-looking arguments were inserted as data, not reparsed. These are parser/formatter calls, not SDK guard or UI execution.

Actual exit: **0**, validator problems: **0**; rejected controls: **4**. Full stdout: `mechanical.json`; stderr: `mechanical.stderr.log` (empty). The fixture was restored and the clean validator rechecked. No plural/select exists, so CLDR arms are **not applicable**.

Invocation (each namespace root has its own retained `verify.py`):

```bash
for name in ${!CMUX_@}; do unset "$name"; done
cd ~/local-operator
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B <this-audit-root>/verify.py
```

The three scoped invocations ran serially in one tool call bounded to **256 MiB / 90 seconds**; all three exited 0. This is not a broad suite or scanner.

### Numeric and safety judgment

Ranges/defaults are exact: **32–256 / 128**, **1–32 / 8**, **4–32 / 16**; reservation cap is **one quarter**, not a quarter of remaining slots. Current work is never stopped; completed history does not consume live slots; child-process memory is explicitly not capped by this setting. No numeric width caps are present; no native-fit pass is inferred.

## Explicit remaining proof and disposition

- **NOT RUN:** native/app fit, screenshots, live language switch, pseudo rendering, SDK guards/settings requests or design review. RTL visual rendering is not applicable to French and was not executed; absence of hidden controls was checked.
- Static Rich widths and source/ICU inspection are **not** native layout or real-path integration proof.
- No source/target/repository/ledger edits, generation, activation, PR/comment, build/install/scanner/full suite. No global completeness, release eligibility or English-free-interface claim.
- Original per-namespace bundle is retained read-only in session scratch for coordinator promotion into canonical tracked `i18n/audits/evidence/fr/<namespace>/...`; it is not claimed on main. Previous policy bundles/addendum and earlier translation packets are untouched.

