# ru / wire.settings — remediation delta audit (le28r2-remed)

Remediation commit `4d38a1d4b8e9a36fa0a38f4833ecf86263da45db` (parent `9495f8a6c3bf2b5438ef5c33d7274cf519435dda`). Verdict: **PASS** for the 4 corrected key(s).

Scope is the delta only: the unchanged keys keep the `le28r2` judgement; this round does not re-audit them.

## Keys re-audited

### `wire.settings.subagents_max_running.label` — fixed

Now: Макс. фоновых заданий

job → «задание» per the ru guide; «задача» stays for todos.

### `wire.settings.supplements_maxcostusd.help` — fixed

Now: Требует включённых графиков. Мягкий потолок расходов на одно задание.

job → «задание».

### `wire.settings.supplements_timeouts.help` — fixed

Now: Требует включённых дополнений. Отказаться от всего задания по истечении этого срока.

job → «задание».

### `wire.settings.librarian_enabled.help` — fixed

Now: Считывается при запуске и в /v1/desktop/librarian; если отключено, его сессия не создаётся.

session → «сессия»; masculine «его» agrees with «Библиотекарь» (m.), as in the sibling librarian keys. The first remediation used «её» (mirroring English "her" and the feminine Aida key); corrected in PR #3123 review round 1 (F1), see `post_review_delta` in `findings.json`.

## Checks

- Key set still equals the English source key set; ICU argument parity checked by `scripts/i18n/check.py --catalogues-only` on this tree.
- Routed `read_public_copy` receipt for the corrected catalogue at `4d38a1d4b8e9a36fa0a38f4833ecf86263da45db` (session `c754bf63ab2a`, 2026-10-29T00:24:37Z); its digest was recomputed from the commit blob and matches.
- No English source, context sidecar, key module or generated artifact changed in the remediation commit.

## Post-review correction (PR #3123 review round 1, F1)

Commit `bd47fce698d3d498c503f047216e4e442f7d0d67` changes exactly one word in one value (`wire.settings.librarian_enabled.help`: «её сессия» → «его сессия»); ru `wire.settings.json` sha256 `5185479b…` → `9156360167e72197d97bfb8caeaa0c7e2795b03d0dceeb7195d24b8f9d1f40bd` (recomputed from the committed blob, 122294 bytes). The `Now:` text above and the 4 verdicts describe the corrected bytes.

Independent delta audit (job `772c338410e1`, translation-reviewer, `anthropic/claude-sonnet-5-5`, no fallback observed; not the translator, not a native speaker): **PASS, 0 findings**. Sibling pronouns read: `librarian_cadence_min_gap_minutes.help` «его», `librarian_cadence_max_extra_per_day.help` «он», `librarian_name.help` «его», `librarian_enabled.label`/`librarian.title` «Библиотекарь» (m.); `aida_enabled.help` «её» is correct (Aida is feminine). The routed receipt (`read_public_copy`, key `wire.settings.librarian_enabled.help`, session `a9c861a89848`) was compared with the direct read by eye and content, not by a mechanical byte diff; the byte claim rests on the implementer's sha256 above. Caveat: the auditor found no ru glossary file; `i18n/glossary.md` line 44 lists session → сессия.

The `route-receipt.json` in this bundle still binds the earlier remediation bytes (`c07e9c46…` at `4d38a1d4b8e`); it is unchanged and is not a receipt for the corrected file.
