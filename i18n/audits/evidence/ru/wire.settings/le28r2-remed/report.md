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

Now: Считывается при запуске и в /v1/desktop/librarian; если отключено, её сессия не создаётся.

session → «сессия»; feminine «её» matches the English "her session" and the aida precedent.

## Checks

- Key set still equals the English source key set; ICU argument parity checked by `scripts/i18n/check.py --catalogues-only` on this tree.
- Routed `read_public_copy` receipt for the corrected catalogue at `4d38a1d4b8e9a36fa0a38f4833ecf86263da45db` (session `c754bf63ab2a`, 2026-10-29T00:24:37Z); its digest was recomputed from the commit blob and matches.
- No English source, context sidecar, key module or generated artifact changed in the remediation commit.
