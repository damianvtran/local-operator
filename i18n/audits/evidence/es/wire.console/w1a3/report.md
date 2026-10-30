# Spanish audit — wire.console — round 2 (w1a3)

- **Verdict: passed** — 0 blocker, 0 major, 2 minor, 2 nit
- Audit id `w1a3`, round 2, 2026-10-29; supersedes `w1a2` (round 1, which bound superseded bytes sha256 `61f0a242644b1adf…`)
- Auditor: translation-auditor (lopdev team delegation es-w1-auditor-r2), session 3da6fd7a1b13, model **radient/auto** (self-reported; disclosed fallback — see Model)
- Revision `efcfbe812609cbf228c01f11a1fb622f3555b487`, branch feat/i18n-es-wire-runtime
- **The es catalogue is untracked at this revision** (`git status --porcelain` → `??`). The audited bytes are pinned by sha256, not by HEAD: `dc48616e35230a56fe064693f8ce8db34e3710c25f51f3cb5eedee5ab369257a`
- Keys: 8 in `en`, 8 in `es`, 8 in the en context sidecar
- Independence: I authored none of this copy, audited none of it before, and edited nothing. My only writes are this directory.

## Round-2 amendment A1 — verdict re-issued

Date 2026-10-29. This section amends the verdict and two severities only; the audit body below was not rewritten.

- **Trigger:** Manager pushback with a same-code control measurement: the English console labels are also over the 29-cell budget, so the clips are not Spanish regressions.
- **Verified here:** Re-ran the control independently with the repository's own truncate_cells: en 32 -> 'Agent terminals per conversa…' and en 33 -> 'Slots reserved for user term…'. Confirmed. The manager's quoted English cut for max_owner ('per conver…') is two cells shorter than what the helper actually produces; the substance is unchanged.
- **What changed:**
    - es-w1-C01 severity blocker -> minor; status deferred (source-side).
    - es-w1-C02 severity blocker -> minor; status deferred (source-side).
    - Both remediation proposals withdrawn: shortening the Spanish would delete scope the English keeps.
    - finding_counts 2/0/0/2 -> 0/0/2/2; unresolved_blocker_major 2 -> 0.
    - verdict failed -> passed; result failed -> clean-with-findings.
- **What did not change:** Every other finding in this namespace (C08, C09 nits) and every other namespace's record. The audit body above was not rewritten.
- **Audit lesson:** A width finding must be run through the same control as the target before it is called a translation defect: measure the source through the identical budget first.

## Verdict reason

All eight keys read. Meaning, ranges, defaults and the never-terminate guarantees are exact, and no placeholder, digit, dash or machine literal was lost. AMENDED (A1): the two label clips this record originally raised as blockers are pre-existing and locale-independent — the same truncate_cells(label, 29) cuts the ENGLISH labels too (32 and 33 cells), and on user_reserved.label the Spanish is one cell SHORTER than the English. They are reclassified minor and deferred to the source. What remains is two nits. Verdict: passed.

## What was verified

1. Every range and default is byte-identical with the en dash intact: 32–256 / 128, 1–32 / 8, 4–32 / 16, 'un cuarto del máximo de terminales activas'.
2. The three behavioural promises survive without strengthening: completed history does not consume process slots, changes affect future starts, running work is never terminated; agent starts cannot consume reserved slots; existing work is never stopped.
3. 'terminales activas' renders 'live terminal(s)' consistently across all four strings, and 'ranuras' renders 'slots' consistently.
4. For the record, the three help strings (187, 125, 171 cells) also exceed the detail band, but so does their English source (151, 102, 133) and the sidecar pins no maxCells here, so the translation is not what makes them clip; that is a source-side budget gap, not a translation defect.

## Findings

### es-w1-C01 — minor — deferred — source-side: the English labels are 32/33 cells against the 29-cell budget, so the clip is pre-existing and locale-independent

Keys: `wire.console.max_owner.label`

Current 'Terminales de agente por conversación' is 37 cells against the renderer's 29-cell row-label budget (_LABEL_BUDGET = 34 - 2 - 2 - 1 = 29, settings_view.py:434). The row is a registry Setting (settings_io.py:4785-4788, key console.max_owner, section 'console') and section 'console' is always visible (team_bridge/availability.py:86-87 returns True for every name except BRIDGE_SECTION), so the row always renders. settings_view.py:4020 passes every row label through truncate_cells(setting.label, 29), which cannot wrap. Measured with the repository's own truncate_cells: 'Terminales de agente por con…'. Round 1 audited this namespace and did not measure this label.

**Same-code control** (the same helper the renderer uses: local_operator.tui.widgets.tool_card.truncate_cells(x, 29)):

- en `Agent terminals per conversation` — 32 cells -> `Agent terminals per conversa…` (clips: True)
- es 37 cells -> `Terminales de agente por con…` (clips: True)
- Spanish minus English: +5 cells

BOTH locales clip. The defect is source-side: the English label is 32 cells against a 29-cell budget. Spanish adds 5 cells, so it loses more of the tail, but it does not introduce the clip and a Spanish-only shortening would not remove it from the screen.

Note: The pushback quoted the English cut as 'Agent terminals per conver…'; the real helper yields 'Agent terminals per conversa…' (29 cells). Substance unchanged.

**Proposed:** WITHDRAWN: 'Terminales de agente' (20 cells) should not be applied — it buys a clip-free label by deleting the per-conversation scope the setting exists to express, a loss the English does not make. Source-side fix: shorten the English label to <=29 cells or raise _LABEL_BUDGET, then let Spanish follow that decision.

### es-w1-C02 — minor — deferred — source-side: the English labels are 32/33 cells against the 29-cell budget, so the clip is pre-existing and locale-independent

Keys: `wire.console.user_reserved.label`

Current 'Ranuras de terminales de usuario' is 32 cells against the same 29-cell budget; it renders as 'Ranuras de terminales de usu…'. The row is settings_io.py:4796-4799 (key console.user_reserved, section 'console'), so it clips on the same always-visible section. Round 1 raised this key as a nit about expansion percentage, under the mistaken premise that no budget applied: its own proposed replacement, 'Ranuras reservadas para el usuario', is 32 cells and would clip identically. The clip is now measured, not inferred.

**Same-code control** (the same helper the renderer uses: local_operator.tui.widgets.tool_card.truncate_cells(x, 29)):

- en `Slots reserved for user terminals` — 33 cells -> `Slots reserved for user term…` (clips: True)
- es 32 cells -> `Ranuras de terminales de usu…` (clips: True)
- Spanish minus English: -1 cells

BOTH locales clip, and Spanish is ONE CELL SHORTER than English (32 vs 33). Spanish is strictly no worse than the source: there is no locale regression here at all, only a pre-existing source-side budget breach.

**Proposed:** WITHDRAWN: 'Ranuras de usuario' (18 cells) should not be applied — it drops 'terminales', which is the object the row governs. Source-side fix only: shorten the English label to <=29 cells or raise _LABEL_BUDGET.

### es-w1-C08 — nit — new

Keys: `wire.console.max_owner.help`

English 'the global live allowance' becomes 'la cuota global de terminales activas'. 'cuota' is this locale's rendering of 'quota' elsewhere in the same wave (wire.settings.account_policy.headroom.description, 'la cuota restante en caché'), so English's allowance/quota distinction collapses onto one Spanish word. Meaning is recoverable from context.

**Proposed:** 'el cupo global de terminales activas' - 'cupo' is the capacity sense and leaves 'cuota' to quota.

### es-w1-C09 — nit — new

Keys: `wire.console.description`

'El historial completado no consume ranuras de proceso' renders 'Completed history does not consume process slots'. It is understandable but 'historial completado' reads as a completed history rather than a history of completed terminals; a native phrasing names what was completed.

**Proposed:** 'El historial de terminales completadas no consume ranuras de proceso' - same facts, the completed thing named.

## Width note on the two over-budget help strings

**Status: not-applicable.** The two over-budget help strings are not in this namespace; no per-key maxCells is declared here either.

## Source-side and display-side risk

- (sidecar) wire.console.context.json declares no maxCells for any key, including the three labels the renderer truncates at 29 cells. The translator therefore had no budget signal for the two labels that clip (C01, C02). This is a sidecar gap that will produce the same defect in every future locale.
- (display) No value in this namespace displayed as a redaction marker, so no sanctioned read_public_copy re-read was needed to judge any of the copy. Verified source-side instead: the reader serves the HEAD blob and REFUSES a working-tree difference or an untracked file (local_operator/tools/public_copy.py:487-496), so it cannot read these untracked es bytes at all; and its catalogue reader rejects any value that is not a string (public_copy.py:343-345), so *.context.json sidecars are unreadable through it by construction. Both are recorded as source-side risks, never worked around.

## Limitations

- LLM semantic judgement, not native-speaker attestation.
- Static copy audit only: no rendered-frame, runtime-service, eligibility or release claim. Width verdicts are derived from the renderer source at the audited revision and from the repository's own truncate_cells, not from screenshots.
- The eight es catalogues are UNTRACKED at efcfbe812609; every record binds on-disk sha256 values rather than a committed blob.
- The audit model is an auto-routing alias; the concrete backend is not identifiable from the session header.
- Evidence files were written but never committed: this role performs no git writes.

## Model

SELF-REPORTED from this session's own model-switch notices; not independently observable. The request opened on anthropic/claude-sonnet-5-5; three provider-failure switches fired before any audit work began and the entire audit ran under radient/auto, an auto-routing alias whose concrete backend the session header does not identify. No pinned model is claimed.

Switch chain: anthropic/claude-sonnet-5-5 -> openai/gpt-6-astra -> kimi/k3.1-pro -> radient/auto; each transition was a session model-switch notice reading "provider failure: quota".
