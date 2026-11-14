# es · Slice C1 · ROUND 2 delta audit — `c1esa2`

- **Verdict: clean-with-findings** — status `passed` (0 blocker, 0 major, 0 minor, 1 nit).
- Locale **es** (neutral Spanish). Group: 22 keys — 15 `cli.config.*`, 7 `tui.session_move.*`.
- Reviewer: `i18n-auditor` (lopdev team), independent audit id `c1esa2`, session `95b953e7b8fd`.
  Served model `deepseek/deepseek-flash` (session model header; no fallback, no model-switch notice).
  I did not translate this copy and did not edit the translation; a translation/MT service was used
  for no comparison or back-translation — every judgement below is a reading of en against es plus
  a recount of the shipped es catalogue.
- Round 1: `audit-es-A.md` (3 minor, 3 nit) and `audit-es-B.md` (2 minor, 2 nit), 10 finding
  instances across the group. Translator v2 dispositions in `es.v2-changes.md`.
- Source: `keys.json` → the shipped en catalogues it mirrors,
  `local_operator/i18n/catalogues/en/{cli.config,tui.session_move}.json` @ worktree
  `~/local-operator/.worktrees/i18n-c1` HEAD `b188079f32e6` (sha256 `40649e74…fda5ac` /
  `f4dd682b…a326a913e`). Target: `c1-prep/es.v2.json` (2656 bytes), re-materialised byte-for-byte
  as `target-audited.json` beside this report (same size, same values, same key order); the shard
  itself sits in another session's scratchpad that this session may not read from a shell, so the
  retained copy — not the shard — is what the recorded sha256 covers.
- Round 1 approved keys are **not** re-audited. Scope is the 9 changed keys plus the unchanged
  neighbours they could now contradict, and the two safety lines the brief names.

## Delta: 9 changed keys (v1 → v2), all 9 judged

| key | v1 → v2 | judgement |
|---|---|---|
| `cli.config.cleared` | `Valor de {key} eliminado: vuelve a usarse el predeterminado` → `Se borró el valor de {key}: vuelve a aplicarse su valor predeterminado` | round‑1 defect fixed (bare noun gone); **R2‑N1** nit |
| `cli.config.updated` | `{key} actualizado a {value}` → `Ajuste {key} actualizado a {value}` | clean |
| `cli.config.running_sessions_unchanged` | `escribir en la configuración no puede reducir sus requisitos de aprobación … quieras desactivarlos` → `un cambio de configuración no puede aflojar sus requisitos de aprobación … quieras que no se pida aprobación` | clean (safety intact, term aligned) |
| `cli.config.list_desc_max_learnings_history` | `entradas de aprendizaje` → `aprendizajes` | clean |
| `cli.config.list_desc_classification_notice` | `no muestran avisos en el chat; … figuran en el registro` → `no dicen nada en el chat; … quedan en el registro` | clean |
| `tui.session_move.ambiguous` | `<dispositivo>` → `<dispositivo par>` | clean |
| `tui.session_move.to_needs_device` | `<dispositivo\|local>` → `<dispositivo par\|local>` | clean |
| `tui.session_move.cannot_wait` | `desde una terminal` → `desde un shell` | clean |
| `tui.session_move.cli_no_answer` | `código {code}` → `código de salida {code}` | clean |

No unchanged neighbour contradicts a changed key: no other key in the group says
`entradas de aprendizaje`, `no muestran avisos`, `desde una terminal`, a bare `código {code}`, or
`<dispositivo>` — each of those spellings occurred **only** in the key that was revised.

## Findings

### R2-N1 — NIT — `cli.config.cleared` (wordiness / doubled noun)

Current: `Se borró el valor de {key}: vuelve a aplicarse su valor predeterminado`

The round‑1 nit (the bare noun `el predeterminado`) is fixed, and the preterite receipt
(`Se borró`) is justified by shipped es (`Se eliminó {name} ({id}).`,
`Se eliminó el sonido seleccionado; {default} vuelve a estar seleccionado.`). What is left is
wording: `valor` appears twice in one short receipt, and `vuelve a aplicarse su valor
predeterminado` is a heavier way to say the same thing than the language needs. It also costs
width in a one-line receipt: 44 → 70 cells (en 44), +59 % — second only to `cli_no_answer`
(36 → 64, +78 %) among the group's short single-line strings.

Suggested: `Se borró el valor de {key}: vuelve a su valor predeterminado` (60 cells vs 70, no repetition,
same meaning, and it keeps the receipt's terse register).

Not scored, and not a finding: if the group wants one receipt shape, the three other
`lop config` confirmations are participle-fronted (`Nuevo archivo … creado`, `Archivo … abierto`,
`Ajuste … actualizado`) while this one is a reflexive preterite. Both shapes ship in the es
catalogue, so this is a preference for the wave to settle, not a defect.

## Mechanical pass on v2 (executed — `checks.py.txt`, output in `audit.log`)

1. `python3 scripts/i18n/check.py` → exit 0:
   `check.py: ok — 217 file(s) scanned, 2540 literal(s) at or below the baseline, catalogues
   consistent; 83 advisory finding(s) (report-only until activated)`.
2. `checks.py.txt` (worktree venv, `rich.cells.cell_len` + the repo's own
   `message_arguments`/`parse_message`/`render_message`) → `mechanical problems: none`:
   - 22/22 keys, key set **and order** identical to the en catalogue; none added, none dropped;
     both sides parse as JSON.
   - placeholder name+type sets identical for every key; no ICU plural/select message exists in
     this group, so no CLDR category question arises; every value parses and renders.
   - backticked shell spans byte-identical to en (2 of them: `cannot_wait`, `deadline_exceeded`).
   - typed tokens present verbatim: `/approvals auto`, `/move`, `/move --to`, `--to`, `--keep`,
     `--queue`, `--drop-session-roles`, `--wait`, `{flag}`, and the literal accepted value
     `local` inside `<dispositivo par|local>`; the angle brackets and the `|` are kept.
   - no empty values, none identical to English (all 22 differ), no edge or double spaces, no tabs,
     no invisible/format characters (`Cf`/`Zs` other than U+0020) — so no bidi control anywhere.
   - widths: es is +0 to +91 cells over en; the three longest rendered strings are
     `running_sessions_unchanged` 148→239, `list_desc_detail_length` 111→126 and
     `deadline_exceeded` 102→123. **No `maxCells` is declared for any key in either
     en context sidecar** (grep count 0/0) and every role says the line may wrap; the
     `lop config` receipts print bare and wrap, the list rows wrap inside the box, and the TUI
     notices are `_system_notice` blocks that fold at their column (`fold_width`). No fit finding —
     recorded, not scored.

## Safety lines (both verified against the source notes and the rendered output)

- `cli.config.running_sessions_unchanged` — all four claims survive: (1) running sessions are
  unchanged (`Las sesiones en ejecución no cambian`); (2) a config write cannot loosen them
  (`un cambio de configuración no puede aflojar sus requisitos de aprobación` — `sus` now resolves
  to the running sessions, the round‑1 antecedent strain is gone); (3) `/approvals auto` must be
  typed **in each session** (`en cada sesión`); (4) new sessions open at auto
  (`Las sesiones nuevas se abren en modo auto`). `/approvals auto` and the mode word `auto` are
  verbatim. Nothing claims the write loosens anything, and nothing tells the user to stop
  approving.
- `tui.session_move.deadline_exceeded` (unchanged by this round, re-checked because it is a safety
  line) — reports an **unknown** outcome: `no terminó a tiempo … para consultar hasta dónde llegó`.
  It says neither that the move failed nor that it succeeded, and the re-run command
  `` `lop sessions move {session} --to {to}` `` is byte-identical to en.

## Adjudication of round-1 dispositions (`es.v2-changes.md`)

The disposition list says **0 declined**; I re-read both round‑1 reports and confirmed all ten
finding instances (A: M1, M2, M3, N1, N2×2, N3; B: M1, M2, N1, N2) and checked each claim against
v2 — **10 of 10 landed**. There is therefore no declined finding to overturn or uphold on the
merits, and I invent none.

- **Upheld — one partial non-adoption.** The translator did not copy auditor A's suggested wording
  `no puede aflojar la de una sesión en ejecución`, writing `no puede aflojar sus requisitos de
  aprobación` instead. A's suggestion is genuinely ungrammatical as written (`aflojar la` has no
  antecedent; shipped says `aflojar una`), so the deviation is correct: the finding is fixed and
  the wording is the translator's, better one. **Upheld.**
- **Overturned — none.**

## Translators' recounts, independently reproduced

The disposition file bases several term decisions on counts of the shipped es catalogue. I
recounted every one myself over all 3978 shipped es strings (`catalogues/es/*.json`), and each
number reproduces exactly:

| claim in `es.v2-changes.md` | my recount |
|---|---|
| `aflojar` 2 keys / 2 occurrences | 2 / 2 (`approvals.description`, `tool_approval_mode.help` — the second is `Aflojarlo`, hence the case-insensitive count) |
| `aprendizajes` 1 / 1 | 1 / 1 (`max_learnings_history.label`) |
| `no dicen nada` 1 / 1 | 1 / 1 (`classification_notice.help`) |
| `shell` 20 keys / 23 occurrences | 20 / 23 |
| `código de salida` 2 / 2, plural also ships, `código de retorno` 0 | 2 / 2, plural 3 strings, 0 |
| `dispositivo par` 2 keys (naive substring 3) | word-bounded 2 (`sessions_help.peer`, `move_to`); substring 3, the third being `dispositivo para` — the exclusion is right |
| `obsoleto` 16 keys / 16 | 16 / 16 |
| `proveedor` 67 / 72 · `alojamiento` 7 / 8 | 67 / 72 · 7 / 8 |

The `Ajuste {key}` choice also has shipped support: `ajuste` for a config setting appears in 40
shipped strings (`ajuste propio de esta sesión`, `Ajustes`, `el ajuste no se guardó`, …).

## Second independent pass

The group is high-visibility (settings/config output and the error tier), so I re-read all nine
deltas from scratch against en, `style-guides/es.md` and `glossary.md`. Effect: the pass‑1
objection to the preterite receipt in `cleared` was dropped once the shipped
`Se eliminó …` receipts were found, the `shell` objection in `cannot_wait` was dropped once the
20-string shipped count was confirmed, and `R2-N1` (the doubled `valor`) surfaced only in pass 2.
No blocker or major appeared in either pass.

## Not executed / not claimed

No native-speaker review. No rendered terminal screenshot (no `maxCells` declared; widths measured
with `rich.cells.cell_len` instead). No translation or MT service used. The v2 shard's own bytes
could not be hashed from this session (another session's scratchpad, refused by the session
guardrail); the recorded target hash is over the retained byte-identical copy. Catalogue
integration and a re-run of `scripts/i18n/check.py` against the *integrated* catalogue remain the
manager's step — this report audits the shard, not a merged catalogue.
