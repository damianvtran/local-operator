# Spanish audit — wire.exec — round 2 (w1a3)

- **Verdict: passed** — 0 blocker, 0 major, 0 minor, 2 nit
- Audit id `w1a3`, round 2, 2026-10-29; supersedes `w1a1` (round 1, which bound superseded bytes sha256 `6f9df86c94723eb6…`)
- Auditor: translation-auditor (lopdev team delegation es-w1-auditor-r2), session 3da6fd7a1b13, model **radient/auto** (self-reported; disclosed fallback — see Model)
- Revision `efcfbe812609cbf228c01f11a1fb622f3555b487`, branch feat/i18n-es-wire-runtime
- **The es catalogue is untracked at this revision** (`git status --porcelain` → `??`). The audited bytes are pinned by sha256, not by HEAD: `6f9df86c94723eb655c0deed39d298e073c2b7f120c961ac794a0ce435af96e8`
- Keys: 17 in `en`, 17 in `es`, no en context sidecar (absent by construction)
- Independence: I authored none of this copy, audited none of it before, and edited nothing. My only writes are this directory.

## Verdict reason

All 17 keys read, including a deliberate separate judgement on the three safety-adjacent keys (approval_advisory, goal_only_sets_objective, clear_goal_needs_start), which all pass: the denial promise, every flag remedy and the 'does not start a run' distinction survive intact with the machine literals byte-identical. The epilog rewraps from 14 to 18 prose lines with identical blank-line structure, all five example commands byte-identical and a longest line of 78 cells. Two round-1 nits remain in the final bytes; both are style, neither gates.

## Separate verdict — safety-adjacent copy (deliberate, independent of the namespace verdict)

**PASS.** All three keys PASS. Each was read against the code path that emits it, not only against the English: the Spanish states the same consequence, names the same remedies, and keeps every flag byte-identical. On this evidence the safety-adjacent copy is safe to ship in Spanish. This is a copy verdict: it does not claim any runtime or behavioural test of the approval gate.

Scope: second independent audit of the safety-adjacent copy, judged separately from the namespace verdict.

### wire.exec.approval_advisory — PASS

- English promise: this run cannot ask for approval (no terminal attached), so any tool call needing approval will be denied; remedies --control / --yolo / --tools / tool_approval_mode: auto
- Spanish text: esta ejecución no puede pedir aprobación (no hay terminal conectada), así que cualquier llamada a herramienta que necesite aprobación se denegará
- Emitted at: exec_mode.py:1120, reached only when --control, --yolo, a --tools inventory, a declaring profile, a tty and a saved auto approval are all absent
- Judgement: PASS. The denial is still stated as a consequence of the missing terminal (the condition the emitter actually checks), not as a blanket rule. All four remedies survive with their flags intact and in the same order, 'every tier' -> 'todos los niveles', and the 'tool_approval_mode: auto answers every call the same way for every run' clause is preserved rather than shortened. No reassurance is added and none is removed: the two-line shape with its two-space indent survives.

### wire.exec.goal_only_sets_objective — PASS

- English promise: --goal sets the objective but does not start work (unlike the TUI's /goal); add a prompt, pipe one on stdin (or '-'), or run a loop with --loop N
- Spanish text: --goal fija el objetivo pero no inicia trabajo (a diferencia de /goal de la TUI). Añade un prompt, envíalo por stdin (o '-') o ejecuta un bucle con --loop N.
- Emitted at: exec_mode.py:1179, when --goal is given with no command and no loop
- Judgement: PASS. The refusal still answers the belief that produced it (that --goal also runs, as the TUI's /goal does), which is what the code comment says the message exists for. All three remedies are present and --loop N is byte-identical.

### wire.exec.clear_goal_needs_start — PASS

- English promise: --clear-goal adjusts a run, it does not start one; pair it with --resume SESSION_ID plus a prompt or --loop
- Spanish text: --clear-goal ajusta una ejecución, no la inicia. Combínalo con --resume SESSION_ID y un prompt o --loop.
- Emitted at: exec_mode.py:1174, when --clear-goal is given with no command and no loop
- Judgement: PASS. 'ajusta una ejecución, no la inicia' preserves the adjust-versus-start distinction that is the whole point of the message; --resume SESSION_ID is byte-identical and the 'plus a prompt or --loop' pair is intact.

Residual risk: The advisory's accuracy depends on the emitter's branch, not on the translation: if a future change lets a run reach a tty-less start that is not denied, both locales would misdescribe it equally. Recorded so a later reader does not read this verdict as a behaviour test.

## What was verified

1. SEPARATE SAFETY VERDICT - all three safety-adjacent keys PASS. (1) approval_advisory: 'esta ejecución no puede pedir aprobación (no hay terminal conectada), así que cualquier llamada a herramienta que necesite aprobación se denegará' states the denial as a consequence of the missing terminal, matching the emitter's branch (exec_mode.py:1096-1120 returns the advisory only when there is no tty, no --control/--yolo, no --tools, no profile declaration and no saved auto-approval). All four remedies survive with their literals byte-identical: --control 'retiene las tarjetas para un supervisor', --yolo 'aprueba automáticamente todos los niveles', --tools NAME[,NAME] 'preaprueba las herramientas indicadas y limita el alcance de esta ejecución a esas herramientas', 'tool_approval_mode: auto en la configuración responde a todas las llamadas igual en todas las ejecuciones'. The two-line shape (newline + two-space indent before 'Soluciones:') is preserved.
2. SEPARATE SAFETY VERDICT - (2) goal_only_sets_objective: emitted at exec_mode.py:1179 when --goal is given with no command and no loop. The Spanish keeps all three remedies (a prompt argument, stdin or '-', --loop N) and keeps the contrast with the TUI's /goal that the code comment says the message exists to answer.
3. SEPARATE SAFETY VERDICT - (3) clear_goal_needs_start: emitted at exec_mode.py:1174. '--clear-goal ajusta una ejecución, no la inicia' preserves 'adjusts a run, it does not start one'; the '--resume SESSION_ID' plus 'un prompt o --loop' pairing is intact and SESSION_ID is byte-identical.
4. The 'exec failed: ' prefix is added by code outside the translation unit and is correctly absent from the Spanish values.
5. exec_epilog: the five example command lines are byte-identical to the English (verified line by line, not by eye); the only pre-blank-line difference is the heading 'Examples:' -> 'Ejemplos:'. Prose rewraps 14 -> 18 lines, longest es line 78 cells against the en's own 84, and the blank-line indices are identical.
6. Machine literals survive byte-identically across the namespace: --control, --yolo, --tools, NAME[,NAME], tool_approval_mode, --resume, SESSION_ID, --loop, --loop-goal, --goal, --clear-goal, --agent, --agent-id, --background, --team, --profile, --name, --status, JOB_ID, stdout/stderr, create.json equivalents, {error}, {log_path}, {job_id}, {jobs_path}, {session_id}, {count}, {names}, {failure}.
7. Terminology matches the glossary and the wider locale: trabajo (job), turno (turn), bucle (loop), recibo (receipt, as in tui.notifications.pointer_unread), transcripción, sesión.

## Findings

### es-w1-C10 — nit — carried-unresolved

Keys: `wire.exec.waiting_on_you`

'Waiting on you? lop sessions shows what a run needs' is rendered '¿Algo te espera? lop sessions muestra lo que necesita una ejecución'. The subject flips: the English says a run is waiting on the reader; the Spanish says something awaits the reader. It lands on the same behaviour and is stderr prose with no width budget. Round-1 nit es-w1-C01, not adopted.

**Proposed:** '¿Te está esperando? lop sessions muestra lo que necesita una ejecución'.

### es-w1-C11 — nit — carried-unresolved

Keys: `wire.exec.approval_advisory`, `wire.exec.exec_epilog`

The same --tools promise is phrased two ways inside one namespace: 'limita el alcance de esta ejecución a esas herramientas' (approval_advisory) and 'limita a qué puede llegar la ejecución' (exec_epilog). The second is a calque of 'bounds what the run can reach' and reads as translated. Both keep the promise; style only. Round-1 nit es-w1-C02, not adopted.

**Proposed:** Use 'limita el alcance de la ejecución a esas herramientas' in both.

## Width note on the two over-budget help strings

**Status: not-applicable.** The two over-budget help strings are not in this namespace; no per-key maxCells is declared here either.

## Source-side and display-side risk

- (sidecar) wire.exec has NO context sidecar (en/wire.exec.context.json is absent), so role, tone and any width budget are unknown by construction for all 17 keys. Recorded as a source-side risk; no context was invented for this record.
- (detector) The mechanical newline-count detector fires on exec_epilog (en 20 -> es 24). Judged NOT a defect and not counted as a finding: the epilog is hard-wrapped provenance copy that argparse prints as-is, the extra lines are a legitimate reflow, and the geometry evidence is in audit.log.
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
