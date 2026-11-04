# Audit record — es / wire.exec / w1a1

- **Verdict: passed** — 0 blocker, 0 major, 0 minor, 2 nit
- Audit date: 2026-10-29
- Auditor: `translation-auditor (lopdev team)`, session `4e01ec3a1b46`, model **radient/auto** (self-reported; disclosed
  fallback — see "Model" below)
- Revision: `efcfbe812609cbf228c01f11a1fb622f3555b487`, branch `feat/i18n-es-wire-runtime`
- **The es catalogue is untracked at this revision** (`?? …/es/wire.exec.json`); the audited bytes are pinned by sha256
  `6f9df86c94723eb655c0deed39d298e073c2b7f120c961ac794a0ce435af96e8`
- Keys: 17 in `en`, 17 in `es`, **0** in a context sidecar (none exists at this revision)
- Independence: I authored none of this copy and repaired none of it. I edited no catalogue, sidecar, ledger, allowance
  or generated file. My only writes are this directory.

## What was verified

All **17 keys** were read against the source. Re-runnable checks in `run-checks.sh`: `parity.py.txt` (key sets,
placeholders, code spans, line counts, ICU, plural categories), `tokens.py.txt` (every `--flag` and every SCREAMING
token, per key, in order), `epilog.py.txt` (the Examples block, line by line).

Mechanical result: key sets identical (17/17); **11 placeholders compared, all identical in name and order**
(`{error}`, `{log_path}`, `{job_id}`, `{session_id}`, `{names}`, `{count}`, `{failure}`, and `{session_id}` twice in
`session_pointer`); no code spans; no ICU plural machinery — the one dynamic count is the source's own literal
`tool(s)` convention, mirrored faithfully as `herramienta(s)`; no illegal `few`/`zero`/`two`; no `!`; no ASCII ellipsis.

**All 15 distinct flags are byte-identical** to the source: `--agent`, `--agent-id`, `--background`, `--clear-goal`,
`--control`, `--goal`, `--loop`, `--loop-goal`, `--name`, `--profile`, `--resume`, `--status`, `--team`, `--tools`,
`--yolo`. So are `SESSION_ID`, `JOB_ID`, `TUI`, `TTY`, `NAME[,NAME]`, `stdout`, `stderr`, `lop exec`, `lop sessions`,
`lop --resume`, `tool_approval_mode`, and the `'-'` stdin sentinel. The only token difference in the whole namespace is
`SETS` → `FIJA` in `exec_epilog`: that is the all-caps *emphasis* being translated, which is correct — the emphasis
survives, no machine token was touched.

### The coordinator's three risk keys

**`wire.exec.exec_epilog` — block structure preserved, command lines byte-identical.** `epilog.py.txt` asserts it
rather than eyeballing it: 5/5 command lines identical, one blank line at the same index (line 7) in both, and the
title translated `Examples:` → `Ejemplos:`. The prose rewraps from 15 lines to 19 because Spanish line lengths differ —
that is a rewrap, not a structural change, and it is why the raw newline count reads 20 vs 24. Clause by clause the prose
still means what the English means, including the two traps: `--goal only SETS the objective` →
`--goal solo FIJA el objetivo` with "no envía además ese texto como mensaje, porque el prompt es el canal de mensajes de
exec"; and `--tools bounds what the run can reach: excluded tools are unreachable` → `--tools limita a qué puede llegar
la ejecución: las herramientas excluidas quedan fuera de alcance` (wording nit C02). `exec refuses a live owner` →
`exec rechaza a un propietario activo`; `Default non-TTY approvals deny` → `Sin TTY, las aprobaciones se deniegan de
forma predeterminada`; `--yolo … never implied by --background or --team` → `nunca implícita por --background o --team`.

**`wire.exec.approval_advisory` — the promise is neither weakened nor strengthened.** English: no terminal attached ⇒
*any* call needing approval *will be denied*; four remedies, each with its exact meaning. Spanish keeps every part:
`no puede pedir aprobación (no hay terminal conectada)` / `cualquier llamada a herramienta que necesite aprobación se
denegará`; `--control retiene las tarjetas para un supervisor` (*parks* cards — the same verb the locale already uses in
`relay.asks.gate.body`: "Ask retiene cada llamada…"); `--yolo aprueba automáticamente todos los niveles`;
`--tools … preaprueba las herramientas indicadas y limita el alcance de esta ejecución a esas herramientas`;
`tool_approval_mode: auto en la configuración responde a todas las llamadas igual en todas las ejecuciones`. Nothing
softens "will be denied" into "may be denied", and nothing promotes `--yolo` from an explicit opt-in.

**`wire.exec.goal_only_sets_objective` and `wire.exec.clear_goal_needs_start` — both promises survive exactly.**
`--goal fija el objetivo pero no inicia trabajo (a diferencia de /goal de la TUI)` keeps *sets* vs *starts* and keeps the
TUI contrast that the English carries parenthetically; the three ways to actually start (prompt argument, stdin or `'-'`,
`--loop N`) are all intact. `--clear-goal ajusta una ejecución, no la inicia` keeps the adjust-vs-start distinction, and
`Combínalo con --resume SESSION_ID y un prompt o --loop` keeps both required partners.

### The rest

`error_line`, `error_preflight` (`falló la comprobación previa`), `receipt_suffix` / `receipt_failure_suffix` (leading
space and ` — ` separator preserved, since they are concatenated onto a receipt), `log_pointer` (`Registro:`),
`status_pointer`, `session_pointer` (`conéctate o reanúdala`), `turn_did_not_complete`, `no_exec_job`, `no_prompt`
(lowercase opening preserved, as in the source), `tools_unreachable` (`ninguna herramienta a la que esta ejecución
pudiera llegar` — the imperfect subjunctive mirrors English "could reach" and is acceptable Spanish; `pueda` would be
the more natural present, but the meaning and the count sentence are exact), `exec_help` (infinitive action, as the
guide requires for labels),
`waiting_on_you`. Register is `tú` throughout (`Añade`, `Pásalo`, `Combínalo`, `Usa`, `vuelve a intentarlo`,
`reanúdala`), sentence case, no exclamation.

## Findings

### es-w1-C01 — nit — `waiting_on_you` flips the subject

en `Waiting on you? lop sessions shows what a run needs` → es `¿Algo te espera? lop sessions muestra lo que necesita una
ejecución`. English says the run is waiting on the reader; Spanish says something awaits the reader. Both land on the
same behaviour and the line is printed to stderr (`exec_mode.py:838`, under `--control`), so no width budget is involved
— the finding is only that the Spanish is less pointed. Proposed: `¿Te está esperando? lop sessions muestra lo que
necesita una ejecución` (70 cells).

### es-w1-C02 — nit — the same `--tools` promise is phrased two ways in one namespace

`approval_advisory` says `limita el alcance de esta ejecución a esas herramientas` (idiomatic); `exec_epilog` says
`limita a qué puede llegar la ejecución`, a calque of "bounds what the run can reach". Both keep the promise, so this is
style. Proposed: use `limita el alcance de la ejecución a esas herramientas` in both.

## Disposition

**Pass.** The three behavioural promises the coordinator flagged are intact, the Examples block is byte-clean where it
must be, and every machine token survived. The two nits are optional polish and do not need a round of their own — fold
them into any other remediation the wave already needs.

## Source-side risk (not a translation finding)

This namespace has **no en context sidecar** at the audited revision, so I judged register and width with no
role/tone/`maxCells` contract — even though several strings are help-text shaped (`exec_help`, `approval_advisory`, the
epilog). Every other namespace in this wave has one. Worth closing, because it is the same gap class as the
`hub_defaults` label budget: a surface contract that exists in code but not in the sidecar cannot be audited from the
sidecar.

## Model

The session header names `anthropic/claude-sonnet-5-5` as the primary model. Three provider-failure switches fired on
`quota HTTP 429` during this session (`openai/gpt-6-astra` → `kimi/k3.1-pro` → `radient/auto`); everything in this
namespace was judged under **radient/auto**. This is a disclosed fallback, not the pinned primary.

## Limitations

- No native-speaker review is claimed; the auditor is an LLM.
- No rendered frame was captured (host under load, heavy runs excluded). `wire.exec` strings are CLI/stderr output, not
  a laid-out screen, so the epilog and advisory were verified as bytes and structure rather than as a rendered block.
