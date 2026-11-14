# es · tui.session_move · Slice C1 audit c1esa1

- Locale: es (neutral Spanish) · namespace: `tui.session_move` · keys: 7
- Audited revision: `b188079f32e6d978ad78fac96edc5c3516d028ab` (translation shard is untracked/pre-merge)
- Auditor: i18n-auditor (lopdev team), c1-au-es-A · session 96bf9fded81d · model deepseek/deepseek-flash
- Translator (not me): agent openai/gpt-6.1-sol
- Verdict: **no blocker, no major — status passed; verdict clean-with-findings (3 minor, 3 nit)**

## Findings

### N2 · NIT · `tui.session_move.ambiguous / tui.session_move.to_needs_device`

- Current: Usa /move <ruta> para un directorio de trabajo, o /move <sesión> --to <dispositivo> para mover una sesión. · --to requiere un dispositivo: /move --to <dispositivo|local>
- Problem: en distinguishes <peer> (ambiguous) from a 'device' (to_needs_device); es renders both as 'dispositivo', which is accurate for the --to target but drops the peer sense the shipped CLI help keeps: cli.sessions_help.peer = 'listar las sesiones de un dispositivo par', cli.sessions_help.move_queue 'al mover a un par'. Not misleading (the value really is a device name), so nit only; the two new strings are at least consistent with each other.
- Suggested: /move <sesión> --to <par> … and keep <dispositivo|local> in to_needs_device as it stands (en says 'device' there), or accept 'dispositivo' in both.

### N3 · NIT · `tui.session_move.cannot_wait`

- Current: esta vista no puede esperar — desde una terminal: `lop sessions move <id> --to <to> --wait <seconds>`
- Problem: en says 'from a shell'; es says 'desde una terminal'. Both words are shipped (the catalogue keeps 'shell' English in 'sin un bucle de shell', 'Presupuesto de consultas de shell', and has a near-identical remedy in cli.permissions.setup.refused.isolated: 'Ejecuta el comando desde tu propio shell de inicio de sesión'), so this is consistency, not meaning: the user is already in a terminal when reading this line.
- Suggested: esta vista no puede esperar — desde un shell: `lop sessions move <id> --to <to> --wait <seconds>`

## Mechanical checks

Mechanical verification (all 22 keys, run in the i18n-c1 worktree @ b188079f32e6):
- python3 scripts/i18n/check.py: ok — 217 files scanned, 2540 literals at/below baseline,
  catalogues consistent; 83 advisory findings (advisory only, outside enforced scope).
- key set/order: 22/22, identical to the shipped en catalogues for cli.config.* and
  tui.session_move.*; no keys added or dropped.
- placeholders: message_arguments() name/kind tuples identical to en for every key
  ({path},{key},{value},{description},{flag},{reason},{session},{to},{code});
  no ICU plural/select in this group; every value parses with parse_message().
- backtick spans byte-identical to en (2 shell commands: `lop sessions move <id> --to <to>
  --wait <seconds>` and `lop sessions move {session} --to {to}`); angle-bracket hint tokens
  translated only where the sidecar allows (<ruta>,<sesión>,<dispositivo>); 'local' and
  every --flag kept English; {flag} still inserted verbatim.
- no empty values, no English-identical values, no leading/trailing/double spaces, no
  tabs, no NBSP/zero-width/bidi-control characters, em dash U+2014 with single spaces,
  final-period presence matches en, no stray English beyond the kept tokens.
- widths (rich.cells.cell_len, informational — the sidecar declares no maxCells for any
  key in this group): es runs +0 to +81 cells over en; the largest are
  running_sessions_unchanged 148→229 and list_desc_classification_notice 92→128. Both are
  boxed/wrapping CLI lines with no cap declared, so no fit finding; the +81 line wraps on a
  narrow terminal where English does not, which the en context already permits ('may wrap').

## Second pass

Second independent pass (high-visibility group: settings/config output + error tier).
Re-read all 22 pairs from scratch against i18n/style-guides/es.md and i18n/glossary.md after
finishing pass 1, and checked each term against the shipped es catalogue with bundled greps.
Pass 1 had proposed four findings on the approvals line alone; pass 2 collapsed them to one
(the safety claim itself is faithful and no wording was added that contradicts it) and demoted
the shell/terminal and <peer> observations from minor to nit after finding both words shipped.
Pass 2 produced no new blocker or major, and did not overturn pass 1's M1-M3.

## Style guide

Style-guide conformance (es.md): tú address throughout (Usa/escribe/quieras/vuelve),
imperative/infinitive labels, sentence case only (no Title Case), no "!", em dash with spaces,
no space before ":", lowercase fragments where en is lowercase, [OBSOLETO] bracket tag upper-case
in square brackets with the locale's word for DEPRECATED, neutral Spanish (archivo, dispositivo,
carpeta not used for a session cwd; 'directorio de trabajo' correctly used for the cwd in
tui.session_move.ambiguous), 'compactación' per glossary, 'traslado' matching the shipped
cli.sessions_help.move_* surface, 'coste' per the wave-1 decision, 'sugerencias inteligentes'
matching wire.settings.classification*.

## Keys reviewed

```
tui.session_move.ambiguous
tui.session_move.to_needs_device
tui.session_move.cannot_wait
tui.session_move.unsupported_flag
tui.session_move.cli_unavailable
tui.session_move.deadline_exceeded
tui.session_move.cli_no_answer
```
