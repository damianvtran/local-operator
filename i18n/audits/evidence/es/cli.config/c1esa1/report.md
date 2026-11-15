# es · cli.config · Slice C1 audit c1esa1

- Locale: es (neutral Spanish) · namespace: `cli.config` · keys: 15
- Audited revision: `b188079f32e6d978ad78fac96edc5c3516d028ab` (translation shard is untracked/pre-merge)
- Auditor: i18n-auditor (lopdev team), c1-au-es-A · session 96bf9fded81d · model deepseek/deepseek-flash
- Translator (not me): agent openai/gpt-6.1-sol
- Verdict: **no blocker, no major — status passed; verdict clean-with-findings (3 minor, 3 nit)**

## Findings

### M1 · MINOR · `cli.config.running_sessions_unchanged`

- Current: Las sesiones en ejecución no cambian — escribir en la configuración no puede reducir sus requisitos de aprobación; escribe /approvals auto en cada sesión en la que quieras desactivarlos. Las sesiones nuevas se abren en modo auto.
- Problem: Safety meaning survives (a config write cannot loosen a running session; the command must be typed per session; new sessions start at auto), but the same operation is named two ways inside one sentence — 'reducir sus requisitos de aprobación' then 'desactivarlos' (antecedent strained: the nearest plural is 'requisitos', not the session) — and neither matches the shipped sibling wire.settings.approvals.description, which names loosening as 'aflojar': 'Un cambio de configuración endurece todas las sesiones en ejecución a la vez; aflojar una requiere un cambio desde el propio proceso de esa sesión (/approvals auto en ella…)'. A reader who has seen the settings page meets a third word for the same action.
- Suggested: Las sesiones en ejecución no cambian — escribir en la configuración no puede aflojar la de una sesión en ejecución; escribe /approvals auto en cada sesión que quieras dejar sin aprobación. Las sesiones nuevas se abren en modo auto.

### M2 · MINOR · `cli.config.list_desc_max_learnings_history`

- Current: [OBSOLETO — sustituido por la compactación] Número máximo de entradas de aprendizaje que conservar
- Problem: Same object named two ways: the shipped label for this very setting is wire.settings.max_learnings_history.label = 'Historial máx. de aprendizajes' (the only occurrence of the term in the es catalogue), while this row — read directly under that setting's key — calls them 'entradas de aprendizaje'. The list row restates the setting for the same reader, so drift is visible in two adjacent surfaces, and the longer form costs cells in a boxed row.
- Suggested: [OBSOLETO — sustituido por la compactación] Número máximo de aprendizajes que conservar

### M3 · MINOR · `cli.config.list_desc_classification_notice`

- Current: [OBSOLETO] Las sugerencias inteligentes no muestran avisos en el chat; la llamada y su coste figuran en el registro de la sesión
- Problem: en says the feature is SILENT in the chat ('Smart hints is silent in the chat'); es narrows that to 'no muestran avisos'. 'Aviso' is also this feature's own noun in the shipped label 'Aviso de sugerencias inteligentes', so the clause can be read as 'they do not show the notice' rather than 'they print nothing'. The shipped sibling already renders the same fact correctly: wire.settings.classification_notice.help = 'Obsoleto. Las sugerencias inteligentes no dicen nada en el chat; la llamada queda en el registro.'
- Suggested: [OBSOLETO] Las sugerencias inteligentes no dicen nada en el chat; la llamada y su coste quedan en el registro de la sesión

### N1 · NIT · `cli.config.cleared`

- Current: Valor de {key} eliminado: vuelve a usarse el predeterminado
- Problem: Meaning is right (the value is removed and the built-in default applies again) but 'el predeterminado' stands bare as a noun; the catalogue elsewhere uses it attributively ('editor predeterminado', 'predeterminado: 10'). The bare form reads slightly clipped.
- Suggested: Valor de {key} eliminado: vuelve a aplicarse su valor predeterminado

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
cli.config.created
cli.config.opened
cli.config.cleared
cli.config.updated
cli.config.running_sessions_unchanged
cli.config.list_title
cli.config.list_description
cli.config.list_unrecognised
cli.config.list_desc_hosting
cli.config.list_desc_model_name
cli.config.list_desc_conversation_length
cli.config.list_desc_detail_length
cli.config.list_desc_max_learnings_history
cli.config.list_desc_classification_notice
cli.config.list_desc_auto_save_conversation
```
