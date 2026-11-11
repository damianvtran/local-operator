# Independent i18n audit — `fr` — Slice C1, suffix A

**Verdict: clean-with-findings** — 0 BLOCKER, 0 MAJOR, 4 MINOR, 3 NIT.
Status recorded to the audit ledger: `passed` (no BLOCKER/MAJOR remains).

## Provenance

| | |
|---|---|
| Auditor agent | `i18n-auditor` (lopdev team), independent of the translator |
| Session | `58d8409a6284` |
| **Actual served model** | **`deepseek/deepseek-flash`** (session model header; no fallback occurred, none observed) |
| Translator of record | `openai/gpt-6.1-sol` (per task brief; not audited by me) |
| Date | 2026-11-11 |
| Repository / head | `~/local-operator/.worktrees/i18n-c1` @ `b188079f32e6d978ad78fac96edc5c3516d028ab` |
| Locale / file group | `fr` / `cli.config` (15 keys) + `tui.session_move` (7 keys) = 22 keys |

Method: reasoning only — no MT service, no back-translation, no dictionary lookup.
I did not translate these strings and did not patch them.

## Input hashes (SHA-256)

| Input | Path | sha256 |
|---|---|---|
| en source map | `local_operator/i18n/catalogues/en/cli.config.json` | `40649e7419d144f7e8a550d5ae3c6d48defa7bb0b23f4a20ceac35f2b0fda5ac` |
| en context sidecar | `local_operator/i18n/catalogues/en/cli.config.context.json` | `9831a776d2f179969666f2271f9bc53ae7de8a18b8567732ba81bc8e2ca90e78` |
| en source map | `local_operator/i18n/catalogues/en/tui.session_move.json` | `f4dd682b3b50d64a4009eac5ac78a34f5734b5dd0a5ef379951065da326a913e` |
| en context sidecar | `local_operator/i18n/catalogues/en/tui.session_move.context.json` | `09ed30a0b11969671c41e24fe255072c961fff1f90791a7b9f05cd998cbee573` |
| candidate fr (audited) | `…/sessions/3ef6295a5545/scratchpad/c1-prep/fr.json` | `ff03aa6542bc2a79e4533c229399946eb07b578e22d564e97ed283b7b77495cc` |
| en+context bundle read | `…/c1-prep/keys.json` | `ac35dc123a4241a3c1fdbd0f4f012636d9706695bacfb3601aa06eed2cae1ffc` |

Style guide `i18n/style-guides/fr.md` and `i18n/glossary.md` were read in full;
shipped `fr` catalogues (`cli.*`, `tui.*`, `wire.settings`) were read for
terminology consistency; call sites `local_operator/cli.py`
(`config_edit_command`, `config_list_command`) and
`local_operator/tui/session_move.py` were read to confirm what each message does.

## 1. Mechanical verification (actually run)

### 1a. Repository validator, with the candidate mounted

The repo's own catalogue checker (`scripts/i18n/check.py` →
`check_catalogues()`) was run against a mounted candidate: an isolated
catalogue root holding all of `en`, all of the shipped `fr`, plus the candidate
written as `fr/cli.config.json` and `fr/tui.session_move.json` (no repo edit).

```
$ python3 …/scratchpad/i18n_sim.py …/scratchpad
wrote cli.config: 15 keys
wrote tui.session_move: 7 keys
catalogue root: …/scratchpad/i18n-sim
check.py: catalogue checks (candidate fr mounted): 0 problem(s)
```
(`check_catalogues()` covers: ICU-subset parse of every message, key grammar and
namespace prefix, per-namespace key-set parity, argument tuples by **name and
kind**, and plural-category sets — so placeholder kind drift would fail here.)

For the record, the worktree as it stands (before the fr files are committed)
also reports 0 problems, because a locale that does not carry a namespace rides
en-fallback rather than failing parity:

```
$ python3 scripts/i18n/check.py --catalogues-only
check.py: catalogue checks: 0 problem(s)
```

### 1b. Independent checks of my own (python3)

```
en keys: 22 fr keys: 22
missing: []   extra: []   order identical: True
placeholder mismatches: none   (\\{name\\} occurrences, per key, compared as sets)
backtick-span mismatches: none   (`…` spans compared byte-for-byte, in order)
angle-bracket hint-count diffs: none
values identical to en: none
empty values: none
straight apostrophes ("'"): none   (13 × U+2019 present)
straight double quotes ("), "...": none
U+202F used before every : and ; in prose: 8 occurrences, 0 ordinary-space violations
em dashes: 5, all " — " spaced
leading/trailing or double spaces: none
hand-written digits outside placeholders: none
terminal punctuation / bracket-tag presence: identical to en for all 22 keys
```

Backticked / literal spans verified byte-identical to en:
`/approvals auto`, `/move`, `--to`, `--keep`, `--queue`,
`--drop-session-roles`, `local`, and the whole runnable
`` `lop sessions move <id> --to <to> --wait <seconds>` `` and
`` `lop sessions move {session} --to {to}` `` commands.

### 1c. Rendering through the real message runtime

Rendered with `local_operator.i18n.messages.render(..., locale='fr')` against the
mounted candidate (`.venv/bin/python`, `rich.cells.cell_len`), sample params:
`{path}=~/.local-operator/config.yml`, `{key}=display.terminal_title|model_effort`,
`{value}=auto`, `{flag}='--force'`, `{session}=3ef6295a5545`, `{to}=build-box`,
`{code}=2`, `{reason}='[Errno 2] No such file or directory'`.

No key has a `maxCells`/cell cap in its sidecar context ("no cell cap; one
terminal line, may wrap"), so no overflow finding is derivable from the repo; the
longest rendered lines are recorded for the render reader:

| key | en cells | fr cells |
|---|---|---|
| `cli.config.running_sessions_unchanged` | 148 | 259 |
| `tui.session_move.deadline_exceeded` | 110 | 147 |
| `tui.session_move.ambiguous` | 91 | 124 |
| `cli.config.list_desc_detail_length` | 111 | 129 |
| `cli.config.list_desc_classification_notice` | 92 | 127 |
| `tui.session_move.cannot_wait` | 89 | 102 |

All values render; no missing-key fallback, no ICU error, no stray English.

## 2. Findings

### F-01 — MINOR — `cli.config.cleared`, `cli.config.updated` — participle agreement against the product's own noun for a key

Current:
- `cli.config.cleared` = « {key} effacé : sa valeur par défaut s’applique à nouveau »
- `cli.config.updated` = « {key} mis à jour avec la valeur {value} »

Rendered: `display.terminal_title effacé : …`, `model_effort mis à jour avec la valeur auto`.

Problem: in this very command family the thing `{key}` names is feminine — the
shipped fr copy calls it « Clé de configuration » (`cli.config_help.edit_key`)
and « clé non reconnue » (`cli.config.list_unrecognised`). The sentence has no
noun of its own, so the masculine default is grammatical, but it contradicts the
corpus's own gender for the same object, and a native reader with the `lop config`
help page in mind reads a mismatch. The corpus also shows agreement *following*
the noun it establishes (`wire.language_receipts.saved`: « langue : {key}
enregistrée »).

Suggested:
- « Clé {key} effacée : sa valeur par défaut s’applique à nouveau »
- « Clé {key} mise à jour : {value} » (or keep « avec la valeur {value} » if the
  team prefers echoing "value"; the agreement is the finding, not the tail)

If the team's recorded decision is the masculine default for a bare identifier
token, this is closable as "rejected — masculine default for an unnamed
referent"; say so in the remediation comment rather than silently ignoring it.

### F-02 — MINOR — `tui.session_move.ambiguous`, `tui.session_move.to_needs_device` — "peer" rendered as "appareil" (device) where the shipped corpus says "pair"

Current:
- `tui.session_move.ambiguous` = « Utilisez /move <chemin> pour définir le répertoire de travail, ou /move <session> --to <appareil> pour déplacer une session. »
- `tui.session_move.to_needs_device` = « --to nécessite un appareil : /move --to <appareil|local> »

Problem: the en hint word is `peer` (another device *on the mesh*), and the
shipped fr corpus renders peer as « pair »: `cli.sessions_help.move_to` = « un
appareil pair (id ou nom) », `cli.sessions_help.move_engage_on_arrival` = « pour
un déplacement vers un pair ». `appareil` is the glossary term for *device*
(glossary §2: device → appareil), so the hint names the wider concept than the
one accepted (a non-peer device is refused). No wrong action follows — a peer
*is* a device — but it is a shipped-term inconsistency, hence MINOR, not MAJOR.

Suggested: `<pair>` and `<pair|local>`, i.e.
- « … ou /move <session> --to <pair> pour déplacer une session. »
- « --to nécessite un appareil : /move --to <pair|local> »
(`local` stays byte-identical — it is typed.)

### F-03 — MINOR — `tui.session_move.deadline_exceeded` — "n’a pas abouti" leans towards failure on a line whose whole point is that the outcome is UNKNOWN

Current: « le déplacement n’a pas abouti dans le délai imparti ; relancez `lop sessions move {session} --to {to}` pour connaître son état d’avancement »

Problem: en is deliberately temporal — "the move did not finish in time" — and
the sidecar note says "Do not say it failed or succeeded". « aboutir » carries
success semantics in French ("le projet n'a pas abouti" = the project failed);
the time qualifier saves it, but the verb pulls the reader towards "it failed",
which is exactly the reading this copy exists to prevent (a second move issued
on that reading). The second clause does carry the unknown-outcome guidance
correctly, which is why this is MINOR and not MAJOR.

Suggested — only the verb changes, the command stays byte-identical:
« le déplacement ne s’est pas terminé dans le délai imparti ; relancez `lop
sessions move {session} --to {to}` pour savoir où il en est » (or keep
« pour connaître son état d’avancement », which is good).

### F-04 — MINOR — `cli.config.list_unrecognised` — "aucun composant ne la lit" is vaguer than "nothing reads it"

Current: « clé non reconnue ; aucun composant ne la lit »

Problem: en says nothing in the program reads the key; « composant » invites the
question "which component?", and the row already establishes the subject (the
key). The rest of this rendered block is crisp; this is the one soft phrase.

Suggested: « clé non reconnue ; rien ne la lit » (tersest, exact) or
« clé non reconnue ; aucune option ne la lit » (matches the block's vocabulary:
« Options de configuration »).

### F-05 — NIT — `cli.config.created`, `cli.config.opened` — "à l’emplacement {path}" is a heavier calque of "at {path}"

Current:
- « Nouveau fichier de configuration créé à l’emplacement {path} »
- « Fichier de configuration ouvert à l’emplacement {path} »

Problem: not wrong, and the two lines do keep the same noun (« fichier de
configuration ») as the sidecar requires; but « à l'emplacement X » literalises
"at <path>" where French product copy usually apposes the path.

Suggested: « Nouveau fichier de configuration créé : {path} » and
« Fichier de configuration ouvert : {path} » (path still verbatim, byte-identical).

### F-06 — NIT — `cli.config.list_desc_model_name` — "specific" dropped

Current: « Modèle à utiliser pour les interactions » (en: "The specific model to use for interactions").

Problem: no meaning change for the setting; only the emphasis is gone. 42 en
cells vs 39 fr, no cap, so there is room.

Suggested: « Modèle précis à utiliser pour les interactions ».

### F-07 — NIT — `cli.config.list_desc_auto_save_conversation` — infinitive reads as an action label inside a block of noun phrases

Current: « Enregistrer automatiquement les conversations » (en: "Whether to automatically save conversations").

Problem: the six sibling rows in the same rendered block (`lop config list`) are
noun phrases — « Plateforme du fournisseur d’IA… », « Modèle à utiliser… »,
« Nombre maximal… » — so this row alone reads like a command the user could run.
(en also breaks shape here with "Whether to…", so this is polish, not fidelity.)

Suggested: « Enregistrement automatique des conversations » (matches the shipped
label `wire.settings.auto_save_conversation.label` = « Enregistrement auto de la
conversation »).

## 3. Verified good — the parts that would have been worst to get wrong

- **`cli.config.running_sessions_unchanged` (safety line) survives intact.**
  Both claims hold in fr: running sessions are unchanged and a config write
  cannot loosen them (« modifier la configuration ne peut pas assouplir leurs
  règles d’approbation » — « assouplir » is the corpus term, cf.
  `wire.settings.tool_approval_mode.help` = « Assouplir exige une écriture dans
  cette session — /approvals auto. »); the per-session instruction is preserved
  with the command byte-identical (« saisissez /approvals auto dans chaque
  session… »); and the new-session default is preserved (« Les nouvelles
  sessions démarrent en mode auto. »). Nothing was weakened, softened into "will
  apply next time", or turned into a global promise.
- **`tui.session_move.deadline_exceeded` does not claim failure or success** — see
  F-03 for the one verb I would tighten; the "check progress again" remedy and
  the runnable command are exact.
- **Terminology against the shipped corpus** (checked value-by-value, not
  assumed): « Suggestions intelligentes » (= `wire.settings.classification.title`),
  « apprentissages » (`max_learnings_history.label`), « compactage » (glossary §2),
  « journal » / « journal de session » (`classification_notice.help`), « coût »
  (glossary), « historique de conversation » (`wire.radient_saver`),
  « fournisseur d’IA » (2 shipped occurrences), « p. ex. » (7 shipped
  occurrences), « fichier de configuration » (`cli.config_help`), « la CLI »
  (feminine, shipped once), « répertoire de travail » (6 shipped occurrences),
  « approbations », « session » (feminine), « appareil » for device, « modèle ».
  Kept-English honoured: `CLI`, `/approvals`, `/move`, every `--flag`, `local`,
  `auto` as the mode word, provider ids, `shell` (fr loanword, 102 shipped uses).
- **Register/formality**: vous throughout (« Utilisez », « saisissez »,
  « relancez », « vous souhaitez »), no « Veuillez », no exclamation, terse
  past-participle statuses — per `style-guides/fr.md`.
- **Casing device** faithfully mirrored from en: en-capitalised sentences stay
  capitalised; en-lowercase fragments (`` this view… ``, `the move…`, `the CLI…`,
  `could not start…`, `not a recognised key…`) stay lowercase. No Title Case.
- **Typography**: U+202F before `:`/`;` in prose (8 places, 0 violations),
  U+2019 apostrophes (13), « — » spaced (5), no straight quotes, no `...`,
  no hand-written digits.
- **`{flag}`** is `repr(token)` at the call site, so en and fr both show the
  offending flag in program quotes (`pas '--force'`); kept verbatim, correct.

## 4. Scope statement

22/22 keys read (all 15 `cli.config.*`, all 7 `tui.session_move.*`), every value
compared against its en source and its sidecar context note, plus the call sites
that consume them. Not done, and not claimed: native-speaker review (I am the
model, not a person), a live `lop config list` / `/move` session in a `fr`
process (rendering was exercised through the message runtime, not the CLI binary
end to end), and any second pass — suffix B is the independent second pass for
this group.
