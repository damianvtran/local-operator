# Independent i18n audit — `ur` — slice C1, group A (core CLI + TUI English paths)

**Auditor:** i18n-auditor (lopdev), session `270cc26c9e35`, agent `c1-au-ur-A`
**Served model:** `deepseek/deepseek-flash` (actual model the runs executed on; no fallback was pinned or used)
**Translator under audit:** `openai/gpt-6.1-sol` — I did not translate these strings and edited nothing.
**Date:** 2026-11-11
**Namespaces:** `cli.config` (15 keys) + `tui.session_move` (7 keys) — 22 keys, all audited.
**Verdict: `changes_requested`** — 0 BLOCKER, 3 MAJOR, 3 MINOR, 3 NIT.

Mechanical layer is clean: key set/order, placeholders, backticks, punctuation, bidi hygiene all pass,
and the repo's own catalogue validator reports zero problems on these bytes. Everything below is
meaning, idiom, terminology or fit — one MAJOR is a term clash with copy that already ships for the
same settings in the same locale.

## Inputs read

| Input | Path | SHA-256 |
|---|---|---|
| English + context/notes | `~/.local-operator/sessions/3ef6295a5545/scratchpad/c1-prep/keys.json` | `ac35dc123a4241a3c1fdbd0f4f012636d9706695bacfb3601aa06eed2cae1ffc` |
| Translation under audit | `~/.local-operator/sessions/3ef6295a5545/scratchpad/c1-prep/ur.json` | `28b59ffb628369bd0ce59d9cf309e8c3d8f27eaf16e20d538bc7947fbbd6e644` |
| en catalogue | `local_operator/i18n/catalogues/en/cli.config.json` | `40649e7419d144f7e8a550d5ae3c6d48defa7bb0b23f4a20ceac35f2b0fda5ac` |
| en context sidecar | `local_operator/i18n/catalogues/en/cli.config.context.json` | `9831a776d2f179969666f2271f9bc53ae7de8a18b8567732ba81bc8e2ca90e78` |
| en catalogue | `local_operator/i18n/catalogues/en/tui.session_move.json` | `f4dd682b3b50d64a4009eac5ac78a34f5734b5dd0a5ef379951065da326a913e` |
| en context sidecar | `local_operator/i18n/catalogues/en/tui.session_move.context.json` | `09ed30a0b11969671c41e24fe255072c961fff1f90791a7b9f05cd998cbee573` |
| Style guide | `i18n/style-guides/ur.md` | `7276e96210fe5f23fd82a39ae4cc0c88b2c1009d3a551d914efc3f61516eddc4` |
| Glossary | `i18n/glossary.md` | `0eb34437a8c3d6f564f278ef957a603b1d9f51bb07d5ddae5ea8f37deba962f8` |

Worktree used: `~/local-operator/.worktrees/i18n-c1` @ `b188079f32e6` (shipped `ur` catalogues for terminology
comparison; source use sites `local_operator/cli.py` `config_create_command`/`config_open_command`,
`local_operator/tui/session_move.py`, `local_operator/i18n/keys/tui_session_move.py`).

## Findings

### U1 — MAJOR — `cli.config.running_sessions_unchanged` — plural collapsed to singular inside a safety line

**Current:** `چلتے سیشن میں کوئی تبدیلی نہیں ہوئی — کنفیگریشن محفوظ کرنے سے ان میں منظوری کی پابندی کم نہیں ہوتی؛ جس سیشن میں ٹولز کے لیے منظوری کی پابندی ہٹانا چاہتے ہیں، اس میں /approvals auto لکھیں۔ نئے سیشن auto موڈ میں شروع ہوتے ہیں۔`

**Problem:** EN is "Running sessions **are** unchanged" — all of them. The Urdu subject is singular
(`چلتے سیشن میں`) yet the very next clause refers back to it with the **plural** pronoun `ان میں`, so the
sentence contradicts itself on number, and the claim narrows from "every running session" to "a running
session". The locale already has the plural form for the same idea in shipped copy —
`wire.settings.approvals.description` ("تمام چلتے سیشنز …") and `wire.settings.static.description`
("چلتے سیشنز کی ورکنگ ڈائریکٹریز"). This is the line that tells the operator a config write did **not**
loosen approvals, so its scope has to be exact.

**Suggested:** `چلتے سیشنز میں کوئی تبدیلی نہیں ہوئی — کنفیگریشن محفوظ کرنے سے ان میں منظوری کی پابندی کم نہیں ہوتی؛ جس سیشن میں ٹولز کے لیے منظوری کی پابندی ہٹانا چاہتے ہیں، اس میں /approvals auto لکھیں۔ نئے سیشن auto موڈ میں شروع ہوتے ہیں۔`

### U2 — MAJOR — deprecated tag: `متروک` vs the shipped `فرسودہ` (4 keys)

**Keys:** `cli.config.list_desc_conversation_length`, `cli.config.list_desc_detail_length`,
`cli.config.list_desc_max_learnings_history` (`[متروک — اب کمپریشن اس کی جگہ لیتا ہے]`) and
`cli.config.list_desc_classification_notice` (`[متروک]`).

**Problem:** the `ur` catalogue already renders DEPRECATED as **فرسودہ** in the settings surface these very
rows belong to — `wire.settings.conversation_length.help`, `detail_length.help`,
`max_learnings_history.help`, `classification_notice.help` ("فرسودہ۔ …") and the retired-section title
`wire.settings.retired.title` = "فرسودہ" (5 occurrences). **متروک appears nowhere else in the `ur`
catalogue**, so the user meets two different words for one status in `lop config list` and the settings
page. متروک is also the more Persianised option, which the guide's "not over-Persianised" line cautions
against. (The bracket-tag form itself is right: square brackets, no case in Urdu script, em dash retained.)

**Suggested:** `[فرسودہ — اب کمپریشن اس کی جگہ لیتا ہے]` and `[فرسودہ]`. If the wave prefers متروک, the
shipped settings strings have to move with it in a separate change — not two spellings in one release.

### U3 — MAJOR — `cli.config.list_desc_max_learnings_history` — `اسباق` vs the shipped `نکات`

**Current:** `[متروک — اب کمپریشن اس کی جگہ لیتا ہے] محفوظ رکھے جانے والے سیکھے ہوئے اسباق کی زیادہ سے زیادہ تعداد`

**Problem:** this is the `max_learnings_history` setting, whose label already ships in `ur` as
`wire.settings.max_learnings_history.label` = "سیکھے ہوئے **نکات** کی زیادہ سے زیادہ تاریخچہ". `اسباق`
occurs nowhere in the `ur` catalogue. The English context note glosses "learning entries" as *saved
lessons*, but a gloss for the translator is not a term decision; the live settings page is, and it says
نکات. Same setting, two nouns.

**Suggested:** `[فرسودہ — اب کمپریشن اس کی جگہ لیتا ہے] محفوظ رکھے جانے والے سیکھے ہوئے نکات کی زیادہ سے زیادہ تعداد`

### U4 — MINOR — `cli.config.cleared` — says the *key* was removed, not its value

**Current:** `{key} ہٹا دی گئی: اب اس کی ڈیفالٹ ویلیو دوبارہ استعمال ہوتی ہے`

**Problem:** "ہٹا دی گئی" (was removed) attaches to `{key}`, i.e. the setting itself. Nothing removed the
setting — the write cleared its value, and the second clause exists precisely to say it still has a value
(its default). A reader can take the first clause as "this setting no longer exists".

**Suggested:** `{key} کی ویلیو صاف کر دی گئی: اب یہ دوبارہ اپنی ڈیفالٹ ویلیو پڑھی جاتی ہے`

### U5 — MINOR — `cli.config.updated` — value shoved into a bare complement verb, "Successfully" dropped

**Current:** `{key} کی ویلیو {value} کر دی گئی`

**Problem:** EN is "Successfully updated {key} to {value}". The Urdu reads "the value of {key} was *made*
{value}" — tolerable when `{value}` is a number, but `کر دی گئی` with a word value (`{value}` can be
`auto`, `True`, a list label per the note) is a calque, and the success is only implied. Shipped copy
uses اپ ڈیٹ for exactly this operation (`cli.config_help.edit_key`: "اپ ڈیٹ کی جانے والی کنفیگریشن کلید").

**Suggested:** `{key} کو {value} پر اپ ڈیٹ کر دیا گیا` (or, if the success word is wanted back:
`کامیابی سے {key} کو {value} پر اپ ڈیٹ کر دیا گیا`)

### U6 — MINOR — `cli.config.list_desc_classification_notice` — one English sentence, two Urdu renderings

**Current:** `[متروک] ذہین اشارے چیٹ میں کچھ نہیں دکھاتے؛ ماڈل کی کال اور اس کی لاگت سیشن لاگ میں درج ہوتی ہیں`

**Problem:** the same sentence already ships as `wire.settings.classification_notice.help` =
"فرسودہ۔ ذہین اشارے چیٹ میں **خاموش ہیں**؛ کال لاگ میں آتی ہے۔". The feature term (ذہین اشارے) is right and
matches the settings page, but "چیٹ میں کچھ نہیں دکھاتے" is a second rendering of "is silent in the chat",
and a feature name that is singular in English takes plural verbs here. Only one `[متروک]` fix (U2) and one
phrasing choice are needed; pick the shipped one.

**Suggested:** `[فرسودہ] ذہین اشارے چیٹ میں خاموش ہیں؛ ماڈل کی کال اور اس کی لاگت سیشن لاگ میں درج ہوتی ہیں`

### U7 — NIT — `tui.session_move.ambiguous` — `استعمال کریں` twice in one sentence

**Current:** `ورکنگ ڈائریکٹری کے لیے /move <پاتھ> استعمال کریں، یا سیشن منتقل کرنے کے لیے /move <سیشن> --to <ڈیوائس> استعمال کریں۔`

**Suggested:** `ورکنگ ڈائریکٹری کے لیے /move <پاتھ> استعمال کریں، یا سیشن منتقل کرنے کے لیے /move <سیشن> --to <ڈیوائس> چلائیں۔`
(The angle-bracket hint words are correctly translated per the note, brackets and `--to` untouched.)

### U8 — NIT — `cli.config.list_desc_hosting` — the provider role lands on the platform

**Current:** `AI فراہم کرنے والا پلیٹ فارم (مثلاً radient, openai, deepseek, anthropic, openrouter)`

**Problem:** `فراہم کرنے والا` modifies `پلیٹ فارم`, i.e. "a platform that provides AI"; EN means the AI
*provider's* platform (the listed values are providers). Readable, but the precision is free.

**Suggested:** `AI فراہم کنندہ کا پلیٹ فارم (مثلاً radient, openai, deepseek, anthropic, openrouter)`

### U9 — NIT — `tui.session_move.cli_no_answer` — heavy insertion, and CLI gender

**Current:** `CLI بغیر جواب دیے ایگزٹ کوڈ {code} کے ساتھ بند ہو گئی`

**Problem:** the phrase `بغیر جواب دیے` and its head `کے ساتھ بند ہو گئی` are split by the whole exit-code
phrase, so the line drags; `گئی` also treats the program as feminine (as does `cli_unavailable`'s `سکی`),
where a program more often takes masculine agreement in Urdu. Either is defensible — but if the wave
changes one, both must change together.

**Suggested:** `CLI ایگزٹ کوڈ {code} کے ساتھ، بغیر جواب دیے بند ہو گئی` (masc alternative: `بند ہو گیا`, with
`cli_unavailable` → `CLI شروع نہیں ہو سکا`).

## Second independent pass (high-visibility group)

`cli.config` is the settings surface, so I re-read all 15 lines **Urdu-only, without the English** — the
operator's acceptance test ("switching to `fr`/`ur` leaves NO English anywhere and the copy still makes
sense"). That pass independently surfaced U1 (the mismatched `ان میں` is only visible from the Urdu side),
confirmed U2/U3 against the shipped settings catalogue, and confirmed U4/U5 as the only two lines whose
meaning does not survive a blind read. It produced no new findings and caused no finding from pass 1 to be
withdrawn.

## Checked and accepted (no finding)

- **Placeholders:** `{path}`, `{key}`, `{value}`, `{description}`, `{flag}`, `{reason}`, `{code}`,
  `{session}`, `{to}` — names, kinds and order identical to `en` in all 22 keys; brace counts equal; no
  ICU `plural`/`select` in this group, so no CLDR category question arises (`ur` = one/other).
- **Code spans:** all 4 backticked spans byte-identical; `/approvals auto`, `/move`, `--to`, `--keep`,
  `--queue`, `--drop-session-roles`, `--wait`, `lop sessions move` verbatim; `local` in
  `/move --to <ڈیوائس|local>` kept Latin (typed value), `|` and angle brackets kept.
- **No added/dropped keys,** same order as `en`; valid JSON; no empty values; no value identical to English.
- **Bidi hygiene:** no LRM/RLM/FSI/PDI/LRI/RLI inserted anywhere (guide ban — the render layer isolates);
  strings that open with an LTR token (`{path}`, `{key}`) carry the correct logical order for an RTL base;
  no compensating hacks for the documented best-effort TUI stance.
- **Punctuation:** `۔` `،` `؛` used correctly, no Hindi `।`, Latin commas only inside Latin identifier
  lists (same shape as shipped `cli.config_help.edit_key`); ending punctuation matches `en` (only the two
  full sentences end in `۔`); em dash used, which shipped `ur` uses 238×.
- **Register:** `آپ` throughout, respectful imperatives (`استعمال کریں`, `لکھیں`), no `تم`/`تو`, no `!`,
  no unwarranted `براہ کرم`.
- **Safety meanings intact:** config write cannot loosen a running session (once U1's number is fixed);
  `/approvals auto` must be typed *in each session*; `deadline_exceeded` reports an **unknown** outcome
  ("مکمل نہیں ہوئی … کہاں تک پہنچی"), never failure; `cli_unavailable` blames nothing.
- **Terminology matching shipped `ur`** (occurrence counts over `catalogues/ur/`): کنفیگریشن فائل (4),
  کنفیگریشن آپشنز (1), کلید (68), ڈیفالٹ (90), گفتگو (182), ہسٹری (6), شیل (17),
  ورکنگ ڈائریکٹری (10), ایگزٹ کوڈ (4), ذہین اشارے (3), چیٹ (3), منتقلی (9, the shipped word for a session
  move), سیشن/ڈیوائس/ماڈل/کمپریشن/لاگت per glossary. Only U2 (فرسودہ 5 / متروک 0) and U3 (نکات 1 / اسباق 0)
  diverge.
- **Fit:** no `maxCells`/`maxChars` on any of these 22; nothing here is pinned (PERSIST_HINT/tip pool/`/help`
  untouched); no truncation risk created by the suggested wordings.
- **Glossary/kept-English:** `AI`, `CLI`, `auto` stay Latin per §1/acronym policy; provider names verbatim.

## Commands actually run (real output, not notes)

1. **Repo validator, catalogue job, on an APFS-cloned fixture** of
   `local_operator/i18n/catalogues/` with `ur/cli.config.json` (15) and `ur/tui.session_move.json` (7)
   written from `ur.json`:
   `check_catalogues()` from `scripts/i18n/check.py` (root monkeypatched to the fixture via
   `catalogues._CATALOGUES`) → **`TOTAL problems: 0`**.
2. **Negative control, same call, same bytes:** I corrupted two values in the fixture — dropped
   `{key}`/`{value}` from `cli.config.updated` and renamed `{path}`→`{url}` in `cli.config.created` — and
   the validator printed exactly:
   `ur/cli.config: cli.config.created: arguments (('url', 'simple'),) != en's (('path', 'simple'),): argument 1 is 'url', en's is 'path' (renamed or reordered)` and
   `ur/cli.config: cli.config.updated: arguments () != en's (('key', 'simple'), ('value', 'simple')): en has 2 argument(s), translation has 0`;
   restoring the real bytes → `0`. The check is live on the bytes I audited.
3. **Own mechanical scan** (key set + order, placeholder name/order, `{`/`}` counts, backtick spans and
   counts, emptiness, English-identical values, leading/trailing whitespace, bidi-control codepoints,
   Hindi danda) → **0 issues**.
4. **Latin-token inventory per key** (`[A-Za-z][A-Za-z0-9_.-]*`) → only paths, keys, flags, commands,
   `AI`, `CLI`, `auto`, provider names. **No stray English.**
5. **Term-consistency greps against the shipped `ur` catalogue** (counts in "Checked and accepted"),
   including `فرسودہ` = 5 / `متروک` = 0 and `اسباق` = 0 with
   `wire.settings.max_learnings_history.label` = «سیکھے ہوئے نکات …» — the evidence behind U2/U3.
6. **Source use sites read** to establish intent: `cli.py::config_create_command` (prints the path actually
   written), `cli.py::config_open_command` (prints after the opener/editor launched), `tui/session_move.py`
   (`_t(_keys.ambiguous())`, `unsupported_flag(flag=repr(token))`, `deadline_exceeded(session=…, to=…)`,
   `cli_no_answer(code=str(child.returncode))`).
7. **Ledger payload validation (dry run, nothing written):**
   `python3 scripts/i18n/ledger.py history-sha --locale ur --namespace cli.config` → `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
   (same for `tui.session_move`); `ledger.py append --dry-run --receipt …/ur-cli.config.receipt.json --expected-history-sha 4f53cda1…`
   → `i18n/audit-history/ur/cli.config.json`, exit 0; the same for `tui.session_move` → exit 0.

## Ledger / evidence: prepared, **not applied** — and why

Per the role contract the result belongs in the repo's audit ledger, and I built and **dry-run validated**
both rows (they are on stdout above). I deliberately did **not** `--apply` them, and I did not create
`i18n/audits/evidence/ur/<ns>/…` bundles, for three reasons:

1. the verdict is `changes_requested` and the strings will change under U1–U6;
2. `catalogues/ur/cli.config.json` and `catalogues/ur/tui.session_move.json` **do not exist in the tree
   yet** — the audited bytes are the prep file, so a record's target-locale `input_hashes` could not bind
   tracked, shipped bytes (the evidence contract requires repo-only evidence);
3. `~/local-operator/.worktrees/i18n-c1` is the manager-owned shared worktree; ledger suffix writes from an
   auditor mid-slice can race the coder's commit.

Ready-to-apply receipts (validated): `$LOCAL_OPERATOR_SCRATCHPAD/ledger-payload/ur-cli.config.receipt.json`
and `…/ur-tui.session_move.receipt.json` — both currently `status: changes_requested`. After remediation,
re-audit and flip the auditor block/status, then apply with
`python3 scripts/i18n/ledger.py append --apply --receipt <file> --expected-history-sha $(python3 scripts/i18n/ledger.py history-sha --locale ur --namespace <ns>)`
inside the worktree that will commit the catalogue, and reference the tracked
`i18n/audits/evidence/ur/<ns>/<id>/record.json` path in the row's `note` (≤280 chars).

## Summary

| Severity | Count | Keys |
|---|---|---|
| BLOCKER | 0 | — |
| MAJOR | 3 | U1 `running_sessions_unchanged`; U2 four `[DEPRECATED]` rows; U3 `max_learnings_history` |
| MINOR | 3 | U4 `cleared`; U5 `updated`; U6 `classification_notice` |
| NIT | 3 | U7 `ambiguous`; U8 `list_desc_hosting`; U9 `cli_no_answer` |

The mechanical layer needs nothing. The two term clashes (U2, U3) are the ones that matter beyond this
slice: they are the only places where these 22 keys disagree with copy that already ships in `ur`, and both
are one-word swaps.

## Appendix — validated ledger receipts (verbatim, `--dry-run` exit 0)

`ur` / `cli.config`:

```json
{
  "locale": "ur",
  "namespace": "cli.config",
  "source_sha256": "40649e7419d144f7e8a550d5ae3c6d48defa7bb0b23f4a20ceac35f2b0fda5ac",
  "translator": { "agent": "i18n-translator (c1 slice)", "model": "openai/gpt-6.1-sol", "date": "2026-11-11" },
  "auditor": {
    "agent": "i18n-auditor (lopdev c1-au-ur-A, session 270cc26c9e35)",
    "model": "deepseek/deepseek-flash",
    "date": "2026-11-11",
    "result": "changes_requested",
    "findings": 7
  },
  "status": "changes_requested",
  "note": "C1 round-A: 15 keys; placeholders/backticks/ICU + validator clean. 3 MAJOR (deprecated tag متروک vs shipped فرسودہ; learnings اسباق vs نکات; running-session plural), 3 MINOR, 1 NIT. Auditor deepseek/deepseek-flash."
}
```

`ur` / `tui.session_move`:

```json
{
  "locale": "ur",
  "namespace": "tui.session_move",
  "source_sha256": "f4dd682b3b50d64a4009eac5ac78a34f5734b5dd0a5ef379951065da326a913e",
  "translator": { "agent": "i18n-translator (c1 slice)", "model": "openai/gpt-6.1-sol", "date": "2026-11-11" },
  "auditor": {
    "agent": "i18n-auditor (lopdev c1-au-ur-A, session 270cc26c9e35)",
    "model": "deepseek/deepseek-flash",
    "date": "2026-11-11",
    "result": "changes_requested",
    "findings": 2
  },
  "status": "changes_requested",
  "note": "C1 round-A: 7 /move refusals; commands, flags, 'local' and angle-bracket tokens verified; no blocker/major. 2 NIT. Remainder of the group's MAJOR sits on cli.config (running_sessions_unchanged)."
}
```

Both were passed to `ledger.py append --dry-run` unchanged and returned
`i18n/audit-history/ur/<namespace>.json` with exit 0. Working copies also sit at
`$LOCAL_OPERATOR_SCRATCHPAD/ledger-payload/ur-<namespace>.receipt.json` (session scratch — copies, not the
artifact of record).
