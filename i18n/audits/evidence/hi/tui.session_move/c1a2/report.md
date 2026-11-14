# Independent round-2 delta audit — `hi` — slice C1 (core CLI + TUI), 22 keys

- **Auditor**: `i18n-auditor` (lopdev team), session `fb68c83d54be`.
- **Actual served model**: `deepseek/deepseek-flash` — taken from this session's own
  system prompt (`Model: deepseek/deepseek-flash`). No model was requested or pinned
  for this lane; no fallback is claimed, and this is the model that produced every
  judgment below.
- **Independence**: this auditor did not translate these strings and did not edit any
  value it judged. Translator lane: session `e8ed223773d2`, model `openai/gpt-6.1-sol`
  (per `hi.v2-changes.md`). No translation/MT service was used for comparison or
  back-translation; every judgment is reasoned from the style guides, the glossary, the
  shipped `hi` catalogues and the use sites.
- **Date**: 2026-11-11. **Locale**: `hi` (LTR — the RTL dimension of the audit template
  is not applicable).
- **Revision audited**: worktree `~/local-operator/.worktrees/i18n-c1` @
  `b188079f32e6d978ad78fac96edc5c3516d028ab` (read-only use; nothing in the tree was
  modified except this audit's own evidence bundle and ledger rows).
- **Inputs**:
  - English source + context: `c1-prep/keys.json` — sha256 `ac35dc12…ae1ffc` (verified here).
  - Shipped English catalogues: `en/cli.config.json` `40649e74…fda5ac`,
    `en/tui.session_move.json` `f4dd682b…a913e` — both identical to the hashes the
    round-1 rows recorded, i.e. the source has not moved under this wave.
  - Candidate under audit: v2, sha256 `d575bebe3038926f3b4eea6f17c945db3f4337f1d1e8de0d4b6a039edf6321c5`
    — **verified byte-for-byte against the header the translator published** in
    `hi.v2-changes.md`. Round-1 v1 was `b67c087d…a355`.
  - `i18n/style-guides/hi.md` `82292dc8…`, `i18n/style-guides/en.md`, `i18n/glossary.md`,
    `docs/design/i18n.md` §6.
  - Use sites: `local_operator/cli.py` (2799, 2834/2847, 2913, 3018, 3028, 3093–3113),
    `local_operator/tui/session_move.py` (143–144, 206, 214–216, 229), `cli.py` L3015–3028
    for the write/echo order.

## Verdict

**`passed`** — 0 BLOCKER, 0 MAJOR, 0 MINOR, **1 NIT**. All 14 findings the round-1
auditors raised and the translator fixed are genuinely fixed; both declined findings
survive an independent recount. Nothing in the delta introduces a seam with the shipped
catalogue, and both safety lines keep every claim.

Counts per group: `cli.config` 15 keys (0 B / 0 M / 0 m / 1 n) ·
`tui.session_move` 7 keys (0 B / 0 M / 0 m / 0 n).

## Scope (what this round judged, and what it did not)

1. **Delta** — the 9 keys whose bytes changed between v1 and v2, plus the unchanged
   neighbours a change could now contradict.
2. **Adjudication** — both declined findings, re-decided from a recount I ran myself.
3. **Full mechanical pass on v2** — all 22 keys (structure, ICU, placeholders, literals).
4. **Safety lines** — `cli.config.running_sessions_unchanged` and
   `tui.session_move.deadline_exceeded`, both checked against their code paths.

Not re-audited: the 13 unchanged keys that round 1 approved and no change touches
(`created`, `opened`, `cleared`, `list_title`, `list_description`, `list_unrecognised`,
`list_desc_conversation_length`, `ambiguous`, `to_needs_device`, `cannot_wait`,
`unsupported_flag`, `cli_unavailable`, `deadline_exceeded`). They were re-read only for
(3) and (4) and for neighbour contradiction.

## Delta: 9 changed keys, judged one by one

| Key | v1 → v2 | Judgment |
|---|---|---|
| `cli.config.updated` | `{key} का मान {value} कर दिया गया` → `{key} को {value} पर सेट कर दिया गया` | **Correct and better.** Renders cleanly for every shape the sidecar promises — verified through the production renderer with `auto`, `True`, `50`, and `['a', 'b']` (`display.nerd_icons को True पर सेट कर दिया गया`, `… को ['a', 'b'] पर सेट कर दिया गया`). Dropped "Successfully" is sanctioned (see adjudication 2). |
| `cli.config.running_sessions_unchanged` | `… बदलाव करने से … शर्तें कम नहीं होतीं; जिस सत्र में …` → `… बदलाव सहेजने से … शर्तें ढीली नहीं हो सकतीं; जिन सत्रों में आप … उनमें से हर एक में …` | **Fixed, all four claims intact.** `ढीली` is the shipped collocation for loosening approvals; the distributive "EACH session" is now explicit (`उनमें से हर एक में`). One NIT on the "config write" verb — see HI-A2-01. |
| `cli.config.list_desc_hosting` | `AI प्रदाता का प्लेटफ़ॉर्म` → `AI प्रोवाइडर प्लेटफ़ॉर्म` | **Fixed.** `प्रोवाइडर` is the settled product term (my recount: 62 boundary hits / 17 files vs `प्रदाता` 0 boundary hits, 1 inflected `प्रदाताओं` in `cli.group_help.search`); matches this setting's own shipped label `डिफ़ॉल्ट प्रोवाइडर`. |
| `cli.config.list_desc_model_name` | `… इस्तेमाल किया जाने वाला मॉडल` → `… इस्तेमाल किया जाने वाला खास मॉडल` | **Acceptable.** The round-1 omission is restored; `खास` is everyday Hindi for "specific" (the en guide asks translators to adapt register, not to copy). Shipped `विशिष्ट` occurs once, in a different collocation (`cli.config_help.edit`: "कोई विशिष्ट कॉन्फ़िगरेशन मान"), so no seam worth a finding. Recorded as considered-and-accepted, not as a finding. |
| `cli.config.list_desc_detail_length` | `बिना सारांश बनाए रखे जाने वाले हाल के संदेशों की संख्या` → `हाल के जिन संदेशों का सारांश नहीं बनाया जाता, उनकी संख्या` | **Fixed.** The negated-gerund pile is gone; the correlative `जिन … उनकी` reads as ordinary Hindi settings copy and keeps "leave unsummarized" exactly. |
| `cli.config.list_desc_max_learnings_history` | `सीखी गई बातों की सहेजी जाने वाली प्रविष्टियों …` → `सहेजी जाने वाली लर्निंग्स प्रविष्टियों …` | **Fixed.** `लर्निंग्स` is this setting's own shipped label word (`wire.settings.max_learnings_history.label` = "लर्निंग्स इतिहास अधिकतम"); the compound `लर्निंग्स प्रविष्टियाँ` matches the telegraphic settings-row style already in the catalogue (`स्टेट सीमा`, `इंतज़ार बजट`). |
| `cli.config.list_desc_classification_notice` | `… कॉल और उसकी लागत … दर्ज होती है` → `… दर्ज होते हैं` | **Fixed.** Coordination `कॉल और उसकी लागत` now takes the plural verb; `स्मार्ट संकेत` stays the shipped feature name (4 shipped hits, `wire.settings.classification*`). |
| `cli.config.list_desc_auto_save_conversation` | `अपने आप` → `अपने-आप` | **Fixed.** Hyphenated form is the shipped norm (34/10 files vs 5/5) and is the label of this very setting (`wire.settings.auto_save_conversation.label` = "बातचीत अपने-आप सहेजें"). |
| `tui.session_move.cli_no_answer` | `निकास कोड` → `एग्ज़िट कोड` | **Fixed.** `निकास` has 0 shipped occurrences; `एग्ज़िट कोड` is the shipped term in the same CLI family (`cli.agents_help.push`, `cli.teams_help.push`, `wire.remote_compute.finished_exit_code`). |

**Neighbours checked for contradiction (none found).** `list_title` "कॉन्फ़िगरेशन विकल्प"
still matches `cli.config_help.list`; `list_description` "विवरण:" is the label the list
loop prints (`cli.py` L3093–3113); `list_unrecognised` keeps the shipped `कुंजी`
(`cli.config_help.edit_key`); the seven `list_desc_*` rows agree with their settings-page
twins (`wire.settings.hosting.label`, `model_name.help`, `max_learnings_history.label`,
`classification.title`, `auto_save_conversation.label`); `to_needs_device` and
`ambiguous` keep `पीयर`/`डिवाइस` consistent with each other and with shipped copy.

## Adjudication of the declined findings

### HI-A07 (`<पाथ>` → `<पथ>`) — **DECLINE UPHELD**, no finding

Recount run here over the shipped `hi` values only (144 catalogue files, 3 978 values;
context sidecars excluded):

- `पाथ` **11 hits / 9 files** — `cli.config_help.instructions` ("केवल पाथ और आकार"),
  `cli.permissions`, `cli.agents_picture.help.file` ("इमेज का पाथ"), `wire.errors.desktop`, …
- `पथ` **8 hits / 6 files** — `wire.memory` ("पथ, URL और ID"), `tui.viewer`, …

So the catalogue is genuinely mixed and the majority — including the *nearest* neighbour
in the same `lop config` surface family — is `पाथ`. `i18n/glossary.md` has no row for
path, and `i18n/style-guides/hi.md` fixes nothing about this spelling; its
product-name policy keeps *code*/paths Latin, which this hint word is not (the source
sidecar explicitly says the word inside the brackets is translated). Round 1 itself
graded this NIT, not a defect. Standardising on `पथ` would have to move
`cli.config_help.instructions` with it — a terminology-scope change, not a 22-key slice
edit. **Deferred as a terminology item**, correctly declined here.

### B:n1 (add `सफलतापूर्वक` to the updated line) — **DECLINE UPHELD**, no finding

`i18n/style-guides/en.md` forbids "successfully" padding ("Saved", not "Successfully
saved"); `hi.md` asks for terse copy. I checked the use site rather than trusting the
argument: `cli.py` L3018 prints `updated` **after** `settings_io.write_setting` returned,
and every failure path (`ConfigUnreadableError`, `ValueError`, generic) returns from its
own handler before that print. A line that only prints on success, in the perfective
`कर दिया गया`, loses no information without the adverb. Decline is right.

Both adjudications agree with the translator's evidence; neither is an agreement on
trust — the counts and the code path were re-derived here.

## Mechanical pass on v2 (all 22 keys)

| Check | Result |
|---|---|
| Repo validator `python3 scripts/i18n/check.py` | `ok — 217 file(s) scanned, 2540 literal(s) at or below the baseline, catalogues consistent; 83 advisory` — rc 0 |
| Repo validator `--catalogues-only` | `check.py: catalogue checks: 0 problem(s)` — rc 0 |
| `check_catalogues()` over a private temp root = shipped `en` + `hi` + the candidate installed as `hi/cli.config.json` + `hi/tui.session_move.json` | **0 problems**; control run on the same root without the candidate: 0 problems |
| Candidate group ↔ shipped namespace key sets | `en/cli.config` 15 = candidate 15; `en/tui.session_move` 7 = candidate 7; no key added or dropped |
| Key set and order vs `keys.json` | identical 22/22, order-identical |
| Placeholder sets/types per key | identical for all 22 (`{path}`, `{key}`, `{value}`, `{description}`, `{flag}`, `{reason}`, `{session}`, `{to}`, `{code}`) |
| ICU plural/select | none in this group, so no CLDR category obligation; every value parses in the shipped ICU subset |
| Production render (`local_operator.i18n.messages.render`, `locale="hi"`) | 22/22 render with `source_locale == "hi"` — **no English fallback, no code-only degradation**; renderings in `audit.log` §3 |
| Backticked spans | byte-identical: `` `lop sessions move <id> --to <to> --wait <seconds>` `` and `` `lop sessions move {session} --to {to}` `` |
| Typed tokens | `/approvals auto`, `auto`, `/move`, `--to`, `--keep`, `--queue`, `--drop-session-roles`, `--wait`, literal `local` inside `<पीयर|local>`, `<id>`, `<to>`, `<seconds>`, the five provider ids, `CLI`, `AI` — all verbatim |
| Empties / English-identical values | none |
| Stray Latin | every Latin run traces to a source token; rendered residue is only kept-literals (paths, setting keys, flags, commands, provider ids, `CLI`, and `{reason}`'s OS text, which the sidecar says may be English) |
| Bidi controls, ZWJ/ZWNJ | none |
| Devanagari digits / `!` / `कृपया` / `तुम`-`तू` | none |
| Nukta encoding | 10 uses, **0 precomposed** (base + U+093C), matching shipped `hi` |
| Final punctuation | danda on the two full sentences, none on the label/fragment rows — matches the English source's own stop pattern |
| Width | no `maxCells`/`maxChars` hint in either en context sidecar for this group; the only metric guidance is `list_title` < ~30 cells → measured **12 cells** (`rich.cells.cell_len`) vs 21 for English, inside the fixed `╭─ … ─────` rule printed at `cli.py` L3093. Longest value is the 177-cell approvals line on a "may wrap" line |

## Safety lines

**`cli.config.running_sessions_unchanged`** (printed only when `tool_approval_mode` was
set to `auto` — `cli.py` L3028). All four claims survive:

1. running sessions unchanged → `चल रहे सत्रों में कोई बदलाव नहीं हुआ` ✔
2. a config write cannot loosen them → `कॉन्फ़िगरेशन में बदलाव सहेजने से उनकी स्वीकृति की शर्तें ढीली नहीं हो सकतीं` ✔ (direction correct; shipped `ढीला` verb)
3. `/approvals auto` must be typed in **each** session → `जिन सत्रों में आप बिना स्वीकृति के टूल चलाना चाहते हैं, उनमें से हर एक में /approvals auto टाइप करें।` ✔ (distributive, command verbatim)
4. new sessions open at auto → `नए सत्र auto मोड में शुरू होते हैं।` ✔ (`auto` stays the Latin mode word)

Nothing in it promises that a loosening reaches a running session — the failure mode this
line exists to prevent.

**`tui.session_move.deadline_exceeded`** — reports an **unknown outcome**, as the code
requires: `session_move.py` L210–217 returns this refusal with `changed=True` after
`subprocess.TimeoutExpired`, i.e. the move may have started. The Hindi says only
`स्थानांतरण तय समय में पूरा नहीं हुआ` ("the move did not finish within the fixed time")
and `… फिर से चलाने पर पता चलेगा कि यह कहाँ तक पहुँचा` ("running it again will tell how far
it got") — no failure claim, no success claim, and the re-run command is byte-identical
(verified in the render output). `स्थानांतरण` matches the shipped noun in
`cli.sessions_help.move_queue` / `move_cancel_queued`.

Also checked in code: `unsupported_flag(flag=repr(token))` (L144) receives an
already-quoted argument, and the Hindi slot `… इस्तेमाल कर सकते हैं, {flag} नहीं` reads
correctly with it (`'--force' नहीं`); `cannot_wait` and `to_needs_device` are refusal
tier with plain, blameless tone and untouched literals; `cli_unavailable` keeps
`CLI` Latin and passes `{reason}` through verbatim.

## Second independent pass (high-visibility group)

Settings copy plus error-tier notices is a high-visibility group, so the whole set was
re-read cold from the English after the first pass, with each string re-derived rather
than compared to my own notes. The second pass found the NIT below and changed nothing
else: it re-confirmed `ढीली` (not a re-invented verb), the four safety claims, the
`खास` qualifier, the `लर्निंग्स` compound, and that `एग्ज़िट`/`प्रोवाइडर` are the shipped
words rather than plausible synonyms. No BLOCKER surfaced on either pass.

## Findings

### HI-A2-01 — NIT — `cli.config.running_sessions_unchanged`

- **Current**: `… कॉन्फ़िगरेशन में बदलाव सहेजने से उनकी स्वीकृति की शर्तें ढीली नहीं हो सकतीं; …`
- **Problem**: for the same actor ("a config write") the shipped copy uses the verb
  `लिखना`, in the two strings this caution line is the CLI twin of —
  `wire.settings.approvals.description` ("कॉन्फ़िग लिखने से हरेक चल रहा सत्र एक साथ सख़्त
  हो जाता है") and `wire.settings.tool_approval_mode.help` ("ढीला करने के लिए इसी सत्र से
  लिखना पड़ेगा — /approvals auto।"). `कॉन्फ़िगरेशन में बदलाव सहेजना` is understandable and
  not wrong, so this is polish, not a gate: meaning, polarity and all four safety claims
  are intact and were read as such above.
- **Suggested**: `चल रहे सत्रों में कोई बदलाव नहीं हुआ — कॉन्फ़िग लिखने से उनकी स्वीकृति की शर्तें ढीली नहीं हो सकतीं; जिन सत्रों में आप बिना स्वीकृति के टूल चलाना चाहते हैं, उनमें से हर एक में /approvals auto टाइप करें। नए सत्र auto मोड में शुरू होते हैं।`

No other finding. Considered and deliberately **not** raised: `खास` vs the single shipped
`विशिष्ट` (different collocation, no glossary row); `सहेजी जाने वाली` rendering "retain"
with the same verb the auto-save row uses for "save" (both transparent, no ambiguity about
what is kept); `अपने-आप सहेजी जाए या नहीं` phrasing (matches the shipped label's shape).

## Commands actually run

```
python3 scripts/i18n/check.py                    # rc 0
python3 scripts/i18n/check.py --catalogues-only  # rc 0, 0 problems
.venv/bin/python validate_hi_v2.py               # own script: structure/ICU/placeholders/
                                                 # literals/encoding/width + check_catalogues
                                                 # on a private temp root (0 problems)
.venv/bin/python render_realpath.py              # messages.render(..., locale="hi"), 22/22
                                                 # provenance hi, rc 0
python3 recount_hi_terms.py                      # shipped-usage recounts quoted above
python3 diff_v1_v2.py                            # 9 changed / 13 unchanged
shasum -a 256 (keys.json, hi.v2.json, hi.json, en catalogues)
python3 scripts/i18n/ledger.py append … --apply  # one row per namespace, rc 0
python3 scripts/i18n/ledger.py validate --base HEAD   # "audit history: valid and append-only"
```

Full outputs: `audit.log` in this bundle (sections 1–5, each with its rc).

## Limitations

- Independent LLM review on the harness-served `deepseek/deepseek-flash`, **not**
  native-speaker certification. Judgments are reasoned from the guides, the glossary, the
  shipped catalogue and the use sites.
- The candidate is not in the tree: `hi/cli.config.json` and `hi/tui.session_move.json` do
  not exist yet, so this audit judges candidate bytes preserved in the bundle. A passing
  ledger row is recorded for those bytes; if the bytes that land differ, the row is stale
  and must be re-audited.
- Scalar text evidence only — no terminal frame was captured; widths are `cell_len`
  measurements, not pixels. Not claimed: rendered-frame or native-speaker coverage.
- Pre-existing shipped defects (`wire.settings.classification_notice.help` agreement slip)
  are cited only as terminology evidence and are not gated by this audit.
