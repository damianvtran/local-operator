# Independent i18n audit — `hi` — slice C1-A (core CLI + TUI)

- **Auditor**: `i18n-auditor` (lopdev team), session `bd8cbf2491b2`, **actual served model `deepseek/deepseek-flash`** (harness-disclosed; no model was requested for this lane).
- **Independence**: did not translate these strings, did not edit them, and has no access to the translator lane's reasoning. Translator of record: `openai/gpt-6.1-sol` (per the brief).
- **Audit date**: 2026-11-11
- **Worktree**: `~/local-operator/.worktrees/i18n-c1` @ `b188079f32e6d978ad78fac96edc5c3516d028ab` (clean at audit time)
- **Scope**: 22 keys — 15 `cli.config.*` (`lop config create/open/edit/list` output) + 7 `tui.session_move.*` (`/move --to` refusals), candidate translation for `hi`.
- **Inputs (read-only)**: en source + context in `…/c1-prep/keys.json` (sha256 `ac35dc123a4241a3c1fdbd0f4f012636d9706695bacfb3601aa06eed2cae1ffc`); candidate `…/c1-prep/hi.json` (sha256 `b67c087da565eba745ccf5b48421f92ac0f27907e9c220b57e59f93a4525b355`); `i18n/style-guides/hi.md` (`82292dc8…`) and `i18n/glossary.md` (`0eb34437…`); shipped `hi` catalogues for terminology consistency; use sites `local_operator/cli.py`, `local_operator/i18n/keys/cli_config.py`, `local_operator/i18n/keys/tui_session_move.py`, `local_operator/tui/session_move.py`.

## Verdict

**changes_requested** — 0 BLOCKER, **2 MAJOR**, 3 MINOR, 4 NIT.

The translation is substantively accurate: no meaning, polarity, negation or safety failure; every placeholder, typed token and backticked command survives byte-identically; the safety line about approvals keeps its "cannot loosen / type the command in each session / new sessions open at auto" content, and the timed-out move is correctly reported as an **unknown outcome**, not a failure. Both MAJOR findings are terminology couplings to shipped `hi` copy: they name an already-settled concept with a second, different word, which the operator's "no seams in the interface" acceptance criterion treats as a defect. Each fix is a one-word swap.

## Mechanical checks (all executed, output in `audit.log`)

| Check | Result |
|---|---|
| Repo validator `python3 scripts/i18n/check.py` | `ok — 217 file(s) scanned, 2540 literal(s) at or below the baseline, catalogues consistent; 83 advisory` (exit 0) |
| Repo catalogue checker `check_catalogues()` over a temp root = shipped `en` + `hi` trees **plus the candidate files** | 0 problems (control run on the untouched repo tree: 0 problems) |
| Key set / order vs `keys.json` | identical, 22/22, order-identical |
| Placeholder set & types per key | identical (`{path}`, `{key}`, `{value}`, `{description}`, `{flag}`, `{reason}`, `{code}`, `{session}`, `{to}`) |
| ICU parse + render of every value with `runtime.render_message(..., "hi")` | all parse; 9 strings render, no syntax/format error. No key carries a plural/select, so no CLDR category obligation arises here (locale table is `one`/`other`, `one` = 0 and 1) |
| Backtick spans | identical modulo placeholders: `` `lop sessions move <id> --to <to> --wait <seconds>` `` and `` `lop sessions move {session} --to {to}` `` verbatim |
| Typed tokens | `/approvals auto`, `auto`, `/move`, `--to`, `--keep`, `--queue`, `--drop-session-roles`, `--wait`, `local`, `<id>`, `<to>`, `<seconds>`, provider ids, `CLI`, `AI` — all preserved |
| Empty / English-identical values | none |
| Stray Latin in target not present in source | none (all Latin runs trace to source tokens) |
| Typography | no ZWJ/ZWNJ, no double spaces, no Devanagari digits, no `!`, no `कृपया`, no तुम/तू; danda । for full stops, Latin `;` `:` `—` as the guide allows; nukta encoded as base+U+093C (9 uses, 0 precomposed) matching the shipped catalogue (1377 uses, 0 precomposed) |
| Width | no `maxCells` caps in this group; the one capped context (`cli.config.list_title`, ~30 cells) measures **12 cells** (`rich.cells.cell_len`) vs 21 for the English — comfortably inside. Longest strings are the 161-cell approvals clarification and the 91-cell `/move` ambiguity hint, both "no cell cap; one terminal line, may wrap" |

## Findings

### HI-A01 — MAJOR — `cli.config.list_desc_hosting`

- **Current**: `AI प्रदाता का प्लेटफ़ॉर्म (जैसे radient, openai, deepseek, anthropic, openrouter)`
- **Problem**: `प्रदाता` is a second name for a concept the shipped `hi` copy has already settled as `प्रोवाइडर` — 68 occurrences across 17 shipped catalogue files, and specifically **this setting's own shipped label and help**: `wire.settings.hosting.label` = "डिफ़ॉल्ट प्रोवाइडर", `wire.settings.hosting.help` = "नई बातचीत के लिए प्रोवाइडर।". `प्रदाता` appears exactly once in shipped copy (a help line in `cli.group_help`), so it is neither the settled term nor the majority form. A user reading `lop config list` and then the settings page meets the same setting under two names. The hi guide's "prefer the settled term" rule covers the loan form too.
- **Suggested**: `AI प्रोवाइडर प्लेटफ़ॉर्म (जैसे radient, openai, deepseek, anthropic, openrouter)`

### HI-A02 — MAJOR — `tui.session_move.cli_no_answer`

- **Current**: `CLI बिना जवाब दिए निकास कोड {code} के साथ बंद हो गया`
- **Problem**: `निकास कोड` is a Sanskritised coinage for "exit code" that appears **nowhere** in the shipped catalogues, while `एग्ज़िट कोड` is the established term in this same CLI surface family: `cli.agents_help.push`, `cli.sessions_help`, `cli.teams_help`, `wire.remote_compute` (one each, and zero uses of `निकास` anywhere). The hi guide rejects exactly this shape of invention — "prefer common Hindi words over heavy Sanskritisation; where an English loan is the living term … keep it rather than inventing a stilted native form". Meaning is not at risk; the seam is.
- **Suggested**: `CLI बिना जवाब दिए एग्ज़िट कोड {code} के साथ बंद हो गया`

### HI-A03 — MINOR — `cli.config.running_sessions_unchanged` (safety line)

- **Current (fragment)**: `… कॉन्फ़िगरेशन में बदलाव करने से उनकी स्वीकृति की शर्तें कम नहीं होतीं; …`
- **Problem**: "cannot loosen" is rendered as "their approval conditions do not become fewer". The direction is right — no polarity error — but `शर्तें कम होना` measures a count, and it diverges from the shipped `hi` idiom for loosening approvals, `ढीला करना`: `wire.settings.approvals.description` ("किसी एक को ढीला करने के लिए …"), `wire.settings.tool_approval_mode.help` ("ढीला करने के लिए इसी सत्र से लिखना पड़ेगा — /approvals auto।"). On a safety clarification the shipped collocation is worth matching verbatim.
- **Suggested (whole line)**: `चल रहे सत्रों में कोई बदलाव नहीं हुआ — कॉन्फ़िगरेशन में बदलाव लिखने से कोई चल रहा सत्र ढीला नहीं होता; जिस सत्र को बिना स्वीकृति के चलाना हो, उसमें /approvals auto टाइप करें। नए सत्र auto मोड में शुरू होते हैं।`

### HI-A04 — MINOR — `cli.config.list_desc_classification_notice`

- **Current**: `[अप्रचलित] स्मार्ट संकेत चैट में कुछ नहीं दिखाते; कॉल और उसकी लागत सत्र लॉग में दर्ज होती है`
- **Problem**: subject–verb agreement. The subject is the coordination `कॉल और उसकी लागत` (the call **and** its cost), so the verb must be plural: `दर्ज होती है` agrees only with `लागत`. (The shipped `wire.settings.classification_notice.help` has a comparable slip with a single subject — that is not a model to copy.) Meaning is unaffected.
- **Suggested**: `… ; कॉल और उसकी लागत सत्र लॉग में दर्ज होते हैं`

### HI-A05 — MINOR — `cli.config.list_desc_max_learnings_history`

- **Current**: `[अप्रचलित — अब इसकी जगह संपीडन है] सीखी गई बातों की सहेजी जाने वाली प्रविष्टियों की अधिकतम संख्या`
- **Problem**: two problems in one string. (a) Terminology: the shipped label for this very setting is `wire.settings.max_learnings_history.label` = "लर्निंग्स इतिहास अधिकतम", i.e. the product word is `लर्निंग्स`; the description renames the concept to `सीखी गई बातें`, so the same setting carries two names. (b) Idiom: three stacked `-की` genitives (`बातों की … प्रविष्टियों की … संख्या`) make a heavy noun pile that reads like a legal clause in a one-line settings row.
- **Suggested**: `[अप्रचलित — अब इसकी जगह संपीडन है] सहेजी जाने वाली लर्निंग्स प्रविष्टियों की अधिकतम संख्या`

### HI-A06 — NIT — `cli.config.list_desc_detail_length`

- **Current**: `[अप्रचलित — अब इसकी जगह संपीडन है] बातचीत के इतिहास में बिना सारांश बनाए रखे जाने वाले हाल के संदेशों की संख्या`
- **Problem**: `बिना सारांश बनाए रखे जाने वाले` packs a negated gerund, a passive participle and a genitive into one modifier before the head noun; a reader has to re-parse it, and it is the same class of pile as HI-A05 rather than the flat phrasing the sibling rows use. Meaning is correct ("messages left unsummarized").
- **Suggested**: `… बातचीत के इतिहास में हाल के जिन संदेशों का सारांश नहीं बनाया जाता, उनकी संख्या`

### HI-A07 — NIT — `tui.session_move.ambiguous`

- **Current**: `कार्यशील डायरेक्टरी के लिए /move <पाथ> या सत्र को दूसरे डिवाइस पर ले जाने के लिए /move <सत्र> --to <पीयर> इस्तेमाल करें।`
- **Problem**: two small points. (a) `<पाथ>` is the transliteration; the shipped copy is genuinely mixed (`पाथ` in `cli.permissions`, `wire.errors.desktop`; `पथ` in `tui.viewer` "y पथ कॉपी करें", `wire.memory` "पथ, URL और ID"), and the standard spelling is `पथ` — a placeholder hint the user reads but does not type is the safer place to standardise. `--to <पीयर|local>` and `<पीयर>` are consistent with shipped `पीयर` (12 occurrences), so no change there. (b) `दूसरे डिवाइस पर` ("to another device") is an addition the English does not make; it is a harmless clarifier and I am not asking for its removal — only recording that it is an addition.
- **Suggested**: `कार्यशील डायरेक्टरी के लिए /move <पथ> या सत्र को दूसरे डिवाइस पर ले जाने के लिए /move <सत्र> --to <पीयर> इस्तेमाल करें।` (`कार्यशील डायरेक्टरी` itself is correct and matches shipped `cli.exec_help.run_in`.)

### HI-A08 — NIT — `cli.config.updated`

- **Current**: `{key} का मान {value} कर दिया गया`
- **Problem**: `का मान {value} कर दिया गया` only reads naturally when `{value}` is a bare noun-ish word. The sidecar says `{value}` may be `True`/`False`, a number, `auto`, or a list, so `का मान True कर दिया गया` / `का मान a, b कर दिया गया` is functional but stiff. The dropped "Successfully" is fine — `कर दिया गया` carries completion, and the guide forbids padding.
- **Suggested**: `{key} को {value} पर सेट कर दिया गया`

### HI-A09 — NIT — `cli.config.list_desc_auto_save_conversation`

- **Current**: `बातचीत अपने आप सहेजी जाए या नहीं`
- **Problem**: spelling of the shipped `auto` adverb. The shipped catalogues use the hyphenated form `अपने-आप` 35 times vs `अपने आप` 5 times, including the label of this very setting (`wire.settings.auto_save_conversation.label` = "बातचीत अपने-आप सहेजें"), which sits two rows from this description in the settings page.
- **Suggested**: `बातचीत अपने-आप सहेजी जाए या नहीं`

## Second independent pass (high-visibility group)

Requested by the manager for settings/error copy, so the whole group was re-read cold after the first pass, with each string re-derived from the English rather than from the first set of notes. The second pass produced HI-A04 (agreement) and HI-A09 (`अपने आप`), and re-confirmed the two MAJOR couplings. No additional BLOCKER was found. Specific re-verification of the safety and outcome claims:

- `cli.config.running_sessions_unchanged` — approximates "a config write cannot loosen a running session" (`कम नहीं होतीं`, direction correct), keeps `/approvals auto` verbatim, keeps "in each session" distributive (`जिस सत्र में … उसमें`), and keeps "new sessions open at auto" (`नए सत्र auto मोड में शुरू होते हैं`, consistent with shipped `टूल स्वीकृति मोड` and the Latin mode word `auto`).
- `tui.session_move.deadline_exceeded` — says the move did not complete within the bound (`तय समय में पूरा नहीं हुआ`) and that re-running reports how far it got (`कहाँ तक पहुँचा`). It does **not** say failed or succeeded; `स्थानांतरण` matches the shipped move vocabulary in `cli.sessions_help.move_*`.
- `tui.session_move.cannot_wait` / `unsupported_flag` / `to_needs_device` / `cli_unavailable` — refusal tone is plain, no blame, no alarm; `local` stays Latin; the runnable command and every flag are untouched.
- No English source sentence survives untranslated anywhere in the group; the only Latin left is kept-English by policy (product/command/flag/provider tokens, `CLI`, `AI`, the mode word `auto`).

## Notes and limitations

- Independent LLM review on the harness-served fallback model (`deepseek/deepseek-flash`), not native-speaker certification. Judgments are reasoned from the guide, the glossary, the shipped catalogue and the use sites; no translation/MT service was used for comparison or back-translation.
- The candidate lives in another session's scratchpad and is **not yet in the tree**: `local_operator/i18n/catalogues/hi/cli.config.json` and `…/tui.session_move.json` do not exist yet, so this audit judges candidate bytes (`target-candidate.json` is the exact audited payload, kept so the verdict is reproducible before the translation lands). One audit-ledger row per namespace was appended to `i18n/audit-history/hi/{cli.config,tui.session_move}.json` with status **`changes_requested`** and `unresolved_blocker_major: 1`, referencing this bundle; `scripts/i18n/ledger.py validate --base HEAD` returns "audit history: valid and append-only". A passing row must supersede these after remediation, whose auditor re-reads the fixed strings.
- `scripts/i18n/check.py` reports an advisory ratchet row for `local_operator/tui/session_move.py` (4 literals against a ceiling of 3). I checked it: the differing literals are machine tokens (refusal `code` values such as `cli_unavailable` / `deadline_exceeded`), not user-facing English; the refusal message bodies come from the keys under audit. Advisory scope only, not a finding against this translation.
- RTL dimension: not applicable (`hi` is LTR).
- Pre-existing shipped strings with their own defects (e.g. `wire.settings.classification_notice.help`) are **not** gated by this audit; they are cited only as evidence of the settled terminology.

## Commands actually run

```
python3 scripts/i18n/check.py                     # exit 0, catalogues consistent
python3 ~/.local-operator/sessions/bd8cbf2491b2/scratchpad/validate_hi.py
    # check_catalogues() over temp root = en+hi trees + candidate files → 0 problems
    # control run on the real repo tree → 0 problems
    # key set/order, placeholder sets, backtick spans, typed tokens, empty/EN-identical,
    # stray-Latin, ICU parse+render(hi) → 0 contract problems
    # rich.cells.cell_len measurements for the capped title row
grep/grep -o over local_operator/i18n/catalogues/{en,hi}/*.json   # term counts cited above
git rev-parse HEAD ; git status --short
for ns in cli.config tui.session_move; do
  expected=$(python3 scripts/i18n/ledger.py history-sha --locale hi --namespace $ns)
  python3 scripts/i18n/ledger.py append --receipt …/receipt-hi-$ns.json \
    --expected-history-sha "$expected" --dry-run        # both: dry rc=0
done
# then the same two appends with --apply          # both: rc=0
python3 scripts/i18n/ledger.py validate --base HEAD   # audit history: valid and append-only
```

Evidence bundle: `i18n/audits/evidence/hi/cli.config/c1a/` and `i18n/audits/evidence/hi/tui.session_move/c1a/`.
