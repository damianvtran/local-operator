# Independent audit — ur / cli.permissions — urp1 (2026-11-09)

- PR: damianvtran/local-operator#3691, worktree HEAD `50badda034d755ebb049c4f542f3073439bf823e`
- Auditor: i18n-auditor (lopdev team subagent), session `ffe80a68ac43`, model `anthropic/claude-opus-5-5`
  (host system prompt; no fallback). Independent of translator (PR coder agent, job `ad99821ab9e1`,
  `anthropic/claude-opus-5-5`) — different agent/session, same model family: not an independent vendor.
- Inputs: en `eba1e1dd…bf8f`, context `aca67def…15ce`, ur `589e1fa1…08bd` (full hashes in record.json).
- Read: docs/I18N.md "Add or extend a locale" 2–6, i18n/glossary.md, i18n/style-guides/ur.md,
  i18n/audits/evidence/README.md, `ledger.py append --help`, `local_operator/permissions_cli.py`
  (plain `print`, no RTL isolation projection), `local_operator/i18n/display.py`.
- No MT/translation service used. One web fetch of Apple's macOS feature-availability page to
  establish which macOS UI languages exist (fact about the OS, not a translation aid).

## Verdict: `changes_requested` — 0 BLOCKER, 2 MAJOR, 1 MINOR, 2 NIT

All 28 keys read against source + context. Mechanical layer clean (check.py 0 problems; placeholders,
backtick spans, `[y/N]` + trailing space, verbatim row names/Developer ID/codesign/guide URL, no bidi
controls, Latin digits, key set and order identical). Register is consistent آپ-respectful imperative;
glossary terms (سیشن، ڈیسک ٹاپ ایپ) consistent. Accuracy of all non-UI-label text is good.

The gating problem is one root cause: **macOS has no Urdu system language.** Apple's
feature-availability "System Language" list (fetched 2026-11-09) contains Arabic, Hebrew, Hindi, … but
not Urdu (Urdu only appears as a keyboard input). A Mac belonging to an Urdu-locale `lop` user therefore
shows System Settings and the keychain dialog in English (or another language). The context note says
"use the localized macOS name"; for `ur` there is none, and the translator coined Urdu labels the user
will never see. In permission/safety copy that tells the user exactly what to click, that is a navigation
failure, not a style choice. (hi is a macOS system language, so this does not transfer to hi.)

### U1 — MAJOR — `cli.permissions.status.screen.denied`
- Current: `… اجازت نہیں ہے۔ سسٹم سیٹنگز > رازداری اور سیکیورٹی > اسکرین ریکارڈنگ میں "{row}" آن کریں۔`
- Problem: the menu path does not exist on any Mac; "رازداری اور سیکیورٹی" in particular is a translation,
  not recognisable as "Privacy & Security", and System Settings search will not match it. Also mixes
  رازداری here with پرائیویسی everywhere else in the namespace for "privacy".
- Suggested: `اسکرین ریکارڈنگ (رن ٹائم پری فلائٹ): اجازت نہیں ہے۔ System Settings > Privacy & Security > Screen Recording میں "{row}" آن کریں۔`
  (Latin path as one LTR island, byte-identical to the macOS labels; same treatment as other kept-Latin UI.)

### U2 — MAJOR — `cli.permissions.setup.explain`, `cli.permissions.setup.always_allow`
- Current: `… "ہمیشہ اجازت دیں" منتخب کریں تاکہ …` and `کیا آپ نے "ہمیشہ اجازت دیں" منتخب کیا (یا کوئی ڈائیلاگ ظاہر نہیں ہوا)؟ [y/N] `
- Problem: the keychain dialog button reads "Always Allow" (English) for this user; the dialog also has
  "Allow" and "Deny". Telling the user to pick a label that is not on screen risks them choosing "Allow"
  (works once, then every later `lop update` re-prompts) — and the follow-up y/N asks whether they
  clicked a button that never existed, so the answer is unreliable.
- Suggested explain tail: `"Always Allow" منتخب کریں تاکہ آئندہ `lop update` بغیر پوچھے سائن کر سکے۔`
- Suggested always_allow: `کیا آپ نے "Always Allow" منتخب کیا (یا کوئی ڈائیلاگ ظاہر نہیں ہوا)؟ [y/N] `
  (Optional gloss if wanted: `"Always Allow" (ہمیشہ اجازت دیں)` — keep the Latin label first.)

### U3 — MINOR — en context sidecar / ur style guide (not a ur catalogue byte)
- `cli.permissions.context.json` notes say "use the localized macOS name/label". For ur this instruction
  has no referent and produced U1/U2. Suggest adding to the context note: "If macOS is not localized in
  the target language (ur), keep the English macOS label verbatim," and recording the call in
  `i18n/style-guides/ur.md` (Product-name policy) so future macOS-facing ur copy is consistent.

### U4 — NIT — `cli.permissions.status.identity.legacy`
- Current: `… پاتھ سے پہچانی جانے والی امیج (یہاں bundle نہیں بن سکا)، …`
- Problem: "bundle" is a common noun here, not an identifier; the same sentence transliterates امیج, پاتھ,
  بلڈ. Mixed treatment.
- Suggested: `(یہاں بنڈل نہیں بن سکا)`.

### U5 — NIT (non-gating; bidi observation) — `help.permissions`, `setup.refused.tty`, `setup.no_app`
- These three are the only ur lines whose first strong character is Latin (logged). `permissions_cli.py`
  prints with plain `print` (no `isolated_status_text` RLI wrap), so a first-strong bidi terminal lays
  these lines out LTR and the Urdu clause order reads scrambled; other lines start in Urdu and resolve RTL.
  The ur guide's degraded-TUI stance forbids contorting copy, so this is optional; where natural Urdu
  order exists anyway, prefer it:
  - help.permissions: `وہ macOS پرائیویسی شناخت دیکھیں یا سیٹ اپ کریں جو Local Operator رن ٹائمز استعمال کرتے ہیں`
  - setup.refused.tty: `کمانڈ `lop permissions setup` انٹرایکٹو ہے۔ اسے ٹرمینل میں چلائیں۔`
  - setup.no_app: leave as is (no natural reorder).

## Not findings (checked)
- Width: no `maxCells` in the sidecar; ur cell lengths track en (max explain 251 vs 221); multi-sentence
  CLI lines that wrap anyway. No overflow finding.
- Straight ASCII quotes around verbatim Latin row names follow the guide ("never wrap Latin names in Urdu
  punctuation"); after U2 the only quoted Urdu text disappears.
- "پری فلائٹ" transliteration in a parenthetical technical label: acceptable.
- Safety meaning: cancelled ("کچھ بھی تبدیل نہیں ہوا"), failed ("قطار ہی رکھیں گے"), done (running sessions
  keep current image until restart) all preserve polarity and scope exactly.
