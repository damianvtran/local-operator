# chrome25-safety-b2 — delta re-audit of `relay.chrome` safety copy

**Audit ID:** `chrome25-safety-b2`  
**Round:** R2, delta-only (seven-locale unified remediation after `chrome25-safety-b1`)  
**Namespace:** `relay.chrome`  
**Locales:** es, fr, hi, ru, ur, vi, zh-CN  
**Auditor:** translation-auditor — independent second safety audit  
**Session / model:** `acd16c49df72` / `radient/auto` (harness-reported selection; underlying provider model not exposed in-session; no fallback selected or used)  
**Date:** 2026-10-25  
**Worktree:** `~/local-operator/.worktrees/relay-chrome-i18n`  
**Base commit:** `535cea359825411741069cbffcee42e2927df9d3`  
**Predecessor:** `chrome25-safety-b1` report `a90d11502f4946e49846225bc4a7fa6616bb715512f7ba08511eb4e93d777812`, record `4c9225e418581c01f41d532c6cd0facb04fe7b7e2babd423a8cbfdd8b6f5ab57`.

**Independence:** this is the same independent auditor as b1, doing a delta round. No translator or first-auditor *narrative* report was used; the structured `remediation.json` records named below were read as instructed. All judgement below is against the immutable English source/context and the current bytes.

## 1. Input integrity

English source and context are immutable and match the pinned values:

| Input | Path | SHA-256 |
|---|---|---|
| English source | `local_operator/i18n/catalogues/en/relay.chrome.json` | `f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac` |
| English context | `local_operator/i18n/catalogues/en/relay.chrome.context.json` | `1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a` |

Current target files (the bytes audited):

| Locale | Current target SHA-256 |
|---|---|
| `es` | `416f3d6c884403ea962f158b1718f3743352674b9e7ffc5a47aed4c7520df333` |
| `fr` | `5d3a3387e7324a1ed5b1b51104fd06da1470285674f3ce62e24e0eaf33bbb6d2` |
| `hi` | `eb9f326ca6689ed0d35311cd75ae8cb5683b515e8b638c938790759eeec96784` |
| `ru` | `fb17616cb7875507eb4b16853cb7b229a1176f46349d9c942d9cc8a427e6c97c` |
| `ur` | `a90b9bea54c17e3b74527bb7c503363df348d1d625b105431ff8bbed56f39fff` |
| `vi` | `a875b7f4177f4e3d9118a5d779734db751e350c384e01ae6059916bec5c8c1d9` |
| `zh-CN` | `ba4215e753320387d61bc9d9661eb2a63420473436e492f6da657ac8595c1855` |

Before snapshots are the actual pre-remediation files referenced by the remediation records. Each full-file SHA-256 equals the b1 target hash, so unchanged values are carried by byte proof, not by re-reading a summary:

| Locale | Before snapshot | SHA-256 | Equals b1 |
|---|---|---|---|
| `es` | `~/.local-operator/sessions/d69e1fa9cd3c/scratchpad/chrome25-es-remediation/relay.chrome.es.before.json` | `9e1b68d5e546317ff912e1aa9d463d9ae6d44d7b109e0e584ef5e82a1a667192` | yes |
| `fr` | `~/.local-operator/sessions/132edc7afe5f/scratchpad/chrome25-fr-remediation/candidate-pristine.json` | `b3a97316412b6b35c0d883a0e79a7cf8b9ed88c99bf298920cd235a2d7a6916f` | yes |
| `hi` | `~/.local-operator/sessions/6f3832cb71c0/scratchpad/hi-relay-chrome.candidate-pre-remediation.json` | `a29563f5dd0a9f425b717639c28e6a8f4c97382d45b0928bacdf5e12d31285b3` | yes |
| `ru` | `~/.local-operator/sessions/635b55fcabb6/scratchpad/chrome25-ru-remediation/candidate-before.json` | `27709627e9deb166ae6954a142400f482876b940681819ab6b49234f88d738fd` | yes |
| `ur` | `~/.local-operator/sessions/d409548ee35d/scratchpad/chrome25-ur-remediation/candidate-before.json` | `6059005113786f1d74fc848a9f254fadd5fba259be11aecfcbf59ebe2cd24d28` | yes |
| `vi` | `~/.local-operator/sessions/70a22e9a7716/scratchpad/chrome25-vi-remediation/relay.chrome.pre-remediation.json` | `97a991ac3245c8b4edd59877eb1d1259affb1c8c77bedd460a0721c08a1d27b8` | yes |
| `zh-CN` | `~/.local-operator/sessions/5217c4e5f8e3/scratchpad/zh-CN-relay.chrome.candidate-before-remediation.json` | `ca2c2df9bf514ef301af0feb86062a96e17f5a2e40d2f8a1febed26f5746646e` | yes |

## 2. Coverage retained and precise delta

Original coverage was 17 keys per locale (119 target values). Current files still carry all 83 keys in English order, with no added/removed keys. **25** of the 119 original safety-scope values changed; **94/119** are byte-identical to b1 (canonical JSON comparison of the unchanged subset; per-locale subset hashes are in `record.json`).

| Locale | Original keys | Changed scope keys | Unchanged (byte proof) | Catalogue keys | Order |
|---|---:|---:|---:|---:|---|
| `es` | 17/17 | 2 | 15 | 83 | identical |
| `fr` | 17/17 | 4 | 13 | 83 | identical |
| `hi` | 17/17 | 5 | 12 | 83 | identical |
| `ru` | 17/17 | 3 | 14 | 83 | identical |
| `ur` | 17/17 | 5 | 12 | 83 | identical |
| `vi` | 17/17 | 3 | 14 | 83 | identical |
| `zh-CN` | 17/17 | 3 | 14 | 83 | identical |

**Read count:** 17 values per locale × 7 = **119 original values**; **25 changed safety-scope values** re-read semantically; **94 unchanged values** carried by byte proof. Out-of-scope changed keys (6 es, 4 fr, 2 hi, 1 ru, 3 ur, 2 vi, 2 zh-CN) are outside this safety delta and were not reviewed.

## 3. Method

- Exact delta produced by comparing each current catalogue value against the before snapshot whose full-file hash equals b1.
- All 25 changed safety-scope keys re-read against the English source and context sidecar; unchanged coverage carried by the same-key byte proof.
- Structured remediation records read: `~/.local-operator/sessions/d69e1fa9cd3c/scratchpad/chrome25-es-remediation/remediation.json`, `~/.local-operator/sessions/132edc7afe5f/scratchpad/chrome25-fr-remediation/remediation.json`, `~/.local-operator/sessions/6f3832cb71c0/scratchpad/remediation.json`, `~/.local-operator/sessions/635b55fcabb6/scratchpad/chrome25-ru-remediation/remediation.json`, `~/.local-operator/sessions/d409548ee35d/scratchpad/chrome25-ur-remediation/remediation.json`, `~/.local-operator/sessions/70a22e9a7716/scratchpad/chrome25-vi-remediation/remediation.json`, `~/.local-operator/sessions/5217c4e5f8e3/scratchpad/remediation.json`.
- ICU argument names/kinds, select/plural branch sets, `#` presence, NFC, bidi controls and `rich.cells.cell_len` widths re-checked on the changed keys; unchanged keys inherit b1's checks plus byte identity.
- **Source-semantic only.** No browser, rendered frame, or runtime. Width numbers are text-budget evidence, not clipping proof; rendered checks remain for QA/design.

## 4. Original b1 findings — per-finding disposition

| b1 ID | Locale / key | b1 severity | Current evidence | Disposition |
|---|---|---|---|---|
| `MAJ-01` | `ur` `relay.chrome.link.since_seconds` | major | `… · سنک # سیکنڈ پہلے`; one/other + `#`; 41/44 at one-digit, 42/44 at two-digit | **fixed** |
| `MAJ-02` | `ur` `relay.chrome.link.since_minutes` | major | `… · سنک # منٹ پہلے`; one/other + `#`; 39/44 at one-digit, 40/44 at two-digit | **fixed** |
| `MAJ-03` | `vi` `relay.chrome.session.ended` | major | `Phiên kết thúc · lịch sử đã lưu`; 31/32; state, not imperative | **fixed** |
| `MAJ-04` | `fr` `relay.chrome.session.ended` | major | `Terminée · historique conservé`; 30/32; subject dropped, history kept | **fixed** |
| `MIN-01` | `es` `relay.chrome.session.ended_hint` | minor | `Enviar un mensaje o Reanudar la reabre…`; 61 cells, no maxCells | **fixed** |
| `MIN-02` | `es` `relay.chrome.session.degraded` | minor | `No responde — los mensajes esperarán`; 36/36 | **fixed** |
| `MIN-03` | `fr` `relay.chrome.session.subtitle.waiting` | minor | `Vous attend`; 11/24 | **fixed** |
| `MIN-04` | `fr` `relay.chrome.session.ended_hint` | minor | `Envoyer un message ou « Reprendre »…`; 74, no maxCells; button word preserved | **fixed** |
| `MIN-05` | `fr` `relay.chrome.list.mark_all` | minor | `Tout marquer comme lu`; 21/20; no clipping measured; b1 fallback ungrammatical | **rejected — accepted width residual** |
| `MIN-06` | `fr` `relay.chrome.session.degraded` | minor | `Ne répond pas — les envois attendent`; 36/36; `envois` corpus-backed | **fixed (width); lexical component withdrawn** |
| `MIN-07` | `hi` `relay.chrome.list.filter.needs_you` | minor | `आपका ध्यान चाहिए · {count, number}`; 15/16, 16/16 at two digits, 17/16 at three | **fixed** |
| `MIN-08` | `hi` `relay.chrome.list.mark_all` | minor | `सभी को पढ़ा हुआ चिह्नित करें`; 18/20 | **fixed** |
| `MIN-09` | `ur` `relay.chrome.session.degraded` | minor | `جواب نہیں دے رہا — پیغامات رکے ہیں`; 34/36; plural intransitive | **fixed** |
| `MIN-10` | `vi` `relay.chrome.list.mark_all` | minor | `Đánh dấu tất cả đã đọc`; 22/20; no clipping measured; alternatives lose idiom/scope | **rejected — accepted width residual** |
| `NIT-01` | `ur` `relay.chrome.list.mark_all` | nit | `سب پڑھا نشان زد کریں`; 20/20; grammatical 23/20 | **rejected — accepted nit** |
| `NIT-02` | `vi` `relay.chrome.session.ended_hint` | nit | `… sẽ khôi phục phiên …`; 70, no maxCells; result now says restore | **fixed** |

**Summary:** 13 fixed; 3 rejected/withdrawn with rationale (all minors/nits, no blocker/major); **0 blocker, 0 major remains**. In particular: Urdu now says `سنک # …` (explicit synced age), Vietnamese now says `lịch sử đã lưu` (history preserved as state), Hindi now says `आपका ध्यान चाहिए` (operator attention), and French ended is naturally shorter at 30/32.

## 5. Changed safety-scope keys reviewed

### es — 2 changed safety-scope key(s)
- `relay.chrome.session.ended_hint` — before: `Un envío o Reanudar la reabre en tu carpeta personal`; after: `Enviar un mensaje o Reanudar la reabre en tu carpeta personal`; width 61; maxCells None.
  b1 MIN-01 fixed: `Un envío` → verb `Enviar un mensaje`; `Reanudar` and `carpeta personal` preserved. 61 cells, no maxCells.
- `relay.chrome.session.degraded` — before: `No responde — los envíos esperarán`; after: `No responde — los mensajes esperarán`; width 36; maxCells 36.
  b1 MIN-02 fixed: `los envíos` → `los mensajes`; wait sense retained. 36/36.

### fr — 4 changed safety-scope key(s)
- `relay.chrome.session.subtitle.waiting` — before: `En attente de vous`; after: `Vous attend`; width 11; maxCells 24.
  b1 MIN-03 fixed: `En attente de vous` → native `Vous attend`. 11/24.
- `relay.chrome.session.ended` — before: `Session terminée · historique conservé`; after: `Terminée · historique conservé`; width 30; maxCells 32.
  b1 MAJ-04 fixed: `Session terminée…` → `Terminée · historique conservé`. 38 → 30/32; history-kept retained.
- `relay.chrome.session.ended_hint` — before: `Un envoi ou Reprendre la rouvre dans votre dossier personnel`; after: `Envoyer un message ou « Reprendre » la rouvre dans votre dossier personnel`; width 74; maxCells None.
  b1 MIN-04 fixed: `Un envoi` → `Envoyer un message`, `Reprendre` quoted with U+202F. 74 cells, no maxCells; button word + home folder preserved.
- `relay.chrome.session.degraded` — before: `Ne répond pas — les envois attendront`; after: `Ne répond pas — les envois attendent`; width 36; maxCells 36.
  b1 MIN-06 width fixed 37 → 36/36; `les envois attendent` accepted after corpus check (`relay.asks.status.delivering`, `wire.settings.send_journal_unconfirmed.label`); no queue/delivery claim.

### hi — 5 changed safety-scope key(s)
- `relay.chrome.drawer.status` — before: `{state, select, online {कनेक्टेड · यह फ़ोन} retrying {ऑफ़लाइन · फिर कोशिश} other {कनेक्ट हो रहा है…}}`; after: `{state, select, online {कनेक्टेड · यह फ़ोन} retrying {ऑफ़लाइन · फिर से कोशिश} other {कनेक्ट हो रहा है…}}`; width online=13, retrying=16, other=12; maxCells 32.
  Retry branch `फिर कोशिश` → `फिर से कोशिश`; still elliptical but corpus-backed and status-readable; select branches unchanged; 16/32. No new semantic finding.
- `relay.chrome.list.filter.needs_you` — before: `आपके लिए · {count, number}`; after: `आपका ध्यान चाहिए · {count, number}`; width 16; maxCells 16.
  b1 MIN-07 fixed: `आपके लिए` → `आपका ध्यान चाहिए`; 15/16 at one digit, 16/16 at two, 17/16 at three.
- `relay.chrome.list.mark_all` — before: `सभी पढ़ा हुआ चिह्नित करें`; after: `सभी को पढ़ा हुआ चिह्नित करें`; width 18; maxCells 20.
  b1 MIN-08 fixed: dative `को` added; 18/20.
- `relay.chrome.session.subtitle.waiting` — before: `आपका इंतज़ार`; after: `आपका इंतज़ार है`; width 10; maxCells 24.
  Predicate completed: `आपका इंतज़ार` → `आपका इंतज़ार है`; 10/24; now distinct from `needs_you`.
- `relay.chrome.session.ended_hint` — before: `संदेश भेजने या “जारी रखें” से यह आपके होम फ़ोल्डर में फिर खुलेगा`; after: `संदेश भेजने पर या “जारी रखें” दबाने पर यह आपके होम फ़ोल्डर में फिर खुलेगा`; width 51; maxCells None.
  Both coordinates now conditional (`…पर … दबाने पर`); `जारी रखें` and home folder retained; 51, no maxCells.

### ru — 3 changed safety-scope key(s)
- `relay.chrome.session.ended_hint` — before: `Отправка сообщения или «Продолжить» снова откроет её в вашей домашней папке`; after: `Отправка сообщения или «Возобновить» снова откроет её в вашей домашней папке`; width 76; maxCells None.
  Term aligned to `Возобновить`; matches the button and home-folder promise; 76, no maxCells.
- `relay.chrome.session.resume` — before: `Продолжить`; after: `Возобновить`; width 11; maxCells 12.
  `Продолжить` → `Возобновить`, matching `wire.slash.resume`; 11/12.
- `relay.chrome.session.resuming` — before: `Продолжаем…`; after: `Возобновляем…`; width 13; maxCells 12.
  `Продолжаем…` → `Возобновляем…`; semantics pass, but 13/12 — new +1 width risk for rendered QA.

### ur — 5 changed safety-scope key(s)
- `relay.chrome.drawer.status` — before: `{state, select, online {منسلک · یہی فون} retrying {آف لائن · دوبارہ کوشش} other {منسلک ہو رہا ہے…}}`; after: `{state, select, online {منسلک · یہی فون} retrying {آف لائن · دوبارہ کوشش ہو رہی ہے} other {منسلک ہو رہا ہے…}}`; width online=15, retrying=31, other=16; maxCells 32.
  Retry branch `دوبارہ کوشش` → `دوبارہ کوشش ہو رہی ہے`; select branches unchanged; 31/32.
- `relay.chrome.link.since_seconds` — before: `{count, plural, one {دوبارہ منسلک ہو رہا ہے · # سیکنڈ پہلے} other {دوبارہ منسلک ہو رہا ہے · # سیکنڈ پہلے}}`; after: `{count, plural, one {دوبارہ منسلک ہو رہا ہے · سنک # سیکنڈ پہلے} other {دوبارہ منسلک ہو رہا ہے · سنک # سیکنڈ پہلے}}`; width one=41, other=41; maxCells 44.
  b1 MAJ-01 fixed: `سنک # سیکنڈ پہلے`; one/other + `#`; 41 cells at one-digit, 42/44 at two-digit.
- `relay.chrome.link.since_minutes` — before: `{count, plural, one {دوبارہ منسلک ہو رہا ہے · # منٹ پہلے} other {دوبارہ منسلک ہو رہا ہے · # منٹ پہلے}}`; after: `{count, plural, one {دوبارہ منسلک ہو رہا ہے · سنک # منٹ پہلے} other {دوبارہ منسلک ہو رہا ہے · سنک # منٹ پہلے}}`; width one=39, other=39; maxCells 44.
  b1 MAJ-02 fixed: `سنک # منٹ پہلے`; one/other + `#`; 39 cells at one-digit, 40/44 at two-digit.
- `relay.chrome.session.resuming` — before: `کھل رہا ہے…`; after: `جاری ہو رہا…`; width 12; maxCells 12.
  `کھل رہا ہے…` → `جاری ہو رہا…`; matches `جاری رکھیں`; 12/12.
- `relay.chrome.session.degraded` — before: `جواب نہیں دے رہا — پیغام روکے ہیں`; after: `جواب نہیں دے رہا — پیغامات رکے ہیں`; width 34; maxCells 36.
  b1 MIN-09 fixed: plural intransitive `پیغامات رکے ہیں`; 34/36; no queue/delivery claim.

### vi — 3 changed safety-scope key(s)
- `relay.chrome.session.ended` — before: `Phiên đã kết thúc · giữ lịch sử`; after: `Phiên kết thúc · lịch sử đã lưu`; width 31; maxCells 32.
  b1 MAJ-03 fixed: `giữ lịch sử` → `lịch sử đã lưu`; 31/32; history preserved as state.
- `relay.chrome.session.ended_hint` — before: `Gửi tin nhắn hoặc Mở lại sẽ mở phiên trong thư mục home của bạn`; after: `Gửi tin nhắn hoặc Mở lại sẽ khôi phục phiên trong thư mục home của bạn`; width 70; maxCells None.
  b1 NIT-02 fixed: result now `khôi phục phiên`; `Mở lại` button and home folder preserved; 70, no maxCells.
- `relay.chrome.session.degraded` — before: `Không phản hồi — tin gửi sẽ chờ`; after: `Không phản hồi — tin nhắn sẽ chờ`; width 32; maxCells 36.
  `tin gửi` → `tin nhắn`; 32/36; messages wait, no delivery claim.

### zh-CN — 3 changed safety-scope key(s)
- `relay.chrome.link.reconnecting` — before: `正在重连……`; after: `重连中……`; width 8; maxCells 32.
  `正在重连……` → `重连中……`; 8/32; state word now stable with `since_*`.
- `relay.chrome.link.since_seconds` — before: `{count, plural, other {重连中 · 上次同步于 # 秒前}}`; after: `{count, plural, other {重连中 · 上次同步：# 秒前}}`; width other=25; maxCells 44.
  `于` → full-width colon; `上次同步：# 秒前`; 25/44; `#` and sync meaning retained.
- `relay.chrome.link.since_minutes` — before: `{count, plural, other {重连中 · 上次同步于 # 分钟前}}`; after: `{count, plural, other {重连中 · 上次同步：# 分钟前}}`; width other=27; maxCells 44.
  `于` → full-width colon; `上次同步：# 分钟前`; 27/44; `#` and sync meaning retained.

## 6. Residual width risks for rendered QA (non-gating)

These are **not** claimed covered by this source audit. Pure maxCells excess is not treated as major without measured rendered clipping; no rendered frame was taken.

| Key | Cells / max | Note |
|---|---|---|
| `fr` `list.mark_all` | 21/20 | b1 MIN-05 rejected; correct French kept; no clipping measured |
| `vi` `list.mark_all` | 22/20 | b1 MIN-10 rejected; alternatives lose idiom/scope; no clipping measured |
| `ru` `session.resuming` | 13/12 | new +1 (was 11); semantic pass; rendered check |
| `hi` `list.filter.needs_you` | 15/16, 16/16, 17/16 at 1/2/3 digits | semantic fix; 3-digit count over cap |
| `ur` `list.filter.needs_you` | 16/16, 17/16, 18/16 at 1/2/3 digits | unchanged; chip row scrolls per sidecar |
| `es` `list.filter.needs_you` | 15/16, 16/16, 17/16 at 1/2/3 digits | unchanged; 3-digit count over cap |
| `ur` `list.mark_all` | 20/20 | b1 NIT-01 rejected; at cap; grammatical form 23/20 |

At cap (confirm no wrap/clip in the rendered pass): `es` `session.degraded` 36/36; `fr` `session.degraded` 36/36; `ur` `session.resuming` 12/12; `vi` `session.resuming` 12/12.

No-maxCells hint growth (not gated; banner wrapping is allowed): `es` `session.ended_hint` 61; `fr` `session.ended_hint` 74; `hi` `session.ended_hint` 51; `ru` `session.ended_hint` 76; `vi` `session.ended_hint` 70; `ur` `session.ended_hint` 85; `zh-CN` `session.ended_hint` 40.

Unreachable: `es` `link.since_*` remain 46/45 cells only in the `many` branch (exact 10^6 counts); the seconds line is <60 s and the minutes counter would need ~2 years. Unreachable; not a risk.

## 7. Record and process notes (not verdict inputs)

- **ru metadata discrepancy:** the ru `remediation.json` `safety-ru` entry says the remediation “does not touch the safety labels”, but the actual byte diff changes `session.ended_hint`, `session.resume` and `session.resuming` — all in the original 17. I reviewed them; the changes are the `Возобновить` term alignment and pass semantically. This is a record-metadata inconsistency, not a copy defect.
- **zh out-of-scope gate note:** the zh record reports `generate_check` exit 1 because `i18n/ledger.json` has 0 `relay.chrome` rows; regenerating would shrink `shipped.json` to English. That is a wave-prerequisite/ledger issue, outside this safety-copy delta; not assessed here.
- b1’s `MIN-06` lexical objection (“les envois” nominalization) is **withdrawn**: the fr corpus already uses `envois` for message sends in `relay.asks.status.delivering` (“envoi en cours”) and `wire.settings.send_journal_unconfirmed.label` (“Journaliser les envois non acquittés”); the width component is fixed at 36/36.

## 8. Verdicts

| Locale | Keys read | b1 findings | Changed scope keys | New blocker/major | Verdict |
|---|---:|---|---:|---:|---|
| `es` | 17 | 2 fixed | 2 | 0 | **PASS** |
| `fr` | 17 | 4 fixed, 1 rejected | 4 | 0 | **PASS** |
| `hi` | 17 | 2 fixed | 5 | 0 | **PASS** |
| `ru` | 17 | 0 b1 findings | 3 | 0 | **PASS** |
| `ur` | 17 | 3 fixed, 1 rejected | 5 | 0 | **PASS** |
| `vi` | 17 | 2 fixed, 1 rejected | 3 | 0 | **PASS** |
| `zh-CN` | 17 | 0 b1 findings | 3 | 0 | **PASS** |

**Overall:** **PASS — terminal PASS.** 0 blocker, 0 major, 0 new semantic minor/nit. 13/16 b1 findings fixed; 3 rejected/withdrawn with rationale; 25 changed safety keys reviewed; 94 unchanged values byte-identical. Residual width risks are listed above for the later rendered QA pass and are **not** claimed as covered.

## 9. Limits

- Source-semantic delta only; no rendered frame, browser, or runtime. `cell_len` against `maxCells` is a text-budget check, not a clipping proof.
- The seven target catalogues are untracked in the worktree; these findings cover the current bytes pinned above, not a committed revision.
- No code, catalogue, ledger, or commit change was made. `report.md` and `record.json` are the only outputs.
