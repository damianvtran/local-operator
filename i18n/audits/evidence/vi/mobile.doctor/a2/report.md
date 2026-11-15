# vi / mobile.doctor — independent translation audit a2 (remediation delta)

- Verdict: **passed** — blocker 0 · major 0 · minor 0 · nit 0 new (unresolved blocker/major: 0)
- Scope: round 2, remediation delta only — 2 changed key(s): `mobile.doctor.files.ok`, `mobile.doctor.files.near`; the other keys were confirmed byte-identical to the a1-audited values and not re-audited
- Revision: a3722be7dadacaf140a2587a0100bd3b26c2ab50 + uncommitted working-tree remediation (branch feat/mobile-doctor, worktree .worktrees/mobile-doctor, node lop-remote-node-2)
- Source sha256 (en): `0ca00b276fdea066228394e3fee336042b1e149c9e7df3ea31be11a24f0beb72`; context `c6baa99bc539ef2d2e03bf7b6b11e83edac37139eede8db82034f5fad98d4614`
- Target sha256 (vi): `a59dd28edf10bc1e03fcad07514cc18ac581d15c1a76d6b1fa1656ef0f87ca83` (a1 audited `2642bc0c30738625eb8b2c99b4410f8d1a9ef1bbef7f20d6ea61bdd4c4042314`)
- Prior audit: a1 `i18n/audits/evidence/vi/mobile.doctor/a1/record.json` — changes_requested (auditor session 2c1a73eaa6cd)
- Translator: i18n-translator subagent, session 8dedc962f71a (remediation of a1 findings); original values by session 398cdb26808f (English source drafted by coder session b0300e2c2c7f), anthropic/claude-opus-5-5, 2026-11-14
- Auditor: session 24933ff37418, anthropic/claude-opus-5-5, 2026-11-14 — independent of the translators and the round-1 auditor; edited no catalogue value. Same model as the translator (disclosed).

## Prior findings

| id | severity | round-2 status |
|---|---|---|
| VI-D1 | major | resolved — files.ok, files.near changed exactly as the a1 suggested fix |
| VI-D2 | nit | carried (accepted in a1; keys unchanged) |

## New findings

None.

## Changed keys

| key | English | a1 value | a2 value | judgement |
|---|---|---|---|---|
| mobile.doctor.files.ok | The phone service has {open} of {limit} files open. | Dịch vụ truy cập điện thoại đang mở {open} tệp trên giới hạn {limit}. | Dịch vụ truy cập điện thoại đang mở {open} tệp (giới hạn {limit}). | ok — Resolves VI-D1. «đang mở {open} tệp (giới hạn {limit})» = “has {open} files open (limit {limit})”; the parenthetical carries “of {limit}” with no above/over reading. Same shape as zh-CN «已打开{open}个文件（上限{limit}个）» and consistent with files.status_line «đang mở {open}/{limit}». ASCII parentheses are correct for Vietnamese (Latin script, style guide §Punctuation). |
| mobile.doctor.files.near | The phone service has {open} of {limit} files open, close to its limit, so new phone connections and database opens can fail. | Dịch vụ truy cập điện thoại đang mở {open} tệp trên giới hạn {limit}, gần chạm giới hạn, nên các kết nối mới từ điện thoại và việc mở cơ sở dữ liệu có thể thất bại. | Dịch vụ truy cập điện thoại đang mở {open} tệp (giới hạn {limit}), gần chạm giới hạn, nên các kết nối mới từ điện thoại và việc mở cơ sở dữ liệu có thể thất bại. | ok — Resolves VI-D1. Same parenthetical; «(giới hạn {limit}), gần chạm giới hạn» now agrees (“limit 256, close to the limit”) instead of contradicting. The repeated «giới hạn» mirrors English “of {limit} … close to its limit” and reads naturally. Rest of the sentence unchanged from a1. |

## Verified

- placeholder sets identical to English for every changed key; NFC; key set and order identical (audit.log)
- only the 2 remediation key(s) differ from the a1-audited file; every other key byte-identical
- each changed value matches the a1 suggested fix in meaning; no command named; calm factual tone kept
- parenthetical «(giới hạn {limit})» consistent with files.status_line «{open}/{limit}» and zh-CN «（上限{limit}个）»

## Ledger

- `i18n/audit-history/vi/mobile.doctor.json`: history sha `9994c7b1e1af535a9ac3dc47f06f5706266d4b1bb32e6fb393f80d8f6fce75d4` → `91b5a9626fbdefc7304d4755f0a97e749ae24fce287ff537e997a05a5a424e94` (append --dry-run then --apply, exit 0; dryrun.log)

## Method and limits

- Independent LLM review on anthropic/claude-opus-5-5, not native-speaker certification; translator and auditor ran the same model (disclosed).
- Text renders via the product renderer only (audit.log); no terminal, desktop or device frame.
- Round-2 scope: only the remediation delta was judged; unchanged keys were confirmed byte-identical to the a1-audited values, not re-audited.
- generate.py / check.py not run by this lane; nothing committed or pushed.
