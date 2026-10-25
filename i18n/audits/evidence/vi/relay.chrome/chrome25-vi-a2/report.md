# Translation audit — round 2 delta — `chrome25-vi-a2` (Vietnamese, `relay.chrome`)

Delta verification only: same locale, same namespace, same ownership as `chrome25-vi-a1`.
No full re-audit; every key whose bytes did not change is retained from R1 by digest.

| | |
|---|---|
| Audit ID | `chrome25-vi-a2` |
| Locale / namespace | `vi` / `relay.chrome` — 83 keys |
| Round | 2 (delta after one unified remediation pass) |
| Worktree | `~/local-operator/.worktrees/relay-chrome-i18n`, base `535cea3598` (HEAD unchanged) |
| Target | `local_operator/i18n/catalogues/vi/relay.chrome.json` (untracked; nothing committed) |
| Auditor | `translation-auditor` (team `lopdev`), session `757de37d01bd` |
| Model | `radient/auto` — the id my runtime header declares here. No underlying provider is inferred or claimed |
| Date | 2026-10-25T18:15Z |
| Remediation record read | `…/sessions/70a22e9a7716/scratchpad/chrome25-vi-remediation/` (`remediation.md`, `remediation.json`, `relay.chrome.pre-remediation.json`) — disk record only; no agent summary used |
| Independence | different session from the translator (`70a22e9a7716`) and from the safety auditor (`acd16c49df72`); claims were re-derived from the bytes and the corpus, not taken on trust |

## Hashes (all recomputed in this session)

| Artefact | SHA256 | Note |
|---|---|---|
| `en/relay.chrome.json` | `f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac` | unchanged from R1 |
| `en/relay.chrome.context.json` | `1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a` | unchanged from R1 |
| vi target, **R1-audited** snapshot | `97a991ac3245c8b4edd59877eb1d1259affb1c8c77bedd460a0721c08a1d27b8` | equals my R1 target hash **and** the hash of the translator's `relay.chrome.pre-remediation.json` — the before-state is bound end to end |
| vi target, **current** | `a875b7f4177f4e3d9118a5d779734db751e350c384e01ae6059916bec5c8c1d9` | recomputed here; not copied from any prose |

## Delta scope — exact changed key list (5 of 83)

| # | Key | R1 audited | Current | EN | Cells |
|---|---|---|---|---|---|
| 1 | `relay.chrome.model.effort` | `Nỗ lực suy luận` | `Mức suy luận` | Reasoning effort | 12 / 24 |
| 2 | `relay.chrome.appearance.wide_view` | `Xem rộng` | `Chế độ rộng` | Wide view | 11 / 28 |
| 3 | `relay.chrome.session.degraded` | `Không phản hồi — tin gửi sẽ chờ` | `Không phản hồi — tin nhắn sẽ chờ` | Not answering — sends will wait | 32 / 36 |
| 4 | `relay.chrome.session.ended` | `Phiên đã kết thúc · giữ lịch sử` | `Phiên kết thúc · lịch sử đã lưu` | Session ended · history kept | 31 / 32 |
| 5 | `relay.chrome.session.ended_hint` | `Gửi tin nhắn hoặc Mở lại sẽ mở phiên trong thư mục home của bạn` | `Gửi tin nhắn hoặc Mở lại sẽ khôi phục phiên trong thư mục home của bạn` | A send or Resume reopens it in your home folder | 70 / — |

No other key differs; key set and order are still 83/83 identical to en.

## R1 carry-forward proof

- Unchanged keys: **78**. Digest over the sorted `key:value` map of exactly those keys,
  serialized identically from both files:
  `586824e7db6c05c3a5a757a7abc08d6eca4e9d28a76abbc88cf8248fde6af9ea` (R1 snapshot) =
  `586824e7db6c05c3a5a757a7abc08d6eca4e9d28a76abbc88cf8248fde6af9ea` (current) — **equal**.
- Changed subset digest differs as expected: `ad9f6adc…` (before) vs `664c4014…` (after).
- Consequence: every R1 statement about the 78 unchanged keys stands on unchanged bytes. I
  re-read semantically the 5 changed keys plus 14 direct neighbours (8 safety, 6 terminology);
  the remaining 64 are covered by the digest, not re-read.

## Finding disposition

| R1 item | Severity | Disposition | Verified in bytes |
|---|---|---|---|
| **F1** `model.effort` | minor | **fixed** | `Mức suy luận`, 12 ≤ 24. Term matches the settings label surface I cited in R1 (`wire.settings.model_effort.label` → "Mức suy luận mặc định") — unchanged file, re-read |
| **F2** `appearance.wide_view` | minor | **fixed** — auditor's second option used | `Chế độ rộng`, 11 ≤ 28. Claim checked: `bố cục` **0** hits in the vi corpus; `chế độ` 13 case-sensitive / 15 case-insensitive hits, used for modes (`relay.asks.gate.notice_auto`, `wire.errors.desktop.fast_mode_unsupported`). `Bố cục rộng` was my primary; the chosen form is corpus-grounded and names a mode, so I accept it rather than ask for cosmetic churn |
| **F3** `session.degraded` | minor | **fixed** | `Không phản hồi — tin nhắn sẽ chờ`, 32 ≤ 36. Claim checked: `tin gửi` occurred exactly **1** time corpus-wide and only in this file at R1 (verified on the snapshot); **0** now. `tin nhắn` is the corpus word for a message (25 case-sensitive hits incl. this one) |
| **F4** `list.mark_all` | minor (width) | **accepted as rejected — residual render check** | Retained `Đánh dấu tất cả đã đọc`, still 22 cells vs 20. The rejection is sound on the file's own evidence: the offered `Đánh dấu đã đọc` would be byte-identical to the per-row `row.mark_read` (`Đánh dấu đã đọc`, verified), `Đã đọc tất cả` is a state not an action, and `đánh dấu hết` has **0** corpus hits. A pure budget overrun without measured clipping is not a major; the residual (rendered ghost-button fit) stays open with the wave, not as a defect |
| **F5** `session.ended_hint` repetition | nit | **fixed** (by the same edit as NIT-02) | `Mở lại` occurs once in the hint; `sẽ khôi phục phiên` no longer repeats `mở` |
| **F6** `thư mục home` | nit | **accepted as rejected — deferred to the guide owner** | The replacement would have broken a real one-term invariant: `wire.settings.static.description` renders EN "The agent home" as "Thư mục home của tác nhân" (verified verbatim). Retaining it is right; recording `home` in `i18n/style-guides/vi.md` is a repo-artifact change outside the translator's writable files and is not a defect in these bytes |
| **n1** `Dismiss` → `Đóng` | note | **no change** — agreed | The close-control reading holds |
| **n2** capitalisation | note | **withdrawn by me** | The translator's explanation is source-driven and checks out: initial-case agreement with EN is `relay.chrome` 83/83, `relay.asks` 77/77, `relay.ask` 11/11, `tui.notices` 30/30 (`relay.session.tags` 18/19). `relay.asks` reads lowercase because its English is lowercase — my R1 "older files drift from the guide" inference was wrong, and no pass over those files is warranted |

## Safety and neighbour review (delta)

The two safety-stream edits (`MAJ-03` `session.ended`, `NIT-02` `ended_hint`) are inside my R1
finding surface, so I verified them independently; separate safety delta signoff belongs to
`chrome25-safety-b1`, not to this receipt.

- `Phiên kết thúc · lịch sử đã lưu` states the preserved state rather than an instruction
  (`giữ lịch sử`), keeps the middle dot, 31 ≤ 32, and `đã lưu` is corpus vocabulary for "saved".
  Neighbour `session.subtitle.ended` stays `Đã kết thúc` — different position (bare state vs
  "Session ended · …"), not drift.
- Approve pill/name invariant: `"Phê duyệt — có yêu cầu đang chờ bạn".startswith("Phê duyệt")`
  → **True**; deny remains `từ chối`.
- The hint names the button's own word (`Mở lại`, exactly once); `degraded` keeps `tin nhắn`
  and still promises only that messages **wait** (no queued/delivered claim added).
- Terminology neighbours re-read and coherent: `Giao diện` / `Chủ đề` / `Bật` / `Tắt` around
  `Chế độ rộng`; `Chi phí & ngữ cảnh` unchanged; no new term introduced by the delta other than
  the two sanctioned ones (`chế độ`, `khôi phục`, both corpus-attested).

## Checks executed (actual output)

```
$ ~/local-operator/.venv/bin/python scripts/i18n/check.py
check.py: ok — 182 file(s) scanned, 2253 literal(s) at or below the baseline,
catalogues consistent; 34 advisory finding(s) (report-only until activated)      [exit 0]
```
(the 34 are the AST ratchet in unrelated modules; no catalogue finding)

```
ICU argument (name AND kind) parity over all 83 ......... clean — 0 mismatches
plural categories: 4 plural keys, vi = (('other',),), every branch keeps '#'
changed keys: 5 — all NFC-clean, all within maxCells (or no maxCells)
carry-forward digest (78 unchanged keys) ................ equal (586824e7…)
structure: 5377 bytes, order == en, trailing newline, no BOM,
           0 combining marks, 0 values identical to en, 0 ascii '...' , 0 ws anomalies
```

Runtime-rendered delta (`runtime.render_message`, actual output):
`Reasoning effort → Mức suy luận` · `Wide view → Chế độ rộng` ·
`Not answering — sends will wait → Không phản hồi — tin nhắn sẽ chờ` ·
`Session ended · history kept → Phiên kết thúc · lịch sử đã lưu` ·
`A send or Resume reopens it… → Gửi tin nhắn hoặc Mở lại sẽ khôi phục phiên trong thư mục home của bạn`.

Two count nits in the translator's prose, immaterial to the disposition: it says `chế độ` 14
hits and `tin nhắn` 26; my measurements are 13/25 case-sensitive (15/27 case-insensitive, which
compares the same way only if their method was case-insensitive). The direction of both claims
is what the findings rested on, and it holds.

## Not claimed here

- **Rendered app evidence** — no screenshot or painted frame; `F4` therefore stays a residual
  width signal pending rendered confirmation, not a pass or a fail.
- **Separate safety signoff** — the safety delta receipt is the other auditor's; this report
  only states what I re-verified.
- Other locales, the ledger and the guide files: untouched by me and outside this delta.

## Verdict

**TERMINAL PASS** — 0 blocker, 0 major in the delta; 3 minors and 1 nit from R1 fixed,
1 width signal correctly retained pending rendered fit, 1 nit correctly rejected for
consistency and deferred to the guide owner, 1 of my own notes withdrawn on the translator's
evidence. No new out-of-scope issue found in the changed bytes or their neighbours.

## Evidence files (this scratchpad, `chrome25-vi-a2/`)

`report.md`, `record.json`, `verify.py`, `verify.out`, `render_safety.py`, `render_safety.out`,
`chrome25-vi-a2-check.log`.
