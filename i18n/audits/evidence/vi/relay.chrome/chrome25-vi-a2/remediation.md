# Translation remediation — `vi` (`relay.chrome`), round 1

| | |
|---|---|
| Locale | `vi` (Vietnamese, LTR) — one locale, one run |
| Namespace | `relay.chrome`, 83 keys |
| Worktree | `~/local-operator/.worktrees/relay-chrome-i18n`, base `535cea3598` |
| Target | `local_operator/i18n/catalogues/vi/relay.chrome.json` (untracked; **no commit**) |
| Translator | `i18n-translator`, session `70a22e9a7716` |
| Model | `radient/auto` — the id this session's header declares; the provider model behind it is not exposed in-session, and no fallback was selected or used |
| Reports read | first audit `chrome25-vi-a1` (session `757de37d01bd`), second safety audit `chrome25-safety-b1` (session `acd16c49df72`) |
| Delta audit inputs | pre-remediation snapshot kept at `relay.chrome.pre-remediation.json` in this directory |

## Hashes

| Artefact | SHA256 |
|---|---|
| `en/relay.chrome.json` (frozen, unchanged) | `f30049d1ec7bce1d0586d8d5d380368e64fb0ac841d91b9b04c1625d17688aac` |
| `en/relay.chrome.context.json` (frozen, unchanged) | `1c727d1c6c8ddff7c33cbd3abb1ed716e89ab84c864cdb9c319b5df2595fb69a` |
| `vi/relay.chrome.json` **old** (audited bytes) | `97a991ac3245c8b4edd59877eb1d1259affb1c8c77bedd460a0721c08a1d27b8` |
| `vi/relay.chrome.json` **new** (this delta) | `a875b7f4177f4e3d9118a5d779734db751e350c384e01ae6059916bec5c8c1d9` |

English and context hashes re-computed here after the edit: both unchanged. No other file
was written: `git status --porcelain` shows no tracked modification and exactly one entry in
`catalogues/vi/` (this new file). Ledger, allowance, generators, gates and the other six
locale catalogues untouched.

## Changed keys (5 of 83 — exact list)

| # | Key | Old | New | Cells | Findings |
|---|---|---|---|---|---|
| 1 | `relay.chrome.model.effort` | `Nỗ lực suy luận` | `Mức suy luận` | 12 / 24 | F1 |
| 2 | `relay.chrome.appearance.wide_view` | `Xem rộng` | `Chế độ rộng` | 11 / 28 | F2 |
| 3 | `relay.chrome.session.degraded` | `Không phản hồi — tin gửi sẽ chờ` | `Không phản hồi — tin nhắn sẽ chờ` | 32 / 36 | F3 |
| 4 | `relay.chrome.session.ended` | `Phiên đã kết thúc · giữ lịch sử` | `Phiên kết thúc · lịch sử đã lưu` | 31 / 32 | MAJ-03 |
| 5 | `relay.chrome.session.ended_hint` | `Gửi tin nhắn hoặc Mở lại sẽ mở phiên trong thư mục home của bạn` | `Gửi tin nhắn hoặc Mở lại sẽ khôi phục phiên trong thư mục home của bạn` | 70 / — (no budget) | NIT-02 (+F5 resolved) |

Everything else is byte-identical to the audited snapshot.

## Finding disposition

### Fixed

- **F1 (minor, `chrome25-vi-a1`) — `model.effort`.** Claim verified: the settings surface that
  carries the same control labels it `Mức suy luận` (`wire.settings.model_effort.label`
  “Default reasoning effort” → “Mức suy luận mặc định”; `.help` → “Mức suy luận cho các cuộc
  trò chuyện mới”; `wire.settings.model.description` → “…và mức suy luận…”). This key is the
  same labelled control row, so it takes the label term. `Mức suy luận`, 12 ≤ 24.
- **F2 (minor) — `appearance.wide_view`.** Accepted the underlying point: a toggle whose state
  renders as `Bật`/`Tắt` must name a mode, not an act of viewing. Chose the auditor's second
  option, `Chế độ rộng`: `bố cục` has **0** occurrences in the vi corpus while `chế độ` has 14
  (`relay.asks.gate.notice_auto`, `tui.notices.fast_mode_refused`, `wire.settings.*`), so it is
  the register the product already uses for modes. 11 ≤ 28.
- **F3 (minor) — `session.degraded`.** Claim verified: `tin gửi` occurs exactly once in the whole
  vi corpus — in this file — while `tin nhắn` is the product's word for a message (26 uses;
  `relay.chrome.composer.label` “Message” → “Tin nhắn”). The banner now reads
  `Không phản hồi — tin nhắn sẽ chờ`, 32 ≤ 36. **No new guarantee is introduced**: it still says
  the messages will wait — not queued, not delivered.
- **MAJ-03 (major, `chrome25-safety-b1`) — `session.ended`.** Accepted: `giữ lịch sử` reads as an
  instruction. The banner now states the preserved state, `Phiên kết thúc · lịch sử đã lưu`
  (“history is saved/kept”), 31 ≤ 32. Vocabulary evidence: `đã lưu` is the corpus form for
  “saved” (`tui.authoring.saved`, `tui.schedule.notice_edited`, `wire.memory.enabled.help`), and
  the corpus states kept-history as a state in `wire.settings.session_compression.description`
  (“Lịch sử được giữ đầy đủ…”). The fully passive `… lịch sử được giữ` is 33–36 cells and would
  break the 32-cell banner, so this is the budget-fitting state form, not an imperative.
- **NIT-02 (nit, `chrome25-safety-b1`) — `session.ended_hint`.** Accepted: the result clause said
  `mở phiên` (open) while the trigger is `Mở lại` (reopen). Now `… sẽ khôi phục phiên trong thư
  mục home của bạn` — “restore the session”, i.e. the source's *reopens*. This single edit also
  settles **F5** (the `Mở lại … mở` repetition): `khôi phục` carries the reopen sense without
  repeating `mở`. `khôi phục` is corpus vocabulary (`tui.projects_edit.refusal_estimate_clear`
  “khôi phục giá trị cũ”). 70 cells, no `maxCells` on this key (it wraps by design).

### Rejected — retained wording, flagged for rendered fit

- **F4 (minor, width) and MIN-10 (minor) — `list.mark_all`, 22 cells vs a 20-cell budget.**
  Retained `Đánh dấu tất cả đã đọc`. Neither report claims measured clipping; the brief's rule is
  that a pure budget overrun is not a major without it, and that accurate idiom outranks a
  misleading abbreviation. The offered alternatives fail on meaning or grammar:
  - `Đánh dấu hết đã đọc` (19) — telegraphese: `đánh dấu hết` reads “mark everything” and leaves
    `đã đọc` unattached; no such construction exists in the vi corpus (`đánh dấu hết`: 0 hits).
  - `Đánh dấu đã đọc` (15) — drops the “all” scope, so a whole-list action would read exactly
    like the per-row action `row.mark_read` (`Đánh dấu đã đọc`, also in this file).
  - `Đã đọc tất cả` (13) — a state, not an action; ambiguous as a button.
  The accurate Vietnamese form is already compressed: Gmail vi uses
  `Đánh dấu tất cả là đã đọc` (25). **Residual: render-check the header ghost button; if it
  clips, the only honest shorter form is `Đánh dấu đã đọc` and the scope loss must be accepted
  by the wave owner, not hidden here.**

- **F6 (nit) — `thư mục home`.** Rejected the replacement `thư mục cá nhân`, on corpus evidence:
  `wire.settings.static.description` renders the agent's home folder as “Thư mục home của tác
  nhân”, so changing it only here would break the one-term invariant the glossary requires. The
  second safety audit independently accepted `thư mục home` as the home-folder term in vi
  (its §8 lists it beside `carpeta personal`, `dossier personnel`, `домашняя папка`, `主目录`).
  **Deferred (not mine to write): the wave should record `home` in the vi guide's kept-English
  list — `i18n/style-guides/vi.md` is outside a translator's writable files and I did not edit it.**

- **n1 (consistency note) — `snackbar.dismiss` `Đóng`.** No change; the auditor reached the same
  conclusion (it is a close control, and `Đóng` matches this file's own close family
  `Đóng điều hướng` / `Đóng tìm kiếm`). The corpus split (`bỏ qua` for dismissing a question vs
  `Đã đóng` for a dismissed ask) is a different action, not drift to fix here.

- **n2 (capitalisation note) — no change inside `relay.chrome`.** The auditor's own note says
  nothing here needs fixing. The underlying explanation is source-driven, and I verified it:
  the vi waves mirror the **English source's** initial capitalisation (100% agreement in
  `relay.asks` 77/77 and `relay.ask` 11/11; 97–99% elsewhere, the exceptions being strings
  where en opens with a product name that vi moves mid-sentence). `relay.asks` reads lowercase
  because its en source is lowercase (`try again`, `approve`), while `relay.chrome`'s English is
  sentence case (the redesign's O1) — my file matches en on 83/83 values. Any harmonisation of
  the older relay files is a separate namespace change, out of this wave's scope.

## Validation of this delta

1. **Key set and order** — 83/83, `order == en` true, no added or removed key.
2. **ICU arguments (name and kind)** — 0 mismatches against en; re-checked for all 17 argument
   slots across 10 keys (`count`/`failed` plural, `done`/`total`/`running` number,
   `state` select, `name`/`label`/`elapsed` simple).
3. **Plural categories** — 4 plural keys, all `other`-only, every branch keeps `#`
   (`link.since_seconds`, `link.since_minutes`, `session.tray.questions`,
   `session.tray.agents_failed`); `vi` CLDR set is `["other"]`.
4. **Runtime rendering** (`runtime.render_message(…, locale='vi')`, actual output):
   `Phiên kết thúc · lịch sử đã lưu` /
   `Gửi tin nhắn hoặc Mở lại sẽ khôi phục phiên trong thư mục home của bạn` /
   `Không phản hồi — tin nhắn sẽ chờ` / `Mức suy luận` / `Chế độ rộng`.
5. **Typography/encoding** — U+2026, spaced ` — `, ` · ` kept; no ASCII `...`; no new `!`;
   NFC-clean, zero combining marks; UTF-8, trailing newline.
6. **Widths** — full rendered table in `widths.json`. Every budgeted key fits except
   `list.mark_all` (22/20, flagged above). Changed keys: 12/24, 11/28, 32/36, 31/32,
   and `ended_hint` 70 with no budget.
7. **Repo gate** — `cd ~/local-operator/.worktrees/relay-chrome-i18n && PYTHONPATH=. ~/local-operator/.venv/bin/python scripts/i18n/check.py` → exit 0:

   ```
   check.py: ok — 182 file(s) scanned, 2253 literal(s) at or below the baseline, catalogues consistent; 34 advisory finding(s) (report-only until activated)
   ```

   Zero `vi/` problem lines. The 34 `advisory:` lines are the AST ratchet in unrelated Python
   modules, not catalogue findings. The other six locales were consistent at the moment of this
   run; I did not touch them and any later mid-write failure there is not this delta.

## Actionable residuals (for the terminal auditor / wave owner)

1. **`relay.chrome.list.mark_all`, 22 cells vs 20** — rendered-fit check only; wording retained
   deliberately (see F4/MIN-10). If it clips, the scope-losing fallback is the wave owner's call.
2. **`home` in the vi style guide's kept-English list** — the corpus uses `thư mục home`
   (`wire.settings.static.description`); recording the call belongs to a guide change, not to
   this catalogue file.
3. Nothing else outstanding: no blocker, no major left open in `vi`; F1–F3, MAJ-03, NIT-02 fixed,
   F4/MIN-10 and F6 rejected with evidence and flagged, n1/n2 require no change.
