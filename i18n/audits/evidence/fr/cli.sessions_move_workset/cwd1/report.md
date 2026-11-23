### Translation audit — #4403 cwd-unknown copy (slot 1)

Reviewer: translation-auditor (lopdev team), session `e507d894afb3`, **slot 1**, actual model anthropic/claude-opus-5-5, no fallback.
Translator: coder session `9a6e88d78421`, anthropic/claude-opus-5-5 (verified from its transcript: only provider `anthropic`, model `claude-opus-5-5`). I am independent of the translator by agent only, since we run on the same model.
Scope: `3030a5e5c` (copy freeze). I read the en catalogue and context from that blob. Keys: `cwd_unknown`, `cwd_unknown_advice`, `yes_cwd_unknown`, `needs_confirmation_cwd_unknown`, `receipt_cwd_unknown`, `receipt_keep_cwd_unknown`.
Inputs: `…/9a6e88d78421/scratchpad/tr4403/` (`index.json` sha256 `51b82e36d16fcd24bf4147b57b458c0f34db006d0385753a9fa82b5e06b38201`). All 8 file sha256 values match the index.

**Source checks (read-only)**
- `generate._en_sha('cli.sessions_move_workset')` = `d782f1ded36d43133e7a1c32db9bd89540030eecdba86be92e068b8a2850a480`, matching the blob. Context sha256 `cc4712832270e6f8080a85343175b555f7db4378e58f9b4f844aacbc8ca26635`.
- The en catalogue minus the 6 new keys hashes to `0f5602723eb9…`.
- In every locale, the catalogue at `3030a5e5c` equals the target of its last `passed` receipt. The 111 existing keys are therefore carried and only the 6 new values were audited.
- I ran with `PYTHONDONTWRITEBYTECODE`, so I created no `__pycache__`. The `__pycache__` directories already present in the worktree predate my run (created 16:05Z, cpython-39/312), so I left them untouched.

**Mechanics, all 42 values**
- Placeholders: the multiset matches en.
- Flags verbatim: `--yes`, `--dry-run`, `--confirm-working-set {digest}`, `--without-working-set`.
- No format or control characters.
- fr uses U+202F before every `:` and `;`.
- zh-CN has no CJK↔Latin spaces.
- hi uses the danda (।) and ur uses ۔ (no Latin full stop).
- Rendering: the repo's `runtime.render_message` with synthetic params rendered all 48 values (en plus 7 locales) with 0 unrendered placeholders (`rendered.json`).

**Meaning and safety, against each context "must"** — every one holds in all 7 locales:
- `cwd_unknown` and `receipt_cwd_unknown` say the listed folders are **not moved and stay on `{source}`**.
- Confirming in `needs_confirmation_cwd_unknown` and `yes_cwd_unknown` means **leaving them on `{source}`**: the purpose clause is bound to `--confirm-working-set`, never to sending.
- `yes_cwd_unknown` says nothing was moved, that `--yes` does not decide for the user, and never blames the destination.
- `cwd_unknown_advice` mirrors the certified `advice_stays` as a decision, not a recommendation.
- The receipts read as a record, not an error.

**Terminology and register**
- "Working folder" (the cwd) uses each locale's established working-directory term, the same as `wire.slash.move` and `relay.session.directory.*`:
  - es directorio de trabajo (the es guide records this rule)
  - fr répertoire de travail (9 uses vs 2)
  - ru рабочий каталог
  - hi कार्य निर्देशिका
  - ur ورکنگ ڈائریکٹری (11 uses)
  - vi thư mục làm việc
  - zh-CN 工作目录
- "The folders it mentions" uses the plain folder word and mirrors the certified `mentioned` row.
- The fixed phrases reuse this namespace's certified wording: "--yes did not confirm", "Nothing was moved", "Working files were not transferred", the frame of `receipt_keep`, and push (subir / pousser / push / 推送).
- zh-CN uses 移动, consistent with the C-ZH-1 direction.
- Address forms: tú, vous, вы, आप, آپ, bạn, 你.
- No width budget applies (no `maxCells`; wrapped CLI lines).

**Findings (non-gating, deferred)**

| ID | Sev | Locale | Keys | Current | Proposed | Reason |
|---|---|---|---|---|---|---|
| L1-VI-1 | nit | vi | `cwd_unknown, receipt_cwd_unknown` | …của cuộc trò chuyện này không được ghi lại, nên các thư mục mà nó nhắc đến… | …nên các thư mục mà cuộc trò chuyện nhắc đến… | The sentence subject is “Thư mục làm việc”, so “nó” first reads as the working folder; naming the conversation (as receipt_keep_cwd_unknown already does) removes the detour. Meaning is still recoverable. |
| L1-ZH-1 | nit | zh-CN | `cwd_unknown, receipt_cwd_unknown` | 此对话的工作目录未记录，因此它提及的文件夹… | 此对话的工作目录未记录，因此此对话提及的文件夹… | Same antecedent drift: 它 follows 工作目录 as topic; receipt_keep_cwd_unknown already writes 此对话提及的. |

**Verdict (`cli.sessions_move_workset`, the 6 new keys):** es, fr, hi, ru, ur and vi PASS; zh-CN PASS. Overall **PASS** — 0 BLOCKER, 0 MAJOR. 42 values read.

**Limitations:** no native-speaker review and no rendered terminal frame, Urdu bidi in particular. Slot 2 (`6fd346121602`) is independent and records its own findings separately.
