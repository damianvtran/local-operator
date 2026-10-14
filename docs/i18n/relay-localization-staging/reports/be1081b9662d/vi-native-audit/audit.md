# PR #2329 — independent Vietnamese final65 native audit

**Native-language verdict: PASS / terminal for this exact 65-key candidate.** No BLOCKER, MAJOR, MINOR or NIT findings. **Rendered certification: BLOCKED by the customer resource hold.** This is not a shipped/activation claim and does not waive the programme's independent visual or second-audit gates.

- Auditor: lopdev task, session `be1081b9662d`; actual model `openai/gpt-6.1-sol`; fallback **none**.
- Translator (from `decisions.json`): `relay-vi-final65-translation`, actual `openai/gpt-6.1-sol`. Different agent/session; this auditor did not author/implement the candidate. Same model disclosed. This is an LLM native-language audit, not a claim of human native-speaker identity.
- Source/context checkout: `~/local-operator/.worktrees/coder-2329-convergence-20261014`, observed HEAD `de9b9a490c839d6de2c2e41ea03ec819f66be489`.
- Candidate: `historical-scratch/590f5d5140c6/vi-final65/vi/`.
- Independently read all English source, candidate strings, English context, `i18n/glossary.md`, `i18n/style-guides/vi.md`, and translator decisions. Source bytes independently matched the checkout. Original validation was input, not substituted for execution.

## Exact coverage and fingerprints

| Namespace | Keys | Native result | Source SHA-256 | Target SHA-256 |
|---|---:|---|---|---|
| relay.shell | 44/44 | PASS | `8b17439e9f31f159bc60a9e2940eb2d88400dacdcda9854ae1b3c8eb0ca00586` | `6f196e2535134079dfec349f645b27a75a132c95bbd40bd98a3f3cd68f18d22f` |
| relay.session | 17/17 | PASS | `a4ae148de969d4a2e8541aca5b4b99ad311d9ef112a0d056fc21fcf90e7e0129` | `7743c7b5d018b84395478e99231714a46cd998e8c78ce465333be8df62fd517a` |
| relay.login | 4/4 | PASS | `9a59686bcd0e3641a5a807ea2b778cb385e591060c26c7fb994bea95e4de5bc5` | `a87e8ed3381143ee8d61dada52b0df604c72191cdde1747b9ea8fd6c46537020` |

`audit.json` enumerates all 65 exact source/target pairs and individual PASS dispositions, not a partial sample.

## Native decisions

- **Credits are tín dụng, not tokens.** In these paid/included allowance and spend messages, tín dụng denotes credit units. `token` separately names model token counts. Vietnamese invariant units work for zero/one/many and decimals without English plural suffixes. `yêu cầu` translates requests; USD remains its currency identifier without a conversion claim.
- **cache/token are deliberate borrowing, not untranslated stubs.** Both are explicitly in the binding vi kept-English technical list. The candidate embeds cache in native syntax (`từ cache`, `trong cache`), and chooses `token`, not raw API `tokens`. TUI is also an allowed acronym; the login brand and provider family identifiers remain identifiers. No accidental English sentences or API-unit stubs were found.
- **Projection waiting** says `đang chờ trạng thái hiện tại của phiên…`, meaning the session state has not arrived. This matches the sidecar's live state/frame meaning and stays distinct from `đang kết nối với phiên…`; it promises neither a prediction nor a reconnection.
- **Cached usage safety** explicitly says no provider update is requested, distinguishes absent versus last-known data, and qualifies staleness. Unknown amount/window/eligibility remain unknown, not zero. Included-quota exhaustion and continuation through credits remain distinct from paid/family allowance limits; family caps still apply.
- **Complete rows** retain Vietnamese category/window/value order, dot separator, colon, and family parentheses only in the family template. `dòng mô hình` distinguishes a family cap from a selected model. Compact punctuation is appropriate here.
- **Session copy** preserves concise source register/casing and action meanings. History search explicitly covers names and conversation content; directory fallback asks for another path; model filtering and loading remain separate.
- **Login** preserves the branded lowercase title and clear `mật khẩu`, `đăng nhập`, `Mật khẩu không đúng.` refusal. No stylistic rewrite is needed.

## Executed evidence (isolated; no installs or repo edits)

Python command: `env -i HOME=<this-audit>/home LOCAL_OPERATOR_CONFIG_DIR=<this-audit>operator-home PATH="$PATH" TERM=xterm-256color source-checkout/.venv/bin/python <this-audit>/probe.py`.

Node command: `env -i HOME=<this-audit>/home LOCAL_OPERATOR_CONFIG_DIR=<this-audit>operator-home PATH="$PATH" TERM=xterm-256color TZ=UTC node --experimental-strip-types <this-audit>/probe.mjs`.

Both final commands exited **0**. The worktree had no `.venv`; the existing primary interpreter was used with the candidate checkout explicitly first on `sys.path`. Node executed the **exact relay runtime.ts** with its existing intl-messageformat dependency; no compiled/bundled shipped claim. `env -i` strips inherited CMUX and live configuration variables. All mutation/outputs stayed in this auditor's scratch or in-memory fixture returns; original candidate files were not edited.

| Surface | Actual output/control | Result |
|---|---|---|
| Key sets/NFC/typed ICU | 44 + 17 + 4 exact keys; full AST type/style/encounter-order comparison matches English; NFC | PASS |
| Repository catalogue checker | `[]` before and after fixture controls | PASS |
| Decimal credits | `Còn lại 12.345,67 tín dụng`; `Đã dùng 12.345,67 tín dụng` | PASS |
| Other units | `Còn lại 12.345,67 USD`, `… token`, `… yêu cầu`; zero/one/two also executed for all four units | PASS |
| Fraction percent .29/.07 | `Đã dùng 29%` / `Đã dùng 7%`; remaining/used-percent also produce 29% / 7% in both Python and exact JS runtime | PASS |
| Date/time | `Đặt lại vào 14 thg 10, 2026, 13:07`; `Dữ liệu tính đến 14 thg 10, 2026, 13:07.` | PASS |
| Complete family row | `Hạn mức theo dòng mô hình (opus) · 5 giờ: Còn lại 12.345,67 tín dụng` | PASS |
| All 65 JS messages | All returned localized output, no key fallback; uiLocaleFor('vi') = vi | PASS |
| CLDR | Vietnamese `other` only; no candidate plural/select branches; injected `one` branch rejected | PASS |
| Dropped numeric placeholder | Actual checker rejects arguments `unit` versus expected `value:number, unit:simple` | PASS negative control |
| Percent skeleton removed | Repository checker returns `[]` because it compares kinds only; independent full-style AST comparison **rejects** the mutation | PASS independent negative control; checker limitation disclosed |
| JS missing numeric input | Actual output `relay.shell.usage.remaining`, not a valid localized result | PASS negative control |
| Login HTML | Exact function/template AST emits `lang="vi" dir="ltr"`, localized refusal and `role="alert"` | PASS text/HTML path only |
| HTML injection | `<script>alert("x")</script>&__TITLE__` becomes `&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;&amp;__TITLE__`; raw markup absent and token-looking copy not substituted again; restored normal output identical | PASS negative and restored controls |

The initially surprising number-style checker acceptance was investigated at `scripts/i18n/check.py:513-519`: `message_arguments` compares argument kind, not skeleton. This is not a candidate defect; our independently executed AST check includes skeletons and rejects the altered percent fixture. No claim that the repository guard alone enforces full style parity.

## Remaining gates / applicability

- **Rendered: BLOCKED**, per delegated customer resource hold. No browser, native app, screenshots, layout/width/truncation or live interaction runs. Login AST output and Node strings are not rendered UI proof.
- **RTL: not applicable** to Vietnamese's LTR native audit. No Urdu/mirroring or mixed-direction visual certification claimed.
- **Pseudo: rendered fit check blocked** by the same resource hold. The unmodified candidate was checked; pseudo text is not native-language approval and no pseudo screenshots were fabricated.
- **No shipped claim**, no ledger/repository changes, no self-audit, no merge/release action.

Evidence: `audit.json`, `mechanical.json`, `js-mechanical.json`, `probe.py`, `probe.mjs`, beside this report. Terminal result is **native PASS for all three exact fingerprints, rendered BLOCKED**.
