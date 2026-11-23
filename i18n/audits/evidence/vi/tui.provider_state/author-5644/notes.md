# Vietnamese provider-state author receipt

Status: **translated**, not independently audited, not passed. Five labels authored by i18n-translator / provider-state-vi-author, session `5644d4838456`. Actual authoring model: `openai/gpt-6.1-sol`, following the host's explicit model-switch message from `anthropic/claude-opus-5-5` due to provider quota failure. No translation API, SDK, MT service, website, extra agent, installation, or model-pin change was used. English/context source author is the separate session identified in the brief as `93b7`.

## Native choices

- `logged_in` → **đã đăng nhập**: completed login status, not the instruction “đăng nhập”. The context specifies an existing stored credential; this is conventional compact UI wording rather than a claim about a new live authentication probe.
- `env_key` → **khóa từ biến môi trường**: API key supplied by an environment variable. “từ” makes the credential source explicit; “biến môi trường” avoids treating the label as an environment name or an actual variable identifier. Existing vi product copy uses “khóa” for key. This is human copy, never a translated machine token.
- `needs_login` → **cần đăng nhập**: requirement/status, not a button or command.
- `local_ready` → **máy chủ đã cấu hình**: completed server configuration. “đã” is essential to contrast with the following instruction; not “sẵn sàng”, “hoạt động” or “trực tuyến”, which could overclaim health/reachability.
- `local_unconfigured` → **cấu hình máy chủ**: direct configure-server instruction, distinct from completed configuration. “máy chủ” follows the vi guide's server terminology.

Labels keep the deliberately lower-case compact status/action voice requested by the English sidecar; no unnecessary pronoun, politeness filler, punctuation, or invented CLDR arms. The vi guide normally uses sentence case rather than title case; these contextual lower-case labels follow the explicitly stated label style. NFC precomposed diacritics verified. No protected tokens occur in the five source messages; token absence/parity is verified, not invented.

Glossary proposals (not glossary edits): login → “đăng nhập”; API key → “khóa”; environment variable → “biến môi trường”; configure → “cấu hình”. These are ordinary established vi developer/UI terms and consistent with existing copy. Glossary/style supplies server → “máy chủ”; credentials remains “thông tin xác thực” when that general noun is actually used (not substituted for the more specific key label here).

## Mechanical execution, not self-audit

`audit.log` is actual captured stdout/stderr from the checked-in `validator.py.txt` recipe. The official `scripts.i18n.check.check_catalogues()` runs against an isolated temporary en/vi, single-namespace fixture via the catalogue module's documented fixture-root hook. This scopes the official parity/context/ICU/plural checks without running the global base/ledger/ratchet checker. Missing-key and added-argument mutations fail as expected; the restored positive control returns `[]`. Worktree module paths and exact supplied source/context SHA-256 pins are asserted. Primary checkout's existing venv is used; `PYTHONPATH` selects this worktree. `env -i` strips inherited CMUX variables; HOME and config are session scratch. Git commands use normal HOME outside that isolated invocation.

Actual command (from worktree root; the shell expanded the variables recorded in the log):

```sh
env -i HOME="$LOCAL_OPERATOR_SCRATCHPAD/vi-author-home" LOCAL_OPERATOR_CONFIG_DIR="$LOCAL_OPERATOR_SCRATCHPAD/vi-author-home/config" PATH="$PATH" PYTHONPATH="$PWD" TERM=xterm-256color /Users/damian/local-operator/.venv/bin/python i18n/audits/evidence/vi/tui.provider_state/author-5644/validator.py.txt > i18n/audits/evidence/vi/tui.provider_state/author-5644/audit.log 2>&1
```

Exit: 0. Five ordered keys, zero arguments and no plural/select branches; ICU rendering, plain Rich text, NFC, no bidi controls, protected-token parity all mechanically pass. This paragraph is notes, **not** an execution log; the actual output is `audit.log`.

## Width diagnostics and deferred integration proof

Rich cell widths in source order: English **9 / 7 / 11 / 17 / 16**; Vietnamese **12 / 23 / 13 / 19 / 16**. Synthetic expanded English control **23 / 21 / 25 / 31 / 30**. The 17-cell openrouter example is diagnostic only, not a universal/fixed cap: the env-key and configured-server labels exceed that illustrative allowance. The sidecar declares no maxCells. Semantic clarity is retained rather than arbitrarily abbreviating “biến môi trường” or dropping completed-state “đã”. Root integration must inspect consumer-specific fitting; no all-width fitness or truncation-policy claim is made here.

The Rich transcript is plain text and is **not** a styled settings frame, real native rendering, font/geometry proof, app startup, backend state-code invariant test, or health probe. No app/server/native/browser/full-test/global-base-check run was performed. Current UI producer remains English until the root integrates the typed code row/resolver and pre-localization machine-code predicate. Existing offline original-provider visual remains uncertified by this author work. Independent approved-reader A and B audits remain required after the root commits all seven target catalogues. No eligibility, shipping, ledger, code, generator, source/context, or other-locale changes were made by this author.
