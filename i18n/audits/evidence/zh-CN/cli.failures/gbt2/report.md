# zh-CN / cli.failures — independent audit gbt2 (2026-11-16)

Worktree `.worktrees/github-borrow-timeout-note` @ 80a55b208 (base f0689a27). Keys audited: 5/5 (new: `cli.failures.github_borrow_timeout`; second independent pass over the whole namespace).
Auditor: Local Operator session 1c05d7f1365f (audit2-zh). Model: started openai/gpt-6.1-sol; host switched to fallback anthropic/claude-opus-5-5 (quota HTTP 429) after the first file-listing command, before any catalogue was read — all audit judgement was made on anthropic/claude-opus-5-5. No MT/translation service used.

## Mechanical
- `scripts/i18n/check.py --catalogues-only`: rc 0, "catalogue checks: 0 problem(s)".
- Own check: key set and order identical to en (5); placeholder multisets identical per key; NFC; no bidi controls; no ICU plurals in this namespace; no backticks/flags. New key width 42 cells excluding placeholders (no maxCells).

## Per-key reading
- `github_borrow_timeout` — 来自{owner}的GitHub凭据未在{seconds}秒内送达；请重试该命令. Meaning exact: not-arrived-within-deadline, user re-runs the command. No denial/revoke/login/auto-retry/lease-extension implication. GitHub kept. 凭据 matches house usage (37 occurrences, 0 凭证; glossary zh cell still `—` per note 6). 送达 matches house delivery wording (browser.site_approval.*). 请 sits at head of the directive clause after ； (style guide). No final 。 is consistent with house precedent for unpunctuated sources (5/5 `；请` values with period-less source carry none). No digit/Chinese spacing. No finding.
- `error_line` — {prog}: 错误: {message}. Shape preserved as required by context. See N1.
- `required` — 缺少以下必需参数：{names}. Correct, idiomatic. No finding.
- `invalid_choice` — 参数{argument}：无效选项：{value}（可选值：{choices}）. Mirrors argparse structure; full-width punctuation correct. No finding.
- `invalid_command` — 无效的{group}命令：{command}. Correct. No finding.

## Findings
- **N1 (NIT)** `cli.failures.error_line`: the colon after the Chinese word 错误 is half-width with a space (`错误: `), while the style guide wants full-width punctuation in Chinese prose and the inserted messages use `：`. Suggested: `{prog}: 错误：{message}` (keep the ASCII colon after the Latin {prog}). Pre-existing, previously passed; non-blocking.

Verdict: **passed** — 0 BLOCKER, 0 MAJOR, 0 MINOR, 1 NIT.

Observation (not a translation finding): in en/cli.failures.json the new line is indented with 1 space against the file's 2-space indent; cosmetic, source-side.
