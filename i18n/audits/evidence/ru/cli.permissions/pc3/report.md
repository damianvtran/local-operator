# Audit ru / cli.permissions — pc3 (round 3)

Auditor: i18n-auditor (lopdev team), session 7b05bb0e2f09, model anthropic/claude-opus-5-5 (served model per session system prompt; no fallback observed). Independent of the ru translator (commit 85fb35811f9) and of the prior auditors (pa1 session 741233d88cd4, pb2 session a6f580403f3a). Worktree @ 85fb35811f9d17fb6dfadfc2d5df3269db04bd12. Date 2026-11-09. PR #3691.

Scope (round 2+ rule): verify the pb2 findings P1–P4 and audit the keys that changed in `788cc399..85fb3581` (en: `status.scope`, `setup.already_signed`, new `setup.always_allow_kept`; ru: the same three plus `setup.explain` and `setup.always_allow` from the pb2 fixes). Keys that did not change, and that pb2 approved, were not re-audited.

Verdict: **passed** — 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT.

## Mechanical (audit.log)
- Input hashes match the brief: en 58e58c3b19a07c02…, context 6f8427bf5dd20f8f…; ru 217cb8cd505af805….
- `check.py --catalogues-only`: 0 problems, exit 0.
- `parity.py.txt` (same script as pb2) found 39/39 keys in en order. Placeholders, backtick spans, the `[y/N] ` trailing space, newline count, bullets and verbatim tokens all match, with 0 problem keys.

## pb2 findings: all fixed
| ID | Sev | Status | Evidence |
|---|---|---|---|
| P1 | MAJOR | fixed | setup.explain now reads «…удалите codesign на вкладке «Доступ»». Apple's KeyEditor.loctable `302.label` gives Access Control → «Доступ». |
| P2 | MAJOR | fixed | «Разрешать всегда» in setup.explain, setup.always_allow and setup.always_allow_kept. ConfirmAccess.loctable `100267.title` gives «Разрешать всегда». No «Всегда разрешать» is left in the ru catalogues (grep returns nothing). |
| P3 | MINOR | fixed | status.scope: «Проверяется только разрешение «Запись экрана».» |
| P4 | NIT | fixed | setup.explain: «подписывает рантайм удостоверением «{name}»». |

## Changed / new keys
- **status.scope** — «Проверяется только разрешение «Запись экрана». «Файлы и папки» и другие службы конфиденциальности используют ту же строку.» The text follows the English delta and drops Accessibility, as the context note requires. On this Mac (27.2) the SecurityPrivacyExtension `ACCESSIBILITY` key reads «Управление устройством и доступ к данным», which confirms the rename the note describes. Removing «Универсальный доступ» is therefore correct, not a loss. «Файлы и папки» matches `FILE_ACCESS_COMBINED`. The meaning is accurate and the tone is natural.
- **setup.already_signed** — «Рантайм этой сборки Python уже был подписан, поэтому сейчас ничего не подписывалось и запроса от macOS не было.» This matches the shortened English and no longer mentions the earlier answer, as the context note requires.
- **setup.always_allow_kept** (new) — «Ваш прежний ответ «Разрешать всегда» для этого ключа сохранён.» The meaning is accurate. The Apple button label is exact, and «этого ключа» refers back to the key in the same way as setup.explain («доступ к этому ключу»). Capital «Ваш» is correct at the start of a sentence; mid-sentence «вы» stays lowercase. The line is short and fits on one line.
- **setup.explain / setup.always_allow** — the only changes are the pb2 fixes above. The rest of each string is byte-identical to pb2. The «Разрешать всегда» safety disclosure is still complete.

## Source observation from pb2: resolved upstream
In permissions_cli.py:303-308, setup.already_signed is now unconditional and setup.always_allow_kept is printed only when `kept` is true, so the contradiction pb2 reported is gone.

No new findings.
