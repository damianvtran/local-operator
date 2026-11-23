# #4403 cwd-unknown copy — slot 2 own findings, recorded BEFORE reading any slot-1 output

Scope: en frozen at 3030a5e5c (catalogue_sha256 d782f1ded36d43133e7a1c32db9bd89540030eecdba86be92e068b8a2850a480, verified by blob sha256 and repo function).
Translations: /home/ec2-user/.local-operator/sessions/9a6e88d78421/scratchpad/tr4403 (index 51b82e36…, all 8 per-file sha256 verified).
Auditor: translation-auditor (lopdev), session 6fd346121602, anthropic/claude-opus-5-5, no fallback.

6 keys × 7 locales = 42 values read against en + context + mirrored certified rows (advice_stays, receipt_keep, receipt_not_transferred, yes_*).
Mechanical: placeholders parity, all flags + `lop` verbatim, no bidi/NBSP, fr U+202F everywhere, zh no CJK↔Latin spaces.
Safety: cwd_unknown says NOT included + stay on {source}; needs_confirmation_cwd_unknown binds --confirm-working-set to LEAVING; yes_cwd_unknown refusal + nothing moved, never "destination cannot take"; receipts non-error, mirror certified frames. PASS 42/42.

- T5-VI-1 nit  vi cwd_unknown «không được tính vào» ("not counted in") → «không được đưa vào»
- T5-RU-1 nit  ru cwd_unknown / receipt_cwd_unknown / receipt_keep_cwd_unknown «не записан» → «не сохранён» (more natural for a stored record); meaning intact
Observations (not findings): hi working-directory term split is pre-existing (कार्य निर्देशिका in wire.slash.move/relay/settings vs कार्यशील डायरेक्टरी in tui.session_move/global_help); translator followed the /move slash precedent. fr/es/ru push phrasing mirrors certified advice_stays. ur yes_cwd_unknown / receipt_keep_cwd_unknown start with a Latin run, same as certified yes_* / receipt_keep (known degraded TUI stance).

## ADDED after mechanical repo check (still before reading any slot-1 finding/verdict)
- T5-HI-1 BLOCKER hi needs_confirmation_cwd_unknown: placeholder ORDER {source},{digest} vs en {digest},{source}; repo `scripts/i18n/check.py --catalogues-only` on the merged target reports "renamed or reordered" (2 problems: hi+ur; same-code control with base catalogues = 42 missing-key problems, merged = only these 2). Replacement: इन फ़ोल्डरों को --dry-run से देखें, फिर --confirm-working-set {digest} के साथ दोबारा चलाएँ ताकि ये {source} पर ही रहें, या --without-working-set के साथ।
- T5-UR-1 BLOCKER ur needs_confirmation_cwd_unknown: same. Replacement: ان فولڈرز کو --dry-run سے دیکھیں، پھر --confirm-working-set {digest} کے ساتھ دوبارہ چلائیں تاکہ یہ {source} پر ہی رہیں، یا --without-working-set کے ساتھ۔
  With both replacements check.py = 0 problems. Replacements are auditor proposals, NOT self-certified: the translator must apply them (or its own order-preserving value) and both slots re-audit that delta.
Merged targets computed independently; byte-identical to slot 1's merged/ files (bytes compared only, no slot-1 findings read).


## After reading slot 1 (7066677170)
- Concur L1-VI-1, L1-ZH-1 (nits).
- DIVERGE: slot 1 passed hi/ur; slot 2 finds the placeholder-order BLOCKER (check.py fails on slot 1's own merged targets, byte-identical to slot 2's).
