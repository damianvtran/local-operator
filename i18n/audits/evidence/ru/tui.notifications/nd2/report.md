# i18n audit round 2 — tui.notifications — ru — PASS

Auditor: independent i18n-auditor / reviewer (lopdev, session b4869f40ac63). Actual served model: kimi/k3.1-pro (provider-quota fallback from anthropic/claude-opus-5-5 via anthropic/claude-sonnet-5-5 and openai/gpt-6-astra). I authored none of this translation delta.
Pinned code: 897ffb54fef0f6a889bb135a22c80f7eebc3e0e5; the requested 0bffa9f447f2c697319defc25ddfd42c90a3f47d moved by one code-only commit. All i18n/catalogue/context bytes are identical at both heads.
English source SHA-256: b405d23feb36ff29a4f75895513ec027cab13e4420a71ea05a83e515d8b20224; context sidecar: 1b4fcb28e063b7e646809d801675dfbda00f691ba7cec57edab6a322d0cab4d4.
Target SHA-256: 85b808d1c3baffc6d4c49cdb24c6642bcd194238d9d3c660fed20d2dc4516dec. This exact target file was parsed and rendered in the isolated export.

## Mechanical and rendered evidence

75 keys in exact English key order. The 71 values outside the four-key delta are byte-identical to the previously audited parent target. Both plurals were checked independently against the runtime's CLDR category set; no empty arm and no missing category. Every one of the 72 combinations of count (0,1,2,3,5,6,11,21,22,25,101,1234) and updates (0,1,2,5,6,11) was rendered using `runtime.render_message`. See `matrix.log`; `0` `!!NUM` markers across all locales. Arguments are `(count plural, updates plural, source simple)` in English first-appearance order.
NFC, no bidi controls, no BOM/CR, one final LF, and JSON re-serialization byte-identical. Hints fit maxCells. Confirm rows fit 96 cells in every sampled arm with source `Fix the relay`; see `audit.log` for exact per-locale widths and empty-source budgets. Meaning and both numbers are preserved.

## Gate counterfactual (same isolated export)
At the audited head before any ledger change: `generate.py` and `--check` rc 0, `check.py` rc 0, `tests/unit/i18n` **77 failed, 760 passed**. After six PASS rows and this locale's recorded verdict: generation rc 0 and `shipped.json` is `[en, es, zh-CN, ru, vi, ur, hi]`; `generate.py --check` says **97 artifact(s) byte-identical**; `check.py` says **184 files, 2264 literals at/below baseline, catalogues consistent, 37 advisories, rc 0**; tests are **27 failed, 810 passed**. Every remaining failure is caused by the deliberate French hold or its hard-coded shipped-set expectation. At requested head 0bffa9f, `check.py` additionally reproduces the unrelated `notifications_render.py` ratchet finding (1 literal/ceiling 0); current head 897ffb54 fixes it with a narrow glyph-chrome ignore, so it is no longer red.
