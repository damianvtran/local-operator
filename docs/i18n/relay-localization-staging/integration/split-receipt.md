### Code-only split correction receipt — 2026-10-14
Implementer: coder / openai/gpt-6.1-sol, actual model/no fallback. This is implementation/cause evidence, **not independent review/QA/rendered sign-off**.
Scope: `209d18ed4a839a44be07270bbe35bd4fb163eca2..57bcb939881066c7e645cdec604fd72c3e667471`.
Commit: `57bcb939881066c7e645cdec604fd72c3e667471` — `fix(i18n): split render-held relay candidates from active catalogues`.

**Latest parent disposition supersedes the translated-ledger integration**
209d was already pushed and is retained as an ancestor—no history rewrite/reset. Its native325 strings/15 pending rows and independent audit findings remain preserved. The current PR is **code-only**: render-held locale assets are no longer active generator/catalogue inputs. No separate locale PR, activation, merge, release or installed-runtime change.

Exactly20 corrective files (+15/-868): remove ONLY the15 locale JSONs introduced by209d (five locales × shell/session/login), restore original ledger55 rows, own2 snapshot edits, generated shipped data, and remove native-file-dependent fixture imports/cases. Every removed JSON was hash-checked and copied outside the generator directories first. No user-owned/older locale file was deleted. Original ledger restored **RAW byte-identically** to pre209d/de9 bytes—no downgrade/erasure/reclassification of existing rows. No validator or eligibility implementation change.

Retained unchanged: three English count-aware unit sources, exact contexts, finite measured-count caller, generated count argument factories/types/bundle, previous whole-row composition and decimal/percent/date formatting, English/synthetic CLDR regression cases. **No outer-plural amount change.**

Final English raw/API SHA-256 unchanged:
- shell44: `f53b23560b88b531ef8fc3274bc396676f26e755ad4ff9ad87ce368cebdee560`
- session17: `a4ae148de969d4a2e8541aca5b4b99ad311d9ef112a0d056fc21fcf90e7e0129`
- login4: `9a59686bcd0e3641a5a807ea2b778cb385e591060c26c7fb994bea95e4de5bc5`
- shell.context: `16746ecd1537006992935bb2aa88fe39c46700d9e0341e7aacd565e2366349ea`

**FIRST actual-main negative-comparison control, before touching fixtures**
Primary shared checkout was NOT currentmain (2115d3e/0.70.16), so it was not falsely presented as the control. Reused existing clean main checkout at `348c6b43479b30fa8190cbcfdfd9ac281f54b510` (`0.71.0`), independently matching Git ls-remote AND GitHub main ref. No new worktree/source copy/dependency tree. Existing primary venv used directly, PYTHONPATH points to actualmain checkout; printed/imported local_operator and catalogues __file__ both inside that checkout. Cacheprovider disabled, Python-B/no-bytecode; peer checkout remained clean.

Exact bounded isolated command (same six parametrized cases for both controls):
```text
python -B -m pytest -p no:cacheprovider -n0 -q
 tests/unit/i18n/test_hold_split.py::test_hindi_team_bridge_delta_uses_audited_copy_not_english_fallback
```
`env -i` scratch HOME/config/TMPDIR, verified PYTHONPATH, TERM=xterm-256color; serial512MiB/60s. Actual-main control **6 passed in2.57s BEFORE correction**. Corrected branch **6 passed in1.03s AFTER correction**, using its own verified source. This specifically proves209d's introduction of carried-but-unpassed namespaces caused the six prior public Hindi→English regressions; their previous failure receipts are historical, not rewritten.

The six unchanged assertions are:
- `[team_bridge.description]`
- `[team_bridge.title]`
- `[team_bridge_enabled.help]`
- `[team_bridge_enabled.label]`
- `[team_bridge_session.help]`
- `[team_bridge_session.label]`
No skip, monkeypatch-for-green, relaxation or source-locale forcing was added to these tests.

**Own generator and exact eligibility proof**
Ran repository `scripts/i18n/generate.py`, never hand-edited generated shipped data. `generate.py --check`: **32 artifacts byte-identical**.
-209d carried candidate state: `[en]` (previously disclosed merge blocker).
-Corrected generated state: `[en,fr,es,ru,vi,hi]`.
-Compare raw generated file to `git show 348c6b43479b30fa8190cbcfdfd9ac281f54b510:local_operator/i18n/data/shipped.json`: **BYTE-IDENTICAL**, SHA-256 `8c37aff60b6cb448a2c1f638b525d24c98b7bb1870fd44d0cbbe2ca12a476d8c`.

Restored only our two209d snapshot edits: test_catalogues.py and test_hold_split.py are raw byte-identical to de9/pre209d files. Narrow rerun **2 passed in0.85s**. Code-only usage formatting/count fixture **13 passed in2.02s**; removed ONLY10 imports and5 candidate-dependent cases whose files are now staged outside the repo, preserving the exact209d fixture for later replay. Real JS gate **0 enforced /668 advisory**. No full local suite, compiler/type retry, browser/rendered/native/backend dispatch or dependency install. Current-head CI provides terminal automated coverage and remains pending, not a green claim.

**Native assets preserved, not published**
All15 native files (325 entries) retain their approved first/second audit raw hashes. All15 translated pending ledger rows, full209d raw ledger, and complete209d actual-native fixture retained under the session scratch `c2329/staged-native-after-split/`. Exact matrix is `integration-verified-matrix.json`; no native message was edited/coerced. Their sourced [aggregated native receipt](https://github.com/damianvtran/local-operator/pull/2329#issuecomment-6175378702), including context addenda, ES65-3/RU-M1 deferrals and historical findings, remains authoritative **native/linguistic evidence only**, not rendered/publication approval. Candidate files can also be recovered exactly from209d history.

**Eligibility explanation; minimum future proposal ONLY**
`generate.py::_shipped_locales` checks EVERY namespace a locale CARRIES: each needs a current-source status=passed row. The2026-10-11 per-namespace amendment narrows the checked set to carried namespaces rather than all English namespaces; it does **not** permit partial locale activation when one carried namespace is translated/render-held. Therefore adding three translated relay namespaces made each of the five locales globally ineligible, and `Msg.envelope(locale=hi)` fell back to EN even for already-audited wire.settings copy. Removing the entire newly staged candidate wave restores the original carried set and eligibility without weakening the rule.

Minimum proposal: keep unready native candidates outside active catalogue roots; after actual rendered/full-publication audit passes, create the requested later locale PR with these exact source/target pins and completed truthful ledger evidence. Do not implement a generator filter that ignores pending rows, selectively delete historical failed namespace evidence, or mark native-only audit passed. A different partial-namespace activation policy would require an explicit contract/security decision/RFC—not an incidental fix. No such mechanism change is made here.

**Remaining gates**
Original Q-V1/StageA resource/compliance/rendered/design obligations remain. Independent correction-head code/QA and actual required CI must be read; no prior review is blindly re-pinned. Head pyproject remains inherited0.70.29 with zero authored version diff; actualmain0.71.0 was verified and future base/version convergence remains for the manager—no unauthorized auto-rebase in this correction. No merge/render/access/locale PR or activation. Exact lease pinned to209d accepted; new remotehead57bcb939 verified;209d ancestor verified; worktree clean and shared .local/raw.log untouched.

Retained new evidence: `split-main-control.log`, `split-branch-control.log`, `split-generate.log`, `split-generate-check.log`, `split-mobile-tests.log`, `split-js-gate.log`, `split-snapshot-tests.log`, `split-ci.json`, plus staged assets and all historical receipts under this session's c2329 scratch.
