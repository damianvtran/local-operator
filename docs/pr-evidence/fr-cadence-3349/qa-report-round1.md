### QA report — round 1

Reviewer: qa-tester / openai/gpt-6.1-sol (actual quota fallback; original assigned model anthropic/claude-sonnet-5-5, intermediate switch notice anthropic/claude-opus-5-5; all execution reported here performed after the gpt-6.1-sol switch)
Head: dcdb0b280553861b72b225695497810d201ef890
Scope: parent..dcdb0b280553861b72b225695497810d201ef890; exactly two French `wire.aida_cadence` keys and allowance/ledger/generated integrity
Verdict: PASS — independent runtime/localization scope. Narrow settings/editor pixels BLOCKED, not visually signed off.

Operator request verbatim: “Make sure to continue the work all the way for all languages to get the entire UI completely translated”. This PR's bounded acceptance: French label/help, no fallback for these two keys, 0 daily-local versus 1–168 whole elapsed hours, example 4, already-owed check-in preserved in prose. Non-goals: other namespaces, new UI/behavior, refactors/version changes. This is not proof of all-language/all-UI completion.

#### Environment and preflight

Primary existing interpreter `/Users/damian/local-operator/.venv/bin/python`, no installs. Explicit feature-tree `sys.path` for the scratch probe and `PYTHONPATH` for targeted gates. Verified `local_operator.__file__` resolves to `/Users/damian/local-operator/.worktrees/french-cadence/local_operator/__init__.py` and Git HEAD matches the reviewed SHA. Native/compiled clients were neither dispatched nor claimed. Node v24.22.1 already installed.

Probe started from an isolated scratch CWD with `env -i`, scratch HOME and explicit config root, no inherited `CMUX_*`/`LOP_*`. No sessions, production data, credentials, provider calls, or backend writes. No TUI/Electron/browser launched. Worktree remained clean.

Repro command (scratch `qa_cadence.py` calls real generated keys, resolver, envelope/render provenance, `tr_record`, registry refresh/`resolve_key`, and metadata validation):

```sh
ISO="$LOCAL_OPERATOR_SCRATCHPAD/cadence-isolated"
mkdir -p "$ISO/config"
cd "$ISO"
env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/config" \
 PATH="/Users/damian/.nvm/versions/node/v24.22.1/bin:/usr/bin:/bin" \
 /Users/damian/local-operator/.venv/bin/python "$LOCAL_OPERATOR_SCRATCHPAD/qa_cadence.py"
```

The negative control takes the actual parent English namespace bytes via `git show HEAD^:local_operator/i18n/catalogues/en/wire.aida_cadence.json`, proves the parent has no French namespace using `git ls-tree`, and points the actual catalogue loader at that isolated fixture. Runtime modules and registry were verified unchanged by `git diff HEAD^ HEAD -- local_operator/i18n/{messages,text,resolve}.py local_operator/settings_io.py`. This is a parent-data counterfactual using byte-unchanged production runtime, not a full parent checkout or fake renderer.

| Surface | Actual execution/response | Result |
|---|---|---|
| Parent negative control, both generated keys | Requested `fr`, actual provenance `en`; label `Check-in interval (hours)`; help `0 keeps the daily local check-in time. 1–168 uses elapsed hours; 4 checks in every four hours. Changes preserve an already-owed check-in.` | PASS: reproduces original fallback |
| Locale resolver/generated envelope | `resolve_language(config='fr-FR', environ={}, system='Linux', home=isolated)` → `fr`; envelope `code` unchanged and `params={}`; actual source provenance `fr` for both keys | PASS |
| French label | Envelope and refreshed registry both return `Intervalle entre les points (heures)` | PASS |
| French help | Envelope and refreshed registry both return `0 conserve l’heure locale du point quotidien. De 1 à 168, l’intervalle se compte en heures entières écoulées ; 4 correspond à un point toutes les quatre heures. Toute modification conserve un point déjà dû.` | PASS |
| English unchanged | Switch `fr → en`; both envelopes, namespace-cached resolver and refreshed registry equal English catalogue/parent values exactly; provenance `en` | PASS |
| Settings metadata path | `resolve_key('aida.cadence.interval_hours')` after live locale refresh: French then English label/help as above, default `0`, min `0`, max `168` | PASS |
| Existing numeric validation | `validate(setting, 0/1/4/168)` → `None` for all four; `-1` → `must be at least 0`; `169` → `must be at most 168`; `1.5` → `expected a whole number`; string `'4'` → `expected a number` | PASS |
| Narrative semantics | Explicit assertions cover `heure locale`, `quotidien`, `1 à 168`, `heures entières écoulées`, `4`, `quatre heures`, `déjà dû` in actual rendered French help | PASS, copy only |
| Narrow settings/editor rendered frames | Not dispatched: brief forbids native overlap with active UI native job `16a2cc0b2f6e` | BLOCKED |

No claim that scheduler behavior changed or that a real owed check-in was rescheduled/preserved by this test: this is translation-only, with existing validator exercised and copy semantics checked. No HTTP authentication/tenant matrix applies to this catalogue-only change; no HTTP server or native UI was exercised.

#### Targeted generated/manifest/shard checks

All commands below used the same `env -i HOME="$ISO" LOCAL_OPERATOR_CONFIG_DIR="$ISO/config"` isolation, `PYTHONPATH="$REPO"`, existing Node/Python, with `REPO=/Users/damian/local-operator/.worktrees/french-cadence`:

```sh
/Users/damian/local-operator/.venv/bin/python "$REPO/scripts/i18n/generate.py" --check
# generate.py --check: 123 artifact(s) byte-identical
/Users/damian/local-operator/.venv/bin/python "$REPO/scripts/i18n/check.py" --catalogues-only
# check.py: catalogue checks: 0 problem(s)
/Users/damian/local-operator/.venv/bin/python -m pytest -q -o addopts='' \
 "$REPO/tests/unit/i18n/test_language_manifest.py::test_manifest_is_schema_versioned_and_matches_the_runtime_list" \
 "$REPO/tests/unit/i18n/test_language_manifest.py::test_every_missing_key_is_covered_by_the_shrink_only_allowance"
# .. [100%]
# 2 passed in 1.93s
```

PASS: additional direct JSON/hash assertions compared `git show HEAD^:i18n/ledger/fr.json` to committed head: previous 133 entries, current 134, entire old prefix equal; exact appended row appears in generated aggregate, status passed, source SHA-256 `3bb421a58f2ffa4fec089421d9cb91257fa626d66b2e215b3e120c068bb8e5dd` equals committed English namespace bytes. Coverage allowance comparison removed exactly one `(fr, wire.aida_cadence)` pair and exactly two key debts, no replacement growth. `git status --porcelain` returned empty; HEAD remained the pinned SHA.

No duplicate 93-test/full suite: these are two selected manifest tests plus targeted generation/catalogue checks and independent real-runtime requests. CI completion is not asserted by this report.

Q-findings: none from executed scope. Remaining evidence gap: narrow settings/editor wrapping/clipping and before/after pixels, intentionally BLOCKED by the native resource hold; manager/design must retain that gap rather than treat runtime strings as rendered proof. No repository edits, review requests, tags, merges or releases performed.
