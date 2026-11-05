# French second-reader audit — `wire.errors.sdk_startup` (4 keys)

**Verdict: PASSED (staged linguistic/mechanical audit).** No BLOCKER, no MAJOR. **One MINOR** finding (F1) covering two remedy clauses. Key-level PASS does not mean the namespace is globally UI-clear or that its rendered output is English-free — see "Remaining proof".

- Locale `fr`; namespace `wire.errors.sdk_startup`; 4/4 keys audited.
- Pinned source commit `0e7cde33ca8008d561e74f4aed774aa5f76ff7d5`.
- Auditor `i18n-auditor`, reader role **fr-second**, session `35aac0a677bc`.
- Translator session `7b2ba4ed501d`; coder actual session `eae0961e4e78` (not job `196c3a94b346`). This reader wrote none of the source, target or context, and read no other reader's report before this verdict.
- **Actual exposed served model:** `radient/auto` — an opaque provider label. Own completed persisted `payload.usage` for this task reports exactly `provider=radient`, `model_id=auto`; no underlying model is exposed, so none is claimed. No route/fallback notice was emitted during this task, and no no-fallback claim is made. Provenance: `provenance.json`.

## Inputs (independently rehashed)

| Input | SHA-256 |
|---|---|
| English `catalogues/en/wire.errors.sdk_startup.json` | `5390ecb2413e0978c34541a9a0db6c2a004ae86446d1c0fa0d2e697e4f9ca6bd` |
| English context sidecar | `ee0bbde5e99de22c04368e6bbfedf913d758ef6ba5075984ddf7d795b1ee7822` |
| French target `wire.errors.sdk_startup.json` | `82b61a755a8580e7cb4e4f86ae64a428fc83232fac92b22fa72f09d77b5f781d` |
| French style guide (snapshot) | `ae1519ab7617d19aa826918d607de1d018595094166edbc18aa58f6a8f0a4226` |
| Glossary (snapshot) | `0eb34437a8c3d6f564f278ef957a603b1d9f51bb07d5ddae5ea8f37deba962f8` |

The canonical packet manifest's **12 entries were each rehashed**: all match, byte counts included (`input-verification.json`). The English catalogue and context sidecar are additionally **byte-identical to the pinned commit blobs**; the French file is **absent at that commit** (the 4-key gap this packet fills). The translator's shortened narrative hashes were not used as authority.

## Caller: this namespace is intended to localize (not English-forced)

Verified in pinned `local_operator/sdk.py:587-598`, not inherited from the brief: `startup_refusal_text()` reads `config_manager.get_config().values["language"]`, resolves it with `resolve_language(...)`, and calls `messages.render(message.code, message.params, locale=locale)`. Nothing pins `locale="en"` on these four producers — unlike the 21 `wire.errors.sdk_spec` guards, which each pass `locale="en"` explicitly.

By inspection of the same pinned code, a real configured `fr` reaches this target with no producer change: `fr` is in the generated shipped set (`local_operator/i18n/data/shipped.json`), `resolve_language("fr")` yields `fr`, and `messages.render(..., locale="fr")` selects the French catalogue when it exists. The four producers are `attach_hosting_unconfigured` (sdk.py:626-633, inside the `except HostingNotConfiguredError` handler for `mode='attach'`), `attach_runtime_missing` (sdk.py:656-660, after `dispose()` of the unexpected cold owner), `profile_attach_failed` (sdk.py:669-673) and `tool_inventory_mismatch` (sdk.py:689-695). This is source inspection; no refusal was executed.

Default English is unchanged: rendering each key with `locale="en"` is **byte-identical** to formatting the source message directly (asserted in the probe).

## Native assessment, key by key

### 1. `attach_hosting_unconfigured`

> aucun runtime actif ne sert 'abc123', et cette racine ne peut en démarrer un : {error} Configurez la plateforme d’hébergement de cette racine (son config.yml), puis démarrez l’identifiant avec spawn_session()/deliver() et rattachez-vous dès qu’un runtime existe.

Meaning is preserved: no live runtime serves the id, this root cannot start one, remedy = configure hosting via `config.yml`, then engage the id with `spawn_session()/deliver()`, then attach once a runtime exists. `runtime`, `racine`, `plateforme d’hébergement` and `config.yml` are correct per the French guide/corpus. Register is `vous`, no "Veuillez". **F1 applies** to the closing clause.

**Opaque `{error}` and the sentence boundary (explicitly checked):** the value is inserted verbatim — braces, quotes, `%`, em dash and non-ASCII all survive (`mechanical.json`), and the ICU parser never re-reads it. The French remedy starts with a capital directly after `{error}`, exactly as the English does, and the French adds no separator of its own. The boundary therefore holds **because the sole current producer ends its sentence**: `session_factory.py:816` raises `HostingNotConfiguredError("Hosting platform is not configured.")` — the period is what makes " Configurez…" a new sentence. An opaque value without terminal punctuation would run into the remedy in **both** languages equally; that is a source-contract property, not a French defect. Conversely, the embedded cause sentence itself remains **English** inside French output (see Remaining proof).

### 2. `attach_runtime_missing`

> aucun runtime actif ne sert 'abc123' ; ouvrez-le avec mode='own' pour l’exécuter dans ce processus, ou démarrez-le d’abord avec spawn_session()/deliver(), puis rattachez-vous dès qu’un runtime existe.

Both remedies survive: run it in-process with `mode='own'`, or engage it first and attach once a runtime exists. `mode='own'` keeps its ICU-escaped source spelling and renders as `mode='own'`. "ouvrez-le"/"démarrez-le" agree with `identifiant` (masculine), consistent within the namespace. The French "puis" makes the two-step order clearer than the English "and", without adding a step. **F1 applies** to the closing clause.

### 3. `profile_attach_failed`

> Impossible de rattacher le profil 'profile-value'

Faithful and native: "Could not attach profile {profile}" → "Impossible de rattacher le profil {profile}". Capitalisation of the source sentence-initial word is preserved in kind (French capital `I`); `{profile}` is repr-quoted opaque data and stays untouched. No lifecycle or exception-shape implication is added. PASS.

### 4. `tool_inventory_mismatch`

> spec.tools ('bash', 'read') et ApprovalPolicy.declared(('bash', 'edit')) désignent des inventaires différents ; ne déclarez la portée de l’exécution qu’une seule fois

Both identifiers and both stringified tuples are preserved verbatim; `désignent des inventaires différents` conveys the disagreement, and `ne déclarez … qu’une seule fois` is an accurate restrictive rendering of "state the run's reach once" (the negative *ne … que* means "only once", not a polarity flip). No inventory widening is implied. "portée de l’exécution" is a slightly abstract but usable rendering of "the run's reach"; it is developer-facing and unambiguous next to `spec.tools`. PASS.

## Findings

**F1 — MINOR (two occurrences, one wording decision).**
Keys: `wire.errors.sdk_startup.attach_hosting_unconfigured` and `wire.errors.sdk_startup.attach_runtime_missing`.
Current text (both, final clause): "… et rattachez-vous dès qu’un runtime existe." / "… puis rattachez-vous dès qu’un runtime existe."
Problem: `se rattacher` is incomplete without its complement. The French corpus always supplies one — "rattachés à un enregistrement", "rattache au runtime", "une vue se rattache à un identifiant", "rattachez les éléments nécessaires" — so the bare reflexive reads as "attach yourself", a calque of the English intransitive "attach". Meaning and action survive, hence MINOR rather than MAJOR; it is still copy a native review would correct before integration.
Suggested wording: "… puis rattachez-vous à la session dès qu’un runtime existe." (both keys), which keeps the source's voice and supplies the object.
Note on severity: if the coordinator treats any non-idiomatic remedy clause as blocking, F1 is the single item to fix; two strings, one decision. I did not inflate it, because no user is misled and no technical behaviour changes.

No other MINOR or NIT is raised. I deliberately do not file the cross-packet vocabulary difference (`démarrez` here vs `sollicitez` in the staged, unintegrated `wire.errors.sdk_spec` French) as a finding: neither packet is integrated, and choosing one verb is a coordinator call across two staged namespaces, not a defect in this one.

## Mechanical execution (actual)

Command (isolated 4-key fixture; own root; no repo catalogue touched):

```bash
for name in ${!CMUX_@}; do unset "$name"; done
cd ~/local-operator
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  "$LOCAL_OPERATOR_SCRATCHPAD/fr-second-sdk-startup-0e7cde33/verify.py"
```

- Tool ceiling **256 MiB**, timeout **90 s**; serial single invocation. **Exit 0**; stderr empty.
- Repository checker `scripts.i18n.check.check_catalogues()` on the fixture: **0 problems**.
- Exact 4-key parity and order; arguments verified against source and additionally pinned per key: `{resume}`+`{error}`, `{resume}`, `{profile}`, `{spec_tools}`+`{policy_tools}`, all `simple`.
- **No plural/select anywhere** → CLDR categories **not applicable** (not "missing"). No markup/Rich tags. No `Cf` character. Technical tokens `config.yml`, `spawn_session()/deliver()`, `mode=''own''`, `spec.tools`, `ApprovalPolicy.declared(` present.
- French typography derived per character: **U+202F before every `:` and `;`**, U+2019 apostrophes, and the only straight apostrophes are the ICU `''` escapes inside `mode=''own''`.
- Production render path exercised (`messages.render`, `render_record`) with synthetic opaque values (braces, `%`, quotes, em dash, non-ASCII): French output selected, not English; opaque data verbatim; ICU never re-parsed. `resolve.eligible_locale("fr") == "fr"`, `eligible_locale(None) == "en"`, `normalise_locale("fr-CA.UTF-8") == "fr"`.
- Missing-parameter behaviour recorded as observed, not assumed: a key with an unsupplied argument degrades to the **code**, never to a half-English string.
- **8 negative controls rejected**: missing key, renamed argument, extra argument and malformed ICU by the repository checker; technical-token drift (`config.yml`→`config.yaml`), English stub, `Cf` injection and ASCII space before `;` by independent checks.
- Widths: no `maxCells` in the sidecar for any of the four keys (measured cells 323/200/40/166 with synthetic opaque values — informative only; **no width budget exists to violate**).
- Execution honesty: one earlier probe run exited 1 on an auditor-authored assertion bug (a brace check applied to keys that take no `{error}`); `verify.py` was corrected and the recorded run above is the post-fix run. No repository or packet file was modified by either run.

## Remaining proof and boundaries

- **NOT RUN:** native/app fit, screenshots, live language switch, pseudo-locale, real refusal paths, settings/API requests, scanner, full suite, builds, installs. RTL is not applicable to French.
- **Known remaining English inside this French string:** `{error}` carries the producer's own English sentence ("Hosting platform is not configured."). The catalogue cannot remove it without changing the producer/exception contract, which this contribution explicitly does not do. Under the operator's "no English anywhere" goal this is a real, separately-owned display-boundary item (the architect assessment already recommends a display-specific message omitting raw cause prose). It is **not** introduced or worsened by the French translation, and it is **not** filed as a translation finding.
- No direct current UI sink for these refusals was established here; the architect assessment reports none, and that is a named source, not my own census. Native/rendered caller fit therefore remains unproven.
- Passing this audit is **not** global UI clearance, namespace activation, ledger entry or release eligibility. No repo/ledger/generation/source-factory/API/PR/native/build/install/scanner/full-suite action was taken.
- Original packet preserved untouched. This audit's own bundle is retained read-only in session scratch for coordinator promotion to `i18n/audits/evidence/fr/wire.errors.sdk_startup/`; it is not claimed on main. Earlier French audit bundles are unchanged and were not re-audited.
