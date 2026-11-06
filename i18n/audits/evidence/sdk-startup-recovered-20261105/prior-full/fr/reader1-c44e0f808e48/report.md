# French `wire.errors.sdk_startup` — first independent audit

**Verdict: passed** for the four French target strings. BLOCKER 0, MAJOR 0, MINOR 1 (context sidecar metadata, not the translated copy), NIT 0. Result label: `clean-with-findings`; the unresolved MINOR is metadata, so the runtime/linguistic verdict stays `passed`.

Auditor: `i18n-auditor`, session `c44e0f808e48`, served route **`radient/auto`** (opaque; underlying model undisclosed). Translator session `7b2ba4ed501d`; coder `eae0961e4e78`. This reader is distinct from both, wrote no target text, and did not read the other new reader's report before the verdict.

## Inputs (canonical bytes verified, not narrative)

| Input | Path | SHA-256 |
|---|---|---|
| English | `local_operator/i18n/catalogues/en/wire.errors.sdk_startup.json` @ `0e7cde33ca8008d561e74f4aed774aa5f76ff7d5` | `5390ecb2413e0978c34541a9a0db6c2a004ae86446d1c0fa0d2e697e4f9ca6bd` |
| Context | `…/en/wire.errors.sdk_startup.context.json` @ same commit | `ee0bbde5e99de22c04368e6bbfedf913d758ef6ba5075984ddf7d795b1ee7822` |
| French target | packet `fr-sdk-startup-0e7cde33/wire.errors.sdk_startup.json` | `82b61a755a8580e7cb4e4f86ae64a428fc83232fac92b22fa72f09d77b5f781d` |

All **12** manifest entries were re-hashed and matched their own manifest; the manager-quoted target/EN/context hashes match the files. Original packet bytes preserved.

## Locale path — derived, and it changes what this namespace is

`local_operator/sdk.py:584–598` (`startup_refusal_text`) reads the already-scoped `ConfigManager`, calls `resolve_language(...)`, then `messages.render(message.code, message.params, locale=locale)`. I exercised that exact boundary on an isolated fixture: `fr` → `Impossible de rattacher le profil 'demo'`; `en` → `Could not attach profile 'demo'`; with the `fr` file removed, the same call falls back to the English source. So these four keys are **not** English-pinned like the 21 `wire.errors.sdk_spec` guards, and `docs/SDK.md:220` says so explicitly (“Only these refusal paths lazily resolve language…”). No producer, exception, attribute or API change is implied by this catalogue.

## Per-key linguistic verdict (all four PASS)

1. **`attach_hosting_unconfigured`** — “aucun runtime actif ne sert {resume}, et cette racine ne peut en démarrer un : {error} Configurez la plateforme d’hébergement de cette racine (son config.yml), puis démarrez l’identifiant avec spawn_session()/deliver() et rattachez-vous dès qu’un runtime existe.”
   Both clauses and the remedy survive: no live runtime, this root cannot start one, embedded cause, then configure hosting in the root's `config.yml`, engage the id, attach. “plateforme d’hébergement” is cohesive with the embedded producer sentence `HostingNotConfiguredError("Hosting platform is not configured.")` (`session_factory.py:816`, verified at the pin) rather than an invention. “Cette racine” matches the sibling locales and the `SessionRoots` object. `{error}` keeps its position and its following sentence boundary; the producer string ends in a full stop, so “Configurez” reads as the intended new sentence.
2. **`attach_runtime_missing`** — “aucun runtime actif ne sert {resume} ; ouvrez-le avec mode=''own'' pour l’exécuter dans ce processus, ou démarrez-le d’abord avec spawn_session()/deliver(), puis rattachez-vous dès qu’un runtime existe.”
   Both alternatives and their order preserved; `mode=''own''` is ICU-escaped exactly as in the source and renders `mode='own'`; “rattachez-vous” keeps the viewer-attaches reading already established in the French `sdk_spec` corpus (“session rattachée”, “se rattache à un identifiant”).
3. **`profile_attach_failed`** — “Impossible de rattacher le profil {profile}”. Natural diagnostic, capitalization preserved, `{profile}` untouched.
4. **`tool_inventory_mismatch`** — “spec.tools {spec_tools} et ApprovalPolicy.declared({policy_tools}) désignent des inventaires différents ; ne déclarez la portée de l’exécution qu’une seule fois”.
   “name different inventories” → “désignent des inventaires différents”; “state the run's reach once” → negative imperative “ne déclarez … qu’une seule fois”, which is the idiomatic French for a one-place declaration and does not invert the remedy. Both identifiers and both opaque placeholders are in place.

Register is consistent *vous* with imperative remedies, no “Veuillez…”, no Title Case. Terminology is consistent with the style guide and glossary: `runtime` kept English (wave-1 addition), `profil`, `inventaires`, “rattacher”, `config.yml`/`spawn_session()`/`deliver()`/`spec.tools`/`ApprovalPolicy.declared` literal. French punctuation is correct: every `:` and `;` is preceded by U+202F, apostrophes are U+2019, and the only ASCII apostrophes are the four ICU-escaped ones in `mode=''own''`.

## Findings

### F1 — MINOR (context sidecar, not the translation)
Key `wire.errors.sdk_startup.attach_hosting_unconfigured` (context entry). The note ends: “English-only child namespace falls back under RFC §6 per-CARRIED policy; future translated consumers still owe the RFC migration boundary.” That is false for this namespace: the verified path renders the resolved locale, and the other three notes in the same file do not say it. It contradicts the packet README and `docs/SDK.md`, and would lead a later translator or auditor to treat a translatable namespace as English-pinned.
Suggested wording: “Renders in the resolved locale after root isolation (`sdk.py::startup_refusal_text`); English is the fallback when the catalogue is absent, not a pin. Keep `{error}` as opaque producer text and preserve its sentence boundary.”
This does not change the target verdict: the French text is correct under either reading.

### Observations (non-gating, not counted)
- **O1** `attach_hosting_unconfigured`: the opaque `{error}` is followed immediately by a capitalised French sentence, exactly as in English. With today's producer (full stop) it reads correctly; if that producer ever stops ending with punctuation, both locales degrade equally — no extra punctuation should be added inside the placeholder.
- **O2** Repository docs still inventory this namespace as English-only: `docs/design/backend-originated-strings.md:60` (“Additional English-only family”) and the `docs/SDK.md:220` heading (“English `wire.errors.sdk_startup` keys”, whose body then describes language resolution). That is a documentation-consistency item for the coordinator, outside this catalogue audit.

## Executed evidence (actual, bounded)

`env -u XPC_FLAGS PYTHONDONTWRITEBYTECODE=1 /Users/damian/local-operator/.venv/bin/python …/sdk-startup-fr-audit.py` — exit 0 under a 256 MiB process limit and 90 s budget; output in `audit.log`.

- Core `scripts/i18n/check.py::check_catalogues()` on a one-namespace, four-key fixture: **zero problems**.
- Negative controls rejected: added argument, missing key, invalid ICU (core checker), and a translated technical literal (`mode=''own''` → `mode=''perso''`, caught by this audit's literal guard — the core checker does not police literals).
- Argument names/types identical for all four (`resume`+`error`, `resume`, `profile`, `spec_tools`+`policy_tools`), no plural/select forms, so CLDR arms are not applicable.
- Opaque values containing braces, `%`, `#`, quotes and « » render verbatim through the core runtime; **Cf count 0** in every string, no ESC/NUL/TAB.
- Width for information only (no `maxCells` in the context; exception text, not a fixed-cell surface): 303, 220, 62, 190 cells.

Three assertion-side corrections were needed while building the harness (negative-control expectation, argument-free render call, token-guard scope); none touched the target, and the canonical target hash was re-verified after each run.

## Limits

**NOT RUN:** pseudo-localization, RTL/native rendering, rendered UI or terminal fit, real SDK guard execution against a configured root, full suite, scanner, build, install, ledger generation and activation. No production `fr` catalogue is integrated by this audit, so the French path is proven at the render boundary and with fixtures, not in a served app. A passing linguistic audit does not claim global interface completeness or release eligibility.

**Remaining proof:** second independent reader; integration of the catalogue behind the per-namespace ledger gate; end-to-end startup-refusal execution with a real configured `fr` root; rendered fit; and the docs/context corrections in F1/O2.
