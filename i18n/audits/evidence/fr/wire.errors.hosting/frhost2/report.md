# French audit — `wire.errors.hosting.not_configured` (1 key)

**Verdict: PASSED, zero findings.** No BLOCKER, MAJOR, MINOR or NIT. One key, one sentence, translated cleanly and idiomatically; the display key names the owned condition without adopting the legacy English exception text.

- Locale `fr`; namespace `wire.errors.hosting`; **1 key**, zero ICU arguments.
- Auditor `i18n-auditor`, session `35aac0a677bc`, **reader 2**; translator/author session `7b2ba4ed501d`; code author `94dcef624dec`. I wrote none of the source, context or target, and did not read the other reader's verdict before forming this one.
- **Requested model:** `radient/auto` (session header). **Served model:** `radient/auto`, from my own completed `payload.usage` for this task (5 entries, all `radient`/`auto`). No active-model-route notice fired. **Underlying model: UNKNOWN** — not inferred, no fallback claimed.
- Source is an **uncommitted candidate**: the English namespace is absent at base `87ca846245b2`, so nothing here is shipped, activated or eligible.

## Hashes computed here (stdlib json/hashlib/unicodedata only)

| Input | SHA-256 | Bound value | Match |
|---|---|---|---|
| English | `458d8f8be304428ccb6a328ce22e5a86515e9f44f869f0138a5cc6dfde964b63` | same | yes |
| Context | `ed0ae50fa465a010a2907bc680771dae80bc716d3a186b0af37e31e0d7063cf9` | same | yes |
| Target | `83432e2af01eff5982e4d7a0fca57ebb2223626b0d5713c136d60bb6bec8aa69` | same | yes |

## Mechanical (verified myself, no parser or process)

- Valid JSON; **exact key parity and order** across English, context and target (1 / 1 / 1).
- **Zero ICU arguments** in both source and target, so placeholder parity, plural/select structure and CLDR category checks are **not applicable** rather than untested: there is nothing for ICU to parse or select.
- NFC-normalised; no combining marks; **no `Cf`** and no bidi controls; no braces; no leading or trailing space.
- Terminal period present; sentence starts with a capital; the only apostrophes are **U+2019** (`d’hébergement`, `n’est`), no ASCII apostrophe.
- Target differs from English, so it is not an English stub. No remedy, path, `config.yml`, `spawn_session()`/`deliver()` or other frame wording was added.

## Native judgement

> EN: “Hosting platform is not configured.”
> FR: “La plateforme d’hébergement n’est pas configurée.”

Natural, complete French statement: definite article, indicative state, feminine agreement with *plateforme*, terminal period. It names the condition and stops — it does not restate the surrounding refusal's remedy, which is exactly the sidecar's rule.

Terminology is the **shipped corpus's own term**, not a coinage: `cli.global_help.hosting` (“Plateforme d’hébergement à utiliser”), `cli.exec_worker_help.hosting` (“Plateforme d’hébergement à forcer”) and `cli.credential_help.credential` (“… plateformes d’hébergement”). I judged French against that corpus rather than against other locales' choice of term; VI using a provider word is not a reason to change French.

## Contract boundary, checked not assumed

- The legacy `session_factory.HostingNotConfiguredError("Hosting platform is not configured.")` string is still English in source (line 816) — this key is **display copy** and does not shadow, replace or alter the exception. No caller, API error, envelope or ICU contract was edited.
- Two other unkeyed `ValueError("Hosting platform is not configured.")` sites exist (`server/routes/speech.py:643`, `bootstrap.py:83`). They are **not** consumers of this key and I make no coverage claim about them; recorded so nobody reads this pass as "the string is gone everywhere".

## Advisories (not findings)

1. **Capital after the colon is deliberate.** French typography normally lowercases after a colon, but the sidecar requires the clause to keep its capital and period as a complete statement spliced mid-sentence, mirroring the English. A later copy edit should not "fix" it.
2. **Definite article mirrors the source.** “Aucune plateforme d’hébergement n’est configurée.” would stress that *no* provider was resolved, which is the preflight's actual result, but it departs from the English definite form. Recorded as an alternative, not requested.
3. **No width evidence.** The sidecar declares no `maxCells`; the value is 49 codepoints and no terminal geometry was measured. No fit claim.

## Native holds (separate, not claimed)

Pseudo-localisation, RTL (not applicable to French), rendered/TUI/CLI frames, real refusal execution, consumer wiring of this display key, ledger row, activation and release all remain **NOT RUN / pending**. This is a source-only linguistic and mechanical pass; it is not a frame clearance and not an English-free claim for any page.

## Scope

One key, read-only: no repository, catalogue, caller, ledger, generated or baseline edit; no process, shell, Git, test, formatter, build, install or CI. Earlier SDK, policy, console, hub-defaults and both TUI chrome R1/R2 bundles are untouched.
