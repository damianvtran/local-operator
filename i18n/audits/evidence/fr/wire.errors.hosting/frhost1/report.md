# French `wire.errors.hosting` — first independent audit

**Verdict: passed — clean.** 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT. One key: `wire.errors.hosting.not_configured`.

- English: `Hosting platform is not configured.`
- French: `La plateforme d’hébergement n’est pas configurée.`

Auditor `i18n-auditor`, session `c44e0f808e48`, served route `radient/auto` (provider `radient`, model id `auto`, from this session's own completed usage); underlying model **UNKNOWN**, no fallback inferred. Translator `7b2ba4ed501d`; coder `94dcef624dec`. Independent of both; the other reader's verdict was not read.

## Hashes computed here

| Input | Path | SHA-256 |
|---|---|---|
| English | `…/sessions/3ef6295a5545/scratchpad/hosting-source-v1/wire.errors.hosting.json` | `458d8f8be304428ccb6a328ce22e5a86515e9f44f869f0138a5cc6dfde964b63` |
| Context | `…/hosting-source-v1/wire.errors.hosting.context.json` | `ed0ae50fa465a010a2907bc680771dae80bc716d3a186b0af37e31e0d7063cf9` |
| French target | `…/sessions/7b2ba4ed501d/scratchpad/hosting-fr/wire.errors.hosting.json` | `83432e2af01eff5982e4d7a0fca57ebb2223626b0d5713c136d60bb6bec8aa69` |

The author's owned worktree copy of the English source hashes identically to the staged source (`458d…`), so the two English copies are byte-identical. I ran no git command; the namespace's absence at base `87ca846245b2` remains the manager/author's statement and is not re-proved here.

## Native assessment

- **Condition only.** The French states the owned condition — the hosting platform is not configured — and adds no remedy, path, `config.yml`, `spawn_session()/deliver()` or dynamic detail. The surrounding refusal keeps that work, exactly as the sidecar requires.
- **Complete sentence after the colon.** Declarative with its own capital and terminal period. A capital after a colon that introduces a full sentence is correct French, and the sidecar asks to keep the capital.
- **Mood and agreement.** Indicative state, not an instruction; feminine agreement with *plateforme* (*configurée*). No warning tone and no added advice.
- **Terminology.** *plateforme d’hébergement* is the shipped French corpus's own rendering for `--hosting`: `cli.global_help.hosting` "Plateforme d’hébergement à utiliser", `cli.exec_worker_help.hosting` "Plateforme d’hébergement à forcer", `cli.credential_help.credential` "plateformes d’hébergement". The Vietnamese contribution using a provider term is not a rule for French; the actual English source says *hosting platform*, and the French corpus settles it.
- **Punctuation.** Two U+2019 apostrophes, no ASCII apostrophe, no added punctuation, terminal period retained, no edge whitespace.
- **No English fallback.** The value is wholly French; nothing leaks from the legacy English exception text into it.

## Mechanical facts

JSON parses; one key with matching key and order across English, context and target. No placeholders, no braces. **Zero ICU arguments**, so plural/select and CLDR categories are **N/A with reason**. NFC-normalised; **0** `Cf` characters; no manual bidi controls. Capitalised, period-terminated, 49 characters. The sidecar declares no `maxCells`, so no width claim is made and no terminal geometry was measured.

## Advisories and holds

- **ADV-1.** The English sentence is also `session_factory.HostingNotConfiguredError`'s text, which stays English by contract. This audit covers the display key only: it does not claim the exception string is localized, and no caller or API shape changes.
- **ADV-2.** The French deliberately does not repeat the surrounding refusal's remedy; any future frame must keep supplying it.
- **Holds:** native/rendered proof **NOT RUN**; pseudo-localization **NOT RUN**; RTL **N/A** for French; the consumer that splices this clause still needs its own rendered proof. No activation, ledger, generated or baseline change, and no whole-page or product English-free claim.

## Limits

Strict source-only pass: no bash, process, subprocess, tests, scanner, generation, formatter, project imports, app or native run, install or CI. Only stdlib `json`/`hashlib`/`unicodedata` were used, for hashing, parity and character facts. Earlier TUI R1/R2 and all other bundles are untouched.
