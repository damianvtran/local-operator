# VI audit vi1r — `wire.errors.sdk_startup` (one changed remedy)

- **Audit id:** vi1r · **Date:** 2026-11-03 · **Reader:** reader1 (first independent pass)
- **Locale / namespace:** `vi` / `wire.errors.sdk_startup`
- **Scope:** the single changed key `wire.errors.sdk_startup.attach_hosting_unconfigured` plus its three direct neighbours; **not** the whole corpus, not the other 3 values' content.
- **Verdict: PASSED** — 0 BLOCKER, 0 MAJOR. Two non-gating NITs.
- **Not run (holds):** native fit / width rendering · ICU MessageFormat execution (namespace has no plural/select) · pseudo-locale · RTL · ledger row write (out of this session's scope). No CLI validator was run; every mechanical claim below comes from the stdout in `audit.log`.

## The delta

Old (vi, in worktree): `Hãy cấu hình hosting cho thư mục gốc này (config.yml của nó), rồi …`
New (target): `Hãy cấu hình nhà cung cấp cho thư mục gốc này (config.yml của nó), rồi …`

Proof it is exactly that: `old_vi` with `Hãy cấu hình hosting` → `Hãy cấu hình nhà cung cấp` reconstructs the new file byte-for-byte; the other three keys are value-identical **and** raw-line byte-identical; the old phrase occurs once in old and zero times in new (`audit.log` §3). Input hashes all match the binding (§1).

## Findings

| ID | Class | Key | Current | Problem | Suggested |
|----|-------|-----|---------|---------|-----------|
| vi1r-N1 | NIT (advisory, non-gating) | `wire.errors.sdk_startup.attach_hosting_unconfigured` | `Hãy cấu hình nhà cung cấp cho thư mục gốc này (config.yml của nó)` | `hosting` is also the literal config key (`config.py:750` `"hosting": ""`, documented at `config.py:630` as "AI model hosting provider"), so «nhà cung cấp» no longer points at the key name. Mitigated: `config.yml` is named in the same clause and the field is the only provider field, so the remedy stays actionable. | Keep «nhà cung cấp». Do **not** add `hosting` back as a gloss — that re-introduces English prose against the operator's rule. |
| vi1r-N2 | NIT (scope note, out of delta) | same key | `… runtime … id … attach khi đã có runtime` | English tokens remain in the same string after this change. Classification: `runtime` is a recorded kept-English call (style guide, wave-2); `id` is an identifier; `attach`/`deliver()` are SDK method vocabulary. None is human prose. | No action in this delta; recorded so the "no English anywhere" sweep knows what is deliberately left. |

No MINOR findings. I did not pad to fill the cap: the wording is idiomatic and consistent with the house corpus.

## Why «nhà cung cấp» is right (independent reasoning, not deference)

1. **Meaning.** The field is documented as "AI model hosting provider" (`config.py:630`), so "provider" is the accurate concept; «nhà cung cấp» is the settled vi term for *provider* and is used for it across the catalogue (login/logout/credentials, `wire.settings.classification_vendor.label`, `tui.analytics…by_provider`, `wire.errors.desktop.aida_no_provider`).
2. **Direct precedent for this very field.** `cli.exec_worker_help.hosting`: en `Hosting platform override` → vi `Ghi đè nhà cung cấp`. The corpus already renders the `hosting` concept as «nhà cung cấp», so this delta aligns with, rather than diverges from, audited wording.
3. **Register and grammar.** `Hãy` + verb-initial imperative is the established remedy voice (`wire.errors.desktop.aida_no_provider`, `wire.errors.session_host.*`); sentence-initial capital after the `{error}` cause clause mirrors the English and reads naturally. No calque: Vietnamese needs no article before «nhà cung cấp», and «cho thư mục gốc này» keeps the dative the English has.
4. **The "English exception" classification is false for vi.** The context sidecar calls English `hosting` in the remedy banner a recorded exception. Even though the token doubles as a config key name, its occurrence here is human remedy prose, and the operator's no-English requirement plus the corpus precedent above leave no basis for keeping it in vi. Removing it is a correction, not a regression.

## Mechanical checks (all from `audit.log`)

- JSON parses; 4 entries; no duplicate keys; key set identical across en/old/new.
- Placeholders `{resume}` and `{error}` identical in name, order and count to en; braces balanced; no `plural`/`select` anywhere in the namespace, so no CLDR category question arises.
- Literals preserved: `config.yml`, `spawn_session()/deliver()`, `{resume}`, `{error}`, `attach` — all present in the new value and unchanged versus old.
- Orthography: NFC, no leading/trailing space, no double space, no ASCII `...`.
- Width: en 210 cells → new vi 252 cells (+42; the phrase itself is +5 over `hosting` because Vietnamese needs two syllables). The en sidecar carries no `maxCells`/`maxChars` for this key; this is a wrapped paragraph, not a button/label, so no budget is breached. Native/rendered fit is **not** claimed — hold.

## Provenance observations for the manager (not vi findings)

- The context sidecar cites `wire.errors.sdk_startup.cause_hosting_unconfigured`, which is not among this namespace's four keys, so the claim that the caller substitutes a separately localized cause sentence cannot be verified from this file. English-side drift; no effect on the vi delta.
- Sibling locales keep an English-shaped term here (hi «होस्टिंग», ur «ہوسٹنگ»; ru translates «размещение»). The vi divergence is deliberate under the operator's rule; other locales are out of scope and should not be touched from this audit.

## Reproduction

`audit.log` is the verbatim stdout of the verification code run in session `ebf74725105d` (hashes vs binding, delta proof, placeholder/token parity, NFC, cell counts, corpus precedent reads). The code is pure stdlib (`hashlib`, `json`, `unicodedata`, `re`); no project imports, no validator, no network, no MT service.
One extra read-only helper (`python3` via the shell) pulled the English/Vietnamese `cli.exec_worker_help.hosting` pair before §6 re-derived the same values in-kernel; no project code was imported there either. Deliberately not run per the brief: the repo CLI validator, project imports, tests, native/rendered checks and any translation/MT service.
