# PR 2329 — why one unpassed namespace drops a whole locale, and the minimal fix

Author: architect, `anthropic/claude-sonnet-5-5` (no fallback model used). Read-only: no code, validator, ledger or PR changes.
Worktree `coder-2329-convergence-20261014`, HEAD `57bcb9398` (parent `209d18ed4`). All line numbers are at HEAD unless a commit is named.

## 1. What happened (derived here)

- `209d18ed4` staged 15 locale files (5 locales × `relay.login`, `relay.session` and `relay.shell`) and 15 ledger rows with `status: translated`. The auditor result is `native_pass_render_pending`, and the row note says "Not passed, not eligible for activation/release".
- I extracted that tree and ran its own `_shipped_locales()`. It returned `['en']`.
  - Ledger non-passed rows at that commit are exactly those 15.
  - The cause is `scripts/i18n/generate.py:269-286`. `carried = catalogues.namespaces(locale)` lists every `*.json` in the locale directory. The locale ships only if `all(...)` carried namespaces have a `passed` row with a fresh hash.
  - Three `translated` namespaces therefore removed `fr/es/ru/vi/hi` entirely. That includes their already-audited `wire.*` and `tui.*` catalogues, so Hindi `wire.settings` fell back to English.
  - At HEAD the same function returns the six locales again, because the candidates are no longer carried (`57bcb9398` deletes the 15 files and 15 rows).
- A separate fact came out of the spike: if the generator had merely ignored the pending rows and listed `fr` as shipped while `fr/relay.shell.json` was still on disk, the runtime would serve and render it.
  - I hand-set `shipped.json` to `["en","fr"]` in a scratch extract of `209d18ed4`.
  - `messages.render("relay.shell.list.connecting", locale="fr")` returned `connexion…`.
  - `catalogues.load_catalogue("fr","relay.shell")` succeeded and `catalogue_sha256` returned `7c33c08b…`.
  - So "ignore pending rows" alone would publish unaudited copy.

## 2. Intentional contract versus mechanism limitation

**Intentional, documented, tested contract (not a bug):**

- `docs/design/i18n.md:580-584`: a locale ships iff every namespace it carries is `passed` with a fresh hash. Stale means not shipped.
- `generate.py:228-230` docstring: "a locale with any of ITS OWN namespaces missing or stale is not shipped"; `:240-241`: "a stale member still un-ships the locale".
- `tests/unit/i18n/test_generate.py:332` (`test_a_carried_namespace_without_a_ledger_entry_un_ships`): "completeness is still the gate's job".
- `docs/I18N.md:35-37`: "Blocked/unaudited carried namespaces cannot ship or activate".
- `docs/design/i18n.md:554-558`: a hold is rows and files together; never drop only the failed namespace.

**What the 2026-10-11 amendment did and did not change:** it narrowed *coverage*. A namespace the locale does not carry no longer un-ships it (`i18n.md:589-603`). It did not narrow *eligibility*: a carried-but-unpassed namespace still un-ships the whole locale. The amendment's own motivation was that a shipped capability "must not regress because a namespace was extracted" (`i18n.md:597-599`). That motivation applies equally to a carried namespace that is merely unaudited, but no document addresses that case.

**Verdict:** the all-carried gate is the current contract. The collateral loss (unrelated audited namespaces falling to English) is a consequence of that contract's granularity, which is locale-wide, whereas the fallback runtime is already per-namespace (`messages.py:113-123`, `text.py:105-114`). I found no document that argues for locale-wide drop beyond "fail closed". I would not call it a contract violation, and I would not call it a proven bug either. It is a product-mechanism limitation, and the missing piece is a stated per-namespace *eligibility* rule. What would settle whether it is intended: an operator statement on whether an unaudited carried namespace should take down its locale's audited namespaces. I did not find that statement.

## 3. Why "just ignore pending rows" is not minimal

Three consumers read the carried tree directly and none checks per-namespace status:

- `routes/i18n.py:46-52` and `mobile/daemon.py:4677-4690` gate only on `is_shipped(locale)` (plus `eligible_locale`, and `wire.` for the daemon), then serve any file in that locale directory with its ETag.
- `messages._lookup` (`messages.py:142-147`) and `text._sources` (`text.py:104-114`) load whatever is on disk for a shipped locale. `text.py:107-110` says explicitly that "production eligibility belongs at the public entrypoints, not here".
- Packaging: `pyproject.toml:247` globs `i18n/catalogues/*/*.json`. Every file in the tree goes into the wheel, so `docs/design/i18n.md:559-562` ("keep held bytes out of the artifact") can only be met by the file not being in that tree, or by a build step that filters. No build step does today.

Eligibility is therefore a property of file *presence* in the package tree, not of the ledger row.

## 4. Options

| | Change | Pros | Cons |
|---|---|---|---|
| A. Stage candidates outside active inputs; promote atomically when `passed` (what the corrected PR now does) | None to code, validator or gate. Candidates live on a held ref or branch. Promotion is one commit: files + `passed` rows + regenerated `shipped.json`/bundle. | Smallest. Meets non-distribution because the bytes are not in the tree, wheel, route or bundle. Consistent with `i18n.md:554-558`. Needs no RFC amendment. | Candidates are not validated by `check.py` while out of tree (it only checks present files, `check.py:487-499`). Not a generic per-namespace policy: a *future* stale existing namespace still un-ships its locale. |
| B. Per-(locale, ns) eligibility in code | `shipped.json` schema 2 listing eligible pairs, plus consumer checks in routes, daemon, `messages._lookup`, `text._sources`, and `OWNED_SPA_NAMESPACES` bundling. | Fixes collateral loss in general, including staleness. | Touches ~6 files, and `shipped.json` is a security-sensitive generated input. Does not by itself keep bytes out of the wheel. Needs an operator decision on whether a stale existing namespace may degrade silently (this is the part that changes §6). Needs an RFC amendment to §6 and `docs/I18N.md`. |
| C. Option A as the rule, plus a thin guard | A staged-candidate directory outside `local_operator/` (for example `i18n/candidates/<locale>/`), an optional `check.py` pass over it for parity, and a wheel/sdist contents negative control. | Keeps A's property and removes its two cons. | Small extra code and CI. Not needed to unblock PR 2329. |

## 5. Recommendation

- **For PR 2329: Option A, which is what `57bcb9398` already does.** It is the truly minimal fix. It uses the existing gate unchanged: an uncarried namespace falls back to English for that namespace only (`messages.py:120-122`), the locale's carried passed namespaces stay shipped, and the candidate bytes are not in the wheel, route or bundle. It needs no RFC or security decision, because `_shipped_locales` still means what `i18n.md:580` says. It does *not* weaken the hold: the candidates were never `passed`, and they are removed rows and files together (`57bcb9398` removes both), which is what `i18n.md:554-556` requires. State in the PR that the staging is by design and where the held copy lives (the ref or branch, and the aggregated receipt URL already in the row note).
- **Do not do B inside this PR.** It changes the §6 gate and the eligibility input for every client. It needs an explicit amendment in the style of the 2026-10-11 note, and an operator decision on whether stale existing namespaces should degrade per namespace or per locale.
- **If the team wants the general rule later, the minimal policy is C, and no gate change is needed.** "A `(locale, namespace)` pair is eligible iff its file is in the active catalogue tree, and a file enters the tree only in the commit that adds its fresh `passed` ledger row." This is the existing gate, with staging made a documented place. Document it as one paragraph in `docs/design/i18n.md` §6 and `docs/I18N.md`.

## 6. Exact touch list and negative controls (for the later change, C; none for PR 2329 itself)

Files:

- `docs/design/i18n.md` §6 and `docs/I18N.md`: the staging paragraph.
- `scripts/i18n/check.py`: optional candidates parity pass. Key parity and arguments only; it does not make a candidate eligible.
- A packaging check, either in `tests/unit/i18n/` or CI: wheel/sdist listing.
- `tests/unit/i18n/test_hold_split.py`: extend the existing pins.
- Not touched: `generate.py::_shipped_locales`, `resolve.py`, routes, daemon, `i18n/ledger.json` (no rows for unpassed pairs).

Negative controls (each must fail when the control is broken):

1. **Pending new namespace, English fallback only.** With a staged `fr/relay.shell` outside the tree, `messages.render(key, locale="fr")` equals the English text, and `_shipped_locales()` still contains `fr`. Mutation: copy the file into the tree without a `passed` row, and `_shipped_locales()` must drop `fr` (this is the existing test at `test_generate.py:332`).
2. **Existing namespaces retained.** `fr/wire.settings` and `hi/wire.settings` render translated text. Mutation: add an unpassed carried namespace, and the same assertion fails. This is the collateral-loss regression test, and it currently does not exist.
3. **Blocked locale held.** `zh-CN` is absent from `shipped.json` and has no catalogue directory (`test_hold_split.py:20-35`). Keep those assertions.
4. **No unpassed content served or bundled.**
   - `GET /v1/i18n/catalogues/fr/relay.shell` returns 404 and `catalogue_sha256("fr","relay.shell")` raises `CatalogueNotFound`.
   - `bundled.gen.ts` contains only `en` copy for the owned namespaces.
   - A wheel or sdist listing contains no candidate path. Run it against a real build, because the glob at `pyproject.toml:247` is what would otherwise pick the files up.
5. **Promotion atomicity.** A candidate file in the tree with `status: translated` must fail `generate.py --check`. That is the existing behaviour, and the test pins it.

## 7. Risks to watch

- **Collateral drop recurring.** Any future `translated` or stale carried namespace will again drop a whole locale silently, since `shipped.json` is just a list. Watch `shipped.json` diffs in every i18n PR; the byte-identity check in `generate.py --check` already catches any change.
- **Staged bytes are unvalidated** until promotion. A hash or placeholder drift against a moving `en` is found late. The promotion commit must re-hash against current `en` (`i18n.md:122-124`).
- **Held-ref drift.** Candidates live only on the held ref or branch plus the PR comment. Record the SHA, or the candidates can be lost with the branch.
- **SPA language contract.** `uiLocaleFor` already requires every owned bundle to be complete for a locale (`runtime.ts`, `i18n.md` "Language contract"). When the relay candidates are promoted, all three `relay.*` namespaces must go together for each locale, or the SPA falls back to the whole English bundle.
- **Spike limits.** The spike ran on an extracted tree, not the running daemon. I did not execute the HTTP route; the route behaviour above is from reading the code at `routes/i18n.py:46-52`.
