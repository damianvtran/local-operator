# Audit report — `wire.errors.sdk_startup` · fr · context-delta confirmation

- **Audit ID:** `frsdkstartup-ctxdelta-604100`
- **Mode:** context-delta confirmation only. The English source and the French target are
  byte-identical to the inputs two prior independent FR full audits approved; only the English
  context sidecar changed (PR #3378, **unmerged**; no shipped claim).
- **Auditor:** `i18n-auditor`, session `604100bf1795`, reader `fr-reader1`.
  Requested route `radient/auto` (Radient Auto). The session header names `radient/auto`; no
  served-usage extract was produced in this run, so the **underlying served model is UNKNOWN**
  and no fallback is claimed or denied (UNKNOWN, not NONE).
- **Independence:** translator session `7b2ba4ed501d` (cited by the prior records); coder
  `eae0961e4e78` (as prior records state, not re-verified). Nothing here was authored or edited
  by this auditor. Only the **prior** FR full-audit records were read, for compatibility; no
  parallel reader report from this round was read.
- **Date:** 2026-11-03.

## Hashes (computed in-session over the frozen bytes)

| input | sha256 |
|---|---|
| en (`sdk-startup-followon/en.json`) | `5390ecb2413e0978c34541a9a0db6c2a004ae86446d1c0fa0d2e697e4f9ca6bd` |
| context (`…/context.json`) | `82f193fd3c3e84526cbfaa0498fc47b776dd84df128cea3442df6d13369bb13e` |
| fr (`…/fr.json`) | `82b61a755a8580e7cb4e4f86ae64a428fc83232fac92b22fa72f09d77b5f781d` |
| style guide `i18n/style-guides/fr.md` | `5315e1ac10c2ce57aac631ba653d08de680ee70c9a022ed3a8345ce2f9902ee7` |
| glossary `i18n/glossary.md` | `0eb34437a8c3d6f564f278ef957a603b1d9f51bb07d5ddae5ea8f37deba962f8` |
| old context (superseded) | `ee0bbde5e99de22c04368e6bbfedf913d758ef6ba5075984ddf7d795b1ee7822` |

The context hash matches the binding (`82f193fd…`), **not** the inventory-chat typo — and both
EN and FR are byte-identical to the blobs the prior audits hashed. The prior context text was read
from the retained packet copy (`…/candidates/zh-CN/context.json`, same `ee0bbde5` digest).

## What changed in the context, and what did not

Only **one** of four notes changed: `attach_hosting_unconfigured` (role and tone unchanged). The removed sentence said the child
namespace is “English-only … falls back under RFC §6 per-CARRIED policy; future translated
consumers still owe the RFC migration boundary”. The replacement says the refusal frame renders in
the **resolved locale, never an English pin**; that when the cause is the product's own
not-configured hosting error (exact type **and** published sentence) the caller substitutes the
**separately localized** cause sentence, so this frame must not carry English cause wording of its
own; that any other producer's reason stays opaque and verbatim; and that a locale which does not
carry the namespace falls back to English per-namespace.

Byte check: `English-only` occurs in the old blob and **nowhere** in the new one. The claim was
confirmed against the producer, read only, `local_operator/sdk.py:545-650` at the PR head:
`startup_refusal_locale()` resolves the scoped config language, `startup_refusal_text()` renders
with `locale=…`, and `startup_cause_text()` substitutes the localized cause only for exact type
`HostingNotConfiguredError` whose `str()` equals the catalogued English sentence of
`wire.errors.hosting.not_configured`; everything else returns `str(error)` untouched.

## Does the French copy survive the new context? (the actual question)

**Yes — no incompatibility found.** The new contract is satisfiable with the retained bytes:

1. The frame carries no English cause wording of its own (no English prose at all; only
   `runtime` and the literal technical tokens).
2. With the owned cause, the composed French refusal is complete and reads end to end:
   « aucun runtime actif ne sert 'x', et cette racine ne peut en démarrer un : **La plateforme
   d’hébergement n’est pas configurée.** Configurez la plateforme d’hébergement de cette racine
   (son config.yml), puis démarrez l’identifiant avec spawn_session()/deliver() et
   rattachez-vous dès qu’un runtime existe. »
   The cause ends in a full stop, so the capitalised remedy that follows starts a real sentence.
3. Terminology now meets across the boundary: the frame's « plateforme d’hébergement » is exactly
   the French wording of the cause key it will sit beside
   (`fr/wire.errors.hosting.json`: « La plateforme d’hébergement n’est pas configurée. » —
   sha `83432e2a…`). The two keys were never adjacent in French before this change; they are now,
   and they agree.
4. Foreign opaque causes keep passing through verbatim, which the frame does not disturb.

Terminology re-checked against the guide and the shipped corpus: `runtime` stays English (fr guide,
wave-1 recorded addition) and is uniform across the fr corpus; « racine » for *root* matches
`cli.browser.config_root_line`, `cli.serve_help.host`, `cli.sessions_help.reclaim_description`;
« rattacher » matches `wire.exec.session_pointer`; `engage` → « démarrer » matches
`cli.sessions_help.move_engage_on_arrival` (no glossary row for *engage*; corpus is the reference).

## Mechanical verification (pure stdlib; no repo validator was run)

| check | result |
|---|---|
| JSON valid, 4 keys, key set == source | PASS (no keys added or dropped) |
| placeholders: names + types vs source | PASS — k1 `{resume},{error}`; k2 `{resume}`; k3 `{profile}`; k4 `{spec_tools},{policy_tools}` |
| ICU argument kinds present | none — all four are plain-argument messages; **no plural/select**, so CLDR categories (and fr's `one`/`many`/`other`) are **not applicable**, confirmed by parse rather than assumed |
| brace balance / escaped-brace leakage | PASS (0 delta; no braces outside placeholders) |
| ICU apostrophe escaping | PASS — `mode=''own''` preserved (2 doubled apostrophes, exactly as source); none added elsewhere |
| technical tokens vs source | PASS — diff empty both directions: `config.yml`, `spawn_session()`, `deliver()`, `mode='own'`, `spec.tools`, `ApprovalPolicy.declared`, `runtime` |
| typography | PASS — one U+202F before each `:`/`;`, U+2019 apostrophes only, no ASCII apostrophes in prose |
| register | PASS — *vous* + imperative; no « Veuillez »; no Title Case |
| width hints | **none exist** — all four sidecar entries carry `role`/`tone`/`note` only, no `maxCells`/`maxChars`; no budget invented. Stdlib approximation (not `rich.cells`, not rendered): 210→262 (1.25), 173→202 (1.17), 34→43 (1.26), 123→160 (1.30) cells |
| RTL | not applicable to `fr` |
| pseudo-localisation | NOT RUN |

## Findings

**0 BLOCKER, 0 MAJOR, 0 new MINOR, 0 NIT.** One prior MINOR is carried, unchanged, still open and
non-blocking; two observations are recorded and neither asks for a change.

- **F1 — MINOR (carried from prior FR reader 2, `35aac0a677bc`; not re-raised, not re-litigated)**
  `attach_hosting_unconfigured`, `attach_runtime_missing`. « … et rattachez-vous dès qu’un runtime
  existe. » — bare reflexive *se rattacher* without a complement; the corpus supplies one
  (`wire.exec.session_pointer`: « rattacher ou reprendre »). Suggested: « … puis rattachez-vous
  **à la session** dès qu’un runtime existe. » Meaning and action survive → MINOR. Unrelated to
  the context delta (`affects_context_delta: false`).
- **O1 — inherited observation (both prior readers).** The frame resumes with a capitalised
  sentence after `{error}`. With the owned cause that now composes cleanly in French; a foreign
  cause without terminal punctuation still runs into the remedy (« … : [Errno 61] Connection
  refused Configurez … »). Byte-identical shape in English; **not fixable from a catalogue file**
  and not a translation defect.
- **O2 — new observation, no change requested.** Only now that the cause renders French does the
  composed sentence show « la plateforme d’hébergement » twice. Kept deliberately: the cause names
  the condition, the frame names the remedy, and the frame must stay self-contained because a
  foreign opaque cause supplies no antecedent — a pronoun variant (« Configurez-la pour cette
  racine ») would dangle on that path.

## Not run (explicit gaps — nothing below is cleared by this pass)

- Repository validator / `scripts.i18n.check`, Node generation and typecheck — **NOT RUN**, by the
  task's source-only constraint; source inspection and stdlib hashing/parsing only. This is not a
  green mechanical result from the repo's own tooling.
- Rendered frame, SDK guard execution, real configured root — **NOT RUN**; no runtime started.
- Rendered UI/terminal fit and overflow — **NOT RUN**; the width figures are an approximation
  against no budget. No fit, no overflow and no whole-frame/runtime clearance is claimed.
- Pseudo-localisation — **NOT RUN**.
- Ledger append (`i18n/ledger/fr.json`) and aggregate regeneration — **NOT DONE** here; the
  coordinator batches ledger and PR writes.
- Tracked evidence commit under
  `i18n/audits/evidence/fr/wire.errors.sdk_startup/frsdkstartup-ctxdelta-604100/` — **NOT
  WRITTEN**; this session-scratch original is the record of this run and no evidence is claimed
  on `main`.

## Verdict

**`passed` — context delta confirmed compatible with the retained French copy; 0 unresolved
BLOCKER/MAJOR; prior approved native judgments preserved.** The old French F1 (false English-only
context note) is closed by the new note; the new note introduces no incompatibility with the
translated frame.
