# French second-reader audit: wire.settings.hub_defaults

**PASSED — independent staged linguistic/mechanical verdict for 9/9 keys.** No BLOCKER, MAJOR, MINOR or NIT findings. This is a separate namespace decision, not blanket approval of the three contributions or interface-wide completeness.

Pinned source: `0e7cde33ca8008d561e74f4aed774aa5f76ff7d5`. Auditor `i18n-auditor`, session `35aac0a677bc`; translator session `7b2ba4ed501d`; coder actual session `eae0961e4e78` (not job `196c3a94b346`). No translator/coder role performed by this reader and no other new reader report inspected.

**Actual served model:** `openai/gpt-6.1-sol`, from own completed persisted usage for probe authorship/execution and source-inventory derivation in `provenance.json`. Known own route notices selected Opus then OpenAI after primary Sonnet quota failures; route notices do not establish completed Opus usage. No requested-header or no-fallback inference.

## Input identity

- en: `8720b5719479b638eab24b081f76f97c44ce8d8554cb1b8280f6980182b6740c`
- context: `a337e7394f3856476044f9accdb8daeb7a864a9a8ecfe0c579ec97ec646712ba`
- fr: `9733a2653109b4e5b30964d0bbc9fb5974fe799bb086ce8d21354517d1936940`

Source and context packet bytes were compared directly with pinned git blobs. All original translator-packet manifest entries were rehashed; style/glossary snapshots are byte-identical to the French guides already read in this session. Original packet bytes remain untouched.

## Native assessment of every key

### `agents.help`
Current: Non : gèle toutes les MAJ agents/compétences. Oui : abonne les nouveaux.
Assessment: Non gèle toutes les mises à jour, not merely new links; Oui subscribes new ones. MAJ is the conventional French abbreviation for mises à jour, not an English fallback. The adjacent full label supplies liens; the terse nouveaux has an intelligible masculine antecedent. Native-acceptable at exactly 72/72 Rich cells.

### `agents.label`
Current: Mises à jour auto des liens d’agents et de compétences
Assessment: Full French expansion mises à jour auto des liens; both agents and compétences included, consistent with glossary compétence.

### `agents.off.description`
Current: suspendre les mises à jour de tous les liens d’agents et de compétences
Assessment: Tous les liens preserves the ALL-existing-agent/skill update ceiling; suspendre is reversible pause, not deletion/unsubscription.

### `agents.on.description`
Current: abonner les nouveaux liens d’agents et de compétences
Assessment: Nouveaux explicitly limits creation-time subscription; does not claim retroactive subscription of every existing link.

### `model_default.note`
Current: Les merges du Hub utilisent votre modèle par défaut actuel. L’ancien choix de modèle propre au Hub est ignoré.
Assessment: Current default model and ignored former Hub override both retained. merges is an explicit kept-English Git/Hub operation under the French guide/glossary, not silent untranslated prose.

### `teams.help`
Current: Par défaut pour les nouveaux liens ; les anciens gardent leurs règles.
Assessment: New-link default contrasted with existing links retaining their own rules; no master-switch claim. Team scope is supplied by its setting label/context. 70/72 cells.

### `teams.label`
Current: Abonner les nouveaux liens d’équipes
Assessment: Abonner les nouveaux liens d’équipes is a natural creation-default action label, not a blanket enable/disable switch.

### `teams.off.description`
Current: suspendre les mises à jour entrantes des nouveaux liens d’équipes
Assessment: Incoming updates for new team links preserved; does not say old team links are paused.

### `teams.on.description`
Current: abonner les nouveaux liens d’équipes
Assessment: Subscription applies to new team links, matching source.

## Actual scoped mechanical execution

The existing repository `scripts.i18n.check.check_catalogues()` ran against this namespace alone, with its retained English/French catalogue and original English context. Independently authored checks validated duplicate-free JSON, exact key/order parity, names/types via the core ICU parser, numeric literals, technical tokens, no `Cf`, and all supplied Rich-cell limits. Core `render_message` exercised every string; opaque syntax-looking arguments were inserted as data, not reparsed. These are parser/formatter calls, not SDK guard or UI execution.

Actual exit: **0**, validator problems: **0**; rejected controls: **3**. Full stdout: `mechanical.json`; stderr: `mechanical.stderr.log` (empty). The fixture was restored and the clean validator rechecked. No plural/select exists, so CLDR arms are **not applicable**.

Invocation (each namespace root has its own retained `verify.py`):

```bash
for name in ${!CMUX_@}; do unset "$name"; done
cd ~/local-operator
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B <this-audit-root>/verify.py
```

The three scoped invocations ran serially in one tool call bounded to **256 MiB / 90 seconds**; all three exited 0. This is not a broad suite or scanner.

### Abbreviation and width judgment

**MAJ is approved here as native French abbreviation for mises à jour**, only in the compact 72-cell help; the adjacent label expands it. `Non` suspends all agent/skill updates, while `Oui` subscribes new links. The separate full descriptions remove any ambiguity. This is not an English fallback or a new machine identifier. `merge`/`merges` is an explicitly allowed kept-English Git/Hub term, and `compétence` matches the glossary.

Actual bounded widths: agents help **72/72**, teams help **70/72** Rich cells. Other keys have no numeric sidecar width cap. Exact-limit static pass is not a rendered-widget guarantee.

## Explicit remaining proof and disposition

- **NOT RUN:** native/app fit, screenshots, live language switch, pseudo rendering, SDK guards/settings requests or design review. RTL visual rendering is not applicable to French and was not executed; absence of hidden controls was checked.
- Static Rich widths and source/ICU inspection are **not** native layout or real-path integration proof.
- No source/target/repository/ledger edits, generation, activation, PR/comment, build/install/scanner/full suite. No global completeness, release eligibility or English-free-interface claim.
- Original per-namespace bundle is retained read-only in session scratch for coordinator promotion into canonical tracked `i18n/audits/evidence/fr/<namespace>/...`; it is not claimed on main. Previous policy bundles/addendum and earlier translation packets are untouched.

