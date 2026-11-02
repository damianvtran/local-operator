# French second independent audit — PR #3222

Verdict: **PASSED** — all 3 keys; zero BLOCKER, MAJOR, MINOR or NIT findings.

Auditor: i18n-auditor (host-fr-audit-b), session `83a2be80f6af`, actual model `openai/gpt-6.1-sol`, 2026-10-31 UTC. Quota fallback from the requested Sonnet, via an Opus switch; the latest switch selected GPT before the substantive audit. Translator session `829abca97a25` is distinct, although its actual model is also GPT. No first-auditor material was inspected. No MT service was used; no translation, ledger, or repository file was edited.

## Inputs and scope

Source revision: `9c979a03d551020ce6ff1b16c8066aa69ce7bb2a`.
Read French style guide and glossary, then committed English/context, then candidate and sibling translator provenance/validator log. Independently verified the supplied English/context digests and target bytes. This is a fresh second semantic audit, not adoption of the translator's mechanical result.

## Per-key assessment

### wire.errors.session_host.unsupported — PASS
Current: « Cet ordinateur ne peut pas exécuter l’hôte de sessions en arrière-plan. La session n’a donc pas été démarrée. Utilisez un ordinateur doté d’un gestionnaire de services en arrière-plan compatible. »
Meaning is preserved: this computer cannot run the host, the session did not start, and another supported computer is required. No promise of retries or reconnection. “Hôte de sessions en arrière-plan” is a defensible technical rendering of the host role, consistently distinguished from “gestionnaire de services”. Splitting the causal English sentence improves readability without dropping its consequence. Formal vous imperative and typographic apostrophes are correct. No correction required.

### wire.errors.session_host.unavailable — PASS
Current: « L’hôte de sessions en arrière-plan n’a pas répondu. La session n’a donc pas été démarrée. Attendez un instant avant de réessayer. »
Preserves the absent answer and explicit failure to start. “Attendez … avant de réessayer” gives the user the retry action; it neither asserts an automatic retry nor a reconnecting session. “N’a pas répondu” does not invent a permanent platform limitation. Native, concise professional error copy. No correction required.

### wire.errors.session_host.refused — PASS
Current: « L’hôte de sessions en arrière-plan a refusé de démarrer cette session. La session n’a pas été démarrée. Consultez logs/session-host.log dans votre dossier de configuration Local Operator avant de réessayer. »
Preserves refusal, explicit non-start, log inspection, and inspect-before-retry sequencing. The literal path and product name are byte-identical. “Consultez” is idiomatic for inspecting logs and does not suggest altering configuration. The repetition is justified safety emphasis inherited from English, not gratuitous padding. No correction required.

## Mechanical execution

Independently reran the repository validator in the translator's existing validation tree, read-only, after verifying its 116 Python source files against the committed revision and checking staged English/context/target byte identity. Used isolated HOME/config, stripped CMUX_*, and disabled bytecode writes. Actual stdout: `check.py: catalogue checks: 0 problem(s)`; exit 0. See `audit.log` for the actual command and output.

An initial audit preparation call timed out while individually checking committed source files and produced no log or completed validator result. A process check found no surviving matching validator/git process. Recovery used one in-memory git archive for source-byte comparison, with no extracted copytree or dependency installation; the subsequent validator run above completed. No success is claimed for the interrupted call.

Independent checks: valid JSON, exactly 3 matching keys in matching order; no placeholders, ICU plural/select forms, markup/Rich tags or backtick spans in source or target. The required log path and Local Operator occurrences match. NFC text and no Unicode Cf controls. No destructive/permission polarity changes.

## Fit and limitations

Source sidecars set no maxCells/maxChars budget. French values contain 195, 129 and 206 Unicode characters respectively (English 145, 110 and 174); these are character counts, not rendered terminal-cell geometry. No numerical overflow finding can be justified from an absent constraint. No real rendered frame, truncation, wrapping or fit proof was captured. French is LTR; RTL-specific review is not applicable. This is a semantic/mechanical approval only, not runtime/render QA or release approval.

## Record handoff

Proposed future tracked bundle: `i18n/audits/evidence/fr/wire.errors.session_host/h3222b/`.
Artifacts currently exist in auditor session scratch only; manager must retain them at that canonical root before claiming tracked evidence. Shared ledger was not changed. RFC §6 and docs/I18N.md specify the ledger JSON row; no audit-ledger CLI is named there, so `ledger-row.proposal.json` is a proposal, not a claimed CLI execution. `audit.log` is actual execution output; this report's semantic reasoning is audit notes, not an execution log.

## Style-source reconciliation

The initially read worktree French guide hashes to `ae1519ab7617d19aa826918d607de1d018595094166edbc18aa58f6a8f0a4226`, unlike the translator’s committed guide `5315e1ac10c2ce57aac631ba653d08de680ee70c9a022ed3a8345ce2f9902ee7`. Investigated with `git diff 9c979a03d551020ce6ff1b16c8066aa69ce7bb2a -- i18n/style-guides/fr.md` and the committed blob: the only difference removes a relay.chrome-specific width-exception paragraph. Read that paragraph in the diff; it does not affect any of these three strings. Record retains both hashes; no discrepancy is hidden or treated as a translation defect.
