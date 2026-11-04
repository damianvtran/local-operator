# vi audit — wire.errors.hosting (reader 2, independent)

- Key: `wire.errors.hosting.not_configured` (1 key). Staged candidate; the English namespace is uncommitted and absent at base `87ca846245b2`; not shipped, not activated.
- EN: "Hosting platform is not configured."
- VI: «Chưa cấu hình nhà cung cấp nào.»
- **Verdict: passed** — 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT. One non-gating advisory outside this key (A1).
- Native/rendered gates are **separate holds**, not run here.

## Inputs (hashes computed here, stdlib hashlib)
- en `458d8f8be304428ccb6a328ce22e5a86515e9f44f869f0138a5cc6dfde964b63`
- context `ed0ae50fa465a010a2907bc680771dae80bc716d3a186b0af37e31e0d7063cf9`
- vi target `4a9d101e5c8ddba0c55f1428427f26174b0e10055701d896d0ce68cc1312c647`
All three equal the assigned values.

## Mechanical (stdlib json/unicodedata; no validator, formatter or project import)
- Key parity: 1 key, same as EN; JSON parses with duplicate-key rejection; UTF-8, trailing newline.
- NFC; no Cf/bidi controls; no combining marks.
- Placeholders/ICU/CLDR: **N/A** — EN and VI have zero arguments and no plural/select, so there are no placeholder types or CLDR arms to check.
- Capital first letter and terminal period kept, as the context requires; no markup, code or identifiers.
- No English fallback: the value is fully Vietnamese and not a copy of the English.

## Accuracy and context
- The context says the preflight resolved **no provider at all**, the clause is spliced after a colon and must read as a complete sentence, and it must name the condition without the remedy or dynamic detail.
- «Chưa cấu hình nhà cung cấp nào.» = "no provider has been configured (yet)". «chưa» is the natural negation for a configuration state; «nào» gives "none at all", matching the context. It names the owned condition, not an opaque external failure. No remedy, cause or detail added; no frame wording pulled in.
- Polarity correct; nothing omitted. The English "platform" is not dropped in meaning: in this product the hosting setting *is* the provider choice.

## Terminology and corpus
- vi corpus renders the same concept as «nhà cung cấp»: `wire.settings.hosting.label` «Nhà cung cấp mặc định», `.help` «Nhà cung cấp cho các cuộc trò chuyện mới», `cli.global_help.hosting`, `cli.global_help.propagated_hosting`, `cli.exec_worker_help.hosting`.
- `wire.errors.desktop.aida_no_provider` already states this exact condition as «Chưa cấu hình nhà cung cấp nào, …». The candidate reuses that clause, so the same condition reads the same everywhere.
- Using a provider term where other locales may say "platform" is justified by this corpus, not by cross-locale uniformity. A literal «nền tảng lưu trữ» would read as storage/web hosting and would be wrong.

## Idiom and register
- Subjectless statement is native Vietnamese for state messages; no «bạn», no "!", sentence case, no Title Case. Consistent with the vi style guide.

## Consumer splice (string substitution only — NOT a render)
Substituting the value into the shipped vi frame `wire.errors.sdk_startup.attach_hosting_unconfigured` gives:
«…không thể khởi động một runtime: Chưa cấu hình nhà cung cấp nào. Hãy cấu hình hosting cho thư mục gốc này…»
The clause reads as a complete sentence after the colon, and the frame supplies the remedy.

## Advisory (non-gating, outside this key)
- **A1** — The shipped vi frame's remedy keeps English «hosting» («Hãy cấu hình hosting…»), while the cause clause and the rest of the vi corpus say «nhà cung cấp». In the spliced sentence the condition and the remedy name the same thing two ways. This is pre-existing frame copy in another namespace; it does not gate this key. Suggested for that namespace's owner: «Hãy cấu hình nhà cung cấp cho thư mục gốc này…» (or keep `hosting` only if it is meant as the literal config key, then mark it as code).

## Contract
- `session_factory.HostingNotConfiguredError.__str__` stays English by contract; this key is display-only. Not touched, no permission inferred.

## Separate holds (not claimed)
- Native/rendered TUI or SDK refusal: NOT RUN.
- Pseudo-localization: NOT RUN. RTL: N/A (vi is LTR).
- No width budget in the context; no cell measurement run.
- No whole-surface or "no English anywhere" claim.

## Provenance
- Agent `i18n-auditor`, session `6431174917d4`, reader 2. Requested model: UNKNOWN (the brief names none). Served model: UNKNOWN — this turn's session state exposes no model header; no fallback is claimed for this turn.
- Independent of translator `15986c0af719` and code author `94dcef624dec`. I read the source, context, author record, style guide, glossary and vi corpus. I did not read the other reader's verdict.
- Method: read/glob/write and stdlib eval (json, hashlib, unicodedata). No process, test, formatter, project import, git, repo/target/ledger edit.
