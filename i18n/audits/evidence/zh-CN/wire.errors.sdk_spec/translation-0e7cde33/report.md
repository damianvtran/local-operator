# zh-CN wire.errors.sdk_spec — staged translation packet

21/21 keys translated from pinned commit 0e7cde33ca8008d561e74f4aed774aa5f76ff7d5. Translation only, NOT audited or activated.

## Contract
Existing SDK exception producers explicitly render English. This packet does not change their legacy ValueError/args/str contract or claim any UI/SDK localization at runtime. English/context pinning is retained; implementation needs separate contract handling. No desktop/mobile sink was found by the named architect, not independently exhaustively assessed by this translator.

## Evidence
Mechanical command: `node --max-old-space-size=256 validate.mjs` (use bundle absolute path). Exit 0. See mechanical.log and mechanical-validation.json for actual execution and eight rejected negative controls. Existing parser loaded through installed UI dependencies; no dependency installation. The opaque-argument smoke uses parser ASTs and is not production renderer proof.

Rendered native fit, pseudo, RTL and production renderer: NOT RUN. No independent audit performed.

See translator-record.json for actual completed-turn model provenance, glossary proposals, semantic decisions and all immutable pinned input hashes. See spec.callers.txt and sdk.callers.txt for pinned producer excerpts. manifest.json hashes every bundle companion except itself.
