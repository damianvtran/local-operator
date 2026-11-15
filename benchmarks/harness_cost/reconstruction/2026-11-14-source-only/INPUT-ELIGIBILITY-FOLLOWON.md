# SOURCE-ONLY input eligibility follow-on — proposed for independent review

Base dd54f8d80fb9; R16 clean terminal per parent. Historical binary-eligibility.json remains unchanged.

No binary/media/weights/encrypted archive fetched; no task/sample/preregistration change. This is operational evidence, not a legal opinion or acquisition authorization.

## Matrix summary

| Source family | Inputs | Local fetch recommendation | Raw/public ZIP | Metadata/hash/recipe |
|---|---:|---|---|---|
| financial-documents | 17 | Specific rights/provenance HOLD | HOLD | Eligible |
| doom-reference | 1 | Specific rights/provenance HOLD | HOLD | Eligible |
| zork-video | 1 | Specific rights/provenance HOLD | HOLD | Eligible |
| git-bundles | 2 | Specific rights/provenance HOLD | HOLD | Eligible |
| pov-reference | 1 | Specific rights/provenance HOLD | HOLD | Eligible |
| opaque-encrypted-archive | 1 | Specific rights/provenance HOLD | HOLD | Eligible |
| mnist-model-assets | 2 | Specific rights/provenance HOLD | HOLD | Eligible |
| authored-code-raster | 1 | Recommend after authorship review | HOLD | Eligible |

**Totals:** 26 exact inputs; 1 local-fetch recommendation pending independent authorship review, 25 specific holds, 0 raw-publication clearances, 26 independently written metadata/hash/recipe entries publishable. None acquired.

The code-raster recommendation relies on the pinned per-task generated-image statement, pseudocode purpose, named contributor and Apache Source/Object grant together. It is NOT inferred merely from public availability/root license. If a reviewer rejects that authorship inference, obtain contributor confirmation before collection. Raw ZIP remains blocked until actual-byte provenance/hash/notice audit.

## Positive source-family grants do not prove opaque-file identity

- ARC original: Apache-2.0; opaque Git bundle contents still require source/object inventory. Explicit benchmark instructions establish intended task use, not a warranty of every embedded object.
- Zork original source: Microsoft 2025 MIT grant; exact recording creator/version/capture rights still unknown.
- MNIST curator dataset card: MIT (curator-card declaration); exact provided image binding and independent weight license remain missing. The TensorFlow page footer license was NOT mistaken for dataset rights.
- POV scene names Tom Price/version2.0. Current AGPL >=3.7 and legacy3.6 terms are NOT substituted for 2.2 rights.

## Per-family evidence, exact missing action and proposed-only alternative

### financial-documents

- Rights/source contact: unknown original document creators; benchmark task maintainer.
- Evidence: Randomized filenames and Dockerfile COPY documents demonstrate packaging, NOT generation or rights.
- Exact next action: Per-document original source URL/name and copyright/privacy grant; task maintainer must recover mapping.
- PROPOSED: publish a new fully synthetic CC0 invoice/document set plus generation source and author attestation; requalify OCR/oracle before outcomes.
- Citations: https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/financial-document-processor/task.toml, https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/financial-document-processor/instruction.md, https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/financial-document-processor/environment/Dockerfile

### doom-reference

- Rights/source contact: id Software/game-data owner; task reference-capture author.
- Evidence: Dockerfile pins doomgeneric source but separately downloads doom1.wad; engine source GPL does not license WAD pixels.
- Exact next action: Exact reference image capture provenance and applicable shareware/WAD screenshot permission; do not infer game-data rights from engine code.
- PROPOSED: Freedoom game data (PROPOSED; verify exact release BSD license first, COPYING lookup was 404) with a newly attributed capture/reference and objective oracle revision; not an exact replacement.
- Citations: https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/make-doom-for-mips/task.toml, https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/make-doom-for-mips/instruction.md, https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/make-doom-for-mips/environment/Dockerfile

### zork-video

- Rights/source contact: recording creator unknown; underlying Zork source Microsoft.
- Evidence: Task explicitly identifies Zork gameplay; Microsoft 2025 MIT source grant is positive underlying-game evidence, NOT proof of this recording owner.
- Exact next action: Identify recording creator and capture source/version; obtain recording grant and bind covered MIT game source/version to rendered text.
- PROPOSED: new deterministic recording of explicitly MIT-licensed historicalsource/zork1, with our capture permission and MIT notices; requalify transcript.
- Citations: https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/extract-moves-from-video/task.toml, https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/extract-moves-from-video/instruction.md, https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/extract-moves-from-video/environment/Dockerfile, https://raw.githubusercontent.com/historicalsource/zork1/master/LICENSE

### git-bundles

- Rights/source contact: Zhi Wang/task maintainer; unknown embedded-object owners.
- Evidence: Instruction explicitly requests both bundles for benchmark merging. Original ARC repository has Apache-2.0; opaque bundle object trees were NOT inspected.
- Exact next action: Source/object manifest and author license covering embedded code/history; bind any ARC examples to the Apache-2.0 original. No blanket redistribution grant inferred.
- PROPOSED: new reproducible bundles made only from explicitly Apache-2.0 ARC grids plus newly authored Apache-2.0 branch code/history; science review before freeze.
- Citations: https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/merge-diff-arc-agi-task/task.toml, https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/merge-diff-arc-agi-task/instruction.md, https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/merge-diff-arc-agi-task/environment/Dockerfile, https://raw.githubusercontent.com/fchollet/ARC-AGI/master/LICENSE

### pov-reference

- Rights/source contact: Tom Price scene author; Jeong Shin/reference capture provider; POV-Ray rights holders.
- Evidence: Retained illum1.pov names Tom Price and POV-Ray v2.0. Current >=3.7 AGPL and legacy 3.6 terms do NOT prove exact 2.2 scene/output grant.
- Exact next action: Original POV-Ray 2.2 scene/include/output distribution terms and capture authorship. Legacy-version mismatch remains exact missing grant, not impossible acquisition.
- PROPOSED: new self-authored CC0 scene/render and documented renderer rights, same build/compare task type; requalify oracle.
- Citations: https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/build-pov-ray/task.toml, https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/build-pov-ray/instruction.md, https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/build-pov-ray/environment/Dockerfile, https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/build-pov-ray/environment/deps/illum1.pov, https://www.povray.org/povlegal.html, https://www.povray.org/povlegal-3.6.html

### opaque-encrypted-archive

- Rights/source contact: jvpoulos/task maintainer; contained rights owners unknown.
- Evidence: Dockerfile copies protected.tar.gz.enc; visible ARS instruction/tests do not identify its contents. Encryption/availability is not licensing.
- Exact next action: Unencrypted CONTENT MANIFEST and permission from maintainer, without downloading/decrypting now; determine whether archive is actually required.
- PROPOSED: after dependency audit, remove only unused archive or replace with newly authored CC0 statistical fixtures; no sample drop, independently reviewed before outcomes.
- Citations: https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/adaptive-rejection-sampler/task.toml, https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/adaptive-rejection-sampler/instruction.md, https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/adaptive-rejection-sampler/environment/Dockerfile

### mnist-model-assets

- Rights/source contact: Jan-Lucas Uslu/model provider; MNIST curators for any source image.
- Evidence: Task README names MNIST and provided model/image. MNIST dataset card declares MIT (curator-card declaration); PyTorch software license does NOT grant trained-weight rights.
- Exact next action: Image: exact MNIST split/index/transformation binding + MIT notice compliance. Weights: explicit model grant and training/source provenance, separately.
- PROPOSED: exact licensed MNIST image with attribution plus newly released, explicitly licensed compatible 784-16-16-10 model after training-data rights review; no training/download now.
- Citations: https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/pytorch-model-cli/task.toml, https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/pytorch-model-cli/instruction.md, https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/pytorch-model-cli/environment/Dockerfile, https://huggingface.co/datasets/ylecun/mnist/raw/main/README.md

### authored-code-raster

- Rights/source contact: Guanghao Ye/task contributor.
- Evidence: Pinned solution lines 5-8 explicitly say code.png was generated intentionally for OCR difficulty; task identifies pseudocode, named author and source implementation. Combined with Apache-2.0 Source/Object contribution grant, this is more than root license alone.
- Exact next action: Independent licensing/science review of authorship inference; separately authorized local acquisition, exact prereg SHA check and notice audit before benchmark use or public ZIP.
- No substitution proposed: retain original task/image if the task-authored-image inference is independently accepted.
- Citations: https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/code-from-image/task.toml, https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/code-from-image/instruction.md, https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/code-from-image/environment/Dockerfile, https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/tasks/code-from-image/solution/solve.sh, https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/7131e4375048a0e408a8fb404b5f499d726b695b/LICENSE

## Review and execution boundaries

Publication of facts/hashes/URLs does not authorize redistributing raw content. No future fetch outside the exact original commit/source URL and prereg hash; local recommendation requires independent science/licensing review, fresh collection authorization and durable private/quarantine handling. Models/archives/bundles must not be executed/decrypted merely to inspect rights.

All alternatives are PROPOSED prospective amendments only; existing 28-task sample and historical bindings stay unchanged. No swap, VM cap, runtime identity, source launcher or operator-decision assumption changed. Runtime-source bytes will be proven equal to R16-clean dd54f8d at commit. Separate scientific/licensing review must assess this derivation.

## Frozen-runtime proof

R16 clean/terminal dd54f8d (review b0e98ae596f9, parent relay). All six runtime/review-hashed files plus historical binary-eligibility.json are byte-identical; exact SHA256 map is in the new derivation. This is a license/provenance-only delta, not a native admission or new resource policy.
