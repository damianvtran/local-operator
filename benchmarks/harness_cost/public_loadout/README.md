# Public loadout — optional methods asset, NOT a v1 reproduction

The analysis in `../regenerate.py` uses frozen usage logs and does not need
this directory. These files support a future publicly inspectable loadout;
they do **not** reproduce v1's private instructions, skills or plugins.
No current-performance, quality, or byte-matching claim is made. The schema
snapshot has **493 tools / 9 OSS servers / 931,038 compact UTF-8 schema bytes**.
Raw file bytes include provenance/pretty-printing and are a different measure.

## Optional setup (no paid model calls)

From this directory:

```sh
bash fetch_public_skills.sh
BENCH_SCRATCH=/chosen/fresh/scratch bash build_basehomes_public.sh
# Optional, separate environment: starlette/uvicorn, not analysis dependencies
python3 mcp_replay_public.py
# Optional schema recapture: downloads pinned packages using npm/npx and uvx
python3 capture_mcp_schemas.py --check
```

The fetch script pins `anthropics/skills` at
`ca45fc496a3436251f191354091d557da31c15b5`; 19 SKILL.md hashes are verified.
No skill files are redistributed here. Four document skills (`docx`, `pdf`,
`pptx`, `xlsx`) are proprietary/source-available, **not OSS**; inspect their
LICENSE.txt before use. Other listed skills are Apache-2.0 per upstream.
`BENCH_PUBLIC_SKILLS_EXCLUDE="docx pptx xlsx pdf"` selects the other 15.

The base-home builder is deliberately portable and refuses an existing output.
It copies the same skill set to six native roots and the same neutral instruction
text to seven native instruction paths, without reading real user config. It
never starts a harness, redirects HOME for a subprocess, touches keychains or
installs environments. A future runner must own its platform-specific safe HOME
and credential isolation. This changes the original macOS-only helper's keychain
and APFS clone implementation, not the intended instruction/skill bytes.

The replay server only serves the frozen OSS tool schemas; `tools/call` is
refused with a tool error. Default bind is loopback. A future harness runner
must write its own native MCP configuration; it is intentionally not supplied
with private v1 configuration. Replay-only dependencies (optional):
`starlette==0.48.0`, `uvicorn==0.38.0` (not installed by analysis reproduction).
Recapture requires Node ≥22/npm and uv/uvx. Dummy credentials are intentionally
non-secret placeholders. Credential-shaped LaunchDarkly startup data is
constructed only in runtime memory; the committed command metadata instead
contains `<offline-dummy-generated-at-runtime>`. The runtime invocation is
identical to the prior validated offline capture, and per-tool schema hashes,
licenses and replay text are unchanged. No real key belongs here, and dummy
literals do not get a push-protection exemption. Recapture starts only MCP schema-listing subprocesses,
not a browser engine or real tool invocation. It is not run in this kit's test.

Capture pins top-level versions and applies a dependency-resolution time cutoff,
not a complete transitive lock. `--check` is the byte-compatibility gate; the
committed snapshot, not a future package resolution, is authoritative for replay.
See `public_mcp_schemas.json` for package, version, source URL, per-server
schema bytes, hashes and capture metadata; see ../THIRD_PARTY_NOTICES.md.

## Differences from private v1

- v1 skill roots contained different partly overlapping sets; the public setup
  has 19 shared names in all six roots. Duplicate handling can differ by harness.
- Public document skills may be task-relevant; v1 skills were irrelevant and
  none read a body. This may change both cost and outcomes.
- Neutral instructions are 3,554 bytes and uniform. v1 Local Operator carried
  19,814 bytes, while other roots differed or had no instruction file.
- More native instruction paths exist here; some harnesses may load duplicates.
- Private v1 schemas had 480 tools / 1,184,565 compact bytes. Public schemas
  have 493 tools / 931,038 bytes, different per-tool distributions and domains.
  Wire payloads including built-ins/plugin tools are yet another quantity.
- Private plugins/marketplaces are not reproduced; server instructions are not
  served, only tool definitions, consistent with the original replay policy.
- No first-request token-count matching or public-loadout cost rerun is supplied.

A future v2 run requires a reviewed selection/preregistration and feasible
budget; no 60-task sample, universal $0.30 cap, quality parity, or 41× current
performance ratio is implied by these files. In particular, genuinely eager
public schema exposure can itself have a high initial cost; it must not be
confused with v1 Codex's metadata-driven compaction loop or filtered away.
