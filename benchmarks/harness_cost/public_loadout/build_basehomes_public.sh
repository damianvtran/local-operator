#!/usr/bin/env bash
# WHY: render the pinned public loadout, never the private v1 home, into native
# skill/instruction locations. This is optional; analysis never needs a home.
# CONSTRAINTS: refuses existing output, no auth/keychain/config reads, no harness
# launch. Copies ordinary skill assets (not environments). No macOS-only tools.
# A future benchmark runner must provide its own safe platform-specific HOME
# isolation/keychain policy; this file intentionally does not run any client.
set -euo pipefail
KIT=$(cd "$(dirname "$0")" && pwd)
: "${BENCH_SCRATCH:?Set BENCH_SCRATCH to a fresh scratch parent}"
SKILLS=${SKILLS_DEST:-$KIT/kit-src/anthropic-skills}
SKILLS_DEST="$SKILLS" bash "$KIT/fetch_public_skills.sh"
python3 - "$KIT" "$SKILLS" "$BENCH_SCRATCH" <<'PY'
import hashlib, os, shutil, sys
from pathlib import Path
kit, skills, scratch = map(Path, sys.argv[1:])
out = scratch / 'basehome-public'
if out.exists():
    raise SystemExit('Refusing existing basehome-public; choose a fresh BENCH_SCRATCH')
exclude = set(os.environ.get('BENCH_PUBLIC_SKILLS_EXCLUDE', '').split())
names = [p.parent.name for p in sorted((skills/'skills').glob('*/SKILL.md'))]
if not exclude <= set(names):
    raise SystemExit('Unknown excluded skill')
names = [n for n in names if n not in exclude]
roots = ['.local-operator/skills','.claude/skills','.codex/skills','.agents/skills','.config/opencode/skills','.omp/agent/skills']
for root in roots:
    for name in names:
        shutil.copytree(skills/'skills'/name, out/root/name)
for rel in ['.claude/CLAUDE.md','.codex/AGENTS.md','.local-operator/system_prompt.md','.omp/agent/AGENTS.md','.config/opencode/AGENTS.md','.hermes/SOUL.md','.dsh/AGENTS.md']:
    dst=out/rel
    dst.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(kit/'public_user_instructions.md',dst)
    if dst.read_bytes() != (kit/'public_user_instructions.md').read_bytes():
        raise SystemExit('Instruction mismatch')
(out/'.codex/config.toml').write_text('')
(out/'.config/opencode/opencode.jsonc').write_text('{"$schema":"https://opencode.ai/config.json"}\n')
print(f'Public basehome: {len(names)} skills per root, {len(roots)} roots; no harness launched')
PY
