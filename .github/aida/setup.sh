#!/usr/bin/env bash
# Install Aida's lop attachments into this runner's config directory:
#
#   * the `aida` team (from `.github/aida/teams/aida/`)
#   * the two roles the team uses that are NOT packaged with the harness
#     (`qa-tester`, `ux-reviewer`, from `.github/aida/agents/`)
#
# The installed `local-operator` resolves teams and roles from the config
# directory (`$LOCAL_OPERATOR_CONFIG_DIR`, default `$HOME/.local-operator`);
# this script copies committed files into place and verifies them with the
# same resolution the runtime uses. Idempotent: re-running replaces the
# installed copies. Exits non-zero when anything does not resolve, so a broken
# install fails HERE rather than inside a model run.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
config_dir="${LOCAL_OPERATOR_CONFIG_DIR:-$HOME/.local-operator}"

mkdir -p "$config_dir/teams" "$config_dir/agents"

install_row() {
  # <family> <source-dir>: copy a row directory to $config_dir/<family>/<id>,
  # taking the id from the row's YAML so directory names stay cosmetic.
  local family="$1" source="$2" yaml id
  if [ "$family" = "teams" ]; then yaml="team.yml"; else yaml="agent.yml"; fi
  id="$(sed -n 's/^id: *//p' "$source/$yaml" | head -n1)"
  if [ -z "$id" ]; then
    echo "setup.sh: no id found in $source/$yaml" >&2
    exit 1
  fi
  rm -rf "${config_dir:?}/$family/$id"
  cp -R "$source" "$config_dir/$family/$id"
}

for team_src in "$repo_root"/.github/aida/teams/*/; do
  [ -d "$team_src" ] && install_row teams "$team_src"
done
for agent_src in "$repo_root"/.github/aida/agents/*/; do
  [ -d "$agent_src" ] && install_row agents "$agent_src"
done

# Verify with the same resolution the runtime uses, not by re-listing files:
# `resolve_profile` is what `task(agent=...)` calls, and the team registry
# lookup is what `--team aida` calls. Both must succeed before a run starts.
LOCAL_OPERATOR_CONFIG_DIR="$config_dir" python - <<'PY'
import os
from pathlib import Path

from local_operator.agent_profiles import resolve_profile
from local_operator.agents import AgentRegistry
from local_operator.teams import TeamRegistry

config_dir = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
registry = AgentRegistry(config_dir)
for role in ("qa-tester", "ux-reviewer"):
    profile = resolve_profile(role, registry=registry)
    if profile is None or getattr(profile, "agent_id", None) is None:
        raise SystemExit(f"setup.sh: role {role!r} does not resolve from the registry")

if TeamRegistry(config_dir).get_team_by_name("aida") is None:
    raise SystemExit("setup.sh: team 'aida' does not resolve")

print("setup.sh: team aida and roles qa-tester/ux-reviewer resolve")
PY
