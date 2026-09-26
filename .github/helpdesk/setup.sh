#!/usr/bin/env bash
# Install the helpdesk team's lop attachments into this runner's config directory:
#
#   * the `helpdesk` team (from `.github/helpdesk/teams/helpdesk/`)
#   * the two roles the team uses that are NOT packaged with the harness
#     (`qa-tester`, `ux-reviewer`, from `.github/helpdesk/agents/`)
#   * a `config.yml` carrying the `shell_environment` allowlist, so the run's
#     provider key is not inherited by the model's `bash`/`eval` children
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

for team_src in "$repo_root"/.github/helpdesk/teams/*/; do
  [ -d "$team_src" ] && install_row teams "$team_src"
done
for agent_src in "$repo_root"/.github/helpdesk/agents/*/; do
  [ -d "$agent_src" ] && install_row agents "$agent_src"
done

# Write the child-process environment policy the run needs. The harness reads
# its provider key from its own process environment and, by default, hands a
# copy to every command the model writes — so without this policy, a command
# the model is induced to write can read the spend credential. Allowlist mode
# starts children from the harness's safe set plus an explicit grant: the
# inherit list is exactly what this engagement needs (gh auth plus the run
# context the prompts name), and the exclude is belt-and-braces against a mode
# flip. Merge into an existing config rather than replacing it (CI should have
# none). `python -I -` keeps the repo checkout off sys.path so a repo-local
# module cannot shadow the installed harness.
LOCAL_OPERATOR_CONFIG_DIR="$config_dir" python -I - <<'PY'
import os
from pathlib import Path

import yaml

config_file = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"]) / "config.yml"

data = {}
if config_file.exists():
    parsed = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        raise SystemExit(f"setup.sh: {config_file} is not a YAML mapping; refusing to replace it")
    data = parsed

values = data.get("values")
if values is None:
    values = data["values"] = {}
elif not isinstance(values, dict):
    raise SystemExit(f"setup.sh: 'values' in {config_file} is not a mapping; refusing to replace it")

shell_env = values.get("shell_environment")
if shell_env is None:
    shell_env = values["shell_environment"] = {}
elif not isinstance(shell_env, dict):
    raise SystemExit(f"setup.sh: 'shell_environment' in {config_file} is not a mapping; refusing to replace it")
# Don't clobber other stored settings; set exactly the three policy keys this
# deployment owns.
shell_env["mode"] = "allowlist"
shell_env["inherit"] = [
    "GH_TOKEN",
    "GH_REPO",
    "PR_NUMBER",
    "ISSUE_NUMBER",
    "RUNNER_TEMP",
    "GITHUB_WORKSPACE",
    "GITHUB_REPOSITORY",
    "NO_COLOR",
]
shell_env["exclude"] = ["RADIENT_API_KEY"]

config_file.parent.mkdir(parents=True, exist_ok=True)
config_file.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
print(f"setup.sh: shell_environment policy written to {config_file}")
PY

# Verify with the same resolution the runtime uses, not by re-listing files:
# `resolve_profile` is what `task(agent=...)` calls, and the team registry
# lookup is what `--team helpdesk` calls. Both must succeed before a run starts.
LOCAL_OPERATOR_CONFIG_DIR="$config_dir" python -I - <<'PY'
import os
from pathlib import Path

from local_operator.agent_profiles import resolve_profile
from local_operator.agents import AgentRegistry
from local_operator.teams import TeamRegistry
from local_operator.tools.shell_env import load_policy

config_dir = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"])
registry = AgentRegistry(config_dir)
for role in ("qa-tester", "ux-reviewer"):
    profile = resolve_profile(role, registry=registry)
    if profile is None or getattr(profile, "agent_id", None) is None:
        raise SystemExit(f"setup.sh: role {role!r} does not resolve from the registry")

if TeamRegistry(config_dir).get_team_by_name("helpdesk") is None:
    raise SystemExit("setup.sh: team 'helpdesk' does not resolve")

# `load_policy` reads once per process and freezes, so this fresh interpreter
# must see exactly what the write above stored: every child the policy governs
# is allowlisted. If it does not resolve, fail HERE rather than inside a run.
policy = load_policy()
assert policy.mode == "allowlist", f"shell_environment.mode resolved to {policy.mode!r}"
print("setup.sh: team helpdesk and roles qa-tester/ux-reviewer resolve")
print(f"setup.sh: shell_environment policy resolves: mode={policy.mode!r}")
PY
