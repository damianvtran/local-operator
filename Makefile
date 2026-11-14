# ============================================================================
# Local Operator Makefile
# ============================================================================
#
# This Makefile provides commands for development, testing, and running the
# Local Operator project. It includes targets for setting up the Python
# environment, running the server, executing tests, and maintaining code quality.
#
# Main targets:
# - install: Complete setup (Python + dependencies)
# - setup-python: Install pyenv and Python 3.12
# - server/dev-server: Run the server
# - test/coverage: Run tests
# - format/lint/type-check: Code quality tools
#

# Declare all targets as phony (not representing files)
.PHONY: server dev-server cli openapi test test-affected merge-gate coverage format lint type-check check-changed adapter-osworld security clean help setup-python install

# Default target when running 'make' without arguments
.DEFAULT_GOAL := help

# ============================================================================
# Setup Commands
# ============================================================================

# Install pyenv and Python 3.12 if not already installed
# This target ensures the correct Python version is available
setup-python: ## Install pyenv and Python 3.12 if not already installed
	@echo "Setting up Python environment..."
	@./scripts/install_pyenv.sh

# Complete installation target that depends on setup-python
# This will set up Python and install all project dependencies
install: setup-python ## Complete setup: install Python 3.12 and all dependencies
	@echo "Creating virtual environment in .venv..."
	@if command -v python3.12 >/dev/null 2>&1; then \
		python3.12 -m venv .venv; \
	else \
		echo "python3.12 not found in PATH, using pyenv to create virtual environment..."; \
		eval "$$(pyenv init -)" && pyenv shell 3.12 && python -m venv .venv; \
	fi
	@echo "Activating virtual environment and installing dependencies..."
	. .venv/bin/activate && pip install -e ".[dev]"

# ============================================================================
# Variables
# ============================================================================
# Python executable to use for commands
PYTHON := python3.12
# Test runner
PYTEST := pytest
# Directory for test coverage reports
COVERAGE_DIR := htmlcov
# Output file for OpenAPI specification
OPENAPI_OUTPUT := docs/openapi.json

# ============================================================================
# Server Commands
# ============================================================================
# Start the server without hot reload (for production-like environments)
server: ## Start the server
	local-operator serve

# Start the server with hot reload (for development)
dev-server: ## Start the server with hot reload
	local-operator serve --reload

# ============================================================================
# CLI Commands
# ============================================================================
# Start the CLI interface
cli: ## Start the CLI
	local-operator

# ============================================================================
# Documentation Commands
# ============================================================================
# Generate OpenAPI specification for API documentation
openapi: ## Generate OpenAPI specification
	$(PYTHON) -m local_operator.server.generate_openapi -o $(OPENAPI_OUTPUT)

# ============================================================================
# Testing Commands
# ============================================================================
# Run all tests
test: ## Run tests
	$(PYTEST)

# Generate test coverage report
coverage: ## Generate test coverage report
	$(PYTEST) --cov=local_operator --cov-report=html
	@echo "Coverage report generated in $(COVERAGE_DIR)/"

# ============================================================================
# Code Quality Commands
# ============================================================================
# Invoke formatters/linters the same way AGENTS.md documents the gates.
# Bare `.venv/bin/black` (and flake8/isort/pyright) console scripts carry a
# shebang baked in at install time; after a worktree that owned the venv
# is deleted they fail with `bad interpreter` and exit 126. `rc=126`
# disappears inside a pipeline (`cmd | tail` reports tail's 0), so a gate
# that never ran looks green. `python -m` uses the interpreter's module
# path; `uvx` fetches a pinned tool into an isolated cache. Neither reads
# those shebangs. black/isort are deliberately NOT in the dev extra, so
# format goes through uvx at the versions CI pins.
format: ## Format code with black and isort
	uvx --from black==26.1.0 black .
	uvx isort==5.13.2 .

# Run linting with flake8
lint: ## Run linting with flake8
	.venv/bin/python -m flake8 .

# Run exactly the CI gates this branch's diff can affect. The selector is
# `scripts/ci_scope.py` — the SAME module the workflow's `changes` job runs and
# whose flags gate its jobs — so the local answer and CI's answer cannot drift
# into two opinions about which gates a diff needs.
#
# `--since` is the merge base with `origin/main`, not `origin/main` itself: a
# plain two-dot diff against a moved `origin/main` sweeps in every commit main
# landed since this branch was cut and would run gates for work this branch
# never wrote (the same staleness `version-bump-guard` documents in ci.yml).
# The module adds the index, the working tree and untracked files on top.
#
# `.venv/bin/python ...` goes through the interpreter, and every gate the module
# then runs is spelled `python -m` or `uvx` for the #423 reason above: a bare
# `.venv/bin/flake8` can exit 126 and be swallowed by a pipeline.
# File-level narrowing happens inside that module (its fourth section), and it
# narrows ONLY where the selection can be proven complete: `lint` over the changed
# files, `type-check` over the changed files plus their reverse dependents. The two
# pytest jobs run whole-tree on this tree — the graph carries hundreds of references
# it cannot place (any one of them stops a selection), and a change inside the app
# boot closure always does. `--no-scope` runs the whole-tree commands anyway and
# `--dry-run` prints the plan without running it; falling back is printed with the
# trigger that caused it, and CI's full matrix stays the authoritative gate.
# AGENTS.md, "Scoping the inner loop", has the rule, the trigger list and the costs.
check-changed: ## Run the CI gates this branch's diff can affect
	@base="$$(git merge-base origin/main HEAD 2>/dev/null || git rev-parse origin/main)"; \
	echo "checking changes since $$base"; \
	.venv/bin/python scripts/ci_scope.py --since "$$base" --run

# Run the unit tests this branch's diff affects and print the PR evidence block.
# Pull-request CI no longer runs the unit suite (it runs on `main` in
# .github/workflows/full-suite.yml), so this is the targeted run a PR records
# under `## Targeted test evidence`. Same merge-base rule as check-changed;
# the selection rules and the always-on guard list live in the script.
test-affected: ## Run the tests this branch's diff affects; print PR evidence
	@base="$$(git merge-base origin/main HEAD 2>/dev/null || git rev-parse origin/main)"; \
	.venv/bin/python scripts/affected_tests.py --since "$$base" --python .venv/bin/python --evidence

# RE-ARM THE MERGE GATE on PR=<number>.
#
# The full suite runs on a pull request ONLY for a `labeled` event, because
# running it per push is the cost this repository paid for years. That makes
# the gate's check a property of the SHA it ran on: any push AFTER the gate
# leaves `Merge gate (full suite)` unreported on the new head, where it sits in
# "Expected" and blocks the merge — correctly, since nothing verified that
# head. Re-dispatching the workflow by hand is not the narrow exit either: a
# `workflow_dispatch` DOES report this workflow's checks on the head (every run
# publishes the verdict of its own `needs`, which is why the gate carries no
# event guard), but dispatch is the `WIDE:` infra route and spends a full suite
# on purpose. Removing and re-adding the label is the narrow, documented exit
# from that block, and it is the REASON no `if:` in full-suite.yml may test the
# label's NAME: a job made conditional on a label reports `skipped` on runs
# started by another label, and GitHub counts `skipped` as a passed required
# check.
#
# It verifies what it armed instead of assuming: the PR must exist and be OPEN,
# removing an already-present label must actually succeed (otherwise the add is
# a no-op, no `labeled` event fires, and the head stays stuck in "Expected"),
# and the run it prints is resolved by head SHA + created-after timestamp —
# never `--limit 1`, whose single row is the PREVIOUS run on any re-arm. No new
# run within the bounded poll is an error with the next commands to run, not a
# `gh run watch null`.
#
# The run list is filtered by the SYSTEM `jq`, not by `gh --jq`: `gh` has no
# `--arg` flag and its `--jq` takes exactly one token, so a query needing three
# parameters cannot be expressed there at all. Measured on gh 2.100.0,
# `--jq --arg sha … --arg since … --arg prev …` exits 1 with `unknown flag:
# --arg`, and since the failed lookup was swallowed, `prev` stayed empty and the
# poll matched nothing — the ritual armed a real run and then reported that it
# had armed none, on every arming. `jq --arg` is the supported way to bind the
# values, and gh's `--json` output is what it consumes.
#
# Before any label write, look for live gates on this head. A status-filtered
# existence query needs only one result (not a bounded slice of old history).
# This guard is advisory and non-atomic: it reduces redundant queued runs but
# does not guarantee their absence, and is not a lock. Simultaneous direct label writes
# can still produce two runs; native per-ref concurrency remains the backstop.
# Retry transient lookups at most 3 times, 1 s apart, 5 s per request, sharing
# a 17 s total advisory budget (3*5 + 2*1), then warn and proceed. This is a
# small operator-wait policy budget, not a measured GitHub latency percentile.
# Auth failures refuse; unusable responses require the manual override. The explicit
# I_KNOW_A_GATE_RUN_EXISTS=1 override skips the lookup with a visible warning,
# so even an unreachable API cannot prevent an intentional arm.
# Prints the triggered run and how to follow it.
merge-gate: ## Re-arm the full-suite merge gate on PR=<number>
	@set -e; \
	if [ -z "$(PR)" ]; then echo "usage: make merge-gate PR=<number>" >&2; exit 2; fi; \
	command -v jq >/dev/null 2>&1 \
	  || { echo "make merge-gate: jq is required (the run list is filtered with 'jq --arg', which gh's --jq cannot take); install it, e.g. 'brew install jq'" >&2; exit 1; }; \
	spec="$$(gh pr view "$(PR)" --json state,headRefOid,headRefName \
	  --jq '.state + " " + .headRefOid + " " + .headRefName')" \
	  || { echo "make merge-gate: PR #$(PR) not found (gh pr view failed); pass the number, e.g. PR=3631" >&2; exit 1; }; \
	set -- $$spec; state="$$1"; head_sha="$$2"; head="$$3"; \
	if [ "$$state" != "OPEN" ]; then \
	  echo "make merge-gate: PR #$(PR) is $$state, not OPEN; only an open PR can be re-armed" >&2; exit 1; \
	fi; \
	if [ "$(I_KNOW_A_GATE_RUN_EXISTS)" = "1" ]; then \
	  echo "WARNING: I_KNOW_A_GATE_RUN_EXISTS=1: existing-gate lookup did not run (skipped); intentionally arming without duplicate detection" >&2; \
	else \
	advisory_helper="$(dir $(abspath $(lastword $(MAKEFILE_LIST))))scripts/merge_gate_advisory.py"; \
	advisory_deadline="$$(python3 -c 'import time; print(time.monotonic() + 17)')"; \
	for active_status in queued in_progress pending waiting requested; do \
	  if active="$$(python3 "$$advisory_helper" "$$advisory_deadline" \
	    gh run list --workflow=full-suite.yml --event=pull_request --branch "$$head" \
	    --commit "$$head_sha" --status "$$active_status" --limit=1 --json databaseId,url)"; then :; \
	  else \
	    if printf '%s\n' "$$active" | grep -Eiq 'HTTP 401|bad credentials|unauthori[sz]ed|authentication failed' \
	      || { printf '%s\n' "$$active" | grep -Eiq 'HTTP 403' && ! printf '%s\n' "$$active" | grep -Eiq 'rate.?limit'; }; then \
	      echo "make merge-gate: existing-gate lookup auth/permission failure: $$active; no label changed" >&2; exit 1; \
	    fi; \
	    if printf '%s\n' "$$active" | grep -Eiq 'rate.?limit|HTTP 429|HTTP 5[0-9][0-9]|timed? out|timeout|network|connection|no response|API unavailable|error connecting|no such host|TLS handshake|EOF' || [ -z "$$active" ]; then \
	      echo "WARNING: existing-gate lookup unavailable after bounded retry ($$active_status): $${active:-no response}; proceeding to arm without duplicate detection; manual recovery for an unusable lookup or owner-diagnosed stuck/invalid run: make merge-gate PR=$(PR) I_KNOW_A_GATE_RUN_EXISTS=1" >&2; \
	      break; \
	    fi; \
	    echo "make merge-gate: cannot check existing $$active_status gates (unrecognised response): $$active; no label changed; lane-owner manual recovery only: make merge-gate PR=$(PR) I_KNOW_A_GATE_RUN_EXISTS=1" >&2; exit 1; \
	  fi; \
	  active="$$(printf '%s\n' "$$active" | jq -er 'if type == "array" and all(.[]; type == "object" and (.databaseId | type == "number") and (.url | type == "string" and length > 0)) then (.[0] // empty | "\(.databaseId) \(.url)"), "" else error("malformed run response") end')" \
	    || { echo "make merge-gate: malformed existing-gate response; no label changed; lane-owner manual recovery only: make merge-gate PR=$(PR) I_KNOW_A_GATE_RUN_EXISTS=1" >&2; exit 1; }; \
	  if [ -n "$$active" ]; then \
	      echo "make merge-gate: refusing to arm PR #$(PR): $$active_status gate run $$active already exists for $$head_sha" >&2; \
	      echo "  follow that run; intentional duplicate only: make merge-gate PR=$(PR) I_KNOW_A_GATE_RUN_EXISTS=1" >&2; \
	      exit 1; \
	  fi; \
	done; \
	fi; \
	prev="$$(gh run list --workflow=full-suite.yml --event=pull_request --branch "$$head" --limit 20 \
	  --json databaseId,headSha \
	  | jq -r --arg sha "$$head_sha" '[.[] | select(.headSha == $$sha)] | .[0].databaseId // empty')" || prev=""; \
	since="$$(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
	gh label create ready-to-merge --color 0E8A16 \
	  --description "Run the full-suite merge gate on this PR" 2>/dev/null || true; \
	if gh pr view "$(PR)" --json labels --jq '.labels[].name' | grep -qx ready-to-merge; then \
	  gh pr edit "$(PR)" --remove-label ready-to-merge >/dev/null \
	    || { echo "make merge-gate: could not remove ready-to-merge from PR #$(PR); with the label still on, the add below is a no-op, no labeled event fires and no run starts" >&2; exit 1; }; \
	  sleep 3; \
	fi; \
	gh pr edit "$(PR)" --add-label ready-to-merge >/dev/null; \
	run=""; i=0; \
	while [ $$i -lt 12 ]; do \
	  i=$$((i + 1)); \
	  run="$$(gh run list --workflow=full-suite.yml --event=pull_request --branch "$$head" --limit 20 \
	    --json databaseId,url,headSha,createdAt \
	    | jq -r --arg sha "$$head_sha" --arg since "$$since" --arg prev "$$prev" \
	      '[.[] | select(.headSha == $$sha and .createdAt >= $$since and ((.databaseId | tostring) != $$prev))] | .[0] | select(. != null) | "\(.databaseId) \(.url)"')" || run=""; \
	  if [ -n "$$run" ]; then break; fi; \
	  sleep 5; \
	done; \
	if [ -z "$$run" ]; then \
	  echo "make merge-gate: armed ready-to-merge on PR #$(PR) ($$head), but no NEW full-suite pull_request run for $$head_sha appeared within 60 s; nothing would follow" >&2; \
	  echo "  check: gh run list --workflow=full-suite.yml --branch $$head --limit 5" >&2; \
	  echo "         (a GH_TOKEN that is an Actions token starts no workflow run from the edit)" >&2; \
	  echo "         gh pr checks $(PR)" >&2; \
	  exit 1; \
	fi; \
	echo "re-armed the merge gate on PR #$(PR) ($$head)"; \
	echo "  run:    $${run#* }"; \
	echo "  follow: gh run watch $${run%% *} --exit-status"; \
	echo "  checks: gh pr checks $(PR)"

# Run type checking with pyright
#
# Through `scripts/run_bounded.py`, not a bare `timeout`, and never unbounded.
# pyright is a Python wrapper around an npm/node analyzer that runs as a
# SEPARATE process, and the fleet has shown analyzers re-parented to launchd
# holding 1-2 GB each, one alive 81 minutes after its parent died. The wrapper
# puts the command in its own process group and signals the whole group on every
# exit path — including the two a bare `timeout` cannot cover: a descendant alive
# when the leader exits by itself, and a group that ignores SIGTERM (measured over
# a SIGTERM-ignoring tree: `timeout 3` fires at 3 s and then WAITS THE TREE OUT —
# 30 s for a 30 s tree, 300 s in an earlier run for a 300 s one — where the
# wrapper's SIGKILL escalation cleared the same tree in 5 s with `--grace 2`; the
# shipped `--grace` default of 10 takes ~13 s for it).
#
# The bound is generous on purpose: a whole-tree pyright measures 508 s quiet and
# 1170 s under load on this fleet, so a bound mirroring ci.yml's
# `timeout-minutes: 15` — which also covers checkout, install and the protocol-sync
# step — would red a gate CI passes. Timing out a legitimately slow host is worse
# than waiting for it, and the bound is overridable for a machine that needs more:
# `make type-check BOUND_TIMEOUT=3600`.
#
# Node heap headroom: whole-tree pyright runs the real analyzer in a node child
# whose default V8 old-space cap is ~4 GB (4192 MB measured on this host), and a
# bare local run can cross it under fleet load — measured: OOMed ~116 s in,
# `rc=250` out of the pyright wrapper. ci.yml's type-check job sets
# `NODE_OPTIONS=--max-old-space-size=6144` for the same reason (its 16 GB runner
# OOMed at heap 3885 MB). run_bounded sizes node's heap for BOTH this recipe
# and the scoped runner, raising a smaller caller cap without lowering a larger
# operator-set value; its constant is guarded against ci.yml's value by tests.
BOUND_TIMEOUT ?= 1800

# --contract=command-status: the gate needs pyright's STATUS, not a cleanup
# certificate, and cleanup-required refuses before spawn on Darwin (no
# authoritative census backend). command-status still enforces the memory grant
# and sweeps the owned group; it never certifies cleanup (docs/design/run-bounded-contract.md).
type-check: ## Run type checking with pyright
	.venv/bin/python scripts/run_bounded.py --contract=command-status --timeout $(BOUND_TIMEOUT) -- .venv/bin/python -m pyright --pythonpath .venv/bin/python .

# Build the OSWorld V2 evaluation adapter: lock, wheel, and the workspace
# materialisation command. Deliberately NOT wired into CI's default job — the
# workspace step needs a human-accepted gated HF dataset and an HF_TOKEN, so
# this is a developer/operator command, not a gate.
adapter-osworld: ## Build the OSWorld V2 adapter wheel and print workspace steps
	cd benchmarks/osworld_v2_adapter && uv lock && uv build --wheel --out-dir dist/
	@echo "Wheel built under benchmarks/osworld_v2_adapter/dist/."
	@echo "Materialise the workspace (needs HF_TOKEN) with:"
	@echo "  python scripts/build_osworld_adapter.py --benchmark-release osworld-v2-2026.08.08 --out <workspace>"

# Run security audit with pip-audit
security: ## Run security audit with pip-audit
	pip-audit

# ============================================================================
# Cleanup Commands
# ============================================================================
# Clean up generated files and directories
clean: ## Clean up generated files
	rm -rf $(COVERAGE_DIR)
	rm -rf .pytest_cache
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# ============================================================================
# Help Command
# ============================================================================
# Display help information about available commands
help: ## Show this help message
	@echo "Local Operator Makefile Commands:"
	@echo
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
