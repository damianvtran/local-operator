#!/usr/bin/env bash
# Slice-0 POC entrypoint: the container half of docs/design/remote-cloud-agents.md
# §9.2 (steps 4 and 5, plus the commit/bundle/upload the driver consumes).
#
# Runs as uid 10001 with a READ-ONLY root filesystem, so /workspace is the only
# writable mount and every path this script writes lives under it.
#
# NEVER add `set -x`. From the first line to the last this script holds the model
# key in a shell variable, and a trace would put it in CloudWatch — the one place
# it must never reach. The key is read out of the environment (the task definition
# injects it as an ECS secret) and the environment variable is UNSET immediately,
# so the agent process gets it only through the explicit subshell export below.
set -euo pipefail

# --- phase 0: the ROOT phase, and the ONLY code in this container that ever runs
# as root. A Fargate task volume is mounted root-owned, so a container started as
# uid 10001 cannot create its own workspace — measured, the first attempt died on
# `mkdir: cannot create directory '/workspace/out': Permission denied` before it
# reached its second step, and the task definition's `user` field cannot fix it.
# So the process starts as root, this block hands the volume to 10001, and the
# script RE-EXECUTES ITSELF as 10001, so nothing that follows — not the probes,
# not the agent, not anything the agent spawns — is privileged. The alternative
# (a non-root container with a writable root filesystem) would have traded the
# read-only rootfs for the same result.
if [ "$(id -u)" = "0" ]; then
    chown -R 10001:10001 /workspace
    echo "phase 0: chowned /workspace to 10001:10001, dropping privileges" >&2
    # python, not setpriv: no dependency on which util-linux the base ships, and
    # setgid before setuid is the one order that cannot fail.
    exec python3 -c '
import os, sys
os.setgid(10001)
os.setuid(10001)
os.execv(sys.argv[1], sys.argv[1:])
' "$0" "$@"
fi

readonly OUT=/workspace/out
readonly WORKSPACE=/workspace
readonly FIFO=/workspace/tmp/stream.fifo
# 2 h, the bound §9.3's cost estimate assumes. Firing it is recorded, not hidden.
readonly AGENT_DEADLINE_SECONDS=7200

mkdir -p "$OUT" "$WORKSPACE/repo" "$WORKSPACE/tmp" "$WORKSPACE/home" \
    "${LOCAL_OPERATOR_CONFIG_DIR:-$WORKSPACE/config/local-operator}"

# Every step stamps an epoch-millisecond line into timings.jsonl. The EXIT trap
# folds that into timings.json, so a run that FAILS still leaves its timings
# behind: the timings of a failure are evidence too.
stamp() {
    printf '{"name":"%s","epoch_ms":%s}\n' "$1" "$(date +%s%3N)" >>"$OUT/timings.jsonl"
}

# Upload to a presigned PUT URL, and REQUIRE a 2xx.
#
# WHY this is not just `curl -fsS --upload-file`: `curl -f` fails on 4xx/5xx and
# treats 3xx as SUCCESS. S3 answers a request for a ca-central-1 bucket sent to the
# legacy global host with `307 TemporaryRedirect` and no body, so the upload silently
# did nothing while the run exited 0 with no artifacts — measured, and the single
# most misleading failure in this POC. A redirect is not a successful upload.
put_file() {
    local url="$1"
    local file="$2"
    local code
    code="$(curl -sS --max-time 300 -o /workspace/tmp/curl-body.txt -w '%{http_code}' \
        -X PUT --upload-file "$file" "$url")" || {
        echo "FATAL: upload of $file to the presigned URL failed (curl exit $?)" >&2
        return 1
    }
    case "$code" in
    2??)
        rm -f /workspace/tmp/curl-body.txt
        return 0
        ;;
    *)
        echo "FATAL: upload of $file got HTTP $code, not a 2xx" >&2
        cat /workspace/tmp/curl-body.txt >&2 || true
        return 1
        ;;
    esac
}

fold_timings() {
    /opt/lop/bin/python - "$OUT/timings.jsonl" "$OUT/timings.json" <<'PY' || true
import json
import sys

events = []
try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                events.append(json.loads(line))
except (OSError, ValueError):
    pass
origin = events[0]["epoch_ms"] if events else None
for event in events:
    event["t_plus_ms"] = None if origin is None else event["epoch_ms"] - origin
with open(sys.argv[2], "w", encoding="utf-8") as handle:
    json.dump({"events": events}, handle, indent=2, sort_keys=True)
PY
}

# The EXIT trap folds the timings and stamps the end — but it is NOT the only caller
# of fold_timings: the trap runs after the results tarball has been sealed, so relying
# on it alone ships a raw timings.jsonl with no folded timings.json. Measured: the
# first artifact-bearing run had every cold-start number null for exactly that reason.
on_exit() {
    rc=$?
    stamp t_container_end
    fold_timings
    exit "$rc"
}
trap on_exit EXIT

stamp t_container_start

# ---------------------------------------------------------------- step 2: key
# Out of the environment before anything else can copy that environment.
MODEL_KEY="${LOP_POC_MODEL_KEY:-}"
unset LOP_POC_MODEL_KEY
stamp t_key_read

POC_MOCK="${POC_MOCK:-0}"
POC_HOSTING="${POC_HOSTING:-}"
POC_MODEL="${POC_MODEL:-}"
PROVIDER_ENV=""

if [ "$POC_MOCK" = "1" ]; then
    # Mock runs use lop's own `test` provider (wire "mock"), which
    # `allows_missing_api_key`, so the run needs no key at all.
    :
else
    if [ -z "$POC_HOSTING" ]; then
        case "$MODEL_KEY" in
        sk-or-*) POC_HOSTING="openrouter" ;;
        sk-ant-*) POC_HOSTING="anthropic" ;;
        *)
            echo "FATAL: POC_HOSTING is empty and the key prefix names no known provider" >&2
            exit 2
            ;;
        esac
    fi
    if [ -z "$POC_MODEL" ]; then
        echo "FATAL: POC_MODEL is required for a non-mock run" >&2
        exit 2
    fi
    # The env-var NAME comes from the provider registry rather than a table here,
    # so a rename upstream cannot leave this script exporting a name nothing reads.
    PROVIDER_ENV="$(/opt/lop/bin/python - "$POC_HOSTING" <<'PY'
import sys

from local_operator.providers.registry import get_provider_definition

definition = get_provider_definition(sys.argv[1])
keys = getattr(definition, "env_keys", None)
if isinstance(keys, str):
    keys = (keys,)
if not keys:
    print("")
elif isinstance(keys, (list, tuple)):
    print(keys[0])
else:
    print("")
PY
)"
    if [ -z "$PROVIDER_ENV" ]; then
        echo "FATAL: provider '$POC_HOSTING' declares no environment key name" >&2
        exit 2
    fi
fi

# What probe 4c searches for beyond the exact injected value: the first 8 characters
# of the key, and only for a REAL key. A mock run's value is a placeholder, so there
# is no credential-shaped prefix in it to hunt for.
KEY_PREFIX_CHARS=0
if [ "$POC_MOCK" != "1" ]; then
    KEY_PREFIX_CHARS=8
fi

# The child environment a MODEL-authored command sees. `allowlist` is what keeps
# the provider key out of every bash/eval child the agent starts; the key is in
# the lop process's own environment, which is the one place it has to be.
# `inherit`/`exclude` are restated because the config layer replaces
# `values.shell_environment` wholesale instead of deep-merging it.
cat >"$LOCAL_OPERATOR_CONFIG_DIR/config.yml" <<'YAML'
values:
  shell_environment:
    mode: allowlist
    inherit: []
    exclude: []
YAML
stamp t_config_written

# ------------------------------------------------------------- step 3: clone
stamp t_clone_start
git clone --quiet "$POC_REPO_URL" "$WORKSPACE/repo"
git -C "$WORKSPACE/repo" checkout --quiet --detach "$POC_SHA"
head_sha="$(git -C "$WORKSPACE/repo" rev-parse HEAD)"
if [ "$head_sha" != "$POC_SHA" ]; then
    echo "FATAL: HEAD is $head_sha, asked for POC_SHA $POC_SHA" >&2
    exit 3
fi
git -C "$WORKSPACE/repo" config user.name "lop-poc"
git -C "$WORKSPACE/repo" config user.email "noreply@lop-poc.invalid"
stamp t_clone_done

# ------------------------------------------------------------ step 4: probes
# BEFORE the agent, deliberately: a script is a deterministic instrument.
# A probe that reports pass=false does NOT abort the run — it is a measurement,
# and the run that discovered it is the run worth keeping — but it IS reflected
# in the exit code through PROBE_RC below, and in probes.json["failed"].
stamp t_probes_start
PROBE_RC=0
printf '%s' "$MODEL_KEY" | /opt/probe/bin/python /opt/probe/probes.py \
    --out "$OUT/probes.json" \
    --key-prefix-chars "$KEY_PREFIX_CHARS" \
    --model-secret-arn "${POC_MODEL_SECRET_ARN:-}" || PROBE_RC=$?
stamp t_probes_done
printf '{"probe_rc":%s}\n' "$PROBE_RC" >"$OUT/probe_rc.json"
put_file "$POC_PROBES_URL" "$OUT/probes.json"

# ------------------------------------------------------------- step 5: agent
stamp t_agent_start
mkfifo "$FIFO"
(
    first_model_event_seen=0
    while IFS= read -r line; do
        printf '%s\n' "$line" >>"$OUT/exec.jsonl"
        printf '%s\n' "$line"
        # t_first_model_event is the wall time the FIRST MODEL-PRODUCED event
        # appeared. The user echo is also a `message_start` and arrives first, so
        # the match is on the event type AND the assistant role — an assistant
        # `message_start`, a delta, or a reasoning delta, and nothing else.
        if [ "$first_model_event_seen" = 0 ] &&
            printf '%s' "$line" | grep -Eq \
                '"type": "message_start", "message": \{"role": "assistant"|"type": "message_update"|"type": "reasoning_delta"'; then
            first_model_event_seen=1
            stamp t_first_model_event
        fi
    done <"$FIFO"
) &
reader_pid=$!

agent_rc=0
if [ "$POC_MOCK" = "1" ]; then
    # `-hosting test` is lop's own mock wire (allows_missing_api_key), so a mock
    # run needs no key: it exercises lifecycle, probes and cold start, and it
    # produces no fix — which is expected and stated in the spec.
    (
        cd "$WORKSPACE/repo"
        exec timeout "$AGENT_DEADLINE_SECONDS" \
            lop exec --json --tools read,write,edit,bash \
            --hosting test --model test-model "$POC_PROMPT"
    ) </dev/null >"$FIFO" 2>"$OUT/agent_stderr.txt" || agent_rc=$?
else
    # The key is exported in a SUBSHELL, then exec'd, so it is in the environment
    # of exactly one process tree and never in any argv vector (`env KEY=… prog`
    # would put it in `ps` output for every process on the task).
    (
        export "$PROVIDER_ENV=$MODEL_KEY"
        cd "$WORKSPACE/repo"
        exec timeout "$AGENT_DEADLINE_SECONDS" \
            lop exec --json --tools read,write,edit,bash \
            --hosting "$POC_HOSTING" --model "$POC_MODEL" "$POC_PROMPT"
    ) </dev/null >"$FIFO" 2>"$OUT/agent_stderr.txt" || agent_rc=$?
fi
wait "$reader_pid" || true
if [ "$agent_rc" = 124 ]; then
    stamp agent_timeout_fired
fi
stamp t_agent_end
printf '{"agent_rc":%s}\n' "$agent_rc" >"$OUT/agent_rc.json"

# ------------------------------------------------------------ step 6: commit
stamp t_commit_start
cd "$WORKSPACE/repo"
git checkout -q -b "lop/$POC_RUN_ID"
git add -A
COMMIT_SHA=""
if git diff --cached --quiet; then
    # No changes is a legitimate outcome of a mock run; it is recorded as null
    # rather than papered over with an empty commit.
    :
else
    git commit -q -m "lop-poc: $POC_RUN_ID"
    COMMIT_SHA="$(git rev-parse HEAD)"
fi
BUNDLE=""
if [ -n "$COMMIT_SHA" ]; then
    # A RANGE, so the bundle carries POC_SHA as its prerequisite and
    # `git bundle verify` succeeds in a fresh clone of the fixture (which has
    # POC_SHA but not our commit).
    git bundle create "$OUT/repo.bundle" "$POC_SHA..lop/$POC_RUN_ID"
    BUNDLE="repo.bundle"
fi
printf '{"branch":"lop/%s","commit":%s,"bundle":%s}\n' \
    "$POC_RUN_ID" \
    "$([ -n "$COMMIT_SHA" ] && printf '"%s"' "$COMMIT_SHA" || printf 'null')" \
    "$([ -n "$BUNDLE" ] && printf '"%s"' "$BUNDLE" || printf 'null')" \
    >"$OUT/git.json"
stamp t_commit_done

# --------------------------------------------------- step 7: session + status
stamp t_session_start
SESSION_ID="$(/opt/lop/bin/python - "$OUT/exec.jsonl" <<'PY'
import json
import sys

for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    line = line.strip()
    if not line:
        continue
    try:
        event = json.loads(line)
    except ValueError:
        continue
    session_id = event.get("session_id")
    if session_id:
        print(session_id)
        break
PY
)"
if [ -n "$SESSION_ID" ] && [ -d "$LOCAL_OPERATOR_CONFIG_DIR/sessions/$SESSION_ID" ]; then
    tar -czf "$OUT/session.tar.gz" -C "$LOCAL_OPERATOR_CONFIG_DIR/sessions" "$SESSION_ID"
fi
/opt/lop/bin/python - "$OUT/status.json" "$SESSION_ID" <<'PY'
"""Write what the jobs ledger holds for this session.

A FOREGROUND `lop exec` keeps no durable job row (the ledger is written by the
detached worker), so `lop exec --status <JOB_ID>` has nothing to read for this
run. Rather than invent a job, this records the session id, the fact that there
is no job id, and every ledger row that names the session — which is the honest
answer to "what does the ledger hold?".
"""
import json
import sys

session_id = sys.argv[2]
payload = {
    "session_id": session_id,
    "job_id": None,
    "note": (
        "a foreground `lop exec` has no durable job row; `lop exec --status <JOB_ID>` "
        "requires a --background run, so this file carries the jobs ledger's own rows "
        "for the session instead (spec §9.2 step 7 anticipates this divergence)"
    ),
    "job_records": [],
}
try:
    from local_operator.exec_mode import read_job_records

    payload["job_records"] = [
        row for row in read_job_records() if row.get("session_id") == session_id
    ]
except Exception as error:  # noqa: BLE001 - a ledger read failure is a fact to record
    payload["ledger_error"] = f"{type(error).__name__}: {error}"
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
PY
stamp t_session_done

# ------------------------------------------------- step 8: rescan before upload
stamp t_rescan_start
SCAN_RC=0
scan_out="$(printf '%s' "$MODEL_KEY" | /opt/probe/bin/python /opt/probe/probes.py \
    --key-prefix-chars "$KEY_PREFIX_CHARS" --scan-dir "$OUT")" || SCAN_RC=$?
if [ "$SCAN_RC" -ne 0 ]; then
    # Refuse the upload rather than shipping whatever contains the key. The
    # refusal is recorded without the key itself, and the task exits non-zero.
    printf '{"refused_upload":true,"scan":%s}\n' "$scan_out" >"$OUT/key_scan.json"
    echo "REFUSING to upload results: the model key's bytes appear under $OUT" >&2
    unset MODEL_KEY
    exit 4
fi
printf '{"refused_upload":false,"scan":%s}\n' "$scan_out" >"$OUT/key_scan.json"
unset MODEL_KEY
stamp t_rescan_done

# ------------------------------------------------------------ step 9: upload
stamp t_upload_start
# BEFORE the tarball: see `on_exit` for why this cannot be the trap's job alone.
fold_timings
tar -czf "$WORKSPACE/results.tar.gz" -C "$OUT" .
put_file "$POC_RESULTS_URL" "$WORKSPACE/results.tar.gz"
stamp t_upload_done

# The exit code answers "did the pipeline work AND did the isolation hold":
# a probe that reports pass=false is a failed acceptance claim, so it makes the
# run non-zero rather than being buried in a JSON artifact nobody reads. 4c's
# pass=null (no key in the run) is NOT a failure and does not set this.
if [ "$PROBE_RC" -ne 0 ]; then
    echo "running with PROBE_RC=$PROBE_RC: at least one isolation probe FAILED" >&2
    exit 5
fi
exit "$agent_rc"
