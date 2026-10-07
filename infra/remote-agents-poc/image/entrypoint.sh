#!/usr/bin/env bash
# Slice-0 POC entrypoint: the container half of docs/design/remote-cloud-agents.md
# §9.2 (steps 4 and 5, plus the commit/bundle/upload the driver consumes).
#
# Runs as uid 10001 with a READ-ONLY root filesystem, so /workspace is the only
# writable mount and every path this script writes lives under it.
#
# NEVER add `set -x`. From the first line to the last this script holds the model
# key in a shell variable, and a trace would put it in CloudWatch — the one place
# it must never reach. The task definition injects it as an ECS secret, it is taken
# out of the environment HERE before anything else runs, and from then on it reaches
# the agent only over a file descriptor, through image/lop_launch.py, which sets it
# in-process (see the key-delivery block below).
set -euo pipefail

# THE KEY LEAVES THE ENVIRONMENT BEFORE ANYTHING ELSE — before the uid check and
# before the first `stamp`, because both used to spawn a child (`id -u`, `date`)
# while the value was still in this process's environment. A child's INITIAL
# environment is a file the model's own bash child can read (`/proc/$PPID/environ`),
# so those two children were the last copies of the exposure this POC closes (agent
# review round 2, SEC-12). `$EUID` is a bash builtin and spawns nothing.
MODEL_KEY="${LOP_POC_MODEL_KEY:-}"
unset LOP_POC_MODEL_KEY

# NO privilege drop here, and that is the point: the task definition runs this as
# uid 10001 from the first instruction, because the image declares
# `VOLUME ["/workspace"]` over a /workspace it already owns — which makes the ECS
# agent copy that ownership into the task volume (see the Dockerfile). An earlier
# revision ran a root `phase 0` that chowned the volume and re-exec'd itself; it
# worked, and it is gone because it should not have been necessary. Probe 4d
# asserts uid 10001 on every run, so a regression here fails the run rather than
# the isolation claim.
if [ "$EUID" != "10001" ]; then
    echo "FATAL: expected uid 10001, got $EUID: the non-root contract is broken" >&2
    exit 1
fi

readonly OUT=/workspace/out
readonly WORKSPACE=/workspace
readonly FIFO=/workspace/tmp/stream.fifo
#: Where the key crosses the re-exec. A FIFO, so the value never reaches a file.
readonly KEY_FIFO=/workspace/tmp/key.pipe
# 2 h, the bound §9.3's cost estimate assumes. Firing it is recorded, not hidden.
readonly AGENT_DEADLINE_SECONDS=7200

mkdir -p "$OUT" "$WORKSPACE/repo" "$WORKSPACE/tmp" "$WORKSPACE/home" \
    "${LOCAL_OPERATOR_CONFIG_DIR:-$WORKSPACE/config/local-operator}"

# Every step stamps an epoch-millisecond line into timings.jsonl, and the EXIT trap
# folds that into timings.json.
#
# WHICH FAILURES THAT ACTUALLY SAVES, named exactly: the ones AFTER the results
# tarball is sealed — a non-zero agent exit (rc), a probe that failed (exit 5). It
# does NOT save the earlier ones, because $OUT only reaches S3 at step 9: the
# pre-upload key-scan refusal (exit 4, deliberate), a failed `git clone`, a failed
# probes PUT, and any other pre-upload abort ship no timings at all. Stated rather
# than implied, because the fold reads like a guarantee it cannot give.
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

# ------------------------------------------------------------- key delivery
#
# THE KEY NEVER ENTERS ANY PROCESS'S ENVIRONMENT, and this block is why.
# `local_operator/tools/shell_env.py` states the boundary this closes: the strict
# mode removes the key from a child's OWN environment and does not make it
# unreadable, because "Linux — `cat /proc/$PPID/environ` does the same", and
# unsetting in place "closes NOTHING, because ps and /proc/PID/environ report the
# environment a process was STARTED with". So the ECS-injected value is taken out of
# the environment here, carried across a re-exec on a pipe, and handed to the
# launcher over a descriptor — never argv, never the environment.
#
# PHASE A is the only process that ever HAS the value in its environment, and it
# does not survive: it execs itself away.
if [ -z "${LOP_POC_KEY_FD:-}" ]; then
    rm -f "$KEY_FIFO"
    mkfifo -m 600 "$KEY_FIFO"
    # The writer is a fork holding the value in its memory; it exits the moment the
    # reader opens, milliseconds from now, and it execs nothing.
    ( printf '%s' "$MODEL_KEY" >"$KEY_FIFO" ) &
    exec 3<"$KEY_FIFO"
    export LOP_POC_KEY_FD=3
    export LOP_POC_KEY_LEN=${#MODEL_KEY}
    stamp t_key_read
    # Re-exec THIS script so /proc/<pid>/environ — the image the kernel copied at
    # exec — stops carrying the key. Without it the entrypoint is one more process
    # whose environment the agent's children can read, and probe 4e goes red on it.
    exec "$0" "$@"
fi

# PHASE B: a clean environment image, and the key only in this shell's memory.
stamp t_entrypoint_rescrubbed
MODEL_KEY="$(head -c "$LOP_POC_KEY_LEN" <&3)"
exec 3<&-
rm -f "$KEY_FIFO"
if [ "${#MODEL_KEY}" != "$LOP_POC_KEY_LEN" ]; then
    echo "FATAL: key delivery truncated (${#MODEL_KEY} of $LOP_POC_KEY_LEN bytes)" >&2
    exit 6
fi
unset LOP_POC_KEY_FD LOP_POC_KEY_LEN

# ------------------------------------------------------- step 2: hosting + config

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
/opt/probe/bin/python /opt/probe/probes.py \
    --out "$OUT/probes.json" \
    --key-fd 3 \
    --key-prefix-chars "$KEY_PREFIX_CHARS" \
    --model-secret-arn "${POC_MODEL_SECRET_ARN:-}" \
    3< <(printf '%s' "$MODEL_KEY") || PROBE_RC=$?
stamp t_probes_done
printf '{"probe_rc":%s}\n' "$PROBE_RC" >"$OUT/probe_rc.json"
put_file "$POC_PROBES_URL" "$OUT/probes.json"

# ---------------------------------------- watcher self-tests (on demand, never by
# the driver): PROVE THE INSTRUMENT AND THE COEXISTENCE CASE, in this container,
# without a rebuild.
#
# POC_ENVIRON_WATCH_SELFTEST=1 — the RED case. A child is launched THE OLD WAY (the key
# exported into its environment, which is what SEC-1 replaced) and the watcher must
# FIND it; its exit code (1) is the proof.
#
# POC_ENVIRON_WATCH_COEXIST=1 — the COEXISTENCE case the five mock runs do not cover
# (they spawn no tool child at all; agent review round 2, SEC-13). The real launcher,
# holding the key in its memory, spawns a bash child through the PRODUCT'S OWN filter
# (`shell_env.child_environment`, what the bash and eval tools use) while the watcher
# samples; the watcher must stay GREEN.
#
# Both modes upload their artifacts before exiting, so the reading is re-derivable from
# the artifact set instead of only from CloudWatch.
if [ "${POC_ENVIRON_WATCH_SELFTEST:-0}" = "1" ] || [ "${POC_ENVIRON_WATCH_COEXIST:-0}" = "1" ]; then
    stamp t_watch_selftest_start
    child_rc=0
    if [ "${POC_ENVIRON_WATCH_SELFTEST:-0}" = "1" ]; then
        ( export LOP_POC_MODEL_KEY="$MODEL_KEY"; exec sleep 60 ) &
        probe_child_pid=$!
    else
        /opt/lop/bin/python /usr/local/bin/lop-launch.py --key-fd 3 \
            --provider-env LOP_POC_MODEL_KEY --selftest-child \
            3< <(printf '%s' "$MODEL_KEY") >"$OUT/selftest_child.json" 2>&1 &
        probe_child_pid=$!
    fi
    sleep 3
    selftest_rc=0
    /opt/probe/bin/python /opt/probe/probes.py --watch-environ \
        --out "$OUT/proc-env-watch.json" --stop-file "$WORKSPACE/tmp/never" \
        --key-fd 3 --key-prefix-chars 0 --max-samples 3 \
        3< <(printf '%s' "$MODEL_KEY") || selftest_rc=$?
    if [ "${POC_ENVIRON_WATCH_SELFTEST:-0}" = "1" ]; then
        kill "$probe_child_pid" 2>/dev/null || true
    else
        wait "$probe_child_pid" || child_rc=$?
    fi
    tar -czf "$WORKSPACE/selftest-results.tar.gz" -C "$OUT" .
    put_file "$POC_RESULTS_URL" "$WORKSPACE/selftest-results.tar.gz" || true
    echo "watcher self-test rc=$selftest_rc (0 green / 1 red); child rc=$child_rc; artifacts uploaded"
    exit "$selftest_rc"
fi

# ------------------------------------------------- probe 4e: the environ watcher
# Started before the agent and stopped when it ends: it samples every process's
# INITIAL environment for the key, which is the read path that matters while the
# agent — and anything the model spawns — is alive. It runs as the same uid and is a
# SIBLING of the agent, which is also what makes its /proc/<agent>/mem probe a
# same-uid NON-descendant read (the case yama/ptrace_scope decides).
WATCH_STOP="$WORKSPACE/tmp/agent.done"
rm -f "$WATCH_STOP"
AGENT_PID_FILE="$WORKSPACE/tmp/agent.pid"
rm -f "$AGENT_PID_FILE"
(
    /opt/probe/bin/python /opt/probe/probes.py \
        --watch-environ --out "$OUT/proc-env-watch.json" --stop-file "$WATCH_STOP" \
        --key-fd 3 --key-prefix-chars "$KEY_PREFIX_CHARS" \
        --agent-pid-file "$AGENT_PID_FILE" \
        3< <(printf '%s' "$MODEL_KEY")
) &
watcher_pid=$!

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
    # `--hosting test` is lop's own mock wire (allows_missing_api_key), so a mock run
    # needs no key: it exercises lifecycle, probes and cold start, and it produces no
    # fix — which is expected. It still goes through the LAUNCHER with the key on a
    # descriptor, so the five acceptance runs exercise the real delivery path (with
    # the placeholder as the value) rather than a path only a real key would take.
    (
        cd "$WORKSPACE/repo"
        exec timeout "$AGENT_DEADLINE_SECONDS" \
            /opt/lop/bin/python /usr/local/bin/lop-launch.py \
            --key-fd 3 --provider-env "" --pid-file "$AGENT_PID_FILE" \
            -- exec --json --tools read,write,edit,bash \
            --hosting test --model test-model "$POC_PROMPT"
    ) 3< <(printf '%s' "$MODEL_KEY") </dev/null >"$FIFO" 2>"$OUT/agent_stderr.txt" || agent_rc=$?
else
    # The launcher reads the key from fd 3 and sets it IN-PROCESS: it is in no
    # process's initial environment, so the model's own bash child cannot read it out
    # of its parent with `cat /proc/$PPID/environ`. See the launcher's module
    # docstring, and probes.py's 4e watcher for the measurement.
    (
        cd "$WORKSPACE/repo"
        exec timeout "$AGENT_DEADLINE_SECONDS" \
            /opt/lop/bin/python /usr/local/bin/lop-launch.py \
            --key-fd 3 --provider-env "$PROVIDER_ENV" --pid-file "$AGENT_PID_FILE" \
            -- exec --json --tools read,write,edit,bash \
            --hosting "$POC_HOSTING" --model "$POC_MODEL" "$POC_PROMPT"
    ) 3< <(printf '%s' "$MODEL_KEY") </dev/null >"$FIFO" 2>"$OUT/agent_stderr.txt" || agent_rc=$?
fi
wait "$reader_pid" || true
if [ "$agent_rc" = 124 ]; then
    stamp agent_timeout_fired
fi
stamp t_agent_end
printf '{"agent_rc":%s}\n' "$agent_rc" >"$OUT/agent_rc.json"

# Stop the watcher and read its verdict. A watcher that FOUND the key in some
# process's environment is a failed isolation claim, so it joins PROBE_RC and makes
# the run exit 5 — the same treatment 4a-4d get.
touch "$WATCH_STOP"
watch_rc=0
wait "$watcher_pid" || watch_rc=$?
stamp t_watch_done
printf '{"watch_rc":%s}\n' "$watch_rc" >"$OUT/watch_rc.json"
if [ "$watch_rc" -ne 0 ]; then
    PROBE_RC=1
fi

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
scan_out="$(/opt/probe/bin/python /opt/probe/probes.py --key-fd 3 --scan-dir "$OUT" \
    --key-prefix-chars "$KEY_PREFIX_CHARS" 3< <(printf '%s' "$MODEL_KEY"))" || SCAN_RC=$?
if [ "$SCAN_RC" -eq 2 ]; then
    # rc 2 is "no key was delivered", not "clean": a scan that inspected nothing
    # must never read as a scan that found nothing (AGENTS.md, "A dead instrument
    # returns a reading, not an error").
    printf '{"refused_upload":true,"reason":"no key delivered to the rescan"}\n' \
        >"$OUT/key_scan.json"
    echo "REFUSING to upload results: no key reached the rescan, so nothing was inspected" >&2
    unset MODEL_KEY
    exit 4
fi
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
