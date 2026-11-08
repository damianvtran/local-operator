#!/usr/bin/env bash
# Install a pinned, checksum-verified ripgrep for CI (called by ci.yml "Install ripgrep").
#
# WHY a script: run 42463603282 lost four test shards before any test ran because
# github.com release downloads returned 504 four times in 7 s and the old inline
# `curl --retry 3` gave up. This retries 5xx/network errors with exponential
# backoff and, only when those are exhausted, falls back to a VERIFIED local rg.
#
# What is deliberately NOT weakened (stub-curl evidence is in the PR body):
#   * 403/404/other 4xx fail immediately: a wrong RG_VERSION must be loud.
#   * SHA256 of the downloaded tarball is always checked; a mismatch fails with no
#     fallback (a corrupt/tampered download is not a transient outage).
#   * No unverified binary is ever used: the fallback requires rg >= RG_VERSION
#     already on PATH, or a distro package install that succeeds; else exit 1 with
#     the original HTTP/curl error repeated.
#
# Env: RG_VERSION, RG_SHA256 (required); RUNNER_TEMP (default /tmp);
#      RG_ATTEMPTS (5), RG_SLEEP_BASE (2s -> 2/4/8/16), RG_MAX_TIME (15s per try)
#      so the worst case stays ~105 s; RG_SUDO ("sudo"), RG_DEST (/usr/local/bin)
#      are overridable for the stub tests.
set -euo pipefail

: "${RG_VERSION:?}" "${RG_SHA256:?}"
tmp="${RUNNER_TEMP:-/tmp}"
attempts="${RG_ATTEMPTS:-5}"
delay="${RG_SLEEP_BASE:-2}"
max_time="${RG_MAX_TIME:-15}"
sudo_cmd="${RG_SUDO-sudo}"
dest="${RG_DEST:-/usr/local/bin}"
name="ripgrep-${RG_VERSION}-x86_64-unknown-linux-musl"
url="https://github.com/BurntSushi/ripgrep/releases/download/${RG_VERSION}/${name}.tar.gz"
tgz="$tmp/rg.tgz"

last_err=""
ok=0
for ((i = 1; i <= attempts; i++)); do
  rc=0
  # No -f: we need the HTTP status to tell a transient 5xx from a permanent 4xx.
  code=$(curl -sSL --max-time "$max_time" -o "$tgz" -w '%{http_code}' "$url" 2>"$tmp/rg.curl.err") || rc=$?
  if [[ $rc -eq 0 && $code =~ ^2 ]]; then ok=1; break; fi
  if [[ $rc -eq 0 && $code =~ ^4 && $code != 408 && $code != 429 ]]; then
    echo "::error::ripgrep download failed with permanent HTTP $code for $url (check RG_VERSION)" >&2
    exit 1
  fi
  last_err="HTTP ${code:-000}, curl exit $rc: $(tr '\n' ' ' <"$tmp/rg.curl.err")"
  echo "attempt $i/$attempts failed ($last_err)" >&2
  if ((i < attempts)); then
    echo "retrying in ${delay}s" >&2
    sleep "$delay"
    delay=$((delay * 2))
  fi
done

if ((ok == 1)); then
  echo "${RG_SHA256}  $tgz" | sha256sum -c -
  tar -xzf "$tgz" -C "$tmp" "${name}/rg"
  $sudo_cmd install -m 0755 "$tmp/${name}/rg" "$dest/rg"
  rg --version
  exit 0
fi

echo "::warning::ripgrep download exhausted $attempts attempts ($last_err); trying verified fallback" >&2
# 1) Preinstalled rg that is at least the pinned version.
if command -v rg >/dev/null 2>&1; then
  have=$(rg --version | head -n1 | awk '{print $2}')
  if [[ -n $have && $(printf '%s\n%s\n' "$RG_VERSION" "$have" | sort -V | head -n1) == "$RG_VERSION" ]]; then
    echo "FALLBACK: using preinstalled rg $have (>= pinned $RG_VERSION)"
    rg --version
    exit 0
  fi
  echo "preinstalled rg ${have:-unknown} is older than pinned $RG_VERSION; not using it" >&2
fi
# 2) Distro package from the local apt index (no `apt-get update`: that is the
#    slow, unbounded step the pinned tarball replaced).
if $sudo_cmd timeout 60 apt-get install -y --no-install-recommends ripgrep >/dev/null 2>&1 && command -v rg >/dev/null 2>&1; then
  echo "FALLBACK: installed distro ripgrep $(rg --version | head -n1)"
  exit 0
fi
echo "::error::ripgrep unavailable: no verified fallback. Original error: $last_err" >&2
exit 1
