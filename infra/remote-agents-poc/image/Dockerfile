# Slice-0 POC image: a lop that runs one task in Fargate and nothing else.
# Spec: docs/design/remote-cloud-agents.md §9.2 step 2.
#
# WHY the base is pinned by INDEX DIGEST and not by tag: §7.1 requires the image
# to be "pinned by digest", and a floating `python:3.12-slim` tag would let a
# rebuild move the substrate under an unchanged task definition. This is the
# arm64-bearing multi-arch index resolved 2026-10-07.
FROM python:3.12-slim@sha256:05cda9777409a9c3ffddd94a4c476b79f0769a0b4857f0c7ed9226b6800b0d6f

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# git for the clone/commit/bundle (entrypoint steps 3 and 6), curl for the two
# presigned PUTs (steps 4 and 9), ca-certificates for both. Nothing else is
# installed because nothing else is used.
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git \
 && rm -rf /var/lib/apt/lists/*

# FAIL THE BUILD IF THIS IMAGE EVER GAINS A `ps`.
#
# Two spawn sites in the product hand the CALLER's environment to `ps`:
# tools/group_reaper.py (`ps -o lstart= -p <pid>` with `env={**os.environ, "LC_ALL": "C"}`)
# and memory_guard._default_runner (`ps -axo pid=,pgid=,rss=`, no `env=` at all, on every
# tick of every guarded command). Either child is a child of the process holding the
# model key, so its own /proc/<pid>/environ would carry the key and the model's same-uid
# bash child could read that file — the read path this POC closed. "No process's initial
# environment carries the key" is therefore CONDITIONAL on there being no `ps` to run,
# and this guard turns that condition into a build failure: adding procps for any reason
# (a debugging tool, a base-image change) stops here rather than silently reopening it.
# Probe 4f asserts the same thing at run time, because the runtime is what the claim is
# about. Closing it product-side — a filtered env at those two sites — is deferred work
# (see the PR thread).
RUN ! command -v ps \
 && ! test -e /bin/ps && ! test -e /usr/bin/ps && ! test -e /sbin/ps \
 && ! test -e /usr/sbin/ps && ! test -e /usr/local/bin/ps \
 && ! test -e /usr/local/sbin/ps \
 && ! command -v busybox && ! command -v toybox
# ^ THE SAME SIX PATHS probe 4f publishes, plus the two multiplexers. `/usr/local/bin`
# is where this Dockerfile itself puts `uv` and the `lop` symlink, so it is exactly the
# directory a future `PATH` change would expose — the build-time guard must not be the
# weaker of the two lists (agent review round 3, r3-4).

# uv, pinned by digest for the same reason as the base, and copied from the
# publisher's own image rather than pip-installed so the binary is the artifact
# upstream signed off on.
COPY --from=ghcr.io/astral-sh/uv:0.11.1@sha256:fc93e9ecd7218e9ec8fba117af89348eef8fd2463c50c13347478769aaedd0ce /uv /uvx /usr/local/bin/

# TWO venvs, deliberately. /opt/lop is the PRODUCT: exactly requirements.txt,
# hash-locked, --require-hashes, nothing added. /opt/probe holds boto3 for
# probes.py only, so adding an instrument cannot change what the thing under
# test has installed (§9.2 step 4 depends on the probes being an instrument
# rather than a perturbation).
COPY requirements.txt /tmp/requirements.txt
COPY requirements-probe.txt /tmp/requirements-probe.txt
RUN python -m venv /opt/lop \
 && /opt/lop/bin/pip install --require-hashes -r /tmp/requirements.txt \
 && python -m venv /opt/probe \
 && /opt/probe/bin/pip install --require-hashes -r /tmp/requirements-probe.txt \
 && ln -s /opt/lop/bin/lop /usr/local/bin/lop \
 && rm -f /tmp/requirements.txt /tmp/requirements-probe.txt

COPY probes.py /opt/probe/probes.py
COPY lop_launch.py /usr/local/bin/lop-launch.py
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh /usr/local/bin/lop-launch.py \
 && chmod 0644 /opt/probe/probes.py

# uid 10001, no home of its own: HOME points into the task volume (see the task
# definition's environment block). -M so the image carries no /home entry.
RUN groupadd --gid 10001 lop \
 && useradd --uid 10001 --gid 10001 --no-create-home \
      --home-dir /workspace/home --shell /bin/bash lop

# /workspace is the task volume's mount point. It is created and chowned to 10001
# HERE, in an image layer, and then declared as a VOLUME — that pair is the whole
# mechanism: when a Dockerfile VOLUME path EQUALS the task definition's
# containerPath, the ECS agent copies the image's data AND ITS OWNERSHIP into the
# bind mount, so the container can start as uid 10001 and still write there
# (docs.aws.amazon.com/AmazonECS/latest/developerguide/bind-mounts.html;
# aws/containers-roadmap#938).
#
# ORDER MATTERS: `chown` must run BEFORE `VOLUME`, because changes to a VOLUME path
# made after the declaration are discarded at run time — the copy happens from the
# image layer as it stood when VOLUME was declared.
#
# Verified in the account, not taken on the agent's word: probe 4d reads uid 10001
# and writes /workspace on every run, and the task definition sets
# `user: "10001:10001"` with no privilege drop anywhere in the entrypoint.
RUN mkdir -p /workspace/home /workspace/tmp /workspace/out /workspace/config \
 && chown -R 10001:10001 /workspace
VOLUME ["/workspace"]

USER 10001:10001
WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
