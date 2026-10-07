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
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 0755 /usr/local/bin/entrypoint.sh \
 && chmod 0644 /opt/probe/probes.py

# uid 10001, no home of its own: HOME points into the task volume (see the task
# definition's environment block). -M so the image carries no /home entry.
RUN groupadd --gid 10001 lop \
 && useradd --uid 10001 --gid 10001 --no-create-home \
      --home-dir /workspace/home --shell /bin/bash lop

# /workspace is the task volume's mount point, and it is created AND chowned here
# so the image is right on its own (the CodeBuild smoke test runs against this
# image with no volume attached). At RUN time the volume shadows it and is mounted
# root-owned, which is why the entrypoint has a root phase that re-chowns it and
# then drops to uid 10001 — see the `phase 0` block there and README.md,
# "Divergences" #16.
RUN mkdir -p /workspace/home /workspace/tmp /workspace/out /workspace/config \
 && chown -R 10001:10001 /workspace

# NO `USER 10001:10001`, deliberately. It looks like the stricter choice and it is
# the one that cannot work: the task volume is root-owned, and a container that
# starts as 10001 can neither write it nor chown it. The entrypoint starts as
# root, chowns the volume, and re-execs itself as 10001, so every phase that
# touches untrusted input — probes, agent, everything the agent spawns — runs
# unprivileged. Probe 4d asserts uid 10001 at that point, on every run.
WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
