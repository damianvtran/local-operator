# Local Operator on Amazon Bedrock AgentCore Runtime.
#
# Constraints from the AgentCore Runtime container contract:
#   * linux/arm64 only (Graviton microVMs) — deploy.sh builds this on an arm64
#     CodeBuild host, so nobody needs a local Docker daemon or a cross-build;
#   * listen on 0.0.0.0:8080 and serve GET /ping and POST /invocations
#     (BedrockAgentCoreApp does both).
#
# Pulled from public.ecr.aws rather than Docker Hub so the CodeBuild build is
# not subject to Docker Hub's anonymous pull rate limit.
FROM public.ecr.aws/docker/library/python:3.12-slim

# The PyPI release of Local Operator to run. Pinned per build so a runtime
# version always maps to one known lop release; deploy.sh --lop-version sets it.
ARG LOP_VERSION=0.92.1

# Tools the agent's own shell commonly reaches for inside the microVM. Kept
# short: every package here is attack surface and image size.
RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates curl git procps \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir "local-operator==${LOP_VERSION}" -r /app/requirements.txt

COPY main.py bedrock_bridge.py lop_runtime.py /app/

# Run as an unprivileged user. Its passwd home (/home/lop) deliberately differs
# from LOP_HOME/HOME (/var/lib/lop): lop's SDK refuses to open sessions in the
# uid's *default* store unless told the caller means it, and pointing HOME at
# the explicit root keeps lop's caches inside the declared roots too.
# /var/lib rather than /tmp: the SDK refuses volatile roots by default.
RUN useradd --create-home --home-dir /home/lop --uid 1000 lop \
    && mkdir -p /var/lib/lop \
    && chown -R lop:lop /var/lib/lop

ENV HOME=/var/lib/lop \
    LOP_HOME=/var/lib/lop \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DOCKER_CONTAINER=1 \
    PORT=8080

USER lop
WORKDIR /var/lib/lop
EXPOSE 8080
CMD ["python", "/app/main.py"]
