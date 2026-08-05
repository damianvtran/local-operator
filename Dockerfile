FROM python:3.13-slim

# Set environment variables
ENV PORT=1111
ENV HOME=/home/appuser

# `make` only: the CMD (and docker-compose) drive the server through the
# Makefile. No gcc/g++ — see the pip install note below.
RUN apt-get update && apt-get install -y --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m appuser

# Create necessary directories
RUN mkdir -p ${HOME}/.local-operator ${HOME}/local-operator-home && \
    chown -R appuser:appuser ${HOME}

# Set working directory
WORKDIR /app

# Copy the application
COPY . .

# The image serves the HTTP API, so it needs the `server` extra on top of the
# lean default install. Dependencies are declared in pyproject.toml only —
# there is no requirements.txt to keep in sync.
#
# No build toolchain is installed on purpose: every dependency in this set
# ships prebuilt wheels for CPython on Linux. If a `pip install` here ever
# starts compiling from source, that is a dependency regression to fix rather
# than a reason to add gcc back to the image.
RUN pip install --no-cache-dir -e ".[server]"

EXPOSE ${PORT}

# Switch to non-root user
USER appuser

# Run the server with the specified port
CMD ["sh", "-c", "make dev-server PORT=${PORT}"]
