FROM python:3.12-slim

# Set environment variables
ENV PORT=1111
ENV HOME=/home/appuser

RUN apt-get update && apt-get install -y \
    make \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m appuser

# Create necessary directories
RUN mkdir -p ${HOME}/.local-operator ${HOME}/local-operator-home && \
    chown -R appuser:appuser ${HOME}

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Install the local-operator package in development mode
RUN pip install -e .

EXPOSE ${PORT}

# Switch to non-root user
USER appuser

# Run the server with the specified port
CMD ["sh", "-c", "make dev-server PORT=${PORT}"]
