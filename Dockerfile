# SpiralDeltaDB Docker Image
# Multi-stage build for optimized production image

FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install SpiralDeltaDB in development mode
RUN pip install -e .

# Run tests to ensure build quality
RUN python basic_test.py

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash spiraldelta

# Create working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY --chown=spiraldelta:spiraldelta . .

# Create data directory
RUN mkdir -p /data && chown spiraldelta:spiraldelta /data

# Switch to non-root user
USER spiraldelta

# Set default data directory
ENV SPIRALDELTA_DATA_DIR=/data

# Expose default port (if needed for future web interface)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from spiraldelta import SpiralDeltaDB; print('✅ SpiralDeltaDB is healthy')" || exit 1

# Default command
CMD ["python", "-c", "from spiraldelta import SpiralDeltaDB; print('🌀 SpiralDeltaDB ready'); import time; time.sleep(3600)"]

# Development stage (can be used with --target development)
FROM production as development

# Switch back to root for dev dependencies
USER root

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

# Switch back to spiraldelta user
USER spiraldelta

# Development command with interactive shell
CMD ["/bin/bash"]

# Labels for metadata
LABEL maintainer="Monarch AI <engineering@monarchai.com>" \
      version="1.0.0" \
      description="SpiralDeltaDB - Geometric Vector Database" \
      license="AGPL-3.0" \
      repository="https://github.com/monarch-ai/spiraldelta-db"