# =============================================================================
# MULTI-STAGE DOCKERFILE FOR CREWAI EMAIL TRIAGE
# =============================================================================

# =============================================================================
# BUILD STAGE - Development dependencies and build tools
# =============================================================================
FROM python:3.13-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies required for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create build user
RUN groupadd --gid 1000 builduser && \
    useradd --uid 1000 --gid builduser --shell /bin/bash --create-home builduser

# Set work directory
WORKDIR /build

# Copy dependency files first for better caching
COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE ./

# Install build dependencies
RUN pip install --upgrade pip setuptools wheel build

# Install package dependencies
RUN pip install -e .[dev,test]

# Copy source code
COPY src/ src/
COPY tests/ tests/
COPY triage.py ./

# Change ownership to build user
RUN chown -R builduser:builduser /build
USER builduser

# Build the package
RUN python -m build

# Run tests during build (optional, can be skipped with --build-arg SKIP_TESTS=true)
ARG SKIP_TESTS=false
RUN if [ "$SKIP_TESTS" = "false" ]; then python -m pytest tests/ -x; fi

# =============================================================================
# SECURITY SCAN STAGE - Vulnerability scanning
# =============================================================================
FROM builder as security-scanner

USER root

# Install security scanning tools
RUN pip install bandit safety

# Run security scans
RUN bandit -r src/ -f json -o /tmp/bandit-report.json || true
RUN safety check --json --output /tmp/safety-report.json || true

# Switch back to build user
USER builduser

# =============================================================================
# PRODUCTION STAGE - Minimal runtime image
# =============================================================================
FROM python:3.13-slim as production

# Set production arguments and labels
ARG BUILD_DATE
ARG VERSION=0.1.0
ARG VCS_REF

LABEL org.opencontainers.image.title="CrewAI Email Triage" \
      org.opencontainers.image.description="Smart email assistant that classifies, prioritizes, summarizes, and drafts replies" \
      org.opencontainers.image.authors="CrewAI Email Triage Team" \
      org.opencontainers.image.vendor="CrewAI" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.source="https://github.com/crewai/email-triage" \
      org.opencontainers.image.documentation="https://crewai-email-triage.readthedocs.io" \
      org.opencontainers.image.licenses="MIT"

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CREWAI_ENV=production \
    LOG_LEVEL=INFO

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application user and group with specific UID/GID
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid appuser --shell /bin/bash --create-home appuser

# Create application directory
WORKDIR /app

# Copy built wheel from builder stage
COPY --from=builder /build/dist/*.whl /tmp/

# Install the application
RUN pip install --upgrade pip && \
    pip install /tmp/*.whl && \
    rm -rf /tmp/*.whl

# Create directories for application data
RUN mkdir -p /app/data /app/logs /app/config && \
    chown -R appuser:appuser /app

# Copy application files
COPY --chown=appuser:appuser triage.py /app/
COPY --chown=appuser:appuser src/crewai_email_triage/default_config.json /app/config/

# Switch to application user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "from crewai_email_triage.core import health_check; health_check()" || exit 1

# Expose port for health checks and metrics (if applicable)
EXPOSE 8000

# Set default command
ENTRYPOINT ["python", "triage.py"]
CMD ["--help"]

# =============================================================================
# DEVELOPMENT STAGE - For development and testing
# =============================================================================
FROM builder as development

USER root

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    less \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python development tools
RUN pip install \
    ipython \
    jupyter \
    pre-commit

# Set development environment variables
ENV CREWAI_ENV=development \
    LOG_LEVEL=DEBUG \
    PYTHONPATH=/app/src

# Create development user
RUN usermod -s /bin/bash builduser

# Copy development configuration
COPY --chown=builduser:builduser .env.example /app/.env.example
COPY --chown=builduser:builduser .pre-commit-config.yaml /app/
COPY --chown=builduser:builduser Makefile /app/

WORKDIR /app

# Install pre-commit hooks
USER builduser
RUN pre-commit install

# Default command for development
CMD ["bash"]

# =============================================================================
# TESTING STAGE - For CI/CD testing
# =============================================================================
FROM development as testing

USER builduser

# Copy test configuration
COPY --chown=builduser:builduser pytest.ini /app/ 2>/dev/null || true
COPY --chown=builduser:builduser .coveragerc /app/ 2>/dev/null || true

# Install testing dependencies
RUN pip install -e .[test,performance]

# Set testing environment
ENV CREWAI_ENV=testing \
    PYTEST_ADDOPTS="--tb=short --strict-markers"

# Run tests by default
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=src/crewai_email_triage"]

# =============================================================================
# DOCUMENTATION STAGE - For building documentation
# =============================================================================
FROM python:3.13-slim as documentation

# Install documentation dependencies
RUN pip install \
    mkdocs \
    mkdocs-material \
    mkdocstrings[python]

WORKDIR /docs

# Copy documentation source
COPY docs/ ./
COPY README.md ./
COPY pyproject.toml ./

# Build documentation
RUN mkdocs build

# Serve documentation
EXPOSE 8000
CMD ["mkdocs", "serve", "--dev-addr", "0.0.0.0:8000"]