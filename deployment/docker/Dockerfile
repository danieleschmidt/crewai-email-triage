FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r app && useradd -r -g app app

# Set work directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .
COPY requirements-dev.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY triage.py .
COPY README.md .
COPY pyproject.toml .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/metrics
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.insert(0, '/app/src'); from crewai_email_triage.resilience import resilience; print(resilience.health_check.get_overall_health()['overall_status']); exit(0 if resilience.health_check.get_overall_health()['overall_status'] in ['healthy', 'degraded'] else 1)"

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "triage.py", "--help"]