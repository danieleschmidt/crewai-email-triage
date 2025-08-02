# Deployment and Build Documentation

## Overview

CrewAI Email Triage provides multiple deployment options with comprehensive build automation, containerization, and infrastructure as code capabilities.

## Build System

### Make Commands

The project uses a comprehensive Makefile for build automation:

```bash
# Development setup
make setup                    # Full development environment setup
make install-dev             # Install with development dependencies
make install-all             # Install with all optional dependencies

# Code quality
make format                  # Format code with Black and Ruff
make lint                    # Run linting checks
make typecheck              # Run type checking with MyPy
make security               # Run security analysis with Bandit
make audit                  # Check dependencies for vulnerabilities
make check                  # Run all quality checks

# Testing
make test                   # Run all tests
make test-unit             # Run unit tests only
make test-integration      # Run integration tests
make test-cov              # Run tests with coverage report
make test-parallel         # Run tests in parallel
make benchmark             # Run performance benchmarks

# Build and release
make clean                 # Clean build artifacts
make build                 # Build distribution packages
make build-check           # Build and validate packages
make release               # Prepare for release

# Docker operations
make docker-build          # Build Docker image
make docker-run            # Run Docker container
make docker-clean          # Clean Docker artifacts

# Documentation
make docs                  # Build documentation
make serve-docs            # Serve documentation locally
```

### Automated Build Pipeline

The CI/CD simulation can be run locally:

```bash
make ci                    # Full CI pipeline simulation
make ci-fast               # Quick checks (lint, test)
```

## Containerization

### Multi-Stage Dockerfile

The project uses a sophisticated multi-stage Dockerfile with the following stages:

1. **Builder Stage**: Compiles dependencies and builds the package
2. **Security Scanner**: Runs security vulnerability scans
3. **Production Stage**: Minimal runtime image
4. **Development Stage**: Full development environment
5. **Testing Stage**: Optimized for running tests
6. **Documentation Stage**: Builds and serves documentation

### Docker Build Targets

```bash
# Production build (default)
docker build -t crewai-email-triage:latest .

# Development build
docker build --target development -t crewai-email-triage:dev .

# Testing build
docker build --target testing -t crewai-email-triage:test .

# Documentation build
docker build --target documentation -t crewai-email-triage:docs .

# Security scanning
docker build --target security-scanner -t crewai-email-triage:security .
```

### Build Arguments

```bash
docker build \
  --build-arg BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
  --build-arg VERSION="1.0.0" \
  --build-arg VCS_REF="$(git rev-parse --short HEAD)" \
  --build-arg SKIP_TESTS=false \
  -t crewai-email-triage:latest .
```

## Docker Compose Orchestration

### Available Profiles

The docker-compose.yml supports multiple profiles for different deployment scenarios:

```bash
# Development environment
docker-compose up

# With monitoring stack
docker-compose --profile monitoring up

# With logging (ELK stack)
docker-compose --profile logging up

# Complete production-like setup
docker-compose --profile production --profile monitoring --profile logging up

# Testing environment
docker-compose --profile testing up

# Documentation server
docker-compose --profile docs up

# Security scanning
docker-compose --profile security up
```

### Core Services

- **crewai-triage**: Main application container
- **redis**: Caching and session storage
- **postgres**: Database for future features
- **prometheus**: Metrics collection
- **grafana**: Monitoring dashboards
- **nginx**: Reverse proxy for production
- **elasticsearch**: Log aggregation
- **kibana**: Log visualization
- **mailhog**: Email testing (development)

### Service Dependencies

```mermaid
graph TD
    A[crewai-triage] --> B[redis]
    A --> C[postgres]
    D[prometheus] --> A
    E[grafana] --> D
    F[nginx] --> A
    F --> E
    G[kibana] --> H[elasticsearch]
```

## Deployment Options

### 1. Local Development

```bash
# Quick start
git clone <repository>
cd crewai-email-triage
make setup
python triage.py --help

# Docker development
docker-compose up
```

### 2. Docker Production

```bash
# Build production image
docker build -t crewai-email-triage:prod .

# Run with environment variables
docker run -d \
  --name crewai-triage \
  -p 8000:8000 \
  -e GMAIL_USER=your-email@gmail.com \
  -e GMAIL_PASSWORD=your-app-password \
  -v /path/to/config:/app/config \
  -v /path/to/logs:/app/logs \
  crewai-email-triage:prod
```

### 3. Docker Compose Production

```bash
# Production deployment
cp .env.example .env
# Edit .env with production values

docker-compose --profile production up -d
```

### 4. Kubernetes Deployment

```bash
# Using Helm (future enhancement)
helm install crewai-triage ./k8s/charts/crewai-triage

# Using kubectl with manifests
kubectl apply -f k8s/manifests/
```

### 5. Cloud Platforms

#### AWS ECS
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ecr-repo>
docker build -t <ecr-repo>/crewai-email-triage:latest .
docker push <ecr-repo>/crewai-email-triage:latest

# Deploy with ECS task definition
aws ecs update-service --cluster crewai-cluster --service crewai-triage --force-new-deployment
```

#### Google Cloud Run
```bash
# Deploy to Cloud Run
gcloud builds submit --tag gcr.io/PROJECT-ID/crewai-email-triage
gcloud run deploy --image gcr.io/PROJECT-ID/crewai-email-triage --platform managed
```

#### Azure Container Instances
```bash
# Deploy to ACI
az acr build --registry myregistry --image crewai-email-triage .
az container create --resource-group myResourceGroup --name crewai-triage --image myregistry.azurecr.io/crewai-email-triage:latest
```

## Environment Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Core application
GMAIL_USER=your-email@gmail.com
GMAIL_PASSWORD=your-app-password
CREWAI_CONFIG=/app/config/custom_config.json
LOG_LEVEL=INFO

# Performance
MAX_CONCURRENT_EMAILS=10
BATCH_SIZE=50
TIMEOUT_SECONDS=30

# Security
ENCRYPTION_KEY=your-encryption-key
RATE_LIMIT_PER_MINUTE=60

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=8080
PROMETHEUS_GATEWAY_URL=http://prometheus:9090

# Database (for future features)
DATABASE_URL=postgresql://crewai:password@postgres:5432/crewai_email_triage
REDIS_URL=redis://redis:6379/0
```

### Configuration Files

```bash
# Application configuration
src/crewai_email_triage/default_config.json  # Default settings
config/production.json                       # Production overrides
config/development.json                      # Development settings
```

## Health Checks and Monitoring

### Health Endpoints

```bash
# Application health
curl http://localhost:8000/health

# Detailed health check
python -c "from crewai_email_triage.core import health_check; print(health_check())"

# Metrics endpoint (if enabled)
curl http://localhost:8080/metrics
```

### Monitoring Stack

When using the monitoring profile:

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Application Metrics**: http://localhost:8080/metrics

### Logging

Structured logging with JSON format:

```json
{
  "timestamp": "2025-08-02T10:30:00Z",
  "level": "INFO",
  "logger": "crewai.classifier",
  "message": "Email classified successfully",
  "email_id": "msg_123",
  "category": "urgent",
  "confidence": 0.92,
  "processing_time": 0.145
}
```

## Security Considerations

### Container Security

- **Non-root user**: Runs as `appuser` (UID 1001)
- **Minimal base image**: python:3.11-slim
- **Security scanning**: Bandit and Safety integrated in build
- **No secrets in image**: Environment variables only
- **Read-only filesystem**: Where possible

### Network Security

```yaml
# Docker Compose network isolation
networks:
  crewai-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

### Secret Management

```bash
# Using Docker secrets (Swarm mode)
echo "secret-value" | docker secret create gmail-password -
docker service create --secret gmail-password crewai-email-triage

# Using Kubernetes secrets
kubectl create secret generic crewai-secrets \
  --from-literal=gmail-password=your-password
```

## Performance Optimization

### Build Optimization

- **Multi-stage builds**: Minimize final image size
- **Layer caching**: Efficient Docker layer reuse
- **Build parallelization**: Concurrent test execution
- **Dependency caching**: pip cache optimization

### Runtime Optimization

```bash
# Resource limits in Docker Compose
services:
  crewai-triage:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
```

### Scaling

```bash
# Horizontal scaling with Docker Compose
docker-compose up --scale crewai-triage=3

# Load balancing with nginx
# See nginx/nginx.conf for configuration
```

## Troubleshooting

### Common Issues

1. **Container won't start**
   ```bash
   # Check logs
   docker logs crewai-email-triage
   
   # Check health
   docker exec crewai-email-triage python -c "from crewai_email_triage.core import health_check; print(health_check())"
   ```

2. **Email authentication failures**
   ```bash
   # Verify environment variables
   docker exec crewai-email-triage env | grep GMAIL
   
   # Test connection
   docker exec crewai-email-triage python -c "from crewai_email_triage.provider import GmailProvider; print(GmailProvider().test_connection())"
   ```

3. **Performance issues**
   ```bash
   # Check resource usage
   docker stats crewai-email-triage
   
   # Enable debug logging
   docker exec crewai-email-triage -e LOG_LEVEL=DEBUG
   ```

4. **Database connection issues**
   ```bash
   # Check PostgreSQL health
   docker exec crewai-postgres pg_isready -U crewai
   
   # Check Redis connectivity
   docker exec crewai-redis redis-cli ping
   ```

### Debug Mode

```bash
# Run in debug mode
docker run -it --rm \
  -e LOG_LEVEL=DEBUG \
  -e CREWAI_ENV=development \
  crewai-email-triage:dev bash

# Access development shell
docker-compose exec crewai-triage bash
```

## Backup and Recovery

### Data Backup

```bash
# Backup PostgreSQL data
docker exec crewai-postgres pg_dump -U crewai crewai_email_triage > backup.sql

# Backup Redis data
docker exec crewai-redis redis-cli --rdb /tmp/dump.rdb
docker cp crewai-redis:/tmp/dump.rdb ./redis-backup.rdb

# Backup application logs
docker cp crewai-email-triage:/app/logs ./logs-backup
```

### Configuration Backup

```bash
# Backup configuration
docker cp crewai-email-triage:/app/config ./config-backup

# Backup environment
docker inspect crewai-email-triage | jq '.[0].Config.Env' > env-backup.json
```

## Maintenance

### Updates

```bash
# Update application
docker build -t crewai-email-triage:latest .
docker-compose up -d --no-deps crewai-triage

# Update dependencies
make upgrade-deps
docker build --no-cache -t crewai-email-triage:latest .
```

### Cleanup

```bash
# Clean Docker artifacts
make docker-clean
docker system prune -a

# Clean application artifacts
make clean
```

### Monitoring

```bash
# Check application metrics
make metrics

# View container logs
docker-compose logs -f crewai-triage

# Monitor resource usage
docker stats --no-stream
```

## Migration and Upgrades

### Version Upgrades

```bash
# Backup before upgrade
./scripts/backup.sh

# Pull new version
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Verify upgrade
make health
```

### Data Migration

```bash
# Run migration scripts
docker exec crewai-email-triage python -m crewai_email_triage.migrations.migrate

# Verify data integrity
docker exec crewai-email-triage python -m crewai_email_triage.migrations.verify
```

---

For additional deployment support or questions, please refer to the [troubleshooting guide](../troubleshooting/) or open an issue in the repository.