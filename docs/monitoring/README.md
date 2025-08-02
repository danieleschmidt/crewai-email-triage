# Monitoring and Observability

## Overview

CrewAI Email Triage includes a comprehensive monitoring and observability stack using modern tools and best practices. The system provides metrics collection, alerting, logging, and distributed tracing capabilities.

## Architecture

```mermaid
graph TB
    subgraph "Application Layer"
        A[CrewAI Email Triage]
        A1[Health Endpoints]
        A2[Metrics Endpoints]
        A3[Structured Logging]
    end
    
    subgraph "Metrics Collection"
        P[Prometheus]
        PG[Push Gateway]
        AM[Alert Manager]
    end
    
    subgraph "Visualization"
        G[Grafana Dashboards]
        K[Kibana]
    end
    
    subgraph "Log Aggregation"
        L[Loki]
        PT[Promtail]
        ES[Elasticsearch]
    end
    
    subgraph "Tracing"
        J[Jaeger]
        OT[OpenTelemetry]
    end
    
    A --> A1
    A --> A2
    A --> A3
    A2 --> P
    A2 --> PG
    P --> AM
    P --> G
    A3 --> L
    A3 --> ES
    L --> PT
    L --> G
    ES --> K
    A --> OT
    OT --> J
    J --> G
```

## Metrics Collection

### Application Metrics

The application exports comprehensive metrics via Prometheus format:

**Endpoint**: `http://localhost:8080/metrics`

#### Core Business Metrics

```python
# Email processing metrics
crewai_emails_processed_total{status="success|failure",category="urgent|work|personal|spam"}
crewai_email_processing_duration_seconds{operation="classify|prioritize|summarize|respond"}
crewai_email_batch_size
crewai_email_queue_length

# Classification metrics
crewai_classification_accuracy_ratio{category="urgent|work|personal|spam"}
crewai_classification_confidence_score{category="urgent|work|personal|spam"}

# Priority scoring metrics
crewai_priority_score_distribution{priority="high|medium|low"}
crewai_priority_processing_duration_seconds

# Provider metrics
crewai_email_provider_requests_total{provider="gmail|imap",status="success|failure"}
crewai_email_provider_response_time_seconds{provider="gmail|imap"}
crewai_email_provider_rate_limit_remaining{provider="gmail|imap"}
```

#### System Performance Metrics

```python
# Performance metrics
crewai_request_duration_seconds{method="GET|POST",endpoint="/health|/metrics"}
crewai_requests_total{method="GET|POST",status="200|400|500"}
crewai_active_connections
crewai_memory_usage_bytes
crewai_cpu_usage_percent

# Circuit breaker metrics
crewai_circuit_breaker_state{circuit="gmail_provider|imap_provider"}
crewai_circuit_breaker_failures_total{circuit="gmail_provider|imap_provider"}

# Cache metrics
crewai_cache_hits_total{cache="classification|priority"}
crewai_cache_misses_total{cache="classification|priority"}
crewai_cache_size_bytes{cache="classification|priority"}
```

### Infrastructure Metrics

- **System Metrics**: CPU, memory, disk, network (via node_exporter)
- **Container Metrics**: Docker container stats (via cAdvisor)
- **Database Metrics**: PostgreSQL performance (via postgres_exporter)
- **Cache Metrics**: Redis performance (via redis_exporter)

### Custom Metric Collection

```python
# Adding custom metrics in your code
from prometheus_client import Counter, Histogram, Gauge

# Business metric examples
EMAIL_PROCESSING_COUNTER = Counter(
    'crewai_emails_processed_total',
    'Total emails processed',
    ['status', 'category']
)

PROCESSING_TIME_HISTOGRAM = Histogram(
    'crewai_email_processing_duration_seconds',
    'Time spent processing emails',
    ['operation']
)

QUEUE_LENGTH_GAUGE = Gauge(
    'crewai_email_queue_length',
    'Number of emails in processing queue'
)

# Usage in application code
with PROCESSING_TIME_HISTOGRAM.labels(operation='classify').time():
    result = classifier.classify(email)
    EMAIL_PROCESSING_COUNTER.labels(
        status='success', 
        category=result['category']
    ).inc()
```

## Health Monitoring

### Health Check Endpoints

#### Primary Health Check
**Endpoint**: `GET /health`

```json
{
  "status": "healthy",
  "timestamp": "2025-08-02T10:30:00Z",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "checks": {
    "database": {"status": "healthy", "response_time_ms": 5},
    "redis": {"status": "healthy", "response_time_ms": 2},
    "email_providers": {
      "gmail": {"status": "healthy", "last_check": "2025-08-02T10:29:00Z"},
      "imap": {"status": "degraded", "last_check": "2025-08-02T10:25:00Z"}
    },
    "disk_space": {"status": "healthy", "available_percent": 75},
    "memory": {"status": "healthy", "usage_percent": 45}
  }
}
```

#### Detailed Health Check
**Endpoint**: `GET /health/detailed`

```json
{
  "status": "healthy",
  "application": {
    "name": "crewai-email-triage",
    "version": "1.0.0",
    "environment": "production",
    "startup_time": "2025-08-02T06:30:00Z",
    "uptime_hours": 4.0
  },
  "system": {
    "cpu_usage": 23.5,
    "memory_usage": 512.3,
    "disk_usage": 1024.7,
    "network_connections": 15
  },
  "dependencies": {
    "postgres": {
      "status": "healthy",
      "connection_pool": {"active": 5, "idle": 15, "max": 20},
      "last_query_ms": 12
    },
    "redis": {
      "status": "healthy",
      "memory_usage": "45MB",
      "connected_clients": 3,
      "last_operation_ms": 1
    }
  },
  "email_providers": {
    "gmail": {
      "status": "healthy",
      "quota_remaining": 2500,
      "last_successful_fetch": "2025-08-02T10:25:00Z",
      "error_rate_percent": 0.1
    }
  },
  "performance": {
    "emails_processed_last_hour": 1250,
    "average_processing_time_ms": 145,
    "queue_length": 3,
    "error_rate_percent": 0.05
  }
}
```

#### Readiness Check
**Endpoint**: `GET /health/ready`

```json
{
  "ready": true,
  "checks": {
    "database_migrations": "completed",
    "cache_warmup": "completed", 
    "configuration_loaded": "success",
    "email_providers_initialized": "success"
  }
}
```

#### Liveness Check
**Endpoint**: `GET /health/live`

```json
{
  "alive": true,
  "last_heartbeat": "2025-08-02T10:30:00Z"
}
```

### Health Check Configuration

```yaml
# Docker Compose health check
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s

# Kubernetes health check
livenessProbe:
  httpGet:
    path: /health/live
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

## Alerting

### Alert Manager Configuration

**File**: `monitoring/alertmanager/alertmanager.yml`

```yaml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@crewai-triage.com'

route:
  group_by: ['alertname', 'instance']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default-receiver'

receivers:
  - name: 'default-receiver'
    email_configs:
      - to: 'ops-team@company.com'
        subject: 'CrewAI Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Instance: {{ .Labels.instance }}
          {{ end }}
```

### Application Alert Rules

**File**: `monitoring/prometheus/rules/application-alerts.yml`

```yaml
groups:
  - name: crewai-application-alerts
    rules:
      # High error rate
      - alert: HighEmailProcessingErrorRate
        expr: rate(crewai_emails_processed_total{status="failure"}[5m]) > 0.1
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High email processing error rate"
          description: "Error rate is {{ $value }} errors per second"

      # Long processing times
      - alert: SlowEmailProcessing
        expr: crewai_email_processing_duration_seconds{quantile="0.95"} > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Email processing is slow"
          description: "95th percentile processing time is {{ $value }}s"

      # Queue backup
      - alert: EmailQueueBacklog
        expr: crewai_email_queue_length > 100
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Email processing queue is backing up"
          description: "Queue length is {{ $value }} emails"

      # Provider failures
      - alert: EmailProviderDown
        expr: up{job="crewai-triage"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Email provider is down"
          description: "Cannot reach email provider for {{ $labels.instance }}"

      # Resource usage
      - alert: HighMemoryUsage
        expr: crewai_memory_usage_bytes / 1024 / 1024 / 1024 > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ $value }}GB"

      # Health check failures
      - alert: HealthCheckFailing
        expr: crewai_health_check_status != 1
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "Application health check failing"
          description: "Health check has been failing for {{ $labels.instance }}"
```

## Logging

### Structured Logging Format

```json
{
  "timestamp": "2025-08-02T10:30:00.123Z",
  "level": "INFO",
  "logger": "crewai.pipeline",
  "message": "Email processing completed",
  "request_id": "req_abc123def456",
  "email_id": "email_789ghi012",
  "user_id": "user_123",
  "operation": "classify",
  "duration_ms": 145,
  "category": "urgent",
  "confidence": 0.92,
  "metadata": {
    "sender": "urgent@company.com",
    "subject_hash": "sha256:...",
    "attachments_count": 2,
    "provider": "gmail"
  },
  "performance": {
    "memory_usage_mb": 23.5,
    "cpu_usage_percent": 12.3
  }
}
```

### Log Levels and Usage

```python
import structlog

logger = structlog.get_logger(__name__)

# Error logging with context
try:
    result = process_email(email)
except Exception as e:
    logger.error(
        "Email processing failed",
        email_id=email.id,
        error=str(e),
        error_type=type(e).__name__,
        stack_trace=traceback.format_exc()
    )

# Performance logging
logger.info(
    "Email processed successfully",
    email_id=email.id,
    category=result.category,
    confidence=result.confidence,
    processing_time_ms=processing_time * 1000,
    queue_size=queue.size()
)

# Business event logging
logger.info(
    "High priority email detected",
    email_id=email.id,
    priority_score=result.priority_score,
    sender=email.sender,
    urgent_keywords=result.urgent_keywords
)
```

### Log Aggregation

#### Loki Configuration

**File**: `monitoring/loki/loki-config.yml`

- **Retention**: 30 days
- **Compression**: gzip
- **Indexing**: timestamp, level, logger, request_id
- **Storage**: Local filesystem (configurable for S3/GCS)

#### Log Queries

```logql
# All error logs in the last hour
{job="crewai-triage"} |= "ERROR" | json | level="ERROR"

# Slow email processing (>1 second)
{job="crewai-triage"} | json | duration_ms > 1000

# Failed email classification
{job="crewai-triage"} | json | logger="crewai.classifier" | message="Classification failed"

# High-priority emails
{job="crewai-triage"} | json | category="urgent" | confidence > 0.8
```

## Distributed Tracing

### OpenTelemetry Integration

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument code
@tracer.start_as_current_span("process_email")
def process_email(email):
    span = trace.get_current_span()
    span.set_attribute("email.id", email.id)
    span.set_attribute("email.category", email.category)
    
    with tracer.start_as_current_span("classify_email") as classify_span:
        classification = classifier.classify(email)
        classify_span.set_attribute("classification.confidence", classification.confidence)
    
    with tracer.start_as_current_span("prioritize_email") as priority_span:
        priority = prioritizer.prioritize(email, classification)
        priority_span.set_attribute("priority.score", priority.score)
    
    return process_result
```

### Trace Analysis

- **Request Flow**: End-to-end email processing traces
- **Performance Bottlenecks**: Identify slow operations
- **Error Correlation**: Link errors across services
- **Dependency Mapping**: Understand service interactions

## Dashboards

### Grafana Dashboards

#### 1. Application Overview Dashboard

**Panels**:
- Email processing rate (emails/minute)
- Processing duration (p50, p95, p99)
- Error rate percentage
- Queue length
- Active connections
- Memory and CPU usage

#### 2. Business Metrics Dashboard

**Panels**:
- Email categories distribution
- Priority score distribution  
- Classification accuracy over time
- Provider performance comparison
- Top senders by volume
- Processing time by category

#### 3. System Performance Dashboard

**Panels**:
- System resource usage (CPU, memory, disk)
- Container metrics
- Database performance
- Cache hit rates
- Network I/O
- Disk I/O

#### 4. Alert Status Dashboard

**Panels**:
- Active alerts summary
- Alert history
- MTTR (Mean Time To Resolution)
- Alert frequency by type
- System availability

### Dashboard Configuration

```json
{
  "dashboard": {
    "id": null,
    "title": "CrewAI Email Triage - Application Overview",
    "tags": ["crewai", "email-triage", "application"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Email Processing Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(crewai_emails_processed_total[5m])",
            "legendFormat": "Emails/second"
          }
        ]
      }
    ]
  }
}
```

## Monitoring Setup

### Docker Compose

```bash
# Start monitoring stack
docker-compose --profile monitoring up -d

# Access monitoring tools
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
# Alert Manager: http://localhost:9093
```

### Configuration

```bash
# Custom monitoring configuration
cp monitoring/prometheus/prometheus.yml.example monitoring/prometheus/prometheus.yml
cp monitoring/grafana/grafana.ini.example monitoring/grafana/grafana.ini
cp monitoring/alertmanager/alertmanager.yml.example monitoring/alertmanager/alertmanager.yml

# Edit configurations for your environment
vim monitoring/prometheus/prometheus.yml
vim monitoring/alertmanager/alertmanager.yml
```

## Monitoring Best Practices

### 1. Metric Design

- **Use meaningful names**: `crewai_emails_processed_total` not `counter1`
- **Include relevant labels**: `{status="success", category="urgent"}`
- **Choose appropriate metric types**: Counter, Gauge, Histogram, Summary
- **Avoid high cardinality**: Don't use email addresses as labels

### 2. Alert Design

- **Be specific**: Alert on symptoms, not causes
- **Avoid alert fatigue**: Set appropriate thresholds
- **Include context**: Provide runbook links and descriptions
- **Test alerts**: Ensure they fire correctly

### 3. Dashboard Design

- **User-focused**: Show business metrics prominently
- **Hierarchical**: Overview → Detail → Debug
- **Consistent**: Use standard colors and conventions
- **Actionable**: Include links to logs and traces

### 4. Log Management

- **Structured logging**: Use JSON format consistently
- **Correlation IDs**: Track requests across services
- **Appropriate levels**: INFO for business events, DEBUG for troubleshooting
- **Retention policies**: Balance cost and compliance requirements

## Performance Monitoring

### Key Performance Indicators (KPIs)

1. **Email Processing Rate**: emails/minute
2. **Processing Latency**: p95 processing time < 2 seconds
3. **Error Rate**: < 0.1% processing failures
4. **Availability**: > 99.9% uptime
5. **Queue Health**: < 10 emails in queue

### SLA Monitoring

```yaml
# SLI/SLO definitions
availability_sli: up{job="crewai-triage"}
availability_slo: 99.9% # 43 minutes downtime per month

latency_sli: crewai_email_processing_duration_seconds
latency_slo: 95% < 2 seconds

error_rate_sli: rate(crewai_emails_processed_total{status="failure"}[5m])
error_rate_slo: < 0.1%
```

## Troubleshooting

### Common Issues

1. **Metrics not appearing**
   ```bash
   # Check metrics endpoint
   curl http://localhost:8080/metrics
   
   # Check Prometheus targets
   curl http://localhost:9090/api/v1/targets
   ```

2. **Alerts not firing**
   ```bash
   # Check alert rules
   curl http://localhost:9090/api/v1/rules
   
   # Check alert manager config
   curl http://localhost:9093/api/v1/status
   ```

3. **Dashboard not loading**
   ```bash
   # Check Grafana logs
   docker logs grafana
   
   # Verify data source connection
   curl http://admin:admin@localhost:3000/api/datasources
   ```

### Debugging Tools

```bash
# Query Prometheus directly
curl 'http://localhost:9090/api/v1/query?query=crewai_emails_processed_total'

# Check alert manager silences
curl http://localhost:9093/api/v1/silences

# Validate Prometheus config
promtool check config monitoring/prometheus/prometheus.yml

# Test alert rules
promtool check rules monitoring/prometheus/rules/*.yml
```

---

For additional monitoring configuration and troubleshooting, refer to the individual tool documentation or contact the operations team.