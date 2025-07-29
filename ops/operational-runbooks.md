# Operational Runbooks for Email Triage Service

## Overview
Comprehensive operational procedures for maintaining, troubleshooting, and optimizing the email triage service in production environments.

## Service Architecture Quick Reference

### Core Components
- **Email Classifier**: AI-powered email categorization
- **Priority Agent**: Business priority assignment
- **Summarizer**: Content summarization engine
- **Response Generator**: Automated reply drafting
- **Pipeline Orchestrator**: Workflow coordination

### Dependencies
- **Database**: PostgreSQL cluster (primary/replica)
- **Message Queue**: Redis cluster
- **AI Models**: TensorFlow Serving endpoints
- **Monitoring**: Prometheus/Grafana/Loki stack
- **Cache**: Redis cache layer

---

## High-Priority Incident Response

### P0: Complete Service Outage
**Symptoms**: Service returns 5xx errors, no email processing

**Immediate Response (0-5 minutes)**:
1. **Acknowledge incident** in PagerDuty
2. **Check service status dashboard**: https://status.company.com/email-triage
3. **Verify infrastructure health**:
   ```bash
   kubectl get pods -n email-triage
   kubectl get services -n email-triage
   kubectl describe ingress email-triage-ingress
   ```

**Investigation Steps (5-15 minutes)**:
1. **Check application logs**:
   ```bash
   kubectl logs -f deployment/email-triage-api --tail=100
   stern email-triage -t --since=10m
   ```

2. **Verify database connectivity**:
   ```bash
   kubectl exec -it deployment/email-triage-api -- python -c "
   from src.crewai_email_triage.config import get_db_connection
   conn = get_db_connection()
   print('DB Status:', conn.info.status)
   "
   ```

3. **Check Redis connectivity**:
   ```bash
   kubectl exec -it deployment/email-triage-api -- redis-cli -h redis-cluster ping
   ```

**Resolution Actions**:
- **If pods are crashlooping**: Check resource limits and recent deployments
- **If database is down**: Engage DBA team, consider read-replica promotion
- **If Redis is down**: Restart Redis cluster, flush corrupted data if needed
- **If AI models unavailable**: Switch to fallback classification mode

**Recovery Validation**:
```bash
# Test critical endpoints
curl -X POST https://api.company.com/email-triage/classify \
  -H "Content-Type: application/json" \
  -d '{"email": {"subject": "Test", "body": "Test email"}}'

# Verify processing pipeline
kubectl exec -it deployment/email-triage-api -- python scripts/health_check.py
```

### P1: Performance Degradation
**Symptoms**: High latency (>2s), increased error rates (>5%)

**Investigation Checklist**:
- [ ] Check CPU/Memory utilization in Grafana
- [ ] Review slow query logs in database
- [ ] Verify AI model inference latency
- [ ] Check for upstream dependency issues
- [ ] Review recent code deployments

**Common Fixes**:
1. **Scale horizontally**:
   ```bash
   kubectl scale deployment/email-triage-api --replicas=10
   ```

2. **Clear cache if corrupted**:
   ```bash
   kubectl exec -it deployment/redis -- redis-cli FLUSHDB
   ```

3. **Restart AI model servers**:
   ```bash
   kubectl rollout restart deployment/email-classifier-model
   ```

### P2: Functional Issues
**Symptoms**: Incorrect classifications, missing features

**Debugging Process**:
1. **Check model performance metrics** in MLflow dashboard
2. **Review recent model deployments** in model registry
3. **Analyze classification accuracy trends** in Grafana
4. **Check for data drift** in monitoring dashboards

---

## Routine Maintenance Procedures

### Daily Operations Checklist
**Morning Health Check (9:00 AM)**:
- [ ] Review overnight alerts and incidents
- [ ] Check service performance metrics
- [ ] Verify backup completion status
- [ ] Review resource utilization trends
- [ ] Check security scan reports

**Commands for Daily Health Check**:
```bash
# Service health overview
kubectl get pods,services,ingress -n email-triage

# Performance metrics check
curl -s http://prometheus:9090/api/v1/query?query=up{job="email-triage"} | jq .

# Database health
kubectl exec -it postgresql-primary -- psql -U postgres -c "SELECT version();"

# Cache status
kubectl exec -it redis-cluster -- redis-cli info stats
```

### Weekly Maintenance Tasks
**Every Monday (10:00 AM)**:
- [ ] Review and rotate application secrets
- [ ] Update security certificates if needed
- [ ] Analyze cost optimization opportunities
- [ ] Review and update monitoring alerts
- [ ] Conduct security vulnerability assessment

**Secret Rotation Procedure**:
```bash
# Generate new database password
NEW_PASSWORD=$(openssl rand -base64 32)

# Update Kubernetes secret
kubectl create secret generic db-credentials \
  --from-literal=password=$NEW_PASSWORD \
  --dry-run=client -o yaml | kubectl apply -f -

# Rolling restart to pick up new credentials
kubectl rollout restart deployment/email-triage-api
```

### Monthly Maintenance Tasks
**First Tuesday of each month**:
- [ ] Perform full system backup validation
- [ ] Review and update disaster recovery procedures
- [ ] Conduct capacity planning analysis
- [ ] Update operational documentation
- [ ] Review third-party service integrations

---

## Performance Optimization Procedures

### Email Processing Performance Tuning

**Identify Bottlenecks**:
```bash
# Check processing queue depth
kubectl exec -it deployment/email-triage-api -- python -c "
from src.crewai_email_triage.metrics_export import get_queue_metrics
print(get_queue_metrics())
"

# Analyze processing latency by component
curl -s http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95, email_processing_duration_seconds_bucket)
```

**Optimization Actions**:

1. **Database Query Optimization**:
   ```sql
   -- Check slow queries
   SELECT query, mean_time, calls 
   FROM pg_stat_statements 
   ORDER BY mean_time DESC 
   LIMIT 10;
   
   -- Add missing indexes
   CREATE INDEX CONCURRENTLY idx_emails_created_at ON emails(created_at);
   CREATE INDEX CONCURRENTLY idx_emails_status ON emails(status);
   ```

2. **Application-Level Optimizations**:
   ```python
   # Enable connection pooling
   DATABASE_CONFIG = {
       'pool_size': 20,
       'max_overflow': 0,
       'pool_pre_ping': True,
       'pool_recycle': 3600
   }
   
   # Optimize batch processing
   BATCH_SIZE = 50  # Adjust based on memory constraints
   CONCURRENT_WORKERS = min(multiprocessing.cpu_count(), 8)
   ```

3. **AI Model Optimization**:
   ```bash
   # Enable model quantization for faster inference
   kubectl patch deployment email-classifier-model -p '{
     "spec": {
       "template": {
         "spec": {
           "containers": [{
             "name": "model-server",
             "env": [{"name": "QUANTIZATION", "value": "int8"}]
           }]
         }
       }
     }
   }'
   ```

### Resource Right-sizing

**Memory Optimization**:
```bash
# Analyze memory usage patterns
kubectl top pods -n email-triage --sort-by=memory

# Check for memory leaks
kubectl exec -it deployment/email-triage-api -- python -c "
import psutil
process = psutil.Process()
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
print(f'Memory %: {process.memory_percent():.2f}%')
"
```

**CPU Optimization**:
```bash
# Profile CPU usage
kubectl exec -it deployment/email-triage-api -- py-spy top --pid 1 --duration 30

# Adjust CPU requests/limits based on usage
kubectl patch deployment email-triage-api -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "api",
          "resources": {
            "requests": {"cpu": "500m", "memory": "1Gi"},
            "limits": {"cpu": "2000m", "memory": "2Gi"}
          }
        }]
      }
    }
  }
}'
```

---

## Monitoring and Alerting Management

### Key Metrics to Monitor

**Application Metrics**:
- Email processing rate (emails/minute)
- Classification accuracy (%)
- Response time percentiles (P50, P95, P99)
- Error rates by endpoint
- Queue depth and processing backlog

**Infrastructure Metrics**:
- CPU utilization across pods
- Memory usage and garbage collection
- Database connection pool status
- Network latency and throughput
- Storage utilization and IOPS

**Business Metrics**:
- User satisfaction scores
- Cost per email processed
- Model drift detection scores
- Compliance audit results

### Alert Management

**Critical Alerts (PagerDuty)**:
```yaml
alerts:
  - name: "ServiceDown"
    condition: "up{job='email-triage'} == 0"
    duration: "1m"
    severity: "critical"
    
  - name: "HighErrorRate"
    condition: "rate(http_requests_total{status=~'5..'}[5m]) > 0.05"
    duration: "5m"
    severity: "critical"
    
  - name: "DatabaseConnections"
    condition: "postgresql_stat_database_numbackends > 80"
    duration: "2m"
    severity: "warning"
```

**Alert Tuning**:
```bash
# Review alert frequency
curl -s http://alertmanager:9093/api/v1/alerts | jq '.data[] | select(.state=="firing")'

# Adjust thresholds based on service baseline
# Edit prometheus rules and reload
kubectl exec -it prometheus-server -- promtool check rules /etc/prometheus/rules.yml
kubectl exec -it prometheus-server -- kill -HUP 1
```

---

## Data Management and Backup Procedures

### Database Backup and Recovery

**Daily Backup Verification**:
```bash
# Check backup status
kubectl exec -it postgresql-primary -- pg_basebackup --list

# Verify backup integrity
kubectl exec -it postgresql-primary -- pg_verifybackup /backup/latest/

# Test point-in-time recovery capability
kubectl exec -it postgresql-replica -- psql -c "SELECT pg_last_wal_replay_lsn();"
```

**Recovery Procedures**:
```bash
# Full database restore from backup
kubectl exec -it postgresql-primary -- pg_restore -d email_triage /backup/dump_$(date +%Y%m%d).sql

# Point-in-time recovery
kubectl exec -it postgresql-primary -- pg_ctl stop -D /var/lib/postgresql/data
kubectl exec -it postgresql-primary -- pg_resetwal -f /var/lib/postgresql/data
# Update recovery.conf with target time
kubectl exec -it postgresql-primary -- pg_ctl start -D /var/lib/postgresql/data
```

### Data Retention Management

**Automated Cleanup Scripts**:
```python
# Archive old processed emails
def cleanup_old_emails():
    cutoff_date = datetime.now() - timedelta(days=90)
    
    # Archive to cold storage
    old_emails = session.query(Email).filter(
        Email.created_at < cutoff_date,
        Email.status == 'processed'
    ).all()
    
    # Export to S3 archive
    archive_to_s3(old_emails)
    
    # Delete from active database
    session.query(Email).filter(
        Email.created_at < cutoff_date
    ).delete(synchronize_session=False)
    
    session.commit()
```

---

## Capacity Planning and Scaling

### Growth Monitoring

**Traffic Growth Analysis**:
```bash
# Analyze email volume trends
curl -s 'http://prometheus:9090/api/v1/query_range?query=rate(emails_processed_total[1h])&start=2025-01-01T00:00:00Z&end=2025-01-31T23:59:59Z&step=1h' | jq .

# Resource utilization trends
kubectl top nodes
kubectl top pods -n email-triage --sort-by=cpu
```

**Scaling Triggers**:
- CPU utilization > 70% for 10+ minutes
- Memory utilization > 80% for 5+ minutes
- Queue depth > 1000 emails for 15+ minutes
- Response time P95 > 2 seconds for 10+ minutes

**Scaling Actions**:
```bash
# Horizontal scaling
kubectl scale deployment/email-triage-api --replicas=15

# Vertical scaling (requires restart)
kubectl patch deployment email-triage-api -p '{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "api",
          "resources": {
            "requests": {"cpu": "1000m", "memory": "2Gi"},
            "limits": {"cpu": "4000m", "memory": "4Gi"}
          }
        }]
      }
    }
  }
}'

# Database scaling (add read replicas)
kubectl apply -f postgres-read-replica.yaml
```

---

## Security Operations

### Security Monitoring

**Daily Security Checks**:
```bash
# Check for security vulnerabilities
trivy image email-triage:latest

# Review access logs for anomalies
kubectl logs -f deployment/email-triage-api | grep -E "(401|403|429)"

# Verify certificate status
openssl s_client -connect api.company.com:443 -servername api.company.com | openssl x509 -noout -dates
```

**Incident Response**:
```bash
# Isolate compromised pod
kubectl cordon node-name
kubectl drain node-name --ignore-daemonsets

# Collect forensic data
kubectl cp suspicious-pod:/var/log ./forensic-logs/
kubectl exec -it suspicious-pod -- netstat -tuln > network-connections.txt

# Apply security patches
kubectl set image deployment/email-triage-api api=email-triage:patched-version
```

---

## Communication Templates

### Incident Communication

**Initial Notification**:
```
Subject: [P1] Email Triage Service Experiencing Issues

We are currently investigating performance issues with the Email Triage Service.

Impact: Users may experience delays in email processing
Start Time: [TIMESTAMP]
Status: Investigating

Updates will be provided every 15 minutes.
```

**Resolution Notification**:
```
Subject: [RESOLVED] Email Triage Service Issue

The Email Triage Service issue has been resolved.

Root Cause: [BRIEF DESCRIPTION]
Resolution: [ACTIONS TAKEN]
Duration: [TOTAL DOWNTIME]

A detailed post-mortem will be available within 48 hours.
```

### Maintenance Communication

**Scheduled Maintenance**:
```
Subject: Scheduled Maintenance - Email Triage Service

Maintenance Window: [DATE] [TIME] - [TIME] [TIMEZONE]
Expected Impact: Minimal disruption expected
Purpose: Security updates and performance improvements

The service will remain available during this window with possible brief interruptions.
```

This comprehensive runbook provides structured procedures for maintaining operational excellence in the email triage service. Regular review and updates ensure procedures remain current with system evolution.