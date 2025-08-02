# Operational Runbooks

## Overview

This document provides step-by-step procedures for common operational scenarios and incident response for CrewAI Email Triage.

## Alert Response Procedures

### 1. High Email Processing Error Rate

**Alert**: `HighEmailProcessingErrorRate`
**Severity**: Warning
**Threshold**: Error rate > 10% for 2 minutes

#### Investigation Steps

1. **Check recent deployments**
   ```bash
   # Check application logs for recent errors
   docker logs crewai-email-triage --since 10m | grep ERROR
   
   # Check recent container restarts
   docker ps -a --filter name=crewai-email-triage
   ```

2. **Identify error patterns**
   ```bash
   # Common error types
   curl -s "http://localhost:9090/api/v1/query?query=rate(crewai_emails_processed_total{status=\"failure\"}[5m])" | jq
   
   # Error breakdown by category
   docker logs crewai-email-triage --since 5m | grep ERROR | awk '{print $4}' | sort | uniq -c
   ```

3. **Check email provider status**
   ```bash
   # Test Gmail connectivity
   docker exec crewai-email-triage python -c "from crewai_email_triage.provider import GmailProvider; print(GmailProvider().test_connection())"
   
   # Check provider rate limits
   curl http://localhost:8080/metrics | grep provider_rate_limit
   ```

#### Resolution Steps

1. **If authentication issues**:
   ```bash
   # Refresh OAuth tokens
   docker exec crewai-email-triage python -c "from crewai_email_triage.provider import refresh_tokens; refresh_tokens()"
   
   # Check environment variables
   docker exec crewai-email-triage env | grep -E "(GMAIL|IMAP)"
   ```

2. **If rate limiting**:
   ```bash
   # Enable circuit breaker
   docker exec crewai-email-triage python -c "from crewai_email_triage.circuit_breaker import enable_circuit_breaker; enable_circuit_breaker()"
   
   # Reduce batch size temporarily
   docker exec crewai-email-triage -e BATCH_SIZE=10
   ```

3. **If system resource issues**:
   ```bash
   # Check resource usage
   docker stats crewai-email-triage --no-stream
   
   # Restart container if needed
   docker restart crewai-email-triage
   ```

#### Escalation Criteria
- Error rate > 25% for 5+ minutes
- Complete service failure
- Data loss suspected
- Security incident suspected

---

### 2. Slow Email Processing

**Alert**: `SlowEmailProcessing`
**Severity**: Warning
**Threshold**: 95th percentile > 5 seconds for 5 minutes

#### Investigation Steps

1. **Check system resources**
   ```bash
   # CPU and memory usage
   docker stats crewai-email-triage --no-stream
   
   # Disk I/O
   docker exec crewai-email-triage iostat -x 1 5
   ```

2. **Analyze processing bottlenecks**
   ```bash
   # Check individual operation timings
   curl -s "http://localhost:9090/api/v1/query?query=crewai_email_processing_duration_seconds{quantile=\"0.95\"}" | jq
   
   # Database query performance
   docker exec crewai-postgres pg_stat_statements
   ```

3. **Check external dependencies**
   ```bash
   # Email provider response times
   curl http://localhost:8080/metrics | grep provider_response_time
   
   # Database connection pool
   curl http://localhost:8080/metrics | grep database_connections
   ```

#### Resolution Steps

1. **If CPU bound**:
   ```bash
   # Scale horizontally
   docker-compose up --scale crewai-triage=2
   
   # Optimize processing parameters
   docker exec crewai-email-triage -e MAX_WORKERS=2
   ```

2. **If memory bound**:
   ```bash
   # Clear caches
   docker exec crewai-redis redis-cli FLUSHALL
   
   # Restart with more memory
   docker-compose down && docker-compose up -d
   ```

3. **If I/O bound**:
   ```bash
   # Check disk space
   df -h
   
   # Optimize database
   docker exec crewai-postgres vacuumdb -U crewai -d crewai_email_triage
   ```

---

### 3. Email Queue Backlog

**Alert**: `EmailQueueBacklog`
**Severity**: Critical
**Threshold**: Queue length > 100 emails for 5 minutes

#### Investigation Steps

1. **Check queue status**
   ```bash
   # Current queue length
   curl -s "http://localhost:9090/api/v1/query?query=crewai_email_queue_length" | jq
   
   # Processing rate
   curl -s "http://localhost:9090/api/v1/query?query=rate(crewai_emails_processed_total[5m])" | jq
   ```

2. **Identify bottlenecks**
   ```bash
   # Worker status
   docker exec crewai-email-triage ps aux | grep python
   
   # Database locks
   docker exec crewai-postgres psql -U crewai -d crewai_email_triage -c "SELECT * FROM pg_locks WHERE NOT granted;"
   ```

#### Resolution Steps

1. **Immediate actions**:
   ```bash
   # Scale processing workers
   docker-compose up --scale crewai-triage=3
   
   # Increase batch size
   docker exec crewai-email-triage -e BATCH_SIZE=100
   
   # Enable parallel processing
   docker exec crewai-email-triage -e PARALLEL_PROCESSING=true
   ```

2. **If persistent**:
   ```bash
   # Drain queue to file for later processing
   docker exec crewai-email-triage python -m crewai_email_triage.utils.drain_queue --output /tmp/drained_emails.json
   
   # Process high-priority emails only
   docker exec crewai-email-triage python -m crewai_email_triage.utils.priority_processing --min-priority 8
   ```

---

### 4. Email Provider Down

**Alert**: `EmailProviderDown`
**Severity**: Critical
**Threshold**: Provider unreachable for 1 minute

#### Investigation Steps

1. **Check provider status**
   ```bash
   # Test connectivity
   curl -I https://imap.gmail.com:993
   ping -c 3 imap.gmail.com
   
   # Check DNS resolution
   nslookup imap.gmail.com
   ```

2. **Check authentication**
   ```bash
   # Verify credentials
   docker exec crewai-email-triage python -c "from crewai_email_triage.provider import test_auth; test_auth()"
   
   # Check token expiry
   docker exec crewai-email-triage python -c "from crewai_email_triage.provider import check_token_expiry; check_token_expiry()"
   ```

#### Resolution Steps

1. **If temporary outage**:
   ```bash
   # Enable circuit breaker
   docker exec crewai-email-triage python -c "from crewai_email_triage.circuit_breaker import enable; enable('gmail_provider')"
   
   # Switch to backup provider
   docker exec crewai-email-triage -e FALLBACK_PROVIDER=imap
   ```

2. **If authentication issues**:
   ```bash
   # Refresh OAuth tokens
   docker exec crewai-email-triage python -m crewai_email_triage.auth.refresh_tokens
   
   # Generate new app password
   # (Manual step - update environment variables)
   ```

---

### 5. High Memory Usage

**Alert**: `HighMemoryUsage`
**Severity**: Warning
**Threshold**: Memory usage > 1GB for 5 minutes

#### Investigation Steps

1. **Check memory breakdown**
   ```bash
   # Container memory usage
   docker stats crewai-email-triage --no-stream
   
   # Process memory usage
   docker exec crewai-email-triage ps -eo pid,ppid,cmd,pmem,rss --sort=-rss
   ```

2. **Identify memory leaks**
   ```bash
   # Python memory profiling
   docker exec crewai-email-triage python -m memory_profiler /app/triage.py --profile-memory
   
   # Check for large objects
   docker exec crewai-email-triage python -c "import gc; print(len(gc.get_objects()))"
   ```

#### Resolution Steps

1. **Immediate relief**:
   ```bash
   # Force garbage collection
   docker exec crewai-email-triage python -c "import gc; gc.collect()"
   
   # Clear application caches
   curl -X POST http://localhost:8080/admin/clear-cache
   ```

2. **Long-term fixes**:
   ```bash
   # Restart container
   docker restart crewai-email-triage
   
   # Reduce batch size
   docker exec crewai-email-triage -e BATCH_SIZE=25
   
   # Increase memory limit
   # Update docker-compose.yml with higher memory limits
   ```

---

## Health Check Troubleshooting

### Health Check Failing

1. **Check health endpoint**
   ```bash
   curl -v http://localhost:8000/health
   ```

2. **Common issues and fixes**:
   ```bash
   # Database connection issues
   docker exec crewai-postgres pg_isready -U crewai
   
   # Redis connection issues
   docker exec crewai-redis redis-cli ping
   
   # Application not responding
   docker exec crewai-email-triage ps aux | grep python
   ```

3. **Force restart if needed**
   ```bash
   docker restart crewai-email-triage
   ```

---

## Database Maintenance

### Regular Maintenance Tasks

1. **Daily tasks**
   ```bash
   # Database statistics update
   docker exec crewai-postgres psql -U crewai -d crewai_email_triage -c "ANALYZE;"
   
   # Clean old logs
   docker exec crewai-email-triage find /app/logs -name "*.log" -mtime +7 -delete
   ```

2. **Weekly tasks**
   ```bash
   # Vacuum database
   docker exec crewai-postgres vacuumdb -U crewai -d crewai_email_triage --analyze
   
   # Update database statistics
   docker exec crewai-postgres psql -U crewai -d crewai_email_triage -c "VACUUM ANALYZE;"
   ```

3. **Monthly tasks**
   ```bash
   # Full vacuum (requires downtime)
   docker-compose stop crewai-triage
   docker exec crewai-postgres vacuumdb -U crewai -d crewai_email_triage --full
   docker-compose start crewai-triage
   ```

### Database Recovery

1. **Point-in-time recovery**
   ```bash
   # Stop application
   docker-compose stop crewai-triage
   
   # Restore from backup
   docker exec crewai-postgres pg_restore -U crewai -d crewai_email_triage /backups/backup_YYYYMMDD.sql
   
   # Restart application
   docker-compose start crewai-triage
   ```

---

## Log Management

### Log Analysis

1. **Find specific events**
   ```bash
   # High-priority emails
   docker logs crewai-email-triage | jq 'select(.category == "urgent")'
   
   # Processing errors
   docker logs crewai-email-triage | jq 'select(.level == "ERROR")'
   
   # Slow operations
   docker logs crewai-email-triage | jq 'select(.duration_ms > 1000)'
   ```

2. **Log aggregation queries (Loki)**
   ```logql
   # Error rate by hour
   sum(rate({job="crewai-triage"} |= "ERROR" [1h])) by (hour)
   
   # Top error messages
   topk(10, sum by (message) (count_over_time({job="crewai-triage"} |= "ERROR" [24h])))
   ```

### Log Rotation

1. **Manual log rotation**
   ```bash
   # Rotate application logs
   docker exec crewai-email-triage logrotate /etc/logrotate.d/crewai
   
   # Clean old log files
   docker exec crewai-email-triage find /app/logs -name "*.log.*" -mtime +30 -delete
   ```

---

## Security Incident Response

### Suspected Security Breach

1. **Immediate actions**
   ```bash
   # Isolate the container
   docker network disconnect crewai-network crewai-email-triage
   
   # Preserve evidence
   docker commit crewai-email-triage crewai-email-triage:incident-$(date +%Y%m%d)
   
   # Collect logs
   docker logs crewai-email-triage > incident-logs-$(date +%Y%m%d).log
   ```

2. **Analysis**
   ```bash
   # Check for suspicious activity
   docker logs crewai-email-triage | grep -E "(authentication|login|access)" | tail -100
   
   # Review access patterns
   curl -s "http://localhost:9090/api/v1/query?query=crewai_requests_total" | jq
   ```

3. **Recovery**
   ```bash
   # Rotate all secrets
   # Update OAuth tokens, database passwords, API keys
   
   # Deploy clean version
   docker-compose down
   docker pull crewai-email-triage:latest
   docker-compose up -d
   ```

### Vulnerability Response

1. **Security scan findings**
   ```bash
   # Run security scan
   docker run --rm -v $(pwd):/app securecodewarrior/scanner /app
   
   # Check for known vulnerabilities
   docker exec crewai-email-triage safety check --json
   ```

2. **Patch deployment**
   ```bash
   # Update dependencies
   docker build --no-cache -t crewai-email-triage:patched .
   
   # Rolling update
   docker-compose up -d --no-deps crewai-triage
   ```

---

## Performance Optimization

### Performance Tuning

1. **Application tuning**
   ```bash
   # Optimize batch size
   docker exec crewai-email-triage -e BATCH_SIZE=50 -e MAX_WORKERS=4
   
   # Enable caching
   docker exec crewai-email-triage -e CACHE_ENABLED=true -e CACHE_TTL=3600
   
   # Connection pool tuning
   docker exec crewai-email-triage -e DATABASE_POOL_SIZE=20 -e DATABASE_MAX_OVERFLOW=30
   ```

2. **Infrastructure tuning**
   ```bash
   # Increase container resources
   # Update docker-compose.yml:
   # deploy:
   #   resources:
   #     limits:
   #       cpus: '2.0'
   #       memory: 2G
   
   # Optimize database
   docker exec crewai-postgres psql -U crewai -d crewai_email_triage -c "
   ALTER SYSTEM SET shared_buffers = '256MB';
   ALTER SYSTEM SET effective_cache_size = '1GB';
   SELECT pg_reload_conf();
   "
   ```

### Load Testing

1. **Generate test load**
   ```bash
   # Synthetic load generation
   docker run --rm -v $(pwd):/app locustio/locust -f /app/tests/load_test.py --host http://crewai-email-triage:8000
   
   # Email processing load test
   docker exec crewai-email-triage python -m crewai_email_triage.tests.load_generator --emails 1000 --concurrent 10
   ```

2. **Monitor during load test**
   ```bash
   # Real-time metrics
   watch -n 1 'curl -s http://localhost:8080/metrics | grep -E "(processing_time|queue_length|memory)"'
   
   # Resource monitoring
   docker stats --no-stream
   ```

---

## Backup and Recovery

### Automated Backups

1. **Database backup**
   ```bash
   # Daily backup script
   #!/bin/bash
   DATE=$(date +%Y%m%d_%H%M%S)
   docker exec crewai-postgres pg_dump -U crewai crewai_email_triage | gzip > backups/db_backup_$DATE.sql.gz
   
   # Cleanup old backups (keep 30 days)
   find backups/ -name "db_backup_*.sql.gz" -mtime +30 -delete
   ```

2. **Configuration backup**
   ```bash
   # Backup configuration files
   tar -czf config_backup_$(date +%Y%m%d).tar.gz \
     .env \
     monitoring/ \
     docker-compose.yml \
     src/crewai_email_triage/default_config.json
   ```

### Recovery Procedures

1. **Application recovery**
   ```bash
   # Restore from backup
   docker-compose down
   
   # Restore database
   zcat backups/db_backup_YYYYMMDD_HHMMSS.sql.gz | docker exec -i crewai-postgres psql -U crewai -d crewai_email_triage
   
   # Restore configuration
   tar -xzf config_backup_YYYYMMDD.tar.gz
   
   # Start services
   docker-compose up -d
   ```

2. **Disaster recovery**
   ```bash
   # Full system rebuild
   git clone <repository>
   cd crewai-email-triage
   
   # Restore configuration
   cp /backup/location/.env .
   cp -r /backup/location/monitoring/ .
   
   # Restore data
   docker-compose up -d postgres redis
   # Wait for services to start
   zcat /backup/location/db_backup_latest.sql.gz | docker exec -i crewai-postgres psql -U crewai -d crewai_email_triage
   
   # Start application
   docker-compose up -d
   ```

---

## Escalation Contacts

### On-Call Rotation

- **Primary**: DevOps Team (+1-555-0123)
- **Secondary**: Platform Engineering (+1-555-0456)
- **Manager**: Engineering Manager (+1-555-0789)

### Escalation Matrix

| Severity | Response Time | Escalation Time | Contact |
|----------|---------------|------------------|---------|
| Critical | 5 minutes     | 15 minutes      | Primary On-Call |
| High     | 15 minutes    | 1 hour          | Primary On-Call |
| Medium   | 1 hour        | 4 hours         | Secondary On-Call |
| Low      | 4 hours       | Next business day | Engineering Team |

### Communication Channels

- **Slack**: #crewai-alerts
- **Email**: ops-team@company.com
- **Phone**: Emergency hotline (+1-555-URGENT)
- **Incident Management**: ServiceNow/PagerDuty

---

## Post-Incident Procedures

### Incident Documentation

1. **Incident report template**
   - Incident timeline
   - Root cause analysis
   - Impact assessment
   - Resolution steps
   - Lessons learned
   - Action items

2. **Post-mortem process**
   - Schedule within 24 hours
   - Include all stakeholders
   - Focus on process improvements
   - No blame culture

### Process Improvements

1. **Update monitoring**
   - Add new alerts based on incident
   - Improve alert sensitivity
   - Update runbooks

2. **Infrastructure hardening**
   - Implement additional safeguards
   - Improve redundancy
   - Update documentation

---

*This runbook should be kept up-to-date with system changes and lessons learned from incidents.*