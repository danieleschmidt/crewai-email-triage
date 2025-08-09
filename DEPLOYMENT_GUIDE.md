# 🚀 CrewAI Email Triage - Production Deployment Guide

## Overview

This comprehensive deployment guide covers all aspects of deploying the CrewAI Email Triage system to production, including Docker containerization, Kubernetes orchestration, monitoring, and CI/CD pipelines.

## 📋 Prerequisites

- **Kubernetes cluster** (1.21+) with sufficient resources
- **Docker** registry access (GitHub Container Registry or similar)
- **kubectl** configured for your cluster
- **Helm** (optional, for easier management)
- **Prometheus** and **Grafana** for monitoring
- **Domain/DNS** setup for external access

## 🏗️ Architecture

```
                    ┌─────────────────┐
                    │   Load Balancer │
                    └─────────┬───────┘
                              │
                    ┌─────────┴───────┐
                    │  Kubernetes     │
                    │  Service        │
                    └─────────┬───────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────┴────┐         ┌─────┴─────┐         ┌────┴────┐
   │  Pod 1  │         │   Pod 2   │         │  Pod 3  │
   │         │         │           │         │         │
   │ Email   │         │  Email    │         │ Email   │
   │ Triage  │         │  Triage   │         │ Triage  │
   │ App     │         │  App      │         │ App     │
   └─────────┘         └───────────┘         └─────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    ┌─────────┴───────┐
                    │   Prometheus    │
                    │   Monitoring    │
                    └─────────────────┘
```

## 🐳 Docker Deployment

### Quick Start with Docker Compose

1. **Build and start the services:**
   ```bash
   cd deployment/docker
   docker-compose up -d
   ```

2. **Verify deployment:**
   ```bash
   docker-compose ps
   curl http://localhost:8000/health
   ```

3. **Access monitoring:**
   - Grafana: http://localhost:3000 (admin/admin123)
   - Prometheus: http://localhost:9090

### Configuration Options

Environment variables for Docker deployment:

```bash
ENVIRONMENT=production
LOG_LEVEL=INFO
METRICS_ENABLED=true
PROMETHEUS_PORT=8001
GLOBAL_REGION=us-east-1
COMPLIANCE_STANDARDS=ccpa,sox
```

## ☸️ Kubernetes Deployment

### Automated Deployment

Use the deployment script for fully automated deployment:

```bash
# Deploy to staging
./deployment/scripts/deploy.sh staging us-east-1 v1.0.0

# Deploy to production
./deployment/scripts/deploy.sh production us-east-1 v1.0.0
```

### Manual Step-by-Step Deployment

1. **Create namespace:**
   ```bash
   kubectl apply -f deployment/kubernetes/namespace.yml
   ```

2. **Apply RBAC:**
   ```bash
   kubectl apply -f deployment/kubernetes/rbac.yml
   ```

3. **Configure regional settings:**
   ```bash
   # For US deployment
   kubectl apply -f deployment/kubernetes/configmap.yml
   
   # For EU deployment
   kubectl patch configmap triage-config -n crewai-email-triage \
     -p '{"data":{"region":"eu-west-1","compliance_standards":"gdpr,iso-27001"}}'
   ```

4. **Deploy persistent volumes:**
   ```bash
   kubectl apply -f deployment/kubernetes/pvc.yml
   ```

5. **Deploy the application:**
   ```bash
   kubectl apply -f deployment/kubernetes/deployment.yml
   kubectl apply -f deployment/kubernetes/service.yml
   ```

6. **Enable auto-scaling:**
   ```bash
   kubectl apply -f deployment/kubernetes/hpa.yml
   ```

### Verification

```bash
# Check pod status
kubectl get pods -n crewai-email-triage

# Check service status
kubectl get svc -n crewai-email-triage

# View logs
kubectl logs -f deployment/email-triage-deployment -n crewai-email-triage

# Run health check
kubectl exec -it -n crewai-email-triage deployment/email-triage-deployment -- \
  python -c "from crewai_email_triage.resilience import resilience; print(resilience.health_check.get_overall_health())"
```

## 📊 Monitoring Setup

### Prometheus Configuration

The monitoring stack includes:
- **Prometheus** for metrics collection
- **Grafana** for visualization
- **Alert Manager** for alerting

Key metrics monitored:
- Processing throughput (messages/second)
- Response time percentiles (P50, P95, P99)
- Error rates
- Resource utilization (CPU, Memory)
- Security threat detection
- Compliance violations
- Auto-scaling events

### Grafana Dashboard

Access the pre-configured dashboard at:
- URL: `http://grafana-service:3000`
- Username: `admin`
- Password: `admin123`

Dashboard includes:
- Real-time processing metrics
- Performance trends
- Security alerts
- Compliance status
- Resource utilization

### Alert Rules

Critical alerts configured:
- **EmailTriageHighErrorRate**: >10% error rate
- **EmailTriageHighLatency**: >5s P95 latency
- **SecurityThreatsDetected**: >10 threats in 5min
- **ComplianceViolations**: Any compliance violations
- **EmailTriageServiceDown**: Service unavailable

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

The complete CI/CD pipeline includes:

1. **Continuous Integration:**
   - Code linting (ruff, black, isort)
   - Type checking (mypy)
   - Unit tests with coverage
   - Security scanning (bandit, safety)
   - Integration tests

2. **Build & Security:**
   - Docker image building
   - Container vulnerability scanning (Trivy)
   - SBOM generation

3. **Deployment:**
   - Staging deployment with health checks
   - Smoke tests
   - Production deployment (blue-green strategy)
   - Post-deployment verification

### Pipeline Triggers

- **Pull Request**: Full CI pipeline
- **Main Branch Push**: Full CI/CD with deployment
- **Release Tags**: Production deployment only

## 🌍 Multi-Region Deployment

### Regional Configurations

#### US East (Primary)
```yaml
region: us-east-1
timezone: America/New_York
compliance_standards: ccpa,sox
data_residency_required: false
```

#### EU West (GDPR Compliant)
```yaml
region: eu-west-1
timezone: Europe/London
compliance_standards: gdpr,iso-27001
data_residency_required: true
```

#### Asia Pacific (PDPA Compliant)
```yaml
region: ap-southeast-1
timezone: Asia/Singapore
compliance_standards: pdpa
data_residency_required: true
```

### Traffic Distribution

Use DNS-based routing for global distribution:
```bash
# US traffic
us.email-triage.company.com → US East cluster

# EU traffic  
eu.email-triage.company.com → EU West cluster

# APAC traffic
asia.email-triage.company.com → AP Southeast cluster
```

## 🔒 Security Considerations

### Container Security
- Non-root user execution
- Read-only root filesystem
- Minimal base image (Python slim)
- Security context constraints

### Network Security
- Network policies for pod-to-pod communication
- TLS termination at load balancer
- Internal service mesh (optional)

### Secret Management
```bash
# Create secrets
kubectl create secret generic triage-secrets \
  --from-literal=database-password=<password> \
  --from-literal=api-key=<api-key> \
  -n crewai-email-triage

# Mount in deployment
volumes:
- name: secrets
  secret:
    secretName: triage-secrets
```

## 📈 Scaling Configuration

### Horizontal Pod Autoscaler (HPA)

```yaml
minReplicas: 2
maxReplicas: 10
targetCPUUtilizationPercentage: 70
targetMemoryUtilizationPercentage: 80
```

### Vertical Pod Autoscaler (VPA)

Optional VPA configuration for resource optimization:
```yaml
updateMode: "Auto"
resourcePolicy:
  containerPolicies:
  - containerName: email-triage
    minAllowed:
      cpu: 250m
      memory: 512Mi
    maxAllowed:
      cpu: 2000m
      memory: 4Gi
```

## 🚨 Disaster Recovery

### Backup Strategy

1. **Configuration Backup:**
   ```bash
   kubectl get configmaps,secrets,pvc -n crewai-email-triage -o yaml > backup/config-$(date +%Y%m%d).yaml
   ```

2. **Data Backup:**
   ```bash
   # Backup persistent volumes
   kubectl exec -n crewai-email-triage deployment/email-triage-deployment -- \
     tar -czf /tmp/data-backup.tar.gz /app/data
   ```

### Recovery Procedures

1. **Service Recovery:**
   ```bash
   # Restart deployment
   kubectl rollout restart deployment/email-triage-deployment -n crewai-email-triage
   
   # Scale up if needed
   kubectl scale deployment email-triage-deployment --replicas=5 -n crewai-email-triage
   ```

2. **Configuration Recovery:**
   ```bash
   kubectl apply -f backup/config-$(date +%Y%m%d).yaml
   ```

## 🔧 Troubleshooting

### Common Issues

1. **Pod CrashLoopBackOff:**
   ```bash
   kubectl describe pod <pod-name> -n crewai-email-triage
   kubectl logs <pod-name> -n crewai-email-triage --previous
   ```

2. **Service Not Accessible:**
   ```bash
   kubectl get endpoints email-triage-service -n crewai-email-triage
   kubectl port-forward svc/email-triage-service 8080:8000 -n crewai-email-triage
   ```

3. **High Memory Usage:**
   ```bash
   kubectl top pods -n crewai-email-triage
   kubectl exec -it <pod-name> -n crewai-email-triage -- htop
   ```

### Performance Tuning

1. **Optimize Worker Count:**
   ```bash
   kubectl patch configmap triage-config -n crewai-email-triage \
     -p '{"data":{"max_workers":"20"}}'
   ```

2. **Increase Batch Size:**
   ```bash
   kubectl patch configmap triage-config -n crewai-email-triage \
     -p '{"data":{"batch_size":"100"}}'
   ```

## 📚 Operational Procedures

### Daily Operations

1. **Health Check:**
   ```bash
   curl -f http://your-domain.com:8000/health
   ```

2. **Performance Metrics:**
   ```bash
   curl http://your-domain.com:8001/metrics | grep processing_throughput
   ```

3. **Log Review:**
   ```bash
   kubectl logs -f deployment/email-triage-deployment -n crewai-email-triage | grep ERROR
   ```

### Weekly Maintenance

1. **Update Deployment:**
   ```bash
   ./deployment/scripts/deploy.sh production us-east-1 v1.1.0
   ```

2. **Review Metrics:**
   - Check Grafana dashboards
   - Review alert history
   - Analyze performance trends

3. **Security Scan:**
   ```bash
   trivy image crewai/email-triage:latest
   ```

## 🎯 Performance Benchmarks

### Expected Performance

- **Throughput**: 30+ messages/second
- **Latency**: <1000ms P95
- **Error Rate**: <1%
- **Availability**: 99.9%

### Load Testing

```bash
# Simple load test
for i in {1..1000}; do
  curl -X POST http://your-domain.com:8000/api/triage \
    -H "Content-Type: application/json" \
    -d '{"message": "Test email content"}' &
done
wait
```

## 📞 Support

### Monitoring Alerts

All alerts are sent to:
- Slack: #email-triage-alerts
- Email: ops-team@company.com
- PagerDuty: Email Triage Service

### Escalation Path

1. **Level 1**: DevOps Team
2. **Level 2**: Platform Engineering
3. **Level 3**: Architecture Team

### Documentation

- **API Documentation**: `/docs/api`
- **Architecture Decision Records**: `/docs/adr`
- **Runbooks**: `/docs/runbooks`

---

## 🏁 Quick Start Checklist

- [ ] Prerequisites installed and configured
- [ ] Kubernetes cluster accessible
- [ ] Docker registry configured
- [ ] Secrets created
- [ ] Monitoring stack deployed
- [ ] CI/CD pipeline configured
- [ ] DNS and load balancer configured
- [ ] Alerting rules configured
- [ ] Backup procedures tested
- [ ] Team access configured

**🎉 Your CrewAI Email Triage system is now production-ready!**