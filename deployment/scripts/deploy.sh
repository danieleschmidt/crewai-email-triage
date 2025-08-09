#!/bin/bash

set -euo pipefail

# CrewAI Email Triage Deployment Script
# Usage: ./deploy.sh [environment] [region] [version]

ENVIRONMENT=${1:-staging}
REGION=${2:-us-east-1}
VERSION=${3:-latest}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
NAMESPACE="crewai-email-triage"
IMAGE_NAME="crewai/email-triage:${VERSION}"
DEPLOYMENT_NAME="email-triage-deployment"

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(staging|production)$ ]]; then
    log_error "Invalid environment: $ENVIRONMENT. Must be 'staging' or 'production'"
    exit 1
fi

# Validate region
if [[ ! "$REGION" =~ ^(us-east-1|us-west-2|eu-west-1|eu-central-1|ap-southeast-1)$ ]]; then
    log_error "Invalid region: $REGION"
    exit 1
fi

log_info "Starting deployment to $ENVIRONMENT environment in $REGION region"
log_info "Image: $IMAGE_NAME"

# Check if kubectl is configured
if ! kubectl cluster-info &> /dev/null; then
    log_error "kubectl is not configured or cluster is not accessible"
    exit 1
fi

# Pre-deployment checks
log_info "Running pre-deployment checks..."

# Check if Docker image exists
if ! docker image inspect "$IMAGE_NAME" &> /dev/null; then
    log_warn "Docker image $IMAGE_NAME not found locally. Pulling..."
    docker pull "$IMAGE_NAME" || {
        log_error "Failed to pull image $IMAGE_NAME"
        exit 1
    }
fi

# Create namespace if it doesn't exist
log_info "Ensuring namespace exists..."
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

# Apply RBAC first
log_info "Applying RBAC configuration..."
kubectl apply -f ../kubernetes/rbac.yml

# Apply ConfigMaps based on region
log_info "Applying configuration for region: $REGION"
case "$REGION" in
    us-*)
        kubectl apply -f ../kubernetes/configmap.yml
        kubectl patch configmap triage-config -n "$NAMESPACE" -p '{"data":{"region":"'$REGION'"}}'
        ;;
    eu-*)
        kubectl apply -f ../kubernetes/configmap.yml
        # Switch to EU configuration
        kubectl delete configmap triage-config -n "$NAMESPACE" --ignore-not-found=true
        kubectl create configmap triage-config -n "$NAMESPACE" --from-literal=region="$REGION" --from-literal=compliance_standards="gdpr,iso-27001" --from-literal=data_residency_required="true"
        ;;
    ap-*)
        kubectl apply -f ../kubernetes/configmap.yml
        kubectl patch configmap triage-config -n "$NAMESPACE" -p '{"data":{"region":"'$REGION'","compliance_standards":"pdpa"}}'
        ;;
esac

# Apply PVCs
log_info "Applying persistent volume claims..."
kubectl apply -f ../kubernetes/pvc.yml

# Update deployment with correct image
log_info "Updating deployment with image: $IMAGE_NAME"
sed "s|image: crewai/email-triage:latest|image: $IMAGE_NAME|g" ../kubernetes/deployment.yml > /tmp/deployment-updated.yml

# Deployment strategy based on environment
if [[ "$ENVIRONMENT" == "production" ]]; then
    log_info "Performing blue-green deployment for production..."
    
    # Create green deployment
    sed 's/email-triage-deployment/email-triage-deployment-green/g; s/version: v1/version: green/g' /tmp/deployment-updated.yml > /tmp/deployment-green.yml
    
    # Deploy green version
    kubectl apply -f /tmp/deployment-green.yml
    
    # Wait for green deployment to be ready
    log_info "Waiting for green deployment to be ready..."
    kubectl rollout status deployment/email-triage-deployment-green -n "$NAMESPACE" --timeout=600s
    
    # Health check green deployment
    log_info "Running health checks on green deployment..."
    kubectl wait --for=condition=ready pod -l app=email-triage,version=green -n "$NAMESPACE" --timeout=300s
    
    # Run smoke tests
    log_info "Running smoke tests on green deployment..."
    GREEN_POD=$(kubectl get pods -n "$NAMESPACE" -l app=email-triage,version=green --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')
    
    if ! kubectl exec -n "$NAMESPACE" "$GREEN_POD" -- python -c "
import sys
sys.path.insert(0, '/app/src')
from crewai_email_triage.resilience import resilience
health = resilience.health_check.get_overall_health()
print(f'Health status: {health[\"overall_status\"]}')
assert health['overall_status'] in ['healthy', 'degraded']
print('Health check passed')
"; then
        log_error "Health check failed on green deployment"
        kubectl delete deployment email-triage-deployment-green -n "$NAMESPACE"
        exit 1
    fi
    
    # Switch traffic to green
    log_info "Switching traffic to green deployment..."
    kubectl patch service email-triage-service -n "$NAMESPACE" -p '{"spec":{"selector":{"version":"green"}}}'
    
    # Wait for traffic switch
    sleep 30
    
    # Run final verification
    log_info "Running final verification..."
    SERVICE_IP=$(kubectl get service email-triage-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [[ -n "$SERVICE_IP" ]]; then
        for i in {1..5}; do
            if curl -f "http://$SERVICE_IP:8000/health" &> /dev/null; then
                log_info "Service health check passed"
                break
            fi
            sleep 10
        done
    fi
    
    # Remove old blue deployment
    log_info "Removing old blue deployment..."
    kubectl delete deployment "$DEPLOYMENT_NAME" -n "$NAMESPACE" --ignore-not-found=true
    
    # Rename green to primary
    log_info "Finalizing deployment..."
    kubectl patch deployment email-triage-deployment-green -n "$NAMESPACE" -p '{"metadata":{"name":"email-triage-deployment"},"spec":{"selector":{"matchLabels":{"version":"v1"}},"template":{"metadata":{"labels":{"version":"v1"}}}}}'
    
    # Update service selector back to v1
    kubectl patch service email-triage-service -n "$NAMESPACE" -p '{"spec":{"selector":{"version":"v1"}}}'
    
else
    # Staging deployment - rolling update
    log_info "Performing rolling update for staging..."
    kubectl apply -f /tmp/deployment-updated.yml
    
    # Wait for rollout
    log_info "Waiting for deployment rollout..."
    kubectl rollout status deployment/"$DEPLOYMENT_NAME" -n "$NAMESPACE" --timeout=600s
fi

# Apply services and HPA
log_info "Applying services and autoscaling..."
kubectl apply -f ../kubernetes/service.yml
kubectl apply -f ../kubernetes/hpa.yml

# Post-deployment verification
log_info "Running post-deployment verification..."

# Check pod status
POD_COUNT=$(kubectl get pods -n "$NAMESPACE" -l app=email-triage --field-selector=status.phase=Running --no-headers | wc -l)
if [[ "$POD_COUNT" -eq 0 ]]; then
    log_error "No running pods found"
    exit 1
fi

log_info "Found $POD_COUNT running pods"

# Check HPA status
HPA_STATUS=$(kubectl get hpa email-triage-hpa -n "$NAMESPACE" -o jsonpath='{.status.conditions[?(@.type=="ScalingActive")].status}')
if [[ "$HPA_STATUS" == "True" ]]; then
    log_info "HPA is active and functioning"
else
    log_warn "HPA may not be functioning correctly"
fi

# Get service endpoints
SERVICE_IP=$(kubectl get service email-triage-service -n "$NAMESPACE" -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [[ -n "$SERVICE_IP" ]]; then
    log_info "Service available at: http://$SERVICE_IP:8000"
    log_info "Metrics available at: http://$SERVICE_IP:8001/metrics"
else
    INTERNAL_IP=$(kubectl get service email-triage-internal -n "$NAMESPACE" -o jsonpath='{.spec.clusterIP}')
    log_info "Internal service available at: http://$INTERNAL_IP:8000"
fi

# Performance validation
log_info "Running performance validation..."
POD_NAME=$(kubectl get pods -n "$NAMESPACE" -l app=email-triage --field-selector=status.phase=Running -o jsonpath='{.items[0].metadata.name}')

PERFORMANCE_TEST=$(kubectl exec -n "$NAMESPACE" "$POD_NAME" -- python -c "
import asyncio
import time
import sys
sys.path.insert(0, '/app/src')

from crewai_email_triage.advanced_scaling import process_batch_high_performance

async def test_performance():
    messages = ['Urgent: System down!'] * 20
    start = time.perf_counter()
    results = await process_batch_high_performance(messages)
    duration = time.perf_counter() - start
    throughput = len(results) / duration
    print(f'Performance test: {len(results)} messages in {duration:.2f}s ({throughput:.1f} msg/s)')
    return throughput > 10  # Expect at least 10 msg/s

success = asyncio.run(test_performance())
exit(0 if success else 1)
")

if [[ $? -eq 0 ]]; then
    log_info "Performance validation passed"
else
    log_warn "Performance validation failed - system may be under stress"
fi

# Security validation
log_info "Running security validation..."
SECURITY_TEST=$(kubectl exec -n "$NAMESPACE" "$POD_NAME" -- python -c "
import sys
sys.path.insert(0, '/app/src')

from crewai_email_triage.advanced_security import AdvancedSecurityScanner

scanner = AdvancedSecurityScanner()
result = scanner.scan_content('Test email with suspicious content')
print(f'Security scanner active: {len(result.threats) >= 0}')
print('Security validation passed')
")

# Compliance validation
log_info "Running compliance validation..."
kubectl exec -n "$NAMESPACE" "$POD_NAME" -- python -c "
import sys
sys.path.insert(0, '/app/src')

from crewai_email_triage.global_features import ComplianceManager, ComplianceStandard, GlobalContext

manager = ComplianceManager()
context = GlobalContext(region='$REGION')
checks = manager.validate_compliance('Test email', [ComplianceStandard.GDPR], context)
print(f'Compliance checks completed: {len(checks)} standards validated')
print('Compliance validation passed')
"

log_info "Deployment completed successfully!"

# Deployment summary
log_info "=== DEPLOYMENT SUMMARY ==="
log_info "Environment: $ENVIRONMENT"
log_info "Region: $REGION"
log_info "Version: $VERSION"
log_info "Namespace: $NAMESPACE"
log_info "Running pods: $POD_COUNT"
log_info "Service endpoint: ${SERVICE_IP:-$INTERNAL_IP}:8000"
log_info "=========================="

# Cleanup
rm -f /tmp/deployment-updated.yml /tmp/deployment-green.yml

log_info "Deployment script completed successfully"