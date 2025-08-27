# 🚀 Terragon Labs - Enhanced Email Triage Usage Examples

This document provides comprehensive examples of using the enhanced CrewAI email triage system with all Generation 4 features.

## 🧠 Basic Intelligent Processing

### Simple Email Analysis

```python
from src.crewai_email_triage.core import process_email_intelligent

# Basic intelligent processing
email_content = """
Dear Team,

Please review the quarterly report by Friday. This is urgent and requires 
immediate attention from all stakeholders.

Best regards,
Project Manager
"""

result = process_email_intelligent(email_content)
print(f"Language detected: {result['analysis']['language_detected']}")
print(f"Sentiment: {result['analysis']['sentiment']}")
print(f"Keywords: {result['analysis']['keywords']}")
print(f"Urgency indicators: {result['analysis']['urgency_indicators']}")
print(f"Confidence score: {result['analysis']['confidence_score']}")
```

**Expected Output:**
```
Language detected: en
Sentiment: neutral
Keywords: ['team', 'review', 'quarterly', 'report', 'friday']
Urgency indicators: ['high:urgent', 'high:immediate']
Confidence score: 1.0
```

## 🔮 Quantum NLP Processing

### Advanced Semantic Analysis

```python
from src.crewai_email_triage.quantum_nlp_engine import get_quantum_nlp_engine

# Initialize quantum NLP engine
quantum_engine = get_quantum_nlp_engine()

# Analyze email with quantum algorithms
email_text = """
URGENT: Security breach detected in our payment system. 
Please implement emergency protocols immediately. 
All team members must verify their credentials.
"""

analysis = quantum_engine.analyze_email_quantum(email_text)

print(f"Language confidence: {analysis.language_confidence:.3f}")
print(f"Topic clusters: {analysis.topic_clusters}")
print(f"Emotion spectrum: {analysis.emotion_spectrum}")
print(f"Intent classification: {analysis.intent_classification}")
print(f"Entity extraction: {analysis.entity_extraction}")
print(f"Quantum coherence: {analysis.quantum_coherence:.3f}")
print(f"Processing time: {analysis.processing_time_ms:.2f}ms")
```

**Expected Output:**
```
Language confidence: 0.900
Topic clusters: [('technical', 0.75), ('business', 0.45)]
Emotion spectrum: {'fear': 0.6, 'anger': 0.3, 'trust': 0.2}
Intent classification: [('request', 0.8), ('information', 0.4)]
Entity extraction: [('payment', 'organization', 0.7), ('system', 'organization', 0.6)]
Quantum coherence: 0.742
Processing time: 1.23ms
```

## 🛡️ Robust Processing with Security

### Enterprise-Grade Email Validation

```python
from src.crewai_email_triage.robust_intelligence_framework import get_robust_framework, ValidationLevel

# Get robust framework instance
framework = get_robust_framework()

def email_processor(content):
    return {"processed": True, "content_length": len(content)}

# Process email with enterprise security
suspicious_email = """
WINNER! You've won $1,000,000! 
Click here to claim: http://suspicious-site.com/claim
Verify your account at http://192.168.1.1/verify immediately!
"""

result = framework.robust_process_email(
    suspicious_email,
    email_processor,
    validation_level=ValidationLevel.STRICT,
    timeout=30.0
)

if result['success']:
    print("Email processed successfully")
    validation = result['validation_result']
    print(f"Confidence: {validation['confidence_score']:.3f}")
    print(f"Security threats: {len(validation['security_threats'])}")
else:
    print(f"Processing failed: {result['error']}")
    if 'security_threats' in result:
        for threat in result['security_threats'][:3]:
            print(f"- {threat['category']}: {threat['level']}")

# Check system health
health = framework.get_health_status()
print(f"System health: {health['overall_health']}")
print(f"Availability: {health['robustness_metrics']['availability']:.1%}")
```

**Expected Output:**
```
Processing failed: Email validation failed
- malicious_urls: high
- phishing_indicators: high
- phishing_indicators: high
System health: unhealthy
Availability: 50.0%
```

## ⚡ HyperScale Performance Processing

### Auto-Scaling and Intelligent Caching

```python
from src.crewai_email_triage.hyperscale_performance_engine import get_performance_engine, hyperscale_optimization
import time

# Initialize performance engine
engine = get_performance_engine()
engine.start()

# Decorator for automatic optimization
@hyperscale_optimization(use_cache=True, use_parallel=True)
def enhanced_email_processor(content):
    # Simulate processing time
    time.sleep(0.01)
    return {
        "processed": True,
        "analysis": {
            "length": len(content),
            "word_count": len(content.split()),
            "processing_timestamp": time.time()
        }
    }

# Process single email (will be cached)
email = "This is a test email for performance optimization."
result = enhanced_email_processor(email)
print(f"Single processing result: {result['processed']}")

# Process batch of emails with optimization
test_emails = [
    f"Test email {i} with various content lengths and patterns."
    for i in range(50)
]

start_time = time.time()
batch_results = engine.process_batch_optimized(
    test_emails, 
    enhanced_email_processor,
    batch_size=10,
    use_cache=True
)
processing_time = (time.time() - start_time) * 1000

print(f"Batch processing completed:")
print(f"- Processed: {len([r for r in batch_results if r])}/50 emails")
print(f"- Total time: {processing_time:.2f}ms")
print(f"- Average per email: {processing_time/50:.2f}ms")

# Get performance report
report = engine.get_performance_report()
print(f"Performance metrics:")
print(f"- Requests/second: {report['current_metrics']['requests_per_second']:.2f}")
print(f"- Cache hit rate: {report['cache_performance']['hit_rate']:.1%}")
print(f"- Active workers: {report['current_metrics']['active_workers']}")

# Stop engine
engine.stop()
```

## 🧪 Comprehensive Testing

### Quality Gates and Validation

```python
from src.crewai_email_triage.comprehensive_test_suite import run_quality_gates, TestDataGenerator
from src.crewai_email_triage.core import process_email_intelligent
from src.crewai_email_triage.robust_intelligence_framework import get_robust_framework

# Define processing and validation functions
def test_processor(content):
    return process_email_intelligent(content)

def test_validator(content):
    framework = get_robust_framework()
    return framework.validator.validate_email(content)

# Run comprehensive quality gates
print("Running quality gates...")
quality_passed, report = run_quality_gates(test_processor, test_validator)

print(f"Quality gates status: {'PASSED' if quality_passed else 'FAILED'}")
print(f"Overall pass rate: {report.pass_rate:.1%}")
print(f"Tests run: {report.total_tests}")
print(f"Passed: {report.passed_tests}")
print(f"Failed: {report.failed_tests}")

# Category breakdown
for category, results in report.results_by_category.items():
    if results:
        passed = len([r for r in results if r.passed])
        total = len(results)
        print(f"{category.value}: {passed}/{total} passed")

# Performance summary
if report.performance_summary:
    print(f"Average response time: {report.performance_summary['avg_response_time_ms']:.2f}ms")

# Security summary  
if report.security_summary:
    print(f"Average detection rate: {report.security_summary['avg_detection_rate']:.1%}")

# Generate test data for various scenarios
print("\nGenerating test data samples:")
normal_email = TestDataGenerator.generate_email_content("normal", 200)
urgent_email = TestDataGenerator.generate_email_content("urgent", 150)
spam_email = TestDataGenerator.generate_email_content("spam", 100)

print(f"Normal: {normal_email[:50]}...")
print(f"Urgent: {urgent_email[:50]}...")  
print(f"Spam: {spam_email[:50]}...")
```

## 🌍 Global Processing with Compliance

### Multi-Language and Compliance-Aware Processing

```python
from src.crewai_email_triage.global_intelligence_framework import get_global_framework

# Initialize global framework
global_framework = get_global_framework()

# English business email
english_email = """
Dear Sir/Madam,

I am writing to request access to my personal data that your organization 
has collected according to GDPR Article 15. Please provide a copy of all 
personal information you hold about me.

Regards,
John Smith
Data Subject
"""

result_en = global_framework.process_global_email(
    english_email,
    processing_region="eu",
    target_regions=["us", "uk"]
)

print("English Email Analysis:")
print(f"Language: {result_en.language_profile.language.value} (confidence: {result_en.language_profile.confidence:.3f})")
print(f"Data classification: {result_en.compliance_profile.data_classification.value}")
print(f"Applicable regions: {[r.value for r in result_en.compliance_profile.regions]}")
print(f"Consent required: {result_en.compliance_profile.consent_required}")
print(f"Right to deletion: {result_en.compliance_profile.right_to_deletion}")
print(f"Retention period: {result_en.compliance_profile.retention_days} days")
print(f"Data residency: {result_en.data_residency_region}")
print(f"Classification tags: {result_en.classification_tags[:5]}")

# Spanish email with sensitive data
spanish_email = """
Estimado equipo de RRHH,

Necesito actualizar mi información personal:
- Número de la Seguridad Social: 123-45-6789
- Información médica: Diabetes tipo 2
- Cuenta bancaria: ES91 2100 0418 4502 0005 1332

Esta información es confidencial.

Saludos,
María García
"""

result_es = global_framework.process_global_email(
    spanish_email,
    processing_region="eu",
    target_regions=["es"]
)

print("\nSpanish Email Analysis:")
print(f"Language: {result_es.language_profile.language.value}")
print(f"Data classification: {result_es.compliance_profile.data_classification.value}")
print(f"Special categories: {result_es.compliance_profile.special_categories}")
print(f"Encryption required: {result_es.compliance_profile.encryption_required}")
print(f"Localized responses: {list(result_es.localized_response.keys())}")

# Framework capabilities
capabilities = global_framework.get_compliance_summary()
print(f"\nFramework Capabilities:")
print(f"Languages supported: {len(capabilities['supported_languages'])}")
print(f"Compliance regions: {len(capabilities['supported_regions'])}")
print(f"Privacy frameworks: {capabilities['privacy_frameworks']}")
print(f"Security features: {capabilities['security_features']}")
```

**Expected Output:**
```
English Email Analysis:
Language: en (confidence: 0.850)
Data classification: personal
Applicable regions: ['eu', 'us', 'uk']
Consent required: True
Right to deletion: True
Retention period: 365 days
Data residency: eu
Classification tags: ['lang:en', 'classification:personal', 'region:eu', 'rights:deletion', 'privacy:consent_required']

Spanish Email Analysis:
Language: es
Data classification: sensitive_personal
Special categories: ['health']
Encryption required: True
Localized responses: ['es', 'en']

Framework Capabilities:
Languages supported: 15
Compliance regions: 12
Privacy frameworks: ['GDPR', 'CCPA', 'PIPEDA', 'PDPA', 'LGPD']
Security features: ['Encryption', 'Audit trails', 'Data residency', 'Access controls']
```

## 🔧 Integration Examples

### CLI Integration

```bash
# Basic triage with enhanced intelligence
python triage.py --message "Urgent: Please review the security incident report" --ai-enhanced --show-insights

# Batch processing with global compliance
python triage.py --batch-file emails.txt --enhanced --adaptive --parallel --max-workers 8

# Security scan with validation
python triage.py --security-scan --message "Click here: http://suspicious.com/verify" --sanitization-level strict

# Performance monitoring
python triage.py --performance-insights --benchmark --verbose

# Global processing with compliance
python triage.py --message "Solicitud de acceso a datos personales" --ai-format detailed --show-insights
```

### API Integration

```python
# FastAPI integration example
from fastapi import FastAPI
from src.crewai_email_triage.global_intelligence_framework import get_global_framework

app = FastAPI()
global_framework = get_global_framework()

@app.post("/api/triage/global")
async def process_email_global(
    content: str,
    processing_region: str = "global",
    target_regions: List[str] = None
):
    result = global_framework.process_global_email(
        content, processing_region, target_regions
    )
    
    return {
        "language": result.language_profile.language.value,
        "confidence": result.language_profile.confidence,
        "data_classification": result.compliance_profile.data_classification.value,
        "compliance_regions": [r.value for r in result.compliance_profile.regions],
        "encryption_required": result.compliance_profile.encryption_required,
        "localized_responses": result.localized_response,
        "audit_metadata": result.audit_metadata
    }
```

## 📊 Monitoring and Analytics

### Performance Dashboard

```python
from src.crewai_email_triage.adaptive_monitoring_system import get_monitoring_system

# Initialize monitoring
monitoring = get_monitoring_system()
monitoring.start_monitoring()

# Add custom metrics
monitoring.add_metric_point("custom_processing_time", 45.2, {"type": "email"})
monitoring.add_metric_point("business_emails_processed", 150, {"priority": "high"})

# Get monitoring status
status = monitoring.get_monitoring_status()
print(f"Monitoring active: {status['monitoring_active']}")
print(f"Tracked metrics: {len(status['tracked_metrics'])}")
print(f"Active alerts: {status['active_alerts_count']}")

# Get alerts summary
alerts = monitoring.get_alerts_summary()
print(f"Active alerts: {alerts['active_alerts']}")
for alert in alerts['recent_alerts'][:3]:
    print(f"- {alert['severity']}: {alert['title']}")

# Custom alert handler
def email_alert_handler(alert):
    print(f"ALERT: {alert.severity.value.upper()} - {alert.title}")
    if alert.severity.value in ['error', 'critical']:
        # Send notification to ops team
        send_notification(alert)

monitoring.add_alert_handler(email_alert_handler)
```

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
# Dockerfile example
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e ".[test,dev,performance]"

# Set environment variables
ENV PYTHONPATH=/app
ENV CREWAI_ENV=production
ENV CREWAI_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "from src.crewai_email_triage.health import get_health_checker; print('healthy' if get_health_checker().check_health().status.name == 'HEALTHY' else exit(1))"

# Start with performance monitoring
CMD ["python", "triage.py", "--start-monitor", "--export-metrics", "--metrics-port", "8080"]
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crewai-email-triage
spec:
  replicas: 3
  selector:
    matchLabels:
      app: crewai-email-triage
  template:
    metadata:
      labels:
        app: crewai-email-triage
    spec:
      containers:
      - name: email-triage
        image: crewai-email-triage:latest
        ports:
        - containerPort: 8080
          name: metrics
        env:
        - name: CREWAI_ENV
          value: "production"
        - name: CREWAI_REGION
          value: "us-east-1"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: crewai-email-triage-service
spec:
  selector:
    app: crewai-email-triage
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
    name: http
  - protocol: TCP
    port: 8080
    targetPort: 8080
    name: metrics
```

This comprehensive usage guide demonstrates all the enhanced capabilities of the Generation 4 CrewAI email triage system, from basic processing to enterprise deployment scenarios.