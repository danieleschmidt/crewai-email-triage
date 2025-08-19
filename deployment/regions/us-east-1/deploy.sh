#!/bin/bash
# Deployment script for us-east-1

echo "Deploying CrewAI Email Triage v2.0.0"
echo "Region: us-east-1"
echo "Languages: en, es"

# Build and deploy application
docker build -t crewai-email-triage:2.0.0 .
docker tag crewai-email-triage:2.0.0 crewai-email-triage:us-east-1-latest

# Deploy with appropriate configuration
docker run -d \
  --name crewai-triage-us-east-1 \
  -p 8080:8080 \
  -v $(pwd)/deployment/regions/us-east-1/app_config.json:/app/config.json \
  -v $(pwd)/deployment/regions/us-east-1/i18n_config.json:/app/i18n.json \
  crewai-email-triage:us-east-1-latest

echo "Deployment to us-east-1 completed"
