#!/bin/bash
# Deployment script for ap-southeast-1

echo "Deploying CrewAI Email Triage v2.0.0"
echo "Region: ap-southeast-1"
echo "Languages: en, zh"

# Build and deploy application
docker build -t crewai-email-triage:2.0.0 .
docker tag crewai-email-triage:2.0.0 crewai-email-triage:ap-southeast-1-latest

# Deploy with appropriate configuration
docker run -d \
  --name crewai-triage-ap-southeast-1 \
  -p 8080:8080 \
  -v $(pwd)/deployment/regions/ap-southeast-1/app_config.json:/app/config.json \
  -v $(pwd)/deployment/regions/ap-southeast-1/i18n_config.json:/app/i18n.json \
  crewai-email-triage:ap-southeast-1-latest

echo "Deployment to ap-southeast-1 completed"
