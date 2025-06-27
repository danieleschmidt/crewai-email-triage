# CrewAI-Email-Triage

Smart email assistant that classifies, prioritizes, summarizes, and drafts replies using CrewAI's multi-agent orchestration.

## Features

- **Multi-Agent Email Processing**: Specialized agents for classification, summarization, and response drafting
- **Intelligent Prioritization**: Automatic urgency scoring and category assignment
- **Context-Aware Responses**: Draft replies that maintain conversation context and tone
- **Email Provider Integration**: Gmail, Outlook, and IMAP/POP3 support
- **Custom Workflows**: Configurable agent workflows for different email types
- **Bulk Processing**: Handle large email volumes with parallel agent execution

## Architecture

```
Incoming Email → Classifier Agent → Priority Agent → Summarizer Agent → Response Agent → Output
                      ↓                ↓               ↓                ↓
                   Category         Urgency        Key Points      Draft Reply
```

The ``triage_email`` function in ``pipeline.py`` exposes this workflow as a
single call, returning the outputs of all agents as a dictionary. For bulk
processing, ``triage_emails`` accepts a list of messages and returns a list of
results.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure email credentials
python setup.py --provider gmail

# Run email triage
python triage.py --inbox --limit 50
```


### Quick Demo
Run the pipeline on a single message or a list of messages:
```bash
python triage.py "Your message here"
python triage.py --high-keywords urgent,asap "Important update"
python triage.py --json "Check status"
python - <<'EOF'
from crewai_email_triage import triage_emails
print(triage_emails(["first email", "second email"]))
EOF
```


## Agent Crew

### Classifier Agent
- **Role**: Email Categorization Specialist
- **Goal**: Accurately classify emails into categories (work, personal, spam, urgent, etc.)
- **Tools**: Text analysis, sender reputation, keyword matching

### Priority Agent  
- **Role**: Urgency Assessment Expert
- **Goal**: Score emails by importance and time sensitivity
- **Tools**: Deadline detection, sender importance, content analysis

### Summarizer Agent
- **Role**: Content Distillation Specialist
- **Goal**: Create concise, actionable summaries of email content
- **Tools**: Key point extraction, context preservation, length optimization

### Response Agent
- **Role**: Communication Specialist
- **Goal**: Draft appropriate replies maintaining tone and context
- **Tools**: Template matching, tone analysis, personalization

## Configuration

```yaml
# config/email_config.yml
crew:
  agents:
    classifier:
      model: "gpt-4"
      temperature: 0.2
      categories: ["urgent", "work", "personal", "newsletter", "spam"]
    
    priority:
      model: "gpt-3.5-turbo"
      scoring_range: [1, 10]
      factors: ["deadline", "sender", "keywords"]
    
    summarizer:
      model: "gpt-4"
      max_length: 150
      include_action_items: true
    
    response:
      model: "gpt-4"
      tone_matching: true
      templates_enabled: true

email:
  provider: "gmail"
  check_interval: 300  # seconds
  auto_reply: false
  backup_originals: true
```

## Usage Examples

### Process Inbox
```bash
# Full inbox processing
python triage.py --all

# Process last 24 hours
python triage.py --since yesterday

# Specific folder
python triage.py --folder "Important"
```

### Interactive Mode
```bash
# Review and approve actions
python triage.py --interactive

# Generate reports
python triage.py --report --date-range "last_week"
```

## Sample Output

```json
{
  "email_id": "abc123",
  "classification": {
    "category": "work",
    "subcategory": "meeting_request",
    "confidence": 0.95
  },
  "priority": {
    "score": 8,
    "reasoning": "Meeting with CEO scheduled for tomorrow",
    "urgency": "high"
  },
  "summary": "Meeting request from Sarah (CEO) for tomorrow 2 PM to discuss Q4 budget. Requires budget document preparation.",
  "suggested_response": "Hi Sarah,\n\nConfirmed for tomorrow at 2 PM. I'll have the Q4 budget analysis ready for review.\n\nBest regards,\n[Your name]"
}
```

## Integrations

### Email Providers
- Gmail API
- Microsoft Graph (Outlook)
- IMAP/POP3
- Exchange Server

### Productivity Tools
- Slack notifications
- Calendar integration
- Task management (Todoist, Asana)
- CRM systems (Salesforce, HubSpot)

## Advanced Features

- **Learning Mode**: Agents improve from user feedback
- **Custom Templates**: Personalized response templates
- **Bulk Operations**: Process thousands of emails efficiently
- **Analytics Dashboard**: Email pattern insights and metrics
- **Security Scanning**: Phishing and malware detection
- **Multi-Language**: Support for international email processing

## Deployment

### Local Development
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### Production
```bash
# Docker deployment
docker-compose up -d

# Or use cloud deployment
python deploy.py --platform aws
```

## Contributing

We welcome contributions! Areas for improvement:
- Additional email provider integrations
- Enhanced natural language processing
- New agent specializations
- Performance optimizations
- Security enhancements

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Privacy & Security

- All email processing happens locally by default
- API keys are encrypted and stored securely
- Optional cloud processing with data encryption
- Full audit trail for compliance requirements
