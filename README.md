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

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
# Install the package in editable mode
pip install -e .

# Configure email credentials
python setup.py --provider gmail

# Run email triage on a single message
python triage.py --message "Quarterly report needed ASAP!"
# => {"category": "urgent", "priority": 10, ...}

# Pretty-print the result
python triage.py --message "Quarterly report needed ASAP!" --pretty

# Save the result to a file
python triage.py --message "Quarterly report needed ASAP!" --output result.json
cat result.json
# => {"category": "urgent", "priority": 10, ...}

# Check the version
python triage.py --version
# => triage.py 0.1.0

# Exactly one of --message, --stdin, --file, --batch-file, or --interactive is required
# These options are mutually exclusive

# Or read the message from standard input
echo "Quarterly report needed ASAP!" | python triage.py --stdin

# Or read the message from a file
echo "Quarterly report needed ASAP!" > email.txt
python triage.py --file email.txt

# Process multiple messages from a file
echo -e "Urgent meeting tomorrow!\nAnother message" > batch.txt
python triage.py --batch-file batch.txt
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
- **Heuristics**: "urgent" or all-caps text yields score 10; keywords like
  "deadline" or an exclamation mark score 8; otherwise 5.

### Summarizer Agent
- **Role**: Content Distillation Specialist
- **Goal**: Create concise, actionable summaries of email content
- **Tools**: Key point extraction, context preservation, length optimization

### Response Agent
- **Role**: Communication Specialist
- **Goal**: Draft appropriate replies maintaining tone and context
- **Tools**: Template matching, tone analysis, personalization

### Triage Pipeline
Use ``triage_email`` to run all agents in sequence:

```python
from crewai_email_triage import triage_email

result = triage_email("Quarterly report needed ASAP!")
print(result)
# {'category': 'urgent', 'priority': 10, 'summary': 'Quarterly report needed ASAP!', 'response': 'Thanks for your email'}
```

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

# Type a message and press Enter
Quarterly report needed ASAP!
{"category": "urgent", "priority": 10, "summary": "Quarterly report needed ASAP!", "response": "Thanks for your email"}

# Submit an empty line to quit
# Or press Ctrl+C

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
