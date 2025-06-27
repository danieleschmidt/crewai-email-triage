# Architecture

This project implements a lightweight email triage workflow using several simple agents. Each agent focuses on one responsibility and exposes a single `run()` method. The `process_email` function in `core.py` powers the base `Agent` but can be replaced with more sophisticated logic in the future.

The basic processing pipeline is as follows:

1. **ClassifierAgent** – categorizes an email using keyword matching.
2. **PriorityAgent** – assigns a priority level based on urgency keywords.
3. **SummarizerAgent** – extracts the first sentence as a brief summary.
4. **ResponseAgent** – drafts a short reply.
5. **triage_email** – helper function that runs the above agents in sequence.
6. **triage_emails** – convenience wrapper to process many emails.

Agents can be composed or used individually as needed. This modular approach keeps each component easy to test and extend.
