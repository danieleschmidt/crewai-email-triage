#!/bin/bash
"""
Autonomous SDLC Scheduler
Sets up cron jobs for continuous value discovery and execution.
"""

REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DISCOVERY_SCRIPT="$REPO_PATH/.terragon/value-discovery.py"
EXECUTOR_SCRIPT="$REPO_PATH/.terragon/autonomous-executor.py"

# Log function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$REPO_PATH/.terragon/scheduler.log"
}

# Create log directory
mkdir -p "$REPO_PATH/.terragon"

log "Setting up autonomous SDLC scheduling..."

# Create cron entries (example - user needs to add these manually)
cat > "$REPO_PATH/.terragon/cron-entries.txt" << 'EOF'
# Terragon Autonomous SDLC Scheduling
# Add these entries to your crontab with: crontab -e

# Hourly value discovery (lightweight scan)
0 * * * * cd REPO_PATH && python3 .terragon/value-discovery.py >> .terragon/discovery.log 2>&1

# Daily comprehensive analysis (full scan with execution)
0 2 * * * cd REPO_PATH && python3 .terragon/autonomous-executor.py >> .terragon/execution.log 2>&1

# Weekly strategic review (comprehensive assessment)
0 3 * * 1 cd REPO_PATH && python3 .terragon/value-discovery.py --comprehensive >> .terragon/strategic.log 2>&1
EOF

# Replace REPO_PATH placeholder
sed -i "s|REPO_PATH|$REPO_PATH|g" "$REPO_PATH/.terragon/cron-entries.txt"

log "Created cron configuration at .terragon/cron-entries.txt"
echo "📅 Autonomous SDLC Scheduling Setup Complete!"
echo ""
echo "To enable automated scheduling, run:"
echo "  crontab -e"
echo ""
echo "Then add the contents of:"
echo "  $REPO_PATH/.terragon/cron-entries.txt"
echo ""
echo "Or run manual discovery anytime with:"
echo "  python3 .terragon/value-discovery.py"
echo ""
echo "Log files will be created at:"
echo "  - .terragon/discovery.log (value discovery runs)"
echo "  - .terragon/execution.log (autonomous execution runs)"
echo "  - .terragon/strategic.log (strategic review runs)"