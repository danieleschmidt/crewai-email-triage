-- =============================================================================
-- DATABASE INITIALIZATION SCRIPT FOR CREWAI EMAIL TRIAGE
-- =============================================================================

-- Create database if it doesn't exist (handled by docker environment)
-- CREATE DATABASE IF NOT EXISTS crewai_email_triage;

-- Use the database
-- \c crewai_email_triage;

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- AUDIT TABLES
-- =============================================================================

-- Table for storing email processing audit logs
CREATE TABLE IF NOT EXISTS email_audit (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email_id VARCHAR(255),
    subject TEXT,
    sender VARCHAR(255),
    classification VARCHAR(100),
    priority_score INTEGER,
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing application metrics
CREATE TABLE IF NOT EXISTS metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4),
    metric_type VARCHAR(50) DEFAULT 'gauge',
    labels JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- CONFIGURATION TABLES (for future features)
-- =============================================================================

-- Table for storing user configurations
CREATE TABLE IF NOT EXISTS user_configurations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    config_name VARCHAR(100) NOT NULL,
    config_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, config_name)
);

-- Table for storing classification rules
CREATE TABLE IF NOT EXISTS classification_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rule_name VARCHAR(100) NOT NULL,
    rule_type VARCHAR(50) NOT NULL,
    keywords TEXT[],
    priority INTEGER DEFAULT 0,
    active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- LEARNING & FEEDBACK TABLES (for future ML features)
-- =============================================================================

-- Table for storing user feedback on classifications
CREATE TABLE IF NOT EXISTS classification_feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email_id VARCHAR(255),
    original_classification VARCHAR(100),
    corrected_classification VARCHAR(100),
    user_id VARCHAR(255),
    feedback_type VARCHAR(50), -- 'correction', 'confirmation', 'suggestion'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Table for storing model training data
CREATE TABLE IF NOT EXISTS training_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email_subject TEXT,
    email_body TEXT,
    email_sender VARCHAR(255),
    true_classification VARCHAR(100),
    true_priority INTEGER,
    source VARCHAR(50), -- 'user_feedback', 'manual', 'imported'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- PERFORMANCE TRACKING TABLES
-- =============================================================================

-- Table for tracking processing performance
CREATE TABLE IF NOT EXISTS performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    operation_type VARCHAR(100), -- 'email_processing', 'batch_processing', etc.
    duration_ms INTEGER,
    email_count INTEGER DEFAULT 1,
    memory_usage_mb DECIMAL(10,2),
    cpu_usage_percent DECIMAL(5,2),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- Indexes for email_audit table
CREATE INDEX IF NOT EXISTS idx_email_audit_created_at ON email_audit(created_at);
CREATE INDEX IF NOT EXISTS idx_email_audit_classification ON email_audit(classification);
CREATE INDEX IF NOT EXISTS idx_email_audit_email_id ON email_audit(email_id);

-- Indexes for metrics table
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_metrics_labels ON metrics USING GIN(labels);

-- Indexes for performance tracking
CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp ON performance_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_operation ON performance_metrics(operation_type);

-- Indexes for user configurations
CREATE INDEX IF NOT EXISTS idx_user_configurations_user_id ON user_configurations(user_id);

-- =============================================================================
-- TRIGGERS FOR UPDATED_AT TIMESTAMPS
-- =============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updating timestamps
CREATE TRIGGER update_email_audit_updated_at
    BEFORE UPDATE ON email_audit
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_configurations_updated_at
    BEFORE UPDATE ON user_configurations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_classification_rules_updated_at
    BEFORE UPDATE ON classification_rules
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- INITIAL DATA
-- =============================================================================

-- Insert default classification rules
INSERT INTO classification_rules (rule_name, rule_type, keywords, priority, active) VALUES
('Urgent Keywords', 'urgent', ARRAY['urgent', 'asap', 'immediately', 'critical', 'emergency'], 10, true),
('Spam Keywords', 'spam', ARRAY['unsubscribe', 'marketing', 'promotion', 'offer', 'deal'], 1, true),
('Work Keywords', 'work', ARRAY['meeting', 'project', 'deadline', 'task', 'assignment'], 5, true),
('Support Keywords', 'support', ARRAY['help', 'support', 'issue', 'problem', 'bug'], 7, true)
ON CONFLICT DO NOTHING;

-- =============================================================================
-- SECURITY AND PERMISSIONS
-- =============================================================================

-- Create read-only user for analytics
CREATE USER IF NOT EXISTS crewai_analytics WITH PASSWORD 'analytics_password_change_in_production';
GRANT CONNECT ON DATABASE crewai_email_triage TO crewai_analytics;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO crewai_analytics;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO crewai_analytics;

-- Create application user with limited permissions
CREATE USER IF NOT EXISTS crewai_app WITH PASSWORD 'app_password_change_in_production';
GRANT CONNECT ON DATABASE crewai_email_triage TO crewai_app;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO crewai_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO crewai_app;

-- =============================================================================
-- MAINTENANCE PROCEDURES
-- =============================================================================

-- Procedure to clean old audit data (keep only last 90 days)
CREATE OR REPLACE FUNCTION cleanup_old_audit_data()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM email_audit 
    WHERE created_at < CURRENT_TIMESTAMP - INTERVAL '90 days';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    DELETE FROM metrics 
    WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '30 days';
    
    DELETE FROM performance_metrics 
    WHERE timestamp < CURRENT_TIMESTAMP - INTERVAL '30 days';
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- COMMENTS FOR DOCUMENTATION
-- =============================================================================

COMMENT ON TABLE email_audit IS 'Audit log for all email processing operations';
COMMENT ON TABLE metrics IS 'Application metrics and monitoring data';
COMMENT ON TABLE user_configurations IS 'User-specific configuration settings';
COMMENT ON TABLE classification_rules IS 'Rules for email classification';
COMMENT ON TABLE classification_feedback IS 'User feedback for improving classification accuracy';
COMMENT ON TABLE training_data IS 'Data for training machine learning models';
COMMENT ON TABLE performance_metrics IS 'Performance tracking and optimization data';

-- =============================================================================
-- GRANTS FOR APPLICATION USER
-- =============================================================================

-- Grant default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE ON TABLES TO crewai_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO crewai_app;

-- Commit all changes
COMMIT;