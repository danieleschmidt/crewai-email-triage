#!/usr/bin/env python3
"""Comprehensive security scanner for the system."""

import sys
import os
import re
import ast
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class SecurityScanner:
    """Comprehensive security scanner."""
    
    def __init__(self):
        self.vulnerabilities = []
        self.security_score = 100.0
        
        # Security patterns to check
        self.dangerous_patterns = [
            (r'eval\s*\(', 'Use of eval() function - code injection risk', 'HIGH'),
            (r'exec\s*\(', 'Use of exec() function - code injection risk', 'HIGH'),
            (r'__import__\s*\(', 'Dynamic imports - potential security risk', 'MEDIUM'),
            (r'subprocess.*shell=True', 'Shell injection vulnerability', 'HIGH'),
            (r'os\.system\s*\(', 'Command injection vulnerability', 'HIGH'),
            (r'pickle\.loads?\s*\(', 'Pickle deserialization - code execution risk', 'HIGH'),
            (r'yaml\.load\s*\(', 'Unsafe YAML loading - code execution risk', 'MEDIUM'),
            (r'input\s*\(.*\)', 'User input without validation', 'LOW'),
            (r'password.*=.*["']\w+["']', 'Hardcoded password detected', 'HIGH'),
            (r'secret.*=.*["']\w+["']', 'Hardcoded secret detected', 'HIGH'),
            (r'key.*=.*["']\w+["']', 'Hardcoded key detected', 'HIGH'),
        ]
    
    def scan_file(self, file_path: Path) -> list:
        """Scan individual file for security issues."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for dangerous patterns
            for pattern, description, severity in self.dangerous_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append({
                        'file': str(file_path.relative_to(self.repo_path)),
                        'line': line_num,
                        'pattern': pattern,
                        'description': description,
                        'severity': severity,
                        'code': content.split('\n')[line_num-1].strip()
                    })
            
            # Basic AST analysis for Python files
            if file_path.suffix == '.py':
                try:
                    tree = ast.parse(content)
                    ast_issues = self._analyze_ast(tree, file_path)
                    issues.extend(ast_issues)
                except SyntaxError:
                    pass  # Skip files with syntax errors
        
        except Exception as e:
            issues.append({
                'file': str(file_path.relative_to(self.repo_path)),
                'line': 0,
                'description': f'Failed to scan file: {e}',
                'severity': 'INFO'
            })
        
        return issues
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path) -> list:
        """Analyze AST for security issues."""
        issues = []
        
        class SecurityVisitor(ast.NodeVisitor):
            def visit_Call(self, node):
                # Check for dangerous function calls
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec']:
                        issues.append({
                            'file': str(file_path.relative_to(Path.cwd())),
                            'line': node.lineno,
                            'description': f'Dangerous function call: {node.func.id}',
                            'severity': 'HIGH'
                        })
                self.generic_visit(node)
            
            def visit_Import(self, node):
                # Check for potentially dangerous imports
                for alias in node.names:
                    if alias.name in ['pickle', 'marshal', 'shelve']:
                        issues.append({
                            'file': str(file_path.relative_to(Path.cwd())),
                            'line': node.lineno,
                            'description': f'Potentially dangerous import: {alias.name}',
                            'severity': 'MEDIUM'
                        })
                self.generic_visit(node)
        
        visitor = SecurityVisitor()
        visitor.visit(tree)
        
        return issues
    
    def scan_directory(self, directory: Path) -> dict:
        """Scan entire directory for security issues."""
        results = {
            'files_scanned': 0,
            'total_issues': 0,
            'issues_by_severity': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'INFO': 0},
            'issues': []
        }
        
        # Scan Python files
        for py_file in directory.rglob('*.py'):
            if '.git' in str(py_file) or '__pycache__' in str(py_file):
                continue
            
            results['files_scanned'] += 1
            file_issues = self.scan_file(py_file)
            results['issues'].extend(file_issues)
            results['total_issues'] += len(file_issues)
            
            for issue in file_issues:
                severity = issue.get('severity', 'INFO')
                results['issues_by_severity'][severity] += 1
        
        # Calculate security score
        high_issues = results['issues_by_severity']['HIGH']
        medium_issues = results['issues_by_severity']['MEDIUM']
        low_issues = results['issues_by_severity']['LOW']
        
        # Deduct points based on severity
        self.security_score = max(0, 100 - (high_issues * 20) - (medium_issues * 10) - (low_issues * 2))
        
        return results
    
    def generate_security_report(self, results: dict) -> str:
        """Generate security report."""
        report_lines = [
            "🔐 SECURITY SCAN REPORT",
            "=" * 60,
            f"Files Scanned: {results['files_scanned']}",
            f"Total Issues: {results['total_issues']}",
            f"Security Score: {self.security_score:.1f}/100",
            "",
            "Issues by Severity:",
            f"  HIGH:   {results['issues_by_severity']['HIGH']}",
            f"  MEDIUM: {results['issues_by_severity']['MEDIUM']}",
            f"  LOW:    {results['issues_by_severity']['LOW']}",
            f"  INFO:   {results['issues_by_severity']['INFO']}",
            ""
        ]
        
        if results['issues']:
            report_lines.append("Detailed Issues:")
            report_lines.append("-" * 40)
            
            # Group by severity
            for severity in ['HIGH', 'MEDIUM', 'LOW', 'INFO']:
                severity_issues = [i for i in results['issues'] if i.get('severity') == severity]
                if severity_issues:
                    report_lines.append(f"\n{severity} SEVERITY:")
                    for issue in severity_issues[:10]:  # Limit to first 10 per severity
                        report_lines.append(f"  📁 {issue['file']}:{issue.get('line', '?')}")
                        report_lines.append(f"     {issue['description']}")
                        if 'code' in issue:
                            report_lines.append(f"     Code: {issue['code']}")
                        report_lines.append("")
        
        # Security recommendations
        report_lines.extend([
            "",
            "🛡️  SECURITY RECOMMENDATIONS:",
            "- Input validation: Always validate and sanitize user input",
            "- Avoid dangerous functions: eval(), exec(), pickle.loads()",
            "- Use parameterized queries to prevent injection attacks",
            "- Implement proper error handling to avoid information leakage",
            "- Regular dependency updates and vulnerability scanning",
            "- Principle of least privilege for system access",
            "",
            "SECURITY GATE STATUS:",
        ])
        
        if self.security_score >= 80:
            report_lines.append("✅ SECURITY GATE PASSED")
        elif self.security_score >= 60:
            report_lines.append("⚠️  SECURITY GATE WARNING - Review recommended")
        else:
            report_lines.append("❌ SECURITY GATE FAILED - Critical issues found")
        
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)

def main():
    """Run security scan."""
    repo_path = Path.cwd()
    scanner = SecurityScanner()
    
    print("🔐 Starting comprehensive security scan...")
    results = scanner.scan_directory(repo_path)
    
    report = scanner.generate_security_report(results)
    print(report)
    
    # Return success based on security score
    return scanner.security_score >= 60

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

        self.repo_path = Path('/root/repo')
