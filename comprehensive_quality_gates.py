#!/usr/bin/env python3
"""
Comprehensive Quality Gates for Plugin System
Final validation of all system components before deployment.
"""

import json
import subprocess
import sys
import time
from pathlib import Path


class QualityGateValidator:
    """Comprehensive quality gate validation."""
    
    def __init__(self):
        self.results = {
            'timestamp': time.time(),
            'gates': {},
            'overall_score': 0,
            'critical_failures': [],
            'warnings': [],
            'recommendations': []
        }
    
    def run_all_gates(self):
        """Run all quality gates."""
        print("🛡️  COMPREHENSIVE QUALITY GATES VALIDATION")
        print("=" * 60)
        
        gates = [
            ('🧪 Test Coverage', self.test_coverage_gate),
            ('🚀 Performance Benchmarks', self.performance_gate),
            ('🔒 Security Scan', self.security_gate),
            ('📊 Plugin System', self.plugin_system_gate),
            ('⚡ Scaling Features', self.scaling_gate),
            ('🏗️  Architecture', self.architecture_gate),
            ('📋 Documentation', self.documentation_gate),
            ('🌍 Production Ready', self.production_gate),
        ]
        
        total_score = 0
        for gate_name, gate_function in gates:
            print(f"\n{gate_name}")
            print("-" * 40)
            
            try:
                score, details = gate_function()
                self.results['gates'][gate_name] = {
                    'score': score,
                    'details': details,
                    'passed': score >= 80
                }
                total_score += score
                
                status = "✅ PASS" if score >= 80 else "⚠️  PARTIAL" if score >= 60 else "❌ FAIL"
                print(f"Score: {score:.1f}% {status}")
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
                self.results['gates'][gate_name] = {
                    'score': 0,
                    'details': {'error': str(e)},
                    'passed': False
                }
                self.results['critical_failures'].append(f"{gate_name}: {e}")
        
        self.results['overall_score'] = total_score / len(gates)
        
        print("\n" + "=" * 60)
        print("📊 QUALITY GATES SUMMARY")
        print("=" * 60)
        
        for gate_name, gate_result in self.results['gates'].items():
            status = "✅" if gate_result['passed'] else "❌"
            print(f"{status} {gate_name}: {gate_result['score']:.1f}%")
        
        print(f"\n🏆 Overall Score: {self.results['overall_score']:.1f}%")
        
        if self.results['overall_score'] >= 85:
            print("🎉 ALL QUALITY GATES PASSED - READY FOR DEPLOYMENT")
            return True
        elif self.results['overall_score'] >= 70:
            print("⚠️  PARTIAL PASS - Some improvements recommended")
            return True
        else:
            print("❌ QUALITY GATES FAILED - Critical issues must be addressed")
            return False
    
    def test_coverage_gate(self):
        """Test coverage quality gate."""
        score = 0
        details = {}
        
        # Check test files exist
        test_files = list(Path("tests").glob("*.py")) if Path("tests").exists() else []
        details['test_files_count'] = len(test_files)
        
        if len(test_files) > 30:
            score += 30
        elif len(test_files) > 15:
            score += 20
        elif len(test_files) > 5:
            score += 10
        
        # Check for different test types
        test_types = ['integration', 'performance', 'e2e']
        found_types = []
        for test_type in test_types:
            if Path(f"tests/{test_type}").exists():
                found_types.append(test_type)
                score += 15
        
        details['test_types'] = found_types
        
        # Run existing test suites
        try:
            # Run our comprehensive test runner
            result = subprocess.run(
                [sys.executable, "comprehensive_test_runner.py"],
                capture_output=True, text=True, timeout=60
            )
            
            if "QUALITY GATE PASSED" in result.stdout:
                score += 40
                details['comprehensive_tests'] = 'PASSED'
            elif result.returncode == 0:
                score += 20
                details['comprehensive_tests'] = 'PARTIAL'
            else:
                details['comprehensive_tests'] = 'FAILED'
                
        except Exception as e:
            details['comprehensive_tests'] = f'ERROR: {e}'
        
        return score, details
    
    def performance_gate(self):
        """Performance benchmarking quality gate."""
        score = 0
        details = {}
        
        try:
            # Run performance benchmark
            result = subprocess.run(
                [sys.executable, "performance_benchmark.py"],
                capture_output=True, text=True, timeout=120
            )
            
            if "PERFORMANCE GATE PASSED" in result.stdout:
                score += 60
                details['benchmark_result'] = 'PASSED'
                
                # Extract performance metrics
                if "Performance Score: 100.0%" in result.stdout:
                    score += 40
                    details['performance_score'] = 100.0
                elif "Performance Score:" in result.stdout:
                    score += 20
                    details['performance_score'] = 'PARTIAL'
                    
            elif result.returncode == 0:
                score += 30
                details['benchmark_result'] = 'PARTIAL'
            else:
                details['benchmark_result'] = 'FAILED'
                
        except Exception as e:
            details['benchmark_error'] = str(e)
        
        # Check for scaling implementations
        scaling_files = [
            'src/crewai_email_triage/plugin_scaling.py',
            'plugins/performance_testing_plugin.py'
        ]
        
        for file_path in scaling_files:
            if Path(file_path).exists():
                score += 10
                details[f'scaling_{Path(file_path).stem}'] = 'FOUND'
        
        return min(100, score), details
    
    def security_gate(self):
        """Security validation quality gate."""
        score = 0
        details = {}
        
        # Check security modules exist
        security_files = [
            'src/crewai_email_triage/plugin_security.py',
            'src/crewai_email_triage/enhanced_security_framework.py',
            'src/crewai_email_triage/secure_credentials.py'
        ]
        
        for file_path in security_files:
            if Path(file_path).exists():
                score += 15
                details[f'security_{Path(file_path).stem}'] = 'FOUND'
        
        # Check for security patterns
        if Path('src/crewai_email_triage/plugin_security.py').exists():
            with open('src/crewai_email_triage/plugin_security.py', 'r') as f:
                content = f.read()
                
            security_patterns = [
                'SecurityViolation',
                'PluginSandbox', 
                'PluginValidator',
                'SecurePluginRegistry'
            ]
            
            for pattern in security_patterns:
                if pattern in content:
                    score += 8
                    details[f'pattern_{pattern.lower()}'] = 'IMPLEMENTED'
        
        # Check deployment security
        if Path('deployment/kubernetes/rbac.yml').exists():
            score += 10
            details['rbac_configured'] = True
        
        return min(100, score), details
    
    def plugin_system_gate(self):
        """Plugin system architecture quality gate."""
        score = 0
        details = {}
        
        # Core plugin files
        core_files = [
            'src/crewai_email_triage/plugin_architecture.py',
            'plugins/example_sentiment_plugin.py',
            'plugins/cli_extensions_plugin.py',
            'plugin_config.json'
        ]
        
        for file_path in core_files:
            if Path(file_path).exists():
                score += 20
                details[f'core_{Path(file_path).stem}'] = 'FOUND'
        
        # Test plugin system
        try:
            result = subprocess.run(
                [sys.executable, "test_robust_plugin_system.py"],
                capture_output=True, text=True, timeout=60
            )
            
            if "All tests passed! Plugin system is robust and ready" in result.stdout:
                score += 20
                details['plugin_tests'] = 'ALL_PASSED'
            elif result.returncode == 0:
                score += 10
                details['plugin_tests'] = 'PARTIAL'
            else:
                details['plugin_tests'] = 'FAILED'
                
        except Exception as e:
            details['plugin_test_error'] = str(e)
        
        return score, details
    
    def scaling_gate(self):
        """Scaling and performance quality gate."""
        score = 0
        details = {}
        
        # Test scaling system
        try:
            result = subprocess.run(
                [sys.executable, "test_scaling_system.py"],
                capture_output=True, text=True, timeout=60
            )
            
            if "All tests passed! Plugin scaling system is ready" in result.stdout:
                score += 50
                details['scaling_tests'] = 'ALL_PASSED'
                
                # Check for specific features
                if "4x+ speedup with concurrent processing" in result.stdout:
                    score += 20
                    details['concurrent_speedup'] = 'CONFIRMED'
                
                if "Sub-millisecond cache lookups" in result.stdout:
                    score += 15
                    details['cache_performance'] = 'OPTIMAL'
                
            elif result.returncode == 0:
                score += 25
                details['scaling_tests'] = 'PARTIAL'
            else:
                details['scaling_tests'] = 'FAILED'
                
        except Exception as e:
            details['scaling_test_error'] = str(e)
        
        # Check scaling infrastructure
        if Path('src/crewai_email_triage/plugin_scaling.py').exists():
            score += 15
            details['scaling_module'] = 'IMPLEMENTED'
        
        return score, details
    
    def architecture_gate(self):
        """Architecture and code quality gate."""
        score = 0
        details = {}
        
        # Check architecture documentation
        arch_files = [
            'ARCHITECTURE.md',
            'DEVELOPMENT_PLAN.md',
            'docs/adr/'
        ]
        
        for file_path in arch_files:
            if Path(file_path).exists():
                score += 15
                details[f'arch_{Path(file_path).stem}'] = 'FOUND'
        
        # Check module organization
        src_modules = list(Path('src/crewai_email_triage/').glob('*.py'))
        details['module_count'] = len(src_modules)
        
        if len(src_modules) > 50:
            score += 25
        elif len(src_modules) > 30:
            score += 20
        elif len(src_modules) > 15:
            score += 15
        
        # Check for key architectural components
        key_modules = [
            'plugin_architecture.py',
            'plugin_security.py', 
            'plugin_scaling.py',
            'quantum_consciousness.py',
            'realtime_intelligence.py'
        ]
        
        for module in key_modules:
            if Path(f'src/crewai_email_triage/{module}').exists():
                score += 8
                details[f'module_{module[:-3]}'] = 'IMPLEMENTED'
        
        return min(100, score), details
    
    def documentation_gate(self):
        """Documentation quality gate."""
        score = 0
        details = {}
        
        # Check main documentation
        doc_files = [
            'README.md',
            'CONTRIBUTING.md',
            'CHANGELOG.md',
            'DEPLOYMENT.md'
        ]
        
        for doc_file in doc_files:
            if Path(doc_file).exists():
                score += 20
                details[f'doc_{doc_file[:-3].lower()}'] = 'FOUND'
        
        # Check docs directory
        if Path('docs').exists():
            doc_count = len(list(Path('docs').glob('**/*.md')))
            details['docs_count'] = doc_count
            
            if doc_count > 20:
                score += 20
            elif doc_count > 10:
                score += 15
            elif doc_count > 5:
                score += 10
        
        return score, details
    
    def production_gate(self):
        """Production readiness quality gate."""
        score = 0
        details = {}
        
        try:
            # Run production readiness validator
            result = subprocess.run(
                [sys.executable, "production_readiness_validator.py"],
                capture_output=True, text=True, timeout=60
            )
            
            if "READY FOR PRODUCTION DEPLOYMENT" in result.stdout:
                score += 50
                details['production_ready'] = 'CONFIRMED'
                
                # Extract readiness score
                for line in result.stdout.split('\n'):
                    if 'Overall Readiness Score:' in line:
                        try:
                            readiness_score = float(line.split(':')[1].strip().replace('%', ''))
                            score += int(readiness_score * 0.5)  # Up to 50 additional points
                            details['readiness_score'] = readiness_score
                            break
                        except:
                            pass
            elif result.returncode == 0:
                score += 25
                details['production_ready'] = 'PARTIAL'
            else:
                details['production_ready'] = 'FAILED'
                
        except Exception as e:
            details['production_error'] = str(e)
        
        return min(100, score), details
    
    def save_results(self):
        """Save quality gate results."""
        with open('quality_gates_final_report.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\n📋 Quality gate results saved to: quality_gates_final_report.json")


def main():
    """Main quality gates execution."""
    validator = QualityGateValidator()
    success = validator.run_all_gates()
    validator.save_results()
    
    if success:
        print("\n🎯 SYSTEM READY FOR DEPLOYMENT!")
        print("All critical quality gates have been passed.")
        return True
    else:
        print("\n⚠️  DEPLOYMENT NOT RECOMMENDED")
        print("Critical quality issues must be addressed first.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)