#!/usr/bin/env python3
"""
Security audit script for SpiralDeltaDB before public git push.

Ensures no proprietary content, API keys, or sensitive information
is exposed in the public repository.
"""

import re
import os
import sys
from pathlib import Path
from typing import List, Dict, Set
import json


class SecurityAuditor:
    """Security auditor for open-source repository."""
    
    def __init__(self, repo_path: str = "."):
        """Initialize security auditor."""
        self.repo_path = Path(repo_path)
        
        # Patterns that should never be in public code
        self.sensitive_patterns = [
            # API Keys and tokens
            r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']?[a-zA-Z0-9_-]{20,}',
            r'(?i)(secret[_-]?key|secretkey)\s*[=:]\s*["\']?[a-zA-Z0-9_-]{20,}',
            r'(?i)(access[_-]?token|accesstoken)\s*[=:]\s*["\']?[a-zA-Z0-9_.-]{20,}',
            r'(?i)(bearer\s+[a-zA-Z0-9_.-]{20,})',
            
            # Database connections
            r'(?i)(password|pwd)\s*[=:]\s*["\'][^"\']{3,}["\']',
            r'(?i)mongodb://[^/\s]+:[^@\s]+@',
            r'(?i)postgres://[^/\s]+:[^@\s]+@',
            
            # Cloud credentials
            r'(?i)aws[_-]?access[_-]?key[_-]?id\s*[=:]\s*["\']?[A-Z0-9]{20}',
            r'(?i)aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*["\']?[A-Za-z0-9/+=]{40}',
            
            # Private emails and domains
            r'[a-zA-Z0-9._%+-]+@(?:monarch-private|internal-monarch|monarch-corp)\.com',
            
            # Internal URLs
            r'https?://(?:internal|private|corp)\.monarch(?:ai)?\.com',
            
            # Monarch private references
            r'(?i)monarch[_-]?(?:private|internal|corp|enterprise)[_-]?(?:key|token|secret)',
        ]
        
        # Proprietary business model indicators
        self.business_sensitive_patterns = [
            # Specific pricing information
            r'\$[0-9,]+\s*(?:per|/)\s*(?:month|year|user|query)',
            
            # Internal project codenames (examples)
            r'(?i)project[_-]?(?:aurora|nebula|quantum|genesis)',
            
            # Internal team names
            r'(?i)team[_-]?(?:alpha|beta|gamma|delta)[_-]?(?:internal|private)',
        ]
        
        # File patterns to exclude from public repo
        self.private_file_patterns = [
            r'.*/?monarch[_-]?(?:private|internal|corp)/',
            r'.*/?(?:private|internal|corp)[_-]?monarch/',
            r'.*/?enterprise[_-]?(?:config|keys?)/',
            r'.*\.(?:key|pem|p12|pfx)$',
            r'.*/?\.env\.(?:production|enterprise|private)',
            r'.*/?secrets?/',
            r'.*/?credentials?/',
        ]
        
        # Directories that should never be public
        self.private_directories = {
            'monarch-core', 'private', 'enterprise', 'soft_armor', 
            'conscious_engine', 'secrets', 'credentials', 'internal'
        }
        
        # Directories to skip during scan (not security issues)
        self.skip_directories = {
            'venv', '.venv', 'env', '.env', 'node_modules', '__pycache__',
            '.git', '.pytest_cache', '.mypy_cache', 'build', 'dist',
            '.tox', '.nox', 'target'
        }
        
        # Files to always exclude
        self.excluded_files = {
            '.monarch-credentials', '.enterprise-config', 
            'private-keys.json', 'enterprise-secrets.yaml',
            'security_audit_report.json'  # Exclude security reports
        }
        
    def scan_file_content(self, file_path: Path) -> List[Dict]:
        """Scan file content for sensitive information."""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Check for sensitive patterns
            for pattern in self.sensitive_patterns:
                matches = re.finditer(pattern, content, re.MULTILINE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append({
                        'type': 'CRITICAL',
                        'file': str(file_path),
                        'line': line_num,
                        'issue': 'Potential API key/secret detected',
                        'pattern': pattern[:50] + '...' if len(pattern) > 50 else pattern,
                        'match': match.group()[:100] + '...' if len(match.group()) > 100 else match.group()
                    })
            
            # Check for business sensitive patterns
            for pattern in self.business_sensitive_patterns:
                matches = re.finditer(pattern, content, re.MULTILINE)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    issues.append({
                        'type': 'WARNING',
                        'file': str(file_path),
                        'line': line_num,
                        'issue': 'Potential business sensitive information',
                        'pattern': pattern[:50] + '...' if len(pattern) > 50 else pattern,
                        'match': match.group()[:100] + '...' if len(match.group()) > 100 else match.group()
                    })
                    
        except Exception as e:
            issues.append({
                'type': 'ERROR',
                'file': str(file_path),
                'issue': f'Could not scan file: {e}'
            })
            
        return issues
    
    def check_file_paths(self) -> List[Dict]:
        """Check for problematic file paths."""
        issues = []
        
        for root, dirs, files in os.walk(self.repo_path):
            # Remove private and skip directories from traversal
            dirs[:] = [d for d in dirs if d not in self.private_directories and d not in self.skip_directories]
            
            current_path = Path(root)
            
            # Check directory names
            for dir_name in dirs:
                if dir_name in self.private_directories:
                    issues.append({
                        'type': 'CRITICAL',
                        'file': str(current_path / dir_name),
                        'issue': 'Private directory should not be in public repo'
                    })
            
            # Check file names and paths
            for file_name in files:
                file_path = current_path / file_name
                relative_path = file_path.relative_to(self.repo_path)
                
                # Check excluded files
                if file_name in self.excluded_files:
                    issues.append({
                        'type': 'CRITICAL',
                        'file': str(relative_path),
                        'issue': 'Private file should not be in public repo'
                    })
                
                # Check private file patterns
                for pattern in self.private_file_patterns:
                    if re.match(pattern, str(relative_path)):
                        issues.append({
                            'type': 'CRITICAL',
                            'file': str(relative_path),
                            'issue': 'File matches private pattern'
                        })
        
        return issues
    
    def scan_repository(self) -> Dict:
        """Perform comprehensive security scan."""
        print("🔒 Starting Security Audit for Public Git Push")
        print("=" * 60)
        
        all_issues = []
        scanned_files = 0
        
        # Scan file contents
        for root, dirs, files in os.walk(self.repo_path):
            # Skip private and unwanted directories
            dirs[:] = [d for d in dirs if d not in self.private_directories and d not in self.skip_directories]
            
            # Skip hidden directories except .github
            dirs[:] = [d for d in dirs if not d.startswith('.') or d in {'.github'}]
            
            for file_name in files:
                file_path = Path(root) / file_name
                
                # Skip binary files and certain extensions
                if file_path.suffix in {'.pyc', '.so', '.dll', '.exe', '.bin', '.db', '.sqlite', '.sqlite3'}:
                    continue
                
                # Skip hidden files
                if file_name.startswith('.'):
                    continue
                
                # Scan text files
                if file_path.suffix in {'.py', '.md', '.txt', '.json', '.yaml', '.yml', '.toml', '.cfg', '.ini', '.sh'}:
                    file_issues = self.scan_file_content(file_path)
                    all_issues.extend(file_issues)
                    scanned_files += 1
        
        # Check file paths
        path_issues = self.check_file_paths()
        all_issues.extend(path_issues)
        
        # Categorize issues
        critical_issues = [i for i in all_issues if i.get('type') == 'CRITICAL']
        warning_issues = [i for i in all_issues if i.get('type') == 'WARNING']
        error_issues = [i for i in all_issues if i.get('type') == 'ERROR']
        
        results = {
            'scan_summary': {
                'files_scanned': scanned_files,
                'total_issues': len(all_issues),
                'critical_issues': len(critical_issues),
                'warning_issues': len(warning_issues),
                'error_issues': len(error_issues)
            },
            'critical_issues': critical_issues,
            'warning_issues': warning_issues,
            'error_issues': error_issues,
            'security_status': 'FAIL' if critical_issues else ('WARNING' if warning_issues else 'PASS')
        }
        
        return results
    
    def generate_security_report(self, results: Dict, output_path: str = "security_audit_report.json"):
        """Generate detailed security report."""
        report_path = self.repo_path / output_path
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Security report saved to: {report_path}")
        return report_path
    
    def print_security_summary(self, results: Dict):
        """Print formatted security summary."""
        summary = results['scan_summary']
        status = results['security_status']
        
        print(f"\n🔍 SECURITY AUDIT RESULTS")
        print("=" * 60)
        print(f"Files Scanned: {summary['files_scanned']}")
        print(f"Total Issues: {summary['total_issues']}")
        print(f"Critical Issues: {summary['critical_issues']}")
        print(f"Warning Issues: {summary['warning_issues']}")
        print(f"Error Issues: {summary['error_issues']}")
        
        if status == 'PASS':
            print(f"\n✅ SECURITY STATUS: {status}")
            print("Repository is safe for public git push!")
        elif status == 'WARNING':
            print(f"\n⚠️ SECURITY STATUS: {status}")
            print("Repository has warnings but may be safe for public push.")
            print("Review warnings below:")
        else:
            print(f"\n❌ SECURITY STATUS: {status}")
            print("CRITICAL ISSUES FOUND - DO NOT PUSH TO PUBLIC GIT!")
        
        # Print critical issues
        if results['critical_issues']:
            print(f"\n🚨 CRITICAL ISSUES:")
            for issue in results['critical_issues'][:10]:  # Show first 10
                print(f"  ❌ {issue['file']}")
                print(f"     Issue: {issue['issue']}")
                if 'line' in issue:
                    print(f"     Line: {issue['line']}")
                if 'match' in issue:
                    print(f"     Match: {issue['match'][:80]}...")
                print()
        
        # Print warnings
        if results['warning_issues']:
            print(f"\n⚠️ WARNINGS:")
            for issue in results['warning_issues'][:5]:  # Show first 5
                print(f"  ⚠️ {issue['file']}")
                print(f"     Issue: {issue['issue']}")
                if 'line' in issue:
                    print(f"     Line: {issue['line']}")
                print()


def main():
    """Main security audit function."""
    auditor = SecurityAuditor()
    
    # Perform security scan
    results = auditor.scan_repository()
    
    # Print summary
    auditor.print_security_summary(results)
    
    # Generate report
    auditor.generate_security_report(results)
    
    # Exit with appropriate code
    if results['security_status'] == 'FAIL':
        print(f"\n🛑 BLOCKING GIT PUSH - Fix critical issues first!")
        sys.exit(1)
    elif results['security_status'] == 'WARNING':
        print(f"\n⚠️ PROCEED WITH CAUTION - Review warnings")
        sys.exit(0)
    else:
        print(f"\n🚀 SAFE TO PUSH - No security issues found!")
        sys.exit(0)


if __name__ == "__main__":
    main()