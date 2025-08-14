#!/usr/bin/env python3
"""
Security Update Verification Script
This script helps verify that security updates have been applied correctly.
"""

import subprocess
import sys
import json
from pathlib import Path

def run_command(cmd):
    """Run a command and return the result."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("🔒 Security Update Verification")
    print("=" * 50)
    
    # Check if we're in the right directory
    requirements_file = Path("services/requirements.txt")
    if not requirements_file.exists():
        print("❌ Error: services/requirements.txt not found")
        print("Please run this script from the workspace root directory")
        sys.exit(1)
    
    # Read requirements.txt
    print("\n📋 Current package versions in services/requirements.txt:")
    with open(requirements_file, 'r') as f:
        requirements = f.read()
    
    # Check key security packages with their exact names in requirements.txt
    security_packages = {
        'python-jose[cryptography]': '3.5.0',
        'gunicorn': '23.0.0', 
        'python-multipart': '0.0.20',
        'requests': '2.32.4',
        'black': '25.1.0'
    }
    
    print("\n🔍 Checking security package versions:")
    all_secure = True
    
    for package, expected_version in security_packages.items():
        # Extract version from requirements.txt
        import re
        # Handle packages with extras like [cryptography]
        base_package = package.split('[')[0]
        pattern = rf"{re.escape(package)}==([0-9.]+)"
        match = re.search(pattern, requirements)
        
        if match:
            current_version = match.group(1)
            if current_version == expected_version:
                print(f"✅ {base_package}: {current_version} (Secure)")
            else:
                print(f"❌ {base_package}: {current_version} (Expected: {expected_version})")
                all_secure = False
        else:
            print(f"⚠️  {base_package}: Not found in requirements.txt")
            all_secure = False
    
    print("\n📊 Security Status Summary:")
    if all_secure:
        print("✅ All security packages are up to date!")
    else:
        print("❌ Some security packages need updates")
    
    print("\n🚀 Recommendations:")
    print("1. Run 'pip install -r services/requirements.txt' to install updated packages")
    print("2. Test your application with the new versions")
    print("3. Consider setting up automated security scanning")
    print("4. Monitor for new security advisories")
    
    print("\n📚 Security Tools to Consider:")
    print("- pip-audit: Official Python security checker")
    print("- safety: Comprehensive security scanning")
    print("- bandit: Security linter for Python code")
    print("- semgrep: Advanced security analysis")
    
    print("\n🔗 Useful Resources:")
    print("- Python Security Advisories: https://github.com/pypa/advisory-database")
    print("- CVE Database: https://cve.mitre.org/")
    print("- OWASP: https://owasp.org/")

if __name__ == "__main__":
    main()