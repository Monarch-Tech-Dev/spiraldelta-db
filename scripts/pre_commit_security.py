#!/usr/bin/env python3
"""
Pre-commit security hook for SpiralDeltaDB.

Runs before each git commit to ensure no sensitive information is committed.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_security_audit():
    """Run security audit script."""
    script_dir = Path(__file__).parent
    security_script = script_dir / "security_audit.py"
    
    try:
        # Run security audit
        result = subprocess.run([
            sys.executable, str(security_script)
        ], capture_output=True, text=True, cwd=script_dir.parent)
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Security audit failed: {e}")
        return False

def check_business_model_protection():
    """Ensure business model elements are protected."""
    protected_files = [
        "README.md",
        "PRIVATE_INTEGRATION.md", 
        "PROJECT_SUMMARY.md"
    ]
    
    business_keywords = [
        "Monarch AI",
        "enterprise@monarchai.com",
        "Soft Armor",
        "Conscious Engine",
        "Enterprise Features"
    ]
    
    print("🏢 Checking business model protection...")
    
    for file_path in protected_files:
        file_path = Path(file_path)
        if file_path.exists():
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Ensure business model elements are present
            for keyword in business_keywords:
                if keyword in content:
                    print(f"✅ Business model element '{keyword}' found in {file_path}")
                    break
            else:
                print(f"⚠️ No business model elements found in {file_path}")
    
    return True

def main():
    """Main pre-commit security check."""
    print("🔒 Pre-Commit Security Check")
    print("=" * 40)
    
    # Check business model protection
    business_ok = check_business_model_protection()
    
    # Run security audit
    security_ok = run_security_audit()
    
    if security_ok and business_ok:
        print("\n✅ PRE-COMMIT SECURITY: PASSED")
        print("Safe to commit!")
        sys.exit(0)
    else:
        print("\n❌ PRE-COMMIT SECURITY: FAILED")
        print("Fix security issues before committing!")
        sys.exit(1)

if __name__ == "__main__":
    main()