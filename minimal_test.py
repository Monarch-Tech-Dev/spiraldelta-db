#!/usr/bin/env python3
"""
Minimal test of SpiralDeltaDB imports and basic structure.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_structure():
    """Test that we can import basic structure."""
    print("Testing package structure...")
    
    try:
        # Test basic imports without numpy dependencies
        print("✓ Python path configured")
        
        # Check that files exist
        src_dir = Path(__file__).parent / "src" / "spiraldelta"
        
        expected_files = [
            "__init__.py",
            "types.py", 
            "spiral_coordinator.py",
            "delta_encoder.py",
            "search_engine.py",
            "storage.py",
            "database.py"
        ]
        
        for file in expected_files:
            file_path = src_dir / file
            if file_path.exists():
                print(f"✓ {file} exists")
            else:
                print(f"✗ {file} missing")
                return False
        
        # Try to import without executing numpy code
        try:
            import importlib.util
            
            # Load types module
            spec = importlib.util.spec_from_file_location("types", src_dir / "types.py")
            types_module = importlib.util.module_from_spec(spec)
            
            print("✓ Module loading works")
            
        except Exception as e:
            print(f"✗ Module loading failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Structure test failed: {e}")
        return False

def test_configuration():
    """Test project configuration files."""
    print("\nTesting configuration files...")
    
    try:
        project_root = Path(__file__).parent
        
        config_files = [
            "pyproject.toml",
            "setup.py", 
            "requirements.txt",
            ".gitignore"
        ]
        
        for file in config_files:
            file_path = project_root / file
            if file_path.exists():
                print(f"✓ {file} exists")
            else:
                print(f"✗ {file} missing")
                return False
        
        # Check test directory
        test_dir = project_root / "tests"
        if test_dir.exists():
            print("✓ tests/ directory exists")
            
            test_files = list(test_dir.glob("test_*.py"))
            print(f"✓ Found {len(test_files)} test files")
        else:
            print("✗ tests/ directory missing")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def check_imports_syntax():
    """Check that Python files have valid syntax."""
    print("\nChecking Python syntax...")
    
    try:
        import ast
        
        src_dir = Path(__file__).parent / "src" / "spiraldelta"
        python_files = list(src_dir.glob("*.py"))
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    source = f.read()
                
                # Parse to check syntax
                ast.parse(source)
                print(f"✓ {py_file.name} syntax OK")
                
            except SyntaxError as e:
                print(f"✗ {py_file.name} syntax error: {e}")
                return False
                
        return True
        
    except Exception as e:
        print(f"✗ Syntax check failed: {e}")
        return False

def main():
    """Run minimal tests."""
    print("SpiralDeltaDB Minimal Test Suite")
    print("=" * 40)
    
    tests = [
        test_structure,
        test_configuration,
        check_imports_syntax,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All minimal tests passed!")
        print("\nTo run full tests, install dependencies:")
        print("python3 -m venv venv")
        print("source venv/bin/activate") 
        print("pip install -e .")
        print("python3 basic_test.py")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())