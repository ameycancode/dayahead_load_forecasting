#!/usr/bin/env python3
"""
Sample test execution script for the Energy Testing Framework

This script demonstrates how to run the generated tests and verify coverage.
"""

import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run all generated tests with coverage reporting"""
    print(" Running generated unit tests...")
   
    # Run pytest with coverage
    cmd = [
        "pytest",
        "tests/",
        "--cov=configs",
        "--cov=pipeline",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-fail-under=85",
        "-v"
    ]
   
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(" All tests passed!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(" Some tests failed:")
        print(e.stdout)
        print(e.stderr)
        return False

def run_linting():
    """Run code quality checks"""
    print(" Running code quality checks...")
   
    # Run flake8
    try:
        subprocess.run(["flake8", "tests/"], check=True)
        print(" Flake8 linting passed!")
    except subprocess.CalledProcessError:
        print(" Flake8 linting failed")
        return False
   
    # Run black check
    try:
        subprocess.run(["black", "--check", "tests/"], check=True)
        print(" Black formatting check passed!")
    except subprocess.CalledProcessError:
        print(" Black formatting check failed")
        return False
   
    return True

if __name__ == "__main__":
    print(" Energy Testing Framework - Test Execution")
    print("=" * 50)
   
    # Check if tests exist
    if not Path("tests").exists():
        print(" Tests directory not found. Run the framework first.")
        sys.exit(1)
   
    # Run tests
    tests_passed = run_tests()
    linting_passed = run_linting()
   
    if tests_passed and linting_passed:
        print("\n All checks passed! Your generated tests are ready for production.")
        sys.exit(0)
    else:
        print("\n  Some checks failed. Please review the output above.")
        sys.exit(1)
