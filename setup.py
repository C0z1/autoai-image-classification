#!/usr/bin/env python3
"""
AutoAI Image Classification - Setup Script
==========================================
Quick setup and environment verification
"""

import sys
import subprocess
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.8+"""
    print("Checking Python version...")
    if sys.version_info < (3, 8):
        print(f"❌ Python 3.8+ required. Current: {sys.version}")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def install_dependencies():
    """Install required packages"""
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def verify_imports():
    """Verify all required packages can be imported"""
    print("\nVerifying imports...")
    packages = [
        ('tensorflow', 'TensorFlow'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('PIL', 'Pillow'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'scikit-learn')
    ]
    
    all_good = True
    for module, name in packages:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"❌ {name} - Failed to import")
            all_good = False
    
    return all_good


def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    dirs = ['dataset', 'outputs']
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ {dir_name}/")
    
    return True


def test_feature_extraction():
    """Test if feature extraction script is accessible"""
    print("\nChecking feature extraction script...")
    script_path = Path('src/1_feature_extraction.py')
    
    if script_path.exists():
        print(f"✓ Feature extraction script found")
        return True
    else:
        print(f"❌ Feature extraction script not found at {script_path}")
        return False


def print_next_steps():
    """Print next steps for user"""
    print("\n" + "="*60)
    print("✅ SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run feature extraction:")
    print("   python src/1_feature_extraction.py")
    print("\n2. Follow Watson Studio guide:")
    print("   docs/WATSON_STUDIO_GUIDE.md")
    print("\n3. Use report template:")
    print("   docs/REPORT_TEMPLATE.md")
    print("\n" + "="*60)


def main():
    """Main setup function"""
    print("="*60)
    print("AutoAI Image Classification - Setup")
    print("="*60)
    
    # Run checks
    checks = [
        ("Python version", check_python_version),
        ("Dependencies", install_dependencies),
        ("Package imports", verify_imports),
        ("Directory structure", create_directories),
        ("Feature extraction script", test_feature_extraction)
    ]
    
    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))
    
    # Print summary
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, result in results:
        status = "✓" if result else "❌"
        print(f"{status} {name}")
        if not result:
            all_passed = False
    
    if all_passed:
        print_next_steps()
        return 0
    else:
        print("\n⚠️  Some checks failed. Please resolve issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
