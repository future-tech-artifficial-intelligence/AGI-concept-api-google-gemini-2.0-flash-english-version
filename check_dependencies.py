#!/usr/bin/env python3
"""
**Dependency Verification Script**
Displays the status of all required modules without installing them
"""

from auto_installer import AutoInstaller
import sys

def main():
    """Checks the status of dependencies"""
    print("🔍 DEPENDENCY VERIFICATION")
    print("="*50)
    
    installer = AutoInstaller()
    
    # Check required modules
    print("\n📋 REQUIRED MODULES:")
    missing_required = 0
    for module_name, package_spec in installer.required_modules.items():
        available = installer.check_module_availability(module_name)
        status = "✅" if available else "❌"
        print(f"{status} {module_name}")
        if not available:
            missing_required += 1
    
    # Check optional modules
    print("\n📋 OPTIONAL MODULES:")
    missing_optional = 0
    for module_name, package_spec in installer.optional_modules.items():
        available = installer.check_module_availability(module_name)
        status = "✅" if available else "⚠️"
        print(f"{status} {module_name}")
        if not available:
            missing_optional += 1
    
    print("\n" + "="*50)
    print(f"📊 SUMMARY:")
    print(f"   Missing required modules: {missing_required}")
    print(f"   Missing optional modules: {missing_optional}")
    
    if missing_required > 0:
        print(f"\n💡 To install missing modules:")
        print(f"   python install_dependencies.py")
    else:
        print(f"\n🎉 All required modules are installed!")
    
    print("="*50)
    
    return missing_required == 0

if __name__ == "__main__":
    main()
