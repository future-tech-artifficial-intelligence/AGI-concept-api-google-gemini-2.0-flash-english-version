
#!/usr/bin/env python3
"""
Script de vérification des dépendances
Affiche l'état de tous les modules requis sans les installer
"""

from auto_installer import AutoInstaller
import sys

def main():
    """Vérifie l'état des dépendances"""
    print("🔍 VÉRIFICATION DES DÉPENDANCES")
    print("="*50)
    
    installer = AutoInstaller()
    
    # Vérifier les modules requis
    print("\n📋 MODULES REQUIS:")
    missing_required = 0
    for module_name, package_spec in installer.required_modules.items():
        available = installer.check_module_availability(module_name)
        status = "✅" if available else "❌"
        print(f"{status} {module_name}")
        if not available:
            missing_required += 1
    
    # Vérifier les modules optionnels
    print("\n📋 MODULES OPTIONNELS:")
    missing_optional = 0
    for module_name, package_spec in installer.optional_modules.items():
        available = installer.check_module_availability(module_name)
        status = "✅" if available else "⚠️"
        print(f"{status} {module_name}")
        if not available:
            missing_optional += 1
    
    print("\n" + "="*50)
    print(f"📊 RÉSUMÉ:")
    print(f"   Modules requis manquants: {missing_required}")
    print(f"   Modules optionnels manquants: {missing_optional}")
    
    if missing_required > 0:
        print(f"\n💡 Pour installer les modules manquants:")
        print(f"   python install_dependencies.py")
    else:
        print(f"\n🎉 Tous les modules requis sont installés!")
    
    print("="*50)
    
    return missing_required == 0

if __name__ == "__main__":
    main()
