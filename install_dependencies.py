
#!/usr/bin/env python3
"""
Script d'installation des dépendances pour les nouveaux utilisateurs
Usage: python install_dependencies.py
"""

import os
import sys
import subprocess
from auto_installer import run_auto_installer

def main():
    """Point d'entrée principal pour l'installation"""
    print("🚀 INSTALLATION DES DÉPENDANCES - Projet AGI/ASI AI")
    print("="*60)
    print("Ce script va installer automatiquement toutes les dépendances")
    print("nécessaires pour faire fonctionner le projet.")
    print("="*60 + "\n")
    
    # Vérifier Python
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ est requis. Version actuelle:", sys.version)
        return False
    
    print(f"✅ Python {python_version.major}.{python_version.minor} détecté")
    
    # Installer les dépendances du requirements.txt d'abord
    if os.path.exists('requirements.txt'):
        print("\n📦 Installation des dépendances de base...")
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Dépendances de base installées avec succès")
            else:
                print(f"⚠️ Avertissement lors de l'installation: {result.stderr}")
        except Exception as e:
            print(f"❌ Erreur lors de l'installation des dépendances de base: {str(e)}")
    
    # Lancer l'auto-installer pour les modules supplémentaires
    print("\n🔧 Vérification et installation des modules supplémentaires...")
    success = run_auto_installer()
    
    print("\n" + "="*60)
    if success:
        print("🎉 INSTALLATION TERMINÉE AVEC SUCCÈS!")
        print("\n📋 Prochaines étapes:")
        print("1. Configurez votre clé API Gemini dans le fichier .env")
        print("2. Lancez l'application avec: python app.py")
        print("3. Ouvrez votre navigateur sur http://localhost:5000")
    else:
        print("⚠️ INSTALLATION PARTIELLEMENT RÉUSSIE")
        print("\n📋 Actions recommandées:")
        print("1. Vérifiez les erreurs ci-dessus")
        print("2. Installez manuellement les modules manquants si nécessaire")
        print("3. Essayez de lancer l'application avec: python app.py")
    print("="*60)
    
    return success

if __name__ == "__main__":
    main()
