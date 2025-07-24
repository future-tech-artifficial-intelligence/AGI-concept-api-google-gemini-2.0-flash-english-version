"""
Script de lancement pour les tests d'interaction Gemini-Searx
"""

import os
import sys
import time
import requests
from pathlib import Path

def check_prerequisites():
    """Vérifie que tous les prérequis sont remplis"""
    print("🔍 Vérification des prérequis...")
    
    # Vérifier que nous sommes dans le bon répertoire
    required_files = [
        'app.py',
        'searx_interface.py', 
        'interactive_web_navigator.py',
        'test_gemini_searx_interaction.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Fichiers manquants: {', '.join(missing_files)}")
        return False
    
    print("✅ Tous les fichiers requis sont présents")
    
    # Vérifier que l'app Flask est en cours d'exécution
    try:
        response = requests.get("http://localhost:5000/", timeout=5)
        if response.status_code == 200:
            print("✅ Application Flask accessible sur localhost:5000")
        else:
            print(f"⚠️  Application Flask répond avec le code: {response.status_code}")
    except Exception as e:
        print(f"❌ Application Flask non accessible: {str(e)}")
        print("💡 Assurez-vous que 'python app.py' est en cours d'exécution")
        return False
    
    # Vérifier Searx (optionnel car peut être intégré dans l'app)
    try:
        response = requests.get("http://localhost:8080/", timeout=5)
        if response.status_code == 200:
            print("✅ Searx accessible sur localhost:8080")
        else:
            print(f"⚠️  Searx répond avec le code: {response.status_code}")
    except Exception as e:
        print(f"⚠️  Searx non accessible sur localhost:8080: {str(e)}")
        print("💡 Searx peut être intégré dans l'application Flask")
    
    return True

def main():
    """Fonction principale"""
    print("🧪 Test des Capacités d'Interaction Gemini-Searx")
    print("=" * 55)
    
    if not check_prerequisites():
        print("\n❌ Prérequis non remplis. Veuillez:")
        print("1. Vous assurer d'être dans le bon répertoire")
        print("2. Démarrer l'application avec: python app.py")
        print("3. Optionnellement démarrer Searx")
        return
    
    # Attendre un moment pour que les services se stabilisent
    print("\n⏳ Attente de stabilisation des services...")
    time.sleep(3)
    
    print("🚀 Lancement du test d'interaction Gemini-Searx...\n")
    
    # Lancer le test
    try:
        import subprocess
        result = subprocess.run([sys.executable, "test_gemini_searx_interaction.py"], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n🎉 Tests terminés avec succès!")
        else:
            print(f"\n⚠️  Tests terminés avec des erreurs (code: {result.returncode})")
            
    except Exception as e:
        print(f"\n❌ Erreur lors du lancement des tests: {str(e)}")
    
    print("📁 Consultez le dossier 'test_results_searx_interaction' pour les rapports détaillés")

if __name__ == "__main__":
    main()
